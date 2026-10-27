"""Fact-check runs: migration 023, dispatch, auto-start, backfill.

Hermetic. Nothing here dials a database, a scheduler, or a model API.
"""

import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from nlp_pillars import spend
from nlp_pillars.cli import app
from nlp_pillars.config import Settings
from nlp_pillars.factcheck.constants import STAGES
from nlp_pillars.factcheck.errors import FactcheckError
from nlp_pillars.schemas import FACTCHECK_STAGES, RunStatus
from webui.services import podcast_factcheck_service, run_service
from webui.services.run_service import KIND_DISCOVER, execute_run

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (ROOT / "docs/migrations/023_podcast_factcheck_runs.sql").read_text()
SCHEMA = (ROOT / "schema.sql").read_text()
RUNNER = CliRunner()
PILLAR = "neural-architectures-language"
SCRIPT_ID = "b4269725-1111-4111-8111-111111111111"


def _budget(spent: float, cap: float = 2.0) -> spend.Budget:
    start = datetime(2026, 10, 25, tzinfo=timezone.utc)
    return spend.Budget(
        cap_usd=cap, spent_usd=spent, day_start=start, resets_at=start + timedelta(days=1),
    )


def _statement(pattern: str) -> str:
    match = re.search(pattern, MIGRATION, re.S | re.I)
    assert match, pattern
    return match.group(0)


def test_stage_slugs_match_the_schema_and_the_engine():
    assert [stage.value for stage in FACTCHECK_STAGES] == list(STAGES)


def test_auto_dispatch_defaults_on():
    assert Settings.model_fields["factcheck_auto_dispatch"].default is True


def test_migration_keeps_every_existing_kind_and_adds_factcheck():
    kind = _statement(r"ADD CONSTRAINT pipeline_runs_kind_check.*?;")
    for value in (
        "run_daily", "process_selected", "discover", "podcast_audio",
        "upload", "podcast_script", "podcast_factcheck",
    ):
        assert f"'{value}'" in kind
    trigger = _statement(r"ADD CONSTRAINT pipeline_runs_trigger_source_check.*?;")
    for value in (
        "ui_pipeline", "ui_select", "ui_discover", "scheduler",
        "ui_podcast_audio", "ui_upload", "ui_podcast_script",
        "ui_podcast_factcheck", "auto_podcast_factcheck",
    ):
        assert f"'{value}'" in trigger
    assert MIGRATION.rstrip().endswith("NOTIFY pgrst, 'reload schema';")
    assert "UPDATE podcast_scripts" not in MIGRATION
    assert "podcast_factcheck" in SCHEMA
    assert "'auto_podcast_factcheck'" in SCHEMA
    # Later kinds join the exemption (028 adds playground_explain); these stay.
    exempt = SCHEMA.split("CREATE UNIQUE INDEX pipeline_runs_one_active_per_pillar", 1)[1]
    exempt = exempt.split(";", 1)[0]
    for kind in ("'upload'", "'podcast_script'", "'podcast_factcheck'"):
        assert kind in exempt


def _pillar_table() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE pipeline_runs ("
        " id INTEGER PRIMARY KEY, pillar_id TEXT NOT NULL,"
        " kind TEXT NOT NULL, status TEXT NOT NULL)"
    )
    conn.execute(_statement(
        r"CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_per_pillar[^;]*;"
    ))
    return conn


def _insert(conn, kind, status="running", pillar=PILLAR):
    conn.execute(
        "INSERT INTO pipeline_runs (pillar_id, kind, status) VALUES (?, ?, ?)",
        (pillar, kind, status),
    )


def test_factcheck_may_overlap_discovery_on_the_same_pillar():
    conn = _pillar_table()
    _insert(conn, KIND_DISCOVER)
    _insert(conn, "podcast_factcheck")
    _insert(conn, "podcast_factcheck")


def test_two_discovery_runs_still_conflict():
    conn = _pillar_table()
    _insert(conn, KIND_DISCOVER)
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, KIND_DISCOVER)


def test_one_active_factcheck_per_script():
    sql = _statement(
        r"CREATE UNIQUE INDEX IF NOT EXISTS "
        r"pipeline_runs_one_active_factcheck_per_script[^;]*;"
    )
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE pipeline_runs ("
        " id INTEGER PRIMARY KEY, pillar_id TEXT, kind TEXT,"
        " status TEXT, result TEXT)"
    )
    conn.execute(sql)
    conn.execute(
        "INSERT INTO pipeline_runs (pillar_id, kind, status, result) "
        "VALUES ('p', 'podcast_factcheck', 'running', ?)",
        ('{"script_id":"abc"}',),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO pipeline_runs (pillar_id, kind, status, result) "
            "VALUES ('p', 'podcast_factcheck', 'pending', ?)",
            ('{"script_id":"abc"}',),
        )
    conn.execute(
        "INSERT INTO pipeline_runs (pillar_id, kind, status, result) "
        "VALUES ('p', 'podcast_factcheck', 'succeeded', ?)",
        ('{"script_id":"abc"}',),
    )
    conn.execute(
        "INSERT INTO pipeline_runs (pillar_id, kind, status, result) "
        "VALUES ('p', 'podcast_factcheck', 'running', ?)",
        ('{"script_id":"def"}',),
    )


def test_dispatch_seeds_factcheck_stages_and_stores_script_id():
    fake = MagicMock(id="run-fc")
    scheduler = MagicMock()
    with patch.object(
        run_service.db, "create_pipeline_run", return_value=fake,
    ) as create:
        run_id = run_service.dispatch_run(
            scheduler,
            {},
            pillar_id=PILLAR,
            trigger_source=podcast_factcheck_service.TRIGGER_UI_PODCAST_FACTCHECK,
            kind=run_service.KIND_PODCAST_FACTCHECK,
            script_id=SCRIPT_ID,
            paper_id="arxiv:1",
            initial_result={"script_id": SCRIPT_ID, "paper_id": "arxiv:1"},
        )
    assert run_id == "run-fc"
    assert create.call_args.args[3] == [stage.value for stage in FACTCHECK_STAGES]
    assert create.call_args.kwargs["result"]["script_id"] == SCRIPT_ID
    job_kwargs = scheduler.add_job.call_args.kwargs["kwargs"]
    assert job_kwargs["scheduler"] is scheduler
    assert "initial_result" not in job_kwargs
    assert podcast_factcheck_service.TRIGGER_UI_PODCAST_FACTCHECK in (
        run_service.WEBUI_TRIGGER_SOURCES
    )
    assert podcast_factcheck_service.TRIGGER_AUTO_PODCAST_FACTCHECK in (
        run_service.WEBUI_TRIGGER_SOURCES
    )


def test_a_saved_script_starts_a_factcheck_and_a_failure_does_not_fail_the_script():
    events = {"run-1": threading.Event()}
    scheduler = MagicMock()
    started = {}

    def dispatch(*_args, **kwargs):
        started["kind"] = kwargs.get("kind")
        started["script_id"] = kwargs.get("script_id")
        raise RuntimeError("factcheck dispatch down")

    with (
        patch.object(run_service.db, "start_pipeline_run"),
        patch.object(run_service.db, "update_pipeline_run_stage"),
        patch.object(run_service.db, "finish_pipeline_run") as finish,
        patch.object(
            run_service.podcast_script_service,
            "run_podcast_script_job",
            return_value={
                "saved": True,
                "script_id": SCRIPT_ID,
                "paper_id": "arxiv:1",
                "title": "AML",
                "word_count": 10,
            },
        ),
        patch.object(run_service, "dispatch_run", side_effect=dispatch),
        patch.object(
            podcast_factcheck_service.db,
            "get_active_factcheck_run",
            return_value=None,
        ),
        patch.object(
            podcast_factcheck_service.spend,
            "current_budget",
            return_value=_budget(spent=0.0),
        ),
        patch.object(
            podcast_factcheck_service, "get_settings",
        ) as settings,
    ):
        settings.return_value.factcheck_auto_dispatch = True
        execute_run(
            "run-1",
            run_service.KIND_PODCAST_SCRIPT,
            PILLAR,
            events["run-1"],
            events,
            paper_id="arxiv:1",
            options=SimpleNamespace(),
            scheduler=scheduler,
        )

    assert finish.call_args.args[1] == RunStatus.SUCCEEDED.value
    assert finish.call_args.kwargs["result"]["saved"] is True
    assert started["kind"] == "podcast_factcheck"
    assert started["script_id"] == SCRIPT_ID


def test_auto_dispatch_stays_off_when_the_flag_is_false():
    events = {"run-1": threading.Event()}
    with (
        patch.object(run_service.db, "start_pipeline_run"),
        patch.object(run_service.db, "update_pipeline_run_stage"),
        patch.object(run_service.db, "finish_pipeline_run") as finish,
        patch.object(
            run_service.podcast_script_service,
            "run_podcast_script_job",
            return_value={"saved": True, "script_id": SCRIPT_ID, "paper_id": "p"},
        ),
        patch.object(run_service, "dispatch_run") as dispatch,
        patch.object(
            podcast_factcheck_service, "get_settings",
        ) as settings,
    ):
        settings.return_value.factcheck_auto_dispatch = False
        execute_run(
            "run-1",
            run_service.KIND_PODCAST_SCRIPT,
            PILLAR,
            events["run-1"],
            events,
            paper_id="p",
            options=SimpleNamespace(),
            scheduler=MagicMock(),
        )
    finish.assert_called_once()
    dispatch.assert_not_called()


def test_factcheck_job_failure_does_not_rewrite_the_script():
    events = {"run-1": threading.Event()}
    recorded = []

    def on_stage(run_id, name, status, detail=None):
        recorded.append((name, status))

    with (
        patch.object(run_service.db, "start_pipeline_run"),
        patch.object(run_service.db, "update_pipeline_run_stage", side_effect=on_stage),
        patch.object(run_service.db, "finish_pipeline_run") as finish,
        patch.object(
            podcast_factcheck_service,
            "execute_factcheck",
            side_effect=FactcheckError("no stored paper text"),
        ),
    ):
        with pytest.raises(FactcheckError):
            execute_run(
                "run-1",
                run_service.KIND_PODCAST_FACTCHECK,
                PILLAR,
                events["run-1"],
                events,
                script_id=SCRIPT_ID,
                paper_id="arxiv:1",
            )
    finish.assert_called_once()
    assert finish.call_args.args[1] == RunStatus.FAILED.value
    assert ("factcheck_extract", "failed") in recorded
    assert ("factcheck_screen", "skipped") in recorded
    text = (
        ROOT / "webui/services/podcast_factcheck_service.py"
    ).read_text()
    assert "update_podcast" not in text
    assert "add_podcast_script" not in text


def test_factcheck_cancel_marks_the_run_cancelled():
    events = {"run-1": threading.Event()}
    with (
        patch.object(run_service.db, "start_pipeline_run"),
        patch.object(run_service.db, "update_pipeline_run_stage"),
        patch.object(run_service.db, "finish_pipeline_run") as finish,
        patch.object(
            podcast_factcheck_service,
            "execute_factcheck",
            side_effect=podcast_factcheck_service.FactcheckCancelled(
                "cancelled before factcheck_extract"
            ),
        ),
    ):
        execute_run(
            "run-1",
            run_service.KIND_PODCAST_FACTCHECK,
            PILLAR,
            events["run-1"],
            events,
            script_id=SCRIPT_ID,
        )
    assert finish.call_args.args[1] == RunStatus.CANCELLED.value


def test_backfill_dry_run_prints_the_cost_and_does_not_call_a_model():
    rows = [{
        "id": SCRIPT_ID,
        "title": "AML",
        "paper_id": "arxiv:1",
        "pillar_id": PILLAR,
    }]
    with (
        patch(
            "nlp_pillars.factcheck.backfill.db.list_scripts_missing_factcheck",
            return_value=rows,
        ),
        patch("nlp_pillars.factcheck.backfill.execute_factcheck") as execute,
    ):
        result = RUNNER.invoke(app, ["factcheck", "backfill", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "$0.6" in result.output
    assert "1.3" in result.output
    assert SCRIPT_ID in result.output
    assert "No model call" in result.output
    assert "Expected cost" in result.output
    execute.assert_not_called()


def test_backfill_stops_on_the_first_failure_without_changing_scripts():
    from nlp_pillars.factcheck.backfill import run_backfill

    rows = [
        {"id": SCRIPT_ID, "title": "AML"},
        {"id": "eeeeeeee-1111-4111-8111-111111111111", "title": "Next"},
    ]
    calls = []

    def execute(script_id):
        calls.append(script_id)
        raise FactcheckError("no stored paper text")

    with patch(
        "nlp_pillars.factcheck.backfill.db.list_scripts_missing_factcheck",
        return_value=rows,
    ):
        report = run_backfill(dry_run=False, execute=execute)
    assert calls == [SCRIPT_ID]
    assert report.failed == 1
    assert report.checked == 0
    assert "No script was changed" in "\n".join(report.lines)


def test_auto_dispatch_is_skipped_once_the_daily_cap_is_spent():
    with (
        patch.object(
            podcast_factcheck_service.db, "get_active_factcheck_run", return_value=None,
        ),
        patch.object(
            podcast_factcheck_service.spend,
            "current_budget",
            return_value=_budget(spent=2.0),
        ),
        patch.object(podcast_factcheck_service, "get_settings") as settings,
        patch.object(run_service, "dispatch_run") as dispatch,
    ):
        settings.return_value.factcheck_auto_dispatch = True
        result = podcast_factcheck_service.maybe_dispatch_after_script(
            MagicMock(), {}, script_id=SCRIPT_ID, paper_id="arxiv:1", pillar_id=PILLAR,
        )
    assert result is None
    dispatch.assert_not_called()


def test_auto_dispatch_is_skipped_when_spend_cannot_be_read():
    with (
        patch.object(
            podcast_factcheck_service.db, "get_active_factcheck_run", return_value=None,
        ),
        patch.object(
            podcast_factcheck_service.spend,
            "current_budget",
            side_effect=spend.SpendUnavailableError("db down"),
        ),
        patch.object(podcast_factcheck_service, "get_settings") as settings,
        patch.object(run_service, "dispatch_run") as dispatch,
    ):
        settings.return_value.factcheck_auto_dispatch = True
        result = podcast_factcheck_service.maybe_dispatch_after_script(
            MagicMock(), {}, script_id=SCRIPT_ID, paper_id="arxiv:1", pillar_id=PILLAR,
        )
    assert result is None
    dispatch.assert_not_called()
