"""Podcast script generation as a background run: migration, stages, outcomes.

Fixtures are local and the basename is unique across tests/. Nothing here
reaches a database, a scheduler or the network.
"""

import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nlp_pillars.agents.podcast_agent import InsufficientSourceMaterialError
from nlp_pillars.db import PodcastScriptSaveError
from nlp_pillars.orchestrator import RunCancelledError
from nlp_pillars.podcast_options import resolve
from nlp_pillars.schemas import (
    PODCAST_SCRIPT_STAGES,
    PodcastScript,
    RunStatus,
    SourceMaterial,
    StageName,
    StageStatus,
)
from webui.services import podcast_script_service, run_service
from webui.services.run_service import (
    KIND_DISCOVER,
    KIND_PODCAST_SCRIPT,
    KIND_UPLOAD,
    dispatch_run,
    execute_run,
)

PILLAR = "neural-architectures-language"

MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "docs" / "migrations" / "016_podcast_script_runs.sql"
).read_text()


def _statement(pattern: str) -> str:
    match = re.search(pattern, MIGRATION, re.S | re.I)
    assert match, f"migration 016 no longer contains a statement matching {pattern!r}"
    return match.group(0)


def test_the_migration_widens_both_check_constraints_without_dropping_anything():
    kind = _statement(r"ADD CONSTRAINT pipeline_runs_kind_check.*?;")
    for value in (
        "run_daily", "process_selected", "discover", "podcast_audio",
        "upload", "podcast_script",
    ):
        assert f"'{value}'" in kind

    trigger = _statement(r"ADD CONSTRAINT pipeline_runs_trigger_source_check.*?;")
    for value in (
        "ui_pipeline", "ui_select", "ui_discover", "scheduler",
        "ui_podcast_audio", "ui_upload", "ui_podcast_script",
    ):
        assert f"'{value}'" in trigger


def test_the_migration_ends_by_reloading_postgrests_schema_cache():
    assert MIGRATION.rstrip().endswith("NOTIFY pgrst, 'reload schema';")


def _guarded_table() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE pipeline_runs ("
        " id INTEGER PRIMARY KEY, pillar_id TEXT NOT NULL,"
        " kind TEXT NOT NULL, status TEXT NOT NULL)"
    )
    conn.execute(_statement(
        r"CREATE UNIQUE INDEX[^;]*pipeline_runs_one_active_per_pillar[^;]*;"
    ))
    return conn


def _insert(conn, kind, status="running", pillar=PILLAR):
    conn.execute(
        "INSERT INTO pipeline_runs (pillar_id, kind, status) VALUES (?, ?, ?)",
        (pillar, kind, status),
    )


def test_a_script_run_may_overlap_a_discovery_run_on_the_same_pillar():
    conn = _guarded_table()
    _insert(conn, KIND_DISCOVER)
    _insert(conn, KIND_PODCAST_SCRIPT)  # must not raise


def test_two_script_runs_on_one_pillar_may_overlap():
    conn = _guarded_table()
    _insert(conn, KIND_PODCAST_SCRIPT)
    _insert(conn, KIND_PODCAST_SCRIPT)


def test_upload_and_script_still_exempt_together():
    conn = _guarded_table()
    _insert(conn, KIND_UPLOAD)
    _insert(conn, KIND_PODCAST_SCRIPT)


def test_two_discovery_runs_still_conflict():
    conn = _guarded_table()
    _insert(conn, KIND_DISCOVER)
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, KIND_DISCOVER)


def test_dispatch_seeds_podcast_script_stages():
    fake_run = MagicMock(id="run-ps-1")
    scheduler = MagicMock()
    events = {}
    with patch.object(run_service.db, "create_pipeline_run", return_value=fake_run) as create:
        run_id = dispatch_run(
            scheduler,
            events,
            pillar_id=PILLAR,
            trigger_source=podcast_script_service.TRIGGER_UI_PODCAST_SCRIPT,
            kind=KIND_PODCAST_SCRIPT,
            paper_id="p1",
            options=resolve(None),
        )
    assert run_id == "run-ps-1"
    stage_names = create.call_args[0][3]
    assert stage_names == [s.value for s in PODCAST_SCRIPT_STAGES]
    assert podcast_script_service.TRIGGER_UI_PODCAST_SCRIPT in run_service.WEBUI_TRIGGER_SOURCES


def _script():
    return PodcastScript(
        paper_id="2403.05525",
        pillar_id=PILLAR,
        title="Deep Dive: Test",
        script="[HOST]: Hello.",
        word_count=2,
        key_points=["a"],
        source_material=SourceMaterial(level="full", full_text_chars=1000),
        created_at=datetime(2026, 8, 30),
    )


def test_execute_run_records_saved_script_in_result():
    events = {"run-1": threading.Event()}
    stages = []

    def on_stage(name, status, detail=None):
        stages.append((name, status, detail))

    with patch.object(run_service.db, "start_pipeline_run"), \
         patch.object(run_service.db, "update_pipeline_run_stage",
                      side_effect=lambda *a, **k: on_stage(a[1], a[2], a[3] if len(a) > 3 else k.get("detail"))), \
         patch.object(run_service.db, "finish_pipeline_run") as finish, \
         patch.object(
             podcast_script_service,
             "run_podcast_script_job",
             return_value={
                 "saved": True,
                 "script_id": "sid-1",
                 "title": "Deep Dive: Test",
                 "word_count": 2,
                 "warnings": [],
             },
         ):
        execute_run(
            "run-1",
            KIND_PODCAST_SCRIPT,
            PILLAR,
            events["run-1"],
            events,
            paper_id="p1",
            options=resolve(None),
        )

    finish.assert_called_once()
    assert finish.call_args[0][1] == RunStatus.SUCCEEDED.value
    assert finish.call_args[1]["result"]["saved"] is True
    assert finish.call_args[1]["result"]["script_id"] == "sid-1"


def test_job_marks_later_stages_skipped_when_prepare_refuses():
    recorded = []

    def on_stage(name, status, detail=None):
        recorded.append((name, status, detail))

    async def boom(self, paper_id, pillar_id, *, on_stage=None, cancel=None):
        if on_stage:
            on_stage(
                StageName.PODCAST_PREPARE.value,
                StageStatus.FAILED.value,
                "nothing to write from",
            )
        raise InsufficientSourceMaterialError("nothing to write from")

    with patch.object(
        podcast_script_service.PodcastAgent, "generate", boom
    ), patch.object(
        podcast_script_service.PodcastAgent, "__init__", lambda self, options=None: None
    ):
        with pytest.raises(InsufficientSourceMaterialError):
            podcast_script_service.run_podcast_script_job(
                "run-1",
                "p1",
                PILLAR,
                resolve(None),
                on_stage,
                threading.Event(),
            )

    by_name = {name: (status, detail) for name, status, detail in recorded}
    assert by_name[StageName.PODCAST_PREPARE.value][0] == StageStatus.FAILED.value
    for stage in PODCAST_SCRIPT_STAGES[1:]:
        assert by_name[stage.value][0] == StageStatus.SKIPPED.value


def test_job_keeps_the_script_when_save_fails():
    script = _script()

    async def ok(self, paper_id, pillar_id, *, on_stage=None, cancel=None):
        if on_stage:
            for stage in PODCAST_SCRIPT_STAGES[:-1]:
                on_stage(stage.value, StageStatus.COMPLETED.value, "ok")
        return script

    with patch.object(
        podcast_script_service.PodcastAgent, "generate", ok
    ), patch.object(
        podcast_script_service.PodcastAgent, "__init__", lambda self, options=None: None
    ), patch.object(
        podcast_script_service.db,
        "add_podcast_script",
        side_effect=PodcastScriptSaveError("connection refused"),
    ):
        result = podcast_script_service.run_podcast_script_job(
            "run-1",
            "p1",
            PILLAR,
            resolve(None),
            lambda *a: None,
            threading.Event(),
        )

    assert result["saved"] is False
    assert result["script"] == "[HOST]: Hello."
    assert any("NOT saved" in w for w in result["warnings"])


def test_cancel_before_start_skips_remaining_stages():
    recorded = []

    def on_stage(name, status, detail=None):
        recorded.append((name, status))

    cancel = threading.Event()
    cancel.set()

    async def never(self, *a, **k):
        raise AssertionError("generate should not run when cancelled at persist only")

    # Cancel is checked inside generate at stage boundaries; simulate cancel
    # raised from generate so the service closes out.
    async def cancelled(self, *a, **k):
        raise RunCancelledError("cancelled")

    with patch.object(
        podcast_script_service.PodcastAgent, "generate", cancelled
    ), patch.object(
        podcast_script_service.PodcastAgent, "__init__", lambda self, options=None: None
    ):
        with pytest.raises(RunCancelledError):
            podcast_script_service.run_podcast_script_job(
                "run-1", "p1", PILLAR, resolve(None), on_stage, cancel
            )

    assert all(status == StageStatus.SKIPPED.value for _, status in recorded)
