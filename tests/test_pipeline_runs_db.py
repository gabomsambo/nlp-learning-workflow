"""Tests for the pipeline_runs / pipeline_run_stages data layer.

Mocking style note: these use `set_client(Mock())` and configure the chained
TableQuery calls. They deliberately do NOT follow `tests/test_db.py`'s
TestClientBootstrap, which patches `nlp_pillars.db.create_client` — a function that
does not exist in db.py and is part of the known-red baseline.

Nothing here touches a real database or network.
"""

from unittest.mock import Mock

import pytest

from nlp_pillars import db
from nlp_pillars.schemas import (
    RUN_DAILY_STAGES,
    PipelineRun,
    RunStatus,
    StageStatus,
)

PILLAR = "llm-theory-practice"
RUN_ID = "11111111-2222-3333-4444-555555555555"


def _run_row(**overrides):
    row = {
        "id": RUN_ID,
        "pillar_id": PILLAR,
        "trigger_source": "ui_pipeline",
        "kind": "run_daily",
        "status": RunStatus.PENDING.value,
        "current_stage": None,
        "papers_processed": 0,
        "papers_failed": 0,
        "error": None,
        "created_at": None,
        "started_at": None,
        "finished_at": None,
        "heartbeat_at": None,
    }
    row.update(overrides)
    return row


def _stage_rows(names):
    return [
        {
            "id": f"stage-{i}",
            "run_id": RUN_ID,
            "seq": i + 1,
            "name": n,
            "status": StageStatus.PENDING.value,
            "detail": None,
            "started_at": None,
            "finished_at": None,
        }
        for i, n in enumerate(names)
    ]


@pytest.fixture
def mock_client():
    """Install a mock PostgREST client and remove it afterwards.

    `params` is a real dict, not a Mock attribute: the query builder is written to
    with `query.params[...] = ...` for the filters PostgREST needs but TableQuery has
    no method for (`in.(...)`), and a bare Mock is not subscriptable.
    """
    client = Mock()
    client.table.return_value.params = {}
    db.set_client(client)
    yield client
    db.set_client(None)


class TestSerializers:
    """Pure functions — call them directly, no mocking needed."""

    def test_dict_to_pipeline_run_sorts_embedded_stages(self):
        row = _run_row()
        # Deliberately out of order, as PostgREST embedding may return them.
        row["pipeline_run_stages"] = [
            {"id": "b", "run_id": RUN_ID, "seq": 3, "name": "enqueue",
             "status": "pending"},
            {"id": "a", "run_id": RUN_ID, "seq": 1, "name": "discovery",
             "status": "completed"},
            {"id": "c", "run_id": RUN_ID, "seq": 2, "name": "search",
             "status": "running"},
        ]
        run = db._dict_to_pipeline_run(row)

        assert isinstance(run, PipelineRun)
        assert [s.seq for s in run.stages] == [1, 2, 3]
        assert [s.name for s in run.stages] == ["discovery", "search", "enqueue"]

    def test_dict_to_pipeline_run_without_stages(self):
        run = db._dict_to_pipeline_run(_run_row())
        assert run.stages == []
        assert run.status == RunStatus.PENDING.value


class TestCreatePipelineRun:

    def test_creates_run_and_seeds_every_stage(self, mock_client):
        names = [s.value for s in RUN_DAILY_STAGES]
        table = mock_client.table.return_value
        table.insert.side_effect = [
            {"data": [_run_row()], "error": None},          # the run
            {"data": _stage_rows(names), "error": None},    # the stage skeleton
        ]

        run = db.create_pipeline_run(PILLAR, "ui_pipeline", "run_daily", names)

        assert run is not None
        assert run.id == RUN_ID
        assert len(run.stages) == len(names) == 11
        assert [s.name for s in run.stages] == names
        assert all(s.status == StageStatus.PENDING.value for s in run.stages)

    def test_returns_none_when_a_run_is_already_active(self, mock_client):
        """The partial unique index surfaces as 409 — not an error, just a refusal."""
        mock_client.table.return_value.insert.return_value = {
            "data": None,
            "error": {"code": 409, "message": "Resource already exists"},
        }

        assert db.create_pipeline_run(PILLAR, "ui_pipeline", "run_daily", ["x"]) is None

    def test_returns_none_on_raw_postgres_unique_violation(self, mock_client):
        """PostgREST can surface the constraint directly as 23505."""
        mock_client.table.return_value.insert.return_value = {
            "data": None,
            "error": {
                "code": "23505",
                "message": 'duplicate key value violates unique constraint '
                           '"pipeline_runs_one_active_per_pillar"',
            },
        }

        assert db.create_pipeline_run(PILLAR, "ui_pipeline", "run_daily", ["x"]) is None

    def test_survives_stage_seeding_failure(self, mock_client):
        """A run with no stage rows is degraded, not fatal — the run still exists."""
        table = mock_client.table.return_value
        table.insert.side_effect = [
            {"data": [_run_row()], "error": None},
            {"data": None, "error": {"message": "boom"}},
        ]

        run = db.create_pipeline_run(PILLAR, "ui_pipeline", "run_daily", ["discovery"])

        assert run is not None
        assert run.stages == []


class TestUpdates:

    def test_stage_update_sets_current_stage_on_the_run(self, mock_client):
        table = mock_client.table.return_value
        table.eq.return_value = table
        table.update.return_value = {"data": [], "error": None}

        assert db.update_pipeline_run_stage(RUN_ID, "summarize", "running") is True

        patches = [c.args[0] for c in table.update.call_args_list]
        stage_patch, run_patch = patches[0], patches[1]
        assert stage_patch["status"] == "running"
        assert "started_at" in stage_patch
        assert run_patch["current_stage"] == "summarize"
        assert "heartbeat_at" in run_patch

    def test_completing_a_stage_stamps_finished_at_not_started_at(self, mock_client):
        table = mock_client.table.return_value
        table.eq.return_value = table
        table.update.return_value = {"data": [], "error": None}

        db.update_pipeline_run_stage(RUN_ID, "ingest", "completed", "2 chunks")

        stage_patch = table.update.call_args_list[0].args[0]
        assert stage_patch["status"] == "completed"
        assert "finished_at" in stage_patch
        assert "started_at" not in stage_patch
        assert stage_patch["detail"] == "2 chunks"

        # A completed stage must not become current_stage.
        run_patch = table.update.call_args_list[1].args[0]
        assert "current_stage" not in run_patch

    def test_update_survives_a_raising_client(self, mock_client):
        """TableQuery.update() re-raises HTTP errors; bookkeeping must absorb them."""
        table = mock_client.table.return_value
        table.eq.return_value = table
        table.update.side_effect = RuntimeError("connection reset")

        assert db.update_pipeline_run_stage(RUN_ID, "ingest", "running") is False

    def test_finish_writes_terminal_status_and_counts(self, mock_client):
        table = mock_client.table.return_value
        table.eq.return_value = table
        table.update.return_value = {"data": [], "error": None}

        assert db.finish_pipeline_run(
            RUN_ID, RunStatus.SUCCEEDED.value, papers_processed=3, papers_failed=1
        ) is True

        patch_arg = table.update.call_args.args[0]
        assert patch_arg["status"] == RunStatus.SUCCEEDED.value
        assert patch_arg["papers_processed"] == 3
        assert patch_arg["papers_failed"] == 1
        assert "finished_at" in patch_arg

    def test_finish_truncates_a_huge_error(self, mock_client):
        table = mock_client.table.return_value
        table.eq.return_value = table
        table.update.return_value = {"data": [], "error": None}

        db.finish_pipeline_run(RUN_ID, RunStatus.FAILED.value, error="x" * 5000)

        assert len(table.update.call_args.args[0]["error"]) == 2000


class TestSweep:

    def test_sweep_marks_open_runs_interrupted(self, mock_client):
        table = mock_client.table.return_value
        table.update.return_value = {
            "data": [_run_row(), _run_row(id="other")],
            "error": None,
        }

        assert db.sweep_interrupted_runs() == 2

        assert table.params["status"] == "in.(pending,running)"
        assert table.update.call_args.args[0]["status"] == RunStatus.INTERRUPTED.value

    def test_sweep_can_be_scoped_to_one_process_trigger_sources(self, mock_client):
        """Webui must not declare a live scheduler run dead."""
        table = mock_client.table.return_value
        table.update.return_value = {"data": [], "error": None}

        db.sweep_interrupted_runs(trigger_sources=["ui_pipeline", "ui_select"])

        assert table.params["trigger_source"] == "in.(ui_pipeline,ui_select)"

    def test_sweep_returns_zero_on_failure(self, mock_client):
        mock_client.table.return_value.update.side_effect = RuntimeError("down")
        assert db.sweep_interrupted_runs() == 0


class TestActiveLookup:

    def test_active_lookup_filters_on_status_not_a_null_column(self, mock_client):
        """eq(None) renders 'eq.None' and matches nothing, so status is used."""
        table = mock_client.table.return_value
        table.select.return_value = table
        table.limit.return_value = table
        table.execute.return_value = {"data": [_run_row()], "error": None}

        run = db.get_active_pipeline_run(PILLAR)

        assert run is not None and run.id == RUN_ID
        assert table.params["status"] == "in.(pending,running)"
        assert table.params["pillar_id"] == f"eq.{PILLAR}"
        assert "finished_at" not in table.params

    def test_active_lookup_returns_none_when_nothing_running(self, mock_client):
        table = mock_client.table.return_value
        table.select.return_value = table
        table.limit.return_value = table
        table.execute.return_value = {"data": [], "error": None}

        assert db.get_active_pipeline_run(PILLAR) is None
