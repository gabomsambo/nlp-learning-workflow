"""Tests for webui.services.run_service — the seam between routes and the pipeline.

This module had no tests at all, which is how the cancellation bug survived: the only
cancel coverage lived in tests/test_orchestrator_progress.py and never exercised the
translation from a raised RunCancelledError into a recorded run status.

Fixtures are local rather than shared. There is no conftest.py in this project (see
tests/test_orchestrator_progress.py's docstring), and the basename must stay unique
across tests/ because there are no __init__.py files, so pytest's default import mode
would collide two modules with the same name.

Nothing here starts a scheduler, touches PostgREST, or builds a real Orchestrator.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from nlp_pillars.db import PipelineRunCreateError
from nlp_pillars.orchestrator import RunCancelledError
from nlp_pillars.schemas import DiscoveryCandidate, PaperRef, RunStatus, StageStatus
from webui.services import run_service
from webui.services.run_service import (
    KIND_DISCOVER,
    KIND_RUN_DAILY,
    RunAlreadyActiveError,
    _count_failed_papers,
    _summarise_errors,
    _terminal_status,
    dispatch_run,
    execute_run,
)

PILLAR = "neural-architectures-language"


class FakeResult:
    """Stands in for PipelineResult; only the fields run_service reads."""

    def __init__(self, papers=None, errors=None, success=None):
        self.papers_processed = papers or []
        self.errors = errors or []
        self.success = (
            len(self.papers_processed) > 0 if success is None else success
        )


# --------------------------------------------------------------- status mapping


def test_a_run_that_found_nothing_is_not_a_failure():
    """The ordinary "no new papers today" outcome must not be recorded as failed.

    PipelineResult.success is `len(papers_processed) > 0`, so this arrives as
    success=False with no errors. Recording it as 'failed' meant the browser rendered
    a bare red "Failed" — with no error text, because there was no error — above a
    column of green completed stages.
    """
    assert _terminal_status(FakeResult()) == RunStatus.SUCCEEDED.value


def test_a_run_whose_every_paper_died_is_a_failure():
    result = FakeResult(errors=[{"message": "boom"}])
    assert _terminal_status(result) == RunStatus.FAILED.value


def test_a_run_that_processed_nothing_because_everything_broke_is_a_failure():
    """"Nothing to do" and "nothing worked" must not look the same.

    Regression on the fix itself. _terminal_status maps an empty `errors` list to
    succeeded, and the orchestrator's search / enqueue / pop helpers each swallow
    their exceptions and degrade to an empty result. Before those helpers recorded
    anything, a run against a completely dead stack produced no papers AND no errors,
    so it was reported as succeeded and rendered "Done — no new papers to process" in
    green. A dead database looked exactly like a pillar that had caught up.
    """
    result = FakeResult(errors=[
        {"paper_id": "pipeline", "step": "search", "message": "SearXNG search failed: timeout"},
        {"paper_id": "pipeline", "step": "pop_queue", "message": "Could not read the queue: refused"},
    ])
    assert _terminal_status(result) == RunStatus.FAILED.value
    # ...and the reason reaches the user, not just the log.
    assert "Could not read the queue" in _summarise_errors(result)


def test_plumbing_failures_are_not_counted_as_failed_papers():
    """papers_failed is a count of PAPERS. A dead search backend is not a paper."""
    result = FakeResult(
        papers=["a"],
        errors=[
            {"paper_id": "pipeline", "step": "search", "message": "backend down"},
            {"paper_id": "arxiv:1", "step": "process_paper", "message": "paper died"},
        ],
    )
    assert _count_failed_papers(result) == 1


def test_a_partial_run_succeeds_but_keeps_its_error():
    """2 of 3 papers is a success with casualties, not a failure."""
    result = FakeResult(papers=["a", "b"], errors=[{"message": "c died"}])
    assert _terminal_status(result) == RunStatus.SUCCEEDED.value
    assert _summarise_errors(result) == "c died"


def test_success_semantics_are_left_alone():
    """_terminal_status must not depend on redefining PipelineResult.success.

    The CLI, the scheduler and fourteen existing tests read `success`; only the
    user-facing run status needed to learn the difference.
    """
    assert FakeResult().success is False
    assert FakeResult(papers=["a"]).success is True


# ------------------------------------------------------------- error summarising


def test_every_failed_paper_is_reported_not_just_the_first():
    """Stage rows are per-run, so the run error string is the only place all the
    per-paper failures of a multi-paper run can be seen."""
    result = FakeResult(errors=[
        {"message": "p1 died"}, {"message": "p2 died"}, {"message": "p3 died"},
    ])
    summary = _summarise_errors(result)
    assert summary.startswith("3 papers failed:")
    for msg in ("p1 died", "p2 died", "p3 died"):
        assert msg in summary


def test_a_clean_run_has_no_error_text():
    assert _summarise_errors(FakeResult(papers=["a"])) is None


def test_error_summary_is_truncated_to_fit_the_column():
    """finish_pipeline_run caps the column at 2000 chars; stay under it."""
    result = FakeResult(errors=[{"message": "x" * 500} for _ in range(20)])
    summary = _summarise_errors(result)
    assert len(summary) < 2000
    assert summary.endswith("…(truncated, see logs)")


def test_non_dict_errors_still_produce_text():
    assert _summarise_errors(FakeResult(errors=["plain string failure"])) == (
        "plain string failure"
    )


# ------------------------------------------------------------------- execute_run


def _run(kind=KIND_RUN_DAILY, **orch_behaviour):
    """Drive execute_run with a stubbed db and Orchestrator; return the db mock."""
    events = {"run-1": threading.Event()}
    with (
        patch("webui.services.run_service.db") as mock_db,
        patch("webui.services.run_service.Orchestrator") as mock_orch,
    ):
        for attr, value in orch_behaviour.items():
            setattr(mock_orch.return_value.run_daily, attr, value)
            setattr(mock_orch.return_value.process_selected_papers, attr, value)
        try:
            execute_run("run-1", kind, PILLAR, events["run-1"], events)
        except Exception:
            pass  # execute_run re-raises for EVENT_JOB_ERROR; not what we assert on
        return mock_db, events


def test_a_cancelled_run_is_recorded_as_cancelled_not_failed():
    """The user pressed Cancel; the run must not claim it crashed."""
    mock_db, _ = _run(side_effect=RunCancelledError("cancelled before stage ingest"))

    status = mock_db.finish_pipeline_run.call_args.args[1]
    assert status == RunStatus.CANCELLED.value
    assert "cancelled" in mock_db.finish_pipeline_run.call_args.kwargs["error"]


def test_a_crashed_run_is_recorded_as_failed():
    mock_db, _ = _run(side_effect=RuntimeError("everything is on fire"))

    status = mock_db.finish_pipeline_run.call_args.args[1]
    assert status == RunStatus.FAILED.value
    assert "on fire" in mock_db.finish_pipeline_run.call_args.kwargs["error"]


def test_a_run_that_found_nothing_is_recorded_as_succeeded():
    mock_db, _ = _run(return_value=FakeResult())

    status = mock_db.finish_pipeline_run.call_args.args[1]
    assert status == RunStatus.SUCCEEDED.value
    assert mock_db.finish_pipeline_run.call_args.kwargs["error"] is None


def test_the_cancel_event_is_always_cleaned_up():
    """Even on the failure path — a leaked event keeps the run id cancellable
    forever and slowly grows app.state.cancel_events."""
    _, events = _run(side_effect=RuntimeError("boom"))
    assert "run-1" not in events


# -------------------------------------------------------------------- discovery


def _candidate(paper_id="arxiv:2401.1"):
    return DiscoveryCandidate(
        paper=PaperRef(id=paper_id, title="A Paper", authors=[]),
        source="arxiv",
        relevance_score=0.9,
        citation_count=3,
        is_influential=False,
    )


def _discover(candidates=None, infra_errors=None, side_effect=None):
    """Drive execute_run for a discovery run; return the db mock."""
    events = {"run-1": threading.Event()}
    with (
        patch("webui.services.run_service.db") as mock_db,
        patch("webui.services.run_service.Orchestrator") as mock_orch,
    ):
        orch = mock_orch.return_value
        if side_effect is not None:
            orch.run_discovery_with_selection.side_effect = side_effect
        else:
            orch.run_discovery_with_selection.return_value = candidates or []
        orch.infra_errors = infra_errors or []
        try:
            execute_run("run-1", KIND_DISCOVER, PILLAR, events["run-1"], events,
                        priority_topics=["mamba"], limit=7)
        except Exception:
            pass
        return mock_db, orch


def test_a_discovery_run_stores_its_candidates_on_the_run():
    """The candidates are the whole product of the run and the request no longer
    carries them back, so they must survive on the row the browser is polling."""
    mock_db, _ = _discover(candidates=[_candidate("arxiv:1"), _candidate("arxiv:2")])

    kwargs = mock_db.finish_pipeline_run.call_args.kwargs
    assert mock_db.finish_pipeline_run.call_args.args[1] == RunStatus.SUCCEEDED.value
    assert [c["paper"]["id"] for c in kwargs["result"]["candidates"]] == [
        "arxiv:1", "arxiv:2"
    ]
    assert kwargs["result"]["sources_used"] == ["arxiv"]


def test_a_discovery_run_reports_no_processed_papers():
    """papers_processed counts papers put through the pipeline. Discovery puts none
    through it, and borrowing the column for a candidate count would claim otherwise."""
    mock_db, _ = _discover(candidates=[_candidate()])

    assert mock_db.finish_pipeline_run.call_args.kwargs["papers_processed"] == 0


def test_the_request_arguments_reach_the_orchestrator():
    _, orch = _discover(candidates=[])
    orch.run_discovery_with_selection.assert_called_once_with(
        PILLAR, priority_topics=["mamba"], limit=7
    )


def test_a_discovery_that_matched_nothing_is_a_success():
    """Same rule as an empty daily run: nothing to find is not a failure."""
    mock_db, _ = _discover(candidates=[])

    assert mock_db.finish_pipeline_run.call_args.args[1] == RunStatus.SUCCEEDED.value
    assert mock_db.finish_pipeline_run.call_args.kwargs["error"] is None


def test_a_discovery_that_found_nothing_because_everything_broke_is_a_failure():
    """The distinction this whole change exists for: an empty result that means
    "broken" must not be recorded the same way as one that means "nothing matched"."""
    mock_db, _ = _discover(candidates=[], infra_errors=[
        {"paper_id": "pipeline", "step": "discover_arxiv",
         "message": "rate-limited — try again in a few minutes"},
    ])

    assert mock_db.finish_pipeline_run.call_args.args[1] == RunStatus.FAILED.value
    assert "rate-limited" in mock_db.finish_pipeline_run.call_args.kwargs["error"]


def test_a_partly_broken_discovery_succeeds_but_keeps_its_reason():
    """Ten papers from three sources with the fourth down is a success — but the
    user is the one who decides whether it is worth running again."""
    mock_db, _ = _discover(candidates=[_candidate()], infra_errors=[
        {"paper_id": "pipeline", "step": "discover_vectors",
         "message": "vector search is switched off — QDRANT_URL is not set"},
    ])

    assert mock_db.finish_pipeline_run.call_args.args[1] == RunStatus.SUCCEEDED.value
    assert "QDRANT_URL" in mock_db.finish_pipeline_run.call_args.kwargs["error"]


def test_a_cancelled_discovery_is_recorded_as_cancelled():
    mock_db, _ = _discover(side_effect=RunCancelledError("cancelled before stage x"))

    assert mock_db.finish_pipeline_run.call_args.args[1] == RunStatus.CANCELLED.value


def test_a_dropped_stage_is_deleted_rather_than_patched():
    """'dropped' is not a status a row may hold — the CHECK constraint refuses it.
    It means the step will not happen, and the honest rendering of that is no row."""
    events = {"run-1": threading.Event()}
    with (
        patch("webui.services.run_service.db") as mock_db,
        patch("webui.services.run_service.Orchestrator") as mock_orch,
    ):
        mock_orch.return_value.infra_errors = []
        mock_orch.return_value.run_discovery_with_selection.return_value = []
        execute_run("run-1", KIND_DISCOVER, PILLAR, events["run-1"], events)
        on_stage = mock_orch.call_args.kwargs["on_stage"]

        on_stage("discover_citations", StageStatus.DROPPED.value)
        on_stage("discover_arxiv", StageStatus.COMPLETED.value, "3 found")

    mock_db.delete_pipeline_run_stage.assert_called_once_with(
        "run-1", "discover_citations"
    )
    mock_db.update_pipeline_run_stage.assert_called_once_with(
        "run-1", "discover_arxiv", StageStatus.COMPLETED.value, "3 found"
    )


# ------------------------------------------------------------------ dispatch_run


def _dispatch(create_return=None, create_side_effect=None):
    scheduler = MagicMock()
    with patch("webui.services.run_service.db") as mock_db:
        if create_side_effect is not None:
            mock_db.create_pipeline_run.side_effect = create_side_effect
        else:
            mock_db.create_pipeline_run.return_value = create_return
        run_id = dispatch_run(
            scheduler, {}, pillar_id=PILLAR,
            trigger_source="ui_pipeline", kind=KIND_RUN_DAILY, papers_limit=1,
        )
    return run_id, scheduler


def test_a_conflict_becomes_run_already_active():
    """None from create_pipeline_run means the partial unique index refused a second
    active run. That is the ONLY thing it may mean now."""
    with pytest.raises(RunAlreadyActiveError):
        _dispatch(create_return=None)


def test_discovery_seeds_its_own_stage_list():
    """Not run_daily's. The two pipelines share a table, not a shape."""
    scheduler = MagicMock()
    with patch("webui.services.run_service.db") as mock_db:
        mock_db.create_pipeline_run.return_value = MagicMock(id="run-9")
        dispatch_run(scheduler, {}, pillar_id=PILLAR,
                     trigger_source="ui_discover", kind=KIND_DISCOVER, limit=10)

    stage_names = mock_db.create_pipeline_run.call_args.args[3]
    assert stage_names[0] == "discover_context"
    assert stage_names[-1] == "discover_rank"
    assert "ingest" not in stage_names


def test_a_broken_database_does_not_masquerade_as_a_conflict():
    """Regression. Every None used to become RunAlreadyActiveError, so a missing
    table or a permission error told the user "a run is already in progress" and sent
    them hunting for a run that did not exist."""
    with pytest.raises(PipelineRunCreateError):
        _dispatch(create_side_effect=PipelineRunCreateError("permission denied"))


def test_dispatch_disables_the_misfire_grace_time():
    """APScheduler's default is ONE SECOND, and a job that reaches the executor later
    is discarded with only a WARNING — the user gets their 202 and the run sits at
    'pending' forever."""
    run_id, scheduler = _dispatch(create_return=MagicMock(id="run-1"))

    assert run_id == "run-1"
    kwargs = scheduler.add_job.call_args.kwargs
    assert kwargs["misfire_grace_time"] is None
    assert kwargs["max_instances"] == 1


def test_an_unknown_kind_is_rejected_before_a_row_is_written():
    with pytest.raises(ValueError):
        dispatch_run(MagicMock(), {}, pillar_id=PILLAR,
                     trigger_source="ui_pipeline", kind="not-a-kind")


def test_the_cancel_event_is_registered_before_the_job_is_scheduled():
    """The job must never start without a way to stop it."""
    scheduler = MagicMock()
    events = {}
    with patch("webui.services.run_service.db") as mock_db:
        mock_db.create_pipeline_run.return_value = MagicMock(id="run-1")
        dispatch_run(scheduler, events, pillar_id=PILLAR,
                     trigger_source="ui_pipeline", kind=KIND_RUN_DAILY, papers_limit=1)

    assert isinstance(events["run-1"], threading.Event)
    assert not events["run-1"].is_set()


# ----------------------------------------------------------------------- cancel


def test_request_cancel_reports_whether_it_reached_a_run():
    events = {"run-1": threading.Event()}
    assert run_service.request_cancel(events, "run-1") is True
    assert events["run-1"].is_set()
    # A run this process does not own — it may belong to the scheduler container.
    assert run_service.request_cancel(events, "run-nope") is False
