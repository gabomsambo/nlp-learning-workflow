"""Ingest hooks and the weekly refresh that keep the paper map's evidence fresh.

Nothing here reaches Semantic Scholar, PostgREST or a real scheduler. The
weekly trigger is checked against a fixed clock: CronTrigger is asked for its
fire times from instants this file chooses.
"""

import threading
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from apscheduler.triggers.cron import CronTrigger

from nlp_pillars import map_refresh
from nlp_pillars.map_refresh import (
    HookOutcome,
    MapBuildsReport,
    build_maps_inline,
    run_weekly_refresh,
    sync_new_paper,
)
from nlp_pillars.services.background_jobs import BackgroundJobsService
from nlp_pillars.works import LibraryPaper, WorksSyncReport
from nlp_pillars.works_store import WorksUnavailable
from webui.services import run_service
from webui.services.run_service import (
    KIND_MAP_BUILD,
    KIND_RUN_DAILY,
    RunAlreadyActiveError,
    execute_run,
)

PILLAR = "neural-architectures-language"
PAPER = LibraryPaper(id="arxiv:1706.03762", work_id=None, title="Attention")


def _report(**kw):
    return WorksSyncReport(dry_run=False, **kw)


# ------------------------------------------------------------ sync_new_paper


def test_sync_runs_the_works_sync_for_just_that_paper():
    other = LibraryPaper(id="arxiv:1810.04805", work_id="w2", title="BERT")
    sync = MagicMock(return_value=_report(reference_edges=30, citer_edges=5))
    tool, store = MagicMock(), MagicMock()

    outcome = sync_new_paper(
        PAPER.id, tool=tool, store=store, sync=sync,
        load_library=lambda: [other, PAPER], load_index=lambda: "index",
    )

    assert outcome.ok and outcome.edges == 35
    papers, index, got_tool, got_store = sync.call_args.args
    assert papers == [PAPER]
    assert (index, got_tool, got_store) == ("index", tool, store)
    assert sync.call_args.kwargs == {"dry_run": False}


def test_a_personal_document_is_skipped_without_a_request():
    sync = MagicMock()
    outcome = sync_new_paper("file:abc123", tool=MagicMock(), sync=sync)
    assert outcome.ok
    sync.assert_not_called()


def test_a_paper_missing_from_the_library_is_reported_not_raised():
    outcome = sync_new_paper(
        PAPER.id, tool=MagicMock(), sync=MagicMock(),
        load_library=lambda: [], load_index=lambda: None, store=MagicMock(),
    )
    assert not outcome.ok and "not in the library" in outcome.detail


@pytest.mark.parametrize(
    "error",
    [RuntimeError("boom"), ConnectionError("S2 down"), WorksUnavailable("031 missing")],
)
def test_hook_failure_never_propagates(error):
    outcome = sync_new_paper(
        PAPER.id, tool=MagicMock(), store=MagicMock(),
        sync=MagicMock(side_effect=error),
        load_library=lambda: [PAPER], load_index=lambda: None,
    )
    assert not outcome.ok
    assert outcome.legacy is isinstance(error, WorksUnavailable)


def test_a_library_read_failure_is_contained():
    def broken():
        raise OSError("postgrest unreachable")

    outcome = sync_new_paper(PAPER.id, tool=MagicMock(), load_library=broken)
    assert not outcome.ok and "postgrest" in outcome.detail


def test_an_incomplete_sync_is_a_failed_outcome_with_the_reason():
    sync = MagicMock(return_value=_report(stopped_early="Semantic Scholar is rate limiting"))
    outcome = sync_new_paper(
        PAPER.id, tool=MagicMock(), store=MagicMock(), sync=sync,
        load_library=lambda: [PAPER], load_index=lambda: None,
    )
    assert not outcome.ok and "rate limiting" in outcome.detail


def test_a_paper_semantic_scholar_does_not_know_is_not_a_failure():
    sync = MagicMock(return_value=_report(not_in_s2=[PAPER.id]))
    outcome = sync_new_paper(
        PAPER.id, tool=MagicMock(), store=MagicMock(), sync=sync,
        load_library=lambda: [PAPER], load_index=lambda: None,
    )
    assert outcome.ok


# --- the three ingest paths call the hook and survive its failure


def test_orchestrator_hook_falls_back_to_the_old_fetch_only_before_migration_031():
    from nlp_pillars.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch.semantic_scholar_tool = MagicMock()
    with patch("nlp_pillars.map_refresh.sync_new_paper") as hook, \
         patch("nlp_pillars.citations.fetch_and_store_citations", return_value=4) as old:
        hook.return_value = HookOutcome(True, edges=9)
        assert orch._fetch_and_store_citations("p") == 9
        old.assert_not_called()

        hook.return_value = HookOutcome(False, "S2 down")
        assert orch._fetch_and_store_citations("p") == 0
        old.assert_not_called()

        hook.return_value = HookOutcome(False, "no tables", legacy=True)
        assert orch._fetch_and_store_citations("p") == 4

        hook.side_effect = RuntimeError("unexpected")
        assert orch._fetch_and_store_citations("p") == 0


def test_upload_hook_failure_does_not_raise():
    from nlp_pillars.services.upload_service import UploadService

    service = UploadService(citation_tool=MagicMock())
    with patch("nlp_pillars.map_refresh.sync_new_paper", side_effect=RuntimeError("x")):
        service._store_citations("arxiv:1706.03762")  # must not raise


# ------------------------------------------------------ build_maps_inline


def test_inline_builds_isolate_a_failing_pillar():
    calls = []

    def job(pillar_id, on_stage, check_cancel, run_id=None):
        calls.append(pillar_id)
        if pillar_id == "b":
            raise RuntimeError("layout exploded")
        return {"pillar_id": pillar_id}

    with patch.object(map_refresh, "db") as db:
        db.create_pipeline_run.side_effect = lambda pillar, *_: SimpleNamespace(id=f"run-{pillar}")
        report = build_maps_inline(["a", "b", "c"], run_job=job)

    assert calls == ["a", "b", "c"]
    assert report.started == ["run-a", "run-c"]
    assert report.failed == ["b: layout exploded"]
    assert db.create_pipeline_run.call_args_list[0].args[:3] == ("a", "scheduler", "map_build")
    statuses = {c.args[0]: c.args[1] for c in db.finish_pipeline_run.call_args_list}
    assert statuses == {"run-a": "succeeded", "run-b": "failed", "run-c": "succeeded"}


def test_inline_build_reports_an_active_build_without_raising():
    with patch.object(map_refresh, "db") as db:
        db.create_pipeline_run.return_value = None
        report = build_maps_inline(["a"], run_job=MagicMock())
    assert report.started == [] and "already running" in report.failed[0]


# ----------------------------------------------------------- weekly refresh


def test_weekly_refresh_syncs_then_rebuilds_every_pillar():
    order = []
    sync = MagicMock(side_effect=lambda: order.append("sync") or _report())
    build = MagicMock(side_effect=lambda ids: order.append(("build", list(ids))) or MapBuildsReport())

    run_weekly_refresh(sync=sync, list_pillars=lambda: ["a", "b"], build=build)

    assert order == ["sync", ("build", ["a", "b"])]


def test_weekly_refresh_still_rebuilds_when_the_sync_fails():
    build = MagicMock(return_value=MapBuildsReport())
    run_weekly_refresh(
        sync=MagicMock(side_effect=RuntimeError("S2 down")),
        list_pillars=lambda: ["a"], build=build,
    )
    build.assert_called_once_with(["a"])


def test_weekly_refresh_survives_an_unreadable_pillar_list():
    build = MagicMock()

    def broken():
        raise OSError("db down")

    run_weekly_refresh(sync=MagicMock(return_value=_report()), list_pillars=broken, build=build)
    build.assert_not_called()


# The clock is fixed: the trigger is asked when it fires after instants chosen here.
UTC = ZoneInfo("UTC")


def _next_fire(service, after):
    job = service.scheduler.get_job("weekly_map_refresh")
    return job.trigger.get_next_fire_time(None, after)


def test_weekly_job_fires_once_a_week_at_the_configured_time():
    service = BackgroundJobsService()
    service.add_weekly_job(
        func=lambda: None, day_of_week="sun", hour=3, minute=0, tzinfo=UTC
    )
    # 2026-10-30 is a Friday.
    first = _next_fire(service, datetime(2026, 10, 30, 12, 0, tzinfo=UTC))
    assert first == datetime(2026, 11, 1, 3, 0, tzinfo=UTC)
    assert first.strftime("%A") == "Sunday"
    # Just after it fires, the next one is seven days on, not tomorrow.
    assert _next_fire(service, first + timedelta(seconds=1)) == datetime(2026, 11, 8, 3, 0, tzinfo=UTC)
    # Firing is not skipped on the minute before.
    assert _next_fire(service, datetime(2026, 11, 1, 2, 59, tzinfo=UTC)) == first


def test_weekly_job_honours_the_schedule_timezone():
    tz = ZoneInfo("America/New_York")
    service = BackgroundJobsService()
    service.add_weekly_job(func=lambda: None, day_of_week="sun", hour=3, minute=0, tzinfo=tz)
    fire = _next_fire(service, datetime(2026, 10, 30, 12, 0, tzinfo=UTC))
    assert fire.utcoffset().total_seconds() in (-4 * 3600, -5 * 3600)
    assert (fire.hour, fire.minute) == (3, 0)
    assert isinstance(service.scheduler.get_job("weekly_map_refresh").trigger, CronTrigger)


# ------------------------------------------------------ scheduler wiring


def _settings(**kw):
    base = dict(
        schedule_enabled=True, schedule_time="08:00", schedule_timezone="UTC",
        weekly_map_refresh_enabled=True, weekly_map_refresh_day="sun",
        weekly_map_refresh_time="03:00", log_level="INFO",
    )
    base.update(kw)
    return MagicMock(**base)


def _main(settings):
    from nlp_pillars import scheduler

    service = MagicMock()
    scheduler._shutdown.set()  # return as soon as main() would block
    try:
        with patch.object(scheduler, "get_settings", return_value=settings), \
             patch.object(scheduler, "get_background_jobs_service", return_value=service), \
             patch.object(scheduler.signal, "signal"):
            code = scheduler.main()
    finally:
        scheduler._shutdown.clear()
    return code, service


def test_a_disabled_scheduler_registers_nothing_not_even_the_weekly_job():
    code, service = _main(_settings(schedule_enabled=False))
    assert code == 0
    service.add_weekly_job.assert_not_called()
    service.add_daily_pillar_run_job.assert_not_called()


def test_an_enabled_scheduler_registers_the_weekly_job_unless_switched_off():
    _, service = _main(_settings())
    kwargs = service.add_weekly_job.call_args.kwargs
    assert kwargs["func"] is run_weekly_refresh
    assert (kwargs["day_of_week"], kwargs["hour"], kwargs["minute"]) == ("sun", 3, 0)

    _, service = _main(_settings(weekly_map_refresh_enabled=False))
    service.add_weekly_job.assert_not_called()


def test_a_bad_weekly_time_refuses_to_start():
    code, _ = _main(_settings(weekly_map_refresh_time="25:99"))
    assert code == 1


def test_the_daily_run_rebuilds_only_pillars_that_gained_papers():
    from nlp_pillars import scheduler

    pillars = [MagicMock(id="a"), MagicMock(id="b")]
    results = [MagicMock(success=True, papers_processed=["p"]),
               MagicMock(success=False, papers_processed=[])]
    with patch.object(scheduler, "get_settings", return_value=MagicMock(papers_per_day=1)), \
         patch.object(scheduler.db, "get_pillars", return_value=pillars), \
         patch.object(scheduler, "Orchestrator") as orch, \
         patch.object(scheduler, "build_maps_inline") as build, \
         patch.object(scheduler, "_refresh_vault"):
        orch.return_value.run_daily.side_effect = results
        scheduler.run_all_pillars()
    build.assert_called_once_with(["a"])


def test_a_failed_map_rebuild_after_the_daily_run_does_not_raise():
    from nlp_pillars import scheduler

    with patch.object(scheduler, "build_maps_inline", side_effect=RuntimeError("x")):
        scheduler._rebuild_maps(["a"])


# ------------------------------------------- webui hook: enqueue a map_build


def _drive(kind=KIND_RUN_DAILY, processed=("p",), scheduler=None):
    events = {"run-1": threading.Event()}
    result = SimpleNamespace(papers_processed=list(processed), errors=[], success=bool(processed))
    with patch.object(run_service, "db"), \
         patch.object(run_service, "Orchestrator") as orch, \
         patch.object(run_service, "dispatch_run", return_value="map-run") as dispatch:
        orch.return_value.run_daily.return_value = result
        orch.return_value.process_selected_papers.return_value = result
        execute_run("run-1", kind, PILLAR, events["run-1"], events,
                    scheduler=scheduler if scheduler is not None else MagicMock())
    return dispatch


def test_a_daily_run_that_added_papers_queues_a_map_build():
    dispatch = _drive()
    assert dispatch.call_args.args[2:] == (PILLAR, "ui_map", KIND_MAP_BUILD)


def test_a_select_run_that_added_papers_queues_a_map_build():
    assert _drive(kind="process_selected").call_count == 1


def test_a_run_that_added_nothing_queues_no_build():
    assert _drive(processed=()).call_count == 0


def test_an_upload_queues_a_map_build():
    events = {"run-1": threading.Event()}
    with patch.object(run_service, "_finish_upload", return_value=True), \
         patch.object(run_service, "db"), \
         patch.object(run_service, "dispatch_run", return_value="m") as dispatch:
        execute_run("run-1", "upload", PILLAR, events["run-1"], events, scheduler=MagicMock())
    assert dispatch.call_args.args[2:] == (PILLAR, "ui_map", KIND_MAP_BUILD)


@pytest.mark.parametrize("error", [RunAlreadyActiveError("busy"), RuntimeError("no 034")])
def test_a_failed_enqueue_never_changes_the_run_that_added_the_paper(error):
    events = {"run-1": threading.Event()}
    result = SimpleNamespace(papers_processed=["p"], errors=[], success=True)
    with patch.object(run_service, "db") as db, \
         patch.object(run_service, "Orchestrator") as orch, \
         patch.object(run_service, "dispatch_run", side_effect=error):
        orch.return_value.run_daily.return_value = result
        execute_run("run-1", KIND_RUN_DAILY, PILLAR, events["run-1"], events,
                    scheduler=MagicMock())
    assert db.finish_pipeline_run.call_args.args[1] == "succeeded"


def test_no_scheduler_means_no_enqueue():
    assert run_service._enqueue_map_build(None, {}, PILLAR) is None
