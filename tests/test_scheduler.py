"""Tests for the local daily-run scheduler (nlp_pillars/scheduler.py).

Database, spend and orchestrator calls are mocked; nothing touches the network
or calls a model. The controller tests drive a real APScheduler instance that is
started paused, so jobs get fire times but never fire; fire times are checked
against a fixed clock.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from nlp_pillars import daily_run, scheduler, spend
from nlp_pillars.daily_run import DailyRunConfig
from nlp_pillars.scheduler import (
    DAILY_JOB_ID,
    SYNC_JOB_ID,
    WEEKLY_JOB_ID,
    DailyRunController,
    main,
    parse_schedule_time,
    resolve_timezone,
    run_all_pillars,
    run_scheduled_daily,
)
from nlp_pillars.services.background_jobs import BackgroundJobsService

UTC = timezone.utc
START = datetime(2026, 11, 24, tzinfo=UTC)


class TestParseScheduleTime:
    """SCHEDULE_TIME parsing."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("08:00", (8, 0)),
            ("00:00", (0, 0)),
            ("23:59", (23, 59)),
            ("7:05", (7, 5)),
            ("  08:30  ", (8, 30)),
        ],
    )
    def test_valid(self, value, expected):
        assert parse_schedule_time(value) == expected

    @pytest.mark.parametrize(
        "value",
        ["", "08", "08:00:00", "8am", "24:00", "08:60", "-1:00", "aa:bb"],
    )
    def test_invalid_raises(self, value):
        with pytest.raises(ValueError):
            parse_schedule_time(value)


class TestResolveTimezone:
    """SCHEDULE_TIMEZONE resolution."""

    def test_valid_iana_name(self):
        assert resolve_timezone("America/New_York") == ZoneInfo("America/New_York")

    def test_whitespace_is_stripped(self):
        assert resolve_timezone(" UTC ") == ZoneInfo("UTC")

    def test_unknown_zone_raises(self):
        with pytest.raises(ValueError):
            resolve_timezone("Mars/Olympus_Mons")


def _config(**kw):
    base = dict(enabled=True, run_time="08:00", timezone="UTC", pillar_ids=None,
                source=daily_run.SOURCE_DATABASE)
    base.update(kw)
    return DailyRunConfig(**base)


def _budget(spent=0.0, cap=2.0):
    return spend.Budget(cap_usd=cap, spent_usd=spent, day_start=START, resets_at=START)


def _pillar(pid, papers=2):
    return SimpleNamespace(id=pid, papers_per_day=papers)


def _result(processed=("p",), errors=()):
    return SimpleNamespace(
        success=bool(processed), papers_processed=list(processed), errors=list(errors),
        lessons_created=[], quizzes_generated=[], total_time_seconds=1.0,
    )


class _Run:
    """Patches everything run_all_pillars reaches. ``db`` and ``orch`` are mocks."""

    def __init__(self, pillars, budgets=None, results=None):
        self.pillars = pillars
        self.budgets = budgets
        self.results = results
        self.recorded = []

    def __enter__(self):
        self._patches = [
            patch.object(scheduler.db, "get_pillars", return_value=self.pillars),
            patch.object(scheduler.db, "create_pipeline_run",
                         side_effect=lambda pid, *a, **k: SimpleNamespace(id=f"run-{pid}")),
            patch.object(scheduler.db, "start_pipeline_run"),
            patch.object(scheduler.db, "finish_pipeline_run"),
            patch.object(scheduler.db, "update_pipeline_run_stage"),
            patch.object(scheduler, "Orchestrator"),
            patch.object(scheduler.spend, "current_budget",
                         side_effect=self.budgets or (lambda: _budget())),
            patch.object(scheduler.daily_run, "record_last_run",
                         side_effect=lambda payload: self.recorded.append(
                             {**payload, "runs": list(payload["runs"])})),
            patch.object(scheduler, "build_maps_inline"),
            patch.object(scheduler, "_refresh_vault"),
        ]
        mocks = [p.start() for p in self._patches]
        (_, self.create, self.start, self.finish, self.stage, self.orch, self.budget,
         _, self.build, self.vault) = mocks
        if self.results is not None:
            self.orch.return_value.run_daily.side_effect = self.results
        else:
            self.orch.return_value.run_daily.return_value = _result()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()


class TestRunAllPillars:
    """The daily run itself: which pillars, how many papers, what is recorded."""

    def test_runs_only_participating_pillars_with_their_own_papers_per_day(self):
        pillars = [_pillar("a", 3), _pillar("b", 1), _pillar("c", 2)]
        with _Run(pillars) as r:
            payload = run_all_pillars(_config(pillar_ids=("a", "c")))

        calls = r.orch.return_value.run_daily.call_args_list
        assert [(c.args[0], c.kwargs["papers_limit"]) for c in calls] == [("a", 3), ("c", 2)]
        assert [e["run_id"] for e in payload["runs"]] == ["run-a", "run-c"]
        assert payload["trigger"] == daily_run.TRIGGER_SCHEDULED
        assert payload["finished_at"]

    def test_every_pillar_is_a_recorded_scheduler_run(self):
        with _Run([_pillar("a")]) as r:
            run_all_pillars(_config())
        args = r.create.call_args.args
        assert args[:3] == ("a", "scheduler", "run_daily")
        assert args[3][0] == "discovery"
        r.start.assert_called_once_with("run-a")
        assert r.finish.call_args.args == ("run-a", "succeeded")
        assert r.finish.call_args.kwargs["papers_processed"] == 1

    def test_progress_reaches_the_stage_rows(self):
        with _Run([_pillar("a")]) as r:
            run_all_pillars(_config())
            on_stage = r.orch.call_args.kwargs["on_stage"]
            on_stage("summarize", "running", "x")
            r.stage.assert_called_with("run-a", "summarize", "running", "x")

    def test_one_failing_pillar_does_not_stop_the_rest(self):
        pillars = [_pillar("a"), _pillar("b"), _pillar("c")]
        results = [_result(), RuntimeError("pillar b exploded"), _result()]
        with _Run(pillars, results=results) as r:
            payload = run_all_pillars(_config())
        assert r.orch.return_value.run_daily.call_count == 3
        statuses = {c.args[0]: c.args[1] for c in r.finish.call_args_list}
        assert statuses == {"run-a": "succeeded", "run-b": "failed", "run-c": "succeeded"}
        assert len(payload["runs"]) == 3

    def test_the_spend_cap_is_checked_before_every_pillar(self):
        pillars = [_pillar("a"), _pillar("b"), _pillar("c")]
        budgets = iter([_budget(0.5), _budget(2.0)])
        with _Run(pillars, budgets=lambda: next(budgets)) as r:
            payload = run_all_pillars(_config())
        assert [c.args[0] for c in r.orch.return_value.run_daily.call_args_list] == ["a"]
        skipped = [e for e in payload["runs"] if e.get("skipped")]
        assert [e["pillar_id"] for e in skipped] == ["b", "c"]
        assert "cap reached" in payload["message"]

    def test_unreadable_spend_starts_nothing(self):
        def unreadable():
            raise spend.SpendUnavailableError("db down")
        with _Run([_pillar("a")], budgets=unreadable) as r:
            payload = run_all_pillars(_config())
        r.orch.assert_not_called()
        r.create.assert_not_called()
        assert "could not be read" in payload["message"]

    def test_an_already_active_pillar_is_skipped_not_doubled(self):
        with _Run([_pillar("a"), _pillar("b")]) as r:
            r.create.side_effect = [None, SimpleNamespace(id="run-b")]
            payload = run_all_pillars(_config())
        assert payload["runs"][0]["skipped"].startswith("A run was already in progress")
        assert [c.args[0] for c in r.orch.return_value.run_daily.call_args_list] == ["b"]

    def test_no_participating_pillars_is_a_recorded_no_op(self):
        with _Run([_pillar("a")]) as r:
            payload = run_all_pillars(_config(pillar_ids=()))
        r.orch.assert_not_called()
        assert payload["message"] == "No pillars take part in the daily run."
        r.vault.assert_not_called()

    def test_database_failure_does_not_propagate(self):
        with _Run([]) as r, patch.object(scheduler.db, "get_pillars", side_effect=Exception("no db")):
            payload = run_all_pillars(_config())
        r.orch.assert_not_called()
        assert "Could not load pillars" in payload["message"]

    def test_last_run_is_recorded_as_it_goes(self):
        with _Run([_pillar("a"), _pillar("b")]) as r:
            run_all_pillars(_config())
        assert r.recorded[0]["runs"] == [] and r.recorded[0]["finished_at"] is None
        assert [len(p["runs"]) for p in r.recorded] == [0, 1, 2, 2]
        assert r.recorded[-1]["finished_at"]

    def test_maps_rebuild_only_for_pillars_that_gained_papers(self):
        with _Run([_pillar("a"), _pillar("b")], results=[_result(), _result(processed=())]) as r:
            run_all_pillars(_config())
        r.build.assert_called_once_with(["a"])
        r.vault.assert_called_once()

    def test_unreadable_settings_abort_without_running(self):
        with _Run([_pillar("a")]) as r, patch.object(
            scheduler.daily_run, "load_config",
            side_effect=daily_run.DailyRunSettingsError("down"),
        ):
            assert run_all_pillars() is None
        r.orch.assert_not_called()


class TestScheduledFire:
    """The cron job re-reads the settings before it spends anything."""

    def test_turned_off_since_the_last_sync_does_not_run(self):
        with patch.object(scheduler.daily_run, "load_config", return_value=_config(enabled=False)), \
             patch.object(scheduler, "run_all_pillars") as run:
            run_scheduled_daily((8, 0, "UTC"))
        run.assert_not_called()

    def test_time_changed_since_the_last_sync_does_not_run_at_the_old_time(self):
        with patch.object(scheduler.daily_run, "load_config", return_value=_config(run_time="09:00")), \
             patch.object(scheduler, "run_all_pillars") as run:
            run_scheduled_daily((8, 0, "UTC"))
        run.assert_not_called()

    def test_unreadable_settings_do_not_run(self):
        with patch.object(scheduler.daily_run, "load_config",
                          side_effect=daily_run.DailyRunSettingsError("down")), \
             patch.object(scheduler, "run_all_pillars") as run:
            run_scheduled_daily((8, 0, "UTC"))
        run.assert_not_called()

    def test_matching_settings_run(self):
        config = _config()
        with patch.object(scheduler.daily_run, "load_config", return_value=config), \
             patch.object(scheduler, "run_all_pillars") as run:
            run_scheduled_daily((8, 0, "UTC"))
        run.assert_called_once_with(config, trigger=daily_run.TRIGGER_SCHEDULED)


@pytest.fixture
def service():
    """A real BackgroundJobsService scheduler, started paused: nothing fires."""
    svc = BackgroundJobsService.__new__(BackgroundJobsService)
    svc.scheduler = BackgroundScheduler(timezone=UTC)
    svc.is_running = False
    # Stand-ins for the FSRS jobs start() adds.
    svc.scheduler.add_job(lambda: None, CronTrigger(hour=3, minute=0), id="daily_fsrs_optimization")
    svc.scheduler.add_job(lambda: None, CronTrigger(day_of_week=6, hour=2), id="weekly_fsrs_optimization")
    svc.scheduler.start(paused=True)
    yield svc
    svc.scheduler.shutdown(wait=False)


class _Settings:
    """Feeds the controller one config per sync."""

    def __init__(self, *configs):
        self.configs = list(configs)

    def __call__(self):
        value = self.configs.pop(0) if len(self.configs) > 1 else self.configs[0]
        if isinstance(value, Exception):
            raise value
        return value


def _sync(controller, *configs):
    feed = _Settings(*configs)
    with patch.object(scheduler.daily_run, "load_config", side_effect=feed), \
         patch.object(scheduler.daily_run, "touch_scheduler_seen") as seen:
        for _ in configs:
            controller.sync()
    return seen


def _fsrs_paused(service):
    return all(
        service.scheduler.get_job(j).next_run_time is None
        for j in ("daily_fsrs_optimization", "weekly_fsrs_optimization")
    )


class TestController:
    """The scheduler honours on/off and picks up changes without a restart."""

    NOW = datetime(2026, 11, 24, 7, 0, tzinfo=UTC)

    def _next(self, service, job_id=DAILY_JOB_ID, after=None):
        job = service.scheduler.get_job(job_id)
        return job.trigger.get_next_fire_time(None, after or self.NOW)

    def test_off_registers_no_daily_job_and_pauses_the_rest(self, service):
        controller = DailyRunController(service, weekly=("sun", 3, 0))
        seen = _sync(controller, _config(enabled=False))
        assert service.scheduler.get_job(DAILY_JOB_ID) is None
        assert service.scheduler.get_job(WEEKLY_JOB_ID) is None
        assert _fsrs_paused(service)
        seen.assert_called_once()  # the page can see the scheduler is alive

    def test_on_registers_the_job_at_the_configured_time_and_zone(self, service):
        controller = DailyRunController(service, weekly=("sun", 3, 0))
        _sync(controller, _config(run_time="09:30", timezone="America/New_York"))
        nxt = self._next(service)
        assert nxt == datetime(2026, 11, 24, 14, 30, tzinfo=UTC)
        assert self._next(service, WEEKLY_JOB_ID).astimezone(ZoneInfo("America/New_York")).hour == 3
        assert not _fsrs_paused(service)

    def test_turning_on_later_takes_effect_without_a_restart(self, service):
        controller = DailyRunController(service)
        _sync(controller, _config(enabled=False), _config(enabled=True))
        assert service.scheduler.get_job(DAILY_JOB_ID) is not None
        assert not _fsrs_paused(service)

    def test_a_changed_time_reschedules(self, service):
        controller = DailyRunController(service)
        _sync(controller, _config(run_time="08:00"), _config(run_time="06:45"))
        assert self._next(service) == datetime(2026, 11, 25, 6, 45, tzinfo=UTC)
        assert len([j for j in service.scheduler.get_jobs() if j.id == DAILY_JOB_ID]) == 1

    def test_turning_off_removes_the_job_and_pauses_the_rest(self, service):
        controller = DailyRunController(service, weekly=("sun", 3, 0))
        _sync(controller, _config(), _config(enabled=False))
        assert service.scheduler.get_job(DAILY_JOB_ID) is None
        assert service.scheduler.get_job(WEEKLY_JOB_ID) is None
        assert _fsrs_paused(service)

    def test_unchanged_settings_do_not_touch_the_job(self, service):
        controller = DailyRunController(service)
        _sync(controller, _config())
        with patch.object(service, "add_daily_pillar_run_job") as add:
            _sync(controller, _config())
        add.assert_not_called()

    def test_a_bad_timezone_schedules_nothing_rather_than_guessing(self, service):
        controller = DailyRunController(service)
        _sync(controller, _config(), _config(timezone="Mars/Olympus_Mons"))
        assert service.scheduler.get_job(DAILY_JOB_ID) is None

    def test_unreadable_settings_keep_the_current_schedule(self, service):
        controller = DailyRunController(service)
        _sync(controller, _config(), daily_run.DailyRunSettingsError("db down"))
        assert service.scheduler.get_job(DAILY_JOB_ID) is not None

    def test_the_job_fires_the_fire_time_check_with_its_own_schedule(self, service):
        controller = DailyRunController(service)
        _sync(controller, _config(run_time="08:00"))
        job = service.scheduler.get_job(DAILY_JOB_ID)
        assert job.func.func is run_scheduled_daily
        assert job.func.args == ((8, 0, "UTC"),)
        assert job.misfire_grace_time is None

    def test_a_table_without_a_row_is_seeded_from_env(self, service):
        controller = DailyRunController(service)
        env_config = _config(enabled=False, source=daily_run.SOURCE_ENV)
        with patch.object(scheduler.daily_run, "load_config", return_value=env_config), \
             patch.object(scheduler.daily_run, "seed_from_env",
                          return_value=_config(enabled=False)) as seed, \
             patch.object(scheduler.daily_run, "touch_scheduler_seen"):
            controller.sync()
        seed.assert_called_once()


class TestMain:
    """Process entry point: it stays up whether the run is on or off."""

    def _settings(self, weekly_time="03:00", weekly=True):
        return MagicMock(
            weekly_map_refresh_enabled=weekly,
            weekly_map_refresh_day="sun",
            weekly_map_refresh_time=weekly_time,
            log_level="INFO",
        )

    def _main(self, settings, config):
        service = MagicMock()
        service.scheduler.get_job.return_value = None
        service.scheduler.get_jobs.return_value = []
        scheduler._shutdown.set()  # return as soon as main() would block
        try:
            with patch.object(scheduler, "get_settings", return_value=settings), \
                 patch.object(scheduler, "get_background_jobs_service", return_value=service), \
                 patch.object(scheduler.signal, "signal"), \
                 patch.object(scheduler.db, "sweep_interrupted_runs") as sweep, \
                 patch.object(scheduler.daily_run, "seed_from_env", return_value=config), \
                 patch.object(scheduler.daily_run, "close_unfinished_last_run"), \
                 patch.object(scheduler.daily_run, "load_config", return_value=config), \
                 patch.object(scheduler.daily_run, "touch_scheduler_seen"):
                code = main()
        finally:
            scheduler._shutdown.clear()
        return code, service, sweep

    def test_off_stays_up_and_idles(self):
        code, service, _ = self._main(self._settings(), _config(enabled=False))
        assert code == 0
        service.start.assert_called_once()
        service.add_daily_pillar_run_job.assert_not_called()
        service.add_weekly_job.assert_not_called()
        sync = [c for c in service.scheduler.add_job.call_args_list if c.kwargs.get("id") == SYNC_JOB_ID]
        assert len(sync) == 1
        assert sync[0].kwargs["trigger"].interval.total_seconds() == scheduler.CONFIG_POLL_SECONDS
        service.stop.assert_called_once()

    def test_on_registers_daily_and_weekly_jobs(self):
        code, service, _ = self._main(
            self._settings(), _config(run_time="08:30", timezone="America/New_York"),
        )
        assert code == 0
        kwargs = service.add_daily_pillar_run_job.call_args.kwargs
        assert (kwargs["hour"], kwargs["minute"]) == (8, 30)
        assert kwargs["tzinfo"] == ZoneInfo("America/New_York")
        weekly = service.add_weekly_job.call_args.kwargs
        assert (weekly["day_of_week"], weekly["hour"], weekly["minute"]) == ("sun", 3, 0)

    def test_sweeps_only_its_own_runs(self):
        _, _, sweep = self._main(self._settings(), _config(enabled=False))
        sweep.assert_called_once_with(trigger_sources=["scheduler"])

    def test_bad_weekly_time_exits_nonzero(self):
        code, service, _ = self._main(self._settings(weekly_time="25:99"), _config())
        assert code == 1
        service.start.assert_not_called()

    def test_missing_migration_still_starts(self):
        settings = self._settings()
        service = MagicMock()
        service.scheduler.get_job.return_value = None
        service.scheduler.get_jobs.return_value = []
        scheduler._shutdown.set()
        try:
            with patch.object(scheduler, "get_settings", return_value=settings), \
                 patch.object(scheduler, "get_background_jobs_service", return_value=service), \
                 patch.object(scheduler.signal, "signal"), \
                 patch.object(scheduler.db, "sweep_interrupted_runs"), \
                 patch.object(scheduler.daily_run, "close_unfinished_last_run"), \
                 patch.object(scheduler.daily_run, "seed_from_env",
                              side_effect=daily_run.DailyRunMigrationMissing()), \
                 patch.object(scheduler.daily_run, "load_config",
                              return_value=_config(enabled=False,
                                                   source=daily_run.SOURCE_ENV_NO_TABLE)):
                assert main() == 0
        finally:
            scheduler._shutdown.clear()
        service.start.assert_called_once()


class TestVaultRefresh:
    """The vault job is off unless VAULT_EXPORT_ENABLED is exactly true."""

    def test_default_settings_do_not_export(self):
        with patch.object(scheduler, "get_settings", return_value=MagicMock()), \
             patch("nlp_pillars.vault_store.export_library") as export:
            scheduler._refresh_vault()
        export.assert_not_called()

    def test_enabled_exports_the_configured_directory(self, tmp_path):
        settings = MagicMock(vault_export_enabled=True, vault_dir=str(tmp_path))
        with patch.object(scheduler, "get_settings", return_value=settings), \
             patch("nlp_pillars.vault_store.export_library") as export:
            scheduler._refresh_vault()
        export.assert_called_once()
        assert export.call_args.args[0] == tmp_path
        assert export.call_args.kwargs["dry_run"] is False

    def test_an_export_failure_does_not_raise(self, tmp_path):
        settings = MagicMock(vault_export_enabled=True, vault_dir=str(tmp_path))
        with patch.object(scheduler, "get_settings", return_value=settings), \
             patch("nlp_pillars.vault_store.export_library", side_effect=RuntimeError("disk full")):
            scheduler._refresh_vault()


def test_compose_keeps_the_scheduler_up():
    from pathlib import Path

    compose = (Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text()
    block = compose[compose.index("  scheduler:"):compose.index("  playground-runner:")]
    assert "restart: unless-stopped" in block
    assert "restart: on-failure" not in block
