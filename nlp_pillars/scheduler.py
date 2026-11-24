"""Local daily-run scheduler — the `scheduler` service in docker-compose.yml.

This replaces the schedule trigger on `.github/workflows/daily.yml`, which can no
longer work: the database is Postgres + PostgREST bound to 127.0.0.1 on the
owner's machine, and GitHub's runners cannot reach it. The workflow file is kept
and re-enablable; see the comment at the top of it.

Run as a process, not as part of the web app:

    python -m nlp_pillars.scheduler

The process always stays up. What it does is decided by the Daily run page
(``/daily-run/``), stored in Postgres and re-read every minute — see
``nlp_pillars/daily_run.py`` for which value wins when the page has never been
saved. While the run is off the process idles: no daily job, and the FSRS
optimization jobs from ``services/background_jobs.py`` and the weekly map refresh
are paused or absent. Turning it on in the page takes effect within a minute,
with no rebuild or restart.

While it is on:

1. The daily learning run fires at the page's time in its timezone, for the
   pillars ticked there, with each pillar's ``papers_per_day``. Each pillar is a
   ``pipeline_runs`` row (trigger ``scheduler``), so it shows on Activity. The
   settings are read again when the job fires, and today's spend is checked
   against ``DAILY_SPEND_CAP_USD`` before every pillar.
2. The FSRS optimization and maintenance jobs run.
3. The weekly map refresh runs when WEEKLY_MAP_REFRESH_ENABLED is true.

After a daily run, and only when VAULT_EXPORT_ENABLED is true, it refreshes the
Obsidian vault at VAULT_DIR. A vault failure is logged and does not fail the run.

KNOWN AND ACCEPTED LIMITATION: this only runs while the machine is on and the
stack is up. The jobstore is APScheduler's default in-memory one, so a run
missed while the stack was down is *skipped, not caught up* — on start the cron
trigger computes its next fire time from now. Stack down at 08:00 and up at
11:00 means no run that day; the next one is 08:00 tomorrow. There is
deliberately no wake-on-schedule, system service, or catch-up mechanism here.
"""

import functools
import logging
import signal
import sys
import threading
from datetime import datetime, timezone
from types import FrameType
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from apscheduler.triggers.interval import IntervalTrigger

from . import daily_run, db, spend
from .config import get_settings
from .map_refresh import build_maps_inline, run_weekly_refresh
from .orchestrator import Orchestrator
from .run_outcome import count_failed_papers, summarise_errors, terminal_status
from .schemas import RUN_DAILY_STAGES, RunStatus, StageStatus
from .services.background_jobs import get_background_jobs_service
from .vault import resolve_vault_dir

logger = logging.getLogger(__name__)

#: How often the settings row is re-read. A change on the page takes effect
#: within this long.
CONFIG_POLL_SECONDS = 60

DAILY_JOB_ID = "daily_pillar_run"
WEEKLY_JOB_ID = "weekly_map_refresh"
SYNC_JOB_ID = "daily_run_settings_sync"

#: pipeline_runs.trigger_source for runs this process starts. Already admitted
#: by the CHECK since migration 009, and outside webui's startup sweep.
TRIGGER_SCHEDULER = "scheduler"
KIND_RUN_DAILY = "run_daily"

# Set when a signal arrives, so the main thread can stop the scheduler cleanly.
_shutdown = threading.Event()


def parse_schedule_time(value: str) -> tuple[int, int]:
    """Parse an "HH:MM" (24-hour) time into (hour, minute). Raises ValueError."""
    return daily_run.parse_run_time(value)


def resolve_timezone(name: str) -> ZoneInfo:
    """Resolve an IANA timezone name to a ZoneInfo. Raises ValueError."""
    return daily_run.resolve_timezone(name)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_writer(run_id: str) -> Callable[[str, str, Optional[str]], None]:
    """The Orchestrator's on_stage sink for one recorded run (as in webui's runs)."""

    def on_stage(name: str, status: str, detail: Optional[str] = None) -> None:
        # DROPPED is not a status a row can hold: it means "this step will not
        # happen", and the honest rendering of that is no row at all.
        if status == StageStatus.DROPPED.value:
            db.delete_pipeline_run_stage(run_id, name)
        else:
            db.update_pipeline_run_stage(run_id, name, status, detail)

    return on_stage


def _run_pillar(pillar: Any) -> tuple[dict, bool]:
    """Run one pillar as a recorded pipeline run.

    Returns the ``last_run`` entry for it and whether it added any papers.
    Never raises for a pipeline failure: that is written on the run row.
    """
    papers_limit = int(getattr(pillar, "papers_per_day", None) or 1)
    entry: dict = {"pillar_id": pillar.id, "papers_limit": papers_limit, "run_id": None}
    try:
        run = db.create_pipeline_run(
            pillar.id, TRIGGER_SCHEDULER, KIND_RUN_DAILY,
            [s.value for s in RUN_DAILY_STAGES],
        )
    except db.PipelineRunCreateError as e:
        logger.error(f"Daily run: pillar {pillar.id} not started: {e}")
        entry["skipped"] = f"Could not record a run: {e}"
        return entry, False
    if run is None:
        entry["skipped"] = "A run was already in progress for this pillar."
        return entry, False

    entry["run_id"] = run.id
    db.start_pipeline_run(run.id)
    try:
        # One Orchestrator per run, as webui does: it holds per-run state.
        orchestrator = Orchestrator(enable_quiz=True, on_stage=_stage_writer(run.id))
        result = orchestrator.run_daily(pillar.id, papers_limit=papers_limit)
    except Exception as e:
        # run_daily catches its own failures, so reaching here means something
        # outside the pipeline broke. Record it and keep going.
        logger.error(f"Daily run: pillar {pillar.id} raised: {e}", exc_info=True)
        db.finish_pipeline_run(run.id, RunStatus.FAILED.value, error=str(e))
        return entry, False
    except BaseException as e:
        # A killed worker must not leave the row 'running' (which also blocks the
        # pillar's next run through the one-active index).
        db.finish_pipeline_run(run.id, RunStatus.INTERRUPTED.value, error=str(e) or type(e).__name__)
        raise

    db.finish_pipeline_run(
        run.id,
        terminal_status(result),
        papers_processed=len(result.papers_processed),
        papers_failed=count_failed_papers(result),
        error=summarise_errors(result),
    )
    logger.info(
        f"Daily run: pillar {pillar.id} "
        f"processed={len(result.papers_processed)} "
        f"lessons={len(result.lessons_created)} "
        f"quizzes={len(result.quizzes_generated)} "
        f"errors={len(result.errors)} success={result.success} "
        f"time={result.total_time_seconds:.1f}s"
    )
    return entry, bool(result.papers_processed)


def _spend_refusal() -> Optional[str]:
    """Why no more paid work may start today, or None when there is room."""
    try:
        budget = spend.current_budget()
    except spend.SpendUnavailableError as e:
        # A spend cap that fails open is not one.
        return f"Today's spend could not be read ({e}), so nothing more was started."
    if budget.exhausted:
        return (
            f"Daily spend cap reached (${budget.spent_usd:.2f} of "
            f"${budget.cap_usd:.2f}); the rest waits for tomorrow."
        )
    return None


def run_all_pillars(
    config: Optional[daily_run.DailyRunConfig] = None,
    trigger: str = daily_run.TRIGGER_SCHEDULED,
) -> Optional[dict]:
    """Run the daily pipeline once for every pillar that takes part.

    Pillars run sequentially and independently: a pillar that fails is recorded
    and the rest still run, matching the old Action's `fail-fast: false`. The
    spend cap is checked before each pillar, because one pillar can spend enough
    to reach it. Returns the ``last_run`` payload it stored.
    """
    if config is None:
        try:
            config = daily_run.load_config()
        except daily_run.DailyRunSettingsError as e:
            logger.error(f"Daily run aborted: {e}")
            return None

    payload: dict = {
        "trigger": trigger,
        "started_at": _now_iso(),
        "finished_at": None,
        "runs": [],
        "message": None,
    }

    def finish(message: Optional[str] = None) -> dict:
        payload["finished_at"] = _now_iso()
        if message:
            payload["message"] = message
        daily_run.record_last_run(payload)
        return payload

    try:
        pillars = db.get_pillars(limit=50)
    except db.PillarLookupError as e:
        logger.error(f"Daily run aborted: could not load pillars from database: {e}")
        return finish(f"Could not load pillars: {e}")
    except Exception as e:
        # Kept deliberately broad below the specific clause: this is a long-lived
        # process and nothing about loading a pillar list may be allowed to kill it.
        logger.error(
            f"Daily run aborted: unexpected error loading pillars: {e}", exc_info=True
        )
        return finish(f"Could not load pillars: {e}")

    chosen = daily_run.participating(config, pillars)
    if not chosen:
        logger.warning("Daily run: no pillars take part, nothing to do.")
        return finish("No pillars take part in the daily run.")

    logger.info(f"Daily run starting for {len(chosen)} pillar(s)")
    daily_run.record_last_run(payload)

    pillars_with_new_papers: list[str] = []
    for index, pillar in enumerate(chosen):
        refusal = _spend_refusal()
        if refusal:
            logger.warning(f"Daily run stopped before pillar {pillar.id}: {refusal}")
            payload["runs"].extend(
                {"pillar_id": p.id, "run_id": None, "skipped": refusal}
                for p in chosen[index:]
            )
            payload["message"] = refusal
            break
        entry, added = _run_pillar(pillar)
        payload["runs"].append(entry)
        if added:
            pillars_with_new_papers.append(pillar.id)
        daily_run.record_last_run(payload)

    started = sum(1 for e in payload["runs"] if e.get("run_id"))
    logger.info(f"Daily run finished: {started}/{len(chosen)} pillars started")
    finish()
    _rebuild_maps(pillars_with_new_papers)
    _refresh_vault()
    return payload


def run_scheduled_daily(expected: Optional[tuple[int, int, str]] = None) -> None:
    """The cron job. Re-reads the settings first: they may have changed since
    the last sync, and turning the run off must stop a run that has not begun.
    """
    try:
        config = daily_run.load_config()
    except daily_run.DailyRunSettingsError as e:
        logger.error(f"Daily run skipped: {e}")
        return
    if not config.enabled:
        logger.info("Daily run skipped: it was turned off.")
        return
    if expected is not None:
        try:
            hour, minute, tz = config.schedule()
        except ValueError as e:
            logger.error(f"Daily run skipped: {e}")
            return
        if (hour, minute, str(tz)) != expected:
            logger.info("Daily run skipped: its time changed; it will run at the new time.")
            return
    run_all_pillars(config, trigger=daily_run.TRIGGER_SCHEDULED)


class DailyRunController:
    """Keeps the APScheduler jobs in step with the settings row.

    ``sync()`` runs at startup and then every CONFIG_POLL_SECONDS. It only
    touches the scheduler when something changed, and logs each transition.
    """

    def __init__(self, service: Any, weekly: Optional[tuple[str, int, int]] = None):
        self.service = service
        self.weekly = weekly
        self._applied: Optional[tuple[int, int, str]] = None
        self._state: Optional[str] = None
        self._aux_paused: Optional[bool] = None

    @property
    def scheduler(self):
        return self.service.scheduler

    def sync(self) -> Optional[daily_run.DailyRunConfig]:
        try:
            config = daily_run.load_config()
            if config.source == daily_run.SOURCE_ENV:
                # Table exists but no row yet (039 applied after this process
                # started): copy the env defaults in, as at startup.
                config = daily_run.seed_from_env()
        except daily_run.DailyRunSettingsError as e:
            # Keep whatever is scheduled. The job re-reads the settings when it
            # fires and refuses to run if they are still unreadable.
            logger.error(f"Could not read the daily-run settings: {e}")
            return None

        if config.source == daily_run.SOURCE_DATABASE:
            daily_run.touch_scheduler_seen()

        if not config.enabled:
            self._turn_off("off")
            return config
        try:
            hour, minute, tz = config.schedule()
        except ValueError as e:
            # A wrong time is worse than no run: do not guess one.
            if self._state != "invalid":
                logger.error(f"Daily run not scheduled: {e}")
            self._turn_off("invalid")
            return config

        key = (hour, minute, str(tz))
        if key != self._applied:
            self.service.add_daily_pillar_run_job(
                func=functools.partial(run_scheduled_daily, key),
                hour=hour, minute=minute, tzinfo=tz, job_id=DAILY_JOB_ID,
            )
            if self.weekly is not None:
                day, week_hour, week_minute = self.weekly
                self.service.add_weekly_job(
                    func=run_weekly_refresh, day_of_week=day, hour=week_hour,
                    minute=week_minute, tzinfo=tz, job_id=WEEKLY_JOB_ID,
                )
            self._applied = key
            job = self.scheduler.get_job(DAILY_JOB_ID)
            logger.info(
                f"Daily run is ON at {hour:02d}:{minute:02d} {tz}; "
                f"next run at {getattr(job, 'next_run_time', None)}"
            )
        self._state = "on"
        self._set_aux_paused(False)
        return config

    def _turn_off(self, state: str) -> None:
        for job_id in (DAILY_JOB_ID, WEEKLY_JOB_ID):
            if self.scheduler.get_job(job_id) is not None:
                self.scheduler.remove_job(job_id)
        self._applied = None
        if state == "off" and self._state != "off":
            logger.info("Daily run is OFF — idle. Turn it on from the Daily run page.")
        self._state = state
        self._set_aux_paused(True)

    def _set_aux_paused(self, paused: bool) -> None:
        """Pause or resume every other job (the FSRS ones) with the run."""
        if self._aux_paused == paused:
            return
        for job in self.scheduler.get_jobs():
            if job.id in (DAILY_JOB_ID, WEEKLY_JOB_ID, SYNC_JOB_ID):
                continue
            if paused:
                self.scheduler.pause_job(job.id)
            else:
                self.scheduler.resume_job(job.id)
        self._aux_paused = paused


def _rebuild_maps(pillar_ids: list[str]) -> None:
    """Rebuild the paper map of every pillar the daily run added papers to.

    Citations were fetched per paper inside the run, so the evidence is in
    place. Never raises: the papers are saved, and a map that did not rebuild
    is rebuilt by the next hook or the weekly refresh.
    """
    if not pillar_ids:
        return
    try:
        report = build_maps_inline(pillar_ids)
    except Exception:
        logger.exception("Map rebuild after the daily run failed")
        return
    logger.info(
        "Map rebuild after the daily run: %d started, %d not: %s",
        len(report.started), len(report.failed), report.failed,
    )


def _refresh_vault() -> None:
    """Rewrite the Obsidian vault after a daily run, when that job is enabled.

    ``vault_export_enabled is True`` is deliberate. Anything else — unset,
    false, or a stand-in that is merely truthy — leaves the vault alone.
    The export reads Postgres and writes files; it does not call a model.
    A failure here is logged. The daily run has already finished.
    """
    try:
        settings = get_settings()
        if settings.vault_export_enabled is not True:
            logger.info(
                "VAULT_EXPORT_ENABLED is not true — skipping vault export"
            )
            return
        out = resolve_vault_dir(str(settings.vault_dir or ""))
        from .vault_store import export_library

        report = export_library(out, dry_run=False)
    except Exception:
        logger.exception("Vault export after the daily run failed")
        return
    logger.info("%s", report.render())


def _warn_if_env_is_ignored(config: daily_run.DailyRunConfig) -> None:
    """Say so when .env still holds SCHEDULE_* values the settings row overrides."""
    if config.source != daily_run.SOURCE_DATABASE:
        return
    env = daily_run.env_defaults()
    if (env.enabled, env.run_time, env.timezone) != (
        config.enabled, config.run_time, config.timezone,
    ):
        logger.warning(
            "SCHEDULE_ENABLED/SCHEDULE_TIME/SCHEDULE_TIMEZONE in the environment "
            f"({env.enabled}, {env.run_time}, {env.timezone}) are ignored: the Daily run "
            f"page's settings ({config.enabled}, {config.run_time}, {config.timezone}) win."
        )


def _handle_signal(signum: int, _frame: Optional[FrameType]) -> None:
    """Ask the main thread to shut the scheduler down."""
    logger.info(f"Received signal {signal.Signals(signum).name}, shutting down")
    _shutdown.set()


def main() -> int:
    """Start the scheduler and block until signalled. Returns a process exit code."""
    settings = get_settings()

    # uvicorn configures logging for the webui container; nothing does here, so
    # without this every logger.info in the pipeline would be invisible.
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    weekly: Optional[tuple[str, int, int]] = None
    if settings.weekly_map_refresh_enabled:
        try:
            week_hour, week_minute = parse_schedule_time(settings.weekly_map_refresh_time)
        except ValueError as e:
            logger.error(f"WEEKLY_MAP_REFRESH_* is invalid: {e}")
            return 1
        weekly = (settings.weekly_map_refresh_day.strip().lower(), week_hour, week_minute)
    else:
        logger.info("WEEKLY_MAP_REFRESH_ENABLED is false — no weekly map refresh")

    # This process now records its runs, so a run it was killed in the middle of
    # must be closed out here — webui's sweep leaves 'scheduler' rows alone.
    db.sweep_interrupted_runs(trigger_sources=[TRIGGER_SCHEDULER])
    daily_run.close_unfinished_last_run(
        "The scheduler stopped before this run finished."
    )

    try:
        _warn_if_env_is_ignored(daily_run.seed_from_env())
    except daily_run.DailyRunMigrationMissing:
        logger.warning(
            f"{daily_run.MIGRATION_FILE} is not applied: using SCHEDULE_ENABLED / "
            "SCHEDULE_TIME / SCHEDULE_TIMEZONE until it is."
        )
    except daily_run.DailyRunSettingsError as e:
        logger.error(f"{e} Retrying every {CONFIG_POLL_SECONDS}s.")

    service = get_background_jobs_service()
    # Starts the FSRS optimization and maintenance jobs; the first sync below
    # pauses them straight away while the run is off. None fires within minutes
    # of start, so there is no window in which an "off" scheduler does work.
    service.start()

    controller = DailyRunController(service, weekly=weekly)
    controller.sync()
    service.scheduler.add_job(
        controller.sync,
        trigger=IntervalTrigger(seconds=CONFIG_POLL_SECONDS),
        id=SYNC_JOB_ID,
        name="Re-read daily-run settings",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=None,
    )

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info(
        f"Scheduler running; re-reading the daily-run settings every {CONFIG_POLL_SECONDS}s."
    )
    _shutdown.wait()

    service.stop()
    logger.info("Scheduler stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
