"""Local daily-run scheduler — the `scheduler` service in docker-compose.yml.

This replaces the schedule trigger on `.github/workflows/daily.yml`, which can no
longer work: the database is Postgres + PostgREST bound to 127.0.0.1 on the
owner's machine, and GitHub's runners cannot reach it. The workflow file is kept
and re-enablable; see the comment at the top of it.

Run as a process, not as part of the web app:

    python -m nlp_pillars.scheduler

It runs two things on one APScheduler instance:

1. The daily learning run, at SCHEDULE_TIME in SCHEDULE_TIMEZONE, iterating every
   pillar in the database — the work the GitHub Action's matrix used to do.
2. The FSRS optimization and maintenance jobs already defined in
   `services/background_jobs.py`. Those were written long ago but
   `start_background_jobs()` had no callers, so they had never actually run.

KNOWN AND ACCEPTED LIMITATION: this only runs while the machine is on and the
stack is up. The jobstore is APScheduler's default in-memory one, so a run
missed while the stack was down is *skipped, not caught up* — on start the cron
trigger computes its next fire time from now. Stack down at 08:00 and up at
11:00 means no run that day; the next one is 08:00 tomorrow. There is
deliberately no wake-on-schedule, system service, or catch-up mechanism here.
"""

import logging
import signal
import sys
import threading
from types import FrameType
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import db
from .config import get_settings
from .orchestrator import Orchestrator
from .services.background_jobs import get_background_jobs_service

logger = logging.getLogger(__name__)

# Set when a signal arrives, so the main thread can stop the scheduler cleanly.
_shutdown = threading.Event()


def parse_schedule_time(value: str) -> tuple[int, int]:
    """Parse SCHEDULE_TIME ("HH:MM", 24-hour) into (hour, minute).

    Raises
    ------
        ValueError: If the value is not a valid 24-hour HH:MM time.

    """
    bad = f"SCHEDULE_TIME must be HH:MM (24-hour), got {value!r}"
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(bad)

    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(bad) from None

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"SCHEDULE_TIME out of range: {value!r}")

    return hour, minute


def resolve_timezone(name: str) -> ZoneInfo:
    """Resolve SCHEDULE_TIMEZONE (an IANA name) to a ZoneInfo.

    Raises
    ------
        ValueError: If the name is not a known IANA timezone.

    """
    try:
        return ZoneInfo(name.strip())
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ValueError(
            f"SCHEDULE_TIMEZONE is not a known IANA timezone: {name!r}"
        ) from e


def run_all_pillars() -> None:
    """Run the daily pipeline once for every pillar in the database.

    This is the scheduled job. It mirrors what the GitHub Action's matrix did,
    with one correction: the Action hardcoded P1-P5, which no longer exist —
    `create_pillars.py` seeds eight pillars with slug ids and `cli.py` validates
    against the database. So the pillar list is read from the database.

    Pillars run sequentially and independently: a pillar that raises is logged
    and the rest still run, matching the Action's `fail-fast: false`.
    """
    settings = get_settings()
    papers_limit = settings.papers_per_day

    try:
        pillars = db.get_pillars(limit=50)
    except db.PillarLookupError as e:
        # Reachable at last. get_pillars() used to swallow connection errors and
        # return [], so this clause never ran and the nightly run instead fell into
        # the "no pillars, seed them with create_pillars.py" branch below — telling
        # the operator to seed a database that was merely unreachable, and doing so
        # only at WARNING.
        logger.error(f"Daily run aborted: could not load pillars from database: {e}")
        return
    except Exception as e:
        # Kept deliberately broad below the specific clause: this is a long-lived
        # process and nothing about loading a pillar list may be allowed to kill it.
        # Covered by test_database_failure_does_not_propagate.
        logger.error(
            f"Daily run aborted: unexpected error loading pillars: {e}", exc_info=True
        )
        return

    if not pillars:
        logger.warning("Daily run: no pillars in the database, nothing to do. "
                       "Seed them with create_pillars.py.")
        return

    logger.info(
        f"Daily run starting for {len(pillars)} pillars, {papers_limit} paper(s) each"
    )

    # One Orchestrator for the whole run: building it creates the Qdrant
    # collection and the tool instances, and there is no per-pillar state on it.
    orchestrator = Orchestrator()

    succeeded = 0
    for pillar in pillars:
        try:
            result = orchestrator.run_daily(pillar.id, papers_limit=papers_limit)
            if result.success:
                succeeded += 1
            logger.info(
                f"Daily run: pillar {pillar.id} "
                f"processed={len(result.papers_processed)} "
                f"lessons={len(result.lessons_created)} "
                f"quizzes={len(result.quizzes_generated)} "
                f"errors={len(result.errors)} success={result.success} "
                f"time={result.total_time_seconds:.1f}s"
            )
        except Exception as e:
            # run_daily catches its own failures, so reaching here means
            # something outside the pipeline broke. Keep going regardless.
            logger.error(f"Daily run: pillar {pillar.id} raised: {e}")

    logger.info(f"Daily run finished: {succeeded}/{len(pillars)} pillars succeeded")


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

    if not settings.schedule_enabled:
        # Exit 0 and stay exited. The compose service uses `restart: on-failure`
        # precisely so that this is not restarted into a loop — `docker compose
        # ps` showing "Exited (0)" is the visible off state.
        logger.info("SCHEDULE_ENABLED is false — scheduler is disabled, exiting. "
                    "Set SCHEDULE_ENABLED=true in .env to enable the daily run.")
        return 0

    try:
        hour, minute = parse_schedule_time(settings.schedule_time)
        tzinfo = resolve_timezone(settings.schedule_timezone)
    except ValueError as e:
        # Fail loudly rather than falling back to a default time: a silently
        # wrong run time is worse than a container that refuses to start.
        logger.error(str(e))
        return 1

    service = get_background_jobs_service()
    service.add_daily_pillar_run_job(
        func=run_all_pillars, hour=hour, minute=minute, tzinfo=tzinfo
    )
    # Also starts the FSRS optimization and maintenance jobs, which had never
    # run anywhere before this service existed.
    service.start()

    for job in service.scheduler.get_jobs():
        logger.info(f"Scheduled job {job.id!r}: next run at {job.next_run_time}")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("Scheduler running. Waiting for scheduled jobs.")
    _shutdown.wait()

    service.stop()
    logger.info("Scheduler stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
