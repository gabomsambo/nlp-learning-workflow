"""The daily run's settings: where they live, which value wins, what it costs.

PRECEDENCE — the one place it is written down:

1. **The ``daily_run_settings`` row** (``docs/migrations/039_daily_run_settings.sql``),
   edited on the Daily run page (``/daily-run/``). When the row exists it wins.
2. **No row yet:** ``SCHEDULE_ENABLED``, ``SCHEDULE_TIME`` and ``SCHEDULE_TIMEZONE``
   from the environment, with every pillar taking part. These are first-boot
   defaults only. The scheduler copies them into the row when it starts, and so
   does the first save from the page. After that, editing them in ``.env`` does
   nothing; the scheduler logs a warning when they disagree with the row.
3. **Table missing** (039 not applied): the env values, exactly as before 039.
   The page says so and refuses to save.

Papers per pillar is always ``pillars.papers_per_day``, edited on the same page.
``PAPERS_PER_DAY`` is no longer read by the daily run.

The scheduler re-reads this every minute and again when the job fires, so a
change on the page needs no restart. Defaults are OFF: the column default is
``false`` and ``SCHEDULE_ENABLED`` defaults to false.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from . import db
from .config import get_settings

logger = logging.getLogger(__name__)

TABLE = "daily_run_settings"
MIGRATION_FILE = "docs/migrations/039_daily_run_settings.sql"
ROW_ID = 1

SOURCE_DATABASE = "database"
SOURCE_ENV = "env"
SOURCE_ENV_NO_TABLE = "env_no_table"

#: last_run["trigger"] values.
TRIGGER_SCHEDULED = "scheduled"
TRIGGER_RUN_NOW = "run_now"

#: Per-paper cost when the database has no measured run to go on: three Haiku
#: calls (summary, lesson, quiz) at the $0.0095 AGENTS.md measured for a paper
#: trimmed to the 60,000-token budget. That paper was near the ceiling, so this
#: leans high for a typical one.
DOCUMENTED_COST_PER_PAPER_USD = 0.0285
#: One discovery call per pillar per run (AGENTS.md: ~$0.00012).
DOCUMENTED_DISCOVERY_COST_USD = 0.00012
#: Recent single-paper runs averaged for the measured estimate.
_MEASURE_SAMPLE = 20


class DailyRunSettingsError(Exception):
    """The settings could not be read or written, for a reason worth showing."""


class DailyRunMigrationMissing(DailyRunSettingsError):
    """Migration 039 has not been applied, so there is no row to read or write."""

    def __init__(self) -> None:
        super().__init__(
            f"The daily-run settings table does not exist. Apply {MIGRATION_FILE} by hand."
        )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def parse_run_time(value: str) -> tuple[int, int]:
    """Parse "HH:MM" (24-hour) into (hour, minute). Raises ValueError."""
    bad = f"Run time must be HH:MM (24-hour), got {value!r}"
    parts = str(value).strip().split(":")
    if len(parts) != 2:
        raise ValueError(bad)
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(bad) from None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Run time out of range: {value!r}")
    return hour, minute


def resolve_timezone(name: str) -> ZoneInfo:
    """Resolve an IANA timezone name. Raises ValueError."""
    try:
        return ZoneInfo(str(name).strip())
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ValueError(f"Not a known IANA timezone: {name!r}") from e


def normalise_run_time(value: str) -> str:
    hour, minute = parse_run_time(value)
    return f"{hour:02d}:{minute:02d}"


# --------------------------------------------------------------------------
# The configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DailyRunConfig:
    """The daily-run settings in force, and where they came from."""

    enabled: bool
    run_time: str
    timezone: str
    #: None means every pillar in the database.
    pillar_ids: Optional[tuple[str, ...]]
    source: str
    updated_at: Optional[str] = None
    scheduler_seen_at: Optional[str] = None
    last_run: Optional[dict] = field(default=None, compare=False)

    @property
    def migration_missing(self) -> bool:
        return self.source == SOURCE_ENV_NO_TABLE

    def includes(self, pillar_id: str) -> bool:
        return self.pillar_ids is None or pillar_id in self.pillar_ids

    def schedule(self) -> tuple[int, int, ZoneInfo]:
        """(hour, minute, tzinfo) for the cron trigger. Raises ValueError."""
        hour, minute = parse_run_time(self.run_time)
        return hour, minute, resolve_timezone(self.timezone)

    def next_run_at(self, now: Optional[datetime] = None) -> Optional[datetime]:
        """When the run fires next, or None when it is off or misconfigured."""
        if not self.enabled:
            return None
        try:
            hour, minute, tz = self.schedule()
        except ValueError:
            return None
        from apscheduler.triggers.cron import CronTrigger

        now = now or datetime.now(timezone.utc)
        return CronTrigger(hour=hour, minute=minute, timezone=tz).get_next_fire_time(None, now)

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "run_time": self.run_time,
            "timezone": self.timezone,
            "pillar_ids": list(self.pillar_ids) if self.pillar_ids is not None else None,
            "source": self.source,
            "migration_missing": self.migration_missing,
            "updated_at": self.updated_at,
            "scheduler_seen_at": self.scheduler_seen_at,
        }


def env_defaults(source: str = SOURCE_ENV) -> DailyRunConfig:
    """The first-boot defaults from SCHEDULE_ENABLED / SCHEDULE_TIME / SCHEDULE_TIMEZONE."""
    settings = get_settings()
    return DailyRunConfig(
        enabled=settings.schedule_enabled is True,
        run_time=str(settings.schedule_time).strip(),
        timezone=str(settings.schedule_timezone).strip(),
        pillar_ids=None,
        source=source,
    )


def _row_to_config(row: dict) -> DailyRunConfig:
    pillar_ids = row.get("pillar_ids")
    return DailyRunConfig(
        enabled=row.get("enabled") is True,
        run_time=str(row.get("run_time") or "08:00"),
        timezone=str(row.get("timezone") or "UTC"),
        pillar_ids=tuple(pillar_ids) if pillar_ids is not None else None,
        source=SOURCE_DATABASE,
        updated_at=row.get("updated_at"),
        scheduler_seen_at=row.get("scheduler_seen_at"),
        last_run=row.get("last_run"),
    )


def _missing_table(error: Any) -> bool:
    text = str(error)
    return "PGRST205" in text or "42P01" in text or (
        TABLE in text and "does not exist" in text
    )


def _read_row() -> Optional[dict]:
    """The settings row, or None when there is none. Raises on failure."""
    try:
        client = db.get_client()
        response = client.table(TABLE).select("*").eq("id", ROW_ID).limit(1).execute()
    except Exception as exc:  # transport, missing env
        raise DailyRunSettingsError(f"Could not read the daily-run settings: {exc}") from exc
    if response["error"]:
        if _missing_table(response["error"]):
            raise DailyRunMigrationMissing()
        raise DailyRunSettingsError(
            f"Could not read the daily-run settings: {response['error']}"
        )
    rows = response["data"] or []
    return rows[0] if rows else None


def load_config() -> DailyRunConfig:
    """The settings in force, following the precedence in the module docstring.

    Raises:
        DailyRunSettingsError: the database could not be read. Callers that are
            about to spend money must treat that as "do not run".
    """
    try:
        row = _read_row()
    except DailyRunMigrationMissing:
        return env_defaults(SOURCE_ENV_NO_TABLE)
    if row is None:
        return env_defaults(SOURCE_ENV)
    return _row_to_config(row)


def _patch(fields: dict) -> list:
    """PATCH the row; returns the rows PostgREST says it changed. Raises."""
    client = db.get_client()
    try:
        response = client.table(TABLE).eq("id", ROW_ID).update(fields)
    except httpx.HTTPStatusError as exc:
        if _missing_table(exc.response.text):
            raise DailyRunMigrationMissing() from exc
        raise DailyRunSettingsError(
            f"Could not save the daily-run settings: {exc.response.text}"
        ) from exc
    except Exception as exc:
        raise DailyRunSettingsError(f"Could not save the daily-run settings: {exc}") from exc
    return response["data"] or []


def _insert(row: dict) -> Optional[list]:
    """INSERT the row. Returns None when it already exists (409). Raises."""
    client = db.get_client()
    try:
        response = client.table(TABLE).insert({"id": ROW_ID, **row})
    except Exception as exc:
        raise DailyRunSettingsError(f"Could not save the daily-run settings: {exc}") from exc
    error = response["error"]
    if error:
        if isinstance(error, dict) and error.get("code") in (409, "409", "23505"):
            return None
        if _missing_table(error):
            raise DailyRunMigrationMissing()
        raise DailyRunSettingsError(f"Could not save the daily-run settings: {error}")
    return response["data"] or []


def _config_row(config: DailyRunConfig) -> dict:
    return {
        "enabled": config.enabled,
        "run_time": config.run_time,
        "timezone": config.timezone,
        "pillar_ids": list(config.pillar_ids) if config.pillar_ids is not None else None,
    }


def seed_from_env() -> DailyRunConfig:
    """Copy the env defaults into the row if there is no row yet; return what is in force.

    A row that already exists is left alone (that is the whole precedence rule).
    Raises DailyRunMigrationMissing / DailyRunSettingsError.
    """
    row = _read_row()
    if row is not None:
        return _row_to_config(row)
    defaults = env_defaults()
    try:
        defaults.schedule()
        seeded = _config_row(defaults)
    except ValueError as exc:
        # A broken SCHEDULE_TIME must not become a broken row. Seed OFF with the
        # column defaults instead, and say why.
        logger.error("Not copying SCHEDULE_* into the daily-run settings: %s", exc)
        seeded = {"enabled": False}
    _insert(seeded)
    row = _read_row()
    return _row_to_config(row) if row else defaults


def validate(
    *,
    enabled: Any,
    run_time: Any,
    timezone_name: Any,
    pillar_ids: Optional[Iterable[str]],
    known_pillar_ids: Optional[Sequence[str]] = None,
) -> dict:
    """Check a page submission and return the normalised row fields. Raises ValueError."""
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be true or false")
    run_time = normalise_run_time(str(run_time or ""))
    tz_name = str(timezone_name or "").strip()
    resolve_timezone(tz_name)
    ids: Optional[list[str]] = None
    if pillar_ids is not None:
        ids = []
        for pid in pillar_ids:
            pid = str(pid).strip()
            if pid and pid not in ids:
                ids.append(pid)
        if known_pillar_ids is not None:
            unknown = [pid for pid in ids if pid not in known_pillar_ids]
            if unknown:
                raise ValueError(f"Unknown pillar(s): {', '.join(unknown)}")
    return {
        "enabled": enabled,
        "run_time": run_time,
        "timezone": tz_name,
        "pillar_ids": ids,
    }


def save_config(fields: dict) -> DailyRunConfig:
    """Write validated fields (from :func:`validate`) and return the stored config."""
    patch = {**fields, "updated_at": datetime.now(timezone.utc).isoformat()}
    changed = _patch(patch)
    if not changed:
        # No row yet: create it. A concurrent seed may win the race, in which
        # case the insert reports a conflict and the PATCH is simply repeated.
        if _insert(patch) is None:
            _patch(patch)
    row = _read_row()
    if row is None:
        raise DailyRunSettingsError("The daily-run settings were saved but could not be read back.")
    return _row_to_config(row)


def touch_scheduler_seen(now: Optional[datetime] = None) -> None:
    """Record that the scheduler just read the settings. Never raises."""
    try:
        _patch({"scheduler_seen_at": (now or datetime.now(timezone.utc)).isoformat()})
    except Exception as exc:
        logger.warning("Could not record the scheduler heartbeat: %s", exc)


def record_last_run(payload: dict) -> None:
    """Store the most recent daily run (see migration 039). Never raises."""
    try:
        _patch({"last_run": payload})
    except DailyRunMigrationMissing:
        logger.info("Not recording the daily run: %s is not applied", MIGRATION_FILE)
    except Exception as exc:
        logger.warning("Could not record the daily run: %s", exc)


def close_unfinished_last_run(reason: str) -> None:
    """Mark a batch this process was killed in the middle of as finished. Never raises.

    The scheduler calls it at startup, next to the sweep that marks the batch's
    open run rows interrupted; otherwise the page would show it running forever.
    """
    try:
        config = load_config()
        last = config.last_run
        if config.source != SOURCE_DATABASE or not last or last.get("finished_at"):
            return
        if last.get("trigger") != TRIGGER_SCHEDULED:
            return
        record_last_run({
            **last,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "message": reason,
        })
    except Exception as exc:
        logger.warning("Could not close out the previous daily run: %s", exc)


def participating(config: DailyRunConfig, pillars: Iterable[Any]) -> list:
    """The pillars that take part, in database order."""
    return [p for p in pillars if config.includes(p.id)]


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------

def _stage_cost(stages: Iterable[dict], skip: Iterable[str] = ("discovery",)) -> float:
    from .spend import parse_stage_costs

    skip = set(skip)
    return parse_stage_costs(
        s.get("detail") for s in stages or [] if s.get("name") not in skip
    )


def measured_cost_per_paper(limit: int = _MEASURE_SAMPLE) -> Optional[tuple[float, int]]:
    """Median recorded cost of one paper, from recent single-paper runs.

    Only runs that processed exactly one paper count: stage rows are per run and
    a second paper overwrites the first's cost (see nlp_pillars/spend.py), so a
    multi-paper run under-reports. Discovery is excluded; it is per pillar, not
    per paper. Returns (median_usd, sample_size), or None when there is nothing
    to measure or the database cannot be read.
    """
    try:
        client = db.get_client()
        query = (
            client.table("pipeline_runs")
            .select("id,pipeline_run_stages(name,detail)")
            .eq("status", "succeeded")
            .eq("papers_processed", 1)
            .order("created_at", desc=True)
            .limit(limit)
        )
        query.params["kind"] = "in.(run_daily,process_selected,upload)"
        response = query.execute()
    except Exception as exc:
        logger.info("Measured paper cost unavailable: %s", exc)
        return None
    if response["error"]:
        logger.info("Measured paper cost unavailable: %s", response["error"])
        return None
    costs = [
        c for c in (_stage_cost(row.get("pipeline_run_stages")) for row in response["data"] or [])
        if c > 0
    ]
    if not costs:
        return None
    return statistics.median(costs), len(costs)


def estimate_daily_cost(
    papers_per_pillar: Sequence[int],
    measured: Optional[tuple[float, int]] = None,
) -> dict:
    """Estimated dollars for one daily run over the given pillars."""
    if measured:
        per_paper, sample = measured
        basis = "measured"
    else:
        per_paper, sample = DOCUMENTED_COST_PER_PAPER_USD, 0
        basis = "documented"
    papers = sum(int(n) for n in papers_per_pillar)
    discovery = DOCUMENTED_DISCOVERY_COST_USD * len(papers_per_pillar)
    total = per_paper * papers + discovery
    return {
        "total_usd": round(total, 4),
        "per_paper_usd": round(per_paper, 4),
        "discovery_usd": round(discovery, 5),
        "papers": papers,
        "pillars": len(papers_per_pillar),
        "basis": basis,
        "sample_size": sample,
    }
