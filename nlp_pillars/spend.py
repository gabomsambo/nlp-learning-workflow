"""What the app has spent today, and whether it may spend more.

There is no ledger table. Every priced model call already writes its dollars
into the stage detail it belongs to (``model · 1,200 in · 300 out · $0.009500``,
see ``llm_client.stage_detail``), and a finished fact-check stores its own
``cost_usd``. This module adds those up for the current day and compares the
total with ``DAILY_SPEND_CAP_USD``.

Two limits worth knowing before trusting the number:

- **It is a lower bound.** Stage rows are per run, not per paper, so when a
  run processes several papers the stage's detail is overwritten by the next
  paper and only the last paper's cost survives. A fact-check that fails
  writes no row at all. The cap is checked when work is *started*, so a single
  large run can overshoot it; it stops the next one.
- **It reads the database.** If the numbers cannot be read, callers get
  :class:`SpendUnavailableError` and must refuse rather than assume zero. A
  spend cap that fails open is not one.

The day boundary is midnight in ``SPEND_CAP_TIMEZONE`` (default UTC).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import db

logger = logging.getLogger(__name__)

DEFAULT_DAILY_CAP_USD = 2.0

#: Rows read per query. A day of runs is tens of stages; this is a ceiling.
_ROW_LIMIT = 10000

_COST_RE = re.compile(r"\$(\d+(?:\.\d+)?)")


class SpendUnavailableError(Exception):
    """Today's spend could not be read, so no decision can be made on it."""


@dataclass(frozen=True)
class Budget:
    """One day's allowance and how much of it is gone."""

    cap_usd: float
    spent_usd: float
    day_start: datetime
    resets_at: datetime

    @property
    def remaining_usd(self) -> float:
        return max(0.0, round(self.cap_usd - self.spent_usd, 6))

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.cap_usd

    def as_dict(self) -> dict:
        return {
            "cap_usd": self.cap_usd,
            "spent_usd": round(self.spent_usd, 6),
            "remaining_usd": self.remaining_usd,
            "exhausted": self.exhausted,
            "day_start": self.day_start.isoformat(),
            "resets_at": self.resets_at.isoformat(),
        }


def daily_cap_usd() -> float:
    """``DAILY_SPEND_CAP_USD``, or the default when unset or unusable.

    ``0`` is a real value: it refuses every costly action.
    """
    raw = os.getenv("DAILY_SPEND_CAP_USD", "").strip()
    if not raw:
        return DEFAULT_DAILY_CAP_USD
    try:
        cap = float(raw)
    except ValueError:
        logger.warning("DAILY_SPEND_CAP_USD=%r is not a number; using %s", raw, DEFAULT_DAILY_CAP_USD)
        return DEFAULT_DAILY_CAP_USD
    if cap < 0 or cap != cap:  # negative, or NaN
        logger.warning("DAILY_SPEND_CAP_USD=%r is not usable; using %s", raw, DEFAULT_DAILY_CAP_USD)
        return DEFAULT_DAILY_CAP_USD
    return cap


def _zone() -> timezone | ZoneInfo:
    name = os.getenv("SPEND_CAP_TIMEZONE", "").strip() or "UTC"
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning("SPEND_CAP_TIMEZONE=%r is unknown; using UTC", name)
        return timezone.utc


def day_bounds(now: Optional[datetime] = None) -> tuple[datetime, datetime]:
    """Start of the current spend day and the start of the next one."""
    local = (now or datetime.now(timezone.utc)).astimezone(_zone())
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    # Wall-clock arithmetic on a naive copy keeps a DST day 23 or 25 hours long.
    nxt = (start.replace(tzinfo=None) + timedelta(days=1)).replace(tzinfo=start.tzinfo)
    return start, nxt


def parse_stage_costs(details: Iterable[Optional[str]]) -> float:
    """Add up every ``$amount`` found in stage detail lines."""
    total = 0.0
    for detail in details:
        if not detail:
            continue
        for match in _COST_RE.finditer(detail):
            total += float(match.group(1))
    return total


def _missing_table(error: object) -> bool:
    text = str(error)
    return "PGRST205" in text or "does not exist" in text


def _stage_spend(client, since_iso: str) -> float:
    query = (
        client.table("pipeline_run_stages")
        .select("detail")
        .limit(_ROW_LIMIT)
    )
    query.params["finished_at"] = f"gte.{since_iso}"
    query.params["detail"] = "like.*$*"
    response = query.execute()
    if response["error"]:
        raise SpendUnavailableError(f"could not read run stages: {response['error']}")
    return parse_stage_costs(row.get("detail") for row in response["data"] or [])


def _factcheck_spend(client, since_iso: str) -> float:
    query = (
        client.table("podcast_fact_checks")
        .select("cost_usd")
        .limit(_ROW_LIMIT)
    )
    query.params["created_at"] = f"gte.{since_iso}"
    response = query.execute()
    if response["error"]:
        if _missing_table(response["error"]):
            # Migration 022 not applied: no fact-check can have been saved.
            return 0.0
        raise SpendUnavailableError(f"could not read fact checks: {response['error']}")
    return sum(float(row.get("cost_usd") or 0) for row in response["data"] or [])


def spent_since(since: datetime) -> float:
    """Dollars recorded at or after ``since``."""
    try:
        client = db.get_client()
        since_iso = since.astimezone(timezone.utc).isoformat()
        return _stage_spend(client, since_iso) + _factcheck_spend(client, since_iso)
    except SpendUnavailableError:
        raise
    except Exception as exc:  # transport, missing env, malformed row
        raise SpendUnavailableError(f"could not read spend: {exc}") from exc


def current_budget(now: Optional[datetime] = None) -> Budget:
    """Today's cap and spend. Raises :class:`SpendUnavailableError`."""
    start, resets_at = day_bounds(now)
    return Budget(
        cap_usd=daily_cap_usd(),
        spent_usd=spent_since(start),
        day_start=start,
        resets_at=resets_at,
    )
