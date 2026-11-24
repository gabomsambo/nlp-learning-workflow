"""The Daily run page's API: read the settings, save them, start a run now.

The settings live in ``daily_run_settings`` (migration 039) and the scheduler
container re-reads them every minute; precedence is documented in
``nlp_pillars/daily_run.py``. Papers per pillar is ``pillars.papers_per_day``.

Every handler does its database work on a thread: ``nlp_pillars.db`` is the
synchronous client and must not run on the event loop.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from nlp_pillars import daily_run, db, spend
from nlp_pillars.schemas import PillarUpdate, RunStatus
from webui.services.run_service import (
    KIND_RUN_DAILY,
    RunAlreadyActiveError,
    dispatch_run,
)
from webui.services.spend_guard import require_spend_room

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/daily-run", tags=["daily-run"])

#: "Run now" goes through the same trigger as the Run pipeline button.
TRIGGER_RUN_NOW = "ui_pipeline"

#: The scheduler re-reads the settings every 60 s. Three missed reads means it is
#: not running (or is an image from before migration 039).
SCHEDULER_STALE_AFTER = timedelta(minutes=3)

_ACTIVE = (RunStatus.PENDING.value, RunStatus.RUNNING.value)


class DailyRunUpdate(BaseModel):
    enabled: bool
    run_time: str
    timezone: str
    pillar_ids: List[str]
    papers_per_day: Dict[str, int] = Field(default_factory=dict)


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _when(dt: Optional[datetime], tz_name: str) -> Optional[str]:
    if dt is None:
        return None
    try:
        tz = daily_run.resolve_timezone(tz_name)
    except ValueError:
        tz = timezone.utc
    return dt.astimezone(tz).strftime("%a %d %b %Y, %H:%M %Z")


def run_url(pillar_id: str, run_id: str) -> str:
    """The page that follows one pipeline run (Add papers, auto-run mode)."""
    return f"/pipeline/?pillar={pillar_id}&run={run_id}"


def _fetch_runs(run_ids: List[str]) -> Dict[str, dict]:
    if not run_ids:
        return {}
    client = db.get_client()
    response = (
        client.table("pipeline_runs")
        .select("id,pillar_id,status,papers_processed,papers_failed,error,finished_at")
        .in_("id", run_ids)
        .execute()
    )
    if response["error"]:
        raise RuntimeError(str(response["error"]))
    return {row["id"]: row for row in response["data"] or []}


def last_run_view(last_run: Optional[dict], tz_name: str) -> Optional[dict]:
    """The stored last_run with each run's live status from pipeline_runs."""
    if not last_run:
        return None
    entries = [e for e in last_run.get("runs") or [] if isinstance(e, dict)]
    ids = [e["run_id"] for e in entries if e.get("run_id")]
    lookup_error = None
    try:
        rows = _fetch_runs(ids)
    except Exception as exc:
        rows, lookup_error = {}, f"Could not read the runs: {exc}"

    runs = []
    for entry in entries:
        run_id = entry.get("run_id")
        row = rows.get(run_id) if run_id else None
        runs.append({
            "pillar_id": entry.get("pillar_id"),
            "run_id": run_id,
            "url": run_url(entry.get("pillar_id"), run_id) if run_id else None,
            "status": (row or {}).get("status") or ("skipped" if not run_id else "unknown"),
            "papers_processed": (row or {}).get("papers_processed"),
            "error": (row or {}).get("error") or entry.get("skipped"),
        })

    statuses = [r["status"] for r in runs]
    # A scheduled batch writes finished_at only after its last pillar, and runs
    # its pillars one after another, so later pillars have no row yet.
    batch_open = not last_run.get("finished_at")
    if any(s in _ACTIVE for s in statuses) or batch_open:
        outcome = "running"
    elif not runs:
        outcome = "skipped" if last_run.get("message") else "unknown"
    elif all(s == RunStatus.SUCCEEDED.value for s in statuses):
        outcome = "succeeded"
    elif any(s == RunStatus.SUCCEEDED.value for s in statuses):
        outcome = "partial"
    elif all(s == "skipped" for s in statuses):
        outcome = "skipped"
    else:
        outcome = "failed"

    return {
        "trigger": last_run.get("trigger"),
        "started_at": last_run.get("started_at"),
        "started_display": _when(_parse_iso(last_run.get("started_at")), tz_name),
        "finished_at": last_run.get("finished_at"),
        "message": last_run.get("message"),
        "outcome": outcome,
        "papers_processed": sum(r["papers_processed"] or 0 for r in runs),
        "runs": runs,
        "lookup_error": lookup_error,
    }


def build_status(now: Optional[datetime] = None) -> dict:
    """Everything the Daily run page shows. Synchronous; call on a thread."""
    now = now or datetime.now(timezone.utc)

    settings_error = None
    try:
        config = daily_run.load_config()
    except daily_run.DailyRunSettingsError as exc:
        config = daily_run.env_defaults()
        settings_error = str(exc)

    pillars_error = None
    try:
        pillars = db.get_pillars(limit=100)
    except db.PillarLookupError as exc:
        pillars, pillars_error = [], str(exc)

    pillar_rows = [
        {
            "id": p.id,
            "name": p.name,
            "papers_per_day": int(p.papers_per_day or 1),
            "included": config.includes(p.id),
        }
        for p in pillars
    ]
    chosen = [p for p in pillar_rows if p["included"]]
    estimate = daily_run.estimate_daily_cost(
        [p["papers_per_day"] for p in chosen], daily_run.measured_cost_per_paper(),
    )

    try:
        budget = spend.current_budget(now).as_dict()
        budget["error"] = None
    except spend.SpendUnavailableError as exc:
        budget = {"cap_usd": spend.daily_cap_usd(), "spent_usd": None, "error": str(exc)}

    next_run = config.next_run_at(now)
    schedule_error = None
    if config.enabled:
        try:
            config.schedule()
        except ValueError as exc:
            schedule_error = str(exc)

    seen = _parse_iso(config.scheduler_seen_at)
    return {
        "config": config.as_dict(),
        "settings_error": settings_error,
        "schedule_error": schedule_error,
        "migration_file": daily_run.MIGRATION_FILE,
        "pillars": pillar_rows,
        "pillars_error": pillars_error,
        "next_run_at": next_run.isoformat() if next_run else None,
        "next_run_display": _when(next_run, config.timezone),
        "last_run": last_run_view(config.last_run, config.timezone),
        "estimate": estimate,
        "budget": budget,
        "estimate_over_cap": budget.get("cap_usd") is not None
        and estimate["total_usd"] > float(budget["cap_usd"]),
        "scheduler": {
            "seen_at": config.scheduler_seen_at,
            "seen_display": _when(seen, config.timezone),
            "stale": seen is None or now - seen > SCHEDULER_STALE_AFTER,
        },
    }


@router.get("")
async def get_daily_run() -> dict:
    return await asyncio.to_thread(build_status)


def _save(body: DailyRunUpdate) -> dict:
    try:
        pillars = db.get_pillars(limit=100)
    except db.PillarLookupError as exc:
        raise HTTPException(status_code=503, detail=f"Could not load pillars: {exc}") from exc
    current = {p.id: int(p.papers_per_day or 1) for p in pillars}

    try:
        fields = daily_run.validate(
            enabled=body.enabled,
            run_time=body.run_time,
            timezone_name=body.timezone,
            pillar_ids=body.pillar_ids,
            known_pillar_ids=list(current),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    papers: Dict[str, int] = {}
    for pid, value in body.papers_per_day.items():
        if pid not in current:
            raise HTTPException(status_code=422, detail=f"Unknown pillar: {pid}")
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10:
            raise HTTPException(
                status_code=422,
                detail=f"Papers per day for {pid} must be a whole number from 1 to 10.",
            )
        if value != current[pid]:
            papers[pid] = value

    try:
        daily_run.save_config(fields)
    except daily_run.DailyRunSettingsError as exc:  # includes the missing migration
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    failed = [
        pid for pid, value in papers.items()
        if db.update_pillar(pid, PillarUpdate(papers_per_day=value)) is None
    ]
    if failed:
        raise HTTPException(
            status_code=502,
            detail=(
                "Settings saved, but papers per day could not be updated for: "
                + ", ".join(failed)
            ),
        )
    return build_status()


@router.put("")
async def save_daily_run(body: DailyRunUpdate) -> dict:
    return await asyncio.to_thread(_save, body)


def _run_now(scheduler, cancel_events) -> dict:
    try:
        config = daily_run.load_config()
    except daily_run.DailyRunSettingsError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        pillars = db.get_pillars(limit=100)
    except db.PillarLookupError as exc:
        raise HTTPException(status_code=503, detail=f"Could not load pillars: {exc}") from exc
    chosen = daily_run.participating(config, pillars)
    if not chosen:
        raise HTTPException(status_code=400, detail="No pillars take part in the daily run.")

    started_at = datetime.now(timezone.utc).isoformat()
    runs = []
    for pillar in chosen:
        papers_limit = int(pillar.papers_per_day or 1)
        entry: dict = {"pillar_id": pillar.id, "papers_limit": papers_limit, "run_id": None}
        try:
            entry["run_id"] = dispatch_run(
                scheduler,
                cancel_events,
                pillar_id=pillar.id,
                trigger_source=TRIGGER_RUN_NOW,
                kind=KIND_RUN_DAILY,
                papers_limit=papers_limit,
            )
        except RunAlreadyActiveError:
            entry["skipped"] = "A run was already in progress for this pillar."
        except Exception as exc:
            logger.error("Run now: pillar %s not started: %s", pillar.id, exc)
            entry["skipped"] = f"Not started: {exc}"
        runs.append(entry)

    payload = {
        "trigger": daily_run.TRIGGER_RUN_NOW,
        "started_at": started_at,
        # The runs finish on their own threads; their rows carry the outcome.
        "finished_at": started_at,
        "runs": runs,
        "message": None,
    }
    daily_run.record_last_run(payload)
    return payload


@router.post("/run-now", dependencies=[Depends(require_spend_room)])
async def run_now(request: Request) -> JSONResponse:
    """Start one daily run now, for the pillars that take part.

    One ``run_daily`` pipeline run per pillar through the same path as the Run
    pipeline button, so each shows on Activity and can be followed. The spend
    guard runs first: a reached cap answers 429 and nothing starts. Works
    whether or not the scheduled run is on.
    """
    payload = await asyncio.to_thread(
        _run_now, request.app.state.scheduler, request.app.state.cancel_events,
    )
    payload["runs"] = [
        {**r, "url": run_url(r["pillar_id"], r["run_id"]) if r.get("run_id") else None}
        for r in payload["runs"]
    ]
    return JSONResponse(payload, status_code=202)
