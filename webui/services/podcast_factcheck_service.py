"""Background fact-check of one stored podcast script.

The route answers 202 with a run id. APScheduler runs the five stages.
A failure here never rewrites the script and never fails a script run:
auto-dispatch is attempted only after that run has already been recorded
succeeded, and any error from this module is logged and swallowed there.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional

from nlp_pillars import db
from nlp_pillars.config import get_settings
from nlp_pillars.factcheck.constants import (
    KIND,
    RUNS_MIGRATION,
    TRIGGER_AUTO,
    TRIGGER_UI,
)
from nlp_pillars.factcheck.engine import execute_factcheck, sha256_text
from nlp_pillars.factcheck.errors import FactcheckCancelled, FactcheckError
from nlp_pillars.orchestrator import RunCancelledError, _first_line
from nlp_pillars.schemas import FACTCHECK_STAGES, StageName, StageStatus

logger = logging.getLogger(__name__)

KIND_PODCAST_FACTCHECK = KIND
TRIGGER_UI_PODCAST_FACTCHECK = TRIGGER_UI
TRIGGER_AUTO_PODCAST_FACTCHECK = TRIGGER_AUTO

MIGRATION_HINT = (
    "Fact-check runs are pipeline runs of kind 'podcast_factcheck'. "
    f"If this database has not had {RUNS_MIGRATION} applied, the insert "
    "fails its CHECK constraint. Apply it with: "
    "docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - "
    f"< {RUNS_MIGRATION}"
)

StageCallback = Callable[[str, str, Optional[str]], None]

_TERMINAL = (
    StageStatus.COMPLETED.value,
    StageStatus.FAILED.value,
    StageStatus.SKIPPED.value,
)


class FactcheckAlreadyRunning(Exception):
    """A fact-check for this script is already pending or running.

    ``run_id`` is the run to attach to. The route answers 409 with it.
    """

    def __init__(self, run_id: str):
        super().__init__(run_id)
        self.run_id = run_id


class _Stages:
    def __init__(self, on_stage: Optional[StageCallback]) -> None:
        self._on_stage = on_stage
        self._closed: set = set()
        self._failed: set = set()

    def note(self, name: str, status: str, detail: Optional[str] = None) -> None:
        stage = _as_stage(name)
        if stage is not None and status in _TERMINAL:
            self._closed.add(stage)
            if status == StageStatus.FAILED.value:
                self._failed.add(stage)
        self._forward(name, status, detail)

    def failed(self, stage: StageName, detail: str) -> None:
        self._closed.add(stage)
        self._failed.add(stage)
        self._forward(stage.value, StageStatus.FAILED.value, detail)

    def skipped(self, stage: StageName, detail: str) -> None:
        if stage in self._closed:
            return
        self._closed.add(stage)
        self._forward(stage.value, StageStatus.SKIPPED.value, detail)

    def fail_if_open(self, reason: str) -> None:
        if self._failed:
            return
        for stage in FACTCHECK_STAGES:
            if stage not in self._closed:
                self.failed(stage, reason)
                return

    def stop_after_current(self, reason: str) -> None:
        for stage in FACTCHECK_STAGES:
            self.skipped(stage, reason)

    def _forward(self, name: str, status: str, detail: Optional[str]) -> None:
        if self._on_stage is None:
            return
        try:
            self._on_stage(name, status, detail)
        except Exception as exc:  # noqa: BLE001 — never lose the run for UI
            logger.warning(
                "factcheck on_stage(%s, %s) raised: %s", name, status, exc,
            )


def _as_stage(name: str) -> Optional[StageName]:
    try:
        return StageName(name)
    except ValueError:
        return None


def result_payload(result) -> Dict[str, Any]:
    """Build the small payload stored on the run row.

    Claims stay in podcast_claims. The poll reads counts, not the quotes.
    """
    return {
        "script_id": result.script_id,
        "paper_id": result.paper_id,
        "fact_check_id": result.fact_check_id,
        "counts": result.counts,
        "cost_usd": result.cost_usd,
        "prompt_version": result.prompt_version,
    }


def _json_time(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _claim_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "line_index": row.get("line_index"),
        "span_start": row.get("span_start"),
        "span_end": row.get("span_end"),
        "span_text": row.get("span_text") or "",
        "claim": row.get("claim") or "",
        "scope": row.get("scope"),
        "category": row.get("category"),
        "verify_marker": bool(row.get("verify_marker")),
        "verdict": row.get("verdict"),
        "error_type": row.get("error_type"),
        "severity": row.get("severity"),
        "rationale": row.get("rationale") or "",
        "source_says": row.get("source_says") or "",
        "evidence": row.get("evidence") or [],
        "evidence_status": row.get("evidence_status"),
        "judge_trail": row.get("judge_trail") or [],
        "dismissed_at": _json_time(row.get("dismissed_at")),
        "dismissed_note": row.get("dismissed_note"),
    }


def script_sha256(script_text: str) -> str:
    """Return the hash the fact-check engine stored for a script's text.

    A fact check is only ever about the text it was run against, and
    ``podcast_fact_checks.script_sha256`` is how a stored check is matched to
    the script it graded. The view needs the same hash for the script it is
    showing, to tell a fresh check from a stale one.

    The browser cannot compute it: ``crypto.subtle`` only exists in a secure
    context, so a plain-HTTP host — the usual way to reach a self-hosted stack
    from another machine — could never compare anything and would have to hide
    every highlight instead.
    """
    return sha256_text(script_text or "")


def public_factcheck(row: Dict[str, Any]) -> Dict[str, Any]:
    """Latest stored check: claims, verdicts, quotes, anchors, counts."""
    cost = row.get("cost_usd")
    try:
        cost_usd = float(cost or 0)
    except (TypeError, ValueError):
        cost_usd = 0.0
    claims = [_claim_payload(claim) for claim in (row.get("claims") or [])]
    return {
        "fact_check_id": row.get("id"),
        "script_id": row.get("script_id"),
        "script_sha256": row.get("script_sha256"),
        "paper_text_sha256": row.get("paper_text_sha256"),
        "status": row.get("status"),
        "models": row.get("models") or {},
        "prompt_version": row.get("prompt_version"),
        "counts": row.get("counts") or {},
        "cost_usd": cost_usd,
        "created_at": _json_time(row.get("created_at")),
        "claims": claims,
    }


def run_factcheck_job(
    script_id: str,
    on_stage: StageCallback,
    cancel: threading.Event,
) -> Dict[str, Any]:
    """Check one script. Raises on failure. Does not touch the script row."""
    stages = _Stages(on_stage)
    try:
        result = execute_factcheck(
            script_id, on_stage=stages.note, cancel=cancel,
        )
    except FactcheckCancelled as exc:
        stages.stop_after_current("cancelled")
        raise RunCancelledError(str(exc)) from exc
    except FactcheckError as exc:
        reason = _first_line(exc)
        stages.fail_if_open(reason)
        stages.stop_after_current(reason)
        raise
    except RunCancelledError:
        stages.stop_after_current("cancelled")
        raise
    except BaseException as exc:
        reason = _first_line(exc)
        stages.fail_if_open(reason)
        stages.stop_after_current(reason)
        raise
    return result_payload(result)


def dispatch_factcheck(
    scheduler,
    cancel_events,
    *,
    script,
    trigger_source: str,
) -> str:
    """Create and schedule a fact-check run for ``script``.

    Raises:
        FactcheckAlreadyRunning: a check for this script is already active.
        RunAlreadyActiveError: the per-script unique index won a race.
        db.PipelineRunCreateError: the row could not be written (023 missing).
        db.FactcheckLookupError: the active-run lookup failed.

    """
    from webui.services.run_service import dispatch_run

    existing = db.get_active_factcheck_run(script.id)
    if existing is not None:
        raise FactcheckAlreadyRunning(existing.id)
    return dispatch_run(
        scheduler,
        cancel_events,
        pillar_id=script.pillar_id,
        trigger_source=trigger_source,
        kind=KIND_PODCAST_FACTCHECK,
        script_id=script.id,
        paper_id=script.paper_id,
        initial_result={
            "script_id": script.id,
            "paper_id": script.paper_id,
        },
    )


def maybe_dispatch_after_script(
    scheduler,
    cancel_events,
    *,
    script_id: str,
    paper_id: str,
    pillar_id: str,
) -> Optional[str]:
    """Start a fact-check after a script was saved.

    Returns the run id, or None when dispatch is off, there is no scheduler,
    or the start failed. A failure is logged and does not propagate: the
    script run has already succeeded, and this must not change that.
    """
    if scheduler is None or not script_id:
        return None
    if not get_settings().factcheck_auto_dispatch:
        logger.info(
            "FACTCHECK_AUTO_DISPATCH is off; script %s was not checked",
            script_id,
        )
        return None
    try:
        existing = db.get_active_factcheck_run(script_id)
        if existing is not None:
            logger.info(
                "fact check already running for %s: %s", script_id, existing.id,
            )
            return existing.id
        from webui.services.run_service import dispatch_run

        run_id = dispatch_run(
            scheduler,
            cancel_events,
            pillar_id=pillar_id,
            trigger_source=TRIGGER_AUTO_PODCAST_FACTCHECK,
            kind=KIND_PODCAST_FACTCHECK,
            script_id=script_id,
            paper_id=paper_id,
            initial_result={"script_id": script_id, "paper_id": paper_id},
        )
        logger.info("Dispatched fact check %s for script %s", run_id, script_id)
        return run_id
    except Exception:
        logger.exception(
            "Fact-check was not started for script %s. "
            "The script run stays succeeded and the script was not changed.",
            script_id,
        )
        return None
