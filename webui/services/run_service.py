"""Dispatch pipeline work to a background thread and record its progress.

This module is the seam between the web layer and the pipeline. Routes call
:func:`dispatch_run` and get a run id back immediately; the actual work happens on an
APScheduler worker thread and reports itself into the ``pipeline_runs`` /
``pipeline_run_stages`` tables, which the browser polls.

Why APScheduler rather than the alternatives, all of which were considered:

- ``await asyncio.to_thread(...)`` keeps the event loop free but the *client* still
  waits for the whole run, which is the problem being solved.
- ``asyncio.create_task(asyncio.to_thread(...))`` returns immediately but produces an
  orphan: no handle, no supervision, and an exception that surfaces only at garbage
  collection as "Task exception was never retrieved".
- FastAPI ``BackgroundTasks`` is awaited *inside* the ASGI cycle
  (``starlette/responses.py``: ``await self.background()``), so a minutes-long task
  blocks graceful shutdown and its exceptions land after the response with nothing
  listening.

APScheduler gives a real job id, a bounded executor, and ``EVENT_JOB_ERROR`` as a
second line of defence, and this project already depends on it.

Note that nothing here can *interrupt* a running job — there is no way to kill a
thread executing synchronous Python. Cancellation is cooperative: a
:class:`threading.Event` is checked at each stage boundary inside the Orchestrator.
For the case where the process dies outright, :func:`sweep_interrupted_runs_on_startup`
marks abandoned rows ``interrupted``, because a killed process cannot record its own
death.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

from nlp_pillars import db
from nlp_pillars.orchestrator import Orchestrator, RunCancelledError
from nlp_pillars.schemas import (
    PROCESS_SELECTED_STAGES,
    RUN_DAILY_STAGES,
    PaperRef,
    RunStatus,
)

logger = logging.getLogger(__name__)

KIND_RUN_DAILY = "run_daily"
KIND_PROCESS_SELECTED = "process_selected"

_STAGES_FOR_KIND = {
    KIND_RUN_DAILY: RUN_DAILY_STAGES,
    KIND_PROCESS_SELECTED: PROCESS_SELECTED_STAGES,
}


class RunAlreadyActiveError(Exception):
    """A run is already pending or running for this pillar."""


def dispatch_run(
    scheduler,
    cancel_events: Dict[str, threading.Event],
    pillar_id: str,
    trigger_source: str,
    kind: str,
    **kwargs: Any,
) -> str:
    """Create a run row and schedule the work. Returns the run id immediately.

    Args:
        scheduler: The APScheduler instance from ``app.state.scheduler``.
        cancel_events: The ``app.state.cancel_events`` registry, keyed by run id.
        pillar_id: Target pillar slug.
        trigger_source: 'ui_pipeline' | 'ui_select' | 'scheduler'.
        kind: KIND_RUN_DAILY or KIND_PROCESS_SELECTED.
        **kwargs: ``papers_limit`` for run_daily, ``papers`` for process_selected.

    Raises:
        RunAlreadyActiveError: if this pillar already has a pending/running run. The
            database enforces this with a partial unique index, so it is reported
            rather than guessed at — a check-then-insert would race.
        ValueError: on an unknown ``kind``.

    """
    if kind not in _STAGES_FOR_KIND:
        raise ValueError(f"Unknown run kind: {kind}")

    stage_names = [s.value for s in _STAGES_FOR_KIND[kind]]
    run = db.create_pipeline_run(pillar_id, trigger_source, kind, stage_names)
    if run is None:
        raise RunAlreadyActiveError(
            f"A pipeline run is already in progress for pillar {pillar_id}"
        )

    cancel_event = threading.Event()
    cancel_events[run.id] = cancel_event

    # The run row exists BEFORE the job is scheduled, deliberately: a fast job that
    # started first could write stage updates against a parent row that does not yet
    # exist, and the client would poll a 404.
    scheduler.add_job(
        execute_run,
        args=[run.id, kind, pillar_id, cancel_event, cancel_events],
        kwargs=kwargs,
        id=f"run:{run.id}",
        name=f"Pipeline run {run.id} ({kind}, {pillar_id})",
        trigger="date",  # no run_date => fire now
        # NOT optional. APScheduler's misfire_grace_time defaults to ONE SECOND, and
        # a job that reaches the executor later than that is discarded silently, with
        # only a WARNING in a log this app does not surface. The user would get their
        # 202 and the run would sit at 'pending' forever.
        misfire_grace_time=None,
        max_instances=1,
        coalesce=True,
    )

    logger.info(f"Dispatched pipeline run {run.id} for pillar {pillar_id} ({kind})")
    return run.id


def execute_run(
    run_id: str,
    kind: str,
    pillar_id: str,
    cancel_event: threading.Event,
    cancel_events: Dict[str, threading.Event],
    **kwargs: Any,
) -> None:
    """Run the pipeline on an APScheduler worker thread, never on the event loop."""
    db.start_pipeline_run(run_id)
    logger.info(f"Run {run_id} started ({kind}, pillar {pillar_id})")

    def on_stage(name: str, status: str, detail: Optional[str] = None) -> None:
        db.update_pipeline_run_stage(run_id, name, status, detail)

    try:
        # Built HERE, inside the worker thread. Never cached on app.state and never
        # passed across the thread boundary: it holds HTTP clients and agent state
        # that belong to the run, and reusing one across runs is how state leaks.
        orchestrator = Orchestrator(
            enable_quiz=True, on_stage=on_stage, cancel=cancel_event
        )

        if kind == KIND_RUN_DAILY:
            result = orchestrator.run_daily(
                pillar_id, papers_limit=kwargs.get("papers_limit", 1)
            )
        else:
            papers = _to_paper_refs(kwargs.get("papers") or [])
            result = orchestrator.process_selected_papers(pillar_id, papers=papers)

        db.finish_pipeline_run(
            run_id,
            RunStatus.SUCCEEDED.value if result.success else RunStatus.FAILED.value,
            papers_processed=len(result.papers_processed),
            papers_failed=len(result.errors),
            error=_first_error(result),
        )
        logger.info(
            f"Run {run_id} finished: {len(result.papers_processed)} processed, "
            f"{len(result.errors)} failed"
        )

    except RunCancelledError as e:
        logger.info(f"Run {run_id} cancelled: {e}")
        db.finish_pipeline_run(run_id, RunStatus.CANCELLED.value, error=str(e))

    except BaseException as e:  # noqa: BLE001 - see below
        # BaseException, not Exception, on purpose. If the worker thread dies of
        # anything at all and this clause does not catch it, the row stays 'running'
        # forever and is indistinguishable from a live run. A stuck row is a worse
        # outcome than a slightly over-broad except.
        logger.error(f"Run {run_id} failed: {e}", exc_info=True)
        db.finish_pipeline_run(run_id, RunStatus.FAILED.value, error=str(e))
        raise  # let EVENT_JOB_ERROR see it too

    finally:
        cancel_events.pop(run_id, None)


def request_cancel(cancel_events: Dict[str, threading.Event], run_id: str) -> bool:
    """Ask a run to stop at its next stage boundary.

    Returns False if the run is not tracked in this process — it may already have
    finished, or it may belong to the scheduler container, which has its own registry.
    Cancellation is cooperative; the run does not stop instantly.
    """
    event = cancel_events.get(run_id)
    if event is None:
        return False
    event.set()
    logger.info(f"Cancellation requested for run {run_id}")
    return True


def on_job_error(event) -> None:
    """APScheduler listener: last-resort marking of a run as failed.

    ``execute_run`` already records its own failure, so this normally sees a run that
    is terminal. It exists for the cases that clause cannot reach — a job discarded
    before it ran, or a failure inside APScheduler itself.
    """
    job_id = getattr(event, "job_id", "") or ""
    if not job_id.startswith("run:"):
        return
    run_id = job_id[len("run:"):]

    run = db.get_pipeline_run(run_id)
    if run is not None and run.status in (
        RunStatus.PENDING.value,
        RunStatus.RUNNING.value,
    ):
        message = str(getattr(event, "exception", None) or "job did not run")
        logger.error(f"Job listener marking run {run_id} failed: {message}")
        db.finish_pipeline_run(run_id, RunStatus.FAILED.value, error=message)


#: Trigger sources this process owns. The scheduler container runs the same pipeline
#: against the same database, so the startup sweep must not touch its runs — see
#: db.sweep_interrupted_runs.
WEBUI_TRIGGER_SOURCES = ["ui_pipeline", "ui_select"]


def sweep_interrupted_runs_on_startup() -> int:
    """Mark runs abandoned by a previous webui process as interrupted.

    Call once at startup, before serving traffic. Scoped to this process's own
    trigger sources so a webui restart cannot declare a live scheduler run dead.
    """
    try:
        return db.sweep_interrupted_runs(trigger_sources=WEBUI_TRIGGER_SOURCES)
    except Exception as e:
        # Never let bookkeeping stop the application from booting.
        logger.error(f"Startup sweep of interrupted runs failed: {e}")
        return 0


def _to_paper_refs(papers: List[Any]) -> List[PaperRef]:
    """Accept PaperRef objects or plain dicts, since the route hands over JSON."""
    return [p if isinstance(p, PaperRef) else PaperRef(**p) for p in papers]


def _first_error(result) -> Optional[str]:
    """Surface one error message on the run row; the rest stay in the logs."""
    if not result.errors:
        return None
    first = result.errors[0]
    return first.get("message") if isinstance(first, dict) else str(first)
