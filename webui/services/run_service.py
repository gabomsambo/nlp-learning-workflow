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

import hashlib
import logging
import threading
from typing import Any, Dict, List, Optional

from nlp_pillars import db, map_store
from nlp_pillars.orchestrator import Orchestrator, RunCancelledError
from nlp_pillars.schemas import (
    DISCOVER_STAGES,
    FACTCHECK_STAGES,
    MAP_BUILD_STAGES,
    PLAYGROUND_ALIGN_STAGES,
    PLAYGROUND_EXPLAIN_STAGES,
    PODCAST_AUDIO_STAGES,
    PODCAST_SCRIPT_STAGES,
    PROCESS_SELECTED_STAGES,
    RUN_DAILY_STAGES,
    UPLOAD_STAGES,
    PaperRef,
    RunStatus,
    StageStatus,
    UploadFileRequest,
    UploadUrlRequest,
)
from nlp_pillars.playground import follow as playground_follow
from nlp_pillars.playground import service as playground_service
from nlp_pillars.services.upload_service import get_upload_service
from webui.services import (
    cloud_publish_service,
    podcast_audio_service,
    podcast_factcheck_service,
    podcast_script_service,
)
from webui.services.discovery_results import candidates_payload

logger = logging.getLogger(__name__)

KIND_RUN_DAILY = "run_daily"
KIND_PROCESS_SELECTED = "process_selected"
KIND_DISCOVER = "discover"
KIND_PODCAST_AUDIO = podcast_audio_service.KIND_PODCAST_AUDIO
KIND_PODCAST_SCRIPT = podcast_script_service.KIND_PODCAST_SCRIPT
KIND_PODCAST_FACTCHECK = podcast_factcheck_service.KIND_PODCAST_FACTCHECK
KIND_UPLOAD = "upload"
KIND_PLAYGROUND_EXPLAIN = playground_service.KIND_PLAYGROUND_EXPLAIN
KIND_PLAYGROUND_ALIGN = playground_follow.KIND_PLAYGROUND_ALIGN
KIND_MAP_BUILD = map_store.KIND_MAP_BUILD

#: The trigger source for both manual upload routes. Uploads are exempt from the
#: one-active-run-per-pillar guard (migration 015) — see dispatch_run.
TRIGGER_UI_UPLOAD = "ui_upload"

#: Which of an upload run's kwargs decides what it fetches. Both sources produce the
#: same stage list and the same result payload; only the first stage differs.
UPLOAD_SOURCE_URL = "url"
UPLOAD_SOURCE_FILE = "file"

_STAGES_FOR_KIND = {
    KIND_RUN_DAILY: RUN_DAILY_STAGES,
    KIND_PROCESS_SELECTED: PROCESS_SELECTED_STAGES,
    KIND_DISCOVER: DISCOVER_STAGES,
    KIND_PODCAST_AUDIO: PODCAST_AUDIO_STAGES,
    KIND_PODCAST_SCRIPT: PODCAST_SCRIPT_STAGES,
    KIND_PODCAST_FACTCHECK: FACTCHECK_STAGES,
    KIND_UPLOAD: UPLOAD_STAGES,
    KIND_PLAYGROUND_EXPLAIN: PLAYGROUND_EXPLAIN_STAGES,
    KIND_PLAYGROUND_ALIGN: PLAYGROUND_ALIGN_STAGES,
    KIND_MAP_BUILD: MAP_BUILD_STAGES,
}


class RunAlreadyActiveError(Exception):
    """A run is already pending or running for this pillar."""


class PodcastAudioQueueError(Exception):
    """A podcast audio request could not be durably queued."""


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
        trigger_source: 'ui_pipeline' | 'ui_select' | 'ui_discover' | 'ui_upload' |
            'ui_podcast_audio' | 'scheduler'.
        kind: one of the KIND_* constants above.
        **kwargs: ``papers_limit`` for run_daily, ``papers`` for process_selected,
            ``priority_topics`` and ``limit`` for discover, ``script_id`` and
            ``voice_path`` for podcast_audio, ``source`` plus that source's fields
            for upload (see :func:`_finish_upload`).

    Raises:
        RunAlreadyActiveError: if this pillar already has a pending/running run of a
            kind the guard covers. The database enforces this with a partial unique
            index, so it is reported rather than guessed at — a check-then-insert
            would race.

            KIND_UPLOAD is deliberately OUTSIDE that guard: migration 015 narrows the
            index to ``kind <> 'upload'``, so an upload is never refused because a
            discovery run is in flight on the same pillar. The guard is about two
            writers driving one pillar's queue at once; an upload is scoped to the one
            paper the user handed over. It can therefore still raise here, but only if
            015 has not been applied — which is also the case where the insert fails
            the kind CHECK first, so in practice this is unreachable for uploads.

            KIND_PODCAST_SCRIPT is exempted the same way (migration 016): generation
            is scoped to one paper, and must not 409 against discovery or TTS.

            KIND_PODCAST_FACTCHECK is exempted the same way (migration 023). One
            script still has at most one active fact-check; that conflict is a
            separate index and the route answers 409 with the run to attach to.

            KIND_PLAYGROUND_EXPLAIN is exempted the same way (migration 028), with
            one active pass per playground enforced by its own index.

            KIND_PLAYGROUND_ALIGN is exempted the same way (migration 029), with
            one active pass per (episode, playground) enforced by its own index.

            KIND_MAP_BUILD is exempted the same way (migration 034): it reads the
            pillar and writes only its snapshot. One pillar still has at most one
            active map build, enforced by its own index.
        ValueError: on an unknown ``kind``.
        db.PipelineRunCreateError: the row could not be written for any other reason
            — including the CHECK constraints migration 015 widens.

    """
    if kind not in _STAGES_FOR_KIND:
        raise ValueError(f"Unknown run kind: {kind}")

    initial_result = kwargs.pop("initial_result", None)
    stage_names = [s.value for s in _STAGES_FOR_KIND[kind]]
    run = db.create_pipeline_run(
        pillar_id, trigger_source, kind, stage_names, result=initial_result,
    )
    if run is None:
        raise RunAlreadyActiveError(
            f"A pipeline run is already in progress for pillar {pillar_id}"
        )

    cancel_event = threading.Event()
    cancel_events[run.id] = cancel_event

    # The run row exists BEFORE the job is scheduled, deliberately: a fast job that
    # started first could write stage updates against a parent row that does not yet
    # exist, and the client would poll a 404.
    job_kwargs = dict(kwargs)
    # The script job starts a fact-check after it succeeds. The scheduler is
    # not stored on the run row; it only rides along on this in-memory job.
    job_kwargs["scheduler"] = scheduler
    scheduler.add_job(
        execute_run,
        args=[run.id, kind, pillar_id, cancel_event, cancel_events],
        kwargs=job_kwargs,
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


def queue_podcast_audio_run(
    pillar_id: str,
    script_id: str,
    voice_path: str,
    script_sha256: str,
    consent: bool,
    trigger_source: str = podcast_audio_service.TRIGGER_UI_PODCAST_AUDIO,
) -> tuple[str, bool]:
    """Persist a podcast audio request without requiring IndexTTS to be online.

    The second return value is true only for a newly-created run. A request for a
    script that already has an active run attaches to that durable run instead.
    """
    if consent is not True:
        raise ValueError("Podcast audio requires explicit local-GPU consent")
    request_result = {
        "script_id": script_id,
        "voice_path": voice_path,
        "script_sha256": script_sha256,
        "consent": True,
        "queue_state": "waiting_for_indextts",
        "queue_message": "Waiting for IndexTTS readiness check.",
    }
    try:
        run = db.create_pipeline_run(
            pillar_id,
            trigger_source,
            KIND_PODCAST_AUDIO,
            [stage.value for stage in PODCAST_AUDIO_STAGES],
            result=request_result,
        )
    except db.PipelineRunCreateError as exc:
        raise PodcastAudioQueueError(
            f"{exc}. Apply docs/migrations/037_podcast_audio_queue.sql."
        ) from exc

    if run is not None:
        logger.info(
            "Queued podcast audio run %s for script %s", run.id, script_id,
        )
        return run.id, True

    try:
        active = db.get_active_podcast_audio_run(script_id)
    except db.PodcastAudioQueueLookupError as exc:
        raise PodcastAudioQueueError(str(exc)) from exc
    if active is not None:
        return active.id, False
    raise PodcastAudioQueueError(
        "Could not create the podcast audio queue row. Apply "
        "docs/migrations/037_podcast_audio_queue.sql."
    )


def pump_podcast_audio_queue(
    scheduler,
    cancel_events: Dict[str, threading.Event],
) -> None:
    """Start at most the oldest pending audio run when IndexTTS is contract-ready.

    The periodic job is single-instance in the webui scheduler. The pending-to-running
    update is also conditional in PostgREST, so a cancellation or second pump cannot
    submit the same request twice.
    """
    try:
        runs = db.list_active_podcast_audio_runs()
    except db.PodcastAudioQueueLookupError as exc:
        logger.error("Podcast audio queue could not be read: %s", exc)
        return

    if any(run.status == RunStatus.RUNNING.value for run in runs):
        return
    pending = [run for run in runs if run.status == RunStatus.PENDING.value]
    if not pending:
        return

    run = pending[0]
    request = dict(run.result or {})
    if (
        not request.get("script_id")
        or not request.get("voice_path")
        or not request.get("script_sha256")
        or request.get("consent") is not True
    ):
        request.update({
            "queue_state": "invalid_request",
            "queue_message": "The stored audio request is incomplete or lacks consent.",
            "start_command": None,
        })
        db.update_pending_podcast_audio_result(run.id, request)
        return

    try:
        status = podcast_audio_service.tts_status_payload()
    except Exception as exc:
        status = {
            "status": "contract_mismatch",
            "ready": False,
            "message": f"Could not verify the IndexTTS contract: {exc}",
            "start_command": None,
        }

    state = str(status.get("status") or "contract_mismatch")
    message = str(status.get("message") or "IndexTTS readiness is unknown.")
    if not status.get("ready"):
        request.update({
            "queue_state": (
                "waiting_for_indextts" if state == "not_running" else state
            ),
            "tts_status": state,
            "queue_message": message,
            # The app must never suggest a start command for a wrong service or an
            # incompatible IndexTTS API.
            "start_command": (
                status.get("start_command") if state == "not_running" else None
            ),
        })
        db.update_pending_podcast_audio_result(run.id, request)
        return

    request.update({
        "queue_state": "starting",
        "tts_status": "ready",
        "queue_message": "IndexTTS is ready; starting this audio request.",
        "start_command": None,
    })
    cancel_event = threading.Event()
    # Register before the conditional claim so a concurrent cancel can either close
    # the pending row or signal the newly claimed worker.
    cancel_events[run.id] = cancel_event
    if not db.claim_pending_podcast_audio_run(run.id, request):
        cancel_events.pop(run.id, None)
        return

    try:
        scheduler.add_job(
            execute_run,
            args=[
                run.id, KIND_PODCAST_AUDIO, run.pillar_id,
                cancel_event, cancel_events,
            ],
            kwargs={
                "scheduler": scheduler,
                "script_id": request["script_id"],
                "voice_path": request["voice_path"],
                "audio_request": request,
            },
            id=f"run:{run.id}",
            name=f"Podcast audio run {run.id} ({run.pillar_id})",
            trigger="date",
            misfire_grace_time=None,
            max_instances=1,
            coalesce=True,
        )
    except Exception as exc:
        cancel_events.pop(run.id, None)
        logger.exception("Could not schedule claimed podcast audio run %s", run.id)
        db.finish_pipeline_run(
            run.id, RunStatus.FAILED.value,
            error=f"Could not schedule podcast audio: {exc}",
        )


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
        # DROPPED is not a status a row can hold — the CHECK constraint does not admit
        # it. It means "this run has decided this step will not happen", and the
        # honest rendering of that is no row at all rather than a step left pending.
        if status == StageStatus.DROPPED.value:
            db.delete_pipeline_run_stage(run_id, name)
        else:
            db.update_pipeline_run_stage(run_id, name, status, detail)

    try:
        if kind == KIND_UPLOAD:
            # No Orchestrator on this path, deliberately. An upload does not discover,
            # search or pop a queue, and building one dials Qdrant on the way in
            # (VectorSearchTool.__init__ -> ensure_collections). An unreachable vector
            # store must not stop a paper being added to the library — that step
            # reports its own failure further down, on the VECTORS stage.
            added = _finish_upload(run_id, pillar_id, cancel_event, kwargs, on_stage)
            if added:
                _enqueue_map_build(kwargs.get("scheduler"), cancel_events, pillar_id)
            return

        if kind == KIND_PODCAST_SCRIPT:
            # Same no-Orchestrator rule: script generation never needs Qdrant.
            _finish_podcast_script(
                run_id, pillar_id, cancel_event, cancel_events, kwargs, on_stage,
            )
            return

        if kind == KIND_PODCAST_FACTCHECK:
            # Same no-Orchestrator rule: a fact-check reads stored text only.
            _finish_podcast_factcheck(run_id, cancel_event, kwargs, on_stage)
            return

        if kind == KIND_PLAYGROUND_EXPLAIN:
            # Same no-Orchestrator rule: the explain pass reads the stored notebook.
            _finish_playground_explain(run_id, cancel_event, kwargs, on_stage)
            return

        if kind == KIND_PLAYGROUND_ALIGN:
            # Same no-Orchestrator rule: alignment reads a stored script and guide.
            _finish_playground_align(run_id, cancel_event, kwargs, on_stage)
            return

        if kind == KIND_MAP_BUILD:
            # Same no-Orchestrator rule: the map is computed from stored rows.
            _finish_map_build(run_id, pillar_id, cancel_event, on_stage)
            return

        # Built HERE, inside the worker thread. Never cached on app.state and never
        # passed across the thread boundary: it holds HTTP clients and agent state
        # that belong to the run, and reusing one across runs is how state leaks.
        orchestrator = Orchestrator(
            enable_quiz=True, on_stage=on_stage, cancel=cancel_event
        )

        if kind == KIND_DISCOVER:
            _finish_discovery(run_id, orchestrator, pillar_id, kwargs)
        elif kind == KIND_PODCAST_AUDIO:
            _finish_podcast_audio(run_id, pillar_id, cancel_event, kwargs, on_stage)
        elif kind == KIND_RUN_DAILY:
            result = orchestrator.run_daily(
                pillar_id, papers_limit=kwargs.get("papers_limit", 1)
            )
            _finish_pipeline(run_id, result)
            if result.papers_processed:
                _enqueue_map_build(kwargs.get("scheduler"), cancel_events, pillar_id)
        else:
            papers = _to_paper_refs(kwargs.get("papers") or [])
            result = orchestrator.process_selected_papers(pillar_id, papers=papers)
            _finish_pipeline(run_id, result)
            if result.papers_processed:
                _enqueue_map_build(kwargs.get("scheduler"), cancel_events, pillar_id)

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
        # A discovery shows candidates and changes nothing the reader holds, and
        # the cloud reader does not carry playgrounds (or their links) or the
        # paper map yet. Every other kind may. notify() only records the fact
        # and cannot raise, so a cloud problem can never change how this run
        # ended.
        if kind not in (
            KIND_DISCOVER,
            KIND_PLAYGROUND_EXPLAIN,
            KIND_PLAYGROUND_ALIGN,
            KIND_MAP_BUILD,
        ):
            cloud_publish_service.notify(f"run:{kind}")


def _enqueue_map_build(scheduler, cancel_events, pillar_id: str) -> Optional[str]:
    """Queue a map build for a pillar that just gained a paper.

    Best effort, like the fact-check after a script: the paper is saved and
    the run that added it has already been recorded, so nothing here may
    change that. A build already pending or running for the pillar is left to
    finish (the unique index says so); the next hook or the weekly refresh
    picks up what it missed. Returns the run id, or None.
    """
    if scheduler is None:
        return None
    try:
        return dispatch_run(
            scheduler,
            cancel_events,
            pillar_id,
            map_store.TRIGGER_UI_MAP,
            KIND_MAP_BUILD,
        )
    except RunAlreadyActiveError:
        logger.info("A map build is already active for %s; not queuing another", pillar_id)
    except Exception:
        logger.exception(
            "Map build was not queued for %s. The paper was saved and the run "
            "that added it is unchanged.", pillar_id,
        )
    return None


def _finish_pipeline(run_id: str, result) -> None:
    """Close out a run_daily / process_selected run from its PipelineResult."""
    db.finish_pipeline_run(
        run_id,
        _terminal_status(result),
        papers_processed=len(result.papers_processed),
        papers_failed=_count_failed_papers(result),
        error=_summarise_errors(result),
    )
    logger.info(
        f"Run {run_id} finished: {len(result.papers_processed)} processed, "
        f"{len(result.errors)} failed"
    )


def _finish_discovery(
    run_id: str, orchestrator: Orchestrator, pillar_id: str, kwargs: Dict[str, Any]
) -> None:
    """Close out a discovery run, storing the candidates it found.

    Discovery does not produce a PipelineResult, and forcing it into one would mean
    reporting candidates as `papers_processed` — a count of papers this run put
    through the pipeline, which is zero. The candidate list goes in `result` instead,
    where the browser picks it up from the same poll it was already making.

    A source that failed does not fail the run: the other three still found papers,
    and the failed stage rows say what went wrong. The run is only FAILED when it has
    nothing to show AND something went wrong — a discovery that legitimately matched
    nothing is a succeeded run, exactly as an empty daily run is.
    """
    candidates = orchestrator.run_discovery_with_selection(
        pillar_id,
        priority_topics=kwargs.get("priority_topics") or [],
        limit=kwargs.get("limit", 10),
    )
    problems = [e.get("message", "") for e in orchestrator.infra_errors]
    problems = [p for p in problems if p]

    status = (
        RunStatus.FAILED.value
        if problems and not candidates
        else RunStatus.SUCCEEDED.value
    )
    db.finish_pipeline_run(
        run_id,
        status,
        # Deliberately 0: nothing was *processed*. The count of what was found lives
        # in the payload, where it cannot be mistaken for pipeline work.
        papers_processed=0,
        papers_failed=0,
        error=_join_problems(problems),
        result=candidates_payload(candidates),
    )
    logger.info(
        f"Run {run_id} finished discovery: {len(candidates)} candidate(s), "
        f"{len(problems)} problem(s)"
    )


def _finish_podcast_audio(
    run_id: str,
    pillar_id: str,
    cancel_event: threading.Event,
    kwargs: Dict[str, Any],
    on_stage,
) -> None:
    script_id = kwargs.get("script_id")
    voice_path = kwargs.get("voice_path")
    if not script_id or not voice_path:
        raise ValueError("podcast_audio run requires script_id and voice_path")

    podcast_audio_service.run_podcast_audio_job(
        run_id,
        script_id,
        voice_path,
        on_stage,
        cancel_event,
    )
    script = db.get_podcast_script_by_id(script_id)
    meta = script.audio_metadata if script else None
    result_payload: Dict[str, Any] = dict(kwargs.get("audio_request") or {})
    if meta and meta.file_name:
        result_payload.update({
            "script_id": script_id,
            "audio_url": f"/api/podcast/audio/{meta.file_name}",
            "file_name": meta.file_name,
            "duration_seconds": meta.duration_seconds,
            "voice_path": meta.voice_path,
            "engine": meta.engine,
        })
    db.finish_pipeline_run(
        run_id,
        RunStatus.SUCCEEDED.value,
        papers_processed=0,
        papers_failed=0,
        result=result_payload or None,
    )
    logger.info("Run %s finished podcast audio for script %s", run_id, script_id)


def _finish_podcast_factcheck(
    run_id: str,
    cancel_event: threading.Event,
    kwargs: Dict[str, Any],
    on_stage,
) -> None:
    """Close out a fact-check. A failure raises and does not touch the script."""
    script_id = kwargs.get("script_id")
    if not script_id:
        raise ValueError("podcast_factcheck run requires script_id")
    payload = podcast_factcheck_service.run_factcheck_job(
        script_id, on_stage, cancel_event,
    )
    db.finish_pipeline_run(
        run_id,
        RunStatus.SUCCEEDED.value,
        papers_processed=0,
        papers_failed=0,
        result=payload,
    )
    logger.info(
        "Run %s finished fact check %s for script %s",
        run_id,
        payload.get("fact_check_id"),
        script_id,
    )


def _finish_playground_explain(
    run_id: str,
    cancel_event: threading.Event,
    kwargs: Dict[str, Any],
    on_stage,
) -> None:
    """Close out an explain pass.

    A guide that covers every cell is a succeeded run. A partial guide is
    saved and the run is FAILED with that payload, the same two-facts rule as
    an upload: the guide exists, and some cells were not explained. A pass
    with nothing usable raises and execute_run records it.
    """
    playground_id = kwargs.get("playground_id")
    if not playground_id:
        raise ValueError("playground_explain run requires playground_id")
    payload = playground_service.run_explain_job(playground_id, on_stage, cancel_event)
    ok = payload.get("status") == "ready"
    db.finish_pipeline_run(
        run_id,
        RunStatus.SUCCEEDED.value if ok else RunStatus.FAILED.value,
        papers_processed=0,
        papers_failed=0,
        error=None if ok else payload.get("error"),
        result=payload,
    )
    logger.info(
        "Run %s finished playground %s: %s, %s of %s cells",
        run_id, playground_id, payload.get("status"),
        payload.get("cells_explained"), payload.get("cells_total"),
    )


def _finish_map_build(
    run_id: str,
    pillar_id: str,
    cancel_event: threading.Event,
    on_stage,
) -> None:
    """Close out one pillar's map build.

    A written snapshot is a succeeded run. A failure raises (the previous
    snapshot is left as it was) and execute_run records it failed with the
    reason. Cancellation is checked as each stage starts, never mid-stage.
    """

    def check_cancel() -> None:
        if cancel_event.is_set():
            raise RunCancelledError("Map build cancelled")

    payload = map_store.run_map_build_job(
        pillar_id, on_stage, check_cancel, run_id=run_id
    )
    db.finish_pipeline_run(
        run_id,
        RunStatus.SUCCEEDED.value,
        papers_processed=0,
        papers_failed=0,
        error=None,
        result=payload,
    )
    logger.info(
        "Run %s built the map for %s: %s",
        run_id, pillar_id, payload.get("payload_hash"),
    )


def _finish_playground_align(
    run_id: str,
    cancel_event: threading.Event,
    kwargs: Dict[str, Any],
    on_stage,
) -> None:
    """Close out a follow-along alignment pass.

    A stored mapping is a succeeded run, including an empty one (the episode
    never talks about a particular cell). A pass with nothing usable raises,
    and execute_run records it failed with the reason.
    """
    script_id = kwargs.get("script_id")
    playground_id = kwargs.get("playground_id")
    if not script_id or not playground_id:
        raise ValueError("playground_align run requires script_id and playground_id")
    payload = playground_follow.run_align_job(script_id, playground_id, on_stage, cancel_event)
    db.finish_pipeline_run(
        run_id,
        RunStatus.SUCCEEDED.value,
        papers_processed=0,
        papers_failed=0,
        error=None,
        result=payload,
    )
    logger.info(
        "Run %s aligned script %s to playground %s: %s spans",
        run_id, script_id, playground_id, payload.get("spans"),
    )


def _maybe_factcheck_after_script(
    scheduler,
    cancel_events,
    *,
    script_id: str,
    paper_id: str,
    pillar_id: str,
) -> None:
    """Start a fact-check without letting it change the script run's outcome."""
    try:
        podcast_factcheck_service.maybe_dispatch_after_script(
            scheduler,
            cancel_events,
            script_id=script_id,
            paper_id=paper_id,
            pillar_id=pillar_id,
        )
    except Exception:
        logger.exception(
            "Fact-check was not started after script %s. "
            "The script run stays succeeded and the script was not changed.",
            script_id,
        )


def _finish_podcast_script(
    run_id: str,
    pillar_id: str,
    cancel_event: threading.Event,
    cancel_events: Dict[str, threading.Event],
    kwargs: Dict[str, Any],
    on_stage,
) -> None:
    """Close out a podcast script generation run.

    Two honesty facts travel in ``result``, matching the old sync route:

    - ``saved`` / ``script_id`` — whether the insert worked
    - full ``script`` text when ``saved`` is false — the only copy of a paid artifact

    A save failure therefore finishes as SUCCEEDED with ``saved: false`` (generation
    did what was asked). Insufficient material / extraction / cancel raise and are
    recorded by ``execute_run`` as failed / cancelled.
    """
    paper_id = kwargs.get("paper_id")
    options = kwargs.get("options")
    if not paper_id or options is None:
        raise ValueError("podcast_script run requires paper_id and options")

    result_payload = podcast_script_service.run_podcast_script_job(
        run_id,
        paper_id,
        pillar_id,
        options,
        on_stage,
        cancel_event,
    )
    # Retain the confirmed audio choice in the durable script-run payload. The
    # finisher may replace the initial request result, and the later audio queue
    # needs these values after the browser has left the page.
    result_payload["voice_path"] = kwargs.get("voice_path")
    result_payload["auto_audio"] = bool(kwargs.get("auto_audio", False))
    db.finish_pipeline_run(
        run_id,
        RunStatus.SUCCEEDED.value,
        papers_processed=0,
        papers_failed=0,
        result=result_payload,
        error=(
            None
            if result_payload.get("saved")
            else "Script generated but not saved to the database"
        ),
    )
    logger.info(
        "Run %s finished podcast script for paper %s (saved=%s)",
        run_id,
        paper_id,
        result_payload.get("saved"),
    )
    if result_payload.get("saved") and result_payload.get("script_id"):
        if kwargs.get("auto_audio") is True:
            try:
                script = db.get_podcast_script_by_id(result_payload["script_id"])
                if script is None:
                    raise PodcastAudioQueueError(
                        "The saved script could not be read for audio queueing."
                    )
                audio_run_id, _created = queue_podcast_audio_run(
                    pillar_id=pillar_id,
                    script_id=result_payload["script_id"],
                    voice_path=kwargs.get("voice_path"),
                    script_sha256=hashlib.sha256(
                        script.script.encode("utf-8")
                    ).hexdigest(),
                    consent=True,
                )
                result_payload["audio_run_id"] = audio_run_id
                db.finish_pipeline_run(
                    run_id,
                    RunStatus.SUCCEEDED.value,
                    result=result_payload,
                )
            except Exception as exc:
                result_payload["audio_queue_error"] = str(exc)
                db.finish_pipeline_run(
                    run_id,
                    RunStatus.SUCCEEDED.value,
                    result=result_payload,
                )
                logger.exception(
                    "Saved script %s but could not queue requested audio",
                    result_payload["script_id"],
                )
        _maybe_factcheck_after_script(
            kwargs.get("scheduler"),
            cancel_events,
            script_id=result_payload["script_id"],
            paper_id=result_payload.get("paper_id") or paper_id,
            pillar_id=pillar_id,
        )


def _finish_upload(
    run_id: str,
    pillar_id: str,
    cancel_event: threading.Event,
    kwargs: Dict[str, Any],
    on_stage,
) -> bool:
    """Close out a manual upload run, from either source.

    Returns True: by the time it returns the paper is in the library, which is
    what the caller's map-build hook needs to know.

    An upload reports TWO facts and they are not the same one, which is the mistake
    this path has made twice already (see UploadService._run_full_pipeline). The paper
    reaching the library is one; whether the summarizer, lesson, quiz, metadata and
    vector steps finished is the other.

    Both reach the user here. The run's terminal STATUS carries the second — a run
    whose post-upload steps failed is ``failed``, exactly as a daily run with errors
    is, because the alternative is a green panel above a red stage row — and the
    ``result`` payload carries the first, so the page can say "the paper is in the
    library, these steps did not finish" instead of implying the upload must be
    retried. ``papers_processed`` follows the orchestrator's meaning: a paper whose
    processing failed was not processed.

    A failure that stops the paper reaching the library at all does NOT come back
    here: run_url_upload_job / run_file_upload_job raise, and execute_run records the
    run as failed with the reason. That is the loud path and it is meant to be.
    """
    source = kwargs.get("source")
    service = get_upload_service()

    if source == UPLOAD_SOURCE_URL:
        result = service.run_url_upload_job(
            pillar_id,
            UploadUrlRequest(
                url=kwargs["url"],
                title=kwargs.get("title"),
                authors=kwargs.get("authors"),
                run_summarizer=kwargs.get("run_summarizer", True),
                generate_quiz=kwargs.get("generate_quiz", True),
            ),
            on_stage=on_stage,
            cancel=cancel_event,
        )
    elif source == UPLOAD_SOURCE_FILE:
        result = service.run_file_upload_job(
            pillar_id,
            kwargs["saved_path"],
            kwargs["filename"],
            UploadFileRequest(
                title=kwargs["title"],
                authors=kwargs.get("authors") or [],
                venue=kwargs.get("venue"),
                year=kwargs.get("year"),
                run_summarizer=kwargs.get("run_summarizer", True),
                generate_quiz=kwargs.get("generate_quiz", True),
            ),
            on_stage=on_stage,
            cancel=cancel_event,
        )
    else:
        raise ValueError(f"upload run needs source={UPLOAD_SOURCE_URL!r} or "
                         f"{UPLOAD_SOURCE_FILE!r}, got {source!r}")

    outcome = result.outcome
    status = RunStatus.SUCCEEDED.value if outcome.ok else RunStatus.FAILED.value
    db.finish_pipeline_run(
        run_id,
        status,
        papers_processed=1 if outcome.ok else 0,
        papers_failed=0 if outcome.ok else 1,
        error=_join_problems(outcome.errors),
        result={
            "paper_id": result.paper.id,
            "title": result.paper.title,
            "pillar_id": pillar_id,
            "source": result.source,
            # Always true by the time we are here: the job raises rather than
            # returning if the papers row was never written. Stated explicitly so the
            # page never has to infer it from a status that is about something else.
            "added": True,
            "actions_triggered": outcome.actions_triggered,
            "errors": outcome.errors,
        },
    )
    logger.info(
        "Run %s finished upload of %s (%s): %d action(s), %d error(s)",
        run_id, result.paper.id, result.source,
        len(outcome.actions_triggered), len(outcome.errors),
    )
    return True


def _join_problems(problems: List[str]) -> Optional[str]:
    """One run-level line for the degradations a run survived. Shared by discovery
    and upload, which both produce several independent problems per run.

    Kept even on a succeeded run: "we found ten papers, but arXiv was rate-limited and
    these are the other three sources' results" is a materially different claim from
    "we found ten papers", and the user is the one who decides whether to rerun.
    """
    if not problems:
        return None
    joined = " | ".join(problems)
    if len(joined) > _MAX_ERROR_CHARS:
        joined = joined[:_MAX_ERROR_CHARS].rstrip() + " …(truncated, see logs)"
    return joined


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


def cancel_run(
    cancel_events: Dict[str, threading.Event],
    run_id: str,
) -> tuple[bool, str]:
    """Cancel a durable audio waiter or request cooperative cancellation if running."""
    if db.cancel_pending_podcast_audio_run(run_id):
        for stage in PODCAST_AUDIO_STAGES:
            db.update_pipeline_run_stage(
                run_id,
                stage.value,
                StageStatus.SKIPPED.value,
                "Cancelled while waiting for IndexTTS",
            )
        cancel_events.pop(run_id, None)
        return True, "Podcast audio cancelled while waiting for IndexTTS."
    if request_cancel(cancel_events, run_id):
        return True, "Cancellation requested; the run will stop after its current stage."
    return False, "No cancellable run with that id in this process."


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
WEBUI_TRIGGER_SOURCES = [
    "ui_pipeline",
    "ui_select",
    "ui_discover",
    TRIGGER_UI_UPLOAD,
    podcast_audio_service.TRIGGER_UI_PODCAST_AUDIO,
    podcast_script_service.TRIGGER_UI_PODCAST_SCRIPT,
    podcast_factcheck_service.TRIGGER_UI_PODCAST_FACTCHECK,
    podcast_factcheck_service.TRIGGER_AUTO_PODCAST_FACTCHECK,
    playground_service.TRIGGER_UI_PLAYGROUND,
    map_store.TRIGGER_UI_MAP,
]


def sweep_interrupted_runs_on_startup() -> int:
    """Mark runs abandoned by a previous webui process as interrupted.

    Call once at startup, before serving traffic. Scoped to this process's own
    trigger sources so a webui restart cannot declare a live scheduler run dead.
    """
    try:
        return db.sweep_interrupted_runs(
            trigger_sources=WEBUI_TRIGGER_SOURCES,
            preserve_pending_kinds=[KIND_PODCAST_AUDIO],
        )
    except Exception as e:
        # Never let bookkeeping stop the application from booting.
        logger.error(f"Startup sweep of interrupted runs failed: {e}")
        return 0


def _to_paper_refs(papers: List[Any]) -> List[PaperRef]:
    """Accept PaperRef objects or plain dicts, since the route hands over JSON."""
    return [p if isinstance(p, PaperRef) else PaperRef(**p) for p in papers]


#: The `step` the orchestrator stamps on a per-paper failure. Everything else in
#: `errors` is a failure of the shared plumbing (search, queue, database), which must
#: not be counted as a dead paper.
_PAPER_FAILURE_STEP = "process_paper"


def _count_failed_papers(result) -> int:
    """How many PAPERS failed, as opposed to how many things went wrong."""
    return sum(
        1 for e in result.errors
        if isinstance(e, dict) and e.get("step") == _PAPER_FAILURE_STEP
    )


def _terminal_status(result) -> str:
    """Map a PipelineResult onto a run status a user can act on.

    ``PipelineResult.success`` is ``len(papers_processed) > 0``, so a run that found
    nothing new — the ordinary outcome once a pillar has caught up — arrives here as
    ``success=False`` with an empty ``errors`` list. That used to be recorded as
    ``failed``, and since there was no error to report the browser rendered a bare red
    "Failed" above eleven green completed stages. Nothing had gone wrong; there was
    simply nothing to do, and the run said so on every stage.

    ``success`` itself is deliberately left alone: the CLI, the scheduler and fourteen
    existing tests read it, and "did this run produce anything" is the right meaning
    for all of them. Only the user-facing run status needed to learn the difference.
    """
    if result.success or not result.errors:
        return RunStatus.SUCCEEDED.value
    return RunStatus.FAILED.value


#: Room for several messages inside the 2000-char cap finish_pipeline_run applies.
_MAX_ERROR_CHARS = 1800


def _summarise_errors(result) -> Optional[str]:
    """Put every paper's failure on the run row, not just the first.

    Stage rows cannot carry this: they are per-run, not per-paper, so when paper 2
    re-enters a stage that paper 1 failed in it overwrites the failure back to
    completed. The run-level error string is the only place a multi-paper run can
    report more than one casualty, and reporting only ``errors[0]`` meant a run with
    three failures looked like it had one.
    """
    if not result.errors:
        return None

    messages = [
        e.get("message") if isinstance(e, dict) else str(e) for e in result.errors
    ]
    messages = [m for m in messages if m]
    if not messages:
        return None
    if len(messages) == 1:
        return messages[0]

    joined = f"{len(messages)} papers failed: " + " | ".join(messages)
    if len(joined) > _MAX_ERROR_CHARS:
        joined = joined[:_MAX_ERROR_CHARS].rstrip() + " …(truncated, see logs)"
    return joined
