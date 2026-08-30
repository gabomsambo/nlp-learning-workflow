"""
API router for paper upload operations.

Both routes answer **202 with a run id** and do the work on an APScheduler worker
thread, modelled on ``POST /api/pillars/{id}/discover``. They used to run the whole
upload inside the request: measured from the live logs, one 4.71 MB arXiv paper held
``POST /upload/url`` open for 3 minutes 20 seconds, during which the single uvicorn
process was frozen for every other request and the page showed nothing at all.

The in-memory ``UploadStatus`` object and its two endpoints are gone with it. They
could not have worked: the status lived on a process-local singleton invisible to the
scheduler container, its id was only returned once the call it described had finished,
and no page ever polled it. ``pipeline_runs`` is now the single source of truth for
what an upload is doing.
"""

import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Request

from nlp_pillars.db import PipelineRunCreateError, pillar_exists
from nlp_pillars.schemas import UploadAccepted, UploadUrlRequest
from nlp_pillars.services.upload_service import get_upload_service
from webui.services.run_service import (
    KIND_UPLOAD,
    TRIGGER_UI_UPLOAD,
    UPLOAD_SOURCE_FILE,
    UPLOAD_SOURCE_URL,
    RunAlreadyActiveError,
    dispatch_run,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pillars", tags=["uploads"])

#: What a failed insert most likely means when the CHECK constraints reject the new
#: values. Migration 015 is hand-applied (AGENTS.md, "docs/migrations/ splits in two"),
#: and this project has twice shipped code whose migration was never run — so the
#: route names the file rather than surfacing a bare Postgres error.
_MIGRATION_HINT = (
    "Uploads run as pipeline runs of kind 'upload'. If this database has not had "
    "docs/migrations/015_upload_runs.sql applied, the insert fails its CHECK "
    "constraint. Apply it with: docker exec -i nlp_postgres psql -U nlp -d nlp "
    "-v ON_ERROR_STOP=1 -f - < docs/migrations/015_upload_runs.sql"
)


def _require_pillar(pillar_id: str) -> None:
    if not pillar_exists(pillar_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pillar '{pillar_id}' not found",
        )


def _dispatch_upload(http_request: Request, pillar_id: str, **kwargs) -> str:
    """Create the run row and schedule the job, or say precisely why not.

    Every failure here is loud. A dispatch that quietly fell back to doing the work in
    the request would reintroduce the frozen page this change exists to remove, and a
    dispatch that failed silently would leave the user looking at a form that appears
    to have done nothing.
    """
    try:
        return dispatch_run(
            http_request.app.state.scheduler,
            http_request.app.state.cancel_events,
            pillar_id=pillar_id,
            trigger_source=TRIGGER_UI_UPLOAD,
            kind=KIND_UPLOAD,
            **kwargs,
        )
    except RunAlreadyActiveError as e:
        # Should be unreachable once 015 is applied: its narrowed partial unique index
        # exempts kind='upload' precisely so an upload is never refused because a
        # discovery run is in flight on the same pillar. Reaching this means the index
        # is still the pre-015 one.
        logger.error(f"Upload refused as a duplicate active run: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{e}. {_MIGRATION_HINT}",
        ) from e
    except PipelineRunCreateError as e:
        logger.error(f"Could not create an upload run for {pillar_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not start the upload: {e}. {_MIGRATION_HINT}",
        ) from e


@router.post(
    "/{pillar_id}/upload/url", response_model=UploadAccepted, status_code=202
)
async def upload_paper_from_url(
    pillar_id: str, request: UploadUrlRequest, http_request: Request
) -> UploadAccepted:
    """
    Start a paper upload from a URL and return a run id immediately.

    - **pillar_id**: Target pillar ID (slug)
    - **url**: URL to download PDF from
    - **title**: Optional title override
    - **authors**: Optional authors list
    - **run_summarizer**: Whether to run summarizer after upload
    - **generate_quiz**: Whether to generate quiz after upload

    The download, metadata lookup, ingest, summariser, lesson, quiz and vector steps
    all happen on the run; poll ``GET /api/pipeline-runs/{run_id}`` for per-stage
    progress and read the finished run's ``result`` for what the upload added.
    """
    _require_pillar(pillar_id)

    run_id = _dispatch_upload(
        http_request,
        pillar_id,
        source=UPLOAD_SOURCE_URL,
        url=request.url,
        title=request.title,
        authors=request.authors,
        run_summarizer=request.run_summarizer,
        generate_quiz=request.generate_quiz,
    )
    return UploadAccepted(
        run_id=run_id,
        pillar_id=pillar_id,
        message=f"Uploading {request.url}",
    )


@router.post(
    "/{pillar_id}/upload/pdf", response_model=UploadAccepted, status_code=202
)
async def upload_paper_from_file(
    pillar_id: str,
    http_request: Request,
    file: UploadFile = File(..., description="PDF file to upload"),
    title: str = Form(..., description="Paper title"),
    authors: Optional[str] = Form(None, description="Comma-separated list of authors"),
    venue: Optional[str] = Form(None, description="Conference or journal"),
    year: Optional[int] = Form(None, description="Publication year"),
    # Defaulted on to match the discovery path, which hardcodes enable_quiz=True and
    # summarises unconditionally. An uploaded paper used to get a bare row unless the
    # user ticked two boxes; matching the pipeline is the least surprising default.
    run_summarizer: bool = Form(True, description="Run summarizer after upload"),
    generate_quiz: bool = Form(True, description="Generate quiz after upload"),
) -> UploadAccepted:
    """
    Accept a PDF, save it, and start the upload run. Returns a run id immediately.

    The bytes are written to disk HERE, before dispatch, and the worker is handed a
    path. That is not an optimisation: Starlette closes the ``UploadFile``'s spooled
    temporary file when the response is sent, and the response is sent long before the
    job runs, so the handle cannot cross the thread boundary. The saved file is also
    the paper's permanent copy — ``papers.url_pdf`` becomes ``file://<path>`` — which
    is why it lives under the retained upload directory and not a temp dir.
    """
    _require_pillar(pillar_id)

    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    authors_list = []
    if authors:
        authors_list = [a.strip() for a in authors.split(',') if a.strip()]

    upload_service = get_upload_service()
    saved_path = await upload_service.save_uploaded_file(file)

    try:
        run_id = _dispatch_upload(
            http_request,
            pillar_id,
            source=UPLOAD_SOURCE_FILE,
            saved_path=saved_path,
            filename=file.filename,
            title=title,
            authors=authors_list,
            venue=venue,
            year=year,
            run_summarizer=run_summarizer,
            generate_quiz=generate_quiz,
        )
    except Exception:
        # No run means no worker will ever look at this file, and no papers row will
        # ever point at it. Retention only applies once something refers to the file.
        if os.path.exists(saved_path):
            try:
                os.unlink(saved_path)
                logger.info(f"Discarded upload that was never dispatched: {saved_path}")
            except OSError as cleanup_error:
                logger.warning(f"Failed to clean up {saved_path}: {cleanup_error}")
        raise

    return UploadAccepted(
        run_id=run_id,
        pillar_id=pillar_id,
        message=f"Uploading {file.filename}",
    )
