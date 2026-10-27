"""Paper Playground API: read a playground, import a notebook, start the explain pass.

Paper ids carry ``:`` and sometimes ``/``, so they travel as a query
parameter, never as a path segment. Playground ids are UUIDs.

The store is synchronous (``nlp_pillars.db``), so every call goes through
``asyncio.to_thread``. A missing ``playgrounds`` table is a 503 whose detail
names ``docs/migrations/028_playgrounds.sql``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from nlp_pillars.config import get_settings
from nlp_pillars.db import PipelineRunCreateError
from nlp_pillars.playground import MIGRATION, store
from nlp_pillars.playground import service as playground_service
from nlp_pillars.playground.notebook import NotebookError, cells_view
from webui.services import run_service
from webui.services.run_service import RunAlreadyActiveError
from webui.services.spend_guard import require_spend_room

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/playgrounds", tags=["playgrounds"])

#: Named when the run insert fails its CHECK: 028 also widens the run kinds.
MIGRATION_HINT = (
    "The explain pass runs as pipeline runs of kind 'playground_explain'. If this "
    f"database has not had {MIGRATION} applied, the insert fails its CHECK "
    "constraint. Apply it with: docker exec -i nlp_postgres psql -U nlp -d nlp "
    f"-v ON_ERROR_STOP=1 -f - < {MIGRATION}"
)


def _unavailable(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


def public_playground(row: Dict[str, Any], notebook: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What the Playground tab renders. The raw notebook stays on the server."""
    return {
        "id": row.get("id"),
        "paper_id": row.get("paper_id"),
        "title": row.get("title"),
        "source_kind": row.get("source_kind"),
        "notebook_name": row.get("notebook_name"),
        "status": row.get("status"),
        "model": row.get("model"),
        "error": row.get("error"),
        "steps": row.get("steps") or [],
        "datasets": row.get("datasets") or [],
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "cells": cells_view(notebook or {}),
    }


@router.get("")
async def get_playground(paper_id: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """The paper's playground and its latest explain run.

    ``{"playground": null}`` means the paper has none yet; a failed read is a
    503, never an empty answer.
    """
    try:
        row = await asyncio.to_thread(store.get_for_paper, paper_id, with_notebook=True)
    except store.PlaygroundStoreError as exc:
        raise _unavailable(exc)
    if row is None:
        return {"playground": None, "latest_run": None}
    try:
        latest = await asyncio.to_thread(store.latest_explain_run, row["id"])
    except store.PlaygroundStoreError as exc:
        # The guide is still worth showing; only the run status is unknown.
        logger.warning("Playground %s: explain run status unreadable: %s", row["id"], exc)
        latest = None
    return {
        "playground": public_playground(row, row.get("notebook")),
        "latest_run": latest,
    }


async def _read_capped(file: UploadFile, cap: int) -> bytes:
    data = await file.read(cap + 1)
    if len(data) > cap:
        raise HTTPException(
            status_code=413,
            detail=(
                f"The notebook is larger than {cap / 1_000_000:.1f} MB "
                "(PLAYGROUND_NOTEBOOK_MAX_BYTES)."
            ),
        )
    return data


@router.post("/import")
async def import_playground(
    paper_id: str = Query(..., min_length=1),
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Attach a .ipynb to a paper (replacing any it had) and cache its datasets.

    Dataset downloads happen here, off the event loop, so a large dataset
    holds this one request and nothing else. No model is called.
    """
    data = await _read_capped(file, get_settings().playground_notebook_max_bytes)
    try:
        row = await asyncio.to_thread(
            playground_service.import_notebook, paper_id, data, file.filename or "",
        )
    except NotebookError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except playground_service.PaperNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except store.PlaygroundStoreError as exc:
        raise _unavailable(exc)
    datasets = row.get("datasets") or []
    return {
        "playground_id": row.get("id"),
        "title": row.get("title"),
        "datasets": datasets,
        "message": _import_message(datasets),
    }


def _import_message(datasets) -> str:
    if not datasets:
        return "Notebook imported."
    ok = sum(1 for d in datasets if not d.get("error"))
    if ok == len(datasets):
        return f"Notebook imported. {ok} dataset{'s' if ok != 1 else ''} saved for offline runs."
    return (
        f"Notebook imported. {ok} of {len(datasets)} datasets saved; "
        "the rest could not be downloaded (see the list)."
    )


def _conflict(playground_id: str) -> JSONResponse:
    try:
        latest = store.latest_explain_run(playground_id)
    except store.PlaygroundStoreError:
        latest = None
    body: Dict[str, Any] = {"detail": "A study guide is already being written for this notebook"}
    if latest and latest.get("status") in ("pending", "running"):
        body["run_id"] = latest["id"]
    return JSONResponse(status_code=409, content=body)


@router.post(
    "/{playground_id}/explain",
    status_code=202,
    dependencies=[Depends(require_spend_room)],
)
async def start_explain(playground_id: str, http_request: Request):
    """Start the explain pass (one or more Claude calls) as a background run.

    202 with the run id. 409 with the active run's id when one is already
    going for this playground. 503 names migration 028.
    """
    try:
        playground_id = str(uuid.UUID(playground_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="Playground not found")
    try:
        row = await asyncio.to_thread(store.get, playground_id, with_notebook=False)
        if row is None:
            raise HTTPException(status_code=404, detail="Playground not found")
        latest = await asyncio.to_thread(store.latest_explain_run, playground_id)
        paper = await asyncio.to_thread(store.paper_row, row["paper_id"])
    except store.PlaygroundStoreError as exc:
        raise _unavailable(exc)
    if latest and latest.get("status") in ("pending", "running"):
        return JSONResponse(
            status_code=409,
            content={
                "detail": "A study guide is already being written for this notebook",
                "run_id": latest["id"],
            },
        )
    if paper is None:
        raise HTTPException(status_code=404, detail="The playground's paper no longer exists")
    try:
        run_id = run_service.dispatch_run(
            http_request.app.state.scheduler,
            http_request.app.state.cancel_events,
            pillar_id=paper["pillar_id"],
            trigger_source=playground_service.TRIGGER_UI_PLAYGROUND,
            kind=run_service.KIND_PLAYGROUND_EXPLAIN,
            playground_id=playground_id,
            initial_result={"playground_id": playground_id, "paper_id": row["paper_id"]},
        )
    except RunAlreadyActiveError:
        # The per-playground unique index won a race with another start.
        return await asyncio.to_thread(_conflict, playground_id)
    except PipelineRunCreateError as exc:
        logger.error("Could not create playground_explain run: %s", exc)
        raise HTTPException(status_code=503, detail=f"{exc}. {MIGRATION_HINT}")
    return {"run_id": run_id, "message": "Writing the study guide"}
