"""Follow-along API: link an episode to a playground and read the link.

The Listen page asks which playground an episode follows and starts the
alignment pass (docs/PLAYGROUND.md, "Follow-along"). The cards themselves come
from ``GET /api/playgrounds?paper_id=…``, the Playground tab's own read, so
the code, explanations and sanitized outputs are rendered from one source.

Everything underneath is synchronous (``nlp_pillars.db``), so every call goes
through ``asyncio.to_thread``. A missing ``playground_links`` table is a 503
whose detail names ``docs/migrations/029_playground_links.sql``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from nlp_pillars import db
from nlp_pillars.db import PipelineRunCreateError
from nlp_pillars.playground import FOLLOW_MIGRATION, follow, links, store
from nlp_pillars.playground.service import TRIGGER_UI_PLAYGROUND
from webui.services import run_service
from webui.services.run_service import RunAlreadyActiveError
from webui.services.spend_guard import require_spend_room

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/follow-along", tags=["follow-along"])

#: Named when the run insert fails its CHECK: 029 also widens the run kinds.
MIGRATION_HINT = (
    "Follow-along passes run as pipeline runs of kind 'playground_align'. If this "
    f"database has not had {FOLLOW_MIGRATION} applied, the insert fails its CHECK "
    "constraint. Apply it with: docker exec -i nlp_postgres psql -U nlp -d nlp "
    f"-v ON_ERROR_STOP=1 -f - < {FOLLOW_MIGRATION}"
)

_ACTIVE = ("pending", "running")


class StartRequest(BaseModel):
    script_id: str
    playground_id: str


def _playground_summary(row: Optional[Dict[str, Any]], paper_title: str = "") -> Optional[Dict[str, Any]]:
    if not row:
        return None
    steps = row.get("steps") or []
    return {
        "id": row.get("id"),
        "paper_id": row.get("paper_id"),
        "paper_title": paper_title,
        "title": row.get("title"),
        "status": row.get("status"),
        "steps": len(steps),
    }


def _read_follow(script_id: str) -> Dict[str, Any]:
    """Everything the Listen page needs to decide what the code panel shows."""
    try:
        script = db.get_podcast_script_by_id(script_id)
    except db.PodcastScriptLookupError as exc:
        raise store.PlaygroundStoreError(str(exc)) from exc
    if script is None:
        raise follow.AlignTargetMissing("Episode not found")
    own = store.get_for_paper(script.paper_id) if script.paper_id else None
    rows = links.links_for_script(script_id)
    playgrounds: Dict[str, Optional[Dict[str, Any]]] = {}
    for row in rows:
        pid = row.get("playground_id")
        if pid not in playgrounds:
            playgrounds[pid] = store.get(pid, with_notebook=False)
    paper_ids = [p.get("paper_id") for p in playgrounds.values() if p]
    if own:
        paper_ids.append(own.get("paper_id"))
    titles = links.paper_titles(paper_ids)
    out_links: List[Dict[str, Any]] = []
    for row in rows:
        pg = playgrounds.get(row.get("playground_id"))
        if pg is None:
            continue  # cascaded away between the two reads
        out_links.append({
            "id": row.get("id"),
            "playground_id": row.get("playground_id"),
            "status": row.get("status"),
            "error": row.get("error"),
            "model": row.get("model"),
            "mapping": row.get("mapping") or [],
            "script_sha256": row.get("script_sha256"),
            "updated_at": row.get("updated_at"),
            "playground": _playground_summary(pg, titles.get(pg.get("paper_id"), "")),
        })
    try:
        latest = links.latest_align_run(script_id)
    except store.PlaygroundStoreError as exc:
        logger.warning("Follow-along %s: run status unreadable: %s", script_id, exc)
        latest = None
    return {
        "script_id": script_id,
        "paper_id": script.paper_id,
        "script_sha256": follow.sha256_text(script.script or ""),
        "own_playground": _playground_summary(own, titles.get(script.paper_id, "")),
        "links": out_links,
        "latest_run": latest,
    }


@router.get("")
async def get_follow_along(script_id: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """The episode's links, its own paper's playground, and the latest pass.

    404 when the episode does not exist. 503 for a failed read, naming
    migration 029 when the links table is missing.
    """
    canonical = links.canonical_id(script_id)
    if canonical is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    try:
        return await asyncio.to_thread(_read_follow, canonical)
    except follow.AlignTargetMissing as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except store.PlaygroundStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/playgrounds")
async def list_playgrounds() -> Dict[str, Any]:
    """Papers with a playground, for the picker. No notebooks."""
    try:
        rows = await asyncio.to_thread(links.list_playgrounds)
    except store.PlaygroundStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "playgrounds": [
            {
                "id": r.get("id"),
                "paper_id": r.get("paper_id"),
                "paper_title": r.get("paper_title") or "",
                "title": r.get("title"),
                "status": r.get("status"),
            }
            for r in rows
        ]
    }


def _active_run_for(script_id: str, playground_id: str) -> Optional[Dict[str, Any]]:
    try:
        latest = links.latest_align_run(script_id)
    except store.PlaygroundStoreError:
        return None
    if not latest or latest.get("status") not in _ACTIVE:
        return None
    result = latest.get("result") or {}
    if result.get("playground_id") not in (None, playground_id):
        return None
    return latest


def _conflict(script_id: str, playground_id: str) -> JSONResponse:
    active = _active_run_for(script_id, playground_id)
    body: Dict[str, Any] = {"detail": "This episode is already being matched to this notebook"}
    if active:
        body["run_id"] = active["id"]
    return JSONResponse(status_code=409, content=body)


@router.post("", status_code=202, dependencies=[Depends(require_spend_room)])
async def start_follow_along(request: StartRequest, http_request: Request):
    """Link an episode to a playground and start the alignment pass.

    One cheap Claude call, as a background run.

    202 with the run id. 409 with the active run's id when this pair is
    already being aligned. 404 for an unknown episode or playground, 422 when
    there is nothing to align (no study guide, no spoken lines), 503 naming
    migration 029.
    """
    script_id = links.canonical_id(request.script_id)
    playground_id = links.canonical_id(request.playground_id)
    if script_id is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    if playground_id is None:
        raise HTTPException(status_code=404, detail="Playground not found")
    try:
        # Every check the job would make, before a run row exists: a refusal
        # here costs nothing and says why.
        prep = await asyncio.to_thread(follow.prepare, script_id, playground_id)
    except follow.AlignTargetMissing as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except follow.PlaygroundAlignError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except store.PlaygroundStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    active = await asyncio.to_thread(_active_run_for, script_id, playground_id)
    if active:
        return JSONResponse(
            status_code=409,
            content={
                "detail": "This episode is already being matched to this notebook",
                "run_id": active["id"],
            },
        )
    try:
        run_id = run_service.dispatch_run(
            http_request.app.state.scheduler,
            http_request.app.state.cancel_events,
            pillar_id=prep.script.pillar_id,
            # 028's 'ui_playground', so the webui startup sweep owns these runs.
            trigger_source=TRIGGER_UI_PLAYGROUND,
            kind=run_service.KIND_PLAYGROUND_ALIGN,
            script_id=script_id,
            playground_id=playground_id,
            initial_result={"script_id": script_id, "playground_id": playground_id},
        )
    except RunAlreadyActiveError:
        # The per-pair unique index won a race with another start.
        return await asyncio.to_thread(_conflict, script_id, playground_id)
    except PipelineRunCreateError as exc:
        logger.error("Could not create playground_align run: %s", exc)
        raise HTTPException(status_code=503, detail=f"{exc}. {MIGRATION_HINT}") from exc
    return {
        "run_id": run_id,
        "link_id": prep.link.get("id"),
        "message": "Matching the episode to the notebook",
    }
