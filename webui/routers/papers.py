"""Library list and the paper page.

``GET /papers/details/{paper_id}`` stays the JSON the modal already fetches.
``GET /papers/{paper_id}`` is the page. Paper ids contain ``:``
(``arxiv:2402.08593``); the page route is a path segment and every link
encodes the id, so the colon is not a route separator.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from nlp_pillars.db import get_pillars_or_empty

from ..library_format import build_library, build_paper_page
from ..map_format import build_map_page
from ..services.postgrest_client import PostgrestClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/papers")

#: The list has always been the hundred newest. The pillar query filters that
#: set in the page; it is not a second query, so the client-side chips can
#: switch pillars without another round trip.
LIBRARY_LIMIT = 100

_LOAD_ERROR = (
    "The papers list could not be loaded. This does not mean the library is empty."
)


@router.get("/", response_class=HTMLResponse)
async def library_page(
    request: Request, pillar: Optional[str] = None
) -> HTMLResponse:
    """Render the library. ``?pillar=`` still narrows the list."""
    client = PostgrestClient()
    load_error = None
    try:
        papers = await client.list_papers_checked(limit=LIBRARY_LIMIT)
    except Exception as exc:
        logger.error("Library could not load papers: %s", exc)
        papers = []
        load_error = _LOAD_ERROR
    lesson_ids, card_ids, episode_ids = await asyncio.gather(
        client.paper_ids_with_rows("lessons"),
        client.paper_ids_with_rows("quiz_cards"),
        client.paper_ids_with_rows("podcast_scripts"),
    )
    view = build_library(
        papers,
        get_pillars_or_empty(limit=100),
        lesson_ids=lesson_ids,
        card_ids=card_ids,
        episode_ids=episode_ids,
        selected_pillar=pillar or "",
    )
    return request.app.state.templates.TemplateResponse(
        request,
        "papers.html",
        {"view": view, "load_error": load_error, "pillar": pillar},
    )


async def _paper_counts_by_pillar(
    client: PostgrestClient, pillars: list
) -> Optional[dict]:
    """Library papers per pillar, or None when they could not be counted.

    None, not zeros: a count that failed must not disable a pillar's chip.
    """
    ids = [str(getattr(p, "id", None) or (p.get("id") if isinstance(p, dict) else "")) for p in pillars]
    ids = [pid for pid in ids if pid]
    if not ids:
        return {}
    try:
        return await client.paper_counts_for_pillars(ids)
    except Exception as exc:
        logger.warning("Map page could not count papers per pillar: %s", exc)
        return None


@router.get("/map", response_class=HTMLResponse)
@router.get("/map/", response_class=HTMLResponse, include_in_schema=False)
async def paper_map_page(
    request: Request, pillar: Optional[str] = None
) -> HTMLResponse:
    """Library > Map. Registered before ``/{paper_id:path}``, which would take "map".

    The page is a shell; the map itself is ``GET /api/map?scope=<pillar>``,
    fetched in the browser so a slow or failed read has somewhere honest to show.
    """
    pillars = get_pillars_or_empty(limit=100)
    counts = await _paper_counts_by_pillar(PostgrestClient(), list(pillars))
    page = build_map_page(pillars, counts, pillar)
    return request.app.state.templates.TemplateResponse(
        request,
        "paper_map.html",
        {"page": page, "title": "Paper map"},
    )


@router.get("/details/{paper_id}")
async def get_paper_details(paper_id: str) -> JSONResponse:
    """Get detailed paper information including notes, lessons, and quiz cards."""
    client = PostgrestClient()
    details = await client.get_paper_details(paper_id)

    if not details:
        raise HTTPException(status_code=404, detail="Paper not found")

    return JSONResponse(content=details)


@router.get("/{paper_id:path}", response_class=HTMLResponse)
async def paper_page(request: Request, paper_id: str) -> HTMLResponse:
    """Render one paper. Registered after ``/details`` so that route still wins."""
    client = PostgrestClient()
    details = await client.get_paper_details(paper_id)
    if not details:
        return request.app.state.templates.TemplateResponse(
            request,
            "paper.html",
            {"found": False, "paper_id": paper_id, "page": None},
            status_code=404,
        )
    episodes: list = []
    episodes_unavailable = False
    try:
        episodes = await client.scripts_for_paper(paper_id)
    except Exception as exc:
        logger.error("Paper page could not load episodes for %s: %s", paper_id, exc)
        episodes_unavailable = True
    page = build_paper_page(
        details,
        episodes,
        get_pillars_or_empty(limit=100),
        episodes_unavailable=episodes_unavailable,
    )
    return request.app.state.templates.TemplateResponse(
        request,
        "paper.html",
        {"found": True, "paper_id": paper_id, "page": page},
    )
