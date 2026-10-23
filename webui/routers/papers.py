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
