"""Lessons reading list.

The paper page's Lesson tab is the reading surface: a 680px serif column
with the takeaways, practice and connections. This page is the list that
opens that tab. A second reading route would repeat it.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from nlp_pillars.db import get_pillars_or_empty

from ..lessons_format import build_lesson_list
from ..services.postgrest_client import PostgrestClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/lessons")

#: The list has always been the hundred newest, narrowed by ``?pillar=``.
LESSONS_LIMIT = 100

_LOAD_ERROR = (
    "The lessons list could not be loaded. This does not mean there are none."
)
_TITLES_NOTE = (
    "Paper titles could not be loaded. Each lesson still shows its paper id."
)


@router.get("/", response_class=HTMLResponse)
async def lessons_page(request: Request, pillar: Optional[str] = None) -> HTMLResponse:
    """Render the reading list. ``?pillar=`` still narrows it."""
    client = PostgrestClient()
    load_error = None
    titles_note = None
    try:
        lessons = await client.list_lessons_checked(pillar=pillar, limit=LESSONS_LIMIT)
    except Exception as exc:
        logger.error("Lessons list could not be loaded: %s", exc)
        lessons = []
        load_error = _LOAD_ERROR
    titles: dict[str, str] = {}
    if lessons:
        paper_ids = [
            str(row.get("paper_id"))
            for row in lessons
            if isinstance(row, dict) and row.get("paper_id")
        ]
        try:
            titles = await client.paper_titles(paper_ids)
        except Exception as exc:
            logger.error("Lesson paper titles could not be loaded: %s", exc)
            titles_note = _TITLES_NOTE
    view = build_lesson_list(
        lessons,
        titles,
        get_pillars_or_empty(limit=100),
        selected_pillar=pillar or "",
    )
    return request.app.state.templates.TemplateResponse(
        request,
        "lessons.html",
        {
            "view": view,
            "load_error": load_error,
            "titles_note": titles_note,
            "pillar": pillar,
        },
    )
