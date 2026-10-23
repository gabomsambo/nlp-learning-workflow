"""
Frontend router for Pillar management pages.
Serves Jinja2 templates for pillar CRUD operations.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

from nlp_pillars.db import (
    get_pillars_or_empty,
    get_pillar_by_id,
    get_papers,
    get_lessons,
    get_quiz_cards_for_review
)
from ..services.postgrest_client import CountUnavailableError, PostgrestClient

# Progress-bar targets for the pillar detail page. These are display goals, not
# limits: nothing enforces them and a pillar may sail past any of them, which is
# why the template clamps the bar rather than letting it overflow.
PROGRESS_GOALS = {
    "papers": 100,
    "lessons": 50,
    "quiz_cards": 200,
}
# PillarID enum no longer used - using dynamic string IDs

router = APIRouter(prefix="/pillars", tags=["pillar_pages"])


async def _progress_by_pillar(pillar_ids: list) -> Dict[str, Optional[Dict[str, int]]]:
    """Papers / lessons / cards per pillar, or None when the count could not be read."""

    async def one(pillar_id: str) -> Optional[Dict[str, int]]:
        try:
            return await PostgrestClient().counts_for_pillar(pillar_id)
        except CountUnavailableError as e:
            logger.warning(f"Progress counts unavailable for pillar {pillar_id}: {e}")
            return None

    if not pillar_ids:
        return {}
    counts = await asyncio.gather(*(one(pillar_id) for pillar_id in pillar_ids))
    return dict(zip(pillar_ids, counts))


@router.get("/", response_class=HTMLResponse)
async def pillars_page(request: Request) -> HTMLResponse:
    """
    Display the pillars management page.
    Shows all pillars with options to create, edit, and delete.
    """
    try:
        # Get all pillars
        pillars = get_pillars_or_empty(limit=100)
        # Per-pillar totals. A failure is None, never 0: an empty pillar and a
        # database that could not answer are different facts. counts_by_pillar()
        # degrades to 0, so it is the wrong call here.
        progress = await _progress_by_pillar([pillar.id for pillar in pillars])

        # Render template
        templates = request.app.state.templates
        return templates.TemplateResponse(
            request,
            "pillars.html",
            {
                "pillars": pillars,
                "progress": progress,
                "now": datetime.now(timezone.utc)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load pillars page: {str(e)}")


@router.get("/{pillar_id}", response_class=HTMLResponse)
async def pillar_detail_page(request: Request, pillar_id: str) -> HTMLResponse:
    """
    Display the pillar detail page.
    Shows comprehensive information about a specific pillar.
    """
    try:
        # Get pillar details
        pillar = get_pillar_by_id(pillar_id)
        if pillar is None:
            raise HTTPException(status_code=404, detail=f"Pillar '{pillar_id}' not found")
        
        # Get related data for the pillar using dynamic string pillar_id
        try:
            recent_papers = get_papers(pillar_id, limit=5)
            recent_lessons = get_lessons(pillar_id, limit=5)
            recent_quiz_cards = get_quiz_cards_for_review(pillar_id, limit=5)
        except Exception as e:
            logger.warning(f"Failed to fetch related data for pillar {pillar_id}: {e}")
            recent_papers = []
            recent_lessons = []
            recent_quiz_cards = []
        
        # Real per-pillar totals, counted server-side. These must not be derived
        # from the lists above: those are capped at 5, so "Papers Processed" could
        # never read higher than 5 no matter how many the pillar had.
        #
        # A failure here is NOT degraded to zero. The page renders an explicit
        # unknown state instead, because "0 papers" and "we could not ask" are
        # different facts and only one of them is a reason to worry.
        try:
            counts = await PostgrestClient().counts_for_pillar(pillar_id)
            stats = {
                'papers_count': counts['papers'],
                'lessons_count': counts['lessons'],
                'quiz_cards_count': counts['quiz_cards'],
            }
        except CountUnavailableError as e:
            logger.warning(f"Progress counts unavailable for pillar {pillar_id}: {e}")
            stats = None
        
        # Render template
        templates = request.app.state.templates
        return templates.TemplateResponse(
            request,
            "pillar_detail.html",
            {
                "pillar": pillar,
                "recent_papers": recent_papers,
                "recent_lessons": recent_lessons,
                "recent_quiz_cards": recent_quiz_cards,
                "stats": stats,
                "progress_goals": PROGRESS_GOALS,
                "now": datetime.now(timezone.utc)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load pillar detail: {str(e)}")


