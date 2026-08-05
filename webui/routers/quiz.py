from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from nlp_pillars.db import get_pillars
from ..services.postgrest_client import PostgrestClient


router = APIRouter(prefix="/quiz")


@router.get("/", response_class=HTMLResponse)
async def quiz_home(request: Request, pillar: Optional[str] = None, difficulty: Optional[str] = None) -> HTMLResponse:
    """Display quiz cards for review using FSRS algorithm."""
    try:
        # Import here to avoid circular imports
        from nlp_pillars.db import get_cards_for_review

        # Get cards due for review using FSRS
        # pillar is now a string ID directly
        cards = get_cards_for_review(
            user_id="default_user",
            pillar_id=pillar,  # Pass string pillar_id directly (can be None)
            limit=20
        )
        
        # Filter by difficulty if specified
        if difficulty:
            diff_map = {"easy": 1, "medium": 2, "hard": 3}
            target_difficulty = diff_map.get(difficulty.lower())
            if target_difficulty:
                cards = [c for c in cards if c.difficulty == target_difficulty]
        
        pillars = get_pillars(limit=100)
        return request.app.state.templates.TemplateResponse(
            request,
            "quiz.html",
            {"cards": cards, "pillar": pillar, "difficulty": difficulty, "pillars": pillars},
        )

    except Exception as e:
        # Fallback to old method if FSRS fails
        client = PostgrestClient()
        diff_map = {"easy": 1, "medium": 2, "hard": 3}
        target_difficulty = diff_map.get(difficulty.lower(), None) if difficulty else None
        cards = await client.list_quiz_cards(pillar=pillar, difficulty=target_difficulty, limit=30)
        pillars = get_pillars(limit=100)
        return request.app.state.templates.TemplateResponse(
            request,
            "quiz.html",
            {"cards": cards, "pillar": pillar, "difficulty": difficulty, "pillars": pillars},
        )





