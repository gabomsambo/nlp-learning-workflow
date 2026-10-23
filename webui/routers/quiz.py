from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from nlp_pillars.db import (
    QuizLookupError,
    get_cards_for_review,
    get_pillars_or_empty,
    list_quiz_schedule_rows,
)
from nlp_pillars.schemas import FSRSRating
from nlp_pillars.services.fsrs_service import get_fsrs_service
from ..services.postgrest_client import PostgrestClient
from ..services.review_session import (
    difficulty_number,
    earliest_future,
    empty_copy,
    json_for_script,
    session_cards,
)

router = APIRouter(prefix="/quiz")

# The page shows at most this many due cards, matching the previous quiz page.
# The fetch is wider because get_cards_for_review applies its limit before it
# drops cards that are not due yet.
_SESSION_LIMIT = 20
_FETCH_LIMIT = 200

_DIFFICULTY = {"easy": 1, "medium": 2, "hard": 3}


def _reason(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    line = text[0] if text else exc.__class__.__name__
    return line[:240]


def _target_difficulty(difficulty: Optional[str]) -> Optional[int]:
    if not difficulty:
        return None
    return _DIFFICULTY.get(difficulty.lower())


def preview_schedule(card, rating: int) -> datetime:
    """Next due time for one rating. Does not write the card or a review log.

    This is the same ``schedule_card_review`` the rating POST uses, called on
    a copy the caller already made.
    """
    _updated, due = get_fsrs_service().schedule_card_review(card, FSRSRating(rating))
    return due


@router.get("/", response_class=HTMLResponse)
async def quiz_home(request: Request, pillar: Optional[str] = None, difficulty: Optional[str] = None) -> HTMLResponse:
    """One-card review session for whatever is due."""
    pillars = get_pillars_or_empty(limit=100)
    names = {p.id: p.name for p in pillars}
    target = _target_difficulty(difficulty)
    now = datetime.now(timezone.utc)

    fsrs_fallback = False
    fallback_reason = None
    schedule = preview_schedule
    try:
        cards = get_cards_for_review(
            user_id="default_user",
            pillar_id=pillar,
            limit=_FETCH_LIMIT,
            raise_on_error=True,
        )
        if target is not None:
            cards = [card for card in cards if difficulty_number(card.difficulty) == target]
        cards = cards[:_SESSION_LIMIT]
    except QuizLookupError as exc:
        fsrs_fallback = True
        fallback_reason = _reason(exc)
        schedule = None
        client = PostgrestClient()
        cards = await client.list_quiz_cards(pillar=pillar, difficulty=target, limit=30)
    except Exception as exc:
        # Anything else that used to fall through to the full list still does,
        # and the page says so instead of presenting that list as the due queue.
        fsrs_fallback = True
        fallback_reason = _reason(exc)
        schedule = None
        client = PostgrestClient()
        cards = await client.list_quiz_cards(pillar=pillar, difficulty=target, limit=30)

    filtered = bool(pillar or (difficulty and target is not None))
    next_due = None
    next_due_error = None
    if not cards and not fsrs_fallback:
        try:
            rows = list_quiz_schedule_rows(user_id="default_user", pillar_id=pillar)
            next_due = earliest_future(rows, now, difficulty=target)
        except QuizLookupError as exc:
            next_due_error = _reason(exc)

    if fsrs_fallback and not cards:
        empty_title, empty_message = (
            "No cards loaded",
            "The review queue could not be read, and the fallback list is empty.",
        )
    elif not cards:
        empty_title, empty_message = empty_copy(next_due=next_due, now=now, filtered=filtered)
    else:
        empty_title, empty_message = ("", "")

    payload = {
        "fsrsFallback": fsrs_fallback,
        "cards": session_cards(cards, now=now, schedule=schedule, pillar_names=names),
    }
    return request.app.state.templates.TemplateResponse(
        request,
        "quiz.html",
        {
            "pillar": pillar,
            "difficulty": difficulty,
            "pillars": pillars,
            "cards": payload["cards"],
            "session_json": json_for_script(payload),
            "fsrs_fallback": fsrs_fallback,
            "fallback_reason": fallback_reason,
            "next_due_error": next_due_error,
            "empty_title": empty_title,
            "empty_message": empty_message,
            "filtered": filtered,
        },
    )
