"""
Quiz API endpoints for FSRS-based spaced repetition.
"""

from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/quiz", tags=["quiz"])


class ReviewRequest(BaseModel):
    card_id: str
    rating: int  # 1-4: Again, Hard, Good, Easy
    response_time_ms: Optional[int] = None
    session_id: Optional[str] = None


class OptimizationRequest(BaseModel):
    user_id: Optional[str] = "default_user"
    pillar_id: Optional[str] = None
    min_reviews: Optional[int] = 50


@router.post("/review")
async def review_card(review_request: ReviewRequest) -> JSONResponse:
    """Submit a quiz card review using FSRS algorithm."""
    try:
        # Import here to avoid circular imports
        from nlp_pillars.db import review_quiz_card_fsrs
        from nlp_pillars.schemas import QuizReviewRequest, FSRSRating
        
        # Convert rating to FSRS rating
        rating_map = {
            1: FSRSRating.AGAIN, 
            2: FSRSRating.HARD, 
            3: FSRSRating.GOOD, 
            4: FSRSRating.EASY
        }
        fsrs_rating = rating_map.get(review_request.rating)
        
        if not fsrs_rating:
            raise HTTPException(status_code=400, detail="Invalid rating. Must be 1-4.")
        
        # Create FSRS review request
        fsrs_request = QuizReviewRequest(
            card_id=review_request.card_id,
            rating=fsrs_rating,
            response_time_ms=review_request.response_time_ms,
            session_id=review_request.session_id
        )
        
        # Process the review
        result = review_quiz_card_fsrs(fsrs_request)
        
        if result.success:
            return JSONResponse({
                "success": True,
                "message": result.message,
                "next_review_date": result.next_review_date.isoformat(),
                "card_state": result.card.state.value if result.card else None
            })
        else:
            return JSONResponse({
                "success": False,
                "message": result.message
            }, status_code=400)

    except HTTPException:
        # Deliberate 4xx (e.g. the invalid-rating check above) must not be
        # swallowed by the catch-all below and re-reported as a 500.
        raise
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Review failed: {str(e)}"
        }, status_code=500)


@router.get("/cards/due")
async def get_cards_due_for_review(
    user_id: str = "default_user",
    pillar: Optional[str] = None,
    limit: int = 10
) -> JSONResponse:
    """Get quiz cards due for review."""
    try:
        from nlp_pillars.db import get_cards_for_review

        # Get cards due for review - pillar is now a string ID
        cards = get_cards_for_review(
            user_id=user_id,
            pillar_id=pillar,  # Pass string pillar_id directly (can be None)
            limit=limit
        )

        # Convert to JSON-serializable format
        cards_data = []
        for card in cards:
            cards_data.append({
                "id": card.id,
                "paper_id": card.paper_id,
                "pillar_id": card.pillar_id,  # Already a string
                "question": card.question,
                "answer": card.answer,
                "difficulty": card.difficulty,
                "state": card.state.value,
                "next_review_date": card.next_review_date.isoformat() if card.next_review_date else None,
                "last_review_date": card.last_review_date.isoformat() if card.last_review_date else None
            })
        
        return JSONResponse({
            "success": True,
            "cards": cards_data,
            "count": len(cards_data)
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.post("/optimize")
async def optimize_fsrs_parameters(request: OptimizationRequest) -> JSONResponse:
    """Trigger FSRS parameter optimization for a user."""
    try:
        from nlp_pillars.services.background_jobs import get_background_jobs_service
        
        # Get background jobs service
        jobs_service = get_background_jobs_service()
        
        # Trigger optimization
        result = jobs_service.trigger_fsrs_optimization_now(
            user_id=request.user_id,
            pillar_id=request.pillar_id
        )
        
        return JSONResponse(result)
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.get("/stats")
async def get_fsrs_stats(user_id: str = "default_user") -> JSONResponse:
    """Get FSRS optimization statistics."""
    try:
        from nlp_pillars.services.fsrs_optimizer import get_fsrs_optimizer
        
        optimizer = get_fsrs_optimizer()
        stats = optimizer.get_optimization_stats()
        
        return JSONResponse({
            "success": True,
            "stats": stats
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.get("/jobs/status")
async def get_job_status() -> JSONResponse:
    """Get status of background optimization jobs."""
    try:
        from nlp_pillars.services.background_jobs import get_background_jobs_service
        
        jobs_service = get_background_jobs_service()
        status = jobs_service.get_job_status()
        
        return JSONResponse({
            "success": True,
            "status": status
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)
