"""FSRS Service - Handles spaced repetition scheduling using the FSRS algorithm.

This service provides:
1. FSRS-based quiz card scheduling
2. Review logging and parameter optimization  
3. Personalized spaced repetition parameters
4. Background optimization jobs
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..schemas import CardState, FSRSRating, QuizCard, ReviewLog, UserFSRSParameters

logger = logging.getLogger(__name__)

# Import FSRS libraries with graceful fallback
# Global flags for library availability
FSRS_AVAILABLE = False
FSRS_OPTIMIZER_AVAILABLE = False

try:
    from fsrs import Card, Rating, Scheduler, State
    from fsrs import ReviewLog as FSRSReviewLog
    FSRS_AVAILABLE = True
    logger.info("FSRS library loaded successfully")
except ImportError:
    logger.warning("fsrs library not available. FSRS scheduling will fall back to SM-2.")
    Scheduler = None
    Card = None
    Rating = None
    State = None
    FSRSReviewLog = None

try:
    from fsrs_optimizer import Optimizer
    FSRS_OPTIMIZER_AVAILABLE = True
except ImportError:
    logger.warning("fsrs-optimizer library not available. Parameter optimization will be disabled.")
    Optimizer = None


class FSRSService:
    """Service for handling FSRS-based spaced repetition."""

    def __init__(self):
        """Initialize the FSRS service."""
        global FSRS_AVAILABLE
        self.fsrs = None
        if FSRS_AVAILABLE:
            try:
                # Initialize with default parameters
                self.fsrs = Scheduler()
                logger.info("FSRS service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize FSRS: {e}")
                FSRS_AVAILABLE = False

    def schedule_card_review(self, card: QuizCard, rating: FSRSRating,
                           parameters: Optional[UserFSRSParameters] = None) -> Tuple[QuizCard, datetime]:
        """Schedule the next review for a card based on FSRS algorithm.
        
        Args:
        ----
            card: Quiz card to schedule
            rating: User's rating of recall difficulty
            parameters: User's personalized FSRS parameters
            
        Returns:
        -------
            Tuple of (updated_card, next_review_date)

        """
        if not FSRS_AVAILABLE:
            return self._fallback_sm2_scheduling(card, rating)

        try:
            # Get user parameters (for now, use default scheduler)
            # TODO: Implement custom parameters when FSRS supports it
            fsrs_scheduler = self.fsrs

            # Convert our card to FSRS Card object
            fsrs_card = self._convert_to_fsrs_card(card)

            # Convert rating to FSRS Rating
            fsrs_rating = self._convert_rating(rating)

            # Review the card
            reviewed_card, review_log = fsrs_scheduler.review_card(
                fsrs_card,
                fsrs_rating,
                review_datetime=datetime.now(timezone.utc)
            )

            # Update our card with new FSRS values
            updated_card = self._update_card_from_fsrs(card, reviewed_card)

            # Calculate days until due
            days_until_due = (reviewed_card.due - datetime.now(timezone.utc)).total_seconds() / (24 * 3600)
            logger.info(f"Scheduled card {card.id} with FSRS: next review in {days_until_due:.1f} days")

            return updated_card, reviewed_card.due

        except Exception as e:
            logger.error(f"FSRS scheduling failed: {e}, falling back to SM-2")
            return self._fallback_sm2_scheduling(card, rating)

    def create_review_log(self, card: QuizCard, rating: FSRSRating,
                         response_time_ms: Optional[int] = None,
                         session_id: Optional[str] = None) -> ReviewLog:
        """Create a review log entry for optimization.
        
        Args:
        ----
            card: Quiz card that was reviewed
            rating: User's rating of recall difficulty
            response_time_ms: Time taken to answer
            session_id: Review session identifier
            
        Returns:
        -------
            ReviewLog entry

        """
        now = datetime.now(timezone.utc)

        # Calculate days overdue
        days_overdue = 0
        if card.next_review_date:
            days_overdue = max(0, (now - card.next_review_date).days)

        return ReviewLog(
            id=str(uuid.uuid4()),
            card_id=str(card.id),
            user_id=card.user_id,
            pillar_id=card.pillar_id,
            paper_id=card.paper_id,
            rating=rating,
            review_timestamp=now,
            difficulty=card.difficulty_fsrs,
            stability=card.stability,
            retrievability=card.retrievability,
            previous_due_date=card.next_review_date,
            days_overdue=days_overdue,
            session_id=session_id,
            response_time_ms=response_time_ms
        )

    def optimize_user_parameters(self, review_logs: List[ReviewLog],
                                current_parameters: Optional[UserFSRSParameters] = None) -> Optional[UserFSRSParameters]:
        """Optimize FSRS parameters based on user's review history.
        
        Args:
        ----
            review_logs: List of review logs for optimization
            current_parameters: Current parameters to improve upon
            
        Returns:
        -------
            Optimized parameters or None if optimization failed

        """
        if not FSRS_OPTIMIZER_AVAILABLE:
            logger.warning("FSRS optimizer not available, cannot optimize parameters")
            return None

        if len(review_logs) < 50:  # Minimum reviews for meaningful optimization
            logger.info(f"Not enough reviews ({len(review_logs)}) for optimization, need at least 50")
            return None

        try:
            # Convert review logs to FSRS optimizer format
            optimizer_logs = self._convert_review_logs_for_optimizer(review_logs)

            if not optimizer_logs:
                logger.warning("No valid review logs for optimization")
                return None

            # Run optimization
            optimizer = Optimizer()
            result = optimizer.optimize(optimizer_logs)

            if result and hasattr(result, 'weights'):
                # Create new parameters from optimized weights
                user_id = review_logs[0].user_id if review_logs else "default_user"
                pillar_id = review_logs[0].pillar_id if review_logs else None

                optimized_params = UserFSRSParameters.from_weights_list(
                    weights=result.weights,
                    user_id=user_id,
                    pillar_id=pillar_id
                )

                optimized_params.review_count = len(review_logs)
                optimized_params.last_optimized = datetime.now(timezone.utc)
                optimized_params.optimization_score = getattr(result, 'score', None)

                logger.info(f"Optimized FSRS parameters for user {user_id} using {len(review_logs)} reviews")
                return optimized_params

        except Exception as e:
            logger.error(f"FSRS optimization failed: {e}")

        return None

    def get_default_parameters(self, user_id: str = "default_user",
                              pillar_id: Optional[str] = None) -> UserFSRSParameters:
        """Get default FSRS parameters for a user.
        
        Args:
        ----
            user_id: User identifier
            pillar_id: Optional pillar-specific parameters
            
        Returns:
        -------
            Default UserFSRSParameters

        """
        return UserFSRSParameters(
            user_id=user_id,
            pillar_id=pillar_id,
            is_default=True
        )

    def _convert_to_fsrs_card(self, card: QuizCard) -> 'Card':
        """Convert our QuizCard to FSRS Card object."""
        if not FSRS_AVAILABLE:
            raise RuntimeError("FSRS not available")

        # Create FSRS card - let FSRS manage the state internally
        fsrs_card = Card()

        # Set the basic properties if available
        if card.next_review_date:
            fsrs_card.due = card.next_review_date
        if card.difficulty_fsrs is not None and card.difficulty_fsrs > 0:
            fsrs_card.difficulty = card.difficulty_fsrs
        if card.stability is not None and card.stability > 0:
            fsrs_card.stability = card.stability
        if card.last_review_date:
            fsrs_card.last_review = card.last_review_date

        # Convert our state to FSRS state (FSRS doesn't have a "New" state)
        if card.state == CardState.NEW or card.state == CardState.LEARNING:
            fsrs_card.state = State.Learning
        elif card.state == CardState.REVIEW:
            fsrs_card.state = State.Review
        elif card.state == CardState.RELEARNING:
            fsrs_card.state = State.Relearning
        else:
            # Default to Learning for uninitialized cards
            fsrs_card.state = State.Learning

        return fsrs_card

    def _update_card_from_fsrs(self, original_card: QuizCard, fsrs_card: 'Card') -> QuizCard:
        """Update our QuizCard with values from FSRS Card."""
        # Convert state back
        if fsrs_card.state == State.Learning:
            # Determine if it's truly new or learning based on review count
            state = CardState.NEW if original_card.review_count == 0 else CardState.LEARNING
        elif fsrs_card.state == State.Review:
            state = CardState.REVIEW
        elif fsrs_card.state == State.Relearning:
            state = CardState.RELEARNING
        else:
            state = CardState.NEW

        # Create updated card
        updated_card = original_card.model_copy()
        updated_card.difficulty_fsrs = fsrs_card.difficulty if fsrs_card.difficulty else 0.0
        updated_card.stability = fsrs_card.stability if fsrs_card.stability else 0.0
        updated_card.state = state
        updated_card.last_review_date = datetime.now(timezone.utc)
        updated_card.next_review_date = fsrs_card.due
        updated_card.review_count = updated_card.review_count + 1

        # Update legacy fields for backward compatibility
        updated_card.last_reviewed = updated_card.last_review_date
        updated_card.due_date = fsrs_card.due

        # Calculate days until next review for legacy field
        days_until_review = (fsrs_card.due - datetime.now(timezone.utc)).total_seconds() / (24 * 3600)
        updated_card.interval_days = max(1, int(days_until_review))

        return updated_card

    def _convert_rating(self, rating: FSRSRating) -> 'Rating':
        """Convert our FSRSRating to FSRS Rating."""
        if not FSRS_AVAILABLE:
            raise RuntimeError("FSRS not available")

        if rating == FSRSRating.AGAIN:
            return Rating.Again
        elif rating == FSRSRating.HARD:
            return Rating.Hard
        elif rating == FSRSRating.GOOD:
            return Rating.Good
        elif rating == FSRSRating.EASY:
            return Rating.Easy
        else:
            return Rating.Good  # Default

    def _convert_review_logs_for_optimizer(self, review_logs: List[ReviewLog]) -> List[Dict[str, Any]]:
        """Convert our review logs to format expected by FSRS optimizer."""
        optimizer_logs = []

        for log in review_logs:
            try:
                optimizer_log = {
                    'card_id': log.card_id,
                    'review_time': log.review_timestamp.isoformat(),
                    'review_rating': int(log.rating),
                    'review_state': 0,  # Will be determined by optimizer
                    'review_duration': log.response_time_ms or 0
                }
                optimizer_logs.append(optimizer_log)
            except Exception as e:
                logger.warning(f"Failed to convert review log {log.id}: {e}")
                continue

        return optimizer_logs

    def _fallback_sm2_scheduling(self, card: QuizCard, rating: FSRSRating) -> Tuple[QuizCard, datetime]:
        """Fallback to SM-2 algorithm when FSRS is not available.
        
        This maintains the existing SM-2 logic from the original system.
        """
        logger.debug(f"Using SM-2 fallback for card {card.id}")

        # Convert FSRS rating to SM-2 quality (0-5 scale)
        if rating == FSRSRating.AGAIN:
            quality = 0
        elif rating == FSRSRating.HARD:
            quality = 3
        elif rating == FSRSRating.GOOD:
            quality = 4
        elif rating == FSRSRating.EASY:
            quality = 5
        else:
            quality = 4  # Default to Good

        # SM-2 algorithm implementation
        updated_card = card.model_copy()

        if quality < 3:
            updated_card.interval_days = 1
            updated_card.repetitions = 0
        else:
            if updated_card.repetitions == 0:
                updated_card.interval_days = 1
            elif updated_card.repetitions == 1:
                updated_card.interval_days = 6
            else:
                updated_card.interval_days = int(updated_card.interval_days * updated_card.ease_factor)

            updated_card.repetitions += 1

            # Update ease factor
            ease_factor = updated_card.ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            updated_card.ease_factor = max(1.3, ease_factor)

        # Update review dates
        now = datetime.now(timezone.utc)
        updated_card.last_reviewed = now
        updated_card.last_review_date = now
        next_review = now + timedelta(days=updated_card.interval_days)
        updated_card.due_date = next_review
        updated_card.next_review_date = next_review
        updated_card.review_count += 1

        return updated_card, next_review

    def is_fsrs_available(self) -> bool:
        """Check if FSRS library is available."""
        return FSRS_AVAILABLE

    def is_optimizer_available(self) -> bool:
        """Check if FSRS optimizer is available."""
        return FSRS_OPTIMIZER_AVAILABLE


# Global service instance
_fsrs_service: Optional[FSRSService] = None


def get_fsrs_service() -> FSRSService:
    """Get or create the global FSRS service instance."""
    global _fsrs_service

    if _fsrs_service is None:
        _fsrs_service = FSRSService()

    return _fsrs_service


def set_fsrs_service(service: FSRSService) -> None:
    """Set the global FSRS service instance (for testing)."""
    global _fsrs_service
    _fsrs_service = service
