"""FSRS Optimizer Service - Handles parameter optimization for personalized spaced repetition.

This service provides:
1. Background optimization of FSRS parameters based on user review history
2. Scheduled optimization jobs
3. Performance analysis and recommendations
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..db import get_review_logs, get_user_fsrs_parameters, upsert_user_fsrs_parameters
from ..pillar_utils import get_all_pillar_ids
from ..schemas import (
    FSRSOptimizationRequest,
    FSRSOptimizationResponse,
    ReviewLog,
    UserFSRSParameters,
)
from .fsrs_service import get_fsrs_service

logger = logging.getLogger(__name__)


class FSRSOptimizer:
    """Service for optimizing FSRS parameters based on user review history."""

    def __init__(self):
        """Initialize the FSRS optimizer."""
        self.fsrs_service = get_fsrs_service()

    def optimize_user_parameters(self, request: FSRSOptimizationRequest) -> FSRSOptimizationResponse:
        """Optimize FSRS parameters for a user based on their review history.
        
        Args:
        ----
            request: Optimization request with user/pillar details
            
        Returns:
        -------
            FSRSOptimizationResponse with optimization results

        """
        try:
            logger.info(f"Starting FSRS optimization for user {request.user_id}, pillar {request.pillar_id}")

            # Get review logs for the user/pillar
            review_logs = get_review_logs(
                user_id=request.user_id,
                pillar_id=request.pillar_id,
                limit=2000  # Get more data for better optimization
            )

            if len(review_logs) < request.min_reviews:
                return FSRSOptimizationResponse(
                    success=False,
                    parameters=None,
                    improvement_score=None,
                    reviews_analyzed=len(review_logs),
                    message=f"Insufficient review history: {len(review_logs)} < {request.min_reviews} required"
                )

            # Get current parameters for comparison
            current_params = get_user_fsrs_parameters(request.user_id, request.pillar_id)

            # Optimize using FSRS service
            optimized_params = self.fsrs_service.optimize_user_parameters(review_logs, current_params)

            if not optimized_params:
                return FSRSOptimizationResponse(
                    success=False,
                    parameters=None,
                    improvement_score=None,
                    reviews_analyzed=len(review_logs),
                    message="Optimization failed - insufficient data or optimizer unavailable"
                )

            # Calculate improvement score
            improvement_score = self._calculate_improvement_score(
                review_logs, current_params, optimized_params
            )

            # Save optimized parameters
            success = upsert_user_fsrs_parameters(optimized_params)

            if not success:
                return FSRSOptimizationResponse(
                    success=False,
                    parameters=optimized_params,
                    improvement_score=improvement_score,
                    reviews_analyzed=len(review_logs),
                    message="Optimization succeeded but failed to save parameters"
                )

            logger.info(f"FSRS optimization completed for user {request.user_id}: "
                       f"{len(review_logs)} reviews analyzed, improvement score: {improvement_score:.3f}")

            return FSRSOptimizationResponse(
                success=True,
                parameters=optimized_params,
                improvement_score=improvement_score,
                reviews_analyzed=len(review_logs),
                message=f"Optimization successful using {len(review_logs)} reviews"
            )

        except Exception as e:
            logger.error(f"FSRS optimization failed for user {request.user_id}: {e}")
            return FSRSOptimizationResponse(
                success=False,
                parameters=None,
                improvement_score=None,
                reviews_analyzed=0,
                message=f"Optimization failed: {str(e)}"
            )

    def optimize_all_users(self, min_reviews: int = 50) -> Dict[str, FSRSOptimizationResponse]:
        """Optimize FSRS parameters for all users with sufficient review history.
        
        Args:
        ----
            min_reviews: Minimum reviews required for optimization
            
        Returns:
        -------
            Dictionary mapping user_id to optimization results

        """
        try:
            # This is a simplified implementation - in a real system, you'd
            # query the database to find all users with sufficient review history

            results = {}

            # For now, just optimize the default user
            # Get all pillar IDs dynamically from database
            pillar_ids = get_all_pillar_ids()
            users_to_optimize = [("default_user", None)]  # Global parameters
            users_to_optimize.extend([("default_user", pid) for pid in pillar_ids])

            for user_id, pillar_id in users_to_optimize:
                request = FSRSOptimizationRequest(
                    user_id=user_id,
                    pillar_id=pillar_id,
                    min_reviews=min_reviews
                )

                result = self.optimize_user_parameters(request)
                key = f"{user_id}_{pillar_id if pillar_id else 'global'}"
                results[key] = result

                if result.success:
                    logger.info(f"Optimized parameters for {key}: {result.reviews_analyzed} reviews")
                else:
                    logger.debug(f"Skipped optimization for {key}: {result.message}")

            return results

        except Exception as e:
            logger.error(f"Batch optimization failed: {e}")
            return {}

    def should_optimize_user(self, user_id: str, pillar_id: Optional[str] = None) -> bool:
        """Check if a user's parameters should be optimized.
        
        Args:
        ----
            user_id: User identifier
            pillar_id: Optional pillar-specific check
            
        Returns:
        -------
            True if optimization is recommended

        """
        try:
            # Get current parameters
            params = get_user_fsrs_parameters(user_id, pillar_id)

            if not params:
                # No parameters exist, optimization would create them
                review_logs = get_review_logs(user_id, pillar_id, limit=100)
                return len(review_logs) >= 50

            # Check if parameters are old
            if params.last_optimized:
                days_since_optimization = (datetime.now(timezone.utc) - params.last_optimized).days
                if days_since_optimization < 7:  # Don't optimize too frequently
                    return False

            # Check if user has new reviews since last optimization
            review_logs = get_review_logs(user_id, pillar_id, limit=100)

            if params.last_optimized:
                # Count reviews since last optimization
                new_reviews = [
                    log for log in review_logs
                    if log.review_timestamp > params.last_optimized
                ]
                return len(new_reviews) >= 20  # Optimize if 20+ new reviews
            else:
                # Never optimized, check if enough total reviews
                return len(review_logs) >= 50

        except Exception as e:
            logger.error(f"Failed to check optimization status for user {user_id}: {e}")
            return False

    def get_optimization_candidates(self) -> List[Tuple[str, Optional[str]]]:
        """Get list of user/pillar combinations that should be optimized.

        Returns
        -------
            List of (user_id, pillar_id) tuples ready for optimization

        """
        candidates = []

        # For this implementation, we'll check the default user across pillars
        # In a real system, you'd query the database for all active users

        # Get all pillar IDs dynamically from database
        pillar_ids = get_all_pillar_ids()
        user_pillar_combinations = [("default_user", None)]  # Global parameters
        user_pillar_combinations.extend([("default_user", pid) for pid in pillar_ids])

        for user_id, pillar_id in user_pillar_combinations:
            if self.should_optimize_user(user_id, pillar_id):
                candidates.append((user_id, pillar_id))

        logger.info(f"Found {len(candidates)} optimization candidates")
        return candidates

    def _calculate_improvement_score(self, review_logs: List[ReviewLog],
                                   current_params: Optional[UserFSRSParameters],
                                   optimized_params: UserFSRSParameters) -> float:
        """Calculate how much the optimized parameters improve over current ones.
        
        This is a simplified implementation that compares parameter differences.
        A more sophisticated approach would simulate prediction accuracy.
        
        Args:
        ----
            review_logs: Review history used for optimization
            current_params: Previous parameters (if any)
            optimized_params: New optimized parameters
            
        Returns:
        -------
            Improvement score (higher is better)

        """
        try:
            if not current_params:
                # No previous parameters, any optimization is an improvement
                return 1.0

            if optimized_params.optimization_score:
                # Use the optimizer's own score if available
                return optimized_params.optimization_score

            # Calculate parameter distance as a proxy for improvement
            current_weights = current_params.to_weights_list()
            optimized_weights = optimized_params.to_weights_list()

            # Calculate mean squared difference
            mse = sum((c - o) ** 2 for c, o in zip(current_weights, optimized_weights)) / len(current_weights)

            # Convert to improvement score (more change = more improvement, capped at 1.0)
            improvement = min(1.0, mse)

            return improvement

        except Exception as e:
            logger.warning(f"Failed to calculate improvement score: {e}")
            return 0.5  # Default moderate improvement

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about FSRS optimization status.
        
        Returns
        -------
            Dictionary with optimization statistics

        """
        try:
            stats = {
                "optimizer_available": self.fsrs_service.is_optimizer_available(),
                "fsrs_available": self.fsrs_service.is_fsrs_available(),
                "candidates_for_optimization": len(self.get_optimization_candidates()),
                "default_user_stats": {}
            }

            # Get stats for default user
            # Get all pillar IDs dynamically from database
            pillar_ids = get_all_pillar_ids()
            pillar_list = [None] + pillar_ids
            for pillar_id in pillar_list:
                key = pillar_id if pillar_id else "global"

                params = get_user_fsrs_parameters("default_user", pillar_id)
                review_logs = get_review_logs("default_user", pillar_id, limit=100)

                stats["default_user_stats"][key] = {
                    "has_parameters": params is not None,
                    "review_count": len(review_logs),
                    "last_optimized": params.last_optimized.isoformat() if params and params.last_optimized else None,
                    "optimization_score": params.optimization_score if params else None,
                    "ready_for_optimization": self.should_optimize_user("default_user", pillar_id)
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get optimization stats: {e}")
            return {"error": str(e)}


# Global optimizer instance
_fsrs_optimizer: Optional[FSRSOptimizer] = None


def get_fsrs_optimizer() -> FSRSOptimizer:
    """Get or create the global FSRS optimizer instance."""
    global _fsrs_optimizer

    if _fsrs_optimizer is None:
        _fsrs_optimizer = FSRSOptimizer()

    return _fsrs_optimizer


def set_fsrs_optimizer(optimizer: FSRSOptimizer) -> None:
    """Set the global FSRS optimizer instance (for testing)."""
    global _fsrs_optimizer
    _fsrs_optimizer = optimizer
