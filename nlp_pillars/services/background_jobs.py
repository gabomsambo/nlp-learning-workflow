"""Background Jobs Service - Handles scheduled tasks for the NLP Learning Workflow.

This service provides:
1. FSRS parameter optimization jobs
2. Scheduled maintenance tasks
3. Job scheduling and management using APScheduler
"""

import logging
from datetime import timezone
from typing import Any, Callable, Dict, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from ..schemas import FSRSOptimizationRequest
from .fsrs_optimizer import get_fsrs_optimizer

logger = logging.getLogger(__name__)


class BackgroundJobsService:
    """Service for managing background jobs and scheduled tasks."""

    def __init__(self):
        """Initialize the background jobs service."""
        self.scheduler = BackgroundScheduler(timezone=timezone.utc)
        self.fsrs_optimizer = get_fsrs_optimizer()
        self.is_running = False

    def start(self) -> None:
        """Start the background job scheduler."""
        try:
            if not self.is_running:
                # Add FSRS optimization jobs
                self._setup_fsrs_optimization_jobs()

                # Add maintenance jobs
                self._setup_maintenance_jobs()

                # Start the scheduler
                self.scheduler.start()
                self.is_running = True
                logger.info("Background jobs service started successfully")
            else:
                logger.warning("Background jobs service is already running")

        except Exception as e:
            logger.error(f"Failed to start background jobs service: {e}")
            raise

    def stop(self) -> None:
        """Stop the background job scheduler."""
        try:
            if self.is_running:
                self.scheduler.shutdown(wait=True)
                self.is_running = False
                logger.info("Background jobs service stopped")
            else:
                logger.warning("Background jobs service is not running")

        except Exception as e:
            logger.error(f"Failed to stop background jobs service: {e}")

    def _setup_fsrs_optimization_jobs(self) -> None:
        """Set up FSRS parameter optimization jobs."""
        try:
            # Daily optimization job - runs at 3 AM UTC
            self.scheduler.add_job(
                func=self._run_fsrs_optimization,
                trigger=CronTrigger(hour=3, minute=0),
                id='daily_fsrs_optimization',
                name='Daily FSRS Parameter Optimization',
                replace_existing=True,
                max_instances=1  # Prevent overlapping runs
            )

            # Weekly comprehensive optimization - runs Sundays at 2 AM UTC
            self.scheduler.add_job(
                func=self._run_comprehensive_fsrs_optimization,
                trigger=CronTrigger(day_of_week=6, hour=2, minute=0),  # Sunday = 6
                id='weekly_fsrs_optimization',
                name='Weekly Comprehensive FSRS Optimization',
                replace_existing=True,
                max_instances=1
            )

            logger.info("FSRS optimization jobs scheduled successfully")

        except Exception as e:
            logger.error(f"Failed to setup FSRS optimization jobs: {e}")

    def add_daily_pillar_run_job(self, func: Callable[[], Any], hour: int, minute: int,
                                 tzinfo: Any, job_id: str = 'daily_pillar_run',
                                 name: str = 'Daily Pillar Learning Run') -> None:
        """Register the daily learning run on this service's scheduler.

        Kept generic (the caller supplies `func`) so this module does not import
        the pipeline: `nlp_pillars.scheduler` owns the work, this owns the timing.
        Call before or after `start()` — APScheduler accepts both.

        The trigger carries its own `timezone`, so the daily run honours
        SCHEDULE_TIMEZONE while the FSRS jobs above stay on the UTC times they
        were always written for.

        Args:
        ----
            func: Zero-argument callable to run
            hour: Hour of day, 0-23, in `tzinfo`
            minute: Minute of hour, 0-59, in `tzinfo`
            tzinfo: tzinfo instance the cron fields are interpreted in
            job_id: APScheduler job id
            name: Human-readable job name

        """
        self.scheduler.add_job(
            func=func,
            trigger=CronTrigger(hour=hour, minute=minute, timezone=tzinfo),
            id=job_id,
            name=name,
            replace_existing=True,
            # A run that overruns 24h must not stack a second one on top of it.
            max_instances=1,
        )
        logger.info(f"Daily pillar run scheduled at {hour:02d}:{minute:02d} {tzinfo}")

    def _setup_maintenance_jobs(self) -> None:
        """Set up general maintenance jobs."""
        try:
            # Log optimization statistics - runs every 6 hours
            self.scheduler.add_job(
                func=self._log_optimization_stats,
                trigger=IntervalTrigger(hours=6),
                id='log_optimization_stats',
                name='Log FSRS Optimization Statistics',
                replace_existing=True
            )

            logger.info("Maintenance jobs scheduled successfully")

        except Exception as e:
            logger.error(f"Failed to setup maintenance jobs: {e}")

    def _run_fsrs_optimization(self) -> None:
        """Run daily FSRS parameter optimization for active users."""
        logger.info("Starting daily FSRS optimization job")

        try:
            # Get candidates for optimization
            candidates = self.fsrs_optimizer.get_optimization_candidates()

            if not candidates:
                logger.info("No optimization candidates found")
                return

            optimized_count = 0
            failed_count = 0

            for user_id, pillar_id in candidates:
                try:
                    request = FSRSOptimizationRequest(
                        user_id=user_id,
                        pillar_id=pillar_id,
                        min_reviews=50  # Standard minimum for daily optimization
                    )

                    result = self.fsrs_optimizer.optimize_user_parameters(request)

                    if result.success:
                        optimized_count += 1
                        logger.info(f"Optimized FSRS parameters for {user_id}/{pillar_id}: "
                                   f"{result.reviews_analyzed} reviews, score: {result.improvement_score:.3f}")
                    else:
                        failed_count += 1
                        logger.debug(f"Optimization failed for {user_id}/{pillar_id}: {result.message}")

                except Exception as e:
                    failed_count += 1
                    logger.error(f"Failed to optimize parameters for {user_id}/{pillar_id}: {e}")

            logger.info(f"Daily FSRS optimization completed: {optimized_count} optimized, {failed_count} failed")

        except Exception as e:
            logger.error(f"Daily FSRS optimization job failed: {e}")

    def _run_comprehensive_fsrs_optimization(self) -> None:
        """Run weekly comprehensive FSRS optimization with lower thresholds."""
        logger.info("Starting weekly comprehensive FSRS optimization job")

        try:
            # Use lower minimum review threshold for comprehensive optimization
            results = self.fsrs_optimizer.optimize_all_users(min_reviews=20)

            optimized_count = sum(1 for r in results.values() if r.success)
            failed_count = len(results) - optimized_count

            logger.info(f"Weekly comprehensive FSRS optimization completed: "
                       f"{optimized_count} optimized, {failed_count} failed")

            # Log detailed results
            for key, result in results.items():
                if result.success:
                    logger.info(f"Comprehensive optimization for {key}: "
                               f"{result.reviews_analyzed} reviews, score: {result.improvement_score:.3f}")
                else:
                    logger.debug(f"Comprehensive optimization failed for {key}: {result.message}")

        except Exception as e:
            logger.error(f"Weekly comprehensive FSRS optimization job failed: {e}")

    def _log_optimization_stats(self) -> None:
        """Log FSRS optimization statistics for monitoring."""
        try:
            stats = self.fsrs_optimizer.get_optimization_stats()

            logger.info(f"FSRS Optimization Stats: "
                       f"Optimizer available: {stats.get('optimizer_available', False)}, "
                       f"FSRS available: {stats.get('fsrs_available', False)}, "
                       f"Candidates: {stats.get('candidates_for_optimization', 0)}")

            # Log user-specific stats
            user_stats = stats.get('default_user_stats', {})
            for pillar, pillar_stats in user_stats.items():
                if pillar_stats.get('has_parameters'):
                    logger.debug(f"FSRS stats for {pillar}: "
                                f"Reviews: {pillar_stats.get('review_count', 0)}, "
                                f"Ready for optimization: {pillar_stats.get('ready_for_optimization', False)}")

        except Exception as e:
            logger.error(f"Failed to log optimization stats: {e}")

    def trigger_fsrs_optimization_now(self, user_id: str = "default_user",
                                     pillar_id: Optional[str] = None) -> Dict[str, Any]:
        """Trigger FSRS optimization immediately for testing/manual execution.

        Args:
        ----
            user_id: User to optimize parameters for
            pillar_id: Optional pillar filter (slug format, e.g., 'models-architectures')

        Returns:
        -------
            Dictionary with optimization results

        """
        try:
            request = FSRSOptimizationRequest(
                user_id=user_id,
                pillar_id=pillar_id,  # Now accepts string directly
                min_reviews=10  # Lower threshold for manual testing
            )

            result = self.fsrs_optimizer.optimize_user_parameters(request)

            return {
                "success": result.success,
                "reviews_analyzed": result.reviews_analyzed,
                "improvement_score": result.improvement_score,
                "message": result.message
            }

        except Exception as e:
            logger.error(f"Manual FSRS optimization failed: {e}")
            return {"success": False, "error": str(e)}

    def get_job_status(self) -> Dict[str, Any]:
        """Get status of scheduled jobs.
        
        Returns
        -------
            Dictionary with job status information

        """
        try:
            jobs = []

            for job in self.scheduler.get_jobs():
                next_run = job.next_run_time
                jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run": next_run.isoformat() if next_run else None,
                    "trigger": str(job.trigger)
                })

            return {
                "scheduler_running": self.is_running,
                "jobs": jobs,
                "optimization_stats": self.fsrs_optimizer.get_optimization_stats()
            }

        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return {"error": str(e)}


# Global service instance
_background_jobs_service: Optional[BackgroundJobsService] = None


def get_background_jobs_service() -> BackgroundJobsService:
    """Get or create the global background jobs service instance."""
    global _background_jobs_service

    if _background_jobs_service is None:
        _background_jobs_service = BackgroundJobsService()

    return _background_jobs_service


def start_background_jobs() -> None:
    """Start the background jobs service."""
    service = get_background_jobs_service()
    service.start()


def stop_background_jobs() -> None:
    """Stop the background jobs service."""
    global _background_jobs_service

    if _background_jobs_service:
        _background_jobs_service.stop()


def set_background_jobs_service(service: BackgroundJobsService) -> None:
    """Set the global background jobs service instance (for testing)."""
    global _background_jobs_service
    _background_jobs_service = service
