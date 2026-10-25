"""Brings the phone's reviews home while the webui is up.

``cloud pull-reviews`` does the work (``nlp_pillars/cloud/reviews.py``); this runs
it. One daemon thread pulls

* when the app starts,
* every ``INTERVAL_SECONDS`` (10 minutes) after that,
* after every publish (``trigger("after publish")``), and
* straight away again if the cloud says more events are waiting.

A pull that applied reviews tells the publisher, so the phone's snapshot catches
up. A failure is logged, shown on Activity and retried sooner than the interval,
and never reaches anything else. Off unless ``CLOUD_PUBLISH_ENABLED`` is true and
the service token is configured: the same switch as publishing, because the same
token is what may read ``/ingest/reviews``.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

INTERVAL_SECONDS = 600.0
# Between a trigger and its pull, so a publish that triggers one does not race the publish's own tail.
SETTLE_SECONDS = 5.0
RETRY_BASE_SECONDS = 60.0


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_pull(settings: Any, progress: Callable[[str], None]) -> Dict[str, Any]:
    """Pull with the configured service token. Returns the report summary."""
    from nlp_pillars.cloud.publish import PublishConfig
    from nlp_pillars.cloud.reviews import pull_reviews

    report = pull_reviews(PublishConfig.from_settings(settings), on_progress=progress)
    summary = report.summary()
    if summary["applied"] or summary["skipped"]:
        logger.warning(
            "cloud reviews: %s applied, %s already recorded, %s skipped; cursor %s",
            summary["applied"], summary["duplicates"], summary["skipped"], summary["cursor"],
        )
    return summary


class ReviewSyncer:
    def __init__(
        self,
        settings: Any,
        *,
        pull: Callable[[Any, Callable[[str], None]], Dict[str, Any]] = default_pull,
        on_applied: Optional[Callable[[], None]] = None,
        clock: Callable[[], float] = time.monotonic,
        interval: float = INTERVAL_SECONDS,
        settle: float = SETTLE_SECONDS,
        retry_base: float = RETRY_BASE_SECONDS,
    ) -> None:
        self._settings = settings
        self._pull = pull
        self._on_applied = on_applied or _tell_publisher
        self._clock = clock
        self._interval, self._settle, self._retry_base = interval, settle, retry_base

        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._stopping = False
        self._requested = False
        self._requested_at = 0.0
        self._next = 0.0
        self._reasons: List[str] = []

        self.state = "idle"
        self.message = ""
        self.last_attempt_at: Optional[str] = None
        self.last_success_at: Optional[str] = None
        self.last_error: Optional[str] = None
        self.last_summary: Optional[Dict[str, Any]] = None
        self.failures = 0

    # -- the hook -----------------------------------------------------------

    def trigger(self, reason: str) -> None:
        """Ask for a pull soon. Never raises, never blocks."""
        try:
            with self._cond:
                if not self._requested:
                    self._requested_at = self._clock()
                self._requested = True
                if reason not in self._reasons:
                    self._reasons.append(reason)
                self._cond.notify_all()
        except Exception:  # noqa: BLE001
            logger.warning("cloud reviews: could not record %r", reason, exc_info=True)

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        with self._cond:
            if self._thread is not None:
                return
            self._stopping = False
            self._next = self._clock()  # pull as soon as the app is up
            self._thread = threading.Thread(target=self._loop, name="cloud-review-sync", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
            thread = self._thread
        # A pull in flight is not interrupted: every step of it is safe to repeat.
        if thread is not None:
            thread.join(timeout)

    # -- the thread ---------------------------------------------------------

    def _deadline(self) -> float:
        if self._requested:
            return min(self._next, self._requested_at + self._settle)
        return self._next

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._stopping:
                    now = self._clock()
                    deadline = self._deadline()
                    if now >= deadline:
                        break
                    self._cond.wait(deadline - now)
                if self._stopping:
                    return
                reasons, self._reasons = self._reasons, []
                self._requested = False
                self.state = "pulling"
                self.message = "starting"
                self.last_attempt_at = _stamp()
            self._run_once(reasons)

    def _run_once(self, reasons: List[str]) -> None:
        applied = False
        try:
            summary = self._pull(self._settings, self._progress)
        except BaseException as exc:  # noqa: BLE001 - nothing may escape the thread
            # A pull that stopped part-way may still have applied events the phone should hear about.
            applied = bool(getattr(getattr(exc, "report", None), "applied", 0))
            self._failed(exc, reasons)
            if not isinstance(exc, Exception):
                raise
        else:
            applied = bool(summary.get("applied"))
            with self._cond:
                self.failures = 0
                self.last_error = None
                self.last_success_at = _stamp()
                self.last_summary = summary
                self.message = ""
                self.state = "idle"
                # More waiting: carry straight on. Otherwise wait out the interval.
                self._next = self._clock() + (self._settle if summary.get("more") else self._interval)
        if applied:
            try:
                self._on_applied()
            except Exception:  # noqa: BLE001
                logger.warning("cloud reviews: could not tell the publisher", exc_info=True)

    def _progress(self, message: str) -> None:
        self.message = message

    def _failed(self, exc: BaseException, reasons: List[str]) -> None:
        text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        logger.warning("cloud reviews failed (%s): %s", ", ".join(reasons) or "scheduled", text)
        with self._cond:
            self.failures += 1
            self.last_error = text[:300]
            self.state = "failed"
            self.message = ""
            self._next = self._clock() + min(self._retry_base * (2 ** (self.failures - 1)), self._interval)

    # -- status -------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        with self._cond:
            retry_in = max(0.0, self._next - self._clock()) if self.state == "failed" else None
            return {
                "enabled": True, "configured": True, "state": self.state, "message": self.message,
                "last_attempt_at": self.last_attempt_at, "last_success_at": self.last_success_at,
                "last_error": self.last_error, "last_summary": self.last_summary,
                "failures": self.failures, "retry_in_seconds": retry_in,
            }


def _tell_publisher() -> None:
    from webui.services import cloud_publish_service

    cloud_publish_service.notify("phone reviews")


# ---------------------------------------------------------------------------
# Module-level instance: the webui lifespan starts it, the publisher triggers it.
# ---------------------------------------------------------------------------

_instance: Optional[ReviewSyncer] = None


def start(settings: Any = None) -> Optional[ReviewSyncer]:
    """Start pulling if publishing is enabled and configured. Never raises."""
    global _instance
    try:
        from nlp_pillars.cloud.publish import PublishConfig
        from nlp_pillars.config import get_settings

        settings = settings or get_settings()
        if not getattr(settings, "cloud_publish_enabled", False) or PublishConfig.missing(settings):
            return None
        _instance = ReviewSyncer(settings)
        _instance.start()
        return _instance
    except Exception:  # noqa: BLE001 - a sync problem must not stop the app starting
        logger.warning("cloud review sync could not start", exc_info=True)
        return None


def stop() -> None:
    global _instance
    syncer, _instance = _instance, None
    if syncer is not None:
        syncer.stop()


def trigger(reason: str) -> None:
    """Ask for a pull. A no-op when sync is off."""
    syncer = _instance
    if syncer is None:
        return
    try:
        syncer.trigger(reason)
    except Exception:  # noqa: BLE001
        logger.warning("cloud reviews: trigger failed", exc_info=True)


def current_status() -> Dict[str, Any]:
    """What Activity shows about the sync. ``enabled`` is false when it is not running."""
    if _instance is not None:
        return _instance.status()
    return {"enabled": False}
