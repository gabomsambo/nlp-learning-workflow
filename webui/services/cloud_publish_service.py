"""Keeps the cloud reader current while the webui is up.

``notify()`` is the hook: a finished run, a saved script, a review. It only
records that something changed and returns, so it can sit in ``finally:`` of a
pipeline run. One daemon thread owns the publish. It waits for a quiet moment
(a burst of notifications becomes one publish, at most ``MAX_WAIT_SECONDS``
late), publishes, and on failure tries again later with a longer wait. A
failure is logged and shown on Activity; it never reaches the run that
triggered it.

Runs that finish in another process (the scheduler container, the CLI) never
call ``notify()``, so the thread also checks ``pipeline_runs`` every
``POLL_SECONDS`` and treats a newer finished run as a change. The publish
itself is a hash diff, so a notification that changed nothing uploads nothing.

Off unless ``CLOUD_PUBLISH_ENABLED`` is true.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 45.0
MAX_WAIT_SECONDS = 300.0
RETRY_BASE_SECONDS = 300.0
RETRY_MAX_SECONDS = 3600.0
POLL_SECONDS = 600.0


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_publish(settings: Any, progress: Callable[[str], None]) -> Dict[str, Any]:
    """Export and publish with the configured credentials. Returns the report summary."""
    from nlp_pillars.cloud.publish import PublishConfig, run_publish

    config = PublishConfig.from_settings(settings)
    report = run_publish(config, Path(settings.cloud_bundle_dir), on_progress=progress)
    summary = report.summary()
    logger.warning(
        "cloud publish: uploaded %s object(s) (%s bytes), %s unchanged, manifest %s, %.1fs",
        summary["uploaded"], summary["uploaded_bytes"], summary["unchanged"],
        "sent" if summary["manifest_sent"] else "unchanged", summary["seconds"],
    )
    return summary


def latest_finished_run() -> Optional[str]:
    """finished_at of the newest finished run that can change what the reader shows."""
    from nlp_pillars import db

    client = db.get_client()
    response = client.client.get(
        f"{client.base_url}/pipeline_runs",
        params={
            "select": "finished_at", "finished_at": "not.is.null",
            "kind": "neq.discover", "order": "finished_at.desc", "limit": "1",
        },
    )
    response.raise_for_status()
    rows = response.json()
    return rows[0]["finished_at"] if rows else None


class CloudPublisher:
    def __init__(
        self,
        settings: Any,
        *,
        publish: Callable[[Any, Callable[[str], None]], Dict[str, Any]] = default_publish,
        watch: Callable[[], Optional[str]] = latest_finished_run,
        clock: Callable[[], float] = time.monotonic,
        debounce: float = DEBOUNCE_SECONDS,
        max_wait: float = MAX_WAIT_SECONDS,
        retry_base: float = RETRY_BASE_SECONDS,
        retry_max: float = RETRY_MAX_SECONDS,
        poll: float = POLL_SECONDS,
    ) -> None:
        self._settings = settings
        self._publish = publish
        self._watch = watch
        self._clock = clock
        self._debounce, self._max_wait = debounce, max_wait
        self._retry_base, self._retry_max, self._poll = retry_base, retry_max, poll

        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._stopping = False
        self._dirty = False
        self._first = 0.0
        self._last = 0.0
        self._retry_at = 0.0
        self._next_poll = 0.0
        self._reasons: List[str] = []
        self._seen_finished: Optional[str] = None
        self._baselined = False

        self.state = "idle"
        self.message = ""
        self.last_attempt_at: Optional[str] = None
        self.last_success_at: Optional[str] = None
        self.last_error: Optional[str] = None
        self.last_summary: Optional[Dict[str, Any]] = None
        self.failures = 0

    # -- the hook -----------------------------------------------------------

    def notify(self, reason: str) -> None:
        """Record that the reader's content may have changed. Never raises, never blocks."""
        try:
            with self._cond:
                now = self._clock()
                if not self._dirty:
                    self._first = now
                self._dirty = True
                self._last = now
                if reason not in self._reasons:
                    self._reasons.append(reason)
                if self.state == "idle":
                    self.state = "waiting"
                self._cond.notify_all()
        except Exception:  # noqa: BLE001 - the caller is a pipeline run
            logger.warning("cloud publish: could not record %r", reason, exc_info=True)

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        with self._cond:
            if self._thread is not None:
                return
            self._stopping = False
            self._next_poll = self._clock() + self._poll
            self._thread = threading.Thread(target=self._loop, name="cloud-publisher", daemon=True)
            self._thread.start()
        self.notify("startup")

    def stop(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
            thread = self._thread
        # A publish in flight is not interrupted: it is idempotent and resumes next time.
        if thread is not None:
            thread.join(timeout)

    # -- the thread ---------------------------------------------------------

    def _due(self) -> Optional[float]:
        if not self._dirty:
            return None
        return max(min(self._last + self._debounce, self._first + self._max_wait), self._retry_at)

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._stopping:
                    now = self._clock()
                    due = self._due()
                    deadline = self._next_poll if due is None else min(due, self._next_poll)
                    if now >= deadline:
                        break
                    self._cond.wait(deadline - now)
                if self._stopping:
                    return
                now = self._clock()
                poll_now = now >= self._next_poll
                if poll_now:
                    self._next_poll = now + self._poll
                due = self._due()
                publish_now = due is not None and now >= due
            if poll_now:
                self._check_other_processes()
            if publish_now:
                self._run_once()

    def _check_other_processes(self) -> None:
        try:
            latest = self._watch()
        except Exception as exc:  # noqa: BLE001
            logger.warning("cloud publish: could not read pipeline_runs (%s)", exc)
            return
        first_look = not self._baselined
        self._baselined = True
        if latest and latest != self._seen_finished:
            self._seen_finished = latest
            if not first_look:
                self.notify("a run finished elsewhere")

    def _run_once(self) -> None:
        with self._cond:
            reasons, self._reasons = self._reasons, []
            self._dirty = False
            self.state = "publishing"
            self.message = "starting"
            self.last_attempt_at = _stamp()
        try:
            # Baseline for the poll: what this publish is about to include.
            self._seen_finished = self._watch() or self._seen_finished
        except Exception:  # noqa: BLE001
            pass
        try:
            summary = self._publish(self._settings, self._progress)
        except BaseException as exc:  # noqa: BLE001 - nothing may escape the thread
            self._failed(exc, reasons)
            if not isinstance(exc, Exception):
                raise
            return
        with self._cond:
            self.failures = 0
            self.last_error = None
            self.last_success_at = _stamp()
            self.last_summary = summary
            self._retry_at = 0.0
            self.message = ""
            self.state = "waiting" if self._dirty else "idle"
        self._after_publish()

    @staticmethod
    def _after_publish() -> None:
        """A publish is a natural moment to ask the cloud for reviews made since. Never raises."""
        try:
            from webui.services import cloud_review_sync_service

            cloud_review_sync_service.trigger("after publish")
        except Exception:  # noqa: BLE001
            logger.warning("cloud publish: could not trigger a review pull", exc_info=True)

    def _progress(self, message: str) -> None:
        self.message = message

    def _failed(self, exc: BaseException, reasons: List[str]) -> None:
        text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        logger.warning("cloud publish failed (%s): %s", ", ".join(reasons) or "no reason", text)
        with self._cond:
            self.failures += 1
            self.last_error = text[:300]
            self.state = "failed"
            self.message = ""
            wait = min(self._retry_base * (2 ** (self.failures - 1)), self._retry_max)
            self._retry_at = self._clock() + wait
            # Whatever was pending is still pending.
            now = self._clock()
            if not self._dirty:
                self._first = now
            self._dirty = True
            self._last = now
            self._reasons = list(dict.fromkeys(reasons + self._reasons))
            self._cond.notify_all()

    # -- status -------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        with self._cond:
            retry_in = max(0.0, self._retry_at - self._clock()) if self.state == "failed" else None
            return {
                "enabled": True, "configured": True, "state": self.state, "message": self.message,
                "last_attempt_at": self.last_attempt_at, "last_success_at": self.last_success_at,
                "last_error": self.last_error, "last_summary": self.last_summary,
                "failures": self.failures, "retry_in_seconds": retry_in,
            }


# ---------------------------------------------------------------------------
# Module-level instance: the webui lifespan starts it, run code notifies it.
# ---------------------------------------------------------------------------

_instance: Optional[CloudPublisher] = None
_misconfigured: List[str] = []


def start(settings: Any = None) -> Optional[CloudPublisher]:
    """Start publishing if CLOUD_PUBLISH_ENABLED is set and configured. Never raises."""
    global _instance, _misconfigured
    try:
        from nlp_pillars.cloud.publish import PublishConfig
        from nlp_pillars.config import get_settings

        settings = settings or get_settings()
        _misconfigured = []
        if not getattr(settings, "cloud_publish_enabled", False):
            return None
        _misconfigured = PublishConfig.missing(settings)
        if _misconfigured:
            logger.warning("cloud publish is enabled but %s is not set; not publishing", ", ".join(_misconfigured))
            return None
        _instance = CloudPublisher(settings)
        _instance.start()
        return _instance
    except Exception:  # noqa: BLE001 - a publisher problem must not stop the app starting
        logger.warning("cloud publish could not start", exc_info=True)
        return None


def stop() -> None:
    global _instance
    publisher, _instance = _instance, None
    if publisher is not None:
        publisher.stop()


def notify(reason: str) -> None:
    """Tell the publisher something changed. A no-op when publishing is off."""
    publisher = _instance
    if publisher is None:
        return
    try:
        publisher.notify(reason)
    except Exception:  # noqa: BLE001 - called from `finally:` of a pipeline run
        logger.warning("cloud publish: notify failed", exc_info=True)


def current_status(settings: Any = None) -> Dict[str, Any]:
    """What Activity shows. Valid whether or not publishing is on."""
    if _instance is not None:
        return _instance.status()
    from nlp_pillars.config import get_settings

    settings = settings or get_settings()
    enabled = bool(getattr(settings, "cloud_publish_enabled", False))
    return {
        "enabled": enabled, "configured": enabled and not _misconfigured, "missing": list(_misconfigured),
        "state": "off" if not enabled else "misconfigured",
    }
