"""Sessions: one kernel each, a concurrency cap, and idle reaping."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Protocol

from .config import RunnerConfig

logger = logging.getLogger(__name__)


class Kernel(Protocol):
    #: Run after every start and restart (docs/PLAYGROUND.md, "Setup code").
    setup_code: str

    async def start(self) -> None: ...
    async def execute(self, code: str, timeout_s: float, abort: Optional[asyncio.Event] = None) -> Dict[str, Any]: ...
    async def interrupt(self) -> None: ...
    async def restart(self) -> None: ...
    async def shutdown(self) -> None: ...


KernelFactory = Callable[[RunnerConfig, str, str], Kernel]


class SessionNotFound(KeyError):
    pass


class SessionBusy(RuntimeError):
    pass


class TooManySessions(RuntimeError):
    pass


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


@dataclass
class Session:
    session_id: str
    playground_id: str
    kernel: Kernel
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    last_active_mono: float = field(default_factory=time.monotonic)
    execution_count: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    abort: asyncio.Event = field(default_factory=asyncio.Event)
    interrupt_requested: bool = False

    @property
    def busy(self) -> bool:
        return self.lock.locked()

    def touch(self) -> None:
        self.last_active_at = time.time()
        self.last_active_mono = time.monotonic()

    def describe(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "playground_id": self.playground_id,
            "created_at": _iso(self.created_at),
            "last_active_at": _iso(self.last_active_at),
            "busy": self.busy,
            "execution_count": self.execution_count,
            "setup": setup_status(self.kernel),
        }


def setup_status(kernel: Kernel) -> Dict[str, Any]:
    """How the playground's setup code went on the kernel's last (re)start:
    ``{"status": "none" | "ok" | "error", "ename"?, "evalue"?}``."""
    result = getattr(kernel, "setup_result", None)
    return dict(result) if isinstance(result, dict) else {"status": "none"}


def _default_factory(config: RunnerConfig, session_id: str, playground_id: str) -> Kernel:
    from .kernel import JupyterKernel

    return JupyterKernel(config, session_id, playground_id)


class SessionManager:
    def __init__(self, config: RunnerConfig, kernel_factory: Optional[KernelFactory] = None):
        self.config = config
        self.kernel_factory = kernel_factory or _default_factory
        self.sessions: Dict[str, Session] = {}
        self._starting = 0
        self._create_lock = asyncio.Lock()
        self._reaper: Optional[asyncio.Task] = None

    # -- lookup -----------------------------------------------------------

    def get(self, session_id: str) -> Session:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise SessionNotFound(session_id) from None

    def list(self) -> List[Dict[str, Any]]:
        return [s.describe() for s in self.sessions.values()]

    # -- lifecycle --------------------------------------------------------

    async def create(self, playground_id: str, setup_code: str = "") -> Session:
        async with self._create_lock:
            if len(self.sessions) + self._starting >= self.config.max_sessions:
                raise TooManySessions(
                    f"The runner already has {self.config.max_sessions} sessions open, which is the limit "
                    "(PLAYGROUND_MAX_SESSIONS). Close one with DELETE /sessions/{id}; idle sessions close "
                    f"on their own after {self.config.idle_timeout_s:g} s."
                )
            self._starting += 1
        session_id = secrets.token_hex(12)
        kernel = self.kernel_factory(self.config, session_id, playground_id)
        kernel.setup_code = setup_code or ""
        try:
            await kernel.start()
        finally:
            async with self._create_lock:
                self._starting -= 1
        session = Session(session_id=session_id, playground_id=playground_id, kernel=kernel)
        self.sessions[session_id] = session
        logger.info("session %s started for playground %s", session_id, playground_id)
        return session

    async def execute(self, session_id: str, code: str, timeout_s: float) -> Dict[str, Any]:
        session = self.get(session_id)
        if session.lock.locked():
            raise SessionBusy(
                "This session is still running earlier code. Wait for it, or interrupt it with "
                f"POST /sessions/{session_id}/interrupt."
            )
        async with session.lock:
            session.touch()
            session.interrupt_requested = False
            session.abort.clear()
            try:
                result = await session.kernel.execute(code, timeout_s, session.abort)
            finally:
                session.touch()
            result["interrupted"] = bool(session.interrupt_requested) and result.get("status") == "error"
            session.interrupt_requested = False
            if result.get("execution_count") is not None:
                session.execution_count = int(result["execution_count"])
            if result.get("kernel_restarted"):
                session.execution_count = 0
            return result

    async def interrupt(self, session_id: str) -> Dict[str, Any]:
        session = self.get(session_id)
        was_busy = session.busy
        if was_busy:
            session.interrupt_requested = True
        await session.kernel.interrupt()
        session.touch()
        return {"session_id": session_id, "interrupted": was_busy}

    async def _stop_running(self, session: Session) -> None:
        """Make an in-flight execute return, then hold its lock."""
        if session.busy:
            session.abort.set()
            try:
                await session.kernel.interrupt()
            except Exception:
                logger.warning("interrupt before stop failed for %s", session.session_id, exc_info=True)

    async def restart(self, session_id: str, setup_code: Optional[str] = None) -> Dict[str, Any]:
        """Fresh kernel. ``setup_code``, when given, replaces the session's
        setup code before it runs again."""
        session = self.get(session_id)
        await self._stop_running(session)
        async with session.lock:
            session.abort.clear()
            if setup_code is not None:
                session.kernel.setup_code = setup_code
            await session.kernel.restart()
            session.execution_count = 0
            session.touch()
        return {"session_id": session_id, "restarted": True, "setup": setup_status(session.kernel)}

    async def delete(self, session_id: str) -> None:
        session = self.get(session_id)
        await self._stop_running(session)
        async with session.lock:
            self.sessions.pop(session_id, None)
            await session.kernel.shutdown()
        logger.info("session %s closed", session_id)

    async def shutdown_all(self) -> None:
        for session_id in list(self.sessions):
            try:
                await self.delete(session_id)
            except Exception:
                logger.warning("could not close session %s", session_id, exc_info=True)

    # -- reaping ----------------------------------------------------------

    async def reap_idle(self, now: Optional[float] = None) -> List[str]:
        now = time.monotonic() if now is None else now
        reaped = []
        for session in list(self.sessions.values()):
            if session.busy:
                continue
            if now - session.last_active_mono >= self.config.idle_timeout_s:
                try:
                    await self.delete(session.session_id)
                    reaped.append(session.session_id)
                    logger.info("session %s reaped after being idle", session.session_id)
                except SessionNotFound:
                    pass
        return reaped

    async def _reap_forever(self) -> None:
        while True:
            await asyncio.sleep(self.config.reap_interval_s)
            try:
                await self.reap_idle()
            except Exception:
                logger.warning("idle reaping failed", exc_info=True)

    def start_reaper(self) -> None:
        if self._reaper is None:
            self._reaper = asyncio.create_task(self._reap_forever())

    async def stop_reaper(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
            try:
                await self._reaper
            except (asyncio.CancelledError, Exception):
                pass
            self._reaper = None
