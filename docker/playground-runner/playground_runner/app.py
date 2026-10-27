"""HTTP API of the playground runner. docs/PLAYGROUND_RUNNER.md is the contract."""

from __future__ import annotations

import logging
import os
import platform
import shutil
from contextlib import asynccontextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import RunnerConfig, valid_playground_id
from .sessions import (
    KernelFactory,
    SessionBusy,
    SessionManager,
    SessionNotFound,
    TooManySessions,
)

logger = logging.getLogger(__name__)

REPORTED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "ipykernel", "jupyter_client")
MPL_CACHE_SEED = Path("/opt/matplotlib-cache")


class CreateSessionRequest(BaseModel):
    playground_id: str = Field(..., min_length=1, max_length=128)
    setup_code: str = ""


class RestartRequest(BaseModel):
    setup_code: Optional[str] = None


class ExecuteRequest(BaseModel):
    code: str
    timeout_s: Optional[float] = Field(None, gt=0)


def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in REPORTED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _memory_limit_bytes() -> Optional[int]:
    """The container's cgroup v2 memory limit, when there is one."""
    try:
        raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    except OSError:
        return None
    return None if raw == "max" else int(raw)


def _seed_matplotlib_cache() -> None:
    """Copy the font cache built into the image onto the writable tmpfs."""
    target = os.environ.get("MPLCONFIGDIR")
    if not target or not MPL_CACHE_SEED.is_dir():
        return
    try:
        shutil.copytree(MPL_CACHE_SEED, target, dirs_exist_ok=True)
    except OSError:
        logger.warning("could not seed the matplotlib cache", exc_info=True)


def create_app(config: Optional[RunnerConfig] = None, kernel_factory: Optional[KernelFactory] = None) -> FastAPI:
    config = config or RunnerConfig.from_env()
    manager = SessionManager(config, kernel_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config.runtime_dir.mkdir(parents=True, exist_ok=True)
        _seed_matplotlib_cache()
        manager.start_reaper()
        try:
            yield
        finally:
            await manager.stop_reaper()
            await manager.shutdown_all()

    app = FastAPI(title="Playground runner", lifespan=lifespan)
    app.state.manager = manager
    app.state.config = config

    def check_code_size(code: Optional[str], what: str) -> None:
        if code is not None and len(code.encode("utf-8")) > config.max_code_bytes:
            raise HTTPException(status_code=413, detail=f"{what} is larger than {config.max_code_bytes} bytes")

    def not_found(session_id: str) -> HTTPException:
        return HTTPException(status_code=404, detail=f"No session {session_id}. It may have been closed after being idle.")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "sessions": len(manager.sessions),
            "max_sessions": config.max_sessions,
            "limits": {
                "exec_timeout_s": config.exec_timeout_s,
                "max_exec_timeout_s": config.max_exec_timeout_s,
                "max_output_bytes": config.max_output_bytes,
                "max_code_bytes": config.max_code_bytes,
                "idle_timeout_s": config.idle_timeout_s,
                "memory_limit_bytes": _memory_limit_bytes(),
            },
            "kernel_seccomp": config.kernel_seccomp,
            "python": platform.python_version(),
            "packages": _package_versions(),
        }

    @app.get("/sessions")
    async def list_sessions() -> Dict[str, Any]:
        return {
            "sessions": manager.list(),
            "max_sessions": config.max_sessions,
            "idle_timeout_s": config.idle_timeout_s,
        }

    @app.post("/sessions", status_code=201)
    async def create_session(body: CreateSessionRequest) -> Dict[str, Any]:
        if not valid_playground_id(body.playground_id):
            raise HTTPException(
                status_code=422,
                detail="playground_id may contain only letters, digits, '.', '_' and '-', and must start with a letter or digit.",
            )
        check_code_size(body.setup_code, "setup_code")
        try:
            session = await manager.create(body.playground_id, body.setup_code)
        except TooManySessions as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("kernel start failed")
            raise HTTPException(status_code=503, detail=f"Could not start a Python kernel: {exc}") from exc
        return session.describe()

    @app.post("/sessions/{session_id}/execute")
    async def execute(session_id: str, body: ExecuteRequest) -> Dict[str, Any]:
        check_code_size(body.code, "code")
        timeout_s = body.timeout_s if body.timeout_s is not None else config.exec_timeout_s
        if timeout_s > config.max_exec_timeout_s:
            raise HTTPException(
                status_code=422,
                detail=f"timeout_s may be at most {config.max_exec_timeout_s:g} (PLAYGROUND_MAX_EXEC_TIMEOUT_S)",
            )
        try:
            return await manager.execute(session_id, body.code, timeout_s)
        except SessionNotFound:
            raise not_found(session_id) from None
        except SessionBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("execute failed in session %s", session_id)
            raise HTTPException(
                status_code=503,
                detail=f"The Python kernel failed: {exc}. Restart the session (POST /sessions/{session_id}/restart).",
            ) from exc

    @app.post("/sessions/{session_id}/interrupt")
    async def interrupt(session_id: str) -> Dict[str, Any]:
        try:
            return await manager.interrupt(session_id)
        except SessionNotFound:
            raise not_found(session_id) from None

    @app.post("/sessions/{session_id}/restart")
    async def restart(session_id: str, body: Optional[RestartRequest] = None) -> Dict[str, Any]:
        setup_code = body.setup_code if body is not None else None
        check_code_size(setup_code, "setup_code")
        try:
            return await manager.restart(session_id, setup_code)
        except SessionNotFound:
            raise not_found(session_id) from None
        except Exception as exc:
            logger.exception("restart failed")
            raise HTTPException(status_code=503, detail=f"Could not restart the Python kernel: {exc}") from exc

    @app.delete("/sessions/{session_id}")
    async def delete(session_id: str) -> JSONResponse:
        try:
            await manager.delete(session_id)
        except SessionNotFound:
            raise not_found(session_id) from None
        return JSONResponse({"session_id": session_id, "deleted": True})

    return app


def app_from_env() -> FastAPI:  # uvicorn --factory entry point
    logging.basicConfig(level=os.environ.get("PLAYGROUND_LOG_LEVEL", "INFO"))
    return create_app()
