"""JSON proxy to the playground runner, for the Playground tab (PR 3).

Each route forwards one runner call (docs/PLAYGROUND_RUNNER.md) and returns the
runner's body unchanged. Status codes:

* the runner's own 4xx (404 unknown session, 409 busy, 413, 422, 429 too many
  sessions) pass through with its ``detail``;
* 503 when the runner cannot be reached, with a ``detail`` naming how to start
  it, or when it could not start a kernel;
* 502 for any other runner 5xx, 504 when it did not answer in time.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from webui.services.playground_runner_client import (
    PlaygroundRunnerClient,
    PlaygroundRunnerError,
    PlaygroundRunnerUnavailable,
)

router = APIRouter(prefix="/api/playground/runner", tags=["playground"])

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class CreateSessionBody(BaseModel):
    playground_id: str = Field(..., min_length=1, max_length=128)


class ExecuteBody(BaseModel):
    code: str
    timeout_s: Optional[float] = Field(None, gt=0)


def get_client() -> PlaygroundRunnerClient:
    """Indirection so tests can swap the transport."""
    return PlaygroundRunnerClient()


async def _call(coro) -> Dict[str, Any]:
    try:
        return await coro
    except PlaygroundRunnerUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PlaygroundRunnerError as exc:
        status = exc.status_code
        if status >= 500 and status not in (503, 504):
            status = 502
        raise HTTPException(status_code=status, detail=exc.detail) from exc


SessionId = Annotated[str, Path(pattern=SESSION_ID_PATTERN)]


@router.get("/health")
async def runner_health() -> Dict[str, Any]:
    return await _call(get_client().health())


@router.get("/sessions")
async def list_sessions() -> Dict[str, Any]:
    return await _call(get_client().list_sessions())


@router.post("/sessions", status_code=201)
async def create_session(body: CreateSessionBody) -> Dict[str, Any]:
    return await _call(get_client().create_session(body.playground_id))


@router.post("/sessions/{session_id}/execute")
async def execute(session_id: SessionId, body: ExecuteBody) -> Dict[str, Any]:
    return await _call(get_client().execute(session_id, body.code, body.timeout_s))


@router.post("/sessions/{session_id}/interrupt")
async def interrupt(session_id: SessionId) -> Dict[str, Any]:
    return await _call(get_client().interrupt(session_id))


@router.post("/sessions/{session_id}/restart")
async def restart(session_id: SessionId) -> Dict[str, Any]:
    return await _call(get_client().restart(session_id))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: SessionId) -> JSONResponse:
    return JSONResponse(await _call(get_client().delete_session(session_id)))
