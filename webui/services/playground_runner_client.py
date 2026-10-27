"""Thin async client for the playground runner (docs/PLAYGROUND_RUNNER.md).

The runner is a separate, locked-down container reachable only over compose's
internal ``playground`` network at ``PLAYGROUND_RUNNER_URL``. This client adds
nothing to the runner's answers: it forwards them and turns transport failures
into :class:`PlaygroundRunnerUnavailable`, whose message says how to start it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx

from nlp_pillars.config import get_settings

#: Used for every call except execute, which waits for the code to finish.
DEFAULT_TIMEOUT_S = 15.0
CONNECT_TIMEOUT_S = 3.0
#: Upper bound the runner allows for one execution when the caller does not
#: name one. The runner's own default is lower; this only sizes the HTTP wait.
EXECUTE_FALLBACK_TIMEOUT_S = 900.0
#: Room on top of the execution timeout for the interrupt grace period and a
#: kernel restart, so the runner's "timeout" answer arrives before ours fires.
EXECUTE_MARGIN_S = 120.0

START_HINT = "Start it with ./workstation up (or docker compose up -d playground-runner), then check ./workstation status."


class PlaygroundRunnerUnavailable(Exception):
    """The runner could not be reached at all."""


class PlaygroundRunnerError(Exception):
    """The runner answered with an error status."""

    def __init__(self, status_code: int, detail: Any):
        super().__init__(f"playground runner answered HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def runner_url() -> str:
    return (get_settings().playground_runner_url or "http://playground-runner:8765").rstrip("/")


def _detail(response: httpx.Response) -> Any:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500] or f"HTTP {response.status_code}"
    if isinstance(body, dict) and "detail" in body:
        return body["detail"]
    return body


class PlaygroundRunnerClient:
    def __init__(self, base_url: Optional[str] = None, *, transport: Optional[httpx.AsyncBaseTransport] = None):
        self.base_url = (base_url or runner_url()).rstrip("/")
        self._transport = transport

    def unavailable_message(self, reason: str = "") -> str:
        why = f" ({reason})" if reason else ""
        return f"The playground runner is not reachable at {self.base_url}{why}. {START_HINT}"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        read_timeout: float = DEFAULT_TIMEOUT_S,
    ) -> Dict[str, Any]:
        timeout = httpx.Timeout(read_timeout, connect=CONNECT_TIMEOUT_S)
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, timeout=timeout, transport=self._transport
            ) as client:
                response = await client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            if isinstance(exc, httpx.ConnectTimeout):
                raise PlaygroundRunnerUnavailable(self.unavailable_message("connection timed out")) from exc
            raise PlaygroundRunnerError(
                504, f"The playground runner did not answer within {read_timeout:g} s."
            ) from exc
        except httpx.TransportError as exc:
            raise PlaygroundRunnerUnavailable(self.unavailable_message(type(exc).__name__)) from exc
        if response.status_code >= 400:
            raise PlaygroundRunnerError(response.status_code, _detail(response))
        try:
            body = response.json()
        except ValueError as exc:
            raise PlaygroundRunnerError(502, "The playground runner answered with something that is not JSON.") from exc
        return body if isinstance(body, dict) else {"result": body}

    @staticmethod
    def _sid(session_id: str) -> str:
        return quote(session_id, safe="")

    async def health(self) -> Dict[str, Any]:
        return await self._request("GET", "/health", read_timeout=5.0)

    async def list_sessions(self) -> Dict[str, Any]:
        return await self._request("GET", "/sessions")

    @staticmethod
    def _start_wait(setup_code: Optional[str]) -> float:
        # Starting a kernel takes a few seconds; the runner's own cap is 60 s.
        # Setup code then runs under the execution timeout.
        return 90.0 + (EXECUTE_FALLBACK_TIMEOUT_S if (setup_code or "").strip() else 0.0)

    async def create_session(self, playground_id: str, setup_code: str = "") -> Dict[str, Any]:
        body = {"playground_id": playground_id, "setup_code": setup_code or ""}
        return await self._request("POST", "/sessions", json=body, read_timeout=self._start_wait(setup_code))

    async def execute(self, session_id: str, code: str, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"code": code}
        if timeout_s is not None:
            body["timeout_s"] = timeout_s
        wait = (timeout_s if timeout_s is not None else EXECUTE_FALLBACK_TIMEOUT_S) + EXECUTE_MARGIN_S
        return await self._request("POST", f"/sessions/{self._sid(session_id)}/execute", json=body, read_timeout=wait)

    async def interrupt(self, session_id: str) -> Dict[str, Any]:
        return await self._request("POST", f"/sessions/{self._sid(session_id)}/interrupt")

    async def restart(self, session_id: str, setup_code: Optional[str] = None) -> Dict[str, Any]:
        body = None if setup_code is None else {"setup_code": setup_code}
        return await self._request(
            "POST", f"/sessions/{self._sid(session_id)}/restart", json=body,
            read_timeout=self._start_wait(setup_code),
        )

    async def delete_session(self, session_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/sessions/{self._sid(session_id)}", read_timeout=30.0)
