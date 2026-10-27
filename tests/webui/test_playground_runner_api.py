"""webui's playground runner client and its /api/playground/runner routes.

The runner is replaced by httpx.MockTransport; nothing here opens a socket.
"""

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from webui.routers.api import playground_runner as routes
from webui.services.playground_runner_client import (
    EXECUTE_MARGIN_S,
    PlaygroundRunnerClient,
    PlaygroundRunnerError,
    PlaygroundRunnerUnavailable,
)

BASE = "http://runner.test:8765"


def _runner(seen):
    """A fake runner that answers like the real API."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "method": request.method,
                "path": request.url.raw_path.decode(),
                "body": json.loads(request.content) if request.content else None,
                "read_timeout": request.extensions.get("timeout", {}).get("read"),
            }
        )
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok", "sessions": 0, "max_sessions": 3})
        if path == "/sessions" and request.method == "GET":
            return httpx.Response(200, json={"sessions": [], "max_sessions": 3, "idle_timeout_s": 1800})
        if path == "/sessions" and request.method == "POST":
            if json.loads(request.content)["playground_id"] == "full":
                return httpx.Response(429, json={"detail": "The runner already has 3 sessions open"})
            return httpx.Response(201, json={"session_id": "abc123", "playground_id": "pg"})
        if path == "/sessions/abc123/execute":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "execution_ms": 12,
                    "execution_count": 1,
                    "outputs": [{"type": "text", "text": "42"}],
                    "truncated": False,
                    "dropped_outputs": 0,
                    "interrupted": False,
                    "kernel_restarted": False,
                },
            )
        if path == "/sessions/rich/execute":
            return httpx.Response(200, json={"status": "ok", "execution_ms": 5, "execution_count": 2,
                                             "outputs": RICH_OUTPUTS, "truncated": True, "dropped_outputs": 1,
                                             "interrupted": False, "kernel_restarted": False})
        if path == "/sessions/busy/execute":
            return httpx.Response(409, json={"detail": "This session is still running earlier code."})
        if path == "/sessions/broken/execute":
            return httpx.Response(500, text="Internal Server Error")
        if path == "/sessions/abc123/interrupt":
            return httpx.Response(200, json={"session_id": "abc123", "interrupted": True})
        if path == "/sessions/abc123/restart":
            return httpx.Response(200, json={"session_id": "abc123", "restarted": True})
        if path == "/sessions/abc123" and request.method == "DELETE":
            return httpx.Response(200, json={"session_id": "abc123", "deleted": True})
        return httpx.Response(404, json={"detail": f"No session at {path}"})

    return handler


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("[Errno 111] Connection refused", request=request)


@pytest.fixture
def api(monkeypatch):
    seen = []
    state = {"transport": httpx.MockTransport(_runner(seen))}
    monkeypatch.setattr(
        routes, "get_client", lambda: PlaygroundRunnerClient(BASE, transport=state["transport"])
    )
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield client, seen, state


# -- client ---------------------------------------------------------------


async def test_client_forwards_calls_and_sizes_the_execute_wait():
    seen = []
    client = PlaygroundRunnerClient(BASE, transport=httpx.MockTransport(_runner(seen)))
    assert (await client.health())["status"] == "ok"
    assert (await client.create_session("pg"))["session_id"] == "abc123"
    result = await client.execute("abc123", "6 * 7", timeout_s=60)
    assert result["outputs"] == [{"type": "text", "text": "42"}]
    assert seen[-1]["body"] == {"code": "6 * 7", "timeout_s": 60}
    assert seen[-1]["read_timeout"] == 60 + EXECUTE_MARGIN_S
    await client.execute("abc123", "1")
    assert seen[-1]["body"] == {"code": "1"}
    assert seen[-1]["read_timeout"] > 900
    assert (await client.interrupt("abc123"))["interrupted"] is True
    assert (await client.restart("abc123"))["restarted"] is True
    assert (await client.delete_session("abc123"))["deleted"] is True


async def test_client_quotes_session_ids_into_one_path_segment():
    seen = []
    client = PlaygroundRunnerClient(BASE, transport=httpx.MockTransport(_runner(seen)))
    with pytest.raises(PlaygroundRunnerError):
        await client.interrupt("../health")
    assert seen[-1]["path"] == "/sessions/..%2Fhealth/interrupt"


async def test_client_down_raises_unavailable_with_how_to_start():
    client = PlaygroundRunnerClient(BASE, transport=httpx.MockTransport(_refuse))
    with pytest.raises(PlaygroundRunnerUnavailable) as info:
        await client.health()
    message = str(info.value)
    assert BASE in message
    assert "./workstation up" in message
    assert "docker compose up -d playground-runner" in message


async def test_client_runner_errors_carry_status_and_detail():
    client = PlaygroundRunnerClient(BASE, transport=httpx.MockTransport(_runner([])))
    with pytest.raises(PlaygroundRunnerError) as info:
        await client.execute("busy", "1")
    assert info.value.status_code == 409
    assert "still running" in info.value.detail


async def test_client_read_timeout_is_504_not_unavailable():
    def slow(request):
        raise httpx.ReadTimeout("timed out", request=request)

    client = PlaygroundRunnerClient(BASE, transport=httpx.MockTransport(slow))
    with pytest.raises(PlaygroundRunnerError) as info:
        await client.execute("abc123", "1", timeout_s=5)
    assert info.value.status_code == 504


def test_default_url_comes_from_settings(monkeypatch):
    from nlp_pillars.config import get_settings

    monkeypatch.setattr(get_settings(), "playground_runner_url", "http://elsewhere:9000/")
    assert PlaygroundRunnerClient().base_url == "http://elsewhere:9000"


# -- routes ---------------------------------------------------------------


def test_routes_proxy_every_call(api):
    client, seen, _ = api
    assert client.get("/api/playground/runner/health").json()["status"] == "ok"
    assert client.get("/api/playground/runner/sessions").json()["max_sessions"] == 3
    created = client.post("/api/playground/runner/sessions", json={"playground_id": "pg"})
    assert created.status_code == 201
    assert created.json()["session_id"] == "abc123"
    executed = client.post("/api/playground/runner/sessions/abc123/execute", json={"code": "6*7", "timeout_s": 30})
    assert executed.status_code == 200
    # Outputs come back in the tab's view shape, not the runner's.
    assert executed.json()["outputs"] == [{"kind": "text", "text": "42"}]
    assert executed.json()["execution_count"] == 1
    assert seen[-1]["body"] == {"code": "6*7", "timeout_s": 30}
    assert client.post("/api/playground/runner/sessions/abc123/interrupt").json()["interrupted"] is True
    assert client.post("/api/playground/runner/sessions/abc123/restart").json()["restarted"] is True
    assert client.delete("/api/playground/runner/sessions/abc123").json()["deleted"] is True


def test_routes_answer_503_naming_how_to_start_when_runner_is_down(api):
    client, _, state = api
    state["transport"] = httpx.MockTransport(_refuse)
    for method, path, body in (
        ("GET", "/api/playground/runner/health", None),
        ("GET", "/api/playground/runner/sessions", None),
        ("POST", "/api/playground/runner/sessions", {"playground_id": "pg"}),
        ("POST", "/api/playground/runner/sessions/abc123/execute", {"code": "1"}),
        ("POST", "/api/playground/runner/sessions/abc123/interrupt", None),
        ("POST", "/api/playground/runner/sessions/abc123/restart", None),
        ("DELETE", "/api/playground/runner/sessions/abc123", None),
    ):
        response = client.request(method, path, json=body)
        assert response.status_code == 503, path
        detail = response.json()["detail"]
        assert "./workstation up" in detail
        assert "not reachable" in detail


def test_routes_pass_runner_4xx_through_and_map_5xx_to_502(api):
    client, _, _ = api
    busy = client.post("/api/playground/runner/sessions/busy/execute", json={"code": "1"})
    assert busy.status_code == 409
    assert "still running" in busy.json()["detail"]
    full = client.post("/api/playground/runner/sessions", json={"playground_id": "full"})
    assert full.status_code == 429
    gone = client.post("/api/playground/runner/sessions/zzz/interrupt")
    assert gone.status_code == 404
    broken = client.post("/api/playground/runner/sessions/broken/execute", json={"code": "1"})
    assert broken.status_code == 502


def test_routes_validate_before_calling_the_runner(api):
    client, seen, _ = api
    assert client.post("/api/playground/runner/sessions/bad.id/execute", json={"code": "1"}).status_code == 422
    assert client.post("/api/playground/runner/sessions/abc123/execute", json={}).status_code == 422
    assert client.post("/api/playground/runner/sessions/abc123/execute", json={"code": "1", "timeout_s": -1}).status_code == 422
    assert client.post("/api/playground/runner/sessions", json={}).status_code == 422
    assert seen == []


def test_execute_refuses_a_non_json_body(api):
    # A cross-site form post cannot set application/json without a CORS
    # preflight, which this app never answers. FastAPI must not parse it.
    client, seen, _ = api
    response = client.post(
        "/api/playground/runner/sessions/abc123/execute",
        content=json.dumps({"code": "1"}),
        headers={"content-type": "text/plain"},
    )
    assert response.status_code == 422
    assert seen == []


def test_app_registers_the_runner_routes():
    from webui.app import create_app

    paths = set(create_app().openapi()["paths"])
    assert "/api/playground/runner/sessions/{session_id}/execute" in paths
    assert "/api/playground/runner/health" in paths


PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
RICH_OUTPUTS = [
    {"type": "stream", "name": "stdout", "text": "\x1b[31mrows: 3\x1b[0m\n"},
    {"type": "html", "untrusted": True, "text": "   a\n0  1",
     "html": '<style>td{}</style><table onclick="x()"><tr><td colspan="2">1</td>'
             '<td><a href="javascript:alert(1)">link</a></td></tr></table><script>alert(1)</script>'},
    {"type": "html", "untrusted": True, "text": "fallback", "html": "<script>only()</script>"},
    {"type": "image", "mime": "image/png", "data": PNG_B64, "text": "<Figure size 640x480>"},
    {"type": "image", "mime": "image/png", "data": "bm90IGEgcG5n", "text": "<Figure>"},
    {"type": "error", "ename": "ValueError", "evalue": "bad", "traceback": "Trace\n\x1b[31mValueError\x1b[0m: bad"},
    {"type": "text", "text": "cut[output truncated: limit of 10 bytes reached]", "truncated": True},
]


def test_execute_turns_runner_outputs_into_safe_view_shapes(api):
    client, _, _ = api
    body = client.post("/api/playground/runner/sessions/rich/execute", json={"code": "x"}).json()
    stream, html, html_fallback, png, bad_png, error, cut = body["outputs"]
    assert stream == {"kind": "stream", "name": "stdout", "text": "rows: 3\n"}
    # The same sanitizer as saved outputs: no raw markup, no attributes but spans, no script.
    assert html["kind"] == "html" and "html" not in html
    assert html["tree"] == [{"t": "table", "c": [{"t": "tr", "c": [
        {"t": "td", "a": {"colspan": "2"}, "c": ["1"]}, {"t": "td", "c": ["link"]}]}]}]
    assert html["plain"] == "   a\n0  1"
    assert html_fallback == {"kind": "text", "text": "fallback"}
    assert png == {"kind": "png", "data": PNG_B64, "alt": "<Figure size 640x480>"}
    assert bad_png == {"kind": "text", "text": "<Figure>"}
    assert error == {"kind": "error", "text": "Trace\nValueError: bad"}
    assert cut == {"kind": "text", "text": "cut[output truncated: limit of 10 bytes reached]", "truncated": True}
    # Everything else about the answer passes through.
    assert body["truncated"] is True and body["dropped_outputs"] == 1
    assert "javascript" not in json.dumps(body) and "script" not in json.dumps(body)


def test_routes_forward_the_setup_code_on_create_and_restart(api):
    client, seen, _ = api
    client.post("/api/playground/runner/sessions", json={"playground_id": "pg", "setup_code": "import numpy as np\n"})
    assert seen[-1]["body"] == {"playground_id": "pg", "setup_code": "import numpy as np\n"}
    # Setup code runs under the execution timeout, so the wait grows with it.
    assert seen[-1]["read_timeout"] > 900
    client.post("/api/playground/runner/sessions", json={"playground_id": "pg"})
    assert seen[-1]["body"] == {"playground_id": "pg", "setup_code": ""}
    assert seen[-1]["read_timeout"] == 90
    client.post("/api/playground/runner/sessions/abc123/restart", json={"setup_code": "import pandas as pd\n"})
    assert seen[-1]["body"] == {"setup_code": "import pandas as pd\n"}
    client.post("/api/playground/runner/sessions/abc123/restart")
    assert seen[-1]["body"] is None
