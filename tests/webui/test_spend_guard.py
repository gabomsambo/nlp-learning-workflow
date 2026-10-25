"""The expensive routes: confirmation through the tunnel, and the daily cap."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from nlp_pillars import spend
from webui.app import create_app
from webui.services import spend_guard

REMOTE = {"Cf-Ray": "8a1b2c3d4e5f-AMS"}
START = datetime(2026, 10, 25, tzinfo=timezone.utc)

#: Every route that spends money or the GPU. The guard runs before the handler,
#: so the body is irrelevant when it refuses; when it lets a request through the
#: handler may answer anything but 428/429/503-from-spend.
EXPENSIVE = [
    ("post", "/pipeline/run?pillar=neural-architectures-language"),
    ("post", "/api/pillars/neural-architectures-language/select"),
    ("post", "/api/pillars/neural-architectures-language/upload/url"),
    ("post", "/api/pillars/neural-architectures-language/upload/pdf"),
    ("post", "/api/podcast/generate"),
    ("post", "/api/podcast/tts/generate"),
    ("post", "/api/podcast/00000000-0000-4000-8000-000000000000/factcheck"),
]


def _budget(spent, cap=2.0):
    return spend.Budget(
        cap_usd=cap, spent_usd=spent, day_start=START, resets_at=START + timedelta(days=1),
    )


@pytest.fixture
def client():
    # Desktop-shaped requests: the Access middleware is tested in test_access.py,
    # and the remote/confirm branch of the guard is exercised on the dependency.
    return TestClient(create_app())


@pytest.mark.parametrize("method,path", EXPENSIVE)
def test_cap_reached_refuses_every_expensive_route(client, method, path):
    with patch.object(spend, "current_budget", return_value=_budget(2.0)):
        response = getattr(client, method)(path, json={})
    assert response.status_code == 429
    assert "cap" in response.json()["detail"].lower()


@pytest.mark.parametrize("method,path", EXPENSIVE)
def test_unreadable_spend_refuses_every_expensive_route(client, method, path):
    with patch.object(spend, "current_budget", side_effect=spend.SpendUnavailableError("down")):
        response = getattr(client, method)(path, json={})
    assert response.status_code == 503


def test_room_left_lets_the_request_reach_the_handler(client):
    with patch.object(spend, "current_budget", return_value=_budget(0.5)):
        response = client.post("/pipeline/run?pillar=not-a-pillar")
    # The handler's own validation answered, so the guard let it through.
    assert response.status_code == 400
    assert "Invalid pillar" in response.json()["error"]


@pytest.mark.asyncio
async def test_remote_request_without_confirmation_is_refused():
    from fastapi import HTTPException
    from starlette.requests import Request

    scope = {
        "type": "http",
        "headers": [(b"cf-ray", b"1")],
        "method": "POST",
        "path": "/x",
    }
    with patch.object(spend, "current_budget", return_value=_budget(0.0)) as current:
        with pytest.raises(HTTPException) as excinfo:
            await spend_guard.require_spend_room(Request(scope))
    assert excinfo.value.status_code == 428
    current.assert_not_called()


@pytest.mark.asyncio
async def test_remote_request_with_confirmation_is_checked_against_the_cap():
    from fastapi import HTTPException
    from starlette.requests import Request

    scope = {
        "type": "http",
        "headers": [(b"cf-ray", b"1"), (b"x-nlp-confirm", b"1")],
        "method": "POST",
        "path": "/x",
    }
    with patch.object(spend, "current_budget", return_value=_budget(0.0)):
        await spend_guard.require_spend_room(Request(scope))  # no raise
    with patch.object(spend, "current_budget", return_value=_budget(3.0)):
        with pytest.raises(HTTPException) as excinfo:
            await spend_guard.require_spend_room(Request(scope))
    assert excinfo.value.status_code == 429


def test_spend_endpoint_reports_the_budget(client):
    with patch.object(spend, "current_budget", return_value=_budget(0.5)):
        body = client.get("/api/spend").json()
    assert body["cap_usd"] == 2.0
    assert body["spent_usd"] == 0.5
    assert body["remaining_usd"] == 1.5
    assert body["exhausted"] is False


def test_spend_endpoint_is_503_when_unreadable(client):
    with patch.object(spend, "current_budget", side_effect=spend.SpendUnavailableError("down")):
        assert client.get("/api/spend").status_code == 503
