"""Tests for the pillar page's Progress Overview counts.

The bug: `stats` was `len()` of three `limit=5` lists, so "Papers Processed"
could never read above 5. The fix counts server-side via `Prefer: count=exact`
and `Content-Range`.

The second half of the contract matters as much as the first: a count that
cannot be obtained must NOT render as 0. `counts_for_pillar` raises
`CountUnavailableError` where the neighbouring `counts_by_pillar()` degrades to
zero, and the router turns that into `stats=None` for an explicit unknown state.

httpx.MockTransport through the existing `client=` injection point, matching
tests/webui/test_pipeline_runs_api.py. Basename is unique across tests/ because
there are no __init__.py files anywhere under it.
"""

import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.services.postgrest_client import (  # noqa: E402
    CountUnavailableError,
    PostgrestClient,
)


def _client(handler):
    return PostgrestClient(
        base_url="http://example",
        token=None,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _counting(totals):
    """Answer each table with one row and a Content-Range naming its total."""

    def handler(request: httpx.Request) -> httpx.Response:
        table = request.url.path.strip("/")
        total = totals[table]
        return httpx.Response(
            200,
            json=[{"id": "x"}] if total else [],
            headers={"Content-Range": f"0-0/{total}" if total else f"*/{total}"},
        )

    return handler


@pytest.mark.asyncio
async def test_counts_are_totals_not_capped_at_the_recent_list_size():
    client = _client(_counting({"papers": 42, "lessons": 17, "quiz_cards": 300}))
    counts = await client.counts_for_pillar("models-architectures")
    assert counts == {"papers": 42, "lessons": 17, "quiz_cards": 300}


@pytest.mark.asyncio
async def test_counts_filter_by_pillar_and_fetch_one_row():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path.strip("/"), dict(request.url.params)))
        return httpx.Response(200, json=[], headers={"Content-Range": "*/0"})

    await _client(handler).counts_for_pillar("efficiency")

    assert {t for t, _ in seen} == {"papers", "lessons", "quiz_cards"}
    for _, params in seen:
        assert params["pillar_id"] == "eq.efficiency"
        assert params["limit"] == "1"
        assert params["select"] == "id"


@pytest.mark.asyncio
async def test_zero_is_a_real_answer():
    client = _client(_counting({"papers": 0, "lessons": 0, "quiz_cards": 0}))
    assert await client.counts_for_pillar("p") == {
        "papers": 0,
        "lessons": 0,
        "quiz_cards": 0,
    }


@pytest.mark.asyncio
async def test_missing_content_range_raises_rather_than_reading_zero():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    with pytest.raises(CountUnavailableError):
        await _client(handler).counts_for_pillar("p")


@pytest.mark.asyncio
async def test_http_error_raises_rather_than_reading_zero():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.strip("/") == "lessons":
            return httpx.Response(503, text="upstream down")
        return httpx.Response(200, json=[], headers={"Content-Range": "*/0"})

    with pytest.raises(CountUnavailableError) as exc:
        await _client(handler).counts_for_pillar("p")
    assert "lessons" in str(exc.value)


@pytest.mark.asyncio
async def test_unparseable_content_range_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[], headers={"Content-Range": "0-0/*"})

    with pytest.raises(CountUnavailableError):
        await _client(handler).counts_for_pillar("p")


# --------------------------------------------------- the page the counts feed


@pytest.fixture
def page_client(monkeypatch):
    """A TestClient over just the pillars page router, with templates wired.

    Deliberately not the whole app: create_app()'s lifespan starts a scheduler and
    sweeps the database, none of which this page needs.
    """
    from fastapi import FastAPI
    from fastapi.templating import Jinja2Templates
    from fastapi.testclient import TestClient

    from datetime import datetime, timezone

    from nlp_pillars.schemas import Pillar
    from webui.routers import pillars as pillars_router

    monkeypatch.setattr(
        pillars_router,
        "get_pillar_by_id",
        lambda pillar_id: Pillar(
            id=pillar_id,
            name="Test Pillar",
            goal="A goal long enough to satisfy the schema's minimum length.",
            focus_areas=[],
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    )
    # The recent lists are capped at 5 by design; the counts must not follow them.
    monkeypatch.setattr(pillars_router, "get_papers", lambda *a, **k: [])
    monkeypatch.setattr(pillars_router, "get_lessons", lambda *a, **k: [])
    monkeypatch.setattr(pillars_router, "get_quiz_cards_for_review", lambda *a, **k: [])

    app = FastAPI()
    app.include_router(pillars_router.router)
    app.state.templates = Jinja2Templates(directory=str(ROOT / "webui" / "templates"))
    return TestClient(app, raise_server_exceptions=False)


def test_page_shows_a_total_well_past_the_recent_list_cap(page_client, monkeypatch):
    async def counts(self, pillar_id):
        return {"papers": 42, "lessons": 17, "quiz_cards": 300}

    monkeypatch.setattr(PostgrestClient, "counts_for_pillar", counts)

    body = page_client.get("/pillars/p1").text
    assert ">42<" in body
    assert ">17<" in body
    assert ">300<" in body


def test_bar_clamps_when_a_total_passes_its_goal(page_client, monkeypatch):
    async def counts(self, pillar_id):
        return {"papers": 100000, "lessons": 0, "quiz_cards": 0}

    monkeypatch.setattr(PostgrestClient, "counts_for_pillar", counts)

    body = page_client.get("/pillars/p1").text
    widths = [float(c.split("%")[0]) for c in body.split('style="width: ')[1:]]
    assert widths, "no progress bars rendered"
    # 100000 papers against a goal of 100 fills the bar; it must not overflow it.
    assert max(widths) == 100.0
    assert all(0.0 <= w <= 100.0 for w in widths)


def test_a_failed_count_renders_unknown_and_never_zero(page_client, monkeypatch):
    async def boom(self, pillar_id):
        raise CountUnavailableError("postgrest is down")

    monkeypatch.setattr(PostgrestClient, "counts_for_pillar", boom)

    res = page_client.get("/pillars/p1")
    assert res.status_code == 200
    body = res.text
    assert "Count unavailable" in body
    assert "&mdash;" in body or "—" in body
    # The page must not claim the pillar is empty.
    assert ">0<" not in body
