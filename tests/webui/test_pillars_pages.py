"""Pillars list progress, and the contracts the pillar page must keep."""

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.templating import Jinja2Templates

from nlp_pillars.schemas import Pillar
from webui.routers import pillars as pillars_router
from webui.services.postgrest_client import CountUnavailableError, PostgrestClient

ROOT = Path(__file__).resolve().parents[2]


def _pillar(pillar_id="ai-agents-tool-use"):
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Pillar(
        id=pillar_id,
        name="Agents",
        goal="A goal long enough to satisfy the schema minimum.",
        focus_areas=["tools", "planning"],
        papers_per_day=2,
        created_at=when,
        updated_at=when,
    )


def _app():
    app = FastAPI()
    app.include_router(pillars_router.router)
    app.state.templates = Jinja2Templates(directory=str(ROOT / "webui" / "templates"))
    return TestClient(app, raise_server_exceptions=False)


def test_list_shows_real_counts_and_never_a_zero_for_an_unknown(monkeypatch):
    monkeypatch.setattr(pillars_router, "get_pillars_or_empty", lambda limit=100: [_pillar(), _pillar("ai-safety-alignment")])

    async def counts(self, pillar_id):
        if pillar_id == "ai-safety-alignment":
            raise CountUnavailableError("down")
        return {"papers": 12, "lessons": 3, "quiz_cards": 9}

    monkeypatch.setattr(PostgrestClient, "counts_for_pillar", counts)

    body = _app().get("/pillars/").text
    assert ">12<" in body
    assert ">3<" in body
    assert ">9<" in body
    assert "Count unavailable" in body
    # The unknown pillar must not be painted as empty. The known one has no zero.
    assert ">0<" not in body
    assert 'class="btn btn-primary btn-new-pillar"' in body


def test_save_control_is_not_a_submit_and_edit_stays_on_the_page():
    dialogs = (ROOT / "webui" / "templates" / "_pillar_dialogs.html").read_text()
    save = dialogs.split('id="modal-save"', 1)[0].rsplit("<button", 1)[1]
    assert 'type="button"' in save

    detail = (ROOT / "webui" / "templates" / "pillar_detail.html").read_text()
    assert "activity-item-link" in detail
    assert "/pillars/${" not in detail
    assert "status_id='upload-run-status'" in detail
    assert 'id="search-form"' in detail
    assert 'data-pillar-id="{{ pillar.id }}"' in detail
    assert "/discovery/?pillar=" in detail
    assert "/papers/details/" not in detail


def test_detail_page_still_has_the_quick_links(page_ready):
    body = page_ready.get("/pillars/ai-agents-tool-use").text
    assert 'href="/discovery/?pillar=ai-agents-tool-use"' in body
    assert 'href="/pipeline/?pillar=ai-agents-tool-use"' in body
    assert 'href="/papers/?pillar=ai-agents-tool-use"' in body
    assert 'href="/lessons/?pillar=ai-agents-tool-use"' in body
    assert 'href="/quiz/?pillar=ai-agents-tool-use"' in body
    assert "btn-edit" in body
    assert 'id="pillar-page"' in body


import pytest


@pytest.fixture
def page_ready(monkeypatch):
    monkeypatch.setattr(pillars_router, "get_pillar_by_id", lambda pillar_id: _pillar(pillar_id))
    monkeypatch.setattr(pillars_router, "get_papers", lambda *a, **k: [])
    monkeypatch.setattr(pillars_router, "get_lessons", lambda *a, **k: [])
    monkeypatch.setattr(pillars_router, "get_quiz_cards_for_review", lambda *a, **k: [])

    async def counts(self, pillar_id):
        return {"papers": 4, "lessons": 1, "quiz_cards": 2}

    monkeypatch.setattr(PostgrestClient, "counts_for_pillar", counts)
    return _app()
