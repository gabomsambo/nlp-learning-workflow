"""POST /api/pillars/{id}/queue wires Enqueue to paper_queue."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nlp_pillars.db import LibraryEnqueueError
from webui.routers.api import pillars as api_pillars


def _client():
    app = FastAPI()
    app.include_router(api_pillars.router)
    return TestClient(app)


def test_enqueue_returns_queued(monkeypatch):
    monkeypatch.setattr(api_pillars, "pillar_exists", lambda pillar_id: True)
    monkeypatch.setattr(
        api_pillars,
        "enqueue_library_paper",
        lambda pillar_id, paper_id: {
            "status": "queued",
            "paper_id": paper_id,
            "title": "Attention",
        },
    )

    response = _client().post(
        "/api/pillars/ai-agents-tool-use/queue",
        json={"paper_id": "file:abc"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert response.json()["paper_id"] == "file:abc"


def test_unknown_pillar_is_404(monkeypatch):
    monkeypatch.setattr(api_pillars, "pillar_exists", lambda pillar_id: False)

    response = _client().post(
        "/api/pillars/missing-pillar/queue",
        json={"paper_id": "file:abc"},
    )

    assert response.status_code == 404


def test_unresolvable_paper_is_422(monkeypatch):
    monkeypatch.setattr(api_pillars, "pillar_exists", lambda pillar_id: True)

    def boom(pillar_id, paper_id):
        raise LibraryEnqueueError("unresolvable", "no pdf")

    monkeypatch.setattr(api_pillars, "enqueue_library_paper", boom)

    response = _client().post(
        "/api/pillars/ai-agents-tool-use/queue",
        json={"paper_id": "file:abc"},
    )

    assert response.status_code == 422
    assert "no pdf" in response.json()["detail"]


def test_queue_outage_is_503(monkeypatch):
    monkeypatch.setattr(api_pillars, "pillar_exists", lambda pillar_id: True)

    def boom(pillar_id, paper_id):
        raise LibraryEnqueueError("unavailable", "postgrest down")

    monkeypatch.setattr(api_pillars, "enqueue_library_paper", boom)

    response = _client().post(
        "/api/pillars/ai-agents-tool-use/queue",
        json={"paper_id": "file:abc"},
    )

    assert response.status_code == 503
