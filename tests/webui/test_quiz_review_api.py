"""POST /api/quiz/review: optional event_id and reviewed_at reach the review, bad ones stop at the door."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nlp_pillars.schemas import QuizReviewResponse
from webui.routers.api import quiz as quiz_api

EVENT = "00000000-0000-4000-8000-000000000001"


@pytest.fixture
def seen(monkeypatch):
    requests = []

    def fake_review(request):
        requests.append(request)
        return QuizReviewResponse(
            success=True, card=None,
            next_review_date=datetime(2026, 3, 12, tzinfo=timezone.utc),
            message="Card reviewed successfully",
            event_id=request.event_id or "generated", duplicate=request.event_id == EVENT,
        )

    monkeypatch.setattr("nlp_pillars.db.review_quiz_card_fsrs", fake_review)
    return requests


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(quiz_api.router)
    return TestClient(app)


def _stamp(**delta):
    return (datetime.now(timezone.utc) - timedelta(**delta)).isoformat()


def test_event_id_and_reviewed_at_are_passed_through(client, seen):
    when = _stamp(hours=2)
    response = client.post("/api/quiz/review", json={
        "card_id": "c1", "rating": 3, "event_id": EVENT, "reviewed_at": when,
    })

    assert response.status_code == 200
    body = response.json()
    assert body["event_id"] == EVENT and body["duplicate"] is True
    assert seen[0].event_id == EVENT
    assert seen[0].reviewed_at == datetime.fromisoformat(when)


def test_both_are_optional(client, seen):
    response = client.post("/api/quiz/review", json={"card_id": "c1", "rating": 3})
    assert response.status_code == 200
    assert seen[0].event_id is None and seen[0].reviewed_at is None
    assert response.json()["duplicate"] is False


@pytest.mark.parametrize("payload, text", [
    ({"event_id": "nope"}, "UUID"),
    ({"reviewed_at": _stamp(days=-1)}, "future"),
    ({"reviewed_at": _stamp(days=60)}, "days old"),
    ({"reviewed_at": "2026-03-10T09:00:00"}, "UTC offset"),
    ({"rating": 7}, "Invalid rating"),
])
def test_unusable_input_is_a_400_and_never_reaches_the_review(client, seen, payload, text):
    body = {"card_id": "c1", "rating": 3, **payload}
    response = client.post("/api/quiz/review", json=body)
    assert response.status_code == 400
    assert text in response.json()["detail"]
    assert seen == []
