"""/api/connections: the learner's confirm / reject / edit. Mounted alone, store patched."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nlp_pillars.connections import ConnectionStoreError, Decision
from webui.routers.api import connections as route


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(route.router)
    return TestClient(app)


def test_decisions_are_listed_with_the_choices_the_form_needs(client, monkeypatch):
    monkeypatch.setattr(
        route, "load_decisions",
        lambda: [Decision("a", "b", "uses", "confirmed", "my words", updated_at="2026-10-30")],
    )
    body = client.get("/api/connections").json()
    assert body["decisions"][0] == {
        "src": "a", "dst": "b", "relation": "uses", "status": "confirmed",
        "phrase": "my words", "provenance": "user", "updated_at": "2026-10-30",
    }
    assert "builds_on" in body["relations"] and "related" in body["relations"]
    assert body["statuses"] == ["confirmed", "rejected"] and body["phrase_max_chars"] == 400


def test_confirming_with_an_edited_relation_and_phrase_stores_a_user_decision(client, monkeypatch):
    seen = {}

    def fake(a, b, *, status, relation, phrase):
        seen.update(a=a, b=b, status=status, relation=relation, phrase=phrase)
        return Decision(a, b, relation, status, phrase)

    monkeypatch.setattr(route, "put_decision", fake)
    response = client.put(
        "/api/connections",
        json={"a": "w1", "b": "w2", "status": "confirmed", "relation": "critiques", "phrase": "disputes the control task"},
    )
    assert response.status_code == 200 and response.json()["provenance"] == "user"
    assert seen == {"a": "w1", "b": "w2", "status": "confirmed", "relation": "critiques", "phrase": "disputes the control task"}


def test_rejecting_needs_only_the_pair(client, monkeypatch):
    monkeypatch.setattr(route, "put_decision", lambda a, b, *, status, relation, phrase: Decision(a, b, relation, status))
    response = client.put("/api/connections", json={"a": "w1", "b": "w2", "status": "rejected"})
    assert response.status_code == 200 and response.json()["status"] == "rejected"


@pytest.mark.parametrize(
    "body",
    [
        {"a": "w1", "b": "w2", "status": "maybe"},
        {"a": "w1", "b": "w2", "status": "confirmed", "relation": "refutes"},
        {"a": "w1", "b": " w1 ", "status": "confirmed"},
        {"a": "", "b": "w2", "status": "confirmed"},
        {"a": "w1", "status": "confirmed"},
    ],
)
def test_a_bad_decision_is_422_and_stores_nothing(client, monkeypatch, body):
    monkeypatch.setattr(route, "put_decision", lambda *a, **k: pytest.fail("nothing may be stored"))
    assert client.put("/api/connections", json=body).status_code == 422


def test_an_overlong_phrase_is_422(client, monkeypatch):
    def fake(a, b, *, status, relation, phrase):
        raise ValueError("The phrase is 401 characters; the limit is 400.")

    monkeypatch.setattr(route, "put_decision", fake)
    response = client.put("/api/connections", json={"a": "w1", "b": "w2", "status": "confirmed", "phrase": "x" * 401})
    assert response.status_code == 422 and "limit is 400" in response.json()["detail"]


def test_a_missing_migration_is_503_and_names_the_file(client, monkeypatch):
    def fail(*a, **k):
        raise ConnectionStoreError("not available. Apply docs/migrations/035_knowledge_layer.sql by hand.")

    monkeypatch.setattr(route, "load_decisions", fail)
    monkeypatch.setattr(route, "put_decision", fail)
    monkeypatch.setattr(route, "delete_decision", fail)
    for response in (
        client.get("/api/connections"),
        client.put("/api/connections", json={"a": "w1", "b": "w2", "status": "rejected"}),
        client.delete("/api/connections", params={"a": "w1", "b": "w2"}),
    ):
        assert response.status_code == 503 and "035_knowledge_layer.sql" in response.json()["detail"]


def test_undoing_a_decision_returns_the_pair_to_suggested(client, monkeypatch):
    monkeypatch.setattr(route, "delete_decision", lambda a, b: 1)
    assert client.delete("/api/connections", params={"a": "w1", "b": "w2"}).json() == {"removed": 1}
    monkeypatch.setattr(route, "delete_decision", lambda a, b: 0)
    assert client.delete("/api/connections", params={"a": "w1", "b": "w2"}).status_code == 404


def test_the_route_is_mounted_by_the_app(monkeypatch):
    from webui.app import create_app

    monkeypatch.setattr(route, "load_decisions", lambda: [])
    response = TestClient(create_app()).get("/api/connections")
    assert response.status_code == 200 and response.json()["decisions"] == []
