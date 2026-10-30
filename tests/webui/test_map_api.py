"""GET /api/map: snapshot plus live overlays. Nothing here reaches a database."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nlp_pillars import map_store, map_view
from nlp_pillars.graph_store import GraphStoreError
from nlp_pillars.map_store import MapStoreError
from webui.routers.api.map import router

PILLAR = "interpretability"


def _client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _payload(nodes=None, queue=None):
    return {
        "version": 1,
        "scope": PILLAR,
        "nodes": nodes
        if nodes is not None
        else [
            {"id": "w1", "kind": "library", "paper_ids": ["arxiv:1"]},
            {"id": "w2", "kind": "library", "paper_ids": ["arxiv:2"]},
            {"id": "w3", "kind": "external", "paper_ids": []},
        ],
        "queue": queue
        if queue is not None
        else [
            {"queue_id": "q1", "node": "w3", "verdict": "close", "processed": False,
             "rank_fraction": 0.02},
            {"queue_id": "q2", "node": None, "verdict": "off_topic", "processed": False},
        ],
    }


def _row(payload):
    return {
        "scope": PILLAR,
        "built_at": "2026-10-30T10:00:00+00:00",
        "params_hash": "p",
        "payload_hash": "h",
        "model_id": "bge",
        "run_id": None,
        "stats": {},
        "payload": payload,
    }


@pytest.fixture
def stub(monkeypatch):
    """Healthy store: one snapshot, a queue, mastery for two concepts."""
    state = {
        "snapshot": _row(_payload()),
        "pillars": [PILLAR],
        "queue": [
            {"id": "q1", "paper_id": "arxiv:9", "title": "Nine", "processed": True},
            {"id": "q3", "paper_id": "arxiv:7", "title": "Seven", "processed": False},
        ],
    }

    def read(table, columns, order, narrow=None):
        if table == "map_snapshots":
            return [state["snapshot"]] if state["snapshot"] else []
        if table == "pillars":
            return [{"id": p} for p in state["pillars"]]
        raise AssertionError(table)

    monkeypatch.setattr(map_store, "_read", read)
    monkeypatch.setattr(map_store, "load_queue", lambda pillar: state["queue"])
    monkeypatch.setattr(
        map_store,
        "_read_in",
        lambda table, *a, **k: [
            {"paper_id": "arxiv:1", "concept_id": "c1"},
            {"paper_id": "arxiv:1", "concept_id": "c2"},
            {"paper_id": "arxiv:2", "concept_id": "c3"},
        ],
    )
    monkeypatch.setattr(
        map_view,
        "load_concept_mastery",
        lambda **_k: [
            {"concept_id": "c1", "retrievability": 0.8, "card_count": 2, "reviewed_card_count": 2},
            {"concept_id": "c2", "retrievability": 0.4, "card_count": 1, "reviewed_card_count": 0},
            {"concept_id": "c3", "retrievability": None, "card_count": 0, "reviewed_card_count": 0},
        ],
    )
    return state


def test_ready_map_carries_snapshot_and_overlays(stub):
    body = _client().get("/api/map", params={"scope": PILLAR}).json()
    assert body["status"] == "ready"
    assert body["snapshot"]["nodes"][0]["id"] == "w1"
    assert body["built_at"] == "2026-10-30T10:00:00+00:00"

    queue = body["overlays"]["queue"]
    assert queue["available"] is True
    by_id = {item["queue_id"]: item for item in queue["items"]}
    # q1 was in the build; it is processed now and the overlay says so
    assert by_id["q1"]["verdict"] == "close"
    assert by_id["q1"]["processed_since_build"] is True
    # q3 was queued after the build: no verdict, not an invented one
    assert by_id["q3"]["in_snapshot"] is False
    assert by_id["q3"]["verdict"] is None
    assert [row["queue_id"] for row in queue["removed_since_build"]] == ["q2"]
    assert queue["counts"] == {
        "queued": 2, "pending": 1, "not_in_snapshot": 1, "removed_since_build": 1,
    }

    mastery = body["overlays"]["mastery"]
    assert mastery["papers"]["arxiv:1"]["retrievability"] == 0.6
    assert mastery["papers"]["arxiv:1"]["cards"] == 3
    # a paper whose only concept has no cards is unknown, not zero
    assert mastery["papers"]["arxiv:2"]["retrievability"] is None
    assert "w3" not in mastery["nodes"]
    assert mastery["nodes"]["w1"]["retrievability"] == 0.6


def test_never_built_is_distinct_from_empty(stub):
    stub["snapshot"] = None
    body = _client().get("/api/map", params={"scope": PILLAR}).json()
    assert body["status"] == "not_built"
    assert body["snapshot"] is None and body["overlays"] is None
    assert f"map build --pillar {PILLAR}" in body["message"]

    stub["snapshot"] = _row(_payload(nodes=[], queue=[]))
    body = _client().get("/api/map", params={"scope": PILLAR}).json()
    assert body["status"] == "empty"
    assert body["snapshot"]["nodes"] == []
    assert body["overlays"]["queue"]["available"] is True


def test_unknown_pillar_is_404_but_all_is_not_built(stub):
    stub["snapshot"] = None
    assert _client().get("/api/map", params={"scope": "nope"}).status_code == 404
    body = _client().get("/api/map", params={"scope": "all"}).json()
    assert body["status"] == "not_built"


def test_missing_table_is_503_and_names_the_migration(monkeypatch):
    def read(*_a, **_k):
        raise map_store._error("relation map_snapshots does not exist")

    monkeypatch.setattr(map_store, "_read", read)
    response = _client().get("/api/map", params={"scope": PILLAR})
    assert response.status_code == 503
    assert "034_map_snapshots.sql" in response.json()["detail"]


def test_a_failed_overlay_degrades_without_losing_the_snapshot(stub, monkeypatch):
    def broken(**_k):
        raise GraphStoreError("Apply docs/migrations/024_vocabulary_concepts.sql by hand.")

    monkeypatch.setattr(map_view, "load_concept_mastery", broken)
    monkeypatch.setattr(
        map_store, "load_queue",
        lambda pillar: (_ for _ in ()).throw(MapStoreError("queue down")),
    )
    response = _client().get("/api/map", params={"scope": PILLAR})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["overlays"]["mastery"]["available"] is False
    assert "024_vocabulary_concepts.sql" in body["overlays"]["mastery"]["reason"]
    assert body["overlays"]["queue"] == {"available": False, "reason": "queue down"}


def test_all_scope_has_no_queue_overlay(stub):
    row = _row(_payload())
    row["scope"] = "all"
    stub["snapshot"] = row
    body = _client().get("/api/map", params={"scope": "all"}).json()
    assert body["overlays"]["queue"]["available"] is False


def test_scope_is_required():
    assert _client().get("/api/map").status_code == 422
    assert _client().get("/api/map", params={"scope": "  "}).status_code == 422
