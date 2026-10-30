"""POST /api/pillars/{id}/queue/external and GET .../queue/fit. No network, no database."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nlp_pillars import map_store, map_view, queue_external
from nlp_pillars.schemas import PaperRef
from nlp_pillars.tools.semantic_scholar_tool import (
    SemanticScholarError,
    SemanticScholarUnavailableError,
)
from webui.routers.api import queue_external as route

PILLAR = "interpretability"
S2_ID = "a" * 40


def _client():
    app = FastAPI()
    app.include_router(route.router)
    return TestClient(app)


def _ref(**overrides):
    base = dict(id="2001.00001", title="A paper", authors=["A"], year=2020,
                url_pdf="https://arxiv.org/pdf/2001.00001.pdf")
    base.update(overrides)
    return PaperRef(**base)


class FakeS2:
    ref = None
    error = None
    calls = []

    def get_paper(self, lookup):
        FakeS2.calls.append(lookup)
        if FakeS2.error:
            raise FakeS2.error
        return FakeS2.ref


@pytest.fixture
def stub(monkeypatch):
    FakeS2.ref, FakeS2.error, FakeS2.calls = _ref(), None, []
    state = {"added": 1, "queued": []}

    def add(pillar_id, papers, priority=5):
        state["queued"].append((pillar_id, [p.id for p in papers]))
        return state["added"]

    monkeypatch.setattr(route, "pillar_exists", lambda pid: pid == PILLAR)
    monkeypatch.setattr(queue_external, "SemanticScholarTool", FakeS2)
    monkeypatch.setattr(queue_external, "add_to_paper_queue", add)
    return state


def _post(s2_id=S2_ID, pillar=PILLAR):
    return _client().post(f"/api/pillars/{pillar}/queue/external", json={"s2_id": s2_id})


def test_queues_an_arxiv_paper_without_a_model_call(stub):
    res = _post()
    assert res.status_code == 200
    assert res.json()["status"] == "queued"
    assert stub["queued"] == [(PILLAR, ["2001.00001"])]
    assert FakeS2.calls == [S2_ID]


def test_open_pdf_without_arxiv_id_is_queueable(stub):
    FakeS2.ref = _ref(id="10.1/abc", url_pdf="https://example.org/p.pdf")
    assert _post().json()["source"] == "url"


def test_no_open_pdf_is_422_and_nothing_is_queued(stub):
    FakeS2.ref = _ref(id="10.1/abc", url_pdf=None)
    res = _post()
    assert res.status_code == 422
    assert "no open PDF" in res.json()["detail"]
    assert stub["queued"] == []


def test_already_queued_is_reported_not_an_error(stub):
    stub["added"] = 0
    assert _post().json()["status"] == "already_queued"


def test_unknown_pillar_and_unknown_paper_are_404(stub):
    assert _post(pillar="nope").status_code == 404
    FakeS2.ref = None
    assert _post().status_code == 404


def test_malformed_id_never_reaches_semantic_scholar(stub):
    assert _post("../../graph/v1/author/1").status_code == 422
    assert FakeS2.calls == []


def test_semantic_scholar_outage_is_503(stub):
    FakeS2.error = SemanticScholarUnavailableError("rate limited")
    assert _post().status_code == 503
    FakeS2.error = SemanticScholarError("boom")
    assert _post().status_code == 503


def test_prefixed_ids_are_accepted(stub):
    assert _post("ARXIV:2001.00001").status_code == 200


# --------------------------------------------------------------------- fit


def _snapshot(queue):
    return {"scope": PILLAR, "built_at": "2026-10-30T10:00:00+00:00",
            "payload": {"nodes": [], "queue": queue}}


def _fit_stub(monkeypatch, snapshot, live):
    monkeypatch.setattr(route, "pillar_exists", lambda pid: pid == PILLAR)
    monkeypatch.setattr(map_view, "read_snapshot", lambda scope: snapshot)
    monkeypatch.setattr(map_store, "load_queue", lambda pillar: live)


def test_fit_keys_verdicts_by_paper_id_and_skips_unscored_rows(monkeypatch):
    live = [
        {"id": "q1", "paper_id": "2001.00001", "title": "one", "processed": False},
        {"id": "q2", "paper_id": "2001.00002", "title": "two", "processed": False},
        {"id": "q3", "paper_id": "2001.00003", "title": "queued after the build",
         "processed": False},
    ]
    snap = _snapshot([
        {"queue_id": "q1", "verdict": "close_strong", "rank_fraction": 0.03},
        {"queue_id": "q2", "verdict": "off_topic", "rank_fraction": 0.8},
    ])
    _fit_stub(monkeypatch, snap, live)
    body = _client().get(f"/api/pillars/{PILLAR}/queue/fit").json()
    assert body["status"] == "ready"
    assert body["fits"] == {
        "2001.00001": {"verdict": "close_strong", "rank_fraction": 0.03},
        "2001.00002": {"verdict": "off_topic", "rank_fraction": 0.8},
    }


def test_fit_before_any_build_is_not_built_and_empty(monkeypatch):
    _fit_stub(monkeypatch, None, [])
    body = _client().get(f"/api/pillars/{PILLAR}/queue/fit").json()
    assert body == {"status": "not_built", "built_at": None, "fits": {}}


def test_fit_missing_table_is_503(monkeypatch):
    monkeypatch.setattr(route, "pillar_exists", lambda pid: True)

    def boom(scope):
        raise map_store.MapStoreError("run docs/migrations/034_map_snapshots.sql")

    monkeypatch.setattr(map_view, "read_snapshot", boom)
    res = _client().get(f"/api/pillars/{PILLAR}/queue/fit")
    assert res.status_code == 503
    assert "034" in res.json()["detail"]
