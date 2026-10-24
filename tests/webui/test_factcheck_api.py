"""POST and GET /api/podcast/{id}/factcheck.

The run itself is covered in tests/test_factcheck_runs.py. These tests only
check the HTTP contract: 202, 409 attach, 503 naming the migration, and a
GET body that keeps quotes and anchors.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nlp_pillars.db import FactcheckLookupError, PipelineRunCreateError
from webui.services.podcast_factcheck_service import FactcheckAlreadyRunning
from webui.services.run_service import RunAlreadyActiveError

SCRIPT_ID = "b4269725-1111-4111-8111-111111111111"


def _script():
    return SimpleNamespace(
        id=SCRIPT_ID,
        paper_id="arxiv:2403.05525",
        pillar_id="neural-architectures-language",
        title="AML",
    )


@pytest.fixture
def api_client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from webui.routers.api import podcast as podcast_api

    monkeypatch.setattr(
        "webui.routers.api.podcast.get_podcast_script_by_id",
        lambda sid: _script(),
    )
    app = FastAPI()
    app.state.scheduler = MagicMock()
    app.state.cancel_events = {}
    app.include_router(podcast_api.router)
    return TestClient(app, raise_server_exceptions=False)


def test_post_starts_a_run(api_client, monkeypatch):
    seen = {}

    def dispatch(scheduler, events, *, script, trigger_source):
        seen["script_id"] = script.id
        seen["trigger"] = trigger_source
        seen["scheduler"] = scheduler
        return "run-new"

    monkeypatch.setattr(
        "webui.routers.api.podcast.podcast_factcheck_service.dispatch_factcheck",
        dispatch,
    )
    response = api_client.post(f"/api/podcast/{SCRIPT_ID}/factcheck")
    assert response.status_code == 202
    assert response.json()["run_id"] == "run-new"
    assert seen["script_id"] == SCRIPT_ID
    assert seen["trigger"] == "ui_podcast_factcheck"
    assert seen["scheduler"] is api_client.app.state.scheduler


def test_post_409_includes_the_active_run_id(api_client, monkeypatch):
    monkeypatch.setattr(
        "webui.routers.api.podcast.podcast_factcheck_service.dispatch_factcheck",
        lambda *a, **k: (_ for _ in ()).throw(FactcheckAlreadyRunning("run-live")),
    )
    response = api_client.post(f"/api/podcast/{SCRIPT_ID}/factcheck")
    assert response.status_code == 409
    body = response.json()
    assert body["run_id"] == "run-live"
    assert "already running" in body["detail"]


def test_post_409_after_a_race_attaches_to_the_winner(api_client, monkeypatch):
    monkeypatch.setattr(
        "webui.routers.api.podcast.podcast_factcheck_service.dispatch_factcheck",
        lambda *a, **k: (_ for _ in ()).throw(RunAlreadyActiveError("conflict")),
    )
    monkeypatch.setattr(
        "webui.routers.api.podcast.get_active_factcheck_run",
        lambda sid: SimpleNamespace(id="run-winner"),
    )
    response = api_client.post(f"/api/podcast/{SCRIPT_ID}/factcheck")
    assert response.status_code == 409
    assert response.json()["run_id"] == "run-winner"


def test_post_503_names_migration_023(api_client, monkeypatch):
    monkeypatch.setattr(
        "webui.routers.api.podcast.podcast_factcheck_service.dispatch_factcheck",
        lambda *a, **k: (_ for _ in ()).throw(
            PipelineRunCreateError("new row violates check constraint")
        ),
    )
    response = api_client.post(f"/api/podcast/{SCRIPT_ID}/factcheck")
    assert response.status_code == 503
    assert "023_podcast_factcheck_runs.sql" in response.json()["detail"]


def test_get_returns_claims_verdicts_quotes_and_anchors(api_client, monkeypatch):
    monkeypatch.setattr(
        "webui.routers.api.podcast.get_latest_podcast_fact_check",
        lambda sid: {
            "id": "fc-1",
            "script_id": sid,
            "script_sha256": "abc",
            "paper_text_sha256": "def",
            "status": "succeeded",
            "models": {"screen": "claude-haiku-5-5"},
            "prompt_version": "v2",
            "counts": {"claims": 1, "high": 1, "contradicted": 1},
            "cost_usd": "0.110000",
            "created_at": "2026-10-24T00:00:00+00:00",
            "claims": [{
                "id": "cl-1",
                "line_index": 12,
                "span_start": 4,
                "span_end": 20,
                "span_text": "six out of eight",
                "claim": "VDCNN wins on six of eight datasets.",
                "scope": "paper",
                "category": "comparison",
                "verify_marker": False,
                "verdict": "contradicted",
                "error_type": "number_mismatch",
                "severity": "high",
                "rationale": "The table shows five wins.",
                "source_says": "Table 1",
                "evidence": [{
                    "quote": "VDCNN is better on 5",
                    "chunk_id": "chunk-9",
                    "chunk_index": 4,
                    "char_start": 120,
                    "char_end": 142,
                    "match_score": 100,
                }],
                "evidence_status": "located",
                "judge_trail": [{
                    "step": "rejudge",
                    "model": "claude-sonnet-5-5",
                    "verdict": "contradicted",
                }],
            }],
        },
    )
    response = api_client.get(f"/api/podcast/{SCRIPT_ID}/factcheck")
    assert response.status_code == 200
    body = response.json()
    assert body["fact_check_id"] == "fc-1"
    assert body["counts"]["high"] == 1
    assert body["cost_usd"] == pytest.approx(0.11)
    claim = body["claims"][0]
    assert claim["verdict"] == "contradicted"
    assert claim["evidence"][0]["quote"] == "VDCNN is better on 5"
    assert claim["evidence"][0]["char_start"] == 120
    assert claim["evidence"][0]["chunk_index"] == 4


def test_get_503_names_migration_022(api_client, monkeypatch):
    def missing(_sid):
        raise FactcheckLookupError(
            "Could not read the fact check: PGRST205. "
            "Apply docs/migrations/022_podcast_fact_checks.sql by hand."
        )

    monkeypatch.setattr(
        "webui.routers.api.podcast.get_latest_podcast_fact_check",
        missing,
    )
    response = api_client.get(f"/api/podcast/{SCRIPT_ID}/factcheck")
    assert response.status_code == 503
    assert "022_podcast_fact_checks.sql" in response.json()["detail"]


def test_get_404_when_the_script_has_no_check(api_client, monkeypatch):
    monkeypatch.setattr(
        "webui.routers.api.podcast.get_latest_podcast_fact_check",
        lambda sid: None,
    )
    response = api_client.get(f"/api/podcast/{SCRIPT_ID}/factcheck")
    assert response.status_code == 404
    assert response.json()["detail"] == "No fact check for this script"


def test_a_non_uuid_script_id_is_404(api_client):
    response = api_client.post("/api/podcast/not-a-uuid/factcheck")
    assert response.status_code == 404
    assert response.json()["detail"] == "Script not found"
    response = api_client.get("/api/podcast/not-a-uuid/factcheck")
    assert response.status_code == 404


def test_unknown_script_is_404(api_client, monkeypatch):
    monkeypatch.setattr(
        "webui.routers.api.podcast.get_podcast_script_by_id",
        lambda sid: None,
    )
    response = api_client.post(f"/api/podcast/{SCRIPT_ID}/factcheck")
    assert response.status_code == 404
    assert response.json()["detail"] == "Script not found"
