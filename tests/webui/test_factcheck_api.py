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
    from webui.services.spend_guard import require_spend_room
    app = FastAPI()
    app.state.scheduler = MagicMock()
    app.state.cancel_events = {}
    app.include_router(podcast_api.router)
    # The spend guard has its own tests (test_spend_guard.py); here it would only
    # try to read the database.
    app.dependency_overrides[require_spend_room] = lambda: None
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


# --------------------------------------------------------------------- staleness


def test_the_script_response_carries_the_hash_the_fact_check_was_run_against(
    api_client, monkeypatch
):
    """The script response carries the hash a fact check was run against.

    Listen needs the hash of the script text to tell a fresh fact check from
    a stale one.

    The browser cannot compute it: ``crypto.subtle`` only exists in a secure
    context, so a plain-HTTP host — the usual way to reach a self-hosted stack
    from another machine — could never compare anything and would have to hide
    every highlight instead.
    """
    import hashlib

    from nlp_pillars.schemas import PodcastScript

    text = "[HOST]: The paper reports an average of 61.7."
    script = PodcastScript(
        id=SCRIPT_ID,
        paper_id="arxiv:2403.05525",
        pillar_id="neural-architectures-language",
        title="AML",
        script=text,
        word_count=8,
    )
    monkeypatch.setattr(
        "webui.routers.api.podcast.get_podcast_script_by_id",
        lambda sid: script,
    )
    response = api_client.get(f"/api/podcast/{SCRIPT_ID}")
    assert response.status_code == 200
    body = response.json()["script"]
    assert body["script_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_view_hash_is_the_hash_the_engine_stored():
    """Two hash functions that disagree would make every check look stale.

    The engine writes ``podcast_fact_checks.script_sha256``; the view compares
    the script's hash against it. They have to be the same function.
    """
    from nlp_pillars.factcheck.engine import sha256_text
    from webui.services.podcast_factcheck_service import script_sha256

    assert script_sha256("the same text") == sha256_text("the same text")
    assert script_sha256("") == sha256_text("")


# --- dismiss with a note ----------------------------------------------------

CLAIM_ID = "c1a1a1a1-1111-4111-8111-111111111111"
DISMISS_URL = f"/api/podcast/{SCRIPT_ID}/factcheck/claims/{CLAIM_ID}/dismissal"


def _stored_claim(**extra):
    row = {
        "id": CLAIM_ID,
        "line_index": 138,
        "span_start": 0,
        "span_end": 10,
        "span_text": "Stage 1",
        "claim": "Stage 1 plus 3 gives 61.7",
        "scope": "paper",
        "category": "number",
        "verdict": "contradicted",
        "error_type": "number_mismatch",
        "severity": "high",
        "evidence": [],
        "evidence_status": "not_required",
        "judge_trail": [],
        "dismissed_at": None,
        "dismissed_note": None,
    }
    row.update(extra)
    return row


def _wire(monkeypatch, *, owner=SCRIPT_ID, store=None):
    monkeypatch.setattr(
        "webui.routers.api.podcast.get_claim_script_id", lambda claim_id: owner,
    )
    calls = []

    def fake_store(claim_id, dismissed, note):
        calls.append((claim_id, dismissed, note))
        if store is not None:
            return store(claim_id, dismissed, note)
        return _stored_claim(
            dismissed_at="2026-10-30T10:00:00+00:00" if dismissed else None,
            dismissed_note=note if dismissed else None,
        )

    monkeypatch.setattr("webui.routers.api.podcast.set_claim_dismissal", fake_store)
    return calls


def test_dismiss_stores_the_trimmed_note(api_client, monkeypatch):
    calls = _wire(monkeypatch)
    response = api_client.put(DISMISS_URL, json={"dismissed": True, "note": "  paper says so  "})
    assert response.status_code == 200
    body = response.json()
    assert body["dismissed_note"] == "paper says so"
    assert body["dismissed_at"]
    # The flag itself is untouched: a dismissal is an opinion, not an edit.
    assert body["verdict"] == "contradicted"
    assert body["severity"] == "high"
    assert calls == [(CLAIM_ID, True, "paper says so")]


def test_dismiss_without_a_note_is_refused_before_the_database(api_client, monkeypatch):
    calls = _wire(monkeypatch)
    for payload in ({"dismissed": True}, {"dismissed": True, "note": "   "}):
        response = api_client.put(DISMISS_URL, json=payload)
        assert response.status_code == 422
    assert calls == []


def test_restore_needs_no_note(api_client, monkeypatch):
    calls = _wire(monkeypatch)
    response = api_client.put(DISMISS_URL, json={"dismissed": False})
    assert response.status_code == 200
    assert response.json()["dismissed_at"] is None
    assert calls == [(CLAIM_ID, False, "")]


def test_a_claim_of_another_script_is_404(api_client, monkeypatch):
    calls = _wire(monkeypatch, owner="99999999-9999-4999-8999-999999999999")
    response = api_client.put(DISMISS_URL, json={"dismissed": True, "note": "x"})
    assert response.status_code == 404
    assert calls == []


def test_an_unknown_or_malformed_claim_is_404(api_client, monkeypatch):
    _wire(monkeypatch, owner=None)
    assert api_client.put(DISMISS_URL, json={"dismissed": True, "note": "x"}).status_code == 404
    bad = f"/api/podcast/{SCRIPT_ID}/factcheck/claims/not-a-uuid/dismissal"
    assert api_client.put(bad, json={"dismissed": True, "note": "x"}).status_code == 404


def test_a_missing_migration_is_503_and_names_it(api_client, monkeypatch):
    from nlp_pillars.db import ClaimDismissalError

    def boom(*args):
        raise ClaimDismissalError("Dismissal is not set up. Apply docs/migrations/033_claim_dismissals.sql by hand.")

    _wire(monkeypatch, store=boom)
    response = api_client.put(DISMISS_URL, json={"dismissed": True, "note": "x"})
    assert response.status_code == 503
    assert "033_claim_dismissals.sql" in response.json()["detail"]


def test_a_note_over_the_limit_is_422(api_client, monkeypatch):
    _wire(monkeypatch)
    response = api_client.put(DISMISS_URL, json={"dismissed": True, "note": "x" * 1001})
    assert response.status_code == 422
