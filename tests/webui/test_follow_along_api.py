"""The follow-along HTTP contract: read an episode's links, list playgrounds, start a pass.

The pass itself is covered in tests/test_playground_follow.py. Nothing here
reaches a database, the network or a model.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nlp_pillars.db import PipelineRunCreateError
from nlp_pillars.playground import follow, links, store
from webui.routers.api import follow_along
from webui.services.run_service import RunAlreadyActiveError

SCRIPT_ID = "0b1c2d3e-0000-4000-8000-000000000001"
PG_ID = "6a1f6c1e-1111-4111-8111-111111111111"
OWN_PG = "6a1f6c1e-3333-4333-8333-333333333333"
MISSING_029 = links.LinkTableMissing(
    "Follow-along is not set up in this database yet (PGRST205). "
    "Apply docs/migrations/029_playground_links.sql by hand."
)


def _script(**over):
    fields = {"id": SCRIPT_ID, "paper_id": "arxiv:study", "pillar_id": "interview-prep",
              "title": "Deep dive", "script": "[HOST]: Hello."}
    fields.update(over)
    return SimpleNamespace(**fields)


def _pg(pid=PG_ID, paper="arxiv:notebook", steps=1):
    return {"id": pid, "paper_id": paper, "title": "aml.ipynb", "status": "ready",
            "steps": [{"id": f"step-{i}"} for i in range(1, steps + 1)]}


@pytest.fixture
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from webui.services.spend_guard import require_spend_room

    app = FastAPI()
    app.state.scheduler = MagicMock()
    app.state.cancel_events = {}
    app.include_router(follow_along.router)
    # The guard has its own tests (test_spend_guard.py includes this route).
    app.dependency_overrides[require_spend_room] = lambda: None
    return TestClient(app, raise_server_exceptions=False)


def _reads(monkeypatch, *, script=None, own=None, rows=(), playgrounds=None, latest=None):
    monkeypatch.setattr(follow_along.db, "get_podcast_script_by_id",
                        lambda sid: script if script is not None else _script())
    monkeypatch.setattr(store, "get_for_paper", lambda pid, with_notebook=False: own)
    monkeypatch.setattr(links, "links_for_script", lambda sid: list(rows))
    monkeypatch.setattr(store, "get", lambda pid, with_notebook=True: (playgrounds or {}).get(pid))
    monkeypatch.setattr(links, "paper_titles",
                        lambda ids: {i: f"Title of {i}" for i in ids if i})
    monkeypatch.setattr(links, "latest_align_run", lambda sid: latest)


# -- GET ---------------------------------------------------------------------------


def test_get_returns_links_with_their_playground_and_the_episode_paper_playground(client, monkeypatch):
    row = {"id": "link-1", "playground_id": PG_ID, "status": "ready", "error": None,
           "model": "claude-haiku-5-5", "script_sha256": "abc", "updated_at": "2026-10-27",
           "mapping": [{"line_start": 1, "line_end": 2, "cell_index": 3, "step_id": "step-1"}]}
    _reads(monkeypatch, own=_pg(OWN_PG, "arxiv:study"), rows=[row],
           playgrounds={PG_ID: _pg()}, latest={"id": "run-1", "status": "succeeded"})
    res = client.get("/api/follow-along", params={"script_id": SCRIPT_ID})
    assert res.status_code == 200
    body = res.json()
    assert body["own_playground"]["id"] == OWN_PG
    assert body["own_playground"]["paper_title"] == "Title of arxiv:study"
    link = body["links"][0]
    assert link["mapping"] == row["mapping"]
    assert link["playground"] == {"id": PG_ID, "paper_id": "arxiv:notebook",
                                  "paper_title": "Title of arxiv:notebook",
                                  "title": "aml.ipynb", "status": "ready", "steps": 1}
    assert body["latest_run"]["id"] == "run-1"
    assert body["script_sha256"] == follow.sha256_text("[HOST]: Hello.")


def test_get_drops_a_link_whose_playground_is_gone(client, monkeypatch):
    _reads(monkeypatch, rows=[{"id": "link-1", "playground_id": PG_ID, "status": "ready"}])
    body = client.get("/api/follow-along", params={"script_id": SCRIPT_ID}).json()
    assert body["links"] == [] and body["own_playground"] is None


def test_get_unknown_or_malformed_episode_is_404(client, monkeypatch):
    assert client.get("/api/follow-along", params={"script_id": "nope"}).status_code == 404
    monkeypatch.setattr(follow_along.db, "get_podcast_script_by_id", lambda sid: None)
    assert client.get("/api/follow-along", params={"script_id": SCRIPT_ID}).status_code == 404


def test_get_names_migration_029_when_the_table_is_missing(client, monkeypatch):
    _reads(monkeypatch)

    def missing(sid):
        raise MISSING_029

    monkeypatch.setattr(links, "links_for_script", missing)
    res = client.get("/api/follow-along", params={"script_id": SCRIPT_ID})
    assert res.status_code == 503
    assert "docs/migrations/029_playground_links.sql" in res.json()["detail"]


def test_get_keeps_the_links_when_only_the_run_status_is_unreadable(client, monkeypatch):
    _reads(monkeypatch)

    def broken(sid):
        raise store.PlaygroundStoreError("down")

    monkeypatch.setattr(links, "latest_align_run", broken)
    res = client.get("/api/follow-along", params={"script_id": SCRIPT_ID})
    assert res.status_code == 200 and res.json()["latest_run"] is None


def test_list_playgrounds_for_the_picker(client, monkeypatch):
    monkeypatch.setattr(links, "list_playgrounds", lambda: [
        {"id": PG_ID, "paper_id": "arxiv:1", "paper_title": "Study guide", "title": "aml",
         "status": "ready", "updated_at": "x"},
    ])
    res = client.get("/api/follow-along/playgrounds")
    assert res.status_code == 200
    assert res.json() == {"playgrounds": [{"id": PG_ID, "paper_id": "arxiv:1",
                                           "paper_title": "Study guide", "title": "aml",
                                           "status": "ready"}]}


def test_list_playgrounds_says_when_028_is_missing(client, monkeypatch):
    def missing():
        raise store.PlaygroundTableMissing("Apply docs/migrations/028_playgrounds.sql by hand.")

    monkeypatch.setattr(links, "list_playgrounds", missing)
    res = client.get("/api/follow-along/playgrounds")
    assert res.status_code == 503 and "028" in res.json()["detail"]


# -- POST --------------------------------------------------------------------------


def _prepared():
    return follow.Prepared(script=_script(), playground=_pg(), link={"id": "link-1"},
                           lines=[(0, "Hello.")], cells={1: "step-1"}, prompt="...")


def _start(client, script_id=SCRIPT_ID, playground_id=PG_ID):
    return client.post("/api/follow-along", json={"script_id": script_id, "playground_id": playground_id})


def test_start_dispatches_a_playground_align_run_with_both_ids(client, monkeypatch):
    monkeypatch.setattr(follow, "prepare", lambda sid, pid: _prepared())
    monkeypatch.setattr(links, "latest_align_run", lambda sid: None)
    seen = {}

    def dispatch(scheduler, cancel_events, **kwargs):
        seen.update(kwargs)
        return "run-9"

    monkeypatch.setattr(follow_along.run_service, "dispatch_run", dispatch)
    res = _start(client)
    assert res.status_code == 202
    assert res.json()["run_id"] == "run-9" and res.json()["link_id"] == "link-1"
    assert seen["kind"] == "playground_align"
    assert seen["trigger_source"] == "ui_playground"
    assert seen["pillar_id"] == "interview-prep"  # the episode's pillar
    assert seen["script_id"] == SCRIPT_ID and seen["playground_id"] == PG_ID
    assert seen["initial_result"] == {"script_id": SCRIPT_ID, "playground_id": PG_ID}


def test_start_attaches_to_the_pass_already_running(client, monkeypatch):
    monkeypatch.setattr(follow, "prepare", lambda sid, pid: _prepared())
    monkeypatch.setattr(links, "latest_align_run", lambda sid: {
        "id": "run-live", "status": "running", "result": {"script_id": SCRIPT_ID, "playground_id": PG_ID},
    })
    dispatch = MagicMock()
    monkeypatch.setattr(follow_along.run_service, "dispatch_run", dispatch)
    res = _start(client)
    assert res.status_code == 409 and res.json()["run_id"] == "run-live"
    dispatch.assert_not_called()


def test_a_running_pass_for_another_notebook_does_not_block(client, monkeypatch):
    monkeypatch.setattr(follow, "prepare", lambda sid, pid: _prepared())
    monkeypatch.setattr(links, "latest_align_run", lambda sid: {
        "id": "run-other", "status": "running", "result": {"script_id": SCRIPT_ID, "playground_id": OWN_PG},
    })
    monkeypatch.setattr(follow_along.run_service, "dispatch_run", lambda *a, **k: "run-new")
    assert _start(client).status_code == 202


def test_start_race_lost_to_the_unique_index_is_409(client, monkeypatch):
    monkeypatch.setattr(follow, "prepare", lambda sid, pid: _prepared())
    monkeypatch.setattr(links, "latest_align_run", lambda sid: None)

    def raced(*a, **k):
        raise RunAlreadyActiveError("taken")

    monkeypatch.setattr(follow_along.run_service, "dispatch_run", raced)
    assert _start(client).status_code == 409


@pytest.mark.parametrize("error,status,needle", [
    (follow.AlignTargetMissing("No playground with id x."), 404, "No playground"),
    (follow.PlaygroundAlignError("This playground has no study guide yet."), 422, "no study guide"),
    (MISSING_029, 503, "029_playground_links.sql"),
])
def test_start_refuses_before_any_run_with_the_reason(client, monkeypatch, error, status, needle):
    def prepare(sid, pid):
        raise error

    monkeypatch.setattr(follow, "prepare", prepare)
    dispatch = MagicMock()
    monkeypatch.setattr(follow_along.run_service, "dispatch_run", dispatch)
    res = _start(client)
    assert res.status_code == status
    assert needle in res.json()["detail"]
    dispatch.assert_not_called()


def test_start_names_029_when_the_run_kind_is_not_allowed_yet(client, monkeypatch):
    monkeypatch.setattr(follow, "prepare", lambda sid, pid: _prepared())
    monkeypatch.setattr(links, "latest_align_run", lambda sid: None)

    def check_violation(*a, **k):
        raise PipelineRunCreateError("23514 violates check constraint pipeline_runs_kind_check")

    monkeypatch.setattr(follow_along.run_service, "dispatch_run", check_violation)
    res = _start(client)
    assert res.status_code == 503
    assert "docs/migrations/029_playground_links.sql" in res.json()["detail"]


def test_start_with_malformed_ids_is_404_without_reading_anything(client, monkeypatch):
    prepare = MagicMock()
    monkeypatch.setattr(follow, "prepare", prepare)
    assert _start(client, script_id="x").status_code == 404
    assert _start(client, playground_id="y").status_code == 404
    prepare.assert_not_called()


def test_the_app_registers_the_router():
    from fastapi.testclient import TestClient

    from webui.app import create_app

    res = TestClient(create_app()).get("/api/follow-along", params={"script_id": "not-a-uuid"})
    # The route's own 404, not the router's "Not Found" for a missing path.
    assert res.status_code == 404 and res.json()["detail"] == "Episode not found"
