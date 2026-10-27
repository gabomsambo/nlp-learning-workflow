"""The Playground HTTP contract: read, import, start the explain pass.

The work itself is covered in tests/test_playground.py. Nothing here reaches a
database, the network or a model.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nlp_pillars.db import PipelineRunCreateError
from nlp_pillars.playground import store
from nlp_pillars.playground import service as pg_service
from nlp_pillars.playground.notebook import parse_notebook
from webui.services.run_service import RunAlreadyActiveError

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/playground/toy_regression.ipynb"
PG_ID = "6a1f6c1e-1111-4111-8111-111111111111"
MISSING = store.PlaygroundTableMissing(
    "Playgrounds are not set up in this database yet (PGRST205). "
    "Apply docs/migrations/028_playgrounds.sql by hand."
)


def _row(**over):
    row = {
        "id": PG_ID, "paper_id": "arxiv:1", "title": "Toy", "source_kind": "notebook",
        "notebook_name": "toy.ipynb", "status": "imported", "model": None, "error": None,
        "steps": [], "datasets": [],
        "notebook": parse_notebook(FIXTURE.read_bytes(), max_bytes=10**8),
    }
    row.update(over)
    return row


@pytest.fixture
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from webui.routers.api import playgrounds
    from webui.services.spend_guard import require_spend_room

    app = FastAPI()
    app.state.scheduler = MagicMock()
    app.state.cancel_events = {}
    app.include_router(playgrounds.router)
    # The guard has its own tests (test_spend_guard.py, which includes this route).
    app.dependency_overrides[require_spend_room] = lambda: None
    return TestClient(app, raise_server_exceptions=False)


# -- GET ---------------------------------------------------------------------------


def test_get_without_a_playground_is_null_not_an_error(client, monkeypatch):
    monkeypatch.setattr(store, "get_for_paper", lambda pid, with_notebook=False: None)
    res = client.get("/api/playgrounds", params={"paper_id": "arxiv:1"})
    assert res.status_code == 200
    assert res.json() == {"playground": None, "latest_run": None}


def test_get_returns_the_cell_view_and_not_the_raw_notebook(client, monkeypatch):
    seen = {}

    def get_for_paper(pid, with_notebook=False):
        seen["pid"] = pid
        return _row()

    monkeypatch.setattr(store, "get_for_paper", get_for_paper)
    monkeypatch.setattr(store, "latest_explain_run", lambda pid: {"id": "run-1", "status": "running"})
    res = client.get("/api/playgrounds", params={"paper_id": "arxiv:2403.05525/v2"})
    assert res.status_code == 200
    body = res.json()
    assert seen["pid"] == "arxiv:2403.05525/v2"
    pg = body["playground"]
    assert "notebook" not in pg
    assert [c["index"] for c in pg["cells"]] == list(range(8))
    html = pg["cells"][2]["outputs"][0]
    assert html["kind"] == "html"
    assert "alert" not in json.dumps(html) and "onclick" not in json.dumps(html)
    assert body["latest_run"] == {"id": "run-1", "status": "running"}


def test_get_names_the_migration_when_the_table_is_missing(client, monkeypatch):
    def missing(*a, **k):
        raise MISSING

    monkeypatch.setattr(store, "get_for_paper", missing)
    res = client.get("/api/playgrounds", params={"paper_id": "arxiv:1"})
    assert res.status_code == 503
    assert "docs/migrations/028_playgrounds.sql" in res.json()["detail"]


def test_get_keeps_the_guide_when_only_the_run_status_is_unreadable(client, monkeypatch):
    monkeypatch.setattr(store, "get_for_paper", lambda pid, with_notebook=False: _row())

    def broken(pid):
        raise store.PlaygroundStoreError("down")

    monkeypatch.setattr(store, "latest_explain_run", broken)
    res = client.get("/api/playgrounds", params={"paper_id": "arxiv:1"})
    assert res.status_code == 200
    assert res.json()["latest_run"] is None


# -- import --------------------------------------------------------------------------


def _upload(client, data=None, name="toy.ipynb", paper="arxiv:1"):
    return client.post(
        "/api/playgrounds/import",
        params={"paper_id": paper},
        files={"file": (name, data if data is not None else FIXTURE.read_bytes(), "application/json")},
    )


def test_import_answers_with_the_dataset_outcome(client, monkeypatch):
    seen = {}

    def fake_import(paper_id, data, filename):
        seen.update(paper_id=paper_id, size=len(data), filename=filename)
        return {"id": PG_ID, "title": "Toy", "datasets": [
            {"url": "https://example.org/a.csv", "path": "files/a", "bytes": 3, "sha256": "0" * 64},
            {"url": "https://example.org/b.csv", "error": "HTTP 404 from example.org."},
        ]}

    monkeypatch.setattr(pg_service, "import_notebook", fake_import)
    res = _upload(client)
    assert res.status_code == 200
    body = res.json()
    assert body["playground_id"] == PG_ID
    assert "1 of 2 datasets saved" in body["message"]
    assert seen == {"paper_id": "arxiv:1", "size": FIXTURE.stat().st_size, "filename": "toy.ipynb"}


def test_import_rejects_a_non_notebook_with_400(client):
    res = _upload(client, b"just text", name="notes.ipynb")
    assert res.status_code == 400
    assert "not JSON" in res.json()["detail"]


def test_import_rejects_the_wrong_extension_with_400(client):
    res = _upload(client, name="toy.txt")
    assert res.status_code == 400
    assert ".ipynb" in res.json()["detail"]


def test_import_rejects_an_oversized_file_before_parsing(client, monkeypatch):
    from nlp_pillars import config

    settings = config.get_settings()
    monkeypatch.setattr(settings, "playground_notebook_max_bytes", 50)
    called = MagicMock()
    monkeypatch.setattr(pg_service, "import_notebook", called)
    res = _upload(client)
    assert res.status_code == 413
    assert "PLAYGROUND_NOTEBOOK_MAX_BYTES" in res.json()["detail"]
    called.assert_not_called()


def test_import_for_an_unknown_paper_is_404(client, monkeypatch):
    def fake_import(*a):
        raise pg_service.PaperNotFound("Paper not found: arxiv:404")

    monkeypatch.setattr(pg_service, "import_notebook", fake_import)
    assert _upload(client, paper="arxiv:404").status_code == 404


def test_import_names_the_migration_when_the_table_is_missing(client, monkeypatch):
    def fake_import(*a):
        raise MISSING

    monkeypatch.setattr(pg_service, "import_notebook", fake_import)
    res = _upload(client)
    assert res.status_code == 503
    assert "028_playgrounds.sql" in res.json()["detail"]


# -- explain ----------------------------------------------------------------------------


def _explain(client, pid=PG_ID):
    return client.post(f"/api/playgrounds/{pid}/explain")


@pytest.fixture
def found(monkeypatch):
    monkeypatch.setattr(store, "get", lambda pid, with_notebook=True: _row())
    monkeypatch.setattr(store, "latest_explain_run", lambda pid: None)
    monkeypatch.setattr(store, "paper_row", lambda pid: {"id": pid, "pillar_id": "neural-architectures-language"})


def test_explain_starts_a_run_with_the_playground_id(client, found, monkeypatch):
    seen = {}

    def dispatch(scheduler, events, **kwargs):
        seen.update(kwargs)
        return "run-new"

    monkeypatch.setattr("webui.services.run_service.dispatch_run", dispatch)
    res = _explain(client)
    assert res.status_code == 202
    assert res.json()["run_id"] == "run-new"
    assert seen["kind"] == "playground_explain"
    assert seen["trigger_source"] == "ui_playground"
    assert seen["pillar_id"] == "neural-architectures-language"
    assert seen["playground_id"] == PG_ID
    assert seen["initial_result"] == {"playground_id": PG_ID, "paper_id": "arxiv:1"}


def test_explain_attaches_to_a_run_already_going(client, found, monkeypatch):
    monkeypatch.setattr(store, "latest_explain_run", lambda pid: {"id": "run-old", "status": "running"})
    dispatch = MagicMock()
    monkeypatch.setattr("webui.services.run_service.dispatch_run", dispatch)
    res = _explain(client)
    assert res.status_code == 409
    assert res.json()["run_id"] == "run-old"
    dispatch.assert_not_called()


def test_explain_race_lost_on_the_index_is_a_409(client, found, monkeypatch):
    def dispatch(*a, **k):
        raise RunAlreadyActiveError("taken")

    monkeypatch.setattr("webui.services.run_service.dispatch_run", dispatch)
    assert _explain(client).status_code == 409


def test_explain_names_the_migration_when_the_kind_is_not_allowed(client, found, monkeypatch):
    def dispatch(*a, **k):
        raise PipelineRunCreateError("new row violates check constraint pipeline_runs_kind_check (23514)")

    monkeypatch.setattr("webui.services.run_service.dispatch_run", dispatch)
    res = _explain(client)
    assert res.status_code == 503
    assert "docs/migrations/028_playgrounds.sql" in res.json()["detail"]


def test_explain_for_an_unknown_or_malformed_id_is_404(client, monkeypatch):
    monkeypatch.setattr(store, "get", lambda pid, with_notebook=True: None)
    assert _explain(client).status_code == 404
    assert _explain(client, "not-a-uuid").status_code == 404


def test_explain_names_the_migration_when_the_table_is_missing(client, monkeypatch):
    def missing(*a, **k):
        raise MISSING

    monkeypatch.setattr(store, "get", missing)
    res = _explain(client)
    assert res.status_code == 503
    assert "028_playgrounds.sql" in res.json()["detail"]


# -- the page --------------------------------------------------------------------------------


def test_paper_page_has_the_playground_tab_and_its_controls():
    html = (ROOT / "webui/templates/paper.html").read_text()
    assert '<a href="#playground" data-panel="playground">Playground</a>' in html
    for needle in ('id="playground"', 'id="playground-explain-btn"', 'id="playground-hide-explanations"',
                   'id="playground-import"', 'id="playground-file"', "/static/playground.js",
                   "/static/run-progress.js", "panel_id='playground-run-card'"):
        assert needle in html
    # No Run affordance yet: running code is a later change.
    assert "playground-run-btn" not in html


def test_playground_js_never_uses_innerhtml():
    js = (ROOT / "webui/static/playground.js").read_text()
    code = "\n".join(line for line in js.splitlines() if not line.strip().startswith("*"))
    assert "innerHTML" not in code
    assert "insertAdjacentHTML" not in code
    assert "outerHTML" not in code


# -- explain the difference --------------------------------------------------------

from nlp_pillars.playground import difference  # noqa: E402

DIFF_BODY = {
    "before_code": "print(df.shape)",
    "after_code": "print(df.head(10).shape)",
    "before_output": "(1000, 4)",
    "after_output": "(10, 4)",
    "explanation": "Shows the table's size.",
    "expect": "Only ten rows are counted.",
}


def test_explain_difference_returns_the_sentences_and_stores_nothing(client, monkeypatch):
    seen = {}

    def fake(**fields):
        seen.update(fields)
        return difference.Difference(explanation="head(10) keeps ten rows, so the shape is (10, 4).",
                                     model="claude-haiku-5-5", cost_usd=0.0004)

    monkeypatch.setattr(difference, "explain_difference", fake)
    for name in ("get", "get_for_paper", "save_steps", "update"):
        if hasattr(store, name):
            monkeypatch.setattr(store, name, MagicMock(side_effect=AssertionError("no database")))
    res = client.post(f"/api/playgrounds/{PG_ID}/explain-difference", json=DIFF_BODY)
    assert res.status_code == 200
    assert res.json() == {
        "explanation": "head(10) keeps ten rows, so the shape is (10, 4).",
        "model": "claude-haiku-5-5",
        "cost_usd": 0.0004,
    }
    assert seen == DIFF_BODY


def test_explain_difference_with_a_mocked_model_sends_the_outputs(client, monkeypatch):
    from nlp_pillars import llm_client

    calls = []

    def fake_structured(route, system, user, out_model, **kwargs):
        calls.append({"route": route, "user": user, "out_model": out_model})
        return llm_client.StructuredCall(
            stage=route.stage, provider="anthropic", model="claude-haiku-5-5",
            parsed=out_model(explanation="The shape went from (1000, 4) to (10, 4)."),
            input_tokens=300, output_tokens=30, cost_usd=0.000045,
        )

    monkeypatch.setattr(difference, "structured", fake_structured)
    res = client.post(f"/api/playgrounds/{PG_ID}/explain-difference", json=DIFF_BODY)
    assert res.status_code == 200
    assert res.json()["explanation"] == "The shape went from (1000, 4) to (10, 4)."
    assert len(calls) == 1
    assert calls[0]["route"].stage == "playground_diff"
    for part in ("(1000, 4)", "(10, 4)", "print(df.shape)", "head(10)", "Shows the table's size."):
        assert part in calls[0]["user"]


def test_explain_difference_maps_failures(client, monkeypatch):
    def unavailable(**_):
        raise difference.DifferenceUnavailable("ANTHROPIC_API_KEY environment variable is required")

    monkeypatch.setattr(difference, "explain_difference", unavailable)
    res = client.post(f"/api/playgrounds/{PG_ID}/explain-difference", json=DIFF_BODY)
    assert res.status_code == 503
    assert "ANTHROPIC_API_KEY" in res.json()["detail"]

    def failed(**_):
        raise difference.DifferenceError("The model could not explain this difference: refusal")

    monkeypatch.setattr(difference, "explain_difference", failed)
    res = client.post(f"/api/playgrounds/{PG_ID}/explain-difference", json=DIFF_BODY)
    assert res.status_code == 502
    assert "could not explain" in res.json()["detail"]


def test_explain_difference_validates_before_calling(client, monkeypatch):
    monkeypatch.setattr(difference, "explain_difference", MagicMock(side_effect=AssertionError("no call")))
    assert client.post("/api/playgrounds/not-a-uuid/explain-difference", json=DIFF_BODY).status_code == 404
    assert client.post(f"/api/playgrounds/{PG_ID}/explain-difference", json={"after_code": "x"}).status_code == 422
    huge = dict(DIFF_BODY, before_output="x" * 200_001)
    assert client.post(f"/api/playgrounds/{PG_ID}/explain-difference", json=huge).status_code == 422


def test_explain_difference_is_behind_the_spend_guard():
    from webui.routers.api import playgrounds
    from webui.services.spend_guard import require_spend_room

    route = next(r for r in playgrounds.router.routes if r.path.endswith("/explain-difference"))
    assert any(d.dependency is require_spend_room for d in route.dependencies)


# -- setup code --------------------------------------------------------------------

NEEDS_NP = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
    {"cell_type": "code", "source": "np.where(x > 1)", "metadata": {}, "outputs": [], "execution_count": 1},
]}


def test_get_carries_the_saved_and_suggested_setup_code(client, monkeypatch):
    monkeypatch.setattr(store, "get_for_paper", lambda pid, with_notebook=False: _row(notebook=NEEDS_NP, setup_code=None))
    monkeypatch.setattr(store, "latest_explain_run", lambda pid: None)
    pg = client.get("/api/playgrounds", params={"paper_id": "arxiv:1"}).json()["playground"]
    assert pg["setup_code"] is None
    assert pg["setup_suggested"] == "import numpy as np\n"
    assert pg["setup_migration"] is None


def test_get_names_migration_030_when_the_column_is_missing(client, monkeypatch):
    monkeypatch.setattr(store, "get_for_paper", lambda pid, with_notebook=False: _row(notebook=NEEDS_NP))
    monkeypatch.setattr(store, "latest_explain_run", lambda pid: None)
    pg = client.get("/api/playgrounds", params={"paper_id": "arxiv:1"}).json()["playground"]
    assert pg["setup_migration"] == "docs/migrations/030_playground_setup_code.sql"
    # The suggestion is still offered, and the tab still runs it.
    assert pg["setup_suggested"] == "import numpy as np\n"


def test_put_setup_saves_it(client, monkeypatch):
    saved = []
    monkeypatch.setattr(store, "get", lambda pid, with_notebook=True: _row(notebook=NEEDS_NP))
    monkeypatch.setattr(store, "save_setup", lambda pid, code: saved.append((pid, code)))
    res = client.put(f"/api/playgrounds/{PG_ID}/setup", json={"setup_code": "import numpy as np\nimport pandas as pd\n"})
    assert res.status_code == 200
    assert saved == [(PG_ID, "import numpy as np\nimport pandas as pd\n")]
    assert res.json()["setup_suggested"] == "import numpy as np\n"
    client.put(f"/api/playgrounds/{PG_ID}/setup", json={"setup_code": None})
    assert saved[-1] == (PG_ID, None)


def test_put_setup_names_migration_030_when_the_column_is_missing(client, monkeypatch):
    monkeypatch.setattr(store, "get", lambda pid, with_notebook=True: _row())

    def missing(pid, code):
        raise store.PlaygroundSetupColumnMissing("Apply docs/migrations/030_playground_setup_code.sql by hand.")

    monkeypatch.setattr(store, "save_setup", missing)
    res = client.put(f"/api/playgrounds/{PG_ID}/setup", json={"setup_code": "x"})
    assert res.status_code == 503
    assert "030_playground_setup_code.sql" in res.json()["detail"]


def test_put_setup_for_an_unknown_or_malformed_id_is_404(client, monkeypatch):
    monkeypatch.setattr(store, "get", lambda pid, with_notebook=True: None)
    assert client.put(f"/api/playgrounds/{PG_ID}/setup", json={"setup_code": "x"}).status_code == 404
    assert client.put("/api/playgrounds/nope/setup", json={"setup_code": "x"}).status_code == 404
