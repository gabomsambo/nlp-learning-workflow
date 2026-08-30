"""Tests for the podcast routes' honesty contracts after background migration.

Script generation is now a 202 + pipeline run. Options validation and paper
lookup still happen before dispatch. Insufficient material, save failure and
partial material live on the job / ``result`` payload — covered in
``tests/test_podcast_script_runs.py`` and the service unit tests there.

Basename is unique across tests/ because there are no __init__.py files anywhere
under it.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp_pillars.db import PodcastScriptLookupError, PipelineRunCreateError
from nlp_pillars.podcast_options import resolve


@pytest.fixture
def api_client(monkeypatch):
    """A TestClient over the podcast API router with a fake scheduler."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from webui.routers.api import podcast as podcast_api

    class _Table:
        def select(self, *a, **k):
            return self

        def eq(self, *a, **k):
            return self

        def execute(self):
            return {"data": [{"pillar_id": "models-architectures"}], "error": None}

    class _Client:
        def table(self, name):
            return _Table()

    monkeypatch.setattr("nlp_pillars.db.get_client", lambda: _Client())

    app = FastAPI()
    app.state.scheduler = MagicMock()
    app.state.cancel_events = {}
    app.include_router(podcast_api.router)
    return TestClient(app, raise_server_exceptions=False)


def test_generate_answers_202_with_a_run_id(api_client, monkeypatch):
    monkeypatch.setattr(
        "webui.routers.api.podcast.run_service.dispatch_run",
        lambda *a, **k: "run-abc",
    )

    r = api_client.post(
        "/api/podcast/generate",
        json={"paper_id": "p1", "pillar_id": "ignored"},
    )

    assert r.status_code == 202
    assert r.json()["run_id"] == "run-abc"


def test_options_are_resolved_before_dispatch(api_client, monkeypatch):
    seen = {}

    def dispatch(*a, **kwargs):
        seen["options"] = kwargs.get("options")
        return "run-1"

    monkeypatch.setattr(
        "webui.routers.api.podcast.run_service.dispatch_run", dispatch
    )

    res = api_client.post(
        "/api/podcast/generate",
        json={
            "paper_id": "p1",
            "pillar_id": "ignored",
            "options": {"field": "biology", "length": "45"},
        },
    )

    assert res.status_code == 202
    assert seen["options"].choices["field"].preset == "biology"
    assert seen["options"].choices["length"].label == "~45 minutes"


def test_omitting_options_still_dispatches_defaults(api_client, monkeypatch):
    from nlp_pillars.agents.podcast_agent import DEFAULT_OPTIONS

    seen = {}

    def dispatch(*a, **kwargs):
        seen["options"] = kwargs.get("options")
        return "run-1"

    monkeypatch.setattr(
        "webui.routers.api.podcast.run_service.dispatch_run", dispatch
    )

    res = api_client.post(
        "/api/podcast/generate", json={"paper_id": "p1", "pillar_id": "ignored"}
    )

    assert res.status_code == 202
    assert seen["options"] == DEFAULT_OPTIONS


def test_an_unknown_option_is_refused_before_dispatch(api_client, monkeypatch):
    calls = []

    def dispatch(*a, **k):
        calls.append(1)
        return "run-1"

    monkeypatch.setattr(
        "webui.routers.api.podcast.run_service.dispatch_run", dispatch
    )

    res = api_client.post(
        "/api/podcast/generate",
        json={
            "paper_id": "p1",
            "pillar_id": "ignored",
            "options": {"field": "astrology-but-quantum"},
        },
    )

    assert res.status_code == 400
    assert "Unknown Field / domain option" in res.json()["detail"]
    assert "biology" in res.json()["detail"]
    assert calls == []


def test_missing_migration_is_503_naming_the_file(api_client, monkeypatch):
    def boom(*a, **k):
        raise PipelineRunCreateError("CHECK constraint failed")

    monkeypatch.setattr(
        "webui.routers.api.podcast.run_service.dispatch_run", boom
    )

    r = api_client.post(
        "/api/podcast/generate", json={"paper_id": "p1", "pillar_id": "x"}
    )
    assert r.status_code == 503
    assert "016_podcast_script_runs.sql" in r.json()["detail"]


def test_missing_paper_is_404_before_dispatch(api_client, monkeypatch):
    class _Empty:
        def select(self, *a, **k):
            return self

        def eq(self, *a, **k):
            return self

        def execute(self):
            return {"data": [], "error": None}

    class _Client:
        def table(self, name):
            return _Empty()

    monkeypatch.setattr("nlp_pillars.db.get_client", lambda: _Client())
    calls = []
    monkeypatch.setattr(
        "webui.routers.api.podcast.run_service.dispatch_run",
        lambda *a, **k: calls.append(1) or "x",
    )

    r = api_client.post(
        "/api/podcast/generate", json={"paper_id": "missing", "pillar_id": "x"}
    )
    assert r.status_code == 404
    assert calls == []


# -------------------------------------------- C: gone is not the same as broken


@pytest.mark.parametrize("path", ["/api/podcast/{}", "/api/podcast/{}/download"])
def test_a_broken_lookup_is_503_and_a_missing_script_is_404(
    api_client, monkeypatch, path
):
    monkeypatch.setattr(
        "webui.routers.api.podcast.get_podcast_script_by_id", lambda sid: None
    )
    assert api_client.get(path.format("00000000-0000-0000-0000-000000000000")).status_code == 404

    def boom(sid):
        raise PodcastScriptLookupError('invalid input syntax for type uuid: "not-a-uuid"')

    monkeypatch.setattr("webui.routers.api.podcast.get_podcast_script_by_id", boom)
    r = api_client.get(path.format("not-a-uuid"))
    assert r.status_code == 503
    assert "invalid input syntax" in r.json()["detail"]


def test_listing_reports_a_dead_database_rather_than_an_empty_list(
    api_client, monkeypatch
):
    def boom(pillar_id=None, limit=20):
        raise PodcastScriptLookupError("Connection refused")

    monkeypatch.setattr("webui.routers.api.podcast.get_podcast_scripts", boom)

    r = api_client.get("/api/podcast/list")
    assert r.status_code == 503
    assert "Connection refused" in r.json()["detail"]


# ------------------------------------ C, one level up: the page itself


@pytest.fixture
def page_client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.templating import Jinja2Templates
    from fastapi.testclient import TestClient

    from webui.routers import podcast as podcast_page

    monkeypatch.setattr(podcast_page, "get_pillars_or_empty", lambda **k: [])

    app = FastAPI()
    app.include_router(podcast_page.router)
    app.state.templates = Jinja2Templates(directory=str(ROOT / "webui" / "templates"))
    return TestClient(app, raise_server_exceptions=False)


def test_the_page_says_the_database_is_unreadable_rather_than_showing_nothing(
    page_client, monkeypatch
):
    from nlp_pillars.db import PaperLookupError
    from webui.routers import podcast as podcast_page

    def papers_boom(limit=100):
        raise PaperLookupError("Connection refused")

    def scripts_boom(limit=20):
        raise PodcastScriptLookupError("Connection refused")

    monkeypatch.setattr(podcast_page, "get_all_papers", papers_boom)
    monkeypatch.setattr(podcast_page, "get_podcast_scripts", scripts_boom)

    body = page_client.get("/podcast/").text

    assert "Some of this page could not be loaded" in body
    assert "The paper list could not be loaded" in body
    assert "No podcast scripts generated yet" not in body


def test_a_genuinely_empty_account_still_reads_as_empty(page_client, monkeypatch):
    from webui.routers import podcast as podcast_page

    monkeypatch.setattr(podcast_page, "get_all_papers", lambda limit=100: [])
    monkeypatch.setattr(podcast_page, "get_podcast_scripts", lambda limit=20: [])

    body = page_client.get("/podcast/").text

    assert "No podcast scripts generated yet" in body
    assert "Some of this page could not be loaded" not in body


def test_the_page_no_longer_ships_the_fake_progress_interval(page_client, monkeypatch):
    from webui.routers import podcast as podcast_page

    monkeypatch.setattr(podcast_page, "get_all_papers", lambda limit=100: [])
    monkeypatch.setattr(podcast_page, "get_podcast_scripts", lambda limit=20: [])

    body = page_client.get("/podcast/").text
    assert "Simulate progress" not in body
    assert "progressBar.value += 2" not in body
    assert "script-stages" in body
    assert "podcast_script" in body
