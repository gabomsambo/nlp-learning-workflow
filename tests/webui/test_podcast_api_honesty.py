"""Tests for the podcast routes' three honesty contracts.

Each corresponds to a measured failure documented in the 2026-08-29 podcast
investigation:

A. Generation against a paper whose body could not be read made five model calls
   (~$0.27) and answered a green "Script generated successfully!" for a script
   whose entire factual basis was the title.
B. A failed insert became `500 Failed to save podcast script to database`, with
   the script — already generated, already paid for — discarded.
C. A malformed script id made PostgREST answer `400 invalid input syntax for
   type uuid`, and the route reported it as 404 "Script not found", which is what
   a genuinely absent script looks like.

Basename is unique across tests/ because there are no __init__.py files anywhere
under it.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp_pillars.db import PodcastScriptLookupError, PodcastScriptSaveError
from nlp_pillars.schemas import PodcastScript, SourceMaterial


@pytest.fixture
def api_client(monkeypatch):
    """A TestClient over just the podcast API router.

    Deliberately not the whole app: create_app()'s lifespan starts a scheduler
    and sweeps the database, none of which these routes need.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from webui.routers.api import podcast as podcast_api

    # The route re-reads the paper's real pillar_id before generating.
    class _Resp(dict):
        pass

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
    app.include_router(podcast_api.router)
    return TestClient(app, raise_server_exceptions=False)


def _script(level="full", warnings=()):
    return PodcastScript(
        paper_id="file:2dd76e910fbc",
        pillar_id="models-architectures",
        title="Deep Dive: Test Paper",
        script="[HOST]: Hello there.",
        word_count=3,
        key_points=["a point"],
        source_material=SourceMaterial(level=level, warnings=list(warnings)),
        created_at=datetime(2026, 8, 29),
    )


def _stub_generate(monkeypatch, script=None, raises=None):
    from nlp_pillars.agents.podcast_agent import PodcastAgent

    async def generate(self, paper_id, pillar_id):
        if raises is not None:
            raise raises
        return script

    monkeypatch.setattr(PodcastAgent, "__init__", lambda self: None)
    monkeypatch.setattr(PodcastAgent, "generate", generate)


# ---------------------------------------------------------------- A: refusal


def test_nothing_to_write_from_is_422_with_the_reason(api_client, monkeypatch):
    from nlp_pillars.agents.podcast_agent import InsufficientSourceMaterialError

    reason = (
        'Cannot generate a podcast for "FastTextB": Local file not found: '
        "/app/.cache/uploads/5690.pdf, and the paper has no abstract and no "
        "extracted notes. There is nothing to write from, so no model calls were made."
    )
    _stub_generate(monkeypatch, raises=InsufficientSourceMaterialError(reason))

    r = api_client.post(
        "/api/podcast/generate",
        json={"paper_id": "file:2dd76e910fbc", "pillar_id": "models-architectures"},
    )

    # Not 500: the request is fine, the paper is the problem, and the user can act.
    assert r.status_code == 422
    assert r.json()["detail"] == reason


def test_partial_material_is_reported_alongside_the_script(api_client, monkeypatch):
    caveat = "The full text of this paper was not available (HTTP 404 fetching the PDF)."
    _stub_generate(monkeypatch, script=_script(level="partial", warnings=[caveat]))
    monkeypatch.setattr(
        "webui.routers.api.podcast.add_podcast_script", lambda s: "new-id"
    )

    body = api_client.post(
        "/api/podcast/generate",
        json={"paper_id": "p", "pillar_id": "models-architectures"},
    ).json()

    assert body["saved"] is True
    assert body["source_material_level"] == "partial"
    assert caveat in body["warnings"]


# ----------------------------------------------------- B: the artifact is kept


def test_a_failed_save_still_hands_back_the_script(api_client, monkeypatch):
    """The one error class where failing loudly is not enough.

    The script already exists and already cost ~$0.27, so it comes back in the
    body — 200, not 5xx, because a 5xx sends it into generic error handling and
    the artifact is thrown away, which is the whole bug.
    """
    _stub_generate(monkeypatch, script=_script())

    def boom(script):
        raise PodcastScriptSaveError("connection refused")

    monkeypatch.setattr("webui.routers.api.podcast.add_podcast_script", boom)

    r = api_client.post(
        "/api/podcast/generate",
        json={"paper_id": "p", "pillar_id": "models-architectures"},
    )
    body = r.json()

    assert r.status_code == 200
    assert body["saved"] is False
    assert body["script_id"] is None
    assert body["script"] == "[HOST]: Hello there."
    assert body["key_points"] == ["a point"]
    # The real reason travels with it, and the warning is unmissable.
    assert any("connection refused" in w for w in body["warnings"])
    assert any("NOT saved" in w for w in body["warnings"])


def test_a_successful_save_says_so_and_does_not_duplicate_the_script(
    api_client, monkeypatch
):
    _stub_generate(monkeypatch, script=_script())
    monkeypatch.setattr(
        "webui.routers.api.podcast.add_podcast_script", lambda s: "new-id"
    )

    body = api_client.post(
        "/api/podcast/generate",
        json={"paper_id": "p", "pillar_id": "models-architectures"},
    ).json()

    assert body["saved"] is True
    assert body["script_id"] == "new-id"
    assert body["warnings"] == []
    # The page re-fetches a stored script, so shipping it twice is dead weight.
    assert body["script"] is None


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
    """An unreachable database used to render as a working, empty account."""
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
    # The false reassurance must be gone.
    assert "No podcast scripts generated yet" not in body


def test_a_genuinely_empty_account_still_reads_as_empty(page_client, monkeypatch):
    """Emptiness is not an error, and must not grow a scary banner."""
    from webui.routers import podcast as podcast_page

    monkeypatch.setattr(podcast_page, "get_all_papers", lambda limit=100: [])
    monkeypatch.setattr(podcast_page, "get_podcast_scripts", lambda limit=20: [])

    body = page_client.get("/podcast/").text

    assert "No podcast scripts generated yet" in body
    assert "Some of this page could not be loaded" not in body
