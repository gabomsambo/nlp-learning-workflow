"""Tests for the podcast routes' honesty contracts after background migration.

Script generation is now a 202 + pipeline run. Options validation and paper
lookup still happen before dispatch. Insufficient material, save failure and
partial material live on the job / ``result`` payload — covered in
``tests/test_podcast_script_runs.py`` and the service unit tests there.

Basename is unique across tests/ because there are no __init__.py files anywhere
under it.
"""

import re
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

    from webui.services.spend_guard import require_spend_room
    app = FastAPI()
    app.state.scheduler = MagicMock()
    app.state.cancel_events = {}
    app.include_router(podcast_api.router)
    # The spend guard has its own tests (test_spend_guard.py); here it would only
    # try to read the database.
    app.dependency_overrides[require_spend_room] = lambda: None
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


def test_audio_route_serves_current_and_legacy_files(api_client, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from webui.routers.api import podcast as podcast_api

    current = tmp_path / "podcasts"
    legacy = tmp_path / "legacy-podcast-audio"
    current.mkdir()
    legacy.mkdir()
    (current / "new.mp3").write_bytes(b"new recording")
    (legacy / "old.mp3").write_bytes(b"retained recording")
    monkeypatch.setattr(
        podcast_api,
        "get_settings",
        lambda: SimpleNamespace(
            podcast_audio_dir=str(current),
            podcast_audio_legacy_dir=str(legacy),
        ),
    )

    assert api_client.get("/api/podcast/audio/new.mp3").content == b"new recording"
    assert api_client.get("/api/podcast/audio/old.mp3").content == b"retained recording"
    assert api_client.get("/api/podcast/audio/missing.mp3").status_code == 404


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


def test_generate_passes_voice_and_explicit_audio_choice_to_the_script_run(api_client, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "webui.routers.api.podcast.podcast_audio_service.validate_voice_for_generation",
        lambda path: seen.setdefault("validated_voice", path),
    )
    monkeypatch.setattr(
        "webui.routers.api.podcast.set_last_podcast_voice_path",
        lambda path: seen.setdefault("remembered_voice", path),
    )

    def dispatch(*args, **kwargs):
        seen["dispatch"] = kwargs
        return "run-voice"

    monkeypatch.setattr(
        "webui.routers.api.podcast.run_service.dispatch_run", dispatch
    )
    response = api_client.post(
        "/api/podcast/generate",
        json={
            "paper_id": "p1",
            "pillar_id": "ignored",
            "voice_path": "Podcasts/voices/host.wav",
            "auto_audio": True,
        },
    )

    assert response.status_code == 202
    assert seen["validated_voice"] == "Podcasts/voices/host.wav"
    assert seen["remembered_voice"] == "Podcasts/voices/host.wav"
    assert seen["dispatch"]["voice_path"] == "Podcasts/voices/host.wav"
    assert seen["dispatch"]["auto_audio"] is True
    assert seen["dispatch"]["initial_result"] == {
        "voice_path": "Podcasts/voices/host.wav",
        "auto_audio": True,
    }


def test_auto_audio_requires_a_selected_usable_voice(api_client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "webui.routers.api.podcast.run_service.dispatch_run",
        lambda *a, **k: calls.append(1) or "run-1",
    )

    response = api_client.post(
        "/api/podcast/generate",
        json={"paper_id": "p1", "pillar_id": "ignored", "auto_audio": True},
    )

    assert response.status_code == 400
    assert "Choose a usable voice" in response.json()["detail"]
    assert calls == []


def test_unusable_voice_is_rejected_before_script_dispatch(api_client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "webui.routers.api.podcast.podcast_audio_service.validate_voice_for_generation",
        MagicMock(side_effect=ValueError("voice is too long")),
    )
    monkeypatch.setattr(
        "webui.routers.api.podcast.run_service.dispatch_run",
        lambda *a, **k: calls.append(1) or "run-1",
    )

    response = api_client.post(
        "/api/podcast/generate",
        json={"paper_id": "p1", "pillar_id": "ignored", "voice_path": "long.wav"},
    )

    assert response.status_code == 400
    assert "too long" in response.json()["detail"]
    assert calls == []


def test_voice_list_includes_the_server_remembered_default(api_client, monkeypatch):
    monkeypatch.setattr(
        "webui.routers.api.podcast.podcast_audio_service.list_voices_payload",
        lambda: {"voices": [{"relative_path": "host.wav", "usability": "usable"}]},
    )
    monkeypatch.setattr(
        "webui.routers.api.podcast.get_last_podcast_voice_path",
        lambda: "host.wav",
    )

    response = api_client.get("/api/podcast/tts/voices")

    assert response.status_code == 200
    assert response.json()["last_voice_path"] == "host.wav"


def test_manual_audio_is_queued_while_indextts_is_offline(api_client, monkeypatch):
    from types import SimpleNamespace
    import hashlib

    script_text = "[HOST] Saved episode text."
    monkeypatch.setattr(
        "webui.routers.api.podcast.get_podcast_script_by_id",
        lambda _script_id: SimpleNamespace(
            id="script-1", pillar_id="models-architectures", script=script_text,
        ),
    )
    monkeypatch.setattr(
        "webui.routers.api.podcast.podcast_audio_service.validate_voice_for_generation",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "webui.routers.api.podcast.podcast_audio_service.tts_status_payload",
        lambda: (_ for _ in ()).throw(AssertionError("route must not gate on TTS")),
    )
    seen = {}

    def queue(**kwargs):
        seen.update(kwargs)
        return "audio-run", True

    monkeypatch.setattr(
        "webui.routers.api.podcast.run_service.queue_podcast_audio_run", queue
    )
    response = api_client.post(
        "/api/podcast/tts/generate",
        json={"script_id": "script-1", "voice_path": "nested/voice.wav"},
    )

    assert response.status_code == 202
    assert response.json()["run_id"] == "audio-run"
    assert seen["script_sha256"] == hashlib.sha256(script_text.encode()).hexdigest()
    assert seen["consent"] is True


def test_manual_audio_queue_names_migration_037_on_database_failure(api_client, monkeypatch):
    from types import SimpleNamespace
    from webui.services.run_service import PodcastAudioQueueError

    monkeypatch.setattr(
        "webui.routers.api.podcast.get_podcast_script_by_id",
        lambda _script_id: SimpleNamespace(
            id="script-1", pillar_id="models-architectures", script="text",
        ),
    )
    monkeypatch.setattr(
        "webui.routers.api.podcast.podcast_audio_service.validate_voice_for_generation",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "webui.routers.api.podcast.run_service.queue_podcast_audio_run",
        MagicMock(side_effect=PodcastAudioQueueError(
            "Apply docs/migrations/037_podcast_audio_queue.sql"
        )),
    )
    response = api_client.post(
        "/api/podcast/tts/generate",
        json={"script_id": "script-1", "voice_path": "nested/voice.wav"},
    )
    assert response.status_code == 503
    assert "037_podcast_audio_queue.sql" in response.json()["detail"]


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
    assert "innerHTML" not in body
    assert "#f5f5f5" not in body


def test_a_paper_deep_link_opens_the_studio_with_that_paper_selected(page_client, monkeypatch):
    from datetime import datetime, timezone

    from nlp_pillars.schemas import AudioMetadata, PaperRef, Pillar, PodcastScript
    from webui.routers import podcast as podcast_page

    pillar = Pillar(
        id="neural-architectures-language",
        name="Neural architectures",
        goal="Learn how language models are built.",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    paper = PaperRef(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        authors=["Vaswani"],
        pillar_id="neural-architectures-language",
    )
    script = PodcastScript(
        id="11111111-1111-1111-1111-111111111111",
        paper_id="arxiv:1706.03762",
        pillar_id="neural-architectures-language",
        title="Attention, spoken",
        script="[HOST]: Hello there",
        word_count=12,
        created_at=datetime(2026, 10, 23, tzinfo=timezone.utc),
        audio_metadata=AudioMetadata(file_name="attention.mp3", duration_seconds=125),
    )
    monkeypatch.setattr(podcast_page, "get_pillars_or_empty", lambda **k: [pillar])
    monkeypatch.setattr(podcast_page, "get_all_papers", lambda limit=100: [paper])
    monkeypatch.setattr(podcast_page, "get_podcast_scripts", lambda limit=20: [script])

    library = page_client.get("/podcast/").text
    assert 'href="/papers/arxiv%3A1706.03762"' in library
    assert "Attention, spoken" in library
    assert "2:05" in library
    assert "[HOST]:" not in library
    assert "No podcast scripts generated yet" not in library

    studio = page_client.get("/podcast/?paper=arxiv:1706.03762").text
    studio_tag = re.search(r'<section id="listen-studio"[^>]*>', studio)
    library_tag = re.search(r'<section id="listen-library"[^>]*>', studio)
    assert studio_tag and "hidden" not in studio_tag.group(0)
    assert library_tag and "hidden" in library_tag.group(0)
    assert 'value="arxiv:1706.03762" data-pillar="neural-architectures-language" selected' in studio
    assert "Generate Podcast Script" in studio
    assert "paid model calls" in studio
    assert "local GPU" in studio
