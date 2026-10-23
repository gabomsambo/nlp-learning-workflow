"""Shell extras: palette data, shortcut markup, and the mini player."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "webui" / "templates"
STATIC = ROOT / "webui" / "static"


class _URL:
    def __init__(self, path: str) -> None:
        self.path = path


class _Request:
    def __init__(self, path: str) -> None:
        self.url = _URL(path)


def _render(path: str) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    return env.get_template("base.html").render(request=_Request(path))


def test_shell_exposes_palette_help_and_player():
    home = _render("/")
    assert 'id="command-palette"' in home
    assert 'id="palette-input"' in home
    assert 'id="shortcuts-dialog"' in home
    assert 'id="mini-player"' in home
    assert 'id="mini-player-audio"' in home
    assert "/static/shell.js" in home
    assert 'data-open-palette' in home
    assert 'data-open-shortcuts' in home
    assert 'data-open-settings' in home
    assert ">0<" not in home
    script = (STATIC / "shell.js").read_text()
    assert "nlp:player" in script
    assert "NlpShortcuts" in script
    podcast = (TEMPLATES / "podcast.html").read_text()
    assert "NlpPlayer.attach" in podcast
    papers = (TEMPLATES / "papers.html").read_text()
    assert "URLSearchParams" in papers
    assert ".get('paper')" in papers


def test_list_papers_route_returns_library_rows():
    from webui.routers.api import papers as papers_api

    app = FastAPI()
    app.include_router(papers_api.router)
    client = TestClient(app)
    rows = [{
        "id": "1706.03762",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani"],
        "year": 2017,
        "citation_count": 1,
        "pillar_id": "llm-theory-practice",
        "added_at": "2026-01-01T00:00:00+00:00",
    }]
    with patch("webui.routers.api.papers.PostgrestClient") as cls:
        cls.return_value.list_papers = AsyncMock(return_value=rows)
        response = client.get("/api/papers/?limit=50")
    assert response.status_code == 200
    assert response.json()["papers"] == rows
    cls.return_value.list_papers.assert_awaited_once_with(limit=50)


def test_list_papers_route_rejects_a_huge_limit():
    from webui.routers.api import papers as papers_api

    app = FastAPI()
    app.include_router(papers_api.router)
    response = TestClient(app).get("/api/papers/?limit=9999")
    assert response.status_code == 422
