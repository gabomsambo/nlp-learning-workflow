import asyncio
import types
import sys
from pathlib import Path

import httpx
import pytest

# Ensure project root is on sys.path for importing 'webui'
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.services.postgrest_client import PostgrestClient, quote_in_filter


class _Resp:
    def __init__(self, headers=None):
        self.headers = headers or {}


@pytest.mark.asyncio
async def test_get_quick_stats_success(monkeypatch):
    client = PostgrestClient(base_url="http://example" , token=None)

    async def fake_get(path, params=None):
        # Simulate distinct totals via Content-Range header
        if path == "papers":
            return [], _Resp({"Content-Range": "items 0-0/12"})
        if path == "lessons":
            return [], _Resp({"Content-Range": "items 0-0/7"})
        if path == "quiz_cards":
            return [], _Resp({"Content-Range": "items 0-0/34"})
        return [], _Resp({})

    monkeypatch.setattr(client, "_get", fake_get)

    stats = await client.get_quick_stats()
    assert stats.total_papers == 12
    assert stats.total_lessons == 7
    assert stats.total_quiz_cards == 34


@pytest.mark.asyncio
async def test_get_quick_stats_edge_no_header(monkeypatch):
    client = PostgrestClient(base_url="http://example" , token=None)

    async def fake_get(path, params=None):
        return [], _Resp({})

    monkeypatch.setattr(client, "_get", fake_get)
    stats = await client.get_quick_stats()
    assert stats.total_papers == 0
    assert stats.total_lessons == 0
    assert stats.total_quiz_cards == 0


@pytest.mark.asyncio
async def test_get_quick_stats_failure(monkeypatch):
    client = PostgrestClient(base_url="http://example" , token=None)

    async def fake_get(path, params=None):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(client, "_get", fake_get)

    # get_quick_stats() deliberately degrades to 0 per table on a transport
    # error (see commit 4f81ba9's count_table try/except, added specifically
    # so the dashboard renders instead of 500ing while the database is
    # unreachable) -- the same tolerate-per-table-failure contract
    # get_recent_activity() already has. It must not propagate here.
    stats = await client.get_quick_stats()
    assert stats.total_papers == 0
    assert stats.total_lessons == 0
    assert stats.total_quiz_cards == 0


@pytest.mark.asyncio
async def test_list_papers_success(monkeypatch):
    client = PostgrestClient(base_url="http://example", token=None)

    async def fake_get(path, params=None):
        data = [
            {"id": "p1", "title": "Paper 1", "authors": ["A"], "pillar_id": "models-architectures"},
            {"id": "p2", "title": "Paper 2", "authors": ["B"], "pillar_id": "data-training-methodologies"},
        ]
        return data, _Resp({})

    monkeypatch.setattr(client, "_get", fake_get)
    items = await client.list_papers(pillar=None, limit=10)
    assert len(items) == 2
    assert items[0]["id"] == "p1"


@pytest.mark.asyncio
async def test_list_papers_checked_raises_and_list_papers_degrades(monkeypatch):
    client = PostgrestClient(base_url="http://example", token=None)

    async def fake_get(path, params=None):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(client, "_get", fake_get)
    with pytest.raises(httpx.HTTPError):
        await client.list_papers_checked()
    assert await client.list_papers() == []


@pytest.mark.asyncio
async def test_paper_ids_with_rows_degrades_to_empty(monkeypatch):
    client = PostgrestClient(base_url="http://example", token=None)

    async def fake_get(path, params=None):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(client, "_get", fake_get)
    assert await client.paper_ids_with_rows("lessons") == set()


@pytest.mark.asyncio
async def test_scripts_for_paper_returns_rows(monkeypatch):
    client = PostgrestClient(base_url="http://example", token=None)
    seen = {}

    async def fake_get(path, params=None):
        seen["path"] = path
        seen["params"] = params
        return [{"id": "script-1", "title": "Episode"}], _Resp({})

    monkeypatch.setattr(client, "_get", fake_get)
    rows = await client.scripts_for_paper("arxiv:2402.08593")
    assert rows[0]["id"] == "script-1"
    assert seen["path"] == "podcast_scripts"
    assert seen["params"]["paper_id"] == "eq.arxiv:2402.08593"
    assert "audio_metadata" not in seen["params"]["select"]


def test_quote_in_filter_quotes_colons_and_escapes():
    assert quote_in_filter(["arxiv:2402.08593"]) == 'in.("arxiv:2402.08593")'
    assert quote_in_filter(['say "hi"', "a,b", "a\\b"]) == 'in.("say \\"hi\\"","a,b","a\\\\b")'


@pytest.mark.asyncio
async def test_list_lessons_checked_raises_and_list_lessons_degrades(monkeypatch):
    client = PostgrestClient(base_url="http://example", token=None)

    async def fake_get(path, params=None):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(client, "_get", fake_get)
    with pytest.raises(httpx.HTTPError):
        await client.list_lessons_checked()
    assert await client.list_lessons() == []


@pytest.mark.asyncio
async def test_paper_titles_quotes_ids_and_skips_blanks(monkeypatch):
    client = PostgrestClient(base_url="http://example", token=None)
    seen = {}

    async def fake_get(path, params=None):
        seen["path"] = path
        seen["params"] = params
        return [
            {"id": "arxiv:2402.08593", "title": "Graph Feature Preprocessor"},
            {"id": "missing-title", "title": ""},
        ], _Resp({})

    monkeypatch.setattr(client, "_get", fake_get)
    titles = await client.paper_titles(
        ["arxiv:2402.08593", "", "arxiv:2402.08593", "missing-title"]
    )
    assert seen["path"] == "papers"
    assert seen["params"]["id"] == 'in.("arxiv:2402.08593","missing-title")'
    assert seen["params"]["limit"] == 2
    assert titles == {
        "arxiv:2402.08593": "Graph Feature Preprocessor",
        "missing-title": "",
    }
    assert await client.paper_titles([]) == {}


