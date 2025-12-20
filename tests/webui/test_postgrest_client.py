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

from webui.services.postgrest_client import PostgrestClient


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

    with pytest.raises(httpx.HTTPError):
        await client.get_quick_stats()


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


