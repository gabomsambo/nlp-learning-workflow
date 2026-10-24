"""Hermetic route tests for pillar and cross-pillar hybrid search."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nlp_pillars.schemas import PaperRef
from webui.routers.api import pillars


@pytest.fixture
def client():
    """Mount only the API router under test."""
    app = FastAPI()
    app.include_router(pillars.router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def search_stubs(monkeypatch):
    """Stub vector search and its single metadata lookup."""
    search = Mock(
        return_value=[
            {
                "paper_id": "paper-1",
                "score": 0.75,
                "snippet": "the matched chunk text",
                "chunk_id": "chunk-1",
                "chunk_index": 4,
            },
            {
                "paper_id": "gone-from-postgres",
                "score": 0.5,
                "snippet": "stale projection",
                "chunk_id": "chunk-old",
                "chunk_index": 0,
            },
        ]
    )
    lookup = Mock(
        return_value=[
            PaperRef(
                id="paper-1",
                title="Hybrid Retrieval",
                authors=["Ada"],
                year=2026,
                venue="ACL",
            )
        ]
    )
    monkeypatch.setattr(pillars, "get_pillar_by_id", lambda _pillar_id: object())
    monkeypatch.setattr(pillars.vectors, "search_similar", search)
    monkeypatch.setattr(pillars, "get_papers_by_ids", lookup)
    return search, lookup


def test_pillar_scope_batches_metadata_and_keeps_response_shape(
    client, search_stubs
):
    """Default scope filters by pillar and returns additive snippet fields."""
    search, lookup = search_stubs

    response = client.get(
        "/api/pillars/models-architectures/search",
        params={"q": "RWKV state space", "top_k": 7},
    )

    assert response.status_code == 200
    search.assert_called_once_with(
        pillar_id="models-architectures",
        query_text="RWKV state space",
        top_k=7,
    )
    lookup.assert_called_once_with(["paper-1", "gone-from-postgres"])
    assert response.json() == [
        {
            "paper_id": "paper-1",
            "title": "Hybrid Retrieval",
            "authors": ["Ada"],
            "year": 2026,
            "venue": "ACL",
            "score": 0.75,
            "cosine_similarity": 0.75,
            "keyword_overlap": 0.0,
            "source": "vector_search",
            "snippet": "the matched chunk text",
            "matched_chunk": {"id": "chunk-1", "index": 4},
        }
    ]


def test_all_scope_omits_the_pillar_filter(client, search_stubs):
    """The all scope asks the vector layer not to build a pillar filter."""
    search, _lookup = search_stubs

    response = client.get(
        "/api/pillars/models-architectures/search",
        params={"q": "LoRA", "scope": "all"},
    )

    assert response.status_code == 200
    assert search.call_args.kwargs["pillar_id"] is None


def test_scope_is_validated_by_fastapi(client, search_stubs):
    """Unknown scope values are rejected at the API boundary."""
    response = client.get(
        "/api/pillars/models-architectures/search",
        params={"q": "LoRA", "scope": "everywhere"},
    )
    assert response.status_code == 422


def test_missing_pillar_still_returns_404(client, monkeypatch):
    """The path pillar remains a real resource even for all-library search."""
    monkeypatch.setattr(pillars, "get_pillar_by_id", lambda _pillar_id: None)
    response = client.get(
        "/api/pillars/missing/search", params={"q": "transformers"}
    )
    assert response.status_code == 404
