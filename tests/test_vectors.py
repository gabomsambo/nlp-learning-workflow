"""
Tests for the Qdrant read path and the upsert_text projection hook.
The client is mocked. The query embedder is a stub, so nothing downloads a model.
"""

import os
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from nlp_pillars import vectors
from nlp_pillars.embeddings import LOCAL_MODEL_ID
from nlp_pillars.projector import PAPER_ID_FIELD, ProjectionError, collection_name
from nlp_pillars.vectors import (
    COLLECTION_NAME,
    PILLAR_ID_FIELD,
    ensure_collections,
    get_client,
    search_similar,
    set_client,
    upsert_text,
)

MODEL_COLLECTION = collection_name(LOCAL_MODEL_ID, 768)


def make_hit(paper_id, score, chunk_index=0):
    """One ScoredPoint-shaped search hit."""
    hit = Mock()
    hit.payload = {
        "paper_id": paper_id,
        "chunk_id": f"chunk-{paper_id}-{chunk_index}",
        "chunk_index": chunk_index,
        "text": f"Matched text for {paper_id}",
    }
    hit.score = score
    return hit


def query_response(hits):
    """query_points() returns a QueryResponse; its hits live on `.points`."""
    response = Mock()
    response.points = hits
    return response


def unexpected_response(status_code, content=b"boom"):
    """A qdrant-client UnexpectedResponse with the given HTTP status."""
    return UnexpectedResponse(
        status_code=status_code,
        reason_phrase="Bad Request" if status_code < 500 else "Server Error",
        content=content,
        headers=None,
    )


# Test fixtures
@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing.

    spec=QdrantClient is load-bearing, not tidiness. This fixture used to be a
    bare Mock(), which happily answered `client.search(...)` long after
    qdrant-client 1.19 removed that method — so the whole search suite passed
    green against a read path that raised AttributeError on every real query.
    A spec'd mock fails the same way the library does.
    """
    mock_client = Mock(spec=QdrantClient)

    # Mock collections response
    mock_collections = Mock()
    mock_collections.collections = []
    mock_client.get_collections.return_value = mock_collections

    # Mock create collection
    mock_client.create_collection.return_value = None

    # Collection info, with both payload indexes already in place so
    # ensure_index() does not create them unless a test says otherwise.
    collection_info = Mock()
    collection_info.payload_schema = {PILLAR_ID_FIELD: Mock(), PAPER_ID_FIELD: Mock()}
    mock_client.get_collection.return_value = collection_info
    mock_client.create_payload_index.return_value = None

    aliases = Mock()
    aliases.aliases = []
    mock_client.get_aliases.return_value = aliases
    mock_client.update_collection_aliases.return_value = True
    mock_client.scroll.return_value = ([], None)

    # Mock upsert
    mock_client.upsert.return_value = None

    # Mock search
    mock_client.query_points.return_value = query_response([])

    return mock_client


@pytest.fixture(autouse=True)
def setup_clean_state():
    """Clean state for each test."""
    set_client(None)
    yield
    set_client(None)


@pytest.fixture(autouse=True)
def stub_index_embedder():
    """Search must not download a model or call OpenAI.

    The stub is the active provider: same model id and dims the collection
    name is built from, and a fixed query vector the search tests assert.
    """
    provider = Mock()
    provider.model_id = LOCAL_MODEL_ID
    provider.dims = 768
    provider.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    sparse = models.SparseVector(indices=[7, 11], values=[1.0, 1.0])
    with patch("nlp_pillars.embeddings.get_embedding_provider", return_value=provider), \
         patch("nlp_pillars.vectors.bm25_query", return_value=sparse):
        yield provider


@contextmanager
def no_qdrant_client():
    """Force vectors.get_client() to return None, deterministically.

    Two things have to be true and neither is guaranteed by default:

    - the module singleton must be cleared. Other tests in this file call
      set_client(mock) and never reset it, so whether these tests saw a client
      depended on execution order.
    - QDRANT_URL must be unset. get_client() only gives up when the variable is
      missing, and it is set both in CI and in .env — so with a URL present the
      client is constructed (unreachable, but not None) and the "Cannot ..."
      warning these tests assert on never fires.

    Restores the previous singleton afterwards so ordering stays irrelevant.
    """
    previous = vectors._client
    set_client(None)
    try:
        with patch.dict(os.environ, {"QDRANT_URL": ""}, clear=False):
            yield
    finally:
        set_client(previous)



class TestClientBootstrap:
    """Test client initialization and configuration."""
    
    @patch('nlp_pillars.vectors.QdrantClient')
    @patch.dict('os.environ', {'QDRANT_URL': 'http://localhost:6333', 'QDRANT_API_KEY': ''})
    def test_get_client_success_no_api_key(self, mock_qdrant_class):
        """Test successful client creation without API key (local deployment)."""
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        
        result = get_client()
        
        assert result == mock_client
        mock_qdrant_class.assert_called_once_with(url='http://localhost:6333')
    
    @patch('nlp_pillars.vectors.QdrantClient')
    @patch.dict('os.environ', {'QDRANT_URL': 'https://xyz.qdrant.io', 'QDRANT_API_KEY': 'test_key'})
    def test_get_client_success_with_api_key(self, mock_qdrant_class):
        """Test successful client creation with API key."""
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        
        result = get_client()
        
        assert result == mock_client
        mock_qdrant_class.assert_called_once_with(url='https://xyz.qdrant.io', api_key='test_key')
    
    @patch.dict('os.environ', {}, clear=True)
    def test_get_client_missing_url(self):
        """Test client creation failure when QDRANT_URL is missing."""
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            result = get_client()
            
            assert result is None
            mock_logger.warning.assert_called_once_with(
                "QDRANT_URL environment variable not set. Vector operations will be disabled."
            )
    
    @patch('nlp_pillars.vectors.QdrantClient')
    @patch.dict('os.environ', {'QDRANT_URL': 'http://bad-url'})
    def test_get_client_connection_error(self, mock_qdrant_class):
        """Test client creation failure when connection fails."""
        mock_qdrant_class.side_effect = Exception("Connection failed")
        
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            result = get_client()
            
            assert result is None
            mock_logger.error.assert_called_once()


class TestEnsureCollections:
    """Test collection management functionality."""
    
    def test_ensure_collections_no_client(self):
        """Test ensure_collections when client is not available."""
        with no_qdrant_client(), patch('nlp_pillars.vectors.logger') as mock_logger:
            ensure_collections()
            
            # Check that the warning about collections was called (may have other warnings too)
            warning_calls = [call for call in mock_logger.warning.call_args_list]
            assert any("Cannot ensure collections" in str(call) for call in warning_calls)
    
    def test_ensure_collections_exists(self, mock_qdrant_client):
        """Test ensure_collections when collection already exists."""
        set_client(mock_qdrant_client)
        
        # Mock existing collection
        mock_collection = Mock()
        mock_collection.name = MODEL_COLLECTION
        mock_qdrant_client.get_collections.return_value.collections = [mock_collection]
        
        with patch('nlp_pillars.projector.logger') as mock_logger:
            ensure_collections()
            
            mock_qdrant_client.create_collection.assert_not_called()
            mock_logger.info.assert_any_call(
                "Collection '%s' already exists", MODEL_COLLECTION
            )
    
    def test_ensure_collections_creates_new(self, mock_qdrant_client):
        """Test ensure_collections creates new collection with correct configuration."""
        set_client(mock_qdrant_client)
        
        # Mock no existing collections
        mock_qdrant_client.get_collections.return_value.collections = []
        
        with patch('nlp_pillars.projector.logger') as mock_logger:
            ensure_collections()
            
            # Verify create_collection was called with correct parameters
            mock_qdrant_client.create_collection.assert_called_once()
            call_args = mock_qdrant_client.create_collection.call_args
            
            assert call_args[1]['collection_name'] == MODEL_COLLECTION
            vectors_config = call_args[1]['vectors_config']
            assert vectors_config.size == 768
            assert vectors_config.distance == models.Distance.COSINE
            
            mock_logger.info.assert_any_call(
                "Created collection '%s' with vector size %s and cosine distance",
                MODEL_COLLECTION,
                768,
            )
    
    def test_ensure_collections_error(self, mock_qdrant_client):
        """Test ensure_collections handles errors gracefully."""
        set_client(mock_qdrant_client)
        mock_qdrant_client.get_collections.side_effect = Exception("Database error")
        
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            ensure_collections()
            
            mock_logger.error.assert_called_once()


class TestUpsertText:
    """upsert_text projects stored embeddings. It does not call an embedder."""

    def test_upsert_text_no_client(self):
        """A missing local target is explicit, never an ambiguous zero."""
        with no_qdrant_client(), pytest.raises(
            ProjectionError, match="local vector index unreachable"
        ):
            upsert_text(
                "linguistic-cognitive-foundations", "test.123", "test text"
            )

    def test_upsert_text_empty_text(self, mock_qdrant_client):
        """Test upsert_text with empty text."""
        set_client(mock_qdrant_client)

        with patch('nlp_pillars.vectors.logger') as mock_logger:
            result = upsert_text("linguistic-cognitive-foundations", "test.123", "")

            assert result == 0
            mock_logger.warning.assert_called_once_with("Empty text provided for upsert")

    def test_upsert_text_projects_stored_chunks(self, mock_qdrant_client):
        """Non-empty text with a client delegates to the projector."""
        set_client(mock_qdrant_client)
        with patch("nlp_pillars.vectors.project_paper", return_value=4) as project:
            result = upsert_text("models-architectures", "test.456", "stored body")
        assert result == 4
        project.assert_called_once_with("models-architectures", "test.456")

    def test_local_projection_failure_propagates(self, mock_qdrant_client):
        """A local index failure is the stage detail, not a silent zero."""
        set_client(mock_qdrant_client)
        with patch(
            "nlp_pillars.vectors.project_paper",
            side_effect=ProjectionError("local vector index unreachable: suspended"),
        ):
            with pytest.raises(ProjectionError, match="unreachable"):
                upsert_text("linguistic-cognitive-foundations", "test.123", "test text")


class TestSearchSimilar:
    """Test similarity search functionality."""
    
    def test_search_similar_no_client(self):
        """Test search_similar when client is not available."""
        with no_qdrant_client(), patch('nlp_pillars.vectors.logger') as mock_logger:
            result = search_similar("linguistic-cognitive-foundations", "test query")

            assert result == []
            # Check that the warning about search was called (may have other warnings too)
            warning_calls = [call for call in mock_logger.warning.call_args_list]
            assert any("Cannot search similar text" in str(call) for call in warning_calls)
    
    def test_search_similar_empty_query(self, mock_qdrant_client):
        """Test search_similar with empty query."""
        set_client(mock_qdrant_client)
        
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            result = search_similar("linguistic-cognitive-foundations", "")

            assert result == []
            mock_logger.warning.assert_called_once_with("Empty query text provided")
    
    def test_search_similar_success(self, mock_qdrant_client):
        """Test successful similarity search with proper filtering."""
        set_client(mock_qdrant_client)
        
        # Mock search results
        mock_qdrant_client.query_points.return_value = query_response([
            make_hit("paper.123", 0.9),
            make_hit("paper.456", 0.8),
        ])

        result = search_similar("data-training-methodologies", "test query", top_k=5)

        # Verify search was called with correct parameters
        mock_qdrant_client.query_points.assert_called_once()
        search_call = mock_qdrant_client.query_points.call_args

        assert search_call[1]['collection_name'] == COLLECTION_NAME
        assert isinstance(search_call[1]['query'], models.FusionQuery)
        assert search_call[1]['query'].fusion == models.Fusion.RRF
        assert search_call[1]['limit'] == 15  # top_k * 3 for deduplication
        assert search_call[1]['with_payload'] is True

        dense, sparse = search_call[1]['prefetch']
        assert dense.query == [0.1, 0.2, 0.3, 0.4]
        assert dense.using is None
        assert sparse.using == "bm25"
        assert sparse.query.indices == [7, 11]
        for leg in (dense, sparse):
            assert leg.limit == 15
            assert len(leg.filter.must) == 1
            field_condition = leg.filter.must[0]
            assert field_condition.key == "pillar_id"
            assert field_condition.match.value == "data-training-methodologies"
        
        # Check results format
        expected_results = [
            {
                "paper_id": "paper.123", "score": 0.9,
                "snippet": "Matched text for paper.123",
                "chunk_id": "chunk-paper.123-0", "chunk_index": 0,
            },
            {
                "paper_id": "paper.456", "score": 0.8,
                "snippet": "Matched text for paper.456",
                "chunk_id": "chunk-paper.456-0", "chunk_index": 0,
            },
        ]
        assert result == expected_results
    
    def test_search_similar_deduplication(self, mock_qdrant_client):
        """Test search_similar deduplicates by paper_id keeping highest score."""
        set_client(mock_qdrant_client)
        
        # Mock search results with duplicates
        mock_qdrant_client.query_points.return_value = query_response([
            make_hit("paper.123", 0.9),
            make_hit("paper.123", 0.7),  # Same paper, lower score
            make_hit("paper.456", 0.8),
        ])

        result = search_similar("linguistic-cognitive-foundations", "test query", top_k=5)
        
        # Should deduplicate and keep highest score for paper.123
        expected_results = [
            {
                "paper_id": "paper.123", "score": 0.9,
                "snippet": "Matched text for paper.123",
                "chunk_id": "chunk-paper.123-0", "chunk_index": 0,
            },
            {
                "paper_id": "paper.456", "score": 0.8,
                "snippet": "Matched text for paper.456",
                "chunk_id": "chunk-paper.456-0", "chunk_index": 0,
            },
        ]
        assert result == expected_results
    
    def test_search_similar_top_k_limit(self, mock_qdrant_client):
        """Test search_similar respects top_k limit."""
        set_client(mock_qdrant_client)
        
        # Mock many search results
        mock_hits = [make_hit(f"paper.{i}", 0.9 - (i * 0.1)) for i in range(10)]

        mock_qdrant_client.query_points.return_value = query_response(mock_hits)

        result = search_similar("linguistic-cognitive-foundations", "test query", top_k=3)
        
        # Should return only top 3
        assert len(result) == 3
        assert result[0]["paper_id"] == "paper.0"
        assert result[1]["paper_id"] == "paper.1"
        assert result[2]["paper_id"] == "paper.2"
    
    def test_search_similar_embedding_failure(self, mock_qdrant_client):
        """Test search_similar when embedding fails."""
        set_client(mock_qdrant_client)

        with patch('nlp_pillars.vectors._query_vector', side_effect=Exception("Embedding failed")):
            with patch('nlp_pillars.vectors.logger') as mock_logger:
                result = search_similar("linguistic-cognitive-foundations", "test query")

                assert result == []
                mock_logger.error.assert_called_once()


class TestSearchSimilarFailsLoudly:
    """A broken read path must not be reported as 'nothing matched'.

    The original bug: search_similar() called client.search(), removed in
    qdrant-client 1.19, and its blanket `except` turned the AttributeError into
    `[]` on every single query for months. Same precedent as
    pdf_loader.chunk_text() after PR #8 — a call the library rejects is a bug,
    not a miss, so it raises; genuine runtime failures still degrade to [].
    """

    def test_installed_client_has_query_points_and_no_search(self):
        """Canary on the pinned library, so a version bump cannot go unnoticed."""
        assert hasattr(QdrantClient, "query_points")
        assert not hasattr(QdrantClient, "search")

    def test_raises_when_method_is_missing(self, mock_qdrant_client):
        """AttributeError (the 1.19 removal) propagates as RuntimeError."""
        set_client(mock_qdrant_client)
        mock_qdrant_client.query_points.side_effect = AttributeError(
            "'QdrantClient' object has no attribute 'query_points'"
        )

        with pytest.raises(RuntimeError, match="incompatible with the installed"):
            search_similar("linguistic-cognitive-foundations", "test query")

    def test_raises_when_signature_is_rejected(self, mock_qdrant_client):
        """TypeError (a renamed/removed keyword) propagates as RuntimeError."""
        set_client(mock_qdrant_client)
        mock_qdrant_client.query_points.side_effect = TypeError(
            "query_points() got an unexpected keyword argument 'query_vector'"
        )

        with pytest.raises(RuntimeError, match="incompatible with the installed"):
            search_similar("linguistic-cognitive-foundations", "test query")

    def test_raises_when_server_rejects_request(self, mock_qdrant_client):
        """A 4xx — e.g. the missing pillar_id payload index under strict mode."""
        set_client(mock_qdrant_client)
        mock_qdrant_client.query_points.side_effect = unexpected_response(
            400, b'{"status":{"error":"Bad request: Index required but not found for \\"pillar_id\\""}}'
        )

        with pytest.raises(RuntimeError, match="Qdrant rejected the search"):
            search_similar("linguistic-cognitive-foundations", "test query")

    def test_returns_empty_on_server_error(self, mock_qdrant_client):
        """A 5xx is a runtime failure, not a contract mismatch: still degrades."""
        set_client(mock_qdrant_client)
        mock_qdrant_client.query_points.side_effect = unexpected_response(503)

        assert search_similar("linguistic-cognitive-foundations", "test query") == []

    def test_raises_on_unexpected_response_shape(self, mock_qdrant_client):
        """A bare list (the old search() shape) is not silently read as no hits."""
        set_client(mock_qdrant_client)
        mock_qdrant_client.query_points.return_value = [make_hit("paper.123", 0.9)]

        with pytest.raises(RuntimeError, match="unexpected query_points"):
            search_similar("linguistic-cognitive-foundations", "test query")


class TestPayloadIndex:
    """Every read filters on pillar_id, so that key must carry a payload index.

    Qdrant Cloud runs strict mode (`unindexed_filtering_retrieve: false`) and
    answers a filtered query on an unindexed key with 400. The live collection
    had 170 points and an empty payload_schema, so the filter could never work.
    """

    def test_creates_index_when_missing(self, mock_qdrant_client):
        set_client(mock_qdrant_client)

        existing = Mock()
        existing.name = MODEL_COLLECTION
        mock_qdrant_client.get_collections.return_value.collections = [existing]
        mock_qdrant_client.get_collection.return_value.payload_schema = {}

        ensure_collections()

        fields = [
            call.kwargs["field_name"]
            for call in mock_qdrant_client.create_payload_index.call_args_list
        ]
        assert fields == [PILLAR_ID_FIELD, PAPER_ID_FIELD]
        for call in mock_qdrant_client.create_payload_index.call_args_list:
            assert call.kwargs["collection_name"] == MODEL_COLLECTION
            assert call.kwargs["field_schema"] == models.PayloadSchemaType.KEYWORD

    def test_does_not_recreate_existing_index(self, mock_qdrant_client):
        set_client(mock_qdrant_client)

        existing = Mock()
        existing.name = MODEL_COLLECTION
        mock_qdrant_client.get_collections.return_value.collections = [existing]
        # fixture already reports pillar_id and paper_id in payload_schema

        ensure_collections()

        mock_qdrant_client.create_payload_index.assert_not_called()

    def test_index_created_for_new_collection(self, mock_qdrant_client):
        set_client(mock_qdrant_client)
        mock_qdrant_client.get_collections.return_value.collections = []
        mock_qdrant_client.get_collection.return_value.payload_schema = {}

        ensure_collections()

        mock_qdrant_client.create_collection.assert_called_once()
        assert mock_qdrant_client.create_payload_index.call_count == 2

    def test_index_failure_is_not_fatal(self, mock_qdrant_client):
        """Writes do not need the index; search_similar() raises if it is absent."""
        set_client(mock_qdrant_client)

        existing = Mock()
        existing.name = MODEL_COLLECTION
        mock_qdrant_client.get_collections.return_value.collections = [existing]
        mock_qdrant_client.get_collection.side_effect = Exception("index lookup failed")

        ensure_collections()  # must not raise


class TestNamespaceEnforcement:
    """Test that pillar isolation is enforced."""
    
    def test_all_searches_include_pillar_filter(self, mock_qdrant_client):
        """Test that all search operations include pillar_id filter."""
        set_client(mock_qdrant_client)
        
        mock_qdrant_client.query_points.return_value = query_response([])

        # Test different pillars
        pillars = [
            "linguistic-cognitive-foundations",
            "data-training-methodologies",
            "ethics-applications"
        ]
        for pillar in pillars:
            search_similar(pillar, "test query")

            # Get the last search call
            search_call = mock_qdrant_client.query_points.call_args
            for leg in search_call[1]['prefetch']:
                assert len(leg.filter.must) == 1
                field_condition = leg.filter.must[0]
                assert field_condition.key == "pillar_id"
                assert field_condition.match.value == pillar

    def test_all_scope_omits_filter_from_both_legs(self, mock_qdrant_client):
        set_client(mock_qdrant_client)
        mock_qdrant_client.query_points.return_value = query_response([])

        search_similar(None, "test query")

        for leg in mock_qdrant_client.query_points.call_args.kwargs["prefetch"]:
            assert leg.filter is None
    
class TestErrorHandling:
    """Test error handling and graceful degradation."""
    
    def test_search_qdrant_error(self, mock_qdrant_client):
        """Test search_similar handles transport-level Qdrant errors gracefully."""
        set_client(mock_qdrant_client)

        mock_qdrant_client.query_points.side_effect = Exception("Qdrant error")

        with patch('nlp_pillars.vectors.logger') as mock_logger:
            result = search_similar("linguistic-cognitive-foundations", "test query")

            assert result == []
            mock_logger.error.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
