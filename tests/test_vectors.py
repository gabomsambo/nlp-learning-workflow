"""
Comprehensive tests for Qdrant vector store utility.
All Qdrant and OpenAI calls are mocked for fast, reliable testing.
"""

import pytest
import hashlib
import math
import os
import uuid
from contextlib import contextmanager
from unittest.mock import Mock, patch, MagicMock
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from nlp_pillars import vectors
from nlp_pillars.vectors import (
    get_client, set_client, set_openai_client, reset_vector_size,
    ensure_collections, upsert_text, search_similar,
    _embed, _embed_batch, _get_vector_size, COLLECTION_NAME, PILLAR_ID_FIELD
)
from nlp_pillars.tools.pdf_loader import chunk_text


def make_hit(paper_id, score, chunk_index=0):
    """One ScoredPoint-shaped search hit."""
    hit = Mock()
    hit.payload = {"paper_id": paper_id, "chunk_index": chunk_index}
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

    # Collection info, with the payload index already in place so
    # _ensure_payload_indexes() is a no-op unless a test says otherwise.
    collection_info = Mock()
    collection_info.payload_schema = {PILLAR_ID_FIELD: Mock()}
    mock_client.get_collection.return_value = collection_info
    mock_client.create_payload_index.return_value = None

    # Mock upsert
    mock_client.upsert.return_value = None

    # Mock search
    mock_client.query_points.return_value = query_response([])

    return mock_client


def embedding_item(vector, index):
    """One item of an OpenAI embeddings response: a vector and the input it answers."""
    item = Mock()
    item.embedding = vector
    item.index = index
    return item


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing.

    Answers a batch, because upsert_text sends one. A fixture that returned a single
    fixed item regardless of input would let a batching bug — the wrong number of
    vectors, or vectors mapped to the wrong chunks — pass green.
    """
    mock_client = Mock()

    def create(model, input, **kwargs):
        texts = input if isinstance(input, list) else [input]
        response = Mock()
        response.data = [
            embedding_item([0.1, 0.2, 0.3, 0.4], idx) for idx in range(len(texts))
        ]
        return response

    mock_client.embeddings.create.side_effect = create

    return mock_client


@pytest.fixture(autouse=True)
def setup_clean_state():
    """Clean state for each test."""
    set_client(None)
    set_openai_client(None) 
    reset_vector_size()
    yield
    set_client(None)
    set_openai_client(None)
    reset_vector_size()


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
    @patch.dict('os.environ', {'QDRANT_URL': 'http://localhost:6333'})
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


class TestEmbeddings:
    """Test embedding generation functionality."""
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'})
    def test_embed_success(self, mock_openai_client):
        """Test successful text embedding."""
        set_openai_client(mock_openai_client)
        
        result = _embed("test text")

        assert result == [0.1, 0.2, 0.3, 0.4]
        mock_openai_client.embeddings.create.assert_called_once_with(
            model='text-embedding-3-small',
            input='test text'
        )
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key', 'EMBEDDING_MODEL': 'custom-model'})
    def test_embed_custom_model(self, mock_openai_client):
        """Test embedding with custom model from environment."""
        set_openai_client(mock_openai_client)
        
        _embed("test text")
        
        mock_openai_client.embeddings.create.assert_called_once_with(
            model='custom-model',
            input='test text'
        )
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_no_api_key(self):
        """Test embedding failure when OpenAI API key is missing."""
        with pytest.raises(RuntimeError, match="OpenAI client not configured"):
            _embed("test text")
    
    def test_embed_openai_error(self, mock_openai_client):
        """Test embedding failure when OpenAI API call fails."""
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")
        set_openai_client(mock_openai_client)
        
        with pytest.raises(RuntimeError, match="Failed to generate embedding"):
            _embed("test text")
    
    def test_get_vector_size_success(self, mock_openai_client):
        """Test vector size determination from embedding."""
        set_openai_client(mock_openai_client)
        
        size = _get_vector_size()
        
        assert size == 4  # Length of mock embedding
    
    def test_get_vector_size_failure_fallback(self):
        """Test vector size fallback when embedding fails.

        The failure is forced explicitly rather than relying on the environment
        lacking a usable OPENAI_API_KEY. It used to depend on that, and it made the
        test both order-dependent and expensive: get_settings() does
        load_dotenv(override=True), so the real key in .env beats any placeholder a
        developer exports, the embedding call SUCCEEDED against the live API — billing
        the project on every local suite run — and the warning this asserts on was
        never logged. Passing on CI and failing locally for that reason is precisely
        the asymmetry AGENTS.md warns about; a unit test must not decide whether it is
        testing the failure path by looking at your shell.

        _vector_size is a cached module global, so it is cleared before and restored
        after, or whichever test ran first would decide the answer.
        """
        previous_size = vectors._vector_size
        vectors._vector_size = None

        exploding = Mock()
        exploding.embeddings.create.side_effect = RuntimeError("no API key")

        try:
            set_openai_client(exploding)
            with patch('nlp_pillars.vectors.logger') as mock_logger:
                size = _get_vector_size()

                assert size == 1536  # Default size for text-embedding-3-small
                mock_logger.warning.assert_called()
        finally:
            vectors._vector_size = previous_size
            set_openai_client(None)


class TestEnsureCollections:
    """Test collection management functionality."""
    
    def test_ensure_collections_no_client(self):
        """Test ensure_collections when client is not available."""
        with no_qdrant_client(), patch('nlp_pillars.vectors.logger') as mock_logger:
            ensure_collections()
            
            # Check that the warning about collections was called (may have other warnings too)
            warning_calls = [call for call in mock_logger.warning.call_args_list]
            assert any("Cannot ensure collections" in str(call) for call in warning_calls)
    
    def test_ensure_collections_exists(self, mock_qdrant_client, mock_openai_client):
        """Test ensure_collections when collection already exists."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
        # Mock existing collection
        mock_collection = Mock()
        mock_collection.name = COLLECTION_NAME
        mock_qdrant_client.get_collections.return_value.collections = [mock_collection]
        
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            ensure_collections()
            
            mock_qdrant_client.create_collection.assert_not_called()
            mock_logger.info.assert_any_call(f"Collection '{COLLECTION_NAME}' already exists")
    
    def test_ensure_collections_creates_new(self, mock_qdrant_client, mock_openai_client):
        """Test ensure_collections creates new collection with correct configuration."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
        # Mock no existing collections
        mock_qdrant_client.get_collections.return_value.collections = []
        
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            ensure_collections()
            
            # Verify create_collection was called with correct parameters
            mock_qdrant_client.create_collection.assert_called_once()
            call_args = mock_qdrant_client.create_collection.call_args
            
            assert call_args[1]['collection_name'] == COLLECTION_NAME
            vectors_config = call_args[1]['vectors_config']
            assert vectors_config.size == 4  # Mock embedding size
            assert vectors_config.distance == models.Distance.COSINE
            
            mock_logger.info.assert_any_call(
                f"Created collection '{COLLECTION_NAME}' with vector size 4 and cosine distance"
            )
    
    def test_ensure_collections_error(self, mock_qdrant_client):
        """Test ensure_collections handles errors gracefully."""
        set_client(mock_qdrant_client)
        mock_qdrant_client.get_collections.side_effect = Exception("Database error")
        
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            ensure_collections()
            
            mock_logger.error.assert_called_once()


class TestUpsertText:
    """Test text upserting functionality."""
    
    def test_upsert_text_no_client(self):
        """Test upsert_text when client is not available."""
        with no_qdrant_client(), patch('nlp_pillars.vectors.logger') as mock_logger:
            result = upsert_text("linguistic-cognitive-foundations", "test.123", "test text")

            assert result == 0
            # Check that the warning about upsert was called (may have other warnings too)
            warning_calls = [call for call in mock_logger.warning.call_args_list]
            assert any("Cannot upsert text" in str(call) for call in warning_calls)
    
    def test_upsert_text_empty_text(self, mock_qdrant_client):
        """Test upsert_text with empty text."""
        set_client(mock_qdrant_client)
        
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            result = upsert_text("linguistic-cognitive-foundations", "test.123", "")

            assert result == 0
            mock_logger.warning.assert_called_once_with("Empty text provided for upsert")
    
    def test_upsert_text_success(self, mock_qdrant_client, mock_openai_client):
        """Test successful text upserting with chunking and embedding."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
        # Test text that will create multiple chunks. chunk_size/overlap are in
        # TOKENS (see upsert_text), so size the text in tokens too: this is
        # ~700 tokens against a 100-token budget.
        test_text = "This is a long test text. " * 100

        result = upsert_text("models-architectures", "test.456", test_text, chunk_size=100, overlap=10)

        # Should have upserted chunks
        assert result > 0

        # Verify upsert was called
        mock_qdrant_client.upsert.assert_called_once()
        upsert_call = mock_qdrant_client.upsert.call_args

        assert upsert_call[1]['collection_name'] == COLLECTION_NAME
        points = upsert_call[1]['points']

        # Verify points structure
        assert len(points) > 0
        first_point = points[0]

        # Check deterministic ID format. vectors.py takes the first 16 bytes
        # of the SHA1 digest and formats them as a UUID (qdrant-client 1.19
        # requires point ids to be an unsigned int or a UUID, not an
        # arbitrary hex string) - see nlp_pillars/vectors.py's upsert_text.
        hash_bytes = hashlib.sha1("models-architectures|test.456|0".encode()).digest()[:16]
        expected_id = str(uuid.UUID(bytes=hash_bytes))
        assert first_point.id == expected_id

        # Check payload
        assert first_point.payload['pillar_id'] == 'models-architectures'
        assert first_point.payload['paper_id'] == 'test.456'
        assert first_point.payload['chunk_index'] == 0
        assert 'len' in first_point.payload
        
        # Check vector
        assert first_point.vector == [0.1, 0.2, 0.3, 0.4]
    
    def test_upsert_text_embedding_failure(self, mock_qdrant_client):
        """Test upsert_text when embedding fails."""
        set_client(mock_qdrant_client)

        with patch('nlp_pillars.vectors._embed_batch', side_effect=RuntimeError("Embedding failed")):
            with patch('nlp_pillars.vectors.logger') as mock_logger:
                result = upsert_text("linguistic-cognitive-foundations", "test.123", "test text")

                assert result == 0
                mock_logger.error.assert_called()

        # Nothing may reach Qdrant when the document could not be embedded.
        mock_qdrant_client.upsert.assert_not_called()

    def test_a_failed_batch_writes_nothing_rather_than_a_partial_document(
        self, mock_qdrant_client
    ):
        """A partial embed is the one outcome with no honest way to report it.

        upsert_text used to catch a failed chunk, log a warning, `continue`, and
        upsert whatever survived — returning a plausible-looking count for a paper
        that was only partly in the collection. Nothing downstream could tell that
        from a short paper. Now the whole upsert is abandoned: no points are written
        and the return is 0, which both callers already treat as a failed vector
        stage for non-empty text.
        """
        set_client(mock_qdrant_client)

        calls = {"n": 0}

        def failing_second_batch(texts):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("Embedding failed")
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        # Comfortably more than one batch of EMBED_BATCH_SIZE chunks, so the second
        # call is reached and the first batch's vectors are already in hand.
        test_text = "Alpha beta gamma delta epsilon. " * 900

        with patch('nlp_pillars.vectors._embed_batch', side_effect=failing_second_batch):
            result = upsert_text(
                "linguistic-cognitive-foundations", "test.123", test_text,
                chunk_size=20, overlap=2,
            )

        assert calls["n"] >= 2, "the text did not span more than one batch"
        assert result == 0
        mock_qdrant_client.upsert.assert_not_called()

    def test_chunks_are_embedded_in_batches_not_one_request_each(
        self, mock_qdrant_client, mock_openai_client
    ):
        """The whole point of the change: one request per EMBED_BATCH_SIZE chunks.

        The captain's 448-chunk paper cost 448 sequential OpenAI round trips and
        dominated a 3m20s upload.
        """
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)

        test_text = "Alpha beta gamma delta epsilon. " * 900

        result = upsert_text(
            "models-architectures", "test.batched", test_text,
            chunk_size=20, overlap=2,
        )

        assert result > vectors.EMBED_BATCH_SIZE, "not enough chunks to test batching"

        calls = mock_openai_client.embeddings.create.call_args_list
        expected_calls = math.ceil(result / vectors.EMBED_BATCH_SIZE)
        assert len(calls) == expected_calls
        assert len(calls) < result, "still one request per chunk"

        # Every request sends a list, and no request exceeds the batch size.
        for call in calls:
            sent = call[1]['input']
            assert isinstance(sent, list)
            assert 0 < len(sent) <= vectors.EMBED_BATCH_SIZE

    def test_a_multi_batch_document_matches_the_one_at_a_time_result(
        self, mock_qdrant_client
    ):
        """Same vectors, same count, same order as embedding one chunk at a time.

        Each chunk is given a vector derived from its own text, so a point holding
        the wrong chunk's vector is detectable — which is the failure batching can
        introduce and the reason _embed_batch places vectors by the response's
        `index` rather than by arrival order.
        """
        set_client(mock_qdrant_client)

        def vector_for(text):
            digest = hashlib.sha1(text.encode()).digest()
            return [b / 255 for b in digest[:4]]

        test_text = "Alpha beta gamma delta epsilon. " * 900
        chunks = chunk_text(test_text, chunk_size=20, chunk_overlap=2)
        assert len(chunks) > vectors.EMBED_BATCH_SIZE, "not a multi-batch document"

        # The one-at-a-time expectation, built without going through upsert_text.
        expected = [vector_for(chunk) for chunk in chunks]

        def batched(texts):
            return [vector_for(t.strip()) for t in texts]

        with patch('nlp_pillars.vectors._embed_batch', side_effect=batched):
            result = upsert_text(
                "models-architectures", "test.order", test_text,
                chunk_size=20, overlap=2,
            )

        assert result == len(chunks)

        points = mock_qdrant_client.upsert.call_args[1]['points']
        assert len(points) == len(chunks)

        # Order, payload and vector-to-chunk mapping, per point.
        for idx, (point, chunk) in enumerate(zip(points, chunks)):
            assert point.payload['chunk_index'] == idx
            assert point.payload['len'] == len(chunk)
            assert point.payload['paper_id'] == 'test.order'
            assert point.payload['pillar_id'] == 'models-architectures'
            assert point.vector == expected[idx]

            id_string = f"models-architectures|test.order|{idx}"
            hash_bytes = hashlib.sha1(id_string.encode()).digest()[:16]
            assert point.id == str(uuid.UUID(bytes=hash_bytes))


class TestEmbedBatch:
    """_embed_batch maps one vector onto every input, or raises."""

    def test_vectors_are_placed_by_the_responses_index_not_arrival_order(
        self, mock_openai_client
    ):
        """Order is read from the response, not assumed.

        A reshuffled response that is silently trusted produces a collection where
        every chunk is present and every one of them answers for a different chunk —
        invisible in the stored payloads, which carry no chunk text.
        """
        response = Mock()
        response.data = [
            embedding_item([2.0], 2),
            embedding_item([0.0], 0),
            embedding_item([1.0], 1),
        ]
        mock_openai_client.embeddings.create.side_effect = None
        mock_openai_client.embeddings.create.return_value = response
        set_openai_client(mock_openai_client)

        assert _embed_batch(["a", "b", "c"]) == [[0.0], [1.0], [2.0]]

    def test_a_short_response_raises_rather_than_returning_what_arrived(
        self, mock_openai_client
    ):
        response = Mock()
        response.data = [embedding_item([0.0], 0), embedding_item([1.0], 1)]
        mock_openai_client.embeddings.create.side_effect = None
        mock_openai_client.embeddings.create.return_value = response
        set_openai_client(mock_openai_client)

        with pytest.raises(RuntimeError, match="returned 2 vectors for 3 inputs"):
            _embed_batch(["a", "b", "c"])

    def test_a_duplicated_index_raises(self, mock_openai_client):
        response = Mock()
        response.data = [embedding_item([0.0], 0), embedding_item([1.0], 0)]
        mock_openai_client.embeddings.create.side_effect = None
        mock_openai_client.embeddings.create.return_value = response
        set_openai_client(mock_openai_client)

        with pytest.raises(RuntimeError, match="index 0 twice"):
            _embed_batch(["a", "b"])

    def test_an_out_of_range_index_raises(self, mock_openai_client):
        response = Mock()
        response.data = [embedding_item([0.0], 0), embedding_item([1.0], 7)]
        mock_openai_client.embeddings.create.side_effect = None
        mock_openai_client.embeddings.create.return_value = response
        set_openai_client(mock_openai_client)

        with pytest.raises(RuntimeError, match="out-of-range index"):
            _embed_batch(["a", "b"])

    def test_an_api_failure_is_a_runtime_error(self, mock_openai_client):
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")
        set_openai_client(mock_openai_client)

        with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
            _embed_batch(["a", "b"])

    def test_an_empty_batch_needs_no_client(self):
        set_openai_client(None)
        assert _embed_batch([]) == []


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
    
    def test_search_similar_success(self, mock_qdrant_client, mock_openai_client):
        """Test successful similarity search with proper filtering."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
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
        assert search_call[1]['query'] == [0.1, 0.2, 0.3, 0.4]
        assert search_call[1]['limit'] == 15  # top_k * 3 for deduplication
        assert search_call[1]['with_payload'] is True

        # Check pillar filter
        query_filter = search_call[1]['query_filter']
        assert len(query_filter.must) == 1
        field_condition = query_filter.must[0]
        assert field_condition.key == "pillar_id"
        assert field_condition.match.value == "data-training-methodologies"
        
        # Check results format
        expected_results = [
            {"paper_id": "paper.123", "score": 0.9},
            {"paper_id": "paper.456", "score": 0.8}
        ]
        assert result == expected_results
    
    def test_search_similar_deduplication(self, mock_qdrant_client, mock_openai_client):
        """Test search_similar deduplicates by paper_id keeping highest score."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
        # Mock search results with duplicates
        mock_qdrant_client.query_points.return_value = query_response([
            make_hit("paper.123", 0.9),
            make_hit("paper.123", 0.7),  # Same paper, lower score
            make_hit("paper.456", 0.8),
        ])

        result = search_similar("linguistic-cognitive-foundations", "test query", top_k=5)
        
        # Should deduplicate and keep highest score for paper.123
        expected_results = [
            {"paper_id": "paper.123", "score": 0.9},  # Higher score kept
            {"paper_id": "paper.456", "score": 0.8}
        ]
        assert result == expected_results
    
    def test_search_similar_top_k_limit(self, mock_qdrant_client, mock_openai_client):
        """Test search_similar respects top_k limit."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
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

        with patch('nlp_pillars.vectors._embed', side_effect=Exception("Embedding failed")):
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

    def test_raises_when_method_is_missing(self, mock_qdrant_client, mock_openai_client):
        """AttributeError (the 1.19 removal) propagates as RuntimeError."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        mock_qdrant_client.query_points.side_effect = AttributeError(
            "'QdrantClient' object has no attribute 'query_points'"
        )

        with pytest.raises(RuntimeError, match="incompatible with the installed"):
            search_similar("linguistic-cognitive-foundations", "test query")

    def test_raises_when_signature_is_rejected(self, mock_qdrant_client, mock_openai_client):
        """TypeError (a renamed/removed keyword) propagates as RuntimeError."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        mock_qdrant_client.query_points.side_effect = TypeError(
            "query_points() got an unexpected keyword argument 'query_vector'"
        )

        with pytest.raises(RuntimeError, match="incompatible with the installed"):
            search_similar("linguistic-cognitive-foundations", "test query")

    def test_raises_when_server_rejects_request(self, mock_qdrant_client, mock_openai_client):
        """A 4xx — e.g. the missing pillar_id payload index under strict mode."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        mock_qdrant_client.query_points.side_effect = unexpected_response(
            400, b'{"status":{"error":"Bad request: Index required but not found for \\"pillar_id\\""}}'
        )

        with pytest.raises(RuntimeError, match="Qdrant rejected the search"):
            search_similar("linguistic-cognitive-foundations", "test query")

    def test_returns_empty_on_server_error(self, mock_qdrant_client, mock_openai_client):
        """A 5xx is a runtime failure, not a contract mismatch: still degrades."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        mock_qdrant_client.query_points.side_effect = unexpected_response(503)

        assert search_similar("linguistic-cognitive-foundations", "test query") == []

    def test_raises_on_unexpected_response_shape(self, mock_qdrant_client, mock_openai_client):
        """A bare list (the old search() shape) is not silently read as no hits."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        mock_qdrant_client.query_points.return_value = [make_hit("paper.123", 0.9)]

        with pytest.raises(RuntimeError, match="unexpected query_points"):
            search_similar("linguistic-cognitive-foundations", "test query")


class TestPayloadIndex:
    """Every read filters on pillar_id, so that key must carry a payload index.

    Qdrant Cloud runs strict mode (`unindexed_filtering_retrieve: false`) and
    answers a filtered query on an unindexed key with 400. The live collection
    had 170 points and an empty payload_schema, so the filter could never work.
    """

    def test_creates_index_when_missing(self, mock_qdrant_client, mock_openai_client):
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)

        existing = Mock()
        existing.name = COLLECTION_NAME
        mock_qdrant_client.get_collections.return_value.collections = [existing]
        mock_qdrant_client.get_collection.return_value.payload_schema = {}

        ensure_collections()

        mock_qdrant_client.create_payload_index.assert_called_once()
        call = mock_qdrant_client.create_payload_index.call_args
        assert call[1]["collection_name"] == COLLECTION_NAME
        assert call[1]["field_name"] == PILLAR_ID_FIELD
        assert call[1]["field_schema"] == models.PayloadSchemaType.KEYWORD

    def test_does_not_recreate_existing_index(self, mock_qdrant_client, mock_openai_client):
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)

        existing = Mock()
        existing.name = COLLECTION_NAME
        mock_qdrant_client.get_collections.return_value.collections = [existing]
        # fixture already reports pillar_id in payload_schema

        ensure_collections()

        mock_qdrant_client.create_payload_index.assert_not_called()

    def test_index_created_for_new_collection(self, mock_qdrant_client, mock_openai_client):
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        mock_qdrant_client.get_collections.return_value.collections = []
        mock_qdrant_client.get_collection.return_value.payload_schema = {}

        ensure_collections()

        mock_qdrant_client.create_collection.assert_called_once()
        mock_qdrant_client.create_payload_index.assert_called_once()

    def test_index_failure_is_not_fatal(self, mock_qdrant_client, mock_openai_client):
        """Writes do not need the index; search_similar() raises if it is absent."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)

        existing = Mock()
        existing.name = COLLECTION_NAME
        mock_qdrant_client.get_collections.return_value.collections = [existing]
        mock_qdrant_client.get_collection.side_effect = Exception("index lookup failed")

        ensure_collections()  # must not raise


class TestNamespaceEnforcement:
    """Test that pillar isolation is enforced."""
    
    def test_all_searches_include_pillar_filter(self, mock_qdrant_client, mock_openai_client):
        """Test that all search operations include pillar_id filter."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
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
            query_filter = search_call[1]['query_filter']

            # Verify pillar filter is present
            assert len(query_filter.must) == 1
            field_condition = query_filter.must[0]
            assert field_condition.key == "pillar_id"
            assert field_condition.match.value == pillar
    
    def test_upsert_includes_pillar_payload(self, mock_qdrant_client, mock_openai_client):
        """Test that all upserts include pillar_id in payload."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
        upsert_text("evaluation-interpretability", "test.789", "test text")

        # Verify upsert was called
        upsert_call = mock_qdrant_client.upsert.call_args
        points = upsert_call[1]['points']

        # Check all points have pillar_id in payload
        for point in points:
            assert point.payload['pillar_id'] == 'evaluation-interpretability'
            assert point.payload['paper_id'] == 'test.789'


class TestErrorHandling:
    """Test error handling and graceful degradation."""
    
    def test_upsert_qdrant_error(self, mock_qdrant_client, mock_openai_client):
        """Test upsert_text handles Qdrant errors gracefully."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
        mock_qdrant_client.upsert.side_effect = Exception("Qdrant error")
        
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            result = upsert_text("linguistic-cognitive-foundations", "test.123", "test text")

            assert result == 0
            mock_logger.error.assert_called_once()
    
    def test_search_qdrant_error(self, mock_qdrant_client, mock_openai_client):
        """Test search_similar handles transport-level Qdrant errors gracefully."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)

        mock_qdrant_client.query_points.side_effect = Exception("Qdrant error")

        with patch('nlp_pillars.vectors.logger') as mock_logger:
            result = search_similar("linguistic-cognitive-foundations", "test query")

            assert result == []
            mock_logger.error.assert_called_once()

    def test_deterministic_ids(self, mock_qdrant_client, mock_openai_client):
        """Test that point IDs are deterministic and stable."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)

        # Upsert same content twice
        upsert_text("linguistic-cognitive-foundations", "test.123", "same text")
        first_call = mock_qdrant_client.upsert.call_args

        mock_qdrant_client.reset_mock()

        upsert_text("linguistic-cognitive-foundations", "test.123", "same text")
        second_call = mock_qdrant_client.upsert.call_args
        
        # IDs should be identical
        first_points = first_call[1]['points']
        second_points = second_call[1]['points']
        
        assert len(first_points) == len(second_points)
        for i in range(len(first_points)):
            assert first_points[i].id == second_points[i].id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
