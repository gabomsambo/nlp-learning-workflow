"""
Comprehensive tests for Qdrant vector store utility.
All Qdrant and OpenAI calls are mocked for fast, reliable testing.
"""

import pytest
import hashlib
from unittest.mock import Mock, patch, MagicMock
from qdrant_client import models

from nlp_pillars.schemas import PillarID
from nlp_pillars.vectors import (
    get_client, set_client, set_openai_client, reset_vector_size,
    ensure_collections, upsert_text, search_similar,
    _embed, _get_vector_size, COLLECTION_NAME
)


# Test fixtures
@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock_client = Mock()
    
    # Mock collections response
    mock_collections = Mock()
    mock_collections.collections = []
    mock_client.get_collections.return_value = mock_collections
    
    # Mock create collection
    mock_client.create_collection.return_value = None
    
    # Mock upsert
    mock_client.upsert.return_value = None
    
    # Mock search
    mock_client.search.return_value = []
    
    return mock_client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    
    # Mock embeddings response
    mock_response = Mock()
    mock_embedding = Mock()
    mock_embedding.embedding = [0.1, 0.2, 0.3, 0.4]  # Small test vector
    mock_response.data = [mock_embedding]
    mock_client.embeddings.create.return_value = mock_response
    
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
        """Test vector size fallback when embedding fails."""
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            size = _get_vector_size()
            
            assert size == 1536  # Default size for text-embedding-3-small
            mock_logger.warning.assert_called()


class TestEnsureCollections:
    """Test collection management functionality."""
    
    def test_ensure_collections_no_client(self):
        """Test ensure_collections when client is not available."""
        with patch('nlp_pillars.vectors.logger') as mock_logger:
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
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            result = upsert_text(PillarID.P1, "test.123", "test text")
            
            assert result == 0
            # Check that the warning about upsert was called (may have other warnings too)
            warning_calls = [call for call in mock_logger.warning.call_args_list]
            assert any("Cannot upsert text" in str(call) for call in warning_calls)
    
    def test_upsert_text_empty_text(self, mock_qdrant_client):
        """Test upsert_text with empty text."""
        set_client(mock_qdrant_client)
        
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            result = upsert_text(PillarID.P1, "test.123", "")
            
            assert result == 0
            mock_logger.warning.assert_called_once_with("Empty text provided for upsert")
    
    def test_upsert_text_success(self, mock_qdrant_client, mock_openai_client):
        """Test successful text upserting with chunking and embedding."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
        # Test text that will create multiple chunks
        test_text = "This is a long test text. " * 100  # Create text > 1000 chars
        
        result = upsert_text(PillarID.P2, "test.456", test_text, chunk_size=500, overlap=50)
        
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
        
        # Check deterministic ID format (SHA1 hex)
        expected_id = hashlib.sha1("P2|test.456|0".encode()).hexdigest()
        assert first_point.id == expected_id
        
        # Check payload
        assert first_point.payload['pillar_id'] == 'P2'
        assert first_point.payload['paper_id'] == 'test.456'
        assert first_point.payload['chunk_index'] == 0
        assert 'len' in first_point.payload
        
        # Check vector
        assert first_point.vector == [0.1, 0.2, 0.3, 0.4]
    
    def test_upsert_text_embedding_failure(self, mock_qdrant_client):
        """Test upsert_text when embedding fails for some chunks."""
        set_client(mock_qdrant_client)
        
        # Mock embedding to fail
        with patch('nlp_pillars.vectors._embed', side_effect=Exception("Embedding failed")):
            with patch('nlp_pillars.vectors.logger') as mock_logger:
                result = upsert_text(PillarID.P1, "test.123", "test text")
                
                assert result == 0
                mock_logger.warning.assert_called()
    
    def test_upsert_text_partial_embedding_failure(self, mock_qdrant_client, mock_openai_client):
        """Test upsert_text when some embeddings fail but others succeed."""
        set_client(mock_qdrant_client)
        
        # Mock embedding to fail on second call only
        embed_calls = 0
        def mock_embed(text):
            nonlocal embed_calls
            embed_calls += 1
            if embed_calls == 2:
                raise Exception("Embedding failed")
            return [0.1, 0.2, 0.3, 0.4]
        
        with patch('nlp_pillars.vectors._embed', side_effect=mock_embed):
            # Create shorter text that generates exactly 2 chunks
            test_text = "A" * 600 + "B" * 600  # Two distinct chunks
            
            result = upsert_text(PillarID.P1, "test.123", test_text, chunk_size=500, overlap=50)
            
            # Should have upserted chunks except the second one that failed
            # Check that we got some successful embeds but not all
            assert result >= 1
            assert result < embed_calls  # Some failed


class TestSearchSimilar:
    """Test similarity search functionality."""
    
    def test_search_similar_no_client(self):
        """Test search_similar when client is not available."""
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            result = search_similar(PillarID.P1, "test query")
            
            assert result == []
            # Check that the warning about search was called (may have other warnings too)
            warning_calls = [call for call in mock_logger.warning.call_args_list]
            assert any("Cannot search similar text" in str(call) for call in warning_calls)
    
    def test_search_similar_empty_query(self, mock_qdrant_client):
        """Test search_similar with empty query."""
        set_client(mock_qdrant_client)
        
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            result = search_similar(PillarID.P1, "")
            
            assert result == []
            mock_logger.warning.assert_called_once_with("Empty query text provided")
    
    def test_search_similar_success(self, mock_qdrant_client, mock_openai_client):
        """Test successful similarity search with proper filtering."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
        # Mock search results
        mock_hit1 = Mock()
        mock_hit1.payload = {"paper_id": "paper.123"}
        mock_hit1.score = 0.9
        
        mock_hit2 = Mock()
        mock_hit2.payload = {"paper_id": "paper.456"}
        mock_hit2.score = 0.8
        
        mock_qdrant_client.search.return_value = [mock_hit1, mock_hit2]
        
        result = search_similar(PillarID.P3, "test query", top_k=5)
        
        # Verify search was called with correct parameters
        mock_qdrant_client.search.assert_called_once()
        search_call = mock_qdrant_client.search.call_args
        
        assert search_call[1]['collection_name'] == COLLECTION_NAME
        assert search_call[1]['query_vector'] == [0.1, 0.2, 0.3, 0.4]
        assert search_call[1]['limit'] == 15  # top_k * 3 for deduplication
        assert search_call[1]['with_payload'] is True
        
        # Check pillar filter
        query_filter = search_call[1]['query_filter']
        assert len(query_filter.must) == 1
        field_condition = query_filter.must[0]
        assert field_condition.key == "pillar_id"
        assert field_condition.match.value == "P3"
        
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
        mock_hit1 = Mock()
        mock_hit1.payload = {"paper_id": "paper.123"}
        mock_hit1.score = 0.9
        
        mock_hit2 = Mock()
        mock_hit2.payload = {"paper_id": "paper.123"}  # Same paper
        mock_hit2.score = 0.7  # Lower score
        
        mock_hit3 = Mock()
        mock_hit3.payload = {"paper_id": "paper.456"}
        mock_hit3.score = 0.8
        
        mock_qdrant_client.search.return_value = [mock_hit1, mock_hit2, mock_hit3]
        
        result = search_similar(PillarID.P1, "test query", top_k=5)
        
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
        mock_hits = []
        for i in range(10):
            hit = Mock()
            hit.payload = {"paper_id": f"paper.{i}"}
            hit.score = 0.9 - (i * 0.1)  # Decreasing scores
            mock_hits.append(hit)
        
        mock_qdrant_client.search.return_value = mock_hits
        
        result = search_similar(PillarID.P1, "test query", top_k=3)
        
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
                result = search_similar(PillarID.P1, "test query")
                
                assert result == []
                mock_logger.error.assert_called_once()


class TestNamespaceEnforcement:
    """Test that pillar isolation is enforced."""
    
    def test_all_searches_include_pillar_filter(self, mock_qdrant_client, mock_openai_client):
        """Test that all search operations include pillar_id filter."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
        mock_qdrant_client.search.return_value = []
        
        # Test different pillars
        for pillar in [PillarID.P1, PillarID.P3, PillarID.P5]:
            search_similar(pillar, "test query")
            
            # Get the last search call
            search_call = mock_qdrant_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            # Verify pillar filter is present
            assert len(query_filter.must) == 1
            field_condition = query_filter.must[0]
            assert field_condition.key == "pillar_id"
            assert field_condition.match.value == pillar.value
    
    def test_upsert_includes_pillar_payload(self, mock_qdrant_client, mock_openai_client):
        """Test that all upserts include pillar_id in payload."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
        upsert_text(PillarID.P4, "test.789", "test text")
        
        # Verify upsert was called
        upsert_call = mock_qdrant_client.upsert.call_args
        points = upsert_call[1]['points']
        
        # Check all points have pillar_id in payload
        for point in points:
            assert point.payload['pillar_id'] == 'P4'
            assert point.payload['paper_id'] == 'test.789'


class TestErrorHandling:
    """Test error handling and graceful degradation."""
    
    def test_upsert_qdrant_error(self, mock_qdrant_client, mock_openai_client):
        """Test upsert_text handles Qdrant errors gracefully."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
        mock_qdrant_client.upsert.side_effect = Exception("Qdrant error")
        
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            result = upsert_text(PillarID.P1, "test.123", "test text")
            
            assert result == 0
            mock_logger.error.assert_called_once()
    
    def test_search_qdrant_error(self, mock_qdrant_client, mock_openai_client):
        """Test search_similar handles Qdrant errors gracefully."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
        mock_qdrant_client.search.side_effect = Exception("Qdrant error")
        
        with patch('nlp_pillars.vectors.logger') as mock_logger:
            result = search_similar(PillarID.P1, "test query")
            
            assert result == []
            mock_logger.error.assert_called_once()
    
    def test_deterministic_ids(self, mock_qdrant_client, mock_openai_client):
        """Test that point IDs are deterministic and stable."""
        set_client(mock_qdrant_client)
        set_openai_client(mock_openai_client)
        
        # Upsert same content twice
        upsert_text(PillarID.P1, "test.123", "same text")
        first_call = mock_qdrant_client.upsert.call_args
        
        mock_qdrant_client.reset_mock()
        
        upsert_text(PillarID.P1, "test.123", "same text")
        second_call = mock_qdrant_client.upsert.call_args
        
        # IDs should be identical
        first_points = first_call[1]['points']
        second_points = second_call[1]['points']
        
        assert len(first_points) == len(second_points)
        for i in range(len(first_points)):
            assert first_points[i].id == second_points[i].id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
