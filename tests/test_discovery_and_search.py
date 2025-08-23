"""
Comprehensive tests for Discovery Agent and Search Tools.
All network calls are mocked for fast, reliable testing.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import asyncio
import httpx
import arxiv

from nlp_pillars.schemas import (
    PillarID, 
    PillarConfig, 
    SearchQuery, 
    PaperRef,
    DiscoveryInput,
    DiscoveryOutput
)
from nlp_pillars.agents.discovery_agent import DiscoveryAgent
from nlp_pillars.tools.arxiv_tool import ArXivTool
from nlp_pillars.tools.searxng_tool import SearXNGTool


# Test fixtures
@pytest.fixture
def sample_pillar():
    """Sample pillar configuration for testing."""
    return PillarConfig(
        id=PillarID.P2,
        name="Models & Architectures",
        goal="Understand cutting-edge model architectures and emerging paradigms",
        focus_areas=["Transformer variants", "Long-context models", "Multimodal architectures"],
        created_at=datetime.now()
    )


@pytest.fixture
def sample_search_query():
    """Sample search query for testing."""
    return SearchQuery(
        pillar_id=PillarID.P2,
        query="transformer attention mechanisms",
        filters={"categories": ["cs.CL", "cs.LG"]},
        max_results=5
    )


@pytest.fixture
def sample_paper_refs():
    """Sample PaperRef objects for testing."""
    return [
        PaperRef(
            id="2401.12345",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            venue="NeurIPS",
            year=2017,
            url_pdf="https://arxiv.org/pdf/1706.03762.pdf",
            abstract="The dominant sequence transduction models...",
            citation_count=50000
        ),
        PaperRef(
            id="2402.67890", 
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Jacob Devlin", "Ming-Wei Chang"],
            venue="NAACL",
            year=2019,
            url_pdf="https://arxiv.org/pdf/1810.04805.pdf",
            abstract="We introduce a new language representation model...",
            citation_count=30000
        )
    ]


class TestArXivTool:
    """Test cases for ArXiv search tool."""
    
    def test_init(self):
        """Test ArXiv tool initialization."""
        tool = ArXivTool()
        assert tool.rate_limit_seconds == 3
        assert tool.client is not None
        assert hasattr(tool, 'last_request_time')
    
    @patch('nlp_pillars.tools.arxiv_tool.arxiv.Client')
    def test_search_success(self, mock_client_class, sample_search_query):
        """Test successful ArXiv search."""
        # Mock arxiv.Result objects
        mock_result1 = Mock()
        mock_result1.entry_id = "http://arxiv.org/abs/2401.12345v1"
        mock_result1.title = "Attention Is All You Need"
        mock_result1.authors = [Mock(__str__ = lambda self: "Ashish Vaswani"), Mock(__str__ = lambda self: "Noam Shazeer")]
        mock_result1.published = datetime(2017, 6, 12)
        mock_result1.pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
        mock_result1.summary = "The dominant sequence transduction models..."
        mock_result1.primary_category = "cs.CL"
        mock_result1.journal_ref = None
        
        mock_result2 = Mock()
        mock_result2.entry_id = "http://arxiv.org/abs/2402.67890v2"
        mock_result2.title = "BERT: Pre-training"
        mock_result2.authors = [Mock(__str__ = lambda self: "Jacob Devlin")]
        mock_result2.published = datetime(2019, 5, 24)
        mock_result2.pdf_url = "https://arxiv.org/pdf/1810.04805.pdf"
        mock_result2.summary = "We introduce a new language representation..."
        mock_result2.primary_category = "cs.CL"
        mock_result2.journal_ref = "NAACL 2019"
        
        # Mock client
        mock_client = Mock()
        mock_client.results.return_value = [mock_result1, mock_result2]
        mock_client_class.return_value = mock_client
        
        # Test search
        tool = ArXivTool()
        results = tool.search(sample_search_query)
        
        # Verify results
        assert len(results) == 2
        assert all(isinstance(paper, PaperRef) for paper in results)
        
        # Check first result
        paper1 = results[0]
        assert paper1.id == "2401.12345"
        assert paper1.title == "Attention Is All You Need"
        assert "Ashish Vaswani" in paper1.authors
        assert paper1.year == 2017
        assert paper1.url_pdf == "https://arxiv.org/pdf/1706.03762.pdf"
        
        # Check second result
        paper2 = results[1]
        assert paper2.id == "2402.67890"
        assert paper2.venue == "NAACL 2019"
    
    @patch('nlp_pillars.tools.arxiv_tool.arxiv.Client')
    def test_search_error_handling(self, mock_client_class, sample_search_query):
        """Test ArXiv search error handling."""
        # Mock client that raises an error
        mock_client = Mock()
        mock_client.results.side_effect = ConnectionError("API error")
        mock_client_class.return_value = mock_client
        
        tool = ArXivTool()
        
        # Should raise RetryError due to retry mechanism
        from tenacity import RetryError
        with pytest.raises(RetryError):
            tool.search(sample_search_query)
    
    @patch('nlp_pillars.tools.arxiv_tool.time.sleep')
    def test_rate_limiting(self, mock_sleep):
        """Test rate limiting functionality."""
        tool = ArXivTool()
        
        # Simulate recent request
        tool.last_request_time = 1000
        
        with patch('nlp_pillars.tools.arxiv_tool.time.time', return_value=1001):
            tool._enforce_rate_limit()
            mock_sleep.assert_called_once()  # Should sleep for ~2 seconds
    
    def test_convert_to_paper_ref(self):
        """Test conversion of arXiv result to PaperRef."""
        tool = ArXivTool()
        
        # Mock arxiv result
        mock_result = Mock()
        mock_result.entry_id = "http://arxiv.org/abs/2401.12345v1"
        mock_result.title = "Test Paper Title"
        mock_result.authors = [Mock(__str__ = lambda self: "Author One"), Mock(__str__ = lambda self: "Author Two")]
        mock_result.published = datetime(2023, 1, 15)
        mock_result.pdf_url = "https://arxiv.org/pdf/2401.12345.pdf"
        mock_result.summary = "This is a test abstract."
        mock_result.primary_category = "cs.CL"
        mock_result.journal_ref = None
        
        paper_ref = tool._convert_to_paper_ref(mock_result)
        
        assert paper_ref.id == "2401.12345"
        assert paper_ref.title == "Test Paper Title"
        assert paper_ref.authors == ["Author One", "Author Two"]
        assert paper_ref.year == 2023
        assert paper_ref.url_pdf == "https://arxiv.org/pdf/2401.12345.pdf"
        assert paper_ref.abstract == "This is a test abstract."
        assert paper_ref.venue == "arXiv:cs.CL"


class TestSearXNGTool:
    """Test cases for SearXNG search tool."""
    
    def test_init(self):
        """Test SearXNG tool initialization."""
        tool = SearXNGTool()
        assert tool.base_url is not None
        assert tool.rate_limit_seconds == 1
        assert tool.client is not None
    
    def test_init_custom_url(self):
        """Test SearXNG tool initialization with custom URL."""
        custom_url = "http://custom.searxng.com"
        tool = SearXNGTool(base_url=custom_url)
        assert tool.base_url == custom_url
    
    @pytest.mark.asyncio
    async def test_search_async_success(self, sample_search_query):
        """Test successful SearXNG async search."""
        # Mock response data
        mock_response_data = {
            "results": [
                {
                    "title": "Attention Is All You Need",
                    "url": "https://arxiv.org/abs/1706.03762",
                    "content": "The dominant sequence transduction models are based on...",
                    "engine": "arxiv"
                },
                {
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "url": "https://arxiv.org/pdf/1810.04805.pdf",
                    "content": "We introduce BERT, a new method for pre-training...",
                    "engine": "semantic scholar"
                }
            ]
        }
        
        # Mock HTTP client
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = Mock()
        
        with patch.object(SearXNGTool, '__init__', lambda x: None):
            tool = SearXNGTool()
            tool.base_url = "http://localhost:8080"
            tool.rate_limit_seconds = 1
            tool.last_request_time = 0
            tool.academic_engines = ["arxiv", "semantic scholar"]
            
            # Mock the HTTP client
            tool.client = AsyncMock()
            tool.client.get.return_value = mock_response
            
            results = await tool.search_async(sample_search_query)
        
        # Verify results
        assert len(results) == 2
        assert all(isinstance(paper, PaperRef) for paper in results)
        
        # Check first result
        paper1 = results[0]
        assert paper1.title == "Attention Is All You Need"
        assert "1706.03762" in paper1.id  # Should extract arXiv ID
        
        # Check second result  
        paper2 = results[1]
        assert paper2.title == "BERT: Pre-training of Deep Bidirectional Transformers"
        assert paper2.url_pdf == "https://arxiv.org/pdf/1810.04805.pdf"
    
    @pytest.mark.asyncio 
    async def test_search_async_error_handling(self, sample_search_query):
        """Test SearXNG search error handling."""
        with patch.object(SearXNGTool, '__init__', lambda x: None):
            tool = SearXNGTool()
            tool.base_url = "http://localhost:8080"
            tool.rate_limit_seconds = 1
            tool.last_request_time = 0
            tool.academic_engines = ["arxiv", "semantic scholar"]
            
            # Mock HTTP client that raises an error
            tool.client = AsyncMock()
            tool.client.get.side_effect = httpx.HTTPError("Connection failed")
            
            from tenacity import RetryError
            with pytest.raises(RetryError):
                await tool.search_async(sample_search_query)
    
    def test_build_search_url(self, sample_search_query):
        """Test SearXNG search URL building."""
        tool = SearXNGTool(base_url="http://localhost:8080")
        
        url = tool._build_search_url(sample_search_query)
        
        assert "http://localhost:8080/search" in url
        assert "transformer+attention+mechanisms" in url or "transformer%20attention%20mechanisms" in url
        assert "format=json" in url
        assert "categories=science" in url
    
    def test_convert_result_to_paper_ref(self):
        """Test conversion of SearXNG result to PaperRef."""
        tool = SearXNGTool()
        
        result = {
            "title": "Test Paper: A Novel Approach (2023)",
            "url": "https://arxiv.org/abs/2301.12345",
            "content": "This paper by John Doe and Jane Smith presents...",
            "engine": "arxiv"
        }
        
        paper_ref = tool._convert_result_to_paper_ref(result)
        
        assert paper_ref is not None
        assert paper_ref.title == "Test Paper: A Novel Approach (2023)"
        assert paper_ref.id == "2301.12345"  # Should extract arXiv ID
        assert paper_ref.venue == "arXiv"
        assert len(paper_ref.authors) > 0  # Should extract some authors
    
    def test_generate_paper_id(self):
        """Test paper ID generation from URLs."""
        tool = SearXNGTool()
        
        # Test arXiv URL
        arxiv_url = "https://arxiv.org/abs/2301.12345"
        assert tool._generate_paper_id(arxiv_url, "title") == "2301.12345"
        
        # Test DOI URL
        doi_url = "https://doi.org/10.1038/nature12345"
        assert tool._generate_paper_id(doi_url, "title") == "10.1038/nature12345"
        
        # Test fallback
        other_url = "https://example.com/paper"
        paper_id = tool._generate_paper_id(other_url, "title")
        assert paper_id.startswith("searxng_")
    
    def test_extract_year(self):
        """Test year extraction from text."""
        tool = SearXNGTool()
        
        # Test extraction from various sources
        assert tool._extract_year("Paper Title (2023)", "", "") == 2023
        assert tool._extract_year("", "Published in 2022", "") == 2022
        assert tool._extract_year("", "", "https://arxiv.org/abs/2021.12345") == 2021
        assert tool._extract_year("Old text 1989 and new 2023", "", "") == 2023  # Should get most recent
        assert tool._extract_year("No years here", "", "") is None


class TestDiscoveryAgent:
    """Test cases for Discovery Agent."""
    
    @patch('nlp_pillars.agents.discovery_agent.instructor.from_openai')
    @patch('nlp_pillars.agents.discovery_agent.OpenAI')
    def test_init(self, mock_openai, mock_instructor):
        """Test Discovery Agent initialization."""
        # Mock the components
        mock_client = Mock()
        mock_instructor.return_value = mock_client
        
        agent = DiscoveryAgent()
        
        assert agent.model == "gpt-4o-mini"  # Default from config
        assert agent.system_prompt is not None
        assert agent.agent is not None
    
    @patch('nlp_pillars.agents.discovery_agent.instructor.from_openai')
    @patch('nlp_pillars.agents.discovery_agent.OpenAI')
    def test_discover_success(self, mock_openai, mock_instructor, sample_pillar):
        """Test successful discovery query generation."""
        # Mock the atomic agent response
        mock_discovery_output = DiscoveryOutput(
            queries=[
                SearchQuery(
                    pillar_id=PillarID.P2,
                    query="transformer architecture variants recent advances",
                    filters={"categories": ["cs.CL"], "time_range": "year"},
                    max_results=10
                ),
                SearchQuery(
                    pillar_id=PillarID.P2,
                    query="long context attention mechanisms",
                    filters={"categories": ["cs.LG"]},
                    max_results=8
                ),
                SearchQuery(
                    pillar_id=PillarID.P2,
                    query="multimodal transformer applications",
                    filters={},
                    max_results=12
                )
            ],
            rationale="Generated diverse queries targeting core architecture research, recent advances, and practical applications to provide comprehensive coverage of the Models & Architectures pillar."
        )
        
        # Mock the agent
        mock_atomic_agent = Mock()
        mock_atomic_agent.run.return_value = mock_discovery_output
        
        with patch.object(DiscoveryAgent, '__init__', lambda x, model=None: None):
            agent = DiscoveryAgent()
            agent.agent = mock_atomic_agent
            
            result = agent.discover(sample_pillar)
        
        # Verify result
        assert isinstance(result, DiscoveryOutput)
        assert len(result.queries) == 3
        assert all(query.pillar_id == PillarID.P2 for query in result.queries)
        assert result.rationale is not None
        assert len(result.rationale) > 0
        
        # Verify query diversity
        query_texts = [q.query for q in result.queries]
        assert "transformer" in query_texts[0].lower()
        assert "long context" in query_texts[1].lower()
        assert "multimodal" in query_texts[2].lower()
    
    @patch('nlp_pillars.agents.discovery_agent.instructor.from_openai')
    @patch('nlp_pillars.agents.discovery_agent.OpenAI')
    def test_discover_insufficient_queries(self, mock_openai, mock_instructor, sample_pillar):
        """Test discovery with insufficient queries (should raise error)."""
        # Mock response with too few queries
        mock_discovery_output = DiscoveryOutput(
            queries=[
                SearchQuery(
                    pillar_id=PillarID.P2,
                    query="transformer variants",
                    filters={},
                    max_results=10
                )
            ],
            rationale="Only one query generated."
        )
        
        mock_atomic_agent = Mock()
        mock_atomic_agent.run.return_value = mock_discovery_output
        
        with patch.object(DiscoveryAgent, '__init__', lambda x, model=None: None):
            agent = DiscoveryAgent()
            agent.agent = mock_atomic_agent
            
            with pytest.raises(ValueError, match="must generate at least 3 queries"):
                agent.discover(sample_pillar)
    
    @patch('nlp_pillars.agents.discovery_agent.instructor.from_openai')
    @patch('nlp_pillars.agents.discovery_agent.OpenAI')
    def test_discover_for_pillar_id(self, mock_openai, mock_instructor):
        """Test discovery by pillar ID."""
        # Mock successful discovery
        mock_discovery_output = DiscoveryOutput(
            queries=[
                SearchQuery(pillar_id=PillarID.P1, query="test1", filters={}, max_results=5),
                SearchQuery(pillar_id=PillarID.P1, query="test2", filters={}, max_results=5),
                SearchQuery(pillar_id=PillarID.P1, query="test3", filters={}, max_results=5)
            ],
            rationale="Test rationale"
        )
        
        mock_atomic_agent = Mock()
        mock_atomic_agent.run.return_value = mock_discovery_output
        
        with patch.object(DiscoveryAgent, '__init__', lambda x, model=None: None):
            agent = DiscoveryAgent()
            agent.agent = mock_atomic_agent
            
            result = agent.discover_for_pillar_id(PillarID.P1)
        
        assert isinstance(result, DiscoveryOutput)
        assert len(result.queries) == 3
        assert all(query.pillar_id == PillarID.P1 for query in result.queries)
    
    def test_get_priority_topics_for_pillar(self):
        """Test getting priority topics for a pillar."""
        with patch.object(DiscoveryAgent, '__init__', lambda x, model=None: None):
            agent = DiscoveryAgent()
            
            topics = agent.get_priority_topics_for_pillar(PillarID.P2)
            
            assert isinstance(topics, list)
            assert len(topics) > 0
            assert "Transformer variants" in topics  # From PILLAR_CONFIGS


class TestIntegration:
    """Integration tests for Discovery Agent and Search Tools."""
    
    @patch('nlp_pillars.agents.discovery_agent.instructor.from_openai')
    @patch('nlp_pillars.agents.discovery_agent.OpenAI')
    @patch('nlp_pillars.tools.arxiv_tool.arxiv.Client')
    def test_discovery_to_arxiv_integration(self, mock_arxiv_client, mock_openai, mock_instructor, sample_pillar):
        """Test that DiscoveryAgent output is compatible with ArXiv tool input."""
        # Mock Discovery Agent
        mock_discovery_output = DiscoveryOutput(
            queries=[
                SearchQuery(
                    pillar_id=PillarID.P2,
                    query="transformer attention mechanisms",
                    filters={"categories": ["cs.CL"]},
                    max_results=5
                ),
                SearchQuery(
                    pillar_id=PillarID.P2,
                    query="multimodal neural networks",
                    filters={},
                    max_results=8
                ),
                SearchQuery(
                    pillar_id=PillarID.P2,
                    query="state space models",
                    filters={"categories": ["cs.LG"]},
                    max_results=10
                )
            ],
            rationale="Comprehensive query strategy targeting different aspects of model architectures."
        )
        
        mock_atomic_agent = Mock()
        mock_atomic_agent.run.return_value = mock_discovery_output
        
        # Mock ArXiv results
        mock_arxiv_result = Mock()
        mock_arxiv_result.entry_id = "http://arxiv.org/abs/2301.12345v1"
        mock_arxiv_result.title = "Test Paper"
        mock_arxiv_result.authors = [Mock(__str__ = lambda self: "Test Author")]
        mock_arxiv_result.published = datetime(2023, 1, 15)
        mock_arxiv_result.pdf_url = "https://arxiv.org/pdf/2301.12345.pdf"
        mock_arxiv_result.summary = "Test abstract"
        mock_arxiv_result.primary_category = "cs.CL"
        mock_arxiv_result.journal_ref = None
        
        mock_client = Mock()
        mock_client.results.return_value = [mock_arxiv_result]
        mock_arxiv_client.return_value = mock_client
        
        # Test integration
        with patch.object(DiscoveryAgent, '__init__', lambda x, model=None: None):
            discovery_agent = DiscoveryAgent()
            discovery_agent.agent = mock_atomic_agent
            
            arxiv_tool = ArXivTool()
            
            # Get queries from discovery agent
            discovery_result = discovery_agent.discover(sample_pillar)
            
            # Use first query with ArXiv tool
            search_query = discovery_result.queries[0]
            arxiv_results = arxiv_tool.search(search_query)
            
            # Verify compatibility
            assert len(arxiv_results) == 1
            assert isinstance(arxiv_results[0], PaperRef)
            assert arxiv_results[0].title == "Test Paper"
            assert arxiv_results[0].id == "2301.12345"
    
    @patch('nlp_pillars.agents.discovery_agent.instructor.from_openai') 
    @patch('nlp_pillars.agents.discovery_agent.OpenAI')
    def test_discovery_to_searxng_integration(self, mock_openai, mock_instructor, sample_pillar):
        """Test that DiscoveryAgent output is compatible with SearXNG tool input."""
        # Mock Discovery Agent (same as above)
        mock_discovery_output = DiscoveryOutput(
            queries=[
                SearchQuery(
                    pillar_id=PillarID.P2,
                    query="neural machine translation",
                    filters={"time_range": "year"},
                    max_results=7
                ),
                SearchQuery(
                    pillar_id=PillarID.P2,
                    query="transformer architectures",
                    filters={},
                    max_results=5
                ),
                SearchQuery(
                    pillar_id=PillarID.P2,
                    query="attention mechanisms",
                    filters={"categories": ["cs.LG"]},
                    max_results=8
                )
            ],
            rationale="Focused queries for recent NMT and architecture research."
        )
        
        mock_atomic_agent = Mock()
        mock_atomic_agent.run.return_value = mock_discovery_output
        
        with patch.object(DiscoveryAgent, '__init__', lambda x, model=None: None):
            discovery_agent = DiscoveryAgent()
            discovery_agent.agent = mock_atomic_agent
            
            # Get queries from discovery agent
            discovery_result = discovery_agent.discover(sample_pillar)
            
            # Verify query structure is compatible with SearXNG
            search_query = discovery_result.queries[0]
            assert hasattr(search_query, 'pillar_id')
            assert hasattr(search_query, 'query')
            assert hasattr(search_query, 'filters')
            assert hasattr(search_query, 'max_results')
            
            # Verify query can be used to build SearXNG URL
            searxng_tool = SearXNGTool(base_url="http://localhost:8080")
            url = searxng_tool._build_search_url(search_query)
            
            assert "neural+machine+translation" in url or "neural%20machine%20translation" in url
            assert "format=json" in url


class TestSchemaValidation:
    """Test schema validation and compliance."""
    
    def test_paper_ref_schema_validation(self):
        """Test PaperRef schema validation."""
        # Valid PaperRef
        valid_paper = PaperRef(
            id="2301.12345",
            title="Test Paper",
            authors=["Author One", "Author Two"]
        )
        assert valid_paper.id == "2301.12345"
        assert len(valid_paper.authors) == 2
        
        # Missing required fields should be handled gracefully
        with pytest.raises(Exception):  # Pydantic validation error
            PaperRef(id="123")  # Missing title and authors
    
    def test_search_query_schema_validation(self):
        """Test SearchQuery schema validation."""
        # Valid SearchQuery
        valid_query = SearchQuery(
            pillar_id=PillarID.P1,
            query="test query",
            filters={"category": "cs.CL"},
            max_results=10
        )
        assert valid_query.pillar_id == PillarID.P1
        assert valid_query.max_results == 10
        
        # Test default values
        minimal_query = SearchQuery(
            pillar_id=PillarID.P2,
            query="minimal query"
        )
        assert minimal_query.max_results == 10  # Default value
        assert minimal_query.filters == {}  # Default empty dict
    
    def test_discovery_output_schema_validation(self):
        """Test DiscoveryOutput schema validation."""
        queries = [
            SearchQuery(pillar_id=PillarID.P1, query="query1"),
            SearchQuery(pillar_id=PillarID.P1, query="query2"),
            SearchQuery(pillar_id=PillarID.P1, query="query3")
        ]
        
        valid_output = DiscoveryOutput(
            queries=queries,
            rationale="Test rationale explaining query selection strategy."
        )
        
        assert len(valid_output.queries) == 3
        assert valid_output.rationale is not None
        assert all(isinstance(q, SearchQuery) for q in valid_output.queries)


# Performance and reliability tests
class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_arxiv_tool_partial_data(self):
        """Test ArXiv tool with partial/missing data."""
        tool = ArXivTool()
        
        # Mock result with missing fields
        mock_result = Mock()
        mock_result.entry_id = "http://arxiv.org/abs/2301.12345v1"
        mock_result.title = "Test Paper"
        mock_result.authors = [Mock(__str__ = lambda self: "Author One")]
        mock_result.published = None  # Missing publication date
        mock_result.pdf_url = None  # Missing PDF URL
        mock_result.summary = None  # Missing abstract
        mock_result.primary_category = "cs.CL"
        mock_result.journal_ref = None
        
        paper_ref = tool._convert_to_paper_ref(mock_result)
        
        # Should handle missing fields gracefully
        assert paper_ref.id == "2301.12345"
        assert paper_ref.title == "Test Paper"
        assert paper_ref.year is None
        assert paper_ref.url_pdf is None
        assert paper_ref.abstract is None
    
    def test_searxng_tool_malformed_results(self):
        """Test SearXNG tool with malformed results."""
        tool = SearXNGTool()
        
        # Test with missing title
        malformed_result = {
            "url": "https://example.com/paper",
            "content": "Some content...",
            "engine": "test"
        }
        
        paper_ref = tool._convert_result_to_paper_ref(malformed_result)
        assert paper_ref is None  # Should return None for invalid results
        
        # Test with missing URL
        malformed_result2 = {
            "title": "Test Paper",
            "content": "Some content...",
            "engine": "test"
        }
        
        paper_ref2 = tool._convert_result_to_paper_ref(malformed_result2)
        assert paper_ref2 is None  # Should return None for invalid results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
