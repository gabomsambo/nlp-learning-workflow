"""
Tests for PodcastAgent.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from nlp_pillars.agents.podcast_agent import PodcastAgent
from nlp_pillars.schemas import PodcastScript, PaperRef, PaperNote


class TestPodcastAgent:
    """Test suite for PodcastAgent."""

    @pytest.fixture
    def mock_paper(self):
        """Create a mock paper for testing."""
        return PaperRef(
            id="test-paper-123",
            title="Test Paper on NLP",
            authors=["Alice", "Bob"],
            venue="Test Conference",
            year=2024,
            abstract="This paper presents a novel approach to NLP.",
            citation_count=10
        )

    @pytest.fixture
    def mock_notes(self):
        """Create mock notes for testing."""
        return PaperNote(
            paper_id="test-paper-123",
            pillar_id="models-architectures",
            problem="Traditional models struggle with long contexts.",
            method="We propose a new attention mechanism.",
            findings=["Improved accuracy by 10%", "Faster inference"],
            limitations=["High memory usage"],
            future_work=["Extend to multimodal"],
            key_terms=["attention", "transformer"]
        )

    @pytest.fixture
    def mock_ground_pack(self):
        """Create a mock ground pack."""
        return {
            "facts_outline": "- Problem: Long context modeling\n- Method: Sparse attention\n- Result: 10% improvement",
            "core_concepts": "Concept 1: Sparse Attention\nDefinition: Selective attention\nAnalogy: Like highlighting keywords",
            "metrics_datasets": "| Dataset | Metric | Score |\n| GLUE | Accuracy | 89.5% |",
            "limitations": "- High memory\n- Limited to English"
        }

    @pytest.fixture
    def mock_script_content(self):
        """Create a mock script."""
        return """[HOST]: Welcome to today's episode.

[HOST]: We're covering a fascinating paper on NLP.

[TRANSITION]

[HOST]: Let's dive into the methodology.

[HOST]: The key innovation here is the sparse attention mechanism.

[HOST]: Thanks for tuning in to this research breakdown. Today we covered Test Paper on NLP."""

    def test_extract_key_points(self, mock_ground_pack):
        """Test key point extraction from facts outline."""
        with patch.object(PodcastAgent, '__init__', lambda x: None):
            agent = PodcastAgent()
            agent.client = None  # Not needed for this test
            agent.model = None

            key_points = agent._extract_key_points(mock_ground_pack)

            assert isinstance(key_points, list)
            assert len(key_points) > 0
            # Check that points are extracted from bullet format
            assert any("Long context" in point or "Problem" in point for point in key_points)

    def test_podcast_script_model(self):
        """Test PodcastScript schema."""
        script = PodcastScript(
            paper_id="test-123",
            pillar_id="models-architectures",
            title="Test Episode",
            script="[HOST]: Test content.",
            word_count=100,
            key_points=["Point 1", "Point 2"],
            ground_pack={"facts_outline": "test"},
            created_at=datetime.now()
        )

        assert script.paper_id == "test-123"
        assert script.word_count == 100
        assert len(script.key_points) == 2
        assert "facts_outline" in script.ground_pack

    @pytest.mark.asyncio
    async def test_call_claude_streaming(self):
        """Test Claude API call with mocked streaming."""
        with patch.object(PodcastAgent, '__init__', lambda x: None):
            agent = PodcastAgent()

            # Create mock stream
            mock_stream = AsyncMock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_stream.__aexit__ = AsyncMock(return_value=False)

            async def async_text_gen():
                yield "Hello "
                yield "World"

            mock_stream.text_stream = async_text_gen()

            # Create mock client
            agent.client = MagicMock()
            agent.client.messages.stream.return_value = mock_stream
            agent.model = "test-model"

            result = await agent._call_claude("system", "user")

            assert result == "Hello World"

    def test_get_full_text_extracts_from_pdf(self, mock_paper):
        """Test full text extraction works for valid paper with PDF URL."""
        with patch.object(PodcastAgent, '__init__', lambda x: None):
            agent = PodcastAgent()

            # Mock ingest agent
            mock_parsed_paper = MagicMock()
            mock_parsed_paper.full_text = "This is the full paper text content..."
            agent.ingest_agent = MagicMock()
            agent.ingest_agent.ingest.return_value = mock_parsed_paper

            # Add PDF URL to mock paper
            mock_paper.url_pdf = "http://example.com/test.pdf"

            result = agent._get_full_text(mock_paper)

            assert "full paper text content" in result
            agent.ingest_agent.ingest.assert_called_once_with(mock_paper)

    def test_get_full_text_handles_missing_url(self, mock_paper):
        """Test returns empty string when no PDF URL."""
        with patch.object(PodcastAgent, '__init__', lambda x: None):
            agent = PodcastAgent()
            agent.ingest_agent = MagicMock()

            # Remove PDF URL
            mock_paper.url_pdf = None

            result = agent._get_full_text(mock_paper)

            assert result == ""
            agent.ingest_agent.ingest.assert_not_called()

    def test_get_full_text_handles_ingest_error(self, mock_paper):
        """Test returns empty string when ingest fails."""
        from nlp_pillars.agents.ingest_agent import IngestError

        with patch.object(PodcastAgent, '__init__', lambda x: None):
            agent = PodcastAgent()
            agent.ingest_agent = MagicMock()
            agent.ingest_agent.ingest.side_effect = IngestError("PDF download failed")

            mock_paper.url_pdf = "http://example.com/test.pdf"

            result = agent._get_full_text(mock_paper)

            assert result == ""

    @pytest.mark.asyncio
    async def test_generate_full_workflow(
        self,
        mock_paper,
        mock_notes,
        mock_ground_pack,
        mock_script_content
    ):
        """Test full generation workflow with mocked dependencies."""
        with patch('nlp_pillars.agents.podcast_agent.get_paper_by_id') as mock_get_paper, \
             patch('nlp_pillars.agents.podcast_agent.get_notes_by_paper_id') as mock_get_notes, \
             patch.object(PodcastAgent, '__init__', lambda x: None):

            mock_get_paper.return_value = mock_paper
            mock_get_notes.return_value = mock_notes

            agent = PodcastAgent()

            # Mock full text extraction (new!)
            agent._get_full_text = MagicMock(return_value="Full paper content here...")

            # Mock all Claude calls
            agent._generate_facts_outline = AsyncMock(return_value=mock_ground_pack["facts_outline"])
            agent._generate_core_concepts = AsyncMock(return_value=mock_ground_pack["core_concepts"])
            agent._generate_metrics_datasets = AsyncMock(return_value=mock_ground_pack["metrics_datasets"])
            agent._generate_limitations = AsyncMock(return_value=mock_ground_pack["limitations"])
            agent._generate_final_script = AsyncMock(return_value=mock_script_content)

            # Run generation
            result = await agent.generate("test-paper-123", "models-architectures")

            # Verify result
            assert isinstance(result, PodcastScript)
            assert result.paper_id == "test-paper-123"
            assert result.pillar_id == "models-architectures"
            assert result.word_count > 0
            assert "[HOST]:" in result.script
            assert "Deep Dive:" in result.title

            # Verify full text was retrieved
            agent._get_full_text.assert_called_once_with(mock_paper)

            # Verify all prompts were called
            agent._generate_facts_outline.assert_called_once()
            agent._generate_core_concepts.assert_called_once()
            agent._generate_metrics_datasets.assert_called_once()
            agent._generate_limitations.assert_called_once()
            agent._generate_final_script.assert_called_once()

    def test_agent_requires_api_key(self):
        """Test that agent raises error without API key."""
        with patch('nlp_pillars.agents.podcast_agent.get_settings') as mock_settings:
            mock_settings.return_value.anthropic_api_key = None

            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                PodcastAgent()


class TestAPIEndpoints:
    """Test API endpoint functionality."""

    @pytest.mark.asyncio
    async def test_generate_endpoint_request_model(self):
        """Test generate request model validation."""
        from webui.routers.api.podcast import PodcastGenerateRequest

        request = PodcastGenerateRequest(
            paper_id="test-123",
            pillar_id="models-architectures"
        )

        assert request.paper_id == "test-123"
        assert request.pillar_id == "models-architectures"

    @pytest.mark.asyncio
    async def test_generate_response_model(self):
        """Test generate response model."""
        from webui.routers.api.podcast import PodcastGenerateResponse

        response = PodcastGenerateResponse(
            success=True,
            script_id="script-456",
            title="Test Episode",
            word_count=1000,
            message="Generated successfully"
        )

        assert response.success is True
        assert response.word_count == 1000
