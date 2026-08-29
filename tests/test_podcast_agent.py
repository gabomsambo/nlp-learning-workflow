"""
Tests for PodcastAgent.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

import pytest

from nlp_pillars.agents import podcast_agent
from nlp_pillars.agents.ingest_agent import IngestAgent
from nlp_pillars.agents.podcast_agent import (
    MAX_FULL_TEXT_CHARS, FullTextResult, InsufficientSourceMaterialError,
    LLMCallResult, PodcastAgent,
)
from nlp_pillars.schemas import PodcastScript, PaperRef, PaperNote, GroundPackCallRecord


def _extraction_mock(section: str, text: str):
    """Return value shape for mocked _generate_* extraction helpers."""
    return (
        text,
        GroundPackCallRecord(
            section=section,
            provider="deepseek",
            model="deepseek-v4-flash",
        ),
    )


# Fixtures are module-level so every class in this file can use them. They were
# class-scoped until a second class needed the same papers.
@pytest.fixture
def mock_paper():
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
def mock_notes():
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
def mock_ground_pack():
    """Create a mock ground pack."""
    return {
        "facts_outline": "- Problem: Long context modeling\n- Method: Sparse attention\n- Result: 10% improvement",
        "core_concepts": "Concept 1: Sparse Attention\nDefinition: Selective attention\nAnalogy: Like highlighting keywords",
        "metrics_datasets": "| Dataset | Metric | Score |\n| GLUE | Accuracy | 89.5% |",
        "limitations": "- High memory\n- Limited to English"
    }


@pytest.fixture
def mock_script_content():
    """Create a mock script."""
    return """[HOST]: Welcome to today's episode.

[HOST]: We're covering a fascinating paper on NLP.

[TRANSITION]

[HOST]: Let's dive into the methodology.

[HOST]: The key innovation here is the sparse attention mechanism.

[HOST]: Thanks for tuning in to this research breakdown. Today we covered Test Paper on NLP."""


class TestPodcastAgent:
    """Test suite for PodcastAgent."""

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
            mock_final = MagicMock()
            mock_final.usage.input_tokens = 1
            mock_final.usage.output_tokens = 2
            mock_final.stop_reason = "end_turn"
            mock_stream.get_final_message = AsyncMock(return_value=mock_final)

            agent.client = MagicMock()
            agent.client.messages.stream.return_value = mock_stream
            agent.synthesis_model = "test-model"

            result = await agent._call_claude("system", "user")

            assert result.text == "Hello World"

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

            assert "full paper text content" in result.text
            assert result.error is None
            agent.ingest_agent.ingest.assert_called_once_with(mock_paper)

    def test_get_full_text_handles_missing_url(self, mock_paper):
        """Empty text plus a reason when there is no PDF URL."""
        with patch.object(PodcastAgent, '__init__', lambda x: None):
            agent = PodcastAgent()
            agent.ingest_agent = MagicMock()

            # Remove PDF URL
            mock_paper.url_pdf = None

            result = agent._get_full_text(mock_paper)

            assert result.text == ""
            assert "no PDF URL" in result.error
            agent.ingest_agent.ingest.assert_not_called()

    def test_get_full_text_handles_ingest_error(self, mock_paper):
        """The ingest failure reason survives, instead of dying at the except.

        Returning a bare "" is exactly as informative as an empty paper: the
        caller could not tell "this PDF 404s" from "this paper has no PDF",
        and neither could the user.
        """
        from nlp_pillars.agents.ingest_agent import IngestError

        with patch.object(PodcastAgent, '__init__', lambda x: None):
            agent = PodcastAgent()
            agent.ingest_agent = MagicMock()
            agent.ingest_agent.ingest.side_effect = IngestError("PDF download failed")

            mock_paper.url_pdf = "http://example.com/test.pdf"

            result = agent._get_full_text(mock_paper)

            assert result.text == ""
            assert "PDF download failed" in result.error

    def test_init_configures_timeout_and_ingest_agent(self):
        """Constructor wires up the HTTP timeout and the IngestAgent.

        The rest of this suite stubs out ``__init__``, so without this test
        neither the client timeout nor the IngestAgent wiring is ever
        exercised.
        """
        with patch('nlp_pillars.agents.podcast_agent.get_settings') as mock_settings:
            mock_settings.return_value.anthropic_api_key = "sk-ant-test-key"
            mock_settings.return_value.deepseek_api_key = "sk-deepseek-test"
            mock_settings.return_value.deepseek_base_url = "https://api.deepseek.com"
            mock_settings.return_value.podcast_extraction_model = "deepseek-v4-flash"
            mock_settings.return_value.podcast_synthesis_model = "claude-sonnet-4-5-20250929"

            agent = PodcastAgent()

            assert agent.synthesis_model == "claude-sonnet-4-5-20250929"
            assert agent.extraction_model == "deepseek-v4-flash"
            assert isinstance(agent.ingest_agent, IngestAgent)
            # Timeouts matter: without them a hung Anthropic call would pin a
            # request forever.
            assert agent.client.timeout.read == 120.0
            assert agent.client.timeout.connect == 10.0

    @staticmethod
    def _embedded_full_text(paper_content: str) -> str:
        """Pull the paper body back out of an assembled prompt."""
        body = paper_content.split("=== FULL PAPER TEXT ===", 1)[1]
        return body.split("=== END FULL TEXT ===", 1)[0].strip()

    @pytest.mark.asyncio
    async def test_full_text_is_truncated_at_guard(
        self, mock_paper, mock_ground_pack, mock_script_content
    ):
        """Over-long full text is cut to the guard before it reaches Claude.

        This is the cost-critical path: paper_content is sent to five separate
        Claude calls, so an unbounded paper would multiply straight onto the
        bill.
        """
        oversized = "A" * (MAX_FULL_TEXT_CHARS + 50_000)

        with patch('nlp_pillars.agents.podcast_agent.get_paper_by_id') as mock_get_paper, \
             patch('nlp_pillars.agents.podcast_agent.get_notes_by_paper_id') as mock_get_notes, \
             patch.object(PodcastAgent, '__init__', lambda x: None):

            mock_get_paper.return_value = mock_paper
            mock_get_notes.return_value = None

            agent = PodcastAgent()
            agent._get_full_text = MagicMock(return_value=FullTextResult(oversized))
            agent._generate_facts_outline = AsyncMock(
                return_value=_extraction_mock("facts_outline", mock_ground_pack["facts_outline"]),
            )
            agent._generate_core_concepts = AsyncMock(
                return_value=_extraction_mock("core_concepts", mock_ground_pack["core_concepts"]),
            )
            agent._generate_metrics_datasets = AsyncMock(
                return_value=_extraction_mock("metrics_datasets", mock_ground_pack["metrics_datasets"]),
            )
            agent._generate_limitations = AsyncMock(
                return_value=_extraction_mock("limitations", mock_ground_pack["limitations"]),
            )
            agent._generate_final_script = AsyncMock(return_value=mock_script_content)

            await agent.generate("test-paper-123", "models-architectures")

            sent = agent._generate_facts_outline.call_args[0][0]
            body = self._embedded_full_text(sent)
            assert body.endswith("[TRUNCATED - paper continues...]")
            kept = body.split("\n\n[TRUNCATED", 1)[0]
            assert kept == "A" * MAX_FULL_TEXT_CHARS
            # Every prompt gets the same truncated content, including the
            # expensive final synthesis call.
            for call in (
                agent._generate_core_concepts,
                agent._generate_metrics_datasets,
                agent._generate_limitations,
            ):
                assert call.call_args[0][0] == sent
            assert agent._generate_final_script.call_args[0][1] == sent

    @pytest.mark.asyncio
    async def test_full_text_is_not_truncated_below_guard(
        self, mock_paper, mock_ground_pack, mock_script_content
    ):
        """Text at or under the guard is passed through untouched."""
        exact = "B" * MAX_FULL_TEXT_CHARS

        with patch('nlp_pillars.agents.podcast_agent.get_paper_by_id') as mock_get_paper, \
             patch('nlp_pillars.agents.podcast_agent.get_notes_by_paper_id') as mock_get_notes, \
             patch.object(PodcastAgent, '__init__', lambda x: None):

            mock_get_paper.return_value = mock_paper
            mock_get_notes.return_value = None

            agent = PodcastAgent()
            agent._get_full_text = MagicMock(return_value=FullTextResult(exact))
            agent._generate_facts_outline = AsyncMock(
                return_value=_extraction_mock("facts_outline", mock_ground_pack["facts_outline"]),
            )
            agent._generate_core_concepts = AsyncMock(
                return_value=_extraction_mock("core_concepts", mock_ground_pack["core_concepts"]),
            )
            agent._generate_metrics_datasets = AsyncMock(
                return_value=_extraction_mock("metrics_datasets", mock_ground_pack["metrics_datasets"]),
            )
            agent._generate_limitations = AsyncMock(
                return_value=_extraction_mock("limitations", mock_ground_pack["limitations"]),
            )
            agent._generate_final_script = AsyncMock(return_value=mock_script_content)

            await agent.generate("test-paper-123", "models-architectures")

            sent = agent._generate_facts_outline.call_args[0][0]
            assert "[TRUNCATED" not in sent
            assert self._embedded_full_text(sent) == exact

    @pytest.mark.asyncio
    async def test_pdf_extraction_does_not_block_event_loop(
        self, mock_paper, mock_ground_pack, mock_script_content
    ):
        """PDF ingest runs off the event loop.

        ``_get_full_text`` downloads and parses a PDF (~7s cold) and
        ``generate`` is awaited straight from a FastAPI route, so running it
        inline made the whole web UI unresponsive for its duration.
        """
        blocking_duration = 0.3

        def slow_ingest(paper):
            time.sleep(blocking_duration)
            return FullTextResult("Full paper content here...")

        with patch('nlp_pillars.agents.podcast_agent.get_paper_by_id') as mock_get_paper, \
             patch('nlp_pillars.agents.podcast_agent.get_notes_by_paper_id') as mock_get_notes, \
             patch.object(PodcastAgent, '__init__', lambda x: None):

            mock_get_paper.return_value = mock_paper
            mock_get_notes.return_value = None

            agent = PodcastAgent()
            agent._get_full_text = slow_ingest
            agent._generate_facts_outline = AsyncMock(
                return_value=_extraction_mock("facts_outline", mock_ground_pack["facts_outline"]),
            )
            agent._generate_core_concepts = AsyncMock(
                return_value=_extraction_mock("core_concepts", mock_ground_pack["core_concepts"]),
            )
            agent._generate_metrics_datasets = AsyncMock(
                return_value=_extraction_mock("metrics_datasets", mock_ground_pack["metrics_datasets"]),
            )
            agent._generate_limitations = AsyncMock(
                return_value=_extraction_mock("limitations", mock_ground_pack["limitations"]),
            )
            agent._generate_final_script = AsyncMock(return_value=mock_script_content)

            # A co-running coroutine standing in for another HTTP request.
            ticks = 0

            async def other_request():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.01)
                    ticks += 1

            ticker = asyncio.create_task(other_request())
            await agent.generate("test-paper-123", "models-architectures")
            ticker.cancel()

            # If the ingest ran inline the loop would have been frozen and the
            # ticker could not have advanced.
            assert ticks > 5, f"event loop appears blocked; only {ticks} ticks"

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

            # Mock full text extraction
            agent._get_full_text = MagicMock(return_value=FullTextResult("Full paper content here..."))

            # Mock all Claude calls
            agent._generate_facts_outline = AsyncMock(
                return_value=_extraction_mock("facts_outline", mock_ground_pack["facts_outline"]),
            )
            agent._generate_core_concepts = AsyncMock(
                return_value=_extraction_mock("core_concepts", mock_ground_pack["core_concepts"]),
            )
            agent._generate_metrics_datasets = AsyncMock(
                return_value=_extraction_mock("metrics_datasets", mock_ground_pack["metrics_datasets"]),
            )
            agent._generate_limitations = AsyncMock(
                return_value=_extraction_mock("limitations", mock_ground_pack["limitations"]),
            )
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

    @pytest.mark.asyncio
    async def test_generate_refuses_before_spending_when_there_is_nothing_to_read(
        self, mock_paper, mock_ground_pack, mock_script_content
    ):
        """No body, no abstract, no notes: raise before the first model call.

        Reproduced live against paper file:2dd76e910fbc, whose url_pdf points at
        a stale .cache/ path that does not exist. Before this guard, generation
        made all five Claude calls (~$0.27) against a "paper" consisting of a
        title, one author name and the string "[Full text not available...]",
        and reported it to the user as a green success.
        """
        mock_paper.abstract = None

        with patch('nlp_pillars.agents.podcast_agent.get_paper_by_id') as mock_get_paper, \
             patch('nlp_pillars.agents.podcast_agent.get_notes_by_paper_id') as mock_get_notes, \
             patch.object(PodcastAgent, '__init__', lambda x: None):

            mock_get_paper.return_value = mock_paper
            mock_get_notes.return_value = None

            agent = PodcastAgent()
            agent._get_full_text = MagicMock(
                return_value=FullTextResult("", "Local file not found: /app/.cache/uploads/x.pdf")
            )
            agent._generate_facts_outline = AsyncMock(
                return_value=_extraction_mock("facts_outline", mock_ground_pack["facts_outline"]),
            )
            agent._generate_core_concepts = AsyncMock(
                return_value=_extraction_mock("core_concepts", mock_ground_pack["core_concepts"]),
            )
            agent._generate_metrics_datasets = AsyncMock(
                return_value=_extraction_mock("metrics_datasets", mock_ground_pack["metrics_datasets"]),
            )
            agent._generate_limitations = AsyncMock(
                return_value=_extraction_mock("limitations", mock_ground_pack["limitations"]),
            )
            agent._generate_final_script = AsyncMock(return_value=mock_script_content)

            with pytest.raises(InsufficientSourceMaterialError) as exc:
                await agent.generate("test-paper-123", "models-architectures")

            # The reason has to be actionable, not "generation failed".
            message = str(exc.value)
            assert "Local file not found" in message
            assert "no abstract" in message
            assert mock_paper.title in message

            # Nothing was spent. This is the entire point of the guard.
            agent._generate_facts_outline.assert_not_called()
            agent._generate_core_concepts.assert_not_called()
            agent._generate_metrics_datasets.assert_not_called()
            agent._generate_limitations.assert_not_called()
            agent._generate_final_script.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_proceeds_from_an_abstract_but_records_that_it_did(
        self, mock_paper, mock_ground_pack, mock_script_content
    ):
        """An abstract with no body is thin, not empty: proceed and say so.

        Refusing here would block every paper whose PDF host is briefly
        unreachable, so this runs — but the caveat is attached to the result and
        stored with the row, because "written from the whole paper" and "written
        from two hundred words of abstract" looking identical is the bug.
        """
        with patch('nlp_pillars.agents.podcast_agent.get_paper_by_id') as mock_get_paper, \
             patch('nlp_pillars.agents.podcast_agent.get_notes_by_paper_id') as mock_get_notes, \
             patch.object(PodcastAgent, '__init__', lambda x: None):

            mock_get_paper.return_value = mock_paper  # has an abstract
            mock_get_notes.return_value = None

            agent = PodcastAgent()
            agent._get_full_text = MagicMock(
                return_value=FullTextResult("", "HTTP 404 fetching the PDF")
            )
            agent._generate_facts_outline = AsyncMock(
                return_value=_extraction_mock("facts_outline", mock_ground_pack["facts_outline"]),
            )
            agent._generate_core_concepts = AsyncMock(
                return_value=_extraction_mock("core_concepts", mock_ground_pack["core_concepts"]),
            )
            agent._generate_metrics_datasets = AsyncMock(
                return_value=_extraction_mock("metrics_datasets", mock_ground_pack["metrics_datasets"]),
            )
            agent._generate_limitations = AsyncMock(
                return_value=_extraction_mock("limitations", mock_ground_pack["limitations"]),
            )
            agent._generate_final_script = AsyncMock(return_value=mock_script_content)

            result = await agent.generate("test-paper-123", "models-architectures")

            assert result.source_material.level == "partial"
            assert result.source_material.full_text_chars == 0
            assert result.source_material.has_abstract is True
            assert result.source_material.has_notes is False
            assert len(result.source_material.warnings) == 1
            assert "HTTP 404 fetching the PDF" in result.source_material.warnings[0]
            agent._generate_final_script.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_proceeds_from_notes_alone(
        self, mock_paper, mock_notes, mock_ground_pack, mock_script_content
    ):
        """A notes row alone is enough: it carries problem, method and findings."""
        mock_paper.abstract = None

        with patch('nlp_pillars.agents.podcast_agent.get_paper_by_id') as mock_get_paper, \
             patch('nlp_pillars.agents.podcast_agent.get_notes_by_paper_id') as mock_get_notes, \
             patch.object(PodcastAgent, '__init__', lambda x: None):

            mock_get_paper.return_value = mock_paper
            mock_get_notes.return_value = mock_notes

            agent = PodcastAgent()
            agent._get_full_text = MagicMock(return_value=FullTextResult("", "no PDF"))
            agent._generate_facts_outline = AsyncMock(
                return_value=_extraction_mock("facts_outline", mock_ground_pack["facts_outline"]),
            )
            agent._generate_core_concepts = AsyncMock(
                return_value=_extraction_mock("core_concepts", mock_ground_pack["core_concepts"]),
            )
            agent._generate_metrics_datasets = AsyncMock(
                return_value=_extraction_mock("metrics_datasets", mock_ground_pack["metrics_datasets"]),
            )
            agent._generate_limitations = AsyncMock(
                return_value=_extraction_mock("limitations", mock_ground_pack["limitations"]),
            )
            agent._generate_final_script = AsyncMock(return_value=mock_script_content)

            result = await agent.generate("test-paper-123", "models-architectures")

            assert result.source_material.level == "partial"
            assert result.source_material.has_notes is True
            assert "extracted notes" in result.source_material.warnings[0]

    @pytest.mark.asyncio
    async def test_generate_from_full_text_records_no_warnings(
        self, mock_paper, mock_ground_pack, mock_script_content
    ):
        """The normal path stays clean — no caveat where there is nothing to caveat."""
        with patch('nlp_pillars.agents.podcast_agent.get_paper_by_id') as mock_get_paper, \
             patch('nlp_pillars.agents.podcast_agent.get_notes_by_paper_id') as mock_get_notes, \
             patch.object(PodcastAgent, '__init__', lambda x: None):

            mock_get_paper.return_value = mock_paper
            mock_get_notes.return_value = None

            agent = PodcastAgent()
            agent._get_full_text = MagicMock(return_value=FullTextResult("Body " * 500))
            agent._generate_facts_outline = AsyncMock(
                return_value=_extraction_mock("facts_outline", mock_ground_pack["facts_outline"]),
            )
            agent._generate_core_concepts = AsyncMock(
                return_value=_extraction_mock("core_concepts", mock_ground_pack["core_concepts"]),
            )
            agent._generate_metrics_datasets = AsyncMock(
                return_value=_extraction_mock("metrics_datasets", mock_ground_pack["metrics_datasets"]),
            )
            agent._generate_limitations = AsyncMock(
                return_value=_extraction_mock("limitations", mock_ground_pack["limitations"]),
            )
            agent._generate_final_script = AsyncMock(return_value=mock_script_content)

            result = await agent.generate("test-paper-123", "models-architectures")

            assert result.source_material.level == "full"
            assert result.source_material.warnings == []
            assert result.source_material.full_text_chars == len("Body " * 500)

    def test_whitespace_only_extraction_counts_as_no_body(self, mock_paper):
        """A PDF that parses to whitespace is an empty paper, not a full one."""
        with patch.object(PodcastAgent, '__init__', lambda x: None):
            agent = PodcastAgent()
            parsed = MagicMock()
            parsed.full_text = "   \n  "
            agent.ingest_agent = MagicMock()
            agent.ingest_agent.ingest.return_value = parsed
            mock_paper.url_pdf = "http://example.com/test.pdf"

            result = agent._get_full_text(mock_paper)

            assert result.text == ""
            assert "no text could be extracted" in result.error

    def test_agent_requires_api_key(self):
        """Test that agent raises error without API key."""
        with patch('nlp_pillars.agents.podcast_agent.get_settings') as mock_settings:
            mock_settings.return_value.anthropic_api_key = None

            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                PodcastAgent()


class TestOptionsReachTheModelAndTheRow:
    """The chosen options must survive from the constructor to the stored row.

    Without this, a configurable feature is configurable in the UI only: the
    prompts still go out with whatever the class defaults are, and the row keeps
    no record of what produced it.
    """

    @pytest.mark.asyncio
    async def test_options_are_recorded_on_the_generated_script(
        self, mock_paper, mock_ground_pack, mock_script_content
    ):
        from nlp_pillars.podcast_options import resolve

        options = resolve({"field": "biology", "length": "45"})

        with patch('nlp_pillars.agents.podcast_agent.get_paper_by_id') as mock_get_paper, \
             patch('nlp_pillars.agents.podcast_agent.get_notes_by_paper_id') as mock_get_notes, \
             patch.object(PodcastAgent, '__init__', lambda x: None):

            mock_get_paper.return_value = mock_paper
            mock_get_notes.return_value = None

            agent = PodcastAgent()
            agent.options = options
            agent._get_full_text = MagicMock(return_value=FullTextResult("Body text."))
            agent._generate_facts_outline = AsyncMock(
                return_value=_extraction_mock("facts_outline", mock_ground_pack["facts_outline"]),
            )
            agent._generate_core_concepts = AsyncMock(
                return_value=_extraction_mock("core_concepts", mock_ground_pack["core_concepts"]),
            )
            agent._generate_metrics_datasets = AsyncMock(
                return_value=_extraction_mock("metrics_datasets", mock_ground_pack["metrics_datasets"]),
            )
            agent._generate_limitations = AsyncMock(
                return_value=_extraction_mock("limitations", mock_ground_pack["limitations"]),
            )
            agent._generate_final_script = AsyncMock(return_value=mock_script_content)

            script = await agent.generate("test-paper-123", "models-architectures")

        assert script.options == options
        assert script.options.choices["field"].preset == "biology"
        assert script.options.choices["length"].label == "~45 minutes"

    def test_constructor_renders_the_prompts_for_the_options_it_is_given(self):
        """The rendered fragments come from the constructor argument."""
        from nlp_pillars.podcast_options import resolve

        with patch('nlp_pillars.agents.podcast_agent.get_settings') as mock_settings:
            mock_settings.return_value.anthropic_api_key = "sk-ant-test-key"
            mock_settings.return_value.deepseek_api_key = "sk-deepseek-test"
            mock_settings.return_value.deepseek_base_url = "https://api.deepseek.com"
            mock_settings.return_value.podcast_extraction_model = "deepseek-v4-flash"
            mock_settings.return_value.podcast_synthesis_model = "claude-sonnet-4-5-20250929"

            agent = PodcastAgent(options=resolve({"field": "economics"}))

            assert "economics" in agent._variables["field_name"]
            assert "Economics" in agent._settings_block
            # ...and the default construction still reproduces the old aiming.
            default_agent = PodcastAgent()
            assert default_agent._variables["field_paper"] == "NLP paper"

    def test_no_options_reproduces_the_default_aiming(self):
        with patch('nlp_pillars.agents.podcast_agent.get_settings') as mock_settings:
            mock_settings.return_value.anthropic_api_key = "sk-ant-test-key"
            mock_settings.return_value.deepseek_api_key = "sk-deepseek-test"
            mock_settings.return_value.deepseek_base_url = "https://api.deepseek.com"
            mock_settings.return_value.podcast_extraction_model = "deepseek-v4-flash"
            mock_settings.return_value.podcast_synthesis_model = "claude-sonnet-4-5-20250929"

            assert PodcastAgent().options == podcast_agent.DEFAULT_OPTIONS

    @pytest.mark.asyncio
    async def test_each_call_sets_its_own_temperature(self, mock_ground_pack):
        """Extraction temperatures reach DeepSeek; synthesis reaches Claude."""
        with patch.object(PodcastAgent, '__init__', lambda x: None):
            agent = PodcastAgent()
            agent._call_deepseek = AsyncMock(
                return_value=LLMCallResult("out", "deepseek", "deepseek-v4-flash"),
            )
            agent._call_claude = AsyncMock(
                return_value=LLMCallResult("out", "anthropic", "claude"),
            )
            agent._variables = podcast_agent.DEFAULT_VARIABLES
            agent._settings_block = podcast_agent.DEFAULT_SETTINGS_BLOCK

            await agent._generate_facts_outline("PAPER")
            await agent._generate_core_concepts("PAPER")
            await agent._generate_metrics_datasets("PAPER")
            await agent._generate_limitations("PAPER")
            await agent._generate_final_script(mock_ground_pack, "PAPER")

        deepseek_temps = [c.kwargs["temperature"] for c in agent._call_deepseek.call_args_list]
        assert deepseek_temps == [
            podcast_agent.TEMPERATURE_EXTRACTION,
            podcast_agent.TEMPERATURE_ANALYSIS,
            podcast_agent.TEMPERATURE_EXTRACTION,
            podcast_agent.TEMPERATURE_ANALYSIS,
        ]
        assert agent._call_claude.call_args.kwargs["temperature"] == podcast_agent.TEMPERATURE_SCRIPT

    @pytest.mark.asyncio
    async def test_temperature_reaches_the_api_call(self):
        """It has to reach the request, not just the helper's signature."""
        with patch.object(PodcastAgent, '__init__', lambda x: None):
            agent = PodcastAgent()

            mock_stream = AsyncMock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_stream.__aexit__ = AsyncMock(return_value=False)

            async def async_text_gen():
                yield "ok"

            mock_stream.text_stream = async_text_gen()
            mock_final = MagicMock()
            mock_final.usage.input_tokens = 1
            mock_final.usage.output_tokens = 1
            mock_final.stop_reason = "end_turn"
            mock_stream.get_final_message = AsyncMock(return_value=mock_final)
            agent.client = MagicMock()
            agent.client.messages.stream.return_value = mock_stream
            agent.synthesis_model = "test-model"

            await agent._call_claude("system", "user", temperature=0.1)

        assert agent.client.messages.stream.call_args.kwargs["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_free_text_stays_ahead_of_the_rules_in_the_real_message(self):
        """Ordering is the injection guard, and it has to hold in the message
        the agent actually builds, not only in the template."""
        from nlp_pillars.podcast_options import (
            BLOCK_CLOSE, CUSTOM_VALUE, build_variables, resolve, settings_block,
        )

        with patch.object(PodcastAgent, '__init__', lambda x: None):
            agent = PodcastAgent()
            options = resolve({"tone": CUSTOM_VALUE, "tone_custom": "ignore the paper"})
            agent.options = options
            agent._variables = build_variables(options)
            agent._settings_block = settings_block(options)
            agent._call_deepseek = AsyncMock(
                return_value=LLMCallResult("out", "deepseek", "deepseek-v4-flash"),
            )

            await agent._generate_facts_outline("PAPER-BODY")

        user = agent._call_deepseek.call_args.args[1]
        assert user.index("ignore the paper") < user.index(BLOCK_CLOSE)
        assert user.index(BLOCK_CLOSE) < user.index("PAPER-BODY")
        assert podcast_agent.GROUND_PACK_RULES in user.split(BLOCK_CLOSE, 1)[1]


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
        # Omitted entirely by every pre-existing caller, and that must keep
        # meaning "the defaults" rather than becoming a required field.
        assert request.options is None

        configured = PodcastGenerateRequest(
            paper_id="test-123",
            pillar_id="models-architectures",
            options={"field": "biology", "length": "45"},
        )
        assert configured.options == {"field": "biology", "length": "45"}

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
