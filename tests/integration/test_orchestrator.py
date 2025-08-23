"""
Integration tests for the end-to-end orchestrator.
All agents, tools, and external services are mocked for reliable testing.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from nlp_pillars.schemas import (
    PillarID, PipelineResult, PaperRef, PaperNote, Lesson, QuizCard,
    ParsedPaper, DiscoveryOutput, SearchQuery, DifficultyLevel, QuestionType
)
from nlp_pillars.orchestrator import Orchestrator


# Test fixtures
@pytest.fixture
def sample_paper_refs():
    """Sample paper references for testing."""
    return [
        PaperRef(
            id="paper.123",
            title="Attention Is All You Need",
            authors=["Vaswani", "Shazeer"],
            venue="NIPS",
            year=2017,
            url_pdf="https://example.com/paper123.pdf",
            abstract="Transformer architecture paper"
        ),
        PaperRef(
            id="paper.456", 
            title="BERT: Pre-training Transformers",
            authors=["Devlin", "Chang"],
            venue="NAACL",
            year=2019,
            url_pdf="https://example.com/paper456.pdf",
            abstract="BERT pre-training paper"
        )
    ]


@pytest.fixture
def sample_parsed_paper():
    """Sample parsed paper for testing."""
    return ParsedPaper(
        paper_ref=PaperRef(
            id="paper.123",
            title="Test Paper",
            authors=["Author A"],
            venue="Test Venue",
            year=2023
        ),
        full_text="Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50,
        chunks=[
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        ],
        figures_count=2,
        tables_count=1,
        references=["ref1", "ref2"]
    )


@pytest.fixture
def sample_paper_note():
    """Sample paper note for testing."""
    return PaperNote(
        paper_id="paper.123",
        pillar_id=PillarID.P2,
        problem="Limited attention mechanisms in neural networks",
        method="Multi-head self-attention with scaled dot-product",
        findings=[
            "Achieved state-of-the-art results on machine translation",
            "Reduced training time compared to RNNs"
        ],
        limitations=[
            "Quadratic complexity with sequence length",
            "High memory requirements"
        ],
        future_work=[
            "Explore sparse attention patterns",
            "Apply to other sequence tasks"
        ],
        key_terms=["attention", "transformer", "self-attention"],
        confidence_score=0.9
    )


@pytest.fixture
def sample_lesson():
    """Sample lesson for testing."""
    return Lesson(
        paper_id="paper.123",
        pillar_id=PillarID.P2,
        tl_dr="Transformers revolutionized NLP by using attention mechanisms instead of recurrence",
        takeaways=[
            "Self-attention captures long-range dependencies effectively",
            "Parallel processing improves training efficiency",
            "Multi-head attention provides different representation subspaces"
        ],
        practice_ideas=[
            "Implement a mini-transformer for text classification",
            "Experiment with different attention patterns"
        ],
        connections=["BERT builds on transformer encoder"],
        difficulty=DifficultyLevel.MEDIUM,
        estimated_time=15
    )


@pytest.fixture
def sample_quiz_cards():
    """Sample quiz cards for testing."""
    return [
        QuizCard(
            paper_id="paper.123",
            pillar_id=PillarID.P2,
            question="What is the key innovation of the Transformer architecture?",
            answer="Multi-head self-attention mechanism that eliminates recurrence",
            difficulty=DifficultyLevel.EASY,
            question_type=QuestionType.CONCEPTUAL
        ),
        QuizCard(
            paper_id="paper.123",
            pillar_id=PillarID.P2,
            question="What is the computational complexity of self-attention?",
            answer="Quadratic with respect to sequence length",
            difficulty=DifficultyLevel.MEDIUM,
            question_type=QuestionType.FACTUAL
        ),
        QuizCard(
            paper_id="paper.123",
            pillar_id=PillarID.P2,
            question="How would you adapt transformers for very long sequences?",
            answer="Use sparse attention patterns or sliding window attention",
            difficulty=DifficultyLevel.HARD,
            question_type=QuestionType.APPLICATION
        ),
        QuizCard(
            paper_id="paper.123",
            pillar_id=PillarID.P2,
            question="What enables parallel processing in transformers?",
            answer="Self-attention allows all positions to be processed simultaneously",
            difficulty=DifficultyLevel.MEDIUM,
            question_type=QuestionType.CONCEPTUAL
        ),
        QuizCard(
            paper_id="paper.123",
            pillar_id=PillarID.P2,
            question="What is positional encoding used for in transformers?",
            answer="To provide sequence position information since there's no inherent order",
            difficulty=DifficultyLevel.EASY,
            question_type=QuestionType.FACTUAL
        )
    ]


class TestOrchestratorHappyPath:
    """Test orchestrator happy path scenarios."""
    
    @patch('nlp_pillars.orchestrator.vectors')
    @patch('nlp_pillars.orchestrator.db')
    @patch('nlp_pillars.orchestrator.QuizAgent')
    @patch('nlp_pillars.orchestrator.SynthesisAgent')
    @patch('nlp_pillars.orchestrator.SummarizerAgent')
    @patch('nlp_pillars.orchestrator.DiscoveryAgent')
    def test_happy_path_with_quiz(
        self,
        mock_discovery_agent,
        mock_summarizer_agent,
        mock_synthesis_agent,
        mock_quiz_agent,
        mock_db,
        mock_vectors,
        sample_paper_refs,
        sample_parsed_paper,
        sample_paper_note,
        sample_lesson,
        sample_quiz_cards
    ):
        """Test successful pipeline execution with quiz generation."""
        # Mock discovery agent
        mock_discovery_output = DiscoveryOutput(
            queries=[
                SearchQuery(pillar_id=PillarID.P2, query="transformer attention mechanisms"),
                SearchQuery(pillar_id=PillarID.P2, query="BERT language models")
            ],
            rationale="Focus on attention and language models"
        )
        mock_discovery_agent.run.return_value = mock_discovery_output
        
        # Mock database operations
        mock_db.get_recent_notes.return_value = []
        mock_db.queue_add_candidates.return_value = 1
        mock_db.queue_pop_next.return_value = [sample_paper_refs[0]]  # Return first paper
        mock_db.upsert_paper.return_value = None
        mock_db.insert_note.return_value = None
        mock_db.insert_lesson.return_value = None
        mock_db.insert_quiz_cards.return_value = None
        mock_db.mark_processed.return_value = None
        
        # Mock agents
        mock_summarizer_agent.run.return_value = sample_paper_note
        mock_synthesis_agent.run.return_value = sample_lesson
        mock_quiz_agent.run.return_value = sample_quiz_cards
        
        # Mock vector operations
        mock_vectors.upsert_text.return_value = 5
        
        # Create orchestrator and mock ingest agent
        orchestrator = Orchestrator(enable_quiz=True)
        orchestrator.ingest_agent.ingest = Mock(return_value=sample_parsed_paper)
        
        # Mock search tools
        orchestrator.searxng_tool.search = Mock(return_value=[sample_paper_refs[0]])
        orchestrator.arxiv_tool.search = Mock(return_value=[sample_paper_refs[1]])
        
        # Run pipeline
        result = orchestrator.run_daily(PillarID.P2, papers_limit=1)
        
        # Verify result structure
        assert isinstance(result, PipelineResult)
        assert result.pillar_id == PillarID.P2
        assert result.success is True
        assert len(result.papers_processed) == 1
        assert result.papers_processed[0] == "paper.123"
        assert len(result.lessons_created) == 1
        assert len(result.quizzes_generated) == 5
        assert len(result.errors) == 0
        assert result.total_time_seconds > 0
        
        # Verify database calls with correct pillar_id
        mock_db.get_recent_notes.assert_called_with(PillarID.P2, limit=5)
        mock_db.queue_add_candidates.assert_called_once()
        queue_call_args = mock_db.queue_add_candidates.call_args
        assert queue_call_args[0][0] == PillarID.P2  # First arg is pillar_id
        
        mock_db.queue_pop_next.assert_called_with(PillarID.P2, limit=1)
        mock_db.upsert_paper.assert_called_with(PillarID.P2, sample_paper_refs[0])
        mock_db.insert_note.assert_called_with(sample_paper_note)
        mock_db.insert_lesson.assert_called_with(sample_lesson)
        mock_db.insert_quiz_cards.assert_called_with(sample_quiz_cards)
        mock_db.mark_processed.assert_called_with(PillarID.P2, "paper.123")
        
        # Verify vector operations
        mock_vectors.upsert_text.assert_called_with(
            PillarID.P2, 
            "paper.123", 
            sample_parsed_paper.full_text
        )
        
        # Verify agents were called
        mock_discovery_agent.run.assert_called_once()
        mock_summarizer_agent.run.assert_called_once()
        mock_synthesis_agent.run.assert_called_once()
        mock_quiz_agent.run.assert_called_once()
    
    @patch('nlp_pillars.orchestrator.vectors')
    @patch('nlp_pillars.orchestrator.db')
    @patch('nlp_pillars.orchestrator.SynthesisAgent')
    @patch('nlp_pillars.orchestrator.SummarizerAgent')
    @patch('nlp_pillars.orchestrator.DiscoveryAgent')
    def test_happy_path_without_quiz(
        self,
        mock_discovery_agent,
        mock_summarizer_agent,
        mock_synthesis_agent,
        mock_db,
        mock_vectors,
        sample_paper_refs,
        sample_parsed_paper,
        sample_paper_note,
        sample_lesson
    ):
        """Test successful pipeline execution without quiz generation."""
        # Mock discovery agent
        mock_discovery_output = DiscoveryOutput(
            queries=[SearchQuery(pillar_id=PillarID.P1, query="test query")],
            rationale="test rationale"
        )
        mock_discovery_agent.run.return_value = mock_discovery_output
        
        # Mock database operations
        mock_db.get_recent_notes.return_value = []
        mock_db.queue_add_candidates.return_value = 1
        mock_db.queue_pop_next.return_value = [sample_paper_refs[0]]
        mock_db.upsert_paper.return_value = None
        mock_db.insert_note.return_value = None
        mock_db.insert_lesson.return_value = None
        mock_db.mark_processed.return_value = None
        
        # Mock agents
        mock_summarizer_agent.run.return_value = sample_paper_note
        mock_synthesis_agent.run.return_value = sample_lesson
        
        # Mock vector operations
        mock_vectors.upsert_text.return_value = 3
        
        # Create orchestrator without quiz
        orchestrator = Orchestrator(enable_quiz=False)
        orchestrator.ingest_agent.ingest = Mock(return_value=sample_parsed_paper)
        orchestrator.searxng_tool.search = Mock(return_value=[])
        orchestrator.arxiv_tool.search = Mock(return_value=[sample_paper_refs[0]])
        
        # Run pipeline
        result = orchestrator.run_daily(PillarID.P1, papers_limit=1)
        
        # Verify result
        assert result.success is True
        assert len(result.papers_processed) == 1
        assert len(result.lessons_created) == 1
        assert len(result.quizzes_generated) == 0  # No quiz cards
        assert len(result.errors) == 0
        
        # Verify quiz operations were not called
        mock_db.insert_quiz_cards.assert_not_called()


class TestOrchestratorErrorHandling:
    """Test orchestrator error handling and resilience."""
    
    @patch('nlp_pillars.orchestrator.vectors')
    @patch('nlp_pillars.orchestrator.db')
    @patch('nlp_pillars.orchestrator.SynthesisAgent')
    @patch('nlp_pillars.orchestrator.SummarizerAgent')
    @patch('nlp_pillars.orchestrator.DiscoveryAgent')
    def test_continue_on_ingest_failure(
        self,
        mock_discovery_agent,
        mock_summarizer_agent,
        mock_synthesis_agent,
        mock_db,
        mock_vectors,
        sample_paper_refs,
        sample_parsed_paper,
        sample_paper_note,
        sample_lesson
    ):
        """Test that pipeline continues when one paper fails ingestion."""
        # Mock discovery agent
        mock_discovery_output = DiscoveryOutput(
            queries=[SearchQuery(pillar_id=PillarID.P3, query="test")],
            rationale="test"
        )
        mock_discovery_agent.run.return_value = mock_discovery_output
        
        # Mock database operations
        mock_db.get_recent_notes.return_value = []
        mock_db.queue_add_candidates.return_value = 2
        mock_db.queue_pop_next.return_value = sample_paper_refs  # Return both papers
        mock_db.upsert_paper.return_value = None
        mock_db.insert_note.return_value = None
        mock_db.insert_lesson.return_value = None
        mock_db.mark_processed.return_value = None
        
        # Mock agents
        mock_summarizer_agent.run.return_value = sample_paper_note
        mock_synthesis_agent.run.return_value = sample_lesson
        mock_vectors.upsert_text.return_value = 3
        
        # Create orchestrator
        orchestrator = Orchestrator(enable_quiz=False)
        orchestrator.searxng_tool.search = Mock(return_value=[])
        orchestrator.arxiv_tool.search = Mock(return_value=sample_paper_refs)
        
        # Mock ingest agent to fail on first paper, succeed on second
        def mock_ingest(paper_ref):
            if paper_ref.id == "paper.123":
                raise Exception("Ingestion failed for first paper")
            return sample_parsed_paper
        
        orchestrator.ingest_agent.ingest = Mock(side_effect=mock_ingest)
        
        # Run pipeline
        result = orchestrator.run_daily(PillarID.P3, papers_limit=2)
        
        # Verify results
        assert result.success is True  # Should succeed because second paper worked
        assert len(result.papers_processed) == 1  # Only second paper processed
        assert result.papers_processed[0] == "paper.456"
        assert len(result.lessons_created) == 1
        assert len(result.errors) == 1  # One error for first paper
        
        # Verify error details
        error = result.errors[0]
        assert error["paper_id"] == "paper.123"
        assert error["step"] == "process_paper"
        assert "Ingestion failed" in error["message"]
    
    @patch('nlp_pillars.orchestrator.vectors')
    @patch('nlp_pillars.orchestrator.db')
    @patch('nlp_pillars.orchestrator.DiscoveryAgent')
    def test_empty_queue_handling(
        self,
        mock_discovery_agent,
        mock_db,
        mock_vectors
    ):
        """Test handling of empty paper queue."""
        # Mock discovery agent
        mock_discovery_output = DiscoveryOutput(
            queries=[SearchQuery(pillar_id=PillarID.P3, query="test")],
            rationale="test"
        )
        mock_discovery_agent.run.return_value = mock_discovery_output
        
        # Mock database operations
        mock_db.get_recent_notes.return_value = []
        mock_db.queue_add_candidates.return_value = 0
        mock_db.queue_pop_next.return_value = []  # Empty queue
        
        # Create orchestrator
        orchestrator = Orchestrator(enable_quiz=True)
        orchestrator.searxng_tool.search = Mock(return_value=[])
        orchestrator.arxiv_tool.search = Mock(return_value=[])
        
        # Run pipeline
        result = orchestrator.run_daily(PillarID.P4, papers_limit=1)
        
        # Verify results
        assert result.success is False  # No papers processed
        assert len(result.papers_processed) == 0
        assert len(result.lessons_created) == 0
        assert len(result.quizzes_generated) == 0
        assert len(result.errors) == 0  # No errors, just no work done
    
    @patch('nlp_pillars.orchestrator.DiscoveryAgent')
    @patch('nlp_pillars.orchestrator.db')
    def test_pipeline_level_failure(self, mock_db, mock_discovery_agent):
        """Test handling of pipeline-level failures."""
        # Mock discovery agent
        mock_discovery_output = DiscoveryOutput(
            queries=[SearchQuery(pillar_id=PillarID.P5, query="test")],
            rationale="test"
        )
        mock_discovery_agent.run.return_value = mock_discovery_output
        
        # Mock database to raise exception on queue_pop_next (which is not in try-catch)
        mock_db.get_recent_notes.return_value = []
        mock_db.queue_add_candidates.return_value = 0
        mock_db.queue_pop_next.side_effect = Exception("Database connection failed")
        
        # Create orchestrator
        orchestrator = Orchestrator(enable_quiz=True)
        
        # Mock search tools to avoid errors there
        orchestrator.searxng_tool.search = Mock(return_value=[])
        orchestrator.arxiv_tool.search = Mock(return_value=[])
        
        # Run pipeline
        result = orchestrator.run_daily(PillarID.P5, papers_limit=1)
        
        # Verify results
        assert result.success is False
        assert len(result.papers_processed) == 0
        assert len(result.errors) == 0  # Errors are logged but pipeline continues gracefully


class TestOrchestratorUtilities:
    """Test orchestrator utility functions."""
    
    def test_dedupe_papers(self):
        """Test paper deduplication logic."""
        # Create duplicates
        papers = [
            PaperRef(id="paper.123", title="Paper A", authors=["Author 1"]),
            PaperRef(id="paper.456", title="Paper B", authors=["Author 2"]),
            PaperRef(id="paper.123", title="Paper A Duplicate", authors=["Author 1"]),  # Duplicate
            PaperRef(id="paper.789", title="Paper C", authors=["Author 3"])
        ]
        
        orchestrator = Orchestrator()
        deduplicated = orchestrator._dedupe_papers(papers)
        
        # Should have 3 unique papers
        assert len(deduplicated) == 3
        paper_ids = {paper.id for paper in deduplicated}
        assert paper_ids == {"paper.123", "paper.456", "paper.789"}
    
    def test_get_pillar_config(self):
        """Test pillar configuration generation."""
        orchestrator = Orchestrator()
        
        # Test different pillars
        for pillar_id in [PillarID.P1, PillarID.P2, PillarID.P3, PillarID.P4, PillarID.P5]:
            config = orchestrator._get_pillar_config(pillar_id)
            
            assert config.id == pillar_id
            assert config.name  # Should have a name
            assert config.goal  # Should have a goal
            assert isinstance(config.focus_areas, list)


class TestPillarIsolation:
    """Test that pillar isolation is enforced throughout the pipeline."""
    
    @patch('nlp_pillars.orchestrator.vectors')
    @patch('nlp_pillars.orchestrator.db')
    @patch('nlp_pillars.orchestrator.QuizAgent')
    @patch('nlp_pillars.orchestrator.SynthesisAgent')
    @patch('nlp_pillars.orchestrator.SummarizerAgent')
    @patch('nlp_pillars.orchestrator.DiscoveryAgent')
    def test_pillar_id_enforcement(
        self,
        mock_discovery_agent,
        mock_summarizer_agent,
        mock_synthesis_agent,
        mock_quiz_agent,
        mock_db,
        mock_vectors,
        sample_paper_refs,
        sample_parsed_paper,
        sample_paper_note,
        sample_lesson,
        sample_quiz_cards
    ):
        """Test that pillar_id is passed to all operations that require it."""
        # Mock discovery agent
        mock_discovery_output = DiscoveryOutput(
            queries=[SearchQuery(pillar_id=PillarID.P3, query="test")],
            rationale="test"
        )
        mock_discovery_agent.run.return_value = mock_discovery_output
        
        # Mock database operations
        mock_db.get_recent_notes.return_value = []
        mock_db.queue_add_candidates.return_value = 1
        mock_db.queue_pop_next.return_value = [sample_paper_refs[0]]
        mock_db.upsert_paper.return_value = None
        mock_db.insert_note.return_value = None
        mock_db.insert_lesson.return_value = None
        mock_db.insert_quiz_cards.return_value = None
        mock_db.mark_processed.return_value = None
        
        # Mock agents
        mock_summarizer_agent.run.return_value = sample_paper_note
        mock_synthesis_agent.run.return_value = sample_lesson
        mock_quiz_agent.run.return_value = sample_quiz_cards
        mock_vectors.upsert_text.return_value = 5
        
        # Create orchestrator
        orchestrator = Orchestrator(enable_quiz=True)
        orchestrator.ingest_agent.ingest = Mock(return_value=sample_parsed_paper)
        orchestrator.searxng_tool.search = Mock(return_value=[])
        orchestrator.arxiv_tool.search = Mock(return_value=[sample_paper_refs[0]])
        
        # Run pipeline
        test_pillar = PillarID.P3
        result = orchestrator.run_daily(test_pillar, papers_limit=1)
        
        # Verify all database operations include correct pillar_id
        mock_db.get_recent_notes.assert_called_with(test_pillar, limit=5)
        
        queue_add_call = mock_db.queue_add_candidates.call_args
        assert queue_add_call[0][0] == test_pillar
        
        queue_pop_call = mock_db.queue_pop_next.call_args
        assert queue_pop_call[0][0] == test_pillar
        assert queue_pop_call[1]['limit'] == 1
        
        upsert_call = mock_db.upsert_paper.call_args
        assert upsert_call[0][0] == test_pillar
        
        mark_processed_call = mock_db.mark_processed.call_args
        assert mark_processed_call[0][0] == test_pillar
        assert mark_processed_call[0][1] == "paper.123"
        
        # Verify vector operations include correct pillar_id
        vector_call = mock_vectors.upsert_text.call_args
        assert vector_call[0][0] == test_pillar
        assert vector_call[0][1] == "paper.123"
        
        # Verify result has correct pillar_id
        assert result.pillar_id == test_pillar


class TestLoggingAndTiming:
    """Test logging and timing functionality."""
    
    @patch('nlp_pillars.orchestrator.vectors')
    @patch('nlp_pillars.orchestrator.db')
    @patch('nlp_pillars.orchestrator.SynthesisAgent')
    @patch('nlp_pillars.orchestrator.SummarizerAgent')
    @patch('nlp_pillars.orchestrator.DiscoveryAgent')
    def test_logging_includes_pillar_and_paper_ids(
        self,
        mock_discovery_agent,
        mock_summarizer_agent,
        mock_synthesis_agent,
        mock_db,
        mock_vectors,
        sample_paper_refs,
        sample_parsed_paper,
        sample_paper_note,
        sample_lesson,
        caplog
    ):
        """Test that logs include pillar_id and paper_id at key steps."""
        # Mock discovery agent
        mock_discovery_output = DiscoveryOutput(
            queries=[SearchQuery(pillar_id=PillarID.P3, query="test")],
            rationale="test"
        )
        mock_discovery_agent.run.return_value = mock_discovery_output
        
        # Mock database operations
        mock_db.get_recent_notes.return_value = []
        mock_db.queue_add_candidates.return_value = 1
        mock_db.queue_pop_next.return_value = [sample_paper_refs[0]]
        mock_db.upsert_paper.return_value = None
        mock_db.insert_note.return_value = None
        mock_db.insert_lesson.return_value = None
        mock_db.mark_processed.return_value = None
        
        # Mock agents
        mock_summarizer_agent.run.return_value = sample_paper_note
        mock_synthesis_agent.run.return_value = sample_lesson
        mock_vectors.upsert_text.return_value = 3
        
        # Create orchestrator
        orchestrator = Orchestrator(enable_quiz=False)
        orchestrator.ingest_agent.ingest = Mock(return_value=sample_parsed_paper)
        orchestrator.searxng_tool.search = Mock(return_value=[])
        orchestrator.arxiv_tool.search = Mock(return_value=[sample_paper_refs[0]])
        
        # Run pipeline with logging capture
        with caplog.at_level("INFO"):
            result = orchestrator.run_daily(PillarID.P2, papers_limit=1)
        
        # Check that logs contain pillar_id
        log_text = caplog.text
        assert "pillar P2" in log_text
        assert "paper.123" in log_text
        
        # Check for key pipeline steps in logs
        assert "Starting daily pipeline" in log_text
        assert "Step 1: Discovery" in log_text
        assert "Step 2: Searching" in log_text
        assert "Step 3: Enqueueing" in log_text
        assert "Step 4: Popping" in log_text
        assert "Step 5: Processing" in log_text
        assert "Pipeline completed" in log_text
    
    @patch('nlp_pillars.orchestrator.time.time')
    @patch('nlp_pillars.orchestrator.vectors')
    @patch('nlp_pillars.orchestrator.db')
    @patch('nlp_pillars.orchestrator.DiscoveryAgent')
    def test_timing_calculation(
        self,
        mock_discovery_agent,
        mock_db,
        mock_vectors,
        mock_time
    ):
        """Test that timing is calculated correctly."""
        # Mock time progression
        mock_time.side_effect = [1000.0, 1005.5]  # Start and end times
        
        # Mock discovery agent
        mock_discovery_output = DiscoveryOutput(
            queries=[SearchQuery(pillar_id=PillarID.P3, query="test")],
            rationale="test"
        )
        mock_discovery_agent.run.return_value = mock_discovery_output
        
        # Mock database operations
        mock_db.get_recent_notes.return_value = []
        mock_db.queue_add_candidates.return_value = 0
        mock_db.queue_pop_next.return_value = []  # Empty queue
        
        # Create orchestrator
        orchestrator = Orchestrator()
        orchestrator.searxng_tool.search = Mock(return_value=[])
        orchestrator.arxiv_tool.search = Mock(return_value=[])
        
        # Run pipeline
        result = orchestrator.run_daily(PillarID.P1, papers_limit=1)
        
        # Verify timing
        assert result.total_time_seconds == 5.5  # 1005.5 - 1000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
