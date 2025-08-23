"""
End-to-end smoke tests for the NLP Learning Workflow orchestrator.

Tests the complete pipeline with full mocking to ensure all components
integrate correctly without any network calls or external dependencies.
"""

import pytest
import sys
import subprocess
from unittest.mock import Mock
from datetime import datetime

from nlp_pillars.schemas import (
    PillarID, PaperRef, PaperNote, Lesson, QuizCard, SearchQuery,
    DiscoveryOutput, ParsedPaper, DifficultyLevel, QuestionType
)
from nlp_pillars.orchestrator import Orchestrator


@pytest.fixture
def fake_data():
    """Create deterministic fake data for all components."""
    
    # Fake PaperRef
    fake_paper_ref = PaperRef(
        id="fake.001",
        title="Smoke Test Paper: Advanced Neural Architecture Analysis",
        authors=["Smoke Tester", "Mock Researcher"],
        url="https://example.com/fake.001.pdf",
        publication_date="2024-01-15",
        abstract="This is a fake abstract for smoke testing the NLP workflow pipeline."
    )
    
    # Fake ParsedPaper
    fake_parsed_paper = ParsedPaper(
        paper_ref=fake_paper_ref,
        full_text="This is a smoke test full text for summarization. It contains multiple sentences that discuss advanced neural architectures, attention mechanisms, and transformer models. The content is designed to be processed by the summarization agent.",
        chunks=["This is a smoke test full text for summarization. It contains multiple sentences that discuss advanced neural architectures, attention mechanisms, and transformer models."],
        figures_count=0,
        tables_count=0,
        references=[]
    )
    
    # Fake PaperNote
    fake_paper_note = PaperNote(
        paper_id="fake.001",
        pillar_id=PillarID.P2,
        problem="Smoke test problem: evaluating neural architecture efficiency",
        method="Smoke test method: comparative analysis framework",
        findings=["Finding 1: architecture performs well", "Finding 2: efficiency gains observed"],
        limitations=["Limitation 1: small dataset", "Limitation 2: specific domain"],
        future_work=["Future work 1: larger experiments", "Future work 2: cross-domain validation"],
        key_terms=["neural architecture", "efficiency", "transformer"],
        confidence_score=0.85
    )
    
    # Fake Lesson
    fake_lesson = Lesson(
        paper_id="fake.001",
        pillar_id=PillarID.P2,
        tl_dr="TL;DR: Smoke test OK.",
        takeaways=[
            "Takeaway 1: Neural architectures can be optimized efficiently",
            "Takeaway 2: Attention mechanisms improve performance",
            "Takeaway 3: Transformer models show promising results"
        ],
        practice_ideas=[
            "Practice 1: Implement attention mechanism variants",
            "Practice 2: Compare architecture efficiency metrics"
        ],
        connections=[],
        difficulty=DifficultyLevel.MEDIUM,
        estimated_time=15
    )
    
    # Fake QuizCards
    fake_quiz_cards = [
        QuizCard(
            paper_id="fake.001",
            pillar_id=PillarID.P2,
            question="What is the main contribution of this smoke test paper?",
            answer="Advanced neural architecture analysis with efficiency evaluation",
            difficulty=DifficultyLevel.EASY,
            question_type=QuestionType.FACTUAL
        ),
        QuizCard(
            paper_id="fake.001",
            pillar_id=PillarID.P2,
            question="How do attention mechanisms improve model performance?",
            answer="By allowing selective focus on relevant input features",
            difficulty=DifficultyLevel.MEDIUM,
            question_type=QuestionType.CONCEPTUAL
        ),
        QuizCard(
            paper_id="fake.001",
            pillar_id=PillarID.P2,
            question="Compare transformer efficiency with traditional architectures",
            answer="Transformers show better parallelization and context modeling",
            difficulty=DifficultyLevel.HARD,
            question_type=QuestionType.APPLICATION
        ),
        QuizCard(
            paper_id="fake.001",
            pillar_id=PillarID.P2,
            question="What are the key limitations mentioned in this study?",
            answer="Small dataset size and domain-specific constraints",
            difficulty=DifficultyLevel.MEDIUM,
            question_type=QuestionType.FACTUAL
        ),
        QuizCard(
            paper_id="fake.001",
            pillar_id=PillarID.P2,
            question="Design an experiment to validate the architecture claims",
            answer="Cross-domain evaluation with larger datasets and baseline comparisons",
            difficulty=DifficultyLevel.HARD,
            question_type=QuestionType.APPLICATION
        )
    ]
    
    # Fake DiscoveryOutput
    fake_discovery_output = DiscoveryOutput(
        queries=[
            SearchQuery(pillar_id=PillarID.P2, query="neural architecture efficiency"),
            SearchQuery(pillar_id=PillarID.P2, query="transformer attention mechanisms")
        ],
        rationale="Focus on architecture analysis and attention mechanisms for P2"
    )
    
    return {
        'paper_ref': fake_paper_ref,
        'parsed_paper': fake_parsed_paper,
        'paper_note': fake_paper_note,
        'lesson': fake_lesson,
        'quiz_cards': fake_quiz_cards,
        'discovery_output': fake_discovery_output
    }


def test_orchestrator_end_to_end_smoke(monkeypatch, fake_data):
    """Test the complete orchestrator pipeline with full mocking."""
    
    # Mock all the agent classes
    mock_discovery_agent_class = Mock()
    mock_discovery_agent_class.run.return_value = fake_data['discovery_output']
    monkeypatch.setattr('nlp_pillars.orchestrator.DiscoveryAgent', mock_discovery_agent_class)
    
    mock_summarizer_agent_class = Mock()
    mock_summarizer_agent_class.run.return_value = fake_data['paper_note']
    monkeypatch.setattr('nlp_pillars.orchestrator.SummarizerAgent', mock_summarizer_agent_class)
    
    mock_synthesis_agent_class = Mock()
    mock_synthesis_agent_class.run.return_value = fake_data['lesson']
    monkeypatch.setattr('nlp_pillars.orchestrator.SynthesisAgent', mock_synthesis_agent_class)
    
    mock_quiz_agent_class = Mock()
    mock_quiz_agent_class.run.return_value = fake_data['quiz_cards']
    monkeypatch.setattr('nlp_pillars.orchestrator.QuizAgent', mock_quiz_agent_class)
    
    # Mock database operations
    mock_db = Mock()
    mock_db.get_recent_notes.return_value = []
    mock_db.queue_add_candidates.return_value = 1
    mock_db.queue_pop_next.return_value = [fake_data['paper_ref']]
    mock_db.upsert_paper.return_value = None
    mock_db.insert_note.return_value = None
    mock_db.insert_lesson.return_value = None
    mock_db.insert_quiz_cards.return_value = None
    mock_db.mark_processed.return_value = None
    monkeypatch.setattr('nlp_pillars.orchestrator.db', mock_db)
    
    # Mock vector operations
    mock_vectors = Mock()
    mock_vectors.upsert_text.return_value = 1
    monkeypatch.setattr('nlp_pillars.orchestrator.vectors', mock_vectors)
    
    # Mock the Orchestrator's __init__ to avoid creating real tools
    original_init = Orchestrator.__init__
    
    def mock_init(self, enable_quiz=True):
        self.enable_quiz = enable_quiz
        # Mock the search tools
        self.searxng_tool = Mock()
        self.searxng_tool.search.return_value = [fake_data['paper_ref']]
        
        self.arxiv_tool = Mock()
        self.arxiv_tool.search.return_value = []
        
        # Mock the ingest agent
        self.ingest_agent = Mock()
        self.ingest_agent.ingest.return_value = fake_data['parsed_paper']
    
    monkeypatch.setattr(Orchestrator, '__init__', mock_init)
    
    try:
        # Create and run orchestrator
        orchestrator = Orchestrator(enable_quiz=True)
        result = orchestrator.run_daily(PillarID.P2, papers_limit=1)
        
        # Verify results
        assert result.success is True, "Pipeline should succeed"
        assert len(result.papers_processed) == 1, "Should process exactly 1 paper"
        assert len(result.lessons_created) == 1, "Should create exactly 1 lesson"
        assert len(result.quizzes_generated) == 5, "Should generate exactly 5 quiz cards"
        assert result.lessons_created[0].tl_dr == "TL;DR: Smoke test OK.", "Should have correct TL;DR"
        assert len(result.errors) == 0, "Should have no errors"
        
        # Verify agent calls
        mock_discovery_agent_class.run.assert_called_once()
        mock_summarizer_agent_class.run.assert_called_once()
        mock_synthesis_agent_class.run.assert_called_once()
        mock_quiz_agent_class.run.assert_called_once()
        
        # Verify database calls
        mock_db.get_recent_notes.assert_called()
        mock_db.queue_add_candidates.assert_called_once()
        mock_db.queue_pop_next.assert_called_once()
        mock_db.upsert_paper.assert_called_once()
        mock_db.insert_note.assert_called_once()
        mock_db.insert_lesson.assert_called_once()
        mock_db.insert_quiz_cards.assert_called_once()
        mock_db.mark_processed.assert_called_once()
        
        # Verify vector calls
        mock_vectors.upsert_text.assert_called_once()
        
        # Verify search tool calls
        orchestrator.searxng_tool.search.assert_called()
        orchestrator.arxiv_tool.search.assert_called()
        orchestrator.ingest_agent.ingest.assert_called_once()
        
    finally:
        # Restore original __init__
        monkeypatch.setattr(Orchestrator, '__init__', original_init)


def test_smoke_script_execution():
    """Test that the smoke script runs successfully and produces expected output."""
    
    # Run the smoke script
    result = subprocess.run(
        [sys.executable, "scripts/smoke_local.py"],
        capture_output=True,
        text=True,
        cwd="."
    )
    
    # Verify exit code
    assert result.returncode == 0, f"Smoke script should exit successfully, got: {result.stderr}"
    
    # Verify expected output
    stdout = result.stdout
    assert "TL;DR: TL;DR: Smoke test OK." in stdout, "Should contain correct TL;DR"
    assert "Quiz cards generated: 5" in stdout, "Should generate 5 quiz cards"
    assert "Papers processed: 1" in stdout, "Should process 1 paper"
    assert "Success: True" in stdout, "Should be successful"
    assert "Errors: 0" in stdout, "Should have no errors"
    assert "Smoke test completed!" in stdout, "Should complete successfully"


def test_orchestrator_with_quiz_disabled(monkeypatch, fake_data):
    """Test orchestrator with quiz generation disabled."""
    
    # Mock all the agent classes (same as before but no quiz agent)
    mock_discovery_agent_class = Mock()
    mock_discovery_agent_class.run.return_value = fake_data['discovery_output']
    monkeypatch.setattr('nlp_pillars.orchestrator.DiscoveryAgent', mock_discovery_agent_class)
    
    mock_summarizer_agent_class = Mock()
    mock_summarizer_agent_class.run.return_value = fake_data['paper_note']
    monkeypatch.setattr('nlp_pillars.orchestrator.SummarizerAgent', mock_summarizer_agent_class)
    
    mock_synthesis_agent_class = Mock()
    mock_synthesis_agent_class.run.return_value = fake_data['lesson']
    monkeypatch.setattr('nlp_pillars.orchestrator.SynthesisAgent', mock_synthesis_agent_class)
    
    mock_quiz_agent_class = Mock()
    monkeypatch.setattr('nlp_pillars.orchestrator.QuizAgent', mock_quiz_agent_class)
    
    # Mock database and vector operations (same as before)
    mock_db = Mock()
    mock_db.get_recent_notes.return_value = []
    mock_db.queue_add_candidates.return_value = 1
    mock_db.queue_pop_next.return_value = [fake_data['paper_ref']]
    mock_db.upsert_paper.return_value = None
    mock_db.insert_note.return_value = None
    mock_db.insert_lesson.return_value = None
    mock_db.mark_processed.return_value = None
    monkeypatch.setattr('nlp_pillars.orchestrator.db', mock_db)
    
    mock_vectors = Mock()
    mock_vectors.upsert_text.return_value = 1
    monkeypatch.setattr('nlp_pillars.orchestrator.vectors', mock_vectors)
    
    # Mock the Orchestrator's __init__
    def mock_init(self, enable_quiz=True):
        self.enable_quiz = enable_quiz
        self.searxng_tool = Mock()
        self.searxng_tool.search.return_value = [fake_data['paper_ref']]
        self.arxiv_tool = Mock()
        self.arxiv_tool.search.return_value = []
        self.ingest_agent = Mock()
        self.ingest_agent.ingest.return_value = fake_data['parsed_paper']
    
    original_init = Orchestrator.__init__
    monkeypatch.setattr(Orchestrator, '__init__', mock_init)
    
    try:
        # Create and run orchestrator with quiz disabled
        orchestrator = Orchestrator(enable_quiz=False)
        result = orchestrator.run_daily(PillarID.P2, papers_limit=1)
        
        # Verify results
        assert result.success is True, "Pipeline should succeed"
        assert len(result.papers_processed) == 1, "Should process exactly 1 paper"
        assert len(result.lessons_created) == 1, "Should create exactly 1 lesson"
        assert len(result.quizzes_generated) == 0, "Should generate no quiz cards when disabled"
        assert len(result.errors) == 0, "Should have no errors"
        
        # Verify quiz agent was NOT called
        mock_quiz_agent_class.run.assert_not_called()
        
        # Verify quiz cards were NOT inserted
        mock_db.insert_quiz_cards.assert_not_called()
        
    finally:
        monkeypatch.setattr(Orchestrator, '__init__', original_init)


def test_orchestrator_error_handling(monkeypatch, fake_data):
    """Test orchestrator error handling when a paper fails to process."""
    
    # Mock agents with one failing
    mock_discovery_agent_class = Mock()
    mock_discovery_agent_class.run.return_value = fake_data['discovery_output']
    monkeypatch.setattr('nlp_pillars.orchestrator.DiscoveryAgent', mock_discovery_agent_class)
    
    # Make summarizer agent fail
    mock_summarizer_agent_class = Mock()
    mock_summarizer_agent_class.run.side_effect = Exception("Summarizer failed")
    monkeypatch.setattr('nlp_pillars.orchestrator.SummarizerAgent', mock_summarizer_agent_class)
    
    # Mock database and vector operations
    mock_db = Mock()
    mock_db.get_recent_notes.return_value = []
    mock_db.queue_add_candidates.return_value = 1
    mock_db.queue_pop_next.return_value = [fake_data['paper_ref']]
    monkeypatch.setattr('nlp_pillars.orchestrator.db', mock_db)
    
    mock_vectors = Mock()
    monkeypatch.setattr('nlp_pillars.orchestrator.vectors', mock_vectors)
    
    # Mock the Orchestrator's __init__
    def mock_init(self, enable_quiz=True):
        self.enable_quiz = enable_quiz
        self.searxng_tool = Mock()
        self.searxng_tool.search.return_value = [fake_data['paper_ref']]
        self.arxiv_tool = Mock()
        self.arxiv_tool.search.return_value = []
        self.ingest_agent = Mock()
        self.ingest_agent.ingest.return_value = fake_data['parsed_paper']
    
    original_init = Orchestrator.__init__
    monkeypatch.setattr(Orchestrator, '__init__', mock_init)
    
    try:
        # Create and run orchestrator
        orchestrator = Orchestrator(enable_quiz=True)
        result = orchestrator.run_daily(PillarID.P2, papers_limit=1)
        
        # Verify error handling
        assert result.success is False, "Pipeline should fail when paper processing fails"
        assert len(result.papers_processed) == 0, "Should process no papers when summarizer fails"
        assert len(result.lessons_created) == 0, "Should create no lessons when processing fails"
        assert len(result.quizzes_generated) == 0, "Should generate no quiz cards when processing fails"
        assert len(result.errors) == 1, "Should have one error"
        assert "Summarizer failed" in str(result.errors[0]), "Should capture the summarizer error"
        
    finally:
        monkeypatch.setattr(Orchestrator, '__init__', original_init)
