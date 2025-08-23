#!/usr/bin/env python3
"""
Local smoke test for the NLP Learning Workflow orchestrator.

Exercises the complete pipeline end-to-end with deterministic mocks,
no network calls or external dependencies.
"""

import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime, timezone

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import schemas and orchestrator
from nlp_pillars.schemas import (
    PillarID, PaperRef, PaperNote, Lesson, QuizCard, SearchQuery,
    DiscoveryOutput, ParsedPaper, DifficultyLevel, QuestionType
)
from nlp_pillars.orchestrator import Orchestrator


def create_fake_data():
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


def run_smoke_test():
    """Run the smoke test with full mocking."""
    
    print("🔬 Starting NLP Workflow Smoke Test...")
    
    # Create fake data
    fake_data = create_fake_data()
    
    try:
        # Apply all the mocks
        with patch('nlp_pillars.orchestrator.DiscoveryAgent') as mock_discovery_agent_class, \
             patch('nlp_pillars.orchestrator.SummarizerAgent') as mock_summarizer_agent_class, \
             patch('nlp_pillars.orchestrator.SynthesisAgent') as mock_synthesis_agent_class, \
             patch('nlp_pillars.orchestrator.QuizAgent') as mock_quiz_agent_class, \
             patch('nlp_pillars.orchestrator.db') as mock_db, \
             patch('nlp_pillars.orchestrator.vectors') as mock_vectors:
            
            # Mock agent class methods (static methods)
            mock_discovery_agent_class.run.return_value = fake_data['discovery_output']
            mock_summarizer_agent_class.run.return_value = fake_data['paper_note']
            mock_synthesis_agent_class.run.return_value = fake_data['lesson']
            mock_quiz_agent_class.run.return_value = fake_data['quiz_cards']
            
            # Mock database operations
            mock_db.get_recent_notes.return_value = []
            mock_db.queue_add_candidates.return_value = 1
            mock_db.queue_pop_next.return_value = [fake_data['paper_ref']]
            mock_db.upsert_paper.return_value = None
            mock_db.insert_note.return_value = None
            mock_db.insert_lesson.return_value = None
            mock_db.insert_quiz_cards.return_value = None
            mock_db.mark_processed.return_value = None
            
            # Mock vector operations
            mock_vectors.upsert_text.return_value = 1
            
            # Create orchestrator with search tool mocks
            with patch.object(Orchestrator, '__init__') as mock_init:
                # Mock the init to not create real tools
                mock_init.return_value = None
                
                # Create orchestrator instance
                orchestrator = Orchestrator(enable_quiz=True)
                
                # Manually set the enable_quiz attribute
                orchestrator.enable_quiz = True
                
                # Mock the search tools
                orchestrator.searxng_tool = Mock()
                orchestrator.searxng_tool.search.return_value = [fake_data['paper_ref']]
                
                orchestrator.arxiv_tool = Mock()
                orchestrator.arxiv_tool.search.return_value = []
                
                # Mock the ingest agent
                orchestrator.ingest_agent = Mock()
                orchestrator.ingest_agent.ingest.return_value = fake_data['parsed_paper']
                
                print("🎯 Running orchestrator for Pillar P2...")
                
                # Run the orchestrator
                result = orchestrator.run_daily(PillarID.P2, papers_limit=1)
                
                # Print results
                if result.lessons_created:
                    print(f"TL;DR: {result.lessons_created[0].tl_dr}")
                else:
                    print("TL;DR: No lessons created")
                
                print(f"Quiz cards generated: {len(result.quizzes_generated)}")
                
                # Print additional smoke test info
                print(f"✅ Papers processed: {len(result.papers_processed)}")
                print(f"✅ Success: {result.success}")
                print(f"✅ Total time: {result.total_time_seconds:.2f}s")
                print(f"✅ Errors: {len(result.errors)}")
                
                return 0
                
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = run_smoke_test()
    print("🏁 Smoke test completed!")
    sys.exit(exit_code)
