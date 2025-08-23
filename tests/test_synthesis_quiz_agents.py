"""
Comprehensive tests for SynthesisAgent and QuizAgent using Atomic Agents v2.0 + Instructor.
All OpenAI calls are mocked for fast, reliable testing.
"""

import pytest
import logging
from unittest.mock import Mock, patch
from pydantic import ValidationError

from nlp_pillars.schemas import (
    SynthesisInput, Lesson, QuizGeneratorInput, QuizCard,
    PaperNote, PaperRef, PillarConfig, PillarID,
    DifficultyLevel, QuestionType
)
from nlp_pillars.agents.synthesis_agent import (
    SynthesisAgentImpl, SynthesisValidationError, synthesize
)
from nlp_pillars.agents.quiz_agent import (
    QuizAgentImpl, QuizValidationError, generate_quiz
)


# Test fixtures
@pytest.fixture
def sample_paper_note():
    """Sample paper note for testing."""
    return PaperNote(
        paper_id="test.12345",
        pillar_id=PillarID.P2,
        problem="Limited effectiveness of traditional attention mechanisms in sequence modeling",
        method="Multi-head attention with scaled dot-product computation",
        findings=[
            "Achieved 28.4 BLEU score on WMT 2014 EN-DE translation",
            "Reduced training time by 75% compared to RNN models",
            "Demonstrated better parallelization capabilities"
        ],
        limitations=[
            "High memory requirements for long sequences",
            "Quadratic complexity with sequence length",
            "Requires large amounts of training data"
        ],
        future_work=[
            "Investigate linear attention mechanisms",
            "Explore sparse attention patterns",
            "Apply to other sequence tasks"
        ],
        key_terms=[
            "multi-head attention",
            "transformer",
            "self-attention",
            "positional encoding",
            "scaled dot-product"
        ],
        confidence_score=0.9
    )


@pytest.fixture
def sample_pillar_config():
    """Sample pillar configuration for testing."""
    return PillarConfig(
        id=PillarID.P2,  # This is the required field name
        name="Models & Architectures",
        goal="Master modern neural network architectures and understand their design principles",
        focus_areas=[
            "Attention mechanisms",
            "Transformer architectures",
            "Model optimization",
            "Architectural innovations"
        ]
    )


@pytest.fixture
def sample_synthesis_input(sample_paper_note, sample_pillar_config):
    """Sample SynthesisInput for testing."""
    return SynthesisInput(
        paper_note=sample_paper_note,
        pillar_config=sample_pillar_config,
        related_lessons=[]
    )


@pytest.fixture
def valid_lesson():
    """Valid Lesson response for testing."""
    return Lesson(
        paper_id="test.12345",
        pillar_id=PillarID.P2,
        tl_dr="Transformers revolutionized sequence modeling by using attention mechanisms instead of recurrence.",
        takeaways=[
            "Multi-head attention allows parallel processing of sequences",
            "Self-attention captures long-range dependencies effectively",
            "Positional encoding is crucial for sequence order understanding",
            "Transformer architecture scales better than RNNs"
        ],
        practice_ideas=[
            "Implement a mini-transformer for text classification",
            "Experiment with different positional encoding schemes",
            "Compare attention patterns across different tasks"
        ],
        connections=[
            "BERT uses transformer encoder architecture",
            "GPT applies transformer decoder for generation"
        ],
        difficulty=DifficultyLevel.MEDIUM,
        estimated_time=15
    )


@pytest.fixture
def sample_quiz_input(sample_paper_note):
    """Sample QuizGeneratorInput for testing."""
    return QuizGeneratorInput(
        paper_note=sample_paper_note,
        num_questions=5,
        difficulty_mix={"easy": 2, "medium": 2, "hard": 1}
    )


@pytest.fixture
def valid_quiz_cards():
    """Valid QuizCard list response for testing."""
    return [
        QuizCard(
            id="test.12345_q1",
            paper_id="test.12345",
            pillar_id=PillarID.P2,
            question="What BLEU score did the model achieve on WMT 2014 EN-DE translation?",
            answer="28.4 BLEU score",
            difficulty=DifficultyLevel.EASY,
            question_type=QuestionType.FACTUAL
        ),
        QuizCard(
            id="test.12345_q2",
            paper_id="test.12345",
            pillar_id=PillarID.P2,
            question="How does multi-head attention improve upon single attention mechanisms?",
            answer="Multi-head attention allows the model to jointly attend to information from different representation subspaces, capturing various types of relationships.",
            difficulty=DifficultyLevel.EASY,
            question_type=QuestionType.CONCEPTUAL
        ),
        QuizCard(
            id="test.12345_q3",
            paper_id="test.12345",
            pillar_id=PillarID.P2,
            question="What is a key limitation of transformers for very long sequences?",
            answer="Quadratic complexity with sequence length due to the attention mechanism.",
            difficulty=DifficultyLevel.MEDIUM,
            question_type=QuestionType.CONCEPTUAL
        ),
        QuizCard(
            id="test.12345_q4",
            paper_id="test.12345",
            pillar_id=PillarID.P2,
            question="How would you modify the transformer to handle sequences longer than the training context?",
            answer="Implement sparse attention patterns or use techniques like sliding window attention to reduce computational complexity.",
            difficulty=DifficultyLevel.MEDIUM,
            question_type=QuestionType.APPLICATION
        ),
        QuizCard(
            id="test.12345_q5",
            paper_id="test.12345",
            pillar_id=PillarID.P2,
            question="Design an experiment to compare transformer attention patterns across different linguistic tasks.",
            answer="Train transformers on tasks like translation, summarization, and question answering, then visualize and analyze attention heads for syntactic vs semantic patterns.",
            difficulty=DifficultyLevel.HARD,
            question_type=QuestionType.APPLICATION
        )
    ]


@pytest.fixture
def mock_instructor_client():
    """Mock Instructor client for testing."""
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    return mock_client


class TestSynthesisAgent:
    """Test cases for SynthesisAgent."""
    
    def test_synthesis_happy_path(self, mock_instructor_client, sample_synthesis_input, valid_lesson):
        """Test successful synthesis on first attempt."""
        # Mock successful response
        mock_instructor_client.chat.completions.create.return_value = valid_lesson
        
        # Create agent
        agent = SynthesisAgentImpl(mock_instructor_client, "gpt-4")
        
        # Run synthesis
        with patch('nlp_pillars.agents.synthesis_agent.logger') as mock_logger:
            result = agent.run(sample_synthesis_input)
        
        # Verify result
        assert isinstance(result, Lesson)
        assert result.paper_id == "test.12345"
        assert result.pillar_id == PillarID.P2
        assert result.tl_dr == valid_lesson.tl_dr
        assert len(result.takeaways) >= 3 and len(result.takeaways) <= 5
        assert len(result.practice_ideas) >= 2 and len(result.practice_ideas) <= 3
        assert isinstance(result.connections, list)
        
        # Verify logging
        mock_logger.info.assert_any_call("Starting synthesis for paper: test.12345")
        mock_logger.info.assert_any_call("Synthesis completed successfully on first attempt")
        
        # Verify client was called correctly
        mock_instructor_client.chat.completions.create.assert_called_once()
        call_args = mock_instructor_client.chat.completions.create.call_args
        assert call_args[1]['model'] == "gpt-4"
        assert call_args[1]['response_model'] == Lesson
        assert call_args[1]['temperature'] == 0.2
    
    def test_synthesis_retry_on_validation_error(self, mock_instructor_client, sample_synthesis_input, valid_lesson):
        """Test retry logic when first attempt fails validation."""
        # Mock first call to fail, second to succeed
        validation_error = ValidationError.from_exception_data(
            "ValidationError", 
            [{"type": "missing", "loc": ("tl_dr",), "msg": "Field required"}]
        )
        mock_instructor_client.chat.completions.create.side_effect = [
            validation_error,
            valid_lesson
        ]
        
        agent = SynthesisAgentImpl(mock_instructor_client, "gpt-4")
        
        with patch('nlp_pillars.agents.synthesis_agent.logger') as mock_logger:
            result = agent.run(sample_synthesis_input)
        
        # Verify successful result after retry
        assert isinstance(result, Lesson)
        assert result.paper_id == "test.12345"
        
        # Verify retry logging
        mock_logger.warning.assert_called_once()
        mock_logger.info.assert_any_call("Attempting retry with corrective message")
        mock_logger.info.assert_any_call("Synthesis completed successfully on retry")
        
        # Verify client was called twice
        assert mock_instructor_client.chat.completions.create.call_count == 2
        
        # Check that retry included corrective suffix
        retry_call_args = mock_instructor_client.chat.completions.create.call_args_list[1]
        retry_message = retry_call_args[1]['messages'][1]['content']
        assert "Your last output was invalid JSON for Lesson" in retry_message
    
    def test_synthesis_fails_after_retry(self, mock_instructor_client, sample_synthesis_input):
        """Test that SynthesisValidationError is raised when both attempts fail."""
        # Mock both calls to fail validation
        validation_error1 = ValidationError.from_exception_data(
            "ValidationError", 
            [{"type": "missing", "loc": ("tl_dr",), "msg": "Field required"}]
        )
        validation_error2 = ValidationError.from_exception_data(
            "ValidationError", 
            [{"type": "missing", "loc": ("takeaways",), "msg": "Field required"}]
        )
        mock_instructor_client.chat.completions.create.side_effect = [
            validation_error1,
            validation_error2
        ]
        
        agent = SynthesisAgentImpl(mock_instructor_client, "gpt-4")
        
        with patch('nlp_pillars.agents.synthesis_agent.logger') as mock_logger:
            with pytest.raises(SynthesisValidationError) as exc_info:
                agent.run(sample_synthesis_input)
        
        # Verify error message contains both validation errors
        error_message = str(exc_info.value)
        assert "Failed to generate valid Lesson after retry" in error_message
        assert "Original error:" in error_message
        assert "Retry error:" in error_message
        
        # Verify error logging
        mock_logger.error.assert_called_once()
        
        # Verify both calls were made
        assert mock_instructor_client.chat.completions.create.call_count == 2


class TestQuizAgent:
    """Test cases for QuizAgent."""
    
    def test_quiz_happy_path_with_difficulty_mix(self, mock_instructor_client, sample_quiz_input, valid_quiz_cards):
        """Test successful quiz generation with proper difficulty distribution."""
        # Mock successful response
        mock_instructor_client.chat.completions.create.return_value = valid_quiz_cards
        
        # Create agent
        agent = QuizAgentImpl(mock_instructor_client, "gpt-4")
        
        # Run quiz generation
        with patch('nlp_pillars.agents.quiz_agent.logger') as mock_logger:
            result = agent.run(sample_quiz_input)
        
        # Verify result structure
        assert isinstance(result, list)
        assert len(result) == 5  # Exactly num_questions
        assert all(isinstance(card, QuizCard) for card in result)
        
        # Verify all cards have required fields
        for card in result:
            assert card.paper_id == "test.12345"
            assert card.pillar_id == PillarID.P2
            assert card.question
            assert card.answer
            assert card.difficulty in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
            assert card.question_type in [QuestionType.FACTUAL, QuestionType.CONCEPTUAL, QuestionType.APPLICATION]
        
        # Verify difficulty mix (should match requested: 2 easy, 2 medium, 1 hard)
        difficulty_counts = {}
        for card in result:
            difficulty_counts[card.difficulty] = difficulty_counts.get(card.difficulty, 0) + 1
        
        assert difficulty_counts.get(DifficultyLevel.EASY, 0) == 2
        assert difficulty_counts.get(DifficultyLevel.MEDIUM, 0) == 2
        assert difficulty_counts.get(DifficultyLevel.HARD, 0) == 1
        
        # Verify question type distribution (should have variety)
        question_types = set(card.question_type for card in result)
        assert len(question_types) >= 2  # At least 2 different types
        
        # Verify logging
        mock_logger.info.assert_any_call("Starting quiz generation for paper: test.12345")
        mock_logger.info.assert_any_call("Requested 5 questions with mix: {'easy': 2, 'medium': 2, 'hard': 1}")
    
    def test_quiz_retry_on_validation_error(self, mock_instructor_client, sample_quiz_input, valid_quiz_cards):
        """Test retry logic when first attempt fails validation."""
        # Mock first call to fail, second to succeed
        validation_error = ValidationError.from_exception_data(
            "ValidationError", 
            [{"type": "missing", "loc": (0, "question"), "msg": "Field required"}]
        )
        mock_instructor_client.chat.completions.create.side_effect = [
            validation_error,
            valid_quiz_cards
        ]
        
        agent = QuizAgentImpl(mock_instructor_client, "gpt-4")
        
        with patch('nlp_pillars.agents.quiz_agent.logger') as mock_logger:
            result = agent.run(sample_quiz_input)
        
        # Verify successful result after retry
        assert isinstance(result, list)
        assert len(result) == 5
        
        # Verify retry logging
        mock_logger.warning.assert_called_once()
        mock_logger.info.assert_any_call("Attempting retry with corrective message")
        
        # Verify client was called twice
        assert mock_instructor_client.chat.completions.create.call_count == 2
        
        # Check that retry included corrective suffix
        retry_call_args = mock_instructor_client.chat.completions.create.call_args_list[1]
        retry_message = retry_call_args[1]['messages'][1]['content']
        assert "Your last output was invalid JSON for QuizCard array" in retry_message
    
    def test_quiz_fails_after_retry(self, mock_instructor_client, sample_quiz_input):
        """Test that QuizValidationError is raised when both attempts fail."""
        # Mock both calls to fail validation
        validation_error1 = ValidationError.from_exception_data(
            "ValidationError", 
            [{"type": "missing", "loc": (0, "question"), "msg": "Field required"}]
        )
        validation_error2 = ValidationError.from_exception_data(
            "ValidationError", 
            [{"type": "missing", "loc": (0, "answer"), "msg": "Field required"}]
        )
        mock_instructor_client.chat.completions.create.side_effect = [
            validation_error1,
            validation_error2
        ]
        
        agent = QuizAgentImpl(mock_instructor_client, "gpt-4")
        
        with patch('nlp_pillars.agents.quiz_agent.logger') as mock_logger:
            with pytest.raises(QuizValidationError) as exc_info:
                agent.run(sample_quiz_input)
        
        # Verify error message contains both validation errors
        error_message = str(exc_info.value)
        assert "Failed to generate valid QuizCard array after retry" in error_message
        
        # Verify error logging
        mock_logger.error.assert_called_once()
        
        # Verify both calls were made
        assert mock_instructor_client.chat.completions.create.call_count == 2
    
    def test_quiz_type_distribution_sanity(self, mock_instructor_client, sample_quiz_input):
        """Test that quiz ensures type diversity even if LLM returns all same type."""
        # Create quiz cards all with FACTUAL type (should be fixed by agent)
        all_factual_cards = [
            QuizCard(
                paper_id="test.12345",
                pillar_id=PillarID.P2,
                question=f"Question {i+1}",
                answer=f"Answer {i+1}",
                difficulty=DifficultyLevel.MEDIUM,
                question_type=QuestionType.FACTUAL  # All same type
            ) for i in range(5)
        ]
        
        mock_instructor_client.chat.completions.create.return_value = all_factual_cards
        
        agent = QuizAgentImpl(mock_instructor_client, "gpt-4")
        result = agent.run(sample_quiz_input)
        
        # Verify that type diversity was enforced
        question_types = set(card.question_type for card in result)
        assert len(question_types) >= 2  # Should have at least 2 different types
        
        # Verify that types cycle through the available options
        expected_types = [QuestionType.FACTUAL, QuestionType.CONCEPTUAL, QuestionType.APPLICATION]
        for i, card in enumerate(result):
            expected_type = expected_types[i % len(expected_types)]
            assert card.question_type == expected_type
    
    def test_quiz_exact_count_enforcement(self, mock_instructor_client, sample_quiz_input):
        """Test that quiz returns exactly the requested number of questions."""
        # Test with too many questions returned by LLM
        too_many_cards = [
            QuizCard(
                paper_id="test.12345",
                pillar_id=PillarID.P2,
                question=f"Question {i+1}",
                answer=f"Answer {i+1}",
                difficulty=DifficultyLevel.MEDIUM,
                question_type=QuestionType.FACTUAL
            ) for i in range(8)  # More than requested 5
        ]
        
        mock_instructor_client.chat.completions.create.return_value = too_many_cards
        
        agent = QuizAgentImpl(mock_instructor_client, "gpt-4")
        result = agent.run(sample_quiz_input)
        
        # Should truncate to exactly 5
        assert len(result) == 5
        
        # Verify the first 5 questions were kept
        for i in range(5):
            assert result[i].question == f"Question {i+1}"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('nlp_pillars.agents.synthesis_agent.SynthesisAgent')
    def test_synthesize_function(self, mock_agent, sample_paper_note, sample_pillar_config, valid_lesson):
        """Test the convenience synthesize function."""
        mock_agent.run.return_value = valid_lesson
        
        result = synthesize(
            paper_note=sample_paper_note,
            pillar_config=sample_pillar_config,
            related_lessons=[]
        )
        
        # Verify result
        assert isinstance(result, Lesson)
        
        # Verify agent was called with correct input
        mock_agent.run.assert_called_once()
        call_args = mock_agent.run.call_args[0][0]
        assert isinstance(call_args, SynthesisInput)
        assert call_args.paper_note == sample_paper_note
        assert call_args.pillar_config == sample_pillar_config
    
    @patch('nlp_pillars.agents.quiz_agent.QuizAgent')
    def test_generate_quiz_function(self, mock_agent, sample_paper_note, valid_quiz_cards):
        """Test the convenience generate_quiz function."""
        mock_agent.run.return_value = valid_quiz_cards
        
        result = generate_quiz(
            paper_note=sample_paper_note,
            num_questions=5,
            difficulty_mix={"easy": 2, "medium": 2, "hard": 1}
        )
        
        # Verify result
        assert isinstance(result, list)
        assert len(result) == 5
        
        # Verify agent was called with correct input
        mock_agent.run.assert_called_once()
        call_args = mock_agent.run.call_args[0][0]
        assert isinstance(call_args, QuizGeneratorInput)
        assert call_args.paper_note == sample_paper_note
        assert call_args.num_questions == 5
        assert call_args.difficulty_mix == {"easy": 2, "medium": 2, "hard": 1}


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_synthesis_to_quiz_pipeline(self, mock_instructor_client, sample_paper_note, sample_pillar_config, valid_lesson, valid_quiz_cards):
        """Test that synthesis output can be used as input for related lesson context."""
        # Mock both agents
        mock_instructor_client.chat.completions.create.side_effect = [
            valid_lesson,  # For synthesis
            valid_quiz_cards  # For quiz
        ]
        
        # Create agents
        synthesis_agent = SynthesisAgentImpl(mock_instructor_client, "gpt-4")
        quiz_agent = QuizAgentImpl(mock_instructor_client, "gpt-4")
        
        # Run synthesis first
        synthesis_input = SynthesisInput(
            paper_note=sample_paper_note,
            pillar_config=sample_pillar_config,
            related_lessons=[]
        )
        lesson = synthesis_agent.run(synthesis_input)
        
        # Run quiz generation
        quiz_input = QuizGeneratorInput(
            paper_note=sample_paper_note,
            num_questions=3,
            difficulty_mix={"easy": 1, "medium": 1, "hard": 1}
        )
        quiz_cards = quiz_agent.run(quiz_input)
        
        # Verify both outputs are valid and compatible
        assert isinstance(lesson, Lesson)
        assert isinstance(quiz_cards, list)
        assert lesson.paper_id == sample_paper_note.paper_id
        assert all(card.paper_id == sample_paper_note.paper_id for card in quiz_cards)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
