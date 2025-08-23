"""
Comprehensive tests for SummarizerAgent using Atomic Agents v2.0 + Instructor.
All OpenAI calls are mocked for fast, reliable testing.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from pydantic import ValidationError

from nlp_pillars.schemas import SummarizerInput, PaperNote, ParsedPaper, PaperRef, PillarID
from nlp_pillars.agents.summarizer_agent import (
    SummarizerAgentImpl, 
    SummarizerValidationError,
    summarize,
    _make_client
)


# Test fixtures
@pytest.fixture
def sample_paper_ref():
    """Sample paper reference for testing."""
    return PaperRef(
        id="test.12345",
        title="Test Paper: Attention Mechanisms",
        authors=["Dr. Test", "Prof. Example"],
        venue="Test Conference",
        year=2023,
        url_pdf="https://example.com/test_paper.pdf"
    )


@pytest.fixture
def sample_parsed_paper(sample_paper_ref):
    """Sample parsed paper for testing."""
    return ParsedPaper(
        paper_ref=sample_paper_ref,
        full_text="""This paper introduces a novel attention mechanism for sequence-to-sequence learning. The proposed method achieves state-of-the-art performance on machine translation tasks.

        The key innovation is a multi-head attention mechanism that allows the model to jointly attend to information from different representation subspaces. This approach significantly improves translation quality while maintaining computational efficiency.

        Experimental results on WMT 2014 English-German translation show a BLEU score improvement of 2.8 points over the previous best methods. The model also demonstrates faster convergence during training.

        Limitations include increased memory requirements and potential difficulty with very long sequences. Future work could explore more efficient attention computations.""",
        chunks=["Attention mechanism chunk", "Experimental results chunk", "Limitations chunk"],
        figures_count=2,
        tables_count=3,
        references=["Bahdanau et al., 2015", "Vaswani et al., 2017"]
    )


@pytest.fixture
def sample_summarizer_input(sample_parsed_paper):
    """Sample SummarizerInput for testing."""
    return SummarizerInput(
        parsed_paper=sample_parsed_paper,
        pillar_id=PillarID.P2,
        recent_notes=[
            "Previous paper on RNNs showed limited parallelization capabilities",
            "Recent work on transformers achieved breakthrough performance"
        ]
    )


@pytest.fixture
def valid_paper_note():
    """Valid PaperNote response for testing."""
    return PaperNote(
        paper_id="test.12345",
        pillar_id=PillarID.P2,
        problem="Limited effectiveness of traditional attention mechanisms in sequence-to-sequence learning",
        method="Multi-head attention mechanism with parallel processing capabilities",
        findings=[
            "Achieved 2.8 BLEU score improvement on WMT 2014 English-German translation",
            "Demonstrated faster convergence during training",
            "Maintained computational efficiency compared to recurrent models"
        ],
        limitations=[
            "Increased memory requirements for longer sequences",
            "Potential difficulty with very long sequences",
            "Requires large amounts of training data"
        ],
        future_work=[
            "Explore more efficient attention computations",
            "Investigate performance on longer sequences",
            "Study application to other sequence tasks"
        ],
        key_terms=[
            "multi-head attention",
            "sequence-to-sequence",
            "machine translation",
            "BLEU score",
            "computational efficiency"
        ],
        confidence_score=0.9
    )


@pytest.fixture
def mock_instructor_client():
    """Mock Instructor client for testing."""
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    return mock_client


class TestSummarizerAgentImpl:
    """Test cases for SummarizerAgentImpl."""
    
    def test_happy_path(self, mock_instructor_client, sample_summarizer_input, valid_paper_note):
        """Test successful summarization on first attempt."""
        # Mock successful response
        mock_instructor_client.chat.completions.create.return_value = valid_paper_note
        
        # Create agent
        agent = SummarizerAgentImpl(mock_instructor_client, "gpt-4")
        
        # Run summarization
        with patch('nlp_pillars.agents.summarizer_agent.logger') as mock_logger:
            result = agent.run(sample_summarizer_input)
        
        # Verify result
        assert isinstance(result, PaperNote)
        assert result.paper_id == "test.12345"
        assert result.pillar_id == PillarID.P2
        assert result.problem == valid_paper_note.problem
        assert result.method == valid_paper_note.method
        assert len(result.findings) == 3
        assert len(result.limitations) == 3
        assert len(result.key_terms) == 5
        
        # Verify logging
        mock_logger.info.assert_any_call("Starting summarization for paper: Test Paper: Attention Mechanisms")
        mock_logger.info.assert_any_call("Summarization completed successfully on first attempt")
        
        # Verify client was called correctly
        mock_instructor_client.chat.completions.create.assert_called_once()
        call_args = mock_instructor_client.chat.completions.create.call_args
        assert call_args[1]['model'] == "gpt-4"
        assert call_args[1]['response_model'] == PaperNote
        assert call_args[1]['temperature'] == 0.2
        assert len(call_args[1]['messages']) == 2
        assert call_args[1]['messages'][0]['role'] == 'system'
        assert call_args[1]['messages'][1]['role'] == 'user'
    
    def test_retry_on_validation_error(self, mock_instructor_client, sample_summarizer_input, valid_paper_note):
        """Test retry logic when first attempt fails validation."""
        # Mock first call to fail, second to succeed
        validation_error = ValidationError.from_exception_data(
            "ValidationError", 
            [{"type": "missing", "loc": ("problem",), "msg": "Field required"}]
        )
        mock_instructor_client.chat.completions.create.side_effect = [
            validation_error,
            valid_paper_note
        ]
        
        agent = SummarizerAgentImpl(mock_instructor_client, "gpt-4")
        
        with patch('nlp_pillars.agents.summarizer_agent.logger') as mock_logger:
            result = agent.run(sample_summarizer_input)
        
        # Verify successful result after retry
        assert isinstance(result, PaperNote)
        assert result.paper_id == "test.12345"
        
        # Verify retry logging
        mock_logger.warning.assert_called_once()
        mock_logger.info.assert_any_call("Attempting retry with corrective message")
        mock_logger.info.assert_any_call("Summarization completed successfully on retry")
        
        # Verify client was called twice
        assert mock_instructor_client.chat.completions.create.call_count == 2
        
        # Check that retry included corrective suffix
        retry_call_args = mock_instructor_client.chat.completions.create.call_args_list[1]
        retry_message = retry_call_args[1]['messages'][1]['content']
        assert "Your last output was invalid JSON for PaperNote" in retry_message
    
    def test_fails_after_retry(self, mock_instructor_client, sample_summarizer_input):
        """Test that SummarizerValidationError is raised when both attempts fail."""
        # Mock both calls to fail validation
        validation_error1 = ValidationError.from_exception_data(
            "ValidationError", 
            [{"type": "missing", "loc": ("problem",), "msg": "Field required"}]
        )
        validation_error2 = ValidationError.from_exception_data(
            "ValidationError", 
            [{"type": "missing", "loc": ("method",), "msg": "Field required"}]
        )
        mock_instructor_client.chat.completions.create.side_effect = [
            validation_error1,
            validation_error2
        ]
        
        agent = SummarizerAgentImpl(mock_instructor_client, "gpt-4")
        
        with patch('nlp_pillars.agents.summarizer_agent.logger') as mock_logger:
            with pytest.raises(SummarizerValidationError) as exc_info:
                agent.run(sample_summarizer_input)
        
        # Verify error message contains both validation errors
        error_message = str(exc_info.value)
        assert "Failed to generate valid PaperNote after retry" in error_message
        assert "Original error:" in error_message
        assert "Retry error:" in error_message
        
        # Verify error logging
        mock_logger.error.assert_called_once()
        
        # Verify both calls were made
        assert mock_instructor_client.chat.completions.create.call_count == 2
    
    def test_context_pass_through(self, mock_instructor_client, sample_summarizer_input, valid_paper_note):
        """Test that recent_notes context is included in the prompt."""
        mock_instructor_client.chat.completions.create.return_value = valid_paper_note
        
        agent = SummarizerAgentImpl(mock_instructor_client, "gpt-4")
        result = agent.run(sample_summarizer_input)
        
        # Verify the context was passed through
        call_args = mock_instructor_client.chat.completions.create.call_args
        user_message = call_args[1]['messages'][1]['content']
        
        # Check that recent notes are included
        assert "Recent paper summaries for consistency:" in user_message
        assert "Previous paper on RNNs" in user_message
        assert "Recent work on transformers" in user_message
        
        # Check that paper content is included
        assert "Test Paper: Attention Mechanisms" in user_message
        assert "Dr. Test, Prof. Example" in user_message
        assert "novel attention mechanism" in user_message
    
    def test_system_message_building(self, mock_instructor_client, sample_summarizer_input, valid_paper_note):
        """Test that system message is built correctly from SystemPromptGenerator."""
        mock_instructor_client.chat.completions.create.return_value = valid_paper_note
        
        agent = SummarizerAgentImpl(mock_instructor_client, "gpt-4")
        agent.run(sample_summarizer_input)
        
        call_args = mock_instructor_client.chat.completions.create.call_args
        system_message = call_args[1]['messages'][0]['content']
        
        # Verify required content from the task specification
        assert "NLP research summarizer; faithful, structured" in system_message
        assert "Cite only what's supported by the text; avoid hallucinations" in system_message
        assert "Extract: problem, method, findings, limitations, future_work, key_terms" in system_message
        assert "Be concise but precise; prefer bullet points for lists" in system_message
        assert "Return valid PaperNote JSON only (no extra text)" in system_message
        assert "You must return a valid PaperNote JSON object" in system_message
    
    def test_content_truncation(self, mock_instructor_client, sample_summarizer_input, valid_paper_note):
        """Test that very long paper content is truncated to avoid token limits."""
        # Create a paper with very long content
        long_content = "Very long paper content. " * 1000  # Much longer than 8000 chars
        sample_summarizer_input.parsed_paper.full_text = long_content
        
        mock_instructor_client.chat.completions.create.return_value = valid_paper_note
        
        agent = SummarizerAgentImpl(mock_instructor_client, "gpt-4")
        agent.run(sample_summarizer_input)
        
        call_args = mock_instructor_client.chat.completions.create.call_args
        user_message = call_args[1]['messages'][1]['content']
        
        # Verify content was truncated (should be much shorter than original)
        assert len(user_message) < len(long_content)
        # Should still contain the truncated content
        assert "Very long paper content." in user_message
    
    def test_instructor_exception_handling(self, mock_instructor_client, sample_summarizer_input):
        """Test that non-ValidationError exceptions from Instructor are handled properly."""
        # Mock Instructor to raise a different kind of error
        mock_instructor_client.chat.completions.create.side_effect = Exception("API Error")
        
        agent = SummarizerAgentImpl(mock_instructor_client, "gpt-4")
        
        # Should wrap in SummarizerValidationError for consistent handling
        with pytest.raises(SummarizerValidationError, match="Instructor completion failed"):
            agent.run(sample_summarizer_input)


class TestConvenienceFunctions:
    """Test convenience functions and module-level components."""
    
    @patch('nlp_pillars.agents.summarizer_agent.SummarizerAgent')
    def test_summarize_function(self, mock_agent, sample_parsed_paper, valid_paper_note):
        """Test the convenience summarize function."""
        # Mock the global SummarizerAgent
        mock_agent.run.return_value = valid_paper_note
        
        result = summarize(
            parsed_paper=sample_parsed_paper,
            pillar_id=PillarID.P2,
            recent_notes=["Recent note"]
        )
        
        # Verify result
        assert isinstance(result, PaperNote)
        
        # Verify agent was called with correct input
        mock_agent.run.assert_called_once()
        call_args = mock_agent.run.call_args[0][0]
        assert isinstance(call_args, SummarizerInput)
        assert call_args.parsed_paper == sample_parsed_paper
        assert call_args.pillar_id == PillarID.P2
        assert call_args.recent_notes == ["Recent note"]
    
    @patch('nlp_pillars.agents.summarizer_agent.SummarizerAgent', None)
    def test_summarize_function_agent_not_initialized(self, sample_parsed_paper):
        """Test summarize function when SummarizerAgent is not initialized."""
        with pytest.raises(ValueError, match="SummarizerAgent is not initialized"):
            summarize(
                parsed_paper=sample_parsed_paper,
                pillar_id=PillarID.P2
            )
    
    @patch('nlp_pillars.agents.summarizer_agent.get_settings')
    @patch('nlp_pillars.agents.summarizer_agent.instructor.from_openai')
    def test_make_client_success(self, mock_from_openai, mock_get_settings):
        """Test successful client creation."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.openai_api_key = "test-api-key"
        mock_get_settings.return_value = mock_settings
        
        # Mock instructor client
        mock_client = Mock()
        mock_from_openai.return_value = mock_client
        
        result = _make_client()
        
        assert result == mock_client
        mock_from_openai.assert_called_once()
    
    @patch('nlp_pillars.agents.summarizer_agent.get_settings')
    def test_make_client_missing_api_key(self, mock_get_settings):
        """Test client creation failure when API key is missing."""
        # Mock settings without API key
        mock_settings = Mock()
        mock_settings.openai_api_key = None
        mock_get_settings.return_value = mock_settings
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is required"):
            _make_client()


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @patch('nlp_pillars.agents.summarizer_agent._make_client')
    @patch('nlp_pillars.agents.summarizer_agent.get_settings')
    def test_agent_initialization_and_usage(self, mock_get_settings, mock_make_client, 
                                           sample_summarizer_input, valid_paper_note):
        """Test full agent initialization and usage flow."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.default_model = "gpt-4"
        mock_get_settings.return_value = mock_settings
        
        # Mock client
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = valid_paper_note
        mock_make_client.return_value = mock_client
        
        # Create agent directly
        agent = SummarizerAgentImpl(mock_client, "gpt-4")
        
        # Test the agent
        result = agent.run(sample_summarizer_input)
        
        # Verify result
        assert isinstance(result, PaperNote)
        assert result.paper_id == "test.12345"
        assert result.pillar_id == PillarID.P2
        
        # Verify all required fields are present
        assert result.problem
        assert result.method
        assert result.findings
        assert result.limitations
        assert result.future_work
        assert result.key_terms
        assert 0.0 <= result.confidence_score <= 1.0


# Test logging configuration
class TestLogging:
    """Test logging functionality."""
    
    def test_logger_configuration(self):
        """Test that logger is properly configured."""
        from nlp_pillars.agents.summarizer_agent import logger
        
        assert logger.name == "nlp_pillars.agents.summarizer_agent"
        assert isinstance(logger, logging.Logger)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
