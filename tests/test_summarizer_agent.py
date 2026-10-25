"""SummarizerAgent, on Claude's native structured output.

The provider is mocked, so nothing here reaches the network. What the file
pins down is the shape of the call: no sampling parameters, a small
model-facing schema, identity and timestamps from this process, and the whole
paper in the prompt.
"""

import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from anthropic import Anthropic
from anthropic.resources.messages import Messages

from nlp_pillars.schemas import (
    NoteDraft,
    PaperNote,
    ParsedPaper,
    PaperRef,
    SummarizerInput,
)
from nlp_pillars.agents.summarizer_agent import (
    SummarizerAgentImpl,
    SummarizerValidationError,
    summarize,
    _make_client,
    _LazySummarizerAgent,
)

MODEL = "claude-haiku-5-5"


def _draft(**overrides):
    values = {
        "problem": "Limited effectiveness of traditional attention mechanisms",
        "method": "Multi-head attention with parallel processing",
        "findings": ["2.8 BLEU improvement on WMT 2014 EN-DE"],
        "limitations": ["Increased memory requirements for longer sequences"],
        "future_work": ["Explore more efficient attention computations"],
        "key_terms": ["multi-head attention", "BLEU score"],
        "concepts": ["multi-head attention"],
        "proposed_concepts": [],
        "related_papers": [],
        "confidence_score": 0.9,
    }
    values.update(overrides)
    return NoteDraft(**values)


def _parsed(value, *, input_tokens=100, output_tokens=20, stop_reason="end_turn"):
    return SimpleNamespace(
        parsed_output=value,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        stop_reason=stop_reason,
    )


def _client(result=None, *, error=None, side_effect=None, input_tokens=100,
            output_tokens=20):
    """A mock Anthropic client, spec'd so a missing method is a test failure."""
    messages = Mock(spec=Messages)
    if error is not None:
        messages.parse.side_effect = error
    elif side_effect is not None:
        messages.parse.side_effect = side_effect
    else:
        messages.parse.return_value = _parsed(
            result, input_tokens=input_tokens, output_tokens=output_tokens
        )
    client = Mock(spec=Anthropic)
    client.messages = messages
    return client, messages


@pytest.fixture
def sample_paper_ref():
    return PaperRef(
        id="test.12345",
        title="Test Paper: Attention Mechanisms",
        authors=["Dr. Test", "Prof. Example"],
        venue="Test Conference",
        year=2023,
        url_pdf="https://example.com/test_paper.pdf",
    )


@pytest.fixture
def sample_parsed_paper(sample_paper_ref):
    return ParsedPaper(
        paper_ref=sample_paper_ref,
        full_text=(
            "This paper introduces a novel attention mechanism for "
            "sequence-to-sequence learning. The proposed method achieves "
            "state-of-the-art performance on machine translation tasks."
        ),
        chunks=["chunk"],
        figures_count=2,
        tables_count=3,
        references=["Bahdanau et al., 2015"],
    )


@pytest.fixture
def sample_summarizer_input(sample_parsed_paper):
    return SummarizerInput(
        parsed_paper=sample_parsed_paper,
        pillar_id="models-architectures",
        recent_notes=[
            "Previous paper on RNNs showed limited parallelization capabilities",
            "Recent work on transformers achieved breakthrough performance",
        ],
        concept_vocabulary=["multi-head attention"],
    )


class TestSummarizerAgentImpl:
    def test_happy_path(self, sample_summarizer_input):
        client, messages = _client(_draft())

        agent = SummarizerAgentImpl(client, MODEL)
        result = agent.run(sample_summarizer_input)

        assert isinstance(result, PaperNote)
        assert result.paper_id == "test.12345"
        assert result.pillar_id == "models-architectures"
        assert result.problem == _draft().problem
        assert result.findings == _draft().findings
        assert result.concepts == ["multi-head attention"]

        messages.parse.assert_called_once()
        kwargs = messages.parse.call_args.kwargs
        assert kwargs["model"] == MODEL
        assert kwargs["output_format"] is NoteDraft
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"

    def test_no_sampling_parameters_are_ever_sent(self, sample_summarizer_input):
        """Claude rejects temperature, top_p, top_k and a forced tool choice."""
        client, messages = _client(_draft())

        SummarizerAgentImpl(client, MODEL).run(sample_summarizer_input)

        kwargs = messages.parse.call_args.kwargs
        for banned in ("temperature", "top_p", "top_k", "tool_choice", "tools"):
            assert banned not in kwargs

    def test_the_model_is_not_asked_for_identity_or_a_date(self):
        """NoteDraft has no paper_id, pillar_id or created_at to get wrong."""
        properties = NoteDraft.model_json_schema()["properties"]
        assert "paper_id" not in properties
        assert "pillar_id" not in properties
        assert "created_at" not in properties

    def test_the_stored_note_carries_this_process_identity(self, sample_summarizer_input):
        client, _ = _client(_draft())

        result = SummarizerAgentImpl(client, MODEL).run(sample_summarizer_input)

        assert result.created_at is not None
        assert result.created_at.year >= 2026

    def test_context_pass_through(self, sample_summarizer_input):
        client, messages = _client(_draft())

        SummarizerAgentImpl(client, MODEL).run(sample_summarizer_input)

        user_message = messages.parse.call_args.kwargs["messages"][0]["content"]
        assert "Recent paper summaries for consistency:" in user_message
        assert "Previous paper on RNNs" in user_message
        assert "Test Paper: Attention Mechanisms" in user_message
        assert "Dr. Test, Prof. Example" in user_message

    def test_the_whole_paper_is_sent_not_the_first_8000_characters(
        self, sample_summarizer_input
    ):
        """The truncation used to hide the conclusion section."""
        long_content = "Very long paper content. " * 1000
        sample_summarizer_input.parsed_paper.full_text = long_content
        client, messages = _client(_draft())

        SummarizerAgentImpl(client, MODEL).run(sample_summarizer_input)

        user_message = messages.parse.call_args.kwargs["messages"][0]["content"]
        assert len(user_message) > 8000
        assert "[TRUNCATED" not in user_message

    def test_a_very_long_paper_is_capped_and_says_so(self, sample_summarizer_input):
        """The cap is in tokens, because Haiku's rate card changes at 100K tokens.

        A character cap cannot see that coming: measured on 2026-10-25, 300K
        characters of a dense paper is 172K Claude tokens, which crosses the
        long-context rate and costs five times as much.
        """
        sample_summarizer_input.parsed_paper.full_text = (
            "attention mechanism transformer encoder decoder " * 40_000
        )
        client, messages = _client(_draft())

        SummarizerAgentImpl(client, MODEL).run(sample_summarizer_input)

        user_message = messages.parse.call_args.kwargs["messages"][0]["content"]
        assert "[TRUNCATED - paper continues]" in user_message
        assert len(user_message) < len(
            sample_summarizer_input.parsed_paper.full_text
        )

    def test_a_failed_call_falls_back_to_the_fallback_model(self, sample_summarizer_input):
        """One model being unavailable must not cost the paper its note."""
        client, messages = _client(
            side_effect=[
                RuntimeError("overloaded"),
                _parsed(_draft()),
            ]
        )

        agent = SummarizerAgentImpl(client, MODEL)
        result = agent.run(sample_summarizer_input)

        assert result.problem == _draft().problem
        assert messages.parse.call_count == 2
        assert messages.parse.call_args_list[1].kwargs["model"] == "claude-sonnet-5-5"
        assert agent.last_call.fallback is True
        assert "overloaded" in (agent.last_call.fallback_reason or "")

    def test_the_cost_of_the_call_is_recorded(self, sample_summarizer_input):
        client, _ = _client(_draft(), input_tokens=34_175, output_tokens=964)

        agent = SummarizerAgentImpl(client, MODEL)
        agent.run(sample_summarizer_input)

        call = agent.last_call
        assert call.model == MODEL
        assert call.input_tokens == 34_175
        assert call.output_tokens == 964
        assert call.cost_usd > 0
        assert call.detail().startswith(MODEL)

    def test_a_failed_call_is_wrapped(self, sample_summarizer_input):
        client, _ = _client(error=RuntimeError("no route to host"))

        agent = SummarizerAgentImpl(client, MODEL)
        with pytest.raises(SummarizerValidationError) as exc_info:
            agent.run(sample_summarizer_input)

        assert "Claude completion failed" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None


class TestConvenienceFunctions:
    @patch('nlp_pillars.agents.summarizer_agent.SummarizerAgent')
    def test_summarize_function(self, mock_agent, sample_parsed_paper):
        note = PaperNote(
            paper_id="test.12345",
            pillar_id="models-architectures",
            problem="p",
            method="m",
            findings=[],
            limitations=[],
            future_work=[],
            key_terms=[],
        )
        mock_agent.run.return_value = note

        result = summarize(
            parsed_paper=sample_parsed_paper,
            pillar_id="models-architectures",
            recent_notes=["Recent note"],
        )

        assert result is note
        mock_agent.run.assert_called_once()
        call_args = mock_agent.run.call_args[0][0]
        assert isinstance(call_args, SummarizerInput)
        assert call_args.pillar_id == "models-architectures"
        assert call_args.recent_notes == ["Recent note"]

    @patch('nlp_pillars.agents.summarizer_agent.SummarizerAgent', None)
    def test_summarize_function_agent_not_initialized(self, sample_parsed_paper):
        with pytest.raises(ValueError, match="SummarizerAgent is not initialized"):
            summarize(
                parsed_paper=sample_parsed_paper,
                pillar_id="models-architectures",
            )

    @patch('nlp_pillars.llm_client.get_settings')
    def test_make_client_missing_api_key(self, mock_get_settings):
        mock_get_settings.return_value = SimpleNamespace(anthropic_api_key=None)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            _make_client()


class TestLazySingleton:
    def test_import_constructs_no_client(self):
        from nlp_pillars.agents.summarizer_agent import SummarizerAgent

        assert isinstance(SummarizerAgent, _LazySummarizerAgent)
        assert SummarizerAgent._impl is None

    @patch('nlp_pillars.agents.summarizer_agent.get_settings')
    def test_missing_api_key_names_the_missing_key(self, mock_get_settings, sample_summarizer_input):
        """With no API key, calling run() names the key - not AttributeError."""
        mock_get_settings.return_value = SimpleNamespace(anthropic_api_key=None)

        agent = _LazySummarizerAgent()

        with pytest.raises(ValueError) as exc_info:
            agent.run(sample_summarizer_input)

        message = str(exc_info.value)
        assert "SummarizerAgent is not initialized" in message
        assert "ANTHROPIC_API_KEY" in message

    @patch('nlp_pillars.agents.summarizer_agent._make_client')
    @patch('nlp_pillars.agents.summarizer_agent.get_settings')
    def test_impl_built_once_and_reused(
        self, mock_get_settings, mock_make_client, sample_summarizer_input
    ):
        mock_get_settings.return_value = SimpleNamespace(anthropic_api_key="sk-ant-test")
        client, _ = _client(_draft())
        mock_make_client.return_value = client

        agent = _LazySummarizerAgent()
        assert agent._impl is None

        agent.run(sample_summarizer_input)
        agent.run(sample_summarizer_input)

        assert agent._impl is not None
        mock_make_client.assert_called_once()
        assert agent.last_call is not None


class TestLogging:
    def test_logger_configuration(self):
        from nlp_pillars.agents.summarizer_agent import logger

        assert logger.name == "nlp_pillars.agents.summarizer_agent"
        assert isinstance(logger, logging.Logger)


class TestPaperTextBudget:
    """The paper text is trimmed to a token budget, not a character count."""

    def test_a_paper_that_fits_is_untouched(self):
        from nlp_pillars.llm_client import fit_paper_text

        text = "A short paper. " * 100
        assert fit_paper_text(text) == text

    def test_the_trimmed_text_is_within_the_budget(self):
        import tiktoken

        from nlp_pillars.llm_client import (
            PAPER_TEXT_TOKEN_BUDGET,
            TRUNCATION_MARKER,
            fit_paper_text,
        )

        text = "attention mechanism transformer encoder decoder " * 40_000
        fitted = fit_paper_text(text)

        assert fitted.endswith(TRUNCATION_MARKER)
        encoding = tiktoken.get_encoding("cl100k_base")
        assert len(encoding.encode(fitted)) <= PAPER_TEXT_TOKEN_BUDGET + 32

    def test_the_budget_leaves_room_under_haikus_long_context_threshold(self):
        """60K tiktoken tokens is about 89K Claude tokens on the densest paper.

        The long-context rate starts at 100K tokens and applies to the whole
        request, so a prompt that crosses it costs five times as much.
        """
        from nlp_pillars.llm_client import PAPER_TEXT_TOKEN_BUDGET

        # The worst Claude/tiktoken ratio measured on the library's papers:
        # 60,000 tiktoken tokens came back as 89,261 Claude tokens.
        assert PAPER_TEXT_TOKEN_BUDGET * 1.49 < 100_000


class TestThinkingParameter:
    """`thinking: disabled` is a 400 on Sonnet, so it is only sent to Haiku."""

    def test_only_the_haiku_family_accepts_thinking_disabled(self):
        from nlp_pillars.llm_client import supports_thinking_disabled

        assert supports_thinking_disabled("claude-haiku-5-5")
        assert not supports_thinking_disabled("claude-sonnet-5")
        assert not supports_thinking_disabled("claude-sonnet-5-5")

    def test_a_sonnet_override_does_not_send_the_thinking_parameter(
        self, sample_summarizer_input
    ):
        """The env override has to keep working for every model."""
        client, messages = _client(_draft())

        SummarizerAgentImpl(client, "claude-sonnet-5-5").run(sample_summarizer_input)

        kwargs = messages.parse.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-5-5"
        assert "thinking" not in kwargs
