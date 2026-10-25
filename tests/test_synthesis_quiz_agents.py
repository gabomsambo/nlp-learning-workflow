"""SynthesisAgent and QuizAgent, on Claude's native structured output.

The provider is mocked, so nothing here reaches the network. What the file
pins down is the shape of the call: no sampling parameters, small
model-facing schemas, and the paper's own text in the prompt.
"""

import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from anthropic import Anthropic
from anthropic.resources.messages import Messages

from nlp_pillars.schemas import (
    CardDraft,
    DifficultyLevel,
    Lesson,
    LessonDraft,
    PaperNote,
    PillarConfig,
    QuestionType,
    QuizDraft,
    QuizGeneratorInput,
    SynthesisInput,
)
from nlp_pillars.agents.synthesis_agent import (
    SynthesisAgentImpl, SynthesisValidationError, synthesize
)
from nlp_pillars.agents.quiz_agent import (
    QuizAgentImpl, QuizValidationError, generate_quiz
)

MODEL = "claude-haiku-5-5"


def _lesson_draft(**overrides):
    values = {
        "title": "Understanding Multi-Head Attention in Transformers",
        "content": "This lesson explores the transformer architecture.",
        "tl_dr": "Transformers replaced recurrence with attention.",
        "takeaways": [
            "Multi-head attention allows parallel processing",
            "Self-attention captures long-range dependencies",
        ],
        "practice_ideas": ["Implement a mini-transformer"],
        "connections": ["BERT uses the encoder"],
        "examples": [],
    }
    values.update(overrides)
    return LessonDraft(**values)


def _quiz_draft(n=5):
    return QuizDraft(
        cards=[
            CardDraft(
                question=f"Question {i + 1}",
                answer=f"Answer {i + 1}",
                question_type=["factual", "conceptual", "application"][i % 3],
            )
            for i in range(n)
        ]
    )


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


def _client(result=None, *, error=None, side_effect=None):
    messages = Mock(spec=Messages)
    if error is not None:
        messages.parse.side_effect = error
    elif side_effect is not None:
        messages.parse.side_effect = side_effect
    else:
        messages.parse.return_value = _parsed(result)
    client = Mock(spec=Anthropic)
    client.messages = messages
    return client, messages


@pytest.fixture
def sample_paper_note():
    return PaperNote(
        paper_id="test.12345",
        pillar_id="models-architectures",
        problem="Limited effectiveness of traditional attention mechanisms",
        method="Multi-head attention with scaled dot-product computation",
        findings=["Achieved 28.4 BLEU score on WMT 2014 EN-DE translation"],
        limitations=["High memory requirements for long sequences"],
        future_work=["Investigate linear attention mechanisms"],
        key_terms=["multi-head attention", "transformer"],
        confidence_score=0.9,
    )


@pytest.fixture
def sample_pillar_config():
    return PillarConfig(
        id="models-architectures",
        name="Models & Architectures",
        goal="Master modern neural network architectures",
        focus_areas=["Attention mechanisms", "Transformer architectures"],
    )


@pytest.fixture
def sample_synthesis_input(sample_paper_note, sample_pillar_config):
    return SynthesisInput(
        paper_note=sample_paper_note,
        pillar_config=sample_pillar_config,
        related_lessons=[],
        paper_title="Attention Is All You Need",
        paper_authors=["Vaswani et al."],
    )


@pytest.fixture
def sample_quiz_input(sample_paper_note):
    return QuizGeneratorInput(
        paper_note=sample_paper_note,
        num_questions=5,
        difficulty_mix={"easy": 2, "medium": 2, "hard": 1},
    )


class TestSynthesisAgent:
    def test_synthesis_happy_path(self, sample_synthesis_input):
        client, messages = _client(_lesson_draft())

        result = SynthesisAgentImpl(client, MODEL).run(sample_synthesis_input)

        assert isinstance(result, Lesson)
        assert result.paper_id == "test.12345"
        assert result.pillar_id == "models-architectures"
        assert result.tl_dr == _lesson_draft().tl_dr
        assert len(result.takeaways) == 2

        messages.parse.assert_called_once()
        kwargs = messages.parse.call_args.kwargs
        assert kwargs["model"] == MODEL
        assert kwargs["output_format"] is LessonDraft

    def test_no_sampling_parameters_are_ever_sent(self, sample_synthesis_input):
        """Claude rejects temperature, top_p, top_k and a forced tool choice."""
        client, messages = _client(_lesson_draft())

        SynthesisAgentImpl(client, MODEL).run(sample_synthesis_input)

        kwargs = messages.parse.call_args.kwargs
        for banned in ("temperature", "top_p", "top_k", "tool_choice", "tools"):
            assert banned not in kwargs

    def test_the_prompt_names_the_paper_it_is_about(self, sample_synthesis_input):
        """Otherwise the model treats the paper as related work."""
        client, messages = _client(_lesson_draft())

        SynthesisAgentImpl(client, MODEL).run(sample_synthesis_input)

        user_message = messages.parse.call_args.kwargs["messages"][0]["content"]
        assert "Attention Is All You Need" in user_message
        assert "not related work" in user_message

    def test_the_paper_text_is_sent_when_there_is_one(self, sample_synthesis_input):
        sample_synthesis_input.full_text = "The paper body, in full."
        client, messages = _client(_lesson_draft())

        SynthesisAgentImpl(client, MODEL).run(sample_synthesis_input)

        user_message = messages.parse.call_args.kwargs["messages"][0]["content"]
        assert "The paper body, in full." in user_message

    def test_a_failed_call_falls_back_to_the_fallback_model(self, sample_synthesis_input):
        client, messages = _client(
            side_effect=[RuntimeError("overloaded"), _parsed(_lesson_draft())]
        )

        agent = SynthesisAgentImpl(client, MODEL)
        result = agent.run(sample_synthesis_input)

        assert result.tl_dr == _lesson_draft().tl_dr
        assert messages.parse.call_count == 2
        assert messages.parse.call_args_list[1].kwargs["model"] == "claude-sonnet-5-5"
        assert agent.last_call.fallback is True

    def test_synthesis_failure_is_wrapped(self, sample_synthesis_input):
        client, _ = _client(error=RuntimeError("no route to host"))

        agent = SynthesisAgentImpl(client, MODEL)
        with pytest.raises(SynthesisValidationError) as exc_info:
            agent.run(sample_synthesis_input)

        assert "Claude completion failed" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None


class TestQuizAgent:
    def test_quiz_happy_path_with_difficulty_mix(self, sample_quiz_input):
        client, messages = _client(_quiz_draft())

        result = QuizAgentImpl(client, MODEL).run(sample_quiz_input)

        assert isinstance(result, list)
        assert len(result) == 5
        assert all(card.paper_id == "test.12345" for card in result)
        assert all(card.pillar_id == "models-architectures" for card in result)
        assert all(card.question and card.answer for card in result)

        kwargs = messages.parse.call_args.kwargs
        assert kwargs["output_format"] is QuizDraft
        for banned in ("temperature", "top_p", "top_k", "tool_choice", "tools"):
            assert banned not in kwargs

    def test_the_stored_quizcard_schema_is_not_the_output_format(self):
        """The full QuizCard is rejected by Claude with 400 Schema is too complex."""
        from nlp_pillars.schemas import QuizCard

        properties = QuizCard.model_json_schema()["properties"]
        assert len(properties) > 20
        # The draft is three fields.
        assert len(CardDraft.model_json_schema()["properties"]) == 3

    def test_quiz_preserves_model_question_types(self, sample_quiz_input):
        """question_type is whatever the model said, never fabricated from index."""
        draft = QuizDraft(
            cards=[
                CardDraft(
                    question=f"Question {i + 1}",
                    answer=f"Answer {i + 1}",
                    question_type="factual",
                )
                for i in range(5)
            ]
        )
        client, _ = _client(draft)

        result = QuizAgentImpl(client, MODEL).run(sample_quiz_input)

        assert [card.question_type for card in result] == [QuestionType.FACTUAL] * 5

    def test_quiz_preserves_mixed_model_question_types(self, sample_quiz_input):
        draft = _quiz_draft()
        expected = [QuestionType(card.question_type) for card in draft.cards]
        client, _ = _client(draft)

        result = QuizAgentImpl(client, MODEL).run(sample_quiz_input)

        assert [card.question_type for card in result] == expected
        assert len(set(expected)) >= 2

    def test_quiz_difficulty_mix_is_still_enforced(self, sample_quiz_input):
        """difficulty IS reassigned - it honours the caller's difficulty_mix."""
        client, _ = _client(_quiz_draft())

        result = QuizAgentImpl(client, MODEL).run(sample_quiz_input)

        assert [card.difficulty for card in result] == [
            DifficultyLevel.EASY,
            DifficultyLevel.EASY,
            DifficultyLevel.MEDIUM,
            DifficultyLevel.MEDIUM,
            DifficultyLevel.HARD,
        ]

    def test_quiz_exact_count_enforcement(self, sample_quiz_input):
        """More cards than asked for are trimmed to the requested number."""
        client, _ = _client(_quiz_draft(8))

        result = QuizAgentImpl(client, MODEL).run(sample_quiz_input)

        assert len(result) == 5
        assert [card.question for card in result] == [
            f"Question {i + 1}" for i in range(5)
        ]

    def test_quiz_failure_is_wrapped(self, sample_quiz_input):
        client, _ = _client(error=RuntimeError("no route to host"))

        with pytest.raises(QuizValidationError) as exc_info:
            QuizAgentImpl(client, MODEL).run(sample_quiz_input)

        assert "Claude completion failed" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    def test_the_paper_text_is_sent_when_there_is_one(self, sample_quiz_input):
        sample_quiz_input.full_text = "The paper body, in full."
        client, messages = _client(_quiz_draft())

        QuizAgentImpl(client, MODEL).run(sample_quiz_input)

        user_message = messages.parse.call_args.kwargs["messages"][0]["content"]
        assert "The paper body, in full." in user_message


class TestConvenienceFunctions:
    @patch('nlp_pillars.agents.synthesis_agent.SynthesisAgent')
    def test_synthesize_function(self, mock_agent, sample_paper_note, sample_pillar_config):
        lesson = Lesson(
            paper_id="test.12345",
            pillar_id="models-architectures",
            title="t",
            content="c",
            tl_dr="x",
            takeaways=[],
            practice_ideas=[],
            connections=[],
        )
        mock_agent.run.return_value = lesson

        result = synthesize(
            paper_note=sample_paper_note,
            pillar_config=sample_pillar_config,
            related_lessons=[],
        )

        assert result is lesson
        mock_agent.run.assert_called_once()
        call_args = mock_agent.run.call_args[0][0]
        assert isinstance(call_args, SynthesisInput)
        assert call_args.paper_note == sample_paper_note
        assert call_args.pillar_config == sample_pillar_config

    @patch('nlp_pillars.agents.quiz_agent.QuizAgent')
    def test_generate_quiz_function(self, mock_agent, sample_paper_note):
        mock_agent.run.return_value = []

        result = generate_quiz(
            paper_note=sample_paper_note,
            num_questions=5,
            difficulty_mix={"easy": 2, "medium": 2, "hard": 1},
        )

        assert result == []
        mock_agent.run.assert_called_once()
        call_args = mock_agent.run.call_args[0][0]
        assert isinstance(call_args, QuizGeneratorInput)
        assert call_args.num_questions == 5
        assert call_args.difficulty_mix == {"easy": 2, "medium": 2, "hard": 1}


class TestIntegration:
    def test_synthesis_output_can_feed_the_next_stage(
        self, sample_paper_note, sample_pillar_config
    ):
        client, messages = _client(
            side_effect=[_parsed(_lesson_draft()), _parsed(_quiz_draft())]
        )

        synthesis_agent = SynthesisAgentImpl(client, MODEL)
        quiz_agent = QuizAgentImpl(client, MODEL)

        lesson = synthesis_agent.run(
            SynthesisInput(
                paper_note=sample_paper_note,
                pillar_config=sample_pillar_config,
                related_lessons=[],
            )
        )
        quiz_cards = quiz_agent.run(
            QuizGeneratorInput(
                paper_note=sample_paper_note,
                num_questions=3,
                difficulty_mix={"easy": 1, "medium": 1, "hard": 1},
            )
        )

        assert isinstance(lesson, Lesson)
        assert isinstance(quiz_cards, list)
        assert lesson.paper_id == sample_paper_note.paper_id
        assert all(card.paper_id == sample_paper_note.paper_id for card in quiz_cards)


class TestLogging:
    def test_logger_configuration(self):
        from nlp_pillars.agents.synthesis_agent import logger

        assert logger.name == "nlp_pillars.agents.synthesis_agent"
        assert isinstance(logger, logging.Logger)
