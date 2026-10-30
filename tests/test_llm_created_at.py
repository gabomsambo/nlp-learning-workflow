"""Lessons and notes must not store a date the model invented.

The response schema omits created_at. Insert stamps the server clock. A model
that returns one anyway is ignored, including by the agent that just parsed it.
The one-off repair for rows already stored is the SQL the dates command prints.
It is not applied here.
"""

import re
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from nlp_pillars.agents.summarizer_agent import SummarizerAgentImpl
from nlp_pillars.agents.synthesis_agent import SynthesisAgentImpl
from nlp_pillars.cli import app
from nlp_pillars.created_at_repair import choose_real_timestamp, repair_sql_text
from nlp_pillars.db import _lesson_to_dict, _paper_note_to_dict
from nlp_pillars.schemas import (
    Lesson,
    LessonDraft,
    NoteDraft,
    PaperNote,
    PaperRef,
    ParsedPaper,
    PillarConfig,
    SummarizerInput,
    SynthesisInput,
)

INVENTED = datetime(2023, 10, 1, tzinfo=timezone.utc)


def _parsed(value):
    """One structured response from a mocked provider."""
    return SimpleNamespace(
        parsed_output=value,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        stop_reason="end_turn",
    )


def _note_draft():
    return NoteDraft(
        problem="Recurrence blocks parallel training.",
        method="Scaled dot-product attention.",
        findings=["BLEU improved."],
        limitations=["Quadratic cost."],
        future_work=["Restricted attention."],
        key_terms=["attention"],
        concepts=["attention"],
    )


def _lesson_draft():
    return LessonDraft(
        title="Attention replaces recurrence",
        content="The model attends over the whole sequence.",
        tl_dr="Attention is all you need.",
        takeaways=["No recurrence."],
        practice_ideas=["Mask the diagonal."],
        connections=[{"work": "Devlin et al. 2018", "relation": "uses", "why": "BERT."}],
    )


def _note() -> PaperNote:
    return PaperNote(
        paper_id="arxiv:1706.03762",
        pillar_id="neural-architectures-language",
        problem="Recurrence blocks parallel training.",
        method="Scaled dot-product attention.",
        findings=["BLEU improved."],
        limitations=["Quadratic cost."],
        future_work=["Restricted attention."],
        key_terms=["attention"],
        concepts=["attention"],
        created_at=INVENTED,
    )


def _lesson() -> Lesson:
    return Lesson(
        paper_id="arxiv:1706.03762",
        pillar_id="neural-architectures-language",
        title="Attention replaces recurrence",
        content="The model attends over the whole sequence.",
        tl_dr="Attention is all you need.",
        takeaways=["No recurrence."],
        practice_ideas=["Mask the diagonal."],
        connections=["BERT."],
        created_at=INVENTED,
    )


def _stamp_is_now(payload_created_at: str) -> None:
    stored = datetime.fromisoformat(payload_created_at)
    assert stored.tzinfo is not None
    assert stored != INVENTED
    assert abs((stored - datetime.now(timezone.utc)).total_seconds()) < 5


def test_llm_supplied_created_at_is_ignored():
    for model in (PaperNote, Lesson):
        schema = model.model_json_schema()
        assert "created_at" not in schema["properties"]
        assert "created_at" in model.model_fields

    note = _note()
    lesson = _lesson()
    # The field still parses if a model returns it. Storage must not trust that.
    assert note.created_at == INVENTED
    assert lesson.created_at == INVENTED

    note_row = _paper_note_to_dict(note)
    lesson_row = _lesson_to_dict(lesson)
    _stamp_is_now(note_row["created_at"])
    _stamp_is_now(lesson_row["created_at"])
    assert note.created_at == INVENTED
    assert lesson.created_at == INVENTED

    parsed = ParsedPaper(
        paper_ref=PaperRef(id="arxiv:1706.03762", title="Attention", authors=["Vaswani"]),
        full_text="Attention is all you need.",
        chunks=["Attention is all you need."],
    )
    summarizer = SummarizerAgentImpl(Mock(), "claude-haiku-5-5")
    summarizer.client.messages.parse.return_value = _parsed(_note_draft())
    summarized = summarizer.run(
        SummarizerInput(
            parsed_paper=parsed,
            pillar_id="neural-architectures-language",
            concept_vocabulary=["attention"],
        )
    )
    assert summarized.created_at != INVENTED
    assert summarized.created_at.year != 2023
    # The model is never asked for a date, an id or a pillar.
    assert summarizer.client.messages.parse.call_args.kwargs["output_format"] is NoteDraft

    synthesis = SynthesisAgentImpl(Mock(), "claude-haiku-5-5")
    synthesis.client.messages.parse.return_value = _parsed(_lesson_draft())
    pillar = PillarConfig(id="neural-architectures-language", name="Neural", goal="Read papers.")
    synthesized = synthesis.run(SynthesisInput(paper_note=_note(), pillar_config=pillar))
    assert synthesized.created_at != INVENTED
    assert synthesized.created_at.year != 2023
    assert synthesis.client.messages.parse.call_args.kwargs["output_format"] is LessonDraft


def test_repair_chooses_processing_time_before_later_fallbacks():
    stage = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    quiz = datetime(2026, 8, 29, 12, 5, tzinfo=timezone.utc)
    added = datetime(2026, 8, 29, 11, 0, tzinfo=timezone.utc)
    created = datetime(2026, 8, 29, 10, 0, tzinfo=timezone.utc)

    assert choose_real_timestamp(
        stage_finished_at=stage,
        quiz_created_at=quiz,
        paper_added_at=added,
        paper_created_at=created,
    ) == stage
    assert choose_real_timestamp(quiz_created_at=quiz, paper_added_at=added, paper_created_at=created) == quiz
    assert choose_real_timestamp(paper_added_at=added, paper_created_at=created) == added
    assert choose_real_timestamp(paper_created_at=created) == created
    assert choose_real_timestamp() is None


def test_repair_sql_matches_that_order_and_the_command_does_not_apply_it():
    code = re.sub(r"--.*?$", "", repair_sql_text(), flags=re.M)
    coalesces = re.findall(
        r"COALESCE\(\s*stage\.finished_at,\s*quiz\.created_at,\s*p\.added_at,\s*p\.created_at\s*\)",
        code,
    )
    assert coalesces
    assert len(coalesces) == 2
    assert "s.name = 'summarize'" in code
    assert "s.name = 'synthesize'" in code
    assert "n2.created_at" not in code

    with patch("nlp_pillars.db.get_client", side_effect=AssertionError("connected")):
        result = CliRunner().invoke(app, ["dates", "repair"])

    assert result.exit_code == 0, result.output
    assert "UPDATE notes" in result.output
    assert "UPDATE lessons" in result.output
    assert "was not applied" in result.output
