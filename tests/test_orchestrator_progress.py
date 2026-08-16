"""Tests for the Orchestrator's on_stage progress callback and cooperative cancel.

Fixtures here are deliberately local rather than imported from
tests/integration/test_orchestrator.py: that module's `sample_lesson` builds a Lesson
without the now-required `title` and `content`, which is part of the known-red
baseline. Reusing it would make these tests fail for a reason that has nothing to do
with what they cover.

Patch targets follow the house style — the names as imported INTO orchestrator, not
their defining modules.
"""

import threading
from unittest.mock import Mock, patch

import pytest

from nlp_pillars.orchestrator import Orchestrator, RunCancelledError
from nlp_pillars.schemas import (
    RUN_DAILY_STAGES,
    Lesson,
    PaperNote,
    PaperRef,
    ParsedPaper,
    QuizCard,
    StageName,
    StageStatus,
)

PILLAR = "neural-architectures-language"


@pytest.fixture
def paper():
    return PaperRef(
        id="arxiv:1234.5678",
        title="A Test Paper",
        authors=["A. Author"],
        url_pdf="https://arxiv.org/pdf/1234.5678",
    )


@pytest.fixture
def parsed(paper):
    return ParsedPaper(
        paper_ref=paper,
        full_text="Body text about attention.",
        chunks=["Body text about attention."],
    )


@pytest.fixture
def note():
    return PaperNote(
        paper_id="arxiv:1234.5678",
        pillar_id=PILLAR,
        problem="p",
        method="m",
        findings=["f"],
        limitations=["l"],
        future_work=["w"],
        key_terms=["k"],
    )


@pytest.fixture
def lesson():
    return Lesson(
        paper_id="arxiv:1234.5678",
        pillar_id=PILLAR,
        title="A Test Lesson",
        content="Lesson content.",
        tl_dr="Short.",
        takeaways=["t"],
        practice_ideas=["p"],
        connections=["c"],
    )


@pytest.fixture
def cards():
    return [
        QuizCard(
            paper_id="arxiv:1234.5678",
            pillar_id=PILLAR,
            question="q?",
            answer="a",
        )
    ]


class StageRecorder:
    """A progress sink that just remembers what it was told."""

    def __init__(self):
        self.events = []

    def __call__(self, name, status, detail=None):
        self.events.append((name, status, detail))

    def started(self):
        return [n for n, s, _ in self.events if s == StageStatus.RUNNING.value]

    def completed(self):
        return [n for n, s, _ in self.events if s == StageStatus.COMPLETED.value]

    def failed(self):
        return [n for n, s, _ in self.events if s == StageStatus.FAILED.value]


def _build(on_stage=None, cancel=None, enable_quiz=True, parsed=None, note=None,
           lesson=None, cards=None, paper=None):
    """Construct an Orchestrator with every external dependency stubbed."""
    orch = Orchestrator(enable_quiz=enable_quiz, on_stage=on_stage, cancel=cancel)
    # searxng_tool / arxiv_tool / ingest_agent are real instances built in __init__,
    # so they are replaced on the instance rather than patched on the class.
    orch.searxng_tool.search = Mock(return_value=[paper] if paper else [])
    orch.arxiv_tool.search = Mock(return_value=[])
    orch.ingest_agent.ingest = Mock(return_value=parsed)
    return orch


@patch("nlp_pillars.orchestrator.vectors")
@patch("nlp_pillars.orchestrator.db")
@patch("nlp_pillars.orchestrator.QuizAgent")
@patch("nlp_pillars.orchestrator.SynthesisAgent")
@patch("nlp_pillars.orchestrator.SummarizerAgent")
@patch("nlp_pillars.orchestrator.DiscoveryAgent")
def test_all_eleven_stages_fire_in_order(
    mock_discovery, mock_summ, mock_synth, mock_quiz, mock_db, mock_vectors,
    paper, parsed, note, lesson, cards,
):
    """A clean run reports every stage, started and completed, in declaration order."""
    mock_db.get_recent_notes.return_value = []
    mock_db.queue_add_candidates.return_value = 1
    mock_db.queue_pop_next.return_value = [paper]
    mock_summ.run.return_value = note
    mock_synth.run.return_value = lesson
    mock_quiz.run.return_value = cards
    mock_vectors.upsert_text.return_value = 3

    recorder = StageRecorder()
    orch = _build(on_stage=recorder, parsed=parsed, paper=paper)
    result = orch.run_daily(PILLAR, papers_limit=1)

    assert result.success is True

    expected = [s.value for s in RUN_DAILY_STAGES]
    # PROCESS starts once for the phase and again per paper, so dedupe before
    # comparing order.
    started_unique = list(dict.fromkeys(recorder.started()))
    assert started_unique == expected

    for stage in expected:
        assert stage in recorder.completed(), f"{stage} never completed"
    assert recorder.failed() == []


@patch("nlp_pillars.orchestrator.vectors")
@patch("nlp_pillars.orchestrator.db")
@patch("nlp_pillars.orchestrator.SummarizerAgent")
@patch("nlp_pillars.orchestrator.DiscoveryAgent")
def test_failure_marks_the_stage_that_died(
    mock_discovery, mock_summ, mock_db, mock_vectors, paper, parsed,
):
    """A paper that blows up in summarize marks SUMMARIZE failed, not something else.

    This is the case _process_paper cannot report for itself — it has no internal
    try/except, so the surrounding handler has to know which stage was in flight.
    """
    mock_db.get_recent_notes.return_value = []
    mock_db.queue_add_candidates.return_value = 1
    mock_db.queue_pop_next.return_value = [paper]
    mock_summ.run.side_effect = RuntimeError("model exploded")

    recorder = StageRecorder()
    orch = _build(on_stage=recorder, parsed=parsed, paper=paper)
    result = orch.run_daily(PILLAR, papers_limit=1)

    assert result.success is False
    assert recorder.failed() == [StageName.SUMMARIZE.value]
    # Everything before it still completed.
    assert StageName.INGEST.value in recorder.completed()
    # And nothing after it was even attempted.
    assert StageName.PERSIST.value not in recorder.started()


@patch("nlp_pillars.orchestrator.vectors")
@patch("nlp_pillars.orchestrator.db")
@patch("nlp_pillars.orchestrator.DiscoveryAgent")
def test_cancel_event_stops_at_the_first_boundary(
    mock_discovery, mock_db, mock_vectors,
):
    """An already-set cancel event stops the run before any stage begins."""
    event = threading.Event()
    event.set()

    recorder = StageRecorder()
    orch = _build(on_stage=recorder, cancel=event)
    result = orch.run_daily(PILLAR, papers_limit=1)

    assert result.success is False
    assert recorder.events == []


@patch("nlp_pillars.orchestrator.vectors")
@patch("nlp_pillars.orchestrator.db")
@patch("nlp_pillars.orchestrator.QuizAgent")
@patch("nlp_pillars.orchestrator.SynthesisAgent")
@patch("nlp_pillars.orchestrator.SummarizerAgent")
@patch("nlp_pillars.orchestrator.DiscoveryAgent")
def test_cancel_midway_raises_run_cancelled(
    mock_discovery, mock_summ, mock_synth, mock_quiz, mock_db, mock_vectors,
    paper, parsed, note,
):
    """Setting the event during a run stops it at the next boundary, not instantly."""
    mock_db.get_recent_notes.return_value = []
    mock_db.queue_add_candidates.return_value = 1
    mock_db.queue_pop_next.return_value = [paper]
    mock_summ.run.return_value = note

    event = threading.Event()
    recorder = StageRecorder()

    def cancel_once_summarizing(name, status, detail=None):
        recorder(name, status, detail)
        if name == StageName.SUMMARIZE.value and status == StageStatus.COMPLETED.value:
            event.set()

    orch = _build(on_stage=cancel_once_summarizing, cancel=event,
                  parsed=parsed, paper=paper)

    with pytest.raises(RunCancelledError):
        orch._process_paper(PILLAR, paper)

    # It got as far as summarize and stopped before synthesize.
    assert StageName.SUMMARIZE.value in recorder.completed()
    assert StageName.SYNTHESIZE.value not in recorder.started()


@patch("nlp_pillars.orchestrator.vectors")
@patch("nlp_pillars.orchestrator.db")
@patch("nlp_pillars.orchestrator.SynthesisAgent")
@patch("nlp_pillars.orchestrator.SummarizerAgent")
@patch("nlp_pillars.orchestrator.DiscoveryAgent")
def test_quiz_disabled_is_skipped_not_silently_absent(
    mock_discovery, mock_summ, mock_synth, mock_db, mock_vectors,
    paper, parsed, note, lesson,
):
    """With quiz off, QUIZ reports 'skipped'.

    The UI must be able to tell "turned off" from "not reached yet".
    """
    mock_db.get_recent_notes.return_value = []
    mock_summ.run.return_value = note
    mock_synth.run.return_value = lesson
    mock_vectors.upsert_text.return_value = 1

    recorder = StageRecorder()
    orch = _build(on_stage=recorder, enable_quiz=False, parsed=parsed, paper=paper)
    orch._process_paper(PILLAR, paper)

    skipped = [n for n, s, _ in recorder.events if s == StageStatus.SKIPPED.value]
    assert skipped == [StageName.QUIZ.value]


@patch("nlp_pillars.orchestrator.vectors")
@patch("nlp_pillars.orchestrator.db")
@patch("nlp_pillars.orchestrator.QuizAgent")
@patch("nlp_pillars.orchestrator.SynthesisAgent")
@patch("nlp_pillars.orchestrator.SummarizerAgent")
@patch("nlp_pillars.orchestrator.DiscoveryAgent")
def test_no_callback_behaves_exactly_as_before(
    mock_discovery, mock_summ, mock_synth, mock_quiz, mock_db, mock_vectors,
    paper, parsed, note, lesson, cards,
):
    """The default must be a true no-op — the CLI and scheduler pass nothing."""
    mock_db.get_recent_notes.return_value = []
    mock_db.queue_add_candidates.return_value = 1
    mock_db.queue_pop_next.return_value = [paper]
    mock_summ.run.return_value = note
    mock_synth.run.return_value = lesson
    mock_quiz.run.return_value = cards
    mock_vectors.upsert_text.return_value = 3

    orch = _build(parsed=parsed, paper=paper)  # no on_stage, no cancel
    result = orch.run_daily(PILLAR, papers_limit=1)

    assert result.success is True
    assert result.papers_processed == [paper.id]


@patch("nlp_pillars.orchestrator.vectors")
@patch("nlp_pillars.orchestrator.db")
@patch("nlp_pillars.orchestrator.QuizAgent")
@patch("nlp_pillars.orchestrator.SynthesisAgent")
@patch("nlp_pillars.orchestrator.SummarizerAgent")
@patch("nlp_pillars.orchestrator.DiscoveryAgent")
def test_broken_callback_does_not_break_the_run(
    mock_discovery, mock_summ, mock_synth, mock_quiz, mock_db, mock_vectors,
    paper, parsed, note, lesson, cards,
):
    """Losing the progress display is bad; losing the pipeline with it is worse."""
    mock_db.get_recent_notes.return_value = []
    mock_db.queue_add_candidates.return_value = 1
    mock_db.queue_pop_next.return_value = [paper]
    mock_summ.run.return_value = note
    mock_synth.run.return_value = lesson
    mock_quiz.run.return_value = cards
    mock_vectors.upsert_text.return_value = 3

    def exploding_sink(name, status, detail=None):
        raise RuntimeError("progress sink is down")

    orch = _build(on_stage=exploding_sink, parsed=parsed, paper=paper)
    result = orch.run_daily(PILLAR, papers_limit=1)

    assert result.success is True
