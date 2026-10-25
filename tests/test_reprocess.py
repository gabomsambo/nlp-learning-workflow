"""Hermetic tests for ``cli reprocess``: selection, dry-run, per-paper failure."""

from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from nlp_pillars import reprocess
from nlp_pillars.cli import app
from nlp_pillars.llm_client import StructuredCall
from nlp_pillars.orchestrator import Orchestrator
from nlp_pillars.schemas import PaperRef

PILLAR = "neural-architectures-language"


def _paper(pid, pillar=PILLAR):
    return PaperRef(id=pid, title=f"Paper {pid}", authors=["A"], pillar_id=pillar)


def _call(cost):
    return StructuredCall(
        stage="x", provider="anthropic", model="claude-haiku-5-5", parsed=None,
        input_tokens=1000, output_tokens=100, cost_usd=cost,
    )


@pytest.fixture
def library():
    """Three papers: 'noted' has a note, 'a' and 'b' do not; 'b' has no text."""
    texts = {"noted": "text n", "a": "word " * 50, "b": None}
    with patch.object(reprocess.db, "list_all_papers",
                      return_value=[_paper("noted"), _paper("a"), _paper("b")]), \
         patch.object(reprocess.db, "list_noted_paper_ids", return_value=["noted"]), \
         patch.object(reprocess.db, "get_stored_paper_text",
                      side_effect=lambda pid: texts[pid]):
        yield


def test_missing_only_selects_only_papers_without_notes(library):
    cands = reprocess.select_candidates(missing_only=True)
    assert [(c.paper.id, c.state) for c in cands] == [("a", "ready"), ("b", "no_text")]


def test_without_missing_only_a_noted_paper_is_reported_not_regenerated(library):
    cands = {c.paper.id: c for c in reprocess.select_candidates()}
    assert cands["noted"].state == "has_note"
    assert "duplicate" in cands["noted"].detail


def test_missing_text_is_reported_not_skipped_silently(library):
    report = reprocess.run_reprocess(dry_run=True, missing_only=True)
    text = report.render()
    assert "no_text b" in text
    assert "would process a" in text


def test_unknown_paper_is_an_error(library):
    with pytest.raises(ValueError):
        reprocess.select_candidates(paper_id="nope")


def test_dry_run_writes_and_calls_nothing(library):
    factory = Mock()
    with patch.object(reprocess.db, "create_pipeline_run") as create, \
         patch.object(reprocess.db, "insert_note") as note:
        report = reprocess.run_reprocess(
            dry_run=True, missing_only=True, orchestrator_factory=factory,
        )
    factory.assert_not_called()
    create.assert_not_called()
    note.assert_not_called()
    assert report.estimate_usd > 0
    assert "No model call. Nothing written." in report.render()


def test_estimate_caps_at_the_token_budget():
    assert reprocess.estimate_paper_usd(10_000_000) == reprocess.estimate_paper_usd(60_000)
    assert reprocess.estimate_paper_usd(1_000) < reprocess.estimate_paper_usd(60_000)


def _agent(cost):
    agent = Mock()
    agent.last_call = None
    agent._cost = cost
    return agent


def test_one_failure_is_reported_and_the_rest_continue(library):
    orch = Mock(spec=Orchestrator)
    summarizer, synth, quiz = _agent(0), _agent(0), _agent(0)

    def generate(pillar_id, paper, parsed):
        if paper.id == "a":
            summarizer.last_call = _call(0.01)
            raise RuntimeError("model said no")
        summarizer.last_call = _call(0.01)
        synth.last_call = _call(0.02)
        quiz.last_call = _call(0.005)
        return Mock(concepts=["x"]), Mock(), [Mock()]

    orch._generate_learning_content.side_effect = generate
    orch._store_learning_content.return_value = (True, True, 1)
    papers = [_paper("a"), _paper("c")]
    with patch.object(reprocess.db, "list_all_papers", return_value=papers), \
         patch.object(reprocess.db, "list_noted_paper_ids", return_value=[]), \
         patch.object(reprocess.db, "get_stored_paper_text", return_value="word " * 20), \
         patch.object(reprocess.db, "create_pipeline_run",
                      return_value=Mock(id="run-1")) as create, \
         patch.object(reprocess.db, "start_pipeline_run"), \
         patch.object(reprocess.db, "update_pipeline_run_stage"), \
         patch.object(reprocess.db, "finish_pipeline_run") as finish:
        report = reprocess.run_reprocess(
            orchestrator_factory=Mock(return_value=orch),
            agents=(summarizer, synth, quiz),
        )
    statuses = {o.paper_id: o.status for o in report.outcomes}
    assert statuses == {"a": "failed", "c": "done"}
    # the failed paper's paid summarize call still counts; the other paper's all three
    assert report.cost_usd == pytest.approx(0.01 + 0.035)
    orch._mark_current_paper_stage_failed.assert_called_once()
    assert create.call_count == 2
    assert [c.args[1] for c in finish.call_args_list] == ["failed", "succeeded"]


def test_a_refused_insert_fails_the_paper(library):
    orch = Mock(spec=Orchestrator)
    orch._generate_learning_content.return_value = (Mock(concepts=[]), Mock(), [Mock(), Mock()])
    orch._store_learning_content.return_value = (True, False, 1)
    with patch.object(reprocess.db, "list_all_papers", return_value=[_paper("a")]), \
         patch.object(reprocess.db, "list_noted_paper_ids", return_value=[]), \
         patch.object(reprocess.db, "get_stored_paper_text", return_value="word " * 20), \
         patch.object(reprocess.db, "create_pipeline_run", return_value=None), \
         patch.object(reprocess.db, "update_pipeline_run_stage"):
        report = reprocess.run_reprocess(
            orchestrator_factory=Mock(return_value=orch),
            agents=(_agent(0), _agent(0), _agent(0)),
        )
    out = report.outcomes[0]
    assert out.status == "failed"
    assert "lesson" in out.detail and "quiz cards (1/2)" in out.detail
    assert "no run row" in out.detail


def test_cli_dry_run_exits_zero(library):
    result = CliRunner().invoke(app, ["reprocess", "--dry-run", "--missing-only"])
    assert result.exit_code == 0, result.output
    assert "would process a" in result.output
