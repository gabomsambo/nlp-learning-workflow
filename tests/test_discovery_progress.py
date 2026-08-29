"""Tests for discovery's progress reporting and its honest failure states.

The behaviour under test is a distinction that did not exist before: a discovery step
that reports nothing found versus one that could not run. Both used to arrive as an
empty list — the query LLM falling back to focus areas, a rate-limited arXiv, an
unreachable vector store and a genuinely empty library were indistinguishable on
screen, which is what made a thirty-second blind wait actively misleading rather than
merely dull.

Local fixtures and a basename unique across tests/ for the reasons given in
tests/test_orchestrator_progress.py. Nothing here touches a network, a database or a
vector store.
"""

from unittest.mock import Mock, patch

import pytest

from nlp_pillars.orchestrator import Orchestrator, RunCancelledError, SourceResult
from nlp_pillars.schemas import (
    DISCOVER_STAGES,
    DiscoveryCandidate,
    PaperNote,
    PaperRef,
    SearchQuery,
    StageName,
    StageStatus,
)

PILLAR = "neural-architectures-language"


@pytest.fixture
def paper():
    return PaperRef(
        id="arxiv:2401.00001",
        title="Mamba: Linear-Time Sequence Modeling",
        authors=["A. Author"],
        url_pdf="https://arxiv.org/pdf/2401.00001",
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


class StageRecorder:
    """A progress sink that just remembers what it was told."""

    def __init__(self):
        self.events = []

    def __call__(self, name, status, detail=None):
        self.events.append((name, status, detail))

    def status_of(self, stage):
        """The last status reported for a stage, or None if it never reported."""
        seen = [s for n, s, _ in self.events if n == stage.value]
        return seen[-1] if seen else None

    def detail_of(self, stage):
        """The detail on that last report."""
        seen = [d for n, s, d in self.events if n == stage.value and s != "running"]
        return seen[-1] if seen else None

    def started(self):
        return [n for n, s, _ in self.events if s == StageStatus.RUNNING.value]


def _queries(*texts):
    """A DiscoveryAgent.run() return value carrying these query strings."""
    return Mock(queries=[Mock(query=t) for t in texts])


def _orchestrator(on_stage=None, cancel=None):
    """An Orchestrator whose four discovery sources are all inert by default."""
    orch = Orchestrator(enable_quiz=False, on_stage=on_stage, cancel=cancel)
    orch.arxiv_tool.search = Mock(return_value=[])
    orch.semantic_scholar_tool = Mock()
    orch.semantic_scholar_tool.search = Mock(return_value=[])
    orch.vector_search_tool = Mock()
    orch.vector_search_tool.search_similar_papers = Mock(return_value=[])
    return orch


@pytest.fixture
def mocked(request):
    """Patch the orchestrator's module-level collaborators for a discovery run."""
    with patch("nlp_pillars.orchestrator.db") as db, \
         patch("nlp_pillars.orchestrator.vectors") as vectors, \
         patch("nlp_pillars.orchestrator.DiscoveryAgent") as agent:
        db.get_recent_notes.return_value = []
        db.get_citation_network_papers.return_value = []
        agent.run.return_value = _queries("state space models", "linear attention")
        yield db, vectors, agent


# ------------------------------------------------------------------ the happy path


def test_every_step_reports_itself_with_a_count(mocked, paper):
    db, vectors, agent = mocked
    recorder = StageRecorder()
    orch = _orchestrator(on_stage=recorder)
    orch.arxiv_tool.search = Mock(return_value=[paper])

    orch.run_discovery_with_selection(PILLAR, limit=10)

    # Citations do not run: this pillar has no recent papers. Every other step does.
    expected = [s.value for s in DISCOVER_STAGES
                if s is not StageName.DISCOVER_CITATIONS]
    assert recorder.started() == expected
    for stage in DISCOVER_STAGES:
        if stage is StageName.DISCOVER_CITATIONS:
            continue
        assert recorder.status_of(stage) == StageStatus.COMPLETED.value

    # The count is the point: "it is spinning" versus "it is working, and here it is".
    # Two, because both queries hit arXiv and both returned this paper — a source
    # reports its own hits, and the duplicate is removed later, by the step that says
    # so.
    assert recorder.detail_of(StageName.DISCOVER_ARXIV) == "2 found"
    assert recorder.detail_of(StageName.DISCOVER_VECTORS) == "0 found"
    assert recorder.detail_of(StageName.DISCOVER_RANK) == "1 kept from 2 hit(s)"


def test_the_generated_queries_are_reported(mocked):
    """They arrive seconds in and are the first evidence the run is alive."""
    db, vectors, agent = mocked
    recorder = StageRecorder()

    _orchestrator(on_stage=recorder).run_discovery_with_selection(PILLAR)

    detail = recorder.detail_of(StageName.DISCOVER_QUERIES)
    assert '"state space models"' in detail
    assert '"linear attention"' in detail


def test_a_pillar_with_recent_papers_keeps_the_citation_step(mocked, note):
    db, vectors, agent = mocked
    db.get_recent_notes.return_value = [note]
    recorder = StageRecorder()

    _orchestrator(on_stage=recorder).run_discovery_with_selection(PILLAR)

    assert StageName.DISCOVER_CITATIONS.value in recorder.started()
    assert "1 recent paper(s) read" == recorder.detail_of(StageName.DISCOVER_CONTEXT)


def test_a_pillar_with_no_recent_papers_drops_the_citation_step(mocked):
    """Not skipped — dropped. A step that will not run should not be on screen at
    all, and 'skipped' still renders as a step that was considered."""
    db, vectors, agent = mocked
    recorder = StageRecorder()

    _orchestrator(on_stage=recorder).run_discovery_with_selection(PILLAR)

    assert recorder.status_of(StageName.DISCOVER_CITATIONS) == StageStatus.DROPPED.value
    assert StageName.DISCOVER_CITATIONS.value not in recorder.started()


# --------------------------------------------------- the three invisible failures


def test_a_query_fallback_says_so_rather_than_passing_focus_areas_off(mocked):
    """Regression. The fallback was a log line, so the page showed the pillar's own
    focus areas as though the discovery agent had written them."""
    db, vectors, agent = mocked
    agent.run.side_effect = RuntimeError("connection reset")
    agent._blend_topics.return_value = ["efficient transformers", "long context"]

    recorder = StageRecorder()
    orch = _orchestrator(on_stage=recorder)
    candidates = orch.run_discovery_with_selection(PILLAR)

    assert recorder.status_of(StageName.DISCOVER_QUERIES) == StageStatus.FAILED.value
    detail = recorder.detail_of(StageName.DISCOVER_QUERIES)
    assert "couldn't reach the model" in detail
    assert "focus areas instead" in detail
    assert "connection reset" in detail
    # The row shows what was substituted, so the queries on screen are never passed
    # off as the agent's.
    assert '"efficient transformers"' in detail
    # The run-level line stays one sentence: it sits beside the run status, where a
    # full query list pushes everything else off the screen.
    run_level = orch.infra_errors[0]["message"]
    assert "couldn't reach the model" in run_level
    assert "efficient transformers" not in run_level
    # And the run carried on regardless, which is the whole reason for the fallback.
    assert candidates == []


def test_a_rate_limited_source_is_not_reported_as_zero_results(mocked):
    """arXiv throttles bursts. The exception used to be swallowed per query, leaving
    a short list and no reason — the one failure with an obvious remedy."""
    db, vectors, agent = mocked
    recorder = StageRecorder()
    orch = _orchestrator(on_stage=recorder)
    orch.arxiv_tool.search = Mock(
        side_effect=ConnectionError("HTTP 429 Too Many Requests")
    )

    orch.run_discovery_with_selection(PILLAR)

    assert recorder.status_of(StageName.DISCOVER_ARXIV) == StageStatus.FAILED.value
    detail = recorder.detail_of(StageName.DISCOVER_ARXIV)
    assert "rate-limited" in detail
    assert "2 of 2 queries failed" in detail
    assert "0 found" != detail


def test_one_failed_query_out_of_two_still_reports_what_it_found(mocked, paper):
    db, vectors, agent = mocked
    recorder = StageRecorder()
    orch = _orchestrator(on_stage=recorder)
    orch.arxiv_tool.search = Mock(
        side_effect=[[paper], ConnectionError("HTTP 429 Too Many Requests")]
    )

    orch.run_discovery_with_selection(PILLAR)

    # Completed, because it did find something — but it says what it lost.
    assert recorder.status_of(StageName.DISCOVER_ARXIV) == StageStatus.COMPLETED.value
    detail = recorder.detail_of(StageName.DISCOVER_ARXIV)
    assert detail.startswith("1 found · ")
    assert "1 of 2 queries failed" in detail


def test_an_unreachable_vector_store_is_not_reported_as_zero_results(mocked):
    """vectors.search_similar() raises for the "this code disagrees with its server"
    cases — a removed client method, or the 4xx Qdrant strict mode answers when the
    pillar_id payload index is missing. Neither is "nothing matched"."""
    db, vectors, agent = mocked
    recorder = StageRecorder()
    orch = _orchestrator(on_stage=recorder)
    orch.vector_search_tool.search_similar_papers = Mock(
        side_effect=RuntimeError("Qdrant rejected the search (400): Index required")
    )

    orch.run_discovery_with_selection(PILLAR)

    assert recorder.status_of(StageName.DISCOVER_VECTORS) == StageStatus.FAILED.value
    assert "Index required" in recorder.detail_of(StageName.DISCOVER_VECTORS)


def test_vectors_switched_off_is_distinguished_from_an_empty_library(mocked):
    """QDRANT_URL unset disables vector operations with a WARNING nobody reads."""
    db, vectors, agent = mocked
    vectors.get_client.return_value = None
    recorder = StageRecorder()

    _orchestrator(on_stage=recorder).run_discovery_with_selection(PILLAR)

    assert recorder.status_of(StageName.DISCOVER_VECTORS) == StageStatus.FAILED.value
    assert "QDRANT_URL" in recorder.detail_of(StageName.DISCOVER_VECTORS)


def test_a_failed_source_does_not_stop_the_others(mocked, paper):
    db, vectors, agent = mocked
    orch = _orchestrator()
    orch.vector_search_tool.search_similar_papers = Mock(
        side_effect=RuntimeError("qdrant is down")
    )
    orch.arxiv_tool.search = Mock(return_value=[paper])

    candidates = orch.run_discovery_with_selection(PILLAR)

    assert [c.paper.id for c in candidates] == [paper.id]
    # ...and the run remembers what went wrong, for the run-level summary.
    assert any("qdrant is down" in e["message"] for e in orch.infra_errors)


def test_a_clean_run_records_no_problems(mocked, paper):
    db, vectors, agent = mocked
    orch = _orchestrator()
    orch.arxiv_tool.search = Mock(return_value=[paper])

    orch.run_discovery_with_selection(PILLAR)

    assert orch.infra_errors == []


# ------------------------------------------------------- the callers that predate this


def test_the_progress_sink_is_optional(mocked, paper):
    """cli.py calls this with no sink at all; it must behave exactly as before."""
    db, vectors, agent = mocked
    orch = _orchestrator()
    orch.arxiv_tool.search = Mock(return_value=[paper])

    candidates = orch.run_discovery_with_selection(PILLAR, limit=5)

    assert [c.paper.id for c in candidates] == [paper.id]
    assert all(isinstance(c, DiscoveryCandidate) for c in candidates)


def test_a_broken_progress_sink_cannot_take_the_run_down(mocked, paper):
    db, vectors, agent = mocked

    def explode(name, status, detail=None):
        raise ValueError("the display is on fire")

    orch = _orchestrator(on_stage=explode)
    orch.arxiv_tool.search = Mock(return_value=[paper])

    assert len(orch.run_discovery_with_selection(PILLAR)) == 1


def test_cancelling_stops_discovery_at_a_stage_boundary(mocked):
    import threading

    db, vectors, agent = mocked
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(RunCancelledError):
        _orchestrator(cancel=cancel).run_discovery_with_selection(PILLAR)


# --------------------------------------------------------------- the source helpers


def test_a_retry_wrapper_does_not_hide_the_real_failure(mocked):
    """Measured live: three 403s from Semantic Scholar reached the page as
    `RetryError[<Future at 0x74f0… state=finished raised HTTPStatusError>]`, which
    tells the reader nothing at all. The status is the part they can act on; the 300
    characters of encoded request URL after it are not."""
    class FakeAttempt:
        def exception(self):
            return RuntimeError(
                "Client error '403 Forbidden' for url "
                "'https://api.semanticscholar.org/graph/v1/paper/search?query=x"
                "&fields=paperId%2CexternalIds%2Ctitle&limit=5'"
            )

    class FakeRetryError(Exception):
        last_attempt = FakeAttempt()

        def __str__(self):
            return "RetryError[<Future at 0x1 state=finished raised HTTPStatusError>]"

    db, vectors, agent = mocked
    recorder = StageRecorder()
    orch = _orchestrator(on_stage=recorder)
    orch.semantic_scholar_tool.search = Mock(side_effect=FakeRetryError())

    orch.run_discovery_with_selection(PILLAR)

    detail = recorder.detail_of(StageName.DISCOVER_S2)
    assert "403 Forbidden" in detail
    assert "RetryError" not in detail
    assert "semanticscholar.org" not in detail


def test_a_search_helper_reports_its_own_unavailability(mocked):
    """A tool that failed to construct is not an empty result set."""
    db, vectors, agent = mocked
    orch = _orchestrator()
    orch.semantic_scholar_tool = None

    outcome = orch._search_semantic_scholar(["anything"])

    assert isinstance(outcome, SourceResult)
    assert outcome.candidates == []
    assert "unavailable" in outcome.failures[0]


def test_arxiv_search_still_builds_pillar_scoped_queries(mocked, paper):
    """The instrumentation must not have changed what is actually searched for."""
    db, vectors, agent = mocked
    orch = _orchestrator()
    orch.arxiv_tool.search = Mock(return_value=[paper])

    outcome = orch._search_arxiv_candidates(PILLAR, ["mamba", "rwkv", "ignored-third"])

    sent = [call.args[0] for call in orch.arxiv_tool.search.call_args_list]
    assert [q.query for q in sent] == ["mamba", "rwkv"]  # still capped at two
    assert all(isinstance(q, SearchQuery) and q.pillar_id == PILLAR for q in sent)
    assert outcome.failures == []
