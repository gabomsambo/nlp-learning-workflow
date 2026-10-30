"""works sync: identity resolution, typed citations, idempotence, dry-run.

The Semantic Scholar tool is ``Mock(spec=SemanticScholarTool)`` answering from
recorded responses (tests/fixtures/semantic_scholar). The store is an in-memory
stand-in for ``apply_work_batch`` with the same rules as the SQL: a write never
blanks a stored value, an ids-only edge never replaces a typed one.
"""

import copy
import itertools
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from nlp_pillars.tools.semantic_scholar_tool import (
    CitationEdge,
    CiterResult,
    EdgePage,
    SemanticScholarError,
    SemanticScholarTool,
    SemanticScholarUnavailableError,
    work_record_from_s2,
)
from nlp_pillars.work_ids import best_lookup_key, s2_lookup_id
from nlp_pillars.works import (
    SOURCE,
    SOURCE_IDS_ONLY,
    Batch,
    LibraryPaper,
    Overlay,
    WorkIndex,
    record_keys,
    run_works_sync,
)
from nlp_pillars.works_store import WorksStoreError

FIXTURES = Path(__file__).parent / "fixtures" / "semantic_scholar"
ATTENTION = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"


def load(name):
    return json.loads((FIXTURES / name).read_text())


class FakeWorks:
    """What ``apply_work_batch`` leaves behind, kept in dicts."""

    def __init__(self):
        self.works = {}
        self.ids = {}
        self.edges = {}
        self.papers = {}
        self.calls = []
        self.fail_call = None

    def apply(self, batch):
        self.calls.append(batch)
        if self.fail_call is not None and len(self.calls) == self.fail_call:
            raise WorksStoreError("write refused")
        for row in batch.works:
            held = self.works.setdefault(row["id"], {})
            for name, value in row.items():
                if value is not None and name not in {"described", "references_synced", "citers_synced"}:
                    held[name] = value
            for flag in ("described", "references_synced", "citers_synced"):
                held[flag] = held.get(flag, False) or bool(row.get(flag))
        for row in batch.ids:
            self.ids.setdefault((row["kind"], row["value"]), row["work_id"])
        for row in batch.edges:
            key = (row["citing"], row["cited"])
            if row["detail"] or key not in self.edges:
                self.edges[key] = dict(row)
        for row in batch.papers:
            held = self.papers.setdefault(row["id"], {})
            held["work_id"] = row["work_id"] or held.get("work_id")
            if row["citation_count"] is not None:
                held["citation_count"] = row["citation_count"]
        return {}

    def citer_work_ids(self, work_id):
        return {
            citing
            for (citing, cited), row in self.edges.items()
            if cited == work_id and row["source"] == SOURCE
        }

    def index(self):
        return WorkIndex(dict(self.ids))

    def library(self, ids):
        return [
            LibraryPaper(id=i, work_id=self.papers.get(i, {}).get("work_id"))
            for i in ids
        ]

    def snapshot(self):
        return (
            copy.deepcopy(self.ids),
            {key: row["source"] for key, row in self.edges.items()},
            set(self.works),
        )


def counter():
    numbers = itertools.count(1)
    return lambda: f"00000000-0000-4000-8000-{next(numbers):012d}"


@pytest.fixture
def records():
    batch = load("paper_batch.json")
    return {
        "attention": work_record_from_s2(batch[0]),
        "explanation": work_record_from_s2(batch[1]),
        "bpe": work_record_from_s2(batch[2]),
    }


def build_tool(records):
    """A tool that answers from the recorded batch, references and citations."""
    explanation, bpe = records["explanation"], records["bpe"]
    by_lookup = {}
    for record in records.values():
        for key in record_keys(record):
            lookup = s2_lookup_id(key)
            if lookup:
                by_lookup[lookup] = record

    page = load("references_1902.10186.json")
    explanation_refs = [
        CitationEdge(
            other=work_record_from_s2(row["citedPaper"]),
            is_influential=row["isInfluential"],
            intents=row["intents"],
            contexts=row["contexts"],
        )
        for row in page["data"]
    ]
    # Attention cites BPE (the real S2 list also does; this keeps the fixture short)
    attention_refs = [CitationEdge(other=bpe, intents=["methodology"], contexts=["BPE segments rare words."])]
    refs = {
        ATTENTION: attention_refs,
        explanation.s2_paper_id: explanation_refs,
        bpe.s2_paper_id: [],
    }
    citers_page = load("citations_1508.07909.json")
    bpe_citers = [
        CitationEdge(
            other=work_record_from_s2(row["citingPaper"]),
            is_influential=row["isInfluential"],
            intents=row["intents"],
            contexts=row["contexts"],
        )
        for row in citers_page["data"]
    ]
    citers = {bpe.s2_paper_id: bpe_citers, ATTENTION: [], explanation.s2_paper_id: []}

    tool = Mock(spec=SemanticScholarTool)
    tool.get_work_records.side_effect = lambda ids: [by_lookup.get(i) for i in ids]
    tool.get_reference_edges.side_effect = lambda lookup: EdgePage(edges=list(refs[lookup]))

    def get_citers(lookup, known_ids=(), max_items=1000):
        edges = [e for e in citers[lookup] if e.other.s2_paper_id not in set(known_ids)]
        return CiterResult(
            edges=edges[:max_items],
            stopped_at_known=bool(set(known_ids) & {e.other.s2_paper_id for e in citers[lookup]}),
        )

    tool.get_citer_edges.side_effect = get_citers
    return tool


LIBRARY = ["arxiv:1706.03762", "arxiv:1902.10186", "arxiv:1508.07909", "file:2dd76e910fbc"]


def sync(tool, store, ids=LIBRARY, **kw):
    return run_works_sync(
        store.library(ids), store.index(), tool, store, dry_run=False, new_id=kw.pop("new_id", counter()), **kw
    )


# -- identity and citations -------------------------------------------------

def test_library_and_s2_spellings_resolve_to_one_work(records):
    store, tool = FakeWorks(), build_tool(records)
    report = sync(tool, store)

    attention_work = store.papers["arxiv:1706.03762"]["work_id"]
    # S2's reference list names it by its paperId and ArXiv 1706.03762; the library
    # by arxiv:1706.03762. They are one work with all of its aliases.
    assert store.ids[("arxiv", "1706.03762")] == attention_work
    assert store.ids[("s2", ATTENTION)] == attention_work
    assert store.ids[("corpus", "13756489")] == attention_work
    works_with_that_arxiv = {w for (k, v), w in store.ids.items() if (k, v) == ("arxiv", "1706.03762")}
    assert works_with_that_arxiv == {attention_work}
    assert report.works_created == len(store.works)  # nothing was created twice


def test_citations_between_library_papers_become_edges_between_library_works(records):
    store = FakeWorks()
    report = sync(build_tool(records), store)

    work = {paper: row["work_id"] for paper, row in store.papers.items()}
    explanation, attention, bpe = (
        work["arxiv:1902.10186"], work["arxiv:1706.03762"], work["arxiv:1508.07909"],
    )
    assert (explanation, attention) in store.edges
    assert (attention, bpe) in store.edges
    assert report.library_edges == 2
    edge = store.edges[(explanation, attention)]
    assert edge["intents"] == ["background", "methodology"]
    assert edge["contexts"] and edge["detail"] is True and edge["source"] == SOURCE


def test_edges_keep_their_evidence_and_citers_are_stored(records):
    store = FakeWorks()
    report = sync(build_tool(records), store)
    assert report.reference_edges == 33 + 1
    assert report.citer_edges == 6
    recorded = (
        load("references_1902.10186.json")["data"] + load("citations_1508.07909.json")["data"]
    )
    influential = [row for row in store.edges.values() if row["is_influential"]]
    assert len(influential) == sum(1 for row in recorded if row["isInfluential"]) > 1
    assert all(row["source"] == SOURCE for row in store.edges.values())


def test_papers_get_their_work_and_citation_count(records):
    store = FakeWorks()
    report = sync(build_tool(records), store)
    assert store.papers["arxiv:1706.03762"]["citation_count"] == 200329
    assert report.counts_written == 3
    assert "file:2dd76e910fbc" not in store.papers  # the captain's document has no work
    assert report.unresolvable == ["file:2dd76e910fbc"]
    assert report.linked == 0 and report.resolved == 3


def test_a_paper_s2_does_not_know_stays_unlinked_and_is_reported(records):
    store, tool = FakeWorks(), build_tool(records)
    report = sync(tool, store, ids=[*LIBRARY, "arxiv:9999.99999"])
    assert report.not_in_s2 == ["arxiv:9999.99999"]
    assert "arxiv:9999.99999" not in store.papers


def test_a_second_run_changes_nothing(records):
    store = FakeWorks()
    first = sync(build_tool(records), store)
    before = store.snapshot()

    again = sync(build_tool(records), store, new_id=counter())
    assert store.snapshot() == before
    assert again.works_created == 0 and again.aliases_added == 0
    assert again.alias_conflicts == 0 and again.ambiguous == 0
    assert again.linked == 3 and again.library_edges == first.library_edges == 2
    assert again.reference_edges == first.reference_edges


def test_a_second_run_asks_for_citers_only_until_it_meets_a_stored_one(records):
    store, tool = FakeWorks(), build_tool(records)
    sync(tool, store)
    tool.get_citer_edges.reset_mock()
    sync(tool, store)

    bpe_work = store.papers["arxiv:1508.07909"]["work_id"]
    calls = {call.args[0]: call.kwargs for call in tool.get_citer_edges.call_args_list}
    known = calls[records["bpe"].s2_paper_id]["known_ids"]
    index = store.index()
    stored_citers = {citing for (citing, cited) in store.edges if cited == bpe_work}
    assert known == {index.s2_id(citing) for citing in stored_citers}
    # the six recorded citers, and Attention itself (it cites BPE too)
    assert len(known) == 7 and ATTENTION in known


def test_an_ids_only_edge_is_not_treated_as_a_stored_citer(records):
    store = FakeWorks()
    sync(build_tool(records), store)
    bpe_work = store.papers["arxiv:1508.07909"]["work_id"]
    for key, row in store.edges.items():
        if key[1] == bpe_work:
            row["source"] = SOURCE_IDS_ONLY  # what the ingest hooks wrote
    assert store.citer_work_ids(bpe_work) == set()


# -- dry run ----------------------------------------------------------------

def test_dry_run_counts_and_touches_neither_tool_nor_store(records):
    store = FakeWorks()
    tool = Mock(spec=SemanticScholarTool)
    mock_store = Mock(spec=["apply", "citer_work_ids"])
    report = run_works_sync(
        store.library(LIBRARY), store.index(), tool, mock_store, dry_run=True
    )
    tool.assert_not_called()
    assert tool.method_calls == [] and mock_store.method_calls == []
    assert report.dry_run and report.library_papers == 4
    assert report.to_link == 3 and report.linked == 0
    assert report.unresolvable == ["file:2dd76e910fbc"]
    assert report.requests_at_least == 1 + 2 * 3
    text = report.render()
    assert "works sync (dry-run)" in text and "to link: 3" in text
    assert "at least 7" in text


def test_dry_run_needs_no_tool_or_store():
    report = run_works_sync([LibraryPaper("arxiv:1706.03762")], WorkIndex(), None, None, dry_run=True)
    assert report.to_link == 1


def test_a_real_run_without_a_tool_is_refused():
    with pytest.raises(RuntimeError):
        run_works_sync([LibraryPaper("arxiv:1706.03762")], WorkIndex(), None, None, dry_run=False)


# -- failure ---------------------------------------------------------------

def test_a_failed_paper_is_reported_and_the_rest_still_sync(records):
    store, tool = FakeWorks(), build_tool(records)
    real = tool.get_reference_edges.side_effect

    def flaky(lookup):
        if lookup == records["explanation"].s2_paper_id:
            raise SemanticScholarError("Server Error 500")
        return real(lookup)

    tool.get_reference_edges.side_effect = flaky
    report = sync(tool, store)
    assert report.failures and "arxiv:1902.10186" in report.failures[0]
    assert report.stopped_early == ""
    assert ("arxiv:1706.03762" in store.papers) and report.reference_edges == 1
    # nothing half-written for the failed paper: its neighbours were never created
    assert not any(row.get("title", "").startswith("Interpretable Structure") for row in store.works.values())
    # and the failed run left the index able to resolve them next time
    assert report.citer_edges == 6


def test_rate_limiting_stops_the_run_and_keeps_what_was_written(records):
    store, tool = FakeWorks(), build_tool(records)
    real = tool.get_reference_edges.side_effect

    def limited(lookup):
        if lookup == ATTENTION:
            raise SemanticScholarUnavailableError("rate limited")
        return real(lookup)

    tool.get_reference_edges.side_effect = limited
    report = sync(tool, store)
    assert report.stopped_early == "rate limited"
    assert "arxiv:1706.03762" in store.papers  # identity was written before the walk
    assert report.render().count("stopped early") == 1


def test_a_failed_write_stops_the_run(records):
    store, tool = FakeWorks(), build_tool(records)
    store.fail_call = 1
    report = sync(tool, store)
    assert report.stopped_early == "write refused" and report.resolved == 0
    tool.get_reference_edges.assert_not_called()
    assert store.snapshot()[0] == {}


def test_a_batch_call_failure_is_reported_not_raised(records):
    tool = build_tool(records)
    tool.get_work_records.side_effect = SemanticScholarUnavailableError("rate limited")
    report = sync(tool, FakeWorks())
    assert report.stopped_early == "rate limited" and report.resolved == 0


# -- the resolution rules ----------------------------------------------------

def make_overlay(by_key=None, s2_by_work=None):
    return Overlay(WorkIndex(by_key or {}, s2_by_work or {}), counter())


def test_a_known_key_finds_its_work_and_new_aliases_attach_to_it():
    overlay = make_overlay({("arxiv", "1706.03762"): "W1"})
    work, add = overlay.resolve([("s2", ATTENTION), ("arxiv", "1706.03762"), ("corpus", "1")])
    assert work == "W1"
    assert add == [("s2", ATTENTION), ("corpus", "1")]
    assert overlay.created == set()


def test_a_new_work_is_created_once_and_found_again_in_the_same_pass():
    overlay = make_overlay()
    first, _ = overlay.resolve([("s2", ATTENTION), ("arxiv", "1706.03762")])
    second, add = overlay.resolve([("arxiv", "1706.03762")])
    assert first == second and add == [] and overlay.created == {first}


def test_a_key_owned_by_another_work_is_left_where_it_is():
    overlay = make_overlay({("arxiv", "1706.03762"): "W1", ("doi", "10.1/x"): "W2"})
    work, add = overlay.resolve([("s2", "a" * 40), ("arxiv", "1706.03762"), ("doi", "10.1/x")])
    assert work == "W1" and overlay.ambiguous == 1 and overlay.alias_conflicts == 1
    assert ("doi", "10.1/x") not in add


def test_the_work_holding_the_s2_id_wins_an_ambiguous_match():
    s2_key = ("s2", "b" * 40)
    overlay = make_overlay({("arxiv", "1"): "W1", s2_key: "W9"})
    work, _ = overlay.resolve([s2_key, ("arxiv", "1")])
    assert work == "W9"


def test_s2_reissuing_an_id_does_not_give_a_work_a_second_one():
    overlay = make_overlay({("arxiv", "1706.03762"): "W1", ("s2", "a" * 40): "W1"}, {"W1": "a" * 40})
    work, add = overlay.resolve([("s2", "c" * 40), ("arxiv", "1706.03762")])
    assert work == "W1" and add == [] and overlay.alias_conflicts == 1


def test_nothing_reaches_the_index_until_commit():
    index = WorkIndex()
    overlay = Overlay(index, counter())
    work, _ = overlay.resolve([("arxiv", "1706.03762")])
    assert index.work_for(("arxiv", "1706.03762")) is None
    overlay.commit()
    assert index.work_for(("arxiv", "1706.03762")) == work


def test_a_batch_holds_one_row_per_work_and_per_edge(records):
    batch = Batch()
    batch.put_work({"id": "W1", "title": None, "described": False, "references_synced": True})
    batch.put_work({"id": "W1", "title": "Attention", "described": True})
    batch.put_edge({"citing": "A", "cited": "B", "is_influential": False, "detail": True})
    batch.put_edge({"citing": "A", "cited": "B", "is_influential": True, "detail": True})
    assert len(batch.works) == 1 and len(batch.edges) == 1
    row = batch.works[0]
    assert row["title"] == "Attention" and row["described"] and row["references_synced"]
    assert batch.edges[0]["is_influential"] is True


def test_a_work_that_cites_itself_adds_no_edge(records):
    store, tool = FakeWorks(), build_tool(records)
    real = tool.get_reference_edges.side_effect

    def with_self_citation(lookup):
        page = real(lookup)
        if lookup == ATTENTION:
            page.edges.append(CitationEdge(other=records["attention"]))
        return page

    tool.get_reference_edges.side_effect = with_self_citation
    sync(tool, store)
    assert not any(citing == cited for citing, cited in store.edges)


def test_best_lookup_key_prefers_the_s2_id_once_known(records):
    key = best_lookup_key(record_keys(records["attention"]))
    assert s2_lookup_id(key) == ATTENTION


# -- CLI ----------------------------------------------------------------------

def test_works_sync_cli_dry_run_prints_the_report_and_exits_zero(monkeypatch):
    from typer.testing import CliRunner

    from nlp_pillars.cli import app
    from nlp_pillars.works import WorksSyncReport

    seen = {}

    def fake(dry_run, max_citers):
        seen.update(dry_run=dry_run, max_citers=max_citers)
        return WorksSyncReport(dry_run=True, library_papers=4, to_link=3, unresolvable=["file:x"])

    monkeypatch.setattr("nlp_pillars.works_store.execute_works_sync", fake)
    result = CliRunner().invoke(app, ["works", "sync", "--dry-run", "--max-citers", "200"])
    assert result.exit_code == 0
    assert seen == {"dry_run": True, "max_citers": 200}
    assert "works sync (dry-run)" in result.output and "to link: 3" in result.output


def test_works_sync_cli_exits_nonzero_when_the_run_stopped_early(monkeypatch):
    from typer.testing import CliRunner

    from nlp_pillars.cli import app
    from nlp_pillars.works import WorksSyncReport

    monkeypatch.setattr(
        "nlp_pillars.works_store.execute_works_sync",
        lambda dry_run, max_citers: WorksSyncReport(dry_run=False, stopped_early="rate limited"),
    )
    result = CliRunner().invoke(app, ["works", "sync"])
    assert result.exit_code == 1 and "stopped early: rate limited" in result.output


def test_works_sync_cli_names_the_migration_when_it_is_missing(monkeypatch):
    from typer.testing import CliRunner

    from nlp_pillars.cli import app
    from nlp_pillars.works_store import MIGRATION, WorksUnavailable

    def missing(dry_run, max_citers):
        raise WorksUnavailable(f"not available. Apply {MIGRATION} by hand.")

    monkeypatch.setattr("nlp_pillars.works_store.execute_works_sync", missing)
    result = CliRunner().invoke(app, ["works", "sync", "--dry-run"])
    assert result.exit_code == 1 and "031_works_identity.sql" in result.output
