"""graph build and the works layer: library->library citations, and migration 031.

No database, no network. The SQL is checked for the objects and grants the code
relies on; its behaviour is exercised by the sync tests against an in-memory
store with the same rules.
"""

import re
from pathlib import Path
from unittest.mock import Mock

import pytest

from nlp_pillars.graph import (
    EXTERNAL_CITES_PER_DIRECTION,
    GraphSnapshot,
    _merge_citations,
    build_plan,
    run_graph_build,
)
from nlp_pillars.graph_store import GraphStoreError, load_snapshot
from nlp_pillars.schemas import PaperCitation
from nlp_pillars.tools.semantic_scholar_tool import SemanticScholarTool

ROOT = Path(__file__).resolve().parents[1]
MIGRATION_SQL = (ROOT / "docs/migrations/031_works_identity.sql").read_text()
SCHEMA_SQL = (ROOT / "schema.sql").read_text()

ATTENTION = "arxiv:1706.03762"
EXPLANATION = "arxiv:1902.10186"


def snapshot(citations, papers=(ATTENTION, EXPLANATION)):
    return GraphSnapshot(
        papers=[{"id": p, "pillar_id": "p", "title": p, "authors": []} for p in papers],
        pillars=[{"id": "p", "name": "P"}],
        citations=citations,
    )


def cites(plan):
    return {
        (e.src_kind, e.src_id, e.dst_kind, e.dst_id)
        for e in plan.edges
        if e.kind == "cites"
    }


# -- library -> library ------------------------------------------------------

def test_a_citation_between_two_library_papers_is_one_edge_between_papers():
    # what the paper_citations view returns for it: one row from each end
    plan = build_plan(
        snapshot(
            [
                {"paper_id": EXPLANATION, "cited_paper_id": ATTENTION, "direction": "outgoing"},
                {"paper_id": ATTENTION, "cited_paper_id": EXPLANATION, "direction": "incoming"},
            ]
        )
    )
    assert cites(plan) == {("paper", EXPLANATION, "paper", ATTENTION)}


def test_the_bare_id_the_old_fetch_stored_is_the_library_paper_not_an_external_one():
    # 1706.03762 is what Semantic Scholar names it; the library says arxiv:1706.03762
    plan = build_plan(
        snapshot([{"paper_id": EXPLANATION, "cited_paper_id": "1706.03762", "direction": "outgoing"}])
    )
    assert cites(plan) == {("paper", EXPLANATION, "paper", ATTENTION)}


def test_a_library_id_with_a_version_or_a_url_is_still_that_paper():
    plan = build_plan(
        snapshot(
            [
                {"paper_id": EXPLANATION, "cited_paper_id": "1706.03762v5", "direction": "outgoing"},
                {"paper_id": EXPLANATION, "cited_paper_id": "https://arxiv.org/abs/1706.03762", "direction": "outgoing"},
            ]
        )
    )
    assert len(cites(plan)) == 1


def test_run_graph_build_does_not_double_a_citation_the_fetch_repeats():
    tool = Mock(spec=SemanticScholarTool)
    tool.get_references.return_value = [(Mock(id="1706.03762"), False)]
    tool.get_citations.return_value = []
    stored = snapshot(
        [{"paper_id": EXPLANATION, "cited_paper_id": ATTENTION, "direction": "outgoing", "source": "semantic_scholar"}]
    )
    plan_edges = {}

    def apply(plan):
        plan_edges["plan"] = plan

    run_graph_build(
        stored,
        dry_run=False,
        citation_tool=tool,
        store_citations=lambda rows: len(rows),
        apply=apply,
    )
    assert cites(plan_edges["plan"]) == {("paper", EXPLANATION, "paper", ATTENTION)}


def test_merge_keeps_one_row_per_identity_and_prefers_the_stored_row():
    fetched = [PaperCitation(paper_id=EXPLANATION, cited_paper_id="1706.03762", citation_direction="outgoing")]
    stored = [
        {
            "paper_id": EXPLANATION,
            "cited_paper_id": ATTENTION,
            "direction": "outgoing",
            "source": "semantic_scholar",
            "is_influential": True,
            "other_citation_count": 200329,
        }
    ]
    (row,) = _merge_citations(stored, fetched)
    assert row["cited_paper_id"] == ATTENTION
    assert row["is_influential"] is True and row["other_citation_count"] == 200329


# -- the graph keeps its size ----------------------------------------------------

def external_rows(count, direction="outgoing"):
    return [
        {
            "paper_id": ATTENTION,
            "cited_paper_id": f"2001.{n:05d}",
            "direction": direction,
            "other_citation_count": n,
        }
        for n in range(count)
    ]


def test_external_papers_are_capped_per_library_paper_and_direction():
    rows = external_rows(60) + external_rows(60, "incoming")
    plan = build_plan(snapshot(rows))
    out = [e for e in plan.edges if e.kind == "cites" and e.src_id == ATTENTION]
    into = [e for e in plan.edges if e.kind == "cites" and e.dst_id == ATTENTION]
    assert len(out) == len(into) == EXTERNAL_CITES_PER_DIRECTION


def test_the_cap_keeps_influential_first_then_the_most_cited():
    rows = external_rows(40)
    rows[3]["is_influential"] = True  # the least cited, but influential
    plan = build_plan(snapshot(rows))
    kept = {e.dst_id for e in plan.edges if e.kind == "cites"}
    assert "2001.00003" in kept
    assert "2001.00039" in kept and "2001.00020" not in kept  # then by citation count
    assert len(kept) == EXTERNAL_CITES_PER_DIRECTION


def test_citations_between_library_papers_are_never_capped():
    papers = [f"arxiv:2001.{n:05d}" for n in range(EXTERNAL_CITES_PER_DIRECTION + 10)]
    rows = [
        {"paper_id": ATTENTION, "cited_paper_id": p, "direction": "outgoing"} for p in papers
    ]
    plan = build_plan(snapshot(rows, papers=[ATTENTION, *papers]))
    assert len(cites(plan)) == len(papers)


def test_the_choice_is_the_same_every_build():
    rows = external_rows(60)
    first = build_plan(snapshot(rows))
    second = build_plan(snapshot(list(reversed(rows))))
    assert [e.key for e in first.edges] == [e.key for e in second.edges]


# -- reading the view ---------------------------------------------------------------

def test_load_snapshot_asks_the_view_for_influence_and_citation_count(monkeypatch):
    asked = {}

    def read(table, columns, order):
        if table == "paper_citations":
            asked["columns"] = columns
            return [
                {
                    "paper_id": EXPLANATION,
                    "cited_paper_id": ATTENTION,
                    "citation_direction": "outgoing",
                    "source": "semantic_scholar",
                    "is_influential": True,
                    "other_citation_count": 200329,
                }
            ]
        return []

    monkeypatch.setattr("nlp_pillars.graph_store._read_table", read)
    loaded, warnings = load_snapshot()
    assert "other_citation_count" in asked["columns"] and "is_influential" in asked["columns"]
    assert loaded.citations[0]["is_influential"] is True
    assert loaded.citations[0]["other_citation_count"] == 200329
    assert loaded.citations[0]["direction"] == "outgoing"


def test_load_snapshot_still_works_against_the_pre_031_table(monkeypatch):
    columns_asked = []

    def read(table, columns, order):
        if table == "paper_citations":
            columns_asked.append(columns)
            if "other_citation_count" in columns:
                raise GraphStoreError("column paper_citations.other_citation_count does not exist")
            return [
                {
                    "paper_id": EXPLANATION,
                    "cited_paper_id": "1706.03762",
                    "citation_direction": "outgoing",
                    "source": "semantic_scholar",
                    "is_influential": False,
                }
            ]
        return []

    monkeypatch.setattr("nlp_pillars.graph_store._read_table", read)
    loaded, _ = load_snapshot()
    assert len(columns_asked) == 2 and "other_citation_count" not in columns_asked[1]
    assert loaded.citations[0]["other_citation_count"] is None


def test_any_other_read_failure_is_not_swallowed(monkeypatch):
    def read(table, columns, order):
        if table == "paper_citations":
            raise GraphStoreError("connection reset")
        return []

    monkeypatch.setattr("nlp_pillars.graph_store._read_table", read)
    with pytest.raises(GraphStoreError, match="connection reset"):
        load_snapshot()


# -- migration 031 ----------------------------------------------------------------------

def test_migration_declares_the_works_tables_with_the_columns_the_code_writes():
    for table in ("works", "work_ids", "work_citations"):
        assert f"CREATE TABLE IF NOT EXISTS {table} (" in MIGRATION_SQL
    assert "ADD COLUMN IF NOT EXISTS work_id UUID REFERENCES works(id)" in MIGRATION_SQL
    for column in (
        "s2_paper_id", "corpus_id", "publication_date", "influential_citation_count",
        "reference_count", "open_pdf_url", "fetched_at", "refreshed_at",
    ):
        assert column in MIGRATION_SQL
    assert "kind IN ('s2', 'arxiv', 'doi', 'acl', 'corpus', 'title_hash')" in MIGRATION_SQL
    assert "PRIMARY KEY (kind, value)" in MIGRATION_SQL
    assert "PRIMARY KEY (citing, cited)" in MIGRATION_SQL
    assert "intents        TEXT[]" in MIGRATION_SQL and "contexts       TEXT[]" in MIGRATION_SQL
    assert "cardinality(contexts) <= 3" in MIGRATION_SQL


def test_the_old_table_is_renamed_not_dropped_and_a_view_takes_its_name():
    assert "ALTER TABLE paper_citations RENAME TO paper_citations_legacy" in MIGRATION_SQL
    assert "DROP TABLE" not in MIGRATION_SQL.upper().replace("-- ", "")
    assert "CREATE OR REPLACE VIEW paper_citations AS" in MIGRATION_SQL
    for column in ("paper_id", "cited_paper_id", "citation_direction", "is_influential",
                   "citation_context", "other_citation_count"):
        assert column in MIGRATION_SQL.split("CREATE OR REPLACE VIEW paper_citations AS")[1]
    assert "'outgoing'" in MIGRATION_SQL and "'incoming'" in MIGRATION_SQL


def test_migration_is_idempotent_grants_explicitly_and_reloads_postgrest():
    assert not re.search(r"CREATE (UNIQUE )?(TABLE|INDEX) (?!IF NOT EXISTS)", MIGRATION_SQL)
    assert not re.search(r"ADD COLUMN (?!IF NOT EXISTS)", MIGRATION_SQL)
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON works, work_ids, work_citations TO web_anon;" in MIGRATION_SQL
    assert "GRANT SELECT ON paper_citations TO web_anon;" in MIGRATION_SQL
    assert "GRANT EXECUTE ON FUNCTION apply_work_batch(JSONB, JSONB, JSONB, JSONB) TO web_anon;" in MIGRATION_SQL
    assert MIGRATION_SQL.rstrip().endswith("NOTIFY pgrst, 'reload schema';")
    assert "MUST BE RUN BY HAND" in MIGRATION_SQL


def test_the_function_never_lets_a_thin_write_make_a_stored_row_worse():
    assert "COALESCE(EXCLUDED.title, works.title)" in MIGRATION_SQL
    assert "COALESCE(works.s2_paper_id, EXCLUDED.s2_paper_id)" in MIGRATION_SQL
    assert "WHERE NOT COALESCE(e.detail, FALSE)" in MIGRATION_SQL
    assert "ON CONFLICT (citing, cited) DO NOTHING" in MIGRATION_SQL


def test_schema_sql_carries_the_same_objects_last():
    for name in (
        "CREATE TABLE works (",
        "CREATE TABLE work_ids (",
        "CREATE TABLE work_citations (",
        "ALTER TABLE paper_citations RENAME TO paper_citations_legacy",
        "CREATE OR REPLACE VIEW paper_citations AS",
        "apply_work_batch",
        "GRANT EXECUTE ON FUNCTION apply_work_batch(JSONB, JSONB, JSONB, JSONB) TO web_anon;",
    ):
        assert name in SCHEMA_SQL
    # the rename comes after everything that still treats paper_citations as a table
    rename = SCHEMA_SQL.index("RENAME TO paper_citations_legacy")
    assert rename > SCHEMA_SQL.index("COMMENT ON TABLE paper_citations IS")
    assert rename > SCHEMA_SQL.index("CREATE INDEX idx_paper_citations_paper_id")
