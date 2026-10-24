"""Knowledge graph builders. Hermetic: no database, model, or network."""

import re
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from nlp_pillars.citations import citations_from_tool
from nlp_pillars.cli import app
from nlp_pillars.graph import (
    AUTHORS_PER_PAPER,
    RELATED_MIN_COSINE,
    RELATED_TOP_K,
    GraphBuildReport,
    GraphEdge,
    GraphSnapshot,
    MemoryGraphStore,
    build_plan,
    canonicalize_term,
    run_graph_build,
    slice_graph,
)
from nlp_pillars.graph_store import load_snapshot
from nlp_pillars.orchestrator import Orchestrator
from nlp_pillars.paper_vectors import SOURCE_TITLE_ABSTRACT
from nlp_pillars.schemas import Lesson, PaperRef
from nlp_pillars.tools.semantic_scholar_tool import SemanticScholarTool

MIGRATION_SQL = (
    Path(__file__).resolve().parents[1] / "docs/migrations/020_knowledge_graph.sql"
).read_text()
SCHEMA_SQL = (Path(__file__).resolve().parents[1] / "schema.sql").read_text()
RUNNER = CliRunner()


def _paper(paper_id, pillar="neural-architectures-language", authors=None, title=None):
    return {
        "id": paper_id,
        "pillar_id": pillar,
        "title": title or paper_id,
        "authors": authors if authors is not None else ["Ada Lovelace"],
    }


def _lesson(paper_id: str) -> Lesson:
    return Lesson(
        paper_id=paper_id,
        pillar_id="neural-architectures-language",
        title="A lesson",
        content="Content.",
        tl_dr="Short.",
        takeaways=["t"],
        practice_ideas=["p"],
        connections=["c"],
    )


def _snapshot(**overrides):
    base = GraphSnapshot(
        papers=[_paper("arxiv:1"), _paper("arxiv:2", authors=["Grace Hopper"])],
        pillars=[{"id": "neural-architectures-language", "name": "Neural"}],
        notes=[
            {
                "paper_id": "arxiv:1",
                "key_terms": ["Self-Attention (SA)", "Transformers"],
            },
            {
                "paper_id": "arxiv:2",
                "key_terms": ["self attentions", "Transformer"],
            },
        ],
        lessons=[{"id": "lesson-1", "paper_id": "arxiv:1"}],
        cards=[{"id": "card-1", "paper_id": "arxiv:1"}],
        episodes=[{"id": "episode-1", "paper_id": "arxiv:2"}],
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_migration_declares_the_graph_and_reloads_postgrest():
    assert "CREATE TABLE IF NOT EXISTS concepts" in MIGRATION_SQL
    assert "CREATE TABLE IF NOT EXISTS paper_concepts" in MIGRATION_SQL
    assert "CREATE TABLE IF NOT EXISTS graph_edges" in MIGRATION_SQL
    assert "src_kind" in MIGRATION_SQL and "provenance" in MIGRATION_SQL
    assert "created_at" in MIGRATION_SQL
    assert "CREATE UNIQUE INDEX IF NOT EXISTS graph_edges_edge_key" in MIGRATION_SQL
    assert "CREATE UNIQUE INDEX IF NOT EXISTS concepts_name_key" in MIGRATION_SQL
    assert "replace_derived_graph" in MIGRATION_SQL
    assert "provenance LIKE 'derived:%'" in MIGRATION_SQL
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON graph_edges TO web_anon;" in MIGRATION_SQL
    assert "GRANT EXECUTE ON FUNCTION replace_derived_graph(JSONB, JSONB, JSONB)" in MIGRATION_SQL
    assert MIGRATION_SQL.rstrip().endswith("NOTIFY pgrst, 'reload schema';")


def test_schema_sql_carries_the_same_objects():
    assert "CREATE TABLE concepts" in SCHEMA_SQL
    assert "CREATE TABLE paper_concepts" in SCHEMA_SQL
    assert "CREATE TABLE graph_edges" in SCHEMA_SQL
    assert "graph_edges_edge_key" in SCHEMA_SQL
    assert "concepts_name_key" in SCHEMA_SQL
    assert "replace_derived_graph" in SCHEMA_SQL
    assert "GRANT EXECUTE ON FUNCTION replace_derived_graph(JSONB, JSONB, JSONB)" in SCHEMA_SQL


def test_edge_identity_is_enforced_by_the_migration_index():
    match = re.search(
        r"CREATE UNIQUE INDEX IF NOT EXISTS graph_edges_edge_key[^;]*;",
        MIGRATION_SQL,
        re.S,
    )
    assert match, "migration 020 no longer creates the edge unique index"
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE graph_edges ("
        " src_kind TEXT NOT NULL, src_id TEXT NOT NULL,"
        " dst_kind TEXT NOT NULL, dst_id TEXT NOT NULL,"
        " kind TEXT NOT NULL, weight REAL, provenance TEXT NOT NULL,"
        " created_at TEXT)"
    )
    conn.execute(match.group(0))
    conn.execute(
        "INSERT INTO graph_edges VALUES"
        " ('paper', 'arxiv:1', 'pillar', 'neural', 'in_pillar', 1, 'derived:structure', 't')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO graph_edges VALUES"
            " ('paper', 'arxiv:1', 'pillar', 'neural', 'in_pillar', 1, 'derived:structure', 't2')"
        )
    conn.execute(
        "INSERT INTO graph_edges VALUES"
        " ('paper', 'arxiv:1', 'paper', 'arxiv:2', 'related_to', 0.5, 'derived:embedding', 't')"
    )
    assert conn.execute("SELECT count(*) FROM graph_edges").fetchone()[0] == 2


def test_canonical_terms_collapse_surface_variants():
    assert canonicalize_term("Self-Attention (SA)") == "self attention"
    assert canonicalize_term("self attentions") == "self attention"
    assert canonicalize_term("Transformers") == "transformer"
    assert canonicalize_term("Transformer") == "transformer"
    assert canonicalize_term("analysis") == "analysis"
    assert canonicalize_term("losses") == "loss"
    assert canonicalize_term("") == ""


def test_string_canonicalisation_shares_a_concept_without_a_model():
    plan = build_plan(_snapshot())
    names = {concept.name for concept in plan.concepts}
    assert names == {"self attention", "transformer"}
    assert plan.concept_merge == "string"
    covers = {
        (edge.src_id, edge.dst_id)
        for edge in plan.edges
        if edge.kind == "covers"
    }
    assert ("arxiv:1", "self attention") in covers
    assert ("arxiv:2", "self attention") in covers
    assert ("arxiv:1", "transformer") in covers
    assert ("arxiv:2", "transformer") in covers


def test_embedding_merge_unions_close_names_and_keeps_the_smaller_one():
    snapshot = GraphSnapshot(
        papers=[_paper("arxiv:1")],
        notes=[
            {
                "paper_id": "arxiv:1",
                "key_terms": ["self attention", "self attention mechanism"],
            }
        ],
    )

    def embed(names):
        table = {
            "self attention": [1.0, 0.0],
            "self attention mechanism": [0.99, 0.01],
        }
        return [table[name] for name in names]

    plan = build_plan(snapshot, embed_names=embed)
    assert [concept.name for concept in plan.concepts] == ["self attention"]
    assert "self attention mechanism" in plan.concepts[0].aliases
    assert plan.concept_merge == "string+embedding"


def test_structural_edges_cap_authors_and_link_lesson_card_episode():
    authors = [f"Author {index}" for index in range(12)]
    snapshot = _snapshot(
        papers=[_paper("arxiv:1", authors=authors)],
        notes=[],
        lessons=[{"id": "lesson-1", "paper_id": "arxiv:1"}],
        cards=[{"id": "card-1", "paper_id": "arxiv:1"}],
        episodes=[{"id": "episode-1", "paper_id": "arxiv:1"}],
    )
    plan = build_plan(snapshot)
    authored = [edge for edge in plan.edges if edge.kind == "authored_by"]
    assert len(authored) == AUTHORS_PER_PAPER
    assert authored[0].dst_id == "Author 0"
    kinds = {(edge.src_kind, edge.dst_kind, edge.kind) for edge in plan.edges}
    assert ("paper", "pillar", "in_pillar") in kinds
    assert ("lesson", "paper", "of") in kinds
    assert ("quiz_card", "paper", "of") in kinds
    assert ("episode", "paper", "of") in kinds


def _title_abstract(paper_id, vector, model_id="bge", source=SOURCE_TITLE_ABSTRACT):
    return {
        "ref_kind": "paper",
        "ref_id": paper_id,
        "model_id": model_id,
        "source": source,
        "vector": vector,
    }


def test_related_floor_is_the_measured_title_abstract_gap():
    """0.70 sits between the unrelated band (≤0.62) and on-topic pairs (≥0.76)."""
    assert RELATED_MIN_COSINE == 0.70
    assert RELATED_TOP_K == 5


def test_unrelated_papers_get_no_related_edge():
    """Cosine 0.60 is the measured cross-pillar band. Top-k must not keep it.

    Two papers only: without the floor, each is the other's neighbour.
    """
    snapshot = GraphSnapshot(
        papers=[_paper("grammars"), _paper("probing")],
        paper_vectors=[
            _title_abstract("grammars", [1.0, 0.0]),
            _title_abstract("probing", [0.6, 0.8]),
        ],
    )
    plan = build_plan(snapshot)
    related = [edge for edge in plan.edges if edge.kind == "related_to"]
    assert related == []
    assert plan.embedding_model == "bge"


def test_related_edges_keep_title_abstract_neighbours_above_the_floor():
    snapshot = GraphSnapshot(
        papers=[
            _paper("arxiv:1"),
            _paper("arxiv:2"),
            _paper("arxiv:3"),
            _paper("arxiv:4"),
        ],
        paper_vectors=[
            _title_abstract("arxiv:1", [1.0, 0.0]),
            _title_abstract("arxiv:2", [0.8, 0.6]),
            _title_abstract("arxiv:4", [-1.0, 0.0]),
            _title_abstract("arxiv:3", [1.0, 0.0], model_id="other"),
            {
                "ref_kind": "external_paper",
                "ref_id": "ext:1",
                "model_id": "bge",
                "source": SOURCE_TITLE_ABSTRACT,
                "vector": [1.0, 0.0],
            },
            _title_abstract("arxiv:1", [0.0, 1.0], source="chunk_mean"),
        ],
    )
    plan = build_plan(snapshot)
    related = [edge for edge in plan.edges if edge.kind == "related_to"]
    assert plan.embedding_model == "bge"
    pairs = {(edge.src_id, edge.dst_id, edge.weight) for edge in related}
    assert pairs == {("arxiv:1", "arxiv:2", 0.8), ("arxiv:2", "arxiv:1", 0.8)}
    assert all(edge.provenance == "derived:embedding:bge" for edge in related)


def test_related_edges_stop_at_top_k_when_every_pair_clears_the_floor():
    papers = [_paper(f"p{index}") for index in range(RELATED_TOP_K + 2)]
    vectors = [
        _title_abstract(paper["id"], [1.0, 0.0]) for paper in papers
    ]
    plan = build_plan(GraphSnapshot(papers=papers, paper_vectors=vectors))
    related = [edge for edge in plan.edges if edge.kind == "related_to"]
    by_src = {}
    for edge in related:
        by_src.setdefault(edge.src_id, set()).add(edge.dst_id)
    assert len(related) == len(papers) * RELATED_TOP_K
    for paper in papers:
        neighbours = by_src[paper["id"]]
        assert len(neighbours) == RELATED_TOP_K
        assert paper["id"] not in neighbours


def test_related_rebuild_does_not_duplicate_edges():
    snapshot = GraphSnapshot(
        papers=[_paper("arxiv:1"), _paper("arxiv:2")],
        paper_vectors=[
            _title_abstract("arxiv:1", [1.0, 0.0]),
            _title_abstract("arxiv:2", [0.8, 0.6]),
        ],
    )
    store = MemoryGraphStore()
    first = run_graph_build(
        snapshot, dry_run=False, citation_tool=None, apply=store.apply
    )
    before = len(store.edges)
    second = run_graph_build(
        snapshot, dry_run=False, citation_tool=None, apply=store.apply
    )
    assert second.edges == first.edges
    assert len(store.edges) == before
    related = [edge for edge in store.edges.values() if edge.kind == "related_to"]
    assert len(related) == 2


def test_load_snapshot_names_the_migration_when_paper_vectors_is_missing(monkeypatch):
    from nlp_pillars.graph_store import GraphStoreError

    def fake_read(table, columns, order):
        if table == "paper_vectors":
            raise GraphStoreError(
                "Could not find the table paper_vectors in the schema cache PGRST205"
            )
        return []

    monkeypatch.setattr("nlp_pillars.graph_store._read_table", fake_read)
    snapshot, warnings = load_snapshot()
    assert snapshot.paper_vectors == []
    assert any("021_paper_vectors" in warning for warning in warnings)
    assert any("related_to" in warning for warning in warnings)


def test_load_snapshot_reads_paper_vectors(monkeypatch):
    tables = {
        "papers": [{"id": "arxiv:1", "pillar_id": "p", "title": "T", "authors": []}],
        "pillars": [],
        "notes": [],
        "lessons": [],
        "quiz_cards": [],
        "podcast_scripts": [],
        "paper_citations": [],
        "paper_vectors": [{
            "ref_kind": "paper",
            "ref_id": "arxiv:1",
            "model_id": "bge",
            "source": "title_abstract",
            "vec": [0.2, 0.3],
        }],
    }

    def fake_read(table, columns, order):
        assert table not in {"chunk_embeddings", "paper_chunks"}
        assert table in tables
        if table == "paper_vectors":
            assert "vec" in columns
        return tables[table]

    monkeypatch.setattr("nlp_pillars.graph_store._read_table", fake_read)
    snapshot, warnings = load_snapshot()
    assert warnings == []
    assert snapshot.paper_vectors == [{
        "ref_kind": "paper",
        "ref_id": "arxiv:1",
        "model_id": "bge",
        "source": "title_abstract",
        "vector": [0.2, 0.3],
    }]


def test_citation_edges_use_library_papers_only():
    snapshot = _snapshot(
        notes=[],
        citations=[
            {
                "paper_id": "arxiv:1",
                "cited_paper_id": "ext:9",
                "direction": "outgoing",
            },
            {
                "paper_id": "arxiv:2",
                "cited_paper_id": "arxiv:1",
                "direction": "incoming",
            },
            {
                "paper_id": "ext:9",
                "cited_paper_id": "arxiv:1",
                "direction": "outgoing",
            },
        ],
    )
    plan = build_plan(snapshot)
    cites = [edge for edge in plan.edges if edge.kind == "cites"]
    assert {
        (edge.src_kind, edge.src_id, edge.dst_kind, edge.dst_id) for edge in cites
    } == {
        ("paper", "arxiv:1", "external_paper", "ext:9"),
        ("paper", "arxiv:1", "paper", "arxiv:2"),
    }


def test_plan_edge_keys_are_unique():
    plan = build_plan(_snapshot())
    keys = [edge.key for edge in plan.edges]
    assert len(keys) == len(set(keys))


def test_rebuild_does_not_duplicate_edges_and_keeps_user_edges():
    store = MemoryGraphStore()
    snapshot = _snapshot()
    first = run_graph_build(
        snapshot, dry_run=False, citation_tool=None, apply=store.apply
    )
    store.edges[("paper", "arxiv:1", "paper", "arxiv:2", "annotated")] = GraphEdge(
        src_kind="paper",
        src_id="arxiv:1",
        dst_kind="paper",
        dst_id="arxiv:2",
        kind="annotated",
        weight=1.0,
        provenance="user",
    )
    before = len(store.edges)
    second = run_graph_build(
        snapshot, dry_run=False, citation_tool=None, apply=store.apply
    )
    assert second.edges == first.edges
    assert len(store.edges) == before
    assert store.edges[
        ("paper", "arxiv:1", "paper", "arxiv:2", "annotated")
    ].provenance == "user"


def test_dry_run_does_not_fetch_embed_or_write():
    tool = Mock(spec=SemanticScholarTool)
    tool.get_references.return_value = []
    tool.get_citations.return_value = []

    def _boom(*_args, **_kwargs):
        raise AssertionError("dry-run must not call this")

    report = run_graph_build(
        _snapshot(),
        dry_run=True,
        citation_tool=tool,
        embed_names=_boom,
        store_citations=_boom,
        apply=_boom,
    )
    tool.get_references.assert_not_called()
    tool.get_citations.assert_not_called()
    assert report.dry_run is True
    assert report.citations_fetched == 0
    assert report.citations_to_fetch == 2
    assert "dry-run" in report.render()


def test_citation_fetch_uses_the_real_tool_interface_and_not_external_ids():
    """Mock(spec=...) fails if the builder calls a method the class lacks."""
    assert callable(SemanticScholarTool.get_references)
    assert callable(SemanticScholarTool.get_citations)
    assert callable(SemanticScholarTool._enforce_rate_limit)

    tool = Mock(spec=SemanticScholarTool)
    external = PaperRef(id="ext:9", title="Outside", authors=["Z"])
    tool.get_references.return_value = [(external, False)]
    tool.get_citations.return_value = []
    rows = citations_from_tool("arxiv:1", tool, limit=20)
    tool.get_references.assert_called_once_with("arxiv:1", limit=20)
    tool.get_citations.assert_called_once_with("arxiv:1", limit=20)
    assert rows[0].cited_paper_id == "ext:9"
    assert rows[0].citation_direction == "outgoing"

    store = MemoryGraphStore()
    snapshot = _snapshot(notes=[], citations=[])
    run_graph_build(
        snapshot,
        dry_run=False,
        citation_tool=tool,
        store_citations=lambda fetched: 0,
        apply=store.apply,
    )
    queried = {call.args[0] for call in tool.get_references.call_args_list}
    assert queried == {"arxiv:1", "arxiv:2"}
    assert "ext:9" not in queried


def test_slice_respects_depth_pillar_and_the_size_cap():
    papers = [
        _paper("arxiv:1", pillar="neural-architectures-language"),
        _paper("arxiv:2", pillar="neural-architectures-language"),
        _paper("arxiv:3", pillar="ai-safety-alignment"),
    ]
    edges = [
        {
            "src_kind": "paper",
            "src_id": "arxiv:1",
            "dst_kind": "paper",
            "dst_id": "arxiv:2",
            "kind": "related_to",
            "weight": 0.9,
            "provenance": "derived:embedding:bge",
        },
        {
            "src_kind": "paper",
            "src_id": "arxiv:2",
            "dst_kind": "paper",
            "dst_id": "arxiv:3",
            "kind": "related_to",
            "weight": 0.4,
            "provenance": "derived:embedding:bge",
        },
        {
            "src_kind": "paper",
            "src_id": "arxiv:1",
            "dst_kind": "author",
            "dst_id": "Ada Lovelace",
            "kind": "authored_by",
            "weight": 1,
            "provenance": "derived:structure",
        },
    ]
    near = slice_graph(
        papers=papers,
        pillars=[],
        concepts=[],
        lessons=[],
        cards=[],
        episodes=[],
        edges=edges,
        pillar="neural-architectures-language",
        depth=1,
        center="arxiv:1",
    )
    ids = {node["id"] for node in near["nodes"]}
    assert "arxiv:1" in ids
    assert "arxiv:2" in ids
    assert "Ada Lovelace" in ids
    assert "arxiv:3" not in ids
    assert near["found"] is True

    chain_edges = []
    chain_papers = []
    for index in range(6):
        chain_papers.append(_paper(f"p{index}"))
        if index:
            chain_edges.append(
                {
                    "src_kind": "paper",
                    "src_id": f"p{index - 1}",
                    "dst_kind": "paper",
                    "dst_id": f"p{index}",
                    "kind": "related_to",
                    "weight": 1,
                    "provenance": "derived:embedding:bge",
                }
            )
    capped = slice_graph(
        papers=chain_papers,
        pillars=[],
        concepts=[],
        lessons=[],
        cards=[],
        episodes=[],
        edges=chain_edges,
        depth=5,
        center="p0",
        max_nodes=2,
        max_edges=10,
    )
    assert capped["truncated"] is True
    assert len(capped["nodes"]) <= 2
    assert len(capped["edges"]) <= 10


def test_graph_build_cli_dry_run_prints_the_report(monkeypatch):
    report = GraphBuildReport(
        dry_run=True,
        library_papers=2,
        concepts=1,
        paper_concepts=1,
        edges=3,
        edges_by_kind={"in_pillar": 2, "covers": 1},
        embedding_model=None,
        concept_merge="string",
        citations_to_fetch=2,
        citations_fetched=0,
        citations_stored=0,
        warnings=[],
    )
    seen = {}

    def fake(dry_run):
        seen["dry_run"] = dry_run
        return report

    monkeypatch.setattr("nlp_pillars.graph_store.execute_graph_build", fake)
    result = RUNNER.invoke(app, ["graph", "build", "--dry-run"])
    assert result.exit_code == 0
    assert seen["dry_run"] is True
    assert "dry-run" in result.output
    assert "concepts: 1" in result.output


def test_select_and_daily_both_fetch_citations_for_the_processed_paper():
    paper = PaperRef(
        id="arxiv:1706.03762",
        title="Attention",
        authors=["Vaswani"],
    )
    selected = Orchestrator.__new__(Orchestrator)
    selected._stage = lambda *_args, **_kwargs: None
    selected._fetch_and_store_citations = Mock(return_value=1)
    selected._process_paper = Mock(return_value=(_lesson(paper.id), []))
    result = Orchestrator.process_selected_papers(
        selected,
        "neural-architectures-language",
        papers=[paper],
    )
    assert result.success is True
    selected._fetch_and_store_citations.assert_called_once_with(paper.id)

    daily = Orchestrator.__new__(Orchestrator)
    daily._on_stage = lambda *_args, **_kwargs: None
    daily._cancel = None
    daily._current_paper_stage = None
    daily._infra_errors = []
    daily._stage = lambda *_args, **_kwargs: None
    daily._run_discovery = Mock(return_value=([], False))
    daily._search_candidates = Mock(return_value=[])
    daily._enqueue_candidates = Mock(return_value=0)
    daily._pop_queue = Mock(return_value=[paper])
    daily._process_paper = Mock(return_value=(_lesson(paper.id), None))
    daily._fetch_and_store_citations = Mock(return_value=2)
    daily_result = Orchestrator.run_daily(
        daily, "neural-architectures-language", papers_limit=1
    )
    assert daily_result.success is True
    daily._fetch_and_store_citations.assert_called_once_with(paper.id)


def test_upload_stores_citations_without_failing_the_paper(tmp_path):
    from nlp_pillars.schemas import ParsedPaper
    from nlp_pillars.services.upload_service import UploadService

    tool = Mock(spec=SemanticScholarTool)
    cited = PaperRef(id="ext:9", title="Cited", authors=["Z"])
    tool.get_references.return_value = [(cited, False)]
    tool.get_citations.return_value = []
    service = UploadService(
        upload_dir=str(tmp_path / "uploads"), citation_tool=tool
    )
    paper = PaperRef(
        id="arxiv:1706.03762", title="A Paper", authors=["A"], url_pdf="x"
    )
    parsed = ParsedPaper(
        paper_ref=paper,
        full_text="Body text that must be stored.",
        chunks=["Body text that must be stored."],
        extractor="pymupdf4llm",
    )
    stored = {}

    def save(rows):
        stored["rows"] = list(rows)
        return len(rows)

    vectors = Mock()
    vectors.upsert_text.return_value = 1
    db_mock = Mock()
    db_mock.replace_paper_text.return_value = 1
    db_mock.embed_paper_chunks.return_value = 1
    with (
        patch.object(service.ingest_agent, "ingest", return_value=parsed),
        patch("nlp_pillars.services.upload_service.db", db_mock),
        patch("nlp_pillars.services.upload_service.vectors", vectors),
        patch("nlp_pillars.citations.db.add_paper_citations", save),
    ):
        outcome = service._run_full_pipeline(
            paper=paper,
            pillar_id="neural-architectures-language",
            run_summarizer=False,
            generate_quiz=False,
        )

    assert outcome.ok, outcome.errors
    assert stored["rows"][0].cited_paper_id == "ext:9"
    assert "citations" not in outcome.actions_triggered


def test_upload_citation_failure_does_not_fail_the_pipeline(tmp_path):
    from nlp_pillars.schemas import ParsedPaper
    from nlp_pillars.services.upload_service import UploadService

    tool = Mock(spec=SemanticScholarTool)
    tool.get_references.side_effect = RuntimeError("semantic scholar down")
    service = UploadService(
        upload_dir=str(tmp_path / "uploads"), citation_tool=tool
    )
    paper = PaperRef(
        id="arxiv:1706.03762", title="A Paper", authors=["A"], url_pdf="x"
    )
    parsed = ParsedPaper(
        paper_ref=paper,
        full_text="Body.",
        chunks=["Body."],
        extractor="pymupdf4llm",
    )
    vectors = Mock()
    vectors.upsert_text.return_value = 1
    with (
        patch.object(service.ingest_agent, "ingest", return_value=parsed),
        patch("nlp_pillars.services.upload_service.db", Mock()),
        patch("nlp_pillars.services.upload_service.vectors", vectors),
    ):
        outcome = service._run_full_pipeline(
            paper=paper,
            pillar_id="neural-architectures-language",
            run_summarizer=False,
            generate_quiz=False,
        )
    assert outcome.ok, outcome.errors
