"""Obsidian vault export. Hermetic: no database, model, or network."""

from pathlib import Path
from unittest.mock import Mock

from typer.testing import CliRunner

from nlp_pillars.cli import app
from nlp_pillars.db import PostgRESTClient, TableQuery
from nlp_pillars.vault import (
    END_MARK,
    START_MARK,
    ExportReport,
    apply_export,
    render_vault,
    resolve_vault_dir,
    short_id,
)
from nlp_pillars.vault_store import (
    VaultStoreError,
    export_library,
    load_vault_snapshot,
)

GOLDEN = Path(__file__).parent / "fixtures" / "vault"
RUNNER = CliRunner()

ATTENTION = "arxiv:1706.03762"
SHARED_A = "arxiv:1111.1111"
SHARED_B = "arxiv:2222.2222"
EPISODE_A = "11111111-1111-1111-1111-111111111111"
EPISODE_B = "22222222-2222-2222-2222-222222222222"
PILLAR_NEURAL = "neural-architectures-language"
PILLAR_FORMAL = "formal-linguistics-nlp"


def library():
    """A small library with a title collision, one concept, and one external cite."""
    from nlp_pillars.vault import VaultSnapshot

    papers = [
        {
            "id": ATTENTION,
            "pillar_id": PILLAR_NEURAL,
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "year": 2017,
            "abstract": "We propose a new network architecture, the Transformer.",
        },
        {
            "id": SHARED_A,
            "pillar_id": PILLAR_FORMAL,
            "title": "Shared Title",
            "authors": ["Ada Lovelace"],
            "year": 2020,
            "abstract": "First paper that shares its title.",
        },
        {
            "id": SHARED_B,
            "pillar_id": PILLAR_FORMAL,
            "title": "Shared Title",
            "authors": ["Ada Lovelace", "Charles Babbage"],
            "year": 2021,
            "abstract": "Second paper that shares its title.",
        },
    ]
    pillars = [
        {
            "id": PILLAR_NEURAL,
            "name": "Neural Architectures for Language",
            "goal": "Understand attention.",
        },
        {
            "id": PILLAR_FORMAL,
            "name": "Formal Linguistics for NLP",
            "goal": "Master syntax.",
        },
    ]
    notes = [
        {
            "id": "note-attn",
            "paper_id": ATTENTION,
            "problem": "Recurrence is slow.",
            "method": "Self-attention.",
            "findings": ["Attention is all you need."],
            "limitations": ["Quadratic cost."],
            "future_work": ["Sparse attention."],
        }
    ]
    lessons = [
        {
            "id": "lesson-attn",
            "paper_id": ATTENTION,
            "tl_dr": "Replace recurrence with attention.",
            "takeaways": ["Attention is parallel.", "Position is encoded."],
        }
    ]
    cards = [
        {
            "id": "card-attn",
            "paper_id": ATTENTION,
            "question": "What replaces recurrence?",
            "answer": "Self-attention.",
        }
    ]
    episodes = [
        {
            "id": EPISODE_A,
            "paper_id": ATTENTION,
            "pillar_id": PILLAR_NEURAL,
            "title": "Shared Title",
        },
        {
            "id": EPISODE_B,
            "paper_id": SHARED_B,
            "pillar_id": PILLAR_FORMAL,
            "title": "Shared Title",
        },
    ]
    concepts = [
        {
            "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "name": "attention",
            "aliases": ["Self-Attention", "Attention"],
            "description": "A weighted sum of values.",
        }
    ]
    edges = [
        _edge("paper", ATTENTION, "pillar", PILLAR_NEURAL, "in_pillar"),
        _edge("paper", SHARED_A, "pillar", PILLAR_FORMAL, "in_pillar"),
        _edge("paper", SHARED_B, "pillar", PILLAR_FORMAL, "in_pillar"),
        _edge("paper", ATTENTION, "author", "Ashish Vaswani", "authored_by"),
        _edge("paper", ATTENTION, "author", "Noam Shazeer", "authored_by"),
        _edge("paper", SHARED_A, "author", "Ada Lovelace", "authored_by"),
        _edge("paper", SHARED_B, "author", "Ada Lovelace", "authored_by"),
        _edge("paper", SHARED_B, "author", "Charles Babbage", "authored_by"),
        _edge("paper", ATTENTION, "concept", "attention", "covers", weight=2),
        _edge("paper", SHARED_A, "concept", "attention", "covers", weight=1),
        _edge(
            "paper",
            ATTENTION,
            "paper",
            SHARED_A,
            "related_to",
            weight=0.84,
            provenance="derived:embedding:BAAI/bge-base-en-v1.5",
        ),
        _edge("paper", ATTENTION, "paper", SHARED_B, "cites"),
        _edge("paper", SHARED_A, "external_paper", "arxiv:9999.00001", "cites"),
        _edge("paper", SHARED_B, "paper", ATTENTION, "cites"),
    ]
    return VaultSnapshot(
        papers=papers,
        pillars=pillars,
        notes=notes,
        lessons=lessons,
        cards=cards,
        episodes=episodes,
        concepts=concepts,
        edges=edges,
    )


def _edge(src_kind, src_id, dst_kind, dst_id, kind, weight=1, provenance="derived:structure"):
    return {
        "src_kind": src_kind,
        "src_id": src_id,
        "dst_kind": dst_kind,
        "dst_id": dst_id,
        "kind": kind,
        "weight": weight,
        "provenance": provenance,
    }


def _without_attention():
    snap = library()
    snap.papers = [row for row in snap.papers if row["id"] != ATTENTION]
    snap.notes = [row for row in snap.notes if row["paper_id"] != ATTENTION]
    snap.lessons = [row for row in snap.lessons if row["paper_id"] != ATTENTION]
    snap.cards = [row for row in snap.cards if row["paper_id"] != ATTENTION]
    snap.episodes = [row for row in snap.episodes if row["paper_id"] != ATTENTION]
    snap.edges = [
        edge
        for edge in snap.edges
        if edge["src_id"] != ATTENTION and edge["dst_id"] != ATTENTION
    ]
    return snap


def _markdown(root: Path) -> dict:
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.md"))
    }


def test_short_id_is_stable_for_uuids_and_paper_ids():
    """Suffixes do not depend on how many other notes were seen first."""
    assert short_id(EPISODE_A) == "11111111"
    assert short_id("arxiv:1706.03762") == "arxiv-1706.03762"
    assert short_id(EPISODE_A) == short_id(EPISODE_A)


def test_title_collision_keeps_both_notes_and_suffixes_the_id():
    paths = set(render_vault(library()))
    assert "Papers/Attention Is All You Need.md" in paths
    assert "Papers/Shared Title (arxiv-1111.1111).md" in paths
    assert "Papers/Shared Title (arxiv-2222.2222).md" in paths
    assert "Episodes/Shared Title (11111111).md" in paths
    assert "Episodes/Shared Title (22222222).md" in paths
    shared = [path for path in paths if path.startswith("Papers/Shared Title")]
    assert len(shared) == 2


def test_render_ignores_input_order():
    """The same rows in a different order produce the same files."""
    original = render_vault(library())
    snap = library()
    for rows in (
        snap.papers,
        snap.pillars,
        snap.notes,
        snap.lessons,
        snap.cards,
        snap.episodes,
        snap.concepts,
        snap.edges,
    ):
        rows.reverse()
    assert render_vault(snap) == original


def test_paper_note_carries_graph_links_and_a_collapsed_quiz_callout():
    text = render_vault(library())["Papers/Attention Is All You Need.md"]
    assert text.startswith("---\n")
    assert 'pillar: "[[Neural Architectures for Language]]"' in text
    assert "year: 2017" in text
    assert "[[Ashish Vaswani]]" in text
    assert "[[attention]]" in text
    assert "[[Shared Title (arxiv-1111.1111)]]" in text
    assert "related-by-embedding 0.840000 (BAAI/bge-base-en-v1.5)" in text
    assert "[[Shared Title (arxiv-2222.2222)]]" in text
    assert "`lesson-attn`" in text
    assert "> **TL;DR** Replace recurrence with attention." in text
    assert "Recurrence is slow." in text
    assert "- Attention is parallel." in text
    assert "> [!faq]- What replaces recurrence?" in text
    assert "> Self-attention." in text
    assert "[[Shared Title (11111111)]]" in text
    assert START_MARK in text
    assert text.endswith(END_MARK)


def test_external_cite_is_a_wikilink_without_a_note():
    text = render_vault(library())["Papers/Shared Title (arxiv-1111.1111).md"]
    assert "[[arxiv:9999.00001]]" in text
    paths = set(render_vault(library()))
    assert not any("9999.00001" in path for path in paths)


def test_golden_matches_the_fixture_library(tmp_path):
    """The fixture library writes the committed notes and nothing else."""
    report = apply_export(tmp_path, library(), dry_run=False)
    assert report.dry_run is False
    assert report.changed == report.notes
    assert report.notes > 0
    assert report.archived == 0
    assert report.by_type["paper"] == 3
    assert report.by_type["episode"] == 2
    assert _markdown(tmp_path) == _markdown(GOLDEN)


def test_second_run_rewrites_nothing(tmp_path):
    apply_export(tmp_path, library(), dry_run=False)
    mtimes = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*.md")}
    report = apply_export(tmp_path, library(), dry_run=False)
    assert report.changed == 0
    assert report.archived == 0
    assert report.unchanged == report.notes
    assert {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*.md")} == mtimes


def test_dry_run_writes_nothing_and_the_second_dry_run_is_clean(tmp_path):
    report = apply_export(tmp_path, library(), dry_run=True)
    assert report.dry_run is True
    assert report.changed == report.notes
    assert list(tmp_path.iterdir()) == []
    assert "would write" in report.render()
    apply_export(tmp_path, library(), dry_run=False)
    again = apply_export(tmp_path, library(), dry_run=True)
    assert again.changed == 0
    assert again.archived == 0
    assert again.unchanged == again.notes


def test_user_block_survives_regeneration(tmp_path):
    apply_export(tmp_path, library(), dry_run=False)
    path = tmp_path / "Papers" / "Attention Is All You Need.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n\n## My notes\nKeep this.\n", encoding="utf-8")
    snap = library()
    for paper in snap.papers:
        if paper["id"] == ATTENTION:
            paper["abstract"] = "Updated abstract."
    apply_export(tmp_path, snap, dry_run=False)
    text = path.read_text(encoding="utf-8")
    generated, user = text.split(END_MARK, 1)
    assert user == "\n\n## My notes\nKeep this.\n"
    assert "Updated abstract." in generated
    assert "We propose a new network architecture" not in generated
    assert START_MARK in generated
    again = apply_export(tmp_path, snap, dry_run=False)
    assert again.changed == 0
    assert path.read_text(encoding="utf-8") == text


def test_a_marker_inside_an_abstract_does_not_eat_the_user_block(tmp_path):
    snap = library()
    for paper in snap.papers:
        if paper["id"] == ATTENTION:
            paper["abstract"] = "before <!-- /nlp:generated --> after"
    apply_export(tmp_path, snap, dry_run=False)
    path = tmp_path / "Papers" / "Attention Is All You Need.md"
    text = path.read_text(encoding="utf-8")
    assert text.count(END_MARK) == 1
    assert "<!-- /nlp:generated - ->" in text
    path.write_text(text + "USER", encoding="utf-8")
    apply_export(tmp_path, snap, dry_run=False)
    assert path.read_text(encoding="utf-8").endswith("USER")


def test_deleted_paper_is_archived_with_its_user_block(tmp_path):
    apply_export(tmp_path, library(), dry_run=False)
    path = tmp_path / "Papers" / "Attention Is All You Need.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nKeep this.\n", encoding="utf-8")
    report = apply_export(tmp_path, _without_attention(), dry_run=False)
    assert not path.exists()
    archived = tmp_path / "_archive" / "Papers" / "Attention Is All You Need.md"
    assert archived.is_file()
    assert "Keep this." in archived.read_text(encoding="utf-8")
    assert report.archived >= 1
    # Authors and the episode that only existed for this paper move too.
    assert (tmp_path / "_archive" / "Authors" / "Ashish Vaswani.md").is_file()
    assert not (tmp_path / "Authors" / "Ashish Vaswani.md").exists()
    again = apply_export(tmp_path, _without_attention(), dry_run=False)
    assert again.archived == 0
    assert again.changed == 0
    assert archived.is_file()


def test_a_user_file_without_the_marker_is_left_alone(tmp_path):
    apply_export(tmp_path, library(), dry_run=False)
    path = tmp_path / "Papers" / "Attention Is All You Need.md"
    path.write_text("# mine\n", encoding="utf-8")
    report = apply_export(tmp_path, library(), dry_run=False)
    assert path.read_text(encoding="utf-8") == "# mine\n"
    assert report.kept == 1
    assert not (tmp_path / "_archive").exists()


def test_blank_vault_dir_defaults_to_the_home_folder():
    assert resolve_vault_dir(None) == Path.home() / "NLP-Vault"
    assert resolve_vault_dir("  ") == Path.home() / "NLP-Vault"
    assert resolve_vault_dir("~/NLP-Vault") == Path.home() / "NLP-Vault"


class _Query:
    """The select/order/limit/offset/execute chain TableQuery actually has."""

    def __init__(self, table, rows, errors):
        self.table = table
        self.rows = rows
        self.errors = errors
        self.params = {}

    def select(self, columns):
        self.params["select"] = columns
        return self

    def order(self, column, desc=False):
        self.params["order"] = column
        return self

    def limit(self, count):
        self.params["limit"] = str(count)
        return self

    def offset(self, count):
        self.params["offset"] = str(count)
        return self

    def execute(self):
        if self.table in self.errors:
            return {"data": None, "error": self.errors[self.table]}
        keys = [part.strip() for part in self.params["order"].split(",")]
        ordered = sorted(
            self.rows,
            key=lambda row: tuple(str(row.get(key) or "") for key in keys),
        )
        start = int(self.params.get("offset", "0"))
        count = int(self.params["limit"])
        return {"data": ordered[start : start + count], "error": None}


def _client(rows, errors=None):
    """Mock(spec=PostgRESTClient). A missing method raises; the real ones are stubbed."""
    assert callable(PostgRESTClient.table)
    assert callable(TableQuery.select)
    assert callable(TableQuery.order)
    assert callable(TableQuery.limit)
    assert callable(TableQuery.offset)
    assert callable(TableQuery.execute)
    seen = []

    def table(name):
        query = _Query(name, list(rows.get(name, [])), errors or {})
        seen.append(query)
        return query

    client = Mock(spec=PostgRESTClient)
    client.table.side_effect = table
    client.seen = seen
    return client


def _rows_for(snap):
    return {
        "papers": snap.papers,
        "pillars": snap.pillars,
        "notes": snap.notes,
        "lessons": snap.lessons,
        "quiz_cards": snap.cards,
        "podcast_scripts": snap.episodes,
        "concepts": snap.concepts,
        "graph_edges": snap.edges,
    }


def test_loader_uses_the_real_client_and_pages(monkeypatch):
    """One-row pages still return the whole fixture, in the render the golden uses."""
    import nlp_pillars.vault_store as vault_store

    monkeypatch.setattr(vault_store, "_PAGE", 1)
    client = _client(_rows_for(library()))
    monkeypatch.setattr(vault_store.db, "get_client", lambda: client)
    loaded = load_vault_snapshot()
    assert render_vault(loaded) == render_vault(library())
    paper_calls = [query for query in client.seen if query.table == "papers"]
    assert len(paper_calls) >= 3
    assert "abstract" in paper_calls[0].params["select"]
    edge_calls = [query for query in client.seen if query.table == "graph_edges"]
    assert edge_calls[0].params["order"] == (
        "src_kind,src_id,dst_kind,dst_id,kind"
    )
    client.rpc.assert_not_called()


def test_missing_graph_table_names_the_migration(monkeypatch):
    import nlp_pillars.vault_store as vault_store

    client = _client(
        _rows_for(library()),
        errors={
            "graph_edges": {
                "code": "PGRST205",
                "message": "Could not find the table graph_edges in the schema cache",
            }
        },
    )
    monkeypatch.setattr(vault_store.db, "get_client", lambda: client)
    try:
        load_vault_snapshot()
    except VaultStoreError as exc:
        message = str(exc)
    else:
        raise AssertionError("missing graph_edges must fail")
    assert "docs/migrations/020_knowledge_graph.sql" in message


def test_cli_dry_run_prints_counts_and_does_not_require_a_model(monkeypatch, tmp_path):
    seen = {}

    def fake(out, dry_run=False):
        seen["out"] = Path(out)
        seen["dry_run"] = dry_run
        return ExportReport(
            dry_run=True,
            notes=1,
            by_type={"paper": 1},
            changed=1,
            unchanged=0,
            archived=0,
            kept=0,
        )

    monkeypatch.setattr("nlp_pillars.vault_store.export_library", fake)
    result = RUNNER.invoke(
        app, ["vault", "export", "--out", str(tmp_path), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert seen["dry_run"] is True
    assert seen["out"] == tmp_path
    assert "dry-run" in result.output
    assert result.output.count("would write: 1") == 1


def test_export_library_does_not_log_the_report(monkeypatch, caplog, tmp_path):
    """The CLI prints the report. Logging it here showed the counts twice."""
    import nlp_pillars.vault_store as vault_store

    monkeypatch.setattr(vault_store, "load_vault_snapshot", lambda: library())
    with caplog.at_level("INFO"):
        report = export_library(tmp_path, dry_run=True)
    assert "would write" in report.render()
    assert report.render() not in caplog.text
