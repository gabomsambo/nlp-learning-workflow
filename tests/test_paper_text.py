"""Paper text and chunks: the Postgres source of truth.

The write tests execute migration 017's unique index on sqlite, the same way
``tests/test_upload_runs.py`` executes 015's concurrency guard. Nothing here
reaches a database, a vector store, or the network.
"""

import re
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from nlp_pillars.agents.podcast_agent import PodcastAgent
from nlp_pillars.cli import app
from nlp_pillars.db import replace_paper_text
from nlp_pillars.paper_text import (
    MIGRATION,
    PaperTextWriteError,
    apply_replace,
    build_chunk_rows,
    content_hash,
    run_backfill,
)
from nlp_pillars.schemas import PaperRef, ParsedPaper, StageName, StageStatus
from nlp_pillars.tools.pdf_loader import (
    EMBEDDING_ENCODING,
    VECTOR_CHUNK_OVERLAP,
    VECTOR_CHUNK_SIZE,
)

MIGRATION_SQL = (
    Path(__file__).resolve().parents[1] / MIGRATION
).read_text()
SCHEMA_SQL = (
    Path(__file__).resolve().parents[1] / "schema.sql"
).read_text()

VERSION = f"semchunk==test|250/25|{EMBEDDING_ENCODING}"


def _statement(pattern: str) -> str:
    match = re.search(pattern, MIGRATION_SQL, re.S | re.I)
    assert match, f"migration 017 no longer contains a statement matching {pattern!r}"
    return match.group(0)


def _chunk(index: int, text: str, version: str = VERSION) -> dict:
    return {
        "chunk_index": index,
        "text": text,
        "token_count": 4,
        "chunker": "semchunk",
        "chunker_version": version,
        "chunk_size": VECTOR_CHUNK_SIZE,
        "chunk_overlap": VECTOR_CHUNK_OVERLAP,
        "encoding": EMBEDDING_ENCODING,
        "content_hash": content_hash(text),
    }


def _conn() -> sqlite3.Connection:
    """paper_texts / paper_chunks carrying migration 017's own unique index."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.execute(
        "CREATE TABLE paper_texts ("
        " paper_id TEXT PRIMARY KEY,"
        " full_text TEXT NOT NULL,"
        " extractor TEXT NOT NULL,"
        " extracted_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE paper_chunks ("
        " id INTEGER PRIMARY KEY,"
        " paper_id TEXT NOT NULL,"
        " chunk_index INTEGER NOT NULL,"
        " text TEXT NOT NULL,"
        " token_count INTEGER NOT NULL,"
        " chunker TEXT NOT NULL,"
        " chunker_version TEXT NOT NULL,"
        " chunk_size INTEGER NOT NULL,"
        " chunk_overlap INTEGER NOT NULL,"
        " encoding TEXT NOT NULL,"
        " content_hash TEXT NOT NULL,"
        " created_at TEXT NOT NULL DEFAULT (datetime('now')))"
    )
    conn.execute(_statement(
        r"CREATE UNIQUE INDEX[^;]*paper_chunks_paper_idx_version[^;]*;"
    ))
    return conn


# ------------------------------------------------------------------- migration


def test_migration_declares_both_tables_grants_and_reloads_postgrest():
    assert "CREATE TABLE IF NOT EXISTS paper_texts" in MIGRATION_SQL
    assert "CREATE TABLE IF NOT EXISTS paper_chunks" in MIGRATION_SQL
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON paper_texts TO web_anon;" in MIGRATION_SQL
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON paper_chunks TO web_anon;" in MIGRATION_SQL
    assert "GRANT EXECUTE ON FUNCTION replace_paper_text_and_chunks" in MIGRATION_SQL
    assert MIGRATION_SQL.rstrip().endswith("NOTIFY pgrst, 'reload schema';")
    # 017 stores text only. The embedding column arrives in 018 / schema.sql.
    assert re.search(r"(?m)^\s+embedding\s", MIGRATION_SQL) is None
    assert "CREATE TABLE chunk_embeddings" in SCHEMA_SQL


def test_migration_replaces_chunks_in_one_function():
    """Delete and insert live in one function so a failed insert rolls back."""
    delete_at = MIGRATION_SQL.index("DELETE FROM paper_chunks WHERE paper_id = p_paper_id")
    insert_at = MIGRATION_SQL.index("INSERT INTO paper_chunks")
    assert delete_at < insert_at
    assert "ON CONFLICT (paper_id) DO UPDATE" in MIGRATION_SQL


def test_schema_sql_carries_the_same_objects_for_a_fresh_volume():
    assert "CREATE TABLE paper_texts" in SCHEMA_SQL
    assert "CREATE TABLE paper_chunks" in SCHEMA_SQL
    assert "paper_chunks_paper_idx_version" in SCHEMA_SQL
    assert "replace_paper_text_and_chunks" in SCHEMA_SQL
    assert "fk_paper_texts_paper_id" in SCHEMA_SQL
    assert "fk_paper_chunks_paper_id" in SCHEMA_SQL


def test_the_migration_index_rejects_a_duplicate_chunk_identity():
    conn = _conn()
    apply_replace(conn, "arxiv:1", "body", "pymupdf4llm", [_chunk(0, "alpha")])
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO paper_chunks ("
            "paper_id, chunk_index, text, token_count, chunker, chunker_version, "
            "chunk_size, chunk_overlap, encoding, content_hash"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "arxiv:1", 0, "other", 1, "semchunk", VERSION,
                250, 25, EMBEDDING_ENCODING, content_hash("other"),
            ),
        )


# ------------------------------------------------------------------- writes


def test_reingest_replaces_chunks_instead_of_leaving_a_tail():
    """A shorter re-chunk must not keep the previous tail. That is the stale-chunk bug."""
    conn = _conn()
    apply_replace(conn, "arxiv:1", "first body", "pymupdf4llm", [
        _chunk(0, "a"), _chunk(1, "b"), _chunk(2, "c"),
    ])
    apply_replace(conn, "arxiv:1", "second body", "pypdf", [_chunk(0, "only")])

    rows = conn.execute(
        "SELECT chunk_index, text FROM paper_chunks ORDER BY chunk_index"
    ).fetchall()
    assert rows == [(0, "only")]
    assert conn.execute(
        "SELECT full_text, extractor FROM paper_texts"
    ).fetchone() == ("second body", "pypdf")


def test_replacing_the_same_chunks_does_not_duplicate_rows():
    conn = _conn()
    chunks = [_chunk(0, "a"), _chunk(1, "b")]
    apply_replace(conn, "arxiv:1", "body", "pymupdf4llm", chunks)
    apply_replace(conn, "arxiv:1", "body", "pymupdf4llm", chunks)

    assert conn.execute("SELECT count(*) FROM paper_texts").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM paper_chunks").fetchone()[0] == 2


def test_a_failed_insert_keeps_the_previous_chunks():
    """The delete and the insert are one transaction. A bad insert leaves the old rows."""
    conn = _conn()
    apply_replace(conn, "arxiv:1", "body", "pymupdf4llm", [
        _chunk(0, "a"), _chunk(1, "b"),
    ])
    with pytest.raises(sqlite3.IntegrityError):
        apply_replace(conn, "arxiv:1", "new body", "pymupdf4llm", [
            _chunk(0, "x"),
            _chunk(0, "y"),
        ])

    assert conn.execute("SELECT full_text FROM paper_texts").fetchone()[0] == "body"
    texts = [
        row[0]
        for row in conn.execute("SELECT text FROM paper_chunks ORDER BY chunk_index")
    ]
    assert texts == ["a", "b"]


def test_empty_text_and_empty_chunks_are_refused_before_any_delete():
    conn = _conn()
    apply_replace(conn, "arxiv:1", "body", "pymupdf4llm", [_chunk(0, "a")])

    with pytest.raises(PaperTextWriteError):
        apply_replace(conn, "arxiv:1", "   ", "pymupdf4llm", [_chunk(0, "a")])
    with pytest.raises(PaperTextWriteError):
        apply_replace(conn, "arxiv:1", "body", "pymupdf4llm", [])

    assert conn.execute("SELECT count(*) FROM paper_chunks").fetchone()[0] == 1
    assert conn.execute("SELECT full_text FROM paper_texts").fetchone()[0] == "body"


def test_stored_chunks_use_the_vector_chunker_config():
    rows = build_chunk_rows("Attention is all you need. " * 40)
    assert rows
    assert [row["chunk_index"] for row in rows] == list(range(len(rows)))
    for row in rows:
        assert row["chunk_size"] == VECTOR_CHUNK_SIZE == 250
        assert row["chunk_overlap"] == VECTOR_CHUNK_OVERLAP == 25
        assert row["encoding"] == EMBEDDING_ENCODING == "cl100k_base"
        assert "250/25" in row["chunker_version"]
        assert row["chunker"] == "semchunk"
        assert row["token_count"] > 0
        assert row["content_hash"] == content_hash(row["text"])


def test_replace_paper_text_is_one_rpc_and_raises_when_it_fails(monkeypatch):
    client = MagicMock()
    client.rpc.return_value = {"data": 2, "error": None}
    monkeypatch.setattr("nlp_pillars.db.get_client", lambda: client)
    monkeypatch.setattr(
        "nlp_pillars.paper_text.build_chunk_rows",
        lambda _text: [_chunk(0, "a"), _chunk(1, "b")],
    )

    stored = replace_paper_text("arxiv:1", "body text", "pymupdf4llm")

    assert stored == 2
    assert client.rpc.call_count == 1
    fn, payload = client.rpc.call_args.args
    assert fn == "replace_paper_text_and_chunks"
    assert payload["p_paper_id"] == "arxiv:1"
    assert payload["p_extractor"] == "pymupdf4llm"
    assert len(payload["p_chunks"]) == 2
    client.table.assert_not_called()

    client.rpc.return_value = {"data": None, "error": {"code": "PGRST202", "message": "missing"}}
    with pytest.raises(PaperTextWriteError, match="017_paper_text"):
        replace_paper_text("arxiv:1", "body text", "pymupdf4llm")


def test_a_short_chunk_count_from_the_database_is_a_failure(monkeypatch):
    client = MagicMock()
    client.rpc.return_value = {"data": 1, "error": None}
    monkeypatch.setattr("nlp_pillars.db.get_client", lambda: client)
    monkeypatch.setattr(
        "nlp_pillars.paper_text.build_chunk_rows",
        lambda _text: [_chunk(0, "a"), _chunk(1, "b")],
    )
    with pytest.raises(PaperTextWriteError, match="expected 2"):
        replace_paper_text("arxiv:1", "body text", "pymupdf4llm")


# ------------------------------------------------------------------- backfill


def _papers():
    return [
        PaperRef(
            id="file:aaa", title="Upload", authors=[],
            url_pdf="file:///app/data/uploads/aaa.pdf",
        ),
        PaperRef(
            id="arxiv:1", title="URL", authors=[],
            url_pdf="https://arxiv.org/pdf/1706.03762",
        ),
        PaperRef(id="bare", title="No PDF", authors=[]),
    ]


def test_url_papers_are_redownloaded_and_uploads_stay_on_disk(monkeypatch):
    from nlp_pillars.paper_text import load_paper_body
    from nlp_pillars.tools.pdf_loader import ExtractedText

    calls = []

    def fake_download(url, cache_dir=".cache/pdfs", *, use_cache=True):
        calls.append((url, use_cache))
        return "/tmp/paper.pdf"

    monkeypatch.setattr("nlp_pillars.paper_text.download_pdf", fake_download)
    monkeypatch.setattr(
        "nlp_pillars.paper_text.extract_document",
        lambda _path: ExtractedText("body", "pymupdf4llm"),
    )

    text, extractor = load_paper_body(_papers()[1])
    assert (text, extractor) == ("body", "pymupdf4llm")
    assert calls == [("https://arxiv.org/pdf/1706.03762", False)]

    calls.clear()
    load_paper_body(_papers()[0])
    assert calls == [("file:///app/data/uploads/aaa.pdf", True)]


def test_backfill_dry_run_counts_and_does_not_extract_or_write():
    extract = MagicMock()
    write = MagicMock()
    report = run_backfill(
        _papers(), ["arxiv:1"], dry_run=True, extract=extract, write=write,
    )

    assert report.papers == 3
    assert report.uploads == 1
    assert report.urls == 1
    assert report.skipped == 1
    assert report.already_stored == 1
    assert report.would_write == 2
    assert report.written == 0
    extract.assert_not_called()
    write.assert_not_called()
    rendered = report.render()
    assert "dry_run: true" in rendered
    assert "would_write: 2" in rendered

    again = run_backfill(
        _papers(), ["arxiv:1"], dry_run=True, extract=extract, write=write,
    )
    assert again.render() == rendered


def test_backfill_replace_is_idempotent():
    conn = _conn()
    papers = [_papers()[1]]

    def extract(_paper):
        return ("Hello chunk. " * 5, "pymupdf4llm")

    def write(paper_id, text, extractor):
        return apply_replace(conn, paper_id, text, extractor, build_chunk_rows(text))

    first = run_backfill(papers, [], dry_run=False, extract=extract, write=write)
    n_chunks = conn.execute("SELECT count(*) FROM paper_chunks").fetchone()[0]
    assert first.failed == 0
    assert first.written == 1
    assert first.chunks == n_chunks >= 1

    second = run_backfill(
        papers, ["arxiv:1"], dry_run=False, extract=extract, write=write,
    )
    assert second.written == 1
    assert second.replaced == 1
    assert second.failed == 0
    assert conn.execute("SELECT count(*) FROM paper_chunks").fetchone()[0] == n_chunks
    assert conn.execute("SELECT count(*) FROM paper_texts").fetchone()[0] == 1


def test_cli_dry_run_prints_counts_without_downloading():
    with (
        patch("nlp_pillars.cli.db.list_all_papers", return_value=_papers()),
        patch("nlp_pillars.cli.db.list_stored_paper_text_ids", return_value=["arxiv:1"]),
        patch("nlp_pillars.paper_text.download_pdf") as download,
        patch("nlp_pillars.paper_text.extract_document") as extract,
        patch("nlp_pillars.db.replace_paper_text") as write,
    ):
        result = CliRunner().invoke(app, ["text", "backfill", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "dry_run: true" in result.output
    assert "would_write: 2" in result.output
    assert "uploads: 1" in result.output
    assert "urls: 1" in result.output
    download.assert_not_called()
    extract.assert_not_called()
    write.assert_not_called()


def test_cli_exits_nonzero_when_a_paper_fails_to_store():
    papers = [_papers()[1]]
    with (
        patch("nlp_pillars.cli.db.list_all_papers", return_value=papers),
        patch("nlp_pillars.cli.db.list_stored_paper_text_ids", return_value=[]),
        patch(
            "nlp_pillars.paper_text.load_paper_body",
            side_effect=PaperTextWriteError("download refused"),
        ),
    ):
        result = CliRunner().invoke(app, ["text", "backfill"])

    assert result.exit_code == 1
    assert "failed: 1" in result.output
    assert "download refused" in result.output


# ------------------------------------------------------------------- podcast read


def test_podcast_reads_stored_text_and_does_not_ingest():
    with patch.object(PodcastAgent, "__init__", lambda self: None):
        agent = PodcastAgent()
        agent.ingest_agent = MagicMock()
        paper = PaperRef(
            id="arxiv:1", title="T", authors=[], url_pdf="https://example.com/p.pdf",
        )
        with patch(
            "nlp_pillars.agents.podcast_agent.get_stored_paper_text",
            return_value="Stored body of the paper.",
        ):
            result = agent._get_full_text(paper)

    assert result.text == "Stored body of the paper."
    assert result.error is None
    agent.ingest_agent.ingest.assert_not_called()


def test_podcast_falls_back_to_stubbed_ingest_when_nothing_is_stored():
    with patch.object(PodcastAgent, "__init__", lambda self: None):
        agent = PodcastAgent()
        parsed = MagicMock()
        parsed.full_text = "Parsed from the PDF."
        agent.ingest_agent = MagicMock()
        agent.ingest_agent.ingest.return_value = parsed
        paper = PaperRef(
            id="arxiv:1", title="T", authors=[], url_pdf="https://example.com/p.pdf",
        )
        with patch(
            "nlp_pillars.agents.podcast_agent.get_stored_paper_text",
            return_value=None,
        ):
            result = agent._get_full_text(paper)

    assert result.text == "Parsed from the PDF."
    agent.ingest_agent.ingest.assert_called_once_with(paper)


def test_podcast_falls_back_to_stubbed_ingest_when_the_read_fails():
    with patch.object(PodcastAgent, "__init__", lambda self: None):
        agent = PodcastAgent()
        parsed = MagicMock()
        parsed.full_text = "Parsed after a missing table."
        agent.ingest_agent = MagicMock()
        agent.ingest_agent.ingest.return_value = parsed
        paper = PaperRef(
            id="arxiv:1", title="T", authors=[], url_pdf="https://example.com/p.pdf",
        )
        with patch(
            "nlp_pillars.agents.podcast_agent.get_stored_paper_text",
            side_effect=PaperTextWriteError("paper_texts is missing"),
        ):
            result = agent._get_full_text(paper)

    assert "Parsed after a missing table." in result.text
    agent.ingest_agent.ingest.assert_called_once()


# ------------------------------------------------------------------- loud failure


class _Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, name, status, detail=None):
        self.events.append((name, status, detail))

    def status_of(self, stage):
        seen = [event for event in self.events if event[0] == stage.value]
        return seen[-1][1] if seen else None


def test_upload_text_write_failure_fails_the_persist_stage(tmp_path):
    from nlp_pillars.services.upload_service import UploadService

    service = UploadService(upload_dir=str(tmp_path / "uploads"))
    paper = PaperRef(
        id="arxiv:1706.03762", title="A Paper", authors=["A"], url_pdf="x",
    )
    parsed = ParsedPaper(
        paper_ref=paper,
        full_text="Body text that must be stored.",
        chunks=["Body text that must be stored."],
        extractor="pymupdf4llm",
    )
    recorder = _Recorder()
    with (
        patch.object(service.ingest_agent, "ingest", return_value=parsed),
        patch("nlp_pillars.services.upload_service.db") as mock_db,
        patch("nlp_pillars.services.upload_service.vectors") as mock_vectors,
    ):
        mock_db.replace_paper_text.side_effect = PaperTextWriteError(
            "paper text for arxiv:1706.03762 was not stored"
        )
        mock_vectors.upsert_text.return_value = 4
        outcome = service._run_full_pipeline(
            paper=paper,
            pillar_id="neural-architectures-language",
            run_summarizer=False,
            generate_quiz=False,
            on_stage=recorder,
        )

    assert not outcome.ok
    assert any(error.startswith("paper_text:") for error in outcome.errors)
    assert recorder.status_of(StageName.PERSIST) == StageStatus.FAILED.value
    # Qdrant upsert is unchanged and still runs; the text failure is its own fact.
    mock_vectors.upsert_text.assert_called_once()
