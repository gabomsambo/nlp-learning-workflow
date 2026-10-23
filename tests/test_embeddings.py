"""Embedding providers and chunk_embeddings.

Hermetic: a fake provider stands in for the model. The local class is not
constructed in a way that downloads weights, and nothing here opens a socket.
"""

import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from nlp_pillars.chunk_embeddings import (
    embed_chunk_rows,
    embed_paper,
    run_embedding_backfill,
)
from nlp_pillars.cli import app
from nlp_pillars.db import (
    insert_chunk_embeddings,
    list_chunks_missing_embeddings,
    paper_has_chunks,
)
from nlp_pillars.embeddings import (
    LOCAL_MODEL_ID,
    ONNX_REVISION,
    EmbeddingConfigError,
    EmbeddingError,
    LocalBgeEmbeddingProvider,
    OpenAIEmbeddingProvider,
    cache_dir,
    ensure_local_model,
    local_model_dir,
    resolve_embedding_selection,
    selection_for_backfill,
)
from nlp_pillars.schemas import PaperRef, ParsedPaper, StageName, StageStatus

MIGRATION_SQL = (
    Path(__file__).resolve().parents[1] / "docs/migrations/018_chunk_embeddings.sql"
).read_text()
SCHEMA_SQL = (Path(__file__).resolve().parents[1] / "schema.sql").read_text()
EVAL_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/eval_embeddings.py"
).read_text()
CI_WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github/workflows/tests.yml"
).read_text()


class FakeProvider:
    """In-process stand-in. No weights, no network."""

    model_id = "fake-model"
    dims = 2
    query_prefix = "Represent this sentence for searching relevant passages: "
    document_prefix = ""
    batch_size = 2

    def __init__(self):
        self.documents = []

    def embed_documents(self, texts):
        self.documents.append(list(texts))
        return [[float(len(text)), 0.0] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([self.query_prefix + text.strip()])[0]


def _chunk(chunk_id, text):
    return {"chunk_id": chunk_id, "chunk_text": text, "paper_id": "arxiv:1", "chunk_index": 0}


# ------------------------------------------------------------------- migration


def test_migration_stores_real_arrays_and_reloads_postgrest():
    assert "CREATE TABLE IF NOT EXISTS chunk_embeddings" in MIGRATION_SQL
    assert "embedding  REAL[] NOT NULL" in MIGRATION_SQL
    assert "bytea" in MIGRATION_SQL
    assert "pgvector" in MIGRATION_SQL
    assert "cardinality(embedding) = dims" in MIGRATION_SQL
    assert "ON DELETE CASCADE" in MIGRATION_SQL
    assert "CREATE UNIQUE INDEX IF NOT EXISTS chunk_embeddings_chunk_model" in MIGRATION_SQL
    assert "(chunk_id, model_id)" in MIGRATION_SQL
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON chunk_embeddings TO web_anon;" in MIGRATION_SQL
    assert "GRANT EXECUTE ON FUNCTION chunks_missing_embeddings(TEXT, TEXT, INTEGER, INTEGER)" in MIGRATION_SQL
    assert MIGRATION_SQL.rstrip().endswith("NOTIFY pgrst, 'reload schema';")


def test_schema_sql_carries_the_same_objects():
    assert "CREATE TABLE chunk_embeddings" in SCHEMA_SQL
    assert "embedding REAL[] NOT NULL" in SCHEMA_SQL
    assert "chunk_embeddings_chunk_model" in SCHEMA_SQL
    assert "chunks_missing_embeddings" in SCHEMA_SQL
    assert "fk_chunk_embeddings_chunk_id" in SCHEMA_SQL
    assert "ON DELETE CASCADE" in SCHEMA_SQL
    assert "GRANT EXECUTE ON FUNCTION chunks_missing_embeddings" in SCHEMA_SQL


def test_unique_pair_is_enforced_by_the_migration_index():
    match = re.search(
        r"CREATE UNIQUE INDEX IF NOT EXISTS chunk_embeddings_chunk_model[^;]*;",
        MIGRATION_SQL,
        re.S,
    )
    assert match, "migration 018 no longer creates the unique index"
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE chunk_embeddings ("
        " chunk_id TEXT NOT NULL,"
        " model_id TEXT NOT NULL,"
        " dims INTEGER NOT NULL,"
        " embedding TEXT NOT NULL,"
        " created_at TEXT NOT NULL)"
    )
    conn.execute(match.group(0))
    conn.execute(
        "INSERT INTO chunk_embeddings VALUES ('c1', 'model-a', 2, '[1,2]', 't')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO chunk_embeddings VALUES ('c1', 'model-a', 2, '[3,4]', 't')"
        )
    conn.execute(
        "INSERT INTO chunk_embeddings VALUES ('c1', 'model-b', 2, '[3,4]', 't')"
    )
    assert conn.execute("SELECT count(*) FROM chunk_embeddings").fetchone()[0] == 2


# ------------------------------------------------------------------- providers


def test_default_selection_is_the_pinned_local_model():
    assert resolve_embedding_selection(None, None) == ("local", LOCAL_MODEL_ID)
    assert resolve_embedding_selection("local", "") == ("local", LOCAL_MODEL_ID)
    # A leftover OpenAI model id belongs to the Qdrant upsert, not this store.
    assert resolve_embedding_selection("local", "text-embedding-3-small") == (
        "local",
        LOCAL_MODEL_ID,
    )
    assert resolve_embedding_selection("local", "BAAI/bge-base-en-v1.5") == (
        "local",
        LOCAL_MODEL_ID,
    )


def test_openai_provider_is_explicit_and_keeps_known_dims():
    assert resolve_embedding_selection("openai", None) == (
        "openai",
        "text-embedding-3-small",
    )
    assert resolve_embedding_selection("openai", "text-embedding-3-large") == (
        "openai",
        "text-embedding-3-large",
    )
    assert selection_for_backfill(
        "text-embedding-3-small", "local", "BAAI/bge-base-en-v1.5",
    ) == ("openai", "text-embedding-3-small")
    with pytest.raises(EmbeddingConfigError):
        resolve_embedding_selection("openai", "not-a-model")
    with pytest.raises(EmbeddingConfigError):
        resolve_embedding_selection("qdrant", None)
    with pytest.raises(EmbeddingConfigError):
        resolve_embedding_selection("local", "some-other-model")


def test_local_provider_class_implements_the_interface_without_loading():
    """Mock(spec=...) fails if the real class drops an interface member.

    Constructing the class must not download weights.
    """
    double = Mock(spec=LocalBgeEmbeddingProvider)
    double.embed_documents(["passage"])
    double.embed_query("query")
    for name in ("model_id", "dims", "query_prefix", "document_prefix", "batch_size"):
        getattr(double, name)
    with pytest.raises(AttributeError):
        double.embed_passage(["no such method"])

    assert LocalBgeEmbeddingProvider.model_id == LOCAL_MODEL_ID
    assert ONNX_REVISION in LocalBgeEmbeddingProvider.model_id
    assert LocalBgeEmbeddingProvider.dims == 768
    assert LocalBgeEmbeddingProvider.batch_size == 32
    assert LocalBgeEmbeddingProvider.query_prefix == (
        "Represent this sentence for searching relevant passages: "
    )
    assert LocalBgeEmbeddingProvider.query_prefix.endswith(" ")
    assert LocalBgeEmbeddingProvider.document_prefix == ""

    provider = LocalBgeEmbeddingProvider(cache_dir=Path("/no/such/embedding-cache"))
    assert provider._onnx is None


def test_local_provider_applies_the_query_prefix_once():
    seen = []

    class _Onnx:
        def embed(self, documents, batch_size=32, parallel=None):
            seen.extend(documents)
            return [[0.1] * 768 for _ in documents]

    provider = LocalBgeEmbeddingProvider(cache_dir=Path("/no/such/embedding-cache"))
    provider._onnx = _Onnx()

    query = provider.embed_query("  what is attention?  ")
    assert seen == [
        "Represent this sentence for searching relevant passages: what is attention?"
    ]
    assert len(query) == 768

    seen.clear()
    docs = provider.embed_documents(["  passage text  "])
    assert seen == ["passage text"]
    assert len(docs) == 1
    assert len(docs[0]) == 768


def test_cached_onnx_files_skip_the_download(tmp_path, monkeypatch):
    dest = local_model_dir(tmp_path)
    dest.mkdir()
    for name in ("model_optimized.onnx", "tokenizer.json"):
        (dest / name).write_bytes(b"present")

    def boom(*_args, **_kwargs):
        raise AssertionError("snapshot_download should not run")

    monkeypatch.setattr("nlp_pillars.embeddings._snapshot_download", boom)
    assert ensure_local_model(tmp_path) == dest


def test_cache_dir_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(tmp_path / "models"))
    path = cache_dir()
    assert path == tmp_path / "models"
    assert path.is_dir()


def test_openai_provider_orders_by_index_and_strips():
    items = [
        SimpleNamespace(index=1, embedding=[0.2] * 1536),
        SimpleNamespace(index=0, embedding=[0.1] * 1536),
    ]
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(data=items)
    provider = OpenAIEmbeddingProvider(client=client)

    vectors = provider.embed_documents(["  alpha  ", "beta"])

    assert vectors[0][0] == 0.1
    assert vectors[1][0] == 0.2
    assert len(vectors[0]) == 1536
    _args, kwargs = client.embeddings.create.call_args
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["input"] == ["alpha", "beta"]

    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[0.3] * 1536)]
    )
    provider.embed_query("  query  ")
    assert client.embeddings.create.call_args.kwargs["input"] == ["query"]


def test_openai_provider_rejects_a_short_batch():
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[0.1] * 1536)]
    )
    provider = OpenAIEmbeddingProvider(client=client)
    with pytest.raises(EmbeddingError, match="2 inputs"):
        provider.embed_documents(["a", "b"])


# ------------------------------------------------------------------- backfill


def test_backfill_is_batched_and_idempotent():
    provider = FakeProvider()
    stored = {}

    def save(rows):
        for row in rows:
            key = (row["chunk_id"], row["model_id"])
            assert key not in stored
            assert row["dims"] == 2
            assert len(row["embedding"]) == 2
            stored[key] = row
        return len(rows)

    chunks = [_chunk("c1", "one"), _chunk("c2", "two"), _chunk("c3", "three")]
    report = run_embedding_backfill(
        model_id=provider.model_id,
        dry_run=False,
        chunks=chunks,
        embed=lambda batch: embed_chunk_rows(batch, provider, save),
    )
    assert report.missing == 3
    assert report.written == 3
    assert report.failed == 0
    assert [len(batch) for batch in provider.documents] == [2, 1]
    assert "written: 3" in report.render()

    again = run_embedding_backfill(
        model_id=provider.model_id,
        dry_run=False,
        chunks=[],
        embed=lambda batch: embed_chunk_rows(batch, provider, save),
    )
    assert again.missing == 0
    assert again.written == 0
    assert len(stored) == 3


def test_dry_run_counts_and_does_not_embed():
    called = []
    report = run_embedding_backfill(
        model_id="fake-model",
        dry_run=True,
        chunks=[_chunk("c1", "one")],
        embed=lambda batch: called.append(batch) or 1,
    )
    assert report.would_write == 1
    assert report.written == 0
    assert called == []
    assert "dry_run: true" in report.render()
    assert "would_write: 1" in report.render()


def test_a_failed_batch_keeps_earlier_rows_and_reports_the_count():
    class Flaky(FakeProvider):
        batch_size = 1

        def embed_documents(self, texts):
            if texts[0] == "boom":
                raise EmbeddingError("model exploded")
            return super().embed_documents(texts)

    saved = []
    report = run_embedding_backfill(
        model_id="fake-model",
        dry_run=False,
        chunks=[_chunk("c1", "ok"), _chunk("c2", "boom")],
        embed=lambda batch: embed_chunk_rows(batch, Flaky(), lambda rows: saved.extend(rows) or len(rows)),
    )
    assert report.written == 1
    assert report.failed == 1
    assert "model exploded" in report.failures[0]
    assert [row["chunk_id"] for row in saved] == ["c1"]


def test_wrong_dims_are_not_stored():
    class Wrong(FakeProvider):
        def embed_documents(self, texts):
            return [[0.0, 0.0, 0.0] for _ in texts]

    with pytest.raises(EmbeddingError, match="expected 2"):
        embed_chunk_rows([_chunk("c1", "t")], Wrong(), lambda rows: len(rows))


# ------------------------------------------------------------------- database client


def test_missing_chunks_are_paged(monkeypatch):
    pages = [
        [{"chunk_id": "c1", "chunk_text": "a"}],
        [{"chunk_id": "c2", "chunk_text": "b"}],
        [],
    ]
    client = Mock()
    client.rpc.side_effect = [{"data": page, "error": None} for page in pages]
    monkeypatch.setattr("nlp_pillars.db.get_client", lambda: client)
    monkeypatch.setattr("nlp_pillars.db._PAGE_SIZE", 1)

    rows = list_chunks_missing_embeddings("model-a", "arxiv:1")

    assert [row["chunk_id"] for row in rows] == ["c1", "c2"]
    # A full page is followed by one more request; the empty page stops the walk.
    assert client.rpc.call_count == 3
    assert client.rpc.call_args_list[0].args[0] == "chunks_missing_embeddings"
    assert client.rpc.call_args_list[0].args[1]["p_paper_id"] == "arxiv:1"
    assert client.rpc.call_args_list[1].args[1]["p_offset"] == 1


def test_a_missing_embeddings_function_names_the_migration(monkeypatch):
    client = Mock()
    client.rpc.return_value = {
        "data": None,
        "error": {"code": "PGRST202", "message": "Could not find the function chunks_missing_embeddings"},
    }
    monkeypatch.setattr("nlp_pillars.db.get_client", lambda: client)
    with pytest.raises(EmbeddingError, match="018_chunk_embeddings"):
        list_chunks_missing_embeddings("model-a")


def test_insert_posts_rows_and_names_a_missing_table(monkeypatch):
    response = Mock(status_code=201)
    client = Mock()
    client.base_url = "http://postgrest"
    client.headers = {"Prefer": "return=representation"}
    client.client.post.return_value = response
    monkeypatch.setattr("nlp_pillars.db.get_client", lambda: client)

    stored = insert_chunk_embeddings([
        {"chunk_id": "c1", "model_id": "m", "dims": 2, "embedding": [0.1, 0.2]},
    ])
    assert stored == 1
    payload = client.client.post.call_args.kwargs["json"]
    assert payload[0]["embedding"] == [0.1, 0.2]
    assert client.client.post.call_args.kwargs["headers"]["Prefer"] == "return=minimal"

    response.status_code = 404
    response.json.return_value = {
        "code": "PGRST205",
        "message": "Could not find the table chunk_embeddings in the schema cache",
    }
    with pytest.raises(EmbeddingError, match="018_chunk_embeddings"):
        insert_chunk_embeddings([
            {"chunk_id": "c1", "model_id": "m", "dims": 2, "embedding": [0.1, 0.2]},
        ])


def test_embed_paper_skips_chunks_that_already_have_the_model(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(
        "nlp_pillars.db.list_chunks_missing_embeddings",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr("nlp_pillars.db.paper_has_chunks", lambda _paper_id: True)
    assert embed_paper("arxiv:1", provider=provider) == 0
    assert provider.documents == []


def test_embed_paper_without_chunks_fails(monkeypatch):
    monkeypatch.setattr(
        "nlp_pillars.db.list_chunks_missing_embeddings",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr("nlp_pillars.db.paper_has_chunks", lambda _paper_id: False)
    with pytest.raises(EmbeddingError, match="no chunks stored"):
        embed_paper("arxiv:1", provider=FakeProvider())


def test_paper_has_chunks_reads_one_row(monkeypatch):
    client = Mock()
    query = client.table.return_value.select.return_value.eq.return_value.limit.return_value
    query.execute.return_value = {"data": [{"id": "c1"}], "error": None}
    monkeypatch.setattr("nlp_pillars.db.get_client", lambda: client)
    assert paper_has_chunks("arxiv:1") is True
    client.table.assert_called_once_with("paper_chunks")


# ------------------------------------------------------------------- CLI


def test_cli_dry_run_does_not_load_a_provider():
    chunks = [_chunk("c1", "one"), _chunk("c2", "two")]
    with (
        patch("nlp_pillars.config.get_settings") as settings,
        patch("nlp_pillars.db.list_chunks_missing_embeddings", return_value=chunks) as listed,
        patch("nlp_pillars.embeddings.get_embedding_provider") as build,
    ):
        settings.return_value.embedding_provider = "local"
        settings.return_value.embedding_model = "BAAI/bge-base-en-v1.5"
        result = CliRunner().invoke(app, ["embeddings", "backfill", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "dry_run: true" in result.output
    assert f"model_id: {LOCAL_MODEL_ID}" in result.output
    assert "missing: 2" in result.output
    assert "would_write: 2" in result.output
    listed.assert_called_once_with(LOCAL_MODEL_ID)
    build.assert_not_called()


def test_cli_model_flag_selects_openai_without_calling_it_on_dry_run():
    with (
        patch("nlp_pillars.config.get_settings") as settings,
        patch("nlp_pillars.db.list_chunks_missing_embeddings", return_value=[]) as listed,
        patch("nlp_pillars.embeddings.get_embedding_provider") as build,
    ):
        settings.return_value.embedding_provider = "local"
        settings.return_value.embedding_model = "BAAI/bge-base-en-v1.5"
        result = CliRunner().invoke(
            app, ["embeddings", "backfill", "--dry-run", "--model", "text-embedding-3-small"],
        )

    assert result.exit_code == 0, result.output
    assert "model_id: text-embedding-3-small" in result.output
    listed.assert_called_once_with("text-embedding-3-small")
    build.assert_not_called()


def test_cli_exits_nonzero_when_the_table_is_missing():
    with (
        patch("nlp_pillars.config.get_settings") as settings,
        patch(
            "nlp_pillars.db.list_chunks_missing_embeddings",
            side_effect=EmbeddingError("missing table. Apply docs/migrations/018_chunk_embeddings.sql by hand."),
        ),
    ):
        settings.return_value.embedding_provider = "local"
        settings.return_value.embedding_model = "BAAI/bge-base-en-v1.5"
        result = CliRunner().invoke(app, ["embeddings", "backfill", "--dry-run"])

    assert result.exit_code == 1
    assert "018_chunk_embeddings" in result.output


# ------------------------------------------------------------------- run stages


class _Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, name, status, detail=None):
        self.events.append((name, status, detail))

    def status_of(self, stage):
        seen = [event for event in self.events if event[0] == stage.value]
        return seen[-1][1] if seen else None

    def detail_of(self, stage):
        seen = [event for event in self.events if event[0] == stage.value]
        return seen[-1][2] if seen else None


def test_upload_embedding_failure_fails_persist_and_still_upserts(tmp_path):
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
        mock_db.replace_paper_text.return_value = 3
        mock_db.embed_paper_chunks.side_effect = EmbeddingError("model weights missing")
        mock_vectors.upsert_text.return_value = 4
        outcome = service._run_full_pipeline(
            paper=paper,
            pillar_id="neural-architectures-language",
            run_summarizer=False,
            generate_quiz=False,
            on_stage=recorder,
        )

    assert not outcome.ok
    assert any(error.startswith("embeddings:") for error in outcome.errors)
    assert recorder.status_of(StageName.PERSIST) == StageStatus.FAILED.value
    mock_vectors.upsert_text.assert_called_once()


def test_upload_stores_embeddings_after_the_text(tmp_path):
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
        mock_db.replace_paper_text.return_value = 3
        mock_db.embed_paper_chunks.return_value = 3
        mock_vectors.upsert_text.return_value = 3
        outcome = service._run_full_pipeline(
            paper=paper,
            pillar_id="neural-architectures-language",
            run_summarizer=False,
            generate_quiz=False,
            on_stage=recorder,
        )

    assert outcome.ok
    mock_db.embed_paper_chunks.assert_called_once_with("arxiv:1706.03762")
    assert recorder.status_of(StageName.PERSIST) == StageStatus.COMPLETED.value
    assert "3 embedding(s)" in recorder.detail_of(StageName.PERSIST)
    mock_vectors.upsert_text.assert_called_once()


# ------------------------------------------------------------------- eval script stays out of CI


def test_eval_script_is_not_part_of_ci():
    assert "SciFact" in EVAL_SCRIPT
    assert "nDCG" in EVAL_SCRIPT
    assert 'if __name__' in EVAL_SCRIPT
    assert "def test_" not in EVAL_SCRIPT
    assert "eval_embeddings" not in CI_WORKFLOW
