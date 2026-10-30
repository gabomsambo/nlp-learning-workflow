"""Title+abstract paper vectors.

Hermetic: a fake provider stands in for the model. Nothing here opens a
socket or constructs the local ONNX class in a way that downloads weights.
"""

import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from typer.testing import CliRunner

from nlp_pillars.cli import app
from nlp_pillars.db import insert_paper_vectors, list_paper_vector_ref_ids
from nlp_pillars.embeddings import (
    LOCAL_MODEL_ID,
    EmbeddingError,
    LocalBgeEmbeddingProvider,
)
from nlp_pillars.paper_vectors import (
    SOURCE_TITLE_ABSTRACT,
    embed_paper_vector_rows,
    pairwise_spread,
    run_paper_vector_backfill,
    title_abstract_text,
)

MIGRATION_SQL = (
    Path(__file__).resolve().parents[1] / "docs/migrations/021_paper_vectors.sql"
).read_text()
SCHEMA_SQL = (Path(__file__).resolve().parents[1] / "schema.sql").read_text()
SPREAD_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/paper_vector_spread.py"
)


class FakeProvider:
    """In-process stand-in. No weights, no network."""

    model_id = "BAAI/bge-base-en-v1.5@test"
    dims = 2
    query_prefix = "Represent this sentence for searching relevant passages: "
    document_prefix = ""
    batch_size = 2

    def __init__(self):
        self.documents = []
        self.queries = []

    def embed_documents(self, texts):
        self.documents.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return [0.0, 1.0]


def _paper(paper_id, title, abstract="About attention."):
    return SimpleNamespace(id=paper_id, title=title, abstract=abstract)


def test_migration_stores_one_vector_per_ref_model_and_source():
    assert "CREATE TABLE IF NOT EXISTS paper_vectors" in MIGRATION_SQL
    assert "vec        REAL[] NOT NULL" in MIGRATION_SQL
    assert "cardinality(vec) = dims" in MIGRATION_SQL
    assert "DEFAULT 'title_abstract'" in MIGRATION_SQL
    assert "'paper', 'external_paper'" in MIGRATION_SQL
    assert "REFERENCES papers" not in MIGRATION_SQL
    assert (
        "CREATE UNIQUE INDEX IF NOT EXISTS paper_vectors_ref_model_source"
        in MIGRATION_SQL
    )
    assert "(ref_kind, ref_id, model_id, source)" in MIGRATION_SQL
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON paper_vectors TO web_anon;" in MIGRATION_SQL
    assert MIGRATION_SQL.rstrip().endswith("NOTIFY pgrst, 'reload schema';")


def test_schema_sql_carries_the_same_objects():
    assert "CREATE TABLE paper_vectors" in SCHEMA_SQL
    assert "vec REAL[] NOT NULL" in SCHEMA_SQL
    assert "paper_vectors_ref_model_source" in SCHEMA_SQL
    assert "paper_vectors_dims_match" in SCHEMA_SQL
    assert "DEFAULT 'title_abstract'" in SCHEMA_SQL


def test_unique_ref_model_source_is_enforced_by_the_migration_index():
    match = re.search(
        r"CREATE UNIQUE INDEX IF NOT EXISTS paper_vectors_ref_model_source[^;]*;",
        MIGRATION_SQL,
        re.S,
    )
    assert match, "migration 021 no longer creates the unique index"
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE paper_vectors ("
        " ref_kind TEXT NOT NULL, ref_id TEXT NOT NULL,"
        " model_id TEXT NOT NULL, source TEXT NOT NULL,"
        " dims INTEGER NOT NULL, vec TEXT NOT NULL, created_at TEXT)"
    )
    conn.execute(match.group(0))
    conn.execute(
        "INSERT INTO paper_vectors VALUES"
        " ('paper', 'arxiv:1', 'model-a', 'title_abstract', 2, '[1,2]', 't')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO paper_vectors VALUES"
            " ('paper', 'arxiv:1', 'model-a', 'title_abstract', 2, '[3,4]', 't')"
        )
    conn.execute(
        "INSERT INTO paper_vectors VALUES"
        " ('paper', 'arxiv:1', 'model-b', 'title_abstract', 2, '[3,4]', 't')"
    )
    conn.execute(
        "INSERT INTO paper_vectors VALUES"
        " ('external_paper', 'arxiv:1', 'model-a', 'title_abstract', 2, '[3,4]', 't')"
    )
    assert conn.execute("SELECT count(*) FROM paper_vectors").fetchone()[0] == 3


def test_title_and_abstract_join_on_a_blank_line_and_blank_papers_are_empty():
    assert title_abstract_text("  Attention  ", "  Is all you need. ") == (
        "Attention\n\nIs all you need."
    )
    assert title_abstract_text("Attention", None) == "Attention"
    assert title_abstract_text("  ", "   ") == ""


def test_fake_provider_embeds_title_abstract_and_a_second_pass_writes_nothing():
    provider = FakeProvider()
    saved = []

    def save(rows):
        saved.extend(rows)
        return len(rows)

    papers = [
        _paper("arxiv:1", "Attention", "Transformers."),
        _paper("arxiv:2", "BERT", None),
        _paper("arxiv:3", "   ", None),
    ]
    first = run_paper_vector_backfill(
        model_id=provider.model_id,
        dry_run=False,
        papers=papers,
        already=set(),
        embed=lambda batch: embed_paper_vector_rows(batch, provider, save),
    )
    assert first.library == 3
    assert first.missing == 2
    assert first.skipped_empty == 1
    assert first.written == 2
    assert first.failed == 0
    assert provider.queries == []
    assert provider.documents == [
        ["Attention\n\nTransformers.", "BERT"],
    ]
    assert saved[0]["ref_kind"] == "paper"
    assert saved[0]["source"] == SOURCE_TITLE_ABSTRACT
    assert saved[0]["model_id"] == provider.model_id
    assert saved[0]["dims"] == 2
    assert saved[0]["vec"] == [float(len("Attention\n\nTransformers.")), 1.0]

    already = {row["ref_id"] for row in saved}
    provider.documents.clear()
    second = run_paper_vector_backfill(
        model_id=provider.model_id,
        dry_run=False,
        papers=papers,
        already=already,
        embed=lambda batch: embed_paper_vector_rows(batch, provider, save),
    )
    assert second.already == 2
    assert second.missing == 0
    assert second.written == 0
    assert provider.documents == []
    assert len(saved) == 2


def test_dry_run_does_not_call_the_provider():
    provider = FakeProvider()

    def embed(_batch):
        raise AssertionError("dry-run must not embed")

    report = run_paper_vector_backfill(
        model_id=LOCAL_MODEL_ID,
        dry_run=True,
        papers=[_paper("arxiv:1", "Attention", "Transformers.")],
        already=set(),
        embed=embed,
    )
    assert report.would_write == 1
    assert report.written == 0
    assert "would_write: 1" in report.render()
    assert provider.documents == []


def test_embed_documents_is_the_real_provider_method():
    """Mock(spec=...) fails if the writer calls a method the class lacks."""
    assert callable(LocalBgeEmbeddingProvider.embed_documents)
    assert LocalBgeEmbeddingProvider.document_prefix == ""
    provider = Mock(spec=LocalBgeEmbeddingProvider)
    provider.model_id = LOCAL_MODEL_ID
    provider.dims = 2
    provider.batch_size = 4
    provider.embed_documents.return_value = [[0.1, 0.2]]
    saved = []
    written = embed_paper_vector_rows(
        [{"ref_id": "arxiv:1", "text": "Attention\n\nTransformers."}],
        provider,
        saved.append,
    )
    assert written == 1
    provider.embed_documents.assert_called_once_with(
        ["Attention\n\nTransformers."]
    )
    provider.embed_query.assert_not_called()
    assert saved[0][0]["vec"] == [0.1, 0.2]


def test_pairwise_spread_uses_population_std():
    # Pairs: 0, 1, 0. mean 1/3. population std sqrt(2)/3.
    stats = pairwise_spread([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert stats.papers == 3
    assert stats.pairs == 3
    assert stats.minimum == 0.0
    assert stats.maximum == 1.0
    assert abs(stats.mean - (1.0 / 3.0)) < 1e-9
    assert abs(stats.std - (2 ** 0.5) / 3.0) < 1e-9
    assert "std: 0.471405" in stats.render()
    empty = pairwise_spread([[1.0, 0.0]])
    assert empty.pairs == 0
    assert empty.std is None
    assert "std: n/a" in empty.render()


def test_spread_script_reports_the_same_std(tmp_path):
    path = tmp_path / "vectors.json"
    path.write_text(json.dumps([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    result = subprocess.run(
        [sys.executable, str(SPREAD_SCRIPT), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "papers: 3" in result.stdout
    assert "pairs: 3" in result.stdout
    assert "std: 0.471405" in result.stdout
    assert "def test_" not in SPREAD_SCRIPT.read_text()


def test_list_and_insert_name_a_missing_table(monkeypatch):
    response = Mock(status_code=201)
    client = Mock()
    client.base_url = "http://postgrest"
    client.headers = {"Prefer": "return=representation"}
    query = client.table.return_value
    query.select.return_value = query
    query.eq.return_value = query
    query.order.return_value = query
    query.limit.return_value = query
    query.offset.return_value = query
    query.execute.side_effect = [
        {"data": [{"ref_id": "arxiv:1"}], "error": None},
        {"data": [], "error": None},
    ]
    client.client.post.return_value = response
    monkeypatch.setattr("nlp_pillars.db.get_client", lambda: client)
    monkeypatch.setattr("nlp_pillars.db._PAGE_SIZE", 1)

    found = list_paper_vector_ref_ids(LOCAL_MODEL_ID, SOURCE_TITLE_ABSTRACT)
    assert found == ["arxiv:1"]
    assert query.eq.call_args_list[0].args == ("ref_kind", "paper")

    stored = insert_paper_vectors([{
        "ref_kind": "paper",
        "ref_id": "arxiv:2",
        "model_id": LOCAL_MODEL_ID,
        "source": SOURCE_TITLE_ABSTRACT,
        "dims": 2,
        "vec": [0.1, 0.2],
    }])
    assert stored == 1
    payload = client.client.post.call_args.kwargs["json"]
    assert payload[0]["vec"] == [0.1, 0.2]
    assert client.client.post.call_args.kwargs["headers"]["Prefer"] == "return=minimal"

    response.status_code = 404
    response.json.return_value = {
        "code": "PGRST205",
        "message": "Could not find the table paper_vectors in the schema cache",
    }
    try:
        insert_paper_vectors([{
            "ref_kind": "paper",
            "ref_id": "arxiv:2",
            "model_id": LOCAL_MODEL_ID,
            "source": SOURCE_TITLE_ABSTRACT,
            "dims": 2,
            "vec": [0.1, 0.2],
        }])
    except EmbeddingError as exc:
        assert "021_paper_vectors" in str(exc)
    else:
        raise AssertionError("missing table did not raise")


def test_cli_dry_run_uses_local_bge_and_does_not_load_it():
    papers = [_paper("arxiv:1", "Attention", "Transformers.")]
    with (
        patch("nlp_pillars.db.list_all_papers", return_value=papers),
        patch("nlp_pillars.db.list_paper_vector_ref_ids", return_value=[]) as listed,
        patch("nlp_pillars.works_store.load_external_works", return_value=([], 0)),
        patch("nlp_pillars.embeddings.get_embedding_provider") as build,
    ):
        result = CliRunner().invoke(app, ["embeddings", "papers", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "dry_run: true" in result.output
    assert f"model_id: {LOCAL_MODEL_ID}" in result.output
    assert "source: title_abstract" in result.output
    assert "would_write: 1" in result.output
    listed.assert_any_call(LOCAL_MODEL_ID, SOURCE_TITLE_ABSTRACT)
    build.assert_not_called()


def test_cli_second_run_does_not_load_the_provider_when_nothing_is_missing():
    papers = [_paper("arxiv:1", "Attention", "Transformers.")]
    with (
        patch("nlp_pillars.db.list_all_papers", return_value=papers),
        patch(
            "nlp_pillars.db.list_paper_vector_ref_ids",
            side_effect=[["arxiv:1"], []],
        ),
        patch("nlp_pillars.works_store.load_external_works", return_value=([], 0)),
        patch("nlp_pillars.embeddings.get_embedding_provider") as build,
    ):
        result = CliRunner().invoke(app, ["embeddings", "papers"])

    assert result.exit_code == 0, result.output
    assert "already: 1" in result.output
    assert "written: 0" in result.output
    build.assert_not_called()


def test_cli_writes_with_the_local_provider(monkeypatch):
    provider = FakeProvider()
    provider.model_id = LOCAL_MODEL_ID
    papers = [_paper("arxiv:1", "Attention", "Transformers.")]
    saved = []

    def _save(rows):
        saved.extend(rows)
        return len(rows)

    monkeypatch.setattr("nlp_pillars.db.list_all_papers", lambda: papers)
    monkeypatch.setattr("nlp_pillars.db.list_paper_vector_ref_ids", lambda *_a: [])
    monkeypatch.setattr("nlp_pillars.db.insert_paper_vectors", _save)
    monkeypatch.setattr(
        "nlp_pillars.works_store.load_external_works", lambda have: ([], 0)
    )
    with patch(
        "nlp_pillars.embeddings.get_embedding_provider",
        return_value=provider,
    ) as build:
        result = CliRunner().invoke(app, ["embeddings", "papers"])

    assert result.exit_code == 0, result.output
    assert "written: 1" in result.output
    build.assert_called_once_with(provider="local", model=LOCAL_MODEL_ID)
    assert provider.queries == []
    assert saved[0]["source"] == SOURCE_TITLE_ABSTRACT
    assert saved[0]["vec"][1] == 1.0


def test_cli_exits_nonzero_when_the_table_is_missing():
    with (
        patch(
            "nlp_pillars.db.list_all_papers",
            return_value=[_paper("arxiv:1", "Attention")],
        ),
        patch(
            "nlp_pillars.db.list_paper_vector_ref_ids",
            side_effect=EmbeddingError(
                "missing table. Apply docs/migrations/021_paper_vectors.sql by hand."
            ),
        ),
        patch("nlp_pillars.works_store.load_external_works", return_value=([], 0)),
    ):
        result = CliRunner().invoke(app, ["embeddings", "papers", "--dry-run"])

    assert result.exit_code == 1
    assert "021_paper_vectors" in result.output


def _external(work_id, title, abstract=None):
    return {"id": work_id, "title": title, "abstract": abstract}


def test_external_works_are_stored_as_external_paper_rows():
    provider = FakeProvider()
    saved = []

    def save(rows):
        saved.extend(rows)
        return len(rows)

    works = [
        _external("w-1", "Probing", "Control tasks."),
        _external("w-2", "Stub known only by an id"),
        _external("w-3", None, None),
    ]
    report = run_paper_vector_backfill(
        model_id=provider.model_id,
        dry_run=False,
        papers=works,
        already=set(),
        embed=lambda batch: embed_paper_vector_rows(
            batch, provider, save, ref_kind="external_paper"
        ),
        ref_kind="external_paper",
    )
    assert report.skipped_empty == 1
    assert report.written == 2
    assert {row["ref_kind"] for row in saved} == {"external_paper"}
    assert [row["ref_id"] for row in saved] == ["w-1", "w-2"]
    assert provider.documents == [["Probing\n\nControl tasks.", "Stub known only by an id"]]
    assert report.render().splitlines()[3] == "external: 3"


def test_cli_embeds_external_works_after_the_library_with_the_local_model(monkeypatch):
    provider = FakeProvider()
    provider.model_id = LOCAL_MODEL_ID
    saved = []

    def _save(rows):
        saved.extend(rows)
        return len(rows)

    lists = []

    def _list(model_id, source, ref_kind="paper"):
        lists.append(ref_kind)
        return ["arxiv:1"] if ref_kind == "paper" else ["w-old"]

    seen = {}

    def _external(have):
        seen["have"] = have
        # three external works in all, one of them already has its vector
        return [{"id": "w-1", "title": "Probing", "abstract": "Control tasks."}], 3

    monkeypatch.setattr(
        "nlp_pillars.db.list_all_papers",
        lambda: [_paper("arxiv:1", "Attention", "Transformers.")],
    )
    monkeypatch.setattr("nlp_pillars.db.list_paper_vector_ref_ids", _list)
    monkeypatch.setattr("nlp_pillars.db.insert_paper_vectors", _save)
    monkeypatch.setattr("nlp_pillars.works_store.load_external_works", _external)
    with patch(
        "nlp_pillars.embeddings.get_embedding_provider", return_value=provider
    ) as build:
        result = CliRunner().invoke(app, ["embeddings", "papers"])

    assert result.exit_code == 0, result.output
    assert lists == ["paper", "external_paper"]
    assert seen["have"] == {"w-old"}
    assert [(row["ref_kind"], row["ref_id"]) for row in saved] == [
        ("external_paper", "w-1")
    ]
    build.assert_called_once_with(provider="local", model=LOCAL_MODEL_ID)
    assert provider.queries == []
    assert "external: 3" in result.output
    assert "already: 2" in result.output
    assert "written: 1" in result.output


def test_cli_skips_external_works_and_says_so_before_migration_031():
    from nlp_pillars.works_store import WorksUnavailable

    with (
        patch(
            "nlp_pillars.db.list_all_papers",
            return_value=[_paper("arxiv:1", "Attention", "Transformers.")],
        ),
        patch("nlp_pillars.db.list_paper_vector_ref_ids", return_value=[]),
        patch(
            "nlp_pillars.works_store.load_external_works",
            side_effect=WorksUnavailable("no works. Apply docs/migrations/031_works_identity.sql by hand."),
        ),
    ):
        result = CliRunner().invoke(app, ["embeddings", "papers", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "library: 1" in result.output
    assert "external: skipped" in result.output
    assert "031_works_identity" in result.output
