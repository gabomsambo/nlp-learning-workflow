"""Projector: Postgres embeddings copied into Qdrant, per target.

Hermetic. The client is Mock(spec=QdrantClient). Nothing here dials a server
or downloads a model. The throwaway-container test lives in
tests/test_projector_integration.py and stays skipped unless
QDRANT_INTEGRATION=1.
"""

import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from typer.testing import CliRunner

from nlp_pillars.cli import app
from nlp_pillars.projector import (
    BM25_VECTOR,
    MIGRATION,
    POINT_NAMESPACE,
    READ_ALIAS,
    ChunkVector,
    ProjectionError,
    apply_projection,
    collection_name,
    commit_projection,
    point_id,
    project_paper,
)

CHUNK = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"
MODEL = "test-model"
DIMS = 4
PILLAR = "pillar-a"
PAPER = "paper-1"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIGRATION_SQL = open(os.path.join(REPO, MIGRATION), encoding="utf-8").read()
SCHEMA_SQL = open(os.path.join(REPO, "schema.sql"), encoding="utf-8").read()


def unexpected_response(status_code, content=b"boom"):
    return UnexpectedResponse(
        status_code=status_code,
        reason_phrase="Not Found" if status_code == 404 else "Bad Request",
        content=content,
        headers=None,
    )


def make_chunk(chunk_id=CHUNK, paper_id=PAPER, text="hello chunk"):
    return ChunkVector(
        chunk_id=chunk_id,
        paper_id=paper_id,
        pillar_id=PILLAR,
        chunk_index=0,
        text=text,
        embedding=[0.1, 0.2, 0.3, 0.4],
        dims=DIMS,
    )


def row_for(chunk: ChunkVector) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "pillar_id": chunk.pillar_id,
        "chunk_index": chunk.chunk_index,
        "chunk_text": chunk.text,
        "dims": chunk.dims,
        "embedding": list(chunk.embedding),
    }


def make_client(existing=False):
    """Mock(spec=QdrantClient). A bare Mock would answer search() and hide a
    call the installed client no longer has."""
    client = Mock(spec=QdrantClient)
    names = []
    if existing:
        found = Mock()
        found.name = collection_name(MODEL, DIMS)
        names.append(found)
    client.get_collections.return_value.collections = names
    info = client.get_collection.return_value
    info.payload_schema = {}
    info.config.params.sparse_vectors = {
        BM25_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
    }
    # A Mock here is iterable forever. The alias walk must see a real list.
    client.get_aliases.return_value.aliases = []
    client.scroll.return_value = ([], None)
    return client


@pytest.fixture(autouse=True)
def stub_bm25_passage():
    sparse = models.SparseVector(indices=[7, 11], values=[0.5, 1.5])
    with patch("nlp_pillars.projector.bm25_passage", return_value=sparse):
        yield


def _apply(client, chunks, **kwargs):
    params = dict(
        target="local",
        model_id=MODEL,
        dims=DIMS,
        chunks=chunks,
        indexed_ids=set(),
        stale_ids=set(),
        rebuild=False,
        dry_run=False,
        scope_complete=True,
        scroll_paper_ids=None,
    )
    params.update(kwargs)
    return apply_projection(client, **params)


@contextmanager
def patched_store(client, rows, state=None, stale=None):
    provider = Mock()
    provider.model_id = MODEL
    provider.dims = DIMS
    mark = Mock()
    forget = Mock()
    with patch("nlp_pillars.embeddings.active_model_id", return_value=MODEL), \
         patch("nlp_pillars.embeddings.get_embedding_provider", return_value=provider), \
         patch("nlp_pillars.db.list_chunks_with_embeddings", return_value=rows), \
         patch("nlp_pillars.db.list_projection_state", return_value=list(state or [])), \
         patch("nlp_pillars.db.list_projection_stale", return_value=list(stale or [])), \
         patch("nlp_pillars.db.mark_projection_state", mark), \
         patch("nlp_pillars.db.forget_projection_state", forget), \
         patch("nlp_pillars.projector.open_client", return_value=client):
        yield mark, forget


def test_qdrant_client_exposes_the_projection_surface():
    for name in (
        "query_points",
        "upsert",
        "delete",
        "scroll",
        "update_collection_aliases",
        "create_payload_index",
        "get_aliases",
        "update_collection",
    ):
        assert hasattr(QdrantClient, name), name
    assert not hasattr(QdrantClient, "search")
    assert (
        "sparse_vectors_config" in QdrantClient.create_collection.__annotations__
    )
    assert "prefetch" in QdrantClient.query_points.__annotations__


def test_point_id_is_uuid5_of_the_chunk_and_stable():
    assert point_id(CHUNK) == str(uuid.uuid5(POINT_NAMESPACE, CHUNK))
    assert point_id(CHUNK) == point_id(CHUNK)
    assert point_id(CHUNK) != point_id(OTHER)


def test_collection_name_sanitizes_model_ids():
    name = collection_name("BAAI/bge-base-en-v1.5@abc", 768)
    assert name.startswith("chunks__")
    assert name.endswith("__768")
    assert "/" not in name
    assert "@" not in name


def test_upsert_writes_text_and_both_payload_indexes():
    client = make_client()
    chunk = make_chunk()
    result = _apply(client, [chunk])

    assert result.upserted_ids == [CHUNK]
    assert result.error is None
    points = client.upsert.call_args.kwargs["points"]
    assert points[0].id == point_id(CHUNK)
    assert points[0].vector[""] == chunk.embedding
    assert points[0].vector[BM25_VECTOR].indices == [7, 11]
    assert points[0].payload["text"] == "hello chunk"
    assert points[0].payload["pillar_id"] == PILLAR
    assert points[0].payload["paper_id"] == PAPER
    assert points[0].payload["chunk_id"] == CHUNK
    assert points[0].payload["chunk_index"] == 0
    assert client.upsert.call_args.kwargs["collection_name"] == collection_name(
        MODEL, DIMS
    )

    fields = [
        call.kwargs["field_name"]
        for call in client.create_payload_index.call_args_list
    ]
    assert fields == ["pillar_id", "paper_id"]
    for call in client.create_payload_index.call_args_list:
        assert call.kwargs["field_schema"] == models.PayloadSchemaType.KEYWORD

    operations = client.update_collection_aliases.call_args.kwargs[
        "change_aliases_operations"
    ]
    op = operations[0]
    assert op.create_alias.alias_name == READ_ALIAS
    assert op.create_alias.collection_name == collection_name(MODEL, DIMS)

    create = client.create_collection.call_args.kwargs
    sparse_config = create["sparse_vectors_config"][BM25_VECTOR]
    assert sparse_config.modifier == models.Modifier.IDF


def test_existing_dense_collection_gets_bm25_config():
    client = make_client(existing=True)
    client.get_collection.return_value.config.params.sparse_vectors = {}

    _apply(client, [make_chunk()], indexed_ids={CHUNK})

    config = client.update_collection.call_args.kwargs["sparse_vectors_config"]
    assert config[BM25_VECTOR].modifier == models.Modifier.IDF


def test_second_sync_is_a_no_op():
    client = make_client(existing=True)
    client.get_collection.return_value.payload_schema = {
        "pillar_id": object(),
        "paper_id": object(),
    }
    chunk = make_chunk()
    record = Mock()
    record.id = point_id(CHUNK)
    record.payload = {"chunk_id": CHUNK, "paper_id": PAPER}
    client.scroll.return_value = ([record], None)

    result = _apply(client, [chunk], indexed_ids={CHUNK})

    assert result.pending == 0
    assert result.upserted_ids == []
    assert result.deleted_ids == []
    client.upsert.assert_not_called()
    client.delete.assert_not_called()


def test_rebuild_upserts_chunks_already_indexed():
    client = make_client(existing=True)
    result = _apply(client, [make_chunk()], indexed_ids={CHUNK}, rebuild=True)
    assert result.upserted_ids == [CHUNK]
    client.upsert.assert_called_once()


def test_deleted_chunk_point_is_removed():
    client = make_client(existing=True)
    result = _apply(client, [make_chunk()], indexed_ids={CHUNK, OTHER})
    assert OTHER in result.deleted_ids
    deleted = client.delete.call_args.kwargs["points_selector"]
    assert point_id(OTHER) in deleted


def test_paper_scope_does_not_scroll_the_whole_collection():
    client = make_client(existing=True)
    orphan = Mock()
    orphan.id = "orphan-point"
    orphan.payload = {"chunk_id": OTHER, "paper_id": PAPER}
    client.scroll.return_value = ([orphan], None)

    result = _apply(
        client,
        [make_chunk()],
        indexed_ids={CHUNK},
        scope_complete=False,
        scroll_paper_ids={PAPER},
    )

    filt = client.scroll.call_args.kwargs["scroll_filter"]
    assert filt.must[0].key == "paper_id"
    assert filt.must[0].match.value == PAPER
    assert OTHER in result.deleted_ids
    assert "orphan-point" in client.delete.call_args.kwargs["points_selector"]


def test_dry_run_probes_and_does_not_write():
    client = make_client()
    result = _apply(client, [make_chunk()], dry_run=True)
    assert result.dry_run is True
    assert result.pending == 1
    assert "would_upsert: 1" in result.render()
    client.create_collection.assert_not_called()
    client.upsert.assert_not_called()
    client.delete.assert_not_called()
    client.get_collections.assert_called()


def test_dimension_mismatch_does_not_upsert():
    client = make_client()
    result = _apply(client, [make_chunk()], dims=8)
    assert result.error
    assert "8-d" in result.error
    client.upsert.assert_not_called()


def test_suspended_cloud_writes_nothing_and_the_replay_does():
    client = make_client()
    client.get_collections.side_effect = unexpected_response(404, b"404 page not found")
    chunk = make_chunk()

    blocked = _apply(client, [chunk], target="cloud")
    assert blocked.unhealthy is True
    assert "suspended" in (blocked.error or "")
    client.upsert.assert_not_called()
    mark = Mock()
    forget = Mock()
    commit_projection(blocked, mark, forget)
    mark.assert_not_called()
    forget.assert_not_called()

    client.get_collections.side_effect = None
    client.get_collections.return_value.collections = []
    replayed = _apply(client, [chunk], target="cloud")
    assert replayed.unhealthy is False
    assert replayed.error is None
    assert replayed.upserted_ids == [CHUNK]
    commit_projection(replayed, mark, forget)
    mark.assert_called_once_with("cloud", MODEL, [CHUNK])


def test_local_outage_still_projects_cloud_then_raises():
    chunk = make_chunk()
    local = make_client()
    local.get_collections.side_effect = unexpected_response(404, b"404 page not found")
    cloud = make_client()

    with patched_store(cloud, [row_for(chunk)]) as (mark, _forget), \
         patch("nlp_pillars.vectors.get_client", return_value=local), \
         patch.dict(os.environ, {"QDRANT_CLOUD_URL": "http://cloud.example"}):
        with pytest.raises(ProjectionError, match="suspended"):
            project_paper(PILLAR, PAPER)

    local.upsert.assert_not_called()
    cloud.upsert.assert_called_once()
    targets = [call.args[0] for call in mark.call_args_list]
    assert targets == ["cloud"]


def test_migration_has_no_foreign_key_and_reloads_postgrest():
    table = re.search(
        r"CREATE TABLE IF NOT EXISTS vector_index_state \(.*?\);",
        MIGRATION_SQL,
        re.S,
    )
    assert table, "migration 019 no longer creates vector_index_state"
    assert "REFERENCES" not in table.group(0)
    assert "PRIMARY KEY (target, chunk_id, model_id)" in table.group(0)
    assert "CHECK (target IN ('local', 'cloud'))" in table.group(0)
    assert "CREATE OR REPLACE FUNCTION chunks_with_embeddings" in MIGRATION_SQL
    assert "CREATE OR REPLACE FUNCTION projection_stale_chunks" in MIGRATION_SQL
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON vector_index_state TO web_anon;" in MIGRATION_SQL
    assert "GRANT EXECUTE ON FUNCTION chunks_with_embeddings" in MIGRATION_SQL
    assert "GRANT EXECUTE ON FUNCTION projection_stale_chunks" in MIGRATION_SQL
    assert MIGRATION_SQL.rstrip().endswith("NOTIFY pgrst, 'reload schema';")


def test_schema_sql_carries_the_projection_objects():
    table = re.search(
        r"CREATE TABLE vector_index_state \(.*?\);",
        SCHEMA_SQL,
        re.S,
    )
    assert table
    assert "REFERENCES" not in table.group(0)
    assert "chunks_with_embeddings" in SCHEMA_SQL
    assert "projection_stale_chunks" in SCHEMA_SQL
    assert "GRANT EXECUTE ON FUNCTION chunks_with_embeddings" in SCHEMA_SQL
    assert "GRANT EXECUTE ON FUNCTION projection_stale_chunks" in SCHEMA_SQL


def test_state_primary_key_rejects_a_duplicate_projection():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE vector_index_state ("
        " target TEXT NOT NULL,"
        " chunk_id TEXT NOT NULL,"
        " model_id TEXT NOT NULL,"
        " indexed_at TEXT NOT NULL,"
        " PRIMARY KEY (target, chunk_id, model_id))"
    )
    conn.execute(
        "INSERT INTO vector_index_state VALUES ('local', ?, 'model', 't')",
        (CHUNK,),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO vector_index_state VALUES ('local', ?, 'model', 't2')",
            (CHUNK,),
        )
    conn.execute(
        "INSERT INTO vector_index_state VALUES ('cloud', ?, 'model', 't')",
        (CHUNK,),
    )
    assert conn.execute("SELECT count(*) FROM vector_index_state").fetchone()[0] == 2


def test_cli_dry_run_does_not_upsert():
    client = make_client()
    chunk = make_chunk()
    with patched_store(client, [row_for(chunk)]):
        result = CliRunner().invoke(app, ["vectors", "sync", "--target", "local", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry_run: true" in result.output
    client.upsert.assert_not_called()


def test_cli_rejects_an_unknown_target():
    result = CliRunner().invoke(app, ["vectors", "sync", "--target", "both"])
    assert result.exit_code == 2
    assert "local, cloud, or all" in result.output


def test_cli_names_the_migration_when_the_read_fails():
    client = make_client()
    boom = ProjectionError(
        "list chunks: function chunks_with_embeddings does not exist. "
        f"Apply {MIGRATION} by hand."
    )
    with patched_store(client, []), \
         patch("nlp_pillars.db.list_chunks_with_embeddings", side_effect=boom):
        result = CliRunner().invoke(app, ["vectors", "sync", "--target", "local", "--dry-run"])
    assert result.exit_code == 1
    assert "019_vector_index_state.sql" in result.output


def test_cli_cloud_outage_exits_zero_and_local_outage_exits_one():
    client = make_client()
    client.get_collections.side_effect = unexpected_response(404, b"404 page not found")
    with patched_store(client, []):
        cloud = CliRunner().invoke(app, ["vectors", "sync", "--target", "cloud"])
        local = CliRunner().invoke(app, ["vectors", "sync", "--target", "local"])
    assert cloud.exit_code == 0, cloud.output
    assert "unhealthy: true" in cloud.output
    assert local.exit_code == 1, local.output
    assert "unhealthy: true" in local.output
