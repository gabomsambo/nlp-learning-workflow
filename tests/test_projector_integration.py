"""Project into a throwaway Qdrant. Skipped in CI.

Enable with QDRANT_INTEGRATION=1. The container name and host port are not
the compose stack's (`nlp_qdrant` / 6333). The fixture removes the container
when the test finishes.
"""

import json
import os
import subprocess
import time
import urllib.request

import pytest
from qdrant_client import QdrantClient, models

from nlp_pillars.projector import (
    READ_ALIAS,
    ChunkVector,
    apply_projection,
    collection_name,
    point_id,
)

pytestmark = pytest.mark.skipif(
    os.getenv("QDRANT_INTEGRATION") != "1",
    reason="set QDRANT_INTEGRATION=1 to project into a throwaway Qdrant",
)

CONTAINER = "fm-nlp-lk-projector-qdrant"
HOST_PORT = 16333
IMAGE = "qdrant/qdrant:v1.19.2"
CHUNK = "33333333-3333-4333-8333-333333333333"
GONE = "44444444-4444-4444-8444-444444444444"
MODEL = "integ-model"
DIMS = 4


def _run(args):
    subprocess.run(args, check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def qdrant_url():
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, text=True)
    _run([
        "docker", "run", "-d",
        "--name", CONTAINER,
        "-p", f"127.0.0.1:{HOST_PORT}:6333",
        IMAGE,
    ])
    url = f"http://127.0.0.1:{HOST_PORT}"
    deadline = time.time() + 40
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/", timeout=2) as resp:
                banner = json.load(resp)
            if str(banner.get("version", "")).startswith("1.19"):
                break
            last = str(banner)
        except Exception as exc:
            last = str(exc)
        time.sleep(0.5)
    else:
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
        pytest.fail(f"throwaway qdrant did not answer 1.19: {last}")
    try:
        yield url
    finally:
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, text=True)


def _chunk(chunk_id, text):
    return ChunkVector(
        chunk_id=chunk_id,
        paper_id="paper-integ",
        pillar_id="pillar-integ",
        chunk_index=0,
        text=text,
        embedding=[0.4, 0.3, 0.2, 0.1],
        dims=DIMS,
    )


def _project(client, chunks, indexed):
    return apply_projection(
        client,
        target="local",
        model_id=MODEL,
        dims=DIMS,
        chunks=chunks,
        indexed_ids=indexed,
        stale_ids=set(),
        rebuild=False,
        dry_run=False,
        scope_complete=True,
        scroll_paper_ids=None,
    )


def _hits(client):
    response = client.query_points(
        collection_name=READ_ALIAS,
        query=[0.4, 0.3, 0.2, 0.1],
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="pillar_id",
                    match=models.MatchValue(value="pillar-integ"),
                )
            ]
        ),
        limit=10,
        with_payload=True,
    )
    return list(response.points)


def test_projects_searches_and_deletes_stale_points(qdrant_url):
    client = QdrantClient(url=qdrant_url, timeout=10)
    kept = _chunk(CHUNK, "kept chunk text")
    dropped = _chunk(GONE, "dropped chunk text")

    first = _project(client, [kept, dropped], set())
    assert set(first.upserted_ids) == {CHUNK, GONE}
    assert first.error is None

    info = client.get_collection(collection_name(MODEL, DIMS))
    schema = info.payload_schema
    names = set(schema.keys()) if hasattr(schema, "keys") else set(schema)
    assert "pillar_id" in names
    assert "paper_id" in names

    hits = _hits(client)
    by_id = {str(hit.id): hit for hit in hits}
    assert point_id(CHUNK) in by_id
    assert by_id[point_id(CHUNK)].payload["text"] == "kept chunk text"

    second = _project(client, [kept, dropped], {CHUNK, GONE})
    assert second.upserted_ids == []
    assert second.deleted_ids == []
    assert len(_hits(client)) == 2

    third = _project(client, [kept], {CHUNK, GONE})
    assert GONE in third.deleted_ids
    left = _hits(client)
    assert [str(hit.id) for hit in left] == [point_id(CHUNK)]
    assert left[0].payload["text"] == "kept chunk text"
