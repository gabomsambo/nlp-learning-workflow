"""A pre-hybrid collection is replaced by a new generation and the alias swapped.

Qdrant 1.19.2 refuses ``update_collection(sparse_vectors_config=...)`` on an
existing collection (``400 Not existing vector name error: bm25``), so
``ensure_index`` builds ``<name>__v2``, copies the points, and swaps the alias.
These tests drive that logic against an in-memory fake: no Docker, no server,
no model download. The same scenario runs against a real Qdrant in
tests/test_projector_integration.py when QDRANT_INTEGRATION=1.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from qdrant_client import models

from nlp_pillars.projector import (
    BM25_VECTOR,
    READ_ALIAS,
    ProjectionError,
    ensure_index,
)
from nlp_pillars.vector_health import collection_version

MODEL = "test-model"
DIMS = 4
BASE = "chunks__test-model__4"
SPARSE = models.SparseVector(indices=[1, 2], values=[0.5, 0.5])


class FakeQdrant:
    """Just enough of QdrantClient for ensure_index, with failure injection."""

    def __init__(self):
        self.collections = {}
        self.aliases = {}
        self.alias_history = []  # alias target after every change; None = missing
        self.fail_upsert_after = None
        self.fail_alias_swap = False
        self.update_collection_calls = 0

    # reads
    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=n) for n in self.collections]
        )

    def get_aliases(self):
        return SimpleNamespace(
            aliases=[
                SimpleNamespace(alias_name=a, collection_name=c)
                for a, c in self.aliases.items()
            ]
        )

    def get_collection(self, name):
        col = self.collections[name]
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(sparse_vectors=col["sparse"])
            ),
            payload_schema={f: object() for f in col["indexes"]},
        )

    def count(self, collection_name, exact=True):
        return SimpleNamespace(count=len(self.collections[collection_name]["points"]))

    def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        points = list(self.collections[collection_name]["points"].values())
        start = int(offset or 0)
        page = points[start:start + limit]
        nxt = start + limit if start + limit < len(points) else None
        return page, nxt

    # writes
    def create_collection(self, collection_name, vectors_config, sparse_vectors_config=None):
        if collection_name in self.collections:
            raise RuntimeError("already exists")
        self.collections[collection_name] = {
            "sparse": dict(sparse_vectors_config or {}) or None,
            "indexes": set(),
            "points": {},
        }

    def delete_collection(self, name):
        self.collections.pop(name, None)

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.collections[collection_name]["indexes"].add(field_name)

    def upsert(self, collection_name, points):
        store = self.collections[collection_name]["points"]
        if (
            self.fail_upsert_after is not None
            and len(store) + len(points) > self.fail_upsert_after
        ):
            raise RuntimeError("upsert exploded")
        for point in points:
            store[point.id] = SimpleNamespace(
                id=point.id, vector=point.vector, payload=point.payload
            )

    def update_collection(self, **kwargs):
        self.update_collection_calls += 1
        raise RuntimeError("400 Not existing vector name error: bm25")

    def update_collection_aliases(self, change_aliases_operations):
        # One call is one atomic change: validate everything, then apply.
        if self.fail_alias_swap:
            raise RuntimeError("alias swap refused")
        aliases = dict(self.aliases)
        for op in change_aliases_operations:
            if isinstance(op, models.DeleteAliasOperation):
                aliases.pop(op.delete_alias.alias_name)
            else:
                create = op.create_alias
                if create.alias_name in aliases:
                    raise RuntimeError("alias already exists")
                aliases[create.alias_name] = create.collection_name
        self.aliases = aliases
        self.alias_history.append(aliases.get(READ_ALIAS))

    # helpers
    def seed_pre_hybrid(self, name=BASE, n=3, alias=True):
        self.collections[name] = {
            "sparse": None,
            "indexes": {"pillar_id", "paper_id"},
            "points": {
                f"id{i}": SimpleNamespace(
                    id=f"id{i}",
                    vector=[0.1 * i] * DIMS,  # unnamed dense vector, as in the live one
                    payload={"chunk_id": f"c{i}", "text": f"text {i}", "paper_id": "p"},
                )
                for i in range(n)
            },
        }
        if alias:
            self.aliases[READ_ALIAS] = name


@pytest.fixture(autouse=True)
def _no_bm25_model():
    with patch("nlp_pillars.projector.bm25_passage", return_value=SPARSE):
        yield


def test_pre_hybrid_collection_is_replaced_and_alias_swapped():
    client = FakeQdrant()
    client.seed_pre_hybrid(n=3)

    name = ensure_index(client, MODEL, DIMS)

    assert name == BASE + "__v2"
    assert client.aliases[READ_ALIAS] == name
    assert client.update_collection_calls == 0
    new = client.collections[name]
    assert BM25_VECTOR in new["sparse"]
    assert new["sparse"][BM25_VECTOR].modifier == models.Modifier.IDF
    assert new["indexes"] == {"pillar_id", "paper_id"}
    # points preserved, dense untouched, bm25 added, payload identical
    old = client.collections[BASE]["points"]
    assert set(new["points"]) == set(old)
    for pid, point in new["points"].items():
        assert point.vector[""] == old[pid].vector
        assert point.vector[BM25_VECTOR] == SPARSE
        assert point.payload == old[pid].payload


def test_old_collection_is_kept_and_the_alias_is_never_missing():
    client = FakeQdrant()
    client.seed_pre_hybrid()

    ensure_index(client, MODEL, DIMS)

    assert BASE in client.collections
    # exactly one alias change, and it went straight from old to new
    assert client.alias_history == [BASE + "__v2"]
    assert None not in client.alias_history


def test_copy_failure_leaves_alias_and_old_collection_and_removes_half_built():
    client = FakeQdrant()
    client.seed_pre_hybrid(n=3)
    client.fail_upsert_after = 1

    with pytest.raises(ProjectionError) as exc:
        ensure_index(client, MODEL, DIMS)

    assert "still serves nlp_chunks" in str(exc.value)
    assert client.aliases[READ_ALIAS] == BASE
    assert client.alias_history == []
    assert set(client.collections) == {BASE}
    assert len(client.collections[BASE]["points"]) == 3


def test_alias_swap_failure_leaves_alias_on_old_and_removes_new():
    client = FakeQdrant()
    client.seed_pre_hybrid()
    client.fail_alias_swap = True

    with pytest.raises(ProjectionError):
        ensure_index(client, MODEL, DIMS)

    assert client.aliases[READ_ALIAS] == BASE
    assert set(client.collections) == {BASE}


def test_count_mismatch_refuses_to_swap():
    client = FakeQdrant()
    client.seed_pre_hybrid(n=3)
    real_count = client.count

    def lying_count(collection_name, exact=True):
        got = real_count(collection_name, exact)
        if collection_name.endswith("__v2"):
            return SimpleNamespace(count=got.count - 1)
        return got

    client.count = lying_count
    with pytest.raises(ProjectionError, match="holds"):
        ensure_index(client, MODEL, DIMS)
    assert client.aliases[READ_ALIAS] == BASE
    assert set(client.collections) == {BASE}


def test_point_without_text_aborts_rather_than_dropping_it():
    client = FakeQdrant()
    client.seed_pre_hybrid(n=2)
    del client.collections[BASE]["points"]["id1"].payload["text"]

    with pytest.raises(ProjectionError, match="no chunk text"):
        ensure_index(client, MODEL, DIMS)
    assert client.aliases[READ_ALIAS] == BASE
    assert set(client.collections) == {BASE}


def test_migrated_collection_is_not_migrated_again():
    client = FakeQdrant()
    client.seed_pre_hybrid()
    first = ensure_index(client, MODEL, DIMS)

    second = ensure_index(client, MODEL, DIMS)

    assert second == first
    assert set(client.collections) == {BASE, BASE + "__v2"}
    assert client.alias_history == [BASE + "__v2"]


def test_interrupted_earlier_attempt_is_started_over():
    client = FakeQdrant()
    client.seed_pre_hybrid(n=2)
    # a v2 left behind by a crashed run: never served, partly filled
    client.collections[BASE + "__v2"] = {
        "sparse": {BM25_VECTOR: object()},
        "indexes": set(),
        "points": {"stale": SimpleNamespace(id="stale", vector={}, payload={})},
    }

    name = ensure_index(client, MODEL, DIMS)

    assert name == BASE + "__v2"
    assert set(client.collections[name]["points"]) == {"id0", "id1"}


def test_next_generation_after_v2_is_v3():
    client = FakeQdrant()
    client.seed_pre_hybrid(name=BASE + "__v2", n=2)  # an old v2 without bm25

    assert ensure_index(client, MODEL, DIMS) == BASE + "__v3"


def test_hybrid_collection_is_left_alone():
    client = FakeQdrant()
    client.seed_pre_hybrid()
    client.collections[BASE]["sparse"] = {BM25_VECTOR: object()}

    assert ensure_index(client, MODEL, DIMS) == BASE
    assert set(client.collections) == {BASE}
    assert client.alias_history == []


def test_fresh_server_creates_the_hybrid_collection_directly():
    client = FakeQdrant()

    assert ensure_index(client, MODEL, DIMS) == BASE
    assert BM25_VECTOR in client.collections[BASE]["sparse"]
    assert client.aliases[READ_ALIAS] == BASE


def test_collection_without_alias_gets_one_after_migration():
    client = FakeQdrant()
    client.seed_pre_hybrid(alias=False)

    name = ensure_index(client, MODEL, DIMS)

    assert client.aliases[READ_ALIAS] == name == BASE + "__v2"


@pytest.mark.parametrize(
    "name,version",
    [
        (BASE, 1),
        (BASE + "__v2", 2),
        (BASE + "__v10", 10),
        (BASE + "__vx", None),
        (BASE + "__v", None),
        ("chunks__other__4", None),
        (BASE + "__v2__v3", None),
    ],
)
def test_collection_version(name, version):
    assert collection_version(name, BASE) == version
