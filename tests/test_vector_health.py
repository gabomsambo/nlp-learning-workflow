"""Hermetic vector-health checks. No server or model is contacted."""

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from qdrant_client import QdrantClient

from nlp_pillars.projector import collection_name
from nlp_pillars.vector_health import (
    CLOUD_SUSPENDED_OR_DELETED,
    COLLECTION_OR_ALIAS_MISSING,
    EMBEDDER_FAILED,
    INDEX_UNREACHABLE,
    PROJECTION_LAG,
    collect_vector_health,
)

MODEL = "test/model@revision"
DIMS = 3
COLLECTION = collection_name(MODEL, DIMS)
ENV = {
    "QDRANT_URL": "http://local.test",
    "QDRANT_CLOUD_URL": "https://cloud.test",
    "QDRANT_CLOUD_API_KEY": "cloud-secret",
}


class FakeProvider:
    model_id = MODEL
    dims = DIMS

    def embed_query(self, text):
        assert text
        return [0.1, 0.2, 0.3]


def qdrant_client(*, collection=True, alias=True):
    client = Mock(spec=QdrantClient)
    client.get_collections.return_value.collections = (
        [SimpleNamespace(name=COLLECTION)] if collection else []
    )
    client.get_aliases.return_value.aliases = (
        [
            SimpleNamespace(
                alias_name="nlp_chunks",
                collection_name=COLLECTION,
            )
        ]
        if alias
        else []
    )
    return client


def live_response():
    return httpx.Response(
        200,
        json={"title": "qdrant - vector search engine", "version": "1.19.2"},
    )


def collect(**overrides):
    args = {
        "provider_factory": FakeProvider,
        "embedded_ids_loader": lambda model: ["a", "b"],
        "state_loader": lambda target, model: ["a", "b"],
        "http_get": lambda *args, **kwargs: live_response(),
        "client_factory": lambda url, key: qdrant_client(),
        "environ": ENV,
    }
    args.update(overrides)
    return collect_vector_health(**args)


def by_target(payload, target):
    return next(item for item in payload["targets"] if item["target"] == target)


def test_qdrant_client_exposes_health_interfaces():
    assert hasattr(QdrantClient, "get_collections")
    assert hasattr(QdrantClient, "get_aliases")


def test_healthy_embedder_and_both_targets():
    payload = collect()

    assert payload["ok"] is True
    assert payload["embedder"]["state"] == "ok"
    assert by_target(payload, "local")["pending_chunks"] == 0
    assert by_target(payload, "cloud")["state"] == "ok"
    assert by_target(payload, "cloud")["fatal"] is False


def test_embedder_load_or_encode_failure_has_its_own_shape():
    class BrokenProvider:
        model_id = MODEL
        dims = DIMS

        def embed_query(self, text):
            raise RuntimeError("ONNX weights cannot load")

    payload = collect(provider_factory=BrokenProvider)

    assert payload["ok"] is False
    assert payload["embedder"]["failure"] == EMBEDDER_FAILED
    assert payload["embedder"]["detail"] == (
        "embedder failed: ONNX weights cannot load"
    )
    assert by_target(payload, "local")["state"] == "ok"
    assert by_target(payload, "cloud")["state"] == "ok"


def test_paid_provider_is_not_called_by_periodic_health():
    provider = FakeProvider()
    provider.model_id = "text-embedding-3-small"
    provider.dims = 3
    provider.embed_query = Mock(side_effect=AssertionError("paid call"))

    payload = collect(provider_factory=lambda: provider)

    assert payload["embedder"]["state"] == "ok"
    assert payload["embedder"]["detail"] == "configured; paid encode probe skipped"
    provider.embed_query.assert_not_called()


def test_connection_refusal_is_index_unreachable():
    def refused(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    payload = collect(http_get=refused, environ={"QDRANT_URL": ENV["QDRANT_URL"]})
    local = by_target(payload, "local")

    assert payload["ok"] is False
    assert local["failure"] == INDEX_UNREACHABLE
    assert "connection refused" in local["detail"]


def test_cloud_plain_text_404_is_suspended_or_deleted_and_nonfatal():
    def root(url, **kwargs):
        if "cloud" in url:
            return httpx.Response(404, text="404 page not found\n")
        return live_response()

    payload = collect(http_get=root)
    cloud = by_target(payload, "cloud")

    assert payload["ok"] is True
    assert cloud["failure"] == CLOUD_SUSPENDED_OR_DELETED
    assert cloud["state"] == "down"
    assert cloud["fatal"] is False
    assert "suspended or deleted" in cloud["detail"]


def test_local_plain_text_404_is_unreachable_not_a_cloud_lifecycle_guess():
    payload = collect(
        http_get=lambda *args, **kwargs: httpx.Response(
            404, text="404 page not found\n"
        ),
        environ={"QDRANT_URL": ENV["QDRANT_URL"]},
    )

    local = by_target(payload, "local")
    assert local["failure"] == INDEX_UNREACHABLE
    assert "HTTP 404" in local["detail"]


def test_missing_collection_and_alias_have_the_same_failure_shape():
    missing_collection = collect(
        client_factory=lambda url, key: qdrant_client(collection=False)
    )
    row = by_target(missing_collection, "local")
    assert row["failure"] == COLLECTION_OR_ALIAS_MISSING
    assert "collection" in row["detail"]

    missing_alias = collect(
        client_factory=lambda url, key: qdrant_client(alias=False)
    )
    row = by_target(missing_alias, "local")
    assert row["failure"] == COLLECTION_OR_ALIAS_MISSING
    assert "alias nlp_chunks is absent" in row["detail"]


def test_projection_lag_uses_id_difference_so_stale_state_cannot_hide_it():
    payload = collect(
        state_loader=lambda target, model: ["a", "stale"],
        environ={"QDRANT_URL": ENV["QDRANT_URL"]},
    )
    local = by_target(payload, "local")

    assert payload["ok"] is False
    assert local["failure"] == PROJECTION_LAG
    assert local["embedded_chunks"] == 2
    assert local["projected_chunks"] == 1
    assert local["pending_chunks"] == 1


def test_unconfigured_cloud_is_omitted():
    payload = collect(environ={"QDRANT_URL": ENV["QDRANT_URL"]})
    assert [row["target"] for row in payload["targets"]] == ["local"]
