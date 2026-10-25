"""GET /api/graph. The router is mounted alone so the app lifespan stays down."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nlp_pillars.graph_store import GraphStoreError
from webui.routers.api.graph import router


def _client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_graph_route_returns_nodes_and_edges(monkeypatch):
    seen = {}

    def fake(*, pillar, depth, center, user_id="default_user"):
        seen["pillar"] = pillar
        seen["depth"] = depth
        seen["center"] = center
        seen["user_id"] = user_id
        return {
            "nodes": [{"id": "arxiv:1", "kind": "paper", "label": "Attention"}],
            "edges": [],
            "truncated": False,
            "found": True,
            "depth": depth,
            "pillar": pillar,
            "center": center,
        }

    monkeypatch.setattr("webui.routers.api.graph.graph_view", fake)
    response = _client().get(
        "/api/graph",
        params={"pillar": "neural-architectures-language", "depth": 2, "center": "arxiv:1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["nodes"][0]["id"] == "arxiv:1"
    assert body["edges"] == []
    assert seen == {
        "pillar": "neural-architectures-language",
        "depth": 2,
        "center": "arxiv:1",
        "user_id": "default_user",
    }


def test_graph_route_unknown_center_is_404(monkeypatch):
    monkeypatch.setattr(
        "webui.routers.api.graph.graph_view",
        lambda **_kwargs: {"nodes": [], "edges": [], "found": False},
    )
    response = _client().get("/api/graph", params={"center": "missing"})
    assert response.status_code == 404


def test_graph_route_missing_table_is_503_and_names_the_migration(monkeypatch):
    monkeypatch.setattr(
        "webui.routers.api.graph.graph_view",
        lambda **_kwargs: (_ for _ in ()).throw(
            GraphStoreError(
                "Knowledge graph is not available. "
                "Apply docs/migrations/020_knowledge_graph.sql by hand."
            )
        ),
    )
    response = _client().get("/api/graph")
    assert response.status_code == 503
    assert "020_knowledge_graph.sql" in response.json()["detail"]


def test_graph_route_rejects_an_unbounded_depth():
    response = _client().get("/api/graph", params={"depth": 9})
    assert response.status_code == 422
