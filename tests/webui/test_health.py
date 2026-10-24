"""The liveness route exposes cached vector readiness without raising."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from webui.app import create_app


def test_health_returns_the_vector_report():
    report = {
        "ok": False,
        "embedder": {"state": "ok"},
        "targets": [
            {
                "target": "local",
                "state": "behind",
                "failure": "projection_lag",
                "pending_chunks": 2,
            }
        ],
    }
    with patch(
        "nlp_pillars.vector_health.cached_vector_health",
        return_value=report,
    ):
        response = TestClient(create_app()).get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": False, "vectors": report}


def test_health_stays_readable_when_collection_itself_raises():
    with patch(
        "nlp_pillars.vector_health.cached_vector_health",
        side_effect=RuntimeError("probe exploded"),
    ):
        response = TestClient(create_app()).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["vectors"]["failure"] == "vector_health_failed"
    assert body["vectors"]["detail"] == "probe exploded"
