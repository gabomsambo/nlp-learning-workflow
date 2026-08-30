"""Tests for paper metadata refresh API route."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from nlp_pillars.services.paper_metadata_refresh import (
    FieldChange,
    MetadataRefreshResult,
    NoResolvableSourceError,
    PaperNotFoundError,
)
from webui.app import create_app


def test_refresh_metadata_route_success():
    app = create_app()
    client = TestClient(app)
    result = MetadataRefreshResult(
        paper_id="2403.05525",
        changed=[FieldChange(field="authors", before=["A"], after=["A", "B"])],
        message="Updated metadata:\n- authors: A → A, B",
    )

    with patch(
        "webui.routers.api.papers.refresh_paper_metadata",
        return_value=result,
    ):
        response = client.post("/api/papers/2403.05525/refresh-metadata")

    assert response.status_code == 200
    body = response.json()
    assert body["updated"] is True
    assert body["paper_id"] == "2403.05525"
    assert body["changed"][0]["field"] == "authors"


def test_refresh_metadata_route_not_found():
    app = create_app()
    client = TestClient(app)

    with patch(
        "webui.routers.api.papers.refresh_paper_metadata",
        side_effect=PaperNotFoundError("Paper 'missing' was not found."),
    ):
        response = client.post("/api/papers/missing/refresh-metadata")

    assert response.status_code == 404


def test_refresh_metadata_route_no_source():
    app = create_app()
    client = TestClient(app)

    with patch(
        "webui.routers.api.papers.refresh_paper_metadata",
        side_effect=NoResolvableSourceError(
            "This paper has no resolvable metadata source."
        ),
    ):
        response = client.post("/api/papers/custom/refresh-metadata")

    assert response.status_code == 422
    assert "no resolvable metadata source" in response.json()["detail"].lower()
