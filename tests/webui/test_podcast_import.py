"""Import practice drills without a model call, and keep fact-check off them.

Basename is unique across tests/ because there are no __init__.py files there.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from nlp_pillars.db import PodcastScriptSaveError
from nlp_pillars.drills import build_drill_row, parse_drill_sheet, sheet_to_script

SHEET = """# Day 1 drill: ML fundamentals

Intro: Day one. ML fundamentals. One question.

## Q1. What is leakage?

**Answer.** A feature that contains information from after the label.

**Key points.** Time order matters. Split before you fit.
"""


@pytest.fixture
def api_client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from webui.routers.api import podcast as podcast_api

    app = FastAPI()
    app.state.scheduler = MagicMock()
    app.state.cancel_events = {}
    app.include_router(podcast_api.router)

    from webui.services.spend_guard import require_spend_room

    # Spend has its own tests; the drill/import paths make no model call.
    app.dependency_overrides[require_spend_room] = lambda: None
    return TestClient(app, raise_server_exceptions=False)


def _drill_row():
    return build_drill_row(
        title="Day 1 drill: ML fundamentals",
        pillar_id="neural-architectures-language",
        script=sheet_to_script(parse_drill_sheet(SHEET)),
    )


def test_import_stores_a_drill_row_without_a_paper(api_client):
    saved = {}

    def add(row):
        saved["row"] = row
        return "11111111-2222-4333-8444-555555555555"

    with (
        patch("webui.routers.api.podcast.pillar_exists", return_value=True),
        patch("webui.routers.api.podcast.add_podcast_script", side_effect=add),
    ):
        response = api_client.post(
            "/api/podcast/scripts/import",
            json={
                "title": "  Day 1 drill  ",
                "pillar_id": "neural-architectures-language",
                "script": "[HOST]: Question one.\n[PAUSE 5s]\n[HOST]: Answer.\n",
            },
        )

    assert response.status_code == 201
    assert response.json()["script_id"] == "11111111-2222-4333-8444-555555555555"
    row = saved["row"]
    assert row.title == "Day 1 drill"
    assert row.paper_id is None
    assert row.source_material.kind == "drill"
    assert "[PAUSE 5s]" in row.script
    assert row.word_count > 0


def test_import_accepts_an_optional_paper_id(api_client):
    saved = {}

    def add(row):
        saved["row"] = row
        return str(uuid.uuid4())

    with (
        patch("webui.routers.api.podcast.pillar_exists", return_value=True),
        patch("webui.routers.api.podcast.add_podcast_script", side_effect=add),
    ):
        response = api_client.post(
            "/api/podcast/scripts/import",
            json={
                "title": "T",
                "pillar_id": "neural-architectures-language",
                "paper_id": "arxiv:1706.03762",
                "script": "[HOST]: hello",
            },
        )

    assert response.status_code == 201
    assert saved["row"].paper_id == "arxiv:1706.03762"


def test_import_rejects_blank_text(api_client):
    response = api_client.post(
        "/api/podcast/scripts/import",
        json={
            "title": "   ",
            "pillar_id": "neural-architectures-language",
            "script": "[HOST]: hello",
        },
    )
    assert response.status_code == 422
    assert "must not be blank" in response.json()["detail"]


def test_import_rejects_an_unknown_pillar(api_client):
    with patch("webui.routers.api.podcast.pillar_exists", return_value=False):
        response = api_client.post(
            "/api/podcast/scripts/import",
            json={
                "title": "T",
                "pillar_id": "no-such-pillar",
                "script": "[HOST]: hello",
            },
        )
    assert response.status_code == 404
    assert "no-such-pillar" in response.json()["detail"]


def test_import_names_the_nullable_paper_migration_when_the_column_is_old(api_client):
    with (
        patch("webui.routers.api.podcast.pillar_exists", return_value=True),
        patch(
            "webui.routers.api.podcast.add_podcast_script",
            side_effect=PodcastScriptSaveError(
                'null value in column "paper_id" violates not-null constraint'
            ),
        ),
    ):
        response = api_client.post(
            "/api/podcast/scripts/import",
            json={
                "title": "T",
                "pillar_id": "neural-architectures-language",
                "script": "[HOST]: hello",
            },
        )
    assert response.status_code == 503
    assert "038_podcast_drill_scripts.sql" in response.json()["detail"]


def test_manual_factcheck_refuses_a_drill(api_client):
    script_id = str(uuid.uuid4())
    with patch(
        "webui.routers.api.podcast.get_podcast_script_by_id",
        return_value=_drill_row(),
    ):
        response = api_client.post(f"/api/podcast/{script_id}/factcheck")
    assert response.status_code == 422
    assert "practice drill" in response.json()["detail"]


def test_automatic_factcheck_skips_a_drill_before_the_spend_gate():
    from webui.services import podcast_factcheck_service

    with (
        patch.object(podcast_factcheck_service, "get_settings") as settings,
        patch.object(
            podcast_factcheck_service.db,
            "get_podcast_script_by_id",
            return_value=_drill_row(),
        ),
        patch.object(
            podcast_factcheck_service.db, "get_active_factcheck_run"
        ) as active,
        patch.object(podcast_factcheck_service.spend, "current_budget") as budget,
        patch("webui.services.run_service.dispatch_run") as dispatch,
    ):
        settings.return_value.factcheck_auto_dispatch = True
        result = podcast_factcheck_service.maybe_dispatch_after_script(
            MagicMock(),
            {},
            script_id="drill-1",
            paper_id=None,
            pillar_id="neural-architectures-language",
        )

    assert result is None
    active.assert_not_called()
    budget.assert_not_called()
    dispatch.assert_not_called()
