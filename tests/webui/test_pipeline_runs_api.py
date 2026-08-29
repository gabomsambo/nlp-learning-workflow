"""Tests for the pipeline-run read path: PostgrestClient + the polling routes.

The behaviour under test is a distinction that used to be collapsed: "this run does
not exist" (404, browser forgets the run id) versus "the database could not be
reached" (503, browser retries). get_pipeline_run wrapped its whole body in
`except Exception: return None`, and the route turned None into 404, so one transient
PostgREST blip made the browser discard a run that was still executing.

Uses httpx.MockTransport through PostgrestClient's existing `client=` injection point
rather than adding respx — the two-file requirements/lock ritual costs more than it
saves for this. Basename is unique across tests/ because there are no __init__.py
files anywhere under it.
"""

import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.services.postgrest_client import PostgrestClient  # noqa: E402


def _client(handler):
    """A PostgrestClient whose transport is driven by `handler`."""
    return PostgrestClient(
        base_url="http://example",
        token=None,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _rows(*rows):
    return lambda request: httpx.Response(200, json=list(rows))


RUN_ROW = {
    "id": "run-1",
    "pillar_id": "neural-architectures-language",
    "status": "running",
    "papers_processed": 0,
    "papers_failed": 0,
    "error": None,
    "pipeline_run_stages": [
        {"seq": 3, "name": "enqueue", "status": "pending"},
        {"seq": 1, "name": "discovery", "status": "completed"},
        {"seq": 2, "name": "search", "status": "running"},
    ],
}


# ------------------------------------------------------------- get_pipeline_run


async def test_an_absent_run_is_none_not_an_error():
    """PostgREST answers a miss with 200 and an empty array, not a 404."""
    client = _client(_rows())
    assert await client.get_pipeline_run("nope") is None


async def test_a_present_run_comes_back_with_stages_sorted_by_seq():
    client = _client(_rows(RUN_ROW))
    run = await client.get_pipeline_run("run-1")

    assert run["id"] == "run-1"
    assert [s["seq"] for s in run["stages"]] == [1, 2, 3]
    # The embed key is renamed, so callers never see the PostgREST table name.
    assert "pipeline_run_stages" not in run


async def test_a_server_error_is_raised_not_swallowed():
    """Regression. Returning None here made the route answer 404, which tells the
    browser the run is gone: it wipes the stored id and stops polling a live run."""
    client = _client(lambda request: httpx.Response(500, json={"message": "boom"}))

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        await client.get_pipeline_run("run-1")
    assert excinfo.value.response.status_code == 500


async def test_a_transport_failure_is_raised_not_swallowed():
    def handler(request):
        raise httpx.ConnectError("no route to host", request=request)

    client = _client(handler)
    with pytest.raises(httpx.RequestError):
        await client.get_pipeline_run("run-1")


# ------------------------------------------------------ the degrading read paths


async def test_the_active_lookup_degrades_quietly():
    """"No active run" is the common answer and null is harmless, so this one may
    degrade — unlike get_pipeline_run, where null means "forget your run id"."""
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    assert await _client(handler).get_active_pipeline_run() is None


async def test_the_history_list_degrades_to_empty():
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    assert await _client(handler).list_pipeline_runs() == []


async def test_the_history_list_does_not_embed_stages_or_payloads():
    """It is a summary view.

    Embedding eleven stage rows per run would be wasteful; so would `result`, which on
    a discovery run is the entire candidate list. Both are fetched by whoever opens the
    single run.
    """
    captured = {}

    def handler(request):
        captured["select"] = request.url.params.get("select")
        return httpx.Response(200, json=[])

    await _client(handler).list_pipeline_runs(limit=5)
    assert "pipeline_run_stages" not in captured["select"]
    assert "result" not in captured["select"].split(",")
    assert "status" in captured["select"].split(",")


# ----------------------------------------------------------------- route mapping


@pytest.fixture
def app_client():
    """A TestClient over just the pipeline-runs router.

    Deliberately not the whole app: create_app()'s lifespan starts a scheduler and
    sweeps the database, none of which this router needs.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from webui.routers.api import pipeline_runs

    app = FastAPI()
    app.include_router(pipeline_runs.router)
    return TestClient(app, raise_server_exceptions=False)


def test_route_answers_404_for_a_run_that_does_not_exist(app_client, monkeypatch):
    async def absent(self, run_id):
        return None

    monkeypatch.setattr(PostgrestClient, "get_pipeline_run", absent)
    res = app_client.get("/api/pipeline-runs/nope")
    assert res.status_code == 404


def test_route_answers_503_when_the_database_is_unreachable(app_client, monkeypatch):
    """Not 404. The browser treats 404 as "this run is gone" and discards it; 503
    routes into the poller's retry path, which is the honest answer when we simply do
    not know the run's state."""
    async def broken(self, run_id):
        raise httpx.ConnectError("down", request=httpx.Request("GET", "http://x"))

    monkeypatch.setattr(PostgrestClient, "get_pipeline_run", broken)
    res = app_client.get("/api/pipeline-runs/run-1")
    assert res.status_code == 503
    assert "run itself is unaffected" in res.json()["detail"]


def test_active_route_is_reachable_and_not_captured_as_a_run_id(app_client, monkeypatch):
    """/active is registered before /{run_id}; if that ordering regresses, "active"
    is parsed as a run id and this route becomes unreachable."""
    async def active(self, pillar_id=None):
        return {"id": "run-1", "stages": []}

    monkeypatch.setattr(PostgrestClient, "get_active_pipeline_run", active)
    res = app_client.get("/api/pipeline-runs/active")
    assert res.status_code == 200
    assert res.json()["id"] == "run-1"
