"""Read-only endpoints the browser polls to follow a pipeline run.

Every handler here is ``async def`` and reads through the async PostgrestClient. That
is not incidental: these are hit roughly once a second per open page, so they must
never touch a worker thread. Writes happen on the pipeline's own thread through the
synchronous ``nlp_pillars.db`` client instead.
"""

import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, status

from webui.services.postgrest_client import PostgrestClient
from webui.services.run_service import request_cancel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline-runs", tags=["pipeline-runs"])


@router.get("/active")
async def get_active_run(
    pillar_id: Optional[str] = Query(None, description="Restrict to one pillar"),
) -> Optional[Dict[str, Any]]:
    """Return the in-flight run, or null.

    Registered before /{run_id} on purpose — otherwise "active" is captured as a run
    id and this route is unreachable.

    Lets a page that has just been reloaded, with no stored run id, still find the run
    it should be showing. For a single-user tool "the active run" is well defined.
    """
    client = PostgrestClient()
    return await client.get_active_pipeline_run(pillar_id)


@router.get("")
async def list_runs(
    pillar_id: Optional[str] = Query(None, description="Restrict to one pillar"),
    limit: int = Query(20, ge=1, le=100),
) -> List[Dict[str, Any]]:
    """Recent runs, newest first, without stage detail. The run history view."""
    client = PostgrestClient()
    return await client.list_pipeline_runs(pillar_id=pillar_id, limit=limit)


@router.get("/{run_id}")
async def get_run(run_id: str) -> Dict[str, Any]:
    """Return one run with its stages in `seq` order.

    Raises:
        HTTPException: 404 if the run does not exist. The browser treats that as
            "forget the stored run id" rather than retrying forever, so it matters
            that a missing run is a 404 and not an empty body.
        HTTPException: 503 if the database could not be reached. This must NOT be a
            404 for exactly the same reason: 404 tells the browser the run is gone and
            it discards the id, so answering a transient outage with 404 permanently
            detaches the UI from a run that is still executing. 503 routes into the
            poller's retry-with-backoff path instead, which is the honest behaviour —
            we do not know the run's state, so we say so and try again.

    """
    client = PostgrestClient()
    try:
        run = await client.get_pipeline_run(run_id)
    except httpx.HTTPError as e:
        logger.warning(f"Could not read pipeline run {run_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Could not reach the database to read this run. The run itself is "
                "unaffected; retrying."
            ),
        ) from e
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pipeline run with id {run_id}",
        )
    return run


@router.post("/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request) -> Dict[str, Any]:
    """Ask a run to stop at its next stage boundary.

    Cooperative, and deliberately honest about it: nothing can interrupt synchronous
    Python mid-call, so the run finishes the stage it is in and stops before the next
    one. `accepted: false` means this process is not running that job — it may have
    finished already, or it may belong to the scheduler container, which has its own
    registry.
    """
    accepted = request_cancel(request.app.state.cancel_events, run_id)
    return {
        "run_id": run_id,
        "accepted": accepted,
        "message": (
            "Cancellation requested; the run will stop after its current stage."
            if accepted
            else "No cancellable run with that id in this process."
        ),
    }
