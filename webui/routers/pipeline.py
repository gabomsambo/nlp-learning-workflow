from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse

from nlp_pillars.pillar_utils import validate_pillar_id
from webui.routers.discovery import add_papers_context
from webui.services.spend_guard import require_spend_room
from webui.services.run_service import (
    KIND_RUN_DAILY,
    RunAlreadyActiveError,
    dispatch_run,
)

router = APIRouter(prefix="/pipeline")


@router.get("/", response_class=HTMLResponse)
async def pipeline_home(request: Request) -> HTMLResponse:
    """Auto-run mode of Add papers.

    ``/pipeline/?pillar=&run=`` keeps working: same page as ``/discovery/``,
    opened on the pipeline form unless ``?mode=`` asks for another one.
    ``?run=`` is read by the page script, so it is left on this URL.
    """
    return request.app.state.templates.TemplateResponse(
        request, "discovery.html", add_papers_context(request, "auto")
    )


@router.post("/run", dependencies=[Depends(require_spend_room)])
async def trigger_run(request: Request, pillar: str, papers: int = 1):
    """Start a daily pipeline run and return its id immediately.

    This used to shell out to `python -m nlp_pillars.cli run` and await
    `proc.communicate()`, so the request stayed open for the whole pipeline — minutes,
    with the browser showing "Running..." and no way to tell work from a hang. The
    subprocess also re-imported torch and layoutparser on every run and could not
    report structured progress without scraping its own stdout.

    Now the work goes to a background thread and the client polls
    GET /api/pipeline-runs/{run_id}.
    """
    if not validate_pillar_id(pillar):
        return JSONResponse(
            {"success": False, "error": f"Invalid pillar: {pillar}"}, status_code=400
        )

    try:
        run_id = dispatch_run(
            request.app.state.scheduler,
            request.app.state.cancel_events,
            pillar_id=pillar,
            trigger_source="ui_pipeline",
            kind=KIND_RUN_DAILY,
            papers_limit=papers,
        )
    except RunAlreadyActiveError as e:
        # The database refused a second active run for this pillar (partial unique
        # index -> 409/23505). Surfaced as 409 so the UI can point at the run in
        # flight instead of silently starting a duplicate.
        return JSONResponse({"success": False, "error": str(e)}, status_code=409)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    return JSONResponse(
        {"success": True, "run_id": run_id, "status": "pending"}, status_code=202
    )
