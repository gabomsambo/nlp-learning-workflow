"""
Discovery page router for web UI.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from nlp_pillars.db import get_pillars_or_empty


router = APIRouter(prefix="/discovery")

ADD_PAPER_MODES = ("find", "upload", "auto")


def add_papers_context(request: Request, default_mode: str = "find") -> dict:
    """Template context for Add papers.

    ``?pillar=`` pre-selects the pillar a quick link came from. ``?mode=`` opens
    Find, upload, or auto-run; ``/pipeline/`` defaults to auto-run so that URL
    still lands on the pipeline form.
    """
    pillars = get_pillars_or_empty(limit=100)
    selected = request.query_params.get("pillar", "")
    mode = request.query_params.get("mode", "")
    if mode not in ADD_PAPER_MODES:
        mode = default_mode if default_mode in ADD_PAPER_MODES else "find"
    return {
        "title": "Add papers",
        "pillars": pillars,
        "selected_pillar": selected,
        "mode": mode,
    }


@router.get("/", response_class=HTMLResponse)
async def discovery_home(request: Request) -> HTMLResponse:
    """Render Add papers, opening on Find unless ``?mode=`` says otherwise."""
    return request.app.state.templates.TemplateResponse(
        request, "discovery.html", add_papers_context(request, "find")
    )
