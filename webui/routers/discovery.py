"""
Discovery page router for web UI.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from nlp_pillars.db import get_pillars_or_empty


router = APIRouter(prefix="/discovery")


@router.get("/", response_class=HTMLResponse)
async def discovery_home(request: Request) -> HTMLResponse:
    """Render the discovery page.

    Honours ``?pillar=<slug>`` so the link from a pillar's Quick Actions arrives with
    that pillar already chosen. The link carried the parameter before this; nothing
    read it, so the user had to re-pick the pillar they had just come from.
    """
    pillars = get_pillars_or_empty(limit=100)
    selected = request.query_params.get("pillar", "")
    return request.app.state.templates.TemplateResponse(
        request,
        "discovery.html",
        {"title": "Paper Discovery", "pillars": pillars, "selected_pillar": selected}
    )
