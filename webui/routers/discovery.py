"""
Discovery page router for web UI.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from nlp_pillars.db import get_pillars


router = APIRouter(prefix="/discovery")


@router.get("/", response_class=HTMLResponse)
async def discovery_home(request: Request) -> HTMLResponse:
    """Render the discovery page."""
    pillars = get_pillars(limit=100)
    return request.app.state.templates.TemplateResponse(
        "discovery.html",
        {"request": request, "title": "Paper Discovery", "pillars": pillars}
    )
