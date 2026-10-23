"""Activity: run history and workstation status."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from nlp_pillars.db import get_pillars_or_empty

router = APIRouter(prefix="/activity")


@router.get("/", response_class=HTMLResponse)
async def activity_home(request: Request) -> HTMLResponse:
    pillars = [
        {"id": pillar.id, "name": pillar.name}
        for pillar in get_pillars_or_empty(limit=100)
    ]
    return request.app.state.templates.TemplateResponse(
        request,
        "activity.html",
        {"pillars": pillars},
    )
