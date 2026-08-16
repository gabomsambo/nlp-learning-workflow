from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from nlp_pillars.db import get_pillars_or_empty
from ..services.postgrest_client import PostgrestClient


router = APIRouter(prefix="/lessons")


@router.get("/", response_class=HTMLResponse)
async def list_lessons(request: Request, pillar: Optional[str] = None) -> HTMLResponse:
    client = PostgrestClient()
    lessons = await client.list_lessons(pillar=pillar, limit=100)
    pillars = get_pillars_or_empty(limit=100)
    return request.app.state.templates.TemplateResponse(
        request, "lessons.html", {"lessons": lessons, "pillar": pillar, "pillars": pillars}
    )


