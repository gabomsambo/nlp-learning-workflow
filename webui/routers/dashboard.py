from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..services.postgrest_client import PostgrestClient


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    client = PostgrestClient()

    stats = await client.get_quick_stats()
    recent = await client.get_recent_activity()

    return request.app.state.templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "stats": stats, "recent": recent},
    )


