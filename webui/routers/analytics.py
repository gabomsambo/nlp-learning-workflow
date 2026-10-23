from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..services.postgrest_client import PostgrestClient
from ..services.today import load_today_view


router = APIRouter(prefix="/analytics")


@router.get("/", response_class=HTMLResponse)
async def analytics_home(request: Request) -> HTMLResponse:
    """Per-pillar counts. The same figures sit on Today; this URL stays.

    A count that could not be read is an em dash. It is not shown as zero.
    """
    view = await load_today_view(PostgrestClient())
    return request.app.state.templates.TemplateResponse(
        request,
        "analytics.html",
        {"view": view},
    )
