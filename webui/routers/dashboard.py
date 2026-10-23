from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..services.postgrest_client import PostgrestClient
from ..services.today import load_today_view


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Today: what is due, what to continue, and what finished recently.

    The page only reads. Start review, Continue, and Play go to the screens
    that already own those actions.
    """
    view = await load_today_view(PostgrestClient())
    return request.app.state.templates.TemplateResponse(
        request,
        "dashboard.html",
        {"view": view},
    )
