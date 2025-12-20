from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..services.postgrest_client import PostgrestClient


router = APIRouter(prefix="/analytics")


@router.get("/", response_class=HTMLResponse)
async def analytics_home(request: Request) -> HTMLResponse:
    client = PostgrestClient()
    counts = await client.counts_by_pillar()
    total = {
        "papers": sum(v["papers"] for v in counts.values()),
        "lessons": sum(v["lessons"] for v in counts.values()),
        "quizzes": sum(v["quizzes"] for v in counts.values()),
    }
    return request.app.state.templates.TemplateResponse("analytics.html", {"request": request, "counts": counts, "total": total})


