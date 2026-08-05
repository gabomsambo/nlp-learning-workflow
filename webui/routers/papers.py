from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from nlp_pillars.db import get_pillars
from ..services.postgrest_client import PostgrestClient


router = APIRouter(prefix="/papers")


@router.get("/", response_class=HTMLResponse)
async def list_papers(request: Request, pillar: Optional[str] = None) -> HTMLResponse:
    client = PostgrestClient()
    data = await client.list_papers(pillar=pillar, limit=100)
    pillars = get_pillars(limit=100)
    return request.app.state.templates.TemplateResponse(
        request, "papers.html", {"papers": data, "pillar": pillar, "pillars": pillars}
    )


@router.get("/details/{paper_id}")
async def get_paper_details(paper_id: str) -> JSONResponse:
    """Get detailed paper information including notes, lessons, and quiz cards."""
    client = PostgrestClient()
    details = await client.get_paper_details(paper_id)
    
    if not details:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    return JSONResponse(content=details)


