"""
Podcast page router for web UI.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from nlp_pillars.db import get_pillars_or_empty, get_all_papers, get_podcast_scripts


router = APIRouter(prefix="/podcast")


@router.get("/", response_class=HTMLResponse)
async def podcast_home(request: Request) -> HTMLResponse:
    """Render the podcast generation page."""
    # Get pillars for filter dropdown
    pillars = get_pillars_or_empty(limit=100)

    # Get papers for selection dropdown
    papers = get_all_papers(limit=100)

    # Get existing podcast scripts
    scripts = get_podcast_scripts(limit=20)

    return request.app.state.templates.TemplateResponse(
        request,
        "podcast.html",
        {
            "title": "Podcast Script Generation",
            "pillars": pillars,
            "papers": papers,
            "scripts": scripts
        }
    )
