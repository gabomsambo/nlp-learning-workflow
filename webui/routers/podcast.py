"""
Podcast page router for web UI.
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from nlp_pillars.db import (
    get_pillars_or_empty, get_all_papers, get_podcast_scripts,
    PaperLookupError, PodcastScriptLookupError,
)
from nlp_pillars.podcast_options import CUSTOM_VALUE, OPTION_SPECS

from webui.listen_view import (
    build_episode, catalog_json, group_episodes, listen_href, view_is_studio,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/podcast")


@router.get("/", response_class=HTMLResponse)
async def podcast_home(
    request: Request,
    studio: str = "",
    paper: str = "",
    script: str = "",
    run: str = "",
) -> HTMLResponse:
    """Render Listen: the episode library, or the studio when asked.

    A database that cannot be read is reported, not rendered as an empty page.
    get_all_papers() and get_podcast_scripts() used to return [] on failure, so
    an unreachable database produced an empty paper dropdown and the words "No
    podcast scripts generated yet" with no error anywhere — the page looked like
    a working account with nothing in it.

    ``?paper=`` opens the studio with that paper selected (the paper page's
    Make episode link). Nothing is generated from the query string.
    """
    # Pillars are only a filter here, so the degrading variant is the right one;
    # it is explicit about swallowing the failure at the call site.
    pillars = get_pillars_or_empty(limit=100)

    load_errors = []

    try:
        papers = get_all_papers(limit=100)
    except PaperLookupError as e:
        logger.error(f"Podcast page could not load papers: {e}")
        papers = []
        load_errors.append("The paper list could not be loaded, so no paper can be selected.")

    try:
        scripts = get_podcast_scripts(limit=20)
    except PodcastScriptLookupError as e:
        logger.error(f"Podcast page could not load scripts: {e}")
        scripts = None  # Distinct from []: unknown, not empty.
        load_errors.append("Existing podcast scripts could not be loaded.")

    paper_titles = {p.id: p.title or "" for p in papers}
    pillar_names = {p.id: p.name for p in pillars}
    episodes = []
    if scripts:
        for item in scripts:
            row = build_episode(item, paper_titles, pillar_names)
            if row:
                episodes.append(row)
    known_ids = {p.id for p in papers}
    selected_paper = paper or ""
    studio_open = view_is_studio({
        "studio": studio,
        "paper": paper,
        "script": script,
        "run": run,
    })

    return request.app.state.templates.TemplateResponse(
        request,
        "podcast.html",
        {
            "title": "Listen",
            "pillars": pillars,
            "papers": papers,
            "scripts": scripts,
            "episode_groups": group_episodes(episodes),
            "load_errors": load_errors,
            # The generation controls are rendered from the registry, so adding
            # a fifth option needs no template change either.
            "option_specs": OPTION_SPECS,
            "custom_value": CUSTOM_VALUE,
            "studio": studio_open,
            "selected_paper": selected_paper,
            "selected_script": script or "",
            "missing_paper": selected_paper if selected_paper and selected_paper not in known_ids else "",
            "episodes_href": listen_href(script=script or ""),
            "studio_href": listen_href(studio=True, paper=paper or "", script=script or ""),
            "catalog_json": catalog_json(papers, pillars),
        }
    )
