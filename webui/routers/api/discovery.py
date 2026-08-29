"""
Discovery API endpoints for enhanced paper discovery with user selection.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from webui.services.run_service import (
    KIND_DISCOVER,
    KIND_PROCESS_SELECTED,
    RunAlreadyActiveError,
    dispatch_run,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pillars", tags=["discovery"])


class DiscoverRequest(BaseModel):
    """Request body for discovery endpoint."""
    priority_topics: List[str] = Field(default_factory=list, description="Optional topic hints")
    limit: int = Field(default=10, ge=1, le=50, description="Number of candidates to return")


class DiscoverResponse(BaseModel):
    """Acknowledgement of a discovery run, returned before the work happens.

    The candidates themselves arrive on the run: the browser polls
    GET /api/pipeline-runs/{run_id} for per-step progress and reads the finished run's
    `result` payload. There is no second request to make.
    """
    run_id: str
    pillar_id: str
    message: str


class PaperData(BaseModel):
    """Paper data for selection."""
    id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    url_pdf: Optional[str] = None
    abstract: Optional[str] = None
    citation_count: Optional[int] = None


class SelectRequest(BaseModel):
    """Request body for selection endpoint."""
    paper_ids: Optional[List[str]] = Field(None, description="Paper IDs to process (deprecated)")
    papers: Optional[List[PaperData]] = Field(None, description="Full paper data to process")


class SelectResponse(BaseModel):
    """Response from selection endpoint.

    `queued` is now the number of papers ACCEPTED for processing, reported before the
    work happens. It previously held a post-completion count, which made the field
    name a lie and gave a caller no way to tell "accepted" from "finished".
    """
    run_id: str
    queued: int
    processing: bool
    message: str


@router.post("/{pillar_id}/discover", response_model=DiscoverResponse, status_code=202)
async def discover_papers(
    pillar_id: str, request: DiscoverRequest, http_request: Request
):
    """
    Start discovery for a pillar and return a run id immediately.

    This endpoint used to answer synchronously, and the comment here used to explain
    why: the user needs the candidates in front of them to choose from, so there was
    nothing useful to do with an early return. That reasoning was about the *result*,
    and it was sound — but it left the page showing one static "Discovering papers…"
    line for the thirty seconds the work takes, with no steps, no counts, and no way
    to tell work from a hang. Worse, everything that can go wrong underneath — the
    query model falling back to focus areas, a rate-limited arXiv, an unreachable
    vector store — arrived as a short list that looked like an honest answer.

    So this is now a background run, deliberately reversing that decision. The
    candidates still reach the user in one round trip's worth of polling they were
    already doing, and the progress, the per-source counts, the generated queries, the
    failure reasons, reload-survival and a cancel button come with it, because the
    pipeline_runs machinery already provides all of them.

    Note the work still must not happen on the event loop — that part of the old
    comment is permanent. It is on an APScheduler worker thread now rather than in
    asyncio.to_thread, which is the same isolation plus supervision.
    """
    try:
        run_id = dispatch_run(
            http_request.app.state.scheduler,
            http_request.app.state.cancel_events,
            pillar_id=pillar_id,
            trigger_source="ui_discover",
            kind=KIND_DISCOVER,
            priority_topics=request.priority_topics,
            limit=request.limit,
        )
        return DiscoverResponse(
            run_id=run_id,
            pillar_id=pillar_id,
            message=f"Discovering papers for {pillar_id}",
        )

    except RunAlreadyActiveError as e:
        # Enforced by the partial unique index, not guessed at. Discovery and
        # processing share the one-active-run-per-pillar rule, which is the behaviour
        # we want: kicking off a new search while the last selection is still being
        # processed would compete for the same pillar.
        raise HTTPException(status_code=409, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Could not start discovery for pillar {pillar_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Could not start discovery: {str(e)}"
        )


@router.post("/{pillar_id}/select", response_model=SelectResponse, status_code=202)
async def select_papers(pillar_id: str, request: SelectRequest, http_request: Request):
    """
    Accept user-selected papers for processing and return a run id immediately.

    This used to call `orchestrator.process_selected_papers(...)` synchronously inside
    the handler — the full ingest/summarize/synthesize/quiz/vector pipeline, for every
    selected paper in sequence, before the response was written. With several papers
    selected that is many minutes of a frozen server.

    The client now polls GET /api/pipeline-runs/{run_id} for per-stage progress.

    Only the full-paper path is dispatched. `paper_ids` is deprecated and still
    resolved inline by the orchestrator, so it is rejected here rather than silently
    behaving differently from the documented path.
    """
    try:
        if not request.papers and not request.paper_ids:
            raise HTTPException(status_code=400, detail="No papers selected")

        if not request.papers:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The deprecated paper_ids field is no longer supported here; "
                    "send full paper objects in `papers`."
                ),
            )

        # Hand plain dicts across the thread boundary rather than PaperRef objects —
        # run_service rebuilds them inside the worker, so nothing model-shaped has to
        # survive the hop.
        papers_payload = [
            {
                "id": p.id,
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "url_pdf": p.url_pdf,
                "abstract": p.abstract,
                "citation_count": p.citation_count,
            }
            for p in request.papers
        ]

        run_id = dispatch_run(
            http_request.app.state.scheduler,
            http_request.app.state.cancel_events,
            pillar_id=pillar_id,
            trigger_source="ui_select",
            kind=KIND_PROCESS_SELECTED,
            papers=papers_payload,
        )

        return SelectResponse(
            run_id=run_id,
            queued=len(papers_payload),
            processing=True,
            message=f"Accepted {len(papers_payload)} paper(s) for processing",
        )

    except RunAlreadyActiveError as e:
        # Enforced by the database, not guessed at — a check-then-insert would race.
        raise HTTPException(status_code=409, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Selection failed for pillar {pillar_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Selection failed: {str(e)}")


@router.get("/{pillar_id}/citations/{paper_id}")
async def get_paper_citations(pillar_id: str, paper_id: str):
    """
    Get citations for a specific paper.
    """
    try:
        from nlp_pillars import db

        # pillar_id is now a string slug - we can validate it exists if needed
        # but for citations, we just need the paper_id

        # Get citations
        citations = db.get_citations_for_paper(paper_id)

        return {
            "paper_id": paper_id,
            "citations": [
                {
                    "cited_paper_id": c.cited_paper_id,
                    "direction": c.citation_direction,
                    "is_influential": c.is_influential,
                    "source": c.source
                }
                for c in citations
            ],
            "total": len(citations)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get citations for {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get citations: {str(e)}")
