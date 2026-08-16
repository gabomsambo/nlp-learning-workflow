"""
Discovery API endpoints for enhanced paper discovery with user selection.
"""

import asyncio
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from nlp_pillars.orchestrator import Orchestrator
from nlp_pillars.schemas import DiscoveryCandidate, PaperRef
from webui.services.run_service import (
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
    """Response from discovery endpoint."""
    candidates: List[dict]
    total_found: int
    sources_used: List[str]


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


@router.post("/{pillar_id}/discover", response_model=DiscoverResponse)
async def discover_papers(pillar_id: str, request: DiscoverRequest):
    """
    Discover papers for a pillar from multiple sources.

    Returns ranked candidates for user selection without processing.
    """
    try:
        # Run discovery - pillar_id is now a string slug.
        #
        # run_discovery_with_selection is synchronous and takes tens of seconds
        # (an LLM call plus arXiv, SearXNG and Semantic Scholar searches). Called
        # directly from this async handler it blocked the event loop outright, so the
        # entire single-process server — /health included — was frozen for the
        # duration. asyncio.to_thread moves it off the loop; same precedent and same
        # reasoning as podcast_agent's _get_full_text, which measured 0 co-running
        # requests inline versus 13/13 through to_thread.
        #
        # This endpoint stays request/response rather than becoming a job: the user
        # needs these candidates in front of them to choose from, so there is nothing
        # useful to do with an early return.
        orchestrator = Orchestrator(enable_quiz=True)
        candidates = await asyncio.to_thread(
            orchestrator.run_discovery_with_selection,
            pillar_id,
            request.priority_topics,
            request.limit,
        )

        # Extract sources used
        sources_used = list(set(c.source for c in candidates))

        # Convert to dict for JSON response
        candidates_data = []
        for c in candidates:
            candidates_data.append({
                "paper": {
                    "id": c.paper.id,
                    "title": c.paper.title,
                    "authors": c.paper.authors[:3] if c.paper.authors else [],  # Limit authors
                    "year": c.paper.year,
                    "abstract": c.paper.abstract[:300] + "..." if c.paper.abstract and len(c.paper.abstract) > 300 else c.paper.abstract,
                    "url_pdf": c.paper.url_pdf,
                    "citation_count": c.paper.citation_count
                },
                "source": c.source,
                "relevance_score": round(c.relevance_score, 3),
                "citation_count": c.citation_count,
                "is_influential": c.is_influential
            })

        return DiscoverResponse(
            candidates=candidates_data,
            total_found=len(candidates),
            sources_used=sources_used
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Discovery failed for pillar {pillar_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")


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
