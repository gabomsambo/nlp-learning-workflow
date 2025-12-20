"""
Discovery API endpoints for enhanced paper discovery with user selection.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from nlp_pillars.orchestrator import Orchestrator
from nlp_pillars.schemas import DiscoveryCandidate, PaperRef


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
    """Response from selection endpoint."""
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
        # Run discovery - pillar_id is now a string slug
        orchestrator = Orchestrator(enable_quiz=True)
        candidates = orchestrator.run_discovery_with_selection(
            pillar_id=pillar_id,
            priority_topics=request.priority_topics,
            limit=request.limit
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


@router.post("/{pillar_id}/select", response_model=SelectResponse)
async def select_papers(pillar_id: str, request: SelectRequest):
    """
    Process user-selected papers.

    Queues selected papers for processing through the pipeline.
    """
    try:
        if not request.papers and not request.paper_ids:
            raise HTTPException(status_code=400, detail="No papers selected")

        # Process selected papers
        orchestrator = Orchestrator(enable_quiz=True)

        # Prefer full paper data over just IDs
        if request.papers:
            # Convert PaperData to PaperRef objects
            paper_refs = [
                PaperRef(
                    id=p.id,
                    title=p.title,
                    authors=p.authors,
                    year=p.year,
                    url_pdf=p.url_pdf,
                    abstract=p.abstract,
                    citation_count=p.citation_count
                )
                for p in request.papers
            ]
            result = orchestrator.process_selected_papers(
                pillar_id=pillar_id,
                papers=paper_refs
            )
        else:
            # Fallback to paper_ids (deprecated)
            result = orchestrator.process_selected_papers(
                pillar_id=pillar_id,
                paper_ids=request.paper_ids
            )

        return SelectResponse(
            queued=len(result.papers_processed),
            processing=result.success,
            message=f"Processed {len(result.papers_processed)} papers with {len(result.errors)} errors"
        )

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
