"""API router for Pillar CRUD operations.
Provides REST endpoints for managing learning pillars.
"""

import logging
from typing import Any, Dict, List, Literal

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from nlp_pillars import vectors
from nlp_pillars.db import (
    LibraryEnqueueError,
    PillarCreateError,
    PillarExistsError,
    create_pillar,
    delete_pillar,
    enqueue_library_paper,
    get_papers_by_ids,
    get_pillar_by_id,
    get_pillars_or_empty,
    pillar_exists,
    update_pillar,
)
from nlp_pillars.schemas import Pillar, PillarCreate, PillarUpdate

router = APIRouter(prefix="/api/pillars", tags=["pillars"])


class EnqueuePaperRequest(BaseModel):
    """A library paper to place on paper_queue for the next auto-run."""

    paper_id: str = Field(..., min_length=1)


@router.post("/", response_model=Pillar, status_code=status.HTTP_201_CREATED)
async def create_new_pillar(pillar_data: PillarCreate) -> Pillar:
    """Create a new pillar.
    
    - **name**: Human-readable pillar name (1-100 chars)
    - **goal**: Learning objective (10-500 chars)  
    - **focus_areas**: Array of topic areas (optional)
    - **papers_per_day**: Target papers per day 1-10 (default: 2)
    
    The `id` field is automatically generated as a URL-friendly slug from the name.
    """
    try:
        pillar = create_pillar(pillar_data)
        return pillar
    except PillarExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except PillarCreateError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get("/", response_model=List[Pillar])
async def list_pillars(limit: int = 50) -> List[Pillar]:
    """List all pillars.
    
    - **limit**: Maximum number of pillars to return (default: 50)
    
    Returns pillars ordered by creation date (oldest first).
    """
    try:
        pillars = get_pillars_or_empty(limit=limit)
        return pillars
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve pillars: {str(e)}"
        )


@router.get("/{pillar_id}", response_model=Pillar)
async def get_pillar(pillar_id: str) -> Pillar:
    """Get a specific pillar by its ID.
    
    - **pillar_id**: The URL-friendly slug of the pillar
    
    Returns the pillar details or 404 if not found.
    """
    try:
        pillar = get_pillar_by_id(pillar_id)
        if pillar is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pillar '{pillar_id}' not found"
            )
        return pillar
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve pillar: {str(e)}"
        )


@router.patch("/{pillar_id}", response_model=Pillar)
async def update_existing_pillar(pillar_id: str, pillar_data: PillarUpdate) -> Pillar:
    """Update an existing pillar.
    
    - **pillar_id**: The URL-friendly slug of the pillar to update
    - **name**: New name (optional) - will regenerate the slug if changed
    - **goal**: New goal (optional)
    - **focus_areas**: New focus areas (optional)
    - **papers_per_day**: New target papers per day (optional)
    
    Only provided fields will be updated. Returns 404 if pillar not found.
    """
    try:
        # Check if pillar exists
        if not pillar_exists(pillar_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pillar '{pillar_id}' not found"
            )

        pillar = update_pillar(pillar_id, pillar_data)
        if pillar is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update pillar"
            )
        return pillar
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update pillar: {str(e)}"
        )


@router.delete("/{pillar_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_existing_pillar(pillar_id: str) -> None:
    """Delete a pillar.
    
    - **pillar_id**: The URL-friendly slug of the pillar to delete
    
    Returns 204 No Content on success, 404 if not found.
    Cannot delete pillars that have associated papers.
    """
    try:
        # Check if pillar exists
        if not pillar_exists(pillar_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pillar '{pillar_id}' not found"
            )

        success = delete_pillar(pillar_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot delete pillar: has associated papers or other dependencies"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete pillar: {str(e)}"
        )


@router.get("/{pillar_id}/exists")
async def check_pillar_exists(pillar_id: str) -> JSONResponse:
    """Check if a pillar exists.
    
    - **pillar_id**: The URL-friendly slug to check
    
    Returns JSON with exists boolean.
    """
    try:
        exists = pillar_exists(pillar_id)
        return JSONResponse(content={"exists": exists})
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check pillar existence: {str(e)}"
        )


@router.get("/{pillar_id}/search")
async def search_pillar(
    pillar_id: str,
    q: str = Query(..., description="Search query string"),
    top_k: int = Query(
        10, ge=1, le=50, description="Maximum number of results to return"
    ),
    scope: Literal["pillar", "all"] = Query(
        "pillar", description="Search this pillar or the whole library"
    ),
) -> List[Dict[str, Any]]:
    """Search this pillar, or optionally the whole library, with hybrid retrieval.
    
    - **pillar_id**: The URL-friendly slug of the pillar to search in
    - **q**: Search query string
    - **top_k**: Maximum number of results (1-50, default 10)
    - **scope**: `pillar` (default) or `all`
    
    Returns a list of search results with paper metadata and scores.
    """
    try:
        pillar = get_pillar_by_id(pillar_id)
        if pillar is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pillar '{pillar_id}' not found"
            )

        vector_results = vectors.search_similar(
            pillar_id=pillar_id if scope == "pillar" else None,
            query_text=q,
            top_k=top_k,
        )

        if not vector_results:
            return []

        papers = get_papers_by_ids([result["paper_id"] for result in vector_results])
        papers_map = {paper.id: paper for paper in papers}

        enhanced_results = []
        for result in vector_results:
            paper_id = result["paper_id"]
            fused_score = result["score"]

            # Get paper details
            paper = papers_map.get(paper_id)
            if not paper:
                # Skip if we can't find paper details
                continue

            enhanced_results.append({
                "paper_id": paper_id,
                "title": paper.title,
                "authors": paper.authors or [],
                "year": paper.year,
                "venue": paper.venue,
                "score": round(fused_score, 4),
                # Compatibility fields retained for existing clients. The score
                # is now Qdrant's RRF fusion score, not a raw cosine value.
                "cosine_similarity": round(fused_score, 4),
                "keyword_overlap": 0.0,
                "source": "vector_search",
                "snippet": result.get("snippet", ""),
                "matched_chunk": {
                    "id": result.get("chunk_id"),
                    "index": result.get("chunk_index"),
                },
            })

        # Sort by final score and limit to top_k
        enhanced_results.sort(key=lambda x: x["score"], reverse=True)

        return enhanced_results[:top_k]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/{pillar_id}/queue")
async def enqueue_pillar_paper(pillar_id: str, body: EnqueuePaperRequest) -> Dict[str, str]:
    """Queue a paper already in this pillar for the next auto-run.

    Search hits are library papers. ``add_to_paper_queue`` skips those, so this
    route calls ``enqueue_library_paper``, which writes the same ``paper_queue``
    table the orchestrator pops.
    """
    try:
        if not pillar_exists(pillar_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pillar '{pillar_id}' not found",
            )
        return enqueue_library_paper(pillar_id, body.paper_id)
    except LibraryEnqueueError as e:
        code = {
            "not_found": status.HTTP_404_NOT_FOUND,
            "unresolvable": 422,
        }.get(e.code, status.HTTP_503_SERVICE_UNAVAILABLE)
        raise HTTPException(status_code=code, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not queue the paper: {e}",
        ) from e
