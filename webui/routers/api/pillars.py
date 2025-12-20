"""API router for Pillar CRUD operations.
Provides REST endpoints for managing learning pillars.
"""

import logging
import re
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

from nlp_pillars import vectors
from nlp_pillars.db import (
    PillarCreateError,
    PillarExistsError,
    create_pillar,
    delete_pillar,
    get_papers,
    get_pillar_by_id,
    get_pillars,
    pillar_exists,
    update_pillar,
)
from nlp_pillars.schemas import Pillar, PillarCreate, PillarUpdate

router = APIRouter(prefix="/api/pillars", tags=["pillars"])


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
        pillars = get_pillars(limit=limit)
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
    top_k: int = Query(10, ge=1, le=50, description="Maximum number of results to return")
) -> List[Dict[str, Any]]:
    """Search for papers within a specific pillar using vector similarity and re-ranking.
    
    Combines vector search with keyword overlap scoring based on pillar focus areas.
    
    - **pillar_id**: The URL-friendly slug of the pillar to search in
    - **q**: Search query string
    - **top_k**: Maximum number of results (1-50, default 10)
    
    Returns a list of search results with paper metadata and scores.
    """
    try:
        # Get pillar to access focus areas
        pillar = get_pillar_by_id(pillar_id)
        if pillar is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pillar '{pillar_id}' not found"
            )

        # Perform vector search using pillar_id directly (now accepts strings)
        vector_results = vectors.search_similar(
            pillar_id=pillar_id,
            query_text=q,
            top_k=top_k * 2  # Get more results to allow for re-ranking
        )

        if not vector_results:
            return []

        # Get paper details for all found papers
        papers_map = {}
        try:
            papers = get_papers(pillar_id, limit=1000)  # Get a large number to find all papers
            papers_map = {paper.id: paper for paper in papers}
        except Exception as e:
            # If we can't get paper details, continue with basic info
            logger.warning(f"Failed to get paper details for pillar {pillar_id}: {e}")
            pass

        # Apply re-ranking with focus area keyword overlap
        enhanced_results = []
        for result in vector_results:
            paper_id = result["paper_id"]
            cosine_score = result["score"]

            # Get paper details
            paper = papers_map.get(paper_id)
            if not paper:
                # Skip if we can't find paper details
                continue

            # Calculate keyword overlap with focus areas
            keyword_overlap_score = _calculate_keyword_overlap(q, pillar.focus_areas)

            # Apply re-ranking formula: 0.75 * cosine + 0.10 * keyword_overlap
            # (reserve 0.15 for potential BM25 component in future)
            final_score = 0.75 * cosine_score + 0.10 * keyword_overlap_score

            enhanced_results.append({
                "paper_id": paper_id,
                "title": paper.title,
                "authors": paper.authors or [],
                "year": paper.year,
                "venue": paper.venue,
                "score": round(final_score, 4),
                "cosine_similarity": round(cosine_score, 4),
                "keyword_overlap": round(keyword_overlap_score, 4),
                "source": "vector_search"
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


def _calculate_keyword_overlap(query: str, focus_areas: List[str]) -> float:
    """Calculate keyword overlap score between query and pillar focus areas.
    
    Args:
    ----
        query: Search query string
        focus_areas: List of focus area strings from the pillar
        
    Returns:
    -------
        Overlap score between 0.0 and 1.0

    """
    if not query or not focus_areas:
        return 0.0

    # Normalize and tokenize query
    query_tokens = set(re.findall(r'\w+', query.lower()))
    if not query_tokens:
        return 0.0

    # Normalize and tokenize focus areas
    focus_tokens = set()
    for area in focus_areas:
        focus_tokens.update(re.findall(r'\w+', area.lower()))

    if not focus_tokens:
        return 0.0

    # Calculate overlap
    overlap = len(query_tokens.intersection(focus_tokens))

    # Normalize by query length (Jaccard-like similarity)
    total_unique = len(query_tokens.union(focus_tokens))

    return overlap / total_unique if total_unique > 0 else 0.0
