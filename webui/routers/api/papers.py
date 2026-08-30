"""API routes for individual paper operations."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from nlp_pillars.services.paper_metadata_refresh import (
    MetadataRefreshError,
    NoResolvableSourceError,
    PaperNotFoundError,
    refresh_paper_metadata,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/papers", tags=["papers"])


class FieldChangeResponse(BaseModel):
    field: str
    before: Any
    after: Any


class MetadataRefreshResponse(BaseModel):
    paper_id: str
    updated: bool
    message: str
    changed: List[FieldChangeResponse] = Field(default_factory=list)


@router.post(
    "/{paper_id}/refresh-metadata",
    response_model=MetadataRefreshResponse,
    status_code=status.HTTP_200_OK,
)
async def refresh_metadata(paper_id: str) -> MetadataRefreshResponse:
    """Re-resolve one paper's metadata from arXiv or Semantic Scholar."""
    try:
        result = await asyncio.to_thread(refresh_paper_metadata, paper_id)
    except PaperNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except NoResolvableSourceError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    except MetadataRefreshError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected metadata refresh failure for %s", paper_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Metadata refresh failed: {e}",
        ) from e

    return MetadataRefreshResponse(
        paper_id=result.paper_id,
        updated=result.updated,
        message=result.message,
        changed=[
            FieldChangeResponse(field=item.field, before=item.before, after=item.after)
            for item in result.changed
        ],
    )
