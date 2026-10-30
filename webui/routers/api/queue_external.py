"""Find papers: queue a Semantic Scholar paper, and score the queue's fit.

Both routes are free: no model call, and the only network call is one paced
Semantic Scholar lookup. Fit comes from the verdicts ``map build`` stored; a
paper without one is shown as unscored, never hidden.
"""

import logging
from typing import Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from nlp_pillars import map_view
from nlp_pillars.db import pillar_exists
from nlp_pillars.map_store import MapStoreError
from nlp_pillars.queue_external import (
    InvalidS2IdError,
    NoOpenPdfError,
    S2PaperNotFoundError,
    queue_external_paper,
)
from nlp_pillars.tools.semantic_scholar_tool import SemanticScholarError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pillars", tags=["queue"])


class ExternalQueueRequest(BaseModel):
    """A Semantic Scholar paper to put on the pillar's discovery queue."""

    s2_id: str = Field(..., min_length=1, max_length=200)


@router.post("/{pillar_id}/queue/external")
def queue_external(pillar_id: str, body: ExternalQueueRequest) -> Dict[str, str]:
    """Queue ``s2_id`` for the next run. 422 when it has no open PDF."""
    try:
        if not pillar_exists(pillar_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pillar '{pillar_id}' not found",
            )
        return queue_external_paper(pillar_id, body.s2_id)
    except HTTPException:
        raise
    except InvalidS2IdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except NoOpenPdfError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except S2PaperNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except SemanticScholarError as e:
        logger.warning("Semantic Scholar lookup failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Semantic Scholar could not be reached: {e}",
        ) from e
    except Exception as e:
        logger.exception("Could not queue external paper")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not queue the paper: {e}",
        ) from e


@router.get("/{pillar_id}/queue/fit")
def queue_fit(pillar_id: str) -> dict:
    """Each queued paper's stored fit verdict, keyed by ``paper_id``.

    ``status`` is ``not_built`` when no map exists for the pillar; ``fits`` is
    then empty and the page shows every candidate unscored.
    """
    try:
        if not pillar_exists(pillar_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pillar '{pillar_id}' not found",
            )
        row = map_view.read_snapshot(pillar_id)
        if row is None:
            return {"status": "not_built", "built_at": None, "fits": {}}
        overlay = map_view.queue_overlay(
            pillar_id, (row.get("payload") or {}).get("queue") or []
        )
    except HTTPException:
        raise
    except MapStoreError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("Could not read queue fit")
        raise HTTPException(status_code=503, detail=f"Could not read fit: {e}") from e

    if not overlay.get("available"):
        raise HTTPException(status_code=503, detail=overlay.get("reason") or "unavailable")
    fits = {
        item["paper_id"]: {
            "verdict": item["verdict"],
            "rank_fraction": item["rank_fraction"],
        }
        for item in overlay["items"]
        if item["paper_id"] and item["verdict"]
    }
    return {"status": "ready", "built_at": row.get("built_at"), "fits": fits}
