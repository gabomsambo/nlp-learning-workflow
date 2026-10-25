"""Read-only neighbourhood of the learner knowledge graph.

The page that draws this is owned by the UI redesign. This route only returns
JSON, and it does not build the graph: ``python -m nlp_pillars.cli graph build``
does that.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from nlp_pillars.graph import MAX_DEPTH
from nlp_pillars.graph_store import GraphStoreError, graph_view

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("")
def get_graph(
    pillar: Optional[str] = Query(default=None),
    depth: int = Query(default=1, ge=0, le=MAX_DEPTH),
    center: Optional[str] = Query(default=None),
    user: str = Query(default="default_user", min_length=1),
):
    """Nodes and edges around a pillar or a center id.

    ``depth`` is the number of hops. The payload is capped inside
    ``slice_graph``. A missing graph table is 503 and names the migration.
    An unknown ``center`` is 404.
    """
    pillar_id = pillar.strip() if isinstance(pillar, str) and pillar.strip() else None
    center_id = center.strip() if isinstance(center, str) and center.strip() else None
    try:
        view = graph_view(
            pillar=pillar_id,
            depth=depth,
            center=center_id,
            user_id=user.strip() or "default_user",
        )
    except GraphStoreError as exc:
        logger.warning("Graph read failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Graph read failed")
        raise HTTPException(
            status_code=503, detail=f"Could not read the knowledge graph: {exc}"
        ) from exc
    if center_id and not view.get("found"):
        raise HTTPException(
            status_code=404,
            detail=f"No graph node with id {center_id!r}",
        )
    return view
