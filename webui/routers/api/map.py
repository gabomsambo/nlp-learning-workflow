"""Read-only paper map: the stored snapshot plus live overlays.

The page that draws this is a separate change. This route only returns JSON and
does not build the map: ``python -m nlp_pillars.cli map build`` does that.
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from nlp_pillars.map_store import MapStoreError
from nlp_pillars.map_view import UnknownScopeError, map_view

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/map", tags=["map"])


@router.get("")
def get_map(
    scope: str = Query(min_length=1, description="A pillar id, or 'all'."),
    user: str = Query(default="default_user", min_length=1),
):
    """The snapshot for ``scope`` with queue and mastery overlays.

    ``status`` is ``not_built`` (no snapshot yet), ``empty`` (built, no works)
    or ``ready``. A missing ``map_snapshots`` table is 503 and names the
    migration; an unknown pillar is 404. An overlay that cannot be read is
    reported inside ``overlays`` and does not fail the request.
    """
    scope_id = scope.strip()
    if not scope_id:
        raise HTTPException(status_code=422, detail="scope must not be blank")
    try:
        return map_view(scope_id, user_id=user.strip() or "default_user")
    except UnknownScopeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MapStoreError as exc:
        logger.warning("Map read failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Map read failed")
        raise HTTPException(
            status_code=503, detail=f"Could not read the map: {exc}"
        ) from exc
