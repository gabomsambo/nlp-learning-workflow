"""The learner's confirm / reject / edit of suggested connections between papers.

A suggestion is a pair of works the map says are connected (``evidence``,
``connections`` and ``bridges`` in the pillar's snapshot). This route stores
what the learner decided about it: confirm it (optionally with another
relation and their own phrase) or reject it. ``provenance`` is always
``user``. A map rebuild reads these decisions and never writes them, so a
decision survives every rebuild; the next rebuild shows a confirmed pair with
the learner's phrase and a rejected pair flagged, with no bridge sentence
written for it.

Nodes are map node ids: a ``works.id`` uuid, or ``paper:<id>`` for a library
paper Semantic Scholar does not know. Handlers are plain ``def`` so FastAPI
runs them in its thread pool: the database client here is the synchronous one.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from nlp_pillars.connections import (
    PHRASE_MAX_CHARS,
    RELATIONS,
    STATUSES,
    ConnectionStoreError,
    delete_decision,
    load_decisions,
    put_decision,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/connections", tags=["connections"])


class DecisionBody(BaseModel):
    a: str = Field(..., min_length=1, max_length=200, description="Map node id of one work")
    b: str = Field(..., min_length=1, max_length=200, description="Map node id of the other work")
    status: str = Field(..., description="confirmed or rejected")
    relation: str = Field("related", description="How a relates to b")
    phrase: Optional[str] = Field(
        None, max_length=PHRASE_MAX_CHARS * 2, description="The learner's own words for the link"
    )


@router.get("")
def list_decisions():
    """Every decision, plus the relations and statuses the form may offer."""
    try:
        decisions = load_decisions()
    except ConnectionStoreError as exc:
        logger.warning("Reading learner connections failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "decisions": [d.as_dict() for d in decisions],
        "relations": list(RELATIONS),
        "statuses": list(STATUSES),
        "phrase_max_chars": PHRASE_MAX_CHARS,
    }


@router.put("")
def decide(body: DecisionBody):
    """Confirm, edit or reject the connection between ``a`` and ``b``.

    The order of ``a`` and ``b`` is the direction of ``relation`` ("a builds on
    b"); a pair has one decision whichever way it is sent.
    """
    if body.status not in STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of {', '.join(STATUSES)}")
    if body.relation not in RELATIONS:
        raise HTTPException(status_code=422, detail=f"relation must be one of {', '.join(RELATIONS)}")
    if body.a.strip() == body.b.strip():
        raise HTTPException(status_code=422, detail="A connection needs two different works.")
    try:
        decision = put_decision(
            body.a, body.b, status=body.status, relation=body.relation, phrase=body.phrase
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ConnectionStoreError as exc:
        logger.warning("Saving a learner connection failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return decision.as_dict()


@router.delete("")
def undo(a: str = Query(..., min_length=1), b: str = Query(..., min_length=1)):
    """Forget the decision for a pair, returning it to suggested. 404 when there was none."""
    try:
        removed = delete_decision(a, b)
    except ConnectionStoreError as exc:
        logger.warning("Removing a learner connection failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="No decision to remove for that pair.")
    return {"removed": removed}
