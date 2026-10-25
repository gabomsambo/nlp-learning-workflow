"""Today's spend against the daily cap, for the budget chip and confirm dialog."""

import asyncio

from fastapi import APIRouter, HTTPException

from nlp_pillars import spend

router = APIRouter(prefix="/api/spend", tags=["spend"])


@router.get("")
async def get_spend() -> dict:
    try:
        budget = await asyncio.to_thread(spend.current_budget)
    except spend.SpendUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return budget.as_dict()
