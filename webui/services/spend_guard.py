"""Gate for the routes that spend money or the GPU.

Used as ``dependencies=[Depends(require_spend_room)]`` so it runs before the
handler: several handlers wrap their whole body in ``except Exception``, which
would otherwise swallow the refusal.

Two checks, in this order:

1. A request that came through the tunnel must say it was confirmed
   (``X-Nlp-Confirm: 1``, added by ``spend-guard.js`` after the person clicks
   through the dialog). A phone tap on a stale page, or a script holding a
   valid Access session, cannot start paid work by accident. The desktop is
   exempt; it gets the same dialog but is not forced through it server-side.
2. The day's recorded spend must be under ``DAILY_SPEND_CAP_USD``. This applies
   to everyone, desktop included.
"""

import asyncio
import logging

from fastapi import HTTPException, Request

from nlp_pillars import spend
from webui.access import is_remote

logger = logging.getLogger(__name__)

CONFIRM_HEADER = "x-nlp-confirm"


async def require_spend_room(request: Request) -> None:
    if is_remote(request.headers) and request.headers.get(CONFIRM_HEADER) != "1":
        raise HTTPException(
            status_code=428,
            detail="This action spends money or uses the GPU and needs an explicit confirmation.",
        )
    try:
        budget = await asyncio.to_thread(spend.current_budget)
    except spend.SpendUnavailableError as exc:
        logger.error("Refusing costly action: spend unreadable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Could not read today's spend, so this was not started.",
        )
    if budget.exhausted:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily spend cap reached (${budget.spent_usd:.2f} of ${budget.cap_usd:.2f}). "
                f"It resets at {budget.resets_at.strftime('%H:%M %Z on %b %d')}. "
                "Raise DAILY_SPEND_CAP_USD to continue."
            ),
        )
