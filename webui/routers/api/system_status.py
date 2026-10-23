"""Read-only workstation status for the Activity page and the shell chip.

One failing check never fails this endpoint: the body is always a list of
OK / DOWN rows, and a broken collector still answers 200 with a single DOWN row.
Nothing here calls a model.
"""

import logging
import os

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from nlp_pillars.config import get_settings
from webui.services import system_status as status_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/status")
async def system_status() -> JSONResponse:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=status_service.CHECK_TIMEOUT_SECONDS) as client:
            payload = await status_service.collect_status(
                client=client,
                settings=settings,
                postgrest_url=status_service.resolve_postgrest_url(),
                auth_token=os.getenv("SUPABASE_KEY"),
            )
    except Exception:
        logger.exception("system status could not be collected")
        payload = status_service.failed_payload()
    return JSONResponse(payload)
