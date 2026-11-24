"""Daily run: turn the scheduled run on or off, set its time, pillars and size."""

import asyncio
from zoneinfo import available_timezones

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from webui.routers.api.daily_run import build_status

router = APIRouter(prefix="/daily-run")

_TIMEZONES = sorted(available_timezones())


@router.get("/", response_class=HTMLResponse)
async def daily_run_home(request: Request) -> HTMLResponse:
    status = await asyncio.to_thread(build_status)
    return request.app.state.templates.TemplateResponse(
        request,
        "daily_run.html",
        {"status": status, "timezones": _TIMEZONES},
    )
