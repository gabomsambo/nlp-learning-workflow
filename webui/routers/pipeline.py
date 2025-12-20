import asyncio
import sys

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from nlp_pillars.db import get_pillars
from nlp_pillars.pillar_utils import validate_pillar_id

router = APIRouter(prefix="/pipeline")


@router.get("/", response_class=HTMLResponse)
async def pipeline_home(request: Request) -> HTMLResponse:
    pillars = get_pillars(limit=100)
    return request.app.state.templates.TemplateResponse("pipeline.html", {"request": request, "pillars": pillars})


@router.post("/run")
async def trigger_run(pillar: str, papers: int = 1):
    if not validate_pillar_id(pillar):
        return JSONResponse({"success": False, "error": f"Invalid pillar: {pillar}"}, status_code=400)

    cmd = [
        sys.executable,
        "-m",
        "nlp_pillars.cli",
        "run",
        "--pillar",
        pillar,
        "--papers",
        str(papers),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        success = proc.returncode == 0
        return JSONResponse(
            {
                "success": success,
                "returncode": proc.returncode,
                "stdout": stdout.decode(errors="ignore"),
                "stderr": stderr.decode(errors="ignore"),
            },
            status_code=200 if success else 500,
        )
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


