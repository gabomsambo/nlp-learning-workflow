"""
API Router for podcast script generation and management.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from nlp_pillars.config import get_settings
from nlp_pillars.podcast_options import PodcastOptionError, resolve
from nlp_pillars.db import (
    get_podcast_scripts, get_podcast_script_by_id,
    PipelineRunCreateError, PodcastScriptLookupError,
)
from webui.services import podcast_audio_service, podcast_script_service, run_service
from webui.services.run_service import RunAlreadyActiveError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/podcast", tags=["podcast"])


# Request/Response models
class PodcastGenerateRequest(BaseModel):
    """Request to generate a podcast script.

    ``options`` is the flat shape the form posts::

        {"field": "nlp", "audience": "graduate", "length": "30",
         "tone": "__custom__", "tone_custom": "dry and sceptical"}

    Omitted entirely (which is what every pre-existing caller sends) means every
    default, and the defaults reproduce the aiming the prompts hardcoded before
    they were configurable. Validation lives in nlp_pillars/podcast_options.py
    rather than in a pydantic model, so adding a fifth option stays a data
    change here too.
    """
    paper_id: str
    pillar_id: str
    options: Optional[Dict[str, str]] = None


class PodcastGenerateResponse(BaseModel):
    """Immediate ack after queueing podcast script generation.

    The script itself comes back on the finished run's ``result`` payload
    (polled via ``GET /api/pipeline-runs/{id}``), not in this response — the
    route used to hold the HTTP connection open for ~4 minutes.
    """
    run_id: str
    message: str


class PodcastAudioRequest(BaseModel):
    """Start background audio generation for a stored script."""

    script_id: str
    voice_path: str = Field(..., description="Library-relative path under /voices")


class PodcastAudioResponse(BaseModel):
    run_id: str
    message: str


#: Named when create_pipeline_run fails the kind/trigger CHECK — migration 016
#: is hand-applied, and a bare Postgres error is not actionable.
_MIGRATION_HINT = (
    "Podcast script generation runs as pipeline runs of kind 'podcast_script'. "
    "If this database has not had docs/migrations/016_podcast_script_runs.sql "
    "applied, the insert fails its CHECK constraint. Apply it with: "
    "docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - "
    "< docs/migrations/016_podcast_script_runs.sql"
)

@router.get("/tts/status")
async def tts_status():
    """IndexTTS liveness with wrong-service detection."""
    return JSONResponse(content=podcast_audio_service.tts_status_payload())


@router.get("/tts/voices")
async def tts_voices():
    """List voice references from the mounted library with usability preflight."""
    return JSONResponse(content=podcast_audio_service.list_voices_payload())


@router.get("/tts/voices/preview")
async def tts_voice_preview(path: str):
    """Return a short preview clip for the selected reference voice."""
    try:
        preview = podcast_audio_service.preview_voice(path)
        return FileResponse(
            preview,
            media_type="audio/wav",
            filename=preview.name,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tts/generate", status_code=202)
async def generate_podcast_audio(request: PodcastAudioRequest, http_request: Request):
    """Queue podcast audio synthesis as a pipeline run."""
    script = get_podcast_script_by_id(request.script_id)
    if not script:
        raise HTTPException(status_code=404, detail="Script not found")

    try:
        podcast_audio_service.validate_voice_for_generation(request.voice_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    status = podcast_audio_service.tts_status_payload()
    if not status.get("ready"):
        raise HTTPException(status_code=503, detail=status.get("message"))

    try:
        run_id = run_service.dispatch_run(
            http_request.app.state.scheduler,
            http_request.app.state.cancel_events,
            pillar_id=script.pillar_id,
            trigger_source=podcast_audio_service.TRIGGER_UI_PODCAST_AUDIO,
            kind=podcast_audio_service.KIND_PODCAST_AUDIO,
            script_id=request.script_id,
            voice_path=request.voice_path,
        )
    except RunAlreadyActiveError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return PodcastAudioResponse(
        run_id=run_id,
        message="Audio generation started",
    )


@router.get("/audio/{filename}")
async def serve_podcast_audio(filename: str):
    """Stream a generated MP3 from the podcast audio directory."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    audio_dir = Path(get_settings().podcast_audio_dir)
    target = (audio_dir / filename).resolve()
    if not str(target).startswith(str(audio_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(target, media_type="audio/mpeg", filename=filename)


@router.post("/generate", status_code=202)
async def generate_podcast(request: PodcastGenerateRequest, http_request: Request):
    """Queue podcast script generation as a pipeline run.

    Returns 202 with a run id immediately. Progress is the seven
    ``podcast_*`` stages on that run; the finished script (or an unsaved
    copy when the insert fails) lands in ``pipeline_runs.result``.
    """
    try:
        logger.info(f"Queueing podcast generation for paper {request.paper_id}")

        # Resolve options BEFORE anything expensive. An unknown option is a 400,
        # never four minutes of the wrong podcast.
        options = resolve(request.options)

        from nlp_pillars.db import get_client
        client = get_client()
        paper_response = (client.table('papers')
                         .select('pillar_id')
                         .eq('id', request.paper_id)
                         .execute())

        if paper_response['error'] or not paper_response['data']:
            raise HTTPException(
                status_code=404,
                detail=f"Paper not found: {request.paper_id}"
            )

        actual_pillar_id = paper_response['data'][0]['pillar_id']
        logger.info(f"Using pillar_id from paper: {actual_pillar_id}")

        try:
            run_id = run_service.dispatch_run(
                http_request.app.state.scheduler,
                http_request.app.state.cancel_events,
                pillar_id=actual_pillar_id,
                trigger_source=podcast_script_service.TRIGGER_UI_PODCAST_SCRIPT,
                kind=podcast_script_service.KIND_PODCAST_SCRIPT,
                paper_id=request.paper_id,
                options=options,
            )
        except RunAlreadyActiveError as e:
            # Should be unreachable once 016 is applied (script runs are exempt).
            raise HTTPException(status_code=409, detail=str(e))
        except PipelineRunCreateError as e:
            logger.error("Could not create podcast_script run: %s", e)
            raise HTTPException(
                status_code=503,
                detail=f"{e}. {_MIGRATION_HINT}",
            )

        return PodcastGenerateResponse(
            run_id=run_id,
            message="Podcast script generation started",
        )

    except PodcastOptionError as e:
        logger.warning(f"Rejected podcast options for {request.paper_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error queueing podcast generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start podcast script generation: {str(e)}"
        )


@router.get("/list")
async def list_podcasts(pillar: Optional[str] = None, limit: int = 20):
    """
    List podcast scripts, optionally filtered by pillar.

    Args:
        pillar: Optional pillar ID to filter by
        limit: Maximum number of scripts to return

    Returns:
        JSONResponse with list of scripts
    """
    try:
        scripts = get_podcast_scripts(pillar_id=pillar, limit=limit)

        # Convert to serializable format
        scripts_data = []
        for script in scripts:
            scripts_data.append({
                "id": script.id,
                "paper_id": script.paper_id,
                "pillar_id": script.pillar_id,
                "title": script.title,
                "word_count": script.word_count,
                "created_at": script.created_at.isoformat() if script.created_at else None
            })

        return JSONResponse(content={
            "success": True,
            "scripts": scripts_data,
            "count": len(scripts_data)
        })

    except PodcastScriptLookupError as e:
        # The database could not be read. Saying so beats an empty list, which
        # is indistinguishable from "you have never generated one".
        logger.error(f"Error listing podcasts: {e}")
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        logger.error(f"Error listing podcasts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list podcast scripts: {str(e)}"
        )


@router.get("/{script_id}")
async def get_podcast(script_id: str):
    """
    Get a specific podcast script by ID.

    Args:
        script_id: The script ID to fetch

    Returns:
        JSONResponse with full script data
    """
    try:
        script = get_podcast_script_by_id(script_id)

        if not script:
            raise HTTPException(status_code=404, detail="Script not found")

        return JSONResponse(content={
            "success": True,
            "script": {
                "paper_id": script.paper_id,
                "pillar_id": script.pillar_id,
                "title": script.title,
                "script": script.script,
                "word_count": script.word_count,
                "key_points": script.key_points,
                "ground_pack": script.ground_pack,
                # So a stored script written from partial material still says so
                # when it is re-opened, not only on the run that produced it.
                "source_material": script.source_material.model_dump(),
                # So a stored script still says what it was aimed at when it is
                # re-opened; without it, two scripts for the same paper differ
                # for no visible reason.
                "options": script.options.model_dump(),
                "audio_metadata": script.audio_metadata.model_dump(mode="json"),
                "created_at": script.created_at.isoformat() if script.created_at else None
            }
        })

    except HTTPException:
        raise

    except PodcastScriptLookupError as e:
        # 503, never 404. A 404 here means "this script does not exist", and
        # answering it for a database blip told the user their script was gone.
        logger.error(f"Error fetching podcast {script_id}: {e}")
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        logger.error(f"Error fetching podcast {script_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch podcast script: {str(e)}"
        )


@router.get("/{script_id}/download")
async def download_podcast(script_id: str):
    """
    Download podcast script as .txt file.

    Args:
        script_id: The script ID to download

    Returns:
        StreamingResponse with text file
    """
    try:
        script = get_podcast_script_by_id(script_id)

        if not script:
            raise HTTPException(status_code=404, detail="Script not found")

        # Format content for download
        content = f"# {script.title}\n\n"
        content += f"Paper ID: {script.paper_id}\n"
        content += f"Word Count: {script.word_count}\n"
        content += f"Generated: {script.created_at.strftime('%Y-%m-%d %H:%M') if script.created_at else 'Unknown'}\n"
        content += "\n" + "=" * 60 + "\n\n"
        content += script.script

        # Encode to bytes
        content_bytes = content.encode('utf-8')

        # Generate safe filename
        safe_title = "".join(c if c.isalnum() or c in ' -_' else '' for c in script.title)
        filename = f"{safe_title[:50]}_podcast.txt"

        return StreamingResponse(
            iter([content_bytes]),
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(content_bytes)),
                "Cache-Control": "no-cache, no-store, must-revalidate"
            }
        )

    except HTTPException:
        raise

    except PodcastScriptLookupError as e:
        logger.error(f"Error downloading podcast {script_id}: {e}")
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        logger.error(f"Error downloading podcast {script_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download podcast script: {str(e)}"
        )
