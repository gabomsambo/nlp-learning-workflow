"""
API Router for podcast script generation and management.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from nlp_pillars.agents.podcast_agent import (
    InsufficientSourceMaterialError, GroundPackExtractionError, PodcastAgent
)
from nlp_pillars.config import get_settings
from nlp_pillars.podcast_options import PodcastOptionError, resolve
from nlp_pillars.db import (
    add_podcast_script, get_podcast_scripts, get_podcast_script_by_id,
    get_paper_by_id, PodcastScriptLookupError, PodcastScriptSaveError
)
from webui.services import podcast_audio_service, run_service
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
    """Response after generating a podcast script.

    ``saved`` is separate from ``success`` on purpose. The script is the
    artifact and it cost ~$0.27 and four minutes to make; whether the insert
    afterwards worked is a different question, and answering it with a bare 500
    used to destroy the script. When ``saved`` is false the full text comes back
    in ``script`` so the user can keep it — the page renders it with an
    unmissable "not saved" banner.

    ``warnings`` carries anything the user needs to know about what the script
    was written from (see schemas.SourceMaterial).
    """
    success: bool
    script_id: Optional[str]
    title: str
    word_count: int
    message: str
    saved: bool = True
    source_material_level: str = "full"
    warnings: List[str] = []
    # What the script was aimed at, echoed back so the page can show the
    # settings that produced it without a second request.
    options: Dict[str, Any] = {}
    # Only populated when the script could not be stored; there is no other copy.
    script: Optional[str] = None
    key_points: List[str] = []


class PodcastAudioRequest(BaseModel):
    """Start background audio generation for a stored script."""

    script_id: str
    voice_path: str = Field(..., description="Library-relative path under /voices")


class PodcastAudioResponse(BaseModel):
    run_id: str
    message: str


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


@router.post("/generate")
async def generate_podcast(request: PodcastGenerateRequest):
    """
    Generate a podcast script for a paper.

    Args:
        request: Contains paper_id and pillar_id

    Returns:
        JSONResponse with script ID and metadata
    """
    try:
        logger.info(f"Generating podcast for paper {request.paper_id}")

        # Resolve the chosen options BEFORE anything expensive happens. An
        # unknown option or preset is a 400, never a guess: four minutes and
        # ~$0.27 of the wrong podcast is worse than a rejected request.
        options = resolve(request.options)

        # Get the paper's actual pillar_id from the database
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

        # Use the paper's actual pillar_id
        actual_pillar_id = paper_response['data'][0]['pillar_id']
        logger.info(f"Using pillar_id from paper: {actual_pillar_id}")

        # Create agent and generate script
        agent = PodcastAgent(options=options)
        script = await agent.generate(request.paper_id, actual_pillar_id)

        warnings = list(script.source_material.warnings)

        # Save to database. A failure here must not destroy the script: it
        # already exists and has already been paid for, so hand it back with
        # the real reason attached instead of answering 500 and dropping it.
        try:
            script_id = add_podcast_script(script)
        except PodcastScriptSaveError as e:
            logger.error(
                f"Generated podcast for {request.paper_id} but could not save it: {e}. "
                f"Returning the script in the response — there is no other copy."
            )
            unsaved = PodcastGenerateResponse(
                success=True,
                script_id=None,
                title=script.title,
                word_count=script.word_count,
                saved=False,
                source_material_level=script.source_material.level,
                warnings=warnings + [
                    f"This script was NOT saved to the database ({e}). It is shown "
                    f"below and nowhere else — download or copy it now, or it is lost."
                ],
                script=script.script,
                key_points=script.key_points,
                options=script.options.model_dump(),
                message=(
                    f"Generated a {script.word_count}-word script, but saving it "
                    f"failed. The script is in this response and has not been stored."
                ),
            )
            # 200, not 500: the generation the caller asked for succeeded and its
            # result is in the body. A 5xx would send this straight into generic
            # error handling and throw the artifact away, which is the bug.
            return JSONResponse(content=unsaved.model_dump())

        response = PodcastGenerateResponse(
            success=True,
            script_id=script_id,
            title=script.title,
            word_count=script.word_count,
            saved=True,
            source_material_level=script.source_material.level,
            warnings=warnings,
            options=script.options.model_dump(),
            message=f"Successfully generated podcast script with {script.word_count} words"
        )

        return JSONResponse(content=response.model_dump())

    except InsufficientSourceMaterialError as e:
        # 422, not 500: the request was well-formed, the paper is the problem,
        # and the message names which part of it is missing so the user can act
        # (re-upload the PDF, or process the paper so it has notes). Nothing was
        # spent — this is raised before the first model call.
        logger.warning(f"Refused to generate a podcast for {request.paper_id}: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except GroundPackExtractionError as e:
        # Extraction returned truncated or empty output after every attempt for
        # at least one section. Surface it — do not hand a broken Ground Pack
        # to synthesis.
        logger.error(f"Ground Pack extraction failed for {request.paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except PodcastOptionError as e:
        # A ValueError subclass, so this must stay above the ValueError branch
        # to keep its specific message (which names the valid choices).
        logger.warning(f"Rejected podcast options for {request.paper_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error generating podcast: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate podcast script: {str(e)}"
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
