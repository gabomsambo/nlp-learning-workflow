"""
API Router for podcast script generation and management.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from nlp_pillars.agents.podcast_agent import PodcastAgent
from nlp_pillars.db import (
    add_podcast_script, get_podcast_scripts, get_podcast_script_by_id,
    get_paper_by_id
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/podcast", tags=["podcast"])


# Request/Response models
class PodcastGenerateRequest(BaseModel):
    """Request to generate a podcast script."""
    paper_id: str
    pillar_id: str


class PodcastGenerateResponse(BaseModel):
    """Response after generating a podcast script."""
    success: bool
    script_id: Optional[str]
    title: str
    word_count: int
    message: str


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
        agent = PodcastAgent()
        script = await agent.generate(request.paper_id, actual_pillar_id)

        # Save to database
        script_id = add_podcast_script(script)

        if not script_id:
            raise HTTPException(
                status_code=500,
                detail="Failed to save podcast script to database"
            )

        response = PodcastGenerateResponse(
            success=True,
            script_id=script_id,
            title=script.title,
            word_count=script.word_count,
            message=f"Successfully generated podcast script with {script.word_count} words"
        )

        return JSONResponse(content=response.model_dump())

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

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
                "created_at": script.created_at.isoformat() if script.created_at else None
            }
        })

    except HTTPException:
        raise

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

    except Exception as e:
        logger.error(f"Error downloading podcast {script_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download podcast script: {str(e)}"
        )
