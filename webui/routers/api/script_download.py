"""
Script Download API Router

Handles downloading podcast scripts as text files with proper encoding,
headers, and error handling.

Usage:
    from webui.routers.api import script_download
    app.include_router(script_download.router)

Then access via: GET /api/scripts/download/{script_id}
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scripts", tags=["Scripts"])


@router.get("/download/{script_id}")
async def download_script(
    script_id: str,
    background_tasks: Optional[BackgroundTasks] = None
) -> StreamingResponse:
    """
    Download a podcast script as a plain text file.

    This endpoint:
    1. Validates the script ID
    2. Fetches script content from database
    3. Formats it as readable text
    4. Returns it as a downloadable file with proper headers
    5. Logs the download in background

    Args:
        script_id: Unique identifier for the script
        background_tasks: Optional background task handler for logging

    Returns:
        StreamingResponse with text file content and download headers

    Raises:
        HTTPException: 400 for invalid input, 404 if not found, 500 on error

    Examples:
        JavaScript:
        >>> fetch('/api/scripts/download/script_123')
        ...     .then(r => r.blob())
        ...     .then(blob => {
        ...         const link = document.createElement('a');
        ...         link.href = URL.createObjectURL(blob);
        ...         link.download = 'script.txt';
        ...         link.click();
        ...     });

        HTML:
        >>> <a href="/api/scripts/download/{{ script_id }}" download>
        ...     Download Script
        ... </a>
    """

    try:
        # Validate input
        if not script_id or len(script_id) > 100:
            raise HTTPException(
                status_code=400,
                detail="Invalid script ID format"
            )

        # Fetch script from database
        # TODO: Replace with actual database call
        script = await get_script_from_db(script_id)

        if not script:
            raise HTTPException(
                status_code=404,
                detail="Script not found"
            )

        # Verify script has content
        if not script.get('content') and not script.get('sections'):
            raise HTTPException(
                status_code=400,
                detail="Script has no content to download"
            )

        # Format content for download
        content = format_script_for_download(script)

        # Encode to UTF-8 bytes
        content_bytes = content.encode('utf-8')

        # Generate safe filename with timestamp
        title_slug = (
            script.get('title', 'script')
            .lower()
            .replace(' ', '_')
            .replace('/', '_')
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"podcast_{title_slug}_{timestamp}.txt"

        # Log download in background if task manager available
        if background_tasks:
            background_tasks.add_task(
                _log_script_download,
                script_id=script_id,
                size_bytes=len(content_bytes),
                filename=filename
            )

        # Return file with proper headers
        return StreamingResponse(
            iter([content_bytes]),
            media_type="text/plain; charset=utf-8",
            headers={
                # Tell browser to download, not display
                "Content-Disposition": f'attachment; filename="{filename}"',

                # File size for progress bars
                "Content-Length": str(len(content_bytes)),

                # Prevent caching of downloads
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )

    except HTTPException:
        # Re-raise HTTP exceptions (already have proper status codes)
        raise

    except Exception as e:
        # Log unexpected errors
        logger.error(
            f"Error downloading script {script_id}: {type(e).__name__}: {e}",
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate script download"
        )


@router.get("/download/{script_id}/preview")
async def preview_script_download(script_id: str) -> Dict[str, Any]:
    """
    Preview what the download would contain without actually downloading.

    Useful for verifying content before download or checking file size.

    Args:
        script_id: Unique identifier for the script

    Returns:
        Dictionary with preview info:
        - content: Formatted text content
        - size_bytes: Size of the final download
        - lines: Number of lines
        - filename: Generated filename

    Raises:
        HTTPException: 404 if script not found
    """

    script = await get_script_from_db(script_id)

    if not script:
        raise HTTPException(status_code=404, detail="Script not found")

    content = format_script_for_download(script)
    content_bytes = content.encode('utf-8')

    title_slug = script.get('title', 'script').lower().replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"podcast_{title_slug}_{timestamp}.txt"

    return {
        "filename": filename,
        "content": content,
        "size_bytes": len(content_bytes),
        "lines": len(content.split('\n')),
        "sections": len(script.get('sections', []))
    }


async def get_script_from_db(script_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch script from database.

    TODO: Replace with actual database implementation.
    This is a placeholder that shows expected data structure.

    Args:
        script_id: Script identifier

    Returns:
        Script data dictionary or None if not found

    Expected structure:
        {
            "id": "script_123",
            "title": "Episode Title",
            "content": "Plain text content",  # OR sections below
            "sections": [
                {
                    "type": "speaker",
                    "speaker": "Host",
                    "text": "Speech content"
                },
                {
                    "type": "instruction",
                    "text": "Pause for 2 seconds"
                },
                {
                    "type": "note",
                    "text": "Background: music playing"
                }
            ],
            "created_at": "2024-01-15T10:30:00Z"
        }
    """

    # Placeholder: would query actual database
    # Examples:
    # - return await db.scripts.find_by_id(script_id)
    # - script = await supabase.table('scripts').select('*').eq('id', script_id)
    # - script = await mongodb.scripts.find_one({'_id': ObjectId(script_id)})

    return None  # Replace with actual implementation


def format_script_for_download(script: Dict[str, Any]) -> str:
    """
    Format script data into readable, downloadable text.

    Handles both structured sections and plain content.
    Adds headers, metadata, and proper formatting.

    Args:
        script: Script data dictionary

    Returns:
        Formatted text string ready for download

    Example output:
        ======================================================================
        PODCAST SCRIPT
        ======================================================================

        TITLE: My Amazing Podcast Episode
        DATE: 2024-01-15

        ======================================================================

        [HOST]
        Welcome everyone to the show! Today we're discussing AI.

        [EXPERT]
        Thanks for having me. It's great to be here.

        (Note: Play background music)

        [HOST]
        Let's dive into the topic...

        ======================================================================
        End of Script
        ======================================================================
    """

    lines = []

    # Header separator
    lines.append("=" * 70)
    lines.append("PODCAST SCRIPT")
    lines.append("=" * 70)
    lines.append("")

    # Metadata section
    if title := script.get('title'):
        lines.append(f"TITLE: {title}")

    if created_at := script.get('created_at'):
        lines.append(f"DATE: {created_at}")

    if duration := script.get('duration'):
        lines.append(f"DURATION: {duration}")

    lines.append("=" * 70)
    lines.append("")

    # Content section
    if sections := script.get('sections'):
        # Structured content with sections
        for i, section in enumerate(sections):
            section_type = section.get('type', 'text').lower()

            if section_type == 'speaker':
                # Speaker dialogue
                speaker = section.get('speaker', 'SPEAKER').upper()
                text = section.get('text', '').strip()

                lines.append(f"[{speaker}]")
                lines.append(text)
                lines.append("")

            elif section_type == 'instruction' or section_type == 'direction':
                # Production instructions
                text = section.get('text', '').strip()
                lines.append(f"[DIRECTION: {text}]")
                lines.append("")

            elif section_type == 'note':
                # Parenthetical notes
                text = section.get('text', '').strip()
                lines.append(f"(Note: {text})")
                lines.append("")

            elif section_type == 'break' or section_type == 'pause':
                # Silence/music breaks
                duration = section.get('duration', 'brief')
                lines.append(f"[PAUSE: {duration}]")
                lines.append("")

            elif section_type == 'music':
                # Music cues
                track = section.get('track', 'Music')
                lines.append(f"[MUSIC: {track}]")
                lines.append("")

            else:
                # Generic text content
                text = section.get('text', '').strip()
                if text:
                    lines.append(text)
                    lines.append("")

    else:
        # Fallback to plain content
        content = script.get('content', '')
        if content:
            lines.append(content)
            lines.append("")

    # Footer
    lines.append("")
    lines.append("=" * 70)
    lines.append("End of Script")
    lines.append("=" * 70)

    return '\n'.join(lines)


async def _log_script_download(
    script_id: str,
    size_bytes: int,
    filename: str
) -> None:
    """
    Log script download in background.

    Called as background task to avoid blocking the response.

    Args:
        script_id: The downloaded script ID
        size_bytes: Size of the downloaded file
        filename: Generated filename for the download
    """

    try:
        logger.info(
            f"Script download",
            extra={
                "script_id": script_id,
                "size_bytes": size_bytes,
                "filename": filename,
                "timestamp": datetime.now().isoformat()
            }
        )

        # TODO: Store analytics in database
        # await db.logs.create({
        #     "event": "script_download",
        #     "script_id": script_id,
        #     "size_bytes": size_bytes,
        #     "timestamp": datetime.now()
        # })

    except Exception as e:
        logger.error(f"Error logging script download: {e}")
