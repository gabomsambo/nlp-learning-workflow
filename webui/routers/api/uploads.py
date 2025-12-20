"""
API router for paper upload operations.
Handles file uploads and URL downloads for papers associated with pillars.
"""

from typing import Optional, List

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse

from nlp_pillars.schemas import (
    UploadResponse, UploadUrlRequest, UploadFileRequest, UploadStatus
)
from nlp_pillars.services.upload_service import get_upload_service, UploadError
from nlp_pillars.db import pillar_exists

router = APIRouter(prefix="/api/pillars", tags=["uploads"])


@router.post("/{pillar_id}/upload/url", response_model=UploadResponse)
async def upload_paper_from_url(
    pillar_id: str,
    request: UploadUrlRequest
) -> UploadResponse:
    """
    Upload a paper from a URL.
    
    - **pillar_id**: Target pillar ID (slug)
    - **url**: URL to download PDF from
    - **title**: Optional title override
    - **authors**: Optional authors list
    - **run_summarizer**: Whether to run summarizer after upload
    - **generate_quiz**: Whether to generate quiz after upload
    """
    try:
        # Validate pillar exists
        if not pillar_exists(pillar_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pillar '{pillar_id}' not found"
            )
        
        upload_service = get_upload_service()
        result = await upload_service.upload_from_url(pillar_id, request)
        return result
        
    except UploadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.post("/{pillar_id}/upload/pdf", response_model=UploadResponse)
async def upload_paper_from_file(
    pillar_id: str,
    file: UploadFile = File(..., description="PDF file to upload"),
    title: str = Form(..., description="Paper title"),
    authors: Optional[str] = Form(None, description="Comma-separated list of authors"),
    venue: Optional[str] = Form(None, description="Conference or journal"),
    year: Optional[int] = Form(None, description="Publication year"),
    run_summarizer: bool = Form(False, description="Run summarizer after upload"),
    generate_quiz: bool = Form(False, description="Generate quiz after upload")
) -> UploadResponse:
    """
    Upload a paper from a file.
    
    Accepts multipart/form-data with:
    - **file**: PDF file to upload
    - **title**: Paper title (required)
    - **authors**: Comma-separated list of authors (optional)
    - **venue**: Conference or journal (optional)
    - **year**: Publication year (optional)
    - **run_summarizer**: Whether to run summarizer after upload
    - **generate_quiz**: Whether to generate quiz after upload
    """
    try:
        # Validate pillar exists
        if not pillar_exists(pillar_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pillar '{pillar_id}' not found"
            )
        
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        # Parse authors from comma-separated string
        authors_list = []
        if authors:
            authors_list = [author.strip() for author in authors.split(',') if author.strip()]
        
        # Create request object
        request = UploadFileRequest(
            title=title,
            authors=authors_list,
            venue=venue,
            year=year,
            run_summarizer=run_summarizer,
            generate_quiz=generate_quiz
        )
        
        upload_service = get_upload_service()
        result = await upload_service.upload_from_file(pillar_id, file, request)
        return result
        
    except UploadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.get("/{pillar_id}/uploads/status/{upload_id}", response_model=UploadStatus)
async def get_upload_status(
    pillar_id: str,
    upload_id: str
) -> UploadStatus:
    """
    Get the status of a specific upload.
    
    - **pillar_id**: Target pillar ID
    - **upload_id**: Upload ID returned from upload endpoint
    """
    try:
        upload_service = get_upload_service()
        upload_status = upload_service.get_upload_status(upload_id)
        
        if not upload_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Upload '{upload_id}' not found"
            )
        
        if upload_status.pillar_id != pillar_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Upload does not belong to this pillar"
            )
        
        return upload_status
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get upload status: {str(e)}"
        )


@router.get("/{pillar_id}/uploads/recent", response_model=List[UploadStatus])
async def get_recent_uploads(
    pillar_id: str,
    limit: int = 10
) -> List[UploadStatus]:
    """
    Get recent uploads for a pillar.
    
    - **pillar_id**: Target pillar ID
    - **limit**: Maximum number of uploads to return (default: 10)
    """
    try:
        # Validate pillar exists
        if not pillar_exists(pillar_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pillar '{pillar_id}' not found"
            )
        
        upload_service = get_upload_service()
        uploads = upload_service.get_recent_uploads(pillar_id, limit)
        
        return uploads
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recent uploads: {str(e)}"
        )
