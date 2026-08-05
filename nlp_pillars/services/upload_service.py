"""Upload Service for Paper Management.

Handles file uploads, URL downloads, and integration with existing
PDF processing and paper management systems.
"""

import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import arxiv
from fastapi import UploadFile

from .. import db, vectors
from ..agents.ingest_agent import IngestAgent, IngestError
from ..agents.quiz_agent import QuizAgent
from ..agents.summarizer_agent import SummarizerAgent
from ..agents.synthesis_agent import SynthesisAgent
from ..db import add_paper, get_pillar_by_id
from ..schemas import (
    PaperRef,
    PillarConfig,
    QuizGeneratorInput,
    SummarizerInput,
    SynthesisInput,
    UploadFileRequest,
    UploadResponse,
    UploadStatus,
    UploadUrlRequest,
)
from ..tools.pdf_loader import (
    PDFDownloadError,
    PDFParseError,
    download_pdf,
    extract_text,
)

logger = logging.getLogger(__name__)


class UploadError(Exception):
    """Custom exception for upload operations."""

    pass


class UploadService:
    """Service for handling paper uploads and processing."""

    def __init__(self, upload_dir: str = ".cache/uploads"):
        """Initialize the upload service with a directory for uploads."""
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.ingest_agent = IngestAgent()
        self.upload_statuses: Dict[str, UploadStatus] = {}

    async def upload_from_url(
        self,
        pillar_id: str,
        request: UploadUrlRequest
    ) -> UploadResponse:
        """Upload a paper from a URL.

        Args:
            pillar_id: Target pillar ID
            request: Upload request with URL and options

        Returns:
            Upload response with paper details

        """
        upload_id = str(uuid.uuid4())
        logger.info(f"URL upload {upload_id} for {pillar_id}: {request.url}")

        # Validate pillar exists
        pillar = get_pillar_by_id(pillar_id)
        if not pillar:
            raise UploadError(f"Pillar '{pillar_id}' not found")

        # Create upload status
        status = UploadStatus(
            id=upload_id,
            pillar_id=pillar_id,
            status="processing",
            url=request.url,
            message="Downloading PDF from URL..."
        )
        self.upload_statuses[upload_id] = status

        try:
            # Step 1: Download PDF
            status.progress = 20
            status.message = "Downloading PDF..."
            pdf_path = download_pdf(request.url, str(self.upload_dir / "downloads"))

            # Step 2: Extract basic metadata or use provided
            status.progress = 40
            status.message = "Extracting metadata..."
            paper = await self._create_paper_ref_from_url(
                request.url, pdf_path, request.title, request.authors
            )

            # Step 3: Add to database
            status.progress = 60
            status.message = "Saving to database..."

            # Add paper using string pillar_id directly
            success = add_paper(pillar_id, paper)
            if not success:
                raise UploadError("Failed to add paper to database")

            # Step 4: Run full pipeline processing
            status.progress = 40
            status.message = "Processing paper through pipeline..."

            # Always run text extraction, conditionally run summarizer/quiz
            actions_triggered = await self._run_full_pipeline(
                paper=paper,
                pillar_id=pillar_id,
                status=status,
                run_summarizer=request.run_summarizer,
                generate_quiz=request.generate_quiz
            )

            # Complete
            status.progress = 100
            status.status = "completed"
            status.message = "Upload completed successfully"

            logger.info(f"Uploaded {paper.id} from URL: {request.url}")

            return UploadResponse(
                success=True,
                paper=paper,
                message="Paper uploaded successfully from URL",
                actions_triggered=actions_triggered
            )

        except (PDFDownloadError, PDFParseError, IngestError) as e:
            status.status = "failed"
            status.message = f"Upload failed: {str(e)}"
            logger.error(f"Upload {upload_id} failed: {e}")
            raise UploadError(f"Failed to upload from URL: {e}") from e

        except Exception as e:
            status.status = "failed"
            status.message = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error in upload {upload_id}: {e}")
            raise UploadError(f"Unexpected error during upload: {e}") from e

    async def upload_from_file(
        self,
        pillar_id: str,
        file: UploadFile,
        request: UploadFileRequest
    ) -> UploadResponse:
        """Upload a paper from a file.

        Args:
            pillar_id: Target pillar ID
            file: Uploaded file
            request: Upload request with metadata and options

        Returns:
            Upload response with paper details

        """
        upload_id = str(uuid.uuid4())
        logger.info(f"File upload {upload_id} for {pillar_id}: {file.filename}")

        # Validate pillar exists
        pillar = get_pillar_by_id(pillar_id)
        if not pillar:
            raise UploadError(f"Pillar '{pillar_id}' not found")

        # Validate file
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise UploadError("Only PDF files are supported")

        # Create upload status
        status = UploadStatus(
            id=upload_id,
            pillar_id=pillar_id,
            status="processing",
            filename=file.filename,
            message="Saving uploaded file..."
        )
        self.upload_statuses[upload_id] = status

        temp_path = None
        try:
            # Step 1: Save uploaded file
            status.progress = 20
            temp_path = await self._save_uploaded_file(file)

            # Step 2: Create paper reference
            status.progress = 40
            status.message = "Creating paper reference..."
            paper = await self._create_paper_ref_from_file(
                temp_path, file.filename, request
            )

            # Step 3: Add to database
            status.progress = 60
            status.message = "Saving to database..."

            # Add paper using string pillar_id directly
            success = add_paper(pillar_id, paper)
            if not success:
                raise UploadError("Failed to add paper to database")

            # Step 4: Run full pipeline processing
            status.progress = 60
            status.message = "Processing paper through pipeline..."

            # Always run text extraction, conditionally run summarizer/quiz
            actions_triggered = await self._run_full_pipeline(
                paper=paper,
                pillar_id=pillar_id,
                status=status,
                run_summarizer=request.run_summarizer,
                generate_quiz=request.generate_quiz
            )

            # Complete
            status.progress = 100
            status.status = "completed"
            status.message = "Upload completed successfully"

            logger.info(f"Uploaded {paper.id} from file: {file.filename}")

            return UploadResponse(
                success=True,
                paper=paper,
                message="Paper uploaded successfully from file",
                actions_triggered=actions_triggered
            )

        except Exception as e:
            status.status = "failed"
            status.message = f"Upload failed: {str(e)}"
            logger.error(f"Upload {upload_id} failed: {e}")
            raise UploadError(f"Failed to upload file: {e}") from e

        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_path}: {e}")

    def get_upload_status(self, upload_id: str) -> Optional[UploadStatus]:
        """Get upload status by ID."""
        return self.upload_statuses.get(upload_id)

    def get_recent_uploads(self, pillar_id: str, limit: int = 10) -> List[UploadStatus]:
        """Get recent uploads for a pillar."""
        pillar_uploads = [
            status for status in self.upload_statuses.values()
            if status.pillar_id == pillar_id
        ]
        # Sort by creation time, most recent first
        pillar_uploads.sort(key=lambda x: x.created_at, reverse=True)
        return pillar_uploads[:limit]

    async def _save_uploaded_file(self, file: UploadFile) -> str:
        """Save uploaded file to temporary location."""
        # Generate unique filename
        file_id = f"{file.filename}_{uuid.uuid4()}"
        file_hash = hashlib.sha256(file_id.encode()).hexdigest()
        temp_path = self.upload_dir / f"{file_hash}.pdf"

        # Save file
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Saved uploaded file to {temp_path}")
        return str(temp_path)

    async def _create_paper_ref_from_url(
        self,
        url: str,
        pdf_path: str,
        title_override: Optional[str],
        authors_override: Optional[List[str]]
    ) -> PaperRef:
        """Create PaperRef from URL with automatic metadata enrichment."""
        # Generate paper ID from URL
        paper_id = self._generate_paper_id_from_url(url)

        # Extract title if not provided
        title = title_override
        default_title = f"Paper from {urlparse(url).netloc}"
        if not title:
            try:
                # Try to extract title from PDF text
                text = extract_text(pdf_path)
                title = self._extract_title_from_text(text) or default_title
            except Exception as e:
                logger.warning(f"Failed to extract title from PDF: {e}")
                title = default_title

        # Use provided authors or empty list
        authors = authors_override or []

        paper = PaperRef(
            id=paper_id,
            title=title,
            authors=authors,
            url_pdf=url,
            venue=None,
            year=None,
            abstract=None,
            citation_count=0
        )

        # Enrich from arXiv if it's an arXiv URL
        logger.info(f"Attempting metadata enrichment for URL: {url}")
        paper = self._enrich_from_arxiv(paper, url)

        # Fallback to Semantic Scholar if still missing metadata
        if not paper.authors or not paper.year or not paper.abstract:
            paper = self._enrich_from_semantic_scholar(paper)

        return paper

    async def _create_paper_ref_from_file(
        self,
        file_path: str,
        filename: str,
        request: UploadFileRequest
    ) -> PaperRef:
        """Create PaperRef from uploaded file with automatic metadata enrichment."""
        # Check if filename contains an arXiv ID first (for better ID generation)
        arxiv_match = re.search(r'(\d{4}\.\d{4,5})', filename)
        if arxiv_match:
            paper_id = f"arxiv:{arxiv_match.group(1)}"
        else:
            paper_id = self._generate_paper_id_from_filename(filename)

        # Use the local file path as the PDF URL
        pdf_url = f"file://{os.path.abspath(file_path)}"

        paper = PaperRef(
            id=paper_id,
            title=request.title,
            authors=request.authors or [],
            venue=request.venue,
            year=request.year,
            url_pdf=pdf_url,
            abstract=None,
            citation_count=0
        )

        # Enrich from arXiv if filename contains arXiv ID
        if arxiv_match:
            logger.info(f"Attempting metadata enrichment for file: {filename}")
            paper = self._enrich_from_arxiv(paper, filename)

        # Fallback to Semantic Scholar if title is provided but missing other metadata
        if paper.title and (not paper.authors or not paper.year or not paper.abstract):
            paper = self._enrich_from_semantic_scholar(paper)

        return paper

    def _generate_paper_id_from_url(self, url: str) -> str:
        """Generate a unique paper ID from URL."""
        # Try to extract arXiv ID if it's an arXiv URL
        if 'arxiv.org' in url.lower():
            # Extract arXiv ID from URL
            import re
            match = re.search(r'(\d{4}\.\d{4,5})', url)
            if match:
                return f"arxiv:{match.group(1)}"

        # For other URLs, create a hash-based ID
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
        return f"url:{url_hash}"

    def _generate_paper_id_from_filename(self, filename: str) -> str:
        """Generate a unique paper ID from filename."""
        # Remove extension and create hash
        name_without_ext = os.path.splitext(filename)[0]
        name_hash = hashlib.sha256(name_without_ext.encode()).hexdigest()[:12]
        return f"file:{name_hash}"

    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """Extract probable title from PDF text."""
        if not text:
            return None

        # Take first non-empty line as potential title
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 10 and len(line) < 200:
                # Basic heuristics for title-like content
                if not line.isdigit() and not line.startswith('http'):
                    return line

        return None


    async def _run_full_pipeline(
        self,
        paper: PaperRef,
        pillar_id: str,
        status: UploadStatus,
        run_summarizer: bool = False,
        generate_quiz: bool = False
    ) -> List[str]:
        """Run the full processing pipeline on an uploaded paper.

        This replicates the logic from Orchestrator._process_paper to ensure
        uploaded papers get the same rich processing as discovered papers.
        """
        actions_triggered = []

        try:
            # Step 5a: Ingest paper (extract full text and metadata)
            status.message = "Extracting text and metadata..."
            status.progress = 40
            logger.info(f"Step 5a: Ingesting paper {paper.id} for pillar {pillar_id}")

            parsed_paper = self.ingest_agent.ingest(paper_ref=paper)
            actions_triggered.append("text_extraction")

            # Step 5b: Summarize paper (if requested)
            if run_summarizer:
                status.message = "Generating summary and notes..."
                status.progress = 60
                logger.info(f"Step 5b: Summarizing {paper.id} for {pillar_id}")

                # Get recent notes for context (following orchestrator pattern)
                recent_notes = db.get_recent_notes(pillar_id, limit=5)

                # Build note context from recent notes
                note_context = [
                    note.problem + " " + note.method
                    for note in recent_notes[-3:]
                ]
                summarizer_input = SummarizerInput(
                    parsed_paper=parsed_paper,
                    pillar_id=pillar_id,
                    recent_notes=note_context
                )

                paper_note = SummarizerAgent.run(summarizer_input)

                # Persist the paper note
                db.insert_note(paper_note)
                actions_triggered.append("summarizer")

                # Step 5c: Synthesize lesson
                status.message = "Synthesizing lesson..."
                status.progress = 70
                logger.info(f"Step 5c: Synthesizing lesson for {paper.id}")

                # Get pillar config
                pillar_config = self._get_pillar_config(pillar_id)

                synthesis_input = SynthesisInput(
                    paper_note=paper_note,
                    pillar_config=pillar_config,
                    related_lessons=[]  # Could get recent lessons from DB
                )

                lesson = SynthesisAgent.run(synthesis_input)

                # Persist the lesson
                db.insert_lesson(lesson)
                actions_triggered.append("lesson_synthesis")

                # Step 5d: Generate quiz (if requested)
                if generate_quiz:
                    status.message = "Generating quiz cards..."
                    status.progress = 85
                    logger.info(f"Step 5d: Generating quiz for {paper.id}")

                    quiz_input = QuizGeneratorInput(
                        paper_note=paper_note,
                        num_questions=5,
                        difficulty_mix={"easy": 2, "medium": 2, "hard": 1}
                    )

                    quiz_cards = QuizAgent.run(quiz_input)

                    # Persist quiz cards
                    if quiz_cards:
                        db.insert_quiz_cards(quiz_cards)

                    actions_triggered.append("quiz_generation")

            # Step 5e: Update paper with enhanced metadata from processing
            status.message = "Persisting to database..."
            status.progress = 92
            logger.info(f"Step 5e: Persisting data for {paper.id}")

            # The ingest agent may have extracted better metadata
            if parsed_paper.paper_ref.authors and not paper.authors:
                paper.authors = parsed_paper.paper_ref.authors
            if parsed_paper.paper_ref.year and not paper.year:
                paper.year = parsed_paper.paper_ref.year
            if parsed_paper.paper_ref.venue and not paper.venue:
                paper.venue = parsed_paper.paper_ref.venue
            if parsed_paper.paper_ref.abstract and not paper.abstract:
                paper.abstract = parsed_paper.paper_ref.abstract

            # Update the paper in the database with enhanced metadata
            db.upsert_paper(pillar_id, paper)

            # Step 5f: Store in vector database for semantic search
            try:
                status.message = "Storing vectors for search..."
                status.progress = 98
                logger.info(f"Step 5f: Upserting vectors for {paper.id}")
                vectors.ensure_collections()
                vectors.upsert_text(pillar_id, paper.id, parsed_paper.full_text)
                actions_triggered.append("vector_storage")
            except Exception as e:
                # Non-fatal - log warning and continue
                logger.warning(f"Failed to store vectors for {paper.id}: {e}")

        except Exception as e:
            logger.error(f"Error in full pipeline processing for {paper.id}: {e}")
            # Don't fail the entire upload, just log the error
            actions_triggered.append(f"pipeline_error: {str(e)}")

        return actions_triggered

    def _enrich_from_arxiv(self, paper: PaperRef, url_or_filename: str) -> PaperRef:
        """Enrich paper metadata from arXiv API if arXiv ID detected."""
        # Extract arXiv ID using regex
        match = re.search(r'(\d{4}\.\d{4,5})', url_or_filename)
        if not match:
            return paper  # Not an arXiv paper

        arxiv_id = match.group(1)
        logger.info(f"Detected arXiv ID: {arxiv_id}, fetching metadata...")

        try:
            # Use id_list parameter for direct lookup (not query)
            search = arxiv.Search(id_list=[arxiv_id])
            client = arxiv.Client()
            result = next(client.results(search))

            # For arXiv papers, API data is authoritative - always use it
            # (PDF text extraction often gets wrong title from headers/footers)
            paper.title = result.title
            paper.authors = [a.name for a in result.authors]
            paper.year = result.published.year
            paper.abstract = result.summary
            paper.venue = result.journal_ref or f"arXiv:{result.primary_category}"

            logger.info(f"Enriched from arXiv: {paper.title[:50]}...")

        except StopIteration:
            logger.warning(f"arXiv paper {arxiv_id} not found")
        except Exception as e:
            logger.warning(f"arXiv enrichment failed: {e}")

        return paper

    def _enrich_from_semantic_scholar(self, paper: PaperRef) -> PaperRef:
        """Enrich paper metadata from Semantic Scholar API.

        This is a best-effort enrichment - failures are logged but don't
        stop the upload process.
        """
        try:
            from ..tools.semantic_scholar_tool import SemanticScholarTool

            s2 = SemanticScholarTool()
            enriched = None

            # Try by arXiv ID first if we have one
            if paper.id and re.match(r'\d{4}\.\d{4,5}', paper.id.replace('arxiv:', '')):
                arxiv_id = paper.id.replace('arxiv:', '')
                logger.info(f"Trying S2 lookup by arXiv ID: {arxiv_id}")
                enriched = s2.get_paper(arxiv_id)  # Tool handles arXiv: prefix

            # Fallback: search by title
            if not enriched and paper.title and len(paper.title) > 10:
                logger.info(f"Trying S2 search by title: {paper.title[:30]}...")
                results = s2.search(paper.title, limit=1)
                if results and self._titles_similar(paper.title, results[0].title):
                    enriched = results[0]

            if enriched:
                # Only fill empty fields (preserve user overrides)
                if not paper.authors and enriched.authors:
                    paper.authors = enriched.authors
                if not paper.year and enriched.year:
                    paper.year = enriched.year
                if not paper.abstract and enriched.abstract:
                    paper.abstract = enriched.abstract
                if enriched.citation_count:
                    paper.citation_count = enriched.citation_count

                logger.info(f"Enriched from S2: citations={paper.citation_count}")

        except Exception as e:
            # Semantic Scholar enrichment is optional - don't fail the upload
            logger.warning(
                f"Semantic Scholar enrichment failed (continuing without): {e}"
            )

        return paper

    def _titles_similar(self, title1: str, title2: str) -> bool:
        """Check if two titles are similar enough to be the same paper."""
        # Simple: lowercase, remove punctuation, check overlap
        clean1 = re.sub(r'[^\w\s]', '', title1.lower())
        clean2 = re.sub(r'[^\w\s]', '', title2.lower())
        words1 = set(clean1.split())
        words2 = set(clean2.split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap > 0.7

    def _get_pillar_config(self, pillar_id: str) -> PillarConfig:
        """Get pillar configuration from database."""
        from ..config import get_pillar_config

        config = get_pillar_config(pillar_id)
        return PillarConfig(
            id=pillar_id,
            name=config['name'],
            goal=config['goal'],
            focus_areas=config.get('focus_areas', []),
            papers_per_day=config.get('papers_per_day', 2)
        )


# Module-level service instance
_upload_service: Optional[UploadService] = None


def get_upload_service() -> UploadService:
    """Get or create the upload service singleton."""
    global _upload_service
    if _upload_service is None:
        _upload_service = UploadService()
    return _upload_service
