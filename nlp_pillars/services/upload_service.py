"""Upload Service for Paper Management.

Handles file uploads, URL downloads, and integration with existing
PDF processing and paper management systems.
"""

import hashlib
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
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


@dataclass
class PipelineOutcome:
    """What the post-upload pipeline actually did, and what it failed to do.

    Two lists, never one. The old return value was a single list of action names
    onto which a crash was appended as ``pipeline_error: {e}``, which meant the only
    way to tell success from failure was to string-match the entries — and nothing
    did, so the route answered ``success=True`` either way.
    """

    actions_triggered: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class UploadService:
    """Service for handling paper uploads and processing."""

    def __init__(self, upload_dir: Optional[str] = None):
        """Initialize the upload service with a directory for uploads.

        Uploaded PDFs are *retained*: ``_create_paper_ref_from_file`` stores
        ``url_pdf = file://<abs path>`` and podcast generation dereferences that
        path later, so the file is the paper's only copy for file-sourced
        papers. The default therefore lives under ``data/`` rather than
        ``.cache/`` — it is not regenerable — and docker-compose backs
        ``/app/data`` with the ``nlp_uploads`` named volume so it survives
        `docker compose down` and image rebuilds. Override with UPLOAD_DIR.
        """
        upload_dir = upload_dir or os.environ.get("UPLOAD_DIR", "data/uploads")
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
            outcome = await self._run_full_pipeline(
                paper=paper,
                pillar_id=pillar_id,
                status=status,
                run_summarizer=request.run_summarizer,
                generate_quiz=request.generate_quiz
            )

            return self._upload_response(
                paper=paper,
                status=status,
                outcome=outcome,
                source_description=f"from URL: {request.url}",
                added_message="Paper uploaded successfully from URL",
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

        saved_path = None
        paper_persisted = False
        try:
            # Step 1: Save uploaded file
            status.progress = 20
            saved_path = await self._save_uploaded_file(file)

            # Step 2: Create paper reference
            status.progress = 40
            status.message = "Creating paper reference..."
            paper = await self._create_paper_ref_from_file(
                saved_path, file.filename, request
            )

            # Step 3: Add to database
            status.progress = 60
            status.message = "Saving to database..."

            # Add paper using string pillar_id directly
            success = add_paper(pillar_id, paper)
            if not success:
                raise UploadError("Failed to add paper to database")

            # From here on the papers row holds url_pdf = file://<saved_path>,
            # so the file must outlive this request (see the finally block).
            paper_persisted = True

            # Step 4: Run full pipeline processing
            status.progress = 60
            status.message = "Processing paper through pipeline..."

            # Always run text extraction, conditionally run summarizer/quiz
            outcome = await self._run_full_pipeline(
                paper=paper,
                pillar_id=pillar_id,
                status=status,
                run_summarizer=request.run_summarizer,
                generate_quiz=request.generate_quiz
            )

            return self._upload_response(
                paper=paper,
                status=status,
                outcome=outcome,
                source_description=f"from file: {file.filename}",
                added_message="Paper uploaded successfully from file",
            )

        except Exception as e:
            status.status = "failed"
            status.message = f"Upload failed: {str(e)}"
            logger.error(f"Upload {upload_id} failed: {e}")
            raise UploadError(f"Failed to upload file: {e}") from e

        finally:
            # The uploaded PDF is deliberately RETAINED once the paper row
            # exists. It used to be deleted here unconditionally, which left
            # every file-uploaded paper pointing at url_pdf = file://<gone>;
            # podcast full-text extraction then silently fell back to the
            # abstract. Only a failure that never reached the database drops
            # the file, because in that case nothing refers to it.
            #
            # No retention/cleanup policy is implied — retained PDFs currently
            # accumulate (~1-5 MB per paper) and pruning them is a separate,
            # deliberate decision.
            if not paper_persisted and saved_path and os.path.exists(saved_path):
                try:
                    os.unlink(saved_path)
                    logger.info(f"Discarded upload for failed request: {saved_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up file {saved_path}: {e}")

    @staticmethod
    def _upload_response(
        paper: PaperRef,
        status: UploadStatus,
        outcome: "PipelineOutcome",
        source_description: str,
        added_message: str,
    ) -> UploadResponse:
        """Turn a finished pipeline into a response that says what happened.

        The paper is in the library either way — that is what ``success`` reports,
        and it is why a failed pipeline is not raised as an upload error. What the
        pipeline did or failed to do is reported alongside it, so the page can say
        "added, but the summary failed" instead of the "uploaded successfully!
        Triggered: pipeline_error: ..." it used to print.
        """
        status.progress = 100

        if outcome.ok:
            status.status = "completed"
            status.message = "Upload completed successfully"
            logger.info(f"Uploaded {paper.id} {source_description}")
            message = added_message
        else:
            # Not "failed": the paper was added. Anything that reads this back must
            # not conclude the upload has to be retried from scratch.
            status.status = "completed_with_errors"
            status.message = "Paper added, but post-upload processing failed"
            logger.error(
                f"Uploaded {paper.id} {source_description}, but the pipeline "
                f"reported: {'; '.join(outcome.errors)}"
            )
            message = (
                "Paper added to the library, but post-upload processing did not "
                "finish"
            )

        return UploadResponse(
            success=True,
            paper=paper,
            message=message,
            actions_triggered=outcome.actions_triggered,
            pipeline_ok=outcome.ok,
            pipeline_errors=outcome.errors,
        )

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
        """Save an uploaded file to the retained upload directory.

        The returned path is what ends up in ``papers.url_pdf`` as a ``file://``
        URL, so it must remain valid for the life of the paper.
        """
        # Generate unique filename
        file_id = f"{file.filename}_{uuid.uuid4()}"
        file_hash = hashlib.sha256(file_id.encode()).hexdigest()
        saved_path = self.upload_dir / f"{file_hash}.pdf"

        # Save file
        with open(saved_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Saved uploaded file to {saved_path}")
        return str(saved_path)

    async def _create_paper_ref_from_url(
        self,
        url: str,
        pdf_path: str,
        title_override: Optional[str],
        authors_override: Optional[List[str]]
    ) -> PaperRef:
        """Create PaperRef from URL with automatic metadata enrichment.

        The PDF is parsed here only when nothing else can name the paper. It used
        to be parsed unconditionally, to guess a title from the first plausible
        line — and then ``_enrich_from_arxiv`` overwrote that title (along with the
        authors, year, abstract and venue) a few lines later, because for an arXiv
        paper the API is authoritative. The guess was thrown away every time, and
        ``_run_full_pipeline`` then parsed the same PDF a second time for the real
        ingest. Measured on a 4.71 MB arXiv PDF inside the container: 8.4 seconds
        per upload, spent to produce a string that was immediately discarded.

        So an arXiv id in the URL skips the guess. The fallback is deliberately
        kept for everything else, and kept *before* the Semantic Scholar lookup:
        S2 searches by title when it has no id to look up, so handing it the
        ``Paper from <host>`` placeholder instead of the extracted title would
        trade 8.4 seconds for worse metadata on exactly the non-arXiv papers that
        need enrichment most. It also still runs when an arXiv lookup was expected
        but failed (paper withdrawn, API down), which is the case that would
        otherwise leave a paper permanently titled after its hostname.
        """
        # Generate paper ID from URL
        paper_id = self._generate_paper_id_from_url(url)

        default_title = f"Paper from {urlparse(url).netloc}"
        arxiv_id = self._extract_arxiv_id(url)

        title = title_override
        if not title and not arxiv_id:
            title = self._guess_title_from_pdf(pdf_path, default_title)

        # Use provided authors or empty list
        authors = authors_override or []

        paper = PaperRef(
            id=paper_id,
            title=title or default_title,
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

        # The arXiv lookup was supposed to supply the title and did not (not found,
        # or the API was unreachable). Pay for the parse now rather than record the
        # hostname as the paper's title forever.
        if not title_override and paper.title == default_title:
            paper.title = self._guess_title_from_pdf(pdf_path, default_title)

        # Fallback to Semantic Scholar if still missing metadata
        if not paper.authors or not paper.year or not paper.abstract:
            paper = self._enrich_from_semantic_scholar(paper)

        return paper

    def _guess_title_from_pdf(self, pdf_path: str, default_title: str) -> str:
        """Best-effort title from the PDF body. Costs a full extraction — call it
        only when no metadata source can name the paper (see the caller)."""
        try:
            text = extract_text(pdf_path)
            return self._extract_title_from_text(text) or default_title
        except Exception as e:
            logger.warning(f"Failed to extract title from PDF: {e}")
            return default_title

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
        run_summarizer: bool = True,
        generate_quiz: bool = True
    ) -> "PipelineOutcome":
        """Run the full processing pipeline on an uploaded paper.

        This replicates the logic from Orchestrator._process_paper to ensure
        uploaded papers get the same rich processing as discovered papers.

        Returns a PipelineOutcome rather than a bare list of action names. The list
        alone could not distinguish "did the work" from "tried and failed", and the
        previous version resolved that by appending ``pipeline_error: {e}`` to it as
        though a crash were a fifth kind of action. Whatever went wrong here does not
        remove the paper — it is already in the database, and deleting it would be
        worse — so the caller reports the two facts separately.
        """
        actions_triggered: List[str] = []
        errors: List[str] = []

        # Requested but impossible: the quiz is generated from the summarizer's
        # PaperNote, so there is nothing to build it from without one. This used to
        # be a silent no-op — the quiz block sat inside `if run_summarizer:`, so
        # asking for cards without a summary produced none and said nothing. The
        # constraint is real; being quiet about it is what was wrong.
        if generate_quiz and not run_summarizer:
            errors.append(
                "quiz_generation: skipped because quiz cards are generated from the "
                "summarizer's notes and 'Run Summarizer' was off"
            )
            generate_quiz = False

        parsed_paper = None
        try:
            # Step 5a: Ingest paper (extract full text and metadata)
            status.message = "Extracting text and metadata..."
            status.progress = 40
            logger.info(f"Step 5a: Ingesting paper {paper.id} for pillar {pillar_id}")

            parsed_paper = self.ingest_agent.ingest(paper_ref=paper)
            actions_triggered.append("text_extraction")
        except Exception as e:
            # Everything downstream reads parsed_paper, so this one is terminal for
            # the pipeline. The paper row stays; the user is told it is bare.
            logger.error(f"Text extraction failed for {paper.id}: {e}")
            errors.append(f"text_extraction: {e}")
            return PipelineOutcome(actions_triggered=actions_triggered, errors=errors)

        paper_note = None
        if run_summarizer:
            try:
                # Step 5b: Summarize paper
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
            except Exception as e:
                logger.error(f"Summarizer failed for {paper.id}: {e}")
                errors.append(f"summarizer: {e}")

        if paper_note is not None:
            try:
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
            except Exception as e:
                # A failed lesson must not take the quiz down with it: both are built
                # from the note, not from each other.
                logger.error(f"Lesson synthesis failed for {paper.id}: {e}")
                errors.append(f"lesson_synthesis: {e}")

        if generate_quiz:
            if paper_note is None:
                errors.append(
                    "quiz_generation: skipped because the summarizer produced no notes"
                )
            else:
                try:
                    # Step 5d: Generate quiz
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
                    else:
                        errors.append("quiz_generation: the model returned no cards")
                except Exception as e:
                    logger.error(f"Quiz generation failed for {paper.id}: {e}")
                    errors.append(f"quiz_generation: {e}")

        try:
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
        except Exception as e:
            logger.error(f"Metadata persistence failed for {paper.id}: {e}")
            errors.append(f"metadata_persistence: {e}")

        # Step 5f: Store in vector database for semantic search
        try:
            status.message = "Storing vectors for search..."
            status.progress = 98
            logger.info(f"Step 5f: Upserting vectors for {paper.id}")
            vectors.ensure_collections()
            chunks = vectors.upsert_text(pillar_id, paper.id, parsed_paper.full_text)
            if chunks:
                actions_triggered.append("vector_storage")
            elif parsed_paper.full_text and parsed_paper.full_text.strip():
                # upsert_text returns 0 for an empty document and for a dead Qdrant
                # alike, and it swallows its own exceptions to get there. Non-empty
                # text in and nothing out is the second case. Same call the
                # orchestrator makes and the same conclusion it draws — see
                # Orchestrator._process_paper.
                errors.append(
                    "vector_storage: no chunks were written for a non-empty paper; "
                    "the paper will not appear in semantic search"
                )
        except Exception as e:
            logger.error(f"Vector storage failed for {paper.id}: {e}")
            errors.append(f"vector_storage: {e}")

        return PipelineOutcome(actions_triggered=actions_triggered, errors=errors)

    @staticmethod
    def _extract_arxiv_id(url_or_filename: str) -> Optional[str]:
        """The arXiv id in a URL or filename, or None.

        One definition, used both by the enrichment below and by the caller that
        decides whether a PDF parse is worth doing. Two copies of this regex would
        drift, and the failure would be silent in the expensive direction: a URL the
        skip recognised but the enrichment did not would leave the paper with no
        title and no parse to recover one from.
        """
        match = re.search(r'(\d{4}\.\d{4,5})', url_or_filename)
        return match.group(1) if match else None

    def _enrich_from_arxiv(self, paper: PaperRef, url_or_filename: str) -> PaperRef:
        """Enrich paper metadata from arXiv API if arXiv ID detected."""
        arxiv_id = self._extract_arxiv_id(url_or_filename)
        if not arxiv_id:
            return paper  # Not an arXiv paper

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
