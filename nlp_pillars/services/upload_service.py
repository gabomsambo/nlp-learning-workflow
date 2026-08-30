"""Upload Service for Paper Management.

Handles file uploads, URL downloads, and integration with existing
PDF processing and paper management systems.

The two entry points here — :meth:`UploadService.run_url_upload_job` and
:meth:`UploadService.run_file_upload_job` — are SYNCHRONOUS and run on an APScheduler
worker thread, never in the request. They used to be ``async def`` and were awaited
inside the route handler, which held the browser for the whole upload: measured from
the live logs, one 4.71 MB arXiv paper kept ``POST /upload/url`` open for 3 minutes
20 seconds while ``/discover`` and ``/select`` next to it answered in milliseconds.
Nothing in them ever awaited anything meaningful — ``download_pdf``, the arXiv and
Semantic Scholar lookups, the ingest and every agent call are all blocking — so the
``async`` was pure cost.

Progress is reported through an ``on_stage`` callback into ``pipeline_run_stages``,
which is a shared, cross-process record the browser already knows how to poll. What
it replaced was an in-memory ``UploadStatus`` object on a process-local singleton
whose id was only returned once the call it described had finished; nothing could
poll it while it still meant anything, and nothing ever did.
"""

import hashlib
import logging
import os
import re
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional
from urllib.parse import urlparse

from fastapi import UploadFile

from .. import db, vectors
from ..agents.ingest_agent import IngestAgent
from ..agents.quiz_agent import QuizAgent
from ..agents.summarizer_agent import SummarizerAgent
from ..agents.synthesis_agent import SynthesisAgent
from ..db import add_paper
# _first_line is imported rather than copied: it strips tenacity's RetryError
# wrapper, httpx's ' for url …' tail and instructor's dangling tag, and a second
# copy of that knowledge would drift. Same package, same purpose — putting an
# exception on screen as a stage detail.
from ..orchestrator import RunCancelledError, _first_line
from ..schemas import (
    PaperRef,
    PillarConfig,
    QuizGeneratorInput,
    UPLOAD_STAGES,
    StageName,
    StageStatus,
    SummarizerInput,
    SynthesisInput,
    UploadFileRequest,
    UploadUrlRequest,
)
from ..tools.pdf_loader import download_pdf, extract_text
from .paper_metadata import (
    enrich_from_arxiv,
    enrich_from_semantic_scholar,
    extract_arxiv_id_from_hint,
    titles_similar,
)

logger = logging.getLogger(__name__)

#: ``(stage_name, status, detail)`` — the same sink the Orchestrator reports into, so
#: an upload's stages land in the same table and render through the same component.
StageCallback = Callable[[str, str, Optional[str]], None]


class UploadError(Exception):
    """Custom exception for upload operations."""

    pass


class _Stages:
    """Report stage transitions, and check for cancellation at each boundary.

    Three rules. The first two are borrowed from the Orchestrator because both were
    learned there:

    - A callback that raises is logged, never propagated. Losing the progress display
      is bad; losing the upload with it is worse.
    - Cancellation is checked when a stage *starts*, never mid-stage, so a stage always
      finishes the work it began. Nothing can interrupt synchronous Python, so this is
      cooperative by necessity, not by choice.
    - A run that stops early closes out the stages it will never reach. A `pending`
      row on a *finished* run reads as work that is still coming, which is precisely
      the shape a silent failure takes on this page — so `stop_after_current` marks
      them `skipped` with the reason the run stopped.
    """

    def __init__(
        self,
        on_stage: Optional[StageCallback] = None,
        cancel: Optional[threading.Event] = None,
    ):
        self._on_stage = on_stage
        self._cancel = cancel
        #: Stages that have already reached a terminal status. stop_after_current
        #: never touches these — overwriting a completed stage with 'skipped' would
        #: erase work that really happened, and the run's own bookkeeping is the only
        #: thing that knows the difference.
        self._closed: set = set()

    def start(self, stage: StageName, detail: Optional[str] = None) -> None:
        if self._cancel is not None and self._cancel.is_set():
            raise RunCancelledError(f"cancelled before {stage.value}")
        self._emit(stage, StageStatus.RUNNING, detail)

    def stop_after_current(self, reason: str) -> None:
        """Mark every stage that never reached a terminal status ``skipped``.

        Walks UPLOAD_STAGES, so a stage added to that list is closed out here without
        a second edit, and skips the ones already completed, failed or skipped.
        """
        for stage in UPLOAD_STAGES:
            if stage not in self._closed:
                self.skipped(stage, reason)

    def done(self, stage: StageName, detail: Optional[str] = None) -> None:
        self._closed.add(stage)
        self._emit(stage, StageStatus.COMPLETED, detail)

    def failed(self, stage: StageName, detail: str) -> None:
        self._closed.add(stage)
        self._emit(stage, StageStatus.FAILED, detail)

    def skipped(self, stage: StageName, detail: str) -> None:
        self._closed.add(stage)
        self._emit(stage, StageStatus.SKIPPED, detail)

    def _emit(
        self, stage: StageName, status: StageStatus, detail: Optional[str]
    ) -> None:
        if self._on_stage is None:
            return
        try:
            self._on_stage(stage.value, status.value, detail)
        except Exception as e:  # noqa: BLE001 - see the class docstring
            logger.warning(f"Stage callback failed for {stage.value}: {e}")


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


@dataclass
class UploadJobResult:
    """What one finished upload job has to report.

    The paper is separate from the outcome because they are separate facts: reaching
    this object at all means the paper is in the library, and ``outcome`` says what the
    post-upload steps did to it. Collapsing the two is the bug this whole path keeps
    relearning — see PipelineOutcome above.
    """

    paper: PaperRef
    outcome: PipelineOutcome
    source: str


@contextmanager
def _closing_out_on_cancel(stages: "_Stages"):
    """Close out the stages a cancelled upload will never reach.

    Cancellation raises out of ``_Stages.start`` before the stage is marked running,
    so without this the run ends ``cancelled`` above a column of `pending` rows —
    which reads as work that is still coming, on a run that has stopped. Re-raises:
    the run status is the caller's to record.
    """
    try:
        yield
    except RunCancelledError:
        stages.stop_after_current("the upload was cancelled")
        raise


def _reason(e: BaseException) -> str:
    """One readable line for a stage detail, following the exception chain if needed.

    ``_first_line`` strips tenacity's ``RetryError[<Future …>]`` wrapper when the
    RetryError *is* the exception. ``pdf_loader.download_pdf`` instead interpolates it
    into a message of its own — ``PDFDownloadError(f"Failed to download PDF from
    {url}: {e}")`` — so the Future repr survives the trim. Measured on a 404: the panel
    read ``RetryError[<Future at 0x7f29… state=finished raised HTTPStatusError>]``,
    which names no status code and tells the user nothing.

    The real exception is still on ``__cause__``/``__context__`` one hop down, and it
    says ``Client error '404 Not Found'``. Only used when the trimmed line still
    carries the wrapper, so an ordinary message is never second-guessed.
    """
    line = _first_line(e)
    seen = 0
    while _looks_like_a_wrapper(line) and seen < _MAX_CHAIN_HOPS:
        inner = e.__cause__ or e.__context__
        if inner is None:
            break
        e = inner
        seen += 1
        line = _first_line(e) or line
    return line


#: The chain is walked rather than followed once because the repr can be nested twice:
#: pdf_loader's PDFDownloadError message contains tenacity's RetryError repr, whose own
#: __context__ is the RetryError object, whose last_attempt holds the real exception.
#: Bounded so a self-referential chain cannot spin.
_MAX_CHAIN_HOPS = 4


def _looks_like_a_wrapper(line: str) -> bool:
    return "RetryError[" in line or "<Future" in line


def _describe_file(path: str) -> str:
    """Human size for a stage detail — "4.7 MB", not 4938211."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return "unknown size"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


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

    # ------------------------------------------------------------------ jobs
    #
    # Both entry points run on an APScheduler worker thread (see
    # webui/services/run_service.py::_finish_upload). They are deliberately
    # synchronous — see the module docstring — and they raise rather than returning a
    # failure object when the paper never reaches the library, so the run row is
    # marked failed by execute_run's handler and the failure is impossible to miss.
    # What the paper's own post-upload processing did or failed to do is a different
    # fact and comes back in PipelineOutcome.

    def run_url_upload_job(
        self,
        pillar_id: str,
        request: UploadUrlRequest,
        on_stage: Optional[StageCallback] = None,
        cancel: Optional[threading.Event] = None,
    ) -> "UploadJobResult":
        """Download a paper from a URL, add it, and process it.

        Raises:
            UploadError: the PDF could not be fetched, or the paper could not be
                written to the database. Nothing was added in either case.
            RunCancelledError: the run was cancelled at a stage boundary.

        """
        logger.info(f"URL upload for {pillar_id}: {request.url}")
        stages = _Stages(on_stage, cancel)
        with _closing_out_on_cancel(stages):
            return self._run_url_upload(pillar_id, request, stages, on_stage, cancel)

    def _run_url_upload(
        self,
        pillar_id: str,
        request: UploadUrlRequest,
        stages: "_Stages",
        on_stage: Optional[StageCallback],
        cancel: Optional[threading.Event],
    ) -> "UploadJobResult":
        stages.start(StageName.UPLOAD_FETCH, f"Downloading {request.url}")
        try:
            pdf_path = download_pdf(request.url, str(self.upload_dir / "downloads"))
        except Exception as e:
            # Broad on purpose. download_pdf raises PDFDownloadError for the cases it
            # anticipates, but a 4xx page served as a PDF, a DNS failure or a disk
            # error arrive as whatever the underlying library raises, and every one of
            # them means the same thing here: there is no PDF, so there is nothing to
            # add. The reason reaches the stage row and the run row either way.
            stages.failed(StageName.UPLOAD_FETCH, _reason(e))
            stages.stop_after_current("the PDF could not be fetched")
            raise UploadError(f"Could not download the PDF: {_reason(e)}") from e
        stages.done(
            StageName.UPLOAD_FETCH,
            f"{_describe_file(pdf_path)} from {urlparse(request.url).netloc}",
        )

        paper = self._add_uploaded_paper(
            pillar_id,
            stages,
            lambda: self._create_paper_ref_from_url(
                request.url, pdf_path, request.title, request.authors
            ),
        )

        outcome = self._run_full_pipeline(
            paper=paper,
            pillar_id=pillar_id,
            run_summarizer=request.run_summarizer,
            generate_quiz=request.generate_quiz,
            stages=stages,
        )
        return UploadJobResult(
            paper=paper, outcome=outcome, source=f"URL: {request.url}"
        )

    def run_file_upload_job(
        self,
        pillar_id: str,
        saved_path: str,
        filename: str,
        request: UploadFileRequest,
        on_stage: Optional[StageCallback] = None,
        cancel: Optional[threading.Event] = None,
    ) -> "UploadJobResult":
        """Process a PDF the route has already written to disk.

        ``saved_path`` rather than an ``UploadFile``: the request's file handle cannot
        cross into the worker thread — Starlette closes the spooled temporary file
        when the response is sent, and this job starts after that. The route therefore
        does the one part of an upload that genuinely belongs in the request (reading
        the bytes it was given) and hands over a path.

        The UPLOAD_FETCH stage still exists on this path and is honest about what it
        checked: the fetch happened in the request, and this verifies the result is
        still there. A vanished spool file would otherwise surface three stages later
        as an unreadable PDF.
        """
        logger.info(f"File upload for {pillar_id}: {filename}")
        stages = _Stages(on_stage, cancel)
        with _closing_out_on_cancel(stages):
            return self._run_file_upload(
                pillar_id, saved_path, filename, request, stages, on_stage, cancel
            )

    def _run_file_upload(
        self,
        pillar_id: str,
        saved_path: str,
        filename: str,
        request: UploadFileRequest,
        stages: "_Stages",
        on_stage: Optional[StageCallback],
        cancel: Optional[threading.Event],
    ) -> "UploadJobResult":
        stages.start(StageName.UPLOAD_FETCH, f"Reading {filename}")
        if not os.path.exists(saved_path):
            detail = "the file saved with the request is no longer on disk"
            stages.failed(StageName.UPLOAD_FETCH, detail)
            stages.stop_after_current(detail)
            raise UploadError(f"{detail}: {saved_path}")
        stages.done(
            StageName.UPLOAD_FETCH,
            f"{filename} — {_describe_file(saved_path)}, saved with the request",
        )

        # The uploaded PDF is deliberately RETAINED once the paper row exists: the row
        # holds url_pdf = file://<saved_path> and podcast full-text extraction
        # dereferences it, so deleting the file leaves every file-uploaded paper
        # pointing at nothing. Only a failure that never reached the database drops it,
        # because in that case nothing refers to it. No retention policy is implied —
        # retained PDFs accumulate at ~1-5 MB/paper and pruning them is a separate,
        # deliberate decision.
        persisted = False
        try:
            paper = self._add_uploaded_paper(
                pillar_id,
                stages,
                lambda: self._create_paper_ref_from_file(
                    saved_path, filename, request
                ),
            )
            persisted = True
        finally:
            if not persisted and os.path.exists(saved_path):
                try:
                    os.unlink(saved_path)
                    logger.info(f"Discarded upload nothing refers to: {saved_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up file {saved_path}: {e}")

        outcome = self._run_full_pipeline(
            paper=paper,
            pillar_id=pillar_id,
            run_summarizer=request.run_summarizer,
            generate_quiz=request.generate_quiz,
            stages=stages,
        )
        return UploadJobResult(
            paper=paper, outcome=outcome, source=f"file: {filename}"
        )

    def _add_uploaded_paper(
        self, pillar_id: str, stages: _Stages, build_paper
    ) -> PaperRef:
        """Resolve the paper's metadata and write its row. Shared by both paths.

        This is UPLOAD_METADATA, and the ``add_paper`` call belongs inside it rather
        than in a stage of its own: the arXiv/S2 lookup and the insert are one
        indivisible outcome for the user — either the library now knows about this
        paper or it does not — and a separate "saving" step that can only fail when the
        database is already unreachable adds a row to the panel and no information.
        """
        stages.start(
            StageName.UPLOAD_METADATA, "Looking up title, authors and abstract"
        )
        try:
            paper = build_paper()
        except Exception as e:
            stages.failed(StageName.UPLOAD_METADATA, _reason(e))
            stages.stop_after_current("the paper's details could not be read")
            raise UploadError(
                f"Could not read the paper's details: {_reason(e)}"
            ) from e

        if not add_paper(pillar_id, paper):
            detail = "the paper could not be written to the papers table"
            stages.failed(StageName.UPLOAD_METADATA, detail)
            stages.stop_after_current(detail)
            raise UploadError(f"Failed to add paper to database: {detail}")

        stages.done(
            StageName.UPLOAD_METADATA, f'Added "{paper.title}" ({paper.id})'
        )
        return paper

    async def save_uploaded_file(self, file: UploadFile) -> str:
        """Save an uploaded file to the retained upload directory.

        The returned path is what ends up in ``papers.url_pdf`` as a ``file://``
        URL, so it must remain valid for the life of the paper.

        Called from the REQUEST, not from the worker: this is the one part of a file
        upload that cannot be deferred, because Starlette closes the spooled temporary
        file behind ``UploadFile`` when the response is sent. The route hands the
        worker this path instead. It is also the only ``async`` left on this service —
        ``file.read()`` genuinely is awaitable.
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

    def _create_paper_ref_from_url(
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
        arxiv_id = extract_arxiv_id_from_hint(url)

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
        paper = enrich_from_arxiv(paper, url)

        # The arXiv lookup was supposed to supply the title and did not (not found,
        # or the API was unreachable). Pay for the parse now rather than record the
        # hostname as the paper's title forever.
        if not title_override and paper.title == default_title:
            paper.title = self._guess_title_from_pdf(pdf_path, default_title)

        # Fallback to Semantic Scholar if still missing metadata
        if not paper.authors or not paper.year or not paper.abstract:
            paper = enrich_from_semantic_scholar(paper)

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

    def _create_paper_ref_from_file(
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
            paper = enrich_from_arxiv(paper, filename)

        # Fallback to Semantic Scholar if title is provided but missing other metadata
        if paper.title and (not paper.authors or not paper.year or not paper.abstract):
            paper = enrich_from_semantic_scholar(paper)

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


    def _run_full_pipeline(
        self,
        paper: PaperRef,
        pillar_id: str,
        run_summarizer: bool = True,
        generate_quiz: bool = True,
        on_stage: Optional[StageCallback] = None,
        cancel: Optional[threading.Event] = None,
        stages: Optional["_Stages"] = None,
    ) -> "PipelineOutcome":
        """Run the full processing pipeline on an uploaded paper.

        This replicates the logic from Orchestrator._process_paper to ensure
        uploaded papers get the same rich processing as discovered papers, and reports
        into the same six stage names so the progress panel reads identically for an
        uploaded paper and a discovered one.

        Returns a PipelineOutcome rather than a bare list of action names. The list
        alone could not distinguish "did the work" from "tried and failed", and the
        previous version resolved that by appending ``pipeline_error: {e}`` to it as
        though a crash were a fifth kind of action. Whatever went wrong here does not
        remove the paper — it is already in the database, and deleting it would be
        worse — so the caller reports the two facts separately.

        Every stage this does not run is marked ``skipped`` WITH ITS REASON rather
        than left pending. A pending row on a finished run is the shape a silent
        failure takes on this page: it looks like work that is still coming.
        """
        actions_triggered: List[str] = []
        errors: List[str] = []
        # The caller's reporter when there is one, so a stage this half of the run
        # already closed is not re-marked by stop_after_current. `on_stage`/`cancel`
        # remain for the tests and any caller that only has a callback.
        stages = stages or _Stages(on_stage, cancel)

        # Requested but impossible: the quiz is generated from the summarizer's
        # PaperNote, so there is nothing to build it from without one. This used to
        # be a silent no-op — the quiz block sat inside `if run_summarizer:`, so
        # asking for cards without a summary produced none and said nothing. The
        # constraint is real; being quiet about it is what was wrong.
        quiz_off_reason = "'Generate Quiz Cards' was off for this upload"
        if generate_quiz and not run_summarizer:
            quiz_off_reason = (
                "quiz cards are generated from the summarizer's notes and "
                "'Run Summarizer' was off"
            )
            errors.append(f"quiz_generation: skipped because {quiz_off_reason}")
            generate_quiz = False

        # Step 5a: Ingest paper (extract full text and metadata)
        stages.start(StageName.INGEST, f"Reading the PDF for {paper.id}")
        logger.info(f"Step 5a: Ingesting paper {paper.id} for pillar {pillar_id}")
        try:
            parsed_paper = self.ingest_agent.ingest(paper_ref=paper)
        except Exception as e:
            # Everything downstream reads parsed_paper, so this one is terminal for
            # the pipeline. The paper row stays; the user is told it is bare.
            logger.error(f"Text extraction failed for {paper.id}: {e}")
            errors.append(f"text_extraction: {e}")
            stages.failed(StageName.INGEST, _reason(e))
            stages.stop_after_current("no text was extracted from the PDF")
            return PipelineOutcome(actions_triggered=actions_triggered, errors=errors)

        actions_triggered.append("text_extraction")
        stages.done(
            StageName.INGEST, f"{len(parsed_paper.full_text):,} characters extracted"
        )

        paper_note = None
        if not run_summarizer:
            stages.skipped(
                StageName.SUMMARIZE, "'Run Summarizer' was off for this upload"
            )
        else:
            # Step 5b: Summarize paper
            stages.start(StageName.SUMMARIZE, "Generating structured notes")
            logger.info(f"Step 5b: Summarizing {paper.id} for {pillar_id}")
            try:
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
                stages.done(
                    StageName.SUMMARIZE,
                    f"{len(paper_note.findings)} finding(s), "
                    f"{len(paper_note.key_terms)} key term(s)",
                )
            except Exception as e:
                logger.error(f"Summarizer failed for {paper.id}: {e}")
                errors.append(f"summarizer: {e}")
                stages.failed(StageName.SUMMARIZE, _reason(e))

        if paper_note is None:
            stages.skipped(
                StageName.SYNTHESIZE,
                "the lesson is written from the summarizer's notes, and there are none",
            )
        else:
            # Step 5c: Synthesize lesson
            stages.start(StageName.SYNTHESIZE, "Writing the lesson")
            logger.info(f"Step 5c: Synthesizing lesson for {paper.id}")
            try:
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
                stages.done(StageName.SYNTHESIZE, lesson.title)
            except Exception as e:
                # A failed lesson must not take the quiz down with it: both are built
                # from the note, not from each other.
                logger.error(f"Lesson synthesis failed for {paper.id}: {e}")
                errors.append(f"lesson_synthesis: {e}")
                stages.failed(StageName.SYNTHESIZE, _reason(e))

        if not generate_quiz:
            stages.skipped(StageName.QUIZ, quiz_off_reason)
        elif paper_note is None:
            errors.append(
                "quiz_generation: skipped because the summarizer produced no notes"
            )
            stages.skipped(
                StageName.QUIZ, "the summarizer produced no notes to build cards from"
            )
        else:
            # Step 5d: Generate quiz
            stages.start(StageName.QUIZ, "Building quiz cards")
            logger.info(f"Step 5d: Generating quiz for {paper.id}")
            try:
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
                    stages.done(StageName.QUIZ, f"{len(quiz_cards)} card(s)")
                else:
                    errors.append("quiz_generation: the model returned no cards")
                    stages.failed(StageName.QUIZ, "the model returned no cards")
            except Exception as e:
                logger.error(f"Quiz generation failed for {paper.id}: {e}")
                errors.append(f"quiz_generation: {e}")
                stages.failed(StageName.QUIZ, _reason(e))

        # Step 5e: Update paper with enhanced metadata from processing
        stages.start(StageName.PERSIST, "Saving the paper's metadata")
        logger.info(f"Step 5e: Persisting data for {paper.id}")
        try:
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
            stages.done(StageName.PERSIST, f"{len(paper.authors)} author(s) recorded")
        except Exception as e:
            logger.error(f"Metadata persistence failed for {paper.id}: {e}")
            errors.append(f"metadata_persistence: {e}")
            stages.failed(StageName.PERSIST, _reason(e))

        # Step 5f: Store in vector database for semantic search
        stages.start(StageName.VECTORS, "Embedding the paper for semantic search")
        logger.info(f"Step 5f: Upserting vectors for {paper.id}")
        try:
            vectors.ensure_collections()
            chunks = vectors.upsert_text(pillar_id, paper.id, parsed_paper.full_text)
            if chunks:
                actions_triggered.append("vector_storage")
                stages.done(StageName.VECTORS, f"{chunks} chunk(s) indexed")
            elif parsed_paper.full_text and parsed_paper.full_text.strip():
                # upsert_text returns 0 for an empty document and for a dead Qdrant
                # alike, and it swallows its own exceptions to get there. Non-empty
                # text in and nothing out is the second case. Same call the
                # orchestrator makes and the same conclusion it draws — see
                # Orchestrator._process_paper.
                detail = (
                    "no chunks were written for a non-empty paper; it will not "
                    "appear in semantic search"
                )
                errors.append(f"vector_storage: {detail}")
                stages.failed(StageName.VECTORS, detail)
            else:
                stages.skipped(
                    StageName.VECTORS, "the paper has no text to index"
                )
        except Exception as e:
            logger.error(f"Vector storage failed for {paper.id}: {e}")
            errors.append(f"vector_storage: {e}")
            stages.failed(StageName.VECTORS, _reason(e))

        return PipelineOutcome(actions_triggered=actions_triggered, errors=errors)

    @staticmethod
    def _extract_arxiv_id(url_or_filename: str) -> Optional[str]:
        return extract_arxiv_id_from_hint(url_or_filename)

    def _enrich_from_arxiv(self, paper: PaperRef, url_or_filename: str) -> PaperRef:
        return enrich_from_arxiv(paper, url_or_filename)

    def _enrich_from_semantic_scholar(self, paper: PaperRef) -> PaperRef:
        return enrich_from_semantic_scholar(paper)

    def _titles_similar(self, title1: str, title2: str) -> bool:
        return titles_similar(title1, title2)

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
