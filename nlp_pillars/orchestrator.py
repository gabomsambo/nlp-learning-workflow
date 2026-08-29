"""
End-to-end orchestrator for running the daily pipeline for a pillar.

Composes existing agents, tools, DAO, and vector operations to process papers
through the complete learning workflow with strict pillar isolation.
"""

import logging
import re
import threading
import time
from typing import Callable, List, NamedTuple, Optional, Tuple

from .schemas import (
    PillarConfig, PipelineResult,
    PaperRef, Lesson, QuizCard, SearchQuery,
    DiscoveryInput, SummarizerInput, SynthesisInput, QuizGeneratorInput,
    DiscoveryCandidate, PaperCitation,
    StageName, StageStatus
)
from .agents.discovery_agent import DiscoveryAgent
from .agents.ingest_agent import IngestAgent
from .agents.summarizer_agent import SummarizerAgent
from .agents.synthesis_agent import SynthesisAgent
from .agents.quiz_agent import QuizAgent
from .tools.searxng_tool import SearXNGTool
from .tools.arxiv_tool import ArXivTool
from .tools.semantic_scholar_tool import SemanticScholarTool
from .tools.vector_search_tool import VectorSearchTool
from .config import get_settings, get_pillar_config
from .paper_ids import resolvable_pdf_url
from . import db
from . import vectors

logger = logging.getLogger(__name__)


class RunCancelledError(Exception):
    """Raised at a stage boundary when the run's cancel event has been set.

    Distinct from a failure: the run did what it was told, it was just told to stop.
    Callers should record it as 'cancelled', not 'failed'.
    """


#: The six stages that run once per paper, inside _process_paper. Used to work out
#: which stage to blame when a paper raises — _process_paper has no try/except of its
#: own, so the surrounding handler needs to be told.
_PER_PAPER_STAGES = frozenset({
    StageName.INGEST,
    StageName.SUMMARIZE,
    StageName.SYNTHESIZE,
    StageName.QUIZ,
    StageName.PERSIST,
    StageName.VECTORS,
})


class SourceResult(NamedTuple):
    """What one discovery source produced, and how it failed if it did.

    A bare list could not answer the only question the user actually has when a
    source returns nothing: was there nothing to find, or did the search not happen?
    ``failures`` holds plain-English reasons, already fit to put on screen.
    """

    candidates: List[DiscoveryCandidate]
    #: Plain-English reasons, empty when the source answered cleanly. No default:
    #: every construction site has to decide what it is claiming.
    failures: List[str]


def _unwrap_retry(e: Exception) -> Exception:
    """Follow a tenacity RetryError down to the failure it is actually reporting.

    Measured live against Semantic Scholar: three 403s from the API arrived on screen
    as ``RetryError[<Future at 0x74f0… state=finished raised HTTPStatusError>]`` — a
    repr of tenacity's own bookkeeping, with the status code that tells the user what
    happened nowhere in it. The real exception is on ``last_attempt``.

    Duck-typed rather than importing tenacity: this is a display helper, and it should
    not care which retry library a search tool happens to use.
    """
    attempt = getattr(e, "last_attempt", None)
    if attempt is None or not hasattr(attempt, "exception"):
        return e
    try:
        inner = attempt.exception()
    except Exception:
        return e
    return inner if isinstance(inner, BaseException) and inner is not e else e


def _first_line(e: Exception) -> str:
    """One short, printable line for an exception, for a stage detail."""
    e = _unwrap_retry(e)
    text = str(e).strip().splitlines()
    line = text[0] if text else ""
    # httpx puts the whole request URL — fields, limits, encoded query and all — after
    # " for url ". The status is the part a reader can act on; the URL is 300
    # characters that push it off the row.
    line = line.split(" for url ")[0].strip()
    # instructor's retry exception opens an XML-ish block on the first line
    # ("Instructor completion failed: <failed_attempts>") and puts the attempts
    # themselves on the lines we just dropped. Left in, that dangling opening tag
    # reads as truncated output rather than as a reason.
    line = re.sub(r"\s*<[^<>\s]*>\s*$", "", line).strip().rstrip(":").strip()
    return (line or e.__class__.__name__)[:160]


def _friendly_source_error(e: Exception) -> str:
    """Say what a failed search means, in the words the user needs.

    Rate limiting is called out by name because it is the common one and the only one
    with an obvious remedy: arXiv throttles a burst of queries, and SearXNG responds
    by suspending the engine for a full hour (AGENTS.md, "SearXNG serves JSON only
    because settings.yml says so"). Both were previously indistinguishable from a
    query that legitimately matched nothing.
    """
    lowered = str(_unwrap_retry(e)).lower()
    if any(t in lowered for t in
           ("429", "too many requests", "rate limit", "rate-limit", "suspended")):
        return "rate-limited — try again in a few minutes"
    return _first_line(e)


def _quote_queries(queries: List[str], limit: int = 6) -> str:
    """Render search queries for display: quoted, comma-free, bounded."""
    shown = [f'"{q}"' for q in queries[:limit] if q]
    text = " · ".join(shown)
    if len(queries) > limit:
        text += f" (+{len(queries) - limit} more)"
    return text or "no queries"


#: How much of a stage detail to keep. The column is unbounded TEXT, but this is read
#: as one line next to a step name, and a wall of exception text there is not a
#: progress display.
_MAX_STAGE_DETAIL = 300


def _summarise_query_failures(failures: List[str], attempted: List[str]) -> List[str]:
    """Collapse per-query failures into one line that keeps the proportion.

    "1 of 2 queries failed" and "2 of 2 queries failed" mean very different things to
    someone looking at a low result count, and both were previously invisible.
    """
    if not failures:
        return []
    return [f"{len(failures)} of {len(attempted)} queries failed: {failures[0]}"]


def _no_op_stage(name: str, status: str, detail: Optional[str] = None) -> None:
    """Default progress sink. Deliberately does nothing.

    A real function rather than ``None`` so every call site can invoke it
    unconditionally — no ``if self._on_stage is not None`` scattered through the
    pipeline, and no chance of one being forgotten.
    """


class Orchestrator:
    """
    End-to-end orchestrator for running the daily learning pipeline.
    
    Composes discovery, search, ingestion, summarization, synthesis, and quiz generation
    agents to process papers through the complete learning workflow.
    """

    def __init__(
        self,
        enable_quiz: bool = True,
        on_stage: Optional[Callable[[str, str, Optional[str]], None]] = None,
        cancel: Optional[threading.Event] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            enable_quiz: Whether to generate quiz cards during processing
            on_stage: Optional progress sink, called ``on_stage(name, status, detail)``
                as each of the eleven pipeline stages starts and finishes. Defaults to
                a no-op, so every existing caller — the CLI, the scheduler, the tests —
                behaves exactly as it did before. Keeping the sink out here rather than
                writing to a database from inside the pipeline is what lets this module
                stay free of any web or job concern.
            cancel: Optional event checked when each stage *starts*. When set, the run
                raises RunCancelledError at the next boundary. Checked on entry
                rather than mid-stage on purpose: a stage always completes the
                work it began, so the database is never left half-written. Nothing
                can interrupt synchronous Python once it is inside a call, so this
                is cooperative by necessity, not by preference.
        """
        self.enable_quiz = enable_quiz
        self._on_stage = on_stage or _no_op_stage
        self._cancel = cancel
        self._current_paper_stage: Optional[StageName] = None
        #: Failures in the shared plumbing (search / enqueue / pop), which each degrade
        #: to an empty result rather than aborting the run. Kept apart from per-paper
        #: errors so `papers_failed` stays a count of papers. See _record_infra_error.
        self._infra_errors: List[dict] = []

        # Initialize tools
        self.searxng_tool = SearXNGTool()
        self.arxiv_tool = ArXivTool()
        self.ingest_agent = IngestAgent()

        # Initialize enhanced discovery tools
        try:
            self.semantic_scholar_tool = SemanticScholarTool()
        except Exception as e:
            logger.warning(f"Could not initialize Semantic Scholar tool: {e}")
            self.semantic_scholar_tool = None

        try:
            self.vector_search_tool = VectorSearchTool()
        except Exception as e:
            logger.warning(f"Could not initialize vector search tool: {e}")
            self.vector_search_tool = None

        logger.info(f"Orchestrator initialized with enable_quiz={enable_quiz}")

    def run_daily(self, pillar_id: str, papers_limit: int = 1) -> PipelineResult:
        """
        Run the daily learning pipeline for a pillar.
        
        Args:
            pillar_id: Target pillar for processing
            papers_limit: Maximum number of papers to process
            
        Returns:
            PipelineResult with processing summary and any errors
        """
        start_time = time.time()
        logger.info(f"Starting daily pipeline for pillar {pillar_id} with limit {papers_limit}")

        # Initialize result tracking
        papers_processed = []
        lessons_created = []
        quizzes_generated = []
        errors = []
        self._infra_errors = []

        try:
            # Step 1: Discovery - Generate search queries
            logger.info(f"Step 1: Discovery for pillar {pillar_id}")
            self._stage(StageName.DISCOVERY, StageStatus.RUNNING.value)
            discovery_queries, used_fallback = self._run_discovery(pillar_id)
            logger.info(f"Generated {len(discovery_queries)} search queries for pillar {pillar_id}")
            # Say so when the LLM call failed. The run still works — that is the whole
            # point of the fallback — but "3 queries" alone hides the fact that these
            # are the pillar's own focus areas rather than anything discovery chose.
            discovery_detail = f"{len(discovery_queries)} queries"
            if used_fallback:
                discovery_detail += (
                    " (discovery agent unavailable, used pillar focus areas)"
                )
            self._stage(StageName.DISCOVERY, StageStatus.COMPLETED.value,
                        discovery_detail)

            # Step 2: Search - Find candidate papers
            logger.info(f"Step 2: Searching for candidates for pillar {pillar_id}")
            self._stage(StageName.SEARCH, StageStatus.RUNNING.value)
            candidates = self._search_candidates(pillar_id, discovery_queries)
            logger.info(f"Found {len(candidates)} candidate papers for pillar {pillar_id}")
            self._stage(StageName.SEARCH, StageStatus.COMPLETED.value,
                        f"{len(candidates)} candidates")

            # Step 3: Dedupe and enqueue candidates
            logger.info(f"Step 3: Enqueueing candidates for pillar {pillar_id}")
            self._stage(StageName.ENQUEUE, StageStatus.RUNNING.value)
            enqueued_count = self._enqueue_candidates(pillar_id, candidates)
            logger.info(f"Enqueued {enqueued_count} new papers for pillar {pillar_id}")
            self._stage(StageName.ENQUEUE, StageStatus.COMPLETED.value,
                        f"{enqueued_count} enqueued")

            # Step 4: Pop papers from queue for processing
            logger.info(f"Step 4: Popping papers from queue for pillar {pillar_id}")
            self._stage(StageName.POP_QUEUE, StageStatus.RUNNING.value)
            papers_to_process = self._pop_queue(pillar_id, papers_limit)
            logger.info(f"Retrieved {len(papers_to_process)} papers to process for pillar {pillar_id}")
            self._stage(StageName.POP_QUEUE, StageStatus.COMPLETED.value,
                        f"{len(papers_to_process)} popped")

            # Step 5: Process each paper through the pipeline
            logger.info(f"Step 5: Processing {len(papers_to_process)} papers for pillar {pillar_id}")
            total_to_process = len(papers_to_process)
            self._stage(StageName.PROCESS, StageStatus.RUNNING.value,
                        f"0/{total_to_process} papers")
            for index, paper in enumerate(papers_to_process, start=1):
                try:
                    logger.info(f"Processing paper {paper.id} for pillar {pillar_id}")
                    self._stage(StageName.PROCESS, StageStatus.RUNNING.value,
                                f"{index}/{total_to_process}: {paper.id}")

                    # Process single paper through complete pipeline
                    lesson, quiz_cards = self._process_paper(pillar_id, paper)

                    # Track successful processing
                    papers_processed.append(paper.id)
                    lessons_created.append(lesson)
                    if quiz_cards:
                        quizzes_generated.extend(quiz_cards)

                    logger.info(f"Successfully processed paper {paper.id} for pillar {pillar_id}")

                except RunCancelledError:
                    # Cancellation is not a paper failure — stop the loop and let the
                    # caller record the run as cancelled rather than partly failed.
                    raise
                except Exception as e:
                    error_msg = f"Failed to process paper {paper.id}: {str(e)}"
                    logger.error(f"Error in pillar {pillar_id}: {error_msg}")
                    errors.append({
                        "paper_id": paper.id,
                        "step": "process_paper",
                        "message": error_msg
                    })
                    # _process_paper has no internal try/except, so the per-paper stage
                    # that was in flight is still marked running. Close it out here.
                    self._mark_current_paper_stage_failed(error_msg)
                    continue

            self._stage(StageName.PROCESS, StageStatus.COMPLETED.value,
                        f"{len(papers_processed)}/{total_to_process} papers processed")

            # Calculate results
            total_time = time.time() - start_time
            success = len(papers_processed) > 0  # Success if at least one paper processed

            # A run that processed nothing because the search, the queue or the
            # database was broken is not the same as one that had nothing to do, and
            # the caller can only tell them apart if the plumbing failures are here.
            errors = errors + self._infra_errors

            result = PipelineResult(
                pillar_id=pillar_id,
                papers_processed=papers_processed,
                lessons_created=lessons_created,
                quizzes_generated=quizzes_generated,
                podcasts_created=[],  # Not implemented in this step
                errors=errors,
                total_time_seconds=total_time,
                success=success
            )

            logger.info(
                f"Pipeline completed for pillar {pillar_id}: "
                f"processed={len(papers_processed)}, errors={len(errors)}, "
                f"success={success}, time={total_time:.2f}s"
            )

            return result

        except RunCancelledError:
            # MUST stay above the broad clause below. RunCancelledError subclasses
            # Exception, so without this the cancel raised by _stage() is caught,
            # converted into PipelineResult(success=False), and run_service records the
            # run as 'failed' — the user asks to stop and is told it crashed.
            # process_selected_papers has no wrapper at all and was always correct;
            # this is what made the two run kinds disagree.
            raise

        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Pipeline failed for pillar {pillar_id}: {str(e)}"
            logger.error(error_msg)

            return PipelineResult(
                pillar_id=pillar_id,
                papers_processed=papers_processed,
                lessons_created=lessons_created,
                quizzes_generated=quizzes_generated,
                podcasts_created=[],
                errors=[{"paper_id": "pipeline", "step": "run_daily", "message": error_msg}],
                total_time_seconds=total_time,
                success=False
            )

    def _stage(
        self,
        name: StageName,
        status: str,
        detail: Optional[str] = None,
    ) -> None:
        """Report a stage transition, and honour cancellation when one starts.

        Raises:
            RunCancelledError: if a cancel event was supplied and is set, and this call
                marks a stage as starting.

        """
        if (
            status == StageStatus.RUNNING.value
            and self._cancel is not None
            and self._cancel.is_set()
        ):
            raise RunCancelledError(f"cancelled before stage {name.value}")

        # DROPPED is a request to remove the stage row, not a status it can hold, so it
        # never becomes the current per-paper stage and never clears one.
        # Remember which per-paper stage is in flight. _process_paper has no internal
        # try/except, so when a paper blows up the handler in the caller knows only
        # that it failed, not where. This is how the right stage gets marked.
        if status == StageStatus.RUNNING.value and name in _PER_PAPER_STAGES:
            self._current_paper_stage = name
        elif (
            status not in (StageStatus.RUNNING.value, StageStatus.DROPPED.value)
            and name is self._current_paper_stage
        ):
            self._current_paper_stage = None

        try:
            self._on_stage(name.value, status, detail)
        except Exception:
            # A broken progress sink must never take the pipeline down with it —
            # losing the display is bad, losing the run is worse.
            logger.warning(
                f"on_stage callback failed for {name.value}/{status}", exc_info=True
            )

    def _mark_current_paper_stage_failed(self, detail: str) -> None:
        """Mark the per-paper stage that was in flight as failed.

        Called from the per-paper except blocks. Without this, a paper that dies in,
        say, summarize leaves that stage stuck at 'running' in the UI forever while
        the run moves on to the next paper.
        """
        stage = self._current_paper_stage
        if stage is None:
            return
        self._current_paper_stage = None
        try:
            self._on_stage(stage.value, StageStatus.FAILED.value, detail)
        except Exception:
            logger.warning(
                f"on_stage callback failed for {stage.value}/failed", exc_info=True
            )

    def _run_discovery(self, pillar_id: str) -> Tuple[List[str], bool]:
        """Run discovery agent to generate search queries.

        Returns:
            (queries, used_fallback). ``used_fallback`` is True when the LLM call
            failed and the pillar's own focus areas were substituted. The caller puts
            that in the stage detail: the fallback keeps the run alive, but a user
            looking at "3 queries" deserves to know the agent never answered.

        """
        try:
            # Get pillar configuration
            pillar_config = self._get_pillar_config(pillar_id)

            # Get recent notes for context
            recent_notes = db.get_recent_notes(pillar_id, limit=5)

            # Build discovery input
            discovery_input = DiscoveryInput(
                pillar=pillar_config,
                recent_papers=[note.paper_id for note in recent_notes],
                priority_topics=[]
            )

            # Run discovery agent
            discovery_output = DiscoveryAgent.run(discovery_input)

            # Extract query strings from SearchQuery objects
            queries = [query.query for query in discovery_output.queries]

            logger.info(f"Discovery generated {len(queries)} queries for pillar {pillar_id}")
            return queries, False

        except Exception as e:
            logger.warning(f"Discovery failed for pillar {pillar_id}: {e}")
            return self._fallback_queries(self._get_pillar_config(pillar_id)), True

    def _fallback_queries(
        self,
        pillar_config,
        priority_topics: Optional[List[str]] = None,
    ) -> List[str]:
        """Deterministic queries for when the discovery LLM is unavailable.

        Discovery is one OpenAI call, so it can fail for reasons that have
        nothing to do with this pillar — no API key, a rate limit, a network
        blip. None of those should stop a daily run, so both callers fall back
        here instead of propagating.

        The queries are the pillar's own focus areas, reordered by
        `_blend_topics` so anything the user asked for comes first. That reuses
        the exact ordering rule the LLM prompt uses, and focus areas are already
        the right shape — short keyword phrases.

        This replaced `["recent advances <pillar_id>", "latest research
        <pillar_id>"]`, which interpolated the *slug*: "recent advances
        neural-architectures-language" is not a phrase in any paper, and arXiv
        matches it on the filler words. Same failure mode measured for
        LLM-written prose queries — see discovery_agent's output_instructions.
        """
        topics = DiscoveryAgent._blend_topics(
            list(pillar_config.focus_areas or []),
            priority_topics or [],
        )
        queries = [t for t in topics[:3] if t and t.strip()]
        if not queries:
            # A pillar with no focus areas at all; its goal is all there is.
            queries = [DiscoveryAgent._extract_keywords(pillar_config.goal)]
        logger.info(f"Using {len(queries)} fallback queries: {queries}")
        return queries

    @property
    def infra_errors(self) -> List[dict]:
        """Failures in the shared plumbing recorded by the most recent run.

        Public because a caller that does not get a PipelineResult back — discovery
        returns a plain candidate list — otherwise has no way to tell a run that found
        ten papers cleanly from one that found ten with two sources down. Returns a
        copy so a caller cannot edit the pipeline's own record of what went wrong.
        """
        return list(self._infra_errors)

    def _record_infra_error(self, step: str, message: str) -> None:
        """Remember a failure in the shared plumbing, not in one paper.

        These three helpers — search, enqueue, pop — each catch their own exceptions
        and degrade to an empty result, which is right: one dead search backend should
        not abort a run. But the failure then existed only in the log, and a run where
        EVERY one of them failed came out indistinguishable from a run that had simply
        caught up: no papers, no errors. run_service._terminal_status() reads an empty
        errors list as success, so a completely dead stack would report
        "Done — no new papers to process" in green.

        Recorded separately from per-paper errors so `papers_failed` stays a count of
        papers. Merged into PipelineResult.errors at the end of the run.
        """
        self._infra_errors.append(
            {"paper_id": "pipeline", "step": step, "message": message}
        )

    def _search_candidates(self, pillar_id: str, queries: List[str]) -> List[PaperRef]:
        """Search for candidate papers using available tools."""
        all_candidates = []

        # Search with each tool
        for query_str in queries:
            # Create SearchQuery objects for the tools
            search_query = SearchQuery(
                pillar_id=pillar_id,
                query=query_str,
                max_results=10
            )

            try:
                # SearXNG search (sync version)
                searxng_results = self.searxng_tool.search(search_query)
                all_candidates.extend(searxng_results)
                logger.debug(f"SearXNG found {len(searxng_results)} results for query: {query_str}")
            except Exception as e:
                logger.warning(f"SearXNG search failed for query '{query_str}': {e}")
                self._record_infra_error("search", f"SearXNG search failed: {e}")

            try:
                # ArXiv search
                arxiv_results = self.arxiv_tool.search(search_query)
                all_candidates.extend(arxiv_results)
                logger.debug(f"ArXiv found {len(arxiv_results)} results for query: {query_str}")
            except Exception as e:
                logger.warning(f"ArXiv search failed for query '{query_str}': {e}")
                self._record_infra_error("search", f"arXiv search failed: {e}")

        # Deduplicate candidates
        deduplicated = self._dedupe_papers(all_candidates)
        logger.info(f"Deduplicated {len(all_candidates)} candidates to {len(deduplicated)}")

        return deduplicated

    def _enqueue_candidates(self, pillar_id: str, candidates: List[PaperRef]) -> int:
        """Enqueue candidates in the database for processing."""
        if not candidates:
            return 0

        try:
            count = db.queue_add_candidates(pillar_id, candidates)
            logger.info(f"Enqueued {count} new candidates for pillar {pillar_id}")
            return count
        except Exception as e:
            logger.error(f"Failed to enqueue candidates for pillar {pillar_id}: {e}")
            self._record_infra_error("enqueue", f"Could not queue candidates: {e}")
            return 0

    def _pop_queue(self, pillar_id: str, limit: int) -> List[PaperRef]:
        """Pop papers from the queue for processing."""
        try:
            papers = db.queue_pop_next(pillar_id, limit=limit)
            logger.info(f"Popped {len(papers)} papers from queue for pillar {pillar_id}")
            return papers
        except Exception as e:
            logger.error(f"Failed to pop papers from queue for pillar {pillar_id}: {e}")
            self._record_infra_error("pop_queue", f"Could not read the queue: {e}")
            return []

    def _process_paper(self, pillar_id: str, paper: PaperRef) -> tuple[Lesson, Optional[List[QuizCard]]]:
        """Process a single paper through the complete pipeline."""
        logger.info(f"Starting paper processing for {paper.id} in pillar {pillar_id}")

        # Step 5a: Ingest paper
        logger.info(f"Step 5a: Ingesting paper {paper.id} for pillar {pillar_id}")
        self._stage(StageName.INGEST, StageStatus.RUNNING.value, paper.id)
        parsed_paper = self.ingest_agent.ingest(paper_ref=paper)
        self._stage(StageName.INGEST, StageStatus.COMPLETED.value, paper.id)

        # Step 5b: Summarize paper
        logger.info(f"Step 5b: Summarizing paper {paper.id} for pillar {pillar_id}")
        self._stage(StageName.SUMMARIZE, StageStatus.RUNNING.value, paper.id)
        recent_notes = db.get_recent_notes(pillar_id, limit=5)
        summarizer_input = SummarizerInput(
            parsed_paper=parsed_paper,
            pillar_id=pillar_id,
            recent_notes=[note.problem + " " + note.method for note in recent_notes[-3:]]
        )
        paper_note = SummarizerAgent.run(summarizer_input)
        self._stage(StageName.SUMMARIZE, StageStatus.COMPLETED.value, paper.id)

        # Step 5c: Synthesize lesson
        logger.info(f"Step 5c: Synthesizing lesson for paper {paper.id} in pillar {pillar_id}")
        self._stage(StageName.SYNTHESIZE, StageStatus.RUNNING.value, paper.id)
        pillar_config = self._get_pillar_config(pillar_id)
        synthesis_input = SynthesisInput(
            paper_note=paper_note,
            pillar_config=pillar_config,
            related_lessons=[]  # Could get recent lessons from DB
        )
        lesson = SynthesisAgent.run(synthesis_input)
        self._stage(StageName.SYNTHESIZE, StageStatus.COMPLETED.value, paper.id)

        # Step 5d: Generate quiz (if enabled)
        quiz_cards = None
        if not self.enable_quiz:
            # Report it as skipped rather than leaving the row pending forever — the
            # UI must be able to tell "turned off" from "not reached yet".
            self._stage(StageName.QUIZ, StageStatus.SKIPPED.value, "quiz disabled")
        if self.enable_quiz:
            logger.info(f"Step 5d: Generating quiz for paper {paper.id} in pillar {pillar_id}")
            self._stage(StageName.QUIZ, StageStatus.RUNNING.value, paper.id)
            quiz_input = QuizGeneratorInput(
                paper_note=paper_note,
                num_questions=5,
                difficulty_mix={"easy": 2, "medium": 2, "hard": 1}
            )
            quiz_cards = QuizAgent.run(quiz_input)
            self._stage(StageName.QUIZ, StageStatus.COMPLETED.value,
                        f"{len(quiz_cards or [])} cards")

        # Step 5e: Persist to database
        logger.info(f"Step 5e: Persisting data for paper {paper.id} in pillar {pillar_id}")
        self._stage(StageName.PERSIST, StageStatus.RUNNING.value, paper.id)
        db.upsert_paper(pillar_id, paper)
        db.insert_note(paper_note)
        db.insert_lesson(lesson)
        if quiz_cards:
            db.insert_quiz_cards(quiz_cards)
        db.mark_processed(pillar_id, paper.id)
        self._stage(StageName.PERSIST, StageStatus.COMPLETED.value, paper.id)

        # Step 5f: Store in vector database
        logger.info(f"Step 5f: Upserting vectors for paper {paper.id} in pillar {pillar_id}")
        self._stage(StageName.VECTORS, StageStatus.RUNNING.value, paper.id)
        # Ensure Qdrant collection exists before upserting
        vectors.ensure_collections()
        chunks_upserted = vectors.upsert_text(
            pillar_id, paper.id, parsed_paper.full_text
        )
        # upsert_text() wraps its whole body in `except Exception: return 0`, so an
        # unreachable Qdrant, a bad API key and a chunker contract break all arrive
        # here as a plain 0 — indistinguishable from "this paper had no text".
        # Reporting that as a completed stage is how a dead vector store stayed
        # invisible. Infer the difference from the text we actually handed it, and
        # mark the stage failed. Non-fatal on purpose: the lesson and quiz are already
        # written, so the paper still counts as processed (see AGENTS.md on why
        # upsert_text must not be made to raise).
        if chunks_upserted == 0 and parsed_paper.full_text.strip():
            self._stage(
                StageName.VECTORS, StageStatus.FAILED.value,
                "no chunks written — vector store unavailable or chunking failed; "
                "this paper will not appear in semantic search",
            )
        else:
            self._stage(StageName.VECTORS, StageStatus.COMPLETED.value,
                        f"{chunks_upserted} chunks")

        logger.info(f"Completed paper processing for {paper.id} in pillar {pillar_id}")
        return lesson, quiz_cards

    def _get_pillar_config(self, pillar_id: str) -> PillarConfig:
        """Get pillar configuration from database or fallback to static config."""
        try:
            config = get_pillar_config(pillar_id)
            return PillarConfig(
                id=config['id'],
                name=config['name'],
                goal=config['goal'],
                focus_areas=config.get('focus_areas', ['research', 'applications']),
                papers_per_day=config.get('papers_per_day', 2)
            )
        except (ValueError, KeyError):
            # Fallback if pillar not found
            return PillarConfig(
                id=pillar_id,
                name=f"Pillar {pillar_id}",
                goal=f"Learning goals for {pillar_id}",
                focus_areas=["research", "applications"]
            )

    def _dedupe_papers(self, candidates: List[PaperRef]) -> List[PaperRef]:
        """Deduplicate papers by ID, preferring DOI/arXiv IDs."""
        seen = set()
        deduplicated = []

        for paper in candidates:
            if paper.id not in seen:
                seen.add(paper.id)
                deduplicated.append(paper)

        return deduplicated

    # =============================================
    # Enhanced Discovery with User Selection
    # =============================================

    def run_discovery_with_selection(
        self,
        pillar_id: str,
        priority_topics: List[str] = None,
        limit: int = 10
    ) -> List[DiscoveryCandidate]:
        """
        Discover papers from multiple sources for user selection.

        Every step reports itself through ``self._stage`` as it starts and finishes,
        carrying the number of candidates it actually produced. That is not
        decoration. This call takes about thirty seconds, each of its steps was
        already logged, none of it was visible, and three of its failure modes — the
        query LLM falling back to focus areas, a rate-limited search backend, an
        unreachable vector store — each produced an empty list that read exactly like
        a genuine "nothing matched". A step here says "0 found" only when it really
        found nothing; every other empty result is a failed stage carrying its reason.

        The progress sink defaults to a no-op, so `cli discover` and every existing
        caller behave exactly as before.

        Args:
            pillar_id: Target pillar
            priority_topics: Optional user-provided topic hints
            limit: Number of candidates to return

        Returns:
            List of DiscoveryCandidate objects ranked by relevance
        """
        logger.info(f"Running enhanced discovery for pillar {pillar_id}")

        self._infra_errors = []

        pillar_config = self._get_pillar_config(pillar_id)

        # --- Step D1: pillar context ---------------------------------------------
        self._stage(StageName.DISCOVER_CONTEXT, StageStatus.RUNNING.value)
        recent_paper_ids: List[str] = []
        try:
            recent_notes = db.get_recent_notes(pillar_id, limit=5)
            recent_paper_ids = [note.paper_id for note in recent_notes]
            self._stage(
                StageName.DISCOVER_CONTEXT, StageStatus.COMPLETED.value,
                f"{len(recent_paper_ids)} recent paper(s) read" if recent_paper_ids
                else "no papers read yet in this pillar",
            )
        except Exception as e:
            # Context is an input to discovery, not a precondition for it: without it
            # the agent gets no "already seen" list and there are no citations to
            # follow, but every search still works. Report it and carry on.
            message = f"couldn't read your recent papers: {_first_line(e)}"
            logger.warning(f"Discovery context failed for pillar {pillar_id}: {e}")
            self._record_infra_error(StageName.DISCOVER_CONTEXT.value, message)
            self._stage(StageName.DISCOVER_CONTEXT, StageStatus.FAILED.value, message)

        # `_search_citations` is guarded by `if recent_paper_ids`. Take its seeded row
        # away now rather than leaving a step on screen that will never run.
        if not recent_paper_ids:
            self._stage(StageName.DISCOVER_CITATIONS, StageStatus.DROPPED.value)

        # --- Step D2: search queries ----------------------------------------------
        #
        # Guarded for the same reason _run_discovery is: discovery is one OpenAI call
        # and this is the user-facing path — reached from
        # `POST /api/pillars/{id}/discover` and from `cli discover`. An unguarded raise
        # turned a transient rate limit into a 500 with no candidates, when usable
        # queries were available without a model.
        #
        # The fallback is now said out loud. It used to be a log line, so the page
        # showed the pillar's own focus areas as though the model had written them.
        self._stage(StageName.DISCOVER_QUERIES, StageStatus.RUNNING.value)
        try:
            discovery_output = DiscoveryAgent.run(
                DiscoveryInput(
                    pillar=pillar_config,
                    recent_papers=recent_paper_ids,
                    priority_topics=priority_topics or [],
                )
            )
            queries = [q.query for q in discovery_output.queries]
            logger.info(f"Generated {len(queries)} queries: {queries}")
            self._stage(
                StageName.DISCOVER_QUERIES, StageStatus.COMPLETED.value,
                _quote_queries(queries),
            )
        except Exception as e:
            logger.warning(
                f"Discovery failed for pillar {pillar_id}, falling back to "
                f"focus-area queries: {e}"
            )
            queries = self._fallback_queries(pillar_config, priority_topics)
            reason = (
                f"couldn't reach the model ({_first_line(e)}) — using your pillar's "
                f"focus areas instead"
            )
            # The stage row shows the substituted queries too; the run-level summary
            # does not. That line is one sentence next to the run's status, and a full
            # query list in it pushes everything else off the screen.
            self._record_infra_error(StageName.DISCOVER_QUERIES.value, reason)
            self._stage(
                StageName.DISCOVER_QUERIES, StageStatus.FAILED.value,
                f"{reason}: {_quote_queries(queries)}",
            )

        # --- Steps D3-D6: the sources ---------------------------------------------
        all_candidates: List[DiscoveryCandidate] = []
        first_query = queries[0] if queries else pillar_config.goal

        all_candidates += self._run_source_stage(
            StageName.DISCOVER_VECTORS,
            lambda: self._search_vectors(pillar_id, first_query),
        )
        all_candidates += self._run_source_stage(
            StageName.DISCOVER_ARXIV,
            lambda: self._search_arxiv_candidates(pillar_id, queries),
        )
        all_candidates += self._run_source_stage(
            StageName.DISCOVER_S2,
            lambda: self._search_semantic_scholar(queries),
        )
        if recent_paper_ids:
            all_candidates += self._run_source_stage(
                StageName.DISCOVER_CITATIONS,
                lambda: self._search_citations(recent_paper_ids),
            )

        # --- Step D7: rank and dedupe ---------------------------------------------
        self._stage(StageName.DISCOVER_RANK, StageStatus.RUNNING.value)
        candidates = self._rank_and_dedupe(all_candidates, limit)
        self._stage(
            StageName.DISCOVER_RANK, StageStatus.COMPLETED.value,
            f"{len(candidates)} kept from {len(all_candidates)} hit(s)",
        )

        logger.info(f"Enhanced discovery returned {len(candidates)} candidates")
        return candidates

    def _run_source_stage(
        self,
        stage: StageName,
        search: Callable[[], SourceResult],
    ) -> List[DiscoveryCandidate]:
        """Run one discovery source and report what it did — including how it failed.

        The four ``_search_*`` helpers return a :class:`SourceResult` rather than a
        bare list precisely so this can tell apart three outcomes that used to look
        identical from outside:

        * it answered, with n candidates      -> completed, "n found"
        * it answered, but part of it broke   -> completed, "n found · <reason>"
        * it could not answer at all          -> FAILED, carrying the reason

        The third is the one that mattered. A rate-limited arXiv, an unreachable
        Qdrant and a genuinely empty library all produced ``[]``, and the page showed
        the same reassuring nothing for each.

        A failed source is not a failed run: the others still have something to say,
        which is why this reports and returns rather than raising.
        """
        self._stage(stage, StageStatus.RUNNING.value)
        try:
            outcome = search()
        except Exception as e:
            # The helpers are not supposed to raise. If one does, the stage is still
            # the right place to say so — losing the whole run over one source is not.
            message = _friendly_source_error(e)
            logger.error(f"{stage.value} raised: {e}", exc_info=True)
            self._record_infra_error(stage.value, message)
            self._stage(stage, StageStatus.FAILED.value, message)
            return []

        for failure in outcome.failures:
            self._record_infra_error(stage.value, failure)

        if outcome.failures and not outcome.candidates:
            self._stage(stage, StageStatus.FAILED.value,
                        "; ".join(outcome.failures)[:_MAX_STAGE_DETAIL])
        elif outcome.failures:
            self._stage(
                stage, StageStatus.COMPLETED.value,
                f"{len(outcome.candidates)} found · "
                f"{outcome.failures[0]}"[:_MAX_STAGE_DETAIL],
            )
        else:
            self._stage(stage, StageStatus.COMPLETED.value,
                        f"{len(outcome.candidates)} found")
        return outcome.candidates

    def process_selected_papers(
        self,
        pillar_id: str,
        paper_ids: List[str] = None,
        papers: List[PaperRef] = None
    ) -> PipelineResult:
        """
        Process user-selected papers.

        Args:
            pillar_id: Target pillar
            paper_ids: Selected paper IDs to process (will fetch paper details)
            papers: Full PaperRef objects to process (preferred, preserves PDF URLs)

        Returns:
            PipelineResult with processing summary
        """
        start_time = time.time()

        self._infra_errors = []

        # Handle both papers and paper_ids for backward compatibility
        papers_to_process = []
        if papers:
            papers_to_process = papers
        elif paper_ids:
            for paper_id in paper_ids:
                paper = self._get_or_fetch_paper(pillar_id, paper_id)
                if paper:
                    papers_to_process.append(paper)
                else:
                    # Silently dropping it would leave a run that resolved nothing
                    # looking exactly like one that had nothing to do.
                    self._record_infra_error(
                        "pop_queue",
                        f"Could not resolve paper {paper_id} to a downloadable PDF",
                    )

        logger.info(f"Processing {len(papers_to_process)} selected papers for pillar {pillar_id}")

        papers_processed = []
        lessons_created = []
        quizzes_generated = []
        errors = []

        total_to_process = len(papers_to_process)
        self._stage(StageName.PROCESS, StageStatus.RUNNING.value,
                    f"0/{total_to_process} papers")
        for index, paper in enumerate(papers_to_process, start=1):
            try:
                self._stage(StageName.PROCESS, StageStatus.RUNNING.value,
                            f"{index}/{total_to_process}: {paper.id}")

                # Process through pipeline
                lesson, quiz_cards = self._process_paper(pillar_id, paper)

                papers_processed.append(paper.id)
                lessons_created.append(lesson)
                if quiz_cards:
                    quizzes_generated.extend(quiz_cards)

                # Fetch and store citations
                self._fetch_and_store_citations(paper.id)

            except RunCancelledError:
                # Not a paper failure — let the caller record the run as cancelled.
                raise
            except Exception as e:
                error_msg = f"Failed to process paper {paper.id}: {str(e)}"
                logger.error(error_msg)
                errors.append({
                    "paper_id": paper.id,
                    "step": "process_paper",
                    "message": error_msg
                })
                # _process_paper has no try/except of its own, so close out whichever
                # per-paper stage was in flight rather than leaving it 'running'.
                self._mark_current_paper_stage_failed(error_msg)

        self._stage(StageName.PROCESS, StageStatus.COMPLETED.value,
                    f"{len(papers_processed)}/{total_to_process} papers processed")

        total_time = time.time() - start_time

        return PipelineResult(
            pillar_id=pillar_id,
            papers_processed=papers_processed,
            lessons_created=lessons_created,
            quizzes_generated=quizzes_generated,
            podcasts_created=[],
            errors=errors + self._infra_errors,
            total_time_seconds=total_time,
            success=len(papers_processed) > 0
        )

    def _search_vectors(self, pillar_id: str, query: str) -> SourceResult:
        """Search the pillar's own indexed papers by meaning.

        Three ways this returns nothing, and the user is told which:

        * the tool could not be built (``__init__`` logged and set it to None),
        * vectors are switched off entirely — ``QDRANT_URL`` unset, which
          ``vectors.get_client()`` answers with None after a WARNING nobody reads,
        * the search ran and the library is empty.

        Only the third is "no results". The first two were reported as an empty
        result set for the whole life of this endpoint.
        """
        if not self.vector_search_tool:
            return SourceResult([], ["vector search is unavailable — the client could "
                                     "not be initialised"])
        if vectors.get_client() is None:
            return SourceResult([], ["vector search is switched off — QDRANT_URL is "
                                     "not set"])

        try:
            settings = get_settings()
            top_k = getattr(settings, 'vector_search_top_k', 5)
            return SourceResult(
                self.vector_search_tool.search_similar_papers(pillar_id, query, top_k),
                [],
            )
        except Exception as e:
            # search_similar_papers now lets RuntimeError through — that is the
            # "this code disagrees with its library or its server" case from
            # vectors.search_similar (a removed method, or a 4xx such as the missing
            # pillar_id payload index under Qdrant strict mode), and it is precisely
            # what must never be reported as "nothing matched".
            logger.warning(f"Vector search failed: {e}")
            return SourceResult([], [_friendly_source_error(e)])

    def _search_arxiv_candidates(
        self, pillar_id: str, queries: List[str]
    ) -> SourceResult:
        """Search arXiv and convert to DiscoveryCandidate.

        Each query is tried independently, and a query that raises is counted rather
        than merely logged: arXiv throttles bursts, and the resulting exception used
        to leave the caller with a shorter list and no idea why.
        """
        candidates = []
        failures: List[str] = []
        attempted = queries[:2]  # Limit to first 2 queries

        for query_str in attempted:
            try:
                search_query = SearchQuery(
                    pillar_id=pillar_id,
                    query=query_str,
                    max_results=5
                )
                papers = self.arxiv_tool.search(search_query)

                for i, paper in enumerate(papers):
                    # Calculate relevance score (higher for earlier results)
                    relevance = 1.0 - (i * 0.1)
                    candidate = DiscoveryCandidate(
                        paper=paper,
                        source="arxiv",
                        relevance_score=max(0.5, relevance),
                        citation_count=paper.citation_count or 0,
                        is_influential=False
                    )
                    candidates.append(candidate)

            except Exception as e:
                logger.warning(f"ArXiv search failed for query '{query_str}': {e}")
                failures.append(_friendly_source_error(e))

        return SourceResult(candidates, _summarise_query_failures(failures, attempted))

    def _search_semantic_scholar(self, queries: List[str]) -> SourceResult:
        """Search Semantic Scholar and convert to DiscoveryCandidate."""
        if not self.semantic_scholar_tool:
            return SourceResult([], ["Semantic Scholar is unavailable — the client "
                                     "could not be initialised"])

        candidates = []
        failures: List[str] = []
        attempted = queries[:2]  # Limit to first 2 queries

        for query in attempted:
            try:
                papers = self.semantic_scholar_tool.search(query, limit=5, year="2023-2024")

                for i, paper in enumerate(papers):
                    # Calculate relevance score
                    relevance = 1.0 - (i * 0.1)
                    candidate = DiscoveryCandidate(
                        paper=paper,
                        source="semantic_scholar",
                        relevance_score=max(0.5, relevance),
                        citation_count=paper.citation_count or 0,
                        is_influential=(paper.citation_count or 0) > 50
                    )
                    candidates.append(candidate)

            except Exception as e:
                logger.warning(f"Semantic Scholar search failed for query '{query}': {e}")
                failures.append(_friendly_source_error(e))

        return SourceResult(candidates, _summarise_query_failures(failures, attempted))

    def _search_citations(self, paper_ids: List[str]) -> SourceResult:
        """Search citation network for related papers."""
        candidates = []
        failures: List[str] = []

        try:
            network_paper_ids = db.get_citation_network_papers(paper_ids, limit=10)
        except Exception as e:
            # Without this the whole step returns [] and reads as "your papers cite
            # nothing we can find", when the truth is that the query never ran.
            logger.warning(f"Citation network lookup failed: {e}")
            return SourceResult([], [f"couldn't read the citation network: "
                                     f"{_first_line(e)}"])

        if not self.semantic_scholar_tool and network_paper_ids:
            return SourceResult([], ["Semantic Scholar is unavailable, so cited "
                                     "papers could not be looked up"])

        for paper_id in network_paper_ids:
            # Try to get paper details from S2
            if self.semantic_scholar_tool:
                try:
                    paper = self.semantic_scholar_tool.get_paper(paper_id)
                    if paper:
                        candidate = DiscoveryCandidate(
                            paper=paper,
                            source="citation",
                            relevance_score=0.75,  # Citation papers are relevant
                            citation_count=paper.citation_count or 0,
                            is_influential=(paper.citation_count or 0) > 50
                        )
                        candidates.append(candidate)
                except Exception as e:
                    logger.warning(f"Could not fetch paper {paper_id} from S2: {e}")
                    failures.append(_friendly_source_error(e))

        if failures:
            failures = [f"{len(failures)} of {len(network_paper_ids)} cited paper(s) "
                        f"could not be fetched: {failures[0]}"]
        return SourceResult(candidates, failures)

    def _rank_and_dedupe(
        self,
        candidates: List[DiscoveryCandidate],
        limit: int
    ) -> List[DiscoveryCandidate]:
        """Deduplicate, normalize scores, and rank candidates."""
        if not candidates:
            return []

        # Deduplicate by paper ID, keeping highest score
        seen = {}
        for c in candidates:
            paper_id = c.paper.id
            if paper_id not in seen:
                seen[paper_id] = c
            else:
                # Keep the one with higher score
                if c.relevance_score > seen[paper_id].relevance_score:
                    seen[paper_id] = c

        unique = list(seen.values())

        # Sort by relevance score (descending)
        unique.sort(key=lambda x: x.relevance_score, reverse=True)

        return unique[:limit]

    def _get_or_fetch_paper(self, pillar_id: str, paper_id: str) -> Optional[PaperRef]:
        """Get paper from database or fetch from API."""
        # Try database first
        try:
            papers = db.get_papers(pillar_id, limit=100)
            for paper in papers:
                if paper.id == paper_id:
                    return paper
        except Exception as e:
            logger.warning(f"Database lookup failed for {paper_id}: {e}")

        # Try Semantic Scholar
        if self.semantic_scholar_tool:
            try:
                paper = self.semantic_scholar_tool.get_paper(paper_id)
                if paper:
                    return paper
            except Exception as e:
                logger.warning(f"Semantic Scholar fetch failed for {paper_id}: {e}")

        # Create reference with constructed PDF URL for arXiv papers. Uses the
        # shared matcher rather than an ad-hoc "has a dot and digits" test,
        # which accepted DOIs like 10.1038/nature12345 and built arXiv URLs for
        # them.
        pdf_url = resolvable_pdf_url(paper_id, None)
        if pdf_url:
            logger.info(f"Constructed arXiv PDF URL for {paper_id}: {pdf_url}")
        else:
            logger.warning(
                f"No PDF URL could be resolved for paper {paper_id!r}; "
                f"ingest will reject it"
            )

        return PaperRef(
            id=paper_id,
            title=f"Paper {paper_id}",
            authors=[],
            url_pdf=pdf_url
        )

    def _fetch_and_store_citations(self, paper_id: str) -> int:
        """Fetch citations from Semantic Scholar and store in database."""
        if not self.semantic_scholar_tool:
            return 0

        citations_stored = 0

        try:
            # Get outgoing citations (papers this one references)
            references = self.semantic_scholar_tool.get_references(paper_id, limit=20)
            outgoing_citations = []

            for ref_paper, is_influential in references:
                citation = PaperCitation(
                    paper_id=paper_id,
                    cited_paper_id=ref_paper.id,
                    citation_direction="outgoing",
                    is_influential=is_influential,
                    source="semantic_scholar"
                )
                outgoing_citations.append(citation)

            citations_stored += db.add_paper_citations(outgoing_citations)

            # Get incoming citations (papers that cite this one)
            citations = self.semantic_scholar_tool.get_citations(paper_id, limit=20)
            incoming_citations = []

            for citing_paper, is_influential in citations:
                citation = PaperCitation(
                    paper_id=paper_id,
                    cited_paper_id=citing_paper.id,
                    citation_direction="incoming",
                    is_influential=is_influential,
                    source="semantic_scholar"
                )
                incoming_citations.append(citation)

            citations_stored += db.add_paper_citations(incoming_citations)

            logger.info(f"Stored {citations_stored} citations for paper {paper_id}")

        except Exception as e:
            logger.warning(f"Failed to fetch/store citations for {paper_id}: {e}")

        return citations_stored


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Example orchestrator usage
    orchestrator = Orchestrator(enable_quiz=True)
    print("Orchestrator module loaded successfully")
