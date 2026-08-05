"""
End-to-end orchestrator for running the daily pipeline for a pillar.

Composes existing agents, tools, DAO, and vector operations to process papers
through the complete learning workflow with strict pillar isolation.
"""

import logging
import time
from typing import List, Optional

from .schemas import (
    PillarConfig, PipelineResult,
    PaperRef, Lesson, QuizCard, SearchQuery,
    DiscoveryInput, SummarizerInput, SynthesisInput, QuizGeneratorInput,
    DiscoveryCandidate, PaperCitation
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


class Orchestrator:
    """
    End-to-end orchestrator for running the daily learning pipeline.
    
    Composes discovery, search, ingestion, summarization, synthesis, and quiz generation
    agents to process papers through the complete learning workflow.
    """

    def __init__(self, enable_quiz: bool = True):
        """
        Initialize the orchestrator.

        Args:
            enable_quiz: Whether to generate quiz cards during processing
        """
        self.enable_quiz = enable_quiz

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

        try:
            # Step 1: Discovery - Generate search queries
            logger.info(f"Step 1: Discovery for pillar {pillar_id}")
            discovery_queries = self._run_discovery(pillar_id)
            logger.info(f"Generated {len(discovery_queries)} search queries for pillar {pillar_id}")

            # Step 2: Search - Find candidate papers
            logger.info(f"Step 2: Searching for candidates for pillar {pillar_id}")
            candidates = self._search_candidates(pillar_id, discovery_queries)
            logger.info(f"Found {len(candidates)} candidate papers for pillar {pillar_id}")

            # Step 3: Dedupe and enqueue candidates
            logger.info(f"Step 3: Enqueueing candidates for pillar {pillar_id}")
            enqueued_count = self._enqueue_candidates(pillar_id, candidates)
            logger.info(f"Enqueued {enqueued_count} new papers for pillar {pillar_id}")

            # Step 4: Pop papers from queue for processing
            logger.info(f"Step 4: Popping papers from queue for pillar {pillar_id}")
            papers_to_process = self._pop_queue(pillar_id, papers_limit)
            logger.info(f"Retrieved {len(papers_to_process)} papers to process for pillar {pillar_id}")

            # Step 5: Process each paper through the pipeline
            logger.info(f"Step 5: Processing {len(papers_to_process)} papers for pillar {pillar_id}")
            for paper in papers_to_process:
                try:
                    logger.info(f"Processing paper {paper.id} for pillar {pillar_id}")

                    # Process single paper through complete pipeline
                    lesson, quiz_cards = self._process_paper(pillar_id, paper)

                    # Track successful processing
                    papers_processed.append(paper.id)
                    lessons_created.append(lesson)
                    if quiz_cards:
                        quizzes_generated.extend(quiz_cards)

                    logger.info(f"Successfully processed paper {paper.id} for pillar {pillar_id}")

                except Exception as e:
                    error_msg = f"Failed to process paper {paper.id}: {str(e)}"
                    logger.error(f"Error in pillar {pillar_id}: {error_msg}")
                    errors.append({
                        "paper_id": paper.id,
                        "step": "process_paper",
                        "message": error_msg
                    })
                    continue

            # Calculate results
            total_time = time.time() - start_time
            success = len(papers_processed) > 0  # Success if at least one paper processed

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

    def _run_discovery(self, pillar_id: str) -> List[str]:
        """Run discovery agent to generate search queries."""
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
            return queries

        except Exception as e:
            logger.warning(f"Discovery failed for pillar {pillar_id}: {e}")
            # Fallback to generic queries
            return [f"recent advances {pillar_id}", f"latest research {pillar_id}"]

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

            try:
                # ArXiv search
                arxiv_results = self.arxiv_tool.search(search_query)
                all_candidates.extend(arxiv_results)
                logger.debug(f"ArXiv found {len(arxiv_results)} results for query: {query_str}")
            except Exception as e:
                logger.warning(f"ArXiv search failed for query '{query_str}': {e}")

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
            return 0

    def _pop_queue(self, pillar_id: str, limit: int) -> List[PaperRef]:
        """Pop papers from the queue for processing."""
        try:
            papers = db.queue_pop_next(pillar_id, limit=limit)
            logger.info(f"Popped {len(papers)} papers from queue for pillar {pillar_id}")
            return papers
        except Exception as e:
            logger.error(f"Failed to pop papers from queue for pillar {pillar_id}: {e}")
            return []

    def _process_paper(self, pillar_id: str, paper: PaperRef) -> tuple[Lesson, Optional[List[QuizCard]]]:
        """Process a single paper through the complete pipeline."""
        logger.info(f"Starting paper processing for {paper.id} in pillar {pillar_id}")

        # Step 5a: Ingest paper
        logger.info(f"Step 5a: Ingesting paper {paper.id} for pillar {pillar_id}")
        parsed_paper = self.ingest_agent.ingest(paper_ref=paper)

        # Step 5b: Summarize paper
        logger.info(f"Step 5b: Summarizing paper {paper.id} for pillar {pillar_id}")
        recent_notes = db.get_recent_notes(pillar_id, limit=5)
        summarizer_input = SummarizerInput(
            parsed_paper=parsed_paper,
            pillar_id=pillar_id,
            recent_notes=[note.problem + " " + note.method for note in recent_notes[-3:]]
        )
        paper_note = SummarizerAgent.run(summarizer_input)

        # Step 5c: Synthesize lesson
        logger.info(f"Step 5c: Synthesizing lesson for paper {paper.id} in pillar {pillar_id}")
        pillar_config = self._get_pillar_config(pillar_id)
        synthesis_input = SynthesisInput(
            paper_note=paper_note,
            pillar_config=pillar_config,
            related_lessons=[]  # Could get recent lessons from DB
        )
        lesson = SynthesisAgent.run(synthesis_input)

        # Step 5d: Generate quiz (if enabled)
        quiz_cards = None
        if self.enable_quiz:
            logger.info(f"Step 5d: Generating quiz for paper {paper.id} in pillar {pillar_id}")
            quiz_input = QuizGeneratorInput(
                paper_note=paper_note,
                num_questions=5,
                difficulty_mix={"easy": 2, "medium": 2, "hard": 1}
            )
            quiz_cards = QuizAgent.run(quiz_input)

        # Step 5e: Persist to database
        logger.info(f"Step 5e: Persisting data for paper {paper.id} in pillar {pillar_id}")
        db.upsert_paper(pillar_id, paper)
        db.insert_note(paper_note)
        db.insert_lesson(lesson)
        if quiz_cards:
            db.insert_quiz_cards(quiz_cards)
        db.mark_processed(pillar_id, paper.id)

        # Step 5f: Store in vector database
        logger.info(f"Step 5f: Upserting vectors for paper {paper.id} in pillar {pillar_id}")
        # Ensure Qdrant collection exists before upserting
        vectors.ensure_collections()
        vectors.upsert_text(pillar_id, paper.id, parsed_paper.full_text)

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

        Args:
            pillar_id: Target pillar
            priority_topics: Optional user-provided topic hints
            limit: Number of candidates to return

        Returns:
            List of DiscoveryCandidate objects ranked by relevance
        """
        logger.info(f"Running enhanced discovery for pillar {pillar_id}")

        # Get pillar config
        pillar_config = self._get_pillar_config(pillar_id)

        # Get recent notes for context
        recent_notes = db.get_recent_notes(pillar_id, limit=5)
        recent_paper_ids = [note.paper_id for note in recent_notes]

        # Build discovery input with user topics
        discovery_input = DiscoveryInput(
            pillar=pillar_config,
            recent_papers=recent_paper_ids,
            priority_topics=priority_topics or []
        )

        # Generate blended queries
        discovery_output = DiscoveryAgent.run(discovery_input)
        queries = [q.query for q in discovery_output.queries]

        logger.info(f"Generated {len(queries)} queries: {queries}")

        # Search all sources
        all_candidates = []

        # 1. Vector search (semantic similarity from existing papers)
        vector_candidates = self._search_vectors(pillar_id, queries[0] if queries else pillar_config.goal)
        all_candidates.extend(vector_candidates)
        logger.info(f"Vector search found {len(vector_candidates)} candidates")

        # 2. ArXiv search
        arxiv_candidates = self._search_arxiv_candidates(pillar_id, queries)
        all_candidates.extend(arxiv_candidates)
        logger.info(f"ArXiv search found {len(arxiv_candidates)} candidates")

        # 3. Semantic Scholar search
        s2_candidates = self._search_semantic_scholar(queries)
        all_candidates.extend(s2_candidates)
        logger.info(f"Semantic Scholar search found {len(s2_candidates)} candidates")

        # 4. Citation network search (if we have papers)
        if recent_paper_ids:
            citation_candidates = self._search_citations(recent_paper_ids)
            all_candidates.extend(citation_candidates)
            logger.info(f"Citation search found {len(citation_candidates)} candidates")

        # Normalize scores, dedupe, and rank
        candidates = self._rank_and_dedupe(all_candidates, limit)

        logger.info(f"Enhanced discovery returned {len(candidates)} candidates")
        return candidates

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

        # Handle both papers and paper_ids for backward compatibility
        papers_to_process = []
        if papers:
            papers_to_process = papers
        elif paper_ids:
            for paper_id in paper_ids:
                paper = self._get_or_fetch_paper(pillar_id, paper_id)
                if paper:
                    papers_to_process.append(paper)

        logger.info(f"Processing {len(papers_to_process)} selected papers for pillar {pillar_id}")

        papers_processed = []
        lessons_created = []
        quizzes_generated = []
        errors = []

        for paper in papers_to_process:
            try:
                # Process through pipeline
                lesson, quiz_cards = self._process_paper(pillar_id, paper)

                papers_processed.append(paper.id)
                lessons_created.append(lesson)
                if quiz_cards:
                    quizzes_generated.extend(quiz_cards)

                # Fetch and store citations
                self._fetch_and_store_citations(paper.id)

            except Exception as e:
                error_msg = f"Failed to process paper {paper.id}: {str(e)}"
                logger.error(error_msg)
                errors.append({
                    "paper_id": paper.id,
                    "step": "process_paper",
                    "message": error_msg
                })

        total_time = time.time() - start_time

        return PipelineResult(
            pillar_id=pillar_id,
            papers_processed=papers_processed,
            lessons_created=lessons_created,
            quizzes_generated=quizzes_generated,
            podcasts_created=[],
            errors=errors,
            total_time_seconds=total_time,
            success=len(papers_processed) > 0
        )

    def _search_vectors(self, pillar_id: str, query: str) -> List[DiscoveryCandidate]:
        """Search for similar papers using vector database."""
        if not self.vector_search_tool:
            return []

        try:
            settings = get_settings()
            top_k = getattr(settings, 'vector_search_top_k', 5)
            return self.vector_search_tool.search_similar_papers(pillar_id, query, top_k)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    def _search_arxiv_candidates(self, pillar_id: str, queries: List[str]) -> List[DiscoveryCandidate]:
        """Search ArXiv and convert to DiscoveryCandidate."""
        candidates = []

        for query_str in queries[:2]:  # Limit to first 2 queries
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

        return candidates

    def _search_semantic_scholar(self, queries: List[str]) -> List[DiscoveryCandidate]:
        """Search Semantic Scholar and convert to DiscoveryCandidate."""
        if not self.semantic_scholar_tool:
            return []

        candidates = []

        for query in queries[:2]:  # Limit to first 2 queries
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

        return candidates

    def _search_citations(self, paper_ids: List[str]) -> List[DiscoveryCandidate]:
        """Search citation network for related papers."""
        candidates = []

        # Get papers from citation network in database
        network_paper_ids = db.get_citation_network_papers(paper_ids, limit=10)

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

        return candidates

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
