"""
End-to-end orchestrator for running the daily pipeline for a pillar.

Composes existing agents, tools, DAO, and vector operations to process papers
through the complete learning workflow with strict pillar isolation.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from .schemas import (
    PillarID, PillarConfig, PipelineResult, 
    PaperRef, PaperNote, Lesson, QuizCard, SearchQuery,
    DiscoveryInput, SummarizerInput, SynthesisInput, QuizGeneratorInput
)
from .agents.discovery_agent import DiscoveryAgent
from .agents.ingest_agent import IngestAgent  
from .agents.summarizer_agent import SummarizerAgent
from .agents.synthesis_agent import SynthesisAgent
from .agents.quiz_agent import QuizAgent
from .tools.searxng_tool import SearXNGTool
from .tools.arxiv_tool import ArXivTool
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
        
        logger.info(f"Orchestrator initialized with enable_quiz={enable_quiz}")
    
    def run_daily(self, pillar_id: PillarID, papers_limit: int = 1) -> PipelineResult:
        """
        Run the daily learning pipeline for a pillar.
        
        Args:
            pillar_id: Target pillar for processing
            papers_limit: Maximum number of papers to process
            
        Returns:
            PipelineResult with processing summary and any errors
        """
        start_time = time.time()
        logger.info(f"Starting daily pipeline for pillar {pillar_id.value} with limit {papers_limit}")
        
        # Initialize result tracking
        papers_processed = []
        lessons_created = []
        quizzes_generated = []
        errors = []
        
        try:
            # Step 1: Discovery - Generate search queries
            logger.info(f"Step 1: Discovery for pillar {pillar_id.value}")
            discovery_queries = self._run_discovery(pillar_id)
            logger.info(f"Generated {len(discovery_queries)} search queries for pillar {pillar_id.value}")
            
            # Step 2: Search - Find candidate papers
            logger.info(f"Step 2: Searching for candidates for pillar {pillar_id.value}")
            candidates = self._search_candidates(pillar_id, discovery_queries)
            logger.info(f"Found {len(candidates)} candidate papers for pillar {pillar_id.value}")
            
            # Step 3: Dedupe and enqueue candidates
            logger.info(f"Step 3: Enqueueing candidates for pillar {pillar_id.value}")
            enqueued_count = self._enqueue_candidates(pillar_id, candidates)
            logger.info(f"Enqueued {enqueued_count} new papers for pillar {pillar_id.value}")
            
            # Step 4: Pop papers from queue for processing
            logger.info(f"Step 4: Popping papers from queue for pillar {pillar_id.value}")
            papers_to_process = self._pop_queue(pillar_id, papers_limit)
            logger.info(f"Retrieved {len(papers_to_process)} papers to process for pillar {pillar_id.value}")
            
            # Step 5: Process each paper through the pipeline
            logger.info(f"Step 5: Processing {len(papers_to_process)} papers for pillar {pillar_id.value}")
            for paper in papers_to_process:
                try:
                    logger.info(f"Processing paper {paper.id} for pillar {pillar_id.value}")
                    
                    # Process single paper through complete pipeline
                    lesson, quiz_cards = self._process_paper(pillar_id, paper)
                    
                    # Track successful processing
                    papers_processed.append(paper.id)
                    lessons_created.append(lesson)
                    if quiz_cards:
                        quizzes_generated.extend(quiz_cards)
                    
                    logger.info(f"Successfully processed paper {paper.id} for pillar {pillar_id.value}")
                    
                except Exception as e:
                    error_msg = f"Failed to process paper {paper.id}: {str(e)}"
                    logger.error(f"Error in pillar {pillar_id.value}: {error_msg}")
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
                f"Pipeline completed for pillar {pillar_id.value}: "
                f"processed={len(papers_processed)}, errors={len(errors)}, "
                f"success={success}, time={total_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Pipeline failed for pillar {pillar_id.value}: {str(e)}"
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
    
    def _run_discovery(self, pillar_id: PillarID) -> List[str]:
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
            
            logger.info(f"Discovery generated {len(queries)} queries for pillar {pillar_id.value}")
            return queries
            
        except Exception as e:
            logger.warning(f"Discovery failed for pillar {pillar_id.value}: {e}")
            # Fallback to generic queries
            return [f"recent advances {pillar_id.value}", f"latest research {pillar_id.value}"]
    
    def _search_candidates(self, pillar_id: PillarID, queries: List[str]) -> List[PaperRef]:
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
    
    def _enqueue_candidates(self, pillar_id: PillarID, candidates: List[PaperRef]) -> int:
        """Enqueue candidates in the database for processing."""
        if not candidates:
            return 0
        
        try:
            count = db.queue_add_candidates(pillar_id, candidates)
            logger.info(f"Enqueued {count} new candidates for pillar {pillar_id.value}")
            return count
        except Exception as e:
            logger.error(f"Failed to enqueue candidates for pillar {pillar_id.value}: {e}")
            return 0
    
    def _pop_queue(self, pillar_id: PillarID, limit: int) -> List[PaperRef]:
        """Pop papers from the queue for processing."""
        try:
            papers = db.queue_pop_next(pillar_id, limit=limit)
            logger.info(f"Popped {len(papers)} papers from queue for pillar {pillar_id.value}")
            return papers
        except Exception as e:
            logger.error(f"Failed to pop papers from queue for pillar {pillar_id.value}: {e}")
            return []
    
    def _process_paper(self, pillar_id: PillarID, paper: PaperRef) -> tuple[Lesson, Optional[List[QuizCard]]]:
        """Process a single paper through the complete pipeline."""
        logger.info(f"Starting paper processing for {paper.id} in pillar {pillar_id.value}")
        
        # Step 5a: Ingest paper
        logger.info(f"Step 5a: Ingesting paper {paper.id} for pillar {pillar_id.value}")
        parsed_paper = self.ingest_agent.ingest(paper_ref=paper)
        
        # Step 5b: Summarize paper
        logger.info(f"Step 5b: Summarizing paper {paper.id} for pillar {pillar_id.value}")
        recent_notes = db.get_recent_notes(pillar_id, limit=5)
        summarizer_input = SummarizerInput(
            parsed_paper=parsed_paper,
            pillar_id=pillar_id,
            recent_notes=[note.problem + " " + note.method for note in recent_notes[-3:]]
        )
        paper_note = SummarizerAgent.run(summarizer_input)
        
        # Step 5c: Synthesize lesson
        logger.info(f"Step 5c: Synthesizing lesson for paper {paper.id} in pillar {pillar_id.value}")
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
            logger.info(f"Step 5d: Generating quiz for paper {paper.id} in pillar {pillar_id.value}")
            quiz_input = QuizGeneratorInput(
                paper_note=paper_note,
                num_questions=5,
                difficulty_mix={"easy": 2, "medium": 2, "hard": 1}
            )
            quiz_cards = QuizAgent.run(quiz_input)
        
        # Step 5e: Persist to database
        logger.info(f"Step 5e: Persisting data for paper {paper.id} in pillar {pillar_id.value}")
        db.upsert_paper(pillar_id, paper)
        db.insert_note(paper_note)
        db.insert_lesson(lesson)
        if quiz_cards:
            db.insert_quiz_cards(quiz_cards)
        db.mark_processed(pillar_id, paper.id)
        
        # Step 5f: Store in vector database
        logger.info(f"Step 5f: Upserting vectors for paper {paper.id} in pillar {pillar_id.value}")
        # Ensure Qdrant collection exists before upserting
        vectors.ensure_collections()
        vectors.upsert_text(pillar_id, paper.id, parsed_paper.full_text)
        
        logger.info(f"Completed paper processing for {paper.id} in pillar {pillar_id.value}")
        return lesson, quiz_cards
    
    def _get_pillar_config(self, pillar_id: PillarID) -> PillarConfig:
        """Get or create pillar configuration."""
        # For now, create a minimal in-memory config
        # In the future, this could load from DB
        pillar_names = {
            PillarID.P1: "Foundations & Concepts",
            PillarID.P2: "Models & Architectures", 
            PillarID.P3: "Training & Optimization",
            PillarID.P4: "Applications & Use Cases",
            PillarID.P5: "Ethics & Society"
        }
        
        pillar_goals = {
            PillarID.P1: "Master fundamental NLP concepts and mathematical foundations",
            PillarID.P2: "Understand modern neural architectures and model design",
            PillarID.P3: "Learn training techniques and optimization strategies",
            PillarID.P4: "Explore practical applications and real-world implementations",
            PillarID.P5: "Consider ethical implications and societal impact"
        }
        
        return PillarConfig(
            id=pillar_id,
            name=pillar_names.get(pillar_id, f"Pillar {pillar_id.value}"),
            goal=pillar_goals.get(pillar_id, f"Learning goals for {pillar_id.value}"),
            focus_areas=["research", "innovation", "applications"]
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


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example orchestrator usage
    orchestrator = Orchestrator(enable_quiz=True)
    print("Orchestrator module loaded successfully")
