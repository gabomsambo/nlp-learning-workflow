"""
Core Pydantic schemas for the NLP Learning Workflow.
All data models inherit from BaseIOSchema for Atomic Agents compatibility.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic import BaseModel as BaseIOSchema


# Enums
class PillarID(str, Enum):
    """The 5 learning pillars."""
    P1 = "P1"  # Linguistic & Cognitive Foundations
    P2 = "P2"  # Models & Architectures  
    P3 = "P3"  # Data, Training & Methodologies
    P4 = "P4"  # Evaluation & Interpretability
    P5 = "P5"  # Ethics & Applications

class QuestionType(str, Enum):
    """Types of quiz questions."""
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    APPLICATION = "application"

class DifficultyLevel(int, Enum):
    """Difficulty levels for content."""
    EASY = 1
    MEDIUM = 2
    HARD = 3


# Core Schemas
class PillarConfig(BaseModel):
    """Configuration for a learning pillar."""
    id: PillarID
    name: str
    goal: str
    papers_per_day: int = Field(default=2, ge=1, le=10)
    focus_areas: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: Optional[datetime] = None


class PaperRef(BaseIOSchema):
    """Reference to a research paper."""
    id: str = Field(..., description="DOI or arXiv ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of authors")
    venue: Optional[str] = Field(None, description="Conference or journal")
    year: Optional[int] = Field(None, description="Publication year")
    url_pdf: Optional[str] = Field(None, description="URL to PDF")
    abstract: Optional[str] = Field(None, description="Paper abstract")
    citation_count: Optional[int] = Field(None, description="Number of citations")


class SearchQuery(BaseIOSchema):
    """Search query for finding papers."""
    pillar_id: PillarID = Field(..., description="Target pillar")
    query: str = Field(..., description="Search query string")
    filters: dict = Field(default_factory=dict, description="Additional filters")
    max_results: int = Field(default=10, description="Maximum results to return")


class ParsedPaper(BaseIOSchema):
    """Parsed paper content."""
    paper_ref: PaperRef = Field(..., description="Paper metadata")
    full_text: str = Field(..., description="Complete paper text")
    chunks: List[str] = Field(..., description="Text chunks for processing")
    figures_count: int = Field(default=0, description="Number of figures")
    tables_count: int = Field(default=0, description="Number of tables")
    references: List[str] = Field(default_factory=list, description="Paper references")


class PaperNote(BaseIOSchema):
    """Structured notes from a paper."""
    paper_id: str = Field(..., description="Paper ID (DOI/arXiv)")
    pillar_id: PillarID = Field(..., description="Associated pillar")
    problem: str = Field(..., description="Problem the paper addresses")
    method: str = Field(..., description="Methodology used")
    findings: List[str] = Field(..., description="Key findings")
    limitations: List[str] = Field(..., description="Limitations identified")
    future_work: List[str] = Field(..., description="Future research directions")
    key_terms: List[str] = Field(..., description="Important technical terms")
    related_papers: List[str] = Field(default_factory=list, description="Related paper IDs")
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0, description="Extraction confidence")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="When the note was created")


class Lesson(BaseIOSchema):
    """Synthesized lesson from a paper."""
    paper_id: str = Field(..., description="Source paper ID")
    pillar_id: PillarID = Field(..., description="Associated pillar")
    title: str = Field(..., description="Lesson title")
    content: str = Field(..., description="Main lesson content")
    tl_dr: str = Field(..., description="One-sentence summary")
    takeaways: List[str] = Field(..., description="Key takeaways (3-5)")
    practice_ideas: List[str] = Field(..., description="Practical applications")
    connections: List[str] = Field(..., description="Connections to other work")
    examples: List[str] = Field(default_factory=list, description="Examples and illustrations")
    podcast_script: Optional[str] = Field(None, description="Generated podcast script")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)
    estimated_time: int = Field(default=10, description="Reading time in minutes")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="When the lesson was created")


class QuizCard(BaseIOSchema):
    """Quiz card for spaced repetition."""
    id: Optional[str] = Field(None, description="Unique card ID")
    paper_id: str = Field(..., description="Source paper ID")
    pillar_id: PillarID = Field(..., description="Associated pillar")
    question: str = Field(..., description="Quiz question")
    answer: str = Field(..., description="Correct answer")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)
    question_type: QuestionType = Field(default=QuestionType.FACTUAL)
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    # Spaced repetition fields (SM-2 algorithm)
    interval: int = Field(default=1, description="Days until next review")
    repetitions: int = Field(default=0, description="Number of successful reviews")
    ease_factor: float = Field(default=2.5, description="Ease factor for SM-2")
    due_date: datetime = Field(default_factory=datetime.now)
    last_reviewed: Optional[datetime] = None
    review_count: int = Field(default=0, description="Total number of reviews")
    interval_days: int = Field(default=1, description="Current interval in days")


class PodcastScript(BaseIOSchema):
    """Generated podcast script from a paper."""
    paper_id: str = Field(..., description="Source paper ID")
    pillar_id: PillarID = Field(..., description="Associated pillar")
    title: str = Field(..., description="Episode title")
    duration_minutes: int = Field(default=12, description="Target duration")
    host_cs: str = Field(..., description="Computer Science host dialogue")
    host_ling: str = Field(..., description="Linguistics host dialogue")
    key_points: List[str] = Field(..., description="Main discussion points")
    conclusion: str = Field(..., description="Joint conclusion")


class LearningProgress(BaseModel):
    """Track learning progress for a pillar."""
    pillar_id: PillarID
    papers_read: int = Field(default=0)
    papers_queued: int = Field(default=0)
    quizzes_completed: int = Field(default=0)
    current_streak: int = Field(default=0)
    longest_streak: int = Field(default=0)
    total_time_minutes: int = Field(default=0)
    last_activity: Optional[datetime] = None
    next_review: Optional[datetime] = None
    mastery_score: float = Field(default=0.0, ge=0.0, le=1.0)


class DailySession(BaseModel):
    """Record of a daily learning session."""
    id: Optional[str] = None
    pillar_id: PillarID
    date: datetime = Field(default_factory=datetime.now)
    papers_processed: List[str] = Field(default_factory=list)
    lessons_generated: int = Field(default=0)
    quizzes_created: int = Field(default=0)
    quizzes_reviewed: int = Field(default=0)
    time_spent_minutes: int = Field(default=0)
    notes: Optional[str] = None


# Agent Input/Output Schemas
class DiscoveryInput(BaseIOSchema):
    """Input for Discovery Agent."""
    pillar: PillarConfig = Field(..., description="Pillar configuration")
    recent_papers: List[str] = Field(default_factory=list, description="Recently processed paper IDs")
    priority_topics: List[str] = Field(default_factory=list, description="Priority research areas")


class DiscoveryOutput(BaseIOSchema):
    """Output from Discovery Agent."""
    queries: List[SearchQuery] = Field(..., description="Generated search queries")
    rationale: str = Field(..., description="Explanation of query choices")


class SummarizerInput(BaseIOSchema):
    """Input for Summarizer Agent."""
    parsed_paper: ParsedPaper = Field(..., description="Parsed paper content")
    pillar_id: PillarID = Field(..., description="Target pillar for context")
    recent_notes: List[str] = Field(default_factory=list, description="Recent paper summaries for consistency")


class SynthesisInput(BaseIOSchema):
    """Input for Synthesis Agent."""
    paper_note: PaperNote = Field(..., description="Structured paper notes")
    pillar_config: PillarConfig = Field(..., description="Pillar configuration")
    related_lessons: List[Lesson] = Field(default_factory=list, description="Related previous lessons")


class QuizGeneratorInput(BaseIOSchema):
    """Input for Quiz Generator Agent."""
    paper_note: PaperNote = Field(..., description="Paper notes to create quiz from")
    num_questions: int = Field(default=5, ge=1, le=10, description="Number of questions to generate")
    difficulty_mix: dict = Field(
        default={"easy": 2, "medium": 2, "hard": 1},
        description="Mix of difficulty levels"
    )


# Orchestrator Schemas
class PipelineConfig(BaseModel):
    """Configuration for the processing pipeline."""
    pillar_id: PillarID
    papers_limit: int = Field(default=2, ge=1, le=10)
    enable_quiz: bool = Field(default=True)
    enable_podcast: bool = Field(default=False)
    cache_pdfs: bool = Field(default=True)
    parallel_processing: bool = Field(default=False)


class PipelineResult(BaseModel):
    """Result from running the pipeline."""
    pillar_id: PillarID
    papers_processed: List[str]
    lessons_created: List[Lesson]
    quizzes_generated: List[QuizCard]
    podcasts_created: List[PodcastScript]
    errors: List[dict] = Field(default_factory=list)
    total_time_seconds: float
    success: bool
