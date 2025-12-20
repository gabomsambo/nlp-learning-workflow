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
    """DEPRECATED: Use string pillar IDs (slugs) instead.

    This enum is kept for backward compatibility only.
    All new code should use dynamic pillar IDs like 'models-architectures'.
    """
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

class FSRSRating(int, Enum):
    """FSRS review ratings."""
    AGAIN = 1    # Completely forgot, need to review again
    HARD = 2     # Struggled to remember, but got it
    GOOD = 3     # Remembered with some effort
    EASY = 4     # Remembered easily

class CardState(str, Enum):
    """FSRS card states."""
    NEW = "new"           # Never reviewed
    LEARNING = "learning" # In initial learning phase
    REVIEW = "review"     # In regular review phase
    RELEARNING = "relearning"  # Failed and relearning


# Core Schemas

# New dynamic pillar schemas
class PillarBase(BaseModel):
    """Base pillar schema."""
    name: str = Field(..., min_length=1, max_length=100, description="Human-readable pillar name")
    goal: str = Field(..., min_length=10, max_length=500, description="Learning objective for this pillar")
    focus_areas: List[str] = Field(default_factory=list, description="Array of focus area topics")
    papers_per_day: int = Field(default=2, ge=1, le=10, description="Target papers per day")


class PillarCreate(PillarBase):
    """Schema for creating a new pillar."""
    pass


class PillarUpdate(BaseModel):
    """Schema for updating an existing pillar."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    goal: Optional[str] = Field(None, min_length=10, max_length=500)
    focus_areas: Optional[List[str]] = None
    papers_per_day: Optional[int] = Field(None, ge=1, le=10)


class Pillar(PillarBase):
    """Full pillar schema with metadata."""
    id: str = Field(..., description="URL-friendly slug")
    created_at: datetime = Field(..., description="When the pillar was created")
    updated_at: datetime = Field(..., description="When the pillar was last updated")
    last_active: Optional[datetime] = Field(None, description="Last activity timestamp")

    class Config:
        from_attributes = True


# Legacy pillar config for backward compatibility
class PillarConfig(BaseModel):
    """Configuration for a learning pillar.

    Note: id field accepts string slugs to support dynamic pillars.
    """
    id: str  # Changed from PillarID to support dynamic pillar slugs
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


# Upload-related schemas
class UploadUrlRequest(BaseModel):
    """Request schema for uploading a paper from URL."""
    url: str = Field(..., description="URL to download PDF from")
    title: Optional[str] = Field(None, description="Optional paper title override")
    authors: Optional[List[str]] = Field(None, description="Optional authors override")
    run_summarizer: bool = Field(default=False, description="Run summarizer after upload")
    generate_quiz: bool = Field(default=False, description="Generate quiz after upload")


class UploadFileRequest(BaseModel):
    """Request schema for uploading a paper file."""
    title: str = Field(..., description="Paper title")
    authors: Optional[List[str]] = Field(default_factory=list, description="Paper authors")
    venue: Optional[str] = Field(None, description="Conference or journal")
    year: Optional[int] = Field(None, description="Publication year")
    run_summarizer: bool = Field(default=False, description="Run summarizer after upload")
    generate_quiz: bool = Field(default=False, description="Generate quiz after upload")


class UploadResponse(BaseModel):
    """Response schema for upload operations."""
    success: bool = Field(..., description="Whether upload was successful")
    paper: Optional[PaperRef] = Field(None, description="Created paper reference")
    message: str = Field(..., description="Status message")
    actions_triggered: List[str] = Field(default_factory=list, description="Post-upload actions triggered")


class UploadStatus(BaseIOSchema):
    """Upload status for tracking progress."""
    id: str = Field(..., description="Upload ID")
    pillar_id: str = Field(..., description="Target pillar ID")
    status: str = Field(..., description="Current status: pending, processing, completed, failed")
    filename: Optional[str] = Field(None, description="Uploaded filename")
    url: Optional[str] = Field(None, description="Source URL if URL upload")
    progress: int = Field(default=0, description="Progress percentage (0-100)")
    message: str = Field(default="", description="Current status message")
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class SearchQuery(BaseIOSchema):
    """Search query for finding papers."""
    pillar_id: str = Field(..., description="Target pillar ID (slug)")
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
    pillar_id: str = Field(..., description="Associated pillar ID (slug)")
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
    pillar_id: str = Field(..., description="Associated pillar ID (slug)")
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
    """Quiz card for spaced repetition using FSRS algorithm."""
    id: Optional[str] = Field(None, description="Unique card ID")
    paper_id: str = Field(..., description="Source paper ID")
    pillar_id: str = Field(..., description="Associated pillar ID (slug)")
    question: str = Field(..., description="Quiz question")
    answer: str = Field(..., description="Correct answer")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)
    question_type: QuestionType = Field(default=QuestionType.FACTUAL)
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    user_id: str = Field(default="default_user", description="User who owns this card")

    # FSRS algorithm fields
    difficulty_fsrs: float = Field(default=0.0, description="FSRS difficulty parameter")
    stability: float = Field(default=0.0, description="FSRS stability parameter")
    retrievability: Optional[float] = Field(None, description="Current retrievability")
    state: CardState = Field(default=CardState.NEW, description="FSRS card state")
    lapses: int = Field(default=0, description="Number of times forgotten")

    # Review scheduling
    last_review_date: Optional[datetime] = Field(None, description="Last review timestamp")
    next_review_date: Optional[datetime] = Field(None, description="Next scheduled review")

    # Legacy SM-2 fields (for backward compatibility)
    interval: int = Field(default=1, description="Days until next review (legacy)")
    repetitions: int = Field(default=0, description="Number of successful reviews (legacy)")
    ease_factor: float = Field(default=2.5, description="Ease factor for SM-2 (legacy)")
    due_date: datetime = Field(default_factory=datetime.now, description="Due date (legacy)")
    last_reviewed: Optional[datetime] = Field(None, description="Last reviewed (legacy)")
    review_count: int = Field(default=0, description="Total number of reviews")
    interval_days: int = Field(default=1, description="Current interval in days (legacy)")


class PodcastScript(BaseIOSchema):
    """Generated podcast script from a paper - single host format."""
    id: Optional[str] = Field(None, description="Unique script ID")
    paper_id: str = Field(..., description="Source paper ID")
    pillar_id: str = Field(..., description="Associated pillar ID (slug)")
    title: str = Field(..., description="Episode title")
    script: str = Field(..., description="Full script in [HOST]: format")
    word_count: int = Field(default=0, description="Script word count")
    key_points: List[str] = Field(default_factory=list, description="Main discussion points")
    ground_pack: dict = Field(default_factory=dict, description="4 prompt outputs for reference")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="When the script was created")


class LearningProgress(BaseModel):
    """Track learning progress for a pillar."""
    pillar_id: str = Field(..., description="Pillar ID (slug)")
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
    pillar_id: str = Field(..., description="Pillar ID (slug)")
    date: datetime = Field(default_factory=datetime.now)
    papers_processed: List[str] = Field(default_factory=list)
    lessons_generated: int = Field(default=0)
    quizzes_created: int = Field(default=0)
    quizzes_reviewed: int = Field(default=0)
    time_spent_minutes: int = Field(default=0)
    notes: Optional[str] = None


class ReviewLog(BaseIOSchema):
    """Log entry for a quiz card review session."""
    id: Optional[str] = Field(None, description="Unique log ID")
    card_id: str = Field(..., description="Quiz card ID that was reviewed")
    user_id: str = Field(default="default_user", description="User who performed the review")
    pillar_id: str = Field(..., description="Associated pillar ID (slug)")
    paper_id: str = Field(..., description="Source paper ID")
    rating: FSRSRating = Field(..., description="User's rating of recall difficulty")
    review_timestamp: datetime = Field(default_factory=datetime.now, description="When the review occurred")

    # Card state at time of review (for FSRS optimization)
    difficulty: float = Field(..., description="FSRS difficulty at review time")
    stability: float = Field(..., description="FSRS stability at review time")
    retrievability: Optional[float] = Field(None, description="Calculated retrievability")

    # Context information
    previous_due_date: Optional[datetime] = Field(None, description="When card was originally due")
    days_overdue: int = Field(default=0, description="Days past due date")
    session_id: Optional[str] = Field(None, description="Review session identifier")
    response_time_ms: Optional[int] = Field(None, description="Time to answer in milliseconds")


class UserFSRSParameters(BaseIOSchema):
    """FSRS algorithm parameters personalized for a user."""
    id: Optional[str] = Field(None, description="Unique parameter set ID")
    user_id: str = Field(default="default_user", description="User these parameters belong to")
    pillar_id: Optional[str] = Field(None, description="Pillar-specific parameters (None = global)")

    # FSRS algorithm weights (17 parameters)
    w0: float = Field(default=0.4, description="Initial stability for new cards")
    w1: float = Field(default=0.6, description="Initial stability for learning cards")
    w2: float = Field(default=2.4, description="Initial stability multiplier")
    w3: float = Field(default=5.8, description="Initial difficulty offset")
    w4: float = Field(default=4.93, description="Difficulty weight for Again")
    w5: float = Field(default=0.94, description="Difficulty weight for Hard")
    w6: float = Field(default=0.86, description="Difficulty weight for Good")
    w7: float = Field(default=0.01, description="Difficulty weight for Easy")
    w8: float = Field(default=1.49, description="Stability multiplier for Again")
    w9: float = Field(default=0.14, description="Stability multiplier for Hard")
    w10: float = Field(default=0.94, description="Stability multiplier for Good")
    w11: float = Field(default=2.18, description="Stability multiplier for Easy")
    w12: float = Field(default=0.05, description="Difficulty decay for Again")
    w13: float = Field(default=0.34, description="Difficulty decay for Hard")
    w14: float = Field(default=0.67, description="Difficulty decay for Good")
    w15: float = Field(default=2.74, description="Difficulty decay for Easy")
    w16: float = Field(default=0.0, description="Forgetting curve parameter")
    w17: float = Field(default=2.0, description="Stability increase parameter")

    # Metadata
    review_count: int = Field(default=0, description="Reviews used for optimization")
    last_optimized: Optional[datetime] = Field(None, description="Last optimization run")
    optimization_score: Optional[float] = Field(None, description="Quality of optimization")
    is_default: bool = Field(default=False, description="Whether these are default parameters")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_weights_list(self) -> List[float]:
        """Convert parameters to list format for FSRS library."""
        return [
            self.w0, self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.w7,
            self.w8, self.w9, self.w10, self.w11, self.w12, self.w13, self.w14, self.w15,
            self.w16, self.w17
        ]

    @classmethod
    def from_weights_list(cls, weights: List[float], user_id: str = "default_user",
                         pillar_id: Optional[str] = None) -> "UserFSRSParameters":
        """Create parameters from weights list."""
        if len(weights) != 18:
            raise ValueError(f"Expected 18 weights, got {len(weights)}")

        return cls(
            user_id=user_id,
            pillar_id=pillar_id,
            w0=weights[0], w1=weights[1], w2=weights[2], w3=weights[3],
            w4=weights[4], w5=weights[5], w6=weights[6], w7=weights[7],
            w8=weights[8], w9=weights[9], w10=weights[10], w11=weights[11],
            w12=weights[12], w13=weights[13], w14=weights[14], w15=weights[15],
            w16=weights[16], w17=weights[17]
        )


class QuizReviewRequest(BaseIOSchema):
    """Request to review a quiz card with FSRS rating."""
    card_id: str = Field(..., description="Quiz card ID to review")
    rating: FSRSRating = Field(..., description="User's rating of recall difficulty")
    response_time_ms: Optional[int] = Field(None, description="Time taken to answer")
    session_id: Optional[str] = Field(None, description="Review session ID")


class QuizReviewResponse(BaseIOSchema):
    """Response after reviewing a quiz card."""
    success: bool = Field(..., description="Whether review was processed successfully")
    card: QuizCard = Field(..., description="Updated card with new FSRS parameters")
    next_review_date: datetime = Field(..., description="When card should be reviewed next")
    message: str = Field(default="", description="Status message")


class FSRSOptimizationRequest(BaseIOSchema):
    """Request to optimize FSRS parameters for a user."""
    user_id: str = Field(default="default_user", description="User to optimize parameters for")
    pillar_id: Optional[str] = Field(None, description="Specific pillar or None for global")
    min_reviews: int = Field(default=50, description="Minimum reviews required for optimization")


class FSRSOptimizationResponse(BaseIOSchema):
    """Response from FSRS parameter optimization."""
    success: bool = Field(..., description="Whether optimization was successful")
    parameters: Optional[UserFSRSParameters] = Field(None, description="Optimized parameters")
    improvement_score: Optional[float] = Field(None, description="How much parameters improved")
    reviews_analyzed: int = Field(default=0, description="Number of reviews used")
    message: str = Field(default="", description="Status message")


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


class DiscoveryCandidate(BaseIOSchema):
    """A candidate paper with discovery metadata."""
    paper: PaperRef = Field(..., description="Paper reference")
    source: str = Field(..., description="Source: 'vector', 'arxiv', 'semantic_scholar', 'citation'")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Normalized relevance score 0-1")
    citation_count: int = Field(default=0, description="Number of citations")
    is_influential: bool = Field(default=False, description="Is this an influential paper")


class EnhancedDiscoveryInput(BaseIOSchema):
    """Enhanced input for Discovery Agent with user guidance."""
    pillar: PillarConfig = Field(..., description="Pillar configuration")
    recent_papers: List[str] = Field(default_factory=list, description="Recently processed paper IDs")
    priority_topics: List[str] = Field(default_factory=list, description="User-provided topic hints (guide, not override)")
    author_filter: Optional[str] = Field(None, description="Optional author to filter by")


class EnhancedDiscoveryOutput(BaseIOSchema):
    """Output from Enhanced Discovery with ranked candidates."""
    candidates: List[DiscoveryCandidate] = Field(..., description="Ranked discovery candidates")
    total_found: int = Field(..., description="Total candidates found before filtering")
    sources_used: List[str] = Field(..., description="Sources that returned results")
    rationale: str = Field(..., description="Explanation of discovery choices")


class PaperCitation(BaseModel):
    """Citation relationship between papers."""
    id: Optional[str] = Field(None, description="Unique citation ID")
    paper_id: str = Field(..., description="Paper ID (DOI/arXiv)")
    cited_paper_id: str = Field(..., description="Cited paper ID (DOI/arXiv)")
    citation_direction: str = Field(..., description="'outgoing' or 'incoming'")
    is_influential: bool = Field(default=False, description="Is this an influential citation")
    citation_context: Optional[str] = Field(None, description="Text context around citation")
    source: str = Field(default="semantic_scholar", description="Source API")
    fetched_at: Optional[datetime] = Field(None, description="When citation was fetched")


class UserSelection(BaseModel):
    """User's paper selection from discovery candidates."""
    pillar_id: str = Field(..., description="Pillar ID")
    selected_paper_ids: List[str] = Field(..., description="Selected paper IDs to process")


class SummarizerInput(BaseIOSchema):
    """Input for Summarizer Agent."""
    parsed_paper: ParsedPaper = Field(..., description="Parsed paper content")
    pillar_id: str = Field(..., description="Target pillar ID (slug)")
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
    pillar_id: str = Field(..., description="Pillar ID (slug)")
    papers_limit: int = Field(default=2, ge=1, le=10)
    enable_quiz: bool = Field(default=True)
    enable_podcast: bool = Field(default=False)
    cache_pdfs: bool = Field(default=True)
    parallel_processing: bool = Field(default=False)


class PipelineResult(BaseModel):
    """Result from running the pipeline."""
    pillar_id: str = Field(..., description="Pillar ID (slug)")
    papers_processed: List[str]
    lessons_created: List[Lesson]
    quizzes_generated: List[QuizCard]
    podcasts_created: List[PodcastScript]
    errors: List[dict] = Field(default_factory=list)
    total_time_seconds: float
    success: bool
