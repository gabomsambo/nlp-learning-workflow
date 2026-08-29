"""
Core Pydantic schemas for the NLP Learning Workflow.
All data models inherit from BaseIOSchema for Atomic Agents compatibility.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
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
    run_summarizer: bool = Field(default=True, description="Run summarizer after upload")
    generate_quiz: bool = Field(default=True, description="Generate quiz after upload")


class UploadFileRequest(BaseModel):
    """Request schema for uploading a paper file."""
    title: str = Field(..., description="Paper title")
    authors: Optional[List[str]] = Field(default_factory=list, description="Paper authors")
    venue: Optional[str] = Field(None, description="Conference or journal")
    year: Optional[int] = Field(None, description="Publication year")
    run_summarizer: bool = Field(default=True, description="Run summarizer after upload")
    generate_quiz: bool = Field(default=True, description="Generate quiz after upload")


class UploadResponse(BaseModel):
    """Response schema for upload operations.

    ``success`` and ``pipeline_ok`` are two different facts and are deliberately not
    collapsed. ``success`` means the paper reached the ``papers`` table; the follow-on
    processing — ingest, summarizer, synthesis, quiz, vectors — is reported separately,
    because it used to be reported not at all. A pipeline exception was appended to
    ``actions_triggered`` as the pseudo-action ``pipeline_error: ...`` and the route
    still answered ``success=True``, so the browser said "uploaded successfully!
    Triggered: pipeline_error: ..." and a paper with no note, no lesson and no quiz
    read as green.
    """
    success: bool = Field(..., description="Whether the paper was added to the library")
    paper: Optional[PaperRef] = Field(None, description="Created paper reference")
    message: str = Field(..., description="Status message")
    actions_triggered: List[str] = Field(default_factory=list, description="Post-upload actions that COMPLETED")
    pipeline_ok: bool = Field(default=True, description="Whether every requested post-upload action completed")
    pipeline_errors: List[str] = Field(default_factory=list, description="Post-upload actions that failed, and why")


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


class SourceMaterial(BaseModel):
    """What the podcast was actually written from.

    A script built from an abstract alone and a script built from the full paper
    are different artifacts, and used to be indistinguishable in the response, on
    the page and in the database. This records the difference so the ambiguity
    does not just move one level down. ``level`` is "full" or "partial"; a run
    with neither body, abstract nor notes never reaches this type — it raises
    (see ``podcast_agent.InsufficientSourceMaterialError``).
    """
    level: str = Field(default="full", description="'full' or 'partial'")
    full_text_chars: int = Field(default=0, description="Chars of paper body used (0 when none)")
    has_abstract: bool = Field(default=False)
    has_notes: bool = Field(default=False)
    warnings: List[str] = Field(
        default_factory=list,
        description="Human-readable caveats, shown to the user and stored with the script",
    )


class OptionChoice(BaseModel):
    """One podcast option as it was actually chosen.

    Both halves are kept: ``preset`` (or ``custom``) is what to re-apply, and
    ``label`` is what was put in front of the model. Storing the label as well
    means a row from a year ago still says what it was aimed at even if the
    preset list has moved on since.
    """
    key: str = Field(..., description="Option key, e.g. 'field'")
    preset: Optional[str] = Field(None, description="Preset value, or None when custom")
    custom: Optional[str] = Field(None, description="Sanitized free text, or None when a preset")
    label: str = Field(..., description="What was shown to the user and sent to the model")


class PodcastOptions(BaseModel):
    """What a podcast script was aimed at: field, audience, length, tone.

    Keyed by option key rather than given four named fields, so adding a fifth
    option is a data change in ``nlp_pillars/podcast_options.py`` and nothing
    else — no schema edit, no migration, and old rows stay readable.

    An empty ``choices`` means the defaults, which are the aiming the prompts
    hardcoded before they were configurable.
    """
    choices: Dict[str, OptionChoice] = Field(
        default_factory=dict,
        description="Chosen options by key; see nlp_pillars/podcast_options.py",
    )


class GroundPackCallRecord(BaseModel):
    """Which model produced one Ground Pack section.

    Stored per section so a quality shift is attributable to provider or
    prompt, not guessed from logs later. ``fallback`` is True when DeepSeek
    failed and Claude answered that call instead.
    """
    section: str = Field(..., description="facts_outline | core_concepts | metrics_datasets | limitations")
    provider: str = Field(..., description="deepseek or anthropic")
    model: str = Field(..., description="Model id that produced the section")
    fallback: bool = Field(default=False, description="True when Claude replaced a failed DeepSeek call")
    fallback_reason: Optional[str] = Field(
        None,
        description="Why DeepSeek was skipped, when fallback is True",
    )
    input_tokens: Optional[int] = Field(None, description="Prompt tokens, when reported by the provider")
    output_tokens: Optional[int] = Field(None, description="Completion tokens, when reported by the provider")
    finish_reason: Optional[str] = Field(None, description="Provider stop reason, e.g. stop or length")


class AudioMetadata(BaseModel):
    """One generated MP3 and how it was produced.

    Follows the same JSONB-on-podcast_scripts pattern as source_material,
    options, and ground_pack_calls.
    """
    engine: Optional[str] = Field(None, description="TTS engine id, e.g. indextts")
    voice_path: Optional[str] = Field(None, description="Library-relative voice file")
    voice_label: Optional[str] = Field(None, description="Display label for the voice")
    file_name: Optional[str] = Field(None, description="MP3 filename under podcast_audio/")
    file_path: Optional[str] = Field(None, description="Absolute path inside the container")
    duration_seconds: Optional[float] = Field(None, description="Final audio duration")
    generated_at: Optional[datetime] = Field(None, description="When audio was generated")
    chunk_count: Optional[int] = Field(None, description="Number of synthesis chunks")

    @property
    def has_audio(self) -> bool:
        return bool(self.file_name)


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
    ground_pack_calls: Dict[str, GroundPackCallRecord] = Field(
        default_factory=dict,
        description="Per-section model provenance; see GroundPackCallRecord",
    )
    source_material: SourceMaterial = Field(
        default_factory=SourceMaterial,
        description="What the script was written from; see SourceMaterial",
    )
    options: PodcastOptions = Field(
        default_factory=PodcastOptions,
        description="What the script was aimed at; see PodcastOptions",
    )
    audio_metadata: AudioMetadata = Field(
        default_factory=AudioMetadata,
        description="Generated episode audio; see AudioMetadata",
    )
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
    # Optional because review_quiz_card_fsrs() builds this response with
    # card=None on its failure branches (card not found, update failed).
    card: Optional[QuizCard] = Field(default=None, description="Updated card with new FSRS parameters")
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


# ==========================================
# Pipeline run tracking (migration 009)
# ==========================================


class StageName(str, Enum):
    """The stage boundaries that already exist in orchestrator.py.

    These are names for log lines that were always there — the comment on each
    member is where it lives, so a rename in one place is easy to follow to the
    other. `seq` in the database is the position within the *run kind's* stage
    list, 1-based, not this enum's declaration order.

    The DISCOVER_* members belong to `run_discovery_with_selection`, which is a
    different pipeline from `run_daily`: its `DISCOVER_QUERIES` is the same work
    `DISCOVERY` does for the daily run, but the surrounding steps are not, and one
    shared member would put the wrong label on whichever run it was not written for.
    """

    DISCOVERY = "discovery"      # Step 1  run_daily
    SEARCH = "search"            # Step 2  run_daily
    ENQUEUE = "enqueue"          # Step 3  run_daily
    POP_QUEUE = "pop_queue"      # Step 4  run_daily
    PROCESS = "process"          # Step 5  run_daily
    INGEST = "ingest"            # Step 5a _process_paper
    SUMMARIZE = "summarize"      # Step 5b _process_paper
    SYNTHESIZE = "synthesize"    # Step 5c _process_paper
    QUIZ = "quiz"                # Step 5d _process_paper
    PERSIST = "persist"          # Step 5e _process_paper
    VECTORS = "vectors"          # Step 5f _process_paper

    # run_discovery_with_selection, in execution order.
    DISCOVER_CONTEXT = "discover_context"          # db.get_recent_notes
    DISCOVER_QUERIES = "discover_queries"          # DiscoveryAgent.run
    DISCOVER_VECTORS = "discover_vectors"          # _search_vectors
    DISCOVER_ARXIV = "discover_arxiv"              # _search_arxiv_candidates
    DISCOVER_S2 = "discover_semantic_scholar"      # _search_semantic_scholar
    DISCOVER_CITATIONS = "discover_citations"      # _search_citations (conditional)
    DISCOVER_RANK = "discover_rank"                # _rank_and_dedupe

    # podcast audio generation (IndexTTS), in execution order.
    TTS_PREPARE = "tts_prepare"
    TTS_SYNTHESIZE = "tts_synthesize"
    TTS_ASSEMBLE = "tts_assemble"
    TTS_ENCODE = "tts_encode"
    TTS_SAVE = "tts_save"


#: Stages for a full `run_daily`, in order. Index + 1 is the stored `seq`.
#:
#: Spelled out rather than `list(StageName)`, which is what it used to be: that made
#: every future enum member a silent extra stage on every daily run, seeded pending
#: and never written to. The DISCOVER_* members below would have been exactly that.
RUN_DAILY_STAGES: List[StageName] = [
    StageName.DISCOVERY,
    StageName.SEARCH,
    StageName.ENQUEUE,
    StageName.POP_QUEUE,
    StageName.PROCESS,
    StageName.INGEST,
    StageName.SUMMARIZE,
    StageName.SYNTHESIZE,
    StageName.QUIZ,
    StageName.PERSIST,
    StageName.VECTORS,
]

#: Stages for `process_selected_papers`, which has no discovery/search/enqueue/pop
#: phase — the papers are already chosen, so it starts at PROCESS.
PROCESS_SELECTED_STAGES: List[StageName] = [
    StageName.PROCESS,
    StageName.INGEST,
    StageName.SUMMARIZE,
    StageName.SYNTHESIZE,
    StageName.QUIZ,
    StageName.PERSIST,
    StageName.VECTORS,
]


#: Stages for `run_discovery_with_selection`, in order.
#:
#: DISCOVER_CITATIONS is seeded like the rest and then DELETED by the orchestrator when
#: the pillar has no recent papers to follow citations from — `_search_citations` is
#: guarded by `if recent_paper_ids:`, and a step that will not run has no business
#: sitting in the list looking pending. Deleting it rather than skipping it keeps the
#: seq of DISCOVER_RANK stable.
DISCOVER_STAGES: List[StageName] = [
    StageName.DISCOVER_CONTEXT,
    StageName.DISCOVER_QUERIES,
    StageName.DISCOVER_VECTORS,
    StageName.DISCOVER_ARXIV,
    StageName.DISCOVER_S2,
    StageName.DISCOVER_CITATIONS,
    StageName.DISCOVER_RANK,
]

PODCAST_AUDIO_STAGES: List[StageName] = [
    StageName.TTS_PREPARE,
    StageName.TTS_SYNTHESIZE,
    StageName.TTS_ASSEMBLE,
    StageName.TTS_ENCODE,
    StageName.TTS_SAVE,
]


class RunStatus(str, Enum):
    """Lifecycle of a pipeline run.

    INTERRUPTED is deliberately distinct from FAILED: it is written only by the
    startup sweep and means "the process died, we do not know what happened".
    FAILED claims knowledge the sweep does not have.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


#: Statuses that mean the run is over. Polling stops here.
TERMINAL_RUN_STATUSES = frozenset({
    RunStatus.SUCCEEDED.value,
    RunStatus.FAILED.value,
    RunStatus.CANCELLED.value,
    RunStatus.INTERRUPTED.value,
})


class StageStatus(str, Enum):
    """Lifecycle of one stage within a run.

    DROPPED is the odd one out and is deliberately NOT a stored value — the
    pipeline_run_stages CHECK constraint does not admit it. It is a signal on the
    on_stage sink meaning "this run has decided this step will not happen, remove its
    row", which run_service turns into a DELETE. Discovery uses it for the citation
    step, which is conditional on the pillar having recent papers: a step that will
    not run should not sit on screen looking pending, and 'skipped' would still render
    it as a step that was considered and passed over.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    DROPPED = "dropped"


class PipelineRunStage(BaseModel):
    """One stage of a run. Rows are seeded PENDING when the run is created."""

    id: Optional[str] = None
    run_id: str
    seq: int
    name: str
    status: str = StageStatus.PENDING.value
    detail: Optional[str] = Field(None, description="Free text, e.g. '2/5 papers'")
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class PipelineRun(BaseModel):
    """One pipeline execution, from any trigger."""

    id: str
    pillar_id: str
    trigger_source: str = Field(..., description="ui_pipeline | ui_select | scheduler")
    kind: str = Field(..., description="run_daily | process_selected")
    status: str = RunStatus.PENDING.value
    current_stage: Optional[str] = None
    papers_processed: int = 0
    papers_failed: int = 0
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    heartbeat_at: Optional[datetime] = None
    stages: List[PipelineRunStage] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Terminal payload for run kinds that produce one. A 'discover' run stores "
            "{'candidates': [...], 'sources_used': [...]}; the others leave it None."
        ),
    )
