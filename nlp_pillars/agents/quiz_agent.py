"""Quiz Agent - Real LLM implementation using instructor + OpenAI.

Generates varied quiz cards (FACTUAL/CONCEPTUAL/APPLICATION) for spaced repetition.
"""

import logging
from typing import List, Optional

import instructor
from openai import OpenAI

from ..config import get_settings
from ..schemas import DifficultyLevel, QuizCard, QuizGeneratorInput
from .discovery_agent import SystemPromptGenerator

logger = logging.getLogger(__name__)


class QuizValidationError(Exception):
    """Raised when quiz generation fails validation."""

    pass


class QuizAgentImpl:
    """Instance-based implementation for generating quiz cards.

    Uses instructor + OpenAI for structured output generation.
    Tests create their own instances with mocked clients.
    """

    def __init__(self, client, model: str):
        """Initialize the quiz agent.

        Args:
            client: Instructor-wrapped OpenAI client
            model: Model name to use for completions

        """
        self.client = client
        self.model = model

        # System prompt for quiz generation
        self.system_prompt = SystemPromptGenerator(
            background=[
                "You are an expert quiz generator for graduate-level NLP researchers.",
                "You create varied quiz questions that test understanding at multiple levels.",
                "You ensure questions are clear, answerable, and educational."
            ],
            steps=[
                "Analyze the paper's key concepts, findings, and implications",
                "Generate FACTUAL questions testing specific facts and metrics",
                "Generate CONCEPTUAL questions testing understanding of ideas",
                "Generate APPLICATION questions testing ability to apply concepts",
                "Ensure questions span different difficulty levels"
            ],
            output_instructions=[
                "Create questions appropriate for graduate students",
                "Each question should have a clear, complete answer",
                "Vary difficulty: some easy (direct recall), some medium (understanding), some hard (synthesis)",
                "Return valid QuizCard JSON array only (no extra text)",
                "Generate at least the requested number of questions"
            ]
        )

    def run(self, input_data: QuizGeneratorInput) -> List[QuizCard]:
        """Run quiz generation and post-processing.

        Retries on invalid model output are handled by instructor itself, which
        re-prompts with the validation feedback up to ``max_retries`` times
        (default 3) before raising. There is deliberately no retry layer here.

        ``question_type`` is whatever the model determined; only ``difficulty``
        is reassigned, to honour the caller's requested ``difficulty_mix``.

        Args:
            input_data: QuizGeneratorInput with paper note and generation params

        Returns:
            List of QuizCard

        Raises:
            QuizValidationError: If quiz generation fails

        """
        logger.info(f"Starting quiz generation for paper: {input_data.paper_note.paper_id}")
        logger.info(f"Requested {input_data.num_questions} questions with mix: {input_data.difficulty_mix}")

        try:
            cards = self._attempt_quiz_generation(input_data)
        except Exception as e:
            logger.error(f"Quiz generation failed: {e}")
            raise QuizValidationError(f"Instructor completion failed: {e}") from e

        # Apply the caller's requested difficulty mix. The prompt asks for the
        # mix but the model does not reliably honour it, and difficulty seeds
        # FSRS scheduling, so the requested mix is enforced here.
        cards = self._apply_difficulty_mix(cards, input_data.difficulty_mix)

        # Enforce exact count - the prompt asks for at least num_questions.
        cards = cards[:input_data.num_questions]

        logger.info(f"Generated {len(cards)} quiz cards")
        return cards

    def _attempt_quiz_generation(self, input_data: QuizGeneratorInput) -> List[QuizCard]:
        """Single call to the model.

        Args:
            input_data: Input data for quiz generation

        Returns:
            List of QuizCard objects

        """
        system_message = self._build_system_message()
        user_message = self._build_user_message(input_data)

        result = self.client.chat.completions.create(
            model=self.model,
            response_model=List[QuizCard],
            temperature=0.3,  # Slightly higher for variety
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )

        # Set paper_id and pillar_id for all cards
        for card in result:
            card.paper_id = input_data.paper_note.paper_id
            card.pillar_id = input_data.paper_note.pillar_id

        return result

    def _apply_difficulty_mix(
        self,
        cards: List[QuizCard],
        difficulty_mix: dict
    ) -> List[QuizCard]:
        """Apply the requested difficulty mix to cards.

        Args:
            cards: List of generated quiz cards
            difficulty_mix: Dict with 'easy', 'medium', 'hard' counts

        Returns:
            Cards with adjusted difficulty levels

        """
        easy_count = difficulty_mix.get("easy", 0)
        medium_count = difficulty_mix.get("medium", 0)
        # Note: difficulty_mix["hard"] is implicit - anything past the easy and
        # medium allotments becomes HARD.

        for i, card in enumerate(cards):
            if i < easy_count:
                card.difficulty = DifficultyLevel.EASY
            elif i < easy_count + medium_count:
                card.difficulty = DifficultyLevel.MEDIUM
            else:
                card.difficulty = DifficultyLevel.HARD

        return cards

    def _build_system_message(self) -> str:
        """Build system message from SystemPromptGenerator.

        Returns:
            Formatted system message string

        """
        parts = []
        if self.system_prompt.background:
            parts.append("Background:\n" + "\n".join(f"- {b}" for b in self.system_prompt.background))
        if self.system_prompt.steps:
            parts.append("Steps:\n" + "\n".join(f"- {s}" for s in self.system_prompt.steps))
        if self.system_prompt.output_instructions:
            parts.append("Output:\n" + "\n".join(f"- {o}" for o in self.system_prompt.output_instructions))
        return "\n\n".join(parts)

    def _build_user_message(self, input_data: QuizGeneratorInput) -> str:
        """Build user message with paper note and generation params.

        Args:
            input_data: Input data containing paper note and params

        Returns:
            Formatted user message string

        """
        note = input_data.paper_note

        parts = [
            "Paper Summary:",
            f"Problem: {note.problem}",
            f"Method: {note.method}",
            f"Key Findings: {'; '.join(note.findings)}",
            f"Limitations: {'; '.join(note.limitations)}",
            f"Future Work: {'; '.join(note.future_work)}",
            f"Key Terms: {', '.join(note.key_terms)}",
            "",
            f"Generate {input_data.num_questions} quiz questions with this difficulty mix:",
            f"- Easy (direct recall): {input_data.difficulty_mix.get('easy', 0)}",
            f"- Medium (understanding): {input_data.difficulty_mix.get('medium', 0)}",
            f"- Hard (synthesis/application): {input_data.difficulty_mix.get('hard', 0)}",
            "",
            "Include a mix of question types:",
            "- FACTUAL: Tests specific facts, metrics, or definitions",
            "- CONCEPTUAL: Tests understanding of ideas and relationships",
            "- APPLICATION: Tests ability to apply concepts to new situations"
        ]

        return "\n".join(parts)


def _make_client():
    """Create instructor-wrapped OpenAI client.

    Returns:
        Instructor-wrapped OpenAI client

    Raises:
        ValueError: If OPENAI_API_KEY is not set

    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return instructor.from_openai(OpenAI(api_key=settings.openai_api_key))


class _LazyQuizAgent:
    """Lazy stand-in for the module-level ``QuizAgent`` singleton.

    See ``summarizer_agent._LazySummarizerAgent`` for why this exists.
    """

    def __init__(self):
        self._impl: Optional[QuizAgentImpl] = None

    def _get_impl(self) -> QuizAgentImpl:
        if self._impl is None:
            settings = get_settings()
            if not settings.openai_api_key:
                raise ValueError(
                    "QuizAgent is not initialized: "
                    "OPENAI_API_KEY environment variable is required"
                )
            self._impl = QuizAgentImpl(_make_client(), settings.default_model)
        return self._impl

    def run(self, input_data: QuizGeneratorInput) -> List[QuizCard]:
        """Generate quiz cards, constructing the underlying agent on first use."""
        return self._get_impl().run(input_data)


# Module-level singleton for backward compatibility with orchestrator
QuizAgent = _LazyQuizAgent()


def generate_quiz(
    paper_note,
    num_questions: int = 5,
    difficulty_mix: Optional[dict] = None
) -> List[QuizCard]:
    """Convenience function for quiz generation.

    Args:
        paper_note: PaperNote object with paper summary
        num_questions: Number of questions to generate (1-10)
        difficulty_mix: Dict with 'easy', 'medium', 'hard' counts

    Returns:
        List of QuizCard with generated questions

    Raises:
        ValueError: If QuizAgent is not initialized

    """
    if QuizAgent is None:
        raise ValueError("QuizAgent is not initialized")

    if difficulty_mix is None:
        difficulty_mix = {"easy": 2, "medium": 2, "hard": 1}

    input_data = QuizGeneratorInput(
        paper_note=paper_note,
        num_questions=num_questions,
        difficulty_mix=difficulty_mix
    )
    return QuizAgent.run(input_data)
