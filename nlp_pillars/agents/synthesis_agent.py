"""Synthesis Agent - Real LLM implementation using instructor + OpenAI.

Converts paper notes into educational lessons with takeaways and practice ideas.
"""

import logging
from typing import List, Optional

import instructor
from openai import OpenAI

from ..config import get_settings
from ..schemas import Lesson, SynthesisInput
from .discovery_agent import SystemPromptGenerator

logger = logging.getLogger(__name__)


class SynthesisValidationError(Exception):
    """Raised when synthesis fails validation."""

    pass


class SynthesisAgentImpl:
    """Instance-based implementation for synthesizing educational lessons.

    Uses instructor + OpenAI for structured output generation.
    Tests create their own instances with mocked clients.
    """

    def __init__(self, client, model: str):
        """Initialize the synthesis agent.

        Args:
            client: Instructor-wrapped OpenAI client
            model: Model name to use for completions

        """
        self.client = client
        self.model = model

        # System prompt for educational synthesis
        self.system_prompt = SystemPromptGenerator(
            background=[
                "You are an expert educational content synthesizer for graduate-level NLP researchers.",
                "You transform research paper notes into clear, actionable learning content.",
                "You focus on practical understanding and real-world applications."
            ],
            steps=[
                "Analyze the paper's problem, method, and findings",
                "Create a concise TL;DR summary capturing the key insight",
                "Extract 3-5 actionable takeaways for researchers",
                "Suggest 2-3 practical exercises or implementation ideas",
                "Identify connections to related work and concepts"
            ],
            output_instructions=[
                "Write for graduate students studying NLP/ML",
                "Be concise but comprehensive in your synthesis",
                "Focus on what the reader can learn and apply",
                "Return valid Lesson JSON only (no extra text)",
                "Limit takeaways to 3-5 items, practice ideas to 2-3 items"
            ]
        )

    def run(self, input_data: SynthesisInput) -> Lesson:
        """Run synthesis.

        Retries on invalid model output are handled by instructor itself, which
        re-prompts with the validation feedback up to ``max_retries`` times
        (default 3) before raising. There is deliberately no retry layer here.

        Args:
            input_data: SynthesisInput with paper note and pillar config

        Returns:
            Lesson with synthesized educational content

        Raises:
            SynthesisValidationError: If synthesis fails

        """
        logger.info(f"Starting synthesis for paper: {input_data.paper_note.paper_id}")

        try:
            result = self._attempt_synthesis(input_data)
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise SynthesisValidationError(f"Instructor completion failed: {e}") from e

        logger.info("Synthesis completed successfully")
        return result

    def _attempt_synthesis(self, input_data: SynthesisInput) -> Lesson:
        """Single call to the model.

        Args:
            input_data: Input data for synthesis

        Returns:
            Lesson with synthesized content

        """
        system_message = self._build_system_message()
        user_message = self._build_user_message(input_data)

        result = self.client.chat.completions.create(
            model=self.model,
            response_model=Lesson,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )

        # Set paper_id and pillar_id from input
        result.paper_id = input_data.paper_note.paper_id
        result.pillar_id = input_data.paper_note.pillar_id

        return result

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

    def _build_user_message(self, input_data: SynthesisInput) -> str:
        """Build user message with paper note and context.

        Args:
            input_data: Input data containing paper note and pillar config

        Returns:
            Formatted user message string

        """
        note = input_data.paper_note
        pillar = input_data.pillar_config

        parts = [
            f"Pillar: {pillar.name}",
            f"Learning Goal: {pillar.goal}",
            f"Focus Areas: {', '.join(pillar.focus_areas) if pillar.focus_areas else 'General'}",
            "",
            "Paper Summary:",
            f"Problem: {note.problem}",
            f"Method: {note.method}",
            f"Key Findings: {'; '.join(note.findings)}",
            f"Limitations: {'; '.join(note.limitations)}",
            f"Future Work: {'; '.join(note.future_work)}",
            f"Key Terms: {', '.join(note.key_terms)}"
        ]

        # Add related lessons context if provided
        if input_data.related_lessons:
            parts.append("\nRelated lessons for context:")
            for lesson in input_data.related_lessons[:3]:  # Limit to 3 for context
                parts.append(f"- {lesson.title}: {lesson.tl_dr}")

        parts.append("\nCreate an educational lesson based on this paper summary.")

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


class _LazySynthesisAgent:
    """Lazy stand-in for the module-level ``SynthesisAgent`` singleton.

    See ``summarizer_agent._LazySummarizerAgent`` for why this exists.
    """

    def __init__(self):
        self._impl: Optional[SynthesisAgentImpl] = None

    def _get_impl(self) -> SynthesisAgentImpl:
        if self._impl is None:
            settings = get_settings()
            if not settings.openai_api_key:
                raise ValueError(
                    "SynthesisAgent is not initialized: "
                    "OPENAI_API_KEY environment variable is required"
                )
            self._impl = SynthesisAgentImpl(_make_client(), settings.default_model)
        return self._impl

    def run(self, input_data: SynthesisInput) -> Lesson:
        """Synthesize, constructing the underlying agent on first use."""
        return self._get_impl().run(input_data)


# Module-level singleton for backward compatibility with orchestrator
SynthesisAgent = _LazySynthesisAgent()


def synthesize(
    paper_note,
    pillar_config,
    related_lessons: Optional[List] = None
) -> Lesson:
    """Convenience function for synthesis.

    Args:
        paper_note: PaperNote object with paper summary
        pillar_config: PillarConfig for the target pillar
        related_lessons: Optional list of related Lesson objects for context

    Returns:
        Lesson with synthesized educational content

    Raises:
        ValueError: If SynthesisAgent is not initialized

    """
    if SynthesisAgent is None:
        raise ValueError("SynthesisAgent is not initialized")
    input_data = SynthesisInput(
        paper_note=paper_note,
        pillar_config=pillar_config,
        related_lessons=related_lessons or []
    )
    return SynthesisAgent.run(input_data)
