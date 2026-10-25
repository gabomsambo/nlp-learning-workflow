"""Synthesis Agent - turns paper notes into an educational lesson, with Claude.

The lesson used to be written from the note alone, which measured worse than
almost anything else about it: a lesson built from a summary fills gaps from
general knowledge and repeats the summary's mistakes. The paper's own text is
now passed alongside the note, and the prompt names the paper, so the model
does not list it as related work.

The call goes through :mod:`nlp_pillars.llm_client`: Claude's native
structured output, a small model-facing schema (``LessonDraft``), and no
sampling parameters — current Claude models reject ``temperature`` with a
400.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from ..config import get_settings
from ..llm_client import StructuredCall, fit_paper_text, structured
from ..llm_routes import SYNTHESIZE_ROUTE
from ..schemas import Lesson, LessonDraft, SynthesisInput, lesson_from_draft
from .discovery_agent import SystemPromptGenerator

logger = logging.getLogger(__name__)


class SynthesisValidationError(Exception):
    """Raised when synthesis fails validation."""

    pass


class SynthesisAgentImpl:
    """Instance-based implementation for synthesizing educational lessons.

    Uses Claude's native structured output. Tests create their own instances
    with a mocked client.
    """

    def __init__(self, client, model: Optional[str] = None):
        """Initialize the synthesis agent.

        Args:
            client: An Anthropic client with a ``messages.parse`` method.
            model: Model id override. ``None`` means the synthesize route's
                resolved model.

        """
        self.client = client
        self.model = model
        #: The last call this agent made, for the run's stage detail.
        self.last_call: Optional[StructuredCall] = None

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

        One call, no hand-rolled retry. The helper falls back to the route's
        fallback model when the call fails outright.

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
            raise SynthesisValidationError(f"Claude completion failed: {e}") from e

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

        call = structured(
            SYNTHESIZE_ROUTE,
            system=system_message,
            user=user_message,
            out_model=LessonDraft,
            client=self.client,
            model=self.model,
        )
        self.last_call = call

        # Identity and the storage timestamp come from this process. A date on
        # the model object is the one the model invented, and it is not kept.
        result = lesson_from_draft(
            call.parsed,
            input_data.paper_note.paper_id,
            input_data.paper_note.pillar_id,
        )
        result.created_at = datetime.now(timezone.utc)

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

        # Name the paper. Without this the model treats the paper it was
        # given as related work, and writes a lesson about the wrong paper.
        title = input_data.paper_title or ""
        if title:
            authors = ", ".join(input_data.paper_authors or [])
            parts.insert(
                0,
                f"This paper is: {title}"
                + (f" by {authors}" if authors else "")
                + ". It is the paper you are writing about, not related work.",
            )

        # The paper's own text, when there is one. A lesson written from the
        # note alone fills gaps from general knowledge.
        full_text = (input_data.full_text or "").strip()
        if full_text:
            parts.extend(
                [
                    "",
                    "Paper Text:",
                    fit_paper_text(full_text),
                ]
            )

        # Add related lessons context if provided
        if input_data.related_lessons:
            parts.append("\nRelated lessons for context:")
            for lesson in input_data.related_lessons[:3]:  # Limit to 3 for context
                parts.append(f"- {lesson.title}: {lesson.tl_dr}")

        parts.append("\nCreate an educational lesson based on this paper summary.")

        return "\n".join(parts)


def _make_client():
    """Create the Anthropic client.

    Returns:
        An Anthropic client

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set

    """
    from ..llm_client import make_client

    return make_client()


class _LazySynthesisAgent:
    """Lazy stand-in for the module-level ``SynthesisAgent`` singleton.

    See ``summarizer_agent._LazySummarizerAgent`` for why this exists.
    """

    def __init__(self):
        self._impl: Optional[SynthesisAgentImpl] = None

    def _get_impl(self) -> SynthesisAgentImpl:
        if self._impl is None:
            settings = get_settings()
            if not settings.anthropic_api_key:
                raise ValueError(
                    "SynthesisAgent is not initialized: "
                    "ANTHROPIC_API_KEY environment variable is required"
                )
            self._impl = SynthesisAgentImpl(_make_client())
        return self._impl

    @property
    def last_call(self) -> Optional[StructuredCall]:
        """The most recent call, for the run's stage detail."""
        return self._impl.last_call if self._impl is not None else None

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
