"""Summarizer Agent - Real LLM implementation using instructor + OpenAI.

Extracts structured information from research papers into PaperNote objects.
"""

import logging
from typing import List, Optional

import instructor
from openai import OpenAI
from pydantic import ValidationError

from ..config import get_settings
from ..schemas import PaperNote, SummarizerInput
from .discovery_agent import SystemPromptGenerator

logger = logging.getLogger(__name__)


class SummarizerValidationError(Exception):
    """Raised when summarization fails validation."""

    pass


class SummarizerAgentImpl:
    """Instance-based implementation for summarizing research papers.

    Uses instructor + OpenAI for structured output extraction.
    Tests create their own instances with mocked clients.
    """

    def __init__(self, client, model: str):
        """Initialize the summarizer agent.

        Args:
            client: Instructor-wrapped OpenAI client
            model: Model name to use for completions

        """
        self.client = client
        self.model = model

        # System prompt - MUST include strings verified by tests (lines 248-252)
        self.system_prompt = SystemPromptGenerator(
            background=[
                "You are an NLP research summarizer; faithful, structured.",
                "Cite only what's supported by the text; avoid hallucinations.",
                "You have deep expertise in NLP, machine learning, and linguistics."
            ],
            steps=[
                "Extract: problem, method, findings, limitations, future_work, key_terms",
                "Focus on specific metrics and results when mentioned",
                "Identify technical innovations and contributions"
            ],
            output_instructions=[
                "Be concise but precise; prefer bullet points for lists",
                "Return valid PaperNote JSON only (no extra text)",
                "You must return a valid PaperNote JSON object",
                "Limit findings and limitations to 3-5 items each"
            ]
        )

    def run(self, input_data: SummarizerInput) -> PaperNote:
        """Run summarization with retry logic.

        Args:
            input_data: SummarizerInput with parsed paper and context

        Returns:
            PaperNote with extracted information

        Raises:
            SummarizerValidationError: If summarization fails after retry

        """
        logger.info(f"Starting summarization for paper: {input_data.parsed_paper.paper_ref.title}")

        try:
            result = self._attempt_summarization(input_data, is_retry=False)
            logger.info("Summarization completed successfully on first attempt")
            return result
        except ValidationError as e:
            logger.warning(f"Validation error on first attempt: {e}")
            logger.info("Attempting retry with corrective message")
            try:
                result = self._attempt_summarization(input_data, is_retry=True, error=e)
                logger.info("Summarization completed successfully on retry")
                return result
            except ValidationError as e2:
                logger.error("Failed to generate valid PaperNote after retry")
                raise SummarizerValidationError(
                    f"Failed to generate valid PaperNote after retry.\n"
                    f"Original error: {e}\n"
                    f"Retry error: {e2}"
                )
        except Exception as e:
            raise SummarizerValidationError(f"Instructor completion failed: {e}")

    def _attempt_summarization(
        self,
        input_data: SummarizerInput,
        is_retry: bool = False,
        error: Optional[ValidationError] = None
    ) -> PaperNote:
        """Single attempt at summarization.

        Args:
            input_data: Input data for summarization
            is_retry: Whether this is a retry attempt
            error: Previous validation error if retrying

        Returns:
            PaperNote with extracted information

        """
        system_message = self._build_system_message()
        user_message = self._build_user_message(input_data)

        if is_retry and error:
            user_message += f"\n\nYour last output was invalid JSON for PaperNote. Please fix: {error}"

        result = self.client.chat.completions.create(
            model=self.model,
            response_model=PaperNote,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )

        # Set paper_id and pillar_id from input
        result.paper_id = input_data.parsed_paper.paper_ref.id
        result.pillar_id = input_data.pillar_id

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

    def _build_user_message(self, input_data: SummarizerInput) -> str:
        """Build user message with paper content.

        Args:
            input_data: Input data containing parsed paper

        Returns:
            Formatted user message string

        """
        paper = input_data.parsed_paper
        ref = paper.paper_ref

        # Truncate content to ~8000 chars (test requirement line 269)
        content = paper.full_text[:8000]
        if len(paper.full_text) > 8000:
            content += "\n[Content truncated...]"

        parts = [
            f"Title: {ref.title}",
            f"Authors: {', '.join(ref.authors) if ref.authors else 'Unknown'}",
            f"Year: {ref.year or 'Unknown'}",
            f"Abstract: {ref.abstract or 'No abstract available'}",
            "",
            "Paper Content:",
            content
        ]

        # Add recent notes context if provided
        if input_data.recent_notes:
            parts.append("\nRecent paper summaries for consistency:")
            for note in input_data.recent_notes:
                parts.append(f"- {note}")

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


# Module-level singleton for backward compatibility with orchestrator
SummarizerAgent = None
try:
    settings = get_settings()
    if settings.openai_api_key:
        SummarizerAgent = SummarizerAgentImpl(_make_client(), settings.default_model)
except Exception:
    pass  # Allow import without API key (for testing)


def summarize(
    parsed_paper,
    pillar_id: str,
    recent_notes: Optional[List[str]] = None
) -> PaperNote:
    """Convenience function for summarization.

    Args:
        parsed_paper: ParsedPaper object with paper content
        pillar_id: Target pillar ID
        recent_notes: Optional list of recent paper summaries for context

    Returns:
        PaperNote with extracted information

    Raises:
        ValueError: If SummarizerAgent is not initialized

    """
    if SummarizerAgent is None:
        raise ValueError("SummarizerAgent is not initialized")
    input_data = SummarizerInput(
        parsed_paper=parsed_paper,
        pillar_id=pillar_id,
        recent_notes=recent_notes or []
    )
    return SummarizerAgent.run(input_data)
