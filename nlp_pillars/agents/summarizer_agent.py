"""Summarizer Agent - Real LLM implementation using instructor + OpenAI.

Extracts structured information from research papers into PaperNote objects.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

import instructor
from openai import OpenAI

from ..config import get_settings
from ..concept_cost import call_from_completion, call_from_exception, require_price
from ..concept_extraction import ConceptExtractionError, normalize_concepts
from ..schemas import PaperNote, SummarizerInput
from .discovery_agent import SystemPromptGenerator

logger = logging.getLogger(__name__)


class SummarizerValidationError(Exception):
    """Raised when summarization fails validation."""

    def __init__(self, message: str, *, call=None):
        super().__init__(message)
        self.call = call


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
                "Extract: problem, method, findings, limitations, future_work, key_terms, concepts, proposed_concepts",
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
        """Run summarization.

        Retries on invalid model output are handled by instructor itself, which
        re-prompts with the validation feedback up to ``max_retries`` times
        (default 3) before raising. There is deliberately no retry layer here.

        Args:
            input_data: SummarizerInput with parsed paper and context

        Returns:
            PaperNote with extracted information

        Raises:
            SummarizerValidationError: If summarization fails

        """
        logger.info(f"Starting summarization for paper: {input_data.parsed_paper.paper_ref.title}")

        require_price(self.model)
        try:
            result = self._attempt_summarization(input_data)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            call = getattr(e, "call", None) or call_from_exception(
                purpose="summarizer",
                paper_id=input_data.parsed_paper.paper_ref.id,
                model=self.model,
                error=e,
            )
            raise SummarizerValidationError(
                f"Instructor completion failed: {e}", call=call
            ) from e

        logger.info("Summarization completed successfully")
        return result

    def _attempt_summarization(self, input_data: SummarizerInput) -> PaperNote:
        """Single call to the model.

        Args:
            input_data: Input data for summarization

        Returns:
            PaperNote with extracted information

        """
        system_message = self._build_system_message()
        user_message = self._build_user_message(input_data)

        result, completion = self.client.chat.completions.create_with_completion(
            model=self.model,
            response_model=PaperNote,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )

        # Identity and the storage timestamp come from this process. A date on
        # the model object is the one the model invented, and it is not kept.
        result.paper_id = input_data.parsed_paper.paper_ref.id
        result.pillar_id = input_data.pillar_id
        result.created_at = datetime.now(timezone.utc)

        call = call_from_completion(
            purpose="summarizer",
            paper_id=result.paper_id,
            model=self.model,
            completion=completion,
        )
        try:
            concepts, proposed = normalize_concepts(
                result.concepts,
                result.proposed_concepts,
                input_data.concept_vocabulary,
            )
        except ConceptExtractionError as exc:
            exc.call = call
            raise
        result.concepts = concepts
        result.proposed_concepts = proposed
        result.concept_call = call.as_dict()

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

        vocabulary = input_data.concept_vocabulary
        parts.extend(
            [
                "",
                "Concept vocabulary (copy exact names into concepts):",
                json.dumps(vocabulary, ensure_ascii=False),
                (
                    "Choose at most 8 concepts from that vocabulary. Prefer an "
                    "existing name whenever it fits. If an important concept has "
                    "no match, put a concise canonical name in proposed_concepts; "
                    "at most 2. Never put a new name in concepts."
                ),
            ]
        )

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


class _LazySummarizerAgent:
    """Lazy stand-in for the module-level ``SummarizerAgent`` singleton.

    ``orchestrator`` and ``upload_service`` do ``from ... import SummarizerAgent``
    at import time, which binds by value. Building the real agent eagerly meant a
    missing API key produced ``AttributeError: 'NoneType' object has no attribute
    'run'`` at call time. This defers construction to the first ``run()`` so the
    failure names the actual problem, and keeps importing this module free of
    OpenAI client construction.
    """

    def __init__(self):
        self._impl: Optional[SummarizerAgentImpl] = None

    def _get_impl(self) -> SummarizerAgentImpl:
        if self._impl is None:
            settings = get_settings()
            if not settings.openai_api_key:
                raise ValueError(
                    "SummarizerAgent is not initialized: "
                    "OPENAI_API_KEY environment variable is required"
                )
            self._impl = SummarizerAgentImpl(_make_client(), settings.default_model)
        return self._impl

    def run(self, input_data: SummarizerInput) -> PaperNote:
        """Summarize, constructing the underlying agent on first use."""
        return self._get_impl().run(input_data)


# Module-level singleton for backward compatibility with orchestrator
SummarizerAgent = _LazySummarizerAgent()


def summarize(
    parsed_paper,
    pillar_id: str,
    recent_notes: Optional[List[str]] = None,
    concept_vocabulary: Optional[List[str]] = None,
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
        recent_notes=recent_notes or [],
        concept_vocabulary=concept_vocabulary or [],
    )
    return SummarizerAgent.run(input_data)
