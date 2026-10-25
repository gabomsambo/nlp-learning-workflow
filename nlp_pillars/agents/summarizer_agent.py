"""Summarizer Agent - extracts structured notes from a paper with Claude.

The agent used to reach OpenAI through ``instructor``. It now calls Claude's
native structured output through :mod:`nlp_pillars.llm_client`, with a small
model-facing schema (``NoteDraft``) rather than the stored ``PaperNote``.

Two consequences worth knowing:

- **No sampling parameters.** Current Claude models reject ``temperature``
  with a 400, so there is none to set.
- **The model fills in content, not identity.** ``NoteDraft`` has no
  ``paper_id``, ``pillar_id`` or ``created_at``: those come from this
  process. The old schema had all three as required fields, so the model was
  asked for a paper id it could not know.

The whole paper is sent, not the first 8,000 characters. Measured on the same
prompts, the truncated version made up more future-work items than any other
change made, because the model's conclusion section was not in the window.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

from ..config import get_settings
from ..concept_extraction import ConceptExtractionError, normalize_concepts
from ..llm_client import StructuredCall, fit_paper_text, structured
from ..llm_routes import SUMMARIZE_ROUTE
from ..schemas import NoteDraft, PaperNote, SummarizerInput, note_from_draft
from .discovery_agent import SystemPromptGenerator

logger = logging.getLogger(__name__)


class SummarizerValidationError(Exception):
    """Raised when summarization fails validation."""

    def __init__(self, message: str, *, call=None):
        super().__init__(message)
        self.call = call


class SummarizerAgentImpl:
    """Instance-based implementation for summarizing research papers.

    Uses Claude's native structured output. Tests create their own instances
    with a mocked client.
    """

    def __init__(self, client, model: Optional[str] = None):
        """Initialize the summarizer agent.

        Args:
            client: An Anthropic client with a ``messages.parse`` method.
            model: Model id override. ``None`` means the summarize route's
                resolved model.

        """
        self.client = client
        self.model = model
        #: The last call this agent made, for the run's stage detail.
        self.last_call: Optional[StructuredCall] = None

        # System prompt - MUST include strings verified by tests
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

        One call. There is no hand-rolled retry: the provider retries
        structured output itself, and the helper falls back to the route's
        fallback model when the call fails outright.

        Args:
            input_data: SummarizerInput with parsed paper and context

        Returns:
            PaperNote with extracted information

        Raises:
            SummarizerValidationError: If summarization fails

        """
        logger.info(f"Starting summarization for paper: {input_data.parsed_paper.paper_ref.title}")

        try:
            result = self._attempt_summarization(input_data)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            call = getattr(e, "call", None) or self.last_call
            raise SummarizerValidationError(
                f"Claude completion failed: {e}", call=call
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

        call = structured(
            SUMMARIZE_ROUTE,
            system=system_message,
            user=user_message,
            out_model=NoteDraft,
            client=self.client,
            model=self.model,
        )
        self.last_call = call

        # Identity and the storage timestamp come from this process. A date on
        # the model object is the one the model invented, and it is not kept.
        result = note_from_draft(
            call.parsed,
            input_data.parsed_paper.paper_ref.id,
            input_data.pillar_id,
        )
        result.created_at = datetime.now(timezone.utc)

        try:
            concepts, proposed = normalize_concepts(
                result.concepts,
                result.proposed_concepts,
                input_data.concept_vocabulary,
            )
        except ConceptExtractionError as exc:
            exc.call = call.as_dict()
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

        content = fit_paper_text(paper.full_text)

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
    """Create the Anthropic client.

    Returns:
        An Anthropic client

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set

    """
    from ..llm_client import make_client

    return make_client()


class _LazySummarizerAgent:
    """Lazy stand-in for the module-level ``SummarizerAgent`` singleton.

    ``orchestrator`` and ``upload_service`` do ``from ... import SummarizerAgent``
    at import time, which binds by value. Building the real agent eagerly meant a
    missing API key produced ``AttributeError: 'NoneType' object has no attribute
    'run'`` at call time. This defers construction to the first ``run()`` so the
    failure names the actual problem, and keeps importing this module free of
    client construction.
    """

    def __init__(self):
        self._impl: Optional[SummarizerAgentImpl] = None

    def _get_impl(self) -> SummarizerAgentImpl:
        if self._impl is None:
            settings = get_settings()
            if not settings.anthropic_api_key:
                raise ValueError(
                    "SummarizerAgent is not initialized: "
                    "ANTHROPIC_API_KEY environment variable is required"
                )
            self._impl = SummarizerAgentImpl(_make_client())
        return self._impl

    @property
    def last_call(self) -> Optional[StructuredCall]:
        """The most recent call, for the run's stage detail."""
        return self._impl.last_call if self._impl is not None else None

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
