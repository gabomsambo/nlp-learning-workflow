"""Which models produce each podcast call, and each fact-check step.

This module is now a compatibility shim over
:mod:`nlp_pillars.llm_routes`, which holds the routing for every generation
stage in the app. The names here are the ones the podcast and fact-check code
imports, and they still resolve the same way.

The four Ground Pack prompts are mechanical extraction and go to DeepSeek's
``deepseek-flash``, with Claude Haiku 5.5 behind it. Call 5 is creative
synthesis and stays on Claude Sonnet 5. Fact-check extraction and screening
default to Haiku 5.5; the re-judge defaults to Sonnet 5.5.

Changing a model: set ``LLM_PODCAST_EXTRACTION_MODEL``,
``LLM_PODCAST_SYNTHESIS_MODEL``, ``LLM_FACTCHECK_EXTRACT_MODEL``,
``LLM_FACTCHECK_SCREEN_MODEL`` or ``LLM_FACTCHECK_REJUDGE_MODEL`` in
``.env``. The older ``PODCAST_*_MODEL`` and ``FACTCHECK_*_MODEL`` names are
still honoured.
"""

from .llm_routes import (  # noqa: F401 — re-exported for existing importers
    CLAUDE_FALLBACK,
    PODCAST_FALLBACK,
    FACTCHECK_EXTRACT,
    FACTCHECK_REJUDGE,
    FACTCHECK_SCREEN,
    ModelRoute,
    PODCAST_EXTRACTION_ROUTE,
    PODCAST_SYNTHESIS_ROUTE,
    Provider,
    ResolvedRoute,
)

#: Calls 1-4: the Ground Pack extractions.
EXTRACTION_ROUTE = PODCAST_EXTRACTION_ROUTE

#: Call 5: the podcast script itself.
SYNTHESIS_ROUTE = PODCAST_SYNTHESIS_ROUTE

__all__ = [
    "CLAUDE_FALLBACK",
    "EXTRACTION_ROUTE",
    "FACTCHECK_EXTRACT",
    "FACTCHECK_REJUDGE",
    "FACTCHECK_SCREEN",
    "ModelRoute",
    "PODCAST_EXTRACTION_ROUTE",
    "PODCAST_FALLBACK",
    "PODCAST_SYNTHESIS_ROUTE",
    "Provider",
    "ResolvedRoute",
    "SYNTHESIS_ROUTE",
]
