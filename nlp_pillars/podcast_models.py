"""Which models produce each podcast call, and each fact-check step.

The four Ground Pack prompts are mechanical extraction; call 5 is creative
synthesis. Fact-check extraction and screening default to Haiku 5.5; the
re-judge defaults to Sonnet 5.5. This registry is the single place that names
the providers and models, and env vars override the defaults without touching
agent code — same extension pattern as ``nlp_pillars/podcast_options.py``.

Changing a model: set ``PODCAST_EXTRACTION_MODEL``, ``PODCAST_SYNTHESIS_MODEL``,
``FACTCHECK_EXTRACT_MODEL``, ``FACTCHECK_SCREEN_MODEL``, or
``FACTCHECK_REJUDGE_MODEL`` in ``.env``. The provider is fixed per call kind;
only the model id is overridable.
"""

from dataclasses import dataclass
from typing import Literal

from .config import get_settings

Provider = Literal["deepseek", "anthropic"]


@dataclass(frozen=True)
class ModelRoute:
    """One call kind's provider and model."""

    kind: str
    provider: Provider
    default_model: str
    settings_attr: str

    def resolved_model(self) -> str:
        """Model id after applying the env override, if any."""
        override = getattr(get_settings(), self.settings_attr, None)
        if override:
            return str(override).strip()
        return self.default_model


# Calls 1–4: cheap extraction. ``deepseek-v4-pro`` is deliberately not the
# default — it is a reasoning model that burns the token budget on thinking
# and truncates before writing the Ground Pack table.
EXTRACTION_ROUTE = ModelRoute(
    kind="extraction",
    provider="deepseek",
    default_model="deepseek-v4-flash",
    settings_attr="podcast_extraction_model",
)

# Call 5: pacing and taste stay on Claude. ``claude-sonnet-5`` is the current
# Sonnet (1M context, 128K max output, $2/$10 per MTok) — cheaper and with a
# far larger output ceiling than ``claude-sonnet-4-5-20250929``. Sonnet 5
# rejects ``temperature`` / ``top_p`` / ``top_k``; see ``_call_claude``.
SYNTHESIS_ROUTE = ModelRoute(
    kind="synthesis",
    provider="anthropic",
    default_model="claude-sonnet-5",
    settings_attr="podcast_synthesis_model",
)

# Fact-check. Haiku extracts and screens; Sonnet re-judges only flagged
# claims. Current Claude models reject temperature — the caller omits it.
FACTCHECK_EXTRACT = ModelRoute(
    kind="factcheck_extract",
    provider="anthropic",
    default_model="claude-haiku-5-5",
    settings_attr="factcheck_extract_model",
)
FACTCHECK_SCREEN = ModelRoute(
    kind="factcheck_screen",
    provider="anthropic",
    default_model="claude-haiku-5-5",
    settings_attr="factcheck_screen_model",
)
FACTCHECK_REJUDGE = ModelRoute(
    kind="factcheck_rejudge",
    provider="anthropic",
    default_model="claude-sonnet-5-5",
    settings_attr="factcheck_rejudge_model",
)
