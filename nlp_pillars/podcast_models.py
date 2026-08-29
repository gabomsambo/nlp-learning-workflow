"""Which models produce each podcast call.

The four Ground Pack prompts are mechanical extraction; call 5 is creative
synthesis. This registry is the single place that names the providers and
models, and env vars override the defaults without touching agent code —
same extension pattern as ``nlp_pillars/podcast_options.py``.

Changing a model: set ``PODCAST_EXTRACTION_MODEL`` or ``PODCAST_SYNTHESIS_MODEL``
in ``.env``. The provider is fixed per call kind (DeepSeek for extraction,
Anthropic for synthesis); only the model id is overridable.
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

# Call 5: pacing and taste stay on Claude.
SYNTHESIS_ROUTE = ModelRoute(
    kind="synthesis",
    provider="anthropic",
    default_model="claude-sonnet-4-5-20250929",
    settings_attr="podcast_synthesis_model",
)
