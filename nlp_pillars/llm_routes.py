"""Which model produces each generation stage.

One registry for every model call this app makes. Before it there were two
places naming models — ``nlp_pillars/podcast_models.py`` for the podcast and
fact-check calls, and ``Settings.default_model`` for everything else — and the
second one pointed at OpenAI, whose account ran out of credit on 2026-10-25.
Every generation stage now runs on Claude, and this module is where that is
decided.

The defaults, and why:

- **summarize, synthesize, quiz, discover** — ``claude-haiku-5-5``. Haiku is
  $0.10/$0.50 per million tokens, and the measured quality gap against Sonnet
  was smaller than the gap between "full paper text" and "the note alone".
- **concepts** (the vocabulary-aware concept extraction) — ``claude-haiku-5-5``,
  same reason: it is a small, mechanical selection over a supplied vocabulary.
- **podcast_extraction** — DeepSeek's ``deepseek-flash``, the four Ground Pack
  calls. The account was topped up on 2026-10-25 and verified working, and
  Flash is cheaper than Haiku at peak hours. Claude Haiku 5.5 is the fallback.
- **podcast_synthesis** — ``claude-sonnet-5``, unchanged. The script is the
  one call where taste is the point and Haiku measurably was not enough.

Every stage is overridable by env without touching code:
``LLM_SUMMARIZE_MODEL``, ``LLM_SYNTHESIZE_MODEL``, ``LLM_QUIZ_MODEL``,
``LLM_DISCOVER_MODEL``, ``LLM_CONCEPTS_MODEL``,
``LLM_PODCAST_EXTRACTION_MODEL``, ``LLM_PODCAST_SYNTHESIS_MODEL``, the
``LLM_FACTCHECK_*`` trio, ``LLM_PLAYGROUND_MODEL`` and
``LLM_PLAYGROUND_ALIGN_MODEL``. The older ``PODCAST_*_MODEL`` and
``FACTCHECK_*_MODEL`` names still work and are consulted second, so an
existing ``.env`` keeps working.

Moving the lesson to Sonnet is therefore a config change, not a code change:
set ``LLM_SYNTHESIZE_MODEL=claude-sonnet-5-5``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import get_settings

Provider = str

#: The Claude model a failed generation call falls back to.
CLAUDE_FALLBACK = "claude-sonnet-5-5"

#: The second Claude model behind a podcast Ground Pack call. Used when the
#: route's own fallback is already the primary model, which happens when
#: Haiku is made primary by configuration.
PODCAST_FALLBACK = "claude-sonnet-5"

#: Model ids that name a DeepSeek model. Used to keep a stale
#: ``PODCAST_EXTRACTION_MODEL=deepseek-v4-flash`` in an existing .env from
#: being sent to Anthropic, where it would 404.
DEEPSEEK_PREFIX = "deepseek"


@dataclass(frozen=True)
class ResolvedRoute:
    """One stage's routing, after env overrides."""

    stage: str
    provider: Provider
    model: str
    fallback_model: Optional[str] = None
    secondary_fallback_model: Optional[str] = None
    effort: Optional[str] = None
    thinking: str = "adaptive"


@dataclass(frozen=True)
class ModelRoute:
    """One stage's routing, before env overrides.

    ``model_attrs`` are settings attribute names consulted in order; the
    first one holding a non-empty string wins. Keeping several is what lets
    ``LLM_PODCAST_EXTRACTION_MODEL`` and the older ``PODCAST_EXTRACTION_MODEL``
    both work without either being special-cased.

    ``default_model`` is the model for the **Anthropic** path. A route that
    may run on DeepSeek carries ``deepseek_model`` as well, and
    ``provider`` says which one is the default.
    """

    stage: str
    provider: Provider
    default_model: str
    fallback_model: Optional[str] = None
    secondary_fallback_model: Optional[str] = None
    model_attrs: tuple[str, ...] = ()
    provider_attrs: tuple[str, ...] = ()
    effort: Optional[str] = None
    thinking: str = "adaptive"
    # Fixed model id used when this route's provider resolves to DeepSeek.
    deepseek_model: Optional[str] = None
    # Whether the model id decides the provider.
    provider_follows_model: bool = False

    # -- resolution -------------------------------------------------------

    def _setting(self, attrs: tuple[str, ...]) -> Optional[str]:
        """First attribute holding a non-empty string.

        Only strings count. A ``Mock`` in a test is truthy, and reading one as
        a model id would send the test's mock object to the API.
        """
        settings = get_settings()
        for attr in attrs:
            value = getattr(settings, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def resolve(self) -> ResolvedRoute:
        """The route as it will actually be called."""
        model = self._setting(self.model_attrs)
        provider = self._setting(self.provider_attrs)

        if provider is None and self.provider_follows_model and model:
            # A model id that names a family picks that family's provider, so
            # "make Haiku primary" is one env var rather than two.
            provider = (
                "deepseek"
                if model.lower().startswith(DEEPSEEK_PREFIX)
                else "anthropic"
            )

        provider = provider or self.provider

        if provider == "deepseek":
            # The DeepSeek id is fixed: `deepseek-v4-flash` is a legacy alias
            # and the model behind it is served as `deepseek-flash`.
            model = self.deepseek_model or model or self.default_model
        else:
            if model and model.lower().startswith(DEEPSEEK_PREFIX):
                # A DeepSeek id under an Anthropic route would 404. Use the
                # route's own default rather than failing the generation.
                model = self.default_model
            model = model or self.default_model

        return ResolvedRoute(
            stage=self.stage,
            provider=provider,
            model=model,
            fallback_model=self.fallback_model,
            secondary_fallback_model=self.secondary_fallback_model,
            effort=self.effort,
            thinking=self.thinking,
        )

    def resolved_model(self) -> str:
        """Model id after applying the env override, if any."""
        return self.resolve().model

    def resolved_provider(self) -> Provider:
        """Provider after applying the env override, if any."""
        return self.resolve().provider

    def resolved_fallback_model(self) -> Optional[str]:
        """Fallback model id, or None when the stage has no fallback."""
        return self.fallback_model


# -- generation stages ----------------------------------------------------

#: Structured notes from a paper. Thinking on: the measured gain was ~0.6 on
#: the rubric for about 8% more output tokens.
SUMMARIZE_ROUTE = ModelRoute(
    stage="summarize",
    provider="anthropic",
    default_model="claude-haiku-5-5",
    fallback_model=CLAUDE_FALLBACK,
    model_attrs=("llm_summarize_model",),
    effort="medium",
    thinking="adaptive",
)

#: The lesson. Haiku by the captain's call (2026-10-25); the fallback is
#: Sonnet, and ``LLM_SYNTHESIZE_MODEL=claude-sonnet-5-5`` moves the stage
#: to Sonnet without a code change.
SYNTHESIZE_ROUTE = ModelRoute(
    stage="synthesize",
    provider="anthropic",
    default_model="claude-haiku-5-5",
    fallback_model=CLAUDE_FALLBACK,
    model_attrs=("llm_synthesize_model",),
    effort="medium",
    thinking="adaptive",
)

#: Quiz cards. Thinking off: the measured best arm was Haiku without it, and
#: thinking made the model return more cards than the app keeps.
QUIZ_ROUTE = ModelRoute(
    stage="quiz",
    provider="anthropic",
    default_model="claude-haiku-5-5",
    fallback_model=CLAUDE_FALLBACK,
    model_attrs=("llm_quiz_model",),
    effort="medium",
    thinking="disabled",
)

#: Search queries for discovery. Thinking off, and the diversity that used to
#: come from ``temperature=0.7`` is asked for in the prompt instead.
DISCOVER_ROUTE = ModelRoute(
    stage="discover",
    provider="anthropic",
    default_model="claude-haiku-5-5",
    fallback_model=CLAUDE_FALLBACK,
    model_attrs=("llm_discover_model",),
    effort="medium",
    thinking="disabled",
)

#: Vocabulary-aware concept extraction, both the summarizer's piggyback and
#: the one-off backfill over historical notes.
CONCEPTS_ROUTE = ModelRoute(
    stage="concepts",
    provider="anthropic",
    default_model="claude-haiku-5-5",
    fallback_model=CLAUDE_FALLBACK,
    model_attrs=("llm_concepts_model",),
    effort="medium",
    thinking="disabled",
)

# -- podcast ---------------------------------------------------------------

#: Calls 1-4, the Ground Pack. **DeepSeek is the default**: the account was
#: topped up on 2026-10-25 and verified working, and DeepSeek Flash is
#: cheaper than Haiku at peak hours. Claude Haiku 5.5 is the fallback — a
#: DeepSeek failure used to fall back to Sonnet, and Haiku 5.5 answers the
#: same prompt at a tenth of the price.
#:
#: The model id is ``deepseek-flash``. The app used to send
#: ``deepseek-v4-flash``; that is a legacy alias and the model behind it is
#: served as ``deepseek-flash`` now. Both are accepted, and the newer id is
#: what this route sends.
#:
#: Haiku can become primary by config, with no code change:
#: ``LLM_PODCAST_EXTRACTION_MODEL=claude-haiku-5-5`` in ``.env``.
PODCAST_EXTRACTION_ROUTE = ModelRoute(
    stage="podcast_extraction",
    provider="deepseek",
    default_model="claude-haiku-5-5",
    fallback_model="claude-haiku-5-5",
    secondary_fallback_model=PODCAST_FALLBACK,
    model_attrs=("llm_podcast_extraction_model", "podcast_extraction_model"),
    provider_attrs=("podcast_extraction_provider",),
    effort="medium",
    thinking="disabled",
    deepseek_model="deepseek-flash",
    provider_follows_model=True,
)

#: Call 5, the script itself. Unchanged: Sonnet 5, because Haiku's script was
#: measurably drier and put a [VERIFY] note inside a [HOST] line.
PODCAST_SYNTHESIS_ROUTE = ModelRoute(
    stage="podcast_synthesis",
    provider="anthropic",
    default_model="claude-sonnet-5",
    fallback_model=None,
    model_attrs=("llm_podcast_synthesis_model", "podcast_synthesis_model"),
    effort="medium",
    thinking="adaptive",
)

# -- fact-check ------------------------------------------------------------

FACTCHECK_EXTRACT = ModelRoute(
    stage="factcheck_extract",
    provider="anthropic",
    default_model="claude-haiku-5-5",
    fallback_model=CLAUDE_FALLBACK,
    model_attrs=("llm_factcheck_extract_model", "factcheck_extract_model"),
    effort="medium",
    thinking="disabled",
)
FACTCHECK_SCREEN = ModelRoute(
    stage="factcheck_screen",
    provider="anthropic",
    default_model="claude-haiku-5-5",
    fallback_model=CLAUDE_FALLBACK,
    model_attrs=("llm_factcheck_screen_model", "factcheck_screen_model"),
    effort="medium",
    thinking="disabled",
)
FACTCHECK_REJUDGE = ModelRoute(
    stage="factcheck_rejudge",
    provider="anthropic",
    default_model="claude-sonnet-5-5",
    fallback_model=CLAUDE_FALLBACK,
    model_attrs=("llm_factcheck_rejudge_model", "factcheck_rejudge_model"),
    effort="medium",
    thinking="adaptive",
)

# -- playground --------------------------------------------------------------

#: The Paper Playground explain pass (docs/PLAYGROUND.md): a notebook's cells
#: become study steps with one-line explanations, line notes and "try this"
#: tweaks. Same Sonnet as the podcast script by default: the explanations are
#: what the captain studies from, so taste matters more than price here.
PLAYGROUND_ROUTE = ModelRoute(
    stage="playground_explain",
    provider="anthropic",
    default_model="claude-sonnet-5",
    fallback_model=CLAUDE_FALLBACK,
    model_attrs=("llm_playground_model",),
    effort="medium",
    thinking="adaptive",
)

#: Follow-along alignment (docs/PLAYGROUND.md, "Follow-along"): which lines of
#: an episode's script are about which code cell. A mechanical matching task
#: over supplied text, so the cheap Haiku the app already uses, thinking off.
#: ``LLM_PLAYGROUND_ALIGN_MODEL`` overrides it.
PLAYGROUND_ALIGN_ROUTE = ModelRoute(
    stage="playground_align",
    provider="anthropic",
    default_model="claude-haiku-5-5",
    fallback_model=CLAUDE_FALLBACK,
    model_attrs=("llm_playground_align_model",),
    effort="medium",
    thinking="disabled",
)


def route_for(stage: str) -> ModelRoute:
    """The route for a stage name. Raises for an unknown stage."""
    routes = {
        r.stage: r
        for r in (
            SUMMARIZE_ROUTE,
            SYNTHESIZE_ROUTE,
            QUIZ_ROUTE,
            DISCOVER_ROUTE,
            CONCEPTS_ROUTE,
            PODCAST_EXTRACTION_ROUTE,
            PODCAST_SYNTHESIS_ROUTE,
            FACTCHECK_EXTRACT,
            FACTCHECK_SCREEN,
            FACTCHECK_REJUDGE,
            PLAYGROUND_ROUTE,
            PLAYGROUND_ALIGN_ROUTE,
        )
    }
    try:
        return routes[stage]
    except KeyError:
        raise KeyError(
            f"Unknown generation stage {stage!r}; known stages: "
            + ", ".join(sorted(routes))
        ) from None
