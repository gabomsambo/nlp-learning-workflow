"""One Claude call, for every stage that wants structured output.

The app used to reach OpenAI through ``instructor``. That path is gone: the
account ran out of credit on 2026-10-25, and even with credit it could not
have been kept as it was, because current Claude models reject
``temperature`` (HTTP 400), Sonnet 5.5 and Opus 5.5 reject the forced tool
call Instructor's Anthropic mode depends on, and Claude's native structured
outputs reject the app's full ``QuizCard`` schema with ``400 Schema is too
complex``.

What replaces it is the Anthropic SDK's own structured output, called with a
small model-facing schema:

- ``client.messages.parse(output_format=SmallModel, output_config={"effort": ...})``
- **never** ``temperature``, ``top_p``, ``top_k`` or ``tool_choice``.

The schemas are deliberately smaller than the database models. The mapping
from a draft to ``PaperNote`` / ``Lesson`` / ``QuizCard`` lives in
``nlp_pillars/schemas.py``; this module only knows about the draft.

Two things this module does that the old path did not:

- **A fallback model.** When a call fails at the API, comes back as a
  refusal, or returns nothing parseable, the route's fallback model answers
  instead. The swap is recorded on the result, so "which model actually wrote
  this" is answerable afterwards.
- **A cost.** Every call is priced from the provider's own token counts
  (``nlp_pillars/llm_prices.py``) and reported in the result, so a stage can
  show what it spent. Generation cost used to be logged at INFO, which never
  reaches ``docker compose logs``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from .config import get_settings
from .llm_prices import price_usage, require_price
from .llm_routes import ModelRoute, ResolvedRoute

logger = logging.getLogger(__name__)

#: Output budget for a structured generation call. The measured lessons came
#: back in about 2,000 output tokens; 16,000 leaves room without letting a
#: runaway call run up a bill.
MAX_TOKENS = 16000

#: Token budget for the paper text in one generation call.
#:
#: Haiku charges $0.10/$0.50 per million tokens up to a 100K-token prompt and
#: $0.50/$2.50 above it — and the long rate applies to the **whole** request,
#: not to the excess. Measured live on 2026-10-25: a 343K-character paper
#: truncated to 300K characters is 117K tiktoken tokens and 172K Claude
#: tokens, which put it on the long rate and cost **$0.089** per call instead
#: of about $0.009. A character cap cannot see that coming.
#:
#: The budget is counted in tiktoken tokens (``cl100k_base``, the counter the
#: chunker already uses) and is 60,000. The densest paper measured tokenizes
#: 1.49x worse on Claude than on tiktoken — measured live, 60,000 tiktoken
#: tokens is 89,261 Claude tokens — which puts it under the threshold with
#: room. Every paper the quality measurements were taken on was 26-34K
#: tokens, so this still sends more text than any of them.
PAPER_TEXT_TOKEN_BUDGET = 60_000

#: What a trimmed paper says at the cut.
TRUNCATION_MARKER = "\n[TRUNCATED - paper continues]"

#: Parameters Claude rejects. Kept as a name so a test can assert they never
#: appear in a request, and so a future edit that adds one is caught.
FORBIDDEN_PARAMETERS = ("temperature", "top_p", "top_k", "tool_choice")


class LLMCallError(RuntimeError):
    """A generation call failed, and no fallback could answer it."""

    def __init__(self, message: str, *, call: Optional["StructuredCall"] = None):
        super().__init__(message)
        self.call = call


def supports_thinking_disabled(model: str) -> bool:
    """Whether this model accepts ``thinking: {"type": "disabled"}``.

    Haiku 5.5 does, and the routes that want thinking off want it there:
    Haiku's reasoning tokens come out of the same budget as the content, which
    is how a Ground Pack section used to come back empty.

    Sonnet 5 runs adaptive when the parameter is omitted, and Sonnet 5.5
    refuses to have its thinking disabled at all — sending the parameter there
    is a 400. So it is only ever sent to the Haiku family, and every other
    model simply runs adaptive. That keeps ``LLM_<STAGE>_MODEL`` a working
    override for every model the routes know about.
    """
    return model.lower().startswith("claude-haiku")


@dataclass(frozen=True)
class StructuredCall:
    """What one structured call returned, and what it cost."""

    stage: str
    provider: str
    model: str
    parsed: Any
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    fallback: bool = False
    fallback_reason: Optional[str] = None

    def detail(self) -> str:
        """One line a person can read: model, tokens, dollars.

        This is what goes into a run's stage detail. It used to be the paper
        id, which said nothing about what the stage cost.
        """
        bits = [self.model]
        if self.fallback:
            bits.append("fallback")
        if self.input_tokens:
            bits.append(f"{self.input_tokens:,} in")
        if self.output_tokens:
            bits.append(f"{self.output_tokens:,} out")
        bits.append(f"${self.cost_usd:.6f}")
        return " · ".join(bits)

    def render(self) -> str:
        """The same information, for a log line."""
        return (
            f"stage={self.stage} model={self.model} provider={self.provider} "
            f"input_tokens={self.input_tokens} output_tokens={self.output_tokens} "
            f"cost_usd={self.cost_usd:.6f}"
            + (f" fallback={self.fallback_reason}" if self.fallback else "")
        )

    def as_dict(self) -> dict[str, Any]:
        """A storable record of the call.

        This is what lands in ``notes.concept_call`` and in the run's stage
        detail, so it carries the model that actually answered — which, after
        a fallback, is not the model the route asked for.
        """
        return {
            "stage": self.stage,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
        }


def make_client():
    """A sync Anthropic client. Imported here so tests never build one."""
    from anthropic import Anthropic

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required for generation"
        )
    return Anthropic(api_key=settings.anthropic_api_key)


def _call_kwargs(
    route: ResolvedRoute,
    system: str,
    user: str,
    out_model: type,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    """The request body. No sampling parameters, ever.

    ``temperature``, ``top_p``, ``top_k`` and ``tool_choice`` are rejected by
    current Claude models with a 400, so they are not optional here — they
    are absent.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_format": out_model,
    }
    if route.effort:
        kwargs["output_config"] = {"effort": route.effort}
    if route.thinking == "disabled" and supports_thinking_disabled(model):
        kwargs["thinking"] = {"type": "disabled"}
    return kwargs


def _parse_once(
    client: Any,
    route: ResolvedRoute,
    model: str,
    system: str,
    user: str,
    out_model: type,
    max_tokens: int,
) -> tuple[Any, int, int, float]:
    """One API call. Raises on anything that is not a usable answer."""
    response = client.messages.parse(
        **_call_kwargs(route, system, user, out_model, model, max_tokens)
    )
    usage = getattr(response, "usage", None)
    cost, detail = price_usage(model, usage)
    stop = getattr(response, "stop_reason", None)
    parsed = getattr(response, "parsed_output", None)
    if stop in {"max_tokens", "refusal"} or parsed is None:
        raise LLMCallError(
            f"{model} returned {stop or 'no parsed output'}"
        )
    return parsed, detail["input_tokens"], detail["output_tokens"], cost


def structured(
    stage: str | ModelRoute,
    system: str,
    user: str,
    out_model: type,
    *,
    client: Any = None,
    model: Optional[str] = None,
    max_tokens: int = MAX_TOKENS,
) -> StructuredCall:
    """Call the stage's model, and its fallback if the first attempt fails.

    Args:
        stage: A stage name or a route from :mod:`nlp_pillars.llm_routes`.
        system: The system prompt.
        user: The user message.
        out_model: The small Pydantic model the provider fills in.
        client: An Anthropic client. Built from settings when omitted.
        model: Override the route's model. Used by the tests.
        max_tokens: Output budget.

    Returns:
        A :class:`StructuredCall` with the parsed value, the token counts and
        the cost.

    Raises:
        LLMCallError: the call failed and the fallback could not answer it
            either.
    """
    route = stage.resolve() if isinstance(stage, ModelRoute) else _route_for(stage)
    target = model or route.model
    require_price(target)

    if client is None:
        client = make_client()

    try:
        parsed, n_in, n_out, cost = _parse_once(
            client, route, target, system, user, out_model, max_tokens
        )
    except Exception as first_error:
        fallback_model = route.fallback_model
        if not fallback_model:
            raise LLMCallError(
                f"{route.stage} call to {target} failed: {first_error}"
            ) from first_error
        try:
            require_price(fallback_model)
            parsed, n_in, n_out, cost = _parse_once(
                client, route, fallback_model, system, user, out_model, max_tokens
            )
        except Exception as second_error:
            raise LLMCallError(
                f"{route.stage} call failed on {target} ({first_error}) and on "
                f"the fallback {fallback_model} ({second_error})"
            ) from second_error
        reason = f"{type(first_error).__name__}: {first_error}"
        result = StructuredCall(
            stage=route.stage,
            provider=route.provider,
            model=fallback_model,
            parsed=parsed,
            input_tokens=n_in,
            output_tokens=n_out,
            cost_usd=cost,
            fallback=True,
            fallback_reason=reason,
        )
        logger.warning(
            "%s fell back from %s to %s (%s)", route.stage, target,
            fallback_model, reason,
        )
    else:
        result = StructuredCall(
            stage=route.stage,
            provider=route.provider,
            model=target,
            parsed=parsed,
            input_tokens=n_in,
            output_tokens=n_out,
            cost_usd=cost,
        )

    # WARNING, not INFO: `docker compose logs webui` never shows INFO from
    # these loggers, so an INFO line is a line nobody reads. This is the
    # record of what the run spent.
    logger.warning("%s", result.render())
    return result


def _route_for(stage: str) -> ResolvedRoute:
    from .llm_routes import route_for

    return route_for(stage).resolve()


def fit_paper_text(text: str, budget: int = PAPER_TEXT_TOKEN_BUDGET) -> str:
    """Trim a paper to the token budget, and say that it was trimmed.

    The cut is on a token boundary, so the model never sees half a word.
    A paper that fits is returned untouched.

    Falls back to a character approximation when tiktoken's encoding cannot
    be loaded — the same degradation the chunker makes, and the same shape of
    risk: an approximate budget rather than a failed generation.
    """
    if not text:
        return text
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # pragma: no cover — the chunker's fallback shape
        logger.warning(
            "tiktoken unavailable (%s); sizing the paper text by characters",
            exc,
        )
        chars = budget * 4
        if len(text) <= chars:
            return text
        return text[:chars] + TRUNCATION_MARKER

    tokens = encoding.encode(text)
    if len(tokens) <= budget:
        return text
    return encoding.decode(tokens[:budget]) + TRUNCATION_MARKER


def stage_detail(call: Optional[StructuredCall], prefix: str = "") -> str:
    """A stage's detail line: what ran, and what it cost.

    The stage detail is what the run panel shows, and it used to be the paper
    id — which said nothing about what the stage cost. This is the "model ·
    in/out tokens · $cost" line, with whatever the caller wants to say about
    the work itself in front of it.

    A ``call`` that is not a ``StructuredCall`` is ignored rather than
    printed. The agents are replaced wholesale in several tests, and a
    ``Mock`` in the detail line is noise, not information.
    """
    bits = []
    if prefix:
        bits.append(prefix if isinstance(prefix, str) else str(prefix))
    if isinstance(call, StructuredCall):
        bits.append(call.detail())
    elif call is None:
        bits.append("no model call recorded")
    return " · ".join(bits)
