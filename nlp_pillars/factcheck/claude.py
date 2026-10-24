"""One Anthropic structured-output call.

Current Claude models reject temperature, top_p, and top_k. Sonnet 5.5 and
Opus 5.5 reject Instructor's forced tool call. This calls ``messages.parse``
with a small schema and no sampling parameters.
"""

from __future__ import annotations

import logging
from typing import Any

from .cost import price_usage, require_price
from .errors import FactcheckBudgetError, FactcheckError

logger = logging.getLogger(__name__)

MAX_TOKENS = 16000
EFFORT = "medium"


class Spend:
    """Running cost of one fact-check. The cap stops the next call."""

    def __init__(self, cap_usd: float | None):
        self.cap_usd = cap_usd
        self.total = 0.0
        self.calls: list[dict[str, Any]] = []

    def before(self, step: str) -> None:
        if self.cap_usd is not None and self.total >= self.cap_usd:
            raise FactcheckBudgetError(
                f"{step} not started: ${self.total:.6f} is already at the "
                f"${self.cap_usd:.2f} cap",
                cost_usd=self.total,
                calls=self.calls,
            )

    def add(self, record: dict[str, Any], step: str) -> None:
        self.calls.append(record)
        self.total = round(sum(call["cost_usd"] for call in self.calls), 6)
        if self.cap_usd is not None and self.total > self.cap_usd:
            raise FactcheckBudgetError(
                f"{step} finished over the ${self.cap_usd:.2f} cap "
                f"(${self.total:.6f}). Nothing was saved.",
                cost_usd=self.total,
                calls=self.calls,
            )


def cached_system(instructions: str, body: str) -> list[dict[str, Any]]:
    """Instructions, then the stable body with a cache breakpoint on it.

    Anthropic caches the prefix through the breakpoint, so later calls with
    the same instructions and body read the paper (or the script) from cache.
    """
    return [
        {"type": "text", "text": instructions},
        {
            "type": "text",
            "text": body,
            "cache_control": {"type": "ephemeral"},
        },
    ]


def make_client():
    """Sync Anthropic client. Imported here so tests never construct one."""
    from anthropic import Anthropic

    from ..config import get_settings

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise FactcheckError("ANTHROPIC_API_KEY is not configured")
    return Anthropic(api_key=settings.anthropic_api_key)


def parse_structured(
    client: Any,
    *,
    model: str,
    system: str | list[dict[str, Any]],
    user: str,
    output_format: type,
    step: str,
    spend: Spend,
) -> Any:
    """Return ``parsed_output``. Record ``cost_usd`` from ``usage`` first."""
    require_price(model)
    spend.before(step)
    kwargs = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_format": output_format,
        "output_config": {"effort": EFFORT},
    }
    try:
        response = client.messages.parse(**kwargs)
    except FactcheckError:
        raise
    except Exception as exc:
        raise FactcheckError(
            f"{step} call failed: {exc}",
            cost_usd=spend.total,
            calls=spend.calls,
        ) from exc

    usage = getattr(response, "usage", None)
    cost, detail = price_usage(model, usage)
    record = {"step": step, "model": model, "cost_usd": cost, **detail}
    logger.info(
        "%s %s cost_usd=%s input=%s output=%s cache_read=%s cache_write=%s",
        step,
        model,
        cost,
        detail["input_tokens"],
        detail["output_tokens"],
        detail["cache_read_input_tokens"],
        detail["cache_creation_input_tokens"],
    )
    spend.add(record, step)

    stop = getattr(response, "stop_reason", None)
    parsed = getattr(response, "parsed_output", None)
    if stop in {"max_tokens", "refusal"} or parsed is None:
        raise FactcheckError(
            f"{step} returned {stop or 'no parsed output'}",
            cost_usd=spend.total,
            calls=spend.calls,
        )
    return parsed
