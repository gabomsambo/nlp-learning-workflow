"""Price and log the OpenAI call that extracts vocabulary-aware concepts.

The normal summarizer piggybacks concepts on its existing call. The backfill
makes one smaller call per historical note. Both use Instructor's raw
completion so actual spend comes from provider token counts, never a character
count. Prices are the approved gpt-4o family rates, in USD per million tokens.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

logger = logging.getLogger(__name__)

_MTOK = Decimal("1000000")
_QUANT = Decimal("0.000001")
_BACKFILL_INPUT_TOKENS = 1500
_BACKFILL_OUTPUT_TOKENS = 200


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: Decimal
    output_per_mtok: Decimal


PRICES = {
    "gpt-4o-mini": ModelPrice(Decimal("0.15"), Decimal("0.60")),
    "gpt-4o": ModelPrice(Decimal("2.50"), Decimal("10.00")),
}


class UnpricedConceptModel(ValueError):
    """The configured model is outside the captain's approved spend."""


@dataclass(frozen=True)
class ConceptCall:
    """Actual cost of one provider response, including Instructor retries."""

    purpose: str
    paper_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        return (
            f"call {self.paper_id} {self.model} input_tokens {self.input_tokens} "
            f"output_tokens {self.output_tokens} cost_usd {self.cost_usd:.6f}"
        )


def _price_for(model: str) -> ModelPrice:
    """Resolve aliases and dated snapshots without guessing another family."""
    if model == "gpt-4o-mini" or model.startswith("gpt-4o-mini-"):
        return PRICES["gpt-4o-mini"]
    if model == "gpt-4o" or model.startswith("gpt-4o-"):
        return PRICES["gpt-4o"]
    known = ", ".join(sorted(PRICES))
    raise UnpricedConceptModel(
        f"No approved concept-extraction price for model {model!r}; "
        f"refusing to spend. Use one of: {known}."
    )


def require_price(model: str) -> None:
    """Refuse before a provider call if its cost could not be recorded."""
    _price_for(model)


def _integer(obj: Any, *names: str) -> int:
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return int(value)
    return 0


def price_tokens(model: str, input_tokens: int, output_tokens: int) -> float:
    price = _price_for(model)
    amount = (
        price.input_per_mtok * Decimal(input_tokens)
        + price.output_per_mtok * Decimal(output_tokens)
    ) / _MTOK
    return float(amount.quantize(_QUANT, rounding=ROUND_HALF_UP))


def call_from_completion(
    *, purpose: str, paper_id: str, model: str, completion: Any
) -> ConceptCall:
    """Build and log one actual call record from a raw OpenAI completion."""
    usage = getattr(completion, "usage", None)
    if usage is None:
        raise RuntimeError(
            f"{purpose} returned no token usage; cost_usd cannot be recorded"
        )
    input_tokens = _integer(usage, "prompt_tokens", "input_tokens")
    output_tokens = _integer(usage, "completion_tokens", "output_tokens")
    record = ConceptCall(
        purpose=purpose,
        paper_id=paper_id,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=price_tokens(model, input_tokens, output_tokens),
    )
    logger.info(
        "%s paper=%s model=%s cost_usd=%.6f input=%s output=%s",
        purpose,
        paper_id,
        model,
        record.cost_usd,
        input_tokens,
        output_tokens,
    )
    return record


def call_from_exception(
    *, purpose: str, paper_id: str, model: str, error: BaseException
) -> ConceptCall | None:
    """Recover cost from Instructor's failed final completion when available."""
    completion = getattr(error, "last_completion", None)
    if completion is None:
        return None
    try:
        return call_from_completion(
            purpose=purpose,
            paper_id=paper_id,
            model=model,
            completion=completion,
        )
    except Exception as usage_error:  # the original model failure stays primary
        logger.warning(
            "%s paper=%s failed and its cost_usd was unavailable: %s",
            purpose,
            paper_id,
            usage_error,
        )
        return None


def estimate_backfill_call(model: str) -> float:
    """Approved report estimate: about 1,500 input and 200 output tokens."""
    return price_tokens(model, _BACKFILL_INPUT_TOKENS, _BACKFILL_OUTPUT_TOKENS)
