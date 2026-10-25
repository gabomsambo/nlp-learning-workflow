"""Price and log the call that extracts vocabulary-aware concepts.

The normal summarizer piggybacks concepts on its existing call. The backfill
makes one smaller call per historical note. Both are Claude calls now, priced
from the provider's own token counts, never a character count.

The prices live in :mod:`nlp_pillars.llm_prices`, which every other model
call in the app uses. What this module adds is the record shape the note and
the backfill report store.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from .llm_prices import PRICES, ModelPrice, UnpricedModel
from .llm_prices import price_tokens as _price_tokens
from .llm_prices import require_price as _require_price

logger = logging.getLogger(__name__)

_BACKFILL_INPUT_TOKENS = 1500
_BACKFILL_OUTPUT_TOKENS = 200

#: Kept as a name because the concept code and its tests talk about it.
#: The same class the shared price table raises.
UnpricedConceptModel = UnpricedModel

__all__ = [
    "ConceptCall",
    "ModelPrice",
    "PRICES",
    "UnpricedConceptModel",
    "estimate_backfill_call",
    "price_tokens",
    "require_price",
]


@dataclass(frozen=True)
class ConceptCall:
    """Actual cost of one provider response."""

    purpose: str
    paper_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

    @classmethod
    def from_structured(cls, call: Any, *, purpose: str, paper_id: str) -> "ConceptCall":
        """Build a record from a ``llm_client.StructuredCall``."""
        return cls(
            purpose=purpose,
            paper_id=paper_id,
            model=call.model,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            cost_usd=call.cost_usd,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        return (
            f"call {self.paper_id} {self.model} input_tokens {self.input_tokens} "
            f"output_tokens {self.output_tokens} cost_usd {self.cost_usd:.6f}"
        )


def require_price(model: str) -> None:
    """Refuse before a provider call if its cost could not be recorded."""
    _require_price(model)


def price_tokens(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one call, from its token counts."""
    return _price_tokens(model, input_tokens, output_tokens)


def estimate_backfill_call(model: str) -> float:
    """Approved report estimate: about 1,500 input and 200 output tokens."""
    return price_tokens(model, _BACKFILL_INPUT_TOKENS, _BACKFILL_OUTPUT_TOKENS)
