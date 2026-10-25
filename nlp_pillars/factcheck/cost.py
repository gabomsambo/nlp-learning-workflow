"""USD from Anthropic ``usage``.

The rates live in :mod:`nlp_pillars.llm_prices`, which every model call in
the app uses — there is one price table, not one per subsystem. What this
module adds is the fact-check's own error type: a call on an unpriced model
raises ``FactcheckError`` here rather than the shared ``UnpricedModel``.
"""

from __future__ import annotations

from typing import Any

from ..llm_prices import PRICES, ModelPrice, UnpricedModel
from ..llm_prices import price_usage as _price_usage
from ..llm_prices import require_price as _require_price
from .errors import FactcheckError

__all__ = ["PRICES", "ModelPrice", "price_usage", "require_price"]


def require_price(model: str) -> None:
    """Refuse to call a model whose spend we could not record."""
    try:
        _require_price(model)
    except UnpricedModel as exc:
        raise FactcheckError(str(exc)) from exc


def price_usage(model: str, usage: Any) -> tuple[float, dict[str, int]]:
    """Price one response. ``usage`` is a mapping or an object with attributes.

    Returns the cost and a token breakdown the caller can store.
    """
    try:
        return _price_usage(model, usage)
    except UnpricedModel as exc:
        raise FactcheckError(str(exc)) from exc
