"""USD from Anthropic ``usage``.

Prices are the 2026-10-06 Claude rates the fact-check research measured with.
Haiku cache reads at $0.01 per million tokens are 0.1× the short-context input
price. A 5-minute cache write is 1.25× input; a 1-hour write is 2×. Haiku
charges the long-context rate on the whole request once the prompt exceeds
100k tokens. Sonnet 5.5 and Opus 5.5 have no separate long rate here.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from .errors import FactcheckError

_MTOK = Decimal("1000000")
_QUANT = Decimal("0.000001")


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    long_input_per_mtok: Decimal | None = None
    long_output_per_mtok: Decimal | None = None
    long_after: int = 100_000


PRICES: dict[str, ModelPrice] = {
    "claude-haiku-5-5": ModelPrice(
        Decimal("0.10"),
        Decimal("0.50"),
        long_input_per_mtok=Decimal("0.50"),
        long_output_per_mtok=Decimal("2.50"),
    ),
    "claude-sonnet-5-5": ModelPrice(Decimal("2"), Decimal("10")),
    "claude-sonnet-5": ModelPrice(Decimal("2"), Decimal("10")),
    "claude-opus-5-5": ModelPrice(Decimal("4"), Decimal("20")),
}

_WRITE_5M = Decimal("1.25")
_WRITE_1H = Decimal("2")
_READ = Decimal("0.1")


def require_price(model: str) -> None:
    """Refuse to call a model whose spend we could not record."""
    if model not in PRICES:
        known = ", ".join(sorted(PRICES))
        raise FactcheckError(
            f"No price for model {model!r}, so its cost_usd would be a guess. "
            f"Known models: {known}."
        )


def money(amount: Decimal) -> float:
    """Six decimal places, half up. That is the stored ``cost_usd`` grain."""
    rounded = amount.quantize(_QUANT, rounding=ROUND_HALF_UP)
    return float(rounded)


def _as_int(obj: Any, name: str) -> int:
    if obj is None:
        return 0
    if isinstance(obj, dict):
        value = obj.get(name, 0)
    else:
        value = getattr(obj, name, 0)
    if value is None:
        return 0
    return int(value)


def _cache_writes(usage: Any) -> tuple[int, int]:
    """Return ``(ephemeral_5m, ephemeral_1h)`` tokens.

    When the breakdown is absent, the whole ``cache_creation_input_tokens``
    count is the default 5-minute write. Anthropic's ephemeral cache is that
    tier unless the request asks for an hour.
    """
    creation = None
    if isinstance(usage, dict):
        creation = usage.get("cache_creation")
    else:
        creation = getattr(usage, "cache_creation", None)
    if creation is not None:
        five = _as_int(creation, "ephemeral_5m_input_tokens")
        hour = _as_int(creation, "ephemeral_1h_input_tokens")
        if five or hour:
            return five, hour
    return _as_int(usage, "cache_creation_input_tokens"), 0


def price_usage(model: str, usage: Any) -> tuple[float, dict[str, int]]:
    """Price one response. ``usage`` is a mapping or an object with attributes."""
    require_price(model)
    price = PRICES[model]
    n_in = _as_int(usage, "input_tokens")
    n_out = _as_int(usage, "output_tokens")
    n_read = _as_int(usage, "cache_read_input_tokens")
    five, hour = _cache_writes(usage)
    prompt = n_in + n_read + five + hour
    input_rate = price.input_per_mtok
    output_rate = price.output_per_mtok
    if (
        price.long_input_per_mtok is not None
        and price.long_output_per_mtok is not None
        and prompt > price.long_after
    ):
        input_rate = price.long_input_per_mtok
        output_rate = price.long_output_per_mtok
    amount = (
        input_rate * Decimal(n_in)
        + output_rate * Decimal(n_out)
        + input_rate * _WRITE_5M * Decimal(five)
        + input_rate * _WRITE_1H * Decimal(hour)
        + input_rate * _READ * Decimal(n_read)
    ) / _MTOK
    detail = {
        "input_tokens": n_in,
        "output_tokens": n_out,
        "cache_creation_input_tokens": five + hour,
        "cache_read_input_tokens": n_read,
    }
    return money(amount), detail
