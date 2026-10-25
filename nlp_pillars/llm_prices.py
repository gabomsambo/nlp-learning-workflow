"""What every model costs, in one table.

There used to be three price tables in this package — one for the fact-check
engine, one for the concept-extraction backfill, and none at all for the
generation agents, whose spend was logged at INFO and therefore invisible.
This module is the single place a model id is turned into dollars, and the
other two now delegate to it.

Prices are USD per million tokens, verified 2026-10-24 against the Anthropic
API reference and DeepSeek's pricing page:

- ``claude-haiku-5-5``  $0.10 in / $0.50 out up to a 100K-token prompt, and
  $0.50 / $2.50 above it. Haiku charges the long rate on the whole request
  once the prompt crosses 100K tokens, not on the excess.
- ``claude-sonnet-5-5`` and ``claude-sonnet-5``  $2 / $10.
- ``claude-opus-5-5``  $4 / $20. Used only as a judge.
- ``gpt-4o-mini``  $0.15 / $0.60, ``gpt-4o``  $2.50 / $10.
- ``deepseek-flash``  $0.30 / $1.20 at peak ($0.15 / $0.60 off-peak).

Cache pricing follows Anthropic's table: a five-minute cache write is 1.25x
the input price, a one-hour write is 2x, and a cache read is 0.1x. A model
that is not in ``PRICES`` has no price, and a call on it is refused before it
is made — spending money whose cost cannot be recorded is the failure this
module exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

_MTOK = Decimal("1000000")
_QUANT = Decimal("0.000001")

_WRITE_5M = Decimal("1.25")
_WRITE_1H = Decimal("2")
_READ = Decimal("0.1")


@dataclass(frozen=True)
class ModelPrice:
    """One model's rates, in USD per million tokens."""

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
    "gpt-4o-mini": ModelPrice(Decimal("0.15"), Decimal("0.60")),
    "gpt-4o": ModelPrice(Decimal("2.50"), Decimal("10.00")),
    "deepseek-flash": ModelPrice(Decimal("0.30"), Decimal("1.20")),
    "deepseek-v4-pro": ModelPrice(Decimal("1.32"), Decimal("3.96")),
}


class UnpricedModel(ValueError):
    """The model is outside the priced set, so its spend could not be recorded."""


def _resolve(model: str) -> ModelPrice:
    """Resolve a model id to its price, including dated snapshots of a family."""
    if model in PRICES:
        return PRICES[model]
    if model.startswith("gpt-4o-mini-"):
        return PRICES["gpt-4o-mini"]
    if model.startswith("gpt-4o-"):
        return PRICES["gpt-4o"]
    if model.startswith("claude-haiku-4"):
        return PRICES["claude-haiku-5-5"]
    known = ", ".join(sorted(PRICES))
    raise UnpricedModel(
        f"No approved price for model {model!r}; refusing to spend. "
        f"Known models: {known}."
    )


def require_price(model: str) -> None:
    """Refuse before a provider call if its cost could not be recorded."""
    _resolve(model)


def money(amount: Decimal) -> float:
    """Six decimal places, half up. That is the stored ``cost_usd`` grain."""
    return float(amount.quantize(_QUANT, rounding=ROUND_HALF_UP))


def price_tokens(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one call from a plain token count, with no cache involved."""
    price = _resolve(model)
    amount = (
        price.input_per_mtok * Decimal(input_tokens)
        + price.output_per_mtok * Decimal(output_tokens)
    ) / _MTOK
    return money(amount)


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
    """Price one Anthropic response. ``usage`` is a mapping or an object.

    Returns the cost and a token breakdown the caller can store. Haiku's long
    rate applies to the whole request once the prompt crosses 100K tokens.
    """
    price = _resolve(model)
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
