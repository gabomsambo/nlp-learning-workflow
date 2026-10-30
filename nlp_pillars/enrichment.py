"""What every paid step of the knowledge layer shares: the switch, the budget, the ledger.

The knowledge layer (community names, bridge sentences, concept anchoring)
is the app's one recurring small model spend. Three rules hold for all of it:

- **One switch.** ``MAP_ENRICHMENT_ENABLED`` (default on) turns every paid call
  off. Off, nothing here constructs a client; the callers fall back to
  templates and exact matches.
- **Refuse rather than assume.** Before a paid pass, the day's spend is read
  (``spend.current_budget``). An exhausted cap, or a spend that cannot be read,
  stops the pass. A cap that fails open is not one.
- **Every call is on the books.** Each call's model, tokens and dollars are
  kept in a :class:`CallLog`, written into the run's stage detail (the line
  ``spend.py`` sums) and logged at WARNING by ``llm_client`` like every other
  priced call in the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .config import get_settings
from .llm_client import StructuredCall


def enabled() -> bool:
    """Whether paid enrichment may run. Anything but an explicit false is on."""
    return getattr(get_settings(), "map_enrichment_enabled", True) is not False


class EnrichmentRefused(Exception):
    """A paid pass was not started. The message says why, in words for the run panel."""


def require_budget(reserve_usd: float = 0.0) -> None:
    """Raise :class:`EnrichmentRefused` unless today's spend leaves room.

    ``reserve_usd`` is what the pass is expected to cost; the pass is refused
    when spend plus the reserve would pass the cap.
    """
    from . import spend

    try:
        budget = spend.current_budget()
    except spend.SpendUnavailableError as exc:
        raise EnrichmentRefused(
            f"Today's spend could not be read ({exc}), so no paid enrichment was started."
        ) from exc
    if budget.exhausted or budget.spent_usd + reserve_usd > budget.cap_usd:
        raise EnrichmentRefused(
            f"The daily spend cap (${budget.cap_usd:.2f}) leaves "
            f"${budget.remaining_usd:.4f}; paid enrichment was not started."
        )


@dataclass
class CallLog:
    """The calls one pass made and what they cost."""

    calls: List[StructuredCall] = field(default_factory=list)
    #: Purpose of each call ("community name", "bridge", "anchor"), parallel to ``calls``.
    purposes: List[str] = field(default_factory=list)

    def add(self, call: StructuredCall, purpose: str) -> None:
        self.calls.append(call)
        self.purposes.append(purpose)

    @property
    def cost_usd(self) -> float:
        return round(sum(call.cost_usd for call in self.calls), 6)

    @property
    def input_tokens(self) -> int:
        return sum(call.input_tokens for call in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.calls)

    def models(self) -> List[str]:
        return sorted({call.model for call in self.calls})

    def detail(self, prefix: str = "") -> str:
        """``prefix · 3 calls · claude-haiku-5-5 · 4,100 in · 620 out · $0.000720``.

        ``spend.py`` adds up every ``$amount`` in a stage detail, so this is
        the line that makes the pass count against the day's cap.
        """
        bits = [prefix] if prefix else []
        if not self.calls:
            bits.append("no paid calls")
            return " · ".join(bits)
        bits.append(f"{len(self.calls)} call{'s' if len(self.calls) != 1 else ''}")
        bits.extend(self.models())
        bits.append(f"{self.input_tokens:,} in")
        bits.append(f"{self.output_tokens:,} out")
        bits.append(f"${self.cost_usd:.6f}")
        return " · ".join(bits)

    def as_dict(self) -> Dict[str, Any]:
        """A storable summary: totals and one record per call."""
        return {
            "calls": len(self.calls),
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "models": self.models(),
            "by_purpose": {
                purpose: round(
                    sum(c.cost_usd for c, p in zip(self.calls, self.purposes) if p == purpose), 6
                )
                for purpose in sorted(set(self.purposes))
            },
        }
