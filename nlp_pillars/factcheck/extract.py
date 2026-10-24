"""Claim extraction over 30-line windows.

The whole numbered script sits in a cached system block. Each call is told
which lines to emit, and a span that is not in its line is dropped or
re-anchored inside that window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .anchor import SpanHit, place_span
from .claude import Spend, cached_system, parse_structured
from .prompts import EXTRACT_INSTRUCTIONS
from .schemas import ExtractionBatch

WINDOW_LINES = 30


@dataclass
class DraftClaim:
    line_index: int
    span_start: int
    span_end: int
    span_text: str
    claim: str
    scope: str
    category: str
    verify_marker: bool
    verdict: str = ""
    error_type: str = ""
    rationale: str = ""
    source_says: str = ""
    quotes: list[str] = field(default_factory=list)
    judge_trail: list[dict[str, str]] = field(default_factory=list)
    severity: str = ""


def numbered_lines(lines: list[str]) -> str:
    return "\n".join(f"L{index}: {line}" for index, line in enumerate(lines))


def windows(n_lines: int, size: int = WINDOW_LINES) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` ranges covering every line."""
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < n_lines:
        end = min(n_lines, start + size)
        ranges.append((start, end))
        start = end
    return ranges


def _user_message(start: int, end: int) -> str:
    last = end - 1
    return (
        f"Extract claims ONLY for lines L{start} through L{last} inclusive.\n"
        "Line numbers start at 0. They are the indexes of script.splitlines(), "
        "already printed as L<n> in the script. Return that same number.\n"
        "Emit no claims for lines outside that range."
    )


def extract_claims(
    client: Any,
    script: str,
    *,
    model: str,
    spend: Spend,
) -> tuple[list[DraftClaim], int]:
    """Return anchored claims and how many model spans were dropped."""
    lines = script.splitlines()
    if not lines or not script.strip():
        return [], 0
    system = cached_system(EXTRACT_INSTRUCTIONS, "SCRIPT\n" + numbered_lines(lines))
    drafts: list[DraftClaim] = []
    dropped = 0
    for start, end in windows(len(lines)):
        batch = parse_structured(
            client,
            model=model,
            system=system,
            user=_user_message(start, end),
            output_format=ExtractionBatch,
            step="extract",
            spend=spend,
        )
        for raw in batch.claims:
            claim_text = (raw.claim or "").strip()
            if not claim_text:
                dropped += 1
                continue
            placed = place_span(lines, raw.line, raw.span, start, end)
            if placed is None:
                dropped += 1
                continue
            drafts.append(_draft(placed, claim_text, raw.scope, raw.category, raw.verify_marker))
    return drafts, dropped


def _draft(
    placed: SpanHit,
    claim: str,
    scope: str,
    category: str,
    verify_marker: bool,
) -> DraftClaim:
    return DraftClaim(
        line_index=placed.line_index,
        span_start=placed.start,
        span_end=placed.end,
        span_text=placed.text,
        claim=claim,
        scope=scope,
        category=category,
        verify_marker=verify_marker,
    )
