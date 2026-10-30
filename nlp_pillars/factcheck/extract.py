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
from .coverage import (
    WindowStat,
    describe,
    number_bearing_lines,
    substantive_host_lines,
)
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


def _retry_message(start: int, end: int, targets: list[int], empty: bool) -> str:
    named = ", ".join(f"L{n}" for n in targets)
    if empty:
        return (
            _user_message(start, end)
            + "\nA first pass over this range returned no claims at all, but "
            f"it has substantive host lines (figures or talk on: {named}).\n"
            "Read the range again and emit every checkable claim in it."
        )
    return (
        _user_message(start, end)
        + "\nA first pass over this range returned no claim for these lines, "
        f"and they state figures: {named}.\n"
        "Emit claims for those lines only, keeping each number with what it "
        "measures. A line with nothing checkable still gets none."
    )


@dataclass
class Extraction:
    claims: list[DraftClaim]
    dropped: int
    windows: list[WindowStat]

    def describe(self) -> str:
        return describe(self.windows, len(self.claims), self.dropped)


def _run_window(
    client: Any,
    system: Any,
    lines: list[str],
    start: int,
    end: int,
    user: str,
    *,
    model: str,
    spend: Spend,
) -> tuple[list[DraftClaim], int]:
    batch = parse_structured(
        client,
        model=model,
        system=system,
        user=user,
        output_format=ExtractionBatch,
        step="extract",
        spend=spend,
    )
    drafts: list[DraftClaim] = []
    dropped = 0
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


def _retry_targets(
    lines: list[str], start: int, end: int, drafts: list[DraftClaim],
) -> list[int]:
    """Lines a second pass should be told about, or none.

    A host line that states a figure but carries no claim is always a target.
    An empty window with substantive host lines but no figures is retried as a
    whole, since a model that returned nothing for thirty lines of talk has
    most likely failed rather than found nothing.
    """
    covered = {claim.line_index for claim in drafts}
    figures = [n for n in number_bearing_lines(lines, start, end) if n not in covered]
    if figures:
        return figures
    if not drafts:
        return substantive_host_lines(lines, start, end)
    return []


def extract_claims(
    client: Any,
    script: str,
    *,
    model: str,
    spend: Spend,
) -> Extraction:
    """Anchored claims, how many model spans were dropped, and per-window stats.

    Each window gets one retry when it left a number-bearing host line without
    a claim (or came back empty over real host talk). The retry is a second
    call on the same cached script block, so it costs output tokens only.
    """
    lines = script.splitlines()
    if not lines or not script.strip():
        return Extraction([], 0, [])
    system = cached_system(EXTRACT_INSTRUCTIONS, "SCRIPT\n" + numbered_lines(lines))
    drafts: list[DraftClaim] = []
    stats: list[WindowStat] = []
    for start, end in windows(len(lines)):
        found, dropped = _run_window(
            client, system, lines, start, end, _user_message(start, end),
            model=model, spend=spend,
        )
        stat = WindowStat(start, end, first_pass=len(found), dropped=dropped)
        targets = _retry_targets(lines, start, end, found)
        if targets:
            stat.retried = True
            again, more = _run_window(
                client, system, lines, start, end,
                _retry_message(start, end, targets, empty=not found),
                model=model, spend=spend,
            )
            stat.dropped += more
            seen = {(c.line_index, c.span_start, c.span_end) for c in found}
            # After a partial pass only the lines it missed are taken, so a
            # claim it already found is not stated twice.
            wanted = set(range(start, end)) if not found else set(targets)
            for claim in again:
                key = (claim.line_index, claim.span_start, claim.span_end)
                if key in seen or claim.line_index not in wanted:
                    continue
                seen.add(key)
                found.append(claim)
        covered = {claim.line_index for claim in found}
        stat.claims = len(found)
        stat.uncovered = [
            n for n in number_bearing_lines(lines, start, end) if n not in covered
        ]
        drafts.extend(found)
        stats.append(stat)
    return Extraction(drafts, sum(s.dropped for s in stats), stats)


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
