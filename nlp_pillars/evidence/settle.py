"""Turn the writer's tagged output into the stored script, and measure it.

``settle`` is deterministic and free. It parses the citation tags, strips them,
checks every cited id against the ledger, and records the numbers the gate
reads: untagged lines, ids the ledger does not have, how much of the ledger the
script used, and which lines state a number their cited entries do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..factcheck.numbers import number_spans, supported
from .ledger import Ledger, _needs_support
from .tags import ParsedScript, is_spoken, parse_citations, resolve_ids

#: Lines listed by index in the stored stats; the counts are always complete.
MAX_LISTED = 25


@dataclass
class Settled:
    script: str
    #: 0-based line index of ``script`` -> ids. ``[]`` is ``[[-]]``.
    citations: dict[int, list[str]]
    stats: dict[str, Any] = field(default_factory=dict)

    def as_storage(self) -> dict[str, list[str]]:
        return {str(i): ids for i, ids in sorted(self.citations.items())}


def _line_number_violations(
    lines: list[str], citations: dict[int, list[str]], ledger: Ledger
) -> list[dict[str, Any]]:
    by_id = ledger.by_id()
    found: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if index not in citations or not is_spoken(line):
            continue
        cited = [by_id[i].statement for i in citations[index] if i in by_id]
        have = {s.value for s in number_spans(" ".join(cited))}
        missing = [
            span.raw for span in number_spans(line)
            if _needs_support(span.value) and not supported(span.value, have)
        ]
        if missing:
            found.append({"line": index, "numbers": missing})
    return found


def settle(raw_script: str, ledger: Ledger) -> Settled:
    """Parse, strip and measure one tagged script against its ledger."""
    parsed: ParsedScript = parse_citations(raw_script)
    citations, unknown, unresolved = resolve_ids(parsed.citations, ledger.ids())

    used = {i for ids in citations.values() for i in ids}
    narrative = {e.id for e in ledger.narrative}
    tables = {e.id for e in ledger.tables}
    lines = parsed.script.splitlines()
    violations = _line_number_violations(lines, citations, ledger)

    stats = {
        "spoken_lines": parsed.spoken_lines,
        "tagged_lines": len(citations),
        "no_fact_lines": sum(1 for ids in citations.values() if not ids),
        "untagged_lines": len(parsed.untagged),
        "untagged_at": parsed.untagged[:MAX_LISTED],
        "invalid_tags": len(parsed.invalid),
        "unknown_ids": sorted({i for _, i in unknown}),
        "unresolved_lines": unresolved[:MAX_LISTED],
        "narrative_entries": len(narrative),
        "narrative_used": len(used & narrative),
        "narrative_used_ratio": (
            round(len(used & narrative) / len(narrative), 4) if narrative else None
        ),
        "table_entries": len(tables),
        "table_used": len(used & tables),
        "number_violation_lines": len(violations),
        "number_violations": violations[:MAX_LISTED],
    }
    return Settled(script=parsed.script, citations=citations, stats=stats)


def warnings_for(ledger: Ledger, stats: dict[str, Any]) -> list[str]:
    """What a reader of an evidence-mode script should be told."""
    out: list[str] = []
    if stats.get("untagged_lines"):
        out.append(
            f"{stats['untagged_lines']} spoken line(s) came back without a ledger "
            f"citation, so they were written without a recorded basis."
        )
    if stats.get("unknown_ids") or stats.get("unresolved_lines"):
        out.append(
            "Some lines cite ledger entries that do not exist: "
            + ", ".join(stats.get("unknown_ids") or ["(none listed)"]) + "."
        )
    rate = ledger.drop_rate
    if rate is not None and rate > 0.10:
        out.append(
            f"{len(ledger.dropped)} of {ledger.proposed} ledger entries failed quote "
            f"verification and were dropped ({rate:.0%}), so the script may leave out "
            f"parts of the paper."
        )
    if ledger.table_rows_omitted:
        out.append(
            f"{ledger.table_rows_omitted} table row(s) did not fit the ledger and were left out."
        )
    return out
