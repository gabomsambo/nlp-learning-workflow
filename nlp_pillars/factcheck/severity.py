"""Deterministic severity from verdict, scope, category, and error type.

The table is the fact-check research report, section 4.4. Supported and
not-checkable claims are not highlighted as wrong.
"""

from __future__ import annotations

SPECIFIC_CATEGORIES = frozenset({
    "number",
    "result",
    "comparison",
    "method",
    "setup",
    "attribution",
})

PARTIAL_OVERSTATEMENT_CATEGORIES = frozenset({
    "result",
    "comparison",
    "number",
})


def assign_severity(
    verdict: str,
    scope: str,
    category: str,
    error_type: str,
) -> str:
    """Return ``high``, ``medium``, ``low``, or ``none``."""
    if verdict in {"supported", "not_checkable"}:
        return "none"
    if verdict == "contradicted":
        if scope == "paper":
            return "high"
        return "medium"
    if verdict == "unsupported":
        if (
            scope == "background"
            or category == "background"
            or error_type == "outside_knowledge"
        ):
            return "low"
        if scope == "paper" and category in SPECIFIC_CATEGORIES:
            return "high"
        if scope == "paper" and category in {"implication", "limitation"}:
            return "medium"
        if scope == "paper":
            return "medium"
        return "low"
    if verdict == "partially_supported":
        if error_type in {"number_mismatch", "wrong_attribution"}:
            return "medium"
        if (
            error_type == "overstatement"
            and category in PARTIAL_OVERSTATEMENT_CATEGORIES
        ):
            return "medium"
        return "low"
    raise ValueError(f"unknown verdict {verdict!r}")
