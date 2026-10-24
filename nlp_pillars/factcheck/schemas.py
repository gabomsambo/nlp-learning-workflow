"""Small model-facing schemas.

Field order on ``ClaimVerdict`` is load-bearing. With the verdict before the
rationale, Haiku emitted "contradicted" while the rationale said the claim
was supported. Rationale, then verdict.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Scope = Literal["paper", "background"]
Category = Literal[
    "number",
    "method",
    "setup",
    "result",
    "comparison",
    "attribution",
    "limitation",
    "implication",
    "background",
]
Verdict = Literal[
    "supported",
    "partially_supported",
    "contradicted",
    "unsupported",
    "not_checkable",
]
ErrorType = Literal[
    "number_mismatch",
    "invented_detail",
    "overstatement",
    "wrong_attribution",
    "method_misdescribed",
    "outside_knowledge",
    "other",
    "none",
]

# Screen keeps these. Everything else is re-judged.
TERMINAL_SCREEN_VERDICTS = frozenset({"supported", "not_checkable"})


class ExtractedClaim(BaseModel):
    """One claim from a 30-line window."""

    model_config = ConfigDict(extra="forbid")

    line: int
    span: str
    claim: str
    scope: Scope
    category: Category
    verify_marker: bool


class ExtractionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ExtractedClaim] = Field(default_factory=list)


class ClaimVerdict(BaseModel):
    """One claim's judgement. Keep this field order."""

    model_config = ConfigDict(extra="forbid")

    id: str
    quotes: list[str]
    source_says: str
    rationale: str
    verdict: Verdict
    error_type: ErrorType


class VerdictBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[ClaimVerdict]
