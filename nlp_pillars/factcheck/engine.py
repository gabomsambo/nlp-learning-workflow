"""Run extraction, screening, re-judgement, anchoring, and severity.

Persistence happens in ``execute_factcheck`` and only after the pipeline
returns. A raised ``FactcheckError`` means no row was written and the script
text was not modified.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any

from .. import db
from ..podcast_models import (
    FACTCHECK_EXTRACT,
    FACTCHECK_REJUDGE,
    FACTCHECK_SCREEN,
)
from .anchor import evidence_for_quotes
from .claude import Spend, make_client
from .constants import PROMPT_VERSION
from .cost import require_price
from .errors import FactcheckError
from .extract import DraftClaim, extract_claims
from .severity import assign_severity
from .verify import rejudge_claims, screen_claims

_COUNT_KEY = {
    "supported": "supported",
    "partially_supported": "partial",
    "contradicted": "contradicted",
    "unsupported": "unsupported",
    "not_checkable": "not_checkable",
}


@dataclass
class FactcheckResult:
    dry_run: bool
    script_id: str
    paper_id: str
    script_sha256: str
    paper_text_sha256: str
    paper_chars: int
    chunk_count: int
    prompt_version: str
    models: dict[str, Any]
    counts: dict[str, int]
    cost_usd: float
    calls: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    fact_check_id: str | None = None

    def header(self) -> dict[str, Any]:
        return {
            "script_id": self.script_id,
            "script_sha256": self.script_sha256,
            "paper_text_sha256": self.paper_text_sha256,
            "status": "succeeded",
            "models": {
                "extract": self.models["extract"],
                "screen": self.models["screen"],
                "rejudge": self.models["rejudge"],
                "calls": self.calls,
            },
            "prompt_version": self.prompt_version,
            "counts": self.counts,
            "cost_usd": self.cost_usd,
        }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def route_models() -> dict[str, str]:
    return {
        "extract": FACTCHECK_EXTRACT.resolved_model(),
        "screen": FACTCHECK_SCREEN.resolved_model(),
        "rejudge": FACTCHECK_REJUDGE.resolved_model(),
    }


def _counts(claims: list[DraftClaim]) -> dict[str, int]:
    counts = {
        "claims": len(claims),
        "supported": 0,
        "partial": 0,
        "contradicted": 0,
        "unsupported": 0,
        "not_checkable": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
    }
    for claim in claims:
        counts[_COUNT_KEY[claim.verdict]] += 1
        if claim.severity in {"high", "medium", "low"}:
            counts[claim.severity] += 1
    return counts


def _claim_row(
    claim: DraftClaim,
    paper_text: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence, status = evidence_for_quotes(
        paper_text, chunks, claim.quotes, verdict=claim.verdict,
    )
    severity = assign_severity(
        claim.verdict, claim.scope, claim.category, claim.error_type,
    )
    claim.severity = severity
    return {
        "line_index": claim.line_index,
        "span_start": claim.span_start,
        "span_end": claim.span_end,
        "span_text": claim.span_text,
        "claim": claim.claim,
        "scope": claim.scope,
        "category": claim.category,
        "verify_marker": claim.verify_marker,
        "verdict": claim.verdict,
        "error_type": claim.error_type,
        "severity": severity,
        "rationale": claim.rationale,
        "source_says": claim.source_says,
        "evidence": evidence,
        "evidence_status": status,
        "judge_trail": claim.judge_trail,
    }


def run_factcheck(
    script_text: str,
    paper_text: str,
    chunks: list[dict[str, Any]],
    client: Any,
    *,
    script_id: str,
    paper_id: str,
    models: dict[str, str] | None = None,
    budget_usd: float | None = None,
) -> FactcheckResult:
    """Check ``script_text`` against the whole ``paper_text``. Do not persist."""
    if not script_text or not script_text.strip():
        raise FactcheckError("script text is empty")
    if not paper_text or not paper_text.strip():
        raise FactcheckError(
            f"no stored paper text for {paper_id}. "
            "Fact-checking reads paper_texts and does not download the PDF."
        )
    chosen = dict(models or route_models())
    for name in ("extract", "screen", "rejudge"):
        if name not in chosen or not chosen[name]:
            raise FactcheckError(f"missing {name} model")
        require_price(chosen[name])

    spend = Spend(budget_usd)
    claims, _dropped = extract_claims(
        client, script_text, model=chosen["extract"], spend=spend,
    )
    screen_claims(
        client,
        claims,
        paper_text=paper_text,
        model=chosen["screen"],
        spend=spend,
    )
    rejudge_claims(
        client,
        claims,
        paper_text=paper_text,
        model=chosen["rejudge"],
        spend=spend,
    )
    rows = [_claim_row(claim, paper_text, chunks) for claim in claims]
    counts = _counts(claims)
    return FactcheckResult(
        dry_run=False,
        script_id=script_id,
        paper_id=paper_id,
        script_sha256=sha256_text(script_text),
        paper_text_sha256=sha256_text(paper_text),
        paper_chars=len(paper_text),
        chunk_count=len(chunks),
        prompt_version=PROMPT_VERSION,
        models=chosen,
        counts=counts,
        cost_usd=spend.total,
        calls=list(spend.calls),
        claims=rows,
    )


def _require_uuid(script_id: str) -> None:
    try:
        uuid.UUID(str(script_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise FactcheckError(f"script id {script_id!r} is not a uuid") from exc


def execute_factcheck(
    script_id: str,
    *,
    dry_run: bool = False,
    budget_usd: float | None = None,
    client: Any = None,
) -> FactcheckResult:
    """Load the script and paper text, check, and insert on success.

    ``--dry-run`` reads both and returns. It does not call a model and does
    not write. The script row is only ever read.
    """
    _require_uuid(script_id)
    script = db.get_podcast_script_by_id(script_id)
    if script is None:
        raise FactcheckError(f"no podcast script {script_id}")
    paper_text = db.get_stored_paper_text(script.paper_id)
    if not paper_text:
        raise FactcheckError(
            f"no stored paper text for {script.paper_id}. "
            "Fact-checking reads paper_texts and does not download the PDF."
        )
    chunks = db.list_paper_chunks(script.paper_id)
    models = route_models()
    if dry_run:
        return FactcheckResult(
            dry_run=True,
            script_id=script_id,
            paper_id=script.paper_id,
            script_sha256=sha256_text(script.script),
            paper_text_sha256=sha256_text(paper_text),
            paper_chars=len(paper_text),
            chunk_count=len(chunks),
            prompt_version=PROMPT_VERSION,
            models=models,
            counts={},
            cost_usd=0.0,
        )
    if client is None:
        client = make_client()
    result = run_factcheck(
        script.script,
        paper_text,
        chunks,
        client,
        script_id=script_id,
        paper_id=script.paper_id,
        models=models,
        budget_usd=budget_usd,
    )
    try:
        result.fact_check_id = db.insert_podcast_fact_check(
            result.header(), result.claims,
        )
    except db.FactcheckSaveError as exc:
        raise FactcheckError(
            f"{exc} The script was not changed and the fact check was not saved.",
            cost_usd=result.cost_usd,
            calls=result.calls,
        ) from exc
    return result
