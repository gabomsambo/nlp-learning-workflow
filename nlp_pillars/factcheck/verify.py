"""Whole-paper screening, then a re-judge of what the screen flagged.

The paper is the cached system suffix, in full. Retrieved chunks are not
sent. Claims whose screen verdict is supported or not_checkable stay as they
are. Every other claim is judged again.
"""

from __future__ import annotations

from typing import Any

from .claude import Spend, cached_system, parse_structured
from .errors import FactcheckError
from .extract import DraftClaim
from .prompts import verification_instructions
from .schemas import TERMINAL_SCREEN_VERDICTS, ClaimVerdict, VerdictBatch

BATCH_SIZE = 15


def _user_message(claims: list[tuple[str, DraftClaim]]) -> str:
    blocks = []
    for claim_id, claim in claims:
        blocks.append(
            f"id: {claim_id}\n"
            f"scope: {claim.scope}\n"
            f"category: {claim.category}\n"
            f"claim: {claim.claim}"
        )
    return "CLAIMS\n\n" + "\n\n".join(blocks)


def _batches(pairs: list[tuple[str, DraftClaim]]) -> list[list[tuple[str, DraftClaim]]]:
    return [pairs[i:i + BATCH_SIZE] for i in range(0, len(pairs), BATCH_SIZE)]


def _apply(batch: list[ClaimVerdict], expected: dict[str, DraftClaim], step: str, model: str) -> None:
    ids = [item.id for item in batch]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        missing = sorted(set(expected) - set(ids))
        extra = sorted(set(ids) - set(expected))
        raise_message = (
            f"{step} verdicts do not match the claims sent "
            f"(missing {missing}, extra {extra})"
        )
        raise ValueError(raise_message)
    for item in batch:
        claim = expected[item.id]
        claim.verdict = item.verdict
        claim.error_type = item.error_type
        claim.rationale = item.rationale.strip()
        claim.source_says = item.source_says.strip()
        claim.quotes = [quote.strip() for quote in item.quotes if quote and quote.strip()][:3]
        claim.judge_trail.append({
            "step": step,
            "model": model,
            "verdict": item.verdict,
        })


def _judge(
    client: Any,
    claims: list[DraftClaim],
    *,
    paper_text: str,
    model: str,
    step: str,
    rejudge: bool,
    spend: Spend,
) -> None:
    if not claims:
        return
    instructions = verification_instructions(rejudge=rejudge)
    system = cached_system(instructions, "SOURCE\n" + paper_text)
    # Ids are stable inside this call's list. A re-judge list is only the
    # flagged claims, so its ids are not the screen ids.
    pairs = [(str(index), claim) for index, claim in enumerate(claims)]
    for batch_pairs in _batches(pairs):
        expected = {claim_id: claim for claim_id, claim in batch_pairs}
        parsed = parse_structured(
            client,
            model=model,
            system=system,
            user=_user_message(batch_pairs),
            output_format=VerdictBatch,
            step=step,
            spend=spend,
        )
        try:
            _apply(parsed.results, expected, step, model)
        except ValueError as exc:
            raise FactcheckError(
                str(exc),
                cost_usd=spend.total,
                calls=spend.calls,
            ) from exc


def screen_claims(
    client: Any,
    claims: list[DraftClaim],
    *,
    paper_text: str,
    model: str,
    spend: Spend,
) -> None:
    _judge(
        client,
        claims,
        paper_text=paper_text,
        model=model,
        step="screen",
        rejudge=False,
        spend=spend,
    )


def rejudge_claims(
    client: Any,
    claims: list[DraftClaim],
    *,
    paper_text: str,
    model: str,
    spend: Spend,
) -> None:
    flagged = [
        claim for claim in claims
        if claim.verdict not in TERMINAL_SCREEN_VERDICTS
    ]
    _judge(
        client,
        flagged,
        paper_text=paper_text,
        model=model,
        step="rejudge",
        rejudge=True,
        spend=spend,
    )
