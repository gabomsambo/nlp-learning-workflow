"""Vocabulary-aware concept extraction, persistence, and one-off backfill."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field

from . import db
from .concept_cost import (
    ConceptCall,
    call_from_completion,
    call_from_exception,
    estimate_backfill_call,
    require_price,
)
from .config import get_settings
from .graph import canonicalize_term

logger = logging.getLogger(__name__)

MIGRATION = "docs/migrations/024_vocabulary_concepts.sql"
MAX_CONCEPTS = 8
MAX_PROPOSALS = 2
MAX_VOCABULARY = 1000
_PAGE = 200


class ConceptExtraction(BaseModel):
    """Small structured result for the historical-note backfill."""

    concepts: list[str] = Field(default_factory=list, max_length=MAX_CONCEPTS)
    proposed_concepts: list[str] = Field(
        default_factory=list, max_length=MAX_PROPOSALS
    )


class ConceptExtractionError(RuntimeError):
    """A call or its locally validated result failed visibly."""

    def __init__(self, message: str, *, call: ConceptCall | None = None):
        super().__init__(message)
        self.call = call


class ConceptStoreError(RuntimeError):
    """Vocabulary concepts could not be read or atomically replaced."""


@dataclass
class BackfillReport:
    dry_run: bool
    model: str
    candidates: int
    estimated_per_call: float
    calls: list[ConceptCall] = field(default_factory=list)
    failed: int = 0
    failure: str = ""

    @property
    def cost_usd(self) -> float:
        return round(sum(call.cost_usd for call in self.calls), 6)

    def render(self) -> str:
        lines = [
            f"concept backfill ({'dry-run' if self.dry_run else 'write'})",
            f"model {self.model}",
            f"notes to process {self.candidates}",
            f"estimated cost_usd per call {self.estimated_per_call:.6f}",
            f"estimated cost_usd total "
            f"{self.estimated_per_call * self.candidates:.6f}",
        ]
        if self.dry_run:
            lines.append("No model call. Nothing written.")
            return "\n".join(lines)
        lines.extend(call.render() for call in self.calls)
        lines.append(f"actual cost_usd total {self.cost_usd:.6f}")
        lines.append(f"saved {len(self.calls) - self.failed}")
        if self.failure:
            lines.append(f"failed {self.failure}")
            lines.append(
                "Stopped. Concepts saved earlier in this run stay saved; "
                "the failed paper's previous concepts are untouched."
            )
        return "\n".join(lines)


def _store_error(error: Any) -> ConceptStoreError:
    text = str(error).lower()
    if any(
        token in text
        for token in (
            "replace_paper_concepts",
            "concept_mastery",
            "pgrst202",
            "schema cache",
            "could not find the function",
        )
    ):
        return ConceptStoreError(
            f"Vocabulary-aware concepts are not available ({error}). "
            f"Apply {MIGRATION} by hand before a model call."
        )
    return ConceptStoreError(f"Could not update vocabulary concepts: {error}")


def _read_rows(table: str, columns: str, order: str) -> list[dict]:
    client = db.get_client()
    rows: list[dict] = []
    offset = 0
    while True:
        response = (
            client.table(table)
            .select(columns)
            .order(order)
            .limit(_PAGE)
            .offset(offset)
            .execute()
        )
        if response["error"]:
            raise _store_error(response["error"])
        page = response["data"] or []
        rows.extend(page)
        if len(page) < _PAGE:
            return rows
        offset += len(page)


def probe_migration() -> None:
    """Fail before spending if migration 024 is not in PostgREST's cache."""
    response = db.get_client().rpc(
        "concept_mastery",
        {"p_user_id": "default_user", "p_pillar_id": None},
    )
    if response["error"]:
        raise _store_error(response["error"])


def load_vocabulary() -> list[str]:
    """Deterministic bounded vocabulary sent to the model."""
    rows = _read_rows("concepts", "name", "name")
    return [
        str(row["name"]).strip()
        for row in rows[:MAX_VOCABULARY]
        if str(row.get("name") or "").strip()
    ]


def normalize_concepts(
    selected: Sequence[str],
    proposed: Sequence[str],
    vocabulary: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Resolve exact vocabulary choices and canonicalize bounded proposals."""
    by_fold = {name.casefold(): name for name in vocabulary}
    chosen: list[str] = []
    unknown: list[str] = []
    for raw in selected:
        key = str(raw).strip().casefold()
        canonical = by_fold.get(key)
        if canonical is None:
            unknown.append(str(raw))
        elif canonical not in chosen:
            chosen.append(canonical)
    if unknown:
        raise ConceptExtractionError(
            "Model selected names outside the supplied vocabulary: "
            + ", ".join(unknown)
        )
    additions: list[str] = []
    for raw in proposed:
        name = canonicalize_term(str(raw))
        if not name:
            continue
        existing = by_fold.get(name.casefold())
        if existing:
            if existing not in chosen:
                chosen.append(existing)
            continue
        if name not in additions:
            additions.append(name)
    if len(chosen) > MAX_CONCEPTS or len(additions) > MAX_PROPOSALS:
        raise ConceptExtractionError(
            f"Concept result exceeds {MAX_CONCEPTS} selections or "
            f"{MAX_PROPOSALS} proposals"
        )
    if not chosen and not additions:
        raise ConceptExtractionError(
            "Model returned no vocabulary concepts and no new proposal"
        )
    return chosen, additions


def save_paper_concepts(
    paper_id: str,
    concepts: Sequence[str],
    proposed: Sequence[str],
    model: str,
) -> dict:
    names = list(dict.fromkeys([*concepts, *proposed]))
    response = db.get_client().rpc(
        "replace_paper_concepts",
        {
            "p_paper_id": paper_id,
            "p_concepts": [{"name": name, "weight": 1.0} for name in names],
            "p_source": f"llm:{model}",
        },
    )
    if response["error"]:
        raise _store_error(response["error"])
    return response["data"] or {}


def latest_notes_without_llm_concepts() -> list[dict]:
    """Latest stored note per paper, excluding papers already backfilled."""
    notes = _read_rows(
        "notes",
        "id,paper_id,pillar_id,problem,method,findings,limitations,"
        "future_work,key_terms,created_at",
        "created_at",
    )
    links = _read_rows("paper_concepts", "paper_id,source", "paper_id")
    paid = {
        str(row.get("paper_id"))
        for row in links
        if str(row.get("source") or "").startswith("llm:")
    }
    latest: dict[str, dict] = {}
    for note in notes:
        paper_id = str(note.get("paper_id") or "")
        if not paper_id or paper_id in paid:
            continue
        previous = latest.get(paper_id)
        marker = (str(note.get("created_at") or ""), str(note.get("id") or ""))
        previous_marker = (
            str(previous.get("created_at") or ""),
            str(previous.get("id") or ""),
        ) if previous else ("", "")
        if previous is None or marker > previous_marker:
            latest[paper_id] = note
    return [latest[key] for key in sorted(latest)]


def _backfill_messages(note: dict, vocabulary: Sequence[str]) -> list[dict]:
    system = (
        "Choose reusable NLP concepts for one existing paper note. Select exact "
        "names from the supplied vocabulary whenever possible. Put those exact "
        "names in concepts. Only when no existing name represents an important "
        f"idea, propose a concise canonical name in proposed_concepts; at most "
        f"{MAX_PROPOSALS}. Return at most {MAX_CONCEPTS} selected concepts."
    )
    body = {
        "note": {
            key: note.get(key)
            for key in (
                "problem",
                "method",
                "findings",
                "limitations",
                "future_work",
                "key_terms",
            )
        },
        "concept_vocabulary": list(vocabulary),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(body, ensure_ascii=False)},
    ]


def extract_note_concepts(
    client: Any, *, note: dict, vocabulary: Sequence[str], model: str
) -> tuple[list[str], list[str], ConceptCall]:
    """Make one structured call, price it, then validate the vocabulary result."""
    paper_id = str(note["paper_id"])
    require_price(model)
    try:
        result, completion = client.chat.completions.create_with_completion(
            model=model,
            response_model=ConceptExtraction,
            temperature=0.2,
            messages=_backfill_messages(note, vocabulary),
        )
    except Exception as exc:
        call = call_from_exception(
            purpose="concept_backfill",
            paper_id=paper_id,
            model=model,
            error=exc,
        )
        raise ConceptExtractionError(
            f"Instructor completion failed: {exc}", call=call
        ) from exc
    call = call_from_completion(
        purpose="concept_backfill",
        paper_id=paper_id,
        model=model,
        completion=completion,
    )
    try:
        selected, proposed = normalize_concepts(
            result.concepts, result.proposed_concepts, vocabulary
        )
    except ConceptExtractionError as exc:
        exc.call = call
        raise
    return selected, proposed, call


def _make_client():
    from .agents.summarizer_agent import _make_client as make_instructor_client

    return make_instructor_client()


def run_backfill(
    *,
    dry_run: bool,
    client_factory: Callable[[], Any] | None = None,
    writer: Callable[[str, Sequence[str], Sequence[str], str], dict] | None = None,
) -> BackfillReport:
    """Backfill historical notes, stopping visibly at the first failed paper."""
    model = get_settings().default_model
    estimate = estimate_backfill_call(model)
    notes = latest_notes_without_llm_concepts()
    report = BackfillReport(
        dry_run=dry_run,
        model=model,
        candidates=len(notes),
        estimated_per_call=estimate,
    )
    if dry_run or not notes:
        return report

    probe_migration()
    vocabulary = load_vocabulary()
    client = (client_factory or _make_client)()
    save = writer or save_paper_concepts
    for note in notes:
        paper_id = str(note["paper_id"])
        try:
            selected, proposed, call = extract_note_concepts(
                client,
                note=note,
                vocabulary=vocabulary,
                model=model,
            )
            report.calls.append(call)
            save(paper_id, selected, proposed, model)
        except ConceptExtractionError as exc:
            if exc.call is not None and exc.call not in report.calls:
                report.calls.append(exc.call)
            report.failed = 1
            report.failure = f"{paper_id}: {exc}"
            break
        except Exception as exc:
            report.failed = 1
            report.failure = f"{paper_id}: {exc}"
            break
        vocabulary = sorted(
            dict.fromkeys([*vocabulary, *selected, *proposed])
        )[:MAX_VOCABULARY]
    return report

