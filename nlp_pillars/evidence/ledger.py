"""The evidence ledger: facts the paper states, each tied to a checked quote.

Two kinds of entry, and the writer sees both as one list:

* ``F#`` — narrative facts. One Haiku call reads the whole paper and returns
  ``quote`` (verbatim), ``where``, ``statement`` and ``kind``. **Code checks
  every quote against the paper text** and drops what it cannot find; a
  statement that carries a number its own quote does not contain is dropped
  too. The first ``F`` entry is built from the paper's metadata, not by the
  model.
* ``T#`` — table rows. No model turns them into facts: each is a row of the
  page-image transcription in ``paper_tables`` (``table_repair``), rendered
  ``column=value``, so a checkmark column cannot be re-attributed by a writer
  guessing at a flattened table.

The ledger is stochastic (55-76 entries across three runs on one paper), so
what it omits the script omits. Completeness is measured, not assumed:
``scripts/podcast_writer_gate.py``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from ..factcheck.anchor import find_in_text
from ..factcheck.numbers import number_spans, supported
from ..llm_client import structured
from ..llm_routes import PODCAST_LEDGER_ROUTE
from .prompts import LEDGER_PROMPT_VERSION, LEDGER_SYSTEM, LEDGER_USER

logger = logging.getLogger(__name__)

KINDS = (
    "identity", "problem", "method", "data", "result", "comparison",
    "ablation", "limitation", "definition",
)
KIND_TABLE = "table_row"
KIND_OTHER = "other"

MIN_QUOTE_CHARS = 15
MAX_QUOTE_CHARS = 400
FUZZY_ACCEPT = 95.0
#: Table rows offered to the writer. A paper with more keeps the first ones
#: and says how many it left out.
MAX_TABLE_ENTRIES = 240
MAX_IDENTITY_AUTHORS = 6

SOURCE_PAPER = "paper"
SOURCE_METADATA = "metadata"
SOURCE_TABLE = "table"


# -- model-facing schema (small on purpose: see llm_client.structured) ---------


class LedgerEntryDraft(BaseModel):
    quote: str = Field(description="Verbatim passage from the paper")
    where: str = Field(description="Section or table")
    statement: str = Field(description="One self-contained fact")
    kind: str = Field(description="identity, problem, method, data, result, comparison, ablation, limitation or definition")


class LedgerDraft(BaseModel):
    entries: List[LedgerEntryDraft]


# -- stored shapes -------------------------------------------------------------


@dataclass(frozen=True)
class LedgerEntry:
    id: str
    kind: str
    statement: str
    where: str = ""
    quote: str = ""
    source: str = SOURCE_PAPER
    #: how the quote was found: exact | markup | fuzzy | "" (metadata, table).
    match: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "statement": self.statement,
            "where": self.where, "quote": self.quote, "source": self.source,
            "match": self.match,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LedgerEntry":
        return cls(
            id=str(data["id"]), kind=str(data.get("kind", KIND_OTHER)),
            statement=str(data.get("statement", "")), where=str(data.get("where", "")),
            quote=str(data.get("quote", "")), source=str(data.get("source", SOURCE_PAPER)),
            match=str(data.get("match", "")),
        )

    @property
    def is_table(self) -> bool:
        return self.id.startswith("T")

    def prompt_line(self) -> str:
        where = f"; {self.where}" if self.where and not self.is_table else ""
        return f"{self.id} [{self.kind}{where}] {self.statement}"


@dataclass
class Ledger:
    entries: List[LedgerEntry] = field(default_factory=list)
    #: entries the model proposed, before verification.
    proposed: int = 0
    dropped: List[dict[str, str]] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    fallback: bool = False
    table_rows_omitted: int = 0
    prompt_version: str = LEDGER_PROMPT_VERSION
    #: writer-time measurements (untagged lines, entries used, ...); filled in
    #: by the agent after the script is parsed.
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def narrative(self) -> List[LedgerEntry]:
        return [e for e in self.entries if not e.is_table]

    @property
    def tables(self) -> List[LedgerEntry]:
        return [e for e in self.entries if e.is_table]

    @property
    def drop_rate(self) -> Optional[float]:
        return len(self.dropped) / self.proposed if self.proposed else None

    def ids(self) -> set[str]:
        return {e.id for e in self.entries}

    def by_id(self) -> dict[str, LedgerEntry]:
        return {e.id: e for e in self.entries}

    def prompt_text(self) -> str:
        return "\n".join(e.prompt_line() for e in self.entries)

    def detail(self) -> str:
        bits = [f"{len(self.narrative)} facts"]
        if self.proposed:
            bits.append(f"{len(self.dropped)} of {self.proposed} proposed dropped")
        if self.tables:
            bits.append(f"{len(self.tables)} table rows")
        if self.table_rows_omitted:
            bits.append(f"{self.table_rows_omitted} table rows left out")
        bits.append(self.model)
        if self.input_tokens:
            bits.append(f"{self.input_tokens:,} in")
        if self.output_tokens:
            bits.append(f"{self.output_tokens:,} out")
        bits.append(f"${self.cost_usd:.6f}")
        return " · ".join(bits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_version": self.prompt_version,
            "model": self.model,
            "fallback": self.fallback,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "proposed": self.proposed,
            "dropped": self.dropped,
            "table_rows_omitted": self.table_rows_omitted,
            "stats": self.stats,
            "entries": [e.as_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Ledger":
        return cls(
            entries=[LedgerEntry.from_dict(e) for e in data.get("entries") or []],
            proposed=int(data.get("proposed", 0)),
            dropped=list(data.get("dropped") or []),
            model=str(data.get("model", "")),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            fallback=bool(data.get("fallback", False)),
            table_rows_omitted=int(data.get("table_rows_omitted", 0)),
            prompt_version=str(data.get("prompt_version", LEDGER_PROMPT_VERSION)),
            stats=dict(data.get("stats") or {}),
        )


# -- quote verification ---------------------------------------------------------

_FOLD = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
})
_SQUASH = re.compile(r"[^a-z0-9.%]+")


def squash(text: str) -> str:
    """Lower-case letters, digits, ``.`` and ``%`` only.

    Whitespace, markdown (``**``, pipes, ``[text](url)``), quotes, commas and
    hyphenation are exactly what a model's copy and the extractor's markdown
    disagree about. Digits and the decimal point stay, so ``91.5`` and ``9.15``
    remain different quotes.
    """
    return _SQUASH.sub("", text.translate(_FOLD).lower())


_NUMERAL = re.compile(r"\d+(?:[.,]\d+)*")


def _numerals(text: str) -> List[str]:
    return [m.group(0).replace(",", "") for m in _NUMERAL.finditer(text)]


@dataclass(frozen=True)
class QuoteCheck:
    ok: bool
    quote: str = ""
    match: str = ""
    reason: str = ""


def verify_quote(quote: str, text: str, squashed: Optional[str] = None) -> QuoteCheck:
    """Is ``quote`` in ``text``? Exact, then markup-insensitive, then fuzzy >= 95.

    The returned quote is the paper's own characters when the match was exact or
    fuzzy, so what is stored is never the model's rendering of the paper.
    """
    q = (quote or "").strip()
    if len(q) < MIN_QUOTE_CHARS:
        return QuoteCheck(False, reason=f"quote shorter than {MIN_QUOTE_CHARS} characters")
    if len(q) > MAX_QUOTE_CHARS:
        return QuoteCheck(False, reason=f"quote longer than {MAX_QUOTE_CHARS} characters")
    if q in text:
        return QuoteCheck(True, q, "exact")
    if squashed is None:
        squashed = squash(text)
    needle = squash(q)
    if needle and needle in squashed:
        hit = find_in_text(text, q)
        if hit is not None and hit[3] >= 100.0:
            return QuoteCheck(True, hit[2], "exact")
        return QuoteCheck(True, q, "markup")
    hit = find_in_text(text, q)
    if hit is not None and hit[3] >= FUZZY_ACCEPT:
        # A fuzzy match is how a copied quote survives a stray character, and
        # also how 9.15 passes for 91.5. The numbers must agree exactly.
        if _numerals(q) != _numerals(hit[2]):
            return QuoteCheck(False, reason="quote differs from the paper in its numbers")
        return QuoteCheck(True, hit[2], "fuzzy")
    return QuoteCheck(False, reason="quote not found in the paper text")


def _needs_support(value: float) -> bool:
    """Small whole numbers are structural ("two baselines"); the rest must be quoted."""
    return not (float(value).is_integer() and abs(value) <= 12)


_GROUPED = re.compile(r"(?<=\d)[ \u2009\u202f\u00a0](?=\d{3}\b)")
_SPLIT_DECIMAL = re.compile(r"(?<=\d)\s*\.\s*(?=\d)")


def numeric_view(text: str) -> str:
    """The quote as its numbers are meant to be read.

    The extractor writes ``0 _._ 6`` for 0.6 and ``30 000`` for 30,000; a
    statement that says 0.6 or 30,000 is right, and must not be dropped for it.
    """
    text = re.sub(r"[*_`]", "", text)
    text = _SPLIT_DECIMAL.sub(".", text)
    return _GROUPED.sub("", text)


def unsupported_numbers(statement: str, quote: str) -> List[str]:
    """Numbers the statement states that its own quote does not contain."""
    have = {s.value for s in number_spans(numeric_view(quote))}
    missing = []
    for span in number_spans(statement):
        if _needs_support(span.value) and not supported(span.value, have):
            missing.append(span.raw)
    return missing


# -- table rows -------------------------------------------------------------------


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def row_statement(
    label: str, caption: str, columns: Sequence[str], cells: Sequence[str]
) -> Optional[str]:
    """``Table 9 (caption), row "name": col=value; col=value`` or None.

    A row with nothing beyond its name is a sub-heading, not a result. Empty
    cells are written ``(empty)`` so a checkmark grid keeps its blanks.
    """
    cells = [" ".join(str(c).split()) for c in cells]
    if len(cells) < 2 or not any(cells[1:]):
        return None
    name = cells[0]
    parts = []
    for j, value in enumerate(cells[1:], start=1):
        column = columns[j] if j < len(columns) and str(columns[j]).strip() else f"column {j + 1}"
        parts.append(f"{' '.join(str(column).split())}={value or '(empty)'}")
    head = f"Table {label}"
    if caption:
        head += f" ({_clip(caption, 90)})"
    row = f'row "{name}"' if name else "row"
    return f"{head}, {row}: " + "; ".join(parts)


def table_entries(
    tables: Iterable[dict[str, Any]], *, cap: int = MAX_TABLE_ENTRIES
) -> tuple[List[LedgerEntry], int]:
    """``T#`` entries from stored ``paper_tables`` rows, and how many were left out.

    Only ``spliced`` tables: those agreed with the PDF's text layer at 95% or
    better. A flagged table is stored for a person to look at; it is not
    evidence.
    """
    entries: List[LedgerEntry] = []
    omitted = 0
    ordered = sorted(
        (t for t in tables if t.get("status") == "spliced"),
        key=lambda t: (int(t.get("table_index") or 0), str(t.get("label"))),
    )
    for table in ordered:
        label = str(table.get("label") or "").strip()
        caption = str(table.get("caption") or "")
        columns = [str(c) for c in table.get("columns") or []]
        for row in table.get("rows") or []:
            statement = row_statement(label, caption, columns, [str(c) for c in row])
            if statement is None:
                continue
            if len(entries) >= cap:
                omitted += 1
                continue
            entries.append(LedgerEntry(
                id=f"T{len(entries) + 1}", kind=KIND_TABLE, statement=statement,
                where=f"Table {label}", quote=" | ".join(str(c) for c in row),
                source=SOURCE_TABLE,
            ))
    return entries, omitted


# -- building ---------------------------------------------------------------------


def paper_view(title: str, authors: Sequence[str], year: Any, abstract: Optional[str], body: str) -> str:
    """The text the ledger model reads, and the text its quotes are checked against."""
    return (
        f"Title: {title}\n"
        f"Authors: {', '.join(authors) if authors else 'Unknown'}\n"
        f"Year: {year or 'Unknown'}\n"
        f"Abstract: {abstract or 'No abstract available'}\n\n"
        f"=== FULL PAPER TEXT ===\n"
        f"{body if body else '[Full text not available - using the abstract only]'}\n"
        f"=== END FULL TEXT ===\n"
    )


def identity_statement(title: str, authors: Sequence[str], year: Any) -> str:
    shown = list(authors)[:MAX_IDENTITY_AUTHORS]
    who = ""
    if shown:
        who = " by " + ", ".join(shown)
        if len(authors) > len(shown):
            who += " and others"
    when = f" ({year})" if year else ""
    return f'The paper is titled "{title}"{who}{when}.'


def entry_target(text_chars: int) -> tuple[int, int]:
    """How many entries to ask for: roughly one per thousand characters."""
    lo = min(70, max(40, text_chars // 1000))
    return lo, lo + 30


def verify_draft(
    draft: LedgerDraft, view: str, *, first_id: int = 1
) -> tuple[List[LedgerEntry], List[dict[str, str]]]:
    """Keep the entries whose quote is in ``view`` and whose numbers are quoted."""
    squashed = squash(view)
    kept: List[LedgerEntry] = []
    dropped: List[dict[str, str]] = []
    seen: set[str] = set()
    for item in draft.entries:
        statement = " ".join(item.statement.split())
        check = verify_quote(item.quote, view, squashed)
        if not statement:
            dropped.append({"quote": item.quote[:250], "reason": "empty statement"})
            continue
        if not check.ok:
            dropped.append({"quote": item.quote[:250], "reason": check.reason})
            continue
        missing = unsupported_numbers(statement, check.quote)
        if missing:
            dropped.append({
                "quote": item.quote[:120],
                "reason": "statement has numbers its quote does not: " + ", ".join(missing),
            })
            continue
        key = squash(check.quote) + "|" + squash(statement)
        if key in seen:
            dropped.append({"quote": item.quote[:250], "reason": "duplicate entry"})
            continue
        seen.add(key)
        kind = item.kind.strip().lower()
        kept.append(LedgerEntry(
            id=f"F{first_id + len(kept)}",
            kind=kind if kind in KINDS else KIND_OTHER,
            statement=statement, where=" ".join(item.where.split()),
            quote=check.quote, source=SOURCE_PAPER, match=check.match,
        ))
    return kept, dropped


def build_ledger(
    *,
    title: str,
    authors: Sequence[str],
    year: Any,
    abstract: Optional[str],
    body: str,
    tables: Iterable[dict[str, Any]] = (),
    client: Any = None,
    route=PODCAST_LEDGER_ROUTE,
) -> Ledger:
    """Build and verify the ledger for one paper (one model call).

    Raises:
        nlp_pillars.llm_client.LLMCallError: the model call failed.
    """
    view = paper_view(title, authors, year, abstract, body)
    lo, hi = entry_target(len(body))
    call = structured(
        route,
        LEDGER_SYSTEM.format(lo=lo, hi=hi),
        LEDGER_USER.format(paper=view),
        LedgerDraft,
        client=client,
    )
    identity = LedgerEntry(
        id="F1", kind="identity", statement=identity_statement(title, authors, year),
        where="metadata", source=SOURCE_METADATA,
    )
    kept, dropped = verify_draft(call.parsed, view, first_id=2)
    rows, omitted = table_entries(tables)
    ledger = Ledger(
        entries=[identity, *kept, *rows],
        proposed=len(call.parsed.entries),
        dropped=dropped,
        model=call.model,
        input_tokens=call.input_tokens,
        output_tokens=call.output_tokens,
        cost_usd=call.cost_usd,
        fallback=call.fallback,
        table_rows_omitted=omitted,
    )
    logger.warning(
        "Ledger for %r: %s", title[:60], ledger.detail(),
    )
    return ledger
