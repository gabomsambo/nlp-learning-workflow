"""Repair a paper's tables at ingestion with a page-image read.

``pymupdf4llm`` (layout off, the production setting) turns most tables into
loose text rows and loses some outright. Measured on 11 library papers
(``data/nlp-factcheck-prevention/report.md`` section 3.2): 321 of 3,432 numbers
visible in the printed tables were absent from the extracted text, and the AML
paper's results tables (2, 3, 13, 14) kept a caption and a header row only. The
podcast writer and the fact-checker read the same text, so neither can see what
the extractor dropped.

The repair, per paper:

1. **Select pages.** Pages whose text layer has a line that starts like a table
   caption (``Table 9:`` / ``Table 9 |`` / ``Table 9.``). Nothing else is sent.
2. **Transcribe.** One Haiku 5.5 call per page, page rendered to PNG, structured
   output (label, caption, columns, rows). Copy-exactly rules, checkmark ->
   ``yes``, blank -> ``""``.
3. **Agree.** Every numeral in a transcribed cell must occur among the numerals
   of that page's PDF *text layer*. The text layer is the second, independent
   path: it is not the extractor's markdown. Checking against the markdown, as
   the report first sketched, would flag exactly the papers that need repair
   (AML scores 64% that way) and block the fix. ``missing_before`` still records
   how many cells the markdown lacked, which is what the splice recovers.
4. **Splice.** Only when the paper-level agreement is at least
   ``AGREEMENT_THRESHOLD`` and the table's own agreement is too. The pipe table
   replaces the garbled rows next to the caption (above it or below it) or, when
   the extractor kept nothing, is inserted after the caption. ``|Col1|`` junk
   tables are stripped and the text never grows past ``MAX_FULL_TEXT_CHARS``.
   A paper below the threshold is *flagged*, not spliced: its tables are stored
   with their agreement so a person can look.

Everything is non-fatal by construction: ``stage_outcome`` never raises.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Sequence

from pydantic import BaseModel, Field

from .config import get_settings
from .llm_client import LLMCallError, structured
from .llm_routes import TABLE_REPAIR_ROUTE

logger = logging.getLogger(__name__)

MIGRATION = "docs/migrations/031_paper_tables.sql"

#: Paper-level and per-table agreement needed before anything is spliced.
AGREEMENT_THRESHOLD = 0.95
#: A table with fewer numerals than this cannot be checked, so it is not spliced.
MIN_VERIFIABLE_NUMERALS = 3
#: Page calls per paper. The largest library paper measured had 11 table pages.
MAX_TABLE_PAGES = 20
RENDER_DPI = 130
#: ``podcast_agent.MAX_FULL_TEXT_CHARS``; read lazily so this module does not
#: import the podcast agent. The literal is only the fallback.
_FALLBACK_TEXT_CAP = 100_000
EXTRACTOR_SUFFIX = "+tables"

STATUS_SPLICED = "spliced"
STATUS_FLAGGED = "flagged"
STATUS_UNVERIFIED = "unverified"
STATUS_UNPLACED = "unplaced"
STATUS_TOO_LARGE = "too_large"

SYSTEM_PROMPT = (
    "This is one page of a research paper, rendered as an image. Transcribe EVERY "
    "table on the page (not figures or plots).\n"
    "Rules: copy every number exactly as printed; write a checkmark as \"yes\" and a "
    "visibly empty cell as \"\"; keep merged header cells as printed; do not infer, "
    "round or correct anything. `label` is the table's number or letter as printed in "
    "its caption (\"9\", \"A1\"); `caption` is the caption text. If the page has no "
    "table, return an empty list."
)

_CAPTION = r"^(?:\W|<u>|<b>|<i>|_){0,12}(?:Table|TABLE)\s+(?P<label>[A-Z]?\d+)\s*[:.|]"
CAPTION_LINE = re.compile(_CAPTION)
NUMERAL = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w])")
TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+")
MARKUP = re.compile(r"<[^>]+>|[*_`|]")


# -- model-facing schema -----------------------------------------------------


class TableDraft(BaseModel):
    label: str = Field(description="Table number as printed in the caption")
    caption: str
    columns: List[str]
    rows: List[List[str]]


class PageTablesDraft(BaseModel):
    tables: List[TableDraft]


# -- numerals ----------------------------------------------------------------


def normalise_numeral(raw: str) -> str:
    return raw.replace(",", "")


def numerals(text: str) -> List[str]:
    return [normalise_numeral(m.group(0)) for m in NUMERAL.finditer(text or "")]


def cell_numerals(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """Numerals in the table body. Header cells are labels, not data."""
    found: List[str] = []
    for row in rows:
        for cell in row:
            found.extend(numerals(str(cell)))
    return found


def coverage(text: str, found: Sequence[str]) -> Optional[float]:
    """Fraction of ``found`` numerals that occur somewhere in ``text``."""
    if not found:
        return None
    have = set(numerals(text))
    return sum(1 for n in found if n in have) / len(found)


# -- PDF access --------------------------------------------------------------


class PageSource(Protocol):
    """The three things the repair needs from a PDF. Tests supply a fake."""

    def captions(self) -> dict[int, List[str]]:
        """Page index -> caption lines found in that page's text layer."""

    def render(self, page: int) -> bytes:
        """PNG bytes of the page."""

    def page_numerals(self, page: int) -> set[str]:
        """Numerals in the page's text layer."""


class PymupdfPages:
    def __init__(self, pdf_path: str):
        import pymupdf

        self._doc = pymupdf.open(pdf_path)

    def captions(self) -> dict[int, List[str]]:
        found: dict[int, List[str]] = {}
        for index, page in enumerate(self._doc):
            lines = [
                line.strip()
                for line in page.get_text().splitlines()
                if CAPTION_LINE.match(line)
            ]
            if lines:
                found[index] = lines
        return found

    def render(self, page: int) -> bytes:
        return self._doc[page].get_pixmap(dpi=RENDER_DPI).tobytes("png")

    def page_numerals(self, page: int) -> set[str]:
        return set(numerals(self._doc[page].get_text()))

    def close(self) -> None:
        self._doc.close()


# -- result types ------------------------------------------------------------


@dataclass
class TableRecord:
    label: str
    caption: str
    page: int  # 1-based
    columns: List[str]
    rows: List[List[str]]
    model: str
    numerals: int = 0
    matched: int = 0
    missing_before: int = 0
    status: str = STATUS_UNVERIFIED
    reason: str = ""

    @property
    def agreement(self) -> Optional[float]:
        return self.matched / self.numerals if self.numerals else None

    def as_row(self, index: int, cost_usd: float = 0.0) -> dict[str, Any]:
        agreement = self.agreement
        return {
            "table_index": index,
            "label": self.label,
            "page": self.page,
            "caption": self.caption,
            "columns": self.columns,
            "rows": self.rows,
            "model": self.model,
            "agreement": None if agreement is None else round(agreement, 4),
            "numerals": self.numerals,
            "matched": self.matched,
            "missing_before": self.missing_before,
            "status": self.status,
            "reason": self.reason,
            "cost_usd": round(cost_usd, 6),
        }


@dataclass
class TableRepair:
    text: str
    tables: List[TableRecord] = field(default_factory=list)
    cost_usd: float = 0.0
    calls: int = 0
    pages: int = 0
    failed_pages: int = 0
    notes: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    junk_removed: int = 0

    @property
    def spliced(self) -> int:
        return sum(1 for t in self.tables if t.status == STATUS_SPLICED)

    @property
    def agreement(self) -> Optional[float]:
        verifiable = [t for t in self.tables if t.numerals >= MIN_VERIFIABLE_NUMERALS]
        total = sum(t.numerals for t in verifiable)
        return sum(t.matched for t in verifiable) / total if total else None

    def rows_for_storage(self) -> List[dict[str, Any]]:
        share = self.cost_usd / len(self.tables) if self.tables else 0.0
        return [t.as_row(i, share) for i, t in enumerate(self.tables)]

    def detail(self) -> str:
        agreement = self.agreement
        bits = [f"{len(self.tables)} table(s) read on {self.pages} page(s)"]
        bits.append(f"{self.spliced} spliced")
        flagged = sum(1 for t in self.tables if t.status == STATUS_FLAGGED)
        if flagged:
            bits.append(f"{flagged} flagged")
        if agreement is not None:
            bits.append(f"{agreement:.1%} agreement")
        if self.failed_pages:
            bits.append(f"{self.failed_pages} page call(s) failed")
        bits.extend(self.notes)
        if self.models:
            bits.append(", ".join(sorted(set(self.models))))
        bits.append(f"${self.cost_usd:.6f}")
        return " · ".join(bits)


# -- text surgery ------------------------------------------------------------


def text_cap() -> int:
    try:
        from .agents.podcast_agent import MAX_FULL_TEXT_CHARS

        return int(MAX_FULL_TEXT_CHARS)
    except Exception:
        return _FALLBACK_TEXT_CAP


def strip_junk_tables(text: str) -> tuple[str, int]:
    """Drop the mostly-empty ``|Col1|...`` tables the extractor emits for plots.

    Returns the text and how many blocks were removed. A block is the header
    row (it contains ``Col1``) and the pipe rows that follow it. A block whose
    cells are mostly filled is a real table with a poor header (the AML
    comparison table is one) and is kept; the splice step replaces it.
    """
    lines = text.split("\n")
    out: List[str] = []
    removed = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|") and re.search(r"\|Col1\|", line):
            j = i + 1
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            cells = [
                c.strip()
                for row in lines[i + 2 : j]
                for c in row.strip("|").split("|")
            ]
            empty = sum(1 for c in cells if not c)
            if cells and empty / len(cells) >= 0.5:
                removed += 1
                i = j
                continue
        out.append(line)
        i += 1
    return "\n".join(out), removed


def render_markdown(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """A pipe table. Rows are padded to the widest row; ``|`` in a cell is escaped."""
    grid = [list(map(str, r)) for r in rows]
    header = list(map(str, columns))
    if not header and grid:
        header, grid = grid[0], grid[1:]
    width = max([len(header)] + [len(r) for r in grid]) if (header or grid) else 0
    if width == 0:
        return []

    def fmt(cells: List[str]) -> str:
        cells = cells + [""] * (width - len(cells))
        return "| " + " | ".join(c.replace("|", "\\|").strip() for c in cells) + " |"

    return [fmt(header), "|" + "---|" * width] + [fmt(r) for r in grid]


def _vocab(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> set[str]:
    vocab: set[str] = set()
    for cell in list(columns) + [c for r in rows for c in r]:
        vocab.update(t.lower() for t in TOKEN.findall(MARKUP.sub(" ", str(cell))))
    return vocab


def _line_state(line: str, vocab: set[str]) -> str:
    """``blank`` (nothing to judge), ``table`` (explained by the table) or ``other``."""
    tokens = TOKEN.findall(MARKUP.sub(" ", line))
    if not tokens:
        return "blank"
    hits = sum(1 for t in tokens if t.lower() in vocab)
    return "table" if hits / len(tokens) >= 0.85 else "other"


def _scan(lines: List[str], start: int, step: int, vocab: set[str]) -> tuple[int, int, int]:
    """Walk away from the caption. Returns (first, last, explained) line indexes."""
    first = last = -1
    explained = 0
    i = start
    while 0 <= i < len(lines) and abs(i - start) <= 80:
        state = _line_state(lines[i], vocab)
        if state == "other":
            break
        if state == "table":
            if first == -1:
                first = i
            last = i
            explained += 1
        i += step
    if first == -1:
        return -1, -1, 0
    return min(first, last), max(first, last), explained


def splice_table(text: str, record: TableRecord) -> Optional[str]:
    """Put ``record`` as a pipe table next to its caption. None when no caption."""
    caption_re = re.compile(
        _CAPTION.replace(r"[A-Z]?\d+", re.escape(record.label)), re.M
    )
    lines = text.split("\n")
    cap = next((i for i, ln in enumerate(lines) if caption_re.match(ln)), None)
    if cap is None:
        return None
    block = render_markdown(record.columns, record.rows)
    if not block:
        return None
    vocab = _vocab(record.columns, record.rows)
    up = _scan(lines, cap - 1, -1, vocab)
    down = _scan(lines, cap + 1, 1, vocab)
    if up[2] == 0 and down[2] == 0:
        lines[cap + 1 : cap + 1] = [""] + block
    elif up[2] > down[2]:
        lines[up[0] : up[1] + 1] = block
        # keep a blank line between the table and the caption that follows
        after = up[0] + len(block)
        if after < len(lines) and lines[after].strip():
            lines.insert(after, "")
    else:
        lines[down[0] : down[1] + 1] = block
    return "\n".join(lines)


# -- transcription -----------------------------------------------------------


def _user_blocks(png: bytes, captions: Sequence[str]) -> list[dict[str, Any]]:
    note = "Captions printed on this page: " + " | ".join(captions)
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(png).decode("ascii"),
            },
        },
        {"type": "text", "text": note},
    ]


def repair_tables(
    pages: PageSource,
    text: str,
    *,
    client: Any = None,
    model: Optional[str] = None,
    max_pages: int = MAX_TABLE_PAGES,
    refuse_over: Optional[int] = None,
) -> TableRepair:
    """Transcribe the paper's table pages and splice the ones that agree.

    ``text`` is the extracted markdown. The returned ``TableRepair.text`` is
    ``text`` with junk tables stripped and agreeing tables spliced in; it is
    identical to the input when nothing could be spliced.
    """
    result = TableRepair(text=text)
    caption_pages = pages.captions()
    if not caption_pages:
        result.notes.append("no table captions")
        return result
    selected = sorted(caption_pages)
    if len(selected) > max_pages:
        result.notes.append(f"{len(selected) - max_pages} page(s) over the {max_pages}-page cap")
        selected = selected[:max_pages]

    seen_labels: set[str] = set()
    in_text = set(numerals(text))
    for page in selected:
        result.pages += 1
        try:
            call = structured(
                TABLE_REPAIR_ROUTE,
                SYSTEM_PROMPT,
                _user_blocks(pages.render(page), caption_pages[page]),
                PageTablesDraft,
                client=client,
                model=model,
                max_tokens=8000,
            )
        except LLMCallError as exc:
            result.failed_pages += 1
            logger.warning("Table read failed on page %s: %s", page + 1, exc)
            continue
        result.calls += 1
        result.cost_usd += call.cost_usd
        result.models.append(call.model)
        layer = pages.page_numerals(page)
        for draft in call.parsed.tables:
            label = draft.label.strip()
            found = cell_numerals(draft.columns, draft.rows)
            record = TableRecord(
                label=label,
                caption=draft.caption.strip(),
                page=page + 1,
                columns=list(draft.columns),
                rows=[list(r) for r in draft.rows],
                model=call.model,
                numerals=len(found),
                matched=sum(1 for n in found if n in layer),
                missing_before=sum(1 for n in found if n not in in_text),
            )
            if label in seen_labels:
                record.status, record.reason = STATUS_UNPLACED, "label already read on an earlier page"
            seen_labels.add(label)
            result.tables.append(record)

    work, result.junk_removed = strip_junk_tables(text)
    paper_agreement = result.agreement
    budget = max(refuse_over if refuse_over is not None else text_cap(), len(work))
    for record in result.tables:
        if record.reason:
            continue
        agreement = record.agreement
        if record.numerals < MIN_VERIFIABLE_NUMERALS:
            record.status, record.reason = STATUS_UNVERIFIED, "too few numbers to check"
        elif paper_agreement is None or paper_agreement < AGREEMENT_THRESHOLD:
            record.status = STATUS_FLAGGED
            record.reason = f"paper agreement {paper_agreement:.1%} is under {AGREEMENT_THRESHOLD:.0%}"
        elif agreement < AGREEMENT_THRESHOLD:
            record.status = STATUS_FLAGGED
            record.reason = f"table agreement {agreement:.1%} is under {AGREEMENT_THRESHOLD:.0%}"
        else:
            spliced = splice_table(work, record) if record.label else None
            if spliced is None:
                record.status, record.reason = STATUS_UNPLACED, "caption not found in the text"
            elif len(spliced) > budget:
                record.status, record.reason = STATUS_TOO_LARGE, f"would pass {budget:,} characters"
            else:
                work, record.status = spliced, STATUS_SPLICED
    result.text = work if result.spliced else text
    if result.spliced == 0 and result.junk_removed:
        # junk removal alone is not worth rewriting stored text
        result.text = text
    return result


# -- the ingestion stage -----------------------------------------------------


@dataclass
class StageOutcome:
    status: str  # completed | skipped | failed
    detail: str
    repair: Optional[TableRepair] = None

    @property
    def text_changed(self) -> bool:
        return self.repair is not None and self.repair.spliced > 0


def stage_outcome(pdf_path: Optional[str], text: str) -> StageOutcome:
    """Run the repair for the ``tables`` stage. Never raises."""
    settings = get_settings()
    if not getattr(settings, "table_repair_enabled", True):
        return StageOutcome("skipped", "disabled by TABLE_REPAIR_ENABLED=false")
    if not pdf_path:
        return StageOutcome("skipped", "no PDF file to read tables from")
    if not getattr(settings, "anthropic_api_key", None):
        return StageOutcome("skipped", "ANTHROPIC_API_KEY is not set")
    pages = None
    try:
        pages = PymupdfPages(pdf_path)
        repair = repair_tables(pages, text)
    except Exception as exc:
        logger.warning("Table repair failed for %s: %s", pdf_path, exc, exc_info=True)
        return StageOutcome("failed", str(exc).splitlines()[0][:300] if str(exc) else type(exc).__name__)
    finally:
        if pages is not None:
            try:
                pages.close()
            except Exception:
                pass
    if repair.pages == 0:
        return StageOutcome("skipped", "no table captions in the PDF", repair)
    if repair.calls == 0:
        return StageOutcome("failed", f"all {repair.failed_pages} page call(s) failed", repair)
    return StageOutcome("completed", repair.detail(), repair)


def store_records(paper_id: str, repair: Optional[TableRepair]) -> Optional[str]:
    """Store the provenance rows. Non-fatal: returns the problem, or None.

    The repaired text is already stored by the time this runs; a database that
    lacks migration 031 loses the page/caption/agreement record, not the paper.
    """
    if repair is None or not repair.tables:
        return None
    from . import db

    try:
        db.replace_paper_tables(paper_id, repair.rows_for_storage())
    except Exception as exc:
        logger.warning("Table provenance for %s was not stored: %s", paper_id, exc)
        return str(exc)
    return None


def tagged_extractor(extractor: str, spliced: bool) -> str:
    if spliced and not extractor.endswith(EXTRACTOR_SUFFIX):
        return extractor + EXTRACTOR_SUFFIX
    return extractor


# -- stored papers: `text repair-tables` -------------------------------------


@dataclass
class PaperReport:
    """What the command did (or, with ``dry_run``, would do) for one paper."""

    paper_id: str
    status: str  # repaired | flagged | no_tables | skipped | failed
    repair: Optional[TableRepair] = None
    chars_before: int = 0
    chars_after: int = 0
    chunks: int = 0
    embeddings: int = 0
    notes: List[str] = field(default_factory=list)
    error: str = ""

    def render(self, dry_run: bool) -> str:
        lines = [f"paper: {self.paper_id}", f"status: {self.status}"]
        if self.error:
            lines.append(f"error: {self.error}")
        if self.repair is not None:
            lines.append(f"summary: {self.repair.detail()}")
            for t in self.repair.tables:
                agreement = "n/a" if t.agreement is None else f"{t.agreement:.1%}"
                lines.append(
                    f"  table {t.label or '?'} (page {t.page}): {t.status} · "
                    f"{t.matched}/{t.numerals} numbers on the page · "
                    f"{t.missing_before} missing from the text · agreement {agreement}"
                    + (f" · {t.reason}" if t.reason else "")
                )
            lines.append(f"characters: {self.chars_before:,} -> {self.chars_after:,}")
        if dry_run:
            lines.append("dry_run: true (nothing written)")
        else:
            lines.append(f"chunks: {self.chunks}")
            lines.append(f"embeddings: {self.embeddings}")
        lines.extend(f"note: {n}" for n in self.notes)
        return "\n".join(lines)


def repair_stored_paper(
    paper: Any,
    *,
    dry_run: bool,
    model: Optional[str] = None,
    client: Any = None,
    open_pages: Any = None,
) -> PaperReport:
    """Repair one library paper's stored text from its PDF.

    Reads the stored ``paper_texts`` row (so nothing else about the text
    changes), transcribes the table pages, and, unless ``dry_run``, stores the
    spliced text, re-chunks it, re-embeds the new chunks and re-projects them.
    A dry run still makes the (cheap) page calls: agreement is the thing to
    look at, and it cannot be known without them. It writes nothing.
    """
    from . import db, vectors
    from .paper_text import classify_pdf_url
    from .tools.pdf_loader import download_pdf, extract_document

    report = PaperReport(paper_id=paper.id, status="failed")
    url = getattr(paper, "url_pdf", None)
    if classify_pdf_url(url) == "skipped":
        report.status = "skipped"
        report.notes.append("no resolvable PDF URL")
        return report
    pages = None
    try:
        path = download_pdf(url)
        text = db.get_stored_paper_text(paper.id)
        extractor = db.get_stored_paper_extractor(paper.id) if text else None
        if text is None:
            extracted = extract_document(path)
            text, extractor = extracted.text, extracted.extractor
            report.notes.append("no stored text; read from the PDF")
        pages = (open_pages or PymupdfPages)(path)
        repair = repair_tables(pages, text, client=client, model=model)
        report.repair = repair
        report.chars_before, report.chars_after = len(text), len(repair.text)
        if repair.pages == 0:
            report.status = "no_tables"
            return report
        if repair.calls == 0:
            report.error = f"all {repair.failed_pages} page call(s) failed"
            return report
        report.status = "repaired" if repair.spliced else "flagged"
        if dry_run:
            return report
        if repair.spliced:
            report.chunks = db.replace_paper_text(
                paper.id, repair.text, tagged_extractor(extractor or "pdf_loader", True)
            )
            report.embeddings = db.embed_paper_chunks(paper.id)
            try:
                vectors.ensure_collections()
                vectors.upsert_text(paper.pillar_id, paper.id, repair.text)
            except Exception as exc:
                report.notes.append(
                    f"vector index not updated ({exc}); run `vectors sync`"
                )
        db.replace_paper_tables(paper.id, repair.rows_for_storage())
    except Exception as exc:
        report.status, report.error = "failed", str(exc).splitlines()[0][:300] if str(exc) else type(exc).__name__
        logger.warning("repair-tables failed for %s: %s", paper.id, exc, exc_info=True)
    finally:
        if pages is not None and hasattr(pages, "close"):
            pages.close()
    return report
