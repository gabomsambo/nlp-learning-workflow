"""Read and write the works tables through PostgREST.

``works.py`` plans; this module loads the rows it plans from and hands finished
rows to ``apply_work_batch`` (docs/migrations/031_works_identity.sql), which
writes them in one transaction.

``store_citations`` is the bridge for callers that still hold bare ids: the
ingest hooks and ``graph build`` fetch a paper's references and citers as
``PaperCitation(paper_id, cited_paper_id, direction)``. It resolves both ends
to works, creating an id-only stub where Semantic Scholar has not described a
work yet, and stores thin edges that never overwrite typed ones.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import db
from .schemas import PaperCitation
from .work_ids import WorkKey, parse_paper_id
from .tools.semantic_scholar_tool import MAX_CITERS
from .works import (
    SOURCE,
    SOURCE_IDS_ONLY,
    Batch,
    LibraryPaper,
    Overlay,
    WorkIndex,
    WorksSyncReport,
    run_works_sync,
    work_row,
)

logger = logging.getLogger(__name__)

MIGRATION = "docs/migrations/031_works_identity.sql"
_PAGE = 1000
_IN_CHUNK = 100


class WorksStoreError(Exception):
    """The works tables could not be read or written."""


class WorksUnavailable(WorksStoreError):
    """Migration 031 is not applied. Callers with an older path fall back."""


def _store_error(error: Any) -> WorksStoreError:
    text = str(error).lower()
    missing = any(
        token in text
        for token in (
            "apply_work_batch",
            "work_ids",
            "work_citations",
            "papers.work_id",
            "pgrst202",
            "pgrst205",
        )
    ) or ("works" in text and ("schema cache" in text or "does not exist" in text))
    if missing:
        return WorksUnavailable(
            f"The works tables are not available ({error}). Apply {MIGRATION} by hand."
        )
    return WorksStoreError(f"Could not use the works tables: {error}")


def _quote(value: str) -> str:
    """A PostgREST ``in.(...)`` list item: DOIs hold commas and parentheses."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _read_all(
    table: str,
    columns: str,
    order: str,
    *,
    narrow: Optional[Callable[[Any], Any]] = None,
) -> List[dict]:
    client = db.get_client()
    rows: List[dict] = []
    offset = 0
    while True:
        query = client.table(table).select(columns).order(order).limit(_PAGE).offset(offset)
        if narrow:
            query = narrow(query)
        response = query.execute()
        if response["error"]:
            raise _store_error(response["error"])
        page = response["data"] or []
        rows.extend(page)
        if len(page) < _PAGE:
            return rows
        offset += len(page)


def _chunks(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


class PostgrestWorks:
    """The real ``works.WorksStore``."""

    def apply(self, batch: Batch) -> dict:
        response = db.get_client().rpc(
            "apply_work_batch",
            {
                "p_works": batch.works,
                "p_ids": batch.ids,
                "p_edges": batch.edges,
                "p_papers": batch.papers,
            },
        )
        if response["error"]:
            raise _store_error(response["error"])
        return response["data"] or {}

    def citer_work_ids(self, work_id: str) -> Set[str]:
        """Works already stored as citing ``work_id`` with a typed edge."""
        rows = _read_all(
            "work_citations",
            "citing",
            "citing",
            narrow=lambda query: query.eq("cited", work_id).eq("source", SOURCE),
        )
        return {str(row["citing"]) for row in rows}


def load_index() -> WorkIndex:
    """Every ``work_ids`` row. Fails with ``WorksUnavailable`` before migration 031."""
    rows = _read_all("work_ids", "kind,value,work_id", "kind,value")
    return WorkIndex({(row["kind"], row["value"]): str(row["work_id"]) for row in rows})


def load_library() -> List[LibraryPaper]:
    rows = _read_all("papers", "id,work_id,title,citation_count", "id")
    return [
        LibraryPaper(
            id=str(row["id"]),
            work_id=str(row["work_id"]) if row.get("work_id") else None,
            title=row.get("title") or "",
            citation_count=row.get("citation_count"),
        )
        for row in rows
    ]


def load_external_works(have: Set[str]) -> Tuple[List[dict], int]:
    """External works that still need a vector, and how many external works there are.

    Returns ``([{id, title, abstract}, ...], total)``. ``have`` is the work ids
    that already carry the vector, so a rerun reads title and abstract only for
    works that still need one. A work is external when no ``papers.work_id``
    points at it. Fails with ``WorksUnavailable`` before migration 031.
    """
    linked = {
        str(row["work_id"])
        for row in _read_all("papers", "id,work_id", "id")
        if row.get("work_id")
    }
    external = [
        str(row["id"])
        for row in _read_all("works", "id", "id")
        if str(row["id"]) not in linked
    ]
    wanted = sorted(work_id for work_id in external if work_id not in have)
    out: List[dict] = []
    for chunk in _chunks(wanted, _IN_CHUNK):
        rows = _read_all(
            "works",
            "id,title,abstract",
            "id",
            narrow=lambda query, chunk=chunk: query.in_("id", list(chunk)),
        )
        out.extend(
            {"id": str(row["id"]), "title": row.get("title"), "abstract": row.get("abstract")}
            for row in rows
        )
    return out, len(external)


def _lookup_keys(keys: Sequence[WorkKey]) -> WorkIndex:
    """A partial index holding only these keys, from one query per kind."""
    by_kind: Dict[str, List[str]] = {}
    for kind, value in keys:
        by_kind.setdefault(kind, []).append(value)
    found: Dict[WorkKey, str] = {}
    for kind, values in by_kind.items():
        for chunk in _chunks(sorted(set(values)), _IN_CHUNK):
            rows = _read_all(
                "work_ids",
                "kind,value,work_id",
                "value",
                narrow=lambda query, kind=kind, chunk=chunk: query.eq("kind", kind).in_(
                    "value", [_quote(value) for value in chunk]
                ),
            )
            found.update({(row["kind"], row["value"]): str(row["work_id"]) for row in rows})
    return WorkIndex(found)


def _paper_works(paper_ids: Sequence[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for chunk in _chunks(sorted(set(paper_ids)), _IN_CHUNK):
        rows = _read_all(
            "papers",
            "id,work_id",
            "id",
            narrow=lambda query, chunk=chunk: query.in_("id", [_quote(i) for i in chunk]),
        )
        out.update({str(row["id"]): row.get("work_id") for row in rows})
    return out


def store_citations(
    citations: Sequence[PaperCitation],
    *,
    store: Optional[PostgrestWorks] = None,
    new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
) -> int:
    """Store ``PaperCitation`` rows as thin ``work_citations``. Returns new edges.

    A row whose library paper or other end names no work (a ``file:`` upload,
    an id of no known shape) is skipped. Raises ``WorksUnavailable`` before
    migration 031, so the caller can use the old table instead.
    """
    rows: List[Tuple[PaperCitation, WorkKey, WorkKey]] = []
    for citation in citations:
        library_key = parse_paper_id(citation.paper_id)
        other_key = parse_paper_id(citation.cited_paper_id)
        if library_key and other_key and library_key != other_key:
            rows.append((citation, library_key, other_key))
    if not rows:
        return 0

    keys = [key for _, library, other in rows for key in (library, other)]
    index = _lookup_keys(keys)
    linked = _paper_works([citation.paper_id for citation, _, _ in rows])
    overlay = Overlay(index, new_id)
    batch = Batch()
    library_work: Dict[str, str] = {}

    def work_of(key: WorkKey) -> str:
        work_id, add = overlay.resolve([key])
        batch.ids.extend({"kind": k, "value": v, "work_id": work_id} for k, v in add)
        return work_id

    for citation, library_key, other_key in rows:
        paper_id = citation.paper_id
        if paper_id not in linked:
            continue  # not a library paper: its citations are nobody's yet
        if paper_id not in library_work:
            existing = linked[paper_id]
            library_work[paper_id] = str(existing) if existing else work_of(library_key)
            if not existing:
                batch.papers.append(
                    {"id": paper_id, "work_id": library_work[paper_id], "citation_count": None}
                )
        mine = library_work[paper_id]
        other = work_of(other_key)
        if other == mine:
            continue
        citing, cited = (mine, other) if citation.citation_direction == "outgoing" else (other, mine)
        edge = {
            "citing": citing,
            "cited": cited,
            "is_influential": bool(citation.is_influential),
            "source": SOURCE_IDS_ONLY,
            "detail": False,
        }
        batch.put_edge(edge)

    for work_id in overlay.created:
        batch.put_work(work_row(work_id, s2_paper_id=overlay.s2_by_work.get(work_id)))
    if not batch.edges and not batch.papers:
        return 0
    result = (store or PostgrestWorks()).apply(batch)
    overlay.commit()
    return int(result.get("edges") or 0)


def execute_works_sync(*, dry_run: bool, max_citers: int = MAX_CITERS) -> WorksSyncReport:
    """CLI entry. Dry-run reads the library and ``work_ids`` and sends no request.

    The tool is built only for a real run and queues behind interactive
    Semantic Scholar calls (``BULK``), through the shared limiter.
    """
    papers = load_library()
    index = load_index()
    tool = None
    if not dry_run:
        from .tools.semantic_scholar_tool import BULK, SemanticScholarTool

        tool = SemanticScholarTool(priority=BULK)
    return run_works_sync(
        papers,
        index,
        tool,
        None if dry_run else PostgrestWorks(),
        dry_run=dry_run,
        max_citers=max_citers,
    )
