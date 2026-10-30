"""Works sync: give every library paper one work, then fetch its typed citations.

``plan``-style code with the I/O injected, like ``graph.py``: the Semantic
Scholar tool and the store are passed in, so a test runs the whole sync
against a recorded fixture and an in-memory store, and ``--dry-run`` touches
neither.

Identity is resolved here, in Python, from the ``work_ids`` rows loaded into a
``WorkIndex``. The database function ``apply_work_batch`` only writes what this
module hands it. A write that fails leaves the index untouched (``Overlay``),
so the next paper is resolved against what is really stored.

Identity rules (the cases a real library hits):

* A record resolves by any of its keys. The first work found wins, preferring
  the one that already holds the record's own S2 id.
* A key already owned by another work is never moved: it is counted as an
  alias conflict and left. Two stubs that turn out to be one work stay two
  rows and are counted as ``ambiguous``; merging them is a separate decision.
* A work keeps the S2 id it has. S2 re-issuing a paper's id is reported, not
  followed.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

from .tools.semantic_scholar_tool import (
    BATCH_LIMIT,
    MAX_CITERS,
    CitationEdge,
    EdgePage,
    SemanticScholarError,
    SemanticScholarUnavailableError,
    WorkRecord,
)
from .work_ids import (
    KIND_S2,
    WorkKey,
    best_lookup_key,
    keys_from_s2,
    parse_paper_id,
    s2_lookup_id,
)

#: ``work_citations.source`` of an edge S2 described with intents and sentences.
SOURCE = "semantic_scholar"
#: An edge known only as two ids (the ingest hooks and ``graph build`` store
#: those). The citer walk does not treat it as stored, so a first sync still
#: replaces it with the typed edge.
SOURCE_IDS_ONLY = "semantic_scholar:ids"
#: Rows per ``apply_work_batch`` call. One citer page is 100 works and 100 edges;
#: this keeps a 1,000-citer walk to a few calls of a few hundred KB each.
WRITE_CHUNK = 400


@dataclass
class LibraryPaper:
    """A ``papers`` row as the sync reads it."""

    id: str
    work_id: Optional[str] = None
    title: str = ""
    citation_count: Optional[int] = None

    def keys(self) -> List[WorkKey]:
        """The key the paper's own id carries. Empty for ``file:`` uploads."""
        key = parse_paper_id(self.id)
        return [key] if key else []


class WorkIndex:
    """``work_ids`` in memory: key -> work uuid, and each work's S2 id.

    A partial index (only the keys involved) resolves exactly like a full one,
    which is how ``store_citations`` avoids loading the whole table per paper.
    """

    def __init__(
        self,
        by_key: Optional[Dict[WorkKey, str]] = None,
        s2_by_work: Optional[Dict[str, str]] = None,
    ) -> None:
        self.by_key: Dict[WorkKey, str] = dict(by_key or {})
        self.s2_by_work: Dict[str, str] = dict(s2_by_work or {})
        if not self.s2_by_work:
            for (kind, value), work in self.by_key.items():
                if kind == KIND_S2:
                    self.s2_by_work[work] = value

    def work_for(self, key: WorkKey) -> Optional[str]:
        return self.by_key.get(key)

    def keys_of(self, work_id: str) -> List[WorkKey]:
        return [key for key, work in self.by_key.items() if work == work_id]

    def s2_id(self, work_id: str) -> Optional[str]:
        return self.s2_by_work.get(work_id)


@dataclass
class Overlay:
    """Resolutions made since the last successful write. Commit or drop."""

    index: WorkIndex
    new_ids: Callable[[], str]
    by_key: Dict[WorkKey, str] = field(default_factory=dict)
    s2_by_work: Dict[str, str] = field(default_factory=dict)
    created: Set[str] = field(default_factory=set)
    alias_conflicts: int = 0
    ambiguous: int = 0

    def work_for(self, key: WorkKey) -> Optional[str]:
        return self.by_key.get(key) or self.index.work_for(key)

    def s2_id(self, work_id: str) -> Optional[str]:
        return self.s2_by_work.get(work_id) or self.index.s2_id(work_id)

    def resolve(self, keys: Sequence[WorkKey], prefer: Optional[str] = None) -> Tuple[str, List[WorkKey]]:
        """Work for these keys (created if none holds them) and the keys to add."""
        matched: List[str] = []
        for key in keys:
            work = self.work_for(key)
            if work and work not in matched:
                matched.append(work)
        s2_key = next((key for key in keys if key[0] == KIND_S2), None)
        if prefer and prefer in matched:
            chosen = prefer
        elif s2_key and self.work_for(s2_key) in matched:
            chosen = self.work_for(s2_key)
        elif matched:
            chosen = sorted(matched)[0]
        else:
            chosen = self.new_ids()
            self.created.add(chosen)
        if len(matched) > 1:
            self.ambiguous += 1
        add: List[WorkKey] = []
        for key in keys:
            owner = self.work_for(key)
            if owner is None:
                if key[0] == KIND_S2 and self.s2_id(chosen):
                    # chosen already has an S2 id; a second one would break
                    # the unique index on works.s2_paper_id.
                    self.alias_conflicts += 1
                    continue
                self.by_key[key] = chosen
                if key[0] == KIND_S2:
                    self.s2_by_work[chosen] = key[1]
                add.append(key)
            elif owner != chosen:
                self.alias_conflicts += 1
        return chosen, add

    def commit(self) -> None:
        self.index.by_key.update(self.by_key)
        self.index.s2_by_work.update(self.s2_by_work)


@dataclass
class Batch:
    """Rows for ``apply_work_batch``, one row per work and one per edge.

    Postgres refuses an ``ON CONFLICT DO UPDATE`` that meets the same key twice
    in one statement, and S2 does repeat a work (two references to one paper,
    a work citing itself), so rows are merged here rather than left to fail
    there.
    """

    works: List[dict] = field(default_factory=list)
    ids: List[dict] = field(default_factory=list)
    edges: List[dict] = field(default_factory=list)
    papers: List[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.works or self.ids or self.edges or self.papers)

    def put_work(self, row: dict) -> None:
        """Add a work row, or merge into the one already staged for that id."""
        for held in self.works:
            if held["id"] != row["id"]:
                continue
            for flag in ("described", "references_synced", "citers_synced"):
                held[flag] = bool(held.get(flag)) or bool(row.get(flag))
            for name, value in row.items():
                if held.get(name) is None and value is not None:
                    held[name] = value
            return
        self.works.append(row)

    def put_edge(self, row: dict) -> None:
        """Add an edge row; a repeat of (citing, cited) replaces the earlier one."""
        for position, held in enumerate(self.edges):
            if held["citing"] == row["citing"] and held["cited"] == row["cited"]:
                self.edges[position] = row
                return
        self.edges.append(row)


class WorksStore(Protocol):
    """What the sync needs from storage. ``works_store.PostgrestWorks`` is the real one."""

    def apply(self, batch: Batch) -> dict: ...

    def citer_work_ids(self, work_id: str) -> Set[str]:
        """Works with a typed (not ids-only) edge to ``work_id``."""
        ...


class CitationSource(Protocol):
    """The slice of ``SemanticScholarTool`` the sync uses."""

    def get_work_records(self, lookup_ids: Sequence[str]) -> List[Optional[WorkRecord]]: ...

    def get_reference_edges(self, lookup_id: str) -> EdgePage: ...

    def get_citer_edges(self, lookup_id: str, *, known_ids=(), max_items: int = MAX_CITERS): ...


def work_row(
    work_id: str,
    record: Optional[WorkRecord] = None,
    *,
    s2_paper_id: Optional[str] = None,
    references_synced: bool = False,
    citers_synced: bool = False,
) -> dict:
    """The ``p_works`` row for one work. Without a record it is a stub."""
    row: dict = {
        "id": work_id,
        "s2_paper_id": s2_paper_id,
        "described": record is not None,
        "references_synced": references_synced,
        "citers_synced": citers_synced,
    }
    if record is not None:
        row.update(
            corpus_id=record.corpus_id,
            title=record.title,
            year=record.year,
            publication_date=record.publication_date,
            venue=record.venue,
            authors=record.authors,
            author_count=record.author_count,
            abstract=record.abstract,
            tldr=record.tldr,
            citation_count=record.citation_count,
            influential_citation_count=record.influential_citation_count,
            reference_count=record.reference_count,
            open_pdf_url=record.open_pdf_url,
        )
    return row


def record_keys(record: WorkRecord) -> List[WorkKey]:
    return keys_from_s2(record.s2_paper_id, record.external_ids, record.corpus_id)


def _chunks(items: List, size: int) -> Iterable[List]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def resolve_record(overlay: Overlay, record: WorkRecord, batch: Batch, *, prefer: Optional[str] = None) -> str:
    """Resolve a described record to its work and stage its rows into ``batch``."""
    keys = record_keys(record)
    work_id, add = overlay.resolve(keys, prefer=prefer)
    s2_id = overlay.s2_id(work_id)
    batch.put_work(
        work_row(
            work_id,
            record,
            s2_paper_id=s2_id if s2_id == record.s2_paper_id else None,
        )
    )
    batch.ids.extend({"kind": kind, "value": value, "work_id": work_id} for kind, value in add)
    return work_id


def edge_row(citing: str, cited: str, edge: CitationEdge) -> dict:
    return {
        "citing": citing,
        "cited": cited,
        "is_influential": edge.is_influential,
        "intents": edge.intents,
        "contexts": edge.contexts,
        "source": SOURCE,
        "detail": True,
    }


@dataclass
class WorksSyncReport:
    """Counts printed by ``works sync``. ``render`` is the CLI output."""

    dry_run: bool
    library_papers: int = 0
    linked: int = 0
    to_link: int = 0
    unresolvable: List[str] = field(default_factory=list)
    works_known: int = 0
    resolved: int = 0
    not_in_s2: List[str] = field(default_factory=list)
    counts_written: int = 0
    works_created: int = 0
    aliases_added: int = 0
    reference_edges: int = 0
    citer_edges: int = 0
    library_edges: int = 0
    skipped_edges: int = 0
    alias_conflicts: int = 0
    ambiguous: int = 0
    requests_at_least: int = 0
    stopped_early: str = ""
    failures: List[str] = field(default_factory=list)

    def render(self) -> str:
        mode = "dry-run" if self.dry_run else "write"
        lines = [
            f"works sync ({mode})",
            f"library papers: {self.library_papers}",
            f"already linked to a work: {self.linked}",
            f"to link: {self.to_link}",
            f"no resolvable id (file: uploads and the like): {len(self.unresolvable)}",
            f"works known: {self.works_known}",
        ]
        if self.dry_run:
            lines.append(
                f"Semantic Scholar requests: at least {self.requests_at_least} "
                "(one batch, one reference page and one citer page per paper)"
            )
        else:
            lines.extend(
                [
                    f"papers resolved by Semantic Scholar: {self.resolved}",
                    f"papers Semantic Scholar does not know: {len(self.not_in_s2)}",
                    f"papers.citation_count written: {self.counts_written}",
                    f"works created: {self.works_created}",
                    f"ids added: {self.aliases_added}",
                    f"reference edges written: {self.reference_edges}",
                    f"citer edges written: {self.citer_edges}",
                    f"library -> library citations: {self.library_edges}",
                    f"entries Semantic Scholar sent without a paperId: {self.skipped_edges}",
                    f"id conflicts left alone: {self.alias_conflicts}",
                    f"works that matched two rows: {self.ambiguous}",
                ]
            )
        if self.stopped_early:
            lines.append(f"stopped early: {self.stopped_early}")
        lines.extend(f"failed: {reason}" for reason in self.failures)
        return "\n".join(lines)


def _library_targets(papers: Sequence[LibraryPaper]):
    """Split the library into papers S2 can be asked about and the rest."""
    asked: List[Tuple[LibraryPaper, str]] = []
    unresolvable: List[str] = []
    for paper in papers:
        key = best_lookup_key(paper.keys())
        lookup = s2_lookup_id(key) if key else None
        if lookup:
            asked.append((paper, lookup))
        else:
            unresolvable.append(paper.id)
    return asked, unresolvable


def run_works_sync(
    papers: Sequence[LibraryPaper],
    index: WorkIndex,
    tool: Optional[CitationSource],
    store: Optional[WorksStore],
    *,
    dry_run: bool,
    max_citers: int = MAX_CITERS,
    new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
) -> WorksSyncReport:
    """Link library papers to works and store their full typed citations.

    Idempotent: a second run resolves every paper to the work it already has,
    writes the same edges and adds no ids. Dry-run reads ``papers`` and
    ``index`` only; it calls neither the tool nor the store.
    """
    asked, unresolvable = _library_targets(papers)
    report = WorksSyncReport(
        dry_run=dry_run,
        library_papers=len(papers),
        linked=sum(1 for paper in papers if paper.work_id),
        unresolvable=unresolvable,
        works_known=len(set(index.by_key.values())),
    )
    report.to_link = sum(1 for paper, _ in asked if not paper.work_id)
    if dry_run:
        report.requests_at_least = math.ceil(len(asked) / BATCH_LIMIT) + 2 * len(asked)
        return report
    if tool is None or store is None:
        raise RuntimeError("works sync needs a Semantic Scholar tool and a store")

    library_works: Dict[str, str] = {}
    described: List[Tuple[LibraryPaper, WorkRecord, str]] = []

    try:
        records = tool.get_work_records([lookup for _, lookup in asked])
    except SemanticScholarError as exc:
        report.stopped_early = str(exc)
        return report

    overlay = Overlay(index, new_id)
    batch = Batch()
    for (paper, _), record in zip(asked, records, strict=True):
        if record is None:
            report.not_in_s2.append(paper.id)
            continue
        work_id = resolve_record(overlay, record, batch, prefer=paper.work_id)
        # the paper's own key points at this work even when S2 names it differently
        own, add = overlay.resolve([*paper.keys()], prefer=work_id)
        batch.ids.extend({"kind": k, "value": v, "work_id": own} for k, v in add)
        batch.papers.append(
            {"id": paper.id, "work_id": work_id, "citation_count": record.citation_count}
        )
        library_works[paper.id] = work_id
        described.append((paper, record, work_id))
    if not _write(store, batch, overlay, report):
        return report
    report.resolved = len(described)
    report.counts_written = sum(1 for p in batch.papers if p["citation_count"] is not None)

    by_work = {work: paper_id for paper_id, work in library_works.items()}
    cited_by_library: Set[Tuple[str, str]] = set()
    for paper, record, work_id in described:
        lookup = best_lookup_key(record_keys(record))
        lookup_id = s2_lookup_id(lookup) if lookup else None
        if not lookup_id:
            continue
        try:
            _sync_references(
                tool, store, index, new_id, report, work_id, lookup_id, by_work, cited_by_library
            )
            if max_citers > 0:
                _sync_citers(
                    tool, store, index, new_id, report, work_id, lookup_id, max_citers
                )
        except SemanticScholarUnavailableError as exc:
            report.failures.append(f"{paper.id}: {exc}")
            report.stopped_early = str(exc)
            break
        except SemanticScholarError as exc:
            report.failures.append(f"{paper.id}: {exc}")
        if report.stopped_early:
            break
    report.library_edges = len(cited_by_library)
    return report


def _write(store: WorksStore, batch: Batch, overlay: Overlay, report: WorksSyncReport) -> bool:
    """Write ``batch`` in chunks. The index learns the overlay only on success."""
    if not batch:
        overlay.commit()
        return True
    try:
        for chunk in _split(batch):
            store.apply(chunk)
    except Exception as exc:  # the store raises its own typed errors
        report.failures.append(str(exc))
        report.stopped_early = str(exc)
        return False
    overlay.commit()
    report.works_created += len(overlay.created)
    report.aliases_added += len(batch.ids)
    report.alias_conflicts += overlay.alias_conflicts
    report.ambiguous += overlay.ambiguous
    return True


def _split(batch: Batch) -> Iterable[Batch]:
    """Calls in foreign-key order: works, then their ids, then edges, then papers."""
    for chunk in _chunks(batch.works, WRITE_CHUNK):
        yield Batch(works=chunk)
    for chunk in _chunks(batch.ids, WRITE_CHUNK * 3):
        yield Batch(ids=chunk)
    for chunk in _chunks(batch.edges, WRITE_CHUNK):
        yield Batch(edges=chunk)
    if batch.papers:
        yield Batch(papers=batch.papers)


def _sync_references(
    tool, store, index, new_id, report, work_id, lookup_id, library_by_work, library_pairs
) -> None:
    overlay = Overlay(index, new_id)
    batch = Batch()
    result = tool.get_reference_edges(lookup_id)
    report.skipped_edges += result.skipped
    for edge in result.edges:
        other = resolve_record(overlay, edge.other, batch)
        if other == work_id:
            continue
        batch.put_edge(edge_row(work_id, other, edge))
        if other in library_by_work:
            library_pairs.add((work_id, other))
    batch.put_work(work_row(work_id, references_synced=True))
    if _write(store, batch, overlay, report):
        report.reference_edges += len(batch.edges)


def _sync_citers(tool, store, index, new_id, report, work_id, lookup_id, max_citers) -> None:
    known = {
        s2_id
        for citer in store.citer_work_ids(work_id)
        if (s2_id := index.s2_id(citer))
    }
    result = tool.get_citer_edges(lookup_id, known_ids=known, max_items=max_citers)
    report.skipped_edges += result.skipped
    overlay = Overlay(index, new_id)
    batch = Batch()
    for edge in result.edges:
        other = resolve_record(overlay, edge.other, batch)
        if other != work_id:
            batch.put_edge(edge_row(other, work_id, edge))
    batch.put_work(work_row(work_id, citers_synced=True))
    if _write(store, batch, overlay, report):
        report.citer_edges += len(batch.edges)
