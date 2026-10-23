"""Persist a paper's full text and chunks in Postgres.

Postgres is the source of truth. ``paper_chunks`` stores the same semchunk
250/25 cl100k cut ``vectors.upsert_text`` embeds. Qdrant is a rebuildable copy
of those rows; this module does not write vectors.

Re-ingest replaces every chunk for the paper (delete, then insert) in one
transaction. A shorter re-chunk must not leave a stale tail. The database
side of that transaction is ``replace_paper_text_and_chunks`` in
``docs/migrations/017_paper_text.sql``. ``apply_replace`` is the same sequence
on a DB-API connection so tests can run it against sqlite.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional, Protocol, Sequence

from .tools.pdf_loader import (
    EMBEDDING_ENCODING,
    VECTOR_CHUNK_OVERLAP,
    VECTOR_CHUNK_SIZE,
    ExtractedText,
    _get_token_counter,
    chunk_text,
    download_pdf,
    extract_document,
)

logger = logging.getLogger(__name__)

MIGRATION = "docs/migrations/017_paper_text.sql"
CHUNKER_NAME = "semchunk"


class PaperTextWriteError(Exception):
    """A paper's text or chunks were not stored.

    Callers surface this on the run stage. Swallowing it is how a missing
    table used to look like a healthy persist.
    """


class PaperTextStore(Protocol):
    def execute(self, sql: str, params: Sequence = ()): ...

    def executemany(self, sql: str, seq_of_params): ...

    def commit(self): ...

    def rollback(self): ...


@dataclass
class BackfillCounts:
    """What a backfill pass saw. ``render`` is the CLI's whole output."""

    dry_run: bool
    papers: int = 0
    uploads: int = 0
    urls: int = 0
    skipped: int = 0
    already_stored: int = 0
    would_write: int = 0
    written: int = 0
    replaced: int = 0
    chunks: int = 0
    failed: int = 0
    failures: List[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"dry_run: {'true' if self.dry_run else 'false'}",
            f"papers: {self.papers}",
            f"uploads: {self.uploads}",
            f"urls: {self.urls}",
            f"skipped: {self.skipped}",
            f"already_stored: {self.already_stored}",
        ]
        if self.dry_run:
            lines.append(f"would_write: {self.would_write}")
        else:
            lines.extend([
                f"written: {self.written}",
                f"replaced: {self.replaced}",
                f"chunks: {self.chunks}",
                f"failed: {self.failed}",
            ])
        for failure in self.failures:
            lines.append(f"failure: {failure}")
        return "\n".join(lines)


def chunker_version() -> str:
    """Identity of the cut stored in ``paper_chunks.chunker_version``.

    The library version and the token budget are both in the string, because
    the unique key is ``(paper_id, chunk_index, chunker_version)`` and two
    configs of the same library are not the same cut.
    """
    try:
        from importlib.metadata import version

        lib = version("semchunk")
    except Exception:
        lib = "unknown"
    return (
        f"{CHUNKER_NAME}=={lib}|{VECTOR_CHUNK_SIZE}/{VECTOR_CHUNK_OVERLAP}"
        f"|{EMBEDDING_ENCODING}"
    )


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_chunk_rows(full_text: str) -> List[dict]:
    """Chunk ``full_text`` with the vector index's config.

    Raises:
        PaperTextWriteError: the text is empty, or the chunker returned nothing.
            An empty chunk list must not be written: the replace deletes the
            paper's existing chunks first.
    """
    if not full_text or not full_text.strip():
        raise PaperTextWriteError("refusing to store empty paper text")

    chunks = chunk_text(
        full_text,
        chunk_size=VECTOR_CHUNK_SIZE,
        chunk_overlap=VECTOR_CHUNK_OVERLAP,
    )
    if not chunks:
        raise PaperTextWriteError(
            "chunker produced no chunks for non-empty text"
        )

    counter = _get_token_counter()
    version = chunker_version()
    rows = []
    for index, text in enumerate(chunks):
        rows.append({
            "chunk_index": index,
            "text": text,
            "token_count": counter(text),
            "chunker": CHUNKER_NAME,
            "chunker_version": version,
            "chunk_size": VECTOR_CHUNK_SIZE,
            "chunk_overlap": VECTOR_CHUNK_OVERLAP,
            "encoding": EMBEDDING_ENCODING,
            "content_hash": content_hash(text),
        })
    return rows


def apply_replace(
    conn: PaperTextStore,
    paper_id: str,
    full_text: str,
    extractor: str,
    chunks: Sequence[dict],
) -> int:
    """Delete the paper's chunks and insert ``chunks``, in one transaction.

    The same sequence as ``replace_paper_text_and_chunks``. A failed insert
    rolls the delete back, so the previous chunks stay.

    Raises:
        PaperTextWriteError: empty text, empty chunks, or a missing identity.
    """
    if not paper_id or not str(paper_id).strip():
        raise PaperTextWriteError("paper text write missing paper_id")
    if not full_text or not str(full_text).strip():
        raise PaperTextWriteError(
            f"refusing to store empty paper text for {paper_id}"
        )
    if not extractor or not str(extractor).strip():
        raise PaperTextWriteError(
            f"paper text write for {paper_id} missing extractor"
        )
    if not chunks:
        raise PaperTextWriteError(
            f"refusing to replace chunks for {paper_id} with an empty set"
        )

    extracted_at = datetime.now(timezone.utc).isoformat()
    chunk_params = [
        (
            paper_id,
            row["chunk_index"],
            row["text"],
            row["token_count"],
            row["chunker"],
            row["chunker_version"],
            row["chunk_size"],
            row["chunk_overlap"],
            row["encoding"],
            row["content_hash"],
        )
        for row in chunks
    ]

    conn.execute("BEGIN")
    try:
        conn.execute(
            "INSERT INTO paper_texts (paper_id, full_text, extractor, extracted_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(paper_id) DO UPDATE SET "
            "full_text = excluded.full_text, "
            "extractor = excluded.extractor, "
            "extracted_at = excluded.extracted_at",
            (paper_id, full_text, extractor, extracted_at),
        )
        conn.execute(
            "DELETE FROM paper_chunks WHERE paper_id = ?",
            (paper_id,),
        )
        conn.executemany(
            "INSERT INTO paper_chunks ("
            "paper_id, chunk_index, text, token_count, chunker, chunker_version, "
            "chunk_size, chunk_overlap, encoding, content_hash"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            chunk_params,
        )
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
    return len(chunks)


def classify_pdf_url(url: Optional[str]) -> str:
    """``upload`` for a retained file, ``url`` for http(s), ``skipped`` otherwise."""
    if not url or not str(url).strip():
        return "skipped"
    if url.startswith("file://"):
        return "upload"
    if url.startswith("http://") or url.startswith("https://"):
        return "url"
    return "skipped"


def load_paper_body(paper) -> tuple[str, str]:
    """Re-extract a retained upload, or re-download a URL paper, and return text.

    URL papers skip the download cache so the backfill reads the file again.
    ``file://`` paths are the uploads volume; ``download_pdf`` returns that
    path without fetching anything.

    Returns:
        ``(full_text, extractor)``

    Raises:
        PaperTextWriteError: there is no PDF, or extraction produced no text.
    """
    url = getattr(paper, "url_pdf", None) or ""
    kind = classify_pdf_url(url)
    if kind == "skipped":
        raise PaperTextWriteError(f"{paper.id} has no resolvable PDF URL")
    # URL papers are fetched again. Retained uploads are already on disk.
    path = download_pdf(url, use_cache=(kind != "url"))
    extracted: ExtractedText = extract_document(path)
    if not extracted.text.strip():
        raise PaperTextWriteError(f"no text extracted from {url}")
    return extracted.text, extracted.extractor


def run_backfill(
    papers: Sequence,
    stored_ids: Sequence[str],
    *,
    dry_run: bool,
    extract: Callable = load_paper_body,
    write: Optional[Callable[[str, str, str], int]] = None,
) -> BackfillCounts:
    """Re-extract every paper that has a PDF and replace its stored chunks.

    Dry-run classifies and counts. It does not extract, download, or write.
    A real run is idempotent: ``write`` replaces the paper's chunks, so a
    second pass stores the same set rather than appending.

    One paper's failure is counted and the pass continues, so the summary is
    complete. ``failed`` is non-zero when any paper was not stored; the CLI
    exits non-zero in that case.
    """
    counts = BackfillCounts(dry_run=dry_run, papers=len(papers))
    stored = set(stored_ids)
    for paper in papers:
        kind = classify_pdf_url(getattr(paper, "url_pdf", None))
        if kind == "skipped":
            counts.skipped += 1
            continue
        if kind == "upload":
            counts.uploads += 1
        else:
            counts.urls += 1
        had_text = paper.id in stored
        if had_text:
            counts.already_stored += 1
        if dry_run:
            counts.would_write += 1
            continue
        if write is None:
            raise PaperTextWriteError("backfill write callable is required")
        try:
            text, extractor = extract(paper)
            n_chunks = write(paper.id, text, extractor)
            counts.written += 1
            counts.chunks += int(n_chunks)
            if had_text:
                counts.replaced += 1
        except Exception as e:
            counts.failed += 1
            counts.failures.append(f"{paper.id}: {e}")
            logger.error("Backfill failed for %s: %s", paper.id, e)
    return counts
