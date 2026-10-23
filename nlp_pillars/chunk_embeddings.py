"""Fill ``chunk_embeddings`` for chunks that do not yet have the active model.

The Qdrant upsert is not involved. Ingest and upload call ``embed_paper``
after the chunk rows are committed. The CLI backfill uses the same write for
chunks that were stored earlier. A second pass finds nothing to do.

A failed batch leaves the rows already stored and raises. The run stage
shows the error. The next pass continues from the chunks that still lack a
row for this model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .embeddings import EmbeddingError, EmbeddingProvider


@dataclass
class EmbeddingBackfillCounts:
    """What one backfill pass did. ``render`` is the CLI's whole output."""

    dry_run: bool
    model_id: str
    missing: int = 0
    would_write: int = 0
    written: int = 0
    failed: int = 0
    failures: List[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"dry_run: {'true' if self.dry_run else 'false'}",
            f"model_id: {self.model_id}",
            f"missing: {self.missing}",
        ]
        if self.dry_run:
            lines.append(f"would_write: {self.would_write}")
        else:
            lines.append(f"written: {self.written}")
            lines.append(f"failed: {self.failed}")
        for failure in self.failures:
            lines.append(f"failure: {failure}")
        return "\n".join(lines)


def embed_chunk_rows(
    chunks: Sequence[dict],
    provider: EmbeddingProvider,
    save: Callable[[Sequence[dict]], int],
) -> int:
    """Embed ``chunks`` in provider-sized batches and ``save`` each batch.

    Each chunk dict needs ``chunk_id`` and ``chunk_text``.

    Raises:
        EmbeddingError: the model or the save failed. ``written`` on the
            exception counts rows saved before the failing batch.
    """
    if provider.batch_size < 1:
        raise EmbeddingError(f"{provider.model_id} batch_size must be positive")

    written = 0
    size = provider.batch_size
    for start in range(0, len(chunks), size):
        batch = list(chunks[start:start + size])
        texts = [row["chunk_text"] for row in batch]
        try:
            vectors = provider.embed_documents(texts)
        except EmbeddingError as e:
            raise EmbeddingError(str(e), written=written) from e
        except Exception as e:
            raise EmbeddingError(
                f"Failed to embed {len(batch)} chunk(s) with {provider.model_id}: {e}",
                written=written,
            ) from e
        if len(vectors) != len(batch):
            raise EmbeddingError(
                f"{provider.model_id} returned {len(vectors)} vectors "
                f"for {len(batch)} chunks",
                written=written,
            )
        rows = []
        for chunk, vec in zip(batch, vectors):
            if len(vec) != provider.dims:
                raise EmbeddingError(
                    f"{provider.model_id} returned {len(vec)} dims for chunk "
                    f"{chunk['chunk_id']}, expected {provider.dims}",
                    written=written,
                )
            rows.append({
                "chunk_id": chunk["chunk_id"],
                "model_id": provider.model_id,
                "dims": provider.dims,
                "embedding": [float(x) for x in vec],
            })
        try:
            save(rows)
        except EmbeddingError as e:
            raise EmbeddingError(str(e), written=written) from e
        written += len(rows)
    return written


def run_embedding_backfill(
    *,
    model_id: str,
    dry_run: bool,
    chunks: Sequence[dict],
    embed: Optional[Callable[[Sequence[dict]], int]] = None,
) -> EmbeddingBackfillCounts:
    """Count or embed ``chunks``. Dry-run does not call ``embed``.

    ``chunks`` are the ones missing ``model_id`` already. Passing the same
    paper twice is safe when the second call's ``chunks`` is empty.
    """
    counts = EmbeddingBackfillCounts(
        dry_run=dry_run,
        model_id=model_id,
        missing=len(chunks),
    )
    if dry_run:
        counts.would_write = len(chunks)
        return counts
    if not chunks:
        return counts
    if embed is None:
        raise EmbeddingError("embedding backfill has no embed function")
    try:
        counts.written = embed(chunks)
    except EmbeddingError as e:
        counts.written = e.written
        counts.failed = 1
        counts.failures.append(str(e))
    return counts


def embed_paper(paper_id: str, *, provider: Optional[EmbeddingProvider] = None) -> int:
    """Embed one paper's chunks that lack the active model's row.

    Returns the number of rows written. Zero means every chunk already had
    this model. No chunk rows at all is an error: the text write that just
    ran is supposed to have stored some.

    Raises:
        EmbeddingError: the read, the model, or the insert failed.
    """
    from . import db
    from .embeddings import active_model_id, get_embedding_provider

    model_id = provider.model_id if provider is not None else active_model_id()
    missing = db.list_chunks_missing_embeddings(model_id, paper_id)
    if not missing:
        if not db.paper_has_chunks(paper_id):
            raise EmbeddingError(
                f"no chunks stored for {paper_id}; embeddings were not computed"
            )
        return 0
    if provider is None:
        provider = get_embedding_provider()
    if provider.model_id != model_id:
        raise EmbeddingError(
            f"provider model {provider.model_id} does not match {model_id}"
        )
    return embed_chunk_rows(missing, provider, db.insert_chunk_embeddings)
