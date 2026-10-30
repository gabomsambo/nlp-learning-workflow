"""One title+abstract vector per library paper and per external work.

Chunk embeddings stay the search copy. These rows are what ``related_to``
reads. The command always uses the pinned local bge model: the cosine floor
in ``graph.py`` was measured on that model, and this path does not call OpenAI.

Library papers are ``ref_kind='paper'`` keyed by ``papers.id``. A work in the
citation neighbourhood that is not a library paper is ``ref_kind=
'external_paper'`` keyed by ``works.id`` (migration 031): the percentile floor
in ``similarity.py`` needs them to know what a typical pair in the scope looks
like. Semantic Scholar's own SPECTER2 vectors are not stored; bge is the one
space (paper-map scout, section 3.2).

A paper that already has ``(paper, id, model_id, title_abstract)`` is left
alone, so a second run writes nothing and does not load the model. A paper
with neither a title nor an abstract is skipped. Title alone is embedded when
the abstract is missing. The text is the stripped title, a blank line, and
the stripped abstract. Passages go through ``embed_documents`` (bge document
prefix is empty).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Set

from .embeddings import EmbeddingError, EmbeddingProvider

MIGRATION = "docs/migrations/021_paper_vectors.sql"
SOURCE_TITLE_ABSTRACT = "title_abstract"
REF_KIND_PAPER = "paper"
REF_KIND_EXTERNAL = "external_paper"


def title_abstract_text(title: Optional[str], abstract: Optional[str]) -> str:
    """Passage text for one paper. Empty when both fields are blank."""
    title_text = (title or "").strip()
    abstract_text = (abstract or "").strip()
    if title_text and abstract_text:
        return f"{title_text}\n\n{abstract_text}"
    return title_text or abstract_text


@dataclass
class PaperVectorReport:
    """What one ``embeddings papers`` pass did. ``render`` is the CLI output."""

    dry_run: bool
    model_id: str
    source: str
    ref_kind: str = REF_KIND_PAPER
    library: int = 0
    already: int = 0
    skipped_empty: int = 0
    missing: int = 0
    would_write: int = 0
    written: int = 0
    failed: int = 0
    failures: List[str] = field(default_factory=list)

    def render(self) -> str:
        """Plain-text summary. Markup is left off so ids are not styled."""
        total = "external" if self.ref_kind == REF_KIND_EXTERNAL else "library"
        lines = [
            f"dry_run: {'true' if self.dry_run else 'false'}",
            f"model_id: {self.model_id}",
            f"source: {self.source}",
            f"{total}: {self.library}",
            f"already: {self.already}",
            f"skipped_empty: {self.skipped_empty}",
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


def _paper_id(paper: Any) -> str:
    if isinstance(paper, dict):
        return str(paper.get("id") or "")
    return str(getattr(paper, "id", "") or "")


def _paper_text(paper: Any) -> str:
    if isinstance(paper, dict):
        return title_abstract_text(paper.get("title"), paper.get("abstract"))
    return title_abstract_text(
        getattr(paper, "title", None),
        getattr(paper, "abstract", None),
    )


def embed_paper_vector_rows(
    papers: Sequence[dict],
    provider: EmbeddingProvider,
    save: Callable[[Sequence[dict]], int],
    ref_kind: str = REF_KIND_PAPER,
) -> int:
    """Embed ``papers`` in provider-sized batches and ``save`` each batch.

    Each dict needs ``ref_id`` and ``text``.

    Raises:
        EmbeddingError: the model or the save failed. ``written`` on the
            exception counts rows saved before the failing batch.

    """
    if provider.batch_size < 1:
        raise EmbeddingError(f"{provider.model_id} batch_size must be positive")

    written = 0
    size = provider.batch_size
    for start in range(0, len(papers), size):
        batch = list(papers[start:start + size])
        texts = [row["text"] for row in batch]
        try:
            vectors = provider.embed_documents(texts)
        except EmbeddingError as exc:
            raise EmbeddingError(str(exc), written=written) from exc
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed {len(batch)} paper(s) with "
                f"{provider.model_id}: {exc}",
                written=written,
            ) from exc
        if len(vectors) != len(batch):
            raise EmbeddingError(
                f"{provider.model_id} returned {len(vectors)} vectors "
                f"for {len(batch)} papers",
                written=written,
            )
        rows = []
        for paper, vec in zip(batch, vectors, strict=True):
            if len(vec) != provider.dims:
                raise EmbeddingError(
                    f"{provider.model_id} returned {len(vec)} dims for paper "
                    f"{paper['ref_id']}, expected {provider.dims}",
                    written=written,
                )
            rows.append({
                "ref_kind": ref_kind,
                "ref_id": paper["ref_id"],
                "model_id": provider.model_id,
                "source": SOURCE_TITLE_ABSTRACT,
                "dims": provider.dims,
                "vec": [float(value) for value in vec],
            })
        try:
            save(rows)
        except EmbeddingError as exc:
            raise EmbeddingError(str(exc), written=written) from exc
        written += len(rows)
    return written


def run_paper_vector_backfill(
    *,
    model_id: str,
    dry_run: bool,
    papers: Sequence[Any],
    already: Set[str],
    embed: Optional[Callable[[Sequence[dict]], int]] = None,
    ref_kind: str = REF_KIND_PAPER,
) -> PaperVectorReport:
    """Count or embed papers (or external works) that lack ``model_id``.

    Dry-run does not call ``embed``. Neither does a second pass whose
    ``already`` set covers every paper that has text. ``embed`` receives only
    the missing rows, each ``{ref_id, text}``.
    """
    report = PaperVectorReport(
        dry_run=dry_run,
        model_id=model_id,
        source=SOURCE_TITLE_ABSTRACT,
        ref_kind=ref_kind,
        library=len(papers),
    )
    pending: List[dict] = []
    for paper in papers:
        paper_id = _paper_id(paper)
        if not paper_id:
            report.skipped_empty += 1
            continue
        if paper_id in already:
            report.already += 1
            continue
        text = _paper_text(paper)
        if not text:
            report.skipped_empty += 1
            continue
        pending.append({"ref_id": paper_id, "text": text})
    report.missing = len(pending)
    if dry_run:
        report.would_write = len(pending)
        return report
    if not pending:
        return report
    if embed is None:
        raise EmbeddingError("paper vector backfill has no embed function")
    try:
        report.written = embed(pending)
    except EmbeddingError as exc:
        report.written = exc.written
        report.failed = 1
        report.failures.append(str(exc))
    return report


@dataclass(frozen=True)
class PairwiseSpread:
    """Population stats of pairwise cosines. ``std`` matches ``statistics.pstdev``."""

    papers: int
    pairs: int
    minimum: Optional[float]
    maximum: Optional[float]
    mean: Optional[float]
    std: Optional[float]

    def render(self) -> str:
        """Plain-text summary with six decimal places, or ``n/a``."""
        def _num(value: Optional[float]) -> str:
            if value is None:
                return "n/a"
            return f"{value:.6f}"

        return "\n".join([
            f"papers: {self.papers}",
            f"pairs: {self.pairs}",
            f"min: {_num(self.minimum)}",
            f"max: {_num(self.maximum)}",
            f"mean: {_num(self.mean)}",
            f"std: {_num(self.std)}",
        ])


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Local cosine so this module can load without importing the graph."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for x_value, y_value in zip(left, right, strict=True):
        dot += x_value * y_value
        left_norm += x_value * x_value
        right_norm += y_value * y_value
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def pairwise_spread(vectors: Sequence[Sequence[float]]) -> PairwiseSpread:
    """Std of every unordered pair's cosine. Fewer than two vectors is empty.

    Population std (divide by the number of pairs), which is what a full
    library's pairwise set is. The graph-page report's title+abstract figure
    on 12 papers was std 0.093 (range 0.43–0.84). Mean-pooled chunk centroids
    on the same papers were std 0.042 (range 0.79–0.97). A later library whose
    std collapses toward that chunk figure needs another look at the floor.
    """
    usable = [list(vector) for vector in vectors if vector]
    if len(usable) < 2:
        return PairwiseSpread(
            papers=len(usable),
            pairs=0,
            minimum=None,
            maximum=None,
            mean=None,
            std=None,
        )
    scores: List[float] = []
    for index, left in enumerate(usable):
        for right in usable[index + 1:]:
            scores.append(_cosine(left, right))
    count = len(scores)
    mean = sum(scores) / count
    variance = sum((score - mean) ** 2 for score in scores) / count
    return PairwiseSpread(
        papers=len(usable),
        pairs=count,
        minimum=min(scores),
        maximum=max(scores),
        mean=mean,
        std=math.sqrt(variance),
    )
