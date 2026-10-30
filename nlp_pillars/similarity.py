"""Calibrated nearest neighbours over one scope of title+abstract vectors.

Pure: vectors in, numbers out. No database, no model, no network.

A fixed cosine floor is a property of one corpus. ``RELATED_MIN_COSINE`` (0.70)
was measured on 12 library papers; over the 1,227-work neighbourhood of the
paper-map scout it passes 20.7% of all 752,151 pairs (p90 0.732, p99 0.804), so
it links half the neighbourhood to everything. The floor therefore comes from
the scope itself: the ``RELATED_PERCENTILE``-th percentile of every pairwise
cosine in it. The rank cut (top-k) is what bounds the degree, and the floor only
drops neighbours that are the best of a bad lot.

Hubness is the other way a kNN graph misleads. In high dimensions a few
vectors turn up in many other vectors' neighbour lists and most turn up in
none. The skewness of that count (the k-occurrence N_k) is the usual health
check; past about 1.4 a hubness correction (CSLS, mutual proximity) is worth
adding. Nothing here applies one. It only reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Percentile of the scope's pairwise cosines used as the floor. The report's
# "floor at the scope's p97" (paper-map scout, section 5.6).
RELATED_PERCENTILE = 97.0

# Below this many vectors the pairs are too few for a percentile to mean
# anything (13 papers is 78 pairs, so p97 is three pairs). Such a scope keeps
# the absolute floor the caller passes in, which was measured on a library of
# exactly that size.
MIN_PERCENTILE_SCOPE = 30

# Neighbours per vector for the k-occurrence health check. Ten is the layout
# size from the report; it is deliberately not the (smaller) edge k.
HUBNESS_K = 10

# Past this skewness a hubness correction is worth adding (report section 3.2).
HUBNESS_SKEWNESS_LIMIT = 1.4

# All pairs are used up to this many vectors (4.5M pairs at 3,000). A larger
# scope samples pairs with a fixed seed, so the floor is the same on a rerun.
EXACT_PAIRS_MAX_VECTORS = 3000
SAMPLED_PAIRS = 2_000_000
SAMPLE_SEED = 0

# Rows of the similarity matrix held at once while counting neighbours.
_BLOCK = 1024

FLOOR_PERCENTILE = "percentile"
FLOOR_ABSOLUTE = "absolute"


@dataclass(frozen=True)
class Floor:
    """The cosine a neighbour must reach, and where that number came from."""

    value: float
    source: str  # FLOOR_PERCENTILE or FLOOR_ABSOLUTE
    scope_size: int
    pairs: int  # pairs the percentile was read from; 0 for an absolute floor
    sampled: bool = False


@dataclass(frozen=True)
class Hubness:
    """Skewness of the k-occurrence of every vector in the scope."""

    k: int
    scope_size: int
    skewness: Optional[float]  # None: too few vectors, or every count is equal
    max_occurrence: int
    never_a_neighbour: int

    @property
    def high(self) -> bool:
        """True when a hubness correction is worth considering."""
        return self.skewness is not None and self.skewness > HUBNESS_SKEWNESS_LIMIT


@dataclass(frozen=True)
class SimilarityHealth:
    """What one related_to pass measured. ``render`` is the report line."""

    floor: Floor
    hubness: Hubness

    def render(self) -> str:
        """One plain line for the build report."""
        floor = self.floor
        if floor.source == FLOOR_PERCENTILE:
            origin = (
                f"p{RELATED_PERCENTILE:g} of "
                f"{'a sample of ' if floor.sampled else ''}"
                f"{floor.pairs:,} pairs"
            )
        else:
            origin = (
                f"absolute: {floor.scope_size} vectors is under "
                f"{MIN_PERCENTILE_SCOPE}, too few for a percentile"
            )
        hub = self.hubness
        if hub.skewness is None:
            skew = "n/a"
        else:
            skew = f"{hub.skewness:.2f}"
        verdict = (
            f"above {HUBNESS_SKEWNESS_LIMIT:g}, consider a hubness correction"
            if hub.high else "ok"
        )
        return (
            f"similarity: floor {floor.value:.4f} ({origin}, "
            f"{floor.scope_size} vectors); "
            f"k-occurrence skewness {skew} at k={hub.k} ({verdict}); "
            f"{hub.never_a_neighbour} never a neighbour, "
            f"max {hub.max_occurrence}"
        )


def unit_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """Row-normalised float64 matrix. A zero row stays zero, so its cosine is 0."""
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("vectors must all have the same width")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def scope_floor(unit: np.ndarray, absolute: float) -> Floor:
    """Percentile floor of the scope's pairwise cosines, else ``absolute``.

    Args:
        unit: ``unit_matrix`` of the whole scope, one row per work.
        absolute: floor for a scope under ``MIN_PERCENTILE_SCOPE`` vectors.

    """
    size = int(unit.shape[0])
    if size < MIN_PERCENTILE_SCOPE:
        return Floor(absolute, FLOOR_ABSOLUTE, size, 0)
    if size <= EXACT_PAIRS_MAX_VECTORS:
        scores = (unit @ unit.T)[np.triu_indices(size, k=1)]
        sampled = False
    else:
        rng = np.random.default_rng(SAMPLE_SEED)
        left = rng.integers(0, size, SAMPLED_PAIRS)
        right = rng.integers(0, size - 1, SAMPLED_PAIRS)
        right = right + (right >= left)  # skip the diagonal
        scores = np.einsum("ij,ij->i", unit[left], unit[right])
        sampled = True
    value = float(np.percentile(scores, RELATED_PERCENTILE))
    return Floor(value, FLOOR_PERCENTILE, size, int(scores.size), sampled)


def top_k_neighbours(
    unit: np.ndarray,
    rows: Sequence[int],
    k: int,
    floor: float,
    names: Sequence[str],
) -> Dict[int, List[Tuple[float, int]]]:
    """Each ``rows`` entry's best ``k`` neighbours among ``rows`` at or above ``floor``.

    A vector is never its own neighbour. Ties break by ``names`` so the result
    does not depend on input order. The cut is the rank, so a scope in which
    every pair is near 1.0 still yields at most ``k`` per vector.
    """
    out: Dict[int, List[Tuple[float, int]]] = {}
    if not rows or k < 1:
        return {row: [] for row in rows}
    sub = unit[list(rows)]
    scores = sub @ sub.T
    for position, row in enumerate(rows):
        scored = [
            (float(scores[position, other]), rows[other])
            for other in range(len(rows))
            if other != position and scores[position, other] >= floor
        ]
        scored.sort(key=lambda item: (-item[0], names[item[1]]))
        out[row] = scored[:k]
    return out


def k_occurrence_skewness(
    unit: np.ndarray, k: int = HUBNESS_K
) -> Hubness:
    """Skewness of how often each vector is in another's top-``k`` (no floor)."""
    size = int(unit.shape[0])
    k = min(k, size - 1)
    if k < 1:
        return Hubness(max(k, 0), size, None, 0, size)
    counts = np.zeros(size, dtype=np.int64)
    for start in range(0, size, _BLOCK):
        block = unit[start:start + _BLOCK] @ unit.T
        # a vector is not its own neighbour
        rows = np.arange(block.shape[0])
        block[rows, rows + start] = -np.inf
        nearest = np.argpartition(block, -k, axis=1)[:, -k:]
        counts += np.bincount(nearest.ravel(), minlength=size)
    return Hubness(
        k=k,
        scope_size=size,
        skewness=_skewness(counts),
        max_occurrence=int(counts.max()),
        never_a_neighbour=int((counts == 0).sum()),
    )


def _skewness(counts: np.ndarray) -> Optional[float]:
    """Fisher-Pearson moment coefficient. None when every count is the same."""
    if counts.size < 3:
        return None
    values = counts.astype(np.float64)
    centred = values - values.mean()
    variance = float((centred ** 2).mean())
    if variance <= 0.0:
        return None
    return float((centred ** 3).mean() / variance ** 1.5)
