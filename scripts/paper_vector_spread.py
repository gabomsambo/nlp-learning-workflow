#!/usr/bin/env python3
"""Report the spread of pairwise cosines for title+abstract paper vectors.

Reads a JSON array of vectors (each an array of floats) from a file, or from
stdin when the path is ``-``. Prints papers, pairs, min, max, mean, and the
population std. The graph-page report measured std 0.093 (range 0.43–0.84)
for title+abstract bge vectors and std 0.042 (range 0.79–0.97) for mean-pooled
chunk vectors. ``related_to`` keeps pairs at or above 0.70 because of that
gap. If a later library's std collapses toward 0.04, the floor needs another
look.

Does not open a database and does not load a model.

    python scripts/paper_vector_spread.py vectors.json
    python scripts/paper_vector_spread.py - < vectors.json

pytest does not collect this file (it lives outside tests/). The stats
function it calls is covered from the test suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp_pillars.paper_vectors import pairwise_spread  # noqa: E402


def _load(path: str):
    if path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise SystemExit("expected a JSON array of vectors")
    vectors = []
    for item in data:
        if not isinstance(item, list):
            raise SystemExit("each vector must be a JSON array of numbers")
        vectors.append([float(value) for value in item])
    return vectors


def main(argv: list[str]) -> int:
    """Print pairwise cosine spread for one JSON file. Returns a status code."""
    if len(argv) != 2:
        print(
            "usage: python scripts/paper_vector_spread.py <vectors.json | ->",
            file=sys.stderr,
        )
        return 2
    try:
        vectors = _load(argv[1])
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(pairwise_spread(vectors).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
