#!/usr/bin/env python3
"""SciFact-subset check for the local embedding model.

Re-runs the comparison described in the local-knowledge report, section 3.5:
the relevance-labelled SciFact abstracts, plus random filler up to 1,500
documents, and the test claims, scored with exact cosine and nDCG@10.

The filler sample uses seed 0. Absolute nDCG will not match one earlier run
unless that run used the same seed; later runs of this script match each other.

Not part of CI. The first run downloads the BEIR SciFact zip and the pinned
ONNX weights (about 208 MB) into the cache directory. It does not call OpenAI.

    python scripts/eval_embeddings.py
    python scripts/eval_embeddings.py --docs 1500 --seed 0

pytest does not collect this file (it lives outside tests/ and defines no
test functions). Do not import it from the suite.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
DEFAULT_DOCS = 1500
DEFAULT_SEED = 0
K = 10


def ndcg_at_k(ranked_ids, relevant, k=K):
    """nDCG@k with graded gains. ``relevant`` maps doc id to a positive grade."""

    def dcg(grades):
        total = 0.0
        for i, grade in enumerate(grades[:k]):
            if grade:
                total += (2 ** grade - 1) / math.log2(i + 2)
        return total

    gained = [relevant.get(doc_id, 0) for doc_id in ranked_ids]
    ideal = sorted(relevant.values(), reverse=True)
    idcg = dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return dcg(gained) / idcg


def _load_jsonl(path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[str(row["_id"])] = row
    return rows


def _load_qrels(path):
    qrels = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            query_id, doc_id, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


def _ensure_scifact(root: Path) -> Path:
    dataset = root / "scifact"
    if (dataset / "corpus.jsonl").is_file() and (dataset / "queries.jsonl").is_file():
        return dataset
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "scifact.zip"
    if not archive.is_file():
        print(f"downloading {SCIFACT_URL}", file=sys.stderr)
        urllib.request.urlretrieve(SCIFACT_URL, archive)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(root)
    if not (dataset / "corpus.jsonl").is_file():
        raise SystemExit(f"SciFact corpus not found under {dataset}")
    return dataset


def _subset(corpus, qrels, n_docs, seed):
    labelled = set()
    for grades in qrels.values():
        labelled.update(grades)
    labelled = [doc_id for doc_id in labelled if doc_id in corpus]
    rest = [doc_id for doc_id in corpus if doc_id not in set(labelled)]
    random.Random(seed).shuffle(rest)
    chosen = labelled + rest[: max(0, n_docs - len(labelled))]
    return chosen, len(labelled)


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _unit(vector):
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=DEFAULT_DOCS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="SciFact cache. Default: EMBEDDING_CACHE_DIR or ~/.cache/nlp-embeddings.",
    )
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from nlp_pillars.embeddings import LocalBgeEmbeddingProvider, cache_dir

    data_root = args.data_dir or (cache_dir() / "beir")
    dataset = _ensure_scifact(data_root)
    corpus = _load_jsonl(dataset / "corpus.jsonl")
    queries = _load_jsonl(dataset / "queries.jsonl")
    qrels_path = dataset / "qrels" / "test.tsv"
    qrels = _load_qrels(qrels_path)

    doc_ids, n_labelled = _subset(corpus, qrels, args.docs, args.seed)
    query_ids = [qid for qid in queries if qid in qrels]
    print(
        f"docs: {len(doc_ids)} (labelled {n_labelled}, seed {args.seed})",
        file=sys.stderr,
    )
    print(f"queries: {len(query_ids)}", file=sys.stderr)

    provider = LocalBgeEmbeddingProvider()

    def passage(doc_id):
        row = corpus[doc_id]
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        return f"{title} {text}".strip()

    started = time.perf_counter()
    doc_vectors = [
        _unit(vector)
        for vector in provider.embed_documents([passage(doc_id) for doc_id in doc_ids])
    ]
    elapsed = time.perf_counter() - started
    rate = len(doc_ids) / elapsed if elapsed else 0.0

    scores = []
    for query_id in query_ids:
        query_vector = _unit(provider.embed_query(queries[query_id]["text"]))
        order = sorted(
            range(len(doc_ids)),
            key=lambda index: _dot(query_vector, doc_vectors[index]),
            reverse=True,
        )
        ranked = [doc_ids[index] for index in order[:K]]
        scores.append(ndcg_at_k(ranked, qrels[query_id], K))

    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"model: {provider.model_id}")
    print(f"dims: {provider.dims}")
    print(f"nDCG@{K}: {mean:.4f}")
    print(f"docs_per_s: {rate:.2f}")
    print(f"queries_scored: {len(scores)}")


if __name__ == "__main__":
    main()
