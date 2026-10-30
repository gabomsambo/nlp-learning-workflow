#!/usr/bin/env python3
"""Evaluate dense versus hybrid retrieval on a SciFact subset.

This is a local, opt-in benchmark and is not run in CI. It downloads BEIR
SciFact plus the pinned bge-base ONNX weights on first use, writes only to the
named Qdrant collection, and makes no paid API calls.

Example (against a throwaway Qdrant, never the application collection):

    python scripts/eval_search.py --qdrant-url http://127.0.0.1:56334

The default 1,500-document subset contains every relevance-labelled document
and deterministic random filler, matching the model comparison in the local
knowledge report. Results are nDCG@10 over SciFact's 300 test claims.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Iterable

from qdrant_client import QdrantClient, models

from nlp_pillars.embeddings import get_embedding_provider
from nlp_pillars.projector import BM25_VECTOR, bm25_passage, bm25_query

DATASET_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/"
    "datasets/scifact.zip"
)
COLLECTION = "nlp_eval_scifact_bge_base"
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "nlp-search-eval/scifact")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _download_scifact(cache: Path) -> Path:
    root = cache / "scifact"
    if (root / "corpus.jsonl").is_file():
        return root
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "scifact.zip"
    if not archive.is_file():
        print(f"Downloading {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(cache)
    return root


def _qrels(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            query_id, doc_id, score = line.rstrip("\n").split("\t")
            rows.setdefault(query_id, {})[doc_id] = int(score)
    return rows


def _subset(
    root: Path, size: int
) -> tuple[list[dict], list[dict], dict[str, dict[str, int]]]:
    corpus = _read_jsonl(root / "corpus.jsonl")
    queries = _read_jsonl(root / "queries.jsonl")
    relevance = _qrels(root / "qrels" / "test.tsv")
    relevant_ids = {doc_id for values in relevance.values() for doc_id in values}
    by_id = {str(row["_id"]): row for row in corpus}
    if size < len(relevant_ids):
        raise ValueError(
            f"corpus size {size} is smaller than {len(relevant_ids)} relevant documents"
        )
    filler = sorted(set(by_id) - relevant_ids)
    random.Random(7).shuffle(filler)
    selected = sorted(relevant_ids) + filler[: max(0, size - len(relevant_ids))]
    selected_set = set(selected)
    filtered_qrels = {
        query_id: {
            doc_id: score
            for doc_id, score in values.items()
            if doc_id in selected_set
        }
        for query_id, values in relevance.items()
    }
    filtered_queries = [row for row in queries if str(row["_id"]) in filtered_qrels]
    return [by_id[doc_id] for doc_id in selected], filtered_queries, filtered_qrels


def _batches(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _text(row: dict) -> str:
    parts = (row.get("title", ""), row.get("text", ""))
    return "\n\n".join(part for part in parts if part)


def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, doc_id))


def _index(client: QdrantClient, collection: str, corpus: list[dict]) -> None:
    if not collection.startswith("nlp_eval_"):
        raise ValueError("eval collection names must start with 'nlp_eval_'")
    provider = get_embedding_provider(provider="local", model="BAAI/bge-base-en-v1.5")
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=provider.dims, distance=models.Distance.COSINE
        ),
        sparse_vectors_config={
            BM25_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    total = len(corpus)
    for batch_no, batch in enumerate(_batches(corpus, 32), start=1):
        texts = [_text(row) for row in batch]
        dense = provider.embed_documents(texts)
        points = [
            models.PointStruct(
                id=_point_id(str(row["_id"])),
                vector={"": vector, BM25_VECTOR: bm25_passage(text)},
                payload={"doc_id": str(row["_id"])},
            )
            for row, text, vector in zip(batch, texts, dense, strict=True)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        print(f"Indexed {min(batch_no * 32, total)}/{total}", end="\r", flush=True)
    print()


def _dcg(ranking: list[str], relevant: dict[str, int]) -> float:
    return sum(
        (2 ** relevant.get(doc_id, 0) - 1) / math.log2(rank + 2)
        for rank, doc_id in enumerate(ranking[:10])
    )


def _ndcg(ranking: list[str], relevant: dict[str, int]) -> float:
    ideal = sorted(relevant.values(), reverse=True)[:10]
    ideal_dcg = sum(
        (2**score - 1) / math.log2(rank + 2)
        for rank, score in enumerate(ideal)
    )
    return _dcg(ranking, relevant) / ideal_dcg if ideal_dcg else 0.0


def _evaluate(
    client: QdrantClient,
    collection: str,
    queries: list[dict],
    relevance: dict[str, dict[str, int]],
) -> tuple[float, float]:
    provider = get_embedding_provider(provider="local", model="BAAI/bge-base-en-v1.5")
    dense_scores: list[float] = []
    hybrid_scores: list[float] = []
    for number, row in enumerate(queries, start=1):
        query_id = str(row["_id"])
        text = str(row["text"])
        dense = provider.embed_query(text)
        dense_hits = client.query_points(
            collection_name=collection,
            query=dense,
            limit=10,
            with_payload=True,
        ).points
        hybrid_hits = client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(query=dense, limit=30),
                models.Prefetch(
                    query=bm25_query(text), using=BM25_VECTOR, limit=30
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=10,
            with_payload=True,
        ).points
        dense_scores.append(
            _ndcg(
                [str(hit.payload["doc_id"]) for hit in dense_hits],
                relevance[query_id],
            )
        )
        hybrid_scores.append(
            _ndcg(
                [str(hit.payload["doc_id"]) for hit in hybrid_hits],
                relevance[query_id],
            )
        )
        print(f"Queried {number}/{len(queries)}", end="\r", flush=True)
    print()
    dense_mean = sum(dense_scores) / len(dense_scores)
    hybrid_mean = sum(hybrid_scores) / len(hybrid_scores)
    return dense_mean, hybrid_mean


def main() -> None:
    """Run the SciFact subset benchmark and print dense/hybrid nDCG@10."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:3023")
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--corpus-size", type=int, default=1500)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "nlp-search-eval",
    )
    args = parser.parse_args()

    root = _download_scifact(args.cache_dir)
    corpus, queries, relevance = _subset(root, args.corpus_size)
    client = QdrantClient(url=args.qdrant_url)
    _index(client, args.collection, corpus)
    dense, hybrid = _evaluate(client, args.collection, queries, relevance)
    print(f"dense nDCG@10={dense:.3f}")
    print(f"hybrid nDCG@10={hybrid:.3f}")
    print(f"delta={hybrid - dense:+.3f}")


if __name__ == "__main__":
    main()
