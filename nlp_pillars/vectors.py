"""
Qdrant read path and the hook that projects stored embeddings.

Postgres holds the chunk text and the vectors (``paper_chunks``,
``chunk_embeddings``). This module does not embed. ``upsert_text`` projects
rows that are already stored: local Qdrant immediately, and Qdrant Cloud
when ``QDRANT_CLOUD_URL`` is set. Search and discovery read the local index
(``QDRANT_URL``) through the ``nlp_chunks`` alias, and embed the query with
the same provider that wrote ``chunk_embeddings``.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from .projector import (
    PAPER_ID_FIELD,
    PILLAR_ID_FIELD,
    READ_ALIAS,
    ProjectionError,
    ensure_index,
    project_paper,
)

logger = logging.getLogger(__name__)

# Module-level singleton. Search and the local projection share it.
_client: Optional[QdrantClient] = None

# What search queries. The physical collection is per model; the alias is
# what a model change flips.
COLLECTION_NAME = READ_ALIAS

__all__ = [
    "COLLECTION_NAME",
    "PAPER_ID_FIELD",
    "PILLAR_ID_FIELD",
    "ProjectionError",
    "ensure_collections",
    "get_client",
    "project_paper",
    "search_similar",
    "set_client",
    "upsert_text",
]


def get_client() -> Optional[QdrantClient]:
    """
    Get or create the Qdrant client singleton.

    This is the local index (``QDRANT_URL``). Cloud is a separate client
    opened by the projector only while syncing.

    Returns:
        QdrantClient instance or None if not configured
    """
    global _client

    if _client is not None:
        return _client

    url = os.getenv('QDRANT_URL')
    if not url:
        logger.warning("QDRANT_URL environment variable not set. Vector operations will be disabled.")
        return None

    api_key = os.getenv('QDRANT_API_KEY')  # Optional for local deployments

    try:
        if api_key:
            _client = QdrantClient(url=url, api_key=api_key)
        else:
            _client = QdrantClient(url=url)

        logger.info(f"Connected to Qdrant at {url}")
        return _client

    except Exception as e:
        logger.error(f"Failed to connect to Qdrant at {url}: {e}")
        return None


def _query_vector(text: str) -> List[float]:
    """Embed a search query with the same provider the index was built from.

    The local model adds its query prefix here. A download or an API error
    is a runtime failure: ``search_similar`` turns it into an empty result,
    the same way a transport error does.
    """
    from .embeddings import get_embedding_provider

    return get_embedding_provider().embed_query(text.strip())


def ensure_collections() -> None:
    """Ensure the active model's collection, payload indexes, and alias exist.

    Does not embed and does not download weights: the provider's ``dims``
    are known without loading the ONNX file. Failures are logged. Search
    raises if the alias is actually missing; an unreachable server at
    startup must not stop the process.
    """
    client = get_client()
    if client is None:
        logger.warning("Qdrant client not available. Cannot ensure collections.")
        return

    try:
        from .embeddings import get_embedding_provider

        provider = get_embedding_provider()
        ensure_index(client, provider.model_id, provider.dims)
    except Exception as e:
        logger.error(f"Failed to ensure vector collection: {e}")


def upsert_text(
    pillar_id: str,
    paper_id: str,
    full_text: str,
    chunk_size: int = 0,
    overlap: int = 0,
) -> int:
    """Project ``paper_id``'s stored chunks into the vector indexes.

    The persist stage has already written ``paper_chunks`` and
    ``chunk_embeddings``. This reads those rows and upserts them. It does
    not call an embedding API. ``chunk_size`` and ``overlap`` are accepted
    so existing callers keep working; the stored chunker config is what
    was used.

    Returns how many of the paper's chunks the local index represents.
    Already-projected chunks count, so a second run is not "zero chunks".

    A missing local client returns 0 (the VECTORS stage treats that as
    failure for non-empty text). A local server that is up but rejects the
    write raises ``ProjectionError`` with the reason. A configured cloud
    target that is down is logged and does not change the return.

    Raises:
        ProjectionError: stored rows are missing, or the local index failed.
    """
    del chunk_size, overlap  # stored chunks already used the vector chunker

    if not full_text or not str(full_text).strip():
        logger.warning("Empty text provided for upsert")
        return 0

    if get_client() is None:
        logger.warning("Qdrant client not available. Cannot upsert text.")
        return 0

    logger.info("Projecting stored embeddings for paper %s in pillar %s", paper_id, pillar_id)
    projected = project_paper(pillar_id, paper_id)
    logger.info(
        "Local index represents %s chunk(s) of paper %s in pillar %s",
        projected,
        paper_id,
        pillar_id,
    )
    return projected


def search_similar(pillar_id: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar text chunks within a pillar namespace.

    The query is embedded by the same provider that built the index.
    Results come from the ``nlp_chunks`` alias on the local server.

    Args:
        pillar_id: Pillar to search within
        query_text: Text to find similar content for
        top_k: Maximum number of results to return

    Returns:
        List of {"paper_id": str, "score": float} dictionaries,
        deduplicated by paper_id (keeping highest score) and limited to top_k

    Raises:
        RuntimeError: If the query call does not match the installed
            qdrant-client's API, or the server rejects the request with a 4xx
            (e.g. the ``pillar_id`` payload index is missing under strict mode).
            Both mean this module disagrees with its library or its server —
            not a search that found nothing — so neither may be reported as an
            empty result set.
    """
    client = get_client()
    if client is None:
        logger.warning("Qdrant client not available. Cannot search similar text.")
        return []

    if not query_text or not query_text.strip():
        logger.warning("Empty query text provided")
        return []

    logger.info(f"Searching for similar text in pillar {pillar_id} with top_k={top_k}")

    # Embedding is a model call: a genuine runtime failure, and degrading
    # to "no vector candidates" is the right answer for it.
    try:
        query_vector = _query_vector(query_text.strip())
    except Exception as e:
        logger.error(f"Failed to embed query text for pillar {pillar_id}: {e}")
        return []

    # query_points(), NOT search(). search()/search_batch()/search_groups()/
    # recommend()/discover() were all removed in qdrant-client 1.19, the pinned
    # version. The call below therefore returns a QueryResponse whose hits live
    # on `.points`, where search() returned the list of hits directly.
    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="pillar_id",
                        match=models.MatchValue(value=pillar_id)
                    )
                ]
            ),
            limit=top_k * 3,  # Get more results for deduplication
            with_payload=True
        )
    except (AttributeError, TypeError) as e:
        # A missing method or a rejected signature means this code and the
        # installed qdrant-client disagree — a bug that no retry or fallback
        # can fix, and one that returning [] hid for months (every query looked
        # like "nothing matched"). Same precedent as pdf_loader.chunk_text().
        raise RuntimeError(
            f"Qdrant query_points() is incompatible with the installed "
            f"qdrant-client: {e}"
        ) from e
    except UnexpectedResponse as e:
        # A 4xx is the server rejecting *this request*: a bad filter, a missing
        # payload index under strict mode, a missing collection, a bad key.
        # None of that is "nothing matched", and no retry fixes it, so it gets
        # the same treatment as the signature mismatch above. 5xx and anything
        # else falls through to the tolerant branch.
        if e.status_code is not None and e.status_code < 500:
            raise RuntimeError(
                f"Qdrant rejected the search in pillar {pillar_id} "
                f"({e.status_code}): {e.reason_phrase}. {e.content!r}"
            ) from e
        logger.error(
            f"Failed to search similar text in pillar {pillar_id}: {e}",
            exc_info=True
        )
        return []
    except Exception as e:
        # Everything else — network, timeout, transport — is a real runtime
        # failure. Callers treat [] as "no candidates" and carry on.
        logger.error(
            f"Failed to search similar text in pillar {pillar_id}: {e}",
            exc_info=True
        )
        return []

    hits = getattr(response, "points", None)
    if hits is None:
        raise RuntimeError(
            f"qdrant-client returned an unexpected query_points() response "
            f"({type(response).__name__}); expected a QueryResponse with .points"
        )

    # Deduplicate by paper_id, keeping highest score
    paper_scores: Dict[str, float] = {}
    for hit in hits:
        paper_id = (hit.payload or {}).get("paper_id")
        score = hit.score

        if paper_id and (paper_id not in paper_scores or score > paper_scores[paper_id]):
            paper_scores[paper_id] = score

    # Sort by score and limit to top_k
    results = [
        {"paper_id": paper_id, "score": score}
        for paper_id, score in sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)
    ][:top_k]

    logger.info(f"Found {len(results)} similar papers in pillar {pillar_id}")
    return results


def set_client(client: Optional[QdrantClient]) -> None:
    """Set the client singleton (for testing)."""
    global _client
    _client = client
