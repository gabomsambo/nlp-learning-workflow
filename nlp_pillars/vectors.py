"""
Qdrant-based vector store utility with strict namespace routing by pillar.

Provides vector storage and search functionality with pillar isolation.
All operations are namespaced by pillar_id to ensure data separation.
"""

import logging
import os
import hashlib
import uuid
from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from openai import OpenAI

from .tools.pdf_loader import chunk_text

logger = logging.getLogger(__name__)

# Module-level singletons
_client: Optional[QdrantClient] = None
_openai_client: Optional[OpenAI] = None
_vector_size: Optional[int] = None

COLLECTION_NAME = "nlp_pillars"

# Payload key every read filters on, and therefore the one key that must carry
# a Qdrant payload index — see _ensure_payload_indexes().
PILLAR_ID_FIELD = "pillar_id"


def get_client() -> Optional[QdrantClient]:
    """
    Get or create the Qdrant client singleton.
    
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


def _get_openai_client() -> Optional[OpenAI]:
    """Get or create OpenAI client for embeddings."""
    global _openai_client

    if _openai_client is not None:
        return _openai_client

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is required for embeddings")
        return None

    _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _embed(text: str) -> List[float]:
    """
    Generate embedding vector for text using OpenAI.
    
    Args:
        text: Text to embed
        
    Returns:
        List of floats representing the embedding vector
        
    Raises:
        RuntimeError: If OpenAI client is not configured or embedding fails
    """
    client = _get_openai_client()
    if not client:
        raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY environment variable.")

    model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

    try:
        response = client.embeddings.create(
            model=model,
            input=text.strip()
        )

        return response.data[0].embedding

    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding with model {model}: {e}")


def _get_vector_size() -> int:
    """Get or determine vector size by embedding a test string."""
    global _vector_size

    if _vector_size is not None:
        return _vector_size

    try:
        # Embed a small test string to determine vector size
        test_vector = _embed("test")
        _vector_size = len(test_vector)
        logger.info(f"Determined vector size: {_vector_size}")
        return _vector_size

    except Exception as e:
        logger.error(f"Failed to determine vector size: {e}")
        # Default size for text-embedding-3-small
        _vector_size = 1536
        logger.warning(f"Using default vector size: {_vector_size}")
        return _vector_size


def _ensure_payload_indexes(client: QdrantClient) -> None:
    """
    Ensure `pillar_id` is a keyword payload index on the collection.

    Every read in this module filters by `pillar_id` — that is the namespace
    isolation the whole module is built around. Qdrant Cloud enables strict
    mode, whose `unindexed_filtering_retrieve: false` rejects a filtered query
    on an unindexed key with `400 Bad Request: Index required but not found`.
    Without this index the read path cannot work at all, however the query is
    spelled.

    Idempotent: the index is created only when the collection reports it
    missing, so this is safe to call on every startup.
    """
    try:
        schema = client.get_collection(COLLECTION_NAME).payload_schema or {}
        if PILLAR_ID_FIELD in schema:
            return

        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=PILLAR_ID_FIELD,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.info(f"Created keyword payload index on '{PILLAR_ID_FIELD}'")

    except Exception as e:
        # Non-fatal here: writes do not need the index, and search_similar()
        # raises with an actionable message if the index is genuinely absent.
        logger.error(f"Failed to ensure payload index on '{PILLAR_ID_FIELD}': {e}")


def ensure_collections() -> None:
    """
    Ensure the nlp_pillars collection exists with proper configuration.

    Creates the collection with cosine distance and correct vector size if it doesn't exist,
    and ensures the `pillar_id` payload index every filtered read depends on.
    """
    client = get_client()
    if client is None:
        logger.warning("Qdrant client not available. Cannot ensure collections.")
        return

    logger.info(f"Ensuring collection '{COLLECTION_NAME}' exists")

    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        if COLLECTION_NAME in collection_names:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists")
            # Still check the index: collections created before this existed
            # (the cloud one has 170 points and no payload index at all) would
            # otherwise never get one.
            _ensure_payload_indexes(client)
            return

        # Get vector size
        vector_size = _get_vector_size()

        # Create collection
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )

        logger.info(f"Created collection '{COLLECTION_NAME}' with vector size {vector_size} and cosine distance")
        _ensure_payload_indexes(client)

    except Exception as e:
        # Handle collection already exists error gracefully
        if "already exists" in str(e).lower():
            logger.info(f"Collection '{COLLECTION_NAME}' already exists (caught in exception)")
            return
        logger.error(f"Failed to ensure collection '{COLLECTION_NAME}': {e}")
        # Don't raise - let the system continue without vector storage if needed


def upsert_text(
    pillar_id: str,
    paper_id: str,
    full_text: str,
    chunk_size: int = 250,
    overlap: int = 25
) -> int:
    """
    Chunk text, embed chunks, and upsert to vector store with pillar isolation.

    Args:
        pillar_id: Target pillar for namespace isolation
        paper_id: Paper identifier
        full_text: Text content to process
        chunk_size: Maximum TOKENS per chunk, counted with the same encoding
            EMBEDDING_MODEL uses (~250 tokens is ~1000 characters of prose).
        overlap: TOKENS of overlap between consecutive chunks

    Returns:
        Number of chunks successfully upserted
    """
    client = get_client()
    if client is None:
        logger.warning("Qdrant client not available. Cannot upsert text.")
        return 0

    if not full_text or not full_text.strip():
        logger.warning("Empty text provided for upsert")
        return 0

    logger.info(f"Upserting text for paper {paper_id} in pillar {pillar_id}")

    try:
        # Chunk the text
        chunks = chunk_text(full_text, chunk_size=chunk_size, chunk_overlap=overlap)

        if not chunks:
            logger.warning("No chunks generated from text")
            return 0

        # Prepare points for upsert
        points = []
        successful_embeds = 0

        for idx, chunk in enumerate(chunks):
            try:
                # Generate embedding
                vector = _embed(chunk)

                # Create deterministic UUID from hash
                id_string = f"{pillar_id}|{paper_id}|{idx}"
                hash_bytes = hashlib.sha1(id_string.encode()).digest()[:16]  # Take first 16 bytes
                point_id = str(uuid.UUID(bytes=hash_bytes))

                # Create point with payload
                point = models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "pillar_id": pillar_id,
                        "paper_id": paper_id,
                        "chunk_index": idx,
                        "len": len(chunk)
                    }
                )

                points.append(point)
                successful_embeds += 1

            except Exception as e:
                logger.warning(f"Failed to embed chunk {idx} for paper {paper_id}: {e}")
                continue

        if not points:
            logger.warning("No chunks could be embedded")
            return 0

        # Upsert points
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        logger.info(f"Successfully upserted {successful_embeds} chunks for paper {paper_id} in pillar {pillar_id}")
        return successful_embeds

    except Exception as e:
        logger.error(f"Failed to upsert text for paper {paper_id} in pillar {pillar_id}: {e}")
        return 0


def search_similar(pillar_id: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar text chunks within a pillar namespace.
    
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
            (e.g. the `pillar_id` payload index is missing under strict mode).
            Both mean this module disagrees with its library or its server —
            not a search that found nothing — so neither may be reported as an
            empty result set. See the comments below.
    """
    client = get_client()
    if client is None:
        logger.warning("Qdrant client not available. Cannot search similar text.")
        return []

    if not query_text or not query_text.strip():
        logger.warning("Empty query text provided")
        return []

    logger.info(f"Searching for similar text in pillar {pillar_id} with top_k={top_k}")

    # Embedding is a network call to OpenAI: a genuine runtime failure, and
    # degrading to "no vector candidates" is the right answer for it.
    try:
        query_vector = _embed(query_text.strip())
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
    paper_scores = {}
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


def set_openai_client(client: Optional[OpenAI]) -> None:
    """Set the OpenAI client singleton (for testing)."""
    global _openai_client
    _openai_client = client


def reset_vector_size() -> None:
    """Reset cached vector size (for testing)."""
    global _vector_size
    _vector_size = None


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Example operations would go here for testing
    print("Vector store module loaded successfully")
