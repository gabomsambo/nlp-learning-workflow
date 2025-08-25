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
from openai import OpenAI

from .schemas import PillarID
from .tools.pdf_loader import chunk_text

logger = logging.getLogger(__name__)

# Module-level singletons
_client: Optional[QdrantClient] = None
_openai_client: Optional[OpenAI] = None
_vector_size: Optional[int] = None

COLLECTION_NAME = "nlp_pillars"


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


def ensure_collections() -> None:
    """
    Ensure the nlp_pillars collection exists with proper configuration.
    
    Creates the collection with cosine distance and correct vector size if it doesn't exist.
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
        
    except Exception as e:
        # Handle collection already exists error gracefully
        if "already exists" in str(e).lower():
            logger.info(f"Collection '{COLLECTION_NAME}' already exists (caught in exception)")
            return
        logger.error(f"Failed to ensure collection '{COLLECTION_NAME}': {e}")
        # Don't raise - let the system continue without vector storage if needed


def upsert_text(
    pillar_id: PillarID, 
    paper_id: str, 
    full_text: str, 
    chunk_size: int = 1000, 
    overlap: int = 100
) -> int:
    """
    Chunk text, embed chunks, and upsert to vector store with pillar isolation.
    
    Args:
        pillar_id: Target pillar for namespace isolation
        paper_id: Paper identifier
        full_text: Text content to process
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
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
    
    logger.info(f"Upserting text for paper {paper_id} in pillar {pillar_id.value}")
    
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
                id_string = f"{pillar_id.value}|{paper_id}|{idx}"
                hash_bytes = hashlib.sha1(id_string.encode()).digest()[:16]  # Take first 16 bytes
                point_id = str(uuid.UUID(bytes=hash_bytes))
                
                # Create point with payload
                point = models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "pillar_id": pillar_id.value,
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
        
        logger.info(f"Successfully upserted {successful_embeds} chunks for paper {paper_id} in pillar {pillar_id.value}")
        return successful_embeds
        
    except Exception as e:
        logger.error(f"Failed to upsert text for paper {paper_id} in pillar {pillar_id.value}: {e}")
        return 0


def search_similar(pillar_id: PillarID, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar text chunks within a pillar namespace.
    
    Args:
        pillar_id: Pillar to search within
        query_text: Text to find similar content for
        top_k: Maximum number of results to return
        
    Returns:
        List of {"paper_id": str, "score": float} dictionaries, 
        deduplicated by paper_id (keeping highest score) and limited to top_k
    """
    client = get_client()
    if client is None:
        logger.warning("Qdrant client not available. Cannot search similar text.")
        return []
    
    if not query_text or not query_text.strip():
        logger.warning("Empty query text provided")
        return []
    
    logger.info(f"Searching for similar text in pillar {pillar_id.value} with top_k={top_k}")
    
    try:
        # Generate query embedding
        query_vector = _embed(query_text.strip())
        
        # Search with pillar filter
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="pillar_id",
                        match=models.MatchValue(value=pillar_id.value)
                    )
                ]
            ),
            limit=top_k * 3,  # Get more results for deduplication
            with_payload=True
        )
        
        # Deduplicate by paper_id, keeping highest score
        paper_scores = {}
        for hit in search_result:
            paper_id = hit.payload.get("paper_id")
            score = hit.score
            
            if paper_id and (paper_id not in paper_scores or score > paper_scores[paper_id]):
                paper_scores[paper_id] = score
        
        # Sort by score and limit to top_k
        results = [
            {"paper_id": paper_id, "score": score}
            for paper_id, score in sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)
        ][:top_k]
        
        logger.info(f"Found {len(results)} similar papers in pillar {pillar_id.value}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to search similar text in pillar {pillar_id.value}: {e}")
        return []


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
