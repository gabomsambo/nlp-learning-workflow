"""Vector Search Tool - Wrapper around Qdrant for paper discovery.
"""

import logging
from typing import List

from .. import db, vectors
from ..schemas import DiscoveryCandidate, PaperRef

logger = logging.getLogger(__name__)


class VectorSearchTool:
    """Tool for searching similar papers using Qdrant vector database."""

    def __init__(self):
        """Initialize the vector search tool."""
        # Ensure Qdrant collection exists
        vectors.ensure_collections()
        logger.info("VectorSearchTool initialized")

    def search_similar_papers(
        self,
        pillar_id: str,
        query_text: str,
        top_k: int = 5
    ) -> List[DiscoveryCandidate]:
        """Search for similar papers using semantic similarity.

        Args:
        ----
            pillar_id: Pillar to search within (slug format, e.g., 'models-architectures')
            query_text: Text to find similar papers for
            top_k: Maximum number of results

        Returns:
        -------
            List of DiscoveryCandidate objects with vector search results

        """
        if not query_text or not query_text.strip():
            logger.warning("Empty query text provided")
            return []

        logger.info(f"Vector search for pillar {pillar_id}: '{query_text[:50]}...'")

        try:
            # Use the existing search_similar function from vectors module
            results = vectors.search_similar(pillar_id, query_text, top_k)

            if not results:
                logger.info("No similar papers found in vector database")
                return []

            # Convert results to DiscoveryCandidate objects
            candidates = []
            for result in results:
                paper_id = result.get("paper_id")
                score = result.get("score", 0.0)

                if not paper_id:
                    continue

                # Try to get paper details from database
                paper_ref = self._get_paper_ref(pillar_id, paper_id)

                if paper_ref:
                    candidate = DiscoveryCandidate(
                        paper=paper_ref,
                        source="vector",
                        relevance_score=min(score, 1.0),  # Normalize score
                        citation_count=paper_ref.citation_count or 0,
                        is_influential=False
                    )
                    candidates.append(candidate)

            logger.info(f"Found {len(candidates)} similar papers from vector search")
            return candidates

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def search_diverse_papers(
        self,
        pillar_id: str,
        query_text: str,
        existing_ids: List[str] = None,
        top_k: int = 5
    ) -> List[DiscoveryCandidate]:
        """Search for diverse papers different from existing ones.

        Uses vector search to find papers that are semantically similar to query
        but different from papers already in the existing_ids list.

        Args:
        ----
            pillar_id: Pillar to search within
            query_text: Text to find similar papers for
            existing_ids: Paper IDs to avoid (for diversity)
            top_k: Maximum number of results

        Returns:
        -------
            List of DiscoveryCandidate objects

        """
        # Get more results than needed for filtering
        all_candidates = self.search_similar_papers(
            pillar_id,
            query_text,
            top_k=top_k * 3
        )

        if not existing_ids:
            return all_candidates[:top_k]

        # Filter out existing papers
        existing_set = set(existing_ids)
        diverse_candidates = [
            c for c in all_candidates
            if c.paper.id not in existing_set
        ]

        return diverse_candidates[:top_k]

    def _get_paper_ref(self, pillar_id: str, paper_id: str) -> PaperRef:
        """Get paper reference from database or create minimal one.

        Args:
        ----
            pillar_id: Pillar ID
            paper_id: Paper ID

        Returns:
        -------
            PaperRef object

        """
        try:
            # Try to get from database
            papers = db.get_papers(pillar_id, limit=100)
            for paper in papers:
                if paper.id == paper_id:
                    return paper

            # If not found, try to get from notes
            notes = db.get_all_notes(pillar_id, limit=100)
            for note in notes:
                if note.paper_id == paper_id:
                    # Create minimal paper ref from note
                    return PaperRef(
                        id=paper_id,
                        title=f"Paper {paper_id}",
                        authors=[]
                    )

        except Exception as e:
            logger.warning(f"Could not get paper details for {paper_id}: {e}")

        # Return minimal paper ref
        return PaperRef(
            id=paper_id,
            title=f"Paper {paper_id}",
            authors=[]
        )


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tool = VectorSearchTool()

    # Example search
    candidates = tool.search_similar_papers(
        "models-architectures",
        "transformer attention mechanism deep learning",
        top_k=5
    )

    for candidate in candidates:
        print(f"Paper: {candidate.paper.id}")
        print(f"Score: {candidate.relevance_score:.3f}")
        print(f"Source: {candidate.source}")
        print("-" * 30)
