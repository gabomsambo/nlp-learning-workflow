"""
Semantic Scholar Tool - Search for papers and get citations with rate limiting.
"""

import time
import logging
from typing import List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..schemas import PaperRef
from ..config import get_settings


logger = logging.getLogger(__name__)


class SemanticScholarTool:
    """Tool for searching Semantic Scholar and fetching citation data with rate limiting."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Semantic Scholar tool with rate limiting.

        Args:
            api_key: Optional API key for higher rate limits
        """
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.last_request_time = 0
        # Rate limit: 100 requests per 5 minutes = 1 per 3 seconds
        self.rate_limit_seconds = 3.0

        # Set up headers
        self.headers = {"Content-Type": "application/json"}

        # Use provided API key or get from settings
        if api_key:
            self.headers["x-api-key"] = api_key
        else:
            settings = get_settings()
            s2_key = getattr(settings, 'semantic_scholar_api_key', None)
            if s2_key:
                self.headers["x-api-key"] = s2_key

        # HTTP client
        self.client = httpx.Client(timeout=30.0)

        logger.info("SemanticScholarTool initialized")

    def _enforce_rate_limit(self):
        """Ensure we respect the rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_seconds:
            sleep_time = self.rate_limit_seconds - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _map_paper_id(self, paper_id: str) -> str:
        """
        Map our paper ID to Semantic Scholar format.

        Args:
            paper_id: Paper ID in our format (e.g., "2301.00001" for arXiv)

        Returns:
            Semantic Scholar paper ID format
        """
        # If it looks like an arXiv ID (numbers with dots)
        if paper_id and '.' in paper_id and paper_id.replace('.', '').replace('v', '').isdigit():
            return f"arXiv:{paper_id.split('v')[0]}"

        # If it looks like a DOI
        if paper_id and paper_id.startswith('10.'):
            return f"DOI:{paper_id}"

        # Otherwise return as-is (might be S2 corpus ID)
        return paper_id

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, ConnectionError, TimeoutError))
    )
    def search(self, query: str, limit: int = 10, year: Optional[str] = None) -> List[PaperRef]:
        """
        Search for papers by keyword.

        Args:
            query: Search query string
            limit: Maximum number of results
            year: Optional year filter (e.g., "2023-2024" or "2024")

        Returns:
            List of PaperRef objects
        """
        try:
            self._enforce_rate_limit()

            params = {
                "query": query,
                "fields": "paperId,externalIds,title,authors,year,citationCount,abstract,influentialCitationCount,openAccessPdf",
                "limit": min(limit, 100)  # S2 max is 100
            }

            if year:
                params["year"] = year

            logger.info(f"Searching Semantic Scholar for: {query}")

            response = self.client.get(
                f"{self.base_url}/paper/search",
                params=params,
                headers=self.headers
            )

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited. Retry after {retry_after}s")
                time.sleep(retry_after)
                raise httpx.HTTPError(f"Rate limited, retry after {retry_after}s")

            response.raise_for_status()
            data = response.json()

            papers = []
            for paper_data in data.get("data", []):
                try:
                    paper_ref = self._convert_to_paper_ref(paper_data)
                    papers.append(paper_ref)
                except Exception as e:
                    logger.warning(f"Failed to convert S2 result: {e}")
                    continue

            logger.info(f"Found {len(papers)} papers from Semantic Scholar")
            return papers

        except httpx.HTTPError as e:
            logger.error(f"Semantic Scholar API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in S2 search: {e}")
            raise ConnectionError(f"Semantic Scholar search failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, ConnectionError, TimeoutError))
    )
    def get_citations(self, paper_id: str, limit: int = 50) -> List[tuple[PaperRef, bool]]:
        """
        Get papers that cite this paper (incoming citations).

        Args:
            paper_id: Paper ID to get citations for
            limit: Maximum number of citations to return

        Returns:
            List of (PaperRef, is_influential) tuples
        """
        try:
            self._enforce_rate_limit()

            s2_id = self._map_paper_id(paper_id)

            response = self.client.get(
                f"{self.base_url}/paper/{s2_id}/citations",
                params={
                    "fields": "paperId,externalIds,title,authors,year,citationCount,abstract",
                    "limit": min(limit, 100)
                },
                headers=self.headers
            )

            if response.status_code == 404:
                logger.warning(f"Paper not found in Semantic Scholar: {paper_id}")
                return []

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                time.sleep(retry_after)
                raise httpx.HTTPError("Rate limited")

            response.raise_for_status()
            data = response.json()

            citations = []
            for cite_data in data.get("data", []):
                try:
                    citing_paper = cite_data.get("citingPaper", {})
                    if citing_paper and citing_paper.get("paperId"):
                        paper_ref = self._convert_to_paper_ref(citing_paper)
                        is_influential = cite_data.get("isInfluential", False)
                        citations.append((paper_ref, is_influential))
                except Exception as e:
                    logger.warning(f"Failed to convert citation: {e}")
                    continue

            logger.info(f"Found {len(citations)} citations for paper {paper_id}")
            return citations

        except httpx.HTTPError as e:
            if "404" not in str(e):
                logger.error(f"Error fetching citations for {paper_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching citations: {e}")
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, ConnectionError, TimeoutError))
    )
    def get_references(self, paper_id: str, limit: int = 50) -> List[tuple[PaperRef, bool]]:
        """
        Get papers that this paper cites (outgoing references).

        Args:
            paper_id: Paper ID to get references for
            limit: Maximum number of references to return

        Returns:
            List of (PaperRef, is_influential) tuples
        """
        try:
            self._enforce_rate_limit()

            s2_id = self._map_paper_id(paper_id)

            response = self.client.get(
                f"{self.base_url}/paper/{s2_id}/references",
                params={
                    "fields": "paperId,externalIds,title,authors,year,citationCount,abstract",
                    "limit": min(limit, 100)
                },
                headers=self.headers
            )

            if response.status_code == 404:
                logger.warning(f"Paper not found in Semantic Scholar: {paper_id}")
                return []

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                time.sleep(retry_after)
                raise httpx.HTTPError("Rate limited")

            response.raise_for_status()
            data = response.json()

            references = []
            for ref_data in data.get("data", []):
                try:
                    cited_paper = ref_data.get("citedPaper", {})
                    if cited_paper and cited_paper.get("paperId"):
                        paper_ref = self._convert_to_paper_ref(cited_paper)
                        is_influential = ref_data.get("isInfluential", False)
                        references.append((paper_ref, is_influential))
                except Exception as e:
                    logger.warning(f"Failed to convert reference: {e}")
                    continue

            logger.info(f"Found {len(references)} references for paper {paper_id}")
            return references

        except httpx.HTTPError as e:
            if "404" not in str(e):
                logger.error(f"Error fetching references for {paper_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching references: {e}")
            return []

    def get_paper(self, paper_id: str) -> Optional[PaperRef]:
        """
        Get a single paper by ID.

        Args:
            paper_id: Paper ID

        Returns:
            PaperRef or None if not found
        """
        try:
            self._enforce_rate_limit()

            s2_id = self._map_paper_id(paper_id)

            response = self.client.get(
                f"{self.base_url}/paper/{s2_id}",
                params={
                    "fields": "paperId,externalIds,title,authors,year,citationCount,abstract,influentialCitationCount,openAccessPdf"
                },
                headers=self.headers
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            return self._convert_to_paper_ref(data)

        except Exception as e:
            logger.error(f"Error fetching paper {paper_id}: {e}")
            return None

    def _convert_to_paper_ref(self, s2_paper: dict) -> PaperRef:
        """
        Convert Semantic Scholar paper data to PaperRef.

        Args:
            s2_paper: Paper data from S2 API

        Returns:
            PaperRef object
        """
        # Extract best paper ID (prefer arXiv, then DOI, then S2 corpus ID)
        external_ids = s2_paper.get("externalIds", {}) or {}

        paper_id = None
        if external_ids.get("ArXiv"):
            paper_id = external_ids["ArXiv"]
        elif external_ids.get("DOI"):
            paper_id = external_ids["DOI"]
        else:
            paper_id = s2_paper.get("paperId", "unknown")

        # Extract authors
        authors = []
        for author in s2_paper.get("authors", []):
            if author and author.get("name"):
                authors.append(author["name"])

        # Extract PDF URL
        pdf_url = None
        open_access = s2_paper.get("openAccessPdf")
        if open_access and isinstance(open_access, dict):
            pdf_url = open_access.get("url")
        elif external_ids.get("ArXiv"):
            # Construct arXiv PDF URL
            arxiv_id = external_ids["ArXiv"]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        return PaperRef(
            id=paper_id,
            title=s2_paper.get("title", "Unknown Title"),
            authors=authors,
            year=s2_paper.get("year"),
            venue=None,  # S2 doesn't return venue in basic fields
            url_pdf=pdf_url,
            abstract=s2_paper.get("abstract"),
            citation_count=s2_paper.get("citationCount", 0)
        )

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tool = SemanticScholarTool()

    # Example search
    papers = tool.search("transformer attention mechanism", limit=5)

    for paper in papers:
        print(f"Title: {paper.title}")
        print(f"Authors: {', '.join(paper.authors[:3])}")
        print(f"Year: {paper.year}")
        print(f"Citations: {paper.citation_count}")
        print("-" * 50)
