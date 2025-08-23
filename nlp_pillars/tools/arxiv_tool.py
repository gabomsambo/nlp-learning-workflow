"""
ArXiv Search Tool - Search for research papers on arXiv with rate limiting.
"""

import time
from typing import List, Optional
import logging
from urllib.parse import urlencode

import arxiv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..schemas import PaperRef, SearchQuery


logger = logging.getLogger(__name__)


class ArXivTool:
    """Tool for searching papers on arXiv with proper rate limiting and error handling."""
    
    def __init__(self):
        """Initialize the ArXiv tool with rate limiting."""
        self.last_request_time = 0
        self.rate_limit_seconds = 3  # ArXiv requests 1 request per 3 seconds
        
        # Configure arxiv client
        self.client = arxiv.Client(
            page_size=100,  # Maximum results per request
            delay_seconds=self.rate_limit_seconds,  # Built-in rate limiting
            num_retries=3  # Built-in retries
        )
    
    def _enforce_rate_limit(self):
        """Ensure we respect the rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_seconds:
            sleep_time = self.rate_limit_seconds - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((arxiv.ArxivError, ConnectionError, TimeoutError))
    )
    def search(self, query: SearchQuery) -> List[PaperRef]:
        """
        Search for papers on arXiv.
        
        Args:
            query: SearchQuery containing the search parameters
            
        Returns:
            List of PaperRef objects matching the search criteria
            
        Raises:
            arxiv.ArxivError: If arXiv API returns an error
            ConnectionError: If network connection fails
        """
        try:
            # Enforce rate limiting
            self._enforce_rate_limit()
            
            # Build arXiv search query
            search_query = self._build_arxiv_query(query.query, query.filters)
            
            logger.info(f"Searching arXiv for: {search_query}")
            
            # Perform search
            search = arxiv.Search(
                query=search_query,
                max_results=query.max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            for result in self.client.results(search):
                try:
                    paper_ref = self._convert_to_paper_ref(result)
                    papers.append(paper_ref)
                except Exception as e:
                    logger.warning(f"Failed to convert arXiv result: {e}")
                    continue
            
            logger.info(f"Found {len(papers)} papers from arXiv")
            return papers
            
        except arxiv.ArxivError as e:
            logger.error(f"ArXiv API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in arXiv search: {e}")
            raise ConnectionError(f"ArXiv search failed: {e}")
    
    def _build_arxiv_query(self, query: str, filters: dict) -> str:
        """
        Build an arXiv-specific query string from the search parameters.
        
        Args:
            query: Base search query
            filters: Additional filters to apply
            
        Returns:
            Formatted arXiv query string
        """
        # Start with the base query
        arxiv_query = query
        
        # Add category filters if specified
        if "categories" in filters:
            categories = filters["categories"]
            if isinstance(categories, list):
                category_query = " OR ".join([f"cat:{cat}" for cat in categories])
                arxiv_query = f"({arxiv_query}) AND ({category_query})"
            else:
                arxiv_query = f"({arxiv_query}) AND cat:{categories}"
        
        # Add author filter if specified
        if "author" in filters:
            author = filters["author"]
            arxiv_query = f"({arxiv_query}) AND au:{author}"
        
        # Add year filter if specified
        if "year" in filters:
            year = filters["year"]
            if isinstance(year, dict):
                if "min" in year:
                    arxiv_query = f"({arxiv_query}) AND submittedDate:[{year['min']}0101 TO 20991231]"
                if "max" in year:
                    arxiv_query = f"({arxiv_query}) AND submittedDate:[19900101 TO {year['max']}1231]"
            else:
                arxiv_query = f"({arxiv_query}) AND submittedDate:[{year}0101 TO {year}1231]"
        
        return arxiv_query
    
    def _convert_to_paper_ref(self, arxiv_result: arxiv.Result) -> PaperRef:
        """
        Convert an arXiv result to a PaperRef object.
        
        Args:
            arxiv_result: Result from arXiv API
            
        Returns:
            PaperRef object with extracted information
        """
        # Extract arXiv ID (remove version if present)
        arxiv_id = arxiv_result.entry_id.split("/")[-1]
        if "v" in arxiv_id:
            arxiv_id = arxiv_id.split("v")[0]
        
        # Extract authors
        authors = [str(author) for author in arxiv_result.authors]
        
        # Extract year from published date
        year = None
        if arxiv_result.published:
            year = arxiv_result.published.year
        
        # Build PDF URL
        pdf_url = None
        if arxiv_result.pdf_url:
            pdf_url = str(arxiv_result.pdf_url)
        
        # Extract venue/journal (usually not available in arXiv)
        venue = None
        if arxiv_result.journal_ref:
            venue = str(arxiv_result.journal_ref)
        
        # Extract primary category as a rough venue indicator
        if not venue and arxiv_result.primary_category:
            venue = f"arXiv:{arxiv_result.primary_category}"
        
        return PaperRef(
            id=arxiv_id,
            title=str(arxiv_result.title).strip(),
            authors=authors,
            venue=venue,
            year=year,
            url_pdf=pdf_url,
            abstract=str(arxiv_result.summary).strip() if arxiv_result.summary else None,
            citation_count=None  # Not available from arXiv API
        )
    
    def search_by_categories(self, categories: List[str], max_results: int = 50) -> List[PaperRef]:
        """
        Search for recent papers in specific arXiv categories.
        
        Args:
            categories: List of arXiv category codes (e.g., ['cs.CL', 'cs.LG'])
            max_results: Maximum number of results to return
            
        Returns:
            List of recent papers in the specified categories
        """
        # Build category query
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])
        
        # Create SearchQuery object
        search_query = SearchQuery(
            pillar_id="P1",  # Dummy pillar ID for direct category search
            query=category_query,
            filters={},
            max_results=max_results
        )
        
        return self.search(search_query)
    
    def search_by_author(self, author: str, max_results: int = 20) -> List[PaperRef]:
        """
        Search for papers by a specific author.
        
        Args:
            author: Author name to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of papers by the specified author
        """
        search_query = SearchQuery(
            pillar_id="P1",  # Dummy pillar ID for direct author search
            query=f"au:{author}",
            filters={},
            max_results=max_results
        )
        
        return self.search(search_query)


# Example usage and testing
if __name__ == "__main__":
    from ..schemas import PillarID
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create tool
    arxiv_tool = ArXivTool()
    
    # Example search
    search_query = SearchQuery(
        pillar_id=PillarID.P2,
        query="attention transformer",
        filters={"categories": ["cs.CL", "cs.LG"]},
        max_results=5
    )
    
    try:
        papers = arxiv_tool.search(search_query)
        
        for paper in papers:
            print(f"Title: {paper.title}")
            print(f"Authors: {', '.join(paper.authors)}")
            print(f"Year: {paper.year}")
            print(f"arXiv ID: {paper.id}")
            print(f"PDF: {paper.url_pdf}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Search failed: {e}")
