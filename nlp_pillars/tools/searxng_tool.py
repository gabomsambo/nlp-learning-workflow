"""
SearXNG Search Tool - Search for research papers using SearXNG metasearch engine.
"""

import asyncio
import logging
import re
from typing import List, Optional, Dict, Any
from urllib.parse import urlencode, urlparse
import time

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..schemas import PaperRef, SearchQuery
from ..config import get_settings


logger = logging.getLogger(__name__)


class SearXNGTool:
    """Tool for searching papers using SearXNG metasearch engine."""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the SearXNG tool.
        
        Args:
            base_url: Optional override for SearXNG URL. If None, uses config setting.
        """
        settings = get_settings()
        self.base_url = base_url or settings.searxng_url
        self.last_request_time = 0
        self.rate_limit_seconds = 1  # Conservative rate limiting for SearXNG
        
        # HTTP client configuration
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "NLP-Learning-Workflow/1.0",
                "Accept": "application/json",
            }
        )
        
        # Academic search engines to prioritize in SearXNG (simplified)
        self.academic_engines = [
            "arxiv",
            "duckduckgo"
        ]
    
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
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((httpx.HTTPError, ConnectionError, TimeoutError))
    )
    async def search_async(self, query: SearchQuery) -> List[PaperRef]:
        """
        Search for papers using SearXNG (async version).
        
        Args:
            query: SearchQuery containing the search parameters
            
        Returns:
            List of PaperRef objects matching the search criteria
            
        Raises:
            httpx.HTTPError: If SearXNG returns an HTTP error
            ConnectionError: If network connection fails
        """
        try:
            # Enforce rate limiting
            self._enforce_rate_limit()
            
            # Build search URL
            search_url = self._build_search_url(query)
            
            logger.info(f"Searching SearXNG: {search_url}")
            
            # Perform search
            response = await self.client.get(search_url)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            papers = self._parse_searxng_results(data, query.max_results)
            
            logger.info(f"Found {len(papers)} papers from SearXNG")
            return papers
            
        except httpx.HTTPError as e:
            logger.error(f"SearXNG HTTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in SearXNG search: {e}")
            raise ConnectionError(f"SearXNG search failed: {e}")
    
    def search(self, query: SearchQuery) -> List[PaperRef]:
        """
        Search for papers using SearXNG (sync wrapper).
        
        Args:
            query: SearchQuery containing the search parameters
            
        Returns:
            List of PaperRef objects matching the search criteria
        """
        # Use synchronous HTTP client to avoid async issues
        try:
            self._enforce_rate_limit()
            
            # Build search parameters for POST request
            search_params = self._build_search_params(query)
            search_url = f"{self.base_url.rstrip('/')}/search"
            
            logger.info(f"Searching SearXNG: {search_url} with params: {search_params}")
            
            # Use synchronous HTTP client with POST request
            with httpx.Client(timeout=30.0, headers={
                "User-Agent": "NLP-Learning-Workflow/1.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Content-Type": "application/x-www-form-urlencoded",
            }) as sync_client:
                response = sync_client.post(search_url, data=search_params)
                response.raise_for_status()
                
                # Parse HTML results since JSON API seems restricted
                return self._parse_html_results(response.text, query.max_results)
                
        except Exception as e:
            logger.error(f"SearXNG search failed: {e}")
            return []
    
    def _build_search_params(self, query: SearchQuery) -> Dict[str, str]:
        """
        Build SearXNG search parameters for POST request.
        
        Args:
            query: SearchQuery object with search parameters
            
        Returns:
            Dictionary of search parameters for POST request
        """
        # Base search parameters following SearXNG form structure
        params = {
            "q": query.query,
            "category_general": "1",  # Enable general search category
            "language": "en",         # Default language
            "time_range": "",         # No time restriction by default
            "safesearch": "0",        # No content filtering
            "theme": "simple"         # Theme (required by SearXNG)
        }
        
        # Use specific engine if requested, otherwise use all available
        if query.filters and "engines" in query.filters:
            engine_name = query.filters["engines"]
            # Enable specific engine categories based on the engine
            if engine_name == "arxiv":
                params["category_science"] = "1"
            elif engine_name == "duckduckgo":
                params["category_general"] = "1"
        else:
            # Enable science category for academic searches
            params["category_science"] = "1"
        
        # Add additional filters from query
        if query.filters:
            # Time filter
            if "time_range" in query.filters:
                params["time_range"] = query.filters["time_range"]
            
            # Language filter
            if "language" in query.filters:
                params["language"] = query.filters["language"]
        
        return params
        
    def _build_search_url(self, query: SearchQuery) -> str:
        """
        Build SearXNG search URL with parameters (legacy method for async).
        
        Args:
            query: SearchQuery object with search parameters
            
        Returns:
            Complete SearXNG search URL
        """
        # Base search parameters - simplified to avoid URL length issues
        params = {
            "q": query.query,
            "format": "json",
            "safesearch": "0",  # No content filtering
            "pageno": "1"  # First page
        }
        
        # Use only one engine at a time to avoid conflicts
        if query.filters and "engines" in query.filters:
            params["engines"] = query.filters["engines"]
        else:
            # Default to just arxiv for academic searches
            params["engines"] = "arxiv"
        
        # Add additional filters from query
        if query.filters:
            # Time filter
            if "time_range" in query.filters:
                params["time_range"] = query.filters["time_range"]
            
            # Language filter
            if "language" in query.filters:
                params["language"] = query.filters["language"]
        
        # Build URL
        search_url = f"{self.base_url.rstrip('/')}/search?{urlencode(params)}"
        return search_url
    
    def _parse_searxng_results(self, data: Dict[str, Any], max_results: int) -> List[PaperRef]:
        """
        Parse SearXNG JSON response into PaperRef objects.
        
        Args:
            data: JSON response from SearXNG
            max_results: Maximum number of results to return
            
        Returns:
            List of PaperRef objects
        """
        papers = []
        results = data.get("results", [])
        
        for i, result in enumerate(results[:max_results]):
            try:
                paper_ref = self._convert_result_to_paper_ref(result)
                if paper_ref:
                    papers.append(paper_ref)
            except Exception as e:
                logger.warning(f"Failed to convert SearXNG result {i}: {e}")
                continue
        
        return papers
    
    def _parse_html_results(self, html_content: str, max_results: int) -> List[PaperRef]:
        """
        Parse SearXNG HTML response into PaperRef objects.
        
        Args:
            html_content: HTML response from SearXNG
            max_results: Maximum number of results to return
            
        Returns:
            List of PaperRef objects
        """
        papers = []
        
        try:
            # Simple HTML parsing to extract search results
            # Look for result articles
            import re
            
            # Extract all result articles
            article_pattern = r'<article class="result[^"]*"[^>]*>(.*?)</article>'
            articles = re.findall(article_pattern, html_content, re.DOTALL)
            
            for i, article_html in enumerate(articles[:max_results]):
                try:
                    paper_ref = self._extract_paper_from_html(article_html)
                    if paper_ref:
                        papers.append(paper_ref)
                except Exception as e:
                    logger.warning(f"Failed to parse HTML result {i}: {e}")
                    continue
            
            logger.info(f"Parsed {len(papers)} papers from HTML results")
            
        except Exception as e:
            logger.error(f"Failed to parse HTML results: {e}")
        
        return papers
    
    def _extract_paper_from_html(self, article_html: str) -> Optional[PaperRef]:
        """
        Extract paper information from a single HTML article.
        
        Args:
            article_html: HTML content of a single search result
            
        Returns:
            PaperRef object or None if extraction fails
        """
        try:
            import re
            
            # Extract title and URL
            title_pattern = r'<h3><a href="([^"]+)"[^>]*>(.+?)</a></h3>'
            title_match = re.search(title_pattern, article_html, re.DOTALL)
            
            if not title_match:
                return None
            
            url = title_match.group(1)
            title_html = title_match.group(2)
            
            # Clean title (remove HTML tags)
            title = re.sub(r'<[^>]+>', '', title_html).strip()
            
            if not title or not url:
                return None
            
            # Extract content/abstract
            content_pattern = r'<p class="content">\s*(.*?)\s*</p>'
            content_match = re.search(content_pattern, article_html, re.DOTALL)
            content = ""
            if content_match:
                content = re.sub(r'<[^>]+>', '', content_match.group(1)).strip()
            
            # Generate paper ID
            paper_id = self._generate_paper_id(url, title)
            
            # Extract basic paper info
            authors = self._extract_authors(title, content)
            year = self._extract_year(title, content, url)
            venue = self._extract_venue_from_url(url)
            
            # Check if it's a PDF
            pdf_url = None
            if url.endswith('.pdf') or 'pdf' in url.lower():
                pdf_url = url
            
            # Use content as abstract if available
            abstract = content[:500] if content else None
            
            return PaperRef(
                id=paper_id,
                title=title,
                authors=authors,
                venue=venue,
                year=year,
                url_pdf=pdf_url,
                abstract=abstract,
                citation_count=None
            )
            
        except Exception as e:
            logger.warning(f"Error extracting paper from HTML: {e}")
            return None
    
    def _extract_venue_from_url(self, url: str) -> Optional[str]:
        """Extract venue/source from URL."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            
            venue_map = {
                "arxiv.org": "arXiv",
                "scholar.google.com": "Google Scholar",
                "pubmed.ncbi.nlm.nih.gov": "PubMed",
                "semanticscholar.org": "Semantic Scholar",
                "ieee.org": "IEEE",
                "acm.org": "ACM",
                "springer.com": "Springer",
                "nature.com": "Nature",
                "science.org": "Science"
            }
            
            for key, venue in venue_map.items():
                if key in domain:
                    return venue
            
            # Fallback to domain name
            return domain.replace('www.', '').title()
            
        except Exception:
            return "Unknown"
    
    def _convert_result_to_paper_ref(self, result: Dict[str, Any]) -> Optional[PaperRef]:
        """
        Convert a SearXNG result to a PaperRef object.
        
        Args:
            result: Single result from SearXNG JSON response
            
        Returns:
            PaperRef object or None if conversion fails
        """
        try:
            # Extract basic fields
            title = result.get("title", "").strip()
            url = result.get("url", "")
            content = result.get("content", "")
            
            if not title or not url:
                return None
            
            # Generate ID from URL or title
            paper_id = self._generate_paper_id(url, title)
            
            # Extract authors from content or title
            authors = self._extract_authors(title, content)
            
            # Extract year from title, content, or URL
            year = self._extract_year(title, content, url)
            
            # Determine if this is a PDF link
            pdf_url = None
            if url.endswith('.pdf') or 'pdf' in url.lower():
                pdf_url = url
            
            # Extract venue/source information
            venue = self._extract_venue(result, url)
            
            # Use content as abstract if available
            abstract = content[:500] if content else None  # Limit abstract length
            
            return PaperRef(
                id=paper_id,
                title=title,
                authors=authors,
                venue=venue,
                year=year,
                url_pdf=pdf_url,
                abstract=abstract,
                citation_count=None  # Not typically available from SearXNG
            )
            
        except Exception as e:
            logger.warning(f"Error converting SearXNG result: {e}")
            return None
    
    def _generate_paper_id(self, url: str, title: str) -> str:
        """Generate a unique ID for the paper."""
        # Try to extract arXiv ID from URL
        arxiv_match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', url)
        if arxiv_match:
            return arxiv_match.group(1)
        
        # Try to extract DOI from URL
        doi_match = re.search(r'doi\.org/(.+)$', url)
        if doi_match:
            return doi_match.group(1)
        
        # Fall back to URL hash
        return f"searxng_{hash(url) % 1000000:06d}"
    
    def _extract_authors(self, title: str, content: str) -> List[str]:
        """Extract author names from title or content."""
        authors = []
        
        # Look for author patterns in content
        text = f"{title} {content}".lower()
        
        # Common author patterns
        author_patterns = [
            r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)',
            r'authors?:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)',
            r'written\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)'
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                author_text = match.group(1)
                # Split by comma and clean up
                authors = [author.strip() for author in author_text.split(',')]
                break
        
        # If no authors found, return a placeholder
        if not authors:
            authors = ["Unknown Author"]
        
        return authors[:5]  # Limit to 5 authors
    
    def _extract_year(self, title: str, content: str, url: str) -> Optional[int]:
        """Extract publication year from various sources."""
        text = f"{title} {content} {url}"
        
        # Look for 4-digit years (1990-2030)
        year_matches = re.findall(r'\b(19[9]\d|20[0-3]\d)\b', text)
        
        if year_matches:
            # Return the most recent year found
            years = [int(year) for year in year_matches]
            return max(years)
        
        return None
    
    def _extract_venue(self, result: Dict[str, Any], url: str) -> Optional[str]:
        """Extract venue/publisher information."""
        # Try to get engine name from SearXNG
        engine = result.get("engine", "")
        
        # Map common domains to venue names
        domain_to_venue = {
            "arxiv.org": "arXiv",
            "pubmed.ncbi.nlm.nih.gov": "PubMed", 
            "scholar.google.com": "Google Scholar",
            "semanticscholar.org": "Semantic Scholar",
            "crossref.org": "CrossRef",
            "core.ac.uk": "CORE"
        }
        
        # Extract domain from URL
        try:
            domain = urlparse(url).netloc.lower()
            for key, venue in domain_to_venue.items():
                if key in domain:
                    return venue
        except Exception:
            pass
        
        # Fall back to engine name if available
        if engine:
            return engine.title()
        
        return "SearXNG"
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            # Only try to close if we have a client and no running event loop
            if hasattr(self, 'client') and self.client and not self.client.is_closed:
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_running():
                        asyncio.run(self.close())
                except:
                    # If we can't close gracefully, just set client to None
                    self.client = None
        except Exception:
            pass  # Ignore cleanup errors


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from ..schemas import PillarID
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    async def test_searxng():
        # Create tool
        searxng_tool = SearXNGTool()
        
        # Example search
        search_query = SearchQuery(
            pillar_id=PillarID.P2,
            query="transformer neural networks",
            filters={"time_range": "year"},
            max_results=5
        )
        
        try:
            papers = await searxng_tool.search_async(search_query)
            
            for paper in papers:
                print(f"Title: {paper.title}")
                print(f"Authors: {', '.join(paper.authors)}")
                print(f"Year: {paper.year}")
                print(f"Venue: {paper.venue}")
                print(f"ID: {paper.id}")
                print(f"PDF: {paper.url_pdf}")
                print("-" * 50)
                
        except Exception as e:
            print(f"Search failed: {e}")
        finally:
            await searxng_tool.close()
    
    # Run test
    asyncio.run(test_searxng())
