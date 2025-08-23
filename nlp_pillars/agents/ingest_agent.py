"""
Ingest Agent - Processes PDF papers into structured ParsedPaper objects.

Follows Atomic Agents v2.0 patterns for robust PDF ingestion pipeline.
"""

import logging
from typing import Optional
import instructor
from openai import OpenAI

# TODO: Replace with proper atomic_agents imports when v2.0+ is available
# from atomic_agents import AtomicAgent, AgentConfig
# from atomic_agents.context import SystemPromptGenerator, ChatHistory

# Using the temporary mock classes from discovery_agent
from .discovery_agent import AtomicAgent, AgentConfig, SystemPromptGenerator, ChatHistory

from ..schemas import PaperRef, ParsedPaper
from ..tools.pdf_loader import download_pdf, extract_text, chunk_text, PDFDownloadError, PDFParseError
from ..config import get_settings

logger = logging.getLogger(__name__)


class IngestError(Exception):
    """Raised when ingestion fails."""
    pass


class IngestAgent:
    """Agent that ingests PDF papers and converts them to ParsedPaper objects."""
    
    def __init__(self, model: Optional[str] = None, cache_dir: str = ".cache/pdfs"):
        """
        Initialize the Ingest Agent.
        
        Args:
            model: Optional model name override. Defaults to config setting.
            cache_dir: Directory for PDF caching
        """
        settings = get_settings()
        self.model = model or settings.default_model
        self.cache_dir = cache_dir
        
        # Note: For this agent, we don't actually need LLM interaction
        # since we're just processing PDFs mechanically. But we maintain
        # the Atomic Agents pattern for consistency.
        
        # Create the system prompt generator (minimal since no LLM needed)
        self.system_prompt = SystemPromptGenerator(
            background=[
                "You are a PDF processing assistant.",
                "You help convert research papers from PDF format into structured data.",
                "You focus on accurate text extraction and proper chunking."
            ],
            steps=[
                "Process the PDF file to extract text content",
                "Validate the extraction quality",
                "Create appropriate text chunks for downstream processing",
                "Return structured ParsedPaper object"
            ],
            output_instructions=[
                "Ensure text extraction is complete and accurate",
                "Create meaningful chunks that preserve context",
                "Handle errors gracefully with clear error messages",
                "Return valid ParsedPaper schema"
            ]
        )
        
        # Initialize the OpenAI client with instructor (though not used for this agent)
        client = instructor.from_openai(OpenAI(api_key=settings.openai_api_key))
        
        # Create the Atomic Agent (maintaining pattern even though not used)
        self.agent = AtomicAgent[PaperRef, ParsedPaper](
            config=AgentConfig(
                client=client,
                model=self.model,
                system_prompt_generator=self.system_prompt,
                history=ChatHistory(),
            )
        )
    
    def ingest(self, paper_ref: PaperRef, *, pdf_url: Optional[str] = None) -> ParsedPaper:
        """
        Ingest a PDF paper and convert it to a ParsedPaper object.
        
        Args:
            paper_ref: Paper metadata
            pdf_url: Optional PDF URL override. If None, uses paper_ref.url_pdf
            
        Returns:
            ParsedPaper object with extracted text and chunks
            
        Raises:
            IngestError: If ingestion fails at any stage
        """
        logger.info(f"Starting ingestion for paper: {paper_ref.title}")
        
        try:
            # Determine PDF URL
            url = pdf_url or paper_ref.url_pdf
            if not url:
                raise IngestError(
                    f"No PDF URL available for paper '{paper_ref.title}'. "
                    f"Provide pdf_url parameter or ensure paper_ref.url_pdf is set."
                )
            
            # Step 1: Download PDF
            try:
                pdf_path = download_pdf(url, self.cache_dir)
                logger.info(f"Downloaded PDF to: {pdf_path}")
            except PDFDownloadError as e:
                raise IngestError(f"Failed to download PDF from {url}: {e}")
            
            # Step 2: Extract text
            try:
                full_text = extract_text(pdf_path)
                if not full_text.strip():
                    raise IngestError(f"No text extracted from PDF: {pdf_path}")
                logger.info(f"Extracted {len(full_text)} characters of text")
            except PDFParseError as e:
                raise IngestError(f"Failed to extract text from {pdf_path}: {e}")
            
            # Step 3: Create chunks
            try:
                chunks = chunk_text(full_text)
                if not chunks:
                    raise IngestError(f"No chunks created from text (length: {len(full_text)})")
                logger.info(f"Created {len(chunks)} text chunks")
            except Exception as e:
                raise IngestError(f"Failed to chunk text: {e}")
            
            # Step 4: Create ParsedPaper object
            parsed_paper = ParsedPaper(
                paper_ref=paper_ref,
                full_text=full_text,
                chunks=chunks,
                figures_count=0,  # Placeholder - could be enhanced later
                tables_count=0,   # Placeholder - could be enhanced later  
                references=[]     # Placeholder - could be enhanced later
            )
            
            logger.info(f"Successfully ingested paper: {paper_ref.title}")
            return parsed_paper
            
        except IngestError:
            # Re-raise our typed errors as-is
            raise
        except Exception as e:
            # Wrap any unexpected errors in our typed exception
            raise IngestError(f"Unexpected error during ingestion of '{paper_ref.title}': {e}")
    
    def ingest_batch(self, papers: list[tuple[PaperRef, Optional[str]]]) -> list[ParsedPaper]:
        """
        Ingest multiple papers in batch.
        
        Args:
            papers: List of (paper_ref, pdf_url) tuples
            
        Returns:
            List of successfully ingested ParsedPaper objects
            
        Note:
            Failures are logged but don't stop batch processing.
            Check logs for individual failures.
        """
        results = []
        
        for i, (paper_ref, pdf_url) in enumerate(papers, 1):
            try:
                logger.info(f"Processing paper {i}/{len(papers)}: {paper_ref.title}")
                parsed_paper = self.ingest(paper_ref, pdf_url=pdf_url)
                results.append(parsed_paper)
            except IngestError as e:
                logger.error(f"Failed to ingest paper {i}/{len(papers)} '{paper_ref.title}': {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error ingesting paper {i}/{len(papers)} '{paper_ref.title}': {e}")
                continue
        
        logger.info(f"Batch ingestion complete: {len(results)}/{len(papers)} successful")
        return results
    
    def validate_pdf_url(self, url: str) -> bool:
        """
        Validate that a PDF URL is accessible and appears to contain a PDF.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL appears valid, False otherwise
        """
        try:
            # Try to download (will use cache if already downloaded)
            pdf_path = download_pdf(url, self.cache_dir)
            return True
        except PDFDownloadError:
            return False
        except Exception:
            return False
    
    def get_cache_stats(self) -> dict:
        """
        Get statistics about the PDF cache.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            from pathlib import Path
            cache_path = Path(self.cache_dir)
            
            if not cache_path.exists():
                return {"cache_dir": str(cache_path), "exists": False}
            
            pdf_files = list(cache_path.glob("*.pdf"))
            total_size = sum(f.stat().st_size for f in pdf_files)
            
            return {
                "cache_dir": str(cache_path),
                "exists": True,
                "file_count": len(pdf_files),
                "total_size_mb": round(total_size / 1024 / 1024, 2)
            }
        except Exception as e:
            return {"cache_dir": self.cache_dir, "error": str(e)}
    
    def clear_cache(self) -> int:
        """
        Clear the PDF cache directory.
        
        Returns:
            Number of files removed
        """
        try:
            from pathlib import Path
            cache_path = Path(self.cache_dir)
            
            if not cache_path.exists():
                return 0
            
            pdf_files = list(cache_path.glob("*.pdf")) + list(cache_path.glob("*.pdf.tmp"))
            removed_count = 0
            
            for pdf_file in pdf_files:
                try:
                    pdf_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {pdf_file}: {e}")
            
            logger.info(f"Cleared {removed_count} files from cache: {cache_path}")
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0
    
    def reset_context(self):
        """Reset the agent's conversation history."""
        self.agent.config.history.clear()


# Utility functions for common ingestion tasks
def ingest_paper_from_url(url: str, paper_ref: PaperRef) -> ParsedPaper:
    """
    Convenience function to ingest a single paper from URL.
    
    Args:
        url: PDF URL
        paper_ref: Paper metadata
        
    Returns:
        ParsedPaper object
        
    Raises:
        IngestError: If ingestion fails
    """
    agent = IngestAgent()
    return agent.ingest(paper_ref, pdf_url=url)


def ingest_paper_from_ref(paper_ref: PaperRef) -> ParsedPaper:
    """
    Convenience function to ingest a paper using its url_pdf field.
    
    Args:
        paper_ref: Paper metadata with url_pdf set
        
    Returns:
        ParsedPaper object
        
    Raises:
        IngestError: If ingestion fails
    """
    agent = IngestAgent()
    return agent.ingest(paper_ref)


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example paper reference
    sample_paper = PaperRef(
        id="2301.12345",
        title="Sample Research Paper",
        authors=["John Doe", "Jane Smith"],
        venue="Test Conference",
        year=2023,
        url_pdf="https://arxiv.org/pdf/2301.12345.pdf"
    )
    
    # Create agent and ingest
    agent = IngestAgent()
    
    try:
        parsed_paper = agent.ingest(sample_paper)
        print(f"Successfully ingested paper with {len(parsed_paper.chunks)} chunks")
        print(f"Full text length: {len(parsed_paper.full_text)} characters")
        
        # Show cache stats
        stats = agent.get_cache_stats()
        print(f"Cache stats: {stats}")
        
    except IngestError as e:
        print(f"Ingestion failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
