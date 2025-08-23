"""
PDF Loading and Processing Utilities

Provides robust PDF downloading, text extraction, and chunking capabilities
with fallback strategies and comprehensive error handling.
"""

import hashlib
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.pdfparser import PDFSyntaxError
except ImportError:
    pdfminer_extract_text = None
    PDFSyntaxError = Exception

logger = logging.getLogger(__name__)


# Custom exceptions
class PDFDownloadError(Exception):
    """Raised when PDF download fails."""
    pass


class PDFParseError(Exception):
    """Raised when PDF parsing/text extraction fails."""
    pass


def download_pdf(url: str, cache_dir: str = ".cache/pdfs") -> str:
    """
    Download a PDF from URL with caching and retries.
    
    Args:
        url: URL to download PDF from
        cache_dir: Directory to cache downloaded PDFs
        
    Returns:
        Local file path to the downloaded PDF
        
    Raises:
        PDFDownloadError: If download fails after retries
    """
    # Handle file:// URLs by short-circuiting
    if url.startswith("file://"):
        file_path = url[7:]  # Remove file:// prefix
        if os.path.exists(file_path):
            logger.info(f"Using local file: {file_path}")
            return file_path
        else:
            raise PDFDownloadError(f"Local file not found: {file_path}")
    
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Generate deterministic filename from URL
    url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
    final_filename = cache_path / f"{url_hash}.pdf"
    temp_filename = cache_path / f"{url_hash}.pdf.tmp"
    
    # Return cached file if it exists
    if final_filename.exists():
        logger.info(f"Using cached PDF: {final_filename}")
        return str(final_filename)
    
    logger.info(f"Downloading PDF from: {url}")
    start_time = time.time()
    
    try:
        # Download with retries
        content = _download_with_retries(url)
        
        # Validate content
        _validate_pdf_content(content, url)
        
        # Write atomically (temp file -> rename)
        with open(temp_filename, 'wb') as f:
            f.write(content)
        
        # Atomic rename
        temp_filename.rename(final_filename)
        
        duration = time.time() - start_time
        logger.info(f"Downloaded PDF in {duration:.2f}s: {final_filename}")
        
        return str(final_filename)
        
    except Exception as e:
        # Clean up temp file on failure
        if temp_filename.exists():
            temp_filename.unlink()
        
        if isinstance(e, PDFDownloadError):
            raise
        else:
            raise PDFDownloadError(f"Failed to download PDF from {url}: {e}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError))
)
def _download_with_retries(url: str) -> bytes:
    """Download PDF content with retry logic for rate limits and server errors."""
    max_size = 50 * 1024 * 1024  # 50 MB limit
    
    with httpx.Client(timeout=30.0) as client:
        # Use streaming to check size and content-type
        with client.stream('GET', url) as response:
            # Check for HTTP errors
            if response.status_code == 429:
                logger.warning(f"Rate limited (429) for {url}, retrying...")
                raise httpx.HTTPStatusError("Rate limited", request=response.request, response=response)
            
            if response.status_code >= 500:
                logger.warning(f"Server error ({response.status_code}) for {url}, retrying...")
                raise httpx.HTTPStatusError("Server error", request=response.request, response=response)
            
            response.raise_for_status()
            
            # Check content-type if available
            content_type = response.headers.get('content-type', '').lower()
            if content_type and not content_type.startswith('application/pdf'):
                logger.warning(f"Unexpected content-type: {content_type} for {url}")
            
            # Download with size limit
            content = b''
            for chunk in response.iter_bytes(8192):
                content += chunk
                if len(content) > max_size:
                    raise PDFDownloadError(f"PDF too large (>{max_size/1024/1024:.1f}MB): {url}")
            
            return content


def _validate_pdf_content(content: bytes, url: str) -> None:
    """Validate that content looks like a PDF and not HTML."""
    if len(content) < 4:
        raise PDFDownloadError(f"Downloaded content too short: {url}")
    
    # Check for PDF magic bytes
    if not content.startswith(b'%PDF'):
        # Check if it looks like HTML (common when URLs redirect to error pages)
        content_str = content[:1000].decode('utf-8', errors='ignore').lower()
        if any(tag in content_str for tag in ['<html', '<body', '<!doctype']):
            raise PDFDownloadError(f"Downloaded HTML instead of PDF: {url}")
        
        logger.warning(f"Content doesn't start with PDF magic bytes, but proceeding: {url}")


def extract_text(pdf_path: str, min_len_threshold: int = 800) -> str:
    """
    Extract text from PDF with fallback strategy.
    
    Args:
        pdf_path: Path to PDF file
        min_len_threshold: Minimum text length to consider extraction successful
        
    Returns:
        Extracted and normalized text
        
    Raises:
        PDFParseError: If text extraction fails
    """
    if not os.path.exists(pdf_path):
        raise PDFParseError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Extracting text from: {pdf_path}")
    start_time = time.time()
    
    try:
        # Primary: Try pypdf
        if pypdf:
            text = _extract_with_pypdf(pdf_path)
            if len(text.strip()) >= min_len_threshold:
                duration = time.time() - start_time
                logger.info(f"Extracted {len(text)} chars with pypdf in {duration:.2f}s")
                return _normalize_whitespace(text)
            else:
                logger.info(f"pypdf extracted only {len(text.strip())} chars, trying fallback")
        
        # Fallback: Try pdfminer.six
        if pdfminer_extract_text:
            text = _extract_with_pdfminer(pdf_path)
            duration = time.time() - start_time
            logger.info(f"Extracted {len(text)} chars with pdfminer in {duration:.2f}s")
            return _normalize_whitespace(text)
        
        # No extraction libraries available
        raise PDFParseError(f"No PDF extraction libraries available (pypdf: {pypdf is not None}, pdfminer: {pdfminer_extract_text is not None})")
        
    except Exception as e:
        if isinstance(e, PDFParseError):
            raise
        else:
            raise PDFParseError(f"Failed to extract text from {pdf_path}: {e}")


def _extract_with_pypdf(pdf_path: str) -> str:
    """Extract text using pypdf library."""
    try:
        text_parts = []
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            
            # Check if encrypted
            if reader.is_encrypted:
                raise PDFParseError(f"PDF is encrypted/password-protected: {pdf_path}")
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
        
        return '\n'.join(text_parts)
        
    except Exception as e:
        logger.warning(f"pypdf extraction failed: {e}")
        raise


def _extract_with_pdfminer(pdf_path: str) -> str:
    """Extract text using pdfminer.six library."""
    try:
        with open(pdf_path, 'rb') as file:
            text = pdfminer_extract_text(file)
            return text or ""
            
    except PDFSyntaxError as e:
        raise PDFParseError(f"PDF syntax error: {e}")
    except Exception as e:
        logger.warning(f"pdfminer extraction failed: {e}")
        raise


def _normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph structure.
    
    - Collapse multiple spaces/tabs into single spaces
    - Preserve paragraph breaks (double newlines)
    - Remove trailing whitespace from lines
    """
    if not text:
        return ""
    
    # Split into lines and process
    lines = text.split('\n')
    normalized_lines = []
    
    for line in lines:
        # Remove trailing whitespace and collapse internal whitespace
        line = re.sub(r'[ \t]+', ' ', line.strip())
        normalized_lines.append(line)
    
    # Join lines back together
    text = '\n'.join(normalized_lines)
    
    # Preserve paragraph breaks by protecting double+ newlines
    # Replace multiple consecutive newlines with double newlines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    return text.strip()


def chunk_text(full_text: str, chunk_size: int = 3000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks using sliding window approach.
    
    Args:
        full_text: Text to chunk
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks (no empty chunks)
    """
    if not full_text or not full_text.strip():
        return []
    
    text = full_text.strip()
    
    # If text is shorter than chunk_size, return as single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # If this is the last chunk, take everything remaining
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            break
        
        # For intermediate chunks, extract the chunk
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move start position forward by (chunk_size - overlap)
        start += chunk_size - chunk_overlap
        
        # Ensure we make progress even with large overlaps
        prev_start = start - (chunk_size - chunk_overlap)
        if start <= prev_start and len(chunks) > 0:
            start = prev_start + 1
    
    logger.info(f"Chunked {len(text)} chars into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    
    return chunks


# Example usage and testing functions
if __name__ == "__main__":
    # Simple test of chunking logic
    test_text = "This is a test document. " * 100
    chunks = chunk_text(test_text, chunk_size=100, chunk_overlap=20)
    print(f"Created {len(chunks)} chunks from {len(test_text)} characters")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1}: {chunk[:50]}...")
