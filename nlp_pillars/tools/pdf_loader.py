"""
PDF Loading and Processing Utilities

Provides robust PDF downloading, text extraction, and chunking capabilities
with fallback strategies and comprehensive error handling.
"""

import hashlib
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    import pymupdf4llm
except ImportError:
    pymupdf4llm = None

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

# Layout-aware processing imports
try:
    import layoutparser as lp
    import torch
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    layout_parser_available = True
except ImportError:
    layout_parser_available = False
    lp = None
    torch = None
    convert_from_path = None
    pytesseract = None
    Image = None

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

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
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
        # Primary: Try Layout-aware extraction (best for multi-column academic PDFs)
        if layout_parser_available:
            try:
                text = _extract_with_layout_parser(pdf_path)
                if len(text.strip()) >= min_len_threshold:
                    duration = time.time() - start_time
                    logger.info(f"Extracted {len(text)} chars with layout-parser in {duration:.2f}s")
                    return _normalize_whitespace(text)
                else:
                    logger.info(f"layout-parser extracted only {len(text.strip())} chars, trying fallback")
            except Exception as e:
                logger.warning(f"Layout-aware extraction failed: {e}, trying fallback")

        # Fallback 1: Try PyMuPDF4LLM (good for general LLM processing)
        if pymupdf4llm:
            text = _extract_with_pymupdf4llm(pdf_path)
            if len(text.strip()) >= min_len_threshold:
                duration = time.time() - start_time
                logger.info(f"Extracted {len(text)} chars with pymupdf4llm in {duration:.2f}s")
                return _normalize_whitespace(text)
            else:
                logger.info(f"pymupdf4llm extracted only {len(text.strip())} chars, trying fallback")

        # Fallback 2: Try pypdf
        if pypdf:
            text = _extract_with_pypdf(pdf_path)
            if len(text.strip()) >= min_len_threshold:
                duration = time.time() - start_time
                logger.info(f"Extracted {len(text)} chars with pypdf in {duration:.2f}s")
                return _normalize_whitespace(text)
            else:
                logger.info(f"pypdf extracted only {len(text.strip())} chars, trying final fallback")

        # Fallback 3: Try pdfminer.six
        if pdfminer_extract_text:
            text = _extract_with_pdfminer(pdf_path)
            duration = time.time() - start_time
            logger.info(f"Extracted {len(text)} chars with pdfminer in {duration:.2f}s")
            return _normalize_whitespace(text)

        # No extraction libraries available
        raise PDFParseError(f"No PDF extraction libraries available (layout-parser: {layout_parser_available}, pymupdf4llm: {pymupdf4llm is not None}, pypdf: {pypdf is not None}, pdfminer: {pdfminer_extract_text is not None})")

    except Exception as e:
        if isinstance(e, PDFParseError):
            raise
        else:
            raise PDFParseError(f"Failed to extract text from {pdf_path}: {e}")


def _extract_with_pymupdf4llm(pdf_path: str) -> str:
    """Extract text using PyMuPDF4LLM library (optimized for LLM processing)."""
    try:
        # PyMuPDF4LLM returns markdown-formatted text which is better for LLM processing
        text = pymupdf4llm.to_markdown(pdf_path)

        # Convert markdown back to clean text for processing
        # Remove markdown formatting but preserve structure
        import re

        # Convert headers to plain text with proper spacing
        text = re.sub(r'^#{1,6}\s*(.+)$', r'\1\n', text, flags=re.MULTILINE)

        # Convert bold/italic to plain text
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)

        # Clean up extra newlines while preserving paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    except Exception as e:
        logger.warning(f"pymupdf4llm extraction failed: {e}")
        raise


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


def _extract_with_layout_parser(pdf_path: str, max_pages: int = 50) -> str:
    """
    Extract text using layout-parser with LayoutLMv3 for layout-aware processing.
    
    This method:
    1. Converts PDF pages to images
    2. Uses LayoutLMv3 to detect layout blocks
    3. Extracts text from each block using OCR
    4. Reassembles text in correct reading order (handles multi-column layouts)
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum number of pages to process (to avoid memory issues)
        
    Returns:
        Extracted and ordered text
        
    Raises:
        PDFParseError: If layout-aware extraction fails
    """
    if not layout_parser_available:
        raise PDFParseError("Layout-parser dependencies not available")

    logger.info(f"Starting layout-aware extraction for: {pdf_path}")

    try:
        # Step 1: Convert PDF to images
        logger.info("Converting PDF pages to images...")
        images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=max_pages)
        logger.info(f"Converted {len(images)} pages to images")

        # Step 2: Initialize LayoutLMv3 model
        model = _get_layout_model()

        # Step 3: Process each page
        all_text_parts = []

        for page_num, image in enumerate(images, 1):
            try:
                logger.info(f"Processing page {page_num}/{len(images)}")
                page_text = _process_page_with_layout(image, model, page_num)
                if page_text.strip():
                    all_text_parts.append(page_text.strip())

            except Exception as e:
                logger.warning(f"Failed to process page {page_num}: {e}")
                continue

        # Step 4: Combine all pages
        full_text = "\n\n".join(all_text_parts)
        logger.info(f"Layout-aware extraction completed. Total text length: {len(full_text)} chars")

        return full_text

    except Exception as e:
        logger.error(f"Layout-aware extraction failed: {e}")
        raise PDFParseError(f"Layout-aware extraction failed: {e}")


def _get_layout_model():
    """Initialize and cache the LayoutLMv3 model for layout detection."""
    global _layout_model_cache

    if hasattr(_get_layout_model, '_cached_model'):
        return _get_layout_model._cached_model

    try:
        # Use a pre-trained LayoutLMv3 model optimized for academic papers
        # This model can detect various layout elements like paragraphs, titles, lists, etc.
        model = lp.Detectron2LayoutModel(
            config_path="lp://PubLayNet/layoutlmv3-large-publaynet/config",
            model_path="lp://PubLayNet/layoutlmv3-large-publaynet/model_final.pth",
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        )
        _get_layout_model._cached_model = model
        logger.info("LayoutLMv3 model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Failed to load LayoutLMv3 model: {e}")
        raise PDFParseError(f"Failed to load layout model: {e}")


def _process_page_with_layout(image, model, page_num: int) -> str:
    """
    Process a single page image to extract layout-aware text.
    
    Args:
        image: PIL Image of the PDF page
        model: LayoutLMv3 model instance
        page_num: Page number for logging
        
    Returns:
        Text extracted from the page in correct reading order
    """
    try:
        # Step 1: Detect layout blocks
        layout = model.detect(image)

        if len(layout) == 0:
            logger.warning(f"No layout blocks detected on page {page_num}")
            return ""

        # Step 2: Filter and sort blocks by reading order
        text_blocks = []

        for block in layout:
            if block.type in ["Text", "Title", "List"]:  # Focus on text content
                # Extract text from this block using OCR
                block_image = image.crop([
                    block.coordinates[0], block.coordinates[1],
                    block.coordinates[2], block.coordinates[3]
                ])

                # Use Tesseract OCR to extract text from the block
                block_text = pytesseract.image_to_string(block_image, config='--psm 6')

                if block_text.strip():
                    text_blocks.append({
                        'text': block_text.strip(),
                        'coordinates': block.coordinates,
                        'type': block.type,
                        'confidence': getattr(block, 'score', 1.0)
                    })

        # Step 3: Sort blocks by reading order (multi-column aware)
        sorted_blocks = _sort_blocks_reading_order(text_blocks, image.width, image.height)

        # Step 4: Combine text from sorted blocks
        page_text_parts = []
        for block in sorted_blocks:
            # Add some formatting based on block type
            if block['type'] == 'Title':
                page_text_parts.append(f"\n{block['text']}\n")
            elif block['type'] == 'List':
                page_text_parts.append(f"{block['text']}")
            else:  # Regular text
                page_text_parts.append(block['text'])

        page_text = " ".join(page_text_parts)
        logger.info(f"Page {page_num}: Extracted {len(page_text)} chars from {len(sorted_blocks)} blocks")

        return page_text

    except Exception as e:
        logger.error(f"Failed to process page {page_num}: {e}")
        return ""


def _sort_blocks_reading_order(blocks: List[dict], page_width: int, page_height: int) -> List[dict]:
    """
    Sort text blocks by natural reading order, handling multi-column layouts.
    
    This function implements a sophisticated reading order algorithm:
    1. Group blocks into columns by x-coordinate clustering
    2. Within each column, sort by y-coordinate (top to bottom)
    3. Sort columns by x-coordinate (left to right)
    
    Args:
        blocks: List of block dictionaries with coordinates
        page_width: Width of the page image
        page_height: Height of the page image
        
    Returns:
        List of blocks sorted in natural reading order
    """
    if not blocks:
        return []

    # Extract coordinates for analysis
    for block in blocks:
        coords = block['coordinates']
        block['left'] = coords[0]
        block['top'] = coords[1]
        block['right'] = coords[2]
        block['bottom'] = coords[3]
        block['center_x'] = (coords[0] + coords[2]) / 2
        block['center_y'] = (coords[1] + coords[3]) / 2
        block['width'] = coords[2] - coords[0]
        block['height'] = coords[3] - coords[1]

    # Detect if this is likely a multi-column layout
    # If blocks are clustered in distinct x-ranges, it's probably multi-column
    center_xs = [block['center_x'] for block in blocks]
    is_multi_column = _detect_multi_column_layout(center_xs, page_width)

    if is_multi_column:
        logger.info("Multi-column layout detected, using column-aware sorting")
        return _sort_blocks_multi_column(blocks, page_width)
    else:
        logger.info("Single-column layout detected, using simple top-to-bottom sorting")
        return _sort_blocks_single_column(blocks)


def _detect_multi_column_layout(center_xs: List[float], page_width: int) -> bool:
    """
    Detect if the layout is multi-column by analyzing x-coordinate distribution.
    
    Returns True if blocks appear to be clustered in distinct columns.
    """
    if len(center_xs) < 4:  # Need at least 4 blocks to detect columns
        return False

    # Simple heuristic: if we have blocks in both left and right halves
    # and minimal overlap in the middle third, it's likely multi-column
    left_third = page_width / 3
    right_third = 2 * page_width / 3

    left_blocks = sum(1 for x in center_xs if x < left_third)
    middle_blocks = sum(1 for x in center_xs if left_third <= x <= right_third)
    right_blocks = sum(1 for x in center_xs if x > right_third)

    # Multi-column if we have blocks in both left and right, with few in middle
    return (left_blocks > 0 and right_blocks > 0 and
            middle_blocks < max(left_blocks, right_blocks) / 2)


def _sort_blocks_multi_column(blocks: List[dict], page_width: int) -> List[dict]:
    """Sort blocks for multi-column layout (typically 2 columns)."""
    # Group blocks into columns based on their center x-coordinate
    left_column = []
    right_column = []
    middle_threshold = page_width / 2

    for block in blocks:
        if block['center_x'] < middle_threshold:
            left_column.append(block)
        else:
            right_column.append(block)

    # Sort each column by y-coordinate (top to bottom)
    left_column.sort(key=lambda b: b['top'])
    right_column.sort(key=lambda b: b['top'])

    # Combine: left column first, then right column
    return left_column + right_column


def _sort_blocks_single_column(blocks: List[dict]) -> List[dict]:
    """Sort blocks for single-column layout (simple top-to-bottom)."""
    return sorted(blocks, key=lambda b: (b['top'], b['left']))


def _clean_pdf_header_noise(text: str) -> str:
    """
    Remove common PDF header noise and artifacts from extracted text.
    
    - Remove scattered single characters from PDF metadata
    - Remove arXiv header artifacts
    - Find the start of meaningful content
    """
    if not text:
        return ""

    lines = text.split('\n')
    meaningful_start = -1

    # Look for the start of meaningful content
    for i, line in enumerate(lines):
        line_clean = line.strip()

        # Skip very short lines and single characters
        if len(line_clean) < 3:
            continue

        # Look for typical paper starting patterns
        if any(pattern in line_clean.lower() for pattern in [
            'abstract', 'introduction', 'title', 'author',
            # Common paper titles or phrases
            'attention is all you need', 'transformer', 'bert', 'gpt', 'mistral', 'minigpt',
            # Or substantial content indicators
        ]) and len(line_clean) > 10:
            meaningful_start = max(0, i - 5)  # Include a few lines before for context
            break

        # If we find a line with substantial content (likely title)
        if len(line_clean) > 15 and len(line_clean.split()) > 2:
            # Check if this looks like a real title/content (letters, numbers, common punctuation)
            if re.match(r'^[a-zA-Z][a-zA-Z0-9\s\[\]\.\-:&,\(\)]+$', line_clean):
                # Additional check: avoid lines that are mostly numbers or single words
                words = line_clean.split()
                if len(words) >= 2 and not all(w.isdigit() for w in words):
                    meaningful_start = max(0, i - 1)
                    break

    # If we found meaningful content, start from there
    if meaningful_start >= 0:
        cleaned_lines = lines[meaningful_start:]
    else:
        # Fallback: remove obvious noise from the beginning
        cleaned_lines = []
        skip_mode = True

        for line in lines:
            line_clean = line.strip()

            # Skip single characters and very short lines at the start
            if skip_mode and len(line_clean) <= 2:
                continue

            # Once we hit substantial content, keep everything
            if len(line_clean) > 10:
                skip_mode = False

            if not skip_mode:
                cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def _normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph structure.
    
    - Collapse multiple spaces/tabs into single spaces
    - Preserve paragraph breaks (double newlines)
    - Remove trailing whitespace from lines
    """
    if not text:
        return ""

    # First clean PDF header noise
    text = _clean_pdf_header_noise(text)

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


# Chunk sizes are denominated in TOKENS, counted with the encoding the embedding
# model actually uses. Chunks produced here are embedded with
# text-embedding-3-small (see nlp_pillars/vectors.py::_embed), which tokenizes
# with cl100k_base, so a cl100k_base count is the size the downstream model sees.
EMBEDDING_ENCODING = "cl100k_base"

# Only used to translate a token budget into a character budget for the naive
# fallback, which cannot count tokens. ~4 characters per token is the usual
# ratio for English prose; it is an approximation and only ever applies when
# semantic chunking is unavailable.
_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=1)
def _get_token_counter() -> Callable[[str], int]:
    """
    Return the token counter semchunk sizes chunks with.

    Prefers tiktoken's ``cl100k_base`` so chunk sizes match what the embedding
    model counts. If tiktoken cannot load its encoding (it fetches the BPE file
    on first use), falls back to a character-based approximation rather than
    failing the whole chunking path — semantic boundaries still hold, only the
    size budget becomes approximate.
    """
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    except Exception as e:
        logger.error(
            f"tiktoken encoding '{EMBEDDING_ENCODING}' unavailable ({e}). Sizing chunks "
            f"with a ~{_CHARS_PER_TOKEN}-chars-per-token approximation instead; "
            f"chunk sizes will be approximate."
        )
        return lambda text: max(1, len(text) // _CHARS_PER_TOKEN)

    # disallowed_special=() so a literal "<|endoftext|>" in a paper is counted as
    # ordinary text instead of raising.
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def chunk_text(full_text: str, chunk_size: int = 750, chunk_overlap: int = 50) -> List[str]:
    """
    Split text into semantically meaningful chunks using the semchunk library.

    semchunk splits on the strongest semantic boundary that fits the budget
    (sections, then paragraphs, then sentences, then clauses), so chunks end at
    natural breaks instead of mid-sentence the way a sliding window does.

    Args:
        full_text: Text to chunk
        chunk_size: Maximum TOKENS per chunk, counted with EMBEDDING_ENCODING.
            Note this is tokens, not characters: ~750 tokens is ~3000 characters
            of English prose.
        chunk_overlap: TOKENS of overlap between consecutive chunks (0 disables).
            Applied by semchunk itself, on token boundaries.

    Returns:
        List of text chunks (no empty chunks)

    Raises:
        RuntimeError: if semchunk is called incorrectly (a wrong-signature bug in
            this code, or an incompatible semchunk). Runtime failures fall back
            to naive chunking instead; see the comments below.
    """
    if not full_text or not full_text.strip():
        return []

    text = full_text.strip()

    try:
        from semchunk import chunk as semantic_chunk
    except ImportError:
        # Logged at ERROR, not WARNING: semchunk is a pinned hard dependency, so
        # a missing import means a broken install, and degrading quietly to
        # naive chunking is exactly how this path stayed broken for months.
        logger.error("semchunk library not available. Falling back to naive chunking.")
        return _chunk_text_naive(
            full_text, chunk_size * _CHARS_PER_TOKEN, chunk_overlap * _CHARS_PER_TOKEN
        )

    token_counter = _get_token_counter()

    # If the whole text already fits the budget, return it as a single chunk.
    if token_counter(text) <= chunk_size:
        return [text]

    try:
        chunks = semantic_chunk(
            text,
            chunk_size,
            token_counter,
            overlap=chunk_overlap if chunk_overlap > 0 else None,
        )
    except TypeError as e:
        # A TypeError here is a call-signature mismatch — this code disagreeing
        # with the installed semchunk — not bad input. That is precisely the bug
        # this path had (semchunk 4.x made `token_counter` required and every
        # call silently fell back), and swallowing it hid the regression for
        # months. Fail loudly so tests and the first upload surface it.
        raise RuntimeError(
            f"semchunk called with an incompatible signature: {e}. This is a bug in "
            f"chunk_text or an incompatible semchunk version, not a data problem."
        ) from e
    except Exception as e:
        # Genuine runtime failures (odd input, tokenizer trouble) still degrade
        # to naive chunking rather than failing an upload, but loudly and with a
        # traceback.
        logger.error(
            f"Semantic chunking failed: {e}. Falling back to naive chunking.", exc_info=True
        )
        return _chunk_text_naive(
            full_text, chunk_size * _CHARS_PER_TOKEN, chunk_overlap * _CHARS_PER_TOKEN
        )

    # Filter out empty chunks
    chunks = [c.strip() for c in chunks if c.strip()]

    logger.info(
        f"Semantic chunking: {len(text)} chars into {len(chunks)} chunks "
        f"(max_tokens={chunk_size}, overlap_tokens={chunk_overlap})"
    )

    return chunks


def _chunk_text_naive(full_text: str, chunk_size: int = 3000, chunk_overlap: int = 200) -> List[str]:
    """
    Fallback naive text chunking using sliding window approach.

    This is the original implementation used when semchunk is not available
    or fails for any reason. Unlike chunk_text(), its sizes are in CHARACTERS —
    chunk_text() converts its token budget with _CHARS_PER_TOKEN before calling
    here. It cuts on a fixed offset, so chunks routinely end mid-sentence.
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

    logger.info(f"Naive chunking: {len(text)} chars into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")

    return chunks


def _add_overlap_to_chunks(chunks: List[str], overlap: int) -> List[str]:
    """
    Add character-level overlap between chunks by prefixing each chunk with the
    tail of its predecessor.

    No longer used by chunk_text(): semchunk applies overlap itself, on token
    boundaries, which keeps the overlap in the same units as the chunk size and
    does not duplicate a partial word into the next chunk. Kept as a helper for
    callers that want character-denominated overlap over an arbitrary chunk list.
    """
    if len(chunks) <= 1 or overlap <= 0:
        return chunks

    overlapped_chunks = [chunks[0]]  # First chunk remains unchanged

    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        current_chunk = chunks[i]

        # Extract overlap from end of previous chunk
        prev_end = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk

        # Create overlapped chunk
        overlapped_chunk = prev_end + " " + current_chunk
        overlapped_chunks.append(overlapped_chunk)

    return overlapped_chunks


def test_layout_parser_availability() -> dict:
    """
    Test if layout-parser and associated dependencies are available and working.
    
    Returns:
        Dictionary with status of each component
    """
    status = {
        "layout_parser_available": layout_parser_available,
        "dependencies": {},
        "system_requirements": {},
        "recommendations": []
    }

    # Check individual dependencies
    try:
        import layoutparser as lp
        status["dependencies"]["layoutparser"] = True
    except ImportError:
        status["dependencies"]["layoutparser"] = False
        status["recommendations"].append("Install layoutparser: pip install layoutparser")

    try:
        import torch
        status["dependencies"]["torch"] = True
        status["system_requirements"]["torch_version"] = torch.__version__
    except ImportError:
        status["dependencies"]["torch"] = False
        status["recommendations"].append("Install PyTorch: pip install torch torchvision")

    try:
        from pdf2image import convert_from_path
        status["dependencies"]["pdf2image"] = True
    except ImportError:
        status["dependencies"]["pdf2image"] = False
        status["recommendations"].append("Install pdf2image: pip install pdf2image")

    try:
        import pytesseract
        status["dependencies"]["pytesseract"] = True
        # Test if Tesseract is actually installed on the system
        try:
            pytesseract.get_tesseract_version()
            status["system_requirements"]["tesseract_installed"] = True
        except Exception:
            status["system_requirements"]["tesseract_installed"] = False
            status["recommendations"].append("Install Tesseract OCR system package")
    except ImportError:
        status["dependencies"]["pytesseract"] = False
        status["recommendations"].append("Install pytesseract: pip install pytesseract")

    try:
        from PIL import Image
        status["dependencies"]["pillow"] = True
    except ImportError:
        status["dependencies"]["pillow"] = False
        status["recommendations"].append("Install Pillow: pip install Pillow")

    return status


def get_installation_instructions() -> str:
    """
    Get detailed installation instructions for layout-aware PDF processing.
    
    Returns:
        String with installation instructions
    """
    instructions = """
Layout-Aware PDF Processing Installation Instructions:

1. Install Python dependencies:
   pip install layoutparser torch torchvision pdf2image pytesseract Pillow

2. Install system dependencies:
   
   On Ubuntu/Debian:
   sudo apt-get update
   sudo apt-get install tesseract-ocr poppler-utils
   
   On macOS:
   brew install tesseract poppler
   
   On Windows:
   - Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
   - Download and install Poppler from: https://poppler.freedesktop.org/
   
3. Verify installation:
   python -c "from nlp_pillars.tools.pdf_loader import test_layout_parser_availability; print(test_layout_parser_availability())"

4. Optional: For better performance, ensure you have CUDA-enabled PyTorch if you have a GPU:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Note: The first time you use layout-aware extraction, LayoutLMv3 models will be automatically 
downloaded (several GB). Ensure you have a stable internet connection and sufficient storage.
"""
    return instructions


# Example usage and testing functions
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test-layout":
        # Test layout-parser availability
        print("Testing layout-parser availability...")
        status = test_layout_parser_availability()

        print("\nDependency Status:")
        for dep, available in status["dependencies"].items():
            print(f"  {dep}: {'✓' if available else '✗'}")

        print("\nSystem Requirements:")
        for req, available in status["system_requirements"].items():
            print(f"  {req}: {'✓' if available else '✗'}")

        if status["recommendations"]:
            print("\nRecommendations:")
            for rec in status["recommendations"]:
                print(f"  - {rec}")

        if status["layout_parser_available"]:
            print("\n✓ Layout-aware PDF processing is ready!")
        else:
            print("\n✗ Layout-aware PDF processing is not available.")
            print("\nInstallation instructions:")
            print(get_installation_instructions())

    else:
        # Simple test of chunking logic
        test_text = "This is a test document. " * 100
        chunks = chunk_text(test_text, chunk_size=100, chunk_overlap=20)
        print(f"Created {len(chunks)} chunks from {len(test_text)} characters")
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i+1}: {chunk[:50]}...")

        print("\nTo test layout-parser availability, run:")
        print("python -m nlp_pillars.tools.pdf_loader test-layout")
