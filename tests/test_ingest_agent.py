"""
Comprehensive tests for PDF Ingest Agent and PDF Loader utilities.
All network calls are mocked for fast, reliable testing.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import hashlib

from nlp_pillars.schemas import PaperRef, ParsedPaper
from nlp_pillars.agents.ingest_agent import IngestAgent, IngestError, ingest_paper_from_url
from nlp_pillars.tools.pdf_loader import (
    download_pdf, extract_text, chunk_text, 
    PDFDownloadError, PDFParseError,
    _normalize_whitespace, _validate_pdf_content
)


# Test fixtures
@pytest.fixture
def sample_paper_ref():
    """Sample paper reference for testing."""
    return PaperRef(
        id="test.12345",
        title="Test Paper: A Comprehensive Study",
        authors=["Dr. Test", "Prof. Example"],
        venue="Test Conference",
        year=2023,
        url_pdf="https://example.com/test_paper.pdf"
    )


@pytest.fixture
def sample_pdf_content():
    """Sample PDF bytes for testing."""
    return b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\nThis is test PDF content...' + b'x' * 1000


@pytest.fixture
def sample_text():
    """Sample extracted text for testing."""
    return "This is a test paper. It contains multiple sentences. " * 50


@pytest.fixture
def temp_cache_dir():
    """Temporary directory for testing caching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestPDFLoader:
    """Test cases for PDF loader utility functions."""
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization function."""
        # Test basic whitespace collapse
        assert _normalize_whitespace("  multiple   spaces  ") == "multiple spaces"
        
        # Test tab handling
        assert _normalize_whitespace("tabs\t\there") == "tabs here"
        
        # Test paragraph preservation
        input_text = "Paragraph one.\n\nParagraph two.\n\n\nParagraph three."
        expected = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        assert _normalize_whitespace(input_text) == expected
        
        # Test empty input
        assert _normalize_whitespace("") == ""
        assert _normalize_whitespace("   \n\n   ") == ""
        
        # Test line trailing whitespace removal
        input_text = "Line one   \nLine two\t\n   Line three  "
        expected = "Line one\nLine two\nLine three"
        assert _normalize_whitespace(input_text) == expected
    
    def test_validate_pdf_content(self):
        """Test PDF content validation."""
        # Valid PDF content
        _validate_pdf_content(b'%PDF-1.4\nvalid content', "test.pdf")
        
        # HTML masquerading as PDF
        with pytest.raises(PDFDownloadError, match="Downloaded HTML instead of PDF"):
            _validate_pdf_content(b'<html><body>Not a PDF</body></html>', "test.pdf")
        
        # Too short content
        with pytest.raises(PDFDownloadError, match="too short"):
            _validate_pdf_content(b'123', "test.pdf")
        
        # Non-PDF but not HTML (should warn but not fail)
        _validate_pdf_content(b'Not a PDF but also not HTML content', "test.pdf")
    
    @patch('nlp_pillars.tools.pdf_loader.httpx.Client')
    def test_download_pdf_success(self, mock_client_class, temp_cache_dir, sample_pdf_content):
        """Test successful PDF download with caching."""
        url = "https://example.com/test.pdf"
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/pdf'}
        mock_response.iter_bytes.return_value = [sample_pdf_content[i:i+100] for i in range(0, len(sample_pdf_content), 100)]
        mock_response.raise_for_status = Mock()
        
        mock_stream_context = Mock()
        mock_stream_context.__enter__ = Mock(return_value=mock_response)
        mock_stream_context.__exit__ = Mock(return_value=None)
        
        mock_client = Mock()
        mock_client.stream.return_value = mock_stream_context
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client
        
        # First download
        result_path = download_pdf(url, temp_cache_dir)
        
        # Verify file was created
        assert os.path.exists(result_path)
        assert result_path.endswith('.pdf')
        
        # Verify content
        with open(result_path, 'rb') as f:
            assert f.read() == sample_pdf_content
        
        # Second download should use cache
        with patch('nlp_pillars.tools.pdf_loader.logger') as mock_logger:
            result_path2 = download_pdf(url, temp_cache_dir)
            assert result_path == result_path2
            mock_logger.info.assert_called_with(f"Using cached PDF: {result_path}")
    
    def test_download_pdf_file_url(self, temp_cache_dir):
        """Test file:// URL handling."""
        # Create a temporary PDF file
        test_file = Path(temp_cache_dir) / "test.pdf"
        test_file.write_bytes(b'%PDF-1.4\ntest content')
        
        file_url = f"file://{test_file}"
        result_path = download_pdf(file_url, temp_cache_dir)
        
        assert result_path == str(test_file)
        
        # Test non-existent file
        with pytest.raises(PDFDownloadError, match="Local file not found"):
            download_pdf("file:///nonexistent.pdf", temp_cache_dir)
    
    @patch('nlp_pillars.tools.pdf_loader.httpx.Client')
    def test_download_pdf_size_limit(self, mock_client_class, temp_cache_dir):
        """Test PDF size limit enforcement."""
        url = "https://example.com/huge.pdf"
        
        # Mock large response
        huge_content = b'%PDF-1.4\n' + b'x' * (60 * 1024 * 1024)  # 60MB
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/pdf'}
        mock_response.iter_bytes.return_value = [huge_content[i:i+8192] for i in range(0, len(huge_content), 8192)]
        mock_response.raise_for_status = Mock()
        
        mock_stream_context = Mock()
        mock_stream_context.__enter__ = Mock(return_value=mock_response)
        mock_stream_context.__exit__ = Mock(return_value=None)
        
        mock_client = Mock()
        mock_client.stream.return_value = mock_stream_context
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client
        
        with pytest.raises(PDFDownloadError, match="PDF too large"):
            download_pdf(url, temp_cache_dir)
    
    @patch('nlp_pillars.tools.pdf_loader.httpx.Client')
    def test_download_pdf_retry_logic(self, mock_client_class, temp_cache_dir):
        """Test retry logic for 429 and 5xx errors."""
        url = "https://example.com/retry_test.pdf"
        
        # Mock responses: 429, 500, then success
        responses = []
        
        # 429 response
        mock_429 = Mock()
        mock_429.status_code = 429
        mock_429.request = Mock()
        responses.append(mock_429)
        
        # 500 response  
        mock_500 = Mock()
        mock_500.status_code = 500
        mock_500.request = Mock()
        responses.append(mock_500)
        
        # Success response
        mock_success = Mock()
        mock_success.status_code = 200
        mock_success.headers = {'content-type': 'application/pdf'}
        mock_success.iter_bytes.return_value = [b'%PDF-1.4\nsample']
        mock_success.raise_for_status = Mock()
        responses.append(mock_success)
        
        # Setup mock client to return responses in sequence
        mock_client = Mock()
        stream_contexts = []
        
        for response in responses:
            mock_stream_context = Mock()
            mock_stream_context.__enter__ = Mock(return_value=response)
            mock_stream_context.__exit__ = Mock(return_value=None)
            stream_contexts.append(mock_stream_context)
        
        mock_client.stream.side_effect = stream_contexts
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client
        
        # Should succeed after retries
        with patch('time.sleep'):  # Speed up test by mocking sleep
            result_path = download_pdf(url, temp_cache_dir)
            assert os.path.exists(result_path)
    
    def test_chunk_text_basic(self):
        """Test basic text chunking functionality."""
        text = "This is a test. " * 100  # 1600 characters
        
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # No empty chunks
        assert all(chunk.strip() for chunk in chunks)
        
        # Check overlaps exist
        if len(chunks) > 1:
            # Last part of first chunk should appear in second chunk
            overlap_check = chunks[0][-50:]  # Last 50 chars of first chunk
            assert any(overlap_check in chunk for chunk in chunks[1:])
    
    def test_chunk_text_short_text(self):
        """Test chunking with text shorter than chunk size."""
        short_text = "This is a short text."
        chunks = chunk_text(short_text, chunk_size=1000)
        
        assert len(chunks) == 1
        assert chunks[0] == short_text
    
    def test_chunk_text_empty_input(self):
        """Test chunking with empty or whitespace-only input."""
        assert chunk_text("") == []
        assert chunk_text("   \n\n   ") == []
    
    def test_chunk_text_large_overlap(self):
        """Test chunking with large overlap values."""
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=90)
        
        # Should still make progress and create multiple chunks
        assert len(chunks) > 5  # Should create many small chunks
        assert all(len(chunk) <= 100 for chunk in chunks)
    
    @patch('nlp_pillars.tools.pdf_loader.pypdf')
    def test_extract_text_pypdf_success(self, mock_pypdf, temp_cache_dir):
        """Test successful text extraction with pypdf."""
        # Create a fake PDF file
        pdf_path = Path(temp_cache_dir) / "test.pdf"
        pdf_path.write_bytes(b'fake pdf content')
        
        # Mock pypdf extraction
        mock_page = Mock()
        mock_page.extract_text.return_value = "This is extracted text from pypdf. " * 30  # > 800 chars
        
        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page]
        
        mock_pypdf.PdfReader.return_value = mock_reader
        
        result = extract_text(str(pdf_path))
        
        assert len(result) > 800
        assert "This is extracted text from pypdf" in result
    
    @patch('nlp_pillars.tools.pdf_loader.pypdf')
    @patch('nlp_pillars.tools.pdf_loader.pdfminer_extract_text')
    def test_extract_text_fallback_to_pdfminer(self, mock_pdfminer, mock_pypdf, temp_cache_dir):
        """Test fallback to pdfminer when pypdf extracts too little text."""
        # Create a fake PDF file
        pdf_path = Path(temp_cache_dir) / "test.pdf"
        pdf_path.write_bytes(b'fake pdf content')
        
        # Mock pypdf to return short text
        mock_page = Mock()
        mock_page.extract_text.return_value = "Short text"  # < 800 chars
        
        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page]
        
        mock_pypdf.PdfReader.return_value = mock_reader
        
        # Mock pdfminer to return longer text
        mock_pdfminer.return_value = "This is longer text from pdfminer fallback. " * 25  # > 800 chars
        
        result = extract_text(str(pdf_path))
        
        assert "pdfminer fallback" in result
        assert len(result) > 800
        mock_pdfminer.assert_called_once()
    
    @patch('nlp_pillars.tools.pdf_loader.pypdf')
    def test_extract_text_encrypted_pdf(self, mock_pypdf, temp_cache_dir):
        """Test handling of encrypted PDFs."""
        # Create a fake PDF file
        pdf_path = Path(temp_cache_dir) / "encrypted.pdf"
        pdf_path.write_bytes(b'fake encrypted pdf')
        
        # Mock pypdf to indicate encryption
        mock_reader = Mock()
        mock_reader.is_encrypted = True
        
        mock_pypdf.PdfReader.return_value = mock_reader
        
        with pytest.raises(PDFParseError, match="encrypted/password-protected"):
            extract_text(str(pdf_path))
    
    def test_extract_text_nonexistent_file(self):
        """Test handling of non-existent PDF files."""
        with pytest.raises(PDFParseError, match="PDF file not found"):
            extract_text("/nonexistent/file.pdf")


class TestIngestAgent:
    """Test cases for the IngestAgent."""
    
    @patch('nlp_pillars.agents.ingest_agent.download_pdf')
    @patch('nlp_pillars.agents.ingest_agent.extract_text')
    @patch('nlp_pillars.agents.ingest_agent.chunk_text')
    def test_ingest_success(self, mock_chunk, mock_extract, mock_download, sample_paper_ref):
        """Test successful paper ingestion."""
        # Mock the pipeline
        mock_download.return_value = "/fake/path/paper.pdf"
        mock_extract.return_value = "This is extracted text. " * 100
        mock_chunks = ["Chunk 1 content", "Chunk 2 content", "Chunk 3 content"]
        mock_chunk.return_value = mock_chunks
        
        agent = IngestAgent()
        result = agent.ingest(sample_paper_ref)
        
        # Verify result structure
        assert isinstance(result, ParsedPaper)
        assert result.paper_ref == sample_paper_ref
        assert result.full_text == mock_extract.return_value
        assert result.chunks == mock_chunks
        assert result.figures_count == 0
        assert result.tables_count == 0
        assert result.references == []
        
        # Verify function calls
        mock_download.assert_called_once_with(sample_paper_ref.url_pdf, agent.cache_dir)
        mock_extract.assert_called_once_with("/fake/path/paper.pdf")
        mock_chunk.assert_called_once_with(mock_extract.return_value)
    
    @patch('nlp_pillars.agents.ingest_agent.download_pdf')
    def test_ingest_custom_pdf_url(self, mock_download, sample_paper_ref):
        """Test ingestion with custom PDF URL override."""
        custom_url = "https://custom.com/paper.pdf"
        mock_download.return_value = "/fake/path/paper.pdf"
        
        # Also need to mock the rest of the pipeline
        with patch('nlp_pillars.agents.ingest_agent.extract_text') as mock_extract, \
             patch('nlp_pillars.agents.ingest_agent.chunk_text') as mock_chunk:
            
            mock_extract.return_value = "Sample text"
            mock_chunk.return_value = ["Sample chunk"]
            
            agent = IngestAgent()
            result = agent.ingest(sample_paper_ref, pdf_url=custom_url)
            
            # Should use custom URL, not paper_ref.url_pdf
            mock_download.assert_called_once_with(custom_url, agent.cache_dir)
    
    def test_ingest_no_pdf_url(self):
        """Test ingestion failure when no PDF URL is available."""
        paper_without_url = PaperRef(
            id="test.123",
            title="Paper Without URL",
            authors=["Test Author"]
        )
        
        agent = IngestAgent()
        
        with pytest.raises(IngestError, match="No PDF URL available"):
            agent.ingest(paper_without_url)
    
    @patch('nlp_pillars.agents.ingest_agent.download_pdf')
    def test_ingest_download_failure(self, mock_download, sample_paper_ref):
        """Test handling of download failures."""
        mock_download.side_effect = PDFDownloadError("Download failed")
        
        agent = IngestAgent()
        
        with pytest.raises(IngestError, match="Failed to download PDF"):
            agent.ingest(sample_paper_ref)
    
    @patch('nlp_pillars.agents.ingest_agent.download_pdf')
    @patch('nlp_pillars.agents.ingest_agent.extract_text')
    def test_ingest_extraction_failure(self, mock_extract, mock_download, sample_paper_ref):
        """Test handling of text extraction failures."""
        mock_download.return_value = "/fake/path/paper.pdf"
        mock_extract.side_effect = PDFParseError("Extraction failed")
        
        agent = IngestAgent()
        
        with pytest.raises(IngestError, match="Failed to extract text"):
            agent.ingest(sample_paper_ref)
    
    @patch('nlp_pillars.agents.ingest_agent.download_pdf')
    @patch('nlp_pillars.agents.ingest_agent.extract_text')
    def test_ingest_empty_text(self, mock_extract, mock_download, sample_paper_ref):
        """Test handling of empty text extraction."""
        mock_download.return_value = "/fake/path/paper.pdf"
        mock_extract.return_value = "   \n\n   "  # Only whitespace
        
        agent = IngestAgent()
        
        with pytest.raises(IngestError, match="No text extracted"):
            agent.ingest(sample_paper_ref)
    
    @patch('nlp_pillars.agents.ingest_agent.download_pdf')
    @patch('nlp_pillars.agents.ingest_agent.extract_text')
    @patch('nlp_pillars.agents.ingest_agent.chunk_text')
    def test_ingest_chunking_failure(self, mock_chunk, mock_extract, mock_download, sample_paper_ref):
        """Test handling of chunking failures."""
        mock_download.return_value = "/fake/path/paper.pdf"
        mock_extract.return_value = "Valid text content"
        mock_chunk.return_value = []  # No chunks created
        
        agent = IngestAgent()
        
        with pytest.raises(IngestError, match="No chunks created"):
            agent.ingest(sample_paper_ref)
    
    @patch('nlp_pillars.agents.ingest_agent.download_pdf')
    @patch('nlp_pillars.agents.ingest_agent.extract_text')
    @patch('nlp_pillars.agents.ingest_agent.chunk_text')
    def test_ingest_batch(self, mock_chunk, mock_extract, mock_download):
        """Test batch ingestion with mixed success/failure."""
        # Setup mocks
        mock_download.side_effect = ["/path1.pdf", PDFDownloadError("Failed"), "/path3.pdf"]
        mock_extract.return_value = "Sample text"
        mock_chunk.return_value = ["Sample chunk"]
        
        # Test papers
        papers = [
            (PaperRef(id="1", title="Paper 1", authors=["A"], url_pdf="http://test1.pdf"), None),
            (PaperRef(id="2", title="Paper 2", authors=["B"], url_pdf="http://test2.pdf"), None),
            (PaperRef(id="3", title="Paper 3", authors=["C"], url_pdf="http://test3.pdf"), None),
        ]
        
        agent = IngestAgent()
        results = agent.ingest_batch(papers)
        
        # Should get 2 successful results (1st and 3rd papers)
        assert len(results) == 2
        assert all(isinstance(r, ParsedPaper) for r in results)
        assert results[0].paper_ref.id == "1"
        assert results[1].paper_ref.id == "3"
    
    def test_validate_pdf_url_success(self, sample_paper_ref):
        """Test PDF URL validation success."""
        with patch('nlp_pillars.agents.ingest_agent.download_pdf') as mock_download:
            mock_download.return_value = "/fake/path.pdf"
            
            agent = IngestAgent()
            assert agent.validate_pdf_url("https://valid.com/paper.pdf") is True
    
    def test_validate_pdf_url_failure(self):
        """Test PDF URL validation failure."""
        with patch('nlp_pillars.agents.ingest_agent.download_pdf') as mock_download:
            mock_download.side_effect = PDFDownloadError("Invalid URL")
            
            agent = IngestAgent()
            assert agent.validate_pdf_url("https://invalid.com/paper.pdf") is False
    
    def test_cache_stats(self, temp_cache_dir):
        """Test cache statistics functionality."""
        agent = IngestAgent(cache_dir=temp_cache_dir)
        
        # Empty cache
        stats = agent.get_cache_stats()
        assert stats["exists"] is True
        assert stats["file_count"] == 0
        
        # Add some fake PDF files (make them larger so size > 0 MB)
        (Path(temp_cache_dir) / "file1.pdf").write_bytes(b"fake pdf content " * 1000)  # ~16KB
        (Path(temp_cache_dir) / "file2.pdf").write_bytes(b"another pdf content " * 1000)  # ~20KB
        
        stats = agent.get_cache_stats()
        assert stats["file_count"] == 2
        assert stats["total_size_mb"] > 0
    
    def test_clear_cache(self, temp_cache_dir):
        """Test cache clearing functionality."""
        agent = IngestAgent(cache_dir=temp_cache_dir)
        
        # Add some fake files
        (Path(temp_cache_dir) / "file1.pdf").write_bytes(b"fake pdf 1")
        (Path(temp_cache_dir) / "file2.pdf").write_bytes(b"fake pdf 2")
        (Path(temp_cache_dir) / "temp.pdf.tmp").write_bytes(b"temp file")
        
        removed_count = agent.clear_cache()
        
        assert removed_count == 3
        
        # Verify files are gone
        assert not any(Path(temp_cache_dir).glob("*.pdf"))
        assert not any(Path(temp_cache_dir).glob("*.pdf.tmp"))


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('nlp_pillars.agents.ingest_agent.IngestAgent')
    def test_ingest_paper_from_url(self, mock_agent_class, sample_paper_ref):
        """Test convenience function for URL-based ingestion."""
        mock_agent = Mock()
        mock_agent.ingest.return_value = ParsedPaper(
            paper_ref=sample_paper_ref,
            full_text="test",
            chunks=["test chunk"]
        )
        mock_agent_class.return_value = mock_agent
        
        url = "https://test.com/paper.pdf"
        result = ingest_paper_from_url(url, sample_paper_ref)
        
        assert isinstance(result, ParsedPaper)
        mock_agent.ingest.assert_called_once_with(sample_paper_ref, pdf_url=url)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @patch('nlp_pillars.tools.pdf_loader.httpx.Client')
    @patch('nlp_pillars.tools.pdf_loader.pypdf')
    def test_full_pipeline_integration(self, mock_pypdf, mock_client_class, temp_cache_dir, sample_paper_ref):
        """Test full ingestion pipeline integration."""
        # Mock network download
        pdf_content = b'%PDF-1.4\nfake pdf content'
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/pdf'}
        mock_response.iter_bytes.return_value = [pdf_content]
        mock_response.raise_for_status = Mock()
        
        mock_stream_context = Mock()
        mock_stream_context.__enter__ = Mock(return_value=mock_response)
        mock_stream_context.__exit__ = Mock(return_value=None)
        
        mock_client = Mock()
        mock_client.stream.return_value = mock_stream_context
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client
        
        # Mock PDF text extraction
        sample_text = "This is a sample research paper. It contains multiple sentences and paragraphs. " * 20
        
        mock_page = Mock()
        mock_page.extract_text.return_value = sample_text
        
        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page]
        
        mock_pypdf.PdfReader.return_value = mock_reader
        
        # Run full pipeline
        agent = IngestAgent(cache_dir=temp_cache_dir)
        result = agent.ingest(sample_paper_ref)
        
        # Verify end-to-end result
        assert isinstance(result, ParsedPaper)
        assert result.paper_ref == sample_paper_ref
        assert len(result.full_text) > 100
        assert len(result.chunks) > 0
        assert all(isinstance(chunk, str) for chunk in result.chunks)
        assert len(result.chunks[0]) <= 3000  # Default chunk size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
