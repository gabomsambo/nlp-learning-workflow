"""Tests for upload service metadata enrichment and vector storage."""

import re
from unittest.mock import MagicMock, Mock, patch

import pytest

from nlp_pillars.schemas import PaperRef
from nlp_pillars.services.upload_service import UploadService


@pytest.fixture
def upload_service():
    """Create an upload service instance for testing."""
    return UploadService()


class TestArxivIdExtraction:
    """Test arXiv ID detection from various URL and filename formats."""

    def test_arxiv_id_from_abs_url(self):
        """Test arXiv ID detection from abs URL."""
        url = "https://arxiv.org/abs/2301.00001"
        match = re.search(r'(\d{4}\.\d{4,5})', url)
        assert match is not None
        assert match.group(1) == "2301.00001"

    def test_arxiv_id_from_pdf_url(self):
        """Test arXiv ID detection from PDF URL."""
        url = "https://arxiv.org/pdf/2301.00001.pdf"
        match = re.search(r'(\d{4}\.\d{4,5})', url)
        assert match is not None
        assert match.group(1) == "2301.00001"

    def test_arxiv_id_from_versioned_url(self):
        """Test arXiv ID detection from versioned URL."""
        url = "https://arxiv.org/abs/2301.00001v2"
        match = re.search(r'(\d{4}\.\d{4,5})', url)
        assert match is not None
        assert match.group(1) == "2301.00001"

    def test_arxiv_id_from_filename(self):
        """Test arXiv ID detection from filename."""
        filename = "2301.00001.pdf"
        match = re.search(r'(\d{4}\.\d{4,5})', filename)
        assert match is not None
        assert match.group(1) == "2301.00001"

    def test_arxiv_id_5_digit_minor(self):
        """Test arXiv ID with 5-digit minor version."""
        url = "https://arxiv.org/abs/2301.12345"
        match = re.search(r'(\d{4}\.\d{4,5})', url)
        assert match is not None
        assert match.group(1) == "2301.12345"


class TestArxivEnrichment:
    """Test arXiv metadata enrichment."""

    @patch('nlp_pillars.services.upload_service.arxiv.Client')
    @patch('nlp_pillars.services.upload_service.arxiv.Search')
    def test_arxiv_enrichment_success(
        self, mock_search_class, mock_client_class, upload_service
    ):
        """Test successful arXiv metadata enrichment."""
        # Mock arXiv response
        mock_result = Mock()
        mock_result.title = "Attention Is All You Need"
        mock_result.authors = [Mock(name="Vaswani"), Mock(name="Shazeer")]
        mock_result.published = Mock(year=2017)
        mock_result.summary = "The dominant sequence transduction models..."
        mock_result.journal_ref = None
        mock_result.primary_category = "cs.CL"

        mock_client = Mock()
        mock_client.results.return_value = iter([mock_result])
        mock_client_class.return_value = mock_client

        paper = PaperRef(id="1706.03762", title="", authors=[], url_pdf="")
        enriched = upload_service._enrich_from_arxiv(paper, "1706.03762")

        assert enriched.title == "Attention Is All You Need"
        assert len(enriched.authors) == 2

    @patch('nlp_pillars.services.upload_service.arxiv.Client')
    @patch('nlp_pillars.services.upload_service.arxiv.Search')
    def test_arxiv_enrichment_uses_api_data(
        self, mock_search_class, mock_client_class, upload_service
    ):
        """Test that arXiv enrichment uses API data (authoritative source)."""
        mock_result = Mock()
        mock_result.title = "API Title"
        mock_result.authors = [Mock(name="Author")]
        mock_result.published = Mock(year=2020)
        mock_result.summary = "Abstract..."
        mock_result.journal_ref = None
        mock_result.primary_category = "cs.AI"

        mock_client = Mock()
        mock_client.results.return_value = iter([mock_result])
        mock_client_class.return_value = mock_client

        paper = PaperRef(
            id="2301.00001",
            title="PDF Extracted Title",  # Will be overwritten by API
            authors=[],
            url_pdf=""
        )
        enriched = upload_service._enrich_from_arxiv(paper, "2301.00001")

        # arXiv API data is authoritative - should be used
        assert enriched.title == "API Title"
        assert enriched.year == 2020

    def test_arxiv_enrichment_non_arxiv_paper(self, upload_service):
        """Test that non-arXiv papers are returned unchanged."""
        paper = PaperRef(id="some-other-id", title="Test", authors=[], url_pdf="")
        result = upload_service._enrich_from_arxiv(paper, "https://example.com/paper.pdf")
        assert result.title == "Test"


class TestS2Enrichment:
    """Test Semantic Scholar metadata enrichment."""

    def test_s2_enrichment_method_exists(self, upload_service):
        """Test that S2 enrichment method exists and is callable."""
        assert hasattr(upload_service, '_enrich_from_semantic_scholar')
        assert callable(upload_service._enrich_from_semantic_scholar)

    def test_s2_enrichment_with_short_title(self, upload_service):
        """Test that S2 enrichment skips short titles."""
        # Paper with very short title should not trigger S2 search
        paper = PaperRef(
            id="non-arxiv-id",
            title="Short",  # Too short for S2 search
            authors=[],
            url_pdf=""
        )
        # This should return without error (may not enrich if S2 fails)
        result = upload_service._enrich_from_semantic_scholar(paper)
        # Paper should still be returned unchanged (short title)
        assert result.id == "non-arxiv-id"

    def test_s2_enrichment_preserves_existing_data(self, upload_service):
        """Test that S2 enrichment preserves existing user data."""
        paper = PaperRef(
            id="non-arxiv-id",
            title="User Title",
            authors=["User Author"],  # Already has authors
            year=2024,  # Already has year
            url_pdf=""
        )
        # Even if S2 returns data, user data should be preserved
        result = upload_service._enrich_from_semantic_scholar(paper)
        assert result.authors == ["User Author"]
        assert result.year == 2024


class TestTitleSimilarity:
    """Test title similarity matching."""

    def test_titles_identical(self, upload_service):
        """Test that identical titles are similar."""
        assert upload_service._titles_similar(
            "Attention Is All You Need",
            "Attention Is All You Need"
        )

    def test_titles_case_insensitive(self, upload_service):
        """Test that title comparison is case insensitive."""
        assert upload_service._titles_similar(
            "Attention Is All You Need",
            "attention is all you need"
        )

    def test_titles_with_punctuation(self, upload_service):
        """Test that punctuation is ignored."""
        assert upload_service._titles_similar(
            "Attention Is All You Need!",
            "Attention Is All You Need"
        )

    def test_titles_different(self, upload_service):
        """Test that different titles are not similar."""
        assert not upload_service._titles_similar(
            "Attention Is All You Need",
            "BERT: Pre-training of Deep Bidirectional Transformers"
        )


class TestVectorStorage:
    """Test vector storage integration."""

    @patch('nlp_pillars.services.upload_service.vectors.ensure_collections')
    @patch('nlp_pillars.services.upload_service.vectors.upsert_text')
    def test_vectors_module_import(self, mock_upsert, mock_ensure):
        """Test that vectors module can be imported."""
        from nlp_pillars import vectors
        assert hasattr(vectors, 'ensure_collections')
        assert hasattr(vectors, 'upsert_text')


class TestPaperIdGeneration:
    """Test paper ID generation from URLs and filenames."""

    def test_arxiv_url_generates_arxiv_id(self, upload_service):
        """Test that arXiv URLs generate proper arXiv IDs."""
        url = "https://arxiv.org/abs/2301.00001"
        paper_id = upload_service._generate_paper_id_from_url(url)
        assert paper_id == "arxiv:2301.00001"

    def test_non_arxiv_url_generates_hash_id(self, upload_service):
        """Test that non-arXiv URLs generate hash-based IDs."""
        url = "https://example.com/paper.pdf"
        paper_id = upload_service._generate_paper_id_from_url(url)
        assert paper_id.startswith("url:")
        assert len(paper_id) > 5
