"""Tests for per-paper metadata refresh."""

from unittest.mock import patch

import pytest

from nlp_pillars.schemas import PaperRef
from nlp_pillars.services.paper_metadata_refresh import (
    MetadataRefreshError,
    NoResolvableSourceError,
    PaperNotFoundError,
    refresh_paper_metadata,
)


@pytest.fixture
def thin_paper_row():
    return {
        "id": "2403.05525",
        "pillar_id": "neural-architectures-language",
        "title": "DeepSeek-VL",
        "authors": ["Haoyu Lu", "Wen Liu", "Bo Zhang"],
        "venue": None,
        "year": 2024,
        "url_pdf": "https://arxiv.org/pdf/2403.05525v2",
        "abstract": "real-world scenarios including...",
        "citation_count": 0,
    }


@pytest.fixture
def full_paper():
    return PaperRef(
        id="2403.05525",
        title="DeepSeek-VL: Towards Real-World Vision-Language Understanding",
        authors=["Haoyu Lu", "Wen Liu", "Bo Zhang", "Fourth Author"],
        venue="arXiv:cs.CV",
        year=2024,
        url_pdf="https://arxiv.org/pdf/2403.05525v2",
        abstract="A complete abstract that is much longer than the truncated row.",
    )


class TestRefreshPaperMetadata:
    @patch("nlp_pillars.services.paper_metadata_refresh.db.update_paper_metadata")
    @patch("nlp_pillars.services.paper_metadata_refresh.resolve_paper_metadata")
    @patch("nlp_pillars.services.paper_metadata_refresh.db.get_paper_row_by_id")
    def test_refresh_updates_changed_fields(
        self, mock_get_row, mock_resolve, mock_update, thin_paper_row, full_paper
    ):
        mock_get_row.return_value = thin_paper_row
        mock_resolve.return_value = full_paper
        mock_update.return_value = True

        result = refresh_paper_metadata("2403.05525")

        assert result.updated is True
        changed_fields = {item.field for item in result.changed}
        assert "authors" in changed_fields
        assert "abstract" in changed_fields
        assert "venue" in changed_fields
        mock_update.assert_called_once()
        patch = mock_update.call_args[0][1]
        assert patch["authors"] == full_paper.authors
        assert "id" not in patch
        assert "url_pdf" not in patch

    @patch("nlp_pillars.services.paper_metadata_refresh.resolve_paper_metadata")
    @patch("nlp_pillars.services.paper_metadata_refresh.db.get_paper_row_by_id")
    def test_refresh_reports_already_current(
        self, mock_get_row, mock_resolve, thin_paper_row, full_paper
    ):
        current = PaperRef(**full_paper.model_dump())
        row = dict(thin_paper_row)
        row.update(full_paper.model_dump())
        mock_get_row.return_value = row
        mock_resolve.return_value = current

        result = refresh_paper_metadata("2403.05525")

        assert result.updated is False
        assert "already current" in result.message.lower()

    @patch("nlp_pillars.services.paper_metadata_refresh.db.get_paper_row_by_id")
    def test_refresh_missing_paper(self, mock_get_row):
        mock_get_row.return_value = None
        with pytest.raises(PaperNotFoundError):
            refresh_paper_metadata("missing-paper")

    @patch("nlp_pillars.services.paper_metadata_refresh.db.get_paper_row_by_id")
    def test_refresh_no_resolvable_source(self, mock_get_row):
        mock_get_row.return_value = {
            "id": "custom-hash",
            "pillar_id": "neural-architectures-language",
            "title": "Short",
            "authors": [],
            "venue": None,
            "year": None,
            "url_pdf": None,
            "abstract": None,
            "citation_count": 0,
        }
        with pytest.raises(NoResolvableSourceError):
            refresh_paper_metadata("custom-hash")

    @patch("nlp_pillars.services.paper_metadata_refresh.db.update_paper_metadata")
    @patch("nlp_pillars.services.paper_metadata_refresh.resolve_paper_metadata")
    @patch("nlp_pillars.services.paper_metadata_refresh.db.get_paper_row_by_id")
    def test_refresh_does_not_blank_existing_fields(
        self, mock_get_row, mock_resolve, mock_update, thin_paper_row
    ):
        mock_get_row.return_value = thin_paper_row
        resolved = PaperRef(
            id="2403.05525",
            title="DeepSeek-VL",
            authors=[],
            venue=None,
            year=2024,
            url_pdf=thin_paper_row["url_pdf"],
            abstract=None,
        )
        mock_resolve.return_value = resolved

        result = refresh_paper_metadata("2403.05525")

        assert result.updated is False
        mock_update.assert_not_called()

    @patch("nlp_pillars.services.paper_metadata_refresh.db.update_paper_metadata")
    @patch("nlp_pillars.services.paper_metadata_refresh.resolve_paper_metadata")
    @patch("nlp_pillars.services.paper_metadata_refresh.db.get_paper_row_by_id")
    def test_refresh_db_failure(
        self, mock_get_row, mock_resolve, mock_update, thin_paper_row, full_paper
    ):
        mock_get_row.return_value = thin_paper_row
        mock_resolve.return_value = full_paper
        mock_update.return_value = False

        with pytest.raises(MetadataRefreshError):
            refresh_paper_metadata("2403.05525")
