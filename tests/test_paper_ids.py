"""Tests for nlp_pillars.paper_ids.

These are the checks that stop a fabricated identifier from reaching the queue.
The bug they guard against was silent: an unparseable URL became
`searxng_<hash>`, which became `https://arxiv.org/pdf/searxng_078015.pdf`, which
404'd at ingest — several stages after the mistake was made.
"""

import pytest

from nlp_pillars.paper_ids import (
    arxiv_pdf_url,
    extract_arxiv_id,
    extract_doi,
    is_arxiv_id,
    looks_like_pdf_url,
    resolvable_pdf_url,
)


class TestExtractArxivId:
    @pytest.mark.parametrize("url,expected", [
        ("https://arxiv.org/abs/2301.12345", "2301.12345"),
        ("https://arxiv.org/abs/2301.1234", "2301.1234"),
        ("https://arxiv.org/abs/2301.12345v3", "2301.12345"),
        ("http://arxiv.org/pdf/2301.12345", "2301.12345"),
        ("https://arxiv.org/pdf/2301.12345v1.pdf", "2301.12345"),
        ("https://www.arxiv.org/abs/1706.03762", "1706.03762"),
        # Old-style ids are still live on arxiv.org and used to be unparseable.
        ("https://arxiv.org/abs/cs/0501001", "cs/0501001"),
        ("https://arxiv.org/abs/math.GT/0309136", "math.GT/0309136"),
    ])
    def test_recognised(self, url, expected):
        assert extract_arxiv_id(url) == expected

    @pytest.mark.parametrize("url", [
        "",
        "https://en.wikipedia.org/wiki/Transformer_(deep_learning)",
        "https://huggingface.co/learn/nlp-course/chapter1/4",
        "https://arxiv.org/list/cs.CL/recent",
        "https://example.com/arxiv.org/abs/notanid",
    ])
    def test_not_recognised(self, url):
        assert extract_arxiv_id(url) is None


class TestExtractDoi:
    @pytest.mark.parametrize("url,expected", [
        ("https://doi.org/10.1038/nature12345", "10.1038/nature12345"),
        ("https://dx.doi.org/10.1145/3292500.3330701", "10.1145/3292500.3330701"),
        ("https://doi.org/10.18653/v1/N19-1423?utm=x", "10.18653/v1/N19-1423"),
    ])
    def test_recognised(self, url, expected):
        assert extract_doi(url) == expected

    def test_not_recognised(self):
        assert extract_doi("https://example.com/paper") is None
        assert extract_doi("") is None


class TestIsArxivId:
    @pytest.mark.parametrize("value", ["2301.12345", "2301.12345v2", "cs/0501001", "math.GT/0309136"])
    def test_true(self, value):
        assert is_arxiv_id(value) is True

    @pytest.mark.parametrize("value", [
        None,
        "",
        # The exact shape the old fallback minted.
        "searxng_078015",
        # A DOI: has a dot and digits, which the orchestrator's old ad-hoc test
        # accepted, producing https://arxiv.org/pdf/10.1038/nature12345.pdf.
        "10.1038/nature12345",
        "test.12345",
    ])
    def test_false(self, value):
        assert is_arxiv_id(value) is False


class TestPdfUrls:
    def test_arxiv_pdf_url_strips_version(self):
        assert arxiv_pdf_url("2301.12345v7") == "https://arxiv.org/pdf/2301.12345.pdf"

    @pytest.mark.parametrize("url,expected", [
        ("https://arxiv.org/pdf/2301.12345.pdf", True),
        ("https://acl.org/paper.PDF?download=1", True),
        # The old `'pdf' in url.lower()` test called both of these PDFs.
        ("https://example.com/what-is-a-pdf-guide", False),
        ("https://arxiv.org/abs/2301.12345", False),
        (None, False),
    ])
    def test_looks_like_pdf_url(self, url, expected):
        assert looks_like_pdf_url(url) is expected

    def test_resolvable_prefers_stored_url(self):
        assert resolvable_pdf_url("2301.12345", "https://acl.org/x.pdf") == "https://acl.org/x.pdf"

    def test_resolvable_falls_back_to_arxiv(self):
        assert resolvable_pdf_url("2301.12345", None) == "https://arxiv.org/pdf/2301.12345.pdf"

    def test_resolvable_returns_none_when_nothing_works(self):
        """The whole point: no URL is a valid answer, not a cue to invent one."""
        assert resolvable_pdf_url("searxng_078015", None) is None
        assert resolvable_pdf_url("10.1038/nature12345", None) is None
