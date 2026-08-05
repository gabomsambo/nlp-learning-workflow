"""Canonical paper-identifier parsing.

One home for "is this a real paper id, and what PDF does it point at?", because
getting it wrong is expensive and was: SearXNG results whose URL did not parse
used to be given a fabricated `searxng_<hash>` id, which downstream code turned
into `https://arxiv.org/pdf/searxng_078015.pdf`. That 404s, so the paper failed
at ingest — several stages after the point where the identifier was invented.

The rule this module exists to enforce: an identifier that cannot be resolved to
a downloadable PDF is not an identifier. Callers get `None` and drop the
candidate instead of enqueuing a row that is guaranteed to fail later.

Three call sites share these helpers: `tools/searxng_tool.py` (parsing search
results), `db.py` (rebuilding a PaperRef from a queue row) and
`orchestrator.py` (fetching a paper by id).
"""

import re
from typing import Optional

# arXiv ids come in two shapes and both are live on arxiv.org.
#   new style (2007-): 2301.12345 — 4-digit YYMM, then 4 or 5 digits
#   old style (pre-2007): cs/0501001, math.GT/0309136 — archive[.subject]/YYMMNNN
# A trailing version suffix (v1, v2, ...) is stripped: the versionless id is what
# `https://arxiv.org/pdf/<id>.pdf` and the arXiv API both accept.
_ARXIV_NEW = r"\d{4}\.\d{4,5}"
_ARXIV_OLD = r"[a-z-]+(?:\.[A-Z]{2})?/\d{7}"

ARXIV_ID_RE = re.compile(rf"^(?:{_ARXIV_NEW}|{_ARXIV_OLD})$")

_ARXIV_URL_RE = re.compile(
    rf"arxiv\.org/(?:abs|pdf)/({_ARXIV_NEW}|{_ARXIV_OLD})(?:v\d+)?",
    re.IGNORECASE,
)

# DOIs are `10.<registrant>/<suffix>`; the suffix may contain almost anything, so
# stop at whitespace and at the query/fragment separators a search engine adds.
_DOI_URL_RE = re.compile(r"(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s?#]+)", re.IGNORECASE)


def extract_arxiv_id(url: str) -> Optional[str]:
    """Return the versionless arXiv id in `url`, or None if there is not one."""
    if not url:
        return None
    match = _ARXIV_URL_RE.search(url)
    return match.group(1) if match else None


def extract_doi(url: str) -> Optional[str]:
    """Return the DOI in `url`, or None if there is not one."""
    if not url:
        return None
    match = _DOI_URL_RE.search(url)
    return match.group(1).rstrip(".").rstrip("/") if match else None


def is_arxiv_id(paper_id: Optional[str]) -> bool:
    """Report whether `paper_id` is a bare arXiv id, versioned or not."""
    if not paper_id:
        return False
    return bool(ARXIV_ID_RE.match(re.sub(r"v\d+$", "", paper_id.strip())))


def arxiv_pdf_url(paper_id: str) -> str:
    """Build the canonical arXiv PDF URL for a bare arXiv id.

    Only meaningful when `is_arxiv_id(paper_id)` — calling it on anything else is
    exactly the bug this module exists to prevent, so callers must check first.
    """
    versionless = re.sub(r"v\d+$", "", paper_id.strip())
    return f"https://arxiv.org/pdf/{versionless}.pdf"


def looks_like_pdf_url(url: Optional[str]) -> bool:
    """Report whether `url`'s path ends in `.pdf`.

    Deliberately stricter than a substring test: `'pdf' in url` matched things
    like `.../how-to-read-a-pdf-tutorial`, which is not a PDF.
    """
    if not url:
        return False
    path = url.split("?", 1)[0].split("#", 1)[0]
    return path.lower().endswith(".pdf")


def resolvable_pdf_url(
    paper_id: Optional[str], url_pdf: Optional[str]
) -> Optional[str]:
    """Best downloadable URL for a paper, or None if there is not one.

    A stored `url_pdf` wins; otherwise an arXiv id is enough to reconstruct one.
    Returning None is a real answer — it means this candidate cannot be ingested
    and should never have been queued.
    """
    if url_pdf:
        return url_pdf
    if is_arxiv_id(paper_id):
        return arxiv_pdf_url(paper_id)
    return None
