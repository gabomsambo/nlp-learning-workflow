"""Shared paper metadata resolution from arXiv and Semantic Scholar.

The upload path and the per-paper metadata refresh action both need the same
lookup rules. One implementation here; callers merge resolved values into an
existing row without blanking fields the APIs did not return.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import arxiv

from ..paper_ids import extract_arxiv_id, is_arxiv_id
from ..schemas import PaperRef

logger = logging.getLogger(__name__)


def extract_arxiv_id_from_hint(url_or_filename: str) -> Optional[str]:
    """Return an arXiv id from a URL, filename, or bare id string."""
    if not url_or_filename:
        return None
    from_url = extract_arxiv_id(url_or_filename)
    if from_url:
        return from_url
    match = re.search(r"(\d{4}\.\d{4,5})", url_or_filename)
    return match.group(1) if match else None


def arxiv_id_for_paper(paper: PaperRef) -> Optional[str]:
    """Best arXiv id for ``paper``, from its id or PDF URL."""
    if paper.id:
        bare = paper.id.replace("arxiv:", "", 1).strip()
        if is_arxiv_id(bare):
            return re.sub(r"v\d+$", "", bare)
    if paper.url_pdf:
        return extract_arxiv_id_from_hint(paper.url_pdf)
    return None


def enrich_from_arxiv(paper: PaperRef, url_or_filename: str) -> PaperRef:
    """Enrich paper metadata from arXiv when an arXiv id is present."""
    arxiv_id = extract_arxiv_id_from_hint(url_or_filename)
    if not arxiv_id:
        return paper

    logger.info("Detected arXiv ID: %s, fetching metadata...", arxiv_id)

    try:
        search = arxiv.Search(id_list=[arxiv_id])
        client = arxiv.Client()
        result = next(client.results(search))

        # For arXiv papers, API data is authoritative.
        paper.title = result.title
        paper.authors = [a.name for a in result.authors]
        paper.year = result.published.year
        paper.abstract = result.summary
        paper.venue = result.journal_ref or f"arXiv:{result.primary_category}"

        logger.info("Enriched from arXiv: %s...", paper.title[:50])

    except StopIteration:
        logger.warning("arXiv paper %s not found", arxiv_id)
    except Exception as e:
        logger.warning("arXiv enrichment failed: %s", e)

    return paper


def enrich_from_semantic_scholar(paper: PaperRef) -> PaperRef:
    """Best-effort enrichment from Semantic Scholar."""
    try:
        from ..tools.semantic_scholar_tool import SemanticScholarTool

        s2 = SemanticScholarTool()
        enriched = None

        if paper.id and re.match(r"\d{4}\.\d{4,5}", paper.id.replace("arxiv:", "")):
            arxiv_id = paper.id.replace("arxiv:", "")
            logger.info("Trying S2 lookup by arXiv ID: %s", arxiv_id)
            enriched = s2.get_paper(arxiv_id)

        if not enriched and paper.title and len(paper.title) > 10:
            logger.info("Trying S2 search by title: %s...", paper.title[:30])
            results = s2.search(paper.title, limit=1)
            if results and titles_similar(paper.title, results[0].title):
                enriched = results[0]

        if enriched:
            if not paper.authors and enriched.authors:
                paper.authors = enriched.authors
            if not paper.year and enriched.year:
                paper.year = enriched.year
            if not paper.abstract and enriched.abstract:
                paper.abstract = enriched.abstract
            if not paper.venue and enriched.venue:
                paper.venue = enriched.venue
            if enriched.citation_count:
                paper.citation_count = enriched.citation_count

            logger.info("Enriched from S2: citations=%s", paper.citation_count)

    except Exception as e:
        logger.warning(
            "Semantic Scholar enrichment failed (continuing without): %s", e
        )

    return paper


def titles_similar(title1: str, title2: str) -> bool:
    """Return whether two titles are similar enough to be the same paper."""
    clean1 = re.sub(r"[^\w\s]", "", title1.lower())
    clean2 = re.sub(r"[^\w\s]", "", title2.lower())
    words1 = set(clean1.split())
    words2 = set(clean2.split())
    if not words1 or not words2:
        return False
    overlap = len(words1 & words2) / max(len(words1), len(words2))
    return overlap > 0.7


def resolve_paper_metadata(paper: PaperRef, *, for_refresh: bool = False) -> PaperRef:
    """Re-resolve metadata for ``paper`` using the same path as upload.

    When ``for_refresh`` is true, enrichment starts from blank metadata fields so
    a failed lookup cannot masquerade as "already current" by echoing the stored row.
    """
    if for_refresh:
        resolved = PaperRef(
            id=paper.id,
            title="",
            authors=[],
            venue=None,
            year=None,
            url_pdf=paper.url_pdf,
            abstract=None,
        )
    else:
        resolved = paper.model_copy(deep=True)

    arxiv_id = arxiv_id_for_paper(paper)

    if arxiv_id:
        resolved = enrich_from_arxiv(resolved, arxiv_id)

    if (
        not arxiv_id
        or not resolved.authors
        or not resolved.year
        or not resolved.abstract
        or not resolved.venue
    ):
        resolved = enrich_from_semantic_scholar(resolved)

    return resolved


def has_resolvable_metadata_source(paper: PaperRef) -> bool:
    """Return whether this paper has any lookup path we can try."""
    if arxiv_id_for_paper(paper):
        return True
    if paper.title and len(paper.title.strip()) > 10:
        return True
    return False


def metadata_fields_resolved(resolved: PaperRef) -> bool:
    """Return whether a lookup produced any metadata worth applying."""
    return bool(
        (resolved.title and resolved.title.strip())
        or resolved.authors
        or (resolved.abstract and resolved.abstract.strip())
        or (resolved.venue and resolved.venue.strip())
        or resolved.year
    )
