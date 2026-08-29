"""Serialise discovery candidates for transport to the browser.

Lives on its own because two very different callers need the identical shape: the
worker thread that stores a finished discovery run's payload in
``pipeline_runs.result``, and — until this changed — the request handler that returned
it directly. One copy, so the page cannot be handed two subtly different candidate
shapes depending on which path produced it.

Nothing here escapes anything, and it must not start: the browser renders every one of
these values with ``textContent`` / ``setAttribute``. Escaping on the way in would
double-encode a paper title with an ampersand in it, and would encourage the belief
that the data is safe to interpolate, which it is not.
"""

from typing import Any, Dict, List

from nlp_pillars.schemas import DiscoveryCandidate

#: Abstracts are shown as a preview, not read in full, and a discovery run carries up
#: to 50 of them into a JSONB column and then over the wire on every poll.
_ABSTRACT_CHARS = 300

#: How many authors the table shows.
_MAX_AUTHORS = 3


def candidate_to_dict(candidate: DiscoveryCandidate) -> Dict[str, Any]:
    """One candidate, in the shape discovery.html renders."""
    paper = candidate.paper
    abstract = paper.abstract
    if abstract and len(abstract) > _ABSTRACT_CHARS:
        abstract = abstract[:_ABSTRACT_CHARS] + "..."

    return {
        "paper": {
            "id": paper.id,
            "title": paper.title,
            "authors": paper.authors[:_MAX_AUTHORS] if paper.authors else [],
            "year": paper.year,
            "abstract": abstract,
            "url_pdf": paper.url_pdf,
            "citation_count": paper.citation_count,
        },
        "source": candidate.source,
        "relevance_score": round(candidate.relevance_score, 3),
        "citation_count": candidate.citation_count,
        "is_influential": candidate.is_influential,
    }


def candidates_payload(candidates: List[DiscoveryCandidate]) -> Dict[str, Any]:
    """The whole result of a discovery run, as stored on the run row.

    ``sources_used`` is sorted rather than set-ordered: it is rendered to a human, and
    a list that reshuffles itself between two runs of the same pillar reads as change
    where there is none.
    """
    return {
        "candidates": [candidate_to_dict(c) for c in candidates],
        "sources_used": sorted({c.source for c in candidates}),
        "total_found": len(candidates),
    }
