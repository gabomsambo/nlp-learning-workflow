"""Serialise discovery candidates for transport to the browser.

Lives on its own because two very different callers need the identical shape: the
worker thread that stores a finished discovery run's payload in
``pipeline_runs.result``, and — until this changed — the request handler that returned
it directly. One copy, so the page cannot be handed two subtly different candidate
shapes depending on which path produced it.

**This payload is not display-only, and that is the whole reason the truncation that
used to live here is gone.** The candidates the browser renders are the same objects
it posts back to ``/select``, which hands them to ``run_service._to_paper_refs`` ->
``orchestrator`` -> ``db.upsert_paper``. So anything shortened here is shortened in
``papers`` permanently. It used to cap abstracts at 300 characters and author lists at
3, both commented as display caps — and the candidates table renders neither field.
Measured on the captain's library: ``2403.05525`` (discovery-ingested) carries exactly
3 authors and a 303-character abstract cut mid-sentence, while a URL-uploaded paper in
the same pillar carries all 319 authors and the complete 1538-character abstract.

The rule now: **this module carries the record; the renderer decides what fits.** If a
future table wants a preview, truncate it in ``discovery.html`` at render time, where
the shortened string cannot escape into a database write. Do not reintroduce a cap
here.

Nothing here escapes anything, and it must not start: the browser renders every one of
these values with ``textContent`` / ``setAttribute``. Escaping on the way in would
double-encode a paper title with an ampersand in it, and would encourage the belief
that the data is safe to interpolate, which it is not.
"""

from typing import Any, Dict, List

from nlp_pillars.schemas import DiscoveryCandidate


def candidate_to_dict(candidate: DiscoveryCandidate) -> Dict[str, Any]:
    """One candidate, in the shape discovery.html renders and posts back.

    Every metadata field the source gave us, in full. ``venue`` is included for the
    same reason as the rest: it survives the round trip into ``papers.venue``, which
    was NULL on every discovery-ingested paper because this dict never carried it.
    """
    paper = candidate.paper

    return {
        "paper": {
            "id": paper.id,
            "title": paper.title,
            "authors": paper.authors or [],
            "venue": paper.venue,
            "year": paper.year,
            "abstract": paper.abstract,
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
