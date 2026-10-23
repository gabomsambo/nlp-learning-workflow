"""Semantic Scholar citations for papers already in the library.

The client in ``SemanticScholarTool`` waits between requests. This module does
not add a second sleep and does not follow citations of papers that are not
in the library: those ids are edge endpoints only.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence

from . import db
from .schemas import PaperCitation

logger = logging.getLogger(__name__)

#: Matches the cap ``Orchestrator._fetch_and_store_citations`` used. The tool's
#: own default is 50; the library fetch stays smaller so a rebuild does not
#: page the whole citation list of a famous paper.
CITATION_LIMIT = 20


def citations_from_tool(
    paper_id: str,
    tool,
    *,
    limit: int = CITATION_LIMIT,
) -> List[PaperCitation]:
    """Outgoing references and incoming citations for one library paper.

    ``tool`` is a ``SemanticScholarTool``. A missing tool returns nothing.
    Papers the tool cannot see come back as an empty list from the tool
    itself (it treats 404 that way) and produce no rows.
    """
    if tool is None or not paper_id:
        return []

    rows: List[PaperCitation] = []
    for ref_paper, influential in tool.get_references(paper_id, limit=limit):
        cited_id = getattr(ref_paper, "id", None)
        if not cited_id or cited_id == paper_id:
            continue
        rows.append(
            PaperCitation(
                paper_id=paper_id,
                cited_paper_id=cited_id,
                citation_direction="outgoing",
                is_influential=bool(influential),
                source="semantic_scholar",
            )
        )
    for citing_paper, influential in tool.get_citations(paper_id, limit=limit):
        citing_id = getattr(citing_paper, "id", None)
        if not citing_id or citing_id == paper_id:
            continue
        rows.append(
            PaperCitation(
                paper_id=paper_id,
                cited_paper_id=citing_id,
                citation_direction="incoming",
                is_influential=bool(influential),
                source="semantic_scholar",
            )
        )
    return rows


def fetch_and_store_citations(
    paper_id: str,
    tool,
    *,
    limit: int = CITATION_LIMIT,
    store: Optional[Callable[[Sequence[PaperCitation]], int]] = None,
) -> int:
    """Fetch one library paper's citations and insert them.

    Duplicates are skipped by ``add_paper_citations`` (the unique index on
    ``paper_citations``). A store that does not return an int counts as
    nothing stored, so a test double cannot break the caller with a bad
    return. The tool's exceptions propagate; callers that must keep going
    catch them.
    """
    rows = citations_from_tool(paper_id, tool, limit=limit)
    if not rows:
        return 0
    save = store if store is not None else db.add_paper_citations
    added = save(rows)
    if isinstance(added, int):
        return added
    logger.warning(
        "Citation store for %s returned %r; treating it as 0 new rows",
        paper_id,
        added,
    )
    return 0


def fetch_library_citations(
    paper_ids: Sequence[str], tool, *, limit: int = CITATION_LIMIT
):
    """Fetch citations for these ids only.

    Returns ``(rows, warnings)``. One paper's failure is a warning. The other
    papers are still fetched. ``paper_ids`` must already be the library:
    this function does not expand to the papers those citations name.
    """
    rows: List[PaperCitation] = []
    warnings: List[str] = []
    if tool is None:
        return rows, warnings
    for paper_id in paper_ids:
        if not paper_id:
            continue
        try:
            rows.extend(citations_from_tool(paper_id, tool, limit=limit))
        except Exception as exc:
            logger.warning("Citation fetch failed for %s: %s", paper_id, exc)
            warnings.append(f"{paper_id}: {exc}")
    return rows, warnings
