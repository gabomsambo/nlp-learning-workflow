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


def is_file_upload(paper_id: str) -> bool:
    """Return whether this id is an uploaded PDF Semantic Scholar cannot resolve."""
    return str(paper_id or "").startswith("file:")


def _reason(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    if " for url " in text:
        text = text.split(" for url ", 1)[0]
    return text[:180]


def _rows_from_batch(paper_id: str, batch, direction: str) -> List[PaperCitation]:
    rows: List[PaperCitation] = []
    for other, influential in batch or []:
        other_id = getattr(other, "id", None)
        if not other_id or other_id == paper_id:
            continue
        rows.append(
            PaperCitation(
                paper_id=paper_id,
                cited_paper_id=other_id,
                citation_direction=direction,
                is_influential=bool(influential),
                source="semantic_scholar",
            )
        )
    return rows


def citations_from_tool(
    paper_id: str,
    tool,
    *,
    limit: int = CITATION_LIMIT,
) -> List[PaperCitation]:
    """Outgoing references and incoming citations for one library paper.

    ``tool`` is a ``SemanticScholarTool``. A missing tool returns nothing.
    Papers the tool cannot see come back as an empty list from the tool
    itself (it treats 404 that way) and produce no rows. ``file:`` uploads
    are not sent: Semantic Scholar answers 404 for every one of them.
    """
    if tool is None or not paper_id or is_file_upload(paper_id):
        return []

    rows = _rows_from_batch(
        paper_id, tool.get_references(paper_id, limit=limit), "outgoing"
    )
    rows.extend(
        _rows_from_batch(
            paper_id, tool.get_citations(paper_id, limit=limit), "incoming"
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

    Returns ``(rows, failure_reasons)``. One call's failure is a reason
    string. The other calls are still made. An empty list of reasons means
    every call succeeded, including a paper Semantic Scholar does not know
    (the tool returns no rows for a 404). ``file:`` uploads are skipped and
    are not failures. ``paper_ids`` must already be the library: this
    function does not expand to the papers those citations name.
    """
    rows: List[PaperCitation] = []
    failures: List[str] = []
    if tool is None:
        return rows, failures
    calls = (
        ("get_references", "outgoing"),
        ("get_citations", "incoming"),
    )
    for paper_id in paper_ids:
        if not paper_id or is_file_upload(paper_id):
            continue
        for method, direction in calls:
            try:
                batch = getattr(tool, method)(paper_id, limit=limit)
            except Exception as exc:
                reason = _reason(exc)
                logger.warning(
                    "Citation fetch failed for %s via %s: %s",
                    paper_id,
                    method,
                    reason,
                )
                failures.append(reason)
                continue
            rows.extend(_rows_from_batch(paper_id, batch, direction))
    return rows, failures
