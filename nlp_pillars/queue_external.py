"""Queue a Semantic Scholar paper on a pillar's discovery queue.

The map's "+" on a suggestion lands here. It only files a ``paper_queue`` row
for the next run to ingest, so it costs nothing: one paced Semantic Scholar
lookup, no model call. A paper with no arXiv id and no open PDF is refused
(``NoOpenPdfError``) rather than queued, because ingest could never download it
(see ``paper_ids``).
"""

import re
from typing import Dict

from .db import add_to_paper_queue
from .paper_ids import is_arxiv_id, resolvable_pdf_url
from .tools.semantic_scholar_tool import SemanticScholarTool

NO_OPEN_PDF = "no open PDF"

# A bare S2 paperId is 40 hex characters; the rest are S2's own id prefixes.
# Anything else would be interpolated into the request path, so it is refused.
_S2_ID_RE = re.compile(
    r"^(?:[0-9a-f]{40}"
    r"|(?:CorpusId|ARXIV|DOI|MAG|ACL|PMID|PMCID):[A-Za-z0-9._/()\-]+)$",
    re.IGNORECASE,
)


class InvalidS2IdError(ValueError):
    """The id is not a Semantic Scholar paper id."""


class S2PaperNotFoundError(Exception):
    """Semantic Scholar does not know this paper."""


class NoOpenPdfError(Exception):
    """The paper has no arXiv id and no open PDF, so it cannot be ingested."""


def normalise_s2_id(raw: str) -> str:
    value = (raw or "").strip()
    if not _S2_ID_RE.match(value) or ".." in value:
        raise InvalidS2IdError(
            "Expected a Semantic Scholar paper id: 40 hex characters or a "
            "CorpusId:/ARXIV:/DOI: prefixed id"
        )
    return value


def queue_external_paper(pillar_id: str, s2_id: str) -> Dict[str, str]:
    """Resolve ``s2_id`` and queue it for ``pillar_id``; no spend.

    Returns ``{"status": "queued" | "already_queued", "paper_id", "title"}``.
    ``SemanticScholarError`` and the database's exceptions propagate.
    """
    lookup = normalise_s2_id(s2_id)
    ref = SemanticScholarTool().get_paper(lookup)
    if ref is None:
        raise S2PaperNotFoundError(f"Semantic Scholar has no paper {lookup!r}")

    if not resolvable_pdf_url(ref.id, ref.url_pdf):
        raise NoOpenPdfError(
            f"{NO_OPEN_PDF}: {ref.title[:80]!r} has no arXiv id or open PDF "
            "that ingest could download"
        )

    added = add_to_paper_queue(pillar_id, [ref])
    # 0 means the paper is already in the library or already queued for this
    # pillar (add_to_paper_queue dedupes on both); the PDF check passed above.
    return {
        "status": "queued" if added else "already_queued",
        "paper_id": ref.id,
        "title": ref.title,
        "source": "arxiv" if is_arxiv_id(ref.id) else "url",
    }
