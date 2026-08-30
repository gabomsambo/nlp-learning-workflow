"""Per-paper metadata refresh: re-resolve and update stored fields in place."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .. import db
from ..schemas import PaperRef
from .paper_metadata import (
    has_resolvable_metadata_source,
    metadata_fields_resolved,
    resolve_paper_metadata,
)

METADATA_FIELDS = ("title", "authors", "abstract", "venue", "year")


class NoResolvableSourceError(Exception):
    """Raised when a paper has no metadata source we can query."""


class PaperNotFoundError(Exception):
    """Raised when the requested paper id does not exist."""


class MetadataRefreshError(Exception):
    """Raised when the database update fails."""


@dataclass
class FieldChange:
    field: str
    before: Any
    after: Any


@dataclass
class MetadataRefreshResult:
    paper_id: str
    changed: List[FieldChange] = field(default_factory=list)
    message: str = ""

    @property
    def updated(self) -> bool:
        return bool(self.changed)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return True


def _values_equal(field: str, left: Any, right: Any) -> bool:
    if field == "authors":
        left_list = left if isinstance(left, list) else []
        right_list = right if isinstance(right, list) else []
        return left_list == right_list
    return left == right


def _format_authors(authors: Any) -> str:
    if not isinstance(authors, list) or not authors:
        return "(none)"
    if len(authors) <= 3:
        return ", ".join(str(a) for a in authors)
    head = ", ".join(str(a) for a in authors[:3])
    return f"{head}, … ({len(authors)} authors)"


def _format_value(field: str, value: Any) -> str:
    if field == "authors":
        return _format_authors(value)
    if field == "abstract" and isinstance(value, str):
        if len(value) <= 120:
            return value
        return value[:117] + "..."
    if value is None or value == "":
        return "(empty)"
    return str(value)


def _build_message(changed: List[FieldChange]) -> str:
    if not changed:
        return "Metadata is already current; nothing was changed."
    lines = ["Updated metadata:"]
    for item in changed:
        lines.append(
            f"- {item.field}: {_format_value(item.field, item.before)}"
            f" → {_format_value(item.field, item.after)}"
        )
    return "\n".join(lines)


def refresh_paper_metadata(paper_id: str) -> MetadataRefreshResult:
    """Re-resolve one paper's metadata and persist only changed fields."""
    row = db.get_paper_row_by_id(paper_id)
    if not row:
        raise PaperNotFoundError(f"Paper '{paper_id}' was not found.")

    current = db.paper_ref_from_row(row)
    if not has_resolvable_metadata_source(current):
        raise NoResolvableSourceError(
            "This paper has no resolvable metadata source. "
            "Refresh needs an arXiv id or a title long enough to search Semantic Scholar."
        )

    resolved = resolve_paper_metadata(current, for_refresh=True)
    if not metadata_fields_resolved(resolved):
        raise NoResolvableSourceError(
            "Could not resolve metadata from arXiv or Semantic Scholar for this paper."
        )

    patch: Dict[str, Any] = {}
    changed: List[FieldChange] = []

    for name in METADATA_FIELDS:
        before = getattr(current, name)
        after = getattr(resolved, name)
        if not _has_value(after):
            continue
        if _values_equal(name, before, after):
            continue
        patch[name] = after
        changed.append(FieldChange(field=name, before=before, after=after))

    if not patch:
        return MetadataRefreshResult(
            paper_id=paper_id,
            changed=[],
            message="Metadata is already current; nothing was changed.",
        )

    if not db.update_paper_metadata(paper_id, patch):
        raise MetadataRefreshError(
            f"Could not save refreshed metadata for paper '{paper_id}'."
        )

    return MetadataRefreshResult(
        paper_id=paper_id,
        changed=changed,
        message=_build_message(changed),
    )
