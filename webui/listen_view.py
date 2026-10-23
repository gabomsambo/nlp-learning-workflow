"""Pure view helpers for the Listen page.

The episodes library and the studio share one URL. ``?paper=`` and ``?studio=1``
open the studio; ``?script=`` opens a stored episode on the library. Paper ids
contain ``:`` and are encoded in every link.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote, urlencode

_MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
_DATE_PREFIX = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def listen_href(
    *,
    studio: bool = False,
    paper: str = "",
    script: str = "",
    run: str = "",
) -> str:
    """Build a /podcast/ URL. Empty means the episodes library."""
    query: dict[str, str] = {}
    if studio:
        query["studio"] = "1"
    if paper:
        query["paper"] = paper
    if script:
        query["script"] = script
    if run:
        query["run"] = run
    if not query:
        return "/podcast/"
    return "/podcast/?" + urlencode(query)


def view_is_studio(params: Mapping[str, str]) -> bool:
    """Studio is explicit, or opened by a paper deep link or a run id.

    ``?script=`` stays on the library so a finished episode opens where you listen.
    """
    if str(params.get("studio") or "") == "1":
        return True
    if params.get("paper"):
        return True
    if params.get("run"):
        return True
    return False


def paper_page_href(paper_id: Optional[str]) -> Optional[str]:
    """Link to the paper page. Ids such as ``arxiv:1706.03762`` are encoded."""
    if not paper_id or not isinstance(paper_id, str):
        return None
    return "/papers/" + quote(paper_id, safe="")


def audio_href(file_name: Optional[str]) -> Optional[str]:
    """Public MP3 URL, or None when the name could escape the audio directory."""
    if not file_name or not isinstance(file_name, str):
        return None
    if "/" in file_name or "\\" in file_name or ".." in file_name:
        return None
    if any(ord(ch) < 32 for ch in file_name):
        return None
    return "/api/podcast/audio/" + quote(file_name, safe="")


def script_download_href(script_id: Optional[str]) -> Optional[str]:
    if not script_id or not isinstance(script_id, str):
        return None
    return "/api/podcast/" + quote(script_id, safe="") + "/download"


def format_duration(seconds: Any) -> str:
    """Clock label. Empty means there is no audio length to show."""
    if seconds is None or isinstance(seconds, bool):
        return ""
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return ""
    if total <= 0:
        return ""
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_created(value: Any) -> str:
    """Day month year from a datetime or an ISO prefix, without locale drift."""
    if value is None:
        return ""
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        month = int(value.month)
        if 1 <= month <= 12:
            return f"{int(value.day)} {_MONTHS[month - 1]} {int(value.year)}"
    match = _DATE_PREFIX.match(str(value))
    if not match:
        return ""
    year, month_s, day_s = match.groups()
    month = int(month_s)
    if not 1 <= month <= 12:
        return ""
    return f"{int(day_s)} {_MONTHS[month - 1]} {year}"


def episode_initial(title: Optional[str]) -> str:
    for ch in title or "":
        if ch.isalnum():
            return ch.upper()
    return "E"


def created_sort(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def build_episode(
    script: Any,
    paper_titles: Mapping[str, str],
    pillar_names: Mapping[str, str],
) -> Optional[dict[str, Any]]:
    """One library row. None when the script has no id to address."""
    script_id = getattr(script, "id", None)
    if not script_id:
        return None
    paper_id = getattr(script, "paper_id", "") or ""
    pillar_id = getattr(script, "pillar_id", "") or ""
    meta = getattr(script, "audio_metadata", None)
    file_name = getattr(meta, "file_name", None) if meta is not None else None
    duration = getattr(meta, "duration_seconds", None) if meta is not None else None
    title = getattr(script, "title", None) or "Untitled episode"
    paper_title = paper_titles.get(paper_id) or paper_id or "Unknown paper"
    created = getattr(script, "created_at", None)
    return {
        "id": str(script_id),
        "title": title,
        "initial": episode_initial(title),
        "paper_id": paper_id,
        "paper_title": paper_title,
        "paper_href": paper_page_href(paper_id),
        "pillar_id": pillar_id,
        "pillar_name": pillar_names.get(pillar_id) or pillar_id or "No pillar",
        "word_count": int(getattr(script, "word_count", 0) or 0),
        "created_label": format_created(created),
        "created_sort": created_sort(created),
        "duration_label": format_duration(duration),
        "audio_href": audio_href(file_name),
        "script_download_href": script_download_href(str(script_id)),
        "studio_href": listen_href(studio=True, script=str(script_id)),
    }


def group_episodes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Newest episode first, grouped by pillar. A pillar's place follows its newest episode."""
    ordered = sorted(rows, key=lambda row: str(row.get("created_sort") or ""), reverse=True)
    groups: list[dict[str, Any]] = []
    index: dict[str, dict[str, Any]] = {}
    for row in ordered:
        pillar_id = str(row.get("pillar_id") or "")
        if pillar_id not in index:
            index[pillar_id] = {
                "pillar_id": pillar_id,
                "pillar_name": row.get("pillar_name") or "No pillar",
                "episodes": [],
            }
            groups.append(index[pillar_id])
        index[pillar_id]["episodes"].append(dict(row))
    return groups


def catalog_json(papers: Sequence[Any], pillars: Sequence[Any]) -> str:
    """Paper and pillar names for the client library. ``<`` is escaped for a script tag."""
    payload = {
        "papers": {
            p.id: {"title": p.title or "", "pillar_id": getattr(p, "pillar_id", None) or ""}
            for p in papers
            if getattr(p, "id", None)
        },
        "pillars": {
            p.id: p.name
            for p in pillars
            if getattr(p, "id", None)
        },
    }
    return json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
