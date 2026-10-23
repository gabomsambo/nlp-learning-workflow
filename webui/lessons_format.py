"""View model for the lessons reading list.

Lessons have no stored title. ``_dict_to_lesson`` invents ``Lesson from {id}``
and keeps the TL;DR as the sentence. The list uses that sentence as the
headline, and the same invented title only when the TL;DR is missing.

The paper page's Lesson tab is the reading surface. This module only builds
the list that links there, with the paper id encoded the same way as the
library (``paper_href``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping
from urllib.parse import quote

from webui.library_format import (
    PillarChip,
    as_strings,
    difficulty_label,
    difficulty_variant,
    humanise_when,
    paper_href,
    pillar_css_class,
)

_PREVIEW = 2


def lesson_href(paper_id: str) -> str:
    """Return the paper page opened on its Lesson tab.

    ``#lesson`` matches the section id on the paper page. The id is encoded
    first, so a colon in ``arxiv:…`` is not left for the browser to guess.
    """
    return paper_href(paper_id) + "#lesson"


def _with_pillar(path: str, pillar: str) -> str:
    if not pillar:
        return path
    return path + "?pillar=" + quote(pillar, safe="")


def _calendar_day(value: Any) -> str:
    if value is None or value == "":
        return ""
    text = str(value).strip()
    try:
        date.fromisoformat(text[:10])
    except ValueError:
        return ""
    return text[:10]


def reading_minutes(value: Any) -> int | None:
    """Return a reading time in minutes, or None when it was not stored."""
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def difficulty_number(level: Any) -> str:
    """Return the stored difficulty as text. Unknown shapes are kept visible."""
    if isinstance(level, bool) or level in (None, ""):
        return ""
    try:
        return str(int(level))
    except (TypeError, ValueError):
        return str(level).strip()


def lesson_headline(tl_dr: Any, paper_id: str) -> str:
    """Return the TL;DR, or the invented title when that sentence is missing."""
    text = str(tl_dr or "").strip()
    if text:
        return text
    if paper_id:
        return f"Lesson from {paper_id}"
    return "Lesson"


def _count_label(count: int) -> str:
    if count == 1:
        return "1 lesson"
    if count:
        return f"{count} lessons"
    return ""


def _pillar_pairs(pillars: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for pillar in pillars or []:
        if isinstance(pillar, dict):
            pid, name = pillar.get("id"), pillar.get("name")
        else:
            pid, name = getattr(pillar, "id", None), getattr(pillar, "name", None)
        if pid:
            pairs.append((str(pid), str(name or pid)))
    return pairs


@dataclass
class LessonRow:
    """One lesson on the reading list."""

    title: str
    href: str
    when: str
    when_iso: str
    paper_id: str
    paper_title: str
    pillar_id: str
    pillar_name: str
    pillar_class: str
    difficulty_number: str
    difficulty_label: str | None
    difficulty_variant: str
    minutes: int | None
    time_compact: str
    preview: list[str]
    more_takeaways: int


@dataclass
class LessonsView:
    """The reading list plus the pillar chips that filter it."""

    rows: list[LessonRow]
    chips: list[PillarChip]
    selected_pillar: str
    count_label: str
    add_href: str
    papers_href: str
    empty_title: str
    empty_message: str


def _chips(names: dict[str, str], seen: list[str], selected: str) -> list[PillarChip]:
    chips = [
        PillarChip(
            id="",
            name="All",
            href="/lessons/",
            pillar_class="",
            current=not selected,
        )
    ]
    ordered = list(names.items())
    known = {pid for pid, _name in ordered}
    for pid in seen:
        if pid not in known:
            ordered.append((pid, names.get(pid, pid)))
    if selected and selected not in {pid for pid, _name in ordered}:
        ordered.append((selected, names.get(selected, selected)))
    for pid, name in ordered:
        chips.append(
            PillarChip(
                id=pid,
                name=name,
                href="/lessons/?pillar=" + quote(pid, safe=""),
                pillar_class=pillar_css_class(pid),
                current=pid == selected,
            )
        )
    return chips


def build_lesson_list(
    lessons: list[dict[str, Any]] | None,
    paper_titles: Mapping[str, str] | None,
    pillars: Any,
    *,
    selected_pillar: str | None = None,
    today: date | None = None,
) -> LessonsView:
    """Build the reading list. ``selected_pillar`` keeps only that pillar."""
    names = dict(_pillar_pairs(pillars))
    titles = paper_titles or {}
    selected = selected_pillar or ""
    rows: list[LessonRow] = []
    seen: list[str] = []
    for raw in lessons or []:
        if not isinstance(raw, dict):
            continue
        pillar_id = str(raw.get("pillar_id") or "")
        if selected and pillar_id != selected:
            continue
        if pillar_id and pillar_id not in seen:
            seen.append(pillar_id)
        paper_id = str(raw.get("paper_id") or "")
        if paper_id in titles:
            paper_title = str(titles[paper_id] or "").strip() or "Untitled"
        else:
            paper_title = paper_id or "Untitled"
        takeaways = as_strings(raw.get("takeaways"))
        minutes = reading_minutes(raw.get("estimated_time"))
        rows.append(
            LessonRow(
                title=lesson_headline(raw.get("tl_dr"), paper_id),
                href=lesson_href(paper_id) if paper_id else "",
                when=humanise_when(raw.get("created_at"), today=today),
                when_iso=_calendar_day(raw.get("created_at")),
                paper_id=paper_id,
                paper_title=paper_title,
                pillar_id=pillar_id,
                pillar_name=names.get(pillar_id, pillar_id),
                pillar_class=pillar_css_class(pillar_id),
                difficulty_number=difficulty_number(raw.get("difficulty")),
                difficulty_label=difficulty_label(raw.get("difficulty")),
                difficulty_variant=difficulty_variant(raw.get("difficulty")),
                minutes=minutes,
                time_compact="" if minutes is None else f"{minutes}m",
                preview=takeaways[:_PREVIEW],
                more_takeaways=max(0, len(takeaways) - _PREVIEW),
            )
        )
    if selected:
        empty_title = "No lessons in this pillar"
        empty_message = (
            "Nothing here has a lesson yet. Add a paper to this pillar, "
            "or reset the filter to see the others."
        )
    else:
        empty_title = "No lessons yet"
        empty_message = (
            "Lessons show up here after a paper is summarised. "
            "Add a paper to start one."
        )
    return LessonsView(
        rows=rows,
        chips=_chips(names, seen, selected),
        selected_pillar=selected,
        count_label=_count_label(len(rows)),
        add_href=_with_pillar("/discovery/", selected),
        papers_href=_with_pillar("/papers/", selected),
        empty_title=empty_title,
        empty_message=empty_message,
    )
