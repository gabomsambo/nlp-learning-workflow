"""View models for the library list and the paper page.

The templates render these and nothing else, so a paper id that contains
``:`` is encoded in one place and the href scheme check lives next to the
same rule in ``paper-modal.js``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import quote

_SCHEME = re.compile(r"^([a-z][a-z0-9+.-]*):", re.IGNORECASE)
_SAFE_SCHEMES = frozenset({"http", "https", "file"})
_PILLAR_CLASS = re.compile(r"^[a-z0-9-]+$")
_MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
_DIFFICULTY = {1: "Easy", 2: "Medium", 3: "Hard"}
_DIFFICULTY_VARIANT = {1: "ok", 2: "warn", 3: "danger"}


def paper_href(paper_id: str) -> str:
    """Return the paper page path with ``:`` and other reserved bytes encoded."""
    return "/papers/" + quote(str(paper_id), safe="")


def study_href(pillar_id: str | None) -> str:
    """Return the review page, narrowed to the paper's pillar when it has one."""
    if not pillar_id:
        return "/quiz/"
    return "/quiz/?pillar=" + quote(str(pillar_id), safe="")


def make_episode_href(paper_id: str) -> str:
    """Return the podcast page with this paper pre-selected."""
    return "/podcast/?paper=" + quote(str(paper_id), safe="")


def episode_href(script_id: str) -> str:
    """Return the Listen page opened on one stored script."""
    return "/podcast/?script=" + quote(str(script_id), safe="")


def safe_pdf_url(url: str | None) -> str | None:
    """Return ``url`` when it is safe to put in an href, else None.

    Relative URLs pass. Absolute ones must be http, https or file — uploaded
    PDFs are stored as ``file://`` and ``javascript:`` must never become a link.
    The same rule is ``isSafeHref`` in ``paper-modal.js``.
    """
    if not isinstance(url, str):
        return None
    text = url.strip()
    if not text:
        return None
    match = _SCHEME.match(text)
    if match is None:
        return text
    if match.group(1).lower() in _SAFE_SCHEMES:
        return text
    return None


def pillar_css_class(slug: str | None) -> str:
    """Return ``pillar-<slug>`` when the slug is a safe CSS class, else ``''``."""
    if slug and _PILLAR_CLASS.fullmatch(slug):
        return "pillar-" + slug
    return ""


def authors_line(authors: Any, limit: int = 3) -> str:
    """Return an author line. ``limit`` truncates with ``+N``; 0 keeps them all."""
    if not isinstance(authors, list):
        return ""
    names = [item.strip() for item in authors if isinstance(item, str) and item.strip()]
    if not names:
        return ""
    shown = names if limit <= 0 else names[:limit]
    extra = len(names) - len(shown)
    text = ", ".join(shown)
    if extra:
        text += f" +{extra}"
    return text


def humanise_when(value: Any, today: date | None = None) -> str:
    """Return today, yesterday, ``N days ago``, or ``15 Jan 2020``.

    The calendar date is the first ten characters of the stored timestamp, so
    the label does not depend on the machine's timezone. Anything that is not
    a date is returned unchanged.
    """
    if value is None or value == "":
        return ""
    text = str(value).strip()
    try:
        day = date.fromisoformat(text[:10])
    except ValueError:
        return text
    current = today or date.today()
    delta = (current - day).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "yesterday"
    if 2 <= delta < 7:
        return f"{delta} days ago"
    return f"{day.day} {_MONTHS[day.month - 1]} {day.year}"


def as_strings(value: Any) -> list[str]:
    """Return the string items of a list, dropping empties. Non-lists are []."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def difficulty_label(level: Any) -> str | None:
    """Map 1–3 to Easy, Medium, Hard. Anything else is omitted."""
    try:
        number = int(level)
    except (TypeError, ValueError):
        return None
    return _DIFFICULTY.get(number)


def difficulty_variant(level: Any) -> str:
    """Return the chip variant for a difficulty, or neutral when it is unknown."""
    try:
        number = int(level)
    except (TypeError, ValueError):
        return "neutral"
    return _DIFFICULTY_VARIANT.get(number, "neutral")


def confidence_label(score: Any) -> str | None:
    """Return ``84%`` for a 0–1 score, else None."""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    return f"{round(float(score) * 100)}%"


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


def _search_blob(*parts: Any) -> str:
    words: list[str] = []
    for part in parts:
        if part is None or part == "":
            continue
        words.append(str(part).lower())
    return " ".join(words)


def _citation_label(count: Any) -> str:
    if isinstance(count, bool) or count is None or count == "":
        return ""
    try:
        number = int(count)
    except (TypeError, ValueError):
        return ""
    suffix = "citation" if number == 1 else "citations"
    return f"{number} {suffix}"


@dataclass
class LibraryRow:
    """One title-first row on the library page."""

    id: str
    href: str
    title: str
    meta: str
    when: str
    pillar_id: str
    pillar_name: str
    pillar_class: str
    search: str
    has_lesson: bool
    has_cards: bool
    has_episode: bool
    concealed: bool


@dataclass
class PillarChip:
    """A pillar filter chip. ``id`` is empty for All."""

    id: str
    name: str
    href: str
    pillar_class: str
    current: bool


@dataclass
class LibraryView:
    """The library page: rows plus the chips that filter them."""

    rows: list[LibraryRow]
    chips: list[PillarChip]
    selected_pillar: str
    no_match: bool
    visible_count: int


def build_library(
    papers: list[dict[str, Any]] | None,
    pillars: Any,
    *,
    lesson_ids: set[str] | None = None,
    card_ids: set[str] | None = None,
    episode_ids: set[str] | None = None,
    selected_pillar: str | None = None,
    today: date | None = None,
) -> LibraryView:
    """Build the library list. ``selected_pillar`` hides every other row."""
    names = dict(_pillar_pairs(pillars))
    selected = selected_pillar or ""
    lessons = lesson_ids or set()
    cards = card_ids or set()
    episodes = episode_ids or set()
    rows: list[LibraryRow] = []
    seen: list[str] = []
    for paper in papers or []:
        paper_id = str(paper.get("id") or "")
        if not paper_id:
            continue
        pillar_id = str(paper.get("pillar_id") or "")
        if pillar_id and pillar_id not in seen:
            seen.append(pillar_id)
        pillar_name = names.get(pillar_id, pillar_id)
        title = str(paper.get("title") or "Untitled")
        author_text = authors_line(paper.get("authors"))
        year = paper.get("year")
        citations = _citation_label(paper.get("citation_count"))
        meta_parts = [part for part in (author_text, year, citations) if part]
        rows.append(
            LibraryRow(
                id=paper_id,
                href=paper_href(paper_id),
                title=title,
                meta=" · ".join(str(part) for part in meta_parts),
                when=humanise_when(paper.get("added_at"), today=today),
                pillar_id=pillar_id,
                pillar_name=pillar_name,
                pillar_class=pillar_css_class(pillar_id),
                search=_search_blob(title, author_text, paper_id, pillar_name, year),
                has_lesson=paper_id in lessons,
                has_cards=paper_id in cards,
                has_episode=paper_id in episodes,
                concealed=bool(selected) and pillar_id != selected,
            )
        )
    chips = [
        PillarChip(
            id="",
            name="All",
            href="/papers/",
            pillar_class="",
            current=not selected,
        )
    ]
    ordered = list(names.items())
    known = {pid for pid, _name in ordered}
    for pid in seen:
        if pid not in known:
            ordered.append((pid, pid))
    if selected and selected not in {pid for pid, _name in ordered}:
        ordered.append((selected, names.get(selected, selected)))
    for pid, name in ordered:
        chips.append(
            PillarChip(
                id=pid,
                name=name,
                href="/papers/?pillar=" + quote(pid, safe=""),
                pillar_class=pillar_css_class(pid),
                current=pid == selected,
            )
        )
    no_match = bool(rows) and all(row.concealed for row in rows)
    visible = sum(1 for row in rows if not row.concealed)
    return LibraryView(
        rows=rows,
        chips=chips,
        selected_pillar=selected,
        no_match=no_match,
        visible_count=visible,
    )


@dataclass
class NoteView:
    """One research-note record."""

    problem: str
    method: str
    findings: list[str]
    limitations: list[str]
    future_work: list[str]
    key_terms: list[str]
    related_papers: list[str]
    confidence: str | None

    @property
    def has_body(self) -> bool:
        """True when the note has anything to show."""
        return bool(
            self.problem
            or self.method
            or self.findings
            or self.limitations
            or self.future_work
            or self.key_terms
            or self.related_papers
            or self.confidence
        )


@dataclass
class LessonView:
    """One lesson, in reading order."""

    tl_dr: str
    takeaways: list[str]
    practice_ideas: list[str]
    connections: list[str]
    difficulty_label: str | None
    difficulty_variant: str
    minutes: int | None


@dataclass
class CardView:
    """One quiz card. The answer is in the page; the toggle hides it."""

    question: str
    answer: str
    difficulty_label: str | None
    difficulty_variant: str


@dataclass
class EpisodeView:
    """One stored script, linked to Listen."""

    id: str
    title: str
    href: str
    words: str
    when: str


@dataclass
class PaperPage:
    """Everything the paper page renders for one paper."""

    id: str
    title: str
    authors: str
    year: str
    venue: str
    citations: str
    pillar_id: str
    pillar_name: str
    pillar_class: str
    pillar_href: str
    added: str
    processed: bool
    pdf_href: str | None
    pdf_unsafe: str
    abstract: str
    tldr: str
    summary: NoteView | None
    notes: list[NoteView]
    lessons: list[LessonView]
    cards: list[CardView]
    episodes: list[EpisodeView]
    episodes_unavailable: bool
    study_href: str
    make_episode_href: str
    first_question: str


def _note_view(raw: dict[str, Any]) -> NoteView:
    return NoteView(
        problem=str(raw.get("problem") or ""),
        method=str(raw.get("method") or ""),
        findings=as_strings(raw.get("findings")),
        limitations=as_strings(raw.get("limitations")),
        future_work=as_strings(raw.get("future_work")),
        key_terms=as_strings(raw.get("key_terms")),
        related_papers=as_strings(raw.get("related_papers")),
        confidence=confidence_label(raw.get("confidence_score")),
    )


def _lesson_view(raw: dict[str, Any]) -> LessonView:
    minutes = raw.get("estimated_time")
    try:
        minutes_int = int(minutes) if minutes not in (None, "") else None
    except (TypeError, ValueError):
        minutes_int = None
    return LessonView(
        tl_dr=str(raw.get("tl_dr") or ""),
        takeaways=as_strings(raw.get("takeaways")),
        practice_ideas=as_strings(raw.get("practice_ideas")),
        connections=as_strings(raw.get("connections")),
        difficulty_label=difficulty_label(raw.get("difficulty")),
        difficulty_variant=difficulty_variant(raw.get("difficulty")),
        minutes=minutes_int,
    )


def _card_view(raw: dict[str, Any]) -> CardView:
    return CardView(
        question=str(raw.get("question") or ""),
        answer=str(raw.get("answer") or ""),
        difficulty_label=difficulty_label(raw.get("difficulty")),
        difficulty_variant=difficulty_variant(raw.get("difficulty")),
    )


def _episode_view(raw: dict[str, Any], today: date | None) -> EpisodeView | None:
    script_id = str(raw.get("id") or "")
    if not script_id:
        return None
    words = raw.get("word_count")
    try:
        words_label = f"{int(words):,} words" if words not in (None, "") else ""
    except (TypeError, ValueError):
        words_label = ""
    return EpisodeView(
        id=script_id,
        title=str(raw.get("title") or "Episode"),
        href=episode_href(script_id),
        words=words_label,
        when=humanise_when(raw.get("created_at"), today=today),
    )


def build_paper_page(
    details: dict[str, Any],
    episodes: list[dict[str, Any]] | None,
    pillars: Any,
    *,
    episodes_unavailable: bool = False,
    today: date | None = None,
) -> PaperPage:
    """Build the paper page from a ``/papers/details`` payload plus scripts."""
    paper = details.get("paper") or {}
    paper_id = str(paper.get("id") or "")
    names = dict(_pillar_pairs(pillars))
    pillar_id = str(paper.get("pillar_id") or "")
    raw_notes = details.get("notes") or []
    notes = [_note_view(item) for item in raw_notes if isinstance(item, dict)]
    notes = [note for note in notes if note.has_body]
    lessons = [
        _lesson_view(item)
        for item in (details.get("lessons") or [])
        if isinstance(item, dict)
    ]
    cards = [
        _card_view(item)
        for item in (details.get("quiz_cards") or [])
        if isinstance(item, dict)
    ]
    episode_views = []
    for raw in episodes or []:
        if isinstance(raw, dict):
            view = _episode_view(raw, today)
            if view is not None:
                episode_views.append(view)
    tldr = next((lesson.tl_dr for lesson in lessons if lesson.tl_dr), "")
    summary = notes[0] if notes else None
    pdf = paper.get("url_pdf")
    safe = safe_pdf_url(pdf if isinstance(pdf, str) else None)
    year = paper.get("year")
    return PaperPage(
        id=paper_id,
        title=str(paper.get("title") or "Untitled"),
        authors=authors_line(paper.get("authors"), limit=0),
        year=str(year) if year else "",
        venue=str(paper.get("venue") or ""),
        citations=_citation_label(paper.get("citation_count")),
        pillar_id=pillar_id,
        pillar_name=names.get(pillar_id, pillar_id),
        pillar_class=pillar_css_class(pillar_id),
        pillar_href=(
            "/papers/?pillar=" + quote(pillar_id, safe="") if pillar_id else "/papers/"
        ),
        added=humanise_when(paper.get("added_at"), today=today),
        processed=bool(paper.get("processed")),
        pdf_href=safe,
        pdf_unsafe="" if safe or not isinstance(pdf, str) else pdf.strip(),
        abstract=str(paper.get("abstract") or ""),
        tldr=tldr,
        summary=summary,
        notes=notes,
        lessons=lessons,
        cards=cards,
        episodes=episode_views,
        episodes_unavailable=episodes_unavailable,
        study_href=study_href(pillar_id),
        make_episode_href=make_episode_href(paper_id),
        first_question=cards[0].question if cards else "",
    )
