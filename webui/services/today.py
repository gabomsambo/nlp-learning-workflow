"""Today: the home page assembled from data the app already stores.

Nothing here starts a run, a review, or a generation. Plan rows point at the
pages that already do that work. A failed read is a missing value, never a
zero — a streak of 0 and a database that did not answer are different facts.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Optional, Sequence
from urllib.parse import quote

from nlp_pillars.config import PILLAR_CONFIGS
from nlp_pillars.db import get_pillars_or_empty

from .postgrest_client import PostgrestClient, QuickStats
from .review_session import json_for_script

logger = logging.getLogger(__name__)

WEEKDAY_LETTERS = ("M", "T", "W", "T", "F", "S", "S")
_QUIZ_LIMIT = 2000
_REVIEW_LIMIT = 2000
_RECENT_HOURS = 24
_RECENT_CAP = 8

_KIND_NOUN = {
    "discover": "Discovery",
    "upload": "Upload",
    "process_selected": "Processing",
    "run_daily": "Daily run",
    "podcast_script": "Script",
    "podcast_audio": "Episode audio",
}


def greeting(hour: int) -> str:
    """Time-of-day line for an hour already in the zone being shown.

    The page script replaces the rendered greeting and date with the browser's
    local clock, which is the clock the streak dots use. This function is the
    value in the HTML before that script runs.
    """
    if hour < 12:
        return "Good morning."
    if hour < 17:
        return "Good afternoon."
    return "Good evening."


def parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def paper_href(paper_id: str) -> str:
    """Paper page. Ids contain ':' and must be encoded so the path stays one segment."""
    return "/papers/" + quote(str(paper_id), safe="")


def postgrest_in(ids: Sequence[str]) -> str:
    """An `in` filter whose values may contain colons."""
    parts = []
    for raw in ids:
        escaped = str(raw).replace('"', '""')
        parts.append(f'"{escaped}"')
    return "in.(" + ",".join(parts) + ")"


def pillar_label(slug: Optional[str], names: dict[str, str]) -> str:
    """A name a person can read. Never the slug."""
    if not slug:
        return "your library"
    if slug in names and names[slug]:
        return names[slug]
    config = PILLAR_CONFIGS.get(slug)
    if config and config.get("name"):
        return str(config["name"])
    return "Untitled pillar"


@dataclass(frozen=True)
class WeekDot:
    label: str
    iso: str
    reviewed: bool
    is_today: bool
    is_future: bool


@dataclass(frozen=True)
class Streak:
    count: int
    dots: tuple[WeekDot, ...]

    @property
    def aria(self) -> str:
        done = [dot.label for dot in self.dots if dot.reviewed]
        if self.count == 1:
            head = "1 day in a row"
        else:
            head = f"{self.count} days in a row"
        if done:
            return head + ". Reviewed " + ", ".join(done) + " this week."
        return head + ". No reviews yet this week."


def streak_from_timestamps(
    timestamps: Iterable[Any], *, now: datetime, tz: datetime.tzinfo
) -> Streak:
    """Consecutive local-calendar days with at least one review.

    Today counts once a review exists. If today has none yet, yesterday still
    keeps the streak — it breaks only after a full local day with no review.
    The dots are the Monday–Sunday week that contains ``now``.
    """
    days: set[date] = set()
    for raw in timestamps:
        parsed = parse_dt(raw if not isinstance(raw, dict) else raw.get("review_timestamp"))
        if parsed is None:
            continue
        days.add(parsed.astimezone(tz).date())
    today = now.astimezone(tz).date()
    monday = today - timedelta(days=today.weekday())
    dots = []
    for offset in range(7):
        day = monday + timedelta(days=offset)
        dots.append(WeekDot(
            label=WEEKDAY_LETTERS[offset],
            iso=day.isoformat(),
            reviewed=day in days,
            is_today=day == today,
            is_future=day > today,
        ))
    cursor = today if today in days else today - timedelta(days=1)
    count = 0
    if cursor in days:
        while cursor in days:
            count += 1
            cursor -= timedelta(days=1)
    return Streak(count=count, dots=tuple(dots))


def reviewed_today(timestamps: Iterable[Any], *, now: datetime, tz: datetime.tzinfo) -> bool:
    today = now.astimezone(tz).date()
    for raw in timestamps:
        parsed = parse_dt(raw if not isinstance(raw, dict) else raw.get("review_timestamp"))
        if parsed is not None and parsed.astimezone(tz).date() == today:
            return True
    return False


def human_when(value: Any, *, now: datetime, tz: datetime.tzinfo) -> Optional[tuple[str, str]]:
    """Relative phrase plus a hover absolute. None when the value is not a time."""
    parsed = parse_dt(value)
    if parsed is None:
        return None
    local = parsed.astimezone(tz)
    current = now.astimezone(tz)
    absolute = f"{local.day} {local.strftime('%b %Y, %H:%M')}"
    delta = (current.date() - local.date()).days
    if delta == 0:
        return "today", absolute
    if delta == 1:
        return "yesterday", absolute
    if delta == -1:
        return "tomorrow", absolute
    if 1 < delta < 14:
        return f"{delta} days ago", absolute
    if -14 < delta < -1:
        return f"in {-delta} days", absolute
    return f"{local.day} {local.strftime('%b %Y')}", absolute


def _as_utc_date_row(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


@dataclass(frozen=True)
class Due:
    """Due cards, or an unknown queue.

    ``known`` is False when the read failed or came back truncated. ``count``
    is then None. A genuine empty queue has ``known`` True and ``count`` 0.
    """

    known: bool
    count: Optional[int] = None
    pillars: int = 0
    papers: int = 0
    next_due: Optional[datetime] = None
    by_pillar: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.by_pillar is None:
            object.__setattr__(self, "by_pillar", {})


def due_from_rows(rows: Optional[Sequence[Any]], *, now: datetime, truncated: bool = False) -> Due:
    """Cards the review page will actually show.

    ``get_cards_for_review`` treats a missing ``next_review_date`` as due, and
    a future one as not. Today uses the same rule so the number on this page
    and the session behind Start review describe one queue. A truncated page
    of rows cannot be counted honestly, so it is unknown rather than short.
    """
    if rows is None or truncated:
        return Due(known=False)
    current = parse_dt(now) or datetime.now(timezone.utc)
    pillars: set[str] = set()
    papers: set[str] = set()
    by_pillar: dict[str, int] = {}
    next_due: Optional[datetime] = None
    count = 0
    for row in rows:
        raw = _as_utc_date_row(row, "next_review_date")
        due_at = parse_dt(raw) if raw else None
        pillar = _as_utc_date_row(row, "pillar_id") or ""
        paper = _as_utc_date_row(row, "paper_id") or ""
        is_due = due_at is None or due_at <= current
        if is_due:
            count += 1
            if pillar:
                pillars.add(str(pillar))
                by_pillar[str(pillar)] = by_pillar.get(str(pillar), 0) + 1
            if paper:
                papers.add(str(paper))
            continue
        if due_at is not None and (next_due is None or due_at < next_due):
            next_due = due_at
    return Due(
        known=True,
        count=count,
        pillars=len(pillars),
        papers=len(papers),
        next_due=next_due,
        by_pillar=by_pillar,
    )


def due_sentence(due: Due, *, now: datetime, tz: datetime.tzinfo) -> tuple[str, str]:
    """Title and detail for the review plan row."""
    if not due.known:
        return "Review", "Due cards could not be loaded"
    count = due.count or 0
    if count <= 0:
        if due.next_due is not None:
            when = human_when(due.next_due, now=now, tz=tz)
            if when:
                return "Nothing due", f"Next card due {when[0]}"
        return "Nothing due", "No cards are scheduled"
    title = "Review 1 card" if count == 1 else f"Review {count} cards"
    minutes = max(1, round(count * 0.5))
    minute_word = "minute" if minutes == 1 else "minutes"
    if due.papers == 1:
        source = "from 1 paper"
    elif due.papers > 1:
        source = f"from {due.papers} papers"
    elif due.pillars == 1:
        source = "from 1 pillar"
    elif due.pillars > 1:
        source = f"from {due.pillars} pillars"
    else:
        source = ""
    detail = f"About {minutes} {minute_word}"
    if source:
        detail += f" · {source}"
    return title, detail


def summary_line(
    *,
    due: Due,
    has_lesson: bool,
    active_clause: Optional[str],
) -> str:
    parts: list[str] = []
    if due.known and due.count:
        count = due.count
        if count == 1:
            parts.append("1 card is due")
        elif due.pillars > 1:
            parts.append(f"{count} cards are due across {due.pillars} pillars")
        else:
            parts.append(f"{count} cards are due")
    elif due.known:
        parts.append("Nothing is due")
    if has_lesson:
        parts.append("a lesson is ready")
    if active_clause:
        parts.append(active_clause)
    if not parts:
        if not due.known:
            return "Some of today's counts could not be loaded."
        return "A quiet day."
    if len(parts) == 1:
        text = parts[0]
        return text[0].upper() + text[1:] + ("." if not text.endswith(".") else "")
    head = parts[0][0].upper() + parts[0][1:]
    if len(parts) == 2:
        return f"{head}, and {parts[1]}."
    return f"{head}, {', '.join(parts[1:-1])}, and {parts[-1]}."


def author_line(authors: Any) -> str:
    if isinstance(authors, str):
        parts = [piece.strip() for piece in authors.split(",") if piece.strip()]
    elif isinstance(authors, list):
        parts = []
        for piece in authors:
            if isinstance(piece, dict):
                name = piece.get("name") or ""
            else:
                name = str(piece or "")
            name = name.strip()
            if name:
                parts.append(name)
    else:
        parts = []
    if not parts:
        return ""
    if len(parts) <= 2:
        return ", ".join(parts)
    return ", ".join(parts[:2]) + f" +{len(parts) - 2}"


def duration_label(seconds: Any, words: Any) -> str:
    try:
        secs = float(seconds) if seconds is not None else 0
    except (TypeError, ValueError):
        secs = 0
    if secs > 0:
        minutes = max(1, round(secs / 60))
        return f"{minutes} min"
    try:
        count = int(words) if words is not None else 0
    except (TypeError, ValueError):
        count = 0
    if count > 0:
        minutes = max(1, round(count / 150))
        return f"about {minutes} min"
    return ""


def audio_src(meta: Any) -> Optional[str]:
    if isinstance(meta, str):
        return None
    if not isinstance(meta, dict):
        return None
    name = meta.get("file_name")
    if not isinstance(name, str):
        return None
    name = name.strip()
    if not name or "/" in name or "\\" in name or ".." in name:
        return None
    return "/api/podcast/audio/" + quote(name, safe="")


def _first_line(value: Any, limit: int = 180) -> str:
    if not value:
        return ""
    line = str(value).strip().splitlines()[0].strip()
    if len(line) > limit:
        return line[: limit - 1].rstrip() + "…"
    return line


def _tri(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return None
    return bool(value)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def run_sentence(run: dict, names: dict[str, str]) -> str:
    kind = str(run.get("kind") or "")
    status = str(run.get("status") or "")
    noun = _KIND_NOUN.get(kind, "Run")
    pillar = pillar_label(run.get("pillar_id"), names)
    title = str(run.get("result_title") or "").strip()
    error = _first_line(run.get("error"))
    added = _tri(run.get("result_added"))
    saved = _tri(run.get("result_saved"))
    if kind == "upload" and added:
        named = f'"{title}"' if title else "a paper"
        if status == "failed":
            return f"Added {named} to {pillar}, but some steps did not finish"
        return f"Added {named} to {pillar}"
    if kind == "upload" and status == "failed":
        return "Upload did not add a paper" + (f" — {error}" if error else "")
    if kind == "discover" and status == "succeeded":
        return f"Discovery finished for {pillar}"
    if kind == "discover" and status == "failed":
        return f"Discovery did not finish for {pillar}" + (f" — {error}" if error else "")
    if kind == "podcast_script" and status == "succeeded" and saved is False:
        return "A script was generated but not saved"
    if kind == "podcast_script" and status == "succeeded":
        return f"Script ready: {title}" if title else "Script ready"
    if kind == "podcast_audio" and status == "succeeded":
        return "Episode audio is ready"
    if status == "succeeded":
        processed = _int(run.get("papers_processed"))
        failed = _int(run.get("papers_failed"))
        if not processed and not failed:
            return f"{noun} finished — no new papers"
        text = f"{noun} finished — {processed} paper(s) processed"
        if failed:
            text += f", {failed} failed"
        return text
    if status == "cancelled":
        return f"{noun} was cancelled"
    if status == "interrupted":
        return f"{noun} was interrupted before it finished"
    if status in {"running", "pending"}:
        return f"{noun} is still going for {pillar}"
    return f"{noun} did not finish" + (f" — {error}" if error else "")


def run_href(run: dict) -> Optional[str]:
    kind = str(run.get("kind") or "")
    run_id = run.get("id")
    if kind == "discover" and run_id:
        pillar = run.get("pillar_id")
        query = []
        if pillar:
            query.append("pillar=" + quote(str(pillar), safe=""))
        query.append("run=" + quote(str(run_id), safe=""))
        return "/discovery/?" + "&".join(query)
    if kind == "upload":
        paper_id = run.get("result_paper_id")
        if paper_id and _tri(run.get("result_added")):
            return paper_href(str(paper_id))
        return "/pipeline/"
    if kind in {"podcast_script", "podcast_audio"}:
        script_id = run.get("result_script_id")
        if isinstance(script_id, str) and script_id and all(ch.isalnum() or ch == "-" for ch in script_id):
            return "/podcast/?script=" + quote(script_id, safe="")
        paper_id = run.get("result_paper_id")
        if paper_id:
            return paper_href(str(paper_id))
        return "/podcast/"
    paper_id = run.get("result_paper_id")
    if paper_id:
        return paper_href(str(paper_id))
    if run_id:
        return "/pipeline/?run=" + quote(str(run_id), safe="")
    return "/pipeline/"


def run_tone(run: dict) -> str:
    status = str(run.get("status") or "")
    if status == "succeeded" and _tri(run.get("result_saved")) is False:
        return "warn"
    if status == "succeeded" and _int(run.get("papers_failed")) > 0:
        return "warn"
    if status == "failed" and _tri(run.get("result_added")):
        return "warn"
    if status == "succeeded":
        return "ok"
    if status == "failed":
        return "danger"
    if status in {"cancelled", "interrupted"}:
        return "warn"
    return ""


def active_clause(run: Optional[dict], names: dict[str, str]) -> Optional[str]:
    if not run:
        return None
    kind = str(run.get("kind") or "")
    pillar = pillar_label(run.get("pillar_id"), names)
    if kind == "discover":
        return f"new papers are being found for {pillar}"
    if kind == "upload":
        return f"a paper is being added to {pillar}"
    if kind == "podcast_script":
        return "a script is being written"
    if kind == "podcast_audio":
        return "an episode is being recorded"
    if kind in {"process_selected", "run_daily"}:
        return f"papers are being processed for {pillar}"
    return "something is still running"


def active_title(run: dict, names: dict[str, str]) -> str:
    noun = _KIND_NOUN.get(str(run.get("kind") or ""), "Run")
    pillar = run.get("pillar_id")
    if pillar:
        return f"{noun} · {pillar_label(pillar, names)}"
    return noun


def recent_runs(runs: Optional[Sequence[dict]], *, now: datetime) -> tuple[bool, list[dict]]:
    """Terminal runs from the last 24 hours. The bool is False when the read failed."""
    if runs is None:
        return False, []
    current = parse_dt(now) or now
    cutoff = current - timedelta(hours=_RECENT_HOURS)
    chosen = []
    for run in runs:
        status = str(run.get("status") or "")
        if status in {"", "pending", "running"}:
            continue
        stamp = parse_dt(run.get("finished_at") or run.get("created_at"))
        if stamp is None or stamp < cutoff:
            continue
        chosen.append(run)
    chosen.sort(
        key=lambda row: parse_dt(row.get("finished_at") or row.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return True, list(chosen[:_RECENT_CAP])


@dataclass(frozen=True)
class LessonCard:
    paper_id: str
    href: str
    title: str
    pillar_name: str
    minutes: Optional[int]
    when: str
    absolute: str
    authors: str


@dataclass(frozen=True)
class EpisodeCard:
    script_id: str
    paper_href: str
    title: str
    pillar_name: str
    src: str
    listen_href: str
    length: str
    duration_seconds: Optional[float]
    when: str
    absolute: str


@dataclass(frozen=True)
class GlanceTile:
    slug: str
    name: str
    papers: Optional[int]
    lessons: Optional[int]
    cards: Optional[int]
    due: Optional[int]


@dataclass(frozen=True)
class ActivityRow:
    kind: str
    title: str
    href: Optional[str]
    pillar_name: str
    relative: str
    absolute: str


@dataclass(frozen=True)
class RunRow:
    sentence: str
    href: Optional[str]
    tone: str
    relative: str
    absolute: str


@dataclass
class TodayInputs:
    pillars: list
    quiz_rows: Optional[list]
    quiz_truncated: bool
    review_timestamps: Optional[list]
    lessons: Optional[list]
    papers_by_id: dict
    episodes: Optional[list]
    runs: Optional[list]
    active_run: Optional[dict]
    counts: dict
    activity_papers: Optional[list]
    activity_quiz: Optional[list]
    stats: QuickStats


@dataclass
class TodayView:
    greeting: str
    date_line: str
    summary: str
    due: Due
    review_title: str
    review_detail: str
    review_done: bool
    up_next: Optional[LessonCard]
    episode: Optional[EpisodeCard]
    continue_rows: list[LessonCard]
    episodes: list[EpisodeCard]
    lessons_known: bool
    episodes_known: bool
    streak: Optional[Streak]
    glance: list[GlanceTile]
    activity: Optional[list[ActivityRow]]
    stats: QuickStats
    runs_known: bool
    recent: list[RunRow]
    active_run: Optional[dict]
    active_title: str
    active_sentence: str
    payload_json: str
    primary: str


def _names(pillars: list) -> dict[str, str]:
    names = {}
    for pillar in pillars:
        slug = getattr(pillar, "id", None) or (pillar.get("id") if isinstance(pillar, dict) else None)
        name = getattr(pillar, "name", None) or (pillar.get("name") if isinstance(pillar, dict) else None)
        if slug and name:
            names[str(slug)] = str(name)
    return names


def _lesson_card(row: dict, papers: dict, names: dict[str, str], *, now: datetime, tz: datetime.tzinfo) -> LessonCard:
    paper_id = str(row.get("paper_id") or "")
    paper = papers.get(paper_id) or {}
    title = str(paper.get("title") or "").strip() or str(row.get("tl_dr") or "").strip() or "Lesson"
    when = human_when(row.get("created_at"), now=now, tz=tz)
    minutes = row.get("estimated_time")
    try:
        minutes = int(minutes) if minutes is not None else None
    except (TypeError, ValueError):
        minutes = None
    return LessonCard(
        paper_id=paper_id,
        href=paper_href(paper_id) if paper_id else "/papers/",
        title=title,
        pillar_name=pillar_label(row.get("pillar_id") or paper.get("pillar_id"), names),
        minutes=minutes,
        when=when[0] if when else "",
        absolute=when[1] if when else "",
        authors=author_line(paper.get("authors")),
    )


def _episode_cards(rows: Sequence[dict], names: dict[str, str], *, now: datetime, tz: datetime.tzinfo) -> list[EpisodeCard]:
    cards = []
    for row in rows:
        src = audio_src(row.get("audio_metadata"))
        if not src:
            continue
        script_id = str(row.get("id") or "")
        paper_id = str(row.get("paper_id") or "")
        meta = row.get("audio_metadata") if isinstance(row.get("audio_metadata"), dict) else {}
        when = human_when(row.get("created_at"), now=now, tz=tz)
        seconds = meta.get("duration_seconds") if isinstance(meta, dict) else None
        try:
            seconds_f = float(seconds) if seconds is not None else None
        except (TypeError, ValueError):
            seconds_f = None
        listen = "/podcast/"
        if script_id and all(ch.isalnum() or ch == "-" for ch in script_id):
            listen = "/podcast/?script=" + quote(script_id, safe="")
        cards.append(EpisodeCard(
            script_id=script_id,
            paper_href=paper_href(paper_id) if paper_id else listen,
            title=str(row.get("title") or "").strip() or "Episode",
            pillar_name=pillar_label(row.get("pillar_id"), names),
            src=src,
            listen_href=listen,
            length=duration_label(seconds_f, row.get("word_count")),
            duration_seconds=seconds_f,
            when=when[0] if when else "",
            absolute=when[1] if when else "",
        ))
    return cards


def _activity(
    papers: Optional[list],
    lessons: Optional[list],
    quiz: Optional[list],
    names: dict[str, str],
    *,
    now: datetime,
    tz: datetime.tzinfo,
) -> Optional[list[ActivityRow]]:
    if papers is None and lessons is None and quiz is None:
        return None
    rows: list[tuple[datetime, ActivityRow]] = []

    def push(stamp: Any, item: ActivityRow) -> None:
        parsed = parse_dt(stamp)
        rows.append((parsed or datetime.min.replace(tzinfo=timezone.utc), item))

    for paper in papers or []:
        when = human_when(paper.get("added_at"), now=now, tz=tz)
        if when is None:
            continue
        paper_id = str(paper.get("id") or "")
        push(paper.get("added_at"), ActivityRow(
            kind="Paper",
            title=str(paper.get("title") or "").strip() or "Untitled paper",
            href=paper_href(paper_id) if paper_id else None,
            pillar_name=pillar_label(paper.get("pillar_id"), names),
            relative=when[0],
            absolute=when[1],
        ))
    for lesson in lessons or []:
        when = human_when(lesson.get("created_at"), now=now, tz=tz)
        if when is None:
            continue
        paper_id = str(lesson.get("paper_id") or "")
        push(lesson.get("created_at"), ActivityRow(
            kind="Lesson",
            title=str(lesson.get("tl_dr") or "").strip() or "Lesson",
            href=paper_href(paper_id) if paper_id else None,
            pillar_name=pillar_label(lesson.get("pillar_id"), names),
            relative=when[0],
            absolute=when[1],
        ))
    for card in quiz or []:
        stamp = card.get("created_at") or card.get("last_reviewed")
        when = human_when(stamp, now=now, tz=tz)
        if when is None:
            continue
        paper_id = str(card.get("paper_id") or "")
        question = str(card.get("question") or "").strip()
        push(stamp, ActivityRow(
            kind="Quiz card",
            title=question or "Quiz card",
            href=paper_href(paper_id) if paper_id else None,
            pillar_name=pillar_label(card.get("pillar_id"), names),
            relative=when[0],
            absolute=when[1],
        ))
    rows.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in rows[:10]]


def _public_run(run: Optional[dict]) -> Optional[dict]:
    if not run:
        return None
    # result on a discovery run is the whole candidate list. The card does not
    # need it, and it must not be embedded into the page.
    return {key: value for key, value in run.items() if key != "result"}


def build_today_view(inputs: TodayInputs, *, now: datetime, tz: datetime.tzinfo) -> TodayView:
    names = _names(inputs.pillars)
    local = now.astimezone(tz)
    due = due_from_rows(inputs.quiz_rows, now=now, truncated=inputs.quiz_truncated)
    review_title, review_detail = due_sentence(due, now=now, tz=tz)
    review_done = False
    streak: Optional[Streak] = None
    if inputs.review_timestamps is not None:
        streak = streak_from_timestamps(inputs.review_timestamps, now=now, tz=tz)
        review_done = reviewed_today(inputs.review_timestamps, now=now, tz=tz)

    lessons_known = inputs.lessons is not None
    lesson_cards = [
        _lesson_card(row, inputs.papers_by_id, names, now=now, tz=tz)
        for row in (inputs.lessons or [])
        if row.get("paper_id")
    ]
    up_next = lesson_cards[0] if lesson_cards else None
    continue_rows = lesson_cards[:3]

    episodes_known = inputs.episodes is not None
    episode_cards = _episode_cards(inputs.episodes or [], names, now=now, tz=tz)
    episode = episode_cards[0] if episode_cards else None

    glance = []
    for pillar in inputs.pillars:
        slug = str(getattr(pillar, "id", "") or "")
        if not slug:
            continue
        counts = inputs.counts.get(slug)
        due_count: Optional[int]
        if not due.known:
            due_count = None
        else:
            due_count = due.by_pillar.get(slug, 0)
        if counts is None:
            papers = lessons = cards = None
        else:
            papers = counts.get("papers")
            lessons = counts.get("lessons")
            cards = counts.get("quiz_cards")
        glance.append(GlanceTile(
            slug=slug,
            name=pillar_label(slug, names),
            papers=papers,
            lessons=lessons,
            cards=cards,
            due=due_count,
        ))

    activity = _activity(
        inputs.activity_papers,
        inputs.lessons,
        inputs.activity_quiz,
        names,
        now=now,
        tz=tz,
    )
    runs_known, raw_recent = recent_runs(inputs.runs, now=now)
    recent = []
    for run in raw_recent:
        when = human_when(run.get("finished_at") or run.get("created_at"), now=now, tz=tz)
        recent.append(RunRow(
            sentence=run_sentence(run, names),
            href=run_href(run),
            tone=run_tone(run),
            relative=when[0] if when else "",
            absolute=when[1] if when else "",
        ))

    active = _public_run(inputs.active_run)
    clause = active_clause(active, names)
    if due.known and due.count:
        primary = "review"
    elif up_next:
        primary = "lesson"
    elif episode:
        primary = "episode"
    else:
        primary = ""

    timestamps = []
    if inputs.review_timestamps is not None:
        for raw in inputs.review_timestamps:
            if isinstance(raw, dict):
                raw = raw.get("review_timestamp")
            parsed = parse_dt(raw)
            if parsed is not None:
                timestamps.append(parsed.isoformat())

    payload = {
        "reviews": timestamps if inputs.review_timestamps is not None else None,
        "upNextPaperId": up_next.paper_id if up_next else "",
        "episode": {
            "scriptId": episode.script_id,
            "duration": episode.duration_seconds,
        } if episode else None,
        "active": active,
    }
    return TodayView(
        # Container zone. webui/static/today.js rewrites both from the browser
        # clock in the same turn that it redraws the streak dots.
        greeting=greeting(local.hour),
        date_line=f"{local.strftime('%A, %B')} {local.day}",
        summary=summary_line(due=due, has_lesson=up_next is not None, active_clause=clause),
        due=due,
        review_title=review_title,
        review_detail=review_detail,
        review_done=review_done,
        up_next=up_next,
        episode=episode,
        continue_rows=continue_rows,
        episodes=episode_cards[:3],
        lessons_known=lessons_known,
        episodes_known=episodes_known,
        streak=streak,
        glance=glance,
        activity=activity,
        stats=inputs.stats,
        runs_known=runs_known,
        recent=recent,
        active_run=active,
        active_title=active_title(active, names) if active else "",
        active_sentence=run_sentence(active, names) if active else "",
        payload_json=json_for_script(payload),
        primary=primary,
    )


def _rows(result: Any) -> tuple[Optional[list], bool]:
    """(rows or None, truncated)."""
    if result is None:
        return None, False
    rows, total = result
    truncated = False
    if total is not None and total > len(rows):
        truncated = True
    elif total is None and len(rows) >= _QUIZ_LIMIT:
        truncated = True
    return rows, truncated


async def load_today_view(
    client: Optional[PostgrestClient] = None,
    *,
    now: Optional[datetime] = None,
    tz: Optional[datetime.tzinfo] = None,
) -> TodayView:
    client = client or PostgrestClient()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    tz = tz or datetime.now().astimezone().tzinfo or timezone.utc

    quiz_params = {
        "select": "next_review_date,pillar_id,paper_id",
        "user_id": "eq.default_user",
        "limit": _QUIZ_LIMIT,
    }
    review_params = {
        "select": "review_timestamp",
        "user_id": "eq.default_user",
        "order": "review_timestamp.desc",
        "limit": _REVIEW_LIMIT,
    }
    lesson_params = {
        "select": "id,paper_id,pillar_id,tl_dr,created_at,estimated_time",
        "order": "created_at.desc",
        "limit": 10,
    }
    episode_params = {
        "select": "id,paper_id,pillar_id,title,created_at,word_count,audio_metadata",
        "order": "created_at.desc",
        "limit": 20,
    }
    run_params = {
        "select": (
            "id,pillar_id,kind,status,current_stage,papers_processed,papers_failed,"
            "error,created_at,started_at,finished_at,"
            "result_title:result->>title,result_added:result->>added,"
            "result_paper_id:result->>paper_id,result_script_id:result->>script_id,"
            "result_saved:result->>saved"
        ),
        "order": "created_at.desc",
        "limit": 40,
    }
    paper_params = {
        "select": "id,title,pillar_id,added_at",
        "order": "added_at.desc",
        "limit": 10,
    }
    quiz_activity_params = {
        "select": "id,paper_id,pillar_id,question,created_at,last_reviewed",
        "order": "created_at.desc",
        "limit": 10,
    }

    pillars, quiz_raw, review_raw, lessons_raw, episodes_raw, runs_raw, papers_raw, quiz_act_raw, active, stats = await asyncio.gather(
        asyncio.to_thread(get_pillars_or_empty, 100),
        client.fetch_rows("quiz_cards", quiz_params),
        client.fetch_rows("review_logs", review_params),
        client.fetch_rows("lessons", lesson_params),
        client.fetch_rows("podcast_scripts", episode_params),
        client.fetch_rows("pipeline_runs", run_params),
        client.fetch_rows("papers", paper_params),
        client.fetch_rows("quiz_cards", quiz_activity_params),
        client.get_active_pipeline_run(),
        client.get_quick_stats(),
    )

    quiz_rows, quiz_truncated = _rows(quiz_raw)
    review_rows, _review_truncated = _rows(review_raw)
    lessons, _ = _rows(lessons_raw)
    episodes, _ = _rows(episodes_raw)
    runs, _ = _rows(runs_raw)
    activity_papers, _ = _rows(papers_raw)
    activity_quiz, _ = _rows(quiz_act_raw)
    # A truncated review history can only shorten a very long streak. Keep it.
    # A truncated due-card page cannot be counted, which _rows already flags
    # for the quiz query via the shared limit. Reviews use the same helper
    # with a lower practical chance of truncation; do not blank the streak
    # just because Content-Range was absent.
    if review_raw is not None and review_rows is not None and len(review_rows) >= _REVIEW_LIMIT:
        review_rows = review_rows  # still usable; the walk stops at the first gap

    paper_ids = []
    for row in lessons or []:
        paper_id = row.get("paper_id")
        if paper_id and paper_id not in paper_ids:
            paper_ids.append(str(paper_id))
    papers_by_id: dict[str, dict] = {}
    if paper_ids:
        found = await client.fetch_rows("papers", {
            "select": "id,title,authors,year,processed,pillar_id",
            "id": postgrest_in(paper_ids),
        })
        found_rows, _ = _rows(found)
        for row in found_rows or []:
            if row.get("id"):
                papers_by_id[str(row["id"])] = row

    async def one_count(pillar_id: str):
        try:
            return pillar_id, await client.counts_for_pillar(pillar_id)
        except Exception as exc:
            logger.warning("Could not count pillar %s: %s", pillar_id, exc)
            return pillar_id, None

    counted = await asyncio.gather(*(one_count(str(pillar.id)) for pillar in pillars))
    counts = dict(counted)

    if review_rows is not None:
        review_values = [row.get("review_timestamp") for row in review_rows]
    else:
        review_values = None

    return build_today_view(TodayInputs(
        pillars=list(pillars),
        quiz_rows=quiz_rows,
        quiz_truncated=quiz_truncated,
        review_timestamps=review_values,
        lessons=lessons,
        papers_by_id=papers_by_id,
        episodes=episodes,
        runs=runs,
        active_run=active,
        counts=counts,
        activity_papers=activity_papers,
        activity_quiz=activity_quiz,
        stats=stats,
    ), now=now, tz=tz)
