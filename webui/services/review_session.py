"""Pure helpers for the one-card review session.

The page still loads due cards and posts ratings through the existing FSRS
path. This module only decides what the session shows: the next interval on
each rating button, the empty-state sentence, and the JSON the browser
renders one card at a time. Nothing here writes a review.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

# Again / Hard / Good / Easy. The numbers are the rating the review POST
# already accepts. The words match the buttons the old page showed.
RATINGS = (
    (1, "Again", "Completely forgot"),
    (2, "Hard", "Struggled to remember"),
    (3, "Good", "Remembered with effort"),
    (4, "Easy", "Remembered easily"),
)

Schedule = Callable[[Any, int], datetime]


def _as_utc(value: Any) -> Optional[datetime]:
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


def difficulty_number(value: Any) -> Optional[int]:
    """1, 2 or 3. None for anything else, including an unparsable enum repr."""
    if hasattr(value, "value") and not isinstance(value, (str, int)):
        value = value.value
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 1 or number > 3:
        return None
    return number


def format_interval(delta: timedelta) -> str:
    """Short interval for a rating button, from a scheduling result.

    Sub-minute stays ``<1m`` so a card the algorithm wants immediately does
    not read as a zero. Months use a half-month threshold rather than
    banker's rounding, so the label is stable.
    """
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "<1m"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    if days < 30:
        return f"{days}d"
    if days < 365:
        months = max(1, (days + 15) // 30)
        if months >= 12:
            return "1y"
        return f"{months}mo"
    return f"{max(1, days // 365)}y"


def format_due(when: datetime, now: datetime) -> str:
    """When the next card is due, in UTC so the sentence does not depend on the host."""
    due = _as_utc(when)
    current = _as_utc(now)
    if due is None or current is None:
        return "later"
    clock = due.strftime("%H:%M UTC")
    if due.date() == current.date():
        return f"today at {clock}"
    if due.date() == current.date() + timedelta(days=1):
        return f"tomorrow at {clock}"
    return due.strftime("%-d %b %Y")


def earliest_future(rows: list, now: datetime, difficulty: Optional[int] = None) -> Optional[datetime]:
    """Soonest next_review_date strictly after ``now``.

    Rows may be dicts or objects. A missing or unparsable date is skipped.
    ``difficulty`` limits the scan to that 1–3 level.
    """
    current = _as_utc(now)
    if current is None:
        return None
    best: Optional[datetime] = None
    for row in rows:
        if isinstance(row, dict):
            raw_difficulty = row.get("difficulty")
            raw_due = row.get("next_review_date")
        else:
            raw_difficulty = getattr(row, "difficulty", None)
            raw_due = getattr(row, "next_review_date", None)
        if difficulty is not None and difficulty_number(raw_difficulty) != difficulty:
            continue
        due = _as_utc(raw_due)
        if due is None or due <= current:
            continue
        if best is None or due < best:
            best = due
    return best


def empty_copy(*, next_due: Optional[datetime], now: datetime, filtered: bool) -> tuple[str, str]:
    """Title and message for an empty queue.

    The title is always "Nothing due". The message names the next card when
    one is scheduled, and otherwise says the filter matched nothing or that
    no card is scheduled at all.
    """
    title = "Nothing due"
    if next_due is not None:
        return title, f"Next card due {format_due(next_due, now)}."
    if filtered:
        return title, "Nothing matches this filter."
    return title, "No cards are scheduled."


def _copy_card(card: Any) -> Any:
    if hasattr(card, "model_copy"):
        return card.model_copy(deep=True)
    return card


def interval_previews(card: Any, *, now: datetime, schedule: Schedule) -> list[dict]:
    """One entry per rating. ``schedule`` must not persist anything.

    A failure on one rating leaves that interval blank. The button is still
    shown — the label is the action, the interval is extra.
    """
    previews = []
    for rating, label, hint in RATINGS:
        interval = None
        try:
            due = _as_utc(schedule(_copy_card(card), rating))
            current = _as_utc(now)
            if due is not None and current is not None:
                interval = format_interval(due - current)
        except Exception:
            interval = None
        previews.append({
            "rating": rating,
            "label": label,
            "hint": hint,
            "interval": interval,
        })
    return previews


def _card_fields(card: Any) -> dict:
    if isinstance(card, dict):
        return {
            "id": card.get("id"),
            "paper_id": card.get("paper_id") or "",
            "pillar_id": card.get("pillar_id") or "",
            "question": card.get("question") or "",
            "answer": card.get("answer") or "",
            "difficulty": difficulty_number(card.get("difficulty")),
            "due": card.get("next_review_date"),
        }
    due = getattr(card, "next_review_date", None)
    return {
        "id": getattr(card, "id", None),
        "paper_id": getattr(card, "paper_id", "") or "",
        "pillar_id": getattr(card, "pillar_id", "") or "",
        "question": getattr(card, "question", "") or "",
        "answer": getattr(card, "answer", "") or "",
        "difficulty": difficulty_number(getattr(card, "difficulty", None)),
        "due": due.isoformat() if isinstance(due, datetime) else due,
    }


def session_cards(cards: list, *, now: datetime, schedule: Optional[Schedule],
                  pillar_names: dict[str, str]) -> list[dict]:
    """JSON-ready cards. Dict rows (the FSRS fallback) get no invented interval."""
    payload = []
    for card in cards:
        fields = _card_fields(card)
        pillar_id = str(fields["pillar_id"])
        if isinstance(card, dict) or schedule is None:
            intervals = [
                {"rating": rating, "label": label, "hint": hint, "interval": None}
                for rating, label, hint in RATINGS
            ]
        else:
            intervals = interval_previews(card, now=now, schedule=schedule)
        due = fields["due"]
        if isinstance(due, datetime):
            due = due.isoformat()
        payload.append({
            "id": str(fields["id"]) if fields["id"] is not None else "",
            "paperId": str(fields["paper_id"]),
            "pillarId": pillar_id,
            "pillarName": pillar_names.get(pillar_id) or pillar_id,
            "question": str(fields["question"]),
            "answer": str(fields["answer"]),
            "difficulty": fields["difficulty"],
            "due": due,
            "intervals": intervals,
        })
    return payload


def json_for_script(payload: Any) -> str:
    """JSON that is safe to embed in a script tag. ``<`` cannot break out."""
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        raw.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
