"""Quiz card state as a fold over its review log.

``review_logs`` is the source of truth for scheduling; the ``quiz_cards`` row is
a cache of what this module derives from it. Three properties make that safe,
and each one is tested:

* **Fuzz is off.** py-fsrs fuzzes every interval by default, so replaying one
  history twice gives different due dates (measured: 20 identical replays, 10
  distinct dates). A fold that is merged across devices must be reproducible.
* **The order is total.** Events sort by ``(reviewed_at, event_id)``, so a late
  or out-of-order event lands where it belongs and every device agrees.
* **The fold carries ``step``.** A ``quiz_cards`` row has no column for it, so a
  card rebuilt from the row restarts its learning steps. Replaying the log from
  an empty card does not.

This module is pure: no database, no clock. ``services/quiz_reviews.py`` does
the reads and writes.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from fsrs import Card, Rating, Scheduler, State

from .schemas import CardState

# A review stamped later than this past "now" is a wrong clock, not a review.
MAX_FUTURE_SKEW = timedelta(minutes=5)
# Older than this and the device sat on it far too long; refuse rather than
# rewrite history that other devices have already folded.
MAX_AGE = timedelta(days=30)

# py-fsrs needs *a* due date on a new card. A fixed one keeps the starting
# card identical on every run; the first review overwrites it.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def build_scheduler() -> Scheduler:
    """The one scheduler: py-fsrs defaults (FSRS-6 weights, 0.9 retention), fuzz off."""
    return Scheduler(enable_fuzzing=False)


@dataclass(frozen=True)
class FoldEvent:
    """One review. Immutable, identified by ``event_id``."""

    event_id: str
    rating: int  # 1=Again 2=Hard 3=Good 4=Easy
    reviewed_at: datetime  # timezone-aware

    def __post_init__(self) -> None:
        if self.rating not in (1, 2, 3, 4):
            raise ValueError(f"rating must be 1-4, got {self.rating!r}")
        if self.reviewed_at.tzinfo is None:
            raise ValueError("reviewed_at must be timezone-aware")


@dataclass(frozen=True)
class FoldedCard:
    """The result of folding a card's events."""

    card: Card
    review_count: int
    lapses: int  # Again pressed on a card that was in the Review state
    last_event: FoldEvent


def parse_timestamp(value: Any) -> datetime:
    """An ISO-8601 string or datetime as an aware UTC datetime."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError(f"not a timestamp: {value!r}")
    if value.tzinfo is None:
        raise ValueError(f"timestamp has no UTC offset: {value!r}")
    return value.astimezone(timezone.utc)


def ordered(events: Iterable[FoldEvent]) -> List[FoldEvent]:
    """Events in fold order, with a repeated ``event_id`` counted once."""
    unique: Dict[str, FoldEvent] = {}
    for event in events:
        unique.setdefault(event.event_id, event)
    return sorted(
        unique.values(),
        key=lambda e: (e.reviewed_at.astimezone(timezone.utc), e.event_id),
    )


def apply_event(
    scheduler: Scheduler, card: Optional[Card], event: FoldEvent
) -> Card:
    """One review applied to a card (``None`` starts a new one)."""
    if card is None:
        card = Card(card_id=0, due=_EPOCH)
    reviewed, _log = scheduler.review_card(
        card,
        Rating(event.rating),
        review_datetime=event.reviewed_at.astimezone(timezone.utc),
    )
    return reviewed


def fold_events(
    events: Iterable[FoldEvent], scheduler: Optional[Scheduler] = None
) -> Optional[FoldedCard]:
    """Fold a card's review history. ``None`` when the card was never reviewed."""
    scheduler = scheduler or build_scheduler()
    card: Optional[Card] = None
    lapses = 0
    count = 0
    last: Optional[FoldEvent] = None
    for event in ordered(events):
        if card is not None and card.state == State.Review and event.rating == 1:
            lapses += 1
        card = apply_event(scheduler, card, event)
        count += 1
        last = event
    if card is None or last is None:
        return None
    return FoldedCard(card=card, review_count=count, lapses=lapses, last_event=last)


def preview_due(events: Iterable[FoldEvent], rating: int, at: datetime) -> datetime:
    """When a card with this history would next be due if rated ``rating`` at ``at``.

    Writes nothing. The card is the fold of ``events``; an empty history is a new card.
    """
    scheduler = build_scheduler()
    folded = fold_events(events, scheduler)
    reviewed = apply_event(
        scheduler,
        folded.card if folded else None,
        FoldEvent("preview", rating, at),
    )
    return reviewed.due


_STATE = {
    State.Learning: CardState.LEARNING,
    State.Review: CardState.REVIEW,
    State.Relearning: CardState.RELEARNING,
}


def cache_fields(folded: FoldedCard) -> Dict[str, Any]:
    """The ``quiz_cards`` columns that cache this fold.

    ``repetitions`` holds the review count because that is how the row is read
    back (``_dict_to_quiz_card`` maps it to ``review_count``). The legacy
    SM-2 ``interval`` is whole days, never below one.
    """
    card = folded.card
    last = card.last_review
    assert last is not None  # a folded card has been reviewed
    interval = max(1, round((card.due - last).total_seconds() / 86400))
    return {
        "state": _STATE[card.state].value,
        "difficulty_fsrs": card.difficulty,
        "stability": card.stability,
        "lapses": folded.lapses,
        "repetitions": folded.review_count,
        "last_review_date": last.isoformat(),
        "last_reviewed": last.isoformat(),
        "next_review_date": card.due.isoformat(),
        "due_date": card.due.isoformat(),
        "interval": interval,
    }


def card_snapshot(card: Card) -> Dict[str, Any]:
    """A py-fsrs card as plain JSON, with epoch milliseconds for exact comparison.

    ``tests/fixtures/fsrs/sequences.json`` is made of these, and the phone's
    ts-fsrs fold is checked against them.
    """
    data = card.to_dict()
    data.pop("card_id", None)
    data["due_ms"] = _epoch_ms(card.due)
    data["last_review_ms"] = _epoch_ms(card.last_review) if card.last_review else None
    return data


def _epoch_ms(moment: datetime) -> int:
    return round(moment.timestamp() * 1000)


def validate_reviewed_at(reviewed_at: datetime, now: datetime) -> datetime:
    """``reviewed_at`` in UTC, or ``ValueError`` if it cannot be a real review time."""
    stamp = parse_timestamp(reviewed_at)
    now = now.astimezone(timezone.utc)
    if stamp > now + MAX_FUTURE_SKEW:
        raise ValueError("reviewed_at is in the future")
    if stamp < now - MAX_AGE:
        raise ValueError(f"reviewed_at is more than {MAX_AGE.days} days old")
    return stamp


def now_ms() -> datetime:
    """The current UTC time truncated to milliseconds.

    The phone stamps events in milliseconds. A desktop stamp with microseconds
    could fold differently there, so the desktop truncates the same way.
    """
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)
