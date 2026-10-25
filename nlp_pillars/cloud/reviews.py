"""``cloud pull-reviews``: bring the phone's reviews home.

The phone posts each review to the cloud reader as an immutable event. This
module asks the Worker for the ones after the cursor, checks each, and writes it
into ``review_logs`` through the same log-first path a review made at the desk
takes (``services.quiz_reviews.record_review``), with ``origin='cloud'`` and the
phone's ``device_id``. ``record_review`` then re-derives the card from its whole
log, so a phone review made yesterday lands where it belongs among today's.

Nothing here needs a transaction, because every step can be repeated:

* an event is a primary-key insert (its ``event_id`` is the row id), so a second
  delivery records nothing and only re-derives the card;
* the card row is a cache of the log, so a crash between the two writes heals the
  next time the event, or a later one for the same card, is applied;
* the cursor moves only after a page is applied, and ``reviews_pending`` is
  raised before the cards are touched, so a pull that stops half-way is visible to
  ``cloud export`` (which will not publish a snapshot of half-applied events) and
  is finished by the next pull.

What is refused, and why it does not stop the pull: an event the computer will
never be able to apply (a card that no longer exists, an id that is not a UUID, a
time outside the window) is skipped and reported, and the cursor moves past it.
What stops the pull: anything that might clear by itself (the database or the
cloud being unreachable). The cursor stays where it was and the next pull retries.

Review events carry no free text. Nothing from the cloud reaches a prompt, a
shell or a SQL string: values are validated here against fixed shapes and then go
through parameterised PostgREST calls.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .. import fsrs_fold
from ..schemas import FSRSRating, QuizReviewRequest
from ..services import quiz_reviews
from .publish import IngestClient, PublishConfig, PublishError
from .review_state import (
    APPLY_LOCK,
    PENDING_KEY,
    SEQ_KEY,
    PostgrestSyncState,
    ReviewCursor,
    SyncStateError,
)

logger = logging.getLogger(__name__)

PAGE = 500
# A pull stops after this many pages and the next one carries on, so one call stays bounded.
MAX_PAGES = 40
_SLUG = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_MAX_RESPONSE_MS = 3_600_000


class PullError(Exception):
    """The pull stopped. What was applied stays applied; the next pull carries on."""

    def __init__(self, message: str, report: Optional["PullReport"] = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class PullReport:
    dry_run: bool = False
    cursor_before: int = 0
    cursor_after: int = 0
    cloud_through_seq: int = 0
    fetched: int = 0
    applied: int = 0
    duplicates: int = 0
    skipped: List[Tuple[int, str, str]] = field(default_factory=list)  # (seq, event_id, reason)
    cards: Set[str] = field(default_factory=set)
    cursor_reset: bool = False
    more: bool = False

    def summary(self) -> Dict[str, Any]:
        return {
            "fetched": self.fetched, "applied": self.applied, "duplicates": self.duplicates,
            "skipped": len(self.skipped), "cards": len(self.cards),
            "cursor": self.cursor_after, "cloud_through_seq": self.cloud_through_seq, "more": self.more,
        }

    def render(self) -> str:
        verb = "would apply" if self.dry_run else "applied"
        lines = [
            f"cloud pull-reviews {'(dry run) ' if self.dry_run else ''}from seq {self.cursor_before}: "
            f"{self.fetched} event(s) fetched, {verb} {self.applied}, already recorded {self.duplicates}, "
            f"skipped {len(self.skipped)}; {len(self.cards)} card(s) touched.",
            f"  cursor {self.cursor_before} -> {self.cursor_after} (the cloud holds events through seq {self.cloud_through_seq})",
        ]
        if self.cursor_reset:
            lines.append("  the cloud holds fewer events than the cursor expected (a new database?); started again from 0")
        if self.more:
            lines.append("  more events are waiting; run it again")
        lines.extend(f"  skipped seq {seq} ({eid}): {why}" for seq, eid, why in self.skipped[:10])
        if len(self.skipped) > 10:
            lines.append(f"  ... and {len(self.skipped) - 10} more skipped")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checking one event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CloudEvent:
    seq: int
    event_id: str
    card_id: str
    rating: int
    reviewed_at: datetime
    device_id: str
    received_at: datetime
    response_time_ms: Optional[int] = None
    session_id: Optional[str] = None


def _uuid(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a UUID")
    try:
        parsed = str(uuid.UUID(value))
    except ValueError:
        raise ValueError(f"{name} must be a UUID") from None
    if parsed != value:
        raise ValueError(f"{name} must be a lowercase UUID")
    return parsed


def parse_event(raw: Any) -> CloudEvent:
    """A cloud event as the computer will apply it, or ``ValueError`` saying why not.

    Re-checks everything the Worker checked: the computer does not take the cloud's word.
    The time window is judged against ``received_at`` (when the Worker got the event), so a
    review the computer pulls a month late is not refused for being old.
    """
    if not isinstance(raw, dict):
        raise ValueError("not an object")
    seq = raw.get("seq")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
        raise ValueError("seq must be a positive integer")
    rating = raw.get("rating")
    if isinstance(rating, bool) or not isinstance(rating, int) or rating not in (1, 2, 3, 4):
        raise ValueError("rating must be 1 to 4")
    device = raw.get("device_id")
    if not isinstance(device, str) or not _SLUG.match(device):
        raise ValueError("device_id has the wrong shape")
    response_time = raw.get("response_time_ms")
    if response_time is not None:
        if isinstance(response_time, bool) or not isinstance(response_time, int) or not 0 <= response_time <= _MAX_RESPONSE_MS:
            raise ValueError("response_time_ms is out of range")
    session = raw.get("session_id")
    if session is not None and (not isinstance(session, str) or not _SLUG.match(session)):
        raise ValueError("session_id has the wrong shape")
    try:
        received = fsrs_fold.parse_timestamp(raw.get("received_at"))
    except (ValueError, TypeError):
        raise ValueError("received_at is not a valid timestamp") from None
    try:
        reviewed = fsrs_fold.validate_reviewed_at(raw.get("reviewed_at"), received)
    except (ValueError, TypeError) as exc:
        text = str(exc)
        raise ValueError(text if text.startswith("reviewed_at") else "reviewed_at is not a valid timestamp") from None
    return CloudEvent(
        seq=seq,
        event_id=_uuid(raw.get("event_id"), "event_id"),
        card_id=_uuid(raw.get("card_id"), "card_id"),
        rating=rating,
        reviewed_at=reviewed,
        device_id=device,
        received_at=received,
        response_time_ms=response_time,
        session_id=session,
    )


# ---------------------------------------------------------------------------
# A pull
# ---------------------------------------------------------------------------


def _known_card_ids(db_client) -> Set[str]:
    ids: Set[str] = set()
    for row in quiz_reviews._fetch_all(lambda: db_client.table("quiz_cards").select("id").order("id")):
        ids.add(str(row["id"]))
    return ids


# record_review failures that mean "this event can never be applied", not "try again".
_PERMANENT = ("Quiz card not found", "Review not saved: event_id belongs to another card")


# record_review's note that the review is logged but the card row was not refreshed.
_CACHE_PENDING = "Review saved; the card's schedule will refresh"


def _apply(event: CloudEvent, db_client) -> str:
    """``applied``, ``duplicate`` or a reason it can never be applied. Raises PullError if it might later."""
    response = quiz_reviews.record_review(
        QuizReviewRequest(
            card_id=event.card_id,
            rating=FSRSRating(event.rating),
            response_time_ms=event.response_time_ms,
            session_id=event.session_id,
            event_id=event.event_id,
            reviewed_at=event.reviewed_at,
        ),
        client=db_client,
        now=event.received_at,
        origin="cloud",
        device_id=event.device_id,
    )
    if response.success:
        if response.message.startswith(_CACHE_PENDING):
            return "applied-cache-pending"
        return "duplicate" if response.duplicate else "applied"
    if response.message in _PERMANENT:
        return response.message
    raise PullError(f"seq {event.seq}: {response.message}")


def pull_reviews(
    config: PublishConfig,
    *,
    ingest: Optional[IngestClient] = None,
    db_client=None,
    state: Optional[PostgrestSyncState] = None,
    dry_run: bool = False,
    max_pages: int = MAX_PAGES,
    page: int = PAGE,
    on_progress: Callable[[str], None] = lambda message: None,
) -> PullReport:
    """Pull the phone's reviews into ``review_logs``. Raises ``PullError``; see the module docstring."""
    own_ingest = ingest is None
    ingest = ingest or IngestClient(config)
    if db_client is None:
        from .. import db

        db_client = db.get_client()
    state = state or PostgrestSyncState(db_client)
    report = PullReport(dry_run=dry_run)
    try:
        try:
            cursor = state.read()
        except SyncStateError as exc:
            raise PullError(str(exc), report) from exc
        report.cursor_before = report.cursor_after = after = cursor.seq

        try:
            for _ in range(max_pages):
                on_progress(f"asking the cloud for reviews after seq {after}")
                data = ingest.reviews(after, page)
                events = data.get("events") if isinstance(data, dict) else None
                tip = data.get("through_seq") if isinstance(data, dict) else None
                if not isinstance(events, list) or isinstance(tip, bool) or not isinstance(tip, int):
                    raise PullError("The cloud reader answered something that is not a list of reviews.", report)
                report.cloud_through_seq = tip
                if after > tip:
                    # The cloud holds fewer events than we had folded: it was emptied or recreated.
                    # Everything is idempotent, so start again rather than wait for seq to catch up.
                    logger.warning("cloud reviews: cursor %s is ahead of the cloud (%s); starting again from 0", after, tip)
                    report.cursor_reset = True
                    after = report.cursor_after = 0
                    if not dry_run:
                        with APPLY_LOCK:
                            state.write(PENDING_KEY, 0)
                            state.write(SEQ_KEY, 0)
                    continue
                if not events:
                    report.more = False
                    break
                report.fetched += len(events)
                top = _apply_page(events, report, db_client, state, dry_run, tip)
                after = report.cursor_after = max(after, top)
                report.more = bool(data.get("more"))
                if not report.more:
                    break
            else:
                report.more = True
        except PublishError as exc:
            raise PullError(str(exc), report) from exc
        return report
    finally:
        if own_ingest:
            ingest.close()


def _apply_page(events: List[Any], report: PullReport, db_client, state, dry_run: bool, tip: int) -> int:
    """Apply one page in seq order. Returns the highest seq handled."""
    parsed: List[CloudEvent] = []
    top = 0
    for raw in events:
        seq = raw.get("seq") if isinstance(raw, dict) else None
        try:
            event = parse_event(raw)
        except ValueError as exc:
            eid = str(raw.get("event_id")) if isinstance(raw, dict) else "?"
            report.skipped.append((seq if isinstance(seq, int) else 0, eid[:40], str(exc)))
            logger.warning("cloud reviews: skipping seq %s: %s", seq, exc)
            if isinstance(seq, int):
                top = max(top, seq)
            continue
        parsed.append(event)
        top = max(top, event.seq)
    if not parsed:
        if not dry_run and top:
            with APPLY_LOCK:
                state.write(SEQ_KEY, top)
        return top
    parsed.sort(key=lambda e: e.seq)
    known = _known_card_ids(db_client)

    def run() -> None:
        stale: List[str] = []
        for event in parsed:
            if event.card_id not in known:
                # The card was deleted or regenerated. The review stays in the cloud log; there is
                # no card row to attach it to (review_logs wants its pillar and paper).
                report.skipped.append((event.seq, event.event_id, "its card no longer exists"))
                continue
            if dry_run:
                report.applied += 1
                report.cards.add(event.card_id)
                continue
            outcome = _apply(event, db_client)
            if outcome in ("applied", "applied-cache-pending"):
                report.applied += 1
                if outcome == "applied-cache-pending":
                    stale.append(event.card_id)
            elif outcome == "duplicate":
                report.duplicates += 1
            else:
                report.skipped.append((event.seq, event.event_id, outcome))
                continue
            report.cards.add(event.card_id)
        if stale:
            # The reviews are in the log; the card rows that cache them were not written.
            # Repair them now rather than leave the desk's due queue behind the log.
            try:
                quiz_reviews.reconcile_cards(card_ids=sorted(set(stale)), client=db_client)
            except Exception as exc:  # noqa: BLE001
                raise PullError(f"reviews were logged but their cards could not be refreshed: {exc}") from exc

    if dry_run:
        run()
        return top
    with APPLY_LOCK:
        # Raised before the cards are touched, so a stop half-way leaves a visible gap.
        state.write(PENDING_KEY, top)
        try:
            run()
        except PullError as exc:
            exc.report = report
            raise
        state.write(SEQ_KEY, top)
    return top
