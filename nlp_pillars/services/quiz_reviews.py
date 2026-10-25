"""Quiz reviews as events: the log is written first, the card row is a cache.

``review_logs`` is the source of truth (see ``fsrs_fold``). A review therefore
does two writes in a fixed order, and PostgREST cannot make them one
transaction:

1. insert the ``review_logs`` row, keyed by the review's ``event_id``;
2. re-derive the card from its whole log and PATCH ``quiz_cards``.

If (1) fails nothing else is touched. If (2) fails the review is still
recorded, the response says the schedule refresh is pending, and the next
review of that card, a repeat of the same ``event_id``, or ``reconcile_cards``
repairs the row. The old order was the reverse, so a failed log insert left a
rescheduled card with no history, and nothing could recompute it.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .. import fsrs_fold
from ..schemas import FSRSRating, QuizReviewRequest, QuizReviewResponse, ReviewLog

logger = logging.getLogger(__name__)

# PostgREST paging, and ids per `in.(...)` filter (they travel in the URL).
_PAGE = 1000
_ID_CHUNK = 100

_CACHE_COLUMNS = (
    "id,state,difficulty_fsrs,stability,lapses,repetitions,last_review_date,"
    "last_reviewed,next_review_date,due_date,interval"
)
_TIMESTAMP_FIELDS = {"last_review_date", "last_reviewed", "next_review_date", "due_date"}


class ReviewStoreError(Exception):
    """The review log or the card table could not be read or written."""


def _error_text(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error.get("details") or error)
    return str(error)


def _is_duplicate(error: Any) -> bool:
    """A primary-key clash on insert: PostgREST's 409, or Postgres 23505."""
    if not isinstance(error, dict):
        return False
    return error.get("code") in (409, "409", "23505")


def _first_line(text: str) -> str:
    lines = text.strip().splitlines()
    return (lines[0] if lines else "unknown error")[:240]


def _event_id(value: Optional[str]) -> str:
    if value is None:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        raise ValueError("event_id must be a UUID") from None


def _events(rows: Iterable[Dict[str, Any]]) -> List[fsrs_fold.FoldEvent]:
    return [
        fsrs_fold.FoldEvent(
            event_id=str(row["id"]),
            rating=int(row["rating"]),
            reviewed_at=fsrs_fold.parse_timestamp(row["review_timestamp"]),
        )
        for row in rows
    ]


def _fetch_all(make_query) -> List[Dict[str, Any]]:
    """Every row of a query, paged. ``make_query`` returns a fresh ordered query."""
    rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        response = make_query().offset(offset).limit(_PAGE).execute()
        if response["error"]:
            raise ReviewStoreError(_first_line(_error_text(response["error"])))
        page = response["data"] or []
        rows.extend(page)
        if len(page) < _PAGE:
            return rows
        offset += _PAGE


def load_events(client, card_ids: List[str]) -> Dict[str, List[fsrs_fold.FoldEvent]]:
    """Each card's review events, keyed by card id. A never-reviewed card maps to ``[]``."""
    found: Dict[str, List[fsrs_fold.FoldEvent]] = {str(c): [] for c in card_ids}
    ids = list(found)
    for start in range(0, len(ids), _ID_CHUNK):
        chunk = ids[start:start + _ID_CHUNK]
        rows = _fetch_all(
            lambda chunk=chunk: client.table("review_logs")
            .select("id,card_id,rating,review_timestamp")
            .in_("card_id", chunk)
            .order("id")
        )
        for row in rows:
            found.setdefault(str(row["card_id"]), []).extend(_events([row]))
    return found


def _write_cache(client, card_id: str, folded: fsrs_fold.FoldedCard) -> Dict[str, Any]:
    fields = fsrs_fold.cache_fields(folded)
    client.table("quiz_cards").eq("id", card_id).update(fields)
    return fields


def _failure(card, now: datetime, message: str, event_id: Optional[str] = None) -> QuizReviewResponse:
    return QuizReviewResponse(
        success=False,
        card=card,
        next_review_date=now,
        message=message,
        event_id=event_id,
    )


def record_review(
    request: QuizReviewRequest, *, client=None, now: Optional[datetime] = None
) -> QuizReviewResponse:
    """Record one review and re-derive the card from its log."""
    from .. import db

    now = now or fsrs_fold.now_ms()
    try:
        event_id = _event_id(request.event_id)
        reviewed_at = (
            fsrs_fold.validate_reviewed_at(request.reviewed_at, now)
            if request.reviewed_at is not None
            else now
        )
    except ValueError as exc:
        return _failure(None, now, f"Review not saved: {exc}", request.event_id)

    try:
        client = client or db.get_client()
        response = (client.table("quiz_cards").select("*")
                    .eq("id", request.card_id).limit(1).execute())
        if response["error"] or not response["data"]:
            return _failure(None, now, "Quiz card not found", event_id)
        row = response["data"][0]
        card = db._dict_to_quiz_card(row)

        def card_events() -> List[fsrs_fold.FoldEvent]:
            return _events(_fetch_all(
                lambda: client.table("review_logs")
                .select("id,card_id,rating,review_timestamp")
                .eq("card_id", request.card_id)
                .order("id")
            ))

        prior = card_events()
        duplicate = any(e.event_id == event_id for e in prior)

        if not duplicate:
            scheduler = fsrs_fold.build_scheduler()
            before = fsrs_fold.fold_events(prior, scheduler)
            before_card = before.card if before else None
            due = before_card.due if before_card else None
            log = ReviewLog(
                id=event_id,
                card_id=str(request.card_id),
                user_id=card.user_id,
                pillar_id=card.pillar_id,
                paper_id=card.paper_id,
                rating=FSRSRating(request.rating),
                review_timestamp=reviewed_at,
                difficulty=(before_card.difficulty or 0.0) if before_card else 0.0,
                stability=(before_card.stability or 0.0) if before_card else 0.0,
                retrievability=(
                    scheduler.get_card_retrievability(before_card, reviewed_at)
                    if before_card and before_card.stability else None
                ),
                previous_due_date=due,
                days_overdue=max(0, (reviewed_at - due).days) if due else 0,
                session_id=request.session_id,
                response_time_ms=request.response_time_ms,
            )
            # The log goes first. If this does not land, nothing else is touched.
            inserted = client.table("review_logs").insert(db._review_log_to_dict(log))
            if inserted["error"]:
                if _is_duplicate(inserted["error"]):
                    # Lost a race with the same event_id, or the id is another card's.
                    if not _owned_by_card(client, event_id, request.card_id):
                        return _failure(card, now, "Review not saved: event_id belongs to another card", event_id)
                    duplicate = True
                    prior = card_events()
                else:
                    reason = _first_line(_error_text(inserted["error"]))
                    logger.error("Review log insert failed for card %s: %s", request.card_id, reason)
                    return _failure(card, now, f"Review not saved: {reason}", event_id)
            else:
                prior = prior + [fsrs_fold.FoldEvent(event_id, int(request.rating), reviewed_at)]
    except ReviewStoreError as exc:
        return _failure(None, now, f"Review failed: {exc}", event_id)
    except Exception as exc:
        logger.error("Failed to record review of card %s: %s", request.card_id, exc)
        return _failure(None, now, f"Review failed: {_first_line(str(exc))}", event_id)

    folded = fsrs_fold.fold_events(prior)
    if folded is None:  # unreachable: prior holds this event
        return _failure(card, now, "Review failed: no events to fold", event_id)

    message = "Card reviewed successfully"
    try:
        fields = _write_cache(client, str(request.card_id), folded)
        card = db._dict_to_quiz_card({**row, **fields})
    except Exception as exc:
        # The review is in the log; the row is a cache that reconcile repairs.
        logger.warning("Card %s review recorded but its schedule was not saved: %s", request.card_id, exc)
        message = "Review saved; the card's schedule will refresh on the next sync"
        card = db._dict_to_quiz_card({**row, **fsrs_fold.cache_fields(folded)})
    if duplicate:
        message = "Review was already recorded"

    return QuizReviewResponse(
        success=True,
        card=card,
        next_review_date=folded.card.due,
        message=message,
        event_id=event_id,
        duplicate=duplicate,
    )


def _owned_by_card(client, event_id: str, card_id: str) -> bool:
    found = client.table("review_logs").select("id,card_id").eq("id", event_id).limit(1).execute()
    if found["error"] or not found["data"]:
        raise ReviewStoreError("event_id clashed but the row could not be read back")
    return str(found["data"][0]["card_id"]) == str(card_id)


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------


@dataclass
class ReconcileReport:
    """What a pass over the cache found. With ``dry_run`` nothing was written."""

    dry_run: bool = False
    checked: int = 0
    unchanged: int = 0
    updated: List[str] = field(default_factory=list)
    # A schedule with no log behind it. Left alone: the cache is all that is left of it.
    unbacked: List[str] = field(default_factory=list)
    # Events whose card row no longer exists. Kept in the log, applied to nothing.
    orphan_events: int = 0

    def render(self) -> str:
        verb = "would update" if self.dry_run else "updated"
        lines = [
            f"Checked {self.checked} cards: {self.unchanged} already match their review log, "
            f"{len(self.updated)} {verb}.",
        ]
        if self.unbacked:
            lines.append(
                f"{len(self.unbacked)} cards have a schedule but no review log; left unchanged."
            )
        if self.orphan_events:
            lines.append(f"{self.orphan_events} log events belong to cards that no longer exist.")
        return "\n".join(lines)


def _same(key: str, stored: Any, derived: Any) -> bool:
    if stored is None or derived is None:
        return stored == derived
    if key in _TIMESTAMP_FIELDS:
        return fsrs_fold.parse_timestamp(stored) == fsrs_fold.parse_timestamp(derived)
    if isinstance(derived, float):
        return abs(float(stored) - derived) < 1e-9
    return stored == derived


def reconcile_cards(
    *, card_ids: Optional[List[str]] = None, dry_run: bool = False, client=None
) -> ReconcileReport:
    """Re-derive every card's schedule from its review log and fix the rows that differ."""
    from .. import db

    client = client or db.get_client()
    report = ReconcileReport(dry_run=dry_run)
    wanted = {str(c) for c in card_ids} if card_ids is not None else None

    cards = _fetch_all(
        lambda: client.table("quiz_cards").select(_CACHE_COLUMNS).order("id")
    )
    log_rows = _fetch_all(
        lambda: client.table("review_logs")
        .select("id,card_id,rating,review_timestamp")
        .order("id")
    )
    by_card: Dict[str, List[Dict[str, Any]]] = {}
    for row in log_rows:
        by_card.setdefault(str(row["card_id"]), []).append(row)

    known = {str(c["id"]) for c in cards}
    report.orphan_events = sum(len(v) for k, v in by_card.items() if k not in known)

    for row in cards:
        card_id = str(row["id"])
        if wanted is not None and card_id not in wanted:
            continue
        report.checked += 1
        folded = fsrs_fold.fold_events(_events(by_card.get(card_id, [])))
        if folded is None:
            if row.get("state") not in (None, "new"):
                report.unbacked.append(card_id)
            else:
                report.unchanged += 1
            continue
        derived = fsrs_fold.cache_fields(folded)
        if all(_same(key, row.get(key), value) for key, value in derived.items()):
            report.unchanged += 1
            continue
        report.updated.append(card_id)
        if not dry_run:
            client.table("quiz_cards").eq("id", card_id).update(derived)
    return report
