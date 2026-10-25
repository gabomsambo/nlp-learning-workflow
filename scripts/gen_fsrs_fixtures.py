#!/usr/bin/env python3
"""Write tests/fixtures/fsrs/sequences.json: review histories and the card py-fsrs derives.

The phone folds the same histories with ts-fsrs (fuzz off), and its test reads
this file, so the two implementations are held to one set of numbers. Each case
lists the events, the card after every event in fold order, and the final card.

Nothing here reads the clock or a random source, so a re-run on the same py-fsrs
writes the same bytes. ``tests/test_fsrs_fold.py`` re-generates in memory and
fails if the committed file has drifted, so a py-fsrs bump shows up as a diff
to review rather than as a silent change in what the phone is checked against.

    python scripts/gen_fsrs_fixtures.py          # rewrite the file
    python scripts/gen_fsrs_fixtures.py --check  # exit 1 if it differs
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nlp_pillars import fsrs_fold  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "fsrs" / "sequences.json"
T0 = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)

Hist = List[Tuple[int, timedelta]]  # (rating, wait after the previous review)


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _uuid(case: int, index: int) -> str:
    return f"00000000-0000-4000-8000-{case:04d}{index:08d}"


def _events(case: int, history: Hist) -> List[fsrs_fold.FoldEvent]:
    moment = T0
    out = []
    for index, (rating, wait) in enumerate(history):
        moment = moment + wait
        out.append(fsrs_fold.FoldEvent(_uuid(case, index), rating, moment))
    return out


def _on_due(ratings: List[int]) -> List[fsrs_fold.FoldEvent]:
    """A history where every review happens exactly when the card is due."""
    scheduler = fsrs_fold.build_scheduler()
    card = None
    moment = T0
    out = []
    for index, rating in enumerate(ratings):
        event = fsrs_fold.FoldEvent(_uuid(len(ratings), index), rating, moment)
        card = fsrs_fold.apply_event(scheduler, card, event)
        out.append(event)
        moment = card.due
    return out


def _lcg_history(seed: int, count: int) -> List[fsrs_fold.FoldEvent]:
    """Mostly-Good ratings, reviewed a little early or late, from a fixed LCG."""
    state = seed
    scheduler = fsrs_fold.build_scheduler()
    card = None
    moment = T0
    out = []
    for index in range(count):
        state = (state * 1103515245 + 12345) % 2**31
        roll = state % 100
        rating = 1 if roll < 15 else 2 if roll < 35 else 3 if roll < 90 else 4
        state = (state * 1103515245 + 12345) % 2**31
        drift = timedelta(minutes=(state % 2881) - 1440)  # +/- one day
        if card is not None:
            moment = max(moment + timedelta(minutes=1), card.due + drift)
        event = fsrs_fold.FoldEvent(_uuid(900, index), rating, moment)
        card = fsrs_fold.apply_event(scheduler, card, event)
        out.append(event)
    return out


def _case(name: str, note: str, events: List[fsrs_fold.FoldEvent], *, shuffle: bool = False,
          extra_events: List[fsrs_fold.FoldEvent] = ()) -> Dict[str, Any]:
    listed = list(events) + list(extra_events)
    if shuffle:
        listed = listed[1::2] + listed[0::2]  # fixed, non-chronological order
    scheduler = fsrs_fold.build_scheduler()
    after_each = []
    card = None
    for event in fsrs_fold.ordered(events):
        card = fsrs_fold.apply_event(scheduler, card, event)
        after_each.append(fsrs_fold.card_snapshot(card))
    folded = fsrs_fold.fold_events(listed)
    return {
        "name": name,
        "note": note,
        "events": [
            {"event_id": e.event_id, "rating": e.rating, "reviewed_at": _stamp(e.reviewed_at)}
            for e in listed
        ],
        "after_each": after_each,
        "expected": {
            **fsrs_fold.card_snapshot(folded.card),
            "review_count": folded.review_count,
            "lapses": folded.lapses,
        },
    }


def build() -> Dict[str, Any]:
    scheduler = fsrs_fold.build_scheduler()
    day, minute = timedelta(days=1), timedelta(minutes=1)
    cases = [
        _case("six_good_on_due", "Good every time, each review exactly when due.",
              _on_due([3] * 6)),
        _case("mixed_with_lapse", "Good, Hard, Good, Again at due, then relearn.",
              _on_due([3, 2, 3, 3, 1, 3, 3, 4])),
        _case("again_then_good", "Again on a new card, then Good.",
              _events(3, [(1, timedelta()), (3, minute * 5), (3, minute * 12), (3, day * 2)])),
        _case("easy_three_times", "Easy on a new card, repeated on due.",
              _on_due([4, 4, 4])),
        _case("hard_at_first_step", "Hard keeps a new card on its first learning step.",
              _events(5, [(2, timedelta()), (2, minute * 2), (3, minute * 4), (3, minute * 11), (3, day * 3)])),
        _case("same_day_burst", "Several reviews inside one day, then a long gap.",
              _events(6, [(3, timedelta()), (3, minute * 3), (2, minute * 40), (3, minute * 200), (3, day * 6)])),
        _case("late_review", "A review 20 days after it was due.",
              _events(7, [(3, timedelta()), (3, minute * 10), (3, day * 1), (3, day * 24), (3, day * 60)])),
        _case("relearning_steps", "Lapse a graduated card, then walk its relearning step.",
              _events(8, [(3, timedelta()), (3, minute * 10), (3, day * 1), (1, day * 3), (2, minute * 11), (3, minute * 30), (3, day * 2)])),
        _case("ends_mid_learning", "A new card rated Good once: still Learning, on step 1.",
              _events(10, [(3, timedelta())])),
        _case("ends_in_relearning", "A graduated card that lapses on the last event.",
              _events(11, [(3, timedelta()), (3, minute * 10), (3, day * 1), (1, day * 3)])),
        _case("tied_timestamps", "Two events with the same timestamp fold in event_id order.",
              _events(9, [(3, timedelta()), (1, minute * 10), (3, timedelta()), (3, day * 2)])),
        _case("listed_out_of_order", "The events are listed out of order; the fold sorts them.",
              _on_due([3, 3, 2, 3, 3]), shuffle=True),
        _case("duplicate_delivery", "One event delivered twice counts once.",
              _on_due([3, 3, 3]), extra_events=[_on_due([3, 3, 3])[1]]),
        _case("forty_reviews", "A seeded history of 40 reviews, early and late.",
              _lcg_history(2026, 40)),
    ]
    return {
        "format": 1,
        "about": (
            "Review histories and the card py-fsrs derives from them. Fold each case's events "
            "ordered by (reviewed_at, event_id), counting a repeated event_id once, starting from "
            "an empty card, and compare with `expected` (and `after_each`, one entry per distinct "
            "event in fold order). due_ms and last_review_ms are epoch milliseconds; stability and "
            "difficulty are exact doubles."
        ),
        "scheduler": {
            "library": "py-fsrs",
            "version": metadata.version("fsrs"),
            "enable_fuzzing": scheduler.enable_fuzzing,
            "desired_retention": scheduler.desired_retention,
            "maximum_interval": scheduler.maximum_interval,
            "learning_steps_seconds": [int(s.total_seconds()) for s in scheduler.learning_steps],
            "relearning_steps_seconds": [int(s.total_seconds()) for s in scheduler.relearning_steps],
            "parameters": list(scheduler.parameters),
        },
        "state_codes": {"1": "Learning", "2": "Review", "3": "Relearning"},
        "cases": cases,
    }


def render() -> str:
    return json.dumps(build(), indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed file differs")
    args = parser.parse_args()
    text = render()
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT.relative_to(ROOT)} is out of date; run scripts/gen_fsrs_fixtures.py")
            return 1
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
