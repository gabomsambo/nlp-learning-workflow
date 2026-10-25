"""The review log folds to a card: reproducibly, in order, and the same as step-by-step.

The committed fixtures (tests/fixtures/fsrs/sequences.json) are what the
phone's ts-fsrs fold is later checked against, so they are tested here from
both sides: the fold must reproduce them, and they must still be what
scripts/gen_fsrs_fixtures.py writes.
"""

import importlib.util
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fsrs import Rating, Scheduler, State

from nlp_pillars import fsrs_fold
from nlp_pillars.fsrs_fold import FoldEvent, build_scheduler, fold_events
from nlp_pillars.schemas import FSRSRating, QuizCard
from nlp_pillars.services.fsrs_service import FSRSService

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "fsrs" / "sequences.json"
T0 = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)


def _load_fixtures():
    return json.loads(FIXTURES.read_text())


def _events(case):
    return [
        FoldEvent(e["event_id"], e["rating"], fsrs_fold.parse_timestamp(e["reviewed_at"]))
        for e in case["events"]
    ]


CASES = _load_fixtures()["cases"]


# ------------------------------------------------------------------ fixtures


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_fold_reproduces_every_fixture(case):
    folded = fold_events(_events(case))
    expected = case["expected"]
    snapshot = fsrs_fold.card_snapshot(folded.card)
    for key, value in snapshot.items():
        assert value == expected[key], key
    assert folded.review_count == expected["review_count"]
    assert folded.lapses == expected["lapses"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_fixture_after_each_matches_a_prefix_fold(case):
    ordered = fsrs_fold.ordered(_events(case))
    assert len(ordered) == len(case["after_each"])
    for index, after in enumerate(case["after_each"]):
        folded = fold_events(ordered[: index + 1])
        assert fsrs_fold.card_snapshot(folded.card) == after


def test_committed_fixtures_are_what_the_generator_writes():
    spec = importlib.util.spec_from_file_location(
        "gen_fsrs_fixtures", ROOT / "scripts" / "gen_fsrs_fixtures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert FIXTURES.read_text() == module.render(), (
        "tests/fixtures/fsrs/sequences.json is stale: run scripts/gen_fsrs_fixtures.py "
        "and review the diff"
    )


def test_fixtures_cover_every_card_state_and_the_hard_cases():
    states = {case["expected"]["state"] for case in CASES}
    assert states == {1, 2, 3}  # Learning, Review, Relearning
    assert any(c["expected"]["step"] == 1 for c in CASES)  # mid-learning step survives
    names = {c["name"] for c in CASES}
    assert {"tied_timestamps", "listed_out_of_order", "duplicate_delivery"} <= names


def test_fixture_scheduler_block_describes_the_scheduler_used():
    block = _load_fixtures()["scheduler"]
    scheduler = build_scheduler()
    assert block["enable_fuzzing"] is False
    assert block["parameters"] == list(scheduler.parameters)
    assert block["desired_retention"] == scheduler.desired_retention
    assert block["learning_steps_seconds"] == [60, 600]
    assert block["relearning_steps_seconds"] == [600]


# ------------------------------------------------------- fold equals incremental


def _histories():
    rng = random.Random(7)
    for _ in range(25):
        moment = T0
        events = []
        for index in range(rng.randint(1, 18)):
            moment += timedelta(minutes=rng.choice([1, 8, 30, 600, 60 * 24, 60 * 24 * 4, 60 * 24 * 40]))
            events.append(FoldEvent(f"{index:08x}", rng.choice([1, 2, 3, 3, 3, 4]), moment))
        yield events


def test_fold_equals_applying_each_review_to_the_card_before_it():
    scheduler = build_scheduler()
    for events in _histories():
        card = None
        for event in events:
            card = fsrs_fold.apply_event(scheduler, card, event)
        folded = fold_events(events)
        assert fsrs_fold.card_snapshot(folded.card) == fsrs_fold.card_snapshot(card)


def test_one_more_review_extends_the_fold_of_the_rest():
    scheduler = build_scheduler()
    for events in _histories():
        *rest, last = events
        base = fold_events(rest)
        extended = fsrs_fold.apply_event(scheduler, base.card if base else None, last)
        assert fsrs_fold.card_snapshot(fold_events(events).card) == fsrs_fold.card_snapshot(extended)


# ------------------------------------------------------------- determinism


def test_replaying_one_history_gives_one_answer():
    events = [FoldEvent(f"e{i}", 3, T0 + timedelta(days=i * i)) for i in range(6)]
    results = {fold_events(events).card.due for _ in range(20)}
    assert len(results) == 1


def test_the_scheduler_has_fuzz_off_and_that_is_what_makes_replay_reproducible():
    assert build_scheduler().enable_fuzzing is False
    assert FSRSService().fsrs.enable_fuzzing is False

    # The default py-fsrs scheduler fuzzes. Replaying it gives different due dates,
    # which is why a merged log could not be folded on two devices.
    fuzzed = Scheduler()
    assert fuzzed.enable_fuzzing is True
    dues = set()
    for _ in range(20):
        card = None
        for i in range(6):
            card = fsrs_fold.apply_event(fuzzed, card, FoldEvent(f"e{i}", 3, T0 + timedelta(days=i * 6)))
        dues.add(card.due)
    assert len(dues) > 1


# ---------------------------------------------------------------- ordering


def test_input_order_does_not_matter():
    events = [FoldEvent(f"e{i}", r, T0 + timedelta(days=i)) for i, r in enumerate([3, 2, 3, 1, 3, 4])]
    expected = fsrs_fold.card_snapshot(fold_events(events).card)
    for seed in range(5):
        shuffled = list(events)
        random.Random(seed).shuffle(shuffled)
        assert fsrs_fold.card_snapshot(fold_events(shuffled).card) == expected


def test_events_at_the_same_instant_fold_in_event_id_order():
    a = FoldEvent("00000000-0000-4000-8000-00000000000a", 1, T0)
    b = FoldEvent("00000000-0000-4000-8000-00000000000b", 3, T0)
    assert [e.event_id for e in fsrs_fold.ordered([b, a])] == [a.event_id, b.event_id]
    assert fsrs_fold.card_snapshot(fold_events([b, a]).card) == fsrs_fold.card_snapshot(fold_events([a, b]).card)


def test_a_repeated_event_id_counts_once():
    event = FoldEvent("e1", 3, T0)
    once = fold_events([event])
    twice = fold_events([event, FoldEvent("e1", 3, T0), event])
    assert twice.review_count == 1
    assert fsrs_fold.card_snapshot(once.card) == fsrs_fold.card_snapshot(twice.card)


def test_offsets_are_compared_as_instants():
    plus_two = timezone(timedelta(hours=2))
    a = FoldEvent("a", 3, datetime(2026, 3, 2, 11, 0, tzinfo=plus_two))  # 09:00 UTC
    b = FoldEvent("b", 3, T0 + timedelta(minutes=10))
    assert [e.event_id for e in fsrs_fold.ordered([b, a])] == ["a", "b"]
    assert fold_events([b, a]).last_event.event_id == "b"


def test_never_reviewed_folds_to_none():
    assert fold_events([]) is None


@pytest.mark.parametrize("rating", [0, 5, -1])
def test_a_rating_outside_one_to_four_is_refused(rating):
    with pytest.raises(ValueError):
        FoldEvent("e", rating, T0)


def test_a_naive_timestamp_is_refused():
    with pytest.raises(ValueError):
        FoldEvent("e", 3, datetime(2026, 3, 2, 9, 0))
    with pytest.raises(ValueError):
        fsrs_fold.parse_timestamp("2026-03-02T09:00:00")


def test_parse_timestamp_takes_z_and_postgres_offsets():
    assert fsrs_fold.parse_timestamp("2026-03-02T09:00:00.250Z") == T0 + timedelta(milliseconds=250)
    assert fsrs_fold.parse_timestamp("2026-03-02T11:00:00+02:00") == T0


# -------------------------------------------------------- the step is kept


def test_a_second_good_in_learning_graduates_because_the_fold_keeps_the_step():
    """The row has no step column; the fold does not need one.

    One Good moves a new card to learning step 1. The second Good, ten minutes
    later, must graduate it to Review. A card rebuilt from the row restarts at
    step 0 and stays in learning on every Good.
    """
    first = FoldEvent("a", 3, T0)
    second = FoldEvent("b", 3, T0 + timedelta(minutes=10))

    after_one = fold_events([first])
    assert after_one.card.state == State.Learning and after_one.card.step == 1

    after_two = fold_events([first, second])
    assert after_two.card.state == State.Review
    assert after_two.card.due - second.reviewed_at >= timedelta(days=1)


def test_preview_due_is_what_the_review_then_produces():
    history = [FoldEvent("a", 3, T0), FoldEvent("b", 3, T0 + timedelta(minutes=10))]
    at = T0 + timedelta(days=2)
    for rating in (1, 2, 3, 4):
        previewed = fsrs_fold.preview_due(history, rating, at)
        actual = fold_events(history + [FoldEvent("c", rating, at)]).card.due
        assert previewed == actual
    assert fsrs_fold.preview_due([], 3, at) == fold_events([FoldEvent("x", 3, at)]).card.due


# --------------------------------------------------------------- the cache


def test_cache_fields_describe_a_learning_card_and_a_lapse():
    learning = fold_events([FoldEvent("a", 3, T0)])
    fields = fsrs_fold.cache_fields(learning)
    assert fields["state"] == "learning"
    assert fields["repetitions"] == 1 and fields["lapses"] == 0
    assert fields["next_review_date"] == fields["due_date"] == "2026-03-02T09:10:00+00:00"
    assert fields["last_review_date"] == fields["last_reviewed"] == "2026-03-02T09:00:00+00:00"
    assert fields["interval"] == 1

    history = [FoldEvent("a", 3, T0), FoldEvent("b", 3, T0 + timedelta(minutes=10)),
               FoldEvent("c", 3, T0 + timedelta(days=3)), FoldEvent("d", 1, T0 + timedelta(days=9))]
    lapsed = fsrs_fold.cache_fields(fold_events(history))
    assert lapsed["state"] == "relearning"
    assert lapsed["lapses"] == 1 and lapsed["repetitions"] == 4


def test_an_again_while_still_learning_is_not_a_lapse():
    folded = fold_events([FoldEvent("a", 1, T0), FoldEvent("b", 1, T0 + timedelta(minutes=2))])
    assert folded.lapses == 0


def test_cache_fields_interval_is_whole_days_and_at_least_one():
    graduated = fold_events([FoldEvent(f"e{i}", 3, T0 + timedelta(days=i * 10)) for i in range(4)])
    fields = fsrs_fold.cache_fields(graduated)
    assert fields["interval"] == round((graduated.card.due - graduated.card.last_review).total_seconds() / 86400)
    assert fields["interval"] > 1


# ------------------------------------------------------ the legacy service


def _card():
    return QuizCard(id="c", paper_id="p", pillar_id="x", question="q", answer="a")


def test_schedule_card_review_takes_the_review_time():
    service = FSRSService()
    when = T0 + timedelta(days=400)
    _card_after, due = service.schedule_card_review(_card(), FSRSRating.GOOD, review_datetime=when)
    assert due == when + timedelta(minutes=10)
    assert _card_after.last_review_date == when


def test_schedule_card_review_is_deterministic_without_fuzz():
    service = FSRSService()
    card = _card().model_copy(update={
        "state": "review", "stability": 40.0, "difficulty_fsrs": 5.0,
        "last_review_date": T0, "next_review_date": T0 + timedelta(days=40),
    })
    results = [service.schedule_card_review(card, FSRSRating.GOOD, review_datetime=T0 + timedelta(days=40))
               for _ in range(10)]
    assert all(updated.stability != 40.0 for updated, _due in results)  # FSRS ran, not the SM-2 fallback
    assert len({due for _updated, due in results}) == 1


# ---------------------------------------------------------- reviewed_at limits


def test_validate_reviewed_at_bounds():
    now = T0
    assert fsrs_fold.validate_reviewed_at(now - timedelta(days=29), now) == now - timedelta(days=29)
    assert fsrs_fold.validate_reviewed_at(now + timedelta(minutes=4), now)
    with pytest.raises(ValueError, match="future"):
        fsrs_fold.validate_reviewed_at(now + timedelta(minutes=6), now)
    with pytest.raises(ValueError, match="days old"):
        fsrs_fold.validate_reviewed_at(now - timedelta(days=31), now)
    with pytest.raises(ValueError, match="UTC offset"):
        fsrs_fold.validate_reviewed_at(datetime(2026, 3, 2, 9, 0), now)


def test_now_ms_has_no_sub_millisecond_part():
    for _ in range(50):
        assert fsrs_fold.now_ms().microsecond % 1000 == 0


def test_rating_enum_values_match_py_fsrs():
    assert [Rating(r).name for r in (1, 2, 3, 4)] == ["Again", "Hard", "Good", "Easy"]
