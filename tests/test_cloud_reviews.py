"""``cloud pull-reviews``: the phone's reviews into ``review_logs``.

The cloud is a fake with the same contract as the Worker's ``/ingest/reviews``
(events after a cursor, a ``through_seq`` tip, ``more``), and Postgres is the
in-memory stand-in ``test_quiz_reviews`` uses. Nothing touches a network.
"""

import copy
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from nlp_pillars import fsrs_fold
from nlp_pillars.cloud import reviews as cloud_reviews
from nlp_pillars.cloud.publish import PublishConfig, PublishError
from nlp_pillars.cloud.review_state import ReviewCursor, SyncStateMissing
from nlp_pillars.cloud.reviews import PullError, parse_event, pull_reviews
from nlp_pillars.schemas import FSRSRating, QuizReviewRequest
from nlp_pillars.services import quiz_reviews
from test_quiz_reviews import CARD, OTHER, FakeClient, _card_row

CONFIG = PublishConfig("https://reader.test", "id.access", "secret-value-123")
T0 = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)
PHONE = "phone-1234abcd"
IPAD = "ipad-5678efgh"


def stamp(moment):
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def cloud_event(seq, rating=3, at=T0, card=CARD, device=PHONE, received=None, **over):
    return {
        "seq": seq, "event_id": f"00000000-0000-4000-8000-{seq:012d}", "card_id": card, "rating": rating,
        "reviewed_at": stamp(at), "device_id": device,
        "response_time_ms": 1200, "session_id": "session-abcdef12",
        "received_at": stamp(received or at + timedelta(minutes=1)), **over,
    }


class FakeCloud:
    """The Worker's /ingest/reviews: events after a cursor, the tip, and whether more wait."""

    def __init__(self, events=()):
        self.events = list(events)
        self.asked = []
        self.fail = None

    def reviews(self, after, limit=500):
        self.asked.append(after)
        if self.fail:
            raise self.fail
        page = [e for e in self.events if e["seq"] > after][:limit]
        tip = max((e["seq"] for e in self.events), default=0)
        last = page[-1]["seq"] if page else after
        return {"events": copy.deepcopy(page), "through_seq": tip, "more": last < tip}

    def close(self):
        pass


class MemState:
    """cloud_sync_state without Postgres."""

    def __init__(self, seq=0, pending=0):
        self.values = {"reviews_seq": seq, "reviews_pending": pending}
        self.writes = []
        self.read_error = None

    def read(self):
        if self.read_error:
            raise self.read_error
        return ReviewCursor(self.values["reviews_seq"], self.values["reviews_pending"])

    def write(self, key, value):
        self.writes.append((key, value))
        self.values[key] = value

    @property
    def cursor(self):
        return self.read()


@pytest.fixture
def db():
    fake = FakeClient()
    fake.rows["quiz_cards"].append(_card_row(CARD))
    fake.rows["quiz_cards"].append(_card_row(OTHER))
    return fake


def pull(cloud, db, state=None, **kw):
    state = state if state is not None else MemState()
    return pull_reviews(CONFIG, ingest=cloud, db_client=db, state=state, **kw), state


def fold_of(db, card_id=CARD):
    rows = db.logs(card_id)
    return fsrs_fold.fold_events(quiz_reviews._events(rows))


def assert_card_matches_log(db, card_id=CARD):
    folded = fold_of(db, card_id)
    row = db.card(card_id)
    for key, value in fsrs_fold.cache_fields(folded).items():
        if key in ("last_review_date", "last_reviewed", "next_review_date", "due_date"):
            assert fsrs_fold.parse_timestamp(row[key]) == fsrs_fold.parse_timestamp(value), key
        elif isinstance(value, float):
            assert row[key] == pytest.approx(value), key
        else:
            assert row[key] == value, key


# --- one event -------------------------------------------------------------------


def test_an_event_is_checked_against_the_time_it_arrived_not_the_time_it_was_pulled():
    month_late = cloud_event(1, at=T0 - timedelta(days=29), received=T0)
    assert parse_event(month_late).received_at == T0
    with pytest.raises(ValueError, match="30 days"):
        parse_event(cloud_event(1, at=T0 - timedelta(days=31), received=T0))
    with pytest.raises(ValueError, match="future"):
        parse_event(cloud_event(1, at=T0 + timedelta(minutes=6), received=T0))


@pytest.mark.parametrize("over,message", [
    ({"rating": 0}, "rating"), ({"rating": 5}, "rating"), ({"rating": True}, "rating"), ({"rating": "3"}, "rating"),
    ({"seq": 0}, "seq"), ({"seq": "1"}, "seq"),
    ({"event_id": "nope"}, "event_id"), ({"card_id": "AAAAAAAA-0000-4000-8000-00000000000B"}, "lowercase"),
    ({"device_id": "has spaces!"}, "device_id"), ({"session_id": "<b>x</b>"}, "session_id"),
    ({"response_time_ms": -5}, "response_time_ms"), ({"response_time_ms": 9_000_000}, "response_time_ms"),
    ({"reviewed_at": "yesterday"}, "valid timestamp"), ({"received_at": None}, "valid timestamp"),
])
def test_a_malformed_event_is_refused(over, message):
    with pytest.raises(ValueError, match=message):
        parse_event({**cloud_event(1), **over})
    with pytest.raises(ValueError):
        parse_event("not an object")


# --- a pull ----------------------------------------------------------------------


def test_events_land_in_the_log_as_cloud_reviews_and_the_card_is_re_derived(db):
    cloud = FakeCloud([
        cloud_event(1, 3, T0), cloud_event(2, 3, T0 + timedelta(minutes=10)),
        cloud_event(3, 1, T0 + timedelta(days=4), device=IPAD),
    ])
    report, state = pull(cloud, db)
    assert (report.fetched, report.applied, report.duplicates, report.skipped) == (3, 3, 0, [])
    assert state.cursor == ReviewCursor(seq=3, pending=3) and report.cursor_after == 3
    logs = db.logs()
    assert [(r["id"][-1], r["origin"], r["device_id"], r["rating"]) for r in logs] == [
        ("1", "cloud", PHONE, 3), ("2", "cloud", PHONE, 3), ("3", "cloud", IPAD, 1),
    ]
    assert logs[0]["session_id"] == "session-abcdef12" and logs[0]["response_time_ms"] == 1200
    assert logs[0]["review_timestamp"].startswith("2026-03-02T09:00:00")
    assert_card_matches_log(db)
    assert db.card()["state"] == "relearning" and db.card()["lapses"] == 1
    assert db.card(OTHER)["state"] == "new"


def test_the_schedule_the_desk_derives_is_the_one_the_phone_derives(db):
    # tests/fixtures/fsrs/sequences.json is the contract; here, one of its cases through the pull.
    import json
    from pathlib import Path

    fixture = json.loads((Path(__file__).parent / "fixtures/fsrs/sequences.json").read_text())
    case = next(c for c in fixture["cases"] if c["name"] == "mixed_with_lapse")
    events = [
        cloud_event(i + 1, e["rating"], fsrs_fold.parse_timestamp(e["reviewed_at"]),
                    received=fsrs_fold.parse_timestamp(e["reviewed_at"]))
        | {"event_id": e["event_id"]}
        for i, e in enumerate(case["events"])
    ]
    pull(FakeCloud(events), db)
    card = fold_of(db).card
    assert round(card.due.timestamp() * 1000) == case["expected"]["due_ms"]
    assert card.stability == case["expected"]["stability"] and card.difficulty == case["expected"]["difficulty"]


def test_a_second_pull_with_nothing_new_writes_nothing(db):
    cloud = FakeCloud([cloud_event(1), cloud_event(2, at=T0 + timedelta(minutes=10))])
    pull(cloud, db)
    db.calls.clear()
    report, state = pull(cloud, db, MemState(seq=2, pending=2))
    assert report.fetched == 0 and report.applied == 0
    assert db.writes() == [] and state.writes == []
    assert cloud.asked[-1] == 2


def test_the_cursor_is_only_a_hint_pulling_everything_again_records_nothing_twice(db):
    cloud = FakeCloud([cloud_event(1), cloud_event(2, at=T0 + timedelta(minutes=10))])
    pull(cloud, db)
    before = copy.deepcopy(db.rows["review_logs"])
    report, _ = pull(cloud, db, MemState())  # the state table was lost
    assert (report.applied, report.duplicates) == (0, 2)
    assert db.rows["review_logs"] == before
    assert_card_matches_log(db)


def test_the_same_event_delivered_twice_is_one_review(db):
    first = cloud_event(1)
    twin = {**first, "seq": 2}  # the same event_id again, as a retry that landed twice could look
    report, _ = pull(FakeCloud([first, twin]), db)
    assert len(db.logs()) == 1
    assert (report.applied, report.duplicates) == (1, 1)


def test_pages_are_followed_until_the_cloud_has_no_more(db):
    cloud = FakeCloud([cloud_event(i, 3, T0 + timedelta(hours=i)) for i in range(1, 8)])
    report, state = pull(cloud, db, page=3)
    assert cloud.asked == [0, 3, 6]
    assert report.applied == 7 and state.cursor.seq == 7 and not report.more
    assert_card_matches_log(db)


def test_a_pull_stops_after_max_pages_and_says_more_is_waiting(db):
    cloud = FakeCloud([cloud_event(i, 3, T0 + timedelta(hours=i)) for i in range(1, 8)])
    report, state = pull(cloud, db, page=3, max_pages=2)
    assert report.applied == 6 and report.more and state.cursor.seq == 6
    again, state = pull(cloud, db, state, page=3, max_pages=2)
    assert again.applied == 1 and not again.more and state.cursor.seq == 7


def test_an_event_that_can_never_be_applied_is_skipped_and_does_not_stall_the_rest(db):
    cloud = FakeCloud([
        cloud_event(1),
        cloud_event(2, rating=9),                                  # bad rating
        cloud_event(3, card="99999999-9999-4999-8999-999999999999"),  # a card that no longer exists
        cloud_event(4, at=T0 - timedelta(days=40), received=T0),   # stamped a month and more before arrival
        cloud_event(5, at=T0 + timedelta(minutes=10)),
    ])
    report, state = pull(cloud, db)
    assert report.applied == 2 and len(report.skipped) == 3
    assert sorted(seq for seq, _, _ in report.skipped) == [2, 3, 4]
    assert state.cursor.seq == 5, "the cursor moves past events that will never apply"
    assert len(db.logs()) == 2


def test_dry_run_reads_and_writes_nothing(db):
    cloud = FakeCloud([cloud_event(1), cloud_event(2, at=T0 + timedelta(minutes=10))])
    report, state = pull(cloud, db, dry_run=True)
    assert report.applied == 2 and report.dry_run
    assert db.writes() == [] and state.writes == []
    assert "would apply" in report.render()


def test_missing_migration_is_named(db):
    state = MemState()
    state.read_error = SyncStateMissing("Could not read the review cursor: cloud_sync_state is missing. Apply docs/migrations/027_cloud_sync_state.sql by hand.")
    with pytest.raises(PullError, match="027_cloud_sync_state"):
        pull(FakeCloud([cloud_event(1)]), db, state)
    assert db.writes() == []


def test_a_cloud_that_is_down_stops_the_pull_and_keeps_the_cursor(db):
    cloud = FakeCloud([cloud_event(1)])
    cloud.fail = PublishError("GET /ingest/reviews: gave up after 3 tries (HTTP 503)")
    with pytest.raises(PullError, match="gave up"):
        pull(cloud, db, MemState(seq=5, pending=5))
    assert db.writes() == []


def test_a_cloud_answer_that_is_not_reviews_is_refused(db):
    class Odd(FakeCloud):
        def reviews(self, after, limit=500):
            return {"events": "no", "through_seq": 1}

    with pytest.raises(PullError, match="not a list of reviews"):
        pull(Odd(), db)


def test_a_cloud_with_fewer_events_than_the_cursor_means_it_was_rebuilt(db):
    cloud = FakeCloud([cloud_event(1), cloud_event(2, at=T0 + timedelta(minutes=10))])
    pull(cloud, db)
    fresh = FakeCloud([cloud_event(1), cloud_event(2, at=T0 + timedelta(minutes=10))])
    report, state = pull(fresh, db, MemState(seq=40, pending=40))
    assert report.cursor_reset and (report.applied, report.duplicates) == (0, 2)
    assert state.cursor == ReviewCursor(2, 2)


# --- two devices -----------------------------------------------------------------


def desk_review(db, rating, at):
    response = quiz_reviews.record_review(
        QuizReviewRequest(card_id=CARD, rating=FSRSRating(rating), reviewed_at=at),
        client=db, now=at + timedelta(seconds=1),
    )
    assert response.success, response.message


def test_a_desk_review_and_phone_reviews_all_count_in_time_order(db):
    desk_review(db, 3, T0 + timedelta(hours=1))
    desk_review(db, 3, T0 + timedelta(days=1, hours=1))
    # The phone was offline: its reviews were made around the desk's, one of them before both.
    cloud = FakeCloud([
        cloud_event(1, 3, T0), cloud_event(2, 2, T0 + timedelta(days=1)),
        cloud_event(3, 3, T0 + timedelta(days=3), device=IPAD),
    ])
    report, _ = pull(cloud, db)
    assert report.applied == 3 and len(db.logs()) == 5
    folded = fold_of(db)
    assert folded.review_count == 5
    assert [e["origin"] for e in sorted(db.logs(), key=lambda r: r["review_timestamp"])] == [
        "cloud", "desktop", "cloud", "desktop", "cloud",
    ]
    assert_card_matches_log(db)


def test_the_result_does_not_depend_on_which_device_got_there_first(db):
    other = FakeClient()
    other.rows["quiz_cards"].append(_card_row(CARD))
    cloud_events = [cloud_event(1, 3, T0), cloud_event(2, 1, T0 + timedelta(days=1)), cloud_event(3, 4, T0 + timedelta(days=2))]

    # Desk first, then the phone's batch.
    desk_review(db, 3, T0 + timedelta(hours=5))
    desk_review(db, 2, T0 + timedelta(days=1, hours=5))
    pull(FakeCloud(cloud_events), db)

    # The phone's batch first, then the desk reviews.
    pull(FakeCloud(cloud_events), other)
    desk_review(other, 3, T0 + timedelta(hours=5))
    desk_review(other, 2, T0 + timedelta(days=1, hours=5))

    a, b = fold_of(db).card, fold_of(other).card
    assert (a.state, a.step, a.stability, a.difficulty, a.due) == (b.state, b.step, b.stability, b.difficulty, b.due)
    assert_card_matches_log(db)
    assert_card_matches_log(other)


def test_a_late_review_is_logged_with_the_card_as_it_was_then_not_as_it_is_now(db):
    desk_review(db, 3, T0 + timedelta(days=2))
    desk_review(db, 3, T0 + timedelta(days=9))
    pull(FakeCloud([cloud_event(1, 3, T0)]), db)  # made before both
    late = next(r for r in db.logs() if r["origin"] == "cloud")
    assert late["stability"] == 0.0 and late["difficulty"] == 0.0 and "previous_due_date" not in late, (
        "the first review of the card: nothing came before it"
    )
    second = sorted(db.logs(), key=lambda r: r["review_timestamp"])[1]
    # The desk review that now sits second was logged as the card's first when it was made;
    # it is the late review that must not be described by events that came after it.
    assert late["review_timestamp"] < second["review_timestamp"]


# --- failure in the middle -------------------------------------------------------


class Crash(BaseException):
    """The process dying: not an Exception, so nothing in the code under test may swallow it."""


def clean_run(events):
    db = FakeClient()
    db.rows["quiz_cards"].append(_card_row(CARD))
    pull(FakeCloud(events), db)
    return db


EVENTS = [cloud_event(i, r, T0 + timedelta(days=i)) for i, r in enumerate([3, 3, 1, 3, 2, 4], start=1)]


def test_a_crash_between_events_leaves_the_cursor_behind_and_the_next_pull_finishes(db):
    seen = {"n": 0}
    original = db.table

    def table(name):
        query = original(name)
        if name == "review_logs":
            insert = query.insert

            def counting_insert(data):
                seen["n"] += 1
                if seen["n"] == 4:
                    raise Crash()
                return insert(data)

            query.insert = counting_insert
        return query

    db.table = table
    state = MemState()
    with pytest.raises(Crash):
        pull(FakeCloud(EVENTS), db, state)
    assert len(db.logs()) == 3
    assert state.cursor.seq == 0 and state.cursor.pending == 6 and state.cursor.unfinished

    db.table = original
    report, state = pull(FakeCloud(EVENTS), db, state)
    assert (report.applied, report.duplicates) == (3, 3)
    assert state.cursor == ReviewCursor(6, 6)
    reference = clean_run(EVENTS)
    assert [r["id"] for r in db.logs()] == [r["id"] for r in reference.logs()]
    for key in ("state", "stability", "difficulty", "lapses", "repetitions", "due_date"):
        assert db.card()[key] == pytest.approx(reference.card()[key]) if key in ("stability", "difficulty") else db.card()[key] == reference.card()[key]


def test_a_crash_after_the_log_write_and_before_the_card_write_heals_on_the_next_pull(db):
    class Down(BaseException):
        pass

    original = db.table
    state = MemState()
    updates = {"n": 0}

    def table(name):
        query = original(name)
        if name == "quiz_cards":
            update = query.update

            def dying_update(data):
                updates["n"] += 1
                if updates["n"] == 3:
                    raise Down()
                return update(data)

            query.update = dying_update
        return query

    db.table = table
    with pytest.raises(Down):
        pull(FakeCloud(EVENTS), db, state)
    # Three logs are in; the third card write never happened. The card row is behind its log.
    assert len(db.logs()) == 3 and db.card()["repetitions"] == 2
    assert state.cursor.unfinished

    db.table = original
    pull(FakeCloud(EVENTS), db, state)
    assert len(db.logs()) == 6
    assert_card_matches_log(db)
    assert state.cursor == ReviewCursor(6, 6)


def test_a_failed_card_write_is_repaired_in_the_same_pull_or_the_pull_stops(db):
    original = db.table

    def table(name):
        query = original(name)
        if name == "quiz_cards":
            query.update = lambda data: (_ for _ in ()).throw(ConnectionError("PostgREST went away"))
        return query

    db.table = table
    state = MemState()
    with pytest.raises(PullError, match="could not be refreshed"):
        pull(FakeCloud(EVENTS[:2]), db, state)
    assert len(db.logs()) == 2, "the reviews are safe in the log"
    assert state.cursor.seq == 0 and state.cursor.unfinished, "and the pull is not claimed done"

    db.table = original
    report, _ = pull(FakeCloud(EVENTS[:2]), db, state)
    assert (report.applied, report.duplicates) == (0, 2)
    assert_card_matches_log(db)
    assert state.cursor == ReviewCursor(2, 2)


def test_a_database_that_refuses_a_log_write_stops_the_pull_without_moving_the_cursor(db):
    db.fail_insert["review_logs"] = {"message": "connection refused"}
    state = MemState()
    with pytest.raises(PullError, match="seq 1"):
        pull(FakeCloud(EVENTS[:2]), db, state)
    assert db.logs() == [] and state.cursor.seq == 0
    db.fail_insert.clear()
    report, _ = pull(FakeCloud(EVENTS[:2]), db, state)
    assert report.applied == 2 and state.cursor == ReviewCursor(2, 2)


def test_an_event_id_that_is_another_cards_is_skipped_not_retried_forever(db):
    # The id is taken by a review of a different card (a clash that can never resolve).
    desk_review(db, 3, T0)
    clash = cloud_event(1, 3, T0 + timedelta(days=1), card=OTHER)
    clash["event_id"] = db.logs()[0]["id"]
    report, state = pull(FakeCloud([clash]), db)
    assert report.applied == 0 and len(report.skipped) == 1 and state.cursor.seq == 1


def test_secrets_never_reach_the_report():
    report = cloud_reviews.PullReport(cursor_before=1, cursor_after=2)
    assert "secret" not in report.render()
    assert uuid  # the module under test imports uuid for strict ids


# --- the command -----------------------------------------------------------------


def _cli_env(monkeypatch, db, cloud, state):
    from typer.testing import CliRunner

    class Cfg:
        cloud_publish_url = CONFIG.url
        cf_access_client_id = "cid.access"
        cf_access_client_secret = "secret-value-123"

    monkeypatch.setattr("nlp_pillars.config.get_settings", lambda: Cfg())
    monkeypatch.setattr("nlp_pillars.db.get_client", lambda: db)
    monkeypatch.setattr(cloud_reviews, "IngestClient", lambda config: cloud)
    monkeypatch.setattr(cloud_reviews, "PostgrestSyncState", lambda client=None: state)
    return CliRunner()


def test_cli_pulls_prints_the_report_and_exits_nonzero_when_it_stops(monkeypatch, db):
    from nlp_pillars.cli import app

    cloud, state = FakeCloud([cloud_event(1), cloud_event(2, at=T0 + timedelta(minutes=10))]), MemState()
    runner = _cli_env(monkeypatch, db, cloud, state)
    dry = runner.invoke(app, ["cloud", "pull-reviews", "--dry-run"])
    assert dry.exit_code == 0 and "would apply 2" in dry.output and db.logs() == []

    done = runner.invoke(app, ["cloud", "pull-reviews"])
    assert done.exit_code == 0 and "applied 2" in done.output and "cursor 0 -> 2" in done.output
    assert "secret-value-123" not in done.output and len(db.logs()) == 2

    db.fail_insert["review_logs"] = {"message": "connection refused"}
    cloud.events.append(cloud_event(3, at=T0 + timedelta(days=1)))
    failed = runner.invoke(app, ["cloud", "pull-reviews"])
    assert failed.exit_code == 1 and "seq 3" in failed.output


def test_cli_without_credentials_says_which_are_missing(monkeypatch):
    from typer.testing import CliRunner

    from nlp_pillars.cli import app

    class Cfg:
        cloud_publish_url = None
        cf_access_client_id = None
        cf_access_client_secret = None

    monkeypatch.setattr("nlp_pillars.config.get_settings", lambda: Cfg())
    result = CliRunner().invoke(app, ["cloud", "pull-reviews"])
    assert result.exit_code == 1 and "CLOUD_PUBLISH_URL" in result.output
