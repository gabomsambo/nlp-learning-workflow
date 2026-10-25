"""Recording a review: the log first, the card row second, both recomputable.

Everything here runs against an in-memory stand-in for the PostgREST client.
It speaks the same builder methods as ``nlp_pillars.db.TableQuery`` and can be
told to fail a write, which is what the failure-injection tests need.
"""

import copy
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from nlp_pillars import fsrs_fold
from nlp_pillars.fsrs_fold import FoldEvent
from nlp_pillars.schemas import FSRSRating, QuizReviewRequest
from nlp_pillars.services import quiz_reviews
from nlp_pillars.services.quiz_reviews import record_review, reconcile_cards

NOW = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
CARD = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"


class FakeTable:
    def __init__(self, client, name):
        self.client = client
        self.name = name
        self.filters = []
        self._offset = 0
        self._limit = None

    # query builder -----------------------------------------------------
    def select(self, columns="*"):
        return self

    def eq(self, column, value):
        self.filters.append((column, lambda v, value=value: str(v) == str(value)))
        return self

    def in_(self, column, values):
        wanted = {str(v) for v in values}
        self.filters.append((column, lambda v: str(v) in wanted))
        return self

    def order(self, column, desc=False):
        return self

    def limit(self, count):
        self._limit = count
        return self

    def offset(self, count):
        self._offset = count
        return self

    def _matching(self):
        rows = self.client.rows[self.name]
        return [r for r in rows if all(test(r.get(col)) for col, test in self.filters)]

    # reads and writes --------------------------------------------------
    def execute(self):
        self.client.calls.append(("select", self.name))
        if self.client.fail_select == self.name:
            return {"data": None, "error": {"message": "select failed"}}
        rows = [copy.deepcopy(r) for r in self._matching()]
        rows = rows[self._offset:]
        if self._limit is not None:
            rows = rows[: self._limit]
        return {"data": rows, "error": None}

    def insert(self, data):
        self.client.calls.append(("insert", self.name))
        failure = self.client.fail_insert.get(self.name)
        if failure is not None:
            if isinstance(failure, Exception):
                raise failure
            return {"data": None, "error": failure}
        rows = self.client.rows[self.name]
        row = {"origin": "desktop", **copy.deepcopy(data)}
        row.setdefault("id", str(uuid.uuid4()))
        if any(r["id"] == row["id"] for r in rows):
            return {"data": None, "error": {"code": 409, "message": "Resource already exists"}}
        rows.append(row)
        return {"data": [row], "error": None}

    def update(self, data):
        self.client.calls.append(("update", self.name))
        if self.name in self.client.fail_update:
            raise self.client.fail_update[self.name]
        for row in self._matching():
            row.update(copy.deepcopy(data))
        return {"data": None, "error": None}


class FakeClient:
    def __init__(self):
        self.rows = {"quiz_cards": [], "review_logs": []}
        self.calls = []
        self.fail_insert = {}
        self.fail_update = {}
        self.fail_select = None

    def table(self, name):
        return FakeTable(self, name)

    def writes(self):
        return [call for call in self.calls if call[0] in ("insert", "update")]

    def logs(self, card_id=CARD):
        return [r for r in self.rows["review_logs"] if r["card_id"] == card_id]

    def card(self, card_id=CARD):
        return next(r for r in self.rows["quiz_cards"] if r["id"] == card_id)


def _card_row(card_id=CARD):
    return {
        "id": card_id, "paper_id": "paper-1", "pillar_id": "llm-theory-practice",
        "question": "Why?", "answer": "Because.", "difficulty": 2, "question_type": "factual",
        "user_id": "default_user", "difficulty_fsrs": 0.0, "stability": 0.0, "state": "new",
        "lapses": 0, "repetitions": 0, "interval": 1, "ease_factor": 2.5,
        "last_review_date": None, "next_review_date": None, "last_reviewed": None,
        "due_date": "2026-03-01T00:00:00+00:00", "created_at": "2026-03-01T00:00:00+00:00",
    }


@pytest.fixture
def client():
    fake = FakeClient()
    fake.rows["quiz_cards"].append(_card_row())
    return fake


def _request(rating=3, event_id=None, reviewed_at=None, card_id=CARD):
    return QuizReviewRequest(
        card_id=card_id, rating=FSRSRating(rating), event_id=event_id, reviewed_at=reviewed_at,
        response_time_ms=900, session_id="s1",
    )


def _eid(n):
    return f"00000000-0000-4000-8000-{n:012d}"


def _at(**delta):
    return NOW - timedelta(**delta)


def _folded_row(client, card_id=CARD):
    events = [
        FoldEvent(r["id"], r["rating"], fsrs_fold.parse_timestamp(r["review_timestamp"]))
        for r in client.logs(card_id)
    ]
    return fsrs_fold.cache_fields(fsrs_fold.fold_events(events))


def _cache_matches(client, card_id=CARD):
    row = client.card(card_id)
    return all(quiz_reviews._same(k, row.get(k), v) for k, v in _folded_row(client, card_id).items())


# --------------------------------------------------------------- the happy path


def test_a_review_writes_the_log_first_then_the_card(client):
    result = record_review(_request(3, _eid(1), _at(minutes=1)), client=client, now=NOW)

    assert result.success and not result.duplicate
    assert result.event_id == _eid(1)
    assert client.writes() == [("insert", "review_logs"), ("update", "quiz_cards")]

    log = client.logs()[0]
    assert log["id"] == _eid(1)
    assert log["rating"] == 3 and log["session_id"] == "s1" and log["response_time_ms"] == 900
    assert log["review_timestamp"] == _at(minutes=1).isoformat()
    assert log["difficulty"] == 0.0 and log["stability"] == 0.0  # the card before the review
    assert "origin" not in log or log["origin"] == "desktop"
    assert "device_id" not in log  # unmigrated databases must still take the row

    row = client.card()
    assert row["state"] == "learning" and row["repetitions"] == 1
    assert row["created_at"] == "2026-03-01T00:00:00+00:00"  # a review does not rewrite it
    assert result.next_review_date == fsrs_fold.parse_timestamp(row["next_review_date"])
    assert result.card.state.value == "learning"


def test_the_log_records_the_card_as_it_was_before_the_review(client):
    record_review(_request(3, _eid(1), _at(days=3)), client=client, now=NOW)
    record_review(_request(3, _eid(2), _at(days=3) + timedelta(minutes=10)), client=client, now=NOW)
    third = record_review(_request(3, _eid(3), _at(days=1)), client=client, now=NOW)

    log = next(r for r in client.logs() if r["id"] == _eid(3))
    before = fsrs_fold.fold_events([
        FoldEvent(_eid(1), 3, _at(days=3)), FoldEvent(_eid(2), 3, _at(days=3) + timedelta(minutes=10)),
    ])
    assert log["stability"] == before.card.stability
    assert log["difficulty"] == before.card.difficulty
    assert log["previous_due_date"] == before.card.due.isoformat()
    assert log["days_overdue"] == max(0, (_at(days=1) - before.card.due).days)
    assert 0 < log["retrievability"] <= 1
    assert third.card.state.value == "review"


def test_a_run_of_reviews_leaves_the_row_equal_to_the_fold_of_the_log(client):
    ratings = [3, 3, 2, 3, 1, 3, 3]
    for index, rating in enumerate(ratings):
        when = _at(days=20) + timedelta(days=index * 2)
        assert record_review(_request(rating, _eid(index), when), client=client, now=NOW).success
    assert len(client.logs()) == len(ratings)
    assert _cache_matches(client)
    assert client.card()["lapses"] == 1


def test_a_review_without_event_id_or_time_gets_both(client):
    result = record_review(_request(3), client=client, now=NOW)
    assert result.success
    log = client.logs()[0]
    uuid.UUID(log["id"])
    assert log["review_timestamp"] == NOW.isoformat()  # injected now, millisecond-clean


def test_an_out_of_order_review_is_folded_where_it_belongs(client):
    record_review(_request(3, _eid(2), _at(days=1)), client=client, now=NOW)
    record_review(_request(3, _eid(1), _at(days=5)), client=client, now=NOW)  # arrives late

    assert _cache_matches(client)
    in_order = fsrs_fold.fold_events([FoldEvent(_eid(1), 3, _at(days=5)), FoldEvent(_eid(2), 3, _at(days=1))])
    assert client.card()["next_review_date"] == in_order.card.due.isoformat()


# ------------------------------------------------------------------- idempotency


def test_the_same_event_twice_records_one_review(client):
    first = record_review(_request(3, _eid(1), _at(minutes=5)), client=client, now=NOW)
    row_after_first = copy.deepcopy(client.card())
    second = record_review(_request(3, _eid(1), _at(minutes=5)), client=client, now=NOW)

    assert len(client.logs()) == 1
    assert second.success and second.duplicate and not first.duplicate
    assert second.next_review_date == first.next_review_date
    assert client.card() == row_after_first
    assert client.calls.count(("insert", "review_logs")) == 1


def test_a_repeated_event_repairs_a_stale_card_row(client):
    client.fail_update["quiz_cards"] = httpx.ConnectError("db hiccup")
    record_review(_request(3, _eid(1), _at(minutes=5)), client=client, now=NOW)
    assert client.card()["state"] == "new"  # the cache missed it

    client.fail_update.clear()
    again = record_review(_request(3, _eid(1), _at(minutes=5)), client=client, now=NOW)
    assert again.duplicate and len(client.logs()) == 1
    assert _cache_matches(client) and client.card()["state"] == "learning"


def test_an_event_id_that_belongs_to_another_card_is_refused(client):
    client.rows["quiz_cards"].append(_card_row(OTHER))
    record_review(_request(3, _eid(1), _at(minutes=5), card_id=OTHER), client=client, now=NOW)
    before = copy.deepcopy(client.rows)

    result = record_review(_request(3, _eid(1), _at(minutes=4)), client=client, now=NOW)

    assert not result.success and "another card" in result.message
    assert client.rows == before
    assert client.logs(CARD) == []


# ---------------------------------------------------- log first: failure injection


def test_when_the_log_insert_is_refused_the_card_is_not_touched(client):
    client.fail_insert["review_logs"] = {"message": "permission denied for table review_logs"}
    before = copy.deepcopy(client.rows)

    result = record_review(_request(3, _eid(1), _at(minutes=1)), client=client, now=NOW)

    assert not result.success
    assert "permission denied" in result.message and "not saved" in result.message
    assert client.rows == before
    assert ("update", "quiz_cards") not in client.calls


def test_when_the_log_insert_raises_the_card_is_not_touched(client):
    client.fail_insert["review_logs"] = httpx.ConnectError("connection refused")
    before = copy.deepcopy(client.rows)

    result = record_review(_request(3, _eid(1), _at(minutes=1)), client=client, now=NOW)

    assert not result.success and "Review failed" in result.message
    assert client.rows == before
    assert ("update", "quiz_cards") not in client.calls


def test_when_the_card_update_fails_the_review_is_still_recorded(client):
    client.fail_update["quiz_cards"] = httpx.HTTPStatusError(
        "boom", request=httpx.Request("PATCH", "http://x"), response=httpx.Response(500)
    )

    result = record_review(_request(3, _eid(1), _at(minutes=1)), client=client, now=NOW)

    assert result.success  # the review is in the log
    assert "will refresh" in result.message
    assert len(client.logs()) == 1
    assert client.card()["state"] == "new"  # stale cache
    assert result.card.state.value == "learning"  # but the answer given is the derived one

    client.fail_update.clear()
    report = reconcile_cards(client=client)
    assert report.updated == [CARD]
    assert _cache_matches(client)


def test_an_unknown_card_writes_nothing(client):
    result = record_review(_request(3, _eid(1), card_id=OTHER), client=client, now=NOW)
    assert not result.success and result.message == "Quiz card not found"
    assert client.writes() == []


def test_an_unreadable_log_stops_the_review_before_any_write(client):
    client.fail_select = "review_logs"
    result = record_review(_request(3, _eid(1), _at(minutes=1)), client=client, now=NOW)
    assert not result.success
    assert client.writes() == []


@pytest.mark.parametrize("kwargs, text", [
    ({"event_id": "not-a-uuid"}, "UUID"),
    ({"reviewed_at": NOW + timedelta(hours=1)}, "future"),
    ({"reviewed_at": NOW - timedelta(days=45)}, "days old"),
    ({"reviewed_at": datetime(2026, 3, 10, 11, 0)}, "UTC offset"),
])
def test_an_unusable_event_id_or_time_writes_nothing(client, kwargs, text):
    result = record_review(_request(3, **kwargs), client=client, now=NOW)
    assert not result.success and text in result.message
    assert client.calls == []


# --------------------------------------------------------------------- reconcile


def _reviewed_client(n=4):
    client = FakeClient()
    client.rows["quiz_cards"] += [_card_row(CARD), _card_row(OTHER)]
    for index in range(n):
        record_review(_request(3, _eid(index), _at(days=20) + timedelta(days=index * 3)), client=client, now=NOW)
    return client


def test_reconcile_repairs_a_stale_row_and_leaves_a_good_one_alone():
    client = _reviewed_client()
    client.card()["next_review_date"] = "2030-01-01T00:00:00+00:00"
    client.card()["state"] = "new"
    client.calls.clear()

    report = reconcile_cards(client=client)

    assert report.checked == 2 and report.updated == [CARD]
    assert report.unchanged == 1  # the never-reviewed card
    assert _cache_matches(client)
    assert client.calls.count(("update", "quiz_cards")) == 1

    client.calls.clear()
    again = reconcile_cards(client=client)
    assert again.updated == [] and again.unchanged == 2
    assert client.writes() == []


def test_reconcile_dry_run_reports_and_writes_nothing():
    client = _reviewed_client()
    client.card()["stability"] = 99.0
    before = copy.deepcopy(client.rows)

    report = reconcile_cards(client=client, dry_run=True)

    assert report.updated == [CARD] and report.dry_run
    assert "would update" in report.render()
    assert client.rows == before


def test_reconcile_does_not_erase_a_schedule_that_has_no_log():
    client = FakeClient()
    row = _card_row()
    row.update(state="review", stability=12.0, next_review_date="2026-04-01T00:00:00+00:00")
    client.rows["quiz_cards"].append(row)

    report = reconcile_cards(client=client)

    assert report.unbacked == [CARD] and report.updated == []
    assert client.card()["state"] == "review" and client.card()["stability"] == 12.0
    assert "no review log" in report.render()


def test_reconcile_counts_events_of_deleted_cards_and_applies_them_to_nothing():
    client = _reviewed_client()
    client.rows["quiz_cards"] = [r for r in client.rows["quiz_cards"] if r["id"] != CARD]
    report = reconcile_cards(client=client)
    assert report.orphan_events == 4
    assert report.checked == 1


def test_reconcile_can_be_limited_to_some_cards():
    client = _reviewed_client()
    client.card()["stability"] = 99.0
    client.card(OTHER)["state"] = "review"
    report = reconcile_cards(client=client, card_ids=[CARD])
    assert report.checked == 1 and report.updated == [CARD]
    assert client.card(OTHER)["state"] == "review"  # not asked about


def test_reconcile_pages_through_more_rows_than_one_request(monkeypatch):
    monkeypatch.setattr(quiz_reviews, "_PAGE", 2)
    client = FakeClient()
    ids = [f"00000000-0000-4000-8000-{n:012d}" for n in range(5)]
    client.rows["quiz_cards"] = [_card_row(i) for i in ids]
    report = reconcile_cards(client=client)
    assert report.checked == 5


def test_load_events_returns_an_entry_for_every_card_asked_about():
    client = _reviewed_client(2)
    found = quiz_reviews.load_events(client, [CARD, OTHER])
    assert [e.event_id for e in found[CARD]] == [_eid(0), _eid(1)]
    assert found[OTHER] == []
