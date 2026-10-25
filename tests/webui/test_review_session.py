"""Review session: one card, intervals, empty state, and the visible FSRS fallback.

Template rendering goes through a tiny app, not create_app(): that lifespan
starts a scheduler and sweeps the database, and this page does not need either.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nlp_pillars.db import QuizLookupError, get_cards_for_review, list_quiz_schedule_rows
from nlp_pillars.schemas import Pillar, QuizCard
from webui.services.review_session import (
    earliest_future,
    empty_copy,
    format_due,
    format_interval,
    interval_previews,
    json_for_script,
    session_cards,
)

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)


def _pillar():
    return Pillar(
        id="llm-theory-practice",
        name="LLM theory and practice",
        goal="Understand how large language models are trained and used.",
        created_at=NOW,
        updated_at=NOW,
    )


def _card(question="Why does attention scale the dot product?", difficulty=1, card_id="card-1"):
    return QuizCard(
        id=card_id,
        paper_id="review-fixture-1",
        pillar_id="llm-theory-practice",
        question=question,
        answer="By the square root of the key dimension.",
        difficulty=difficulty,
        next_review_date=NOW - timedelta(hours=1),
    )


def test_format_interval_uses_the_scheduling_delta():
    assert format_interval(timedelta(seconds=20)) == "<1m"
    assert format_interval(timedelta(seconds=-5)) == "<1m"
    assert format_interval(timedelta(minutes=8)) == "8m"
    assert format_interval(timedelta(minutes=70)) == "1h"
    assert format_interval(timedelta(days=1)) == "1d"
    assert format_interval(timedelta(days=6)) == "6d"
    assert format_interval(timedelta(days=45)) == "2mo"
    assert format_interval(timedelta(days=400)) == "1y"


def test_format_due_names_today_tomorrow_and_a_later_date():
    assert format_due(NOW + timedelta(hours=3), NOW) == "today at 15:00 UTC"
    assert format_due(NOW + timedelta(days=1), NOW) == "tomorrow at 12:00 UTC"
    assert format_due(datetime(2026, 11, 2, 9, tzinfo=timezone.utc), NOW) == "2 Nov 2026"


def test_empty_copy_names_the_next_card_and_a_filter_miss():
    title, message = empty_copy(next_due=NOW + timedelta(days=1), now=NOW, filtered=False)
    assert title == "Nothing due"
    assert message == "Next card due tomorrow at 12:00 UTC."
    assert empty_copy(next_due=None, now=NOW, filtered=True)[1] == "Nothing matches this filter."
    assert empty_copy(next_due=None, now=NOW, filtered=False)[1] == "No cards are scheduled."


def test_earliest_future_skips_due_cards_and_other_difficulties():
    rows = [
        {"next_review_date": (NOW - timedelta(hours=1)).isoformat(), "difficulty": 1},
        {"next_review_date": (NOW + timedelta(days=3)).isoformat(), "difficulty": 2},
        {"next_review_date": (NOW + timedelta(days=1)).isoformat(), "difficulty": 1},
    ]
    assert earliest_future(rows, NOW) == NOW + timedelta(days=1)
    assert earliest_future(rows, NOW, difficulty=2) == NOW + timedelta(days=3)
    assert earliest_future(rows, NOW, difficulty=3) is None


def test_interval_previews_call_scheduling_without_changing_the_card():
    card = _card()
    seen = []

    def schedule(copy, rating):
        seen.append((copy, rating))
        assert copy is not card
        return NOW + timedelta(days=rating)

    previews = interval_previews(card, now=NOW, schedule=schedule)
    assert [item["rating"] for item in previews] == [1, 2, 3, 4]
    assert [item["label"] for item in previews] == ["Again", "Hard", "Good", "Easy"]
    assert previews[0]["interval"] == "1d"
    assert previews[3]["interval"] == "4d"
    assert card.next_review_date == NOW - timedelta(hours=1)
    assert len(seen) == 4


def test_script_json_cannot_break_out_of_the_script_tag():
    payload = session_cards(
        [_card(question='</script><script>alert(1)</script>')],
        now=NOW,
        schedule=lambda _card, rating: NOW + timedelta(days=rating),
        pillar_names={"llm-theory-practice": "LLM theory and practice"},
    )
    encoded = json_for_script({"cards": payload})
    assert "<script>" not in encoded
    assert "\\u003c" in encoded
    decoded = json.loads(encoded)
    assert decoded["cards"][0]["difficulty"] == 1
    assert decoded["cards"][0]["pillarName"] == "LLM theory and practice"
    assert "<script>" in decoded["cards"][0]["question"]


def test_fallback_rows_do_not_invent_an_interval():
    rows = session_cards(
        [{
            "id": "abc",
            "paper_id": "p",
            "pillar_id": "llm-theory-practice",
            "question": "Q?",
            "answer": "A.",
            "difficulty": "2",
        }],
        now=NOW,
        schedule=lambda *_args: (_ for _ in ()).throw(AssertionError("schedule must not run")),
        pillar_names={},
    )
    assert rows[0]["difficulty"] == 2
    assert all(item["interval"] is None for item in rows[0]["intervals"])
    assert rows[0]["pillarName"] == "llm-theory-practice"


def test_lookup_failure_is_not_an_empty_queue(monkeypatch):
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr("nlp_pillars.db.get_client", boom)
    assert get_cards_for_review() == []
    with pytest.raises(QuizLookupError):
        get_cards_for_review(raise_on_error=True)
    with pytest.raises(QuizLookupError):
        list_quiz_schedule_rows()


# ---------------------------------------------------------------- the page


@pytest.fixture
def page_client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.templating import Jinja2Templates
    from fastapi.testclient import TestClient

    from webui.routers import quiz as quiz_router

    monkeypatch.setattr(quiz_router, "get_pillars_or_empty", lambda limit=100: [_pillar()])
    # No review history: every card previews from its row, and nothing dials the database.
    monkeypatch.setattr(quiz_router, "get_card_review_events", lambda ids: {})
    monkeypatch.setattr(
        quiz_router,
        "preview_schedule",
        lambda card, rating: datetime.now(timezone.utc) + timedelta(minutes={1: 8, 2: 70, 3: 60 * 24, 4: 60 * 24 * 6}[rating]),
    )

    app = FastAPI()
    app.include_router(quiz_router.router)
    app.state.templates = Jinja2Templates(directory=str(ROOT / "webui" / "templates"))
    return TestClient(app)


def _payload(body: str) -> dict:
    start = body.index('id="review-session-data">') + len('id="review-session-data">')
    end = body.index("</script>", start)
    return json.loads(body[start:end])


def test_session_is_one_card_with_intervals_and_a_human_difficulty(page_client, monkeypatch):
    from webui.routers import quiz as quiz_router

    monkeypatch.setattr(quiz_router, "get_cards_for_review", lambda **kwargs: [_card()])
    body = page_client.get("/quiz/").text

    assert "DifficultyLevel" not in body
    assert 'name="pillar"' in body
    assert 'name="difficulty"' in body
    assert "Space reveals the answer" in body
    assert ">1 of 1<" in body
    assert "paper-modal.js" in body
    assert body.index("paper-modal.js") < body.index("review-session.js")
    assert "NLPReviewShortcuts" in (ROOT / "webui" / "static" / "review-session.js").read_text()

    payload = _payload(body)
    card = payload["cards"][0]
    assert card["difficulty"] == 1
    assert card["question"] == "Why does attention scale the dot product?"
    assert [item["interval"] for item in card["intervals"]] == ["8m", "1h", "1d", "6d"]
    assert payload["fsrsFallback"] is False


def test_filters_keep_their_url_params_and_narrow_the_session(page_client, monkeypatch):
    from webui.routers import quiz as quiz_router

    cards = [
        _card("Easy question", difficulty=1, card_id="easy"),
        _card("Hard question", difficulty=3, card_id="hard"),
    ]
    monkeypatch.setattr(quiz_router, "get_cards_for_review", lambda **kwargs: list(cards))
    body = page_client.get("/quiz/?pillar=llm-theory-practice&difficulty=hard").text

    assert 'value="hard" selected' in body or 'value="hard"  selected' in body
    assert "LLM theory and practice" in body
    payload = _payload(body)
    assert [card["id"] for card in payload["cards"]] == ["hard"]
    assert "Hard question" in body
    assert "Easy question" not in body


def test_empty_queue_names_the_next_card(page_client, monkeypatch):
    from webui.routers import quiz as quiz_router

    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).replace(
        hour=9, minute=0, second=0, microsecond=0
    )
    monkeypatch.setattr(quiz_router, "get_cards_for_review", lambda **kwargs: [])
    monkeypatch.setattr(
        quiz_router,
        "list_quiz_schedule_rows",
        lambda **kwargs: [{"next_review_date": tomorrow.isoformat(), "difficulty": 2}],
    )
    body = page_client.get("/quiz/").text
    assert "Nothing due" in body
    assert "Next card due tomorrow at 09:00 UTC." in body
    assert "Scheduling lookup failed" not in body


def test_filtered_empty_offers_a_way_back(page_client, monkeypatch):
    from webui.routers import quiz as quiz_router

    monkeypatch.setattr(quiz_router, "get_cards_for_review", lambda **kwargs: [])
    monkeypatch.setattr(quiz_router, "list_quiz_schedule_rows", lambda **kwargs: [])
    body = page_client.get("/quiz/?difficulty=easy").text
    assert "Nothing due" in body
    assert "Nothing matches this filter." in body
    assert 'href="/quiz/"' in body
    assert "Clear filters" in body


def test_fsrs_fallback_is_visible_and_does_not_print_the_enum(page_client, monkeypatch):
    from webui.routers import quiz as quiz_router
    from webui.services.postgrest_client import PostgrestClient

    def fail(**kwargs):
        raise QuizLookupError("connection refused")

    async def fallback(self, pillar=None, difficulty=None, limit=50):
        return [{
            "id": "fallback-1",
            "paper_id": "review-fixture-1",
            "pillar_id": "llm-theory-practice",
            "question": "Fallback question",
            "answer": "Fallback answer",
            "difficulty": 1,
        }]

    monkeypatch.setattr(quiz_router, "get_cards_for_review", fail)
    monkeypatch.setattr(PostgrestClient, "list_quiz_cards", fallback)
    body = page_client.get("/quiz/?pillar=llm-theory-practice").text

    assert "Scheduling lookup failed" in body
    assert "connection refused" in body
    assert "not only the ones that are due" in body
    assert "DifficultyLevel" not in body
    payload = _payload(body)
    assert payload["fsrsFallback"] is True
    assert payload["cards"][0]["difficulty"] == 1
    assert payload["cards"][0]["intervals"][0]["interval"] is None
    assert "Fallback question" in body


def test_intervals_come_from_the_review_log_when_the_log_is_readable(page_client, monkeypatch):
    """A card on learning step 1 graduates on Good; the row alone would say ten minutes."""
    from nlp_pillars import fsrs_fold
    from webui.routers import quiz as quiz_router

    started = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=11)
    history = [fsrs_fold.FoldEvent("e1", 3, started)]
    monkeypatch.setattr(quiz_router, "get_cards_for_review", lambda **kwargs: [_card()])
    monkeypatch.setattr(quiz_router, "get_card_review_events", lambda ids: {"card-1": history})

    payload = _payload(page_client.get("/quiz/").text)
    intervals = {item["rating"]: item["interval"] for item in payload["cards"][0]["intervals"]}

    assert intervals[3].endswith("d")  # graduated to review, not a learning step in minutes
    assert intervals[1] in {"<1m", "1m"}  # Again goes back to the first learning step
    assert payload["fsrsFallback"] is False


def test_unreadable_review_log_falls_back_to_the_rows_without_failing_the_page(page_client, monkeypatch):
    from webui.routers import quiz as quiz_router

    def unreadable(ids):
        raise RuntimeError("log unreadable")

    monkeypatch.setattr(quiz_router, "get_cards_for_review", lambda **kwargs: [_card()])
    monkeypatch.setattr(quiz_router, "get_card_review_events", unreadable)

    payload = _payload(page_client.get("/quiz/").text)
    assert [i["interval"] for i in payload["cards"][0]["intervals"]] == ["8m", "1h", "1d", "6d"]


def test_a_card_the_log_does_not_cover_previews_from_its_row(monkeypatch):
    from webui.routers import quiz as quiz_router

    monkeypatch.setattr(quiz_router, "preview_schedule", lambda card, rating: NOW + timedelta(days=rating))
    schedule = quiz_router.schedule_from_log({"other": []}, NOW)
    assert schedule(_card(), 2) == NOW + timedelta(days=2)
