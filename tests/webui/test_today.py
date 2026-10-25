"""Today: plan rows, streak, and honest counts.

The page only reads. These tests build the view from rows and render it, so
nothing here needs a database.
"""

from __future__ import annotations

import html as html_lib
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from webui.services.postgrest_client import QuickStats
from webui.services.today import (
    TodayInputs,
    build_today_view,
    due_from_rows,
    human_when,
    load_today_view,
    paper_href,
    postgrest_in,
    run_href,
    run_sentence,
    streak_from_timestamps,
)

ROOT = Path(__file__).resolve().parents[2]
BERLIN = ZoneInfo("Europe/Berlin")
NOW = datetime(2026, 10, 23, 8, 0, tzinfo=timezone.utc)  # Friday 10:00 in Berlin


def _pillar(slug="ml-systems-production", name="ML Systems & Production"):
    return SimpleNamespace(id=slug, name=name)


def _stats():
    return QuickStats(total_papers=4, total_lessons=2, total_quiz_cards=6)


def _inputs(**overrides) -> TodayInputs:
    base = dict(
        pillars=[_pillar()],
        quiz_rows=[{
            "next_review_date": "2026-10-22T12:00:00+00:00",
            "pillar_id": "ml-systems-production",
            "paper_id": "arxiv:2402.08593",
        }],
        quiz_truncated=False,
        review_timestamps=["2026-10-22T23:30:00+00:00", "2026-10-21T10:00:00+00:00"],
        lessons=[{
            "id": "lesson-1",
            "paper_id": "arxiv:2402.08593",
            "pillar_id": "ml-systems-production",
            "tl_dr": "Count the laundering shapes.",
            "created_at": "2026-10-22T18:00:00+00:00",
            "estimated_time": 15,
        }],
        papers_by_id={
            "arxiv:2402.08593": {
                "id": "arxiv:2402.08593",
                "title": "Graph Feature Preprocessor",
                "authors": ["Ada Lovelace", "Grace Hopper"],
                "pillar_id": "ml-systems-production",
            },
        },
        episodes=[{
            "id": "11111111-1111-1111-1111-111111111111",
            "paper_id": "arxiv:2402.08593",
            "pillar_id": "ml-systems-production",
            "title": "Deep Dive: Graph Feature Preprocessor",
            "created_at": "2026-10-22T19:00:00+00:00",
            "word_count": 1800,
            "audio_metadata": {"file_name": "episode.mp3", "duration_seconds": 780},
        }],
        runs=[{
            "id": "run-1",
            "pillar_id": "ml-systems-production",
            "kind": "discover",
            "status": "succeeded",
            "finished_at": "2026-10-23T07:00:00+00:00",
            "created_at": "2026-10-23T06:00:00+00:00",
            "papers_processed": 0,
            "papers_failed": 0,
        }],
        active_run=None,
        counts={"ml-systems-production": {"papers": 3, "lessons": 1, "quiz_cards": 5}},
        activity_papers=[{
            "id": "arxiv:2402.08593",
            "title": "Graph Feature Preprocessor",
            "pillar_id": "ml-systems-production",
            "added_at": "2026-10-22T15:00:00+00:00",
        }],
        activity_quiz=[{
            "paper_id": "arxiv:2402.08593",
            "pillar_id": "ml-systems-production",
            "question": "Why can't a GNN score this live?",
            "created_at": "2026-10-22T16:00:00+00:00",
        }],
        stats=_stats(),
    )
    base.update(overrides)
    return TodayInputs(**base)


def _view(**overrides):
    return build_today_view(_inputs(**overrides), now=NOW, tz=BERLIN)


def _visible(html: str) -> str:
    without_scripts = re.sub(r"<script\b[^>]*>.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", without_scripts)
    return html_lib.unescape(text)


def test_streak_uses_the_local_calendar_and_keeps_yesterday():
    # 23:30 UTC is already Friday in Berlin. A UTC calendar would call it Thursday.
    alone = streak_from_timestamps(
        ["2026-10-22T23:30:00+00:00"], now=NOW, tz=BERLIN,
    )
    assert alone.count == 1
    assert alone.dots[4].is_today and alone.dots[4].reviewed

    # Tue–Thu reviewed, Friday not yet: the streak is still 3, and today is open.
    streak = streak_from_timestamps(
        [
            "2026-10-20T10:00:00+00:00",
            "2026-10-21T10:00:00+00:00",
            "2026-10-22T10:00:00+00:00",
        ],
        now=NOW,
        tz=BERLIN,
    )
    assert streak.count == 3
    assert [dot.reviewed for dot in streak.dots] == [False, True, True, True, False, False, False]
    assert streak.dots[4].is_today

    # A gap yesterday ends it, even if earlier days were full.
    broken = streak_from_timestamps(
        ["2026-10-20T10:00:00+00:00", "2026-10-23T07:00:00+00:00"],
        now=NOW,
        tz=BERLIN,
    )
    assert broken.count == 1
    assert broken.dots[4].reviewed


def test_a_failed_review_read_is_not_a_zero_streak():
    view = _view(review_timestamps=None)
    assert view.streak is None
    assert view.review_done is False


def test_due_matches_the_review_queue_and_failure_is_not_zero():
    rows = [
        {"next_review_date": None, "pillar_id": "ml-systems-production", "paper_id": "a"},
        {"next_review_date": "2026-10-22T00:00:00+00:00", "pillar_id": "ml-systems-production", "paper_id": "b"},
        {"next_review_date": "2026-10-24T09:00:00+00:00", "pillar_id": "formal-linguistics-nlp", "paper_id": "c"},
    ]
    due = due_from_rows(rows, now=NOW)
    assert due.known and due.count == 2
    assert due.papers == 2
    assert due.by_pillar == {"ml-systems-production": 2}

    empty = due_from_rows([], now=NOW)
    assert empty.known and empty.count == 0
    assert empty.next_due is None

    later = due_from_rows(
        [{"next_review_date": (NOW + timedelta(days=1)).isoformat(), "pillar_id": "x", "paper_id": "y"}],
        now=NOW,
    )
    assert later.count == 0
    when = human_when(later.next_due, now=NOW, tz=BERLIN)
    assert when[0] == "tomorrow"

    failed = due_from_rows(None, now=NOW)
    assert failed.known is False and failed.count is None
    truncated = due_from_rows(rows, now=NOW, truncated=True)
    assert truncated.known is False


def test_plan_links_do_not_encode_a_slug_as_the_label_and_runs_have_a_place_to_go():
    assert paper_href("arxiv:2402.08593") == "/papers/arxiv%3A2402.08593"
    assert 'arxiv:1' in postgrest_in(["arxiv:1"])
    view = _view()
    assert view.up_next.href == "/papers/arxiv%3A2402.08593"
    assert view.up_next.pillar_name == "ML Systems & Production"
    assert view.primary == "review"
    assert view.review_done is True
    assert view.episode.src == "/api/podcast/audio/episode.mp3"
    assert view.recent[0].sentence == "Discovery finished for ML Systems & Production"
    assert view.recent[0].href.startswith("/discovery/?")
    assert "run=run-1" in view.recent[0].href
    assert view.recent[0].relative == "today"
    assert view.glance[0].papers == 3
    assert view.glance[0].due == 1
    assert view.activity[0].pillar_name == "ML Systems & Production"
    assert "T03:" not in view.activity[0].relative

    upload = {
        "id": "u",
        "kind": "upload",
        "status": "failed",
        "result_added": "true",
        "result_title": "Attention",
        "result_paper_id": "arxiv:1706.03762",
        "pillar_id": "ml-systems-production",
    }
    assert "some steps did not finish" in run_sentence(upload, {"ml-systems-production": "ML Systems & Production"})
    assert run_href(upload) == "/papers/arxiv%3A1706.03762"


def test_unknown_pillar_counts_stay_unknown():
    view = _view(counts={"ml-systems-production": None}, quiz_rows=None)
    tile = view.glance[0]
    assert tile.papers is None and tile.lessons is None and tile.cards is None
    assert tile.due is None
    assert "could not be loaded" in view.review_detail.lower() or "could not" in view.review_detail


@pytest.fixture
def client():
    from webui.routers import analytics, dashboard

    app = FastAPI()
    app.include_router(dashboard.router)
    app.include_router(analytics.router)
    app.state.templates = Jinja2Templates(directory=str(ROOT / "webui" / "templates"))
    return TestClient(app)


def test_today_page_renders_plan_streak_and_hides_slugs_and_timestamps(client, monkeypatch):
    from webui.routers import dashboard

    view = _view()

    async def fake(client=None, **kwargs):
        return view

    monkeypatch.setattr(dashboard, "load_today_view", fake)
    html = client.get("/").text
    visible = _visible(html)

    assert "<title>Today</title>" in html
    assert "Today's plan" in html
    assert "Start review" in html
    assert 'href="/quiz/"' in html
    assert "Graph Feature Preprocessor" in html
    assert "/papers/arxiv%3A2402.08593" in html
    assert 'id="streak"' in html
    assert 'id="service-status-slot"' in html
    assert 'data-status-endpoint="/api/system/status"' in html
    assert 'id="today-run-card"' in html
    assert "RunProgress.attach(" not in (ROOT / "webui" / "static" / "today.js").read_text()
    assert "/api/pipeline-runs/active" in (ROOT / "webui" / "static" / "today.js").read_text()
    assert "ml-systems-production" not in visible
    assert "2026-10-22T15:00:00" not in visible
    assert "ML Systems & Production" in visible
    assert "Why can't a GNN score this live?" in visible


def test_failed_streak_renders_nothing_rather_than_zero(client, monkeypatch):
    from webui.routers import dashboard

    view = _view(review_timestamps=None)

    async def fake(client=None, **kwargs):
        return view

    monkeypatch.setattr(dashboard, "load_today_view", fake)
    html = client.get("/").text
    assert 'id="streak"' not in html
    assert "Review streak" not in html


def test_analytics_shows_pillar_names_and_a_dash_for_unknown(client, monkeypatch):
    from webui.routers import analytics

    view = _view(counts={"ml-systems-production": None}, quiz_rows=None)

    async def fake(client=None, **kwargs):
        return view

    monkeypatch.setattr(analytics, "load_today_view", fake)
    html = client.get("/analytics/").text
    visible = _visible(html)
    assert "<title>Analytics</title>" in html
    assert "By pillar" in html
    assert "ML Systems & Production" in visible
    assert "ml-systems-production" not in visible
    assert "—" in html
    assert ">0<" not in html


@pytest.mark.asyncio
async def test_loader_treats_a_failed_read_as_unknown(monkeypatch):
    class Fake:
        def __init__(self):
            self.calls = []

        async def fetch_rows(self, table, params):
            self.calls.append((table, params))
            if table == "quiz_cards" and "next_review_date" in params["select"]:
                return None
            if table == "review_logs":
                return None
            return [], 0

        async def get_active_pipeline_run(self, pillar_id=None):
            return None

        async def get_quick_stats(self):
            return _stats()

        async def counts_for_pillar(self, pillar_id):
            raise RuntimeError("down")

    monkeypatch.setattr(
        "webui.services.today.get_pillars_or_empty",
        lambda limit=100: [_pillar()],
    )
    fake = Fake()
    view = await load_today_view(fake, now=NOW, tz=BERLIN)
    quiz_call = next(params for table, params in fake.calls if table == "quiz_cards" and "next_review_date" in params["select"])
    assert quiz_call["user_id"] == "eq.default_user"
    assert view.due.known is False
    assert view.due.count is None
    assert view.streak is None
    assert view.glance[0].papers is None
    assert view.glance[0].due is None


@pytest.mark.asyncio
async def test_loader_does_not_pull_the_read_along_timeline_for_the_episode_card(monkeypatch):
    class Fake:
        def __init__(self):
            self.calls = []

        async def fetch_rows(self, table, params):
            self.calls.append((table, params))
            if table == "podcast_scripts":
                return [{
                    "id": "s1", "paper_id": "p1", "pillar_id": "ml-systems-production",
                    "title": "Deep Dive", "created_at": "2026-10-22T13:50:00+00:00", "word_count": 1900,
                    "audio_file": "episode.mp3", "audio_seconds": 654.7,
                }], 1
            return [], 0

        async def get_active_pipeline_run(self, pillar_id=None):
            return None

        async def get_quick_stats(self):
            return _stats()

        async def counts_for_pillar(self, pillar_id):
            return {"papers": 0, "lessons": 0, "quiz_cards": 0}

    monkeypatch.setattr("webui.services.today.get_pillars_or_empty", lambda limit=100: [_pillar()])
    fake = Fake()
    view = await load_today_view(fake, now=NOW, tz=BERLIN)
    select = next(p for t, p in fake.calls if t == "podcast_scripts")["select"]
    assert "audio_metadata->file_name" in select
    assert "audio_metadata," not in select and not select.endswith("audio_metadata")
    assert view.episode is not None
    assert view.episode.src == "/api/podcast/audio/episode.mp3"
