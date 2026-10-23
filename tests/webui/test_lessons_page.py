"""Lessons reading list.

The paper page Lesson tab stays the reading surface. This page is the list
that opens it, and every field the old table showed is still in the row.
"""

from datetime import date
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from webui.lessons_format import build_lesson_list, lesson_headline, lesson_href
from webui.routers import lessons as lessons_router

ROOT = Path(__file__).resolve().parents[2]

PAPER_ID = "arxiv:2402.08593"
OTHER_ID = "arxiv:1706.03762"


class _Pillar:
    def __init__(self, id, name):
        self.id = id
        self.name = name


PILLARS = [
    _Pillar("ml-systems-production", "ML Systems & Production"),
    _Pillar("neural-architectures-language", "Neural Architectures"),
]


def _lessons():
    return [
        {
            "id": "lesson-1",
            "paper_id": PAPER_ID,
            "pillar_id": "ml-systems-production",
            "tl_dr": "Streams the graph and feeds counts to a booster.",
            "takeaways": [
                "Structure can be a column.",
                "Counts beat a windowed GNN.",
                "Keep the window short.",
            ],
            "difficulty": 2,
            "estimated_time": 15,
            "created_at": "2020-01-15T12:00:00+00:00",
        },
        {
            "id": "lesson-2",
            "paper_id": OTHER_ID,
            "pillar_id": "neural-architectures-language",
            "tl_dr": "<img src=x onerror=alert(1)>",
            "takeaways": "not-a-list",
            "difficulty": 1,
            "estimated_time": 8,
            "created_at": "2020-02-02T00:00:00+00:00",
        },
        {
            "id": "lesson-3",
            "paper_id": PAPER_ID,
            "pillar_id": "ml-systems-production",
            "tl_dr": "",
            "takeaways": [],
            "difficulty": None,
            "estimated_time": None,
            "created_at": "not a date",
        },
    ]


def test_lesson_href_encodes_the_colon_and_opens_the_lesson_tab():
    assert lesson_href(PAPER_ID) == "/papers/arxiv%3A2402.08593#lesson"


def test_headline_uses_the_tl_dr_and_invents_a_title_only_when_it_is_missing():
    assert lesson_headline("A sentence.", PAPER_ID) == "A sentence."
    assert lesson_headline("  ", PAPER_ID) == f"Lesson from {PAPER_ID}"
    assert lesson_headline(None, "") == "Lesson"


def test_build_lesson_list_keeps_names_preview_and_the_old_fields():
    view = build_lesson_list(
        _lessons(),
        {PAPER_ID: "Graph Feature Preprocessor", OTHER_ID: ""},
        PILLARS,
        today=date(2026, 10, 23),
    )
    first, second, third = view.rows
    assert first.title == "Streams the graph and feeds counts to a booster."
    assert first.href == "/papers/arxiv%3A2402.08593#lesson"
    assert first.paper_title == "Graph Feature Preprocessor"
    assert first.pillar_name == "ML Systems & Production"
    assert first.pillar_id == "ml-systems-production"
    assert first.when == "15 Jan 2020"
    assert first.when_iso == "2020-01-15"
    assert first.preview == [
        "Structure can be a column.",
        "Counts beat a windowed GNN.",
    ]
    assert first.more_takeaways == 1
    assert first.minutes == 15
    assert first.time_compact == "15m"
    assert first.difficulty_label == "Medium"
    assert first.difficulty_number == "2"
    assert second.paper_title == "Untitled"
    assert second.preview == []
    assert second.difficulty_label == "Easy"
    assert third.title == f"Lesson from {PAPER_ID}"
    assert third.when == "not a date"
    assert third.when_iso == ""
    assert third.minutes is None
    assert third.time_compact == ""
    assert third.difficulty_number == ""
    assert view.count_label == "3 lessons"
    assert view.chips[0].name == "All"
    assert view.chips[0].current is True
    assert view.add_href == "/discovery/"
    assert view.papers_href == "/papers/"


def test_build_lesson_list_filters_to_the_pillar_and_points_onward():
    view = build_lesson_list(
        _lessons(),
        {},
        PILLARS,
        selected_pillar="neural-architectures-language",
        today=date(2026, 10, 23),
    )
    assert [row.paper_id for row in view.rows] == [OTHER_ID]
    assert view.rows[0].paper_title == OTHER_ID
    assert view.count_label == "1 lesson"
    assert view.empty_title == "No lessons in this pillar"
    assert view.add_href == "/discovery/?pillar=neural-architectures-language"
    assert view.papers_href == "/papers/?pillar=neural-architectures-language"
    current = [chip for chip in view.chips if chip.current]
    assert len(current) == 1
    assert current[0].id == "neural-architectures-language"
    assert view.chips[0].current is False


def test_unknown_pillar_query_still_gets_a_chip():
    view = build_lesson_list([], {}, PILLARS, selected_pillar="not-a-pillar")
    assert view.rows == []
    assert view.count_label == ""
    assert view.chips[-1].id == "not-a-pillar"
    assert view.chips[-1].current is True
    assert view.chips[-1].href == "/lessons/?pillar=not-a-pillar"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(lessons_router, "get_pillars_or_empty", lambda limit=100: PILLARS)

    async def lessons(self, pillar=None, limit=50):
        assert limit == 100
        rows = _lessons()
        if pillar:
            rows = [row for row in rows if row["pillar_id"] == pillar]
        return rows

    async def titles(self, paper_ids):
        return {
            PAPER_ID: "Graph Feature Preprocessor",
            OTHER_ID: "Attention Is All You Need",
        }

    monkeypatch.setattr(lessons_router.PostgrestClient, "list_lessons_checked", lessons)
    monkeypatch.setattr(lessons_router.PostgrestClient, "paper_titles", titles)

    app = FastAPI()
    app.include_router(lessons_router.router)
    app.state.templates = Jinja2Templates(directory=str(ROOT / "webui" / "templates"))
    return TestClient(app)


def test_lessons_page_is_a_reading_list_linked_to_the_paper(client):
    body = client.get("/lessons/").text
    assert "Streams the graph and feeds counts to a booster." in body
    assert "Graph Feature Preprocessor" in body
    assert "ML Systems &amp; Production" in body
    assert "Neural Architectures" in body
    assert "Structure can be a column." in body
    assert "Counts beat a windowed GNN." in body
    assert "Keep the window short." not in body
    assert "+1 more" in body
    assert "15 min read" in body
    assert ">Medium<" in body
    assert 'class="lesson-difficulty">2<' in body
    assert 'class="lesson-time">15m<' in body
    assert 'class="lesson-paper-id">arxiv:2402.08593<' in body
    assert 'class="lesson-pillar-id">ml-systems-production<' in body
    assert 'href="/papers/arxiv%3A2402.08593#lesson"' in body
    assert "15 Jan 2020" in body
    assert "2020-01-15T12:00:00" not in body
    assert 'datetime="2020-01-15"' in body
    assert "3 lessons" in body
    assert "Newest first" in body
    assert 'href="/lessons/"' in body
    assert ">Reset<" in body
    assert 'href="/papers/"' in body
    assert 'href="/discovery/"' in body
    assert "Add papers" in body
    assert "Browse papers" in body
    assert 'id="lesson-list"' in body
    assert "/static/lessons.js" in body
    assert ">j<" in body and ">k<" in body
    assert "&lt;img src=x onerror=alert(1)&gt;" in body
    assert "onerror=alert" not in body.split("&lt;img", 1)[0]
    assert f"Lesson from {PAPER_ID}" in body
    assert "No lessons yet" not in body


def test_pillar_query_filters_and_keeps_reset(client):
    body = client.get("/lessons/?pillar=neural-architectures-language").text
    assert "Attention Is All You Need" in body
    assert "Graph Feature Preprocessor" not in body
    assert 'href="/lessons/?pillar=neural-architectures-language"' in body
    assert 'aria-current="true"' in body
    assert 'href="/lessons/"' in body
    assert ">Reset<" in body
    assert 'href="/papers/?pillar=neural-architectures-language"' in body
    assert 'href="/discovery/?pillar=neural-architectures-language"' in body
    assert "1 lesson" in body


def test_empty_lessons_point_at_add_papers(client, monkeypatch):
    async def empty(self, pillar=None, limit=50):
        return []

    monkeypatch.setattr(lessons_router.PostgrestClient, "list_lessons_checked", empty)
    body = client.get("/lessons/").text
    assert "No lessons yet" in body
    assert "Add a paper to start one." in body
    assert 'href="/discovery/"' in body
    assert "ML Systems &amp; Production" in body
    assert ">Reset<" in body
    assert "/static/lessons.js" not in body
    assert "could not be loaded" not in body


def test_filtered_empty_keeps_the_pillar_and_offers_add_papers(client, monkeypatch):
    async def empty(self, pillar=None, limit=50):
        return []

    monkeypatch.setattr(lessons_router.PostgrestClient, "list_lessons_checked", empty)
    body = client.get("/lessons/?pillar=ml-systems-production").text
    assert "No lessons in this pillar" in body
    assert 'href="/discovery/?pillar=ml-systems-production"' in body
    assert 'href="/lessons/"' in body
    assert 'aria-current="true"' in body
    assert "No lessons yet" not in body


def test_load_error_is_not_an_empty_shelf(client, monkeypatch):
    async def boom(self, pillar=None, limit=50):
        raise RuntimeError("down")

    monkeypatch.setattr(lessons_router.PostgrestClient, "list_lessons_checked", boom)
    body = client.get("/lessons/").text
    assert "could not be loaded" in body
    assert "This does not mean there are none." in body
    assert "No lessons yet" not in body
    assert ">Reset<" in body
    assert "ML Systems &amp; Production" in body


def test_missing_titles_keep_the_paper_id(client, monkeypatch):
    async def boom(self, paper_ids):
        raise RuntimeError("down")

    monkeypatch.setattr(lessons_router.PostgrestClient, "paper_titles", boom)
    body = client.get("/lessons/").text
    assert "Paper titles could not be loaded" in body
    assert "Streams the graph and feeds counts to a booster." in body
    assert 'class="lesson-paper-id">arxiv:2402.08593<' in body
    assert "Graph Feature Preprocessor" not in body
    assert 'href="/papers/arxiv%3A2402.08593#lesson"' in body
