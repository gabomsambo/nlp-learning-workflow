"""Pure formatting for the library and the paper page."""

from datetime import date

from webui.library_format import (
    authors_line,
    build_library,
    build_paper_page,
    humanise_when,
    paper_href,
    pillar_css_class,
    safe_pdf_url,
)


def test_paper_href_encodes_colon():
    assert paper_href("arxiv:2402.08593") == "/papers/arxiv%3A2402.08593"


def test_safe_pdf_url_matches_the_modal_rule():
    assert safe_pdf_url("https://arxiv.org/pdf/1") == "https://arxiv.org/pdf/1"
    assert safe_pdf_url(" file:///app/data/uploads/a.pdf ") == "file:///app/data/uploads/a.pdf"
    assert safe_pdf_url("/papers/") == "/papers/"
    assert safe_pdf_url("javascript:alert(1)") is None
    assert safe_pdf_url("data:text/html,hi") is None
    assert safe_pdf_url("") is None
    assert safe_pdf_url(None) is None


def test_pillar_class_rejects_anything_that_is_not_a_slug():
    assert pillar_css_class("ml-systems-production") == "pillar-ml-systems-production"
    assert pillar_css_class("Foo Bar") == ""
    assert pillar_css_class("a/b") == ""
    assert pillar_css_class(None) == ""


def test_authors_line_truncates_and_can_keep_all():
    names = ["A", "B", "C", "D"]
    assert authors_line(names) == "A, B, C +1"
    assert authors_line(names, limit=0) == "A, B, C, D"
    assert authors_line("Ada") == ""
    assert authors_line([" Ada ", "", None]) == "Ada"


def test_humanise_when_uses_the_calendar_date():
    today = date(2026, 10, 23)
    assert humanise_when("2026-10-23T23:30:00+00:00", today=today) == "today"
    assert humanise_when("2026-10-22T00:00:00Z", today=today) == "yesterday"
    assert humanise_when("2026-10-20", today=today) == "3 days ago"
    assert humanise_when("2020-01-15T00:00:00+00:00", today=today) == "15 Jan 2020"
    assert humanise_when("not a date") == "not a date"
    assert humanise_when(None) == ""


def test_build_library_conceals_other_pillars():
    view = build_library(
        [
            {"id": "arxiv:1", "title": "One", "pillar_id": "ml-systems-production", "authors": ["A"]},
            {"id": "arxiv:2", "title": "Two", "pillar_id": "ai-safety-alignment"},
        ],
        [{"id": "ml-systems-production", "name": "ML Systems"}],
        lesson_ids={"arxiv:1"},
        card_ids=set(),
        episode_ids={"arxiv:2"},
        selected_pillar="ai-safety-alignment",
    )
    by_id = {row.id: row for row in view.rows}
    assert by_id["arxiv:1"].concealed is True
    assert by_id["arxiv:1"].has_lesson is True
    assert by_id["arxiv:1"].pillar_name == "ML Systems"
    assert by_id["arxiv:2"].concealed is False
    assert by_id["arxiv:2"].pillar_name == "ai-safety-alignment"
    assert by_id["arxiv:2"].has_episode is True
    assert view.no_match is False
    assert view.visible_count == 1
    assert view.chips[0].name == "All"
    assert any(chip.current and chip.id == "ai-safety-alignment" for chip in view.chips)


def test_build_paper_page_keeps_notes_lesson_cards_and_episode():
    page = build_paper_page(
        {
            "paper": {
                "id": "arxiv:2402.08593",
                "title": "Graph",
                "pillar_id": "ml-systems-production",
                "url_pdf": "javascript:alert(1)",
                "processed": False,
                "added_at": "2020-01-15",
            },
            "notes": [{"problem": "Blind.", "confidence_score": 0.5, "findings": ["One"]}],
            "lessons": [{"tl_dr": "Short.", "difficulty": 1, "estimated_time": 8}],
            "quiz_cards": [{"question": "Q?", "answer": "A.", "difficulty": 2}],
        },
        [{"id": "script-1", "title": "Ep", "word_count": 12, "created_at": "2020-02-01"}],
        [{"id": "ml-systems-production", "name": "ML Systems"}],
        today=date(2026, 10, 23),
    )
    assert page.pdf_href is None
    assert page.pdf_unsafe == "javascript:alert(1)"
    assert page.tldr == "Short."
    assert page.summary.problem == "Blind."
    assert page.summary.confidence == "50%"
    assert page.lessons[0].difficulty_label == "Easy"
    assert page.cards[0].answer == "A."
    assert page.first_question == "Q?"
    assert page.study_href == "/quiz/?pillar=ml-systems-production"
    assert page.make_episode_href == "/podcast/?paper=arxiv%3A2402.08593"
    assert page.episodes[0].href == "/podcast/?script=script-1"
    assert page.episodes[0].words == "12 words"
    assert page.added == "15 Jan 2020"
    assert page.pillar_name == "ML Systems"
    assert page.processed is False
