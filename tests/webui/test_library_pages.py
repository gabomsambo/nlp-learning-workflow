"""Library list and the paper page.

The details JSON stays the modal's payload. The new page is HTML, and a paper
id that contains ':' round-trips through the path.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from webui.routers import papers as papers_router

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


def _papers():
    return [
        {
            "id": PAPER_ID,
            "title": "Graph Feature Preprocessor",
            "authors": ["Ada Lovelace", "Grace Hopper", "Edsger Dijkstra", "Alan Turing"],
            "year": 2024,
            "citation_count": 12,
            "pillar_id": "ml-systems-production",
            "added_at": "2020-01-15T12:00:00+00:00",
            "url_pdf": "https://arxiv.org/pdf/2402.08593",
            "abstract": "A streaming library.",
            "venue": "arXiv cs.LG",
            "processed": True,
        },
        {
            "id": OTHER_ID,
            "title": "Attention Is All You Need",
            "authors": ["Vaswani"],
            "year": 2017,
            "citation_count": 1,
            "pillar_id": "neural-architectures-language",
            "added_at": "2020-02-02T00:00:00+00:00",
            "url_pdf": "javascript:alert(1)",
            "abstract": "",
            "venue": "",
            "processed": False,
        },
    ]


def _details(paper_id=PAPER_ID):
    paper = next(item for item in _papers() if item["id"] == paper_id)
    return {
        "paper": paper,
        "notes": [
            {
                "problem": "Tabular models are blind to structure.",
                "method": "Mine subgraphs on the CPU.",
                "findings": ["Beats a GNN on F1."],
                "limitations": ["Needs a window."],
                "future_work": ["Try a longer window."],
                "key_terms": ["fan-in"],
                "related_papers": ["arxiv:1706.03762"],
                "confidence_score": 0.84,
            }
        ],
        "lessons": [
            {
                "tl_dr": "Streams the graph and feeds counts to a booster.",
                "takeaways": ["Structure can be a column."],
                "practice_ideas": ["Add fan-in features."],
                "connections": ["GNNs want the whole graph."],
                "difficulty": 2,
                "estimated_time": 15,
            }
        ],
        "quiz_cards": [
            {
                "question": "Why can't a GNN score this live?",
                "answer": "It needs the whole graph in memory.",
                "difficulty": 3,
            }
        ],
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(papers_router, "get_pillars_or_empty", lambda limit=100: PILLARS)

    async def papers(self, pillar=None, limit=50):
        assert pillar is None
        return _papers()

    async def ids(self, table):
        return {
            "lessons": {PAPER_ID},
            "quiz_cards": {PAPER_ID},
            "podcast_scripts": {PAPER_ID},
        }.get(table, set())

    async def details(self, paper_id):
        if paper_id in (PAPER_ID, OTHER_ID):
            return _details(paper_id)
        return None

    async def scripts(self, paper_id):
        if paper_id != PAPER_ID:
            return []
        return [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "title": "Deep Dive episode",
                "word_count": 1948,
                "created_at": "2020-03-01T00:00:00+00:00",
            }
        ]

    monkeypatch.setattr(papers_router.PostgrestClient, "list_papers_checked", papers)
    monkeypatch.setattr(papers_router.PostgrestClient, "paper_ids_with_rows", ids)
    monkeypatch.setattr(papers_router.PostgrestClient, "get_paper_details", details)
    monkeypatch.setattr(papers_router.PostgrestClient, "scripts_for_paper", scripts)

    app = FastAPI()
    app.include_router(papers_router.router)
    app.state.templates = Jinja2Templates(directory=str(ROOT / "webui" / "templates"))
    return TestClient(app)


def test_library_rows_are_title_first_with_names_and_badges(client):
    body = client.get("/papers/").text
    assert "Graph Feature Preprocessor" in body
    assert "ML Systems &amp; Production" in body
    assert ">Lesson<" in body
    assert ">Cards<" in body
    assert ">Episode<" in body
    assert "Ada Lovelace, Grace Hopper, Edsger Dijkstra +1" in body
    assert "12 citations" in body
    assert "15 Jan 2020" in body
    assert "2020-01-15" not in body
    assert 'href="/papers/arxiv%3A2402.08593"' in body
    assert 'btn-sm paper-peek' in body
    assert "triggerSelector: '.paper-peek'" in body
    assert "get('paper')" in body
    assert 'id="library-search"' in body
    assert "Has lesson" in body
    assert 'href="/discovery/"' in body
    assert 'href="/lessons/"' in body
    assert 'href="/pipeline/"' in body
    assert "Press" in body and ">j<" in body


def test_pillar_query_hides_other_rows_and_keeps_filters(client):
    body = client.get("/papers/?pillar=neural-architectures-language").text
    assert "Attention Is All You Need" in body
    assert "Neural Architectures" in body
    hidden = body.split('data-paper-id="arxiv:2402.08593"')[1].split(">", 1)[0]
    assert "hidden" in hidden
    assert 'data-pillar="neural-architectures-language"' in body
    assert 'aria-current="true"' in body
    assert 'href="/lessons/?pillar=neural-architectures-language"' in body
    assert 'id="library-search"' in body


def test_library_load_error_is_not_an_empty_shelf(client, monkeypatch):
    async def boom(self, pillar=None, limit=50):
        raise RuntimeError("down")

    monkeypatch.setattr(papers_router.PostgrestClient, "list_papers_checked", boom)
    body = client.get("/papers/").text
    assert "could not be loaded" in body
    assert "No papers yet" not in body
    assert 'id="library-search"' in body


def test_empty_library_keeps_the_filters(client, monkeypatch):
    async def empty(self, pillar=None, limit=50):
        return []

    monkeypatch.setattr(papers_router.PostgrestClient, "list_papers_checked", empty)
    body = client.get("/papers/").text
    assert "No papers yet" in body
    assert 'id="library-search"' in body
    assert "ML Systems &amp; Production" in body
    assert 'href="/discovery/"' in body


def test_details_json_is_unchanged_for_a_colon_id(client):
    response = client.get(f"/papers/details/{PAPER_ID}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert set(body) == {"paper", "notes", "lessons", "quiz_cards"}
    assert body["paper"]["id"] == PAPER_ID
    assert body["notes"][0]["problem"].startswith("Tabular")


def test_paper_page_renders_sections_and_actions(client):
    response = client.get("/papers/" + PAPER_ID.replace(":", "%3A"))
    assert response.status_code == 200
    body = response.text
    assert "<h1 class=\"paper-title\">Graph Feature Preprocessor</h1>" in body
    assert "arxiv:2402.08593" in body
    assert "ML Systems &amp; Production" in body
    assert 'href="https://arxiv.org/pdf/2402.08593"' in body
    assert "Open PDF" in body
    assert "In one paragraph" in body
    assert "Streams the graph" in body
    assert "Tabular models are blind" in body
    assert "Mine subgraphs" in body
    assert "Beats a GNN" in body
    assert "fan-in" in body
    assert "Needs a window." in body
    assert "Try a longer window." in body
    assert "Add fan-in features." in body
    assert "Why can&#39;t a GNN score this live?" in body or "Why can't a GNN score this live?" in body
    assert "It needs the whole graph in memory." in body
    assert 'data-quiz-answer' in body
    assert "nlp:quizAnswersVisible" in body
    assert 'id="quiz-answers-toggle"' in body
    assert "Hide answers" in body
    assert 'href="/quiz/?pillar=ml-systems-production"' in body
    assert "Study these cards" in body
    assert 'href="/podcast/?paper=arxiv%3A2402.08593"' in body
    assert "Make episode" in body
    assert 'href="/podcast/?script=11111111-1111-1111-1111-111111111111"' in body
    assert "Deep Dive episode" in body
    assert "1,948 words" in body
    assert 'id="refresh-metadata"' in body
    assert "15 min" in body
    assert "Medium" in body
    assert "On this browser" in body
    assert "not saved to your library" in body
    assert 'href="#lesson"' in body
    assert 'href="#episode"' in body


def test_raw_colon_path_resolves_the_same_page(client):
    response = client.get("/papers/arxiv:2402.08593")
    assert response.status_code == 200
    assert "Graph Feature Preprocessor" in response.text


def test_unsafe_pdf_is_text_not_a_link(client):
    body = client.get("/papers/" + OTHER_ID.replace(":", "%3A")).text
    assert "javascript:alert(1)" in body
    assert 'href="javascript:' not in body
    assert "Not processed" in body
    assert "No episode for this paper yet." in body


def test_missing_paper_is_a_page_not_json(client):
    response = client.get("/papers/does-not-exist")
    assert response.status_code == 404
    assert "text/html" in response.headers["content-type"]
    assert "not in the library" in response.text


def test_episode_read_failure_is_not_no_episode(client, monkeypatch):
    async def boom(self, paper_id):
        raise RuntimeError("down")

    monkeypatch.setattr(papers_router.PostgrestClient, "scripts_for_paper", boom)
    body = client.get("/papers/arxiv:2402.08593").text
    assert "could not be loaded" in body
    assert "No episode for this paper yet." not in body


def test_podcast_page_reads_paper_query():
    text = (ROOT / "webui" / "templates" / "podcast.html").read_text()
    assert ".get('paper')" in text
    assert "paperSelect.value = urlPaperId" in text
