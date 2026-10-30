"""The paper map page shell: routing, the pillar it opens on, and the links to it.

The map itself is drawn in the browser from GET /api/map (see tests/js/paper-map.test.js
for the helpers behind it), so this file only pins what the server decides.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from webui.map_format import build_map_page, map_href
from webui.routers import papers as papers_router
from webui.services.postgrest_client import CountUnavailableError, PostgrestClient

ROOT = Path(__file__).resolve().parents[2]


class _Pillar:
    def __init__(self, id, name):
        self.id = id
        self.name = name


PILLARS = [
    _Pillar("formal-linguistics-nlp", "Formal Linguistics"),
    _Pillar("model-interpretability", "Model Interpretability"),
    _Pillar("ml-systems-production", "ML Systems"),
]
COUNTS = {"formal-linguistics-nlp": 0, "model-interpretability": 4, "ml-systems-production": 7}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(papers_router, "get_pillars_or_empty", lambda limit=100: PILLARS)

    async def counts(self, pillar_ids):
        return {pid: COUNTS.get(pid, 0) for pid in pillar_ids}

    monkeypatch.setattr(PostgrestClient, "paper_counts_for_pillars", counts)

    async def details(self, paper_id):
        return None  # a paper page request would 404

    monkeypatch.setattr(PostgrestClient, "get_paper_details", details)
    app = FastAPI()
    app.include_router(papers_router.router)
    app.state.templates = Jinja2Templates(directory=str(ROOT / "webui" / "templates"))
    return TestClient(app, raise_server_exceptions=False)


def test_map_is_a_page_and_not_a_paper_called_map(client):
    for path in ("/papers/map", "/papers/map/"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert 'id="map-page"' in response.text, path
        assert "paper-map.js" in response.text
        assert "vendor/d3.min.js" in response.text


def test_route_order_keeps_the_catch_all_after_the_map():
    paths = [route.path for route in papers_router.router.routes]
    assert paths.index("/papers/map") < paths.index("/papers/{paper_id:path}")
    assert paths.index("/papers/details/{paper_id}") < paths.index("/papers/{paper_id:path}")


def test_a_real_paper_page_still_resolves_through_the_catch_all(client):
    response = client.get("/papers/arxiv:2402.08593")
    assert response.status_code == 404  # the stub knows no paper, which is the catch-all answering
    assert 'id="map-page"' not in response.text


def test_deep_link_opens_on_that_pillar(client):
    body = client.get("/papers/map?pillar=ml-systems-production").text
    assert 'data-pillar="ml-systems-production"' in body
    assert 'data-pillar-name="ML Systems"' in body
    assert body.count('aria-current="true"') == 1
    assert 'href="/papers/map?pillar=ml-systems-production"' in body


def test_without_a_pillar_it_opens_on_the_one_with_most_papers(client):
    body = client.get("/papers/map").text
    assert 'data-pillar="ml-systems-production"' in body


def test_an_unknown_pillar_is_kept_so_the_page_can_say_so(client):
    body = client.get("/papers/map?pillar=nope").text
    assert 'data-pillar="nope"' in body
    assert 'href="/papers/map?pillar=nope"' in body


def test_a_pillar_with_no_papers_is_marked_empty_but_stays_a_link(client):
    body = client.get("/papers/map?pillar=ml-systems-production").text
    chip = body.split('href="/papers/map?pillar=formal-linguistics-nlp"')[1].split("</a>")[0]
    assert "No papers in this pillar yet" in body
    assert "is-empty" in body.split('href="/papers/map?pillar=formal-linguistics-nlp"')[0].rsplit("<a", 1)[1]
    assert ">0<" in chip


def test_a_failed_count_does_not_disable_any_chip(client, monkeypatch):
    async def broken(self, pillar_ids):
        raise CountUnavailableError("down")

    monkeypatch.setattr(PostgrestClient, "paper_counts_for_pillars", broken)
    body = client.get("/papers/map?pillar=model-interpretability").text
    assert "is-empty" not in body
    assert "map-chip-count" not in body


def test_no_pillars_says_so_instead_of_rendering_an_empty_row(client, monkeypatch):
    monkeypatch.setattr(papers_router, "get_pillars_or_empty", lambda limit=100: [])
    body = client.get("/papers/map").text
    assert "No pillars to map" in body
    assert 'data-pillar=""' in body


def test_the_sidebar_has_a_map_link_under_library(client):
    body = client.get("/papers/map").text
    assert 'href="/papers/map"' in body
    nav = body.split('aria-label="Primary"')[1]
    # on the map page Map is current and Library is not
    map_link = nav.split('href="/papers/map"')[1].split("</a>")[0]
    library_link = nav.split('href="/papers/"')[1].split("</a>")[0]
    assert 'aria-current="page"' in map_link
    assert 'aria-current="page"' not in library_link


def test_chip_ids_are_encoded_in_links():
    assert map_href("a b/c") == "/papers/map?pillar=a%20b%2Fc"
    assert map_href(None) == "/papers/map"


def test_build_map_page_ties_go_to_the_first_pillar_listed():
    page = build_map_page(PILLARS, {p.id: 3 for p in PILLARS}, None)
    assert page.selected == "formal-linguistics-nlp"
    assert page.counts_known is True


def test_build_map_page_with_unknown_counts_picks_the_first_and_disables_none():
    page = build_map_page(PILLARS, None, None)
    assert page.selected == "formal-linguistics-nlp"
    assert page.counts_known is False
    assert all(chip.count is None and not chip.empty for chip in page.chips)


def test_pillar_page_and_library_link_to_the_map():
    detail = (ROOT / "webui" / "templates" / "pillar_detail.html").read_text()
    assert "/papers/map?pillar={{ pillar.id|urlencode }}" in detail
    assert "Open in map" in detail
    library = (ROOT / "webui" / "templates" / "papers.html").read_text()
    assert "/papers/map" in library


def test_vendored_d3_is_the_pinned_release_with_its_licence():
    vendor = ROOT / "webui" / "static" / "vendor"
    assert (vendor / "d3.min.js").read_text().startswith("// https://d3js.org v7.9.0")
    licence = (vendor / "d3-LICENSE.txt").read_text()
    assert "Copyright 2010-2023 Mike Bostock" in licence
    assert "f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539" in licence
