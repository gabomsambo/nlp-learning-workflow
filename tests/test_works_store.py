"""works_store over a PostgREST client whose HTTP layer is a Mock(spec=httpx.Client).

Checks what goes over the wire (filters, quoting, the RPC body) and the
bridge that lets callers holding bare ids keep storing citations.
"""

from unittest.mock import Mock

import httpx
import pytest

from nlp_pillars import db, works_store
from nlp_pillars.db import PostgRESTClient
from nlp_pillars.schemas import PaperCitation
from nlp_pillars.works import SOURCE, SOURCE_IDS_ONLY
from nlp_pillars.works_store import (
    MIGRATION,
    PostgrestWorks,
    WorksStoreError,
    WorksUnavailable,
    _quote,
    execute_works_sync,
    load_index,
    load_library,
    store_citations,
)

ATTENTION_WORK = "00000000-0000-4000-8000-0000000000a1"
EXPLANATION_WORK = "00000000-0000-4000-8000-0000000000a2"


def reply(status, body):
    return httpx.Response(status, json=body, request=httpx.Request("GET", "http://x/"))


class FakeHttp:
    """Routes PostgREST requests to canned rows and records them."""

    def __init__(self):
        self.routes = {}
        self.gets = []
        self.posts = []
        self.client = Mock(spec=httpx.Client)
        self.client.get.side_effect = self._get
        self.client.post.side_effect = self._post

    def _get(self, url, params=None):
        self.gets.append((url, dict(params or {})))
        table = url.rsplit("/", 1)[-1]
        rows = self.routes.get(table, [])
        if callable(rows):
            rows = rows(dict(params or {}))
        if isinstance(rows, httpx.Response):
            return rows
        return reply(200, rows)

    def _post(self, url, json=None):
        self.posts.append((url, json))
        answer = self.routes.get(url.rsplit("/", 1)[-1], {})
        return answer if isinstance(answer, httpx.Response) else reply(200, answer)


@pytest.fixture
def http():
    fake = FakeHttp()
    client = PostgRESTClient("http://postgrest.test", "token")
    client.client = fake.client
    previous = db._client
    db.set_client(client)
    yield fake
    db._client = previous


def citation(paper_id, other, direction="outgoing", influential=False):
    return PaperCitation(
        paper_id=paper_id,
        cited_paper_id=other,
        citation_direction=direction,
        is_influential=influential,
    )


# -- reading ----------------------------------------------------------------

def test_values_that_hold_commas_and_parentheses_are_quoted_for_an_in_filter():
    assert _quote("10.1002/(sici)1097-4571(199806)49:8<693::aid-asi4>") == (
        '"10.1002/(sici)1097-4571(199806)49:8<693::aid-asi4>"'
    )
    assert _quote('a"b,c') == '"a\\"b,c"'


def test_the_index_is_every_work_id_row(http):
    http.routes["work_ids"] = [
        {"kind": "s2", "value": "a" * 40, "work_id": ATTENTION_WORK},
        {"kind": "arxiv", "value": "1706.03762", "work_id": ATTENTION_WORK},
    ]
    index = load_index()
    assert index.work_for(("arxiv", "1706.03762")) == ATTENTION_WORK
    assert index.s2_id(ATTENTION_WORK) == "a" * 40
    assert http.gets[0][1]["order"] == "kind,value"


def test_a_large_table_is_read_in_pages(http):
    rows = [{"kind": "arxiv", "value": f"2001.{n:05d}", "work_id": ATTENTION_WORK} for n in range(1500)]
    http.routes["work_ids"] = lambda params: rows[int(params["offset"]):int(params["offset"]) + int(params["limit"])]
    assert len(load_index().by_key) == 1500
    assert [g[1]["offset"] for g in http.gets] == ["0", "1000"]


def test_the_library_is_papers_with_their_work(http):
    http.routes["papers"] = [
        {"id": "arxiv:1706.03762", "work_id": ATTENTION_WORK, "title": "Attention", "citation_count": 5},
        {"id": "file:abc", "work_id": None, "title": "Notes", "citation_count": 0},
    ]
    library = load_library()
    assert [(p.id, p.work_id) for p in library] == [("arxiv:1706.03762", ATTENTION_WORK), ("file:abc", None)]


@pytest.mark.parametrize(
    "error",
    [
        {"code": "PGRST205", "message": "Could not find the table 'public.work_ids' in the schema cache"},
        {"code": "42703", "message": "column papers.work_id does not exist"},
        {"code": "PGRST202", "message": "Could not find the function public.apply_work_batch"},
    ],
)
def test_before_the_migration_every_read_and_write_names_it(http, error):
    http.routes["work_ids"] = reply(404, error)
    http.routes["papers"] = reply(400, error)
    http.routes["apply_work_batch"] = reply(404, error)
    with pytest.raises(WorksUnavailable, match=MIGRATION):
        load_index()
    with pytest.raises(WorksUnavailable, match=MIGRATION):
        load_library()
    with pytest.raises(WorksUnavailable, match=MIGRATION):
        PostgrestWorks().apply(works_store.Batch(works=[{"id": ATTENTION_WORK}]))


def test_another_failure_is_not_blamed_on_the_migration(http):
    http.routes["work_ids"] = reply(500, {"message": "connection reset"})
    with pytest.raises(WorksStoreError) as raised:
        load_index()
    assert not isinstance(raised.value, WorksUnavailable) and MIGRATION not in str(raised.value)


# -- writing ----------------------------------------------------------------

def test_apply_sends_the_four_arrays_in_one_rpc(http):
    http.routes["apply_work_batch"] = {"works": 1, "ids_added": 1, "edges": 0, "papers": 0}
    batch = works_store.Batch(
        works=[{"id": ATTENTION_WORK}], ids=[{"kind": "arxiv", "value": "1", "work_id": ATTENTION_WORK}]
    )
    assert PostgrestWorks().apply(batch)["works"] == 1
    url, body = http.posts[0]
    assert url.endswith("/rpc/apply_work_batch")
    assert set(body) == {"p_works", "p_ids", "p_edges", "p_papers"}
    assert body["p_edges"] == [] and body["p_ids"][0]["value"] == "1"


def test_stored_citers_are_the_typed_edges_into_one_work(http):
    http.routes["work_citations"] = [{"citing": EXPLANATION_WORK}]
    assert PostgrestWorks().citer_work_ids(ATTENTION_WORK) == {EXPLANATION_WORK}
    params = http.gets[0][1]
    assert params["cited"] == f"eq.{ATTENTION_WORK}" and params["source"] == f"eq.{SOURCE}"


# -- the bridge for callers that hold bare ids --------------------------------

def bridge(http, *, work_ids=(), papers=None):
    """Make the lookups answer from a small fake database."""
    http.routes["work_ids"] = lambda params: [
        row for row in work_ids if f"eq.{row['kind']}" == params.get("kind")
    ]
    http.routes["papers"] = papers if papers is not None else []
    http.routes["apply_work_batch"] = {"edges": 1}


def only_batch(http):
    return http.posts[-1][1]


def test_a_bare_id_and_a_prefixed_id_land_on_the_same_work(http):
    bridge(
        http,
        work_ids=[{"kind": "arxiv", "value": "1706.03762", "work_id": ATTENTION_WORK}],
        papers=[{"id": "arxiv:1902.10186", "work_id": EXPLANATION_WORK}],
    )
    # S2 names the cited paper by its bare arXiv id; the library says arxiv:1706.03762
    for spelling in ("1706.03762", "arxiv:1706.03762", "1706.03762v5"):
        http.posts.clear()
        assert store_citations([citation("arxiv:1902.10186", spelling)]) == 1
        batch = only_batch(http)
        assert [(e["citing"], e["cited"]) for e in batch["p_edges"]] == [(EXPLANATION_WORK, ATTENTION_WORK)]
        assert batch["p_works"] == [] and batch["p_ids"] == []  # nothing new was invented


def test_an_unseen_id_becomes_a_stub_work_with_its_key(http):
    bridge(http, papers=[{"id": "arxiv:1902.10186", "work_id": EXPLANATION_WORK}])
    store_citations([citation("arxiv:1902.10186", "1508.07909")], new_id=lambda: "W-new")
    batch = only_batch(http)
    assert batch["p_ids"] == [{"kind": "arxiv", "value": "1508.07909", "work_id": "W-new"}]
    stub = batch["p_works"][0]
    assert stub["id"] == "W-new" and stub["described"] is False and "title" not in stub


def test_incoming_citations_point_the_other_way_and_are_ids_only(http):
    bridge(
        http,
        work_ids=[{"kind": "arxiv", "value": "1706.03762", "work_id": ATTENTION_WORK}],
        papers=[{"id": "arxiv:1902.10186", "work_id": EXPLANATION_WORK}],
    )
    store_citations([citation("arxiv:1902.10186", "1706.03762", "incoming", influential=True)])
    (edge,) = only_batch(http)["p_edges"]
    assert (edge["citing"], edge["cited"]) == (ATTENTION_WORK, EXPLANATION_WORK)
    assert edge["detail"] is False and edge["source"] == SOURCE_IDS_ONLY
    assert edge["is_influential"] is True


def test_a_library_paper_without_a_work_gets_one(http):
    bridge(http, papers=[{"id": "arxiv:1902.10186", "work_id": None}])
    store_citations([citation("arxiv:1902.10186", "1706.03762")], new_id=iter(["W1", "W2"]).__next__)
    batch = only_batch(http)
    assert batch["p_papers"] == [{"id": "arxiv:1902.10186", "work_id": "W1", "citation_count": None}]
    assert {(i["kind"], i["value"]) for i in batch["p_ids"]} == {("arxiv", "1902.10186"), ("arxiv", "1706.03762")}


def test_rows_that_name_no_work_or_no_library_paper_are_skipped(http):
    bridge(http, papers=[{"id": "arxiv:1902.10186", "work_id": EXPLANATION_WORK}])
    assert store_citations([citation("file:abc", "1706.03762")]) == 0
    assert store_citations([citation("arxiv:1902.10186", "searxng_078015")]) == 0
    assert http.posts == []
    # a paper that is not in the library has no citations of its own to store
    assert store_citations([citation("arxiv:2001.00001", "1706.03762")]) == 0
    assert http.posts == []


def test_before_the_migration_the_bridge_says_so(http):
    http.routes["work_ids"] = reply(404, {"code": "PGRST205", "message": "Could not find the table 'public.work_ids'"})
    with pytest.raises(WorksUnavailable):
        store_citations([citation("arxiv:1902.10186", "1706.03762")])


# -- db.add_paper_citations goes through it ----------------------------------

def test_add_paper_citations_stores_works_edges(monkeypatch):
    seen = {}
    monkeypatch.setattr(works_store, "store_citations", lambda rows: seen.setdefault("n", len(rows)) or 3)
    assert db.add_paper_citations([citation("arxiv:1902.10186", "1706.03762")]) == 1
    assert seen["n"] == 1


def test_add_paper_citations_falls_back_to_the_old_table_until_the_migration(monkeypatch):
    def unavailable(rows):
        raise WorksUnavailable("not applied")

    legacy = Mock(return_value=2)
    monkeypatch.setattr(works_store, "store_citations", unavailable)
    monkeypatch.setattr(db, "_add_paper_citations_legacy", legacy)
    rows = [citation("arxiv:1902.10186", "1706.03762")]
    assert db.add_paper_citations(rows) == 2
    legacy.assert_called_once_with(rows)


def test_a_store_failure_after_the_migration_does_not_raise_into_the_caller(monkeypatch):
    def broken(rows):
        raise RuntimeError("boom")

    legacy = Mock()
    monkeypatch.setattr(works_store, "store_citations", broken)
    monkeypatch.setattr(db, "_add_paper_citations_legacy", legacy)
    assert db.add_paper_citations([citation("arxiv:1902.10186", "1706.03762")]) == 0
    legacy.assert_not_called()


def test_nothing_to_add_is_zero():
    assert db.add_paper_citations([]) == 0


# -- the CLI entry --------------------------------------------------------------

def test_dry_run_entry_reads_but_builds_no_tool_and_writes_nothing(http, monkeypatch):
    http.routes["papers"] = [
        {"id": "arxiv:1706.03762", "work_id": None, "title": "Attention", "citation_count": 0},
        {"id": "file:abc", "work_id": None, "title": "Notes", "citation_count": 0},
    ]
    http.routes["work_ids"] = []
    monkeypatch.setattr(
        "nlp_pillars.tools.semantic_scholar_tool.SemanticScholarTool",
        Mock(side_effect=AssertionError("a dry run must not build the S2 client")),
    )
    report = execute_works_sync(dry_run=True)
    assert report.to_link == 1 and report.unresolvable == ["file:abc"]
    assert http.posts == []
