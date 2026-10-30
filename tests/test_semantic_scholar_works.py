"""Semantic Scholar client methods for the works layer.

Answers are recorded S2 responses (tests/fixtures/semantic_scholar), trimmed
only in abstract length. No network, no key, no real waiting.
"""

import copy
import json
from pathlib import Path

import httpx
import pytest

from nlp_pillars.tools import semantic_scholar_tool as s2
from nlp_pillars.tools.semantic_scholar_tool import (
    BATCH_LIMIT,
    CONTEXT_CHARS,
    EDGE_FIELDS,
    EDGE_PAGE_LIMIT,
    RequestPacer,
    SemanticScholarError,
    SemanticScholarTool,
    SemanticScholarUnavailableError,
    WORK_BATCH_FIELDS,
    WORK_FIELDS,
    work_record_from_s2,
)

FIXTURES = Path(__file__).parent / "fixtures" / "semantic_scholar"
ATTENTION_S2 = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"


def load(name):
    return json.loads((FIXTURES / name).read_text())


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def _clean_shared_state():
    s2.reset_shared_state()
    yield
    s2.reset_shared_state()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def tool(clock):
    pacer = RequestPacer(clock=clock, sleep=clock.sleep)
    instance = SemanticScholarTool(api_key=None, pacer=pacer)
    instance._api_key = None
    return instance


def response(code, body=None):
    request = httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/x")
    return httpx.Response(code, json=body if body is not None else {}, request=request)


def script(tool, responses, seen=None):
    queue = list(responses)

    def request(method, url, params=None, json=None, headers=None):
        if seen is not None:
            seen.append({"method": method, "url": url, "params": params, "json": json})
        return queue.pop(0)

    tool.client.request = request
    return queue


def citer_page(count, start=0, next_offset=None):
    """A recorded citations page cut to ``count`` distinct citers."""
    base = load("citations_1508.07909.json")["data"]
    rows = []
    for number in range(start, start + count):
        row = copy.deepcopy(base[number % len(base)])
        row["citingPaper"]["paperId"] = f"{number:040x}"
        rows.append(row)
    body = {"offset": start, "data": rows}
    if next_offset is not None:
        body["next"] = next_offset
    return body


# -- batch details ----------------------------------------------------------

def test_batch_details_are_aligned_and_keep_what_works_stores(tool):
    seen = []
    script(tool, [response(200, load("paper_batch.json"))], seen)
    ids = ["ARXIV:1706.03762", "ARXIV:1902.10186", "ARXIV:1508.07909", "ARXIV:9999.99999"]
    records = tool.get_work_records(ids)

    assert seen[0]["method"] == "POST" and seen[0]["url"].endswith("/paper/batch")
    assert seen[0]["json"] == {"ids": ids}
    assert seen[0]["params"] == {"fields": WORK_BATCH_FIELDS}
    assert "references.paperId" in WORK_BATCH_FIELDS and "tldr" in WORK_BATCH_FIELDS

    attention, explanation, bpe, unknown = records
    assert unknown is None
    assert attention.s2_paper_id == ATTENTION_S2
    assert attention.corpus_id == 13756489
    assert attention.external_ids["ArXiv"] == "1706.03762"
    assert attention.title == "Attention is All you Need"
    assert attention.publication_date == "2017-06-12"
    assert attention.venue == "Neural Information Processing Systems"
    assert attention.citation_count == 200329
    assert attention.reference_count == 41
    assert attention.tldr.startswith("A new simple network architecture")
    assert len(attention.reference_ids) == 6
    assert attention.author_count == 8 and len(attention.authors) == 8
    assert explanation.external_ids["DOI"] == "10.18653/v1/N19-1357"
    assert bpe.title.startswith("Neural Machine Translation of Rare Words")


def test_batch_details_send_ids_in_chunks_of_500(tool):
    ids = [f"ARXIV:2001.{n:05d}" for n in range(BATCH_LIMIT + 1)]
    seen = []
    script(tool, [response(200, [None] * BATCH_LIMIT), response(200, [None])], seen)
    assert tool.get_work_records(ids) == [None] * len(ids)
    assert [len(call["json"]["ids"]) for call in seen] == [BATCH_LIMIT, 1]


def test_a_batch_answer_of_the_wrong_length_is_an_error_not_misaligned_data(tool):
    script(tool, [response(200, [None])])
    with pytest.raises(SemanticScholarError, match="did not match"):
        tool.get_work_records(["ARXIV:1", "ARXIV:2"])


def test_no_paper_id_means_no_record():
    assert work_record_from_s2({"title": "ghost", "paperId": None}) is None
    assert work_record_from_s2({}) is None


def test_authors_are_capped_but_counted():
    record = work_record_from_s2(
        {"paperId": "p", "authors": [{"name": f"A{n}"} for n in range(319)] + [{"name": None}]}
    )
    assert len(record.authors) == s2.AUTHORS_KEPT and record.author_count == 319


# -- reference edges --------------------------------------------------------

def test_reference_edges_carry_intents_contexts_and_influence(tool):
    seen = []
    script(tool, [response(200, load("references_1902.10186.json"))], seen)
    page = tool.get_reference_edges("ARXIV:1902.10186")

    assert len(page.edges) == 33 and page.skipped == 0 and page.next_offset is None
    fields = seen[0]["params"]["fields"].split(",")
    for name in EDGE_FIELDS.split(","):
        assert name in fields
    assert "citedPaper.paperId" in fields and "citedPaper.abstract" in fields
    assert not any(name.endswith(".tldr") for name in fields)  # S2 answers 400 for it
    assert seen[0]["params"]["limit"] == EDGE_PAGE_LIMIT

    attention = next(e for e in page.edges if e.other.s2_paper_id == ATTENTION_S2)
    assert attention.intents == ["background", "methodology"]
    assert attention.is_influential is False
    assert attention.contexts and len(attention.contexts[0]) <= CONTEXT_CHARS
    assert attention.other.external_ids["ArXiv"] == "1706.03762"


def test_reference_edges_follow_next_until_the_last_page(tool):
    one = load("references_1902.10186.json")
    first = copy.deepcopy(one)
    first["data"] = first["data"][:2]
    first["next"] = 2
    last = copy.deepcopy(one)
    last["data"] = last["data"][2:5]
    last.pop("next", None)
    seen = []
    script(tool, [response(200, first), response(200, last)], seen)
    page = tool.get_reference_edges("ARXIV:1902.10186")
    assert [call["params"]["offset"] for call in seen] == [0, 2]
    assert len(page.edges) == 5


def test_an_entry_without_a_paper_id_is_skipped_and_counted(tool):
    body = load("references_1902.10186.json")
    body["data"][0]["citedPaper"]["paperId"] = None
    script(tool, [response(200, body)])
    page = tool.get_reference_edges("ARXIV:1902.10186")
    assert page.skipped == 1 and len(page.edges) == 32


def test_an_unknown_paper_has_no_references_but_a_server_error_is_an_error(tool):
    script(tool, [response(404)])
    assert tool.get_reference_edges("ARXIV:9999.99999").edges == []
    script(tool, [response(403, {"message": "Forbidden"})])
    with pytest.raises(SemanticScholarError, match="403"):
        tool.get_reference_edges("ARXIV:1706.03762")


def test_contexts_are_kept_to_three_short_sentences(tool):
    body = load("references_1902.10186.json")
    body["data"][0]["contexts"] = ["x" * 900, "b", "c", "d"]
    script(tool, [response(200, body)])
    edge = tool.get_reference_edges("ARXIV:1902.10186").edges[0]
    assert len(edge.contexts) == s2.CONTEXTS_KEPT
    assert len(edge.contexts[0]) == CONTEXT_CHARS


# -- citers, newest first, known-id stop ------------------------------------

def test_recorded_citers_carry_their_edge_fields(tool):
    script(tool, [response(200, load("citations_1508.07909.json"))])
    page = tool.get_citer_edges_page("ARXIV:1508.07909", limit=6)
    assert len(page.edges) == 6 and page.next_offset == 6  # S2 says there are more
    assert [e.is_influential for e in page.edges].count(True) == 1
    assert page.edges[-1].is_influential and page.edges[-1].intents == []


def test_citer_walk_stops_after_the_page_with_a_known_citer(tool):
    seen = []
    known = {f"{150:040x}"}
    script(
        tool,
        [response(200, citer_page(100, 0, next_offset=100)),
         response(200, citer_page(100, 100, next_offset=200)),
         response(200, citer_page(100, 200, next_offset=300))],
        seen,
    )
    result = tool.get_citer_edges("ARXIV:1508.07909", known_ids=known)
    assert len(seen) == 2  # the third page is never asked for
    assert result.stopped_at_known is True and result.truncated is False
    assert f"{150:040x}" not in {e.other.s2_paper_id for e in result.edges}
    assert len(result.edges) == 199


def test_with_nothing_known_the_walk_ends_at_the_last_page(tool):
    script(tool, [response(200, citer_page(100, 0, next_offset=100)), response(200, citer_page(40, 100))])
    result = tool.get_citer_edges("ARXIV:1508.07909")
    assert len(result.edges) == 140
    assert result.stopped_at_known is False and result.truncated is False


def test_citer_walk_stops_at_max_items_and_says_so(tool):
    seen = []
    script(tool, [response(200, citer_page(100, 0, next_offset=100)),
                  response(200, citer_page(50, 100, next_offset=150))], seen)
    result = tool.get_citer_edges("ARXIV:1508.07909", max_items=150)
    assert len(result.edges) == 150
    assert [call["params"]["limit"] for call in seen] == [100, 50]
    assert result.truncated is True


def test_a_rate_limited_citer_walk_fails_loudly_rather_than_returning_a_short_list(tool):
    script(tool, [response(429)] * s2.MAX_ATTEMPTS)
    with pytest.raises(SemanticScholarUnavailableError):
        tool.get_citer_edges("ARXIV:1508.07909")


def test_every_new_call_goes_through_the_shared_pacer(tool, clock):
    script(tool, [response(200, load("paper_batch.json")),
                  response(200, load("references_1902.10186.json")),
                  response(200, citer_page(3))])
    tool.get_work_records(["ARXIV:1706.03762"] * 4)
    tool.get_reference_edges("ARXIV:1902.10186")
    tool.get_citer_edges("ARXIV:1508.07909")
    # three calls, so the second and third each waited for the keyless interval
    assert clock.slept == [pytest.approx(s2.KEYLESS_INTERVAL_SECONDS)] * 2


def test_work_fields_are_what_the_migration_stores():
    for name in (
        "paperId", "corpusId", "externalIds", "title", "year", "publicationDate",
        "venue", "authors", "abstract", "citationCount", "influentialCitationCount",
        "referenceCount", "openAccessPdf",
    ):
        assert name in WORK_FIELDS.split(",")
