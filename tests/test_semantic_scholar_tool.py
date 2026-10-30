"""Semantic Scholar client: key fallback, pacing, 429 backoff, queue bound.

No network, no .env key, and no real waiting: the pacer takes a fake clock.
"""

import logging
import threading

import httpx
import pytest

from nlp_pillars.tools import semantic_scholar_tool as s2
from nlp_pillars.tools.semantic_scholar_tool import (
    BULK,
    INTERACTIVE,
    KEY_REJECTED_LOG,
    KEYED_INTERVAL_SECONDS,
    KEYLESS_INTERVAL_SECONDS,
    MAX_ATTEMPTS,
    RequestPacer,
    SemanticScholarError,
    SemanticScholarTool,
    SemanticScholarUnavailableError,
)


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
def pacer(clock):
    return RequestPacer(clock=clock, sleep=clock.sleep)


def _tool(pacer, api_key="rejected-key-value", **kw):
    return SemanticScholarTool(api_key=api_key, pacer=pacer, **kw)


def _response(code, body=None, headers=None):
    request = httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/x")
    return httpx.Response(code, json=body if body is not None else {}, headers=headers, request=request)


def _links(paper_id="abc", wrapper="citingPaper"):
    return {
        "data": [
            {
                wrapper: {
                    "paperId": paper_id,
                    "externalIds": {"ArXiv": "1706.03762"},
                    "title": "Attention",
                    "authors": [{"name": "Vaswani"}],
                    "year": 2017,
                },
                "isInfluential": False,
            }
        ]
    }


def _script(tool, responses, seen=None):
    """Make the tool's HTTP client answer from a list, recording each call."""
    queue = list(responses)

    def request(method, url, params=None, json=None, headers=None):
        if seen is not None:
            seen.append({"method": method, "url": url, "params": params, "json": json,
                         "key": (headers or {}).get("x-api-key")})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    tool.client.request = request
    return queue


# -- key fallback -----------------------------------------------------------

def test_403_with_key_falls_back_to_keyless_and_logs_once(pacer, caplog):
    tool = _tool(pacer)
    seen = []
    _script(tool, [_response(403), _response(200, _links()), _response(200, _links())], seen)
    with caplog.at_level(logging.ERROR):
        first = tool.get_citations("1706.03762", limit=1)
        second = tool.get_citations("1706.03762", limit=1)
    assert [c["key"] for c in seen] == ["rejected-key-value", None, None]
    assert first[0][0].id == "1706.03762" and second[0][0].id == "1706.03762"
    assert tool.key_rejected is True
    assert "x-api-key" not in tool.headers
    assert "rejected-key-value" not in caplog.text
    assert caplog.text.count(KEY_REJECTED_LOG) == 1


def test_rejected_key_is_remembered_by_other_instances(pacer):
    _script(first := _tool(pacer), [_response(401), _response(200, _links())])
    first.get_citations("1706.03762", limit=1)
    seen = []
    other = _tool(pacer)
    _script(other, [_response(200, _links())], seen)
    other.get_citations("1706.03762", limit=1)
    assert other.key_rejected is True
    assert seen[0]["key"] is None


def test_a_different_key_is_still_tried(pacer):
    _script(first := _tool(pacer, "bad"), [_response(403), _response(200, _links())])
    first.get_citations("1706.03762", limit=1)
    seen = []
    fresh = _tool(pacer, "good")
    _script(fresh, [_response(200, _links())], seen)
    fresh.get_citations("1706.03762", limit=1)
    assert seen[0]["key"] == "good"


def test_403_without_a_key_is_a_failure_not_zero_citations(pacer):
    tool = _tool(pacer, api_key=None)
    tool._api_key = None
    _script(tool, [_response(403, {"message": "Forbidden"})])
    with pytest.raises(SemanticScholarError, match="403"):
        tool.get_citations("1706.03762", limit=1)


def test_404_is_still_zero_citations_and_none_paper(pacer):
    tool = _tool(pacer, "present-key")
    _script(tool, [_response(404), _response(404)])
    assert tool.get_citations("1706.03762", limit=1) == []
    assert tool.get_paper("1706.03762") is None


# -- pacing -----------------------------------------------------------------

def test_pacer_spaces_calls_by_the_interval(pacer, clock):
    start = clock.now
    stamps = []
    for _ in range(3):
        pacer.acquire(INTERACTIVE, 1.0)
        stamps.append(clock.now - start)
    assert stamps == [0.0, 1.0, 2.0]


def test_keyless_and_keyed_intervals_differ(pacer, clock):
    keyless = _tool(pacer, api_key=None)
    keyless._api_key = None
    _script(keyless, [_response(200, _links())] * 2)
    keyless.get_citations("1706.03762", limit=1)
    keyless.get_citations("1706.03762", limit=1)
    assert clock.slept == [pytest.approx(KEYLESS_INTERVAL_SECONDS)]

    clock.slept.clear()
    s2.reset_shared_state()
    clock2 = FakeClock()
    p2 = RequestPacer(clock=clock2, sleep=clock2.sleep)
    keyed = _tool(p2, "ok-key")
    _script(keyed, [_response(200, _links())] * 2)
    keyed.get_citations("1706.03762", limit=1)
    keyed.get_citations("1706.03762", limit=1)
    assert clock2.slept == [pytest.approx(KEYED_INTERVAL_SECONDS)]
    assert KEYED_INTERVAL_SECONDS < KEYLESS_INTERVAL_SECONDS


# -- 429 --------------------------------------------------------------------

def test_429_honours_retry_after_then_succeeds(pacer, clock):
    tool = _tool(pacer)
    seen = []
    _script(tool, [_response(429, headers={"Retry-After": "7"}), _response(200, _links())], seen)
    rows = tool.get_citations("1706.03762", limit=1)
    assert len(rows) == 1 and len(seen) == 2
    assert sum(clock.slept) >= 7


def test_429_without_header_backs_off_exponentially(pacer, clock):
    tool = _tool(pacer)
    _script(tool, [_response(429), _response(429), _response(200, _links())])
    tool.get_citations("1706.03762", limit=1)
    # 2s then 4s of back-off, each on top of the pacing interval already owed
    assert sum(clock.slept) >= 2 + 4


def test_429_past_the_budget_is_unavailable_not_empty(pacer):
    tool = _tool(pacer)
    seen = []
    _script(tool, [_response(429)] * 10, seen)
    with pytest.raises(SemanticScholarUnavailableError, match="temporarily unavailable"):
        tool.get_references("1706.03762", limit=1)
    assert len(seen) == MAX_ATTEMPTS


def test_a_very_long_retry_after_is_not_waited_out(pacer, clock):
    tool = _tool(pacer)
    _script(tool, [_response(429, headers={"Retry-After": "600"})])
    with pytest.raises(SemanticScholarUnavailableError):
        tool.search("transformers")
    assert sum(clock.slept) < 600


def test_429_penalty_is_shared_with_other_callers(pacer, clock):
    tool = _tool(pacer)
    _script(tool, [_response(429, headers={"Retry-After": "10"}), _response(200, _links())])
    tool.get_citations("1706.03762", limit=1)
    # A second caller arriving right after waits out the interval, not zero.
    other = _tool(pacer)
    _script(other, [_response(200, _links())])
    clock.slept.clear()
    other.get_citations("1706.03762", limit=1)
    assert clock.slept and clock.slept[0] > 0


def test_5xx_retries_then_fails_with_the_status(pacer):
    tool = _tool(pacer)
    _script(tool, [_response(503)] * 10)
    with pytest.raises(SemanticScholarError, match="503"):
        tool.get_paper("1706.03762")


def test_transport_errors_retry_then_fail(pacer):
    tool = _tool(pacer)
    _script(tool, [httpx.ConnectError("boom"), _response(200, _links())])
    assert len(tool.get_citations("1706.03762", limit=1)) == 1
    tool2 = _tool(pacer)
    _script(tool2, [httpx.ConnectError("boom")] * 10)
    with pytest.raises(SemanticScholarError):
        tool2.get_citations("1706.03762", limit=1)


# -- queue ------------------------------------------------------------------

def _park(pacer, priority, results, name):
    def run():
        try:
            pacer.acquire(priority, 1.0)
            results.append((name, "ok"))
        except SemanticScholarUnavailableError:
            results.append((name, "refused"))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_queue_bound_refuses_instead_of_growing():
    gate = threading.Event()
    clock = FakeClock()
    clock.now = 1000.0
    pacer = RequestPacer(clock=clock, sleep=lambda s: (gate.wait(5), clock.sleep(s)), max_waiting=2, max_bulk_waiting=1)
    pacer.acquire(INTERACTIVE, 1.0)  # takes the slot; the next caller must wait
    results = []
    threads = [_park(pacer, INTERACTIVE, results, "a"), _park(pacer, INTERACTIVE, results, "b")]
    while pacer.waiting() < 2:
        pass
    with pytest.raises(SemanticScholarUnavailableError, match="queued"):
        pacer.acquire(INTERACTIVE, 1.0)
    gate.set()
    for t in threads:
        t.join(5)


def test_bulk_cannot_fill_the_queue_against_interactive():
    gate = threading.Event()
    clock = FakeClock()
    pacer = RequestPacer(clock=clock, sleep=lambda s: (gate.wait(5), clock.sleep(s)), max_waiting=4, max_bulk_waiting=1)
    pacer.acquire(INTERACTIVE, 1.0)
    results = []
    first = _park(pacer, BULK, results, "bulk1")
    while pacer.waiting() < 1:
        pass
    with pytest.raises(SemanticScholarUnavailableError):
        pacer.acquire(BULK, 1.0)
    # interactive is still admitted
    second = _park(pacer, INTERACTIVE, results, "interactive")
    while pacer.waiting() < 2:
        pass
    gate.set()
    for t in (first, second):
        t.join(5)


def test_interactive_overtakes_queued_bulk():
    order = []
    gate = threading.Event()
    clock = FakeClock()
    pacer = RequestPacer(clock=clock, sleep=lambda s: (gate.wait(5), clock.sleep(s)))
    pacer.acquire(INTERACTIVE, 1.0)
    bulk = _park(pacer, BULK, order, "bulk")
    while pacer.waiting() < 1:
        pass
    inter = _park(pacer, INTERACTIVE, order, "interactive")
    while pacer.waiting() < 2:
        pass
    gate.set()
    for t in (bulk, inter):
        t.join(5)
    assert [n for n, _ in order][0] == "interactive"


# -- pagination and batch ---------------------------------------------------

def test_references_paginate_with_offset_and_next(pacer):
    tool = _tool(pacer)
    page1 = _links("p1", "citedPaper")
    page1["next"] = 100
    page2 = _links("p2", "citedPaper")
    seen = []
    _script(tool, [_response(200, page1), _response(200, page2)], seen)
    got = list(tool.iter_references("1706.03762"))
    assert [p.id for p, _ in got] == ["1706.03762", "1706.03762"]
    assert [c["params"]["offset"] for c in seen] == [0, 100]


def test_get_references_stops_at_limit_and_asks_for_only_what_it_needs(pacer):
    tool = _tool(pacer)
    seen = []
    page = _links("p1", "citedPaper")
    page["next"] = 100
    _script(tool, [_response(200, page)], seen)
    assert len(tool.get_references("1706.03762", limit=1)) == 1
    assert len(seen) == 1 and seen[0]["params"]["limit"] == 1


def test_batch_posts_ids_in_chunks_and_aligns_missing(pacer):
    tool = _tool(pacer)
    seen = []
    paper = _links()["data"][0]["citingPaper"]
    _script(tool, [_response(200, [paper, None])], seen)
    out = tool.get_papers_batch(["1706.03762", "9999.99999"])
    assert out[0].id == "1706.03762" and out[1] is None
    assert seen[0]["method"] == "POST"
    assert seen[0]["json"] == {"ids": ["arXiv:1706.03762", "arXiv:9999.99999"]}
