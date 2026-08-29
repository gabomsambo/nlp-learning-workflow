"""A discovered paper must reach `papers` with the metadata its source gave it.

The bug this pins: `webui/services/discovery_results.py` capped abstracts at 300
characters and author lists at 3, commented as *display* caps — and the payload it
produces is not a display. The browser renders from it, then posts the very same
objects back to `/select`, which persists them through run_service -> orchestrator ->
`db.upsert_paper`. So a cap written for a table column became the paper's permanent
record.

Measured on the captain's library before the fix: `2403.05525`, ingested by discovery,
carries exactly 3 authors and an abstract of exactly 300 characters plus `"..."`, cut
mid-sentence at `"real-world scenarios including..."`. A URL-uploaded paper in the same
pillar carries all 319 of its authors and the full 1538-character abstract, because
that path never went through this serialiser.

The chain is walked here in the order the real one runs, so a regression at any single
link fails a test rather than being papered over by the next one. Nothing touches a
network, a database or a vector store.
"""

import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp_pillars.schemas import DiscoveryCandidate, PaperRef  # noqa: E402
from webui.services.discovery_results import (  # noqa: E402
    candidate_to_dict,
    candidates_payload,
)
from webui.services.run_service import _to_paper_refs  # noqa: E402

PILLAR = "neural-architectures-language"

#: Both comfortably over the caps that used to be applied here (300 chars, 3 authors),
#: and shaped like the real thing: the abstract's tail is what a 300-char cut removes.
LONG_ABSTRACT = (
    "We present a model that does a great many things. " * 12
    + "The final sentence names the artifact that only a complete abstract mentions."
)
MANY_AUTHORS = [f"Author {n:03d}" for n in range(1, 320)]


@pytest.fixture
def candidate():
    assert len(LONG_ABSTRACT) > 300
    assert len(MANY_AUTHORS) > 3
    return DiscoveryCandidate(
        paper=PaperRef(
            id="arxiv:2403.05525",
            title="DeepSeek-VL: Towards Real-World Vision-Language Understanding",
            authors=MANY_AUTHORS,
            venue="arXiv:cs.CL",
            year=2024,
            url_pdf="https://arxiv.org/pdf/2403.05525v2",
            abstract=LONG_ABSTRACT,
            citation_count=42,
        ),
        source="arxiv",
        relevance_score=0.87,
        citation_count=42,
        is_influential=False,
    )


def test_the_stored_run_payload_keeps_the_whole_abstract_and_every_author(candidate):
    """Link 1: what the worker writes to pipeline_runs.result."""
    stored = candidates_payload([candidate])["candidates"][0]["paper"]

    assert stored["abstract"] == LONG_ABSTRACT
    assert not stored["abstract"].endswith("...")
    assert stored["authors"] == MANY_AUTHORS
    assert stored["venue"] == "arXiv:cs.CL"


def test_the_select_request_model_accepts_every_field_the_payload_carries(candidate):
    """Link 2: the browser posts the candidate's paper object back verbatim.

    FastAPI drops any key the model does not declare, silently, so a field that
    survives serialisation can still be lost here. `venue` was: it had no home on
    PaperData, which is why every discovery-ingested paper had `venue = NULL`.
    """
    from webui.routers.api.discovery import PaperData

    paper_dict = candidate_to_dict(candidate)["paper"]
    parsed = PaperData(**paper_dict)

    assert set(paper_dict) <= set(parsed.model_dump()), (
        "the /select request model drops a field the discovery payload carries"
    )
    assert parsed.abstract == LONG_ABSTRACT
    assert parsed.authors == MANY_AUTHORS
    assert parsed.venue == "arXiv:cs.CL"


def test_the_paper_persisted_by_select_is_the_paper_discovery_found(candidate):
    """The whole chain, ending at the PaperRef handed to db.upsert_paper.

    Fails against the pre-fix code at the first assertion: the abstract arrives as
    303 characters ending in "...".
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from webui.routers.api import discovery as discovery_router

    app = FastAPI()
    app.include_router(discovery_router.router)
    app.state.scheduler = object()
    app.state.cancel_events = {}

    # What a finished run stores, and therefore what the page renders and posts back.
    posted = [candidates_payload([candidate])["candidates"][0]["paper"]]

    with patch.object(discovery_router, "dispatch_run", return_value="run-1") as dispatch:
        response = TestClient(app).post(
            f"/api/pillars/{PILLAR}/select", json={"papers": posted}
        )

    assert response.status_code == 202
    papers = dispatch.call_args.kwargs["papers"]

    # run_service rebuilds PaperRefs on the worker thread; this is the object the
    # orchestrator hands to db.upsert_paper.
    (persisted,) = _to_paper_refs(papers)

    assert persisted.abstract == LONG_ABSTRACT
    assert persisted.authors == MANY_AUTHORS
    assert len(persisted.authors) == 319
    assert persisted.venue == "arXiv:cs.CL"
    assert persisted.id == "arxiv:2403.05525"
    assert persisted.year == 2024
    assert persisted.url_pdf == "https://arxiv.org/pdf/2403.05525v2"


def test_no_length_cap_survives_in_the_serialiser():
    """The caps are gone, and a future one must be a deliberate act.

    Cheap, and it names the failure directly: the old constants were `_ABSTRACT_CHARS`
    and `_MAX_AUTHORS`, and a slice on either field is what shortened the record.
    """
    source = (ROOT / "webui" / "services" / "discovery_results.py").read_text()
    body = source.split('"""', 2)[-1]  # skip the module docstring, which discusses them

    assert "_ABSTRACT_CHARS" not in body
    assert "_MAX_AUTHORS" not in body
    assert "[:" not in body, "a slice in this module truncates a persisted record"


def test_the_browser_posts_back_every_field_it_was_given():
    """The JS link, which no other test can reach.

    `discovery.html` rebuilds each selected candidate by hand before POSTing it, so a
    field added to the serialiser and to PaperData is still dropped unless it is added
    here too — and the loss is invisible: the paper saves, just thinner.
    """
    template = (ROOT / "webui" / "templates" / "discovery.html").read_text()

    mapping = re.search(
        r"const selectedPapers = candidates.*?\.map\(c => \((\{.*?\})\)\);",
        template,
        re.S,
    )
    assert mapping, "the /select payload is no longer built where this test looks"

    posted_keys = set(re.findall(r"^\s*(\w+):", mapping.group(1), re.M))
    serialised_keys = set(
        candidate_to_dict(
            DiscoveryCandidate(
                paper=PaperRef(id="x", title="t", authors=[]),
                source="arxiv",
                relevance_score=0.5,
            )
        )["paper"]
    )

    assert serialised_keys <= posted_keys, (
        "discovery.html drops "
        f"{sorted(serialised_keys - posted_keys)} on the way back to /select"
    )
    # And nothing on that path may shorten a value.
    assert ".slice(" not in mapping.group(1)
    assert ".substring(" not in mapping.group(1)
