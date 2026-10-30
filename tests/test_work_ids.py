"""Work identity keys. Pure functions: no database, no network."""

import json
from pathlib import Path

import pytest

from nlp_pillars.work_ids import (
    best_lookup_key,
    display_id,
    keys_from_s2,
    parse_paper_id,
    s2_lookup_id,
)

FIXTURES = Path(__file__).parent / "fixtures" / "semantic_scholar"
ATTENTION_S2 = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"


@pytest.mark.parametrize(
    "spelling",
    [
        "arxiv:1706.03762",
        "arXiv:1706.03762",
        "ARXIV:1706.03762v5",
        "1706.03762",
        "1706.03762v2",
        "  1706.03762  ",
        "https://arxiv.org/abs/1706.03762v7",
        "https://arxiv.org/pdf/1706.03762.pdf",
    ],
)
def test_every_spelling_of_an_arxiv_id_is_one_key(spelling):
    assert parse_paper_id(spelling) == ("arxiv", "1706.03762")


def test_old_style_arxiv_ids_keep_the_shape_arxiv_prints():
    assert parse_paper_id("arxiv:cs/0501001v2") == ("arxiv", "cs/0501001")
    assert parse_paper_id("Math.GT/0309136") == ("arxiv", "math.GT/0309136")
    assert parse_paper_id("math.gt/0309136") == ("arxiv", "math.GT/0309136")


@pytest.mark.parametrize(
    "spelling",
    [
        "10.18653/v1/N19-1357",
        "doi:10.18653/V1/N19-1357",
        "DOI:10.18653/v1/N19-1357",
        "https://doi.org/10.18653/v1/N19-1357",
        "https://dx.doi.org/10.18653/v1/N19-1357.",
    ],
)
def test_dois_are_lowercase_and_lose_their_resolver(spelling):
    assert parse_paper_id(spelling) == ("doi", "10.18653/v1/n19-1357")


def test_a_doi_with_a_colon_and_parentheses_survives():
    doi = "10.1002/(sici)1097-4571(199806)49:8<693::aid-asi4>3.0.co;2-o"
    assert parse_paper_id(doi) == ("doi", doi)


def test_s2_paper_ids_are_lowercase_hex():
    assert parse_paper_id(ATTENTION_S2.upper()) == ("s2", ATTENTION_S2)
    assert parse_paper_id(f"S2:{ATTENTION_S2}") == ("s2", ATTENTION_S2)
    assert parse_paper_id("204e3073") is None


def test_acl_and_corpus_prefixes():
    assert parse_paper_id("ACL:P18-1128") == ("acl", "p18-1128")
    assert parse_paper_id("CorpusId:13756489") == ("corpus", "13756489")
    assert parse_paper_id("CorpusId:abc") is None


@pytest.mark.parametrize(
    "raw", [None, "", "   ", "file:2dd76e910fbc", "searxng_078015", 123, "arxiv:", "doi:nope"]
)
def test_an_id_that_names_no_work_is_none(raw):
    assert parse_paper_id(raw) is None


def test_keys_from_a_recorded_s2_record_hold_every_alias_s2_first():
    batch = json.loads((FIXTURES / "paper_batch.json").read_text())
    attention, explanation = batch[0], batch[1]
    keys = keys_from_s2(attention["paperId"], attention["externalIds"], attention["corpusId"])
    assert keys == [("s2", ATTENTION_S2), ("arxiv", "1706.03762"), ("corpus", "13756489")]
    assert keys[0] == parse_paper_id(attention["paperId"])
    # the library spelling and the S2 spelling meet at one key
    assert parse_paper_id("arxiv:1706.03762") in keys

    keys = keys_from_s2(explanation["paperId"], explanation["externalIds"], explanation["corpusId"])
    assert ("doi", "10.18653/v1/n19-1357") in keys and ("acl", "n19-1357") in keys


def test_keys_are_deduplicated_and_junk_dropped():
    keys = keys_from_s2(
        ATTENTION_S2,
        {"ArXiv": "1706.03762v3", "DOI": "", "MAG": "2626778328", "CorpusId": 1},
        corpus_id=1,
    )
    assert keys == [("s2", ATTENTION_S2), ("arxiv", "1706.03762"), ("corpus", "1")]
    assert keys_from_s2(None, None, None) == []
    assert keys_from_s2("not-an-s2-id", {}) == []


def test_lookup_ids_use_the_syntax_s2_accepts():
    assert s2_lookup_id(("arxiv", "1706.03762")) == "ARXIV:1706.03762"
    assert s2_lookup_id(("doi", "10.1/x")) == "DOI:10.1/x"
    assert s2_lookup_id(("s2", ATTENTION_S2)) == ATTENTION_S2
    assert s2_lookup_id(("title_hash", "abc")) is None


def test_the_key_asked_about_is_the_most_specific_one():
    assert best_lookup_key([("arxiv", "1706.03762"), ("s2", ATTENTION_S2)]) == ("s2", ATTENTION_S2)
    assert best_lookup_key([("doi", "10.1/x"), ("arxiv", "1706.03762")]) == ("arxiv", "1706.03762")
    assert best_lookup_key([("title_hash", "abc")]) is None
    assert best_lookup_key([]) is None


def test_display_id_matches_what_the_app_always_called_an_external_paper():
    assert display_id([("s2", ATTENTION_S2), ("doi", "10.1/x"), ("arxiv", "1706.03762")]) == "1706.03762"
    assert display_id([("s2", ATTENTION_S2), ("doi", "10.1/x")]) == "10.1/x"
    assert display_id([("s2", ATTENTION_S2)]) == ATTENTION_S2
    assert display_id([("corpus", "1")]) is None
