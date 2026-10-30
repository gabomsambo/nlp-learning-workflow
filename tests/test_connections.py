"""Typed connections: lesson mentions resolved to works, and the learner's decisions."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_map_store import FakeHttp, reply

from nlp_pillars import connections, db
from nlp_pillars.connections import (
    ConnectionStoreError,
    Mention,
    Reference,
    clean_phrase,
    guess_relation,
    legacy_links,
    lesson_links,
    pair_key,
    parse_mention,
    resolve_lessons,
    resolve_mention,
)
from nlp_pillars.db import PostgRESTClient
from nlp_pillars.schemas import (
    CONNECTION_RELATIONS,
    ConnectionDraft,
    Lesson,
    LessonDraft,
    connection_text,
    lesson_from_draft,
)

ROOT = Path(__file__).resolve().parents[1]
MIGRATION_SQL = (ROOT / "docs/migrations/035_knowledge_layer.sql").read_text()
SCHEMA = (ROOT / "schema.sql").read_text()

REFS = [
    Reference("w1", "Designing and Interpreting Probes with Control Tasks", ("John Hewitt", "Percy Liang"), 2019),
    Reference("w2", "Information-Theoretic Probing for Linguistic Structure", ("Tiago Pimentel", "Josef Valvoda"), 2020),
    Reference("w3", "Attention is All you Need", ("Ashish Vaswani", "Noam Shazeer"), 2017),
    Reference("w4", "A study of fraud", ("Yu Zhang",), 2020),
    Reference("w5", "Another study of fraud", ("Yue Zhang",), 2020),
    Reference("w6", "Élan Vital in Language", ("Hélène Müller",), 2021),
]


# ----------------------------------------------------------- resolution


@pytest.mark.parametrize(
    "text,surnames,year",
    [
        ("Hewitt and Liang 2019", ("hewitt", "liang"), 2019),
        ("Hewitt & Liang (2019)", ("hewitt", "liang"), 2019),
        ("Pimentel et al., 2020", ("pimentel",), 2020),
        ("Vaswani 2017", ("vaswani",), 2017),
        ("Vaswani et al. 2017b", ("vaswani",), 2017),
        ("something with no year", (), None),
    ],
)
def test_a_mention_is_parsed_into_surnames_and_year(text, surnames, year):
    mention = parse_mention(text)
    assert mention.surnames == surnames and mention.year == year


@pytest.mark.parametrize(
    "work,title,expected",
    [
        ("Hewitt and Liang 2019", "", "w1"),
        ("Hewitt and Liang 2018", "", "w1"),  # a preprint year one off still matches
        ("Hewitt and Liang 2016", "", None),  # three years off does not
        ("Pimentel et al. 2020", "", "w2"),
        ("Muller 2021", "", "w6"),  # accents fold
        ("X", "Attention Is All You Need", "w3"),  # a stated title is enough
        ("Voita 2020", "", None),  # not a reference of this paper
        ("Zhang 2020", "", None),  # two Zhangs: ambiguous, not guessed
        ("", "", None),
    ],
)
def test_a_mention_resolves_to_one_reference_or_to_none(work, title, expected):
    assert resolve_mention(parse_mention(work, title), REFS).work_id == expected


def test_the_reason_a_mention_did_not_resolve_is_kept():
    assert resolve_mention(parse_mention("Zhang 2020"), REFS).how == "ambiguous"
    assert resolve_mention(parse_mention("Zhang 2020"), REFS).candidates == ("w4", "w5")
    assert resolve_mention(parse_mention("Voita 2020"), REFS).how == "no_match"
    assert resolve_mention(Mention(raw=""), REFS).how == "unparsed"


def test_a_second_surname_breaks_a_tie_between_two_first_authors():
    refs = [
        Reference("a", "T one", ("Yu Zhang", "Li Wang"), 2020),
        Reference("b", "T two", ("Yue Zhang", "Wei Chen"), 2020),
    ]
    assert resolve_mention(parse_mention("Zhang and Wang 2020"), refs).work_id == "a"


# ---------------------------------------------------------------- lessons


def test_a_typed_draft_gives_the_prose_line_and_the_typed_copy():
    draft = ConnectionDraft(
        work="Hewitt and Liang 2019", title="Control Tasks", relation="critiques", why="penalises memorisation"
    )
    assert connection_text(draft) == "Hewitt and Liang 2019, Control Tasks (critiques): penalises memorisation"
    lesson = lesson_from_draft(
        LessonDraft(
            title="t", content="c", tl_dr="d", takeaways=["x"], practice_ideas=["y"], connections=[draft]
        ),
        "arxiv:1",
        "p",
    )
    assert isinstance(lesson, Lesson)
    assert lesson.connections == [connection_text(draft)]
    assert lesson.connection_links == [draft.model_dump()]
    assert set(CONNECTION_RELATIONS) == {"builds_on", "uses", "critiques", "compares", "alternative", "background"}


def test_a_relation_outside_the_list_is_refused_by_the_draft_schema():
    with pytest.raises(ValueError):
        ConnectionDraft(work="A 2020", relation="refutes", why="x")


def test_legacy_prose_is_read_for_author_year_mentions_and_marked():
    links = legacy_links(
        [
            "Hewitt and Liang (2019): control tasks, which this paper critiques for penalizing memorization",
            "BERT uses the encoder",
            "Builds on Vaswani et al., 2017 and (Bahdanau, 2014)",
            "In 2019 the field moved on",
        ]
    )
    assert [(l["work"], l["relation"]) for l in links] == [
        ("Hewitt and Liang 2019", "critiques"),
        ("Vaswani et al. 2017", "builds_on"),
        ("Bahdanau 2014", "builds_on"),
    ]
    assert all(l["legacy"] for l in links)


@pytest.mark.parametrize(
    "text,relation",
    [
        ("which this paper critiques", "critiques"),
        ("an alternative to attention", "alternative"),
        ("compared with a baseline", "compares"),
        ("extends their method", "builds_on"),
        ("uses their encoder", "uses"),
        ("mentioned in passing", "background"),
    ],
)
def test_the_relation_is_guessed_from_cue_words_or_is_background(text, relation):
    assert guess_relation(text) == relation


def test_stored_typed_links_win_over_prose():
    row = {
        "connection_links": [{"work": "A 2019", "relation": "uses", "why": "x"}],
        "connections": ["Hewitt and Liang (2019) critiques"],
    }
    assert lesson_links(row)[0]["work"] == "A 2019"
    assert lesson_links({"connections": ["Hewitt and Liang (2019) critiques"]})[0]["legacy"] is True


def test_lessons_resolve_to_typed_edges_and_report_what_they_could_not():
    lessons = [
        {
            "paper_id": "arxiv:p",
            "connection_links": [
                {"work": "Hewitt and Liang 2019", "title": "", "relation": "critiques", "why": "memorisation"},
                {"work": "Hewitt and Liang 2019", "title": "", "relation": "critiques", "why": "again"},
                {"work": "Zhang 2020", "title": "", "relation": "uses", "why": "x"},
                {"work": "Voita 2020", "title": "", "relation": "uses", "why": "x"},
                {"work": "Odd 2020", "title": "", "relation": "refutes", "why": "x"},
            ],
        },
        {"paper_id": "file:doc", "connection_links": [{"work": "Hewitt and Liang 2019", "relation": "uses", "why": ""}]},
        {"paper_id": "arxiv:unknown", "connection_links": [{"work": "A 2020", "relation": "uses", "why": ""}]},
    ]
    report = resolve_lessons(
        lessons, {"arxiv:p": "node-p", "file:doc": "paper:file:doc"}, {"node-p": REFS, "paper:file:doc": []}
    )
    assert [(e.src, e.dst, e.relation, e.how) for e in report.edges] == [
        ("node-p", "w1", "critiques", "author_year")
    ]  # the duplicate is one edge; the document has no reference list
    reasons = sorted((r["mention"], r["reason"]) for r in report.unresolved)
    assert reasons == [
        ("Hewitt and Liang 2019", "no_references"),
        ("Odd 2020", "no_match"),
        ("Voita 2020", "no_match"),
        ("Zhang 2020", "ambiguous"),
    ]
    assert report.counts() == {"resolved": 1, "unresolved": 4, "legacy": 0}


def test_a_connection_to_itself_is_not_an_edge():
    refs = [Reference("node-p", "Self", ("A Self",), 2020)]
    report = resolve_lessons(
        [{"paper_id": "p", "connection_links": [{"work": "Self 2020", "relation": "uses", "why": ""}]}],
        {"p": "node-p"},
        {"node-p": refs},
    )
    assert report.edges == [] and report.unresolved[0]["reason"] == "self"


# --------------------------------------------------------------- decisions


@pytest.fixture
def store():
    state = {"rows": {}}
    fake = FakeHttp({"learner_connections": []})

    def post(url, json=None):
        name = url.rsplit("/", 1)[-1]
        fake.posts.append((name, json))
        if name == "put_learner_connection":
            key = pair_key(json["p_src"], json["p_dst"])
            row = {
                "src": json["p_src"], "dst": json["p_dst"], "relation": json["p_relation"],
                "status": json["p_status"], "phrase": json["p_phrase"], "provenance": "user",
                "updated_at": "2026-10-30T00:00:00+00:00",
            }
            state["rows"][key] = row
            return reply(200, row)
        if name == "delete_learner_connection":
            removed = 1 if state["rows"].pop(pair_key(json["p_a"], json["p_b"]), None) else 0
            return reply(200, removed)
        raise AssertionError(name)

    fake.client.post.side_effect = post
    fake.tables["learner_connections"] = state["rows"]
    client = PostgRESTClient("http://postgrest.test", "token")
    client.client = fake.client
    previous = db._client
    db.set_client(client)
    fake.state = state
    yield fake
    db._client = previous


def test_a_decision_is_stored_through_the_rpc_and_comes_back(store):
    decision = connections.put_decision(
        "w1", "w2", status="confirmed", relation="builds_on", phrase="  my   words \n here "
    )
    assert decision.status == "confirmed" and decision.provenance == "user"
    name, body = store.posts[-1]
    assert name == "put_learner_connection"
    assert body == {
        "p_src": "w1", "p_dst": "w2", "p_relation": "builds_on", "p_status": "confirmed", "p_phrase": "my words here",
    }


def test_a_pair_has_one_decision_whichever_way_it_is_sent(store):
    connections.put_decision("w1", "w2", status="confirmed", relation="uses")
    connections.put_decision("w2", "w1", status="rejected")
    assert len(store.state["rows"]) == 1
    assert connections.delete_decision("w2", "w1") == 1
    assert connections.delete_decision("w1", "w2") == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": "maybe"},
        {"status": "confirmed", "relation": "refutes"},
        {"status": "confirmed", "phrase": "x" * 401},
    ],
)
def test_bad_decisions_are_refused_before_any_write(store, kwargs):
    with pytest.raises(ValueError):
        connections.put_decision("w1", "w2", **kwargs)
    assert store.posts == []


def test_two_different_works_are_required(store):
    with pytest.raises(ValueError):
        connections.put_decision("w1", " w1 ", status="confirmed")


def test_a_missing_table_names_the_migration():
    fake = FakeHttp({})
    fake.missing = {"learner_connections", "put_learner_connection"}
    client = PostgRESTClient("http://postgrest.test", "token")
    client.client = fake.client
    previous = db._client
    db.set_client(client)
    try:
        with pytest.raises(ConnectionStoreError, match="035_knowledge_layer.sql"):
            connections.load_decisions()
        with pytest.raises(ConnectionStoreError, match="035_knowledge_layer.sql"):
            connections.put_decision("w1", "w2", status="confirmed")
    finally:
        db._client = previous


def test_decisions_load_as_values(store):
    store.tables["learner_connections"] = [
        {"src": "b", "dst": "a", "relation": "uses", "status": "confirmed", "phrase": "p", "provenance": "user"}
    ]
    [decision] = connections.load_decisions()
    assert decision.pair == ("a", "b") and decision.phrase == "p"


def test_clean_phrase():
    assert clean_phrase(None) is None and clean_phrase("  \n ") is None
    assert clean_phrase("a\x00b\tc") == "a b c"


# --------------------------------------------------------------- migration


def test_the_migration_is_hand_applied_idempotent_and_granted():
    sql = MIGRATION_SQL
    assert "MUST BE RUN BY HAND" in sql and "NOTIFY pgrst, 'reload schema'" in sql
    for statement in ("ADD COLUMN IF NOT EXISTS", "CREATE TABLE IF NOT EXISTS learner_connections", "CREATE OR REPLACE FUNCTION"):
        assert statement in sql
    for grant in (
        "GRANT SELECT, INSERT, UPDATE, DELETE ON learner_connections TO web_anon",
        "GRANT EXECUTE ON FUNCTION apply_concept_anchoring(JSONB, JSONB, JSONB) TO web_anon",
        "GRANT EXECUTE ON FUNCTION put_learner_connection(TEXT, TEXT, TEXT, TEXT, TEXT) TO web_anon",
        "GRANT EXECUTE ON FUNCTION delete_learner_connection(TEXT, TEXT) TO web_anon",
    ):
        assert grant in sql
    assert "provenance = 'user'" in sql  # a learner decision is always the learner's


def test_the_relations_in_the_database_are_the_ones_in_code():
    for relation in connections.RELATIONS:
        assert f"'{relation}'" in MIGRATION_SQL


def test_schema_sql_carries_the_same_objects_for_a_fresh_volume():
    for name in (
        "connection_links JSONB",
        "learner_connections",
        "apply_concept_anchoring",
        "put_learner_connection",
        "delete_learner_connection",
        "anchor_relation",
        "concepts_anchor_relation_check",
    ):
        assert name in SCHEMA


# --------------------------------------------------------------------- CLI


def test_cli_connections_resolve_prints_counts_and_unresolved_reasons():
    from unittest.mock import patch

    from typer.testing import CliRunner

    from nlp_pillars.cli import app

    report = resolve_lessons(
        [{"paper_id": "p", "connection_links": [
            {"work": "Hewitt and Liang 2019", "relation": "uses", "why": ""},
            {"work": "Voita 2020", "relation": "uses", "why": ""},
        ]}],
        {"p": "node-p"},
        {"node-p": REFS},
    )
    with patch("nlp_pillars.map_store.resolve_library_connections", return_value=(report, ["a warning"])):
        result = CliRunner().invoke(app, ["connections", "resolve"])
    assert result.exit_code == 0, result.output
    assert "1 of 2 resolved" in result.output and "warning: a warning" in result.output
    assert "unresolved 'Voita 2020' in p: no_match" in result.output
