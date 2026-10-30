"""The enrichment stage inside a map build: what it reads, what it writes, and that it can't take the build down."""

from __future__ import annotations

import pytest
from test_map_build import OTHER_LIB, PILLAR, SEED_B
from test_map_enrich import ABSTRACT_A1, ABSTRACT_A2, ABSTRACT_B, ABSTRACT_O, Stub
from test_map_store import fake_embedder, http, reply  # noqa: F401  (http is a pytest fixture)

from nlp_pillars import enrichment, map_store
from nlp_pillars.map_enrich import BridgeDraft
from nlp_pillars.schemas import MAP_BUILD_STAGES


@pytest.fixture(autouse=True)
def paid_on(monkeypatch):
    monkeypatch.setattr(enrichment, "enabled", lambda: True)
    monkeypatch.setattr(enrichment, "require_budget", lambda reserve: None)


ABSTRACTS = {"arxiv:a1": ABSTRACT_A1, "arxiv:a2": ABSTRACT_A2, "arxiv:b": ABSTRACT_B, "arxiv:o": ABSTRACT_O}


def seed(http):  # noqa: F811
    for paper in http.tables["papers"]:
        paper["abstract"] = ABSTRACTS.get(paper["id"])
    for work in http.tables["works"]:
        if work["id"] == OTHER_LIB:
            work["authors"] = ["Ann Hewitt"]
            work["year"] = 2018
    http.tables["lessons"] = [
        {
            "paper_id": "arxiv:b",
            "created_at": "2026-10-01",
            "connections": ["Hewitt (2018): the message passing, which this paper builds on"],
            "connection_links": [
                {"work": "Hewitt 2018", "title": "", "relation": "builds_on", "why": "It reuses their message passing."}
            ],
        }
    ]
    http.tables["learner_connections"] = []


def build(http, stub=None, **kwargs):  # noqa: F811
    (embed, model), _ = fake_embedder(http.inputs)
    stages = []
    report = map_store.execute_map_build(
        pillar=PILLAR,
        embedder=(embed, model),
        enrich_call=stub,
        on_stage=lambda name, status, detail: stages.append((name, status, detail)),
        run_id="run-1",
        **kwargs,
    )
    return report, stages


def test_the_stage_reports_the_dollars_and_the_snapshot_row_keeps_the_cost(http):  # noqa: F811
    seed(http)
    stub = Stub()
    report, stages = build(http, stub)
    detail = next(d for n, s, d in stages if n == "map_enrich" and s == "completed")
    assert f"${report.pillars[0].enrichment.calls.cost_usd:.6f}" in detail and "claude-haiku-5-5" in detail
    [(name, body)] = [p for p in http.posts if p[0] == "put_map_snapshot"]
    assert body["p_stats"]["enrichment"]["cost_usd"] == report.pillars[0].enrichment.calls.cost_usd > 0
    assert "cost_usd" not in str(body["p_payload"]["enrichment"])
    assert [n for n, s, _ in stages if s == "completed"] == [s.value for s in MAP_BUILD_STAGES]


def test_the_lessons_typed_connection_is_resolved_against_the_references_and_bridged(http):  # noqa: F811
    seed(http)
    stub = Stub()
    report, _ = build(http, stub)
    payload = report.pillars[0].result.payload
    [entry] = [c for c in payload["connections"] if c["provenance"] == "lesson"]
    assert (entry["a"], entry["b"], entry["relation"]) == (SEED_B, OTHER_LIB, "builds_on")
    prompt = next(c["user"] for c in stub.calls if c["out"] is BridgeDraft and "It reuses their message passing." in c["user"])
    assert "builds on" in prompt
    bridge = next(b for b in payload["bridges"] if {b["a"], b["b"]} == {SEED_B, OTHER_LIB})
    assert bridge["status"] == "generated" and bridge["generated"] is True


def test_the_previous_snapshot_is_the_cache_so_a_second_build_costs_nothing(http):  # noqa: F811
    seed(http)
    first, _ = build(http, Stub())
    http.tables["map_snapshots"] = [{"scope": PILLAR, "payload": first.pillars[0].result.payload, "positions": {}, "layout_key": None}]
    stub = Stub()
    second, _ = build(http, stub)
    assert stub.calls == []
    assert second.pillars[0].enrichment.calls.cost_usd == 0
    assert second.pillars[0].result.payload_hash == first.pillars[0].result.payload_hash


def test_a_database_without_migration_035_reads_lesson_prose_and_says_so(http):  # noqa: F811
    seed(http)
    real_get = http._get

    def get(url, params=None):
        table = url.rsplit("/", 1)[-1]
        if table == "lessons" and "connection_links" in (params or {}).get("select", ""):
            return reply(400, {"code": "42703", "message": "column lessons.connection_links does not exist"})
        if table == "learner_connections":
            return reply(404, {"code": "PGRST205", "message": "Could not find the table 'public.learner_connections' in the schema cache"})
        response = real_get(url, params)
        if table == "lessons":  # the column does not exist, so the rows do not carry it
            rows = [{k: v for k, v in row.items() if k != "connection_links"} for row in response.json()]
            return reply(200, rows)
        return response

    http.client.get.side_effect = get
    report, _ = build(http, Stub())
    result = report.pillars[0].result
    text = " ".join(result.warnings)
    assert "035_knowledge_layer.sql" in text
    legacy = [c for c in result.payload["connections"] if c.get("legacy")]
    assert [(c["a"], c["b"], c["relation"]) for c in legacy] == [(SEED_B, OTHER_LIB, "builds_on")]


def test_an_unexpected_error_fails_only_the_stage(http, monkeypatch):  # noqa: F811
    seed(http)

    def broken(*args, **kwargs):
        raise RuntimeError("abstracts exploded")

    monkeypatch.setattr(map_store, "load_abstracts", broken)
    report, stages = build(http, Stub())
    assert report.pillars[0].written is True
    failed = [(n, s, d) for n, s, d in stages if s == "failed"]
    assert failed and failed[0][0] == "map_enrich" and "abstracts exploded" in failed[0][2]
    assert any("Enrichment failed" in w for w in report.pillars[0].result.warnings)
    assert "bridges" not in report.pillars[0].result.payload  # the free snapshot, untouched


def test_a_dry_run_prices_the_pass_and_calls_nothing(http):  # noqa: F811
    seed(http)
    stub = Stub()
    report, stages = build(http, stub, dry_run=True)
    assert stub.calls == [] and not [p for p in http.posts if p[0] == "put_map_snapshot"]
    detail = next(d for n, s, d in stages if n == "map_enrich" and s == "completed")
    assert detail.startswith("Would make")
    assert report.pillars[0].enrichment.estimated_cost_usd > 0
