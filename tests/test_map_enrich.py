"""The knowledge layer on a built map: names, typed connections, grounded bridges.

The model is a stub that answers from the prompt it is given, so a test can
make it honest (quotes copied from the sources), dishonest (a quote that is
not there), or invent a number. Nothing here reaches the network.
"""

from __future__ import annotations

import copy
import re
from functools import lru_cache

import pytest
from test_map_build import (
    OTHER_LIB,
    SEED_A1,
    SEED_A2,
    SEED_B,
    fixture_inputs,
)

from nlp_pillars import enrichment, map_enrich
from nlp_pillars.connections import STATUS_CONFIRMED, STATUS_REJECTED, Decision, TypedEdge
from nlp_pillars.enrichment import EnrichmentRefused
from nlp_pillars.llm_client import LLMCallError, StructuredCall
from nlp_pillars.llm_prices import price_tokens
from nlp_pillars.map_build import build_map, sha256_of
from nlp_pillars.map_enrich import (
    BRIDGE_ABSTAINED,
    BRIDGE_DEFERRED,
    BRIDGE_GENERATED,
    BRIDGE_NO_ABSTRACT,
    BRIDGE_OFF,
    BRIDGE_REFUSED,
    BRIDGE_REJECTED,
    BridgeDraft,
    BridgeQuote,
    CommunityName,
    EnrichInputs,
    check_bridge,
    clean_name,
    clean_summary,
    enrich,
)

MODEL = "claude-haiku-5-5"
ABSTRACT_A1 = (
    "We study probing classifiers on contextual embeddings and show that control tasks "
    "expose how much a probe memorises instead of reading linguistic structure."
)
ABSTRACT_A2 = (
    "This paper applies a minimum description length view to probing classifiers, "
    "trading accuracy against the cost of describing the probe itself."
)
ABSTRACT_B = (
    "A graph network for transactions that scores accounts by aggregating the "
    "neighbourhood of each transaction over 12 hops."
)
ABSTRACT_O = (
    "A library paper on graph networks for relational data that reuses message passing "
    "over the transaction graph to detect anomalies."
)


@lru_cache(maxsize=1)
def _built():
    return build_map(fixture_inputs())


def fresh_result():
    result = _built()
    clone = copy.deepcopy(result)
    return clone


def abstracts():
    return {SEED_A1: ABSTRACT_A1, SEED_A2: ABSTRACT_A2, SEED_B: ABSTRACT_B, OTHER_LIB: ABSTRACT_O}


class Stub:
    """Stands in for ``llm_client.structured``. Answers from the prompt it receives."""

    def __init__(self, mode="good", name="Probing Classifiers", input_tokens=1000, output_tokens=100):
        self.mode = mode
        self.name = name
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = []

    def __call__(self, route, *, system, user, out_model, client=None, max_tokens=0, **kwargs):
        self.calls.append({"stage": route.stage, "system": system, "user": user, "out": out_model})
        if out_model is CommunityName:
            parsed = CommunityName(name=self.name, summary="Papers that study probing classifiers.")
        else:
            parsed = self._bridge(user)
        return StructuredCall(
            stage=route.stage,
            provider="anthropic",
            model=MODEL,
            parsed=parsed,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=price_tokens(MODEL, self.input_tokens, self.output_tokens),
        )

    def _bridge(self, user):
        sources = dict(re.findall(r"\[S(\d+)\] [^:]*: (.+)", user))
        first = sources["1"]
        words = first.split()
        real_quote = " ".join(words[:8])
        if self.mode == "abstain":
            return BridgeDraft(sentence="", quotes=[])
        if self.mode == "fabricated":
            return BridgeDraft(
                sentence="Both papers probe language models.",
                quotes=[BridgeQuote(source=1, text="this sentence is nowhere in the source text at all")],
            )
        if self.mode == "number":
            return BridgeDraft(
                sentence="Both papers report a 97 percent gain.",
                quotes=[BridgeQuote(source=1, text=real_quote)],
            )
        if self.mode == "bad_source":
            return BridgeDraft(
                sentence="Both papers probe language models.",
                quotes=[BridgeQuote(source=9, text=real_quote)],
            )
        if self.mode == "no_quotes":
            return BridgeDraft(sentence="Both papers probe language models.", quotes=[])
        return BridgeDraft(
            sentence="Both papers probe what a classifier reads out of embeddings.",
            quotes=[BridgeQuote(source=1, text=real_quote.upper())],
        )


def run(stub=None, inputs=None, **kwargs):
    result = fresh_result()
    stub = stub or Stub()
    inputs = inputs or EnrichInputs(abstracts=abstracts())
    kwargs.setdefault("use_model", True)
    kwargs.setdefault("budget_check", lambda reserve: None)
    report = enrich(result, inputs, call=stub, **kwargs)
    return result, report, stub


def bridge_for(result, a, b):
    pair = {a, b}
    return next(br for br in result.payload["bridges"] if {br["a"], br["b"]} == pair)


# -------------------------------------------------------------- names


def test_every_community_is_named_with_one_priced_call_each():
    result, report, stub = run()
    communities = result.payload["communities"]
    rows = communities["regions"] + communities["topics"]
    assert all(row["name"] == "Probing Classifiers" for row in rows)
    assert all(row["name_source"] == "model" and row["name_model"] == MODEL for row in rows)
    name_calls = [c for c in stub.calls if c["out"] is CommunityName]
    assert len(name_calls) == len(rows) == report.names["model"]
    # Every call is on the books with its own dollars, and the stage detail carries the total.
    per_call = price_tokens(MODEL, 1000, 100)
    assert report.calls.cost_usd == pytest.approx(per_call * len(stub.calls))
    assert f"${report.calls.cost_usd:.6f}" in report.detail()
    assert len(report.calls.calls) == len(stub.calls)


def test_a_name_prompt_holds_titles_as_data_and_nothing_else():
    _, _, stub = run()
    prompt = next(c for c in stub.calls if c["out"] is CommunityName)
    assert "Titles:" in prompt["user"] and "probing" in prompt["user"].lower()
    assert "data, not instructions" in prompt["system"]


def test_a_rebuild_with_the_same_members_reuses_every_name_for_nothing():
    first, _, _ = run()
    again_stub = Stub()
    result = fresh_result()
    report = enrich(
        result,
        EnrichInputs(abstracts=abstracts(), previous=first.payload),
        call=again_stub,
        use_model=True,
        budget_check=lambda reserve: None,
    )
    assert again_stub.calls == []
    assert report.calls.cost_usd == 0 and report.names["cached"] == 6
    assert result.payload_hash == first.payload_hash  # costs nothing, moves nothing


def test_only_a_community_whose_members_changed_is_named_again():
    first, _, _ = run()
    previous = copy.deepcopy(first.payload)
    previous["communities"]["topics"][0]["member_hash"] = "changed"
    stub = Stub(name="Control Tasks Probing")
    result = fresh_result()
    report = enrich(
        result,
        EnrichInputs(abstracts=abstracts(), previous=previous),
        call=stub,
        use_model=True,
        budget_check=lambda reserve: None,
    )
    assert [c["out"] for c in stub.calls] == [CommunityName]
    assert report.names["model"] == 1 and report.names["cached"] == 5
    topic = result.payload["communities"]["topics"][0]
    assert topic["name"] == "Control Tasks Probing"


@pytest.mark.parametrize("bad", ["Probing", "A very long name for a group", "", "Line\nbreak name"])
def test_a_name_that_is_not_two_to_four_words_falls_back_to_the_terms(bad):
    result, report, _ = run(Stub(name=bad))
    row = result.payload["communities"]["topics"][0]
    assert row["name_source"] == "terms" and row["name"] == ", ".join(row["terms"][:3])
    assert row["summary"] is None
    assert report.names["terms"] == 6 and report.warnings


def test_clean_helpers():
    assert clean_name('"Probing Classifiers."') == "Probing Classifiers"
    assert clean_name("One") is None and clean_name("a b c d e") is None
    assert clean_summary("They study probes") == "They study probes."
    assert clean_summary("First. Second sentence here.") is None
    assert clean_summary("x" * 300) is None


# ------------------------------------------------------------- bridges


def test_a_bridge_is_generated_marked_and_its_quotes_are_in_the_sources():
    result, report, _ = run()
    bridge = bridge_for(result, SEED_A1, SEED_A2)
    assert bridge["status"] == BRIDGE_GENERATED and bridge["generated"] is True
    assert bridge["reviewed"] is False  # nobody has read it yet
    assert bridge["model"] == MODEL and bridge["text"].startswith("Both papers probe")
    quote = bridge["quotes"][0]
    assert "abstract of" in quote["of"]
    excerpt = quote["text"].split(". ", 1)[-1].lower()  # the source opens with the paper's title
    assert excerpt in (ABSTRACT_A1 + ABSTRACT_A2).lower()  # a real excerpt of a stored abstract
    assert "sources" not in bridge  # the snapshot does not carry the source text again
    assert report.bridges[BRIDGE_GENERATED] == 2  # the third pair is a document with no abstract
    assert bridge_for(result, "paper:file:own-notes", SEED_A2)["status"] == BRIDGE_NO_ABSTRACT


def test_the_bridge_prompt_carries_only_the_stored_evidence():
    edge = TypedEdge(
        src=SEED_B, dst=OTHER_LIB, relation="builds_on", why="It reuses their message passing.",
        source_paper_id="arxiv:b", mention="Other 2018", how="author_year", score=1.0,
    )
    _, _, stub = run(inputs=EnrichInputs(abstracts=abstracts(), typed_edges=[edge]))
    prompt = next(c["user"] for c in stub.calls if c["out"] is BridgeDraft and "[S3]" in c["user"])
    assert "Abstract of" in prompt
    assert "It reuses their message passing." in prompt  # the lesson's own relation
    assert "builds on" in prompt
    assert "from memory" in stub.calls[-1]["system"]


def test_a_fabricated_quote_rejects_the_bridge_and_keeps_the_template():
    result, report, _ = run(Stub(mode="fabricated"))
    bridge = bridge_for(result, SEED_A1, SEED_A2)
    assert bridge["status"] == BRIDGE_REJECTED and "not found" in bridge["reason"]
    assert "text" not in bridge and "generated" not in bridge
    assert report.rejected and report.bridges[BRIDGE_REJECTED] == 2
    evidence = next(e for e in result.payload["evidence"] if {e["a"], e["b"]} == {SEED_A1, SEED_A2})
    assert evidence["sentence"]  # the template explanation is still there


def test_a_number_that_is_in_no_source_rejects_the_bridge():
    result, _, _ = run(Stub(mode="number"))
    bridge = bridge_for(result, SEED_A1, SEED_A2)
    assert bridge["status"] == BRIDGE_REJECTED and "97" in bridge["reason"]


@pytest.mark.parametrize("mode,reason", [("bad_source", "does not exist"), ("no_quotes", "0 quotes")])
def test_a_bridge_with_a_missing_source_or_no_quotes_is_rejected(mode, reason):
    result, _, _ = run(Stub(mode=mode))
    bridge = bridge_for(result, SEED_A1, SEED_A2)
    assert bridge["status"] == BRIDGE_REJECTED and reason in bridge["reason"]


def test_a_model_that_finds_no_connection_may_say_so_without_penalty():
    result, report, _ = run(Stub(mode="abstain"))
    assert bridge_for(result, SEED_A1, SEED_A2)["status"] == BRIDGE_ABSTAINED
    assert report.rejected == []


def test_a_rejected_outcome_is_cached_so_the_same_evidence_is_not_paid_for_twice():
    first, _, _ = run(Stub(mode="fabricated"))
    stub = Stub()
    result = fresh_result()
    enrich(
        result,
        EnrichInputs(abstracts=abstracts(), previous=first.payload),
        call=stub,
        use_model=True,
        budget_check=lambda reserve: None,
    )
    assert [c for c in stub.calls if c["out"] is BridgeDraft] == []
    assert bridge_for(result, SEED_A1, SEED_A2)["status"] == BRIDGE_REJECTED


def test_changed_evidence_is_asked_again():
    first, _, _ = run()
    changed = {**abstracts(), SEED_A1: ABSTRACT_A1 + " It also covers probe selectivity."}
    stub = Stub()
    enrich(
        fresh_result(),
        EnrichInputs(abstracts=changed, previous=first.payload),
        call=stub,
        use_model=True,
        budget_check=lambda reserve: None,
    )
    assert len([c for c in stub.calls if c["out"] is BridgeDraft]) == 1  # only the pair with SEED_A1


def test_a_pair_without_both_abstracts_is_not_bridged_and_costs_nothing():
    partial = {k: v for k, v in abstracts().items() if k != SEED_A2}
    result, _, stub = run(inputs=EnrichInputs(abstracts=partial))
    assert bridge_for(result, SEED_A1, SEED_A2)["status"] == BRIDGE_NO_ABSTRACT
    assert all(SEED_A2 not in c["user"] for c in stub.calls if c["out"] is BridgeDraft)


def test_only_library_pairs_are_bridged():
    result, _, _ = run()
    nodes = {n["id"]: n for n in result.payload["nodes"]}
    for bridge in result.payload["bridges"]:
        assert nodes[bridge["a"]]["kind"] in ("library", "document")
        assert nodes[bridge["b"]]["kind"] in ("library", "document")


def test_the_per_build_limit_defers_the_rest():
    result, report, stub = run(max_bridges=1)
    statuses = sorted(b["status"] for b in result.payload["bridges"])
    assert statuses.count(BRIDGE_GENERATED) == 1 and statuses.count(BRIDGE_DEFERRED) == 1
    assert any("deferred" in w for w in report.warnings)
    assert len([c for c in stub.calls if c["out"] is BridgeDraft]) == 1


def test_a_failed_bridge_call_is_reported_and_does_not_stop_the_pass():
    class Failing(Stub):
        def __call__(self, route, **kwargs):
            if kwargs["out_model"] is BridgeDraft:
                raise LLMCallError("boom")
            return super().__call__(route, **kwargs)

    result, report, _ = run(Failing())
    assert sorted(b["status"] for b in result.payload["bridges"]) == ["failed", "failed", BRIDGE_NO_ABSTRACT]
    assert report.bridges["failed"] == 2 and report.warnings


def test_a_broken_client_ends_the_paid_pass_after_one_attempt_and_keeps_the_templates():
    class NoKey:
        calls = 0

        def __call__(self, route, **kwargs):
            NoKey.calls += 1
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for generation")

    result, report, _ = run(NoKey())
    assert NoKey.calls == 1  # every later call is refused without being attempted
    assert all(r["name_source"] == "terms" for r in result.payload["communities"]["topics"])
    assert any("ANTHROPIC_API_KEY" in w for w in report.warnings)
    assert report.calls.cost_usd == 0


# --------------------------------------------------------- the switch


def test_with_the_switch_off_nothing_is_called_and_templates_explain():
    stub = Stub()
    result = fresh_result()
    report = enrich(result, EnrichInputs(abstracts=abstracts()), call=stub, use_model=False)
    assert stub.calls == [] and report.calls.cost_usd == 0
    rows = result.payload["communities"]["topics"]
    assert all(row["name_source"] == "terms" for row in rows)
    assert {b["status"] for b in result.payload["bridges"]} == {BRIDGE_OFF, BRIDGE_NO_ABSTRACT}
    assert all(e["sentence"] for e in result.payload["evidence"] if e["a"] in (SEED_A1, SEED_B))
    assert "Paid enrichment is off" in report.detail()


def test_the_default_follows_the_config_switch(monkeypatch):
    monkeypatch.setattr(enrichment, "enabled", lambda: False)
    stub = Stub()
    enrich(fresh_result(), EnrichInputs(abstracts=abstracts()), call=stub)
    assert stub.calls == []


def test_a_spent_cap_refuses_the_paid_pass_and_says_so():
    def refuse(reserve):
        raise EnrichmentRefused("The daily spend cap ($2.00) leaves $0.0000; paid enrichment was not started.")

    stub = Stub()
    result = fresh_result()
    report = enrich(
        result, EnrichInputs(abstracts=abstracts()), call=stub, use_model=True, budget_check=refuse
    )
    assert stub.calls == []
    assert {b["status"] for b in result.payload["bridges"]} == {BRIDGE_REFUSED, BRIDGE_NO_ABSTRACT}
    assert any("spend cap" in w for w in result.payload["warnings"])
    assert report.warnings


def test_the_budget_is_checked_with_a_reserve_for_the_calls_it_would_make():
    seen = []
    run(budget_check=seen.append)
    assert seen and seen[0] > 0


def test_a_cached_name_is_reused_even_with_the_switch_off():
    first, _, _ = run()
    stub = Stub()
    result = fresh_result()
    enrich(
        result,
        EnrichInputs(abstracts=abstracts(), previous=first.payload),
        call=stub,
        use_model=False,
    )
    assert stub.calls == []
    assert result.payload["communities"]["topics"][0]["name"] == "Probing Classifiers"


def test_a_dry_run_counts_and_prices_but_calls_nothing_and_changes_nothing():
    result = fresh_result()
    before = copy.deepcopy(result.payload)
    stub = Stub()
    report = enrich(
        result, EnrichInputs(abstracts=abstracts()), call=stub, use_model=True, dry_run=True
    )
    assert stub.calls == []
    assert report.estimated_calls == 6 + 2 and report.estimated_cost_usd > 0
    assert result.payload == before
    assert "Would make 8 paid call(s)" in report.detail()
    assert "$" not in report.detail()  # spend.py would count an estimate as spend


# ---------------------------------------------------------- connections


def edge(src, dst, relation="critiques", why="It disputes the result."):
    return TypedEdge(
        src=src, dst=dst, relation=relation, why=why, source_paper_id="arxiv:x",
        mention="Someone 2018", how="author_year", score=0.8,
    )


def test_a_lesson_connection_becomes_a_typed_entry_with_a_template_sentence():
    result, _, _ = run(inputs=EnrichInputs(abstracts=abstracts(), typed_edges=[edge(SEED_B, OTHER_LIB)]))
    entry = result.payload["connections"][0]
    assert entry["provenance"] == "lesson" and entry["status"] == "suggested"
    assert entry["relation"] == "critiques" and entry["a"] == SEED_B
    assert "critiques" in entry["sentence"] and "It disputes the result." in entry["sentence"]


def test_a_confirmed_connection_carries_the_learners_phrase_and_overrides_the_lesson():
    decision = Decision(OTHER_LIB, SEED_B, "uses", STATUS_CONFIRMED, "I use their message passing")
    inputs = EnrichInputs(
        abstracts=abstracts(), typed_edges=[edge(SEED_B, OTHER_LIB)], decisions=[decision]
    )
    result, report, _ = run(inputs=inputs)
    entries = result.payload["connections"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["provenance"] == "user" and entry["status"] == "confirmed"
    assert entry["relation"] == "uses" and entry["sentence"] == "I use their message passing"
    assert report.connections["confirmed"] == 1 and report.connections["lesson"] == 0
    evidence = next(e for e in result.payload["evidence"] if {e["a"], e["b"]} == {SEED_B, OTHER_LIB})
    assert evidence["status"] == "confirmed"


def test_a_rejected_connection_stays_visible_flagged_and_is_never_explained_or_billed():
    decision = Decision(SEED_B, OTHER_LIB, "related", STATUS_REJECTED)
    result, report, stub = run(inputs=EnrichInputs(abstracts=abstracts(), decisions=[decision]))
    entry = result.payload["connections"][0]
    assert entry["status"] == "rejected" and "rejected" in entry["sentence"]
    assert all({b["a"], b["b"]} != {SEED_B, OTHER_LIB} for b in result.payload["bridges"])
    bridge_prompts = [c["user"] for c in stub.calls if c["out"] is BridgeDraft]
    assert len(bridge_prompts) == 1 and report.connections["rejected"] == 1
    evidence = next(e for e in result.payload["evidence"] if {e["a"], e["b"]} == {SEED_B, OTHER_LIB})
    assert evidence["status"] == "rejected"  # flagged, not removed


def test_a_decision_about_works_outside_this_map_is_left_for_its_own_map():
    decision = Decision("elsewhere-1", "elsewhere-2", "related", STATUS_CONFIRMED)
    result, _, _ = run(inputs=EnrichInputs(abstracts=abstracts(), decisions=[decision]))
    assert result.payload["connections"] == []


def test_the_pass_is_deterministic_and_keeps_cost_out_of_the_payload():
    one, report_one, _ = run()
    two, _, _ = run()
    assert one.payload_hash == two.payload_hash == sha256_of(one.payload)
    assert "cost" not in str(one.payload["enrichment"]) and report_one.calls.cost_usd > 0
    assert one.payload["enrichment"] == {"version": map_enrich.ENRICH_VERSION, "paid": True}


def test_the_free_snapshot_is_untouched_apart_from_the_added_keys():
    base = _built().payload
    result, _, _ = run()
    added = {"connections", "bridges", "enrichment"}
    assert set(result.payload) - set(base) == added
    for key in set(base) - {"communities", "evidence", "warnings"}:
        assert result.payload[key] == base[key]


# --------------------------------------------------------- check_bridge


SOURCES = {1: "We study Probing classifiers on contextual embeddings, with 12 layers.", 2: "Second source text here ok."}


def test_a_quote_matches_up_to_case_and_whitespace_but_not_fuzzily():
    ok = BridgeDraft(sentence="They probe embeddings.", quotes=[BridgeQuote(source=1, text="probing  classifiers\non contextual embeddings")])
    assert check_bridge(ok, SOURCES).ok
    fuzzy = BridgeDraft(sentence="They probe embeddings.", quotes=[BridgeQuote(source=1, text="probing classifier on contextual embedding")])
    assert not check_bridge(fuzzy, SOURCES).ok


def test_the_numbers_in_a_sentence_must_come_from_a_source_or_the_builds_own_counts():
    quote = [BridgeQuote(source=1, text="Probing classifiers on contextual embeddings")]
    assert check_bridge(BridgeDraft(sentence="They use 12 layers.", quotes=quote), SOURCES).ok
    assert not check_bridge(BridgeDraft(sentence="They use 13 layers.", quotes=quote), SOURCES).ok
    assert check_bridge(BridgeDraft(sentence="They share 9 references.", quotes=quote), SOURCES, "They share 9 references.").ok


def test_a_year_in_a_paper_label_is_not_an_invented_number():
    """The first real run rejected 35 of 43 bridges for repeating "Tenney et al. 2019"."""
    quote = [BridgeQuote(source=1, text="Probing classifiers on contextual embeddings")]
    draft = BridgeDraft(sentence="Tenney et al. 2019 probe BERT.", quotes=quote)
    assert not check_bridge(draft, SOURCES).ok
    assert check_bridge(draft, SOURCES, "Tenney et al. 2019 Devlin et al. 2018").ok
    assert not check_bridge(BridgeDraft(sentence="Tenney et al. 2020 probe BERT.", quotes=quote), SOURCES, "Tenney et al. 2019").ok


def test_the_labels_of_the_pair_are_allowed_numbers_in_a_real_pass():
    class Years(Stub):
        def _bridge(self, user):
            first = dict(re.findall(r"\[S(\d+)\] [^:]*: (.+)", user))["1"]
            label = re.search(r"Paper A is labelled “([^”]+)”", user).group(1)
            return BridgeDraft(
                sentence=f"{label} studies probing.", quotes=[BridgeQuote(source=1, text=" ".join(first.split()[:8]))]
            )

    result, report, _ = run(Years())
    assert bridge_for(result, SEED_A1, SEED_A2)["status"] == BRIDGE_GENERATED  # "wA-seed1 2015" has a year


def test_short_quotes_long_sentences_and_too_many_quotes_are_refused():
    quote = [BridgeQuote(source=1, text="Probing classifiers on contextual embeddings")]
    assert not check_bridge(BridgeDraft(sentence="x" * 400, quotes=quote), SOURCES).ok
    assert not check_bridge(BridgeDraft(sentence="Fine.", quotes=[BridgeQuote(source=1, text="Probing")]), SOURCES).ok
    assert not check_bridge(BridgeDraft(sentence="Fine.\nTwo lines.", quotes=quote), SOURCES).ok
    assert not check_bridge(BridgeDraft(sentence="They probe. Then they compare.", quotes=quote), SOURCES).ok
    assert check_bridge(BridgeDraft(sentence="Pimentel et al. critique the probes of Hewitt and Liang.", quotes=quote), SOURCES).ok
    assert not check_bridge(BridgeDraft(sentence="Fine.", quotes=quote * 4), SOURCES).ok
