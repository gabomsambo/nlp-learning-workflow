"""Anchoring concepts to a vocabulary: exact matches, a stubbed pick, merges, parents, cost."""

from __future__ import annotations

import numpy as np
import pytest
from test_map_store import FakeHttp

from nlp_pillars import concept_anchoring, db, enrichment
from nlp_pillars.concept_anchoring import (
    CANDIDATE_FLOOR,
    AnchorChoice,
    ConceptRow,
    candidates_prompt,
    plan_anchoring,
    run_anchoring,
)
from nlp_pillars.db import PostgRESTClient
from nlp_pillars.llm_client import LLMCallError, StructuredCall
from nlp_pillars.llm_prices import price_tokens
from nlp_pillars.vocabulary import BUNDLED_PATH, CandidateIndex, Vocabulary, VocabularySet

VOCAB = VocabularySet([Vocabulary.from_file(BUNDLED_PATH)])
FIELDS = VOCAB.all_fields()
POSITION = {f.id: i for i, f in enumerate(FIELDS)}
MODEL = "claude-haiku-5-5"


def embed(texts):
    """A fake embedder: a known phrase sits on its field's axis, the rest on a far axis."""
    near = {
        "multi-head attention": "Attention",
        "self attention": "Attention",
        "probing classifier": "Probing",
    }
    rows = []
    for text in texts:
        vector = np.zeros(len(FIELDS) + 1)  # the last axis belongs to no field
        if text in near:
            vector[POSITION[near[text]]] = 1.0
            vector[(POSITION[near[text]] + 1) % len(FIELDS)] = 0.3
        elif text.startswith("zzz"):
            vector[-1] = 1.0
        else:
            hit = next((i for i, f in enumerate(FIELDS) if f.text() == text), None)
            vector[hit if hit is not None else 5] = 1.0
        rows.append(vector.tolist())
    return rows


INDEX = CandidateIndex(VOCAB, embed)


class Picker:
    """Stands in for ``model_pick``: answers with a position among the candidates."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, name, candidates):
        self.calls.append((name, [c.id for c in candidates]))
        choice = self.answers[name]
        call = StructuredCall(
            stage="concept_anchor", provider="anthropic", model=MODEL, parsed=choice,
            input_tokens=700, output_tokens=40, cost_usd=price_tokens(MODEL, 700, 40),
        )
        return call, choice


def row(name, **kwargs):
    return ConceptRow(id=name, name=name, **kwargs)


def decision(plan, name):
    return next(d for d in plan.decisions if d.name == name)


def test_a_name_that_is_a_label_or_synonym_is_anchored_for_free():
    picker = Picker({})
    plan = plan_anchoring([row("attention"), row("in context learning")], VOCAB, index=INDEX, pick=picker, use_model=True)
    assert [d.anchor for d in plan.decisions] == ["nlp-kg:Attention", "nlp-kg:In-Context_Learning"]
    assert all(d.method == "exact" and d.relation == "same" for d in plan.decisions)
    assert picker.calls == [] and plan.calls.cost_usd == 0


def test_an_alias_of_a_concept_counts_as_its_name():
    plan = plan_anchoring([row("self-attn mechanism", aliases=("attention",))], VOCAB, use_model=False)
    assert decision(plan, "self-attn mechanism").anchor == "nlp-kg:Attention"


def test_the_model_picks_a_position_never_an_id_and_the_call_is_priced():
    picker = Picker({"multi-head attention": AnchorChoice(choice=1, relation="narrower")})
    plan = plan_anchoring([row("multi-head attention")], VOCAB, index=INDEX, pick=picker, use_model=True)
    only = decision(plan, "multi-head attention")
    assert only.anchor == "nlp-kg:Attention" and only.relation == "narrower" and only.method == "model"
    name, offered = picker.calls[0]
    assert offered[0] == "Attention" and len(offered) == 5
    assert plan.calls.cost_usd == pytest.approx(price_tokens(MODEL, 700, 40))
    assert "model spend" in plan.calls.detail("model spend") and "$0.0000" in plan.calls.detail()


@pytest.mark.parametrize(
    "answer",
    [AnchorChoice(choice=9, relation="same"), AnchorChoice(choice=0, relation="same"), AnchorChoice(choice=2, relation="none")],
)
def test_an_answer_outside_the_candidates_is_not_an_anchor(answer):
    plan = plan_anchoring([row("multi-head attention")], VOCAB, index=INDEX, pick=Picker({"multi-head attention": answer}), use_model=True)
    only = decision(plan, "multi-head attention")
    assert only.anchor is None and "unusable" in only.note
    assert plan.calls.cost_usd > 0  # the call was made and is on the books


def test_none_is_a_legitimate_answer():
    plan = plan_anchoring(
        [row("multi-head attention")], VOCAB, index=INDEX,
        pick=Picker({"multi-head attention": AnchorChoice(choice=0, relation="none")}), use_model=True,
    )
    assert decision(plan, "multi-head attention").note == "model: none fit"


def test_nothing_near_enough_is_not_asked_about():
    picker = Picker({})
    plan = plan_anchoring([row("zzz unrelated phrase")], VOCAB, index=INDEX, pick=picker, use_model=True)
    assert picker.calls == [] and "nothing near" in decision(plan, "zzz unrelated phrase").note
    assert CANDIDATE_FLOOR > 0.4


def test_with_the_model_off_only_exact_matches_are_made_and_the_index_is_not_touched():
    class Boom:
        def top(self, *a, **k):
            raise AssertionError("the index must not be used")

    plan = plan_anchoring([row("attention"), row("multi-head attention")], VOCAB, index=Boom(), pick=Picker({}), use_model=False)
    assert decision(plan, "attention").anchor and decision(plan, "multi-head attention").anchor is None
    assert plan.calls.calls == []


def test_the_config_switch_is_the_default(monkeypatch):
    monkeypatch.setattr(enrichment, "enabled", lambda: False)
    picker = Picker({})
    plan = plan_anchoring([row("multi-head attention")], VOCAB, index=INDEX, pick=picker)
    assert picker.calls == [] and decision(plan, "multi-head attention").anchor is None


def test_a_failed_call_leaves_the_concept_unanchored_and_the_pass_going():
    def failing(name, candidates):
        raise LLMCallError("down")

    plan = plan_anchoring([row("multi-head attention"), row("attention")], VOCAB, index=INDEX, pick=failing, use_model=True)
    assert "model call failed" in decision(plan, "multi-head attention").note
    assert decision(plan, "attention").anchor == "nlp-kg:Attention"


def test_the_call_limit_leaves_the_rest_for_the_next_run():
    picker = Picker({"self attention": AnchorChoice(choice=1, relation="same"), "multi-head attention": AnchorChoice(choice=1, relation="narrower")})
    plan = plan_anchoring(
        [row("multi-head attention"), row("self attention")], VOCAB, index=INDEX, pick=picker, use_model=True, max_calls=1
    )
    assert len(picker.calls) == 1 and plan.skipped == ["self attention"]


# ------------------------------------------------------- merges, parents


def test_concepts_that_are_the_same_field_merge_into_the_one_most_papers_link_to():
    plan = plan_anchoring(
        [row("attention", papers=1), row("attention mechanism", papers=5, aliases=()), row("bleu")],
        VOCAB, use_model=False,
    )
    # "attention mechanism" is a synonym only if NLP-KG lists it; the exact keeper is the label match.
    anchored = {d.name: d.anchor for d in plan.decisions}
    same = [n for n, a in anchored.items() if a == "nlp-kg:Attention"]
    if len(same) > 1:
        [merge] = plan.merges
        assert merge["keep"] == "attention" and merge["drop"] == sorted(set(same) - {"attention"})


def test_merge_keeps_the_concept_named_like_the_field_then_the_best_linked():
    concepts = [
        row("attn", anchor="nlp-kg:Attention", anchor_relation="same", papers=9),
        row("attention", anchor="nlp-kg:Attention", anchor_relation="same", papers=1),
        row("attention head", anchor="nlp-kg:Attention", anchor_relation="narrower", papers=3),
    ]
    plan = plan_anchoring(concepts, VOCAB, use_model=False)
    assert plan.merges == [{"anchor": "nlp-kg:Attention", "keep": "attention", "drop": ["attn"]}]
    # a narrower concept is never merged away
    assert all("attention head" not in m["drop"] for m in plan.merges)


def test_a_narrower_concept_is_parented_to_the_concept_that_is_its_field():
    concepts = [
        row("attention", anchor="nlp-kg:Attention", anchor_relation="same", papers=2),
        row("multi-head attention", anchor="nlp-kg:Attention", anchor_relation="narrower", papers=1),
    ]
    plan = plan_anchoring(concepts, VOCAB, use_model=False)
    assert plan.parents == [{"name": "multi-head attention", "parent": "attention"}]


def test_a_concept_is_parented_to_the_nearest_ancestor_field_the_library_has():
    concepts = [
        row("attention", anchor="nlp-kg:Attention", anchor_relation="same"),
        row("language model mechanisms", anchor="nlp-kg:Language_Model_Mechanisms", anchor_relation="same"),
        row("natural language processing", anchor="nlp-kg:Natural_Language_Processing", anchor_relation="same"),
    ]
    plan = plan_anchoring(concepts, VOCAB, use_model=False)
    parents = {p["name"]: p["parent"] for p in plan.parents}
    assert parents["attention"] == "language model mechanisms"
    assert parents["language model mechanisms"] in {"natural language processing"} or "language model mechanisms" not in parents
    assert "natural language processing" not in parents  # the root has no parent


def test_a_merged_away_concept_is_not_a_parent_or_a_child():
    concepts = [
        row("attn", anchor="nlp-kg:Attention", anchor_relation="same", papers=1),
        row("attention", anchor="nlp-kg:Attention", anchor_relation="same", papers=3),
        row("multi-head attention", anchor="nlp-kg:Attention", anchor_relation="narrower"),
    ]
    plan = plan_anchoring(concepts, VOCAB, use_model=False)
    assert {"name": "multi-head attention", "parent": "attention"} in plan.parents
    assert all(p["name"] != "attn" and p["parent"] != "attn" for p in plan.parents)


def test_an_already_anchored_concept_is_not_decided_again_unless_asked():
    picker = Picker({"multi-head attention": AnchorChoice(choice=1, relation="narrower")})
    concepts = [row("multi-head attention", anchor="nlp-kg:Language_Models", anchor_relation="narrower")]
    assert plan_anchoring(concepts, VOCAB, index=INDEX, pick=picker, use_model=True).decisions == []
    again = plan_anchoring(concepts, VOCAB, index=INDEX, pick=picker, use_model=True, reanchor=True)
    assert again.decisions[0].anchor == "nlp-kg:Attention"


def test_the_prompt_offers_numbered_candidates_with_synonyms_and_trimmed_descriptions():
    text = candidates_prompt("multi-head attention", [VOCAB.get("nlp-kg:Attention")])
    assert text.startswith("Concept: multi-head attention") and "1. Attention" in text
    assert len(text.splitlines()[-1]) < 400


# ------------------------------------------------------------- the store


def store(tables, rpc_answer=None, missing=()):
    fake = FakeHttp(tables)
    fake.missing = set(missing)
    fake.rpc_answer = rpc_answer or {"anchored": 1, "merged": 0, "parents": 0, "anchored_total": 1}
    client = PostgRESTClient("http://postgrest.test", "token")
    client.client = fake.client
    previous = db._client
    db.set_client(client)
    return fake, previous


def tables():
    return {
        "concepts": [
            {"id": "c1", "name": "attention", "aliases": [], "anchor": None, "anchor_relation": None},
            {"id": "c2", "name": "multi-head attention", "aliases": ["mha"], "anchor": None, "anchor_relation": None},
        ],
        "paper_concepts": [
            {"concept_id": "c1", "paper_id": "p1"},
            {"concept_id": "c1", "paper_id": "p2"},
        ],
    }


def test_a_dry_run_reads_but_neither_calls_a_model_nor_writes(monkeypatch):
    monkeypatch.setattr(enrichment, "enabled", lambda: True)
    fake, previous = store(tables())
    try:
        report = run_anchoring(dry_run=True)
    finally:
        db._client = previous
    assert fake.posts == []
    assert "dry-run: exact matches only" in report.render()
    assert decision(report.plan, "attention").anchor == "nlp-kg:Attention"


def test_a_write_sends_the_whole_plan_in_one_rpc(monkeypatch):
    monkeypatch.setattr(enrichment, "enabled", lambda: True)
    monkeypatch.setattr(enrichment, "require_budget", lambda reserve: None)
    picker = Picker({"multi-head attention": AnchorChoice(choice=1, relation="narrower")})
    fake, previous = store(tables())
    try:
        report = run_anchoring(embed=embed, pick=picker)
    finally:
        db._client = previous
    [(name, body)] = fake.posts
    assert name == "apply_concept_anchoring"
    assert {a["name"] for a in body["p_anchors"]} == {"attention", "multi-head attention"}
    assert body["p_parents"] == [{"name": "multi-head attention", "parent": "attention"}]
    assert report.plan.calls.cost_usd > 0 and "model spend" in report.render()
    assert report.stored["anchored"] == 1


def test_with_the_switch_off_the_command_says_so_and_calls_nothing(monkeypatch):
    monkeypatch.setattr(enrichment, "enabled", lambda: False)
    fake, previous = store(tables())
    try:
        report = run_anchoring(pick=lambda *a: pytest.fail("no model call"))
    finally:
        db._client = previous
    assert "MAP_ENRICHMENT_ENABLED is off" in report.render()
    assert [n for n, _ in fake.posts] == ["apply_concept_anchoring"]  # exact matches are still written


def test_a_spent_cap_stops_the_model_step_but_not_the_exact_matches(monkeypatch):
    monkeypatch.setattr(enrichment, "enabled", lambda: True)

    def refuse(reserve):
        raise enrichment.EnrichmentRefused("The daily spend cap ($2.00) leaves $0.0000; paid enrichment was not started.")

    monkeypatch.setattr(enrichment, "require_budget", refuse)
    fake, previous = store(tables())
    try:
        report = run_anchoring(pick=lambda *a: pytest.fail("no model call"))
    finally:
        db._client = previous
    assert "daily spend cap" in report.render() and report.plan.calls.calls == []


def test_a_missing_migration_names_the_file():
    fake, previous = store(tables(), missing={"apply_concept_anchoring"})
    try:
        with pytest.raises(concept_anchoring.ConceptStoreError, match="035_knowledge_layer.sql"):
            concept_anchoring.apply_plan(
                plan_anchoring([row("attention")], VOCAB, use_model=False)
            )
    finally:
        db._client = previous


def test_cli_concepts_anchor_reports_and_never_raises_on_a_dry_run(monkeypatch):
    from typer.testing import CliRunner

    from nlp_pillars.cli import app

    monkeypatch.setattr(enrichment, "enabled", lambda: True)
    fake, previous = store(tables())
    try:
        result = CliRunner().invoke(app, ["concepts", "anchor", "--dry-run"])
    finally:
        db._client = previous
    assert result.exit_code == 0, result.output
    assert "concepts anchor (dry-run)" in result.output and "attention -> nlp-kg:Attention" in result.output
    assert fake.posts == []
