"""The paper map's pure build (nlp_pillars/map_build.py).

One synthetic pillar with a known answer:

- three communities of external works, A (probing), B (graphs) and C
  (fraud), each citing densely inside itself and with vectors around its
  own direction. The pillar's library holds two works in A and one in B; C
  has none of yours, so it is a gap;
- a dated chain in A, a1 2010 -> a2 2012 -> SEED_A1 2015 -> a3 2017 ->
  SEED_A2 2019 -> a4 2021, so the line of development runs through both
  library works in that order;
- HIT, cited by both A seeds and by most of A, citing what the seeds cite,
  and nearest to them in text: the best suggestion by construction;
- a queue covering every badge.

Nothing here touches a database, a model or the network.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from nlp_pillars import map_build as mb
from nlp_pillars.map_build import (
    Citation,
    LibraryPaper,
    MapInputs,
    Pillar,
    QueueItem,
    Work,
)

PILLAR = "model-interpretability"
OTHER = "neural-architectures-language"
DIMS = 12
SEED_A1, SEED_A2, SEED_B = "wA-seed1", "wA-seed2", "wB-seed"
OTHER_LIB = "wB-other-pillar"
HIT = "wA-hit"
DOC = "paper:file:own-notes"
A, B, C = (
    [f"wA{n:02d}" for n in range(20)],
    [f"wB{n:02d}" for n in range(20)],
    [f"wC{n:02d}" for n in range(14)],
)
CHAIN = ["wA-chain1", "wA-chain2", SEED_A1, "wA-chain3", SEED_A2, "wA-chain4"]
CHAIN_YEARS = [2010, 2012, 2015, 2017, 2019, 2021]


def _direction(axis: int, rng: np.random.Generator, spread: float = 0.15) -> list:
    vector = np.zeros(DIMS)
    vector[axis] = 1.0
    vector += rng.normal(0.0, spread, DIMS)
    return [float(value) for value in vector]


def fixture_inputs() -> MapInputs:
    """The pillar described in the module docstring."""
    rng = np.random.default_rng(7)
    picker = random.Random(7)
    works = {}
    citations = []
    vectors = {}

    def work(node, title, year, cites=0):
        works[node] = Work(
            id=node,
            title=title,
            year=year,
            citation_count=cites,
            authors=(f"Author {node}",),
        )

    for group, axis, topic, base_year in (
        (A, 0, "probing classifier", 2005),
        (B, 1, "graph network", 2005),
        (C, 2, "fraud detection", 2005),
    ):
        for position, node in enumerate(group):
            work(
                node,
                f"{topic} study {position}",
                base_year + position,
                cites=10 * (20 - position),
            )
            vectors[node] = _direction(axis, rng)
        for position, node in enumerate(group[1:], start=1):
            for cited in picker.sample(group[:position], min(4, position)):
                citations.append(Citation(node, cited))

    for node, year in zip(CHAIN, CHAIN_YEARS, strict=True):
        work(node, f"probing chain {node}", year, cites=500)
        vectors[node] = _direction(0, rng)
    for earlier, later in zip(CHAIN, CHAIN[1:], strict=False):
        citations.append(
            Citation(
                later,
                earlier,
                is_influential=True,
                intents=("methodology",),
                contexts=(f"{later} builds directly on {earlier}.",),
            )
        )
    # The A seeds cite the first half of community A and the second half
    # cites them, so all of A is their neighbourhood. Same for B below.
    for seed in (SEED_A1, SEED_A2):
        for cited in A[:8]:
            citations.append(Citation(seed, cited, intents=("background",)))
    for node in A[8:]:
        citations.append(Citation(node, SEED_A2))
    # B's seed and the other pillar's library work live in B.
    work(SEED_B, "graph network for transactions", 2020, cites=40)
    work(OTHER_LIB, "graph network library paper", 2018, cites=90)
    vectors[SEED_B] = _direction(1, rng)
    vectors[OTHER_LIB] = _direction(1, rng)
    for cited in B[:10]:
        citations.append(Citation(SEED_B, cited))
        citations.append(Citation(OTHER_LIB, cited))
    for node in B[10:]:
        citations.append(Citation(node, SEED_B))
    citations.append(Citation(SEED_B, OTHER_LIB, intents=("methodology",)))
    # C cites SEED_A1 once each (that is how it is in scope) and itself densely.
    for node in C:
        citations.append(Citation(node, SEED_A1))
    # HIT: cited by both A seeds and by most of A, shares the seeds'
    # references, and is nearest to them in text.
    work(HIT, "probing classifier with control tasks", 2008, cites=300)
    vectors[HIT] = [
        float(value)
        for value in (np.asarray(vectors[SEED_A1]) + np.asarray(vectors[SEED_A2])) / 2
    ]
    for seed in (SEED_A1, SEED_A2):
        citations.append(
            Citation(
                seed,
                HIT,
                is_influential=True,
                intents=("methodology",),
                contexts=("We follow the control-task design of this work.",),
            )
        )
    for cited in A[:3]:
        citations.append(Citation(HIT, cited))
    for node in A[3:]:
        citations.append(Citation(node, HIT))

    library = [
        LibraryPaper(
            paper_id="arxiv:a1",
            pillar_id=PILLAR,
            work_id=SEED_A1,
            title="probing chain seed 1",
        ),
        LibraryPaper(
            paper_id="arxiv:a2",
            pillar_id=PILLAR,
            work_id=SEED_A2,
            title="probing chain seed 2",
        ),
        LibraryPaper(
            paper_id="arxiv:b",
            pillar_id=PILLAR,
            work_id=SEED_B,
            title="graph network for transactions",
        ),
        LibraryPaper(
            paper_id="arxiv:o",
            pillar_id=OTHER,
            work_id=OTHER_LIB,
            title="graph network library paper",
        ),
        LibraryPaper(
            paper_id="file:own-notes",
            pillar_id=PILLAR,
            title="my probing notes",
            year=2024,
        ),
    ]
    vectors[DOC] = _direction(0, rng)
    # Queue: every badge once.
    anchor_axis = 5
    vectors["queue:q-new"] = _direction(anchor_axis, rng, 0.05)
    vectors["queue:q-off"] = _direction(9, rng, 0.05)
    vectors["queue:q-dup"] = list(vectors[SEED_A2])
    queue = [
        QueueItem(
            "q-hit", "s2:hit", "probing classifier with control tasks", work_id=HIT
        ),
        QueueItem("q-new", "arxiv:new", "mechanistic circuits in transformers"),
        QueueItem("q-off", "arxiv:off", "bottlenose dolphin acoustics"),
        QueueItem("q-dup", "arxiv:dup", f"probing chain {SEED_A2}"),
        QueueItem(
            "q-mine",
            "arxiv:a1",
            "probing chain seed 1",
            processed=True,
            work_id=SEED_A1,
        ),
        QueueItem("q-none", "arxiv:none", "a title nothing embedded"),
    ]
    anchors = [_direction(anchor_axis, rng, 0.02), _direction(anchor_axis, rng, 0.02)]
    return MapInputs(
        pillar=Pillar(
            PILLAR,
            "Model interpretability",
            "Learn to analyse models",
            ("Mechanistic interpretability",),
        ),
        library=library,
        works=works,
        citations=citations,
        vectors=vectors,
        model_id="test-model",
        queue=queue,
        anchors=anchors,
    )


@pytest.fixture(scope="module")
def built():
    return mb.build_map(fixture_inputs())


def _nodes(result):
    return {node["id"]: node for node in result.payload["nodes"]}


# ---------------------------------------------------------------- the build


def test_scope_holds_every_community_and_the_document(built):
    nodes = _nodes(built)
    for group in (A, B, C):
        assert set(group) <= set(nodes)
    assert {SEED_A1, SEED_A2, SEED_B, HIT, DOC, OTHER_LIB} <= set(nodes)
    assert nodes[DOC]["kind"] == mb.KIND_DOCUMENT
    assert nodes[SEED_A1]["kind"] == mb.KIND_LIBRARY
    assert nodes[OTHER_LIB]["pillars"] == [OTHER]
    assert built.payload["clustered"] is True


def _purity(members, topics):
    counts = {}
    for node in members:
        counts[topics[node]] = counts.get(topics[node], 0) + 1
    topic, count = max(counts.items(), key=lambda item: item[1])
    return topic, count / len(members)


def test_leiden_recovers_the_planted_communities(built):
    topics = {node["id"]: node["topic"] for node in built.payload["nodes"]}
    found = [_purity(group, topics) for group in (A, B, C)]
    assert all(purity >= 0.9 for _, purity in found), found
    assert len({topic for topic, _ in found}) == 3
    regions = {row["id"] for row in built.payload["communities"]["regions"]}
    assert all(
        row["region"] in regions for row in built.payload["communities"]["topics"]
    )


def test_a_community_with_none_of_yours_is_a_gap(built):
    topics = {node["id"]: node["topic"] for node in built.payload["nodes"]}
    gap_topic = _purity(C, topics)[0]
    gaps = [
        insight for insight in built.payload["insights"] if insight["kind"] == "gap"
    ]
    assert [gap["topic"] for gap in gaps] == [gap_topic]
    assert "none of them are yours" in gaps[0]["text"]


def test_the_line_of_development_runs_through_both_library_works_in_order(built):
    lineage = built.payload["lineage"]
    assert lineage["kind"] == "key_node"
    first = lineage["paths"][0]
    assert first["library"][:2] == [SEED_A1, SEED_A2] or set(first["library"]) >= {
        SEED_A1,
        SEED_A2,
    }
    path = first["nodes"]
    assert path.index(SEED_A1) < path.index(SEED_A2)
    years = [_nodes(built)[node]["year"] for node in path]
    assert years == sorted(years)
    assert set(first["condensed"]) >= {SEED_A1, SEED_A2}
    assert all(node in path for node in first["condensed"])
    arcs = {(row["cited"], row["citing"]) for row in built.payload["citations"]}
    assert all(
        (left, right) in arcs for left, right in zip(path, path[1:], strict=False)
    )


def test_spc_weights_ride_on_the_citation_arcs(built):
    weights = [row["spc"] for row in built.payload["citations"]]
    assert max(weights) == 1.0
    assert all(0.0 <= weight <= 1.0 for weight in weights)


def test_the_best_connected_work_is_the_top_suggestion(built):
    suggestions = built.payload["suggestions"]
    assert suggestions["pillar"][0]["id"] == HIT
    top = suggestions["pillar"][0]
    assert top["cited_by_yours"] == [SEED_A1, SEED_A2]
    assert top["direction"] == "foundation"
    a_topic = _nodes(built)[SEED_A1]["topic"]
    a_group = next(
        group for group in suggestions["communities"] if group["topic"] == a_topic
    )
    # Your own document joins A by text alone; it seeds A's suggestions too.
    assert set(a_group["seeds"]) == {SEED_A1, SEED_A2, DOC}
    assert a_group["items"][0]["id"] == HIT


def test_library_works_are_never_suggested(built):
    library = {SEED_A1, SEED_A2, SEED_B, OTHER_LIB, DOC}
    suggestions = built.payload["suggestions"]
    for group in [
        suggestions["pillar"],
        *[group["items"] for group in suggestions["communities"]],
    ]:
        assert not library & {item["id"] for item in group}


def test_every_queue_badge(built):
    verdicts = {row["queue_id"]: row["verdict"] for row in built.payload["queue"]}
    assert verdicts == {
        "q-hit": mb.VERDICT_STRONG,
        "q-new": mb.VERDICT_NEW_DIRECTION,
        "q-off": mb.VERDICT_OFF_TOPIC,
        "q-dup": mb.VERDICT_DUPLICATE,
        "q-mine": mb.VERDICT_ALREADY_YOURS,
        "q-none": mb.VERDICT_UNSCORED,
    }
    order = [row["verdict"] for row in built.payload["queue"]]
    assert order == sorted(order, key=mb.VERDICT_ORDER.index)
    dup = next(row for row in built.payload["queue"] if row["queue_id"] == "q-dup")
    assert dup["duplicate_of"] == SEED_A2


def test_without_the_pillar_anchor_far_papers_are_not_called_off_topic():
    inputs = fixture_inputs()
    inputs.anchors = []
    result = mb.build_map(inputs)
    verdicts = {row["queue_id"]: row["verdict"] for row in result.payload["queue"]}
    assert verdicts["q-new"] == verdicts["q-off"] == mb.VERDICT_FAR
    assert any("goal is not embedded" in warning for warning in result.warnings)


def test_insights_name_the_pair_the_bridge_and_the_furthest(built):
    kinds = {insight["kind"]: insight for insight in built.payload["insights"]}
    assert set(kinds["most_aligned"]["nodes"]) == {SEED_A1, SEED_A2}
    assert kinds["furthest"]["nodes"] == [SEED_B]
    assert kinds["cross_pillar"]["nodes"] == [SEED_B, OTHER_LIB]
    assert OTHER in kinds["cross_pillar"]["text"]


def test_evidence_explains_a_seed_and_its_suggestion(built):
    rows = {(row["a"], row["b"]): row for row in built.payload["evidence"]}
    row = rows[tuple(sorted((SEED_A1, HIT)))]
    assert row["evidence"]["cites"]["influential"] is True
    assert "builds on" in row["sentence"]
    assert "We follow the control-task design" in row["sentence"]
    assert "share" in row["sentence"]


def test_nodes_carry_importance_and_a_position(built):
    nodes = _nodes(built)
    assert nodes[A[0]]["local_citations"] > 0
    assert all(
        -1.0001 <= node["x"] <= 1.0001 and -1.0001 <= node["y"] <= 1.0001
        for node in nodes.values()
    )
    assert any(node["on_lineage"] for node in nodes.values())
    assert nodes[HIT]["cohort_percentile"] is not None


# ---------------------------------------------------------------- determinism


def _shuffled(inputs: MapInputs, seed: int) -> MapInputs:
    rng = random.Random(seed)
    data = inputs.to_dict()
    for key in ("library", "works", "citations", "queue"):
        rng.shuffle(data[key])
    vectors = list(data["vectors"].items())
    rng.shuffle(vectors)
    data["vectors"] = dict(vectors)
    return MapInputs.from_dict(json.loads(json.dumps(data)))


def test_same_input_gives_the_same_snapshot_hash(built):
    again = mb.build_map(fixture_inputs())
    assert again.payload_hash == built.payload_hash
    assert again.params_hash == built.params_hash


def test_input_order_does_not_change_the_snapshot(built):
    for seed in (1, 2):
        assert (
            mb.build_map(_shuffled(fixture_inputs(), seed)).payload_hash
            == built.payload_hash
        )


def test_the_hash_is_over_the_canonical_payload(built):
    assert built.payload_hash == mb.sha256_of(json.loads(json.dumps(built.payload)))


def test_a_constant_change_changes_params_hash(monkeypatch, built):
    monkeypatch.setattr(mb, "RRF_K", 60)
    assert mb.params_hash() != built.params_hash
    assert "RRF_K" in mb.map_params()


def test_inputs_round_trip_through_json():
    inputs = fixture_inputs()
    again = MapInputs.from_dict(json.loads(json.dumps(inputs.to_dict())))
    assert again.to_dict() == inputs.to_dict()


def test_a_warm_start_keeps_the_map_where_it_was(built):
    previous = {node["id"]: (node["x"], node["y"]) for node in built.payload["nodes"]}
    inputs = fixture_inputs()
    inputs.previous_positions = previous
    warm = _nodes(mb.build_map(inputs))
    moved = np.mean(
        [np.hypot(warm[n]["x"] - x, warm[n]["y"] - y) for n, (x, y) in previous.items()]
    )
    assert moved < 0.2


def test_an_unchanged_graph_keeps_its_layout_and_its_hash(built):
    inputs = fixture_inputs()
    inputs.previous_positions = {
        node["id"]: (node["x"], node["y"]) for node in built.payload["nodes"]
    }
    inputs.previous_layout_key = built.payload["layout"]["key"]
    again = mb.build_map(inputs)
    assert again.layout["reused"] is True
    assert again.payload_hash == built.payload_hash
    # A changed graph is laid out again, starting from where it was.
    inputs.citations = inputs.citations[:-1]
    changed = mb.build_map(inputs)
    assert changed.layout["reused"] is False
    assert changed.layout["warm_started"] == len(changed.payload["nodes"])


def test_progress_reports_every_stage_in_order():
    seen = []
    mb.build_map(
        fixture_inputs(),
        progress=lambda stage, status, detail: seen.append((stage, status)),
    )
    stages = [stage for stage, status in seen if status == mb.STATUS_COMPLETED]
    assert stages == [
        mb.STAGE_GRAPH,
        mb.STAGE_COMMUNITIES,
        mb.STAGE_LINEAGE,
        mb.STAGE_LAYOUT,
        mb.STAGE_SUGGEST,
    ]


def test_a_progress_callback_can_stop_the_build():
    class Stop(Exception):
        pass

    def progress(stage, status, detail):
        if stage == mb.STAGE_LINEAGE:
            raise Stop

    with pytest.raises(Stop):
        mb.build_map(fixture_inputs(), progress=progress)


# ----------------------------------------------------------- small and empty


def test_a_tiny_pillar_is_one_community_and_says_it_is_sparse():
    inputs = MapInputs(
        pillar=Pillar("formal-linguistics-nlp"),
        library=[
            LibraryPaper("p1", "formal-linguistics-nlp", "w1", "Minimalist grammars")
        ],
        works={
            "w1": Work("w1", "Minimalist grammars", 2011),
            "w0": Work("w0", "Aspects of syntax", 1971),
            "w2": Work("w2", "Minimalist semantics", 2015),
        },
        citations=[Citation("w1", "w0"), Citation("w2", "w1")],
    )
    result = mb.build_map(inputs)
    assert result.payload["clustered"] is False
    assert {node["topic"] for node in result.payload["nodes"]} == {0}
    assert [i["kind"] for i in result.payload["insights"]] == ["sparse"]
    assert result.payload["lineage"]["paths"][0]["nodes"] == ["w0", "w1", "w2"]
    assert any("No title+abstract vectors" in warning for warning in result.warnings)


def test_a_pillar_with_no_library_still_scores_its_queue_by_the_anchor():
    inputs = fixture_inputs()
    inputs.library = [paper for paper in inputs.library if paper.pillar_id != PILLAR]
    result = mb.build_map(inputs)
    verdicts = {row["queue_id"]: row["verdict"] for row in result.payload["queue"]}
    assert verdicts["q-new"] == mb.VERDICT_NEW_DIRECTION
    assert verdicts["q-off"] == mb.VERDICT_OFF_TOPIC
    assert result.payload["suggestions"]["pillar"] == []


# ------------------------------------------------------------- pure helpers


def test_search_path_counts_on_a_known_dag():
    import networkx as nx

    dag = nx.DiGraph([("s1", "x"), ("s2", "x"), ("x", "t1"), ("x", "t2"), ("s1", "t1")])
    assert mb.search_path_counts(dag) == {
        ("s1", "t1"): 1,
        ("s1", "x"): 2,
        ("s2", "x"): 2,
        ("x", "t1"): 2,
        ("x", "t2"): 2,
    }


def test_backward_citations_and_cycles_are_dropped():
    index = mb.CitationIndex(
        [Citation("old", "new"), Citation("a", "b"), Citation("b", "a")]
    )
    dag, backwards, cycles = mb.citation_dag(
        ["old", "new", "a", "b"],
        index,
        {"old": 2000, "new": 2010, "a": 2015, "b": 2015},
    )
    assert backwards == 1 and cycles == 1
    assert dag.number_of_edges() == 1


def test_rank_positions_puts_unscored_last_and_breaks_ties_by_id():
    ranks = mb.rank_positions({"b": 1.0, "a": 1.0, "c": 2.0}, ["a", "b", "c", "z", "y"])
    assert ranks == {"c": 1, "a": 2, "b": 3, "y": 4, "z": 5}


def test_rrf_fuses_reciprocal_ranks():
    fused = mb.rrf_fuse({"x": 1, "y": 2}, {"x": 2, "y": 1}, k=10)
    assert fused["x"] == pytest.approx(1 / 11 + 1 / 12)
    assert fused["x"] == pytest.approx(fused["y"])


def test_cohort_percentile_widens_a_thin_year():
    works = {f"w{n}": Work(f"w{n}", year=2020, citation_count=n) for n in range(5)}
    works["lonely"] = Work("lonely", year=1990, citation_count=7)
    out = mb.cohort_percentiles(sorted(works), works)
    assert out["w0"] == 10.0 and out["w4"] == 90.0
    assert out["lonely"] == 50.0  # its own cohort of one, after widening finds nobody


@pytest.mark.parametrize(
    "kwargs,verdict",
    [
        (dict(in_library=True), mb.VERDICT_ALREADY_YOURS),
        (dict(duplicate_of="w"), mb.VERDICT_DUPLICATE),
        (dict(scored=False), mb.VERDICT_UNSCORED),
        (dict(rank_fraction=0.05), mb.VERDICT_STRONG),
        (dict(rank_fraction=0.2), mb.VERDICT_RELATED),
        (dict(rank_fraction=0.5, anchor=0.8), mb.VERDICT_NEW_DIRECTION),
        (dict(rank_fraction=0.5, anchor=0.5), mb.VERDICT_OFF_TOPIC),
        (dict(rank_fraction=0.01, has_seeds=False, anchor=0.5), mb.VERDICT_OFF_TOPIC),
        (dict(rank_fraction=0.5, anchor=None), mb.VERDICT_FAR),
    ],
)
def test_queue_verdict_rules(kwargs, verdict):
    base = dict(
        in_library=False,
        duplicate_of=None,
        scored=True,
        rank_fraction=None,
        anchor=None,
        anchor_cut=0.677,
        has_seeds=True,
    )
    assert mb.queue_verdict(**{**base, **kwargs}) == verdict


def test_evidence_sentence_is_a_template_over_counts():
    evidence = {
        "cites": {
            "citing": "p",
            "cited": "h",
            "influential": True,
            "intents": ["background", "methodology"],
            "context": "Hewitt and Liang likely had an intuition.",
        },
        "shared_references": 3,
        "cocited": 1,
        "text_percentile": 99.5,
        "same_topic": 2,
    }
    sentence = mb.evidence_sentence(
        evidence, {"p": "Pimentel 2020", "h": "Hewitt 2019"}.get
    )
    assert sentence == (
        "Pimentel 2020 cites Hewitt 2019 (background, builds on, influential): "
        "“Hewitt and Liang likely had an intuition.”; they share 3 references; "
        "1 paper cite both; their abstracts are in the top 0.5% most similar pairs on this map; "
        "both are in topic 2."
    )
    assert mb.evidence_sentence({}, str) == ""


def test_canonical_community_labels_do_not_depend_on_leiden_ids():
    assert mb.canonical_communities({"a": 7, "b": 7, "c": 3, "d": 3, "e": 1}) == {
        "a": 0,
        "b": 0,
        "c": 1,
        "d": 1,
        "e": 2,
    }


def test_titles_match_needs_most_words_in_common():
    assert mb.titles_match("Attention Is All You Need", "Attention is all you need.")
    assert not mb.titles_match(
        "Attention Is All You Need", "Attention is not explanation"
    )
