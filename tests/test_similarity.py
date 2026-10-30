"""Percentile floors, top-k and hubness. Pure numpy: no database, model or network."""

import numpy as np
import pytest

from nlp_pillars.similarity import (
    FLOOR_ABSOLUTE,
    FLOOR_PERCENTILE,
    HUBNESS_SKEWNESS_LIMIT,
    MIN_PERCENTILE_SCOPE,
    RELATED_PERCENTILE,
    Hubness,
    SimilarityHealth,
    k_occurrence_skewness,
    scope_floor,
    top_k_neighbours,
    unit_matrix,
)


def _clustered(rng, size, spread, dims=32):
    """Vectors scattered ``spread`` around one direction: a scope where every pair is similar."""
    base = rng.normal(size=dims)
    return unit_matrix(base + spread * rng.normal(size=(size, dims)))


def test_unit_matrix_normalises_rows_and_leaves_a_zero_row_alone():
    unit = unit_matrix([[3.0, 4.0], [0.0, 0.0]])
    assert unit[0].tolist() == pytest.approx([0.6, 0.8])
    assert unit[1].tolist() == [0.0, 0.0]


def test_a_small_scope_keeps_the_absolute_floor():
    unit = unit_matrix(np.eye(MIN_PERCENTILE_SCOPE - 1))
    floor = scope_floor(unit, 0.70)
    assert (floor.value, floor.source, floor.pairs) == (0.70, FLOOR_ABSOLUTE, 0)
    assert floor.scope_size == MIN_PERCENTILE_SCOPE - 1


def test_the_floor_is_the_percentile_of_every_pair():
    rng = np.random.default_rng(1)
    unit = _clustered(rng, 60, spread=0.5)
    floor = scope_floor(unit, 0.70)
    scores = (unit @ unit.T)[np.triu_indices(60, k=1)]
    assert floor.source == FLOOR_PERCENTILE
    assert floor.pairs == 60 * 59 // 2
    assert floor.value == pytest.approx(np.percentile(scores, RELATED_PERCENTILE))
    # about 3% of pairs clear it, whatever the absolute level of the scope
    share = (scores >= floor.value).mean()
    assert 0.02 <= share <= 0.04


def test_the_floor_moves_with_the_scope_not_with_a_constant():
    rng = np.random.default_rng(2)
    tight = scope_floor(_clustered(rng, 60, spread=0.3), 0.70).value
    loose = scope_floor(_clustered(rng, 60, spread=2.0), 0.70).value
    assert tight > 0.70 > loose


def test_a_huge_scope_samples_pairs_and_gets_the_same_floor_twice(monkeypatch):
    monkeypatch.setattr("nlp_pillars.similarity.EXACT_PAIRS_MAX_VECTORS", 40)
    monkeypatch.setattr("nlp_pillars.similarity.SAMPLED_PAIRS", 5000)
    unit = _clustered(np.random.default_rng(3), 80, spread=1.0)
    first = scope_floor(unit, 0.70)
    assert first.sampled and first.pairs == 5000
    assert scope_floor(unit, 0.70) == first
    exact = np.percentile((unit @ unit.T)[np.triu_indices(80, k=1)], RELATED_PERCENTILE)
    assert first.value == pytest.approx(exact, abs=0.05)


def test_top_k_is_ranked_floored_and_never_returns_the_vector_itself():
    unit = unit_matrix([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]])
    names = ["a", "b", "c", "d"]
    found = top_k_neighbours(unit, [0, 1, 2, 3], k=2, floor=0.5, names=names)
    assert [(round(score, 3), other) for score, other in found[0]] == [(0.8, 1)]
    assert [(round(score, 3), other) for score, other in found[1]] == [(0.8, 0), (0.6, 2)]
    assert found[3] == []
    assert all(row != other for row, hits in found.items() for _, other in hits)


def test_ties_break_by_name_not_by_input_order():
    unit = unit_matrix([[1.0, 0.0]] * 4)
    names = ["z", "m", "a", "q"]
    found = top_k_neighbours(unit, [0, 1, 2, 3], k=2, floor=0.0, names=names)
    assert [other for _, other in found[0]] == [2, 1]  # a, then m


def test_a_scope_of_uniformly_high_cosines_yields_at_most_k_neighbours_each():
    """The regression this PR exists for.

    Every pair here is above 0.9, so any absolute floor passes all of them. The
    rank cut has to bound the degree, whatever the floor says.
    """
    rng = np.random.default_rng(4)
    unit = _clustered(rng, 80, spread=0.05)
    scores = (unit @ unit.T)[np.triu_indices(80, k=1)]
    assert scores.min() > 0.9
    names = [f"p{index}" for index in range(80)]
    k = 5
    for floor in (0.0, 0.70):
        found = top_k_neighbours(unit, list(range(80)), k, floor, names)
        assert [len(hits) for hits in found.values()] == [k] * 80  # full, and no more
    found = top_k_neighbours(unit, list(range(80)), k, scope_floor(unit, 0.70).value, names)
    assert 0 < max(len(hits) for hits in found.values()) <= k


def test_skewness_of_an_even_scope_is_small_and_of_a_hub_scope_is_large():
    rng = np.random.default_rng(5)
    even = unit_matrix(rng.normal(size=(120, 64)))
    assert abs(k_occurrence_skewness(even).skewness) < HUBNESS_SKEWNESS_LIMIT

    # One direction every other vector leans towards: a hub that is nearest to
    # everything, and papers that are nobody's neighbour.
    spokes = rng.normal(size=(120, 64))
    spokes[0] = 0.0
    hub = unit_matrix(spokes + 3.0 * np.eye(64)[0])
    hub[0] = np.eye(64)[0]
    result = k_occurrence_skewness(hub)
    assert result.high
    assert result.max_occurrence > 40
    assert result.never_a_neighbour > 0


def test_skewness_counts_each_nearest_neighbour_once():
    # a, b, c in a line: b is a's and c's nearest, a is b's, b is a's (k=1)
    unit = unit_matrix([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    result = k_occurrence_skewness(unit, k=1)
    assert result.k == 1 and result.scope_size == 3
    assert result.max_occurrence == 2
    assert result.never_a_neighbour == 1


def test_skewness_is_none_when_it_is_undefined():
    assert k_occurrence_skewness(unit_matrix([[1.0, 0.0]])).skewness is None
    # every count equal (a ring): no skew to report
    ring = unit_matrix([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)])
    assert k_occurrence_skewness(ring, k=2).skewness is None


def test_health_line_names_the_floor_source_and_the_verdict():
    absolute = SimilarityHealth(
        scope_floor(unit_matrix(np.eye(5)), 0.70),
        Hubness(k=4, scope_size=5, skewness=0.2, max_occurrence=5, never_a_neighbour=0),
    )
    assert "floor 0.7000 (absolute: 5 vectors is under 30" in absolute.render()
    assert "skewness 0.20 at k=4 (ok)" in absolute.render()

    unit = _clustered(np.random.default_rng(6), 40, spread=1.0)
    percentile = SimilarityHealth(
        scope_floor(unit, 0.70),
        Hubness(k=10, scope_size=40, skewness=2.5, max_occurrence=33, never_a_neighbour=12),
    )
    line = percentile.render()
    assert f"p{RELATED_PERCENTILE:g} of 780 pairs" in line
    assert "40 vectors" in line
    assert "above 1.4, consider a hubness correction" in line
    assert "12 never a neighbour, max 33" in line
