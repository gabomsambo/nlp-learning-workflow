"""Vocabulary concept extraction, spend accounting, and migration contracts."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import instructor
import pytest
from openai import OpenAI

from nlp_pillars.concept_cost import estimate_backfill_call, price_tokens
from nlp_pillars.concept_extraction import (
    ConceptExtraction,
    ConceptExtractionError,
    normalize_concepts,
    run_backfill,
)

ROOT = Path(__file__).parents[1]
MIGRATION = ROOT / "docs/migrations/024_vocabulary_concepts.sql"


class CompletionEndpoint:
    """Spec for the one Instructor method production calls."""

    def create_with_completion(self, **_kwargs):
        raise NotImplementedError


def _client(result=None, error=None):
    endpoint = Mock(spec=CompletionEndpoint)
    if error is not None:
        endpoint.create_with_completion.side_effect = error
    else:
        raw = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1500, completion_tokens=200)
        )
        endpoint.create_with_completion.return_value = (result, raw)
    client = Mock(spec=["chat"])
    client.chat = Mock(spec=["completions"])
    client.chat.completions = endpoint
    return client


def _note():
    return {
        "id": "note-1",
        "paper_id": "arxiv:1",
        "problem": "Sequence models need reusable context.",
        "method": "Self attention.",
        "findings": ["Attention works."],
        "limitations": [],
        "future_work": [],
        "key_terms": ["Self-Attention"],
        "created_at": "2026-01-01T00:00:00+00:00",
    }


def test_openai_concept_prices_match_the_approved_models():
    assert price_tokens("gpt-4o-mini", 1500, 200) == pytest.approx(0.000345)
    assert price_tokens("gpt-4o", 1500, 200) == pytest.approx(0.00575)
    assert estimate_backfill_call("gpt-4o-mini") == pytest.approx(0.000345)


def test_normalize_reuses_vocabulary_and_bounds_new_names():
    selected, proposed = normalize_concepts(
        ["SELF ATTENTION", "transformer"],
        ["Retrieval-Augmented Generation", "self attention"],
        ["self attention", "transformer"],
    )
    assert selected == ["self attention", "transformer"]
    assert proposed == ["retrieval augmented generation"]


def test_normalize_rejects_an_invented_vocabulary_selection():
    with pytest.raises(ConceptExtractionError, match="outside"):
        normalize_concepts(["invented"], [], ["attention"])


def test_backfill_fixture_uses_spec_client_and_records_actual_cost(monkeypatch):
    from nlp_pillars import concept_extraction as module

    monkeypatch.setattr(module, "latest_notes_without_llm_concepts", lambda: [_note()])
    monkeypatch.setattr(module, "probe_migration", lambda: None)
    monkeypatch.setattr(module, "load_vocabulary", lambda: ["self attention"])
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(default_model="gpt-4o-mini"),
    )
    client = _client(
        ConceptExtraction(
            concepts=["self attention"],
            proposed_concepts=["retrieval augmented generation"],
        )
    )
    writer = Mock(return_value={"concepts": 2})

    report = run_backfill(
        dry_run=False,
        client_factory=lambda: client,
        writer=writer,
    )

    assert report.failed == 0
    assert report.cost_usd == pytest.approx(0.000345)
    writer.assert_called_once_with(
        "arxiv:1",
        ["self attention"],
        ["retrieval augmented generation"],
        "gpt-4o-mini",
    )
    client.chat.completions.create_with_completion.assert_called_once()
    assert "actual cost_usd total 0.000345" in report.render()


def test_backfill_dry_run_counts_and_estimates_without_client_or_write(monkeypatch):
    from nlp_pillars import concept_extraction as module

    monkeypatch.setattr(
        module, "latest_notes_without_llm_concepts", lambda: [_note(), _note()]
    )
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(default_model="gpt-4o-mini"),
    )
    factory = Mock()
    writer = Mock()
    report = run_backfill(
        dry_run=True,
        client_factory=factory,
        writer=writer,
    )
    assert report.candidates == 2
    assert "estimated cost_usd total 0.000690" in report.render()
    assert "No model call. Nothing written." in report.render()
    factory.assert_not_called()
    writer.assert_not_called()


def test_failed_model_call_is_visible_and_leaves_existing_concepts_untouched(
    monkeypatch,
):
    from nlp_pillars import concept_extraction as module

    monkeypatch.setattr(module, "latest_notes_without_llm_concepts", lambda: [_note()])
    monkeypatch.setattr(module, "probe_migration", lambda: None)
    monkeypatch.setattr(module, "load_vocabulary", lambda: ["self attention"])
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(default_model="gpt-4o-mini"),
    )
    writer = Mock()
    report = run_backfill(
        dry_run=False,
        client_factory=lambda: _client(error=RuntimeError("model unavailable")),
        writer=writer,
    )
    assert report.failed == 1
    assert "model unavailable" in report.render()
    assert "previous concepts are untouched" in report.render()
    writer.assert_not_called()


def test_instructor_real_client_exposes_raw_completion_interface():
    """Canary the pinned library surface without making a network request."""
    client = instructor.from_openai(OpenAI(api_key="test-key-not-real"))
    assert callable(client.chat.completions.create_with_completion)


def test_migration_has_atomic_replacement_mastery_grants_and_reload():
    sql = MIGRATION.read_text()
    assert "CREATE OR REPLACE FUNCTION replace_paper_concepts" in sql
    assert "CREATE OR REPLACE FUNCTION concept_mastery" in sql
    assert "paper_concepts.source NOT LIKE 'llm:%'" in sql
    assert "graph_edges.provenance NOT LIKE 'llm:%'" in sql
    assert "19.0 / 81.0" in sql
    assert "-0.5" in sql
    assert "GRANT EXECUTE ON FUNCTION replace_paper_concepts" in sql
    assert "GRANT EXECUTE ON FUNCTION concept_mastery" in sql
    assert "NOTIFY pgrst, 'reload schema'" in sql

