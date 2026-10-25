"""Vocabulary concept extraction, spend accounting, and migration contracts."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from anthropic import Anthropic
from anthropic.resources.messages import Messages

from nlp_pillars.concept_cost import estimate_backfill_call, price_tokens
from nlp_pillars.concept_extraction import (
    ConceptExtraction,
    ConceptExtractionError,
    normalize_concepts,
    run_backfill,
)

ROOT = Path(__file__).parents[1]
MIGRATION = ROOT / "docs/migrations/024_vocabulary_concepts.sql"


def _parsed(value, *, input_tokens=1500, output_tokens=200):
    return SimpleNamespace(
        parsed_output=value,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        stop_reason="end_turn",
    )


def _client(result=None, error=None):
    """A mock Anthropic client. Spec'd, so a missing method is a failure."""
    messages = Mock(spec=Messages)
    if error is not None:
        messages.parse.side_effect = error
    else:
        messages.parse.return_value = _parsed(result)
    client = Mock(spec=Anthropic)
    client.messages = messages
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


def test_the_prices_cover_every_model_the_app_calls():
    """An unpriced model refuses to spend, so the table has to be complete."""
    assert price_tokens("claude-haiku-5-5", 1500, 200) == pytest.approx(0.000250)
    assert price_tokens("gpt-4o-mini", 1500, 200) == pytest.approx(0.000345)
    assert price_tokens("gpt-4o", 1500, 200) == pytest.approx(0.00575)
    assert price_tokens("deepseek-flash", 1500, 200) == pytest.approx(0.00069)
    assert estimate_backfill_call("claude-haiku-5-5") == pytest.approx(0.000250)


def test_an_unpriced_model_still_refuses():
    with pytest.raises(ValueError, match="No approved price"):
        price_tokens("some-model-nobody-priced", 1500, 200)


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
    assert report.cost_usd == pytest.approx(0.000250)
    writer.assert_called_once_with(
        "arxiv:1",
        ["self attention"],
        ["retrieval augmented generation"],
        "claude-haiku-5-5",
    )
    client.messages.parse.assert_called_once()
    # No sampling parameters: Claude rejects them.
    kwargs = client.messages.parse.call_args.kwargs
    for banned in ("temperature", "top_p", "top_k", "tool_choice"):
        assert banned not in kwargs
    assert "actual cost_usd total 0.000250" in report.render()


def test_backfill_dry_run_counts_and_estimates_without_client_or_write(monkeypatch):
    from nlp_pillars import concept_extraction as module

    monkeypatch.setattr(
        module, "latest_notes_without_llm_concepts", lambda: [_note(), _note()]
    )
    factory = Mock()
    writer = Mock()
    report = run_backfill(
        dry_run=True,
        client_factory=factory,
        writer=writer,
    )
    assert report.candidates == 2
    assert "estimated cost_usd total 0.000500" in report.render()
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


def test_the_anthropic_client_can_parse_structured_output():
    """Canary the pinned library surface without making a network request."""
    from anthropic import Anthropic as _Anthropic

    client = _Anthropic(api_key="test-key-not-real")
    assert callable(client.messages.parse)


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

