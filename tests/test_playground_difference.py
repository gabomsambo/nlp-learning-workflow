"""The "Explain the difference" call behind a Playground Before / After.

The route is covered in tests/webui/test_playground_api.py. Nothing here calls
a model: the Anthropic client is a stand-in that records the request.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nlp_pillars.llm_client import FORBIDDEN_PARAMETERS
from nlp_pillars.llm_routes import PLAYGROUND_DIFF_ROUTE, route_for
from nlp_pillars.playground import difference
from nlp_pillars.playground.difference import (
    CODE_CHARS,
    OUTPUT_CHARS,
    SYSTEM_PROMPT,
    DifferenceDraft,
    build_prompt,
    clip,
    explain_difference,
    tidy,
)

ROOT = Path(__file__).resolve().parents[1]

FIELDS = dict(
    before_code="model = IsolationForest(n_estimators=100, random_state=0)",
    after_code="model = IsolationForest(n_estimators=10, random_state=0)",
    before_output="flagged: 120",
    after_output="flagged: 134",
    explanation="Fits the forest and counts flagged accounts.",
    expect="Fewer trees make the scores noisier.",
)


def _client(answer="Fewer trees changed the count from 120 to 134.", stop="end_turn"):
    client = MagicMock()
    client.messages.parse.return_value = SimpleNamespace(
        parsed_output=DifferenceDraft(explanation=answer) if answer is not None else None,
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=400, output_tokens=40,
                              cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )
    return client


def test_route_is_haiku_by_default_and_env_overridable(monkeypatch):
    assert route_for("playground_diff") is PLAYGROUND_DIFF_ROUTE
    assert PLAYGROUND_DIFF_ROUTE.resolve().model == "claude-haiku-5-5"
    assert PLAYGROUND_DIFF_ROUTE.model_attrs == ("llm_playground_diff_model",)
    from nlp_pillars.config import get_settings

    monkeypatch.setattr(get_settings(), "llm_playground_diff_model", "claude-sonnet-5-5")
    assert PLAYGROUND_DIFF_ROUTE.resolve().model == "claude-sonnet-5-5"


def test_prompt_copy_in_docs_matches_the_code():
    assert SYSTEM_PROMPT in (ROOT / "docs/PROMPTS.md").read_text()


def test_prompt_states_the_grounding_rules():
    for phrase in (
        "Use only the code and the two outputs",
        "Quote numbers only as they appear in the outputs",
        "say what changed and that the outputs do not show the reason",
        "two or three short, plain sentences",
    ):
        assert phrase in SYSTEM_PROMPT


def test_user_message_carries_both_sides_in_delimited_blocks():
    text = build_prompt(**FIELDS)
    order = [
        "What the cell does: Fits the forest",
        "What the change was expected to do: Fewer trees",
        "=== BEFORE: code ===", "n_estimators=100",
        "=== BEFORE: output ===", "flagged: 120",
        "=== AFTER: code ===", "n_estimators=10,",
        "=== AFTER: output ===", "flagged: 134",
        "=== END ===",
    ]
    positions = [text.index(part) for part in order]
    assert positions == sorted(positions)


def test_empty_parts_are_named_not_blank():
    text = build_prompt(before_code="x", after_code="y", before_output="", after_output="  ")
    assert text.count("(no output)") == 2
    assert "What the cell does" not in text


def test_long_outputs_are_cut_in_the_middle_and_marked():
    long = "head " + "x" * 10_000 + " tail"
    cut = clip(long, OUTPUT_CHARS)
    assert cut.startswith("head ") and cut.endswith(" tail")
    assert "[cut:" in cut
    assert len(cut) < OUTPUT_CHARS + 60
    assert clip("short", OUTPUT_CHARS) == "short"
    text = build_prompt(before_code="c" * (CODE_CHARS * 2), after_code="d",
                        before_output=long, after_output="ok")
    assert len(text) < CODE_CHARS + OUTPUT_CHARS + 1_000


def test_tidy_makes_one_short_paragraph():
    assert tidy("- The count\n went up.\n\n") == "The count went up."
    long = ("The count went up. " * 100).strip()
    out = tidy(long)
    assert len(out) <= difference.MAX_ANSWER_CHARS and out.endswith(".")


def test_one_call_on_haiku_with_no_sampling_parameters():
    client = _client()
    result = explain_difference(client=client, **FIELDS)
    assert result.explanation == "Fewer trees changed the count from 120 to 134."
    assert result.model == "claude-haiku-5-5"
    assert result.cost_usd > 0
    assert client.messages.parse.call_count == 1
    kwargs = client.messages.parse.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-5-5"
    assert kwargs["system"] == SYSTEM_PROMPT
    assert kwargs["output_format"] is DifferenceDraft
    assert kwargs["thinking"] == {"type": "disabled"}
    assert kwargs["max_tokens"] == difference.MAX_TOKENS
    for name in FORBIDDEN_PARAMETERS:
        assert name not in kwargs


def test_a_failed_call_is_not_retried_on_a_dearer_model():
    client = _client(answer=None, stop="refusal")
    with pytest.raises(difference.DifferenceError) as excinfo:
        explain_difference(client=client, **FIELDS)
    assert not isinstance(excinfo.value, difference.DifferenceUnavailable)
    assert client.messages.parse.call_count == 1


def test_an_empty_answer_is_an_error():
    with pytest.raises(difference.DifferenceError):
        explain_difference(client=_client(answer="   "), **FIELDS)


def test_missing_key_is_unavailable(monkeypatch):
    from nlp_pillars.config import get_settings

    monkeypatch.setattr(get_settings(), "anthropic_api_key", None)
    with pytest.raises(difference.DifferenceUnavailable) as excinfo:
        explain_difference(**FIELDS)
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)
