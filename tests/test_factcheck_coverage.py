"""Extraction coverage: a host line with a figure must not silently escape."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from anthropic import Anthropic
from anthropic.resources.messages import Messages

from nlp_pillars.factcheck.claude import Spend
from nlp_pillars.factcheck.constants import PROMPT_VERSION
from nlp_pillars.factcheck.coverage import (
    WindowStat,
    describe,
    has_nontrivial_number,
    number_bearing_lines,
    substantive_host_lines,
)
from nlp_pillars.factcheck.extract import extract_claims
from nlp_pillars.factcheck.prompts import VERIFY_INSTRUCTIONS
from nlp_pillars.factcheck.schemas import ExtractedClaim, ExtractionBatch

MODEL = "claude-haiku-5-5"


def _claim(line, span, text="A claim."):
    return ExtractedClaim(
        line=line,
        span=span,
        claim=text,
        scope="paper",
        category="number",
        verify_marker=False,
    )


def _client(batches):
    """Answer each extract call in turn from ``batches`` (a list of claim lists)."""
    users = []
    pending = list(batches)

    def parse(**kwargs):
        users.append(kwargs["messages"][0]["content"])
        claims = pending.pop(0) if pending else []
        return SimpleNamespace(
            parsed_output=ExtractionBatch(claims=claims),
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=10,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                cache_creation=None,
            ),
            stop_reason="end_turn",
        )

    client = Mock(spec=Anthropic)
    client.messages = Mock(spec=Messages)
    client.messages.parse.side_effect = parse
    return client, users


def _run(script, batches):
    client, users = _client(batches)
    result = extract_claims(client, script, model=MODEL, spend=Spend(None))
    return result, users


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("[HOST]: It scores 57.4 on the benchmark.", True),
        ("[HOST]: About 62 percent of them.", True),
        ("[HOST]: It was trained on 2,340 examples.", True),
        ("[HOST]: A 7B model.", True),
        ("[HOST]: It has 180 million parameters.", True),
        ("[HOST]: Roughly eighty thousand times larger.", True),
        ("[HOST]: There are three stages.", False),
        ("[HOST]: See Table 9 for the rows.", False),
        ("[HOST]: Welcome back to the show.", False),
        ("[HOST]: Question fourteen. How did they avoid leakage?", False),
        ("[HOST]: Step 40 is where it gets hard.", False),
        ("[HOST]: Question fourteen asks about 57.4 percent.", True),
    ],
)
def test_number_detection(line, expected):
    assert has_nontrivial_number(line) is expected


def test_only_host_lines_count():
    lines = [
        "[MUSIC] 44 beats",
        "[HOST]: It scores 57.4 here.",
        "",
        "[HOST]: No figure in this one.",
    ]
    assert number_bearing_lines(lines, 0, 4) == [1]
    assert substantive_host_lines(lines, 0, 4) == []


def test_an_empty_window_over_figures_is_read_again_and_the_retry_is_kept():
    script = "\n".join([
        "[HOST]: It scores 57.4 on the benchmark after stage two.",
        "[HOST]: Hello again.",
    ])
    retry = [
        _claim(0, "scores 57.4", "The model scores 57.4."),
        _claim(1, "Hello again", "A greeting."),
    ]
    result, users = _run(script, [[], retry])
    assert len(users) == 2
    assert "returned no claims at all" in users[1]
    assert [c.line_index for c in result.claims] == [0, 1]
    (window,) = result.windows
    assert (window.first_pass, window.claims, window.retried) == (0, 2, True)
    assert window.uncovered == []
    assert result.describe().startswith("2 claims; windows L0-1 0>2")


def test_a_partial_window_retries_only_the_missed_figure_lines():
    script = "\n".join([
        "[HOST]: The first line says nothing numeric at all.",
        "[HOST]: It scores 57.4 on the benchmark after stage two.",
    ])
    first = [_claim(0, "first line says", "Line zero.")]
    retry = [
        _claim(0, "first line says", "Line zero again."),
        _claim(1, "scores 57.4", "The model scores 57.4."),
    ]
    result, users = _run(script, [first, retry])
    assert "state figures: L1" in users[1]
    assert sorted(c.line_index for c in result.claims) == [0, 1]
    assert result.windows[0].first_pass == 1
    assert result.windows[0].claims == 2


def test_the_retry_happens_once_and_a_still_uncovered_line_is_reported():
    script = "[HOST]: It scores 57.4 on the benchmark after stage two."
    result, users = _run(script, [[], []])
    assert len(users) == 2
    assert result.claims == []
    assert result.windows[0].uncovered == [0]
    assert "numbers with no claim on L0" in result.describe()


def test_dropped_spans_are_summed_across_both_passes():
    script = "[HOST]: It scores 57.4 on the benchmark after stage two."
    result, _users = _run(script, [
        [_claim(0, "not on this line")],
        [_claim(0, "also missing"), _claim(0, "scores 57.4")],
    ])
    assert result.dropped == 2
    assert len(result.claims) == 1
    assert "2 spans dropped" in result.describe()


def test_an_empty_window_of_cues_and_short_lines_is_not_retried():
    script = "\n".join(["[MUSIC]", "[HOST]: Welcome back.", "[PAUSE]"])
    result, users = _run(script, [[]])
    assert len(users) == 1
    assert result.windows[0].retried is False


def test_an_empty_window_over_a_long_host_line_without_figures_is_retried():
    script = (
        "[HOST]: The authors describe a careful method for building the data "
        "and then explain how every part of it was collected and cleaned."
    )
    result, users = _run(script, [[], [_claim(0, "careful method")]])
    assert len(users) == 2
    assert len(result.claims) == 1


def test_a_covered_script_makes_one_call_per_window():
    script = "\n".join(
        ["[HOST]: It scores 57.4 on the benchmark after stage two."]
        + ["[HOST]: short."] * 30
    )
    result, users = _run(script, [[_claim(0, "scores 57.4")], []])
    assert len(users) == 2
    assert [w.retried for w in result.windows] == [False, False]


def test_describe_names_each_window():
    windows = [
        WindowStat(0, 30, first_pass=12, claims=12, dropped=1),
        WindowStat(30, 60, first_pass=0, claims=43, retried=True, dropped=2),
    ]
    assert describe(windows, 55, 3) == (
        "55 claims, 3 spans dropped; windows L0-29 12, L30-59 0>43"
    )


def test_prompt_version_and_contradicted_wording():
    assert PROMPT_VERSION == "v3"
    assert "You must be able to quote the incompatible passage" in VERIFY_INSTRUCTIONS
    assert "does not say what the claim says is not a contradiction" in VERIFY_INSTRUCTIONS
    assert "Claims of ABSENCE" in VERIFY_INSTRUCTIONS
