"""Number lint: derived and unsupported numbers in a script. Hermetic, no I/O."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from nlp_pillars.cli import app
from nlp_pillars.factcheck.numbers import (
    KIND_COUNT,
    KIND_RATIO,
    KIND_UNMATCHED,
    MAX_WARNINGS,
    WARNING_PREFIX,
    lint_script,
    merge_warnings,
    number_spans,
    warnings_for,
)
from nlp_pillars.schemas import SourceMaterial

RUNNER = CliRunner()

PAPER = (
    "We train on 2,340 transactions in MLDP. Our large dataset has 175 M - 180 M "
    "transactions. The UN estimates $0.8 - $2.0 trillion are laundered. "
    "Accuracy is 62.4% on AG News and 87,109 tokens were used. "
    "We compare against 8 datasets and win on 5 of the 8."
)


def _host(*lines: str) -> str:
    return "\n\n".join(f"[HOST]: {line}" for line in lines)


def _flags(script: str, paper: str = PAPER):
    return lint_script(script, paper).flags


def test_the_two_known_derived_number_errors_are_flagged():
    fasttext = _flags(_host("It beats the baseline on six out of eight datasets."))
    assert [(f.kind, f.raw) for f in fasttext] == [(KIND_COUNT, "six out of eight")]

    aml = _flags(
        _host("This generator makes datasets eighty thousand times larger, at minimum.")
    )
    assert [(f.kind, f.raw) for f in aml] == [(KIND_UNMATCHED, "eighty thousand")]
    assert KIND_RATIO not in {f.kind for f in aml}


def test_a_small_ratio_the_paper_never_states_is_flagged():
    flags = _flags(_host("It is seven times faster."), "It runs in 40 ms and 3 s.")
    assert [(f.kind, f.raw) for f in flags] == [(KIND_RATIO, "seven times")]
    assert _flags(_host("It is three times faster."), "It runs in 40 ms and 3 s.") == []


def test_numbers_the_paper_states_pass():
    script = _host(
        "There are 2,340 transactions.",
        "The large set has one hundred seventy-five to one hundred eighty million.",
        "The UN says eight hundred billion to two trillion dollars.",
        "Accuracy is about sixty-two percent, and 87 thousand tokens.",
        "It wins on five of the eight datasets.",
    )
    assert _flags(script) == []


def test_small_integers_years_and_non_host_lines_are_not_flagged():
    script = (
        "[MUSIC]\n\n[PAUSE: 2s]\n"
        "[HOST]: Three stages, in twenty twenty-four, published in 2011, over seventeen years.\n"
        "[GUEST]: eighty thousand times\n"
    )
    result = lint_script(script, PAPER)
    assert result.host_lines == 1
    assert [f.raw for f in result.flags] == ["seventeen"]


def test_a_count_the_paper_states_is_not_flagged_but_a_new_one_is():
    assert _flags(_host("It wins on 5 of 8 datasets.")) == []
    assert [f.raw for f in _flags(_host("It wins on four of the eight datasets."))] == [
        "four of the eight"
    ]


def test_idioms_and_fragments_are_not_counts():
    script = _host(
        "One of the eight datasets, and one of ten million samples, plus 30 of 5000."
    )
    assert KIND_COUNT not in {f.kind for f in _flags(script)}


def test_known_false_positive_shapes_are_handled():
    spans = {s.raw: s.value for s in number_spans("half a million, 175 _M_, twenty twenty-four")}
    assert spans["half a million"] == 500_000
    assert 175_000_000 in spans.values()
    assert 2024 in spans.values()
    assert _flags(_host("Roughly half a million rows."), "It has 500,000 rows.") == []


def test_every_flag_has_a_reason_and_a_warning_names_its_line():
    result = lint_script(
        _host("It wins on six out of eight datasets.", "Fine."), PAPER,
    )
    assert result.flags and all(f.reason for f in result.flags)
    warnings = warnings_for(result)
    assert len(warnings) == 1
    assert warnings[0].startswith(f"{WARNING_PREFIX}, script line 1:")
    assert "six out of eight" in warnings[0]


def test_warnings_are_capped_and_replace_only_their_own_earlier_run():
    script = _host(*[f"It is {n} times bigger." for n in range(900, 900 + MAX_WARNINGS + 3)])
    warnings = warnings_for(lint_script(script, PAPER))
    assert len(warnings) == MAX_WARNINGS + 1
    assert warnings[-1].startswith(WARNING_PREFIX) and "3 more" in warnings[-1]

    merged = merge_warnings(["Abstract only.", f"{WARNING_PREFIX}, script line 9: x"], ["new"])
    assert merged == ["Abstract only.", "new"]


def test_the_rate_is_flagged_lines_per_ten_host_lines():
    result = lint_script(_host(*["Plain."] * 9, "It wins on six out of eight."), PAPER)
    assert result.host_lines == 10 and result.flagged_lines == 1
    assert result.per_ten_host_lines == pytest.approx(1.0)


def _row(warnings=None):
    return SimpleNamespace(
        paper_id="p1",
        script=_host("It wins on six out of eight datasets."),
        source_material=SourceMaterial(warnings=warnings or ["Abstract only."]),
    )


def test_cli_saves_merged_warnings_and_never_touches_the_script():
    with patch("nlp_pillars.cli.db") as db:
        db.get_podcast_script_by_id.return_value = _row()
        db.get_stored_paper_text.return_value = PAPER
        db.PodcastScriptSaveError = RuntimeError
        result = RUNNER.invoke(app, ["factcheck", "lint", "--script", "s1"])
    assert result.exit_code == 0, result.output
    script_id, saved = db.update_podcast_source_material_warnings.call_args.args
    assert script_id == "s1"
    assert saved[0] == "Abstract only." and saved[1].startswith(WARNING_PREFIX)
    assert "six out of eight" in result.output


def test_cli_dry_run_writes_nothing_and_missing_text_fails():
    with patch("nlp_pillars.cli.db") as db:
        db.get_podcast_script_by_id.return_value = _row()
        db.get_stored_paper_text.return_value = PAPER
        result = RUNNER.invoke(app, ["factcheck", "lint", "--script", "s1", "--dry-run"])
        assert result.exit_code == 0
        db.update_podcast_source_material_warnings.assert_not_called()

        db.get_stored_paper_text.return_value = None
        result = RUNNER.invoke(app, ["factcheck", "lint", "--script", "s1"])
        assert result.exit_code == 1
