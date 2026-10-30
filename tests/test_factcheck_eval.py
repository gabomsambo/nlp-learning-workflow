"""The calibration set, its scorer and the gate. Offline: no database, no model.

The committed dataset is checked here on every CI run, so a label edited into
an invalid state, or edited without re-measuring the baseline, fails the build
instead of silently changing what the gate compares against.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

from nlp_pillars.factcheck import evaluation as ev
from nlp_pillars.factcheck.severity import assign_severity

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evals" / "factcheck" / "calibration.json"
BASELINE = ROOT / "evals" / "factcheck" / "baseline.json"

sys.path.insert(0, str(ROOT / "scripts"))
import factcheck_eval  # noqa: E402

SID = "11111111-2222-4333-8444-555555555555"
LINE0 = "[HOST]: The model scores 62.4 on the benchmark and beats the baseline."
LINE1 = "[HOST]: It was trained for ten days on eight GPUs, which is cheap."


def _label(lid, line, text, span, verdict, error_type="none", scope="paper", category="number"):
    start = text.index(span)
    return {
        "id": lid,
        "line_index": line,
        "span_start": start,
        "span_end": start + len(span),
        "span_text": span,
        "claim": f"claim {lid}",
        "scope": scope,
        "category": category,
        "verdict": verdict,
        "error_type": error_type,
        "severity": assign_severity(verdict, scope, category, error_type),
        "evidence": [{"quote": "a verbatim quote"}] if verdict != "unsupported" else [],
        "absence_note": "searched the results section" if verdict == "unsupported" else "",
        "note": "because",
        "provenance": {
            "labelled_by": "test", "basis": "paper-text", "date": "2026-10-30",
            "audited_by": "test", "audit": "agreed",
        },
    }


def _data():
    labels = [
        _label("a", 0, LINE0, "scores 62.4 on the benchmark", "contradicted", "number_mismatch"),
        _label("b", 0, LINE0, "beats the baseline", "supported"),
        _label("c", 1, LINE1, "ten days on eight GPUs", "supported"),
        _label("d", 1, LINE1, "which is cheap", "unsupported", "invented_detail", category="implication"),
    ]
    return {
        "schema_version": 1,
        "name": "t",
        "created": "2026-10-30",
        "description": "t",
        "labelling": "t",
        "scripts": [{
            "script_id": SID,
            "paper_id": "p",
            "title": "T",
            "script_sha256": "a" * 64,
            "paper_text_sha256": "b" * 64,
            "coverage": [[0, 1]],
            "lines": {"0": LINE0, "1": LINE1},
            "labels": labels,
        }],
    }


def _pred(line, text, span, severity):
    start = text.index(span)
    return {"line_index": line, "span_start": start, "span_end": start + len(span),
            "span_text": span, "claim": span, "severity": severity}


def _perfect(data):
    return {
        s["script_id"]: [
            {"line_index": l["line_index"], "span_start": l["span_start"], "span_end": l["span_end"],
             "span_text": l["span_text"], "claim": l["claim"], "severity": l["severity"]}
            for l in s["labels"]
        ]
        for s in data["scripts"]
    }


# --- the committed dataset ---------------------------------------------------


@pytest.fixture(scope="module")
def committed():
    return ev.load_dataset(DATASET)


def test_committed_dataset_is_valid_and_the_size_the_spec_asks_for(committed):
    assert ev.validate_dataset(committed, min_labels=150, max_labels=200) == []
    assert 4 <= len(committed["scripts"]) <= 5


def test_committed_dataset_has_each_kind_of_error(committed):
    summary = ev.dataset_summary(committed)
    for severity in ("high", "medium", "low", "none"):
        assert summary["by_severity"][severity] >= 5, summary["by_severity"]
    assert summary["by_verdict"].get("contradicted", 0) >= 3


def test_every_label_has_a_second_reader(committed):
    for _script, label in ev.iter_labels(committed):
        assert label["provenance"]["audited_by"] != "pending", label["id"]


def test_perfect_predictions_score_one_on_the_committed_labels(committed):
    report = ev.score(committed, _perfect(committed))
    for name in ("high", "medium", "low", "any"):
        assert report["metrics"][name]["precision"] == 1.0, name
        assert report["metrics"][name]["recall"] == 1.0, name
    assert report["misses"] == [] and report["false_alarms"] == []


def test_the_baseline_was_measured_on_this_dataset(committed):
    """Editing a label without re-measuring makes the gate compare apples to pears."""
    if not BASELINE.exists():
        pytest.skip("no baseline committed yet")
    baseline = json.loads(BASELINE.read_text())
    assert baseline["dataset_sha256"] == ev.dataset_sha256(committed), (
        "calibration.json changed since baseline.json was measured. Re-measure "
        "with scripts/factcheck_eval.py --live --write-baseline (docs/FACTCHECK_EVAL.md)."
    )
    assert baseline["gate"]["metrics"] == list(ev.GATE_METRICS)


def test_dry_run_exits_zero_on_the_committed_dataset(capsys):
    assert factcheck_eval.main(["--dry-run"]) == 0
    assert "dataset is valid" in capsys.readouterr().out


# --- dataset validation ------------------------------------------------------


def _problems(mutate):
    data = _data()
    mutate(data["scripts"][0]["labels"][0], data["scripts"][0], data)
    return ev.validate_dataset(data)


def test_a_valid_small_dataset_has_no_problems():
    assert ev.validate_dataset(_data()) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda l, s, d: l.update(severity="low"), "severity table"),
    (lambda l, s, d: l.update(span_end=l["span_end"] + 1), "span_text is not"),
    (lambda l, s, d: l.update(verdict="maybe"), "verdict"),
    (lambda l, s, d: l.update(evidence=[]), "needs at least one verbatim paper quote"),
    (lambda l, s, d: l["provenance"].pop("audited_by"), "provenance.audited_by"),
    (lambda l, s, d: l["provenance"].update(audit="skipped"), "provenance.audit"),
    (lambda l, s, d: s.update(coverage=[[5, 6]]), "outside every coverage window"),
    (lambda l, s, d: l.update(line_index=9), "no entry in the script's lines"),
    (lambda l, s, d: l.update(id="b"), "duplicate label id"),
    (lambda l, s, d: l.update(error_type="none"), "needs a real error_type"),
    (lambda l, s, d: s.update(script_sha256="xyz"), "sha256"),
])
def test_validation_names_the_problem(mutate, needle):
    assert any(needle in p for p in _problems(mutate))


def test_unsupported_without_a_quote_needs_an_absence_note():
    data = _data()
    data["scripts"][0]["labels"][3]["absence_note"] = ""
    assert any("absence_note" in p for p in ev.validate_dataset(data))


def test_label_count_bounds():
    assert any("fewer than" in p for p in ev.validate_dataset(_data(), min_labels=10))
    assert any("more than" in p for p in ev.validate_dataset(_data(), max_labels=2))


# --- scoring -----------------------------------------------------------------


def test_exact_severity_precision_and_recall():
    data = _data()
    preds = {SID: [
        _pred(0, LINE0, "scores 62.4 on the benchmark", "high"),   # right
        _pred(1, LINE1, "which is cheap", "high"),                 # gold is medium: FP for high, miss for medium
    ]}
    report = ev.score(data, preds)
    high = report["metrics"]["high"]
    assert (high["labels"], high["predictions"]) == (1, 2)
    assert high["recall"] == 1.0 and high["precision"] == 0.5
    medium = report["metrics"]["medium"]
    assert medium["labels"] == 1 and medium["recall"] == 0.0
    # Pooled "was it flagged at all": both labelled flags were flagged.
    assert report["metrics"]["any"]["recall"] == 1.0
    assert report["metrics"]["any"]["precision"] == 1.0
    assert report["confusion"]["medium"]["high"] == 1


def test_an_unflagged_error_is_a_miss_and_names_whether_it_was_extracted():
    report = ev.score(_data(), {SID: [_pred(0, LINE0, "beats the baseline", "none")]})
    assert report["metrics"]["high"]["recall"] == 0.0
    miss = report["misses"][0]
    assert miss["label_id"] in {"a", "d"} and miss["got"] == "not extracted"


def test_a_flag_in_a_labelled_window_with_no_label_is_a_false_alarm():
    data = _data()
    data["scripts"][0]["lines"]["1"] = LINE1 + " Extra sentence here."
    preds = {SID: [_pred(1, data["scripts"][0]["lines"]["1"], "Extra sentence here.", "high")]}
    report = ev.score(data, preds)
    assert report["metrics"]["high"]["predictions"] == 1
    assert report["metrics"]["high"]["precision"] == 0.0
    assert report["false_alarms"][0]["gold"] == ["no label"]
    ignored = ev.score(data, preds, extras="ignore")
    assert ignored["metrics"]["high"]["predictions"] == 0


def test_predictions_outside_every_window_are_not_judged():
    data = _data()
    data["scripts"][0]["coverage"] = [[0, 0]]
    del data["scripts"][0]["labels"][2:]  # labels only on line 0
    preds = {SID: [_pred(1, LINE1, "which is cheap", "high")]}
    report = ev.score(data, preds)
    assert report["unscored_predictions"] == 1
    assert report["metrics"]["high"]["predictions"] == 0


def test_a_span_that_merges_two_clauses_still_counts_for_both():
    """Many-to-many on purpose: merging two clauses is not a miss of either."""
    wide = _pred(0, LINE0, "scores 62.4 on the benchmark and beats the baseline", "high")
    report = ev.score(_data(), {SID: [wide]})
    assert report["metrics"]["high"]["recall"] == 1.0
    assert report["metrics"]["high"]["precision"] == 1.0


def test_one_character_of_overlap_is_not_a_match():
    stub = {"line_index": 0, "span_start": 0, "span_end": 12, "severity": "high"}
    report = ev.score(_data(), {SID: [stub]})
    assert report["metrics"]["high"]["recall"] == 0.0


def test_a_prediction_without_offsets_is_located_by_its_text():
    pred = {"line_index": 0, "span_text": "scores 62.4 on the benchmark", "severity": "high"}
    assert ev.score(_data(), {SID: [pred]})["metrics"]["high"]["recall"] == 1.0


def test_missing_predictions_for_a_script_is_an_error_not_a_zero():
    with pytest.raises(ValueError, match="no predictions"):
        ev.score(_data(), {})


def test_unknown_severity_is_an_error():
    with pytest.raises(ValueError, match="severity"):
        ev.score(_data(), {SID: [{"line_index": 0, "severity": "critical"}]})


def test_zero_denominators_are_none_not_zero():
    report = ev.score(_data(), {SID: []})
    assert report["metrics"]["high"]["precision"] is None
    assert report["metrics"]["high"]["recall"] == 0.0


# --- baseline and gate -------------------------------------------------------


def _baseline(data, report, tolerance=0.05):
    return ev.make_baseline(data, report, config={"source": "test"}, cost_usd=0.5,
                            created="2026-10-30", tolerance=tolerance)


def test_gate_passes_on_the_same_numbers():
    data = _data()
    report = ev.score(data, _perfect(data))
    assert ev.check_gate(data, report, _baseline(data, report)).passed


def test_gate_fails_when_serious_recall_drops():
    data = _data()
    good = ev.score(data, _perfect(data))
    worse = ev.score(data, {SID: [_pred(0, LINE0, "beats the baseline", "none")]})
    result = ev.check_gate(data, worse, _baseline(data, good))
    assert not result.passed
    assert any("high.recall" in line and "FAIL" in line for line in result.lines)


def test_gate_fails_when_serious_precision_drops():
    data = _data()
    good = ev.score(data, _perfect(data))
    preds = _perfect(data)
    preds[SID].append(_pred(1, LINE1, "ten days on eight GPUs", "high"))
    noisy = ev.score(data, preds)
    result = ev.check_gate(data, noisy, _baseline(data, good))
    assert not result.passed
    assert any("high.precision" in line and "FAIL" in line for line in result.lines)


def test_gate_tolerance_lets_noise_through_but_not_more():
    data = _data()
    base = ev.score(data, _perfect(data))
    baseline = _baseline(data, base, tolerance=0.0)
    noisy = copy.deepcopy(base)
    noisy["metrics"]["high"]["recall"] = 0.97
    assert not ev.check_gate(data, noisy, baseline).passed
    assert ev.check_gate(data, noisy, baseline, tolerance=0.05).passed


def test_a_baseline_for_other_labels_is_stale_not_a_pass():
    data = _data()
    report = ev.score(data, _perfect(data))
    baseline = _baseline(data, report)
    data["scripts"][0]["labels"][0]["note"] = "edited"
    result = ev.check_gate(data, report, baseline)
    assert result.stale and not result.passed


def test_a_run_with_no_value_cannot_pass():
    data = _data()
    report = ev.score(data, _perfect(data))
    baseline = _baseline(data, report)
    empty = ev.score(data, {SID: []})
    assert not ev.check_gate(data, empty, baseline).passed


# --- the command line --------------------------------------------------------


def _write(tmp_path, data):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_cli_scores_a_predictions_file_and_gates(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(factcheck_eval, "MIN_LABELS", 1)
    data = _data()
    dataset = _write(tmp_path, data)
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"verifier": "good", "predictions": _perfect(data)}))
    base = str(tmp_path / "baseline.json")
    args = ["--dataset", dataset, "--baseline", base]

    assert factcheck_eval.main(args + ["--predictions", str(good), "--write-baseline"]) == 0
    assert factcheck_eval.main(args + ["--predictions", str(good)]) == 0
    assert "GATE PASSED" in capsys.readouterr().out

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"verifier": "bad", "predictions": {SID: []}}))
    assert factcheck_eval.main(args + ["--predictions", str(bad)]) == 1
    assert "GATE FAILED" in capsys.readouterr().out
    assert factcheck_eval.main(args + ["--predictions", str(bad), "--no-gate"]) == 0


def test_cli_refuses_a_stale_baseline_and_a_missing_one(tmp_path, monkeypatch):
    monkeypatch.setattr(factcheck_eval, "MIN_LABELS", 1)
    data = _data()
    dataset = _write(tmp_path, data)
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"predictions": _perfect(data)}))
    base = tmp_path / "baseline.json"
    args = ["--dataset", dataset, "--baseline", str(base), "--predictions", str(good)]
    assert factcheck_eval.main(args) == 2  # no baseline yet
    factcheck_eval.main(args + ["--write-baseline"])
    data["scripts"][0]["labels"][1]["note"] = "edited"
    other = _write(tmp_path, data)
    assert factcheck_eval.main(["--dataset", other, "--baseline", str(base), "--predictions", str(good)]) == 2


def test_cli_rejects_an_invalid_dataset(tmp_path):
    data = _data()
    data["scripts"][0]["labels"][0]["severity"] = "low"
    assert factcheck_eval.main(["--dataset", _write(tmp_path, data), "--dry-run"]) == 2


def test_cli_baseline_needs_the_whole_set(tmp_path, monkeypatch):
    monkeypatch.setattr(factcheck_eval, "MIN_LABELS", 1)
    data = _data()
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"predictions": _perfect(data)}))
    code = factcheck_eval.main([
        "--dataset", _write(tmp_path, data), "--baseline", str(tmp_path / "b.json"),
        "--predictions", str(good), "--scripts", SID[:8], "--write-baseline",
    ])
    assert code == 2


def test_live_mode_is_never_the_default():
    parser = factcheck_eval.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # a mode is required: nothing paid runs by accident
    assert parser.parse_args(["--dry-run"]).live is False


def test_a_crash_is_exit_two_never_one(tmp_path, monkeypatch, capsys):
    """Exit 1 is reserved for a real regression; a dead database must not read as one."""
    monkeypatch.setattr(factcheck_eval, "MIN_LABELS", 1)

    def boom(*args, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(factcheck_eval, "mode_from_stored", boom)
    code = factcheck_eval.main(["--dataset", _write(tmp_path, _data()), "--from-stored"])
    assert code == 2
    assert "ConnectionError" in capsys.readouterr().err


def test_pooling_adds_counts_instead_of_averaging_ratios():
    data = _data()
    hit = ev.score(data, _perfect(data))
    quiet = ev.score(data, {SID: []})  # flags nothing: precision undefined, recall 0
    pooled = ev.pool_reports([hit, quiet])
    high = pooled["metrics"]["high"]
    assert pooled["runs"] == 2
    assert (high["labels"], high["true_positive_labels"]) == (2, 1)
    assert high["recall"] == 0.5
    # The silent run made no high predictions, so it must not drag precision to 0.5.
    assert high["precision"] == 1.0
    assert ev.pool_reports([hit]) is hit
    assert {m["run"] for m in pooled["misses"]} == {2}


def test_default_live_pass_count_is_more_than_one():
    assert factcheck_eval.DEFAULT_LIVE_RUNS >= 2
