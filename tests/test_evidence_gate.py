"""The evidence writer's gate arithmetic (nlp_pillars/evidence/gate.py) and its runner.

Records are built by hand; nothing calls a model, a database or the network.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from nlp_pillars.evidence import gate

ROOT = Path(__file__).resolve().parents[1]


def _claim(line, severity="high", span="x"):
    return {"line_index": line, "span_text": span, "severity": severity, "claim": f"claim on {line}"}


def _run(paper, arm, sample, words, high=0, medium=0, low=0, claims=100, flags=(), stats=None, ledger=None):
    return {
        "paper": paper, "arm": arm, "sample": sample, "words": words,
        "counts": {"claims": claims, "high": high, "medium": medium, "low": low},
        "claims": list(flags), "stats": stats or {}, "ledger": ledger or {},
        "cost": {"generate": 0.1, "check": 0.05 if arm == "evidence" else 0.12},
    }


def _good_stats(**kw):
    return {"narrative_used_ratio": 0.7, "untagged_lines": 0, "spoken_lines": 100,
            "unknown_ids": [], "number_violation_lines": 1, **kw}


def _arms(n_papers=8, samples=2, ev_words=900, free_words=1000, ev_flags=0, free_flags=2, **ev_kw):
    free, evidence = [], []
    for p in range(n_papers):
        for s in range(1, samples + 1):
            free.append(_run(f"p{p}", "free", s, free_words, high=free_flags,
                             flags=[_claim(i) for i in range(free_flags)]))
            evidence.append(_run(f"p{p}", "evidence", s, ev_words, high=ev_flags,
                                 flags=[_claim(i) for i in range(ev_flags)],
                                 stats=_good_stats(**ev_kw.get("stats", {})),
                                 ledger={"proposed": 60, "dropped": [{}] * ev_kw.get("dropped", 2)}))
    return free, evidence


def _label_all(*arms, verdict="real"):
    return {f["fid"]: verdict for runs in arms for f in gate.serious_flags(runs)}


def test_spoken_words_leaves_out_prefixes_cues_and_tags():
    script = "[MUSIC]\n[HOST]: One two three.\n\n[PAUSE]\n[HOST]: Four five. [[F1]]\n"
    assert gate.spoken_words(script) == 5


class TestFlagIds:
    def test_ids_are_stable_and_distinct_per_claim(self):
        run = _run("p", "free", 1, 100, flags=[_claim(3), _claim(3, "medium"), _claim(5, "low")])
        first = gate.serious_flags([run])
        again = gate.serious_flags([run])
        assert [f["fid"] for f in first] == [f["fid"] for f in again]
        assert len({f["fid"] for f in first}) == 2  # the low flag is not serious

    def test_the_order_hides_the_arm(self):
        free = _run("p", "free", 1, 100, flags=[_claim(i) for i in range(6)])
        ev = _run("p", "evidence", 1, 100, flags=[_claim(i) for i in range(6)])
        arms = [f["arm"] for f in gate.serious_flags([free, ev])]
        assert arms != sorted(arms)  # shuffled by hash, not grouped


class TestArmSummary:
    def test_rates_per_thousand_words(self):
        runs = [_run("p", "free", 1, 1500, high=2, medium=1, low=3), _run("p", "free", 2, 500, high=0, medium=1, low=1)]
        s = gate.arm_summary(runs)
        assert s["words"] == 2000 and (s["high"], s["medium"], s["low"]) == (2, 2, 4)
        assert s["serious_per_1000_words"] == pytest.approx(2.0)
        assert s["flags_per_1000_words"] == pytest.approx(4.0)

    def test_labels_give_precision_and_a_confirmed_rate(self):
        run = _run("p", "free", 1, 1000, high=3, flags=[_claim(1), _claim(2), _claim(3)])
        fids = [f["fid"] for f in gate.serious_flags([run])]
        labels = dict(zip(fids, ["real", "minor", "wrong"]))
        got = gate.arm_summary([run], labels)["labelled"]
        assert (got["real"], got["minor"], got["wrong"]) == (1, 1, 1)
        assert got["precision_strict"] == pytest.approx(1 / 3)
        assert got["precision_lenient"] == pytest.approx(2 / 3)
        assert got["confirmed_per_1000_words"] == pytest.approx(1.0)

    def test_no_words_means_no_rate_not_a_zero_division(self):
        assert gate.per_1000(3, 0) is None


class TestPairedDifference:
    def test_it_resamples_papers_and_is_deterministic(self):
        free, evidence = _arms(free_flags=2, ev_flags=0)
        a = gate.paired_difference(free, evidence, lambda runs: gate.per_1000(sum(r["counts"]["high"] for r in runs), sum(r["words"] for r in runs)))
        b = gate.paired_difference(free, evidence, lambda runs: gate.per_1000(sum(r["counts"]["high"] for r in runs), sum(r["words"] for r in runs)))
        assert a == b and a["papers"] == 8
        assert a["mean"] == pytest.approx(-2.0) and a["lo"] <= a["mean"] <= a["hi"] < 0

    def test_no_shared_papers_is_empty(self):
        free = [_run("a", "free", 1, 100)]
        ev = [_run("b", "evidence", 1, 100)]
        assert gate.paired_difference(free, ev, lambda r: 1.0)["mean"] is None


class TestGate:
    def test_a_good_run_passes_every_criterion(self):
        free, evidence = _arms()
        result = gate.evaluate(free, evidence, _label_all(free, evidence))
        assert result["passed"], [c for c in result["checks"] if not c["passed"]]
        assert result["completeness"]["words_ratio"] == pytest.approx(0.9)

    def test_too_few_papers_or_samples_cannot_pass(self):
        free, evidence = _arms(n_papers=7)
        result = gate.evaluate(free, evidence, _label_all(free, evidence))
        assert not result["passed"]
        assert result["checks"][0]["name"] == "enough papers and samples"
        assert "7 papers" in result["checks"][0]["detail"]
        free, evidence = _arms(samples=1)
        assert any("samples" in p for p in gate.coverage_problems(free, evidence))

    def test_saying_less_does_not_win(self):
        free, evidence = _arms(ev_words=800)
        result = gate.evaluate(free, evidence, _label_all(free, evidence))
        failed = {c["name"] for c in result["checks"] if not c["passed"]}
        assert failed == {"spoken words >= 85% of free"}

    def test_an_untagged_line_fails_the_zero_untagged_rule(self):
        free, evidence = _arms(stats={"untagged_lines": 1})
        result = gate.evaluate(free, evidence, _label_all(free, evidence))
        assert {c["name"] for c in result["checks"] if not c["passed"]} == {"0 untagged lines"}

    def test_a_high_quote_drop_rate_fails(self):
        free, evidence = _arms(dropped=9)
        result = gate.evaluate(free, evidence, _label_all(free, evidence))
        assert [c["name"] for c in result["checks"] if not c["passed"]] == ["quote-verification drop rate <= 10%"]

    def test_unused_ledger_fails(self):
        free, evidence = _arms(stats={"narrative_used_ratio": 0.4})
        result = gate.evaluate(free, evidence, _label_all(free, evidence))
        assert [c["name"] for c in result["checks"] if not c["passed"]] == ["ledger entries used >= 60%"]

    def test_more_confirmed_errors_than_free_fails_even_if_the_engine_flags_fewer(self):
        free, evidence = _arms(free_flags=1, ev_flags=3)
        labels = _label_all(free, evidence)
        labels.update({f["fid"]: "wrong" for f in gate.serious_flags(free)})
        result = gate.evaluate(free, evidence, labels)
        assert "confirmed serious flags per 1,000 words no higher than free" in {
            c["name"] for c in result["checks"] if not c["passed"]
        }

    def test_an_unlabelled_serious_flag_blocks_the_gate(self):
        free, evidence = _arms()
        labels = _label_all(free, evidence)
        dropped = next(iter(labels))
        del labels[dropped]
        result = gate.evaluate(free, evidence, labels)
        assert result["unlabelled"] == [dropped] and not result["passed"]


class TestRunner:
    @pytest.fixture
    def runner(self):
        spec = importlib.util.spec_from_file_location("podcast_writer_gate", ROOT / "scripts" / "podcast_writer_gate.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_plan_is_offline_and_estimates_under_the_budget(self, runner, tmp_path, capsys):
        assert runner.main(["plan", "--out", str(tmp_path)]) == 0
        text = capsys.readouterr().out
        assert "8 papers x 2 samples x 2 arms = 32 scripts" in text
        assert "estimated total" in text

    def test_the_default_set_is_eight_distinct_papers(self, runner):
        papers = [p for p, _ in runner.DEFAULT_PAPERS]
        assert len(papers) == len(set(papers)) >= gate.MIN_PAPERS

    def test_spend_is_summed_and_the_budget_is_a_hard_stop(self, runner, tmp_path):
        out = runner.Out(str(tmp_path))
        out.charge("a", 1.25)
        out.charge("b", 0.5)
        assert out.spent() == pytest.approx(1.75)
        out.require_budget(2.0)
        with pytest.raises(runner.GateError, match="would be exceeded"):
            out.require_budget(2.0, 0.5)

    @pytest.fixture(autouse=True)
    def _restore_provider_env(self, monkeypatch):
        """load_key() sets PODCAST_EXTRACTION_PROVIDER; make sure the test run undoes it."""
        monkeypatch.setenv("PODCAST_EXTRACTION_PROVIDER", "anthropic")
        monkeypatch.delenv("PODCAST_EXTRACTION_PROVIDER")

    def test_the_key_is_read_from_the_file_without_being_printed(self, runner, tmp_path, monkeypatch, capsys):
        env = tmp_path / ".env"
        env.write_text("ANTHROPIC_API_KEY=sk-ant-not-a-real-key\nOTHER_SECRET=nope\n")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OTHER_SECRET", raising=False)
        runner.load_key(str(env))
        import os

        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-not-a-real-key"
        assert "OTHER_SECRET" not in os.environ
        captured = capsys.readouterr()
        assert "sk-ant" not in captured.out + captured.err

    def test_a_placeholder_key_does_not_count(self, runner, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
        runner.load_key(str(env))
        import os

        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-file"

    def test_a_missing_key_is_an_error_not_a_crash(self, runner, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("SOMETHING=1\n")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(runner.GateError, match="no ANTHROPIC_API_KEY"):
            runner.load_key(str(env))

    def test_report_reads_the_labels_and_writes_json(self, runner, tmp_path, capsys):
        out = runner.Out(str(tmp_path))
        free, evidence = _arms(n_papers=2)
        for record in [*free, *evidence]:
            record["script"] = "[HOST]: hi."
            record["check"] = {}
            out.run_path(record["paper"], record["arm"], record["sample"]).write_text(json.dumps(record))
        labels = tmp_path / "labels.json"
        labels.write_text(json.dumps(_label_all(free, evidence)))
        assert runner.main(["report", "--out", str(tmp_path), "--labels", str(labels)]) == 0
        text = capsys.readouterr().out
        assert "GATE NOT PASSED" in text and "need 8" in text  # two papers is not a gate
        assert json.loads((tmp_path / "report.json").read_text())["passed"] is False
