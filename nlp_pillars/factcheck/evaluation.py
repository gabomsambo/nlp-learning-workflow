"""Score a fact-check verifier against the hand-labelled calibration set.

Everything in this module is pure: it reads dicts and returns dicts. Loading
scripts and papers, calling models and printing live in
``scripts/factcheck_eval.py``, so the dataset check and the scoring run offline
(and in CI) with no database and no key.

Scoring unit and its rules, because a number without them is not comparable:

* A **label** is one clause of one script line with a hand-decided verdict.
  Its severity is not typed in by the labeller: it is
  ``assign_severity(verdict, scope, category, error_type)``, the same
  deterministic table the product uses, and the dataset check rejects a label
  whose stored severity disagrees. A change to that table therefore shows up
  here as a dataset error, not as a silent shift in the metrics.
* A **prediction** is any claim a verifier emitted for a labelled script:
  ``line_index`` plus a character span (or ``span_text``) and a ``severity``.
* They **match** when they sit on the same line and their spans overlap by at
  least ``OVERLAP`` of the shorter span. Matching is many-to-many on purpose:
  an extractor that merges two clauses into one span should neither lose the
  recall of both nor be paid twice for one flag.
* Labelling is **exhaustive inside a script's ``coverage`` windows** and only
  there. A flagged prediction inside a window that overlaps no label is a false
  positive (the labeller read that line and found no such claim). Predictions
  outside every window cannot be judged and are counted as ``unscored``.
* Metrics are per **exact severity**: a "high" prediction on a claim labelled
  "medium" is a false positive for high and a miss for medium. ``any`` pools
  high, medium and low into "was this claim flagged at all".
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

from .schemas import Category, ErrorType, Scope, Verdict
from .severity import assign_severity

SCHEMA_VERSION = 1
REPORT_VERSION = 1

OVERLAP = 0.5
SEVERITIES = ("high", "medium", "low")
ALL_SEVERITIES = ("high", "medium", "low", "none")
FLAGGED = frozenset(SEVERITIES)
# The gate: serious errors. Precision is "when it says high, is it", recall
# is "does it find the high ones".
GATE_METRICS = ("high.precision", "high.recall")
# Measured, not chosen: two runs of the unchanged pipeline on the same labels
# differed by 0.15 in high precision (0.75 vs 0.60 on 8 high labels), because the
# models are not deterministic. Pooling several runs narrows that; the
# tolerance covers what is left.
DEFAULT_TOLERANCE = 0.10

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_QUOTE_MAX = 300

SCOPES = frozenset(get_args(Scope))
CATEGORIES = frozenset(get_args(Category))
VERDICTS = frozenset(get_args(Verdict))
ERROR_TYPES = frozenset(get_args(ErrorType))
# Provenance every label must carry. ``audited_by`` is the second reader.
PROVENANCE_KEYS = ("labelled_by", "basis", "date", "audited_by", "audit")
AUDIT_OUTCOMES = frozenset({"agreed", "changed", "added"})


class DatasetError(ValueError):
    """The calibration file is unusable. Carries every problem found."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        shown = "\n  - ".join(self.problems[:20])
        more = len(self.problems) - 20
        tail = f"\n  ... and {more} more" if more > 0 else ""
        super().__init__(f"{len(self.problems)} dataset problem(s):\n  - {shown}{tail}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_dataset(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def dataset_sha256(data: dict[str, Any]) -> str:
    """Hash of the labels, independent of key order and whitespace.

    A baseline is only comparable to the dataset it was measured on, so the
    baseline records this and the gate refuses a mismatch.
    """
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def iter_labels(data: dict[str, Any]):
    for script in data.get("scripts", []):
        for label in script.get("labels", []):
            yield script, label


def _in_coverage(coverage: list[list[int]], line: int) -> bool:
    return any(start <= line <= end for start, end in coverage)


def _check_label(
    script: dict[str, Any], label: dict[str, Any], where: str, problems: list[str],
) -> None:
    def bad(message: str) -> None:
        problems.append(f"{where}: {message}")

    for key in ("id", "claim", "note"):
        if not isinstance(label.get(key), str) or not label[key].strip():
            bad(f"missing {key}")
    line = label.get("line_index")
    lines = script.get("lines", {})
    text = lines.get(str(line)) if isinstance(line, int) else None
    if not isinstance(line, int) or line < 0:
        bad(f"line_index {line!r} is not a non-negative integer")
    elif text is None:
        bad(f"line {line} has no entry in the script's lines")
    else:
        coverage = script.get("coverage", [])
        if not _in_coverage(coverage, line):
            bad(f"line {line} is outside every coverage window")
        start, end = label.get("span_start"), label.get("span_end")
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
            bad(f"span {start!r}..{end!r} does not fit a {len(text)}-character line")
        elif text[start:end] != label.get("span_text"):
            bad(f"span_text is not line[{start}:{end}] ({text[start:end]!r})")

    scope = label.get("scope")
    category = label.get("category")
    verdict = label.get("verdict")
    error_type = label.get("error_type")
    if scope not in SCOPES:
        bad(f"scope {scope!r}")
    if category not in CATEGORIES:
        bad(f"category {category!r}")
    if verdict not in VERDICTS:
        bad(f"verdict {verdict!r}")
    if error_type not in ERROR_TYPES:
        bad(f"error_type {error_type!r}")
    if scope in SCOPES and category in CATEGORIES and verdict in VERDICTS and error_type in ERROR_TYPES:
        derived = assign_severity(verdict, scope, category, error_type)
        if label.get("severity") != derived:
            bad(
                f"severity {label.get('severity')!r} but the severity table gives "
                f"{derived!r} for {verdict}/{scope}/{category}/{error_type}"
            )
        if verdict in {"supported", "not_checkable"} and error_type != "none":
            bad(f"{verdict} must have error_type 'none'")
        if verdict in {"partially_supported", "contradicted", "unsupported"} and error_type == "none":
            bad(f"{verdict} needs a real error_type")

    evidence = label.get("evidence")
    if not isinstance(evidence, list):
        bad("evidence must be a list")
        evidence = []
    for item in evidence:
        quote = item.get("quote") if isinstance(item, dict) else None
        if not isinstance(quote, str) or not quote.strip():
            bad("evidence item without a quote")
        elif len(quote) > _QUOTE_MAX:
            bad(f"evidence quote is {len(quote)} characters (max {_QUOTE_MAX})")
    if verdict in {"supported", "partially_supported", "contradicted"} and not evidence:
        bad(f"{verdict} needs at least one verbatim paper quote")
    if verdict == "unsupported":
        if not evidence and not str(label.get("absence_note", "")).strip():
            bad("unsupported with no quote needs an absence_note saying what was searched")

    prov = label.get("provenance")
    if not isinstance(prov, dict):
        bad("missing provenance")
        return
    for key in PROVENANCE_KEYS:
        if not isinstance(prov.get(key), str) or not prov[key].strip():
            bad(f"provenance.{key} missing")
    if prov.get("audit") not in AUDIT_OUTCOMES:
        bad(f"provenance.audit must be one of {sorted(AUDIT_OUTCOMES)}")


def validate_dataset(
    data: dict[str, Any],
    *,
    min_labels: int = 0,
    max_labels: int | None = None,
) -> list[str]:
    """Return every problem found. An empty list means the file is usable."""
    problems: list[str] = []
    if data.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    for key in ("name", "created", "description", "labelling"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            problems.append(f"top-level {key} missing")
    scripts = data.get("scripts")
    if not isinstance(scripts, list) or not scripts:
        problems.append("scripts must be a non-empty list")
        return problems

    seen_scripts: set[str] = set()
    seen_ids: set[str] = set()
    total = 0
    for script in scripts:
        sid = script.get("script_id")
        where = f"script {str(sid)[:8]}"
        if not isinstance(sid, str) or not _UUID.match(sid):
            problems.append(f"{where}: script_id is not a lowercase uuid")
            continue
        if sid in seen_scripts:
            problems.append(f"{where}: listed twice")
        seen_scripts.add(sid)
        for key in ("paper_id", "title"):
            if not isinstance(script.get(key), str) or not script[key].strip():
                problems.append(f"{where}: missing {key}")
        for key in ("script_sha256", "paper_text_sha256"):
            if not _HEX64.match(str(script.get(key, ""))):
                problems.append(f"{where}: {key} is not a sha256 hex digest")
        coverage = script.get("coverage")
        if not isinstance(coverage, list) or not coverage:
            problems.append(f"{where}: coverage must list at least one [first, last] window")
            coverage = []
        for window in coverage:
            if not (
                isinstance(window, list)
                and len(window) == 2
                and all(isinstance(n, int) and n >= 0 for n in window)
                and window[0] <= window[1]
            ):
                problems.append(f"{where}: bad coverage window {window!r}")
        lines = script.get("lines")
        if not isinstance(lines, dict) or not lines:
            problems.append(f"{where}: lines must map line index to line text")
        labels = script.get("labels")
        if not isinstance(labels, list) or not labels:
            problems.append(f"{where}: no labels")
            continue
        seen_spans: set[tuple[int, int, int]] = set()
        for label in labels:
            lid = label.get("id")
            lwhere = f"{where} {lid}"
            if isinstance(lid, str):
                if lid in seen_ids:
                    problems.append(f"{lwhere}: duplicate label id")
                seen_ids.add(lid)
            _check_label(script, label, lwhere, problems)
            key = (label.get("line_index"), label.get("span_start"), label.get("span_end"))
            if key in seen_spans:
                problems.append(f"{lwhere}: another label has the same line and span")
            seen_spans.add(key)
            total += 1
    if total < min_labels:
        problems.append(f"{total} labels, fewer than the {min_labels} required")
    if max_labels is not None and total > max_labels:
        problems.append(f"{total} labels, more than the {max_labels} allowed")
    return problems


def require_valid(data: dict[str, Any], **kwargs: Any) -> None:
    problems = validate_dataset(data, **kwargs)
    if problems:
        raise DatasetError(problems)


def dataset_summary(data: dict[str, Any]) -> dict[str, Any]:
    by_severity = {name: 0 for name in ALL_SEVERITIES}
    by_verdict: dict[str, int] = {}
    per_script = []
    for script in data.get("scripts", []):
        count = 0
        for label in script.get("labels", []):
            count += 1
            by_severity[label.get("severity", "none")] = by_severity.get(label.get("severity", "none"), 0) + 1
            by_verdict[label.get("verdict", "?")] = by_verdict.get(label.get("verdict", "?"), 0) + 1
        per_script.append({
            "script_id": script.get("script_id"),
            "title": script.get("title"),
            "labels": count,
        })
    return {
        "labels": sum(item["labels"] for item in per_script),
        "by_severity": by_severity,
        "by_verdict": by_verdict,
        "scripts": per_script,
    }


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def normalise_predictions(
    raw: list[dict[str, Any]], script: dict[str, Any],
) -> list[dict[str, Any]]:
    """Give every prediction an integer span and a known severity.

    A verifier that has no character offsets can send ``span_text`` and the
    first occurrence in the line is used; one that sends neither is scored on
    the whole line. Unknown severities are an error, not a silent "none".
    """
    lines = script.get("lines", {})
    out: list[dict[str, Any]] = []
    for item in raw:
        severity = item.get("severity")
        if severity not in ALL_SEVERITIES:
            raise ValueError(f"prediction severity {severity!r} on line {item.get('line_index')}")
        line = int(item["line_index"])
        start, end = item.get("span_start"), item.get("span_end")
        if start is None or end is None:
            text = lines.get(str(line))
            span_text = item.get("span_text") or ""
            found = text.find(span_text) if text is not None and span_text else -1
            if found >= 0:
                start, end = found, found + len(span_text)
            else:
                start, end = 0, len(text) if text else 1
        start, end = int(start), int(end)
        if end <= start:
            end = start + 1
        kept = dict(item)
        kept.update(line_index=line, span_start=start, span_end=end, severity=severity)
        out.append(kept)
    return out


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    shared = min(a_end, b_end) - max(a_start, b_start)
    if shared <= 0:
        return False
    return shared / min(a_end - a_start, b_end - b_start) >= OVERLAP


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% interval. The calibration set is small and a reader should see how small."""
    if total == 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class _Tally:
    tp_recall: int = 0
    gold: int = 0
    tp_precision: int = 0
    predicted: int = 0

    def metrics(self) -> dict[str, Any]:
        precision = _ratio(self.tp_precision, self.predicted)
        recall = _ratio(self.tp_recall, self.gold)
        interval = wilson(self.tp_recall, self.gold)
        return {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "labels": self.gold,
            "predictions": self.predicted,
            "true_positive_labels": self.tp_recall,
            "true_positive_predictions": self.tp_precision,
            "recall_95ci": list(interval) if interval else None,
        }


def _severity_rank(name: str) -> int:
    return {"none": 0, "low": 1, "medium": 2, "high": 3}[name]


def score(
    data: dict[str, Any],
    predictions: dict[str, list[dict[str, Any]]],
    *,
    extras: str = "penalize",
    scripts: set[str] | None = None,
) -> dict[str, Any]:
    """Compare predictions to labels. ``predictions`` is keyed by script id.

    ``extras`` is ``penalize`` (a flag inside a covered window that matches no
    label is a false positive) or ``ignore`` (only predictions that overlap a
    label are judged, which measures severity agreement but not noise).
    """
    if extras not in {"penalize", "ignore"}:
        raise ValueError("extras must be 'penalize' or 'ignore'")
    classes = list(SEVERITIES) + ["any"]
    total = {name: _Tally() for name in classes}
    per_script: dict[str, dict[str, _Tally]] = {}
    confusion = {gold: {pred: 0 for pred in ALL_SEVERITIES + ("missed",)} for gold in ALL_SEVERITIES}
    misses: list[dict[str, Any]] = []
    false_alarms: list[dict[str, Any]] = []
    unscored = 0
    scored_predictions = 0

    for script in data["scripts"]:
        sid = script["script_id"]
        if scripts is not None and sid not in scripts:
            continue
        if sid not in predictions:
            raise ValueError(f"no predictions for script {sid}")
        preds = normalise_predictions(predictions[sid], script)
        coverage = script["coverage"]
        labels = script["labels"]
        tallies = per_script.setdefault(sid, {name: _Tally() for name in classes})

        in_window = [p for p in preds if _in_coverage(coverage, p["line_index"])]
        unscored += len(preds) - len(in_window)
        scored_predictions += len(in_window)

        def hits(label: dict[str, Any], pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                p for p in pool
                if p["line_index"] == label["line_index"]
                and _overlaps(label["span_start"], label["span_end"], p["span_start"], p["span_end"])
            ]

        for label in labels:
            matched = hits(label, in_window)
            gold = label["severity"]
            best = max((p["severity"] for p in matched), key=_severity_rank, default=None)
            confusion[gold][best if best is not None else "missed"] += 1
            if gold in FLAGGED:
                for name in (gold, "any"):
                    for bucket in (total[name], tallies[name]):
                        bucket.gold += 1
                got = gold in {p["severity"] for p in matched}
                got_any = any(p["severity"] in FLAGGED for p in matched)
                if got:
                    total[gold].tp_recall += 1
                    tallies[gold].tp_recall += 1
                else:
                    misses.append(_describe_miss(sid, script, label, best))
                if got_any:
                    total["any"].tp_recall += 1
                    tallies["any"].tp_recall += 1

        for pred in in_window:
            sev = pred["severity"]
            if sev not in FLAGGED:
                continue
            near = [
                lab for lab in labels
                if lab["line_index"] == pred["line_index"]
                and _overlaps(lab["span_start"], lab["span_end"], pred["span_start"], pred["span_end"])
            ]
            if not near and extras == "ignore":
                continue
            exact = any(lab["severity"] == sev for lab in near)
            flagged_gold = any(lab["severity"] in FLAGGED for lab in near)
            for name, ok in ((sev, exact), ("any", flagged_gold)):
                for bucket in (total[name], tallies[name]):
                    bucket.predicted += 1
                    if ok:
                        bucket.tp_precision += 1
            if not exact:
                false_alarms.append(_describe_false_alarm(sid, script, pred, near))

    return {
        "report_version": REPORT_VERSION,
        "extras": extras,
        "overlap": OVERLAP,
        "metrics": {name: total[name].metrics() for name in classes},
        "per_script": {
            sid: {name: tally.metrics() for name, tally in tallies.items()}
            for sid, tallies in per_script.items()
        },
        "confusion": confusion,
        "scored_predictions": scored_predictions,
        "unscored_predictions": unscored,
        "misses": misses,
        "false_alarms": false_alarms,
    }


def pool_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool several scored runs of the same verifier into one report.

    Counts are added and the ratios recomputed, so a run that flags nothing
    does not average a 0 into a precision it never had. The misses and false
    alarms are kept per run (``run`` is the 1-based run number).
    """
    if not reports:
        raise ValueError("no reports to pool")
    if len(reports) == 1:
        return reports[0]

    def merge(group: list[dict[str, Any]]) -> dict[str, Any]:
        tally = _Tally()
        for item in group:
            tally.tp_recall += item["true_positive_labels"]
            tally.gold += item["labels"]
            tally.tp_precision += item["true_positive_predictions"]
            tally.predicted += item["predictions"]
        return tally.metrics()

    pooled = dict(reports[0])
    pooled["runs"] = len(reports)
    pooled["metrics"] = {
        name: merge([r["metrics"][name] for r in reports]) for name in reports[0]["metrics"]
    }
    pooled["per_script"] = {
        sid: {name: merge([r["per_script"][sid][name] for r in reports]) for name in groups}
        for sid, groups in reports[0]["per_script"].items()
    }
    pooled["confusion"] = {
        gold: {pred: sum(r["confusion"][gold][pred] for r in reports) for pred in row}
        for gold, row in reports[0]["confusion"].items()
    }
    pooled["scored_predictions"] = sum(r["scored_predictions"] for r in reports)
    pooled["unscored_predictions"] = sum(r["unscored_predictions"] for r in reports)
    pooled["misses"] = [dict(m, run=i + 1) for i, r in enumerate(reports) for m in r["misses"]]
    pooled["false_alarms"] = [dict(f, run=i + 1) for i, r in enumerate(reports) for f in r["false_alarms"]]
    return pooled


def _describe_miss(
    sid: str, script: dict[str, Any], label: dict[str, Any], got: str | None,
) -> dict[str, Any]:
    return {
        "script_id": sid,
        "title": script.get("title"),
        "label_id": label["id"],
        "line_index": label["line_index"],
        "claim": label["claim"],
        "gold": label["severity"],
        "got": got or "not extracted",
    }


def _describe_false_alarm(
    sid: str, script: dict[str, Any], pred: dict[str, Any], near: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "script_id": sid,
        "title": script.get("title"),
        "line_index": pred["line_index"],
        "claim": pred.get("claim") or pred.get("span_text") or "",
        "predicted": pred["severity"],
        "gold": sorted({lab["severity"] for lab in near}) or ["no label"],
        "label_ids": [lab["id"] for lab in near],
    }


# ---------------------------------------------------------------------------
# Baseline and gate
# ---------------------------------------------------------------------------


def _metric(report: dict[str, Any], dotted: str) -> float | None:
    group, _, name = dotted.partition(".")
    return report["metrics"][group][name]


def make_baseline(
    data: dict[str, Any],
    report: dict[str, Any],
    *,
    config: dict[str, Any],
    cost_usd: float,
    created: str,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    summary = {
        name: {key: report["metrics"][name][key] for key in (
            "precision", "recall", "f1", "labels", "predictions",
            "true_positive_labels", "true_positive_predictions",
        )}
        for name in report["metrics"]
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created": created,
        "dataset_sha256": dataset_sha256(data),
        "dataset_labels": dataset_summary(data)["labels"],
        "extras": report["extras"],
        "config": config,
        "cost_usd": cost_usd,
        "gate": {"metrics": list(GATE_METRICS), "tolerance": tolerance},
        "metrics": summary,
    }


@dataclass
class GateResult:
    passed: bool
    lines: list[str]
    stale: bool = False


def check_gate(
    data: dict[str, Any],
    report: dict[str, Any],
    baseline: dict[str, Any],
    *,
    tolerance: float | None = None,
) -> GateResult:
    """Compare a run to the committed baseline. Lower than baseline - tolerance fails."""
    lines: list[str] = []
    if baseline.get("dataset_sha256") != dataset_sha256(data):
        return GateResult(
            False,
            [
                "The baseline was measured on a different version of the calibration set.",
                "Metrics on different labels are not comparable. Re-measure with",
                "--write-baseline on the unchanged verifier, review the diff, and commit it.",
            ],
            stale=True,
        )
    if baseline.get("extras") != report["extras"]:
        return GateResult(
            False,
            [f"baseline used extras={baseline.get('extras')!r}, this run used {report['extras']!r}"],
            stale=True,
        )
    gate = baseline.get("gate", {})
    slack = gate.get("tolerance", DEFAULT_TOLERANCE) if tolerance is None else tolerance
    passed = True
    for dotted in gate.get("metrics", list(GATE_METRICS)):
        now = _metric(report, dotted)
        then = baseline["metrics"][dotted.partition(".")[0]][dotted.partition(".")[2]]
        if then is None:
            lines.append(f"{dotted}: baseline has no value (zero labels or predictions); cannot gate")
            passed = False
            continue
        if now is None:
            lines.append(f"{dotted}: FAIL no value this run (baseline {then:.3f})")
            passed = False
            continue
        floor = then - slack
        ok = now >= floor - 1e-9
        passed = passed and ok
        lines.append(
            f"{dotted}: {'ok  ' if ok else 'FAIL'} {now:.3f} vs baseline {then:.3f} "
            f"(floor {max(floor, 0):.3f}, tolerance {slack:.3f})"
        )
    return GateResult(passed, lines)


def format_report(report: dict[str, Any], data: dict[str, Any]) -> str:
    """Plain-text table of per-severity precision and recall."""
    def cell(value: float | None) -> str:
        return "  n/a" if value is None else f"{value:5.2f}"

    out = [
        f"{'severity':<9}{'precision':>10}{'recall':>8}{'f1':>7}{'labels':>8}{'flags':>7}  recall 95% CI",
    ]
    for name in list(SEVERITIES) + ["any"]:
        m = report["metrics"][name]
        ci = m["recall_95ci"]
        span = f"{ci[0]:.2f}-{ci[1]:.2f}" if ci else "-"
        out.append(
            f"{name:<9}{cell(m['precision']):>10}{cell(m['recall']):>8}{cell(m['f1']):>7}"
            f"{m['labels']:>8}{m['predictions']:>7}  {span}"
        )
    titles = {s["script_id"]: s.get("title", "") for s in data["scripts"]}
    out.append("")
    out.append("per script (high precision / recall, any precision / recall):")
    for sid, metrics in report["per_script"].items():
        h, a = metrics["high"], metrics["any"]
        out.append(
            f"  {sid[:8]} {titles.get(sid, '')[:44]:<44} "
            f"high {cell(h['precision'])}/{cell(h['recall'])}  any {cell(a['precision'])}/{cell(a['recall'])}"
        )
    out.append("")
    out.append(
        f"scored predictions {report['scored_predictions']}, "
        f"outside the labelled windows {report['unscored_predictions']}"
    )
    return "\n".join(out)
