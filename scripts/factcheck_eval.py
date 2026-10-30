#!/usr/bin/env python3
"""Measure the fact-checker against the hand-labelled calibration set.

Run this before changing a fact-check prompt, model route or severity rule,
and again after. See docs/FACTCHECK_EVAL.md. One mode per run:

    --dry-run           validate the dataset and print the plan. Offline: no
                        database, no key, no model call. This is what CI runs.
    --check-sources     read-only database check that every label still matches
                        its stored script and quotes the paper verbatim. Free.
    --from-stored       score the fact checks already stored for the labelled
                        scripts. Free.
    --predictions FILE  score any verifier's output, offline. FILE is JSON:
                        {"verifier": "name", "cost_usd": 0.0,
                         "predictions": {"<script_id>": [claim, ...]}}
                        where a claim has line_index, severity and span_start /
                        span_end or span_text.
    --verifier M:FN     run any verifier callable and score it:
                        FN(script_text, paper_text, chunks, context) ->
                        {"claims": [...], "cost_usd": float}
    --live              run the app's pipeline (the ModelRoute config) on every
                        labelled script. PAID. Never run by CI.
    --export-dismissals FILE
                        write the claims people dismissed, with their notes, as
                        candidates for new labels. Free.

Scoring modes then compare against evals/factcheck/baseline.json and exit
1 when serious-error (high severity) precision or recall falls below it by
more than the tolerance. Exit 2 means the run itself could not be trusted
(bad dataset, stale baseline, a failed model call), never a pass.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp_pillars.factcheck import evaluation as ev  # noqa: E402

DEFAULT_DATASET = ROOT / "evals" / "factcheck" / "calibration.json"
DEFAULT_BASELINE = ROOT / "evals" / "factcheck" / "baseline.json"
MIN_LABELS = 150
DEFAULT_BUDGET_USD = 4.0
# The models are not deterministic, so one run is too noisy to gate on (see
# docs/FACTCHECK_EVAL.md). Live runs pool this many passes unless told otherwise.
DEFAULT_LIVE_RUNS = 2

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_ERROR = 2


class RunError(Exception):
    """The run cannot produce a trustworthy score."""


# ---------------------------------------------------------------------------
# Sources of script text, paper text and stored checks
# ---------------------------------------------------------------------------


def _db():
    from nlp_pillars import db

    return db


def load_inputs(script: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    """Script text, paper text and chunks for one labelled script, verified.

    A label is about one exact script and paper text. If either hash moved the
    labels no longer describe what would be checked, so this refuses.
    """
    from nlp_pillars.factcheck.engine import sha256_text

    db = _db()
    sid = script["script_id"]
    row = db.get_podcast_script_by_id(sid)
    if row is None:
        raise RunError(f"script {sid} ({script['title']}) is not in the database")
    if sha256_text(row.script) != script["script_sha256"]:
        raise RunError(
            f"script {sid[:8]} changed since it was labelled (sha256 differs). "
            "Its labels no longer apply."
        )
    paper = db.get_stored_paper_text(row.paper_id)
    if not paper:
        raise RunError(f"no stored paper text for {row.paper_id}")
    if sha256_text(paper) != script["paper_text_sha256"]:
        raise RunError(
            f"paper text for {row.paper_id} changed since the script was labelled "
            "(sha256 differs). Its labels no longer apply."
        )
    return row.script, paper, db.list_paper_chunks(row.paper_id)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_check_sources(data: dict[str, Any]) -> list[str]:
    from nlp_pillars.factcheck.anchor import find_in_text

    problems: list[str] = []
    for script in data["scripts"]:
        tag = f"{script['script_id'][:8]} {script['title'][:40]}"
        try:
            text, paper, _chunks = load_inputs(script)
        except RunError as exc:
            problems.append(f"{tag}: {exc}")
            continue
        lines = text.splitlines()
        for label in script["labels"]:
            line = label["line_index"]
            if line >= len(lines) or lines[line] != script["lines"][str(line)]:
                problems.append(f"{tag} {label['id']}: line {line} differs from the stored script")
            for item in label["evidence"]:
                found = find_in_text(paper, item["quote"])
                if found is None or found[3] < 100:
                    problems.append(
                        f"{tag} {label['id']}: quote is not verbatim in the paper: {item['quote'][:60]!r}"
                    )
    return problems


def predictions_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = (
        "line_index", "span_start", "span_end", "span_text", "claim", "verdict",
        "severity", "scope", "category", "error_type",
    )
    return [{key: row.get(key) for key in keep} for row in rows]


def mode_from_stored(data: dict[str, Any], wanted: set[str] | None) -> tuple[dict, dict]:
    db = _db()
    predictions: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, Any] = {"source": "stored", "checks": {}, "cost_usd": 0.0}
    for script in data["scripts"]:
        sid = script["script_id"]
        if wanted is not None and sid not in wanted:
            continue
        check = db.get_latest_podcast_fact_check(sid)
        if check is None:
            raise RunError(f"script {sid[:8]} ({script['title']}) has no stored fact check")
        if check.get("script_sha256") != script["script_sha256"]:
            raise RunError(f"stored check for {sid[:8]} was made on different script text")
        predictions[sid] = predictions_from_rows(check["claims"])
        meta["checks"][sid] = {
            "fact_check_id": check["id"],
            "prompt_version": check.get("prompt_version"),
            "models": {k: v for k, v in (check.get("models") or {}).items() if k != "calls"},
        }
    return predictions, meta


def mode_predictions_file(path: str) -> tuple[dict, dict]:
    with open(path, encoding="utf-8") as handle:
        blob = json.load(handle)
    if "predictions" not in blob or not isinstance(blob["predictions"], dict):
        raise RunError("predictions file needs a 'predictions' object keyed by script id")
    meta = {
        "source": "predictions",
        "verifier": blob.get("verifier", Path(path).name),
        "cost_usd": float(blob.get("cost_usd", 0.0)),
    }
    return blob["predictions"], meta


def _load_callable(spec: str):
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise RunError("--verifier must look like package.module:function")
    return getattr(importlib.import_module(module_name), attr)


def mode_verifier(
    data: dict[str, Any], spec: str, wanted: set[str] | None, budget: float,
) -> tuple[dict, dict]:
    fn = _load_callable(spec)
    predictions: dict[str, list[dict[str, Any]]] = {}
    spent = 0.0
    for script in data["scripts"]:
        sid = script["script_id"]
        if wanted is not None and sid not in wanted:
            continue
        if spent >= budget:
            raise RunError(f"budget ${budget:.2f} reached after ${spent:.4f}; stopped before {sid[:8]}")
        text, paper, chunks = load_inputs(script)
        out = fn(text, paper, chunks, {"script_id": sid, "paper_id": script["paper_id"], "remaining_usd": budget - spent})
        predictions[sid] = list(out["claims"])
        spent += float(out.get("cost_usd", 0.0))
    return predictions, {"source": "verifier", "verifier": spec, "cost_usd": round(spent, 6)}


def parse_models(text: str | None) -> dict[str, str]:
    """``extract=claude-haiku-5-5,rejudge=claude-sonnet-5-5`` over the route defaults."""
    from nlp_pillars.factcheck.engine import route_models

    models = route_models()
    for part in filter(None, (text or "").split(",")):
        stage, _, model = part.partition("=")
        if stage not in models or not model:
            raise RunError(f"--models: expected extract|screen|rejudge=<model id>, got {part!r}")
        models[stage] = model.strip()
    return models


def mode_live(
    data: dict[str, Any], models: dict[str, str], wanted: set[str] | None, budget: float,
    runs: int = 1,
) -> tuple[list[dict], dict]:
    """``runs`` full passes over the labelled scripts. Returns one prediction set per pass."""
    from nlp_pillars.factcheck.claude import make_client
    from nlp_pillars.factcheck.constants import PROMPT_VERSION
    from nlp_pillars.factcheck.engine import run_factcheck
    from nlp_pillars.factcheck.errors import FactcheckBudgetError, FactcheckError

    client = make_client()
    inputs = {
        script["script_id"]: load_inputs(script)
        for script in data["scripts"]
        if wanted is None or script["script_id"] in wanted
    }
    passes: list[dict[str, list[dict[str, Any]]]] = []
    spent = 0.0
    per_script: dict[str, float] = {}
    failures: list[dict[str, Any]] = []
    last_error = ""
    for number in range(1, runs + 1):
        predictions: dict[str, list[dict[str, Any]]] = {}
        for script in data["scripts"]:
            sid = script["script_id"]
            if sid not in inputs:
                continue
            text, paper, chunks = inputs[sid]
            print(f"  run {number}/{runs}: {sid[:8]} {script['title'][:50]} ...", flush=True)
            result = None
            for attempt in (1, 2):
                try:
                    result = run_factcheck(
                        text, paper, chunks, client,
                        script_id=sid, paper_id=script["paper_id"],
                        models=models, budget_usd=max(budget - spent, 0.0),
                    )
                    break
                except FactcheckBudgetError as exc:
                    spent += exc.cost_usd
                    raise RunError(f"{exc} (spent ${spent:.4f} before stopping)") from exc
                except FactcheckError as exc:
                    spent = round(spent + exc.cost_usd, 6)
                    print(f"    attempt {attempt} failed: {str(exc).splitlines()[0][:140]}", flush=True)
                    last_error = str(exc).splitlines()[0][:300]
            if result is None:
                # The product saves nothing when a check fails, so a script it
                # cannot check yields no flags. Scoring it as that (and saying
                # so) keeps one flaky call from making the gate unusable, and
                # keeps a verifier that fails often from looking good.
                failures.append({"run": number, "script_id": sid, "error": last_error})
                predictions[sid] = []
                continue
            spent = round(spent + result.cost_usd, 6)
            per_script[sid] = round(per_script.get(sid, 0.0) + result.cost_usd, 6)
            predictions[sid] = predictions_from_rows(result.claims)
        passes.append(predictions)
    meta = {
        "source": "live",
        "models": models,
        "prompt_version": PROMPT_VERSION,
        "runs": runs,
        "cost_usd": spent,
        "cost_by_script": per_script,
        "failed_checks": failures,
    }
    return passes, meta


def mode_export_dismissals(path: str) -> int:
    """Write dismissed claims with their notes as candidates for new labels."""
    db = _db()
    client = db.get_client()
    reviews = (
        client.table("podcast_claim_reviews").select("*").eq("action", "dismissed")
        .order("created_at").execute()
    )
    if reviews["error"]:
        raise RunError(f"could not read podcast_claim_reviews: {reviews['error']}. Apply {db.DISMISSAL_MIGRATION}")
    claim_ids = sorted({row["claim_id"] for row in reviews["data"] or []})
    out: list[dict[str, Any]] = []
    for claim_id in claim_ids:
        claim = client.table("podcast_claims").select("*").eq("id", claim_id).limit(1).execute()["data"]
        if not claim or not claim[0].get("dismissed_at"):
            continue  # restored since: no longer a disagreement
        claim = claim[0]
        check = client.table("podcast_fact_checks").select("script_id,script_sha256").eq(
            "id", claim["fact_check_id"]).limit(1).execute()["data"][0]
        out.append({
            "script_id": check["script_id"],
            "script_sha256": check["script_sha256"],
            "claim_id": claim_id,
            "line_index": claim["line_index"],
            "span_start": claim["span_start"],
            "span_end": claim["span_end"],
            "span_text": claim["span_text"],
            "claim": claim["claim"],
            "verifier_said": {k: claim[k] for k in ("verdict", "severity", "error_type", "scope", "category")},
            "dismissed_at": claim["dismissed_at"],
            "dismissed_note": claim["dismissed_note"],
            "todo": "Read the paper, decide the true verdict, then add a label to calibration.json. "
                    "A dismissal says the flag was wrong; it does not say what is right.",
        })
    Path(path).write_text(json.dumps({"candidates": out}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} candidate(s) to {path}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_details(report: dict[str, Any], limit: int = 15) -> None:
    for title, rows, fmt in (
        ("missed (labelled flag the verifier did not raise at that severity)", report["misses"],
         lambda r: f"  {r['script_id'][:8]} L{r['line_index']} gold {r['gold']}, got {r['got']}: {r['claim'][:90]}"),
        ("false alarms (flag with no label of that severity)", report["false_alarms"],
         lambda r: f"  {r['script_id'][:8]} L{r['line_index']} said {r['predicted']}, gold {'/'.join(r['gold'])}: {r['claim'][:90]}"),
    ):
        if not rows:
            continue
        print(f"\n{title}: {len(rows)}")
        for row in rows[:limit]:
            print(fmt(row))
        if len(rows) > limit:
            print(f"  ... {len(rows) - limit} more (use --out for all)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="validate the dataset offline")
    mode.add_argument("--check-sources", action="store_true", help="verify labels against the stored scripts and papers")
    mode.add_argument("--from-stored", action="store_true", help="score the stored fact checks")
    mode.add_argument("--predictions", metavar="FILE", help="score a predictions file")
    mode.add_argument("--verifier", metavar="MODULE:FN", help="run and score another verifier")
    mode.add_argument("--live", action="store_true", help="run the app's pipeline (PAID)")
    mode.add_argument("--export-dismissals", metavar="FILE", help="write dismissed claims as label candidates")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--models", help="override routes: extract=ID,screen=ID,rejudge=ID")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD, help="stop before spending more than this, all runs together (USD)")
    parser.add_argument("--runs", type=int, help=f"live passes to pool (default {DEFAULT_LIVE_RUNS}; the baseline records how many it used)")
    parser.add_argument("--scripts", help="comma-separated script id prefixes; implies --no-gate")
    parser.add_argument("--extras", choices=["penalize", "ignore"], default="penalize")
    parser.add_argument("--tolerance", type=float, help="override the baseline's tolerance")
    parser.add_argument("--no-gate", action="store_true", help="report only, always exit 0 on success")
    parser.add_argument("--write-baseline", action="store_true", help="write the baseline from this run")
    parser.add_argument("--out", metavar="FILE", help="write the full JSON report here")
    return parser


def select_scripts(data: dict[str, Any], prefixes: str | None) -> set[str] | None:
    if not prefixes:
        return None
    wanted: set[str] = set()
    for prefix in filter(None, (p.strip() for p in prefixes.split(","))):
        matches = [s["script_id"] for s in data["scripts"] if s["script_id"].startswith(prefix)]
        if len(matches) != 1:
            raise RunError(f"--scripts {prefix!r} matches {len(matches)} labelled scripts")
        wanted.add(matches[0])
    return wanted


def run(args: argparse.Namespace) -> int:
    data = ev.load_dataset(args.dataset)
    problems = ev.validate_dataset(data, min_labels=MIN_LABELS)
    if problems:
        print(str(ev.DatasetError(problems)), file=sys.stderr)
        return EXIT_ERROR
    summary = ev.dataset_summary(data)
    print(f"dataset {Path(args.dataset).name}: {summary['labels']} labels over {len(summary['scripts'])} scripts "
          f"({', '.join(f'{k} {v}' for k, v in summary['by_severity'].items())})")

    if args.dry_run:
        from nlp_pillars.factcheck.engine import route_models

        print("dataset is valid (offline checks only).")
        print(f"live run would use: {route_models()}")
        print(f"sha256 {ev.dataset_sha256(data)[:16]}...  baseline: "
              f"{'present' if Path(args.baseline).exists() else 'MISSING'}")
        print("A live run costs about the 'cost_usd' recorded in the baseline; nothing was called.")
        return EXIT_OK
    if args.export_dismissals:
        return mode_export_dismissals(args.export_dismissals)

    wanted = select_scripts(data, args.scripts)
    subset = wanted is not None
    if args.check_sources:
        found = mode_check_sources(data)
        for line in found:
            print("PROBLEM", line)
        print("sources match." if not found else f"{len(found)} problem(s).")
        return EXIT_ERROR if found else EXIT_OK

    if args.from_stored:
        predictions, meta = mode_from_stored(data, wanted)
    elif args.predictions:
        predictions, meta = mode_predictions_file(args.predictions)
    elif args.verifier:
        predictions, meta = mode_verifier(data, args.verifier, wanted, args.budget)
    else:
        models = parse_models(args.models)
        runs = args.runs or DEFAULT_LIVE_RUNS
        if runs < 1:
            raise RunError("--runs must be at least 1")
        print(f"live run x{runs}, models {models}, budget ${args.budget:.2f}")
        predictions, meta = mode_live(data, models, wanted, args.budget, runs)

    passes = predictions if isinstance(predictions, list) else [predictions]
    report = ev.pool_reports([
        ev.score(data, one, extras=args.extras, scripts=wanted) for one in passes
    ])
    report["run"] = meta
    print()
    print(ev.format_report(report, data))
    print_details(report)
    for failure in meta.get("failed_checks", []):
        print(f"\nFAILED CHECK (scored as no flags): run {failure['run']} {failure['script_id'][:8]}: {failure['error']}")
    print(f"\ncost this run: ${meta.get('cost_usd', 0.0):.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.write_baseline:
        if subset:
            raise RunError("--write-baseline needs every labelled script, not --scripts")
        baseline = ev.make_baseline(
            data, report, config=meta, cost_usd=float(meta.get("cost_usd", 0.0)),
            created=datetime.date.today().isoformat(),
            tolerance=args.tolerance if args.tolerance is not None else ev.DEFAULT_TOLERANCE,
        )
        Path(args.baseline).write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(f"wrote baseline {args.baseline}")
        return EXIT_OK

    if args.no_gate or subset:
        print("\n(gate skipped)")
        return EXIT_OK
    if not Path(args.baseline).exists():
        raise RunError(f"no baseline at {args.baseline}. Measure one with --write-baseline.")
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    gate = ev.check_gate(data, report, baseline, tolerance=args.tolerance)
    print("\ngate vs", Path(args.baseline).name)
    for line in gate.lines:
        print(" ", line)
    if gate.stale:
        return EXIT_ERROR
    print("GATE PASSED" if gate.passed else "GATE FAILED: serious-error precision or recall dropped")
    return EXIT_OK if gate.passed else EXIT_REGRESSION


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (RunError, ev.DatasetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001
        # Exit 1 means "serious-error recall or precision regressed". A crash
        # (database down, API error) must never look like that, or like a pass.
        print(f"error: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
