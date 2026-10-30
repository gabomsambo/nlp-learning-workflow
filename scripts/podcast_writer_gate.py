#!/usr/bin/env python3
"""Measure the evidence writer against the free writer. Paid; never run by CI.

Both arms write a script for the same papers, the same number of times, and the
app's own fact-check engine (the one `scripts/factcheck_eval.py` calibrates)
checks every script against the same paper text. Then every serious flag is
read against the paper and labelled by hand, blind to the arm. The criteria are
in `nlp_pillars/evidence/gate.py` and docs/PODCAST_EVIDENCE_WRITER.md.

    plan      offline: papers, samples and a spend estimate. Calls nothing.
    prepare   PAID (cents): read each paper's table pages once and keep the
              repaired text and table rows in OUT/inputs. Writes nothing to the
              database: the library's stored text is the captain's, and the
              calibration labels are pinned to its hash.
    generate  PAID: one script per paper x sample x arm, through the real
              PodcastAgent. `--arm` and `--limit` narrow it.
    check     PAID: the production fact-check on every script that has none.
    flags     free: write OUT/flags_blind.md (serious flags, shuffled, arm
              hidden) for hand labelling, and OUT/flags_key.json.
    report    free: metrics and the gate, from OUT and a labels file
              ({"<flag id>": "real" | "minor" | "wrong"}).

Everything lands in OUT (outside the repository), one JSON per script, so a
stopped run resumes where it stopped. Spend is added up in OUT/spend.json and
every paid step refuses to start once --budget is reached.

The Anthropic key is read from --env-file at run time, that one variable only;
it is never printed or copied. The database is the local compose stack:

    SUPABASE_URL=http://localhost:3000 POSTGREST_URL=http://localhost:3000 \\
    SUPABASE_KEY=<LOCAL_API_TOKEN from docker-compose.yml> \\
    python scripts/podcast_writer_gate.py generate --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ENV_FILE = "/home/gabo/firstmate/projects/nlp-learning-workflow/.env"
DEFAULT_BUDGET_USD = 7.0
SAMPLES = 2
ARMS = ("free", "evidence")
#: (library paper id, PDF url when the stored one is not fetchable).
DEFAULT_PAPERS = [
    ("file:8e5131670980", "https://arxiv.org/pdf/1607.01759"),  # Bag of Tricks (fastText)
    ("arxiv:1905.05950", None),   # BERT Rediscovers the Classical NLP Pipeline
    ("arxiv:1706.03762", None),   # Attention Is All You Need
    ("arxiv:1508.07909", None),   # Subword units for NMT
    ("arxiv:1909.01185", None),   # automated feature engineering, credit cards
    ("1108.5096", None),          # Minimalist Grammars
    ("2004.03061", None),         # Information-theoretic probing
    ("arxiv:2306.16424", None),   # AMLworld (tables the extractor loses)
]
#: Rough dollars per script, for `plan` only. Measured by the research report.
ESTIMATE = {"free": (0.10, 0.12), "evidence": (0.08, 0.05)}  # (generate, check)


class GateError(Exception):
    pass


def slug(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", paper_id).strip("_")


def load_key(env_file: str) -> None:
    """Put ANTHROPIC_API_KEY, and only that, into this process's environment."""
    # The free arm's Ground Pack runs on Claude Haiku in this comparison;
    # DeepSeek is a different provider and is not part of this measurement.
    os.environ.setdefault("PODCAST_EXTRACTION_PROVIDER", "anthropic")
    if os.environ.get("ANTHROPIC_API_KEY", "").startswith("test-key"):
        del os.environ["ANTHROPIC_API_KEY"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    from dotenv import dotenv_values

    value = dotenv_values(env_file).get("ANTHROPIC_API_KEY")
    if not value:
        raise GateError(f"no ANTHROPIC_API_KEY in {env_file}")
    os.environ["ANTHROPIC_API_KEY"] = value


# -- OUT directory ----------------------------------------------------------------


class Out:
    def __init__(self, path: str):
        self.root = Path(path)
        for sub in ("inputs", "runs", "pdfs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def input_path(self, paper_id: str) -> Path:
        return self.root / "inputs" / f"{slug(paper_id)}.json"

    def run_path(self, paper_id: str, arm: str, sample: int) -> Path:
        return self.root / "runs" / f"{slug(paper_id)}__{arm}__{sample}.json"

    def load_inputs(self) -> dict[str, dict[str, Any]]:
        found = {}
        for path in sorted((self.root / "inputs").glob("*.json")):
            data = json.loads(path.read_text())
            found[data["paper_id"]] = data
        return found

    def load_runs(self) -> list[dict[str, Any]]:
        return [json.loads(p.read_text()) for p in sorted((self.root / "runs").glob("*.json"))]

    # -- spend ----------------------------------------------------------------
    @property
    def spend_path(self) -> Path:
        return self.root / "spend.json"

    def spent(self) -> float:
        if not self.spend_path.exists():
            return 0.0
        return float(json.loads(self.spend_path.read_text())["total"])

    def charge(self, tag: str, usd: float) -> None:
        data = {"total": 0.0, "entries": []}
        if self.spend_path.exists():
            data = json.loads(self.spend_path.read_text())
        data["entries"].append({"tag": tag, "usd": round(usd, 6), "at": int(time.time())})
        data["total"] = round(sum(e["usd"] for e in data["entries"]), 6)
        self.spend_path.write_text(json.dumps(data, indent=1))

    def require_budget(self, budget: float, about_to_spend: float = 0.0) -> None:
        if self.spent() + about_to_spend > budget:
            raise GateError(
                f"budget ${budget:.2f} would be exceeded (spent ${self.spent():.4f}, "
                f"next step about ${about_to_spend:.2f}); stopping"
            )


# -- commands ----------------------------------------------------------------------


def cmd_plan(args, out: Out) -> int:
    papers = [p for p, _ in DEFAULT_PAPERS]
    total = 0.0
    print(f"{len(papers)} papers x {args.samples} samples x {len(ARMS)} arms = "
          f"{len(papers) * args.samples * len(ARMS)} scripts")
    for arm in ARMS:
        gen, chk = ESTIMATE[arm]
        sub = len(papers) * args.samples * (gen + chk)
        total += sub
        print(f"  {arm:9s} generate ~${gen:.2f} + check ~${chk:.2f} per script -> ~${sub:.2f}")
    print("  prepare   ~$0.05 (table pages, Haiku 5.5)")
    print(f"estimated total ~${total + 0.05:.2f}; budget ${args.budget:.2f}; spent so far ${out.spent():.4f}")
    return 0


def cmd_prepare(args, out: Out) -> int:
    from nlp_pillars import db, table_repair
    from nlp_pillars.tools.pdf_loader import download_pdf

    for paper_id, url_override in DEFAULT_PAPERS:
        path = out.input_path(paper_id)
        if path.exists() and not args.force:
            print(f"{paper_id}: already prepared")
            continue
        out.require_budget(args.budget, 0.02)
        paper = db.get_paper_by_id(paper_id)
        text = db.get_stored_paper_text(paper_id) if paper else None
        if not paper or not text:
            raise GateError(f"{paper_id}: not in the library with stored text")
        pdf = download_pdf(url_override or paper.url_pdf, cache_dir=str(out.root / "pdfs"))
        pages = table_repair.PymupdfPages(pdf)
        try:
            repair = table_repair.repair_tables(pages, text)
        finally:
            pages.close()
        out.charge(f"prepare {paper_id}", repair.cost_usd)
        path.write_text(json.dumps({
            "paper_id": paper_id, "title": paper.title, "pillar_id": paper.pillar_id,
            "stored_chars": len(text), "text": repair.text, "tables": repair.rows_for_storage(),
            "detail": repair.detail(), "cost_usd": repair.cost_usd,
        }, ensure_ascii=False))
        print(f"{paper_id}: {repair.detail()}  ({len(text):,} -> {len(repair.text):,} chars)")
    print(f"spent ${out.spent():.4f}")
    return 0


def _gate_agent(inputs: dict[str, dict[str, Any]], arm: str):
    from nlp_pillars.agents.podcast_agent import FullTextResult, PodcastAgent

    class GateAgent(PodcastAgent):
        """The real agent, reading the repaired text from OUT instead of the database."""

        def _get_full_text(self, paper):
            return FullTextResult(inputs[paper.id]["text"])

        def _get_paper_tables(self, paper_id):
            return inputs[paper_id]["tables"]

    return GateAgent(writer_mode=arm)


async def _generate_one(out: Out, inputs, paper_id: str, arm: str, sample: int) -> float:
    from nlp_pillars.evidence.gate import spoken_words

    agent = _gate_agent(inputs, arm)
    started = time.time()
    script = await agent.generate(paper_id, inputs[paper_id]["pillar_id"])
    synth = agent.last_synthesis_call
    synth_cost = float(synth.cost_usd or 0.0) if synth else 0.0
    if arm == "evidence":
        pre = float(script.ledger.get("cost_usd", 0.0))
    else:
        pre = sum(float(c.cost_usd or 0.0) for c in script.ground_pack_calls.values())
    record = {
        "paper": paper_id, "arm": arm, "sample": sample,
        "script": script.script, "words": spoken_words(script.script),
        "writer_mode": script.writer_mode,
        "line_citations": script.line_citations,
        "ledger": script.ledger,
        "stats": script.ledger.get("stats", {}) if arm == "evidence" else {},
        "source_warnings": script.source_material.warnings,
        "cost": {"generate": round(pre + synth_cost, 6), "pre": round(pre, 6), "script": round(synth_cost, 6)},
        "seconds": round(time.time() - started, 1),
    }
    out.run_path(paper_id, arm, sample).write_text(json.dumps(record, ensure_ascii=False, indent=1))
    out.charge(f"generate {paper_id} {arm} {sample}", record["cost"]["generate"])
    return record["cost"]["generate"]


def cmd_generate(args, out: Out) -> int:
    inputs = out.load_inputs()
    arms = [args.arm] if args.arm else list(ARMS)
    todo = [
        (p, arm, s)
        for s in range(1, args.samples + 1)
        for p, _ in DEFAULT_PAPERS
        for arm in arms
        if not out.run_path(p, arm, s).exists()
    ]
    if args.limit:
        todo = todo[: args.limit]
    missing = sorted({p for p, _a, _s in todo if p not in inputs})
    if missing:
        raise GateError(f"run `prepare` first: no inputs for {', '.join(missing)}")
    print(f"{len(todo)} script(s) to write")

    async def run_all():
        gate = asyncio.Semaphore(args.parallel)
        failures: list[str] = []

        async def one(task):
            paper_id, arm, sample = task
            async with gate:
                try:
                    out.require_budget(args.budget, 0.25)
                    cost = await _generate_one(out, inputs, paper_id, arm, sample)
                    print(f"  {paper_id} {arm} #{sample}: ${cost:.4f}  (total ${out.spent():.4f})", flush=True)
                except Exception as exc:  # noqa: BLE001 — one script must not stop the rest
                    failures.append(f"{paper_id} {arm} #{sample}: {str(exc).splitlines()[0][:200]}")
                    print(f"  FAILED {failures[-1]}", flush=True)

        await asyncio.gather(*(one(t) for t in todo))
        return failures

    failures = asyncio.run(run_all())
    print(f"spent ${out.spent():.4f}; {len(failures)} failure(s)")
    return 1 if failures else 0


def _trim_claim(row: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "line_index", "span_start", "span_end", "span_text", "claim", "scope", "category",
        "verdict", "error_type", "severity", "rationale", "source_says",
    )
    trimmed = {k: row.get(k) for k in keep}
    trimmed["quotes"] = [
        (e.get("quote") or e.get("text") or "")[:400] for e in (row.get("evidence") or [])
    ][:2]
    return trimmed


def cmd_check(args, out: Out) -> int:
    from nlp_pillars.factcheck.claude import make_client
    from nlp_pillars.factcheck.engine import run_factcheck

    inputs = out.load_inputs()
    client = make_client()
    todo = [p for p in sorted((out.root / "runs").glob("*.json"))
            if "check" not in json.loads(p.read_text())]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} script(s) to check")

    def one(path: Path):
        record = json.loads(path.read_text())
        key = f"{record['paper']} {record['arm']} #{record['sample']}"
        out.require_budget(args.budget, 0.3)
        result = run_factcheck(
            record["script"], inputs[record["paper"]]["text"], [], client,
            script_id=key, paper_id=record["paper"], budget_usd=0.8,
        )
        record["check"] = {
            "counts": result.counts, "cost_usd": result.cost_usd,
            "models": result.models, "prompt_version": result.prompt_version,
            "claims": [_trim_claim(c) for c in result.claims],
            "extraction": result.extraction,
        }
        record["counts"] = result.counts
        record["claims"] = record["check"]["claims"]
        record["cost"]["check"] = result.cost_usd
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
        out.charge(f"check {key}", result.cost_usd)
        print(f"  {key}: {result.counts['claims']} claims, "
              f"{result.counts['high']}H {result.counts['medium']}M {result.counts['low']}L, "
              f"${result.cost_usd:.4f}  (total ${out.spent():.4f})", flush=True)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for future in [pool.submit(one, p) for p in todo]:
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAILED {str(exc).splitlines()[0][:200]}", flush=True)
    print(f"spent ${out.spent():.4f}; {failures} failure(s)")
    return 1 if failures else 0


def _checked(out: Out):
    runs = [r for r in out.load_runs() if "check" in r]
    return ([r for r in runs if r["arm"] == "free"], [r for r in runs if r["arm"] == "evidence"])


def cmd_flags(args, out: Out) -> int:
    from nlp_pillars.evidence.gate import serious_flags

    free, evidence = _checked(out)
    scripts = {(r["paper"], r["arm"], r["sample"]): r["script"].splitlines() for r in [*free, *evidence]}
    flags = serious_flags([*free, *evidence])
    key = {f["fid"]: f"{f['paper']}|{f['arm']}|{f['sample']}" for f in flags}
    (out.root / "flags_key.json").write_text(json.dumps(key, indent=1))
    inputs = out.load_inputs()
    lines = [f"# {len(flags)} serious flags (high/medium), arm hidden, shuffled\n",
             "Label each: real (the line states something the paper does not support, "
             "contradicts or overstates materially), minor (right that it goes past the "
             "paper, not serious), wrong (the paper supports it).\n"]
    for f in flags:
        c = f["claim"]
        script = scripts[(f["paper"], f["arm"], f["sample"])]
        lines += [
            f"\n## {f['fid']}  [{inputs[f['paper']]['title'][:50]}]  {c['severity']}/{c['verdict']}/{c['error_type']}",
            f"LINE: {script[c['line_index']][:400]}",
            f"SPAN: {c['span_text']}",
            f"CLAIM: {c['claim']}",
            f"WHY: {c['rationale']}",
            f"PAPER SAYS: {c['source_says']}",
        ]
        lines += [f"QUOTE: {q}" for q in c.get("quotes", [])[:2]]
    (out.root / "flags_blind.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(flags)} flags to {out.root / 'flags_blind.md'}")
    return 0


def cmd_report(args, out: Out) -> int:
    from nlp_pillars.evidence import gate

    free, evidence = _checked(out)
    labels = json.loads(Path(args.labels).read_text()) if args.labels else {}
    result = gate.evaluate(free, evidence, labels)
    (out.root / "report.json").write_text(json.dumps(result, indent=1, default=str))
    f, e, comp = result["free"], result["evidence"], result["completeness"]

    def row(name, a, b, fmt="{:.2f}"):
        show = lambda v: "n/a" if v is None else fmt.format(v)  # noqa: E731
        print(f"  {name:44s} {show(a):>10s} {show(b):>10s}")

    print(f"scripts: free {f['scripts']}, evidence {e['scripts']}; papers {f['papers']}")
    print(f"  {'':44s} {'free':>10s} {'evidence':>10s}")
    row("spoken words", f["words"], e["words"], "{:,.0f}")
    row("claims", f["claims"], e["claims"], "{:,.0f}")
    row("high / medium / low flags", None, None)
    print(f"  {'  high':44s} {f['high']:>10d} {e['high']:>10d}")
    print(f"  {'  medium':44s} {f['medium']:>10d} {e['medium']:>10d}")
    print(f"  {'  low':44s} {f['low']:>10d} {e['low']:>10d}")
    row("all flags per 1,000 words", f["flags_per_1000_words"], e["flags_per_1000_words"])
    row("high+medium per 1,000 words (raw)", f["serious_per_1000_words"], e["serious_per_1000_words"])
    if labels:
        for name in ("flags", "labelled", "real", "minor", "wrong"):
            print(f"  {'  serious flags ' + name:44s} {f['labelled'][name]:>10d} {e['labelled'][name]:>10d}")
        row("precision of serious flags (real)", f["labelled"]["precision_strict"], e["labelled"]["precision_strict"], "{:.1%}")
        row("precision (real + minor)", f["labelled"]["precision_lenient"], e["labelled"]["precision_lenient"], "{:.1%}")
        row("CONFIRMED serious per 1,000 words", f["labelled"]["confirmed_per_1000_words"], e["labelled"]["confirmed_per_1000_words"])
    row("generation cost $", f["cost_generate"], e["cost_generate"], "{:.3f}")
    row("fact-check cost $", f["cost_check"], e["cost_check"], "{:.3f}")
    for name, diff in (("raw serious", result["raw_serious_difference"]),
                       ("confirmed serious", result["confirmed_serious_difference"])):
        if diff["mean"] is not None:
            print(f"paired {name} per 1,000 words, evidence minus free: {diff['mean']:+.2f} "
                  f"(95% CI {diff['lo']:+.2f} to {diff['hi']:+.2f}, {diff['papers']} papers)")
    print("completeness:", {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in comp.items() if k != "words_ratio_by_paper"})
    print("\ngate:")
    for check in result["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}")
    print("GATE PASSED" if result["passed"] else "GATE NOT PASSED")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "prepare", "generate", "check", "flags", "report"])
    parser.add_argument("--out", required=True, help="working directory, outside the repository")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--labels", help="JSON {flag id: real|minor|wrong} (report)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Out(args.out)
    try:
        if args.command in ("prepare", "generate", "check"):
            load_key(args.env_file)
        return {
            "plan": cmd_plan, "prepare": cmd_prepare, "generate": cmd_generate,
            "check": cmd_check, "flags": cmd_flags, "report": cmd_report,
        }[args.command](args, out)
    except GateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
