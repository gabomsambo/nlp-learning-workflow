"""Run the bridge-sentence pass on a file of paper pairs and print what it wrote.

This is the check that has to be passed *by a person* before bridge sentences
are shown unreviewed (report section 7, row M8). It runs the production path
(``map_enrich.enrich``) over pairs you supply, with the real model, and writes
a results file to read and label by hand:

    ANTHROPIC_API_KEY=... python scripts/eval_bridges.py items.json results.json

``items.json`` is a list of::

    {"id": "…", "a": {"label": "Vaswani 2017", "title": "…", "abstract": "…"},
     "b": {…}, "citing": "a" | "b" | null, "context": "the citing sentence",
     "intents": ["background"], "influential": false, "lesson_why": "optional"}

Not part of CI: it calls a paid model, one call per item, and stops at
``--max-cost``. ``--dry-run`` prices the pass and calls nothing. The items file
is data you fetched; this script does not fetch anything and the repository
does not ship abstracts.

Each result records the status the production checks gave it (``generated``
passed the quote and number checks; ``rejected`` failed them; ``abstained``
is the model declining), the sentence, the quotes, and the cost of the call,
so the human labels can be set against what the checks decided.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlp_pillars.connections import TypedEdge  # noqa: E402
from nlp_pillars.map_build import MapResult, sha256_of  # noqa: E402
from nlp_pillars.map_enrich import EnrichInputs, enrich  # noqa: E402


def payload_for(item: dict) -> tuple[MapResult, EnrichInputs]:
    """One item as the smallest map the production pass accepts."""
    a, b = f"{item['id']}:a", f"{item['id']}:b"
    nodes = [
        {"id": node_id, "kind": "library", "label": side["label"], "title": side["title"], "paper_ids": [node_id]}
        for node_id, side in ((a, item["a"]), (b, item["b"]))
    ]
    evidence = {}
    if item.get("context") and item.get("citing") in ("a", "b"):
        citing, cited = (a, b) if item["citing"] == "a" else (b, a)
        evidence["cites"] = {
            "citing": citing,
            "cited": cited,
            "influential": bool(item.get("influential")),
            "intents": list(item.get("intents") or []),
            "context": item["context"],
        }
    payload = {
        "version": 1,
        "scope": "eval",
        "pillar": {"id": "eval", "name": "eval"},
        "clustered": False,
        "nodes": nodes,
        "communities": {"regions": [], "topics": []},
        "evidence": [{"a": a, "b": b, "weight": 0.0, "evidence": evidence, "sentence": "t"}],
        "warnings": [],
    }
    result = MapResult("eval", payload, sha256_of(payload), "", {}, [])
    typed = []
    if item.get("lesson_why"):
        typed.append(
            TypedEdge(a, b, "builds_on", item["lesson_why"], item["id"], item["b"]["label"], "author_year", 1.0)
        )
    inputs = EnrichInputs(
        abstracts={a: item["a"]["abstract"], b: item["b"]["abstract"]}, typed_edges=typed
    )
    return result, inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("items", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-cost", type=float, default=1.0, help="Stop after this many dollars.")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N items.")
    parser.add_argument("--only", default="", help="Comma-separated item ids to run.")
    parser.add_argument("--dry-run", action="store_true", help="Price the pass; call nothing.")
    args = parser.parse_args(argv)

    items = json.loads(args.items.read_text())
    if args.only:
        wanted = set(args.only.split(","))
        items = [item for item in items if item["id"] in wanted]
    if args.limit:
        items = items[: args.limit]
    rows, spent, estimate = [], 0.0, 0.0
    for item in items:
        result, inputs = payload_for(item)
        report = enrich(
            result, inputs, use_model=True, dry_run=args.dry_run, budget_check=lambda reserve: None
        )
        estimate += report.estimated_cost_usd
        if args.dry_run:
            continue
        bridge = result.payload["bridges"][0]
        cost = report.calls.cost_usd
        spent += cost
        draft = report.rejected[0] if report.rejected else None
        rows.append({"id": item["id"], "cost_usd": cost, "draft": draft, **bridge})
        text = bridge.get("text") or (
            f"[{bridge['status']}] {bridge.get('reason', '')}"
            + (f" | draft: {draft['sentence']}" if draft else "")
        )
        print(f"{item['id']:>6} {bridge['status']:<10} ${cost:.6f}  {text}")
        if spent >= args.max_cost:
            print(f"stopped: spent ${spent:.4f} >= --max-cost {args.max_cost}", file=sys.stderr)
            break
    if args.dry_run:
        print(f"{len(items)} items, estimated ${estimate:.4f}")
        return 0
    args.out.write_text(json.dumps(rows, indent=1))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"{len(rows)} items: {counts}; spent ${spent:.6f}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
