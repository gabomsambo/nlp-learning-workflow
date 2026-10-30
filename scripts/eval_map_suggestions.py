"""Offline check of the paper map's suggestion scorer. Not run in CI.

Two protocols, both scored with the functions ``map build`` ships
(``nlp_pillars/map_build.py``), so this measures the code that runs:

``loo`` (default; the paper-map scout's protocol, report section 4.5)
    For each library work L whose pillar has another library work: drop L from
    the library, drop L's own reference list (it was fetched only because L was
    yours), rebuild the pillar's scope from the remaining works, add L to the
    candidate pool, and rank the pool from L's pillar-mates. Records L's rank.
    ``--keep-own-references`` keeps L's references, which is what the scout's
    prototype did (median rank 40.5 over 12 trials).

``hide-refs`` (the larger protocol the report recommends, Kucuktunc et al.)
    For each library work L, hide ``--hide-fraction`` of its references (at
    least one, fixed seed), and rank the pool from L's pillar library. Records
    the rank of every hidden reference. Hundreds of trials instead of a dozen.

Scorers reported: the shipped RRF(PPR, bge) and its two inputs alone.

Inputs come from the database (``--from-db``, read-only, no model, no
network beyond PostgREST) or from ``MapInputs.to_dict()`` JSON files
(``--inputs``), one per pillar or one holding the whole library.

    python scripts/eval_map_suggestions.py --from-db
    python scripts/eval_map_suggestions.py --inputs inputs/*.json --protocol hide-refs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp_pillars.map_build import (  # noqa: E402
    CitationIndex,
    MapInputs,
    VectorSpace,
    rank_pool,
    select_scope,
    walk_graph,
)

SCORERS = ("rrf", "ppr", "bge")
HIDE_FRACTION = 0.10
HIDE_SEED = 0


def load_inputs(args) -> MapInputs:
    """One MapInputs with the whole library. The pillar field is not used."""
    if args.from_db:
        from nlp_pillars.map_store import assemble_inputs, load_evidence

        evidence = load_evidence()
        any_pillar = sorted({paper.pillar_id for paper in evidence.library})[0]
        inputs, _ = assemble_inputs(
            evidence, any_pillar, with_queue=False, with_previous=False
        )
        # assemble_inputs fetched vectors for one pillar's scope; fetch them all.
        from nlp_pillars.map_store import load_vectors

        if evidence.model_id:
            inputs.vectors = load_vectors(
                evidence.model_id, evidence.library, evidence.works.keys()
            )
        return inputs
    merged: MapInputs | None = None
    for path in args.inputs:
        data = MapInputs.from_dict(json.loads(Path(path).read_text()))
        if merged is None:
            merged = data
            continue
        merged.works.update(data.works)
        merged.vectors.update(data.vectors)
        known = {(c.citing, c.cited) for c in merged.citations}
        merged.citations.extend(
            c for c in data.citations if (c.citing, c.cited) not in known
        )
        have = {(p.paper_id, p.pillar_id) for p in merged.library}
        merged.library.extend(
            p for p in data.library if (p.paper_id, p.pillar_id) not in have
        )
    if merged is None:
        sys.exit("No inputs: pass --from-db or --inputs FILE...")
    return merged


def ranks_for(
    pool: Sequence[str], seeds: Sequence[str], citations, space, scope_nodes
) -> Dict[str, Dict[str, int]]:
    """Rank ``pool`` from ``seeds`` with RRF and with each of its two inputs."""
    index = CitationIndex(citations)
    walk = walk_graph(sorted(set(scope_nodes) | set(pool)), index)
    ranking = rank_pool(sorted(pool), seeds, walk, space)
    return {"rrf": ranking.rank, "ppr": ranking.ppr_rank, "bge": ranking.text_rank}


def neighbourhood(pillar: str, library, works, index, pool_scope: str):
    """Scope nodes: the pillar's (what ``map build`` ranks) or the whole library's."""
    if pool_scope == "pillar":
        return select_scope(pillar, library, works, index).nodes
    nodes = set()
    for each in sorted({paper.pillar_id for paper in library}):
        nodes.update(select_scope(each, library, works, index).nodes)
    return sorted(nodes)


def loo_trials(
    inputs: MapInputs, keep_own_references: bool, pool_scope: str
) -> List[dict]:
    """Hold out each library work in turn and rank it from its pillar-mates."""
    space = VectorSpace(inputs.vectors)
    trials = []
    works_nodes = sorted({p.node_id for p in inputs.library if p.work_id})
    for target in works_nodes:
        pillars = sorted({p.pillar_id for p in inputs.library if p.node_id == target})
        for pillar in pillars:
            library = [p for p in inputs.library if p.node_id != target]
            seeds = sorted(
                {p.node_id for p in library if p.pillar_id == pillar and p.work_id}
            )
            if not seeds:
                continue
            citations = [
                c for c in inputs.citations if keep_own_references or c.citing != target
            ]
            scope = neighbourhood(
                pillar, library, inputs.works, CitationIndex(citations), pool_scope
            )
            library_nodes = {p.node_id for p in library}
            pool = sorted((set(scope) - library_nodes) | {target})
            # A citer of L is known only through L's own fetch unless the
            # remaining scope already holds it.
            members = set(scope) | {target}
            citations = [
                c for c in citations if c.cited != target or c.citing in members
            ]
            ranks = ranks_for(pool, seeds, citations, space, scope)
            trials.append(
                {
                    "target": target,
                    "title": (
                        inputs.works.get(target).title if target in inputs.works else ""
                    )[:60],
                    "pillar": pillar,
                    "seeds": len(seeds),
                    "pool": len(pool),
                    **{name: ranks[name][target] for name in SCORERS},
                }
            )
    return trials


def hide_ref_trials(inputs: MapInputs, fraction: float, pool_scope: str) -> List[dict]:
    """Hide part of each library work's references and rank them back."""
    space = VectorSpace(inputs.vectors)
    library_nodes = {p.node_id for p in inputs.library}
    trials = []
    for target in sorted({p.node_id for p in inputs.library if p.work_id}):
        refs = sorted(
            {
                c.cited
                for c in inputs.citations
                if c.citing == target and c.cited in inputs.works
            }
            - library_nodes
        )
        if len(refs) < 2:
            continue
        seed = int(hashlib.sha256(f"{HIDE_SEED}:{target}".encode()).hexdigest()[:8], 16)
        hidden = sorted(
            random.Random(seed).sample(refs, max(1, round(len(refs) * fraction)))
        )
        hidden_set = set(hidden)
        citations = [
            c
            for c in inputs.citations
            if not (c.citing == target and c.cited in hidden_set)
        ]
        for pillar in sorted(
            {p.pillar_id for p in inputs.library if p.node_id == target}
        ):
            seeds = sorted(
                {
                    p.node_id
                    for p in inputs.library
                    if p.pillar_id == pillar and p.work_id
                }
            )
            scope = neighbourhood(
                pillar,
                inputs.library,
                inputs.works,
                CitationIndex(citations),
                pool_scope,
            )
            pool = sorted((set(scope) - library_nodes) | hidden_set)
            ranks = ranks_for(pool, seeds, citations, space, scope)
            for ref in hidden:
                trials.append(
                    {
                        "target": ref,
                        "from": target,
                        "pillar": pillar,
                        "seeds": len(seeds),
                        "pool": len(pool),
                        **{name: ranks[name][ref] for name in SCORERS},
                    }
                )
    return trials


def summary(trials: Sequence[dict]) -> List[str]:
    """Median rank, MRR and hit rates per scorer."""
    if not trials:
        return ["no trials: the library needs two works with citations in one pillar"]
    pools = [t["pool"] for t in trials]
    lines = [f"{len(trials)} trials; pool size median {statistics.median(pools):.0f}"]
    header = ("scorer", "median", "MRR", "hit@10", "hit@25", "top5%")
    lines.append(
        f"{header[0]:8s} {header[1]:>7s} {header[2]:>6s} "
        f"{header[3]:>7s} {header[4]:>7s} {header[5]:>6s}"
    )
    for name in SCORERS:
        ranks = [t[name] for t in trials]
        mrr = statistics.fmean(1 / r for r in ranks)
        hit10 = statistics.fmean(r <= 10 for r in ranks)
        hit25 = statistics.fmean(r <= 25 for r in ranks)
        top5 = statistics.fmean(
            r / t["pool"] <= 0.05 for r, t in zip(ranks, trials, strict=True)
        )
        lines.append(
            f"{name:8s} {statistics.median(ranks):7.1f} {mrr:6.3f} "
            f"{hit10:7.2f} {hit25:7.2f} {top5:6.2f}"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run one protocol, print every trial and the summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-db", action="store_true", help="read the evidence through PostgREST"
    )
    source.add_argument("--inputs", nargs="+", help="MapInputs JSON file(s)")
    parser.add_argument("--protocol", choices=("loo", "hide-refs"), default="loo")
    parser.add_argument(
        "--keep-own-references",
        action="store_true",
        help="loo: the scout's leakier variant",
    )
    parser.add_argument("--hide-fraction", type=float, default=HIDE_FRACTION)
    parser.add_argument(
        "--pool",
        choices=("pillar", "library"),
        default="pillar",
        help=(
            "candidates: the pillar's scope (what map build ranks) "
            "or every pillar's (the scout's pool)"
        ),
    )
    parser.add_argument("--json", type=Path, help="also write every trial here")
    args = parser.parse_args(argv)

    inputs = load_inputs(args)
    if args.protocol == "loo":
        trials = loo_trials(inputs, args.keep_own_references, args.pool)
    else:
        trials = hide_ref_trials(replace(inputs), args.hide_fraction, args.pool)
    for trial in trials:
        print({key: trial[key] for key in trial if key != "target"})
    print()
    print("\n".join(summary(trials)))
    if args.json:
        args.json.write_text(json.dumps(trials, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
