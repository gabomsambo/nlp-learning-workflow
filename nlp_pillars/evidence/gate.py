"""The gate the evidence writer must pass before it can become the default.

Pure functions over run records, so the arithmetic is tested and the paid
runner (``scripts/podcast_writer_gate.py``) only has to produce records.

A **run record** is one script from one arm::

    {"paper": "arxiv:1706.03762", "arm": "free" | "evidence", "sample": 1,
     "words": 3100,                      # spoken words, tags and cues excluded
     "counts": {"claims": 120, "high": 1, "medium": 2, "low": 9, ...},
     "claims": [ {line_index, span_text, claim, verdict, severity, ...} ],
     "stats": { ... evidence arm only: Ledger.stats ... },
     "ledger": {"proposed": 60, "dropped": [...]},          # evidence arm only
     "cost": {"generate": 0.1, "check": 0.1}}

The gate (docs/PODCAST_EVIDENCE_WRITER.md): at least 8 papers x 2 samples per
arm; hand-confirmed serious flags per 1,000 words no higher than ``free``;
spoken words at least 85% of ``free``; at least 60% of the ledger's narrative
entries used; no untagged line; at most 10% of proposed ledger entries dropped
by quote verification.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any, Iterable, Optional

MIN_PAPERS = 8
MIN_SAMPLES = 2
WORDS_RATIO_MIN = 0.85
ENTRIES_USED_MIN = 0.60
DROP_RATE_MAX = 0.10
UNTAGGED_MAX = 0

SERIOUS = ("high", "medium")
ARMS = ("free", "evidence")
VERDICTS = ("real", "minor", "wrong")


def spoken_words(script: str) -> int:
    """Words the narrator would say: no ``[HOST]:`` prefix, no cue lines."""
    from ..tts.cue_parser import spoken_line_text

    total = 0
    for line in script.splitlines():
        spoken = spoken_line_text(line)
        if spoken:
            total += len(spoken.split())
    return total


def per_1000(count: float, words: int) -> Optional[float]:
    return 1000.0 * count / words if words else None


def flag_id(run: dict[str, Any], claim: dict[str, Any], ordinal: int) -> str:
    """A short, stable id for one flag, so labels survive a re-run of ``flags``."""
    raw = f"{run['paper']}|{run['arm']}|{run['sample']}|{claim['line_index']}|{ordinal}"
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def serious_flags(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every high/medium claim, with its flag id, sorted by id (a fixed shuffle)."""
    found = []
    for run in runs:
        seen_line: dict[int, int] = defaultdict(int)
        for claim in run.get("claims") or []:
            if claim.get("severity") not in SERIOUS:
                continue
            line = int(claim["line_index"])
            ordinal = seen_line[line]
            seen_line[line] += 1
            found.append({
                "fid": flag_id(run, claim, ordinal),
                "paper": run["paper"], "arm": run["arm"], "sample": run["sample"],
                "claim": claim,
            })
    return sorted(found, key=lambda f: f["fid"])


def _sum(runs: list[dict[str, Any]], key: str) -> int:
    return sum(int((r.get("counts") or {}).get(key, 0)) for r in runs)


def arm_summary(
    runs: list[dict[str, Any]], labels: Optional[dict[str, str]] = None
) -> dict[str, Any]:
    """Totals and per-1,000-word rates for one arm."""
    words = sum(int(r["words"]) for r in runs)
    high, medium, low = _sum(runs, "high"), _sum(runs, "medium"), _sum(runs, "low")
    claims = _sum(runs, "claims")
    out: dict[str, Any] = {
        "scripts": len(runs),
        "papers": len({r["paper"] for r in runs}),
        "words": words,
        "claims": claims,
        "high": high, "medium": medium, "low": low,
        "flags_per_1000_words": per_1000(high + medium + low, words),
        "serious_per_1000_words": per_1000(high + medium, words),
        "high_per_1000_words": per_1000(high, words),
        "claims_per_1000_words": per_1000(claims, words),
        "cost_generate": round(sum(float((r.get("cost") or {}).get("generate", 0)) for r in runs), 6),
        "cost_check": round(sum(float((r.get("cost") or {}).get("check", 0)) for r in runs), 6),
    }
    if labels is not None:
        flags = [f for f in serious_flags(runs)]
        counted = [labels[f["fid"]] for f in flags if f["fid"] in labels]
        real, minor, wrong = (counted.count(v) for v in VERDICTS)
        out["labelled"] = {
            "flags": len(flags), "labelled": len(counted),
            "real": real, "minor": minor, "wrong": wrong,
            "precision_strict": real / len(counted) if counted else None,
            "precision_lenient": (real + minor) / len(counted) if counted else None,
            "confirmed_per_1000_words": per_1000(real, words),
        }
    return out


def _by_paper(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["paper"]].append(run)
    return grouped


def paired_difference(
    free: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    value,
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Mean evidence-minus-free difference of ``value(runs of one paper)``, with a 95% bootstrap CI.

    Resamples papers, not scripts: two samples of one paper are not independent.
    """
    free_by, ev_by = _by_paper(free), _by_paper(evidence)
    papers = sorted(set(free_by) & set(ev_by))
    diffs = []
    for paper in papers:
        a, b = value(free_by[paper]), value(ev_by[paper])
        if a is not None and b is not None:
            diffs.append(b - a)
    if not diffs:
        return {"papers": 0, "mean": None, "lo": None, "hi": None}
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(resamples)
    )
    return {
        "papers": len(diffs),
        "mean": sum(diffs) / len(diffs),
        "lo": means[int(0.025 * resamples)],
        "hi": means[int(0.975 * resamples) - 1],
        "per_paper": dict(zip(papers, diffs)),
    }


def completeness(free: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """The numbers that stop prevention from winning by saying less."""
    ratio = paired_ratio(free, evidence)
    stats = [r.get("stats") or {} for r in evidence]
    used = [s["narrative_used_ratio"] for s in stats if s.get("narrative_used_ratio") is not None]
    proposed = sum(int((r.get("ledger") or {}).get("proposed", 0)) for r in evidence)
    dropped = sum(len((r.get("ledger") or {}).get("dropped") or []) for r in evidence)
    return {
        "words_ratio": ratio["pooled"],
        "words_ratio_by_paper": ratio["by_paper"],
        "words_ratio_worst_paper": min(ratio["by_paper"].values()) if ratio["by_paper"] else None,
        "narrative_used_mean": sum(used) / len(used) if used else None,
        "narrative_used_min": min(used) if used else None,
        "untagged_lines": sum(int(s.get("untagged_lines", 0)) for s in stats),
        "unresolved_or_unknown": sum(len(s.get("unknown_ids") or []) for s in stats),
        "spoken_lines": sum(int(s.get("spoken_lines", 0)) for s in stats),
        "ledger_proposed": proposed,
        "ledger_dropped": dropped,
        "drop_rate": dropped / proposed if proposed else None,
        "number_violation_lines": sum(int(s.get("number_violation_lines", 0)) for s in stats),
        "claims_per_1000_words_free": arm_summary(free)["claims_per_1000_words"] if free else None,
        "claims_per_1000_words_evidence": arm_summary(evidence)["claims_per_1000_words"] if evidence else None,
    }


def paired_ratio(free: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Evidence words over free words: pooled, and per paper."""
    free_by, ev_by = _by_paper(free), _by_paper(evidence)
    by_paper = {}
    for paper in sorted(set(free_by) & set(ev_by)):
        a = sum(r["words"] for r in free_by[paper]) / len(free_by[paper])
        b = sum(r["words"] for r in ev_by[paper]) / len(ev_by[paper])
        if a:
            by_paper[paper] = b / a
    total_free = sum(r["words"] for r in free)
    total_ev = sum(r["words"] for r in evidence)
    return {
        "pooled": (total_ev / len(evidence)) / (total_free / len(free)) if free and evidence and total_free else None,
        "by_paper": by_paper,
    }


def coverage_problems(free: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> list[str]:
    """Why the run is too small to gate on, if it is."""
    problems = []
    for name, runs in (("free", free), ("evidence", evidence)):
        samples = defaultdict(set)
        for run in runs:
            samples[run["paper"]].add(run["sample"])
        if len(samples) < MIN_PAPERS:
            problems.append(f"{name}: {len(samples)} papers, need {MIN_PAPERS}")
        thin = sorted(p for p, s in samples.items() if len(s) < MIN_SAMPLES)
        if thin:
            problems.append(f"{name}: fewer than {MIN_SAMPLES} samples for {', '.join(thin)}")
    if {r["paper"] for r in free} != {r["paper"] for r in evidence}:
        problems.append("the two arms did not run on the same papers")
    return problems


def evaluate(
    free: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    labels: dict[str, str],
) -> dict[str, Any]:
    """Everything the gate reports, and whether each criterion passed."""
    free_sum = arm_summary(free, labels)
    ev_sum = arm_summary(evidence, labels)
    comp = completeness(free, evidence)
    unlabelled = [
        f["fid"] for f in serious_flags([*free, *evidence]) if f["fid"] not in labels
    ]

    def confirmed(runs):
        flags = {f["fid"]: f for f in serious_flags(runs)}
        real = sum(1 for fid in flags if labels.get(fid) == "real")
        words = sum(r["words"] for r in runs)
        return per_1000(real, words)

    serious_diff = paired_difference(free, evidence, confirmed)
    raw_diff = paired_difference(
        free, evidence,
        lambda runs: per_1000(
            sum(_sum([r], "high") + _sum([r], "medium") for r in runs),
            sum(r["words"] for r in runs),
        ),
    )

    free_rate = free_sum["labelled"]["confirmed_per_1000_words"]
    ev_rate = ev_sum["labelled"]["confirmed_per_1000_words"]
    problems = coverage_problems(free, evidence)
    checks = [
        ("enough papers and samples", not problems, "; ".join(problems) or "ok"),
        ("every serious flag is hand-labelled", not unlabelled, f"{len(unlabelled)} unlabelled"),
        (
            "confirmed serious flags per 1,000 words no higher than free",
            free_rate is not None and ev_rate is not None and ev_rate <= free_rate,
            f"evidence {_fmt(ev_rate)} vs free {_fmt(free_rate)}",
        ),
        (
            f"spoken words >= {WORDS_RATIO_MIN:.0%} of free",
            comp["words_ratio"] is not None and comp["words_ratio"] >= WORDS_RATIO_MIN,
            f"pooled {_fmt(comp['words_ratio'], pct=True)}, worst paper {_fmt(comp['words_ratio_worst_paper'], pct=True)}",
        ),
        (
            f"ledger entries used >= {ENTRIES_USED_MIN:.0%}",
            comp["narrative_used_mean"] is not None and comp["narrative_used_mean"] >= ENTRIES_USED_MIN,
            f"mean {_fmt(comp['narrative_used_mean'], pct=True)}, min {_fmt(comp['narrative_used_min'], pct=True)}",
        ),
        (
            "0 untagged lines",
            comp["untagged_lines"] <= UNTAGGED_MAX,
            f"{comp['untagged_lines']} of {comp['spoken_lines']} spoken lines",
        ),
        (
            f"quote-verification drop rate <= {DROP_RATE_MAX:.0%}",
            comp["drop_rate"] is not None and comp["drop_rate"] <= DROP_RATE_MAX,
            f"{comp['ledger_dropped']} of {comp['ledger_proposed']} ({_fmt(comp['drop_rate'], pct=True)})",
        ),
    ]
    return {
        "free": free_sum,
        "evidence": ev_sum,
        "completeness": comp,
        "raw_serious_difference": raw_diff,
        "confirmed_serious_difference": serious_diff,
        "unlabelled": unlabelled,
        "checks": [{"name": n, "passed": bool(ok), "detail": d} for n, ok, d in checks],
        "passed": all(ok for _n, ok, _d in checks),
    }


def _fmt(value: Optional[float], pct: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}" if pct else f"{value:.2f}"
