"""What ``GET /api/map`` serves: the stored snapshot plus live overlays.

The snapshot (``map_snapshots``, written by ``map build``) is a projection and
can be hours old. Two things the learner changes between builds are read live
and laid beside it, never merged into it:

- **queue** — the pillar's ``paper_queue`` rows as they are now. A paper
  queued after the build has no verdict yet (``in_snapshot: false``); a row
  removed or processed since is reported as such.
- **mastery** — FSRS retrievability per concept (``concept_mastery``, migration
  024), rolled up to each library paper through ``paper_concepts``.

An overlay that cannot be read does not take the map with it: it answers
``available: false`` with the reason, because the snapshot is still true. A
missing ``map_snapshots`` table is different — there is nothing to serve — and
raises ``MapStoreError`` naming migration 034.

``status`` keeps three cases apart that a single empty payload would blur:
``not_built`` (no snapshot row for the scope), ``empty`` (built, and the map
holds no works) and ``ready``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from . import map_store
from .graph_store import GraphStoreError, load_concept_mastery
from .map_store import MapStoreError

logger = logging.getLogger(__name__)

STATUS_NOT_BUILT = "not_built"
STATUS_EMPTY = "empty"
STATUS_READY = "ready"

SCOPE_ALL = "all"

_SNAPSHOT_COLUMNS = "scope,built_at,params_hash,payload_hash,model_id,run_id,stats,payload"


class UnknownScopeError(Exception):
    """The scope is neither ``all`` nor an existing pillar."""


def read_snapshot(scope: str) -> Optional[dict]:
    """The stored ``map_snapshots`` row for ``scope``, or None when never built."""
    rows = map_store._read(
        "map_snapshots",
        _SNAPSHOT_COLUMNS,
        "scope",
        lambda query: query.eq("scope", scope),
    )
    return rows[0] if rows else None


def pillar_exists(pillar_id: str) -> bool:
    rows = map_store._read(
        "pillars", "id", "id", lambda query: query.eq("id", pillar_id)
    )
    return any(str(row.get("id")) == pillar_id for row in rows)


def _unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason}


# --------------------------------------------------------------------- queue


def queue_overlay(scope: str, snapshot_queue: Sequence[dict]) -> dict:
    """The pillar's queue now, against the verdicts the snapshot holds.

    ``all`` has no single queue, so it is reported as unavailable rather than
    as an empty one.
    """
    if scope == SCOPE_ALL:
        return _unavailable("The queue belongs to one pillar; pick a pillar scope.")
    try:
        live = map_store.load_queue(scope)
    except MapStoreError as exc:
        logger.warning("Map queue overlay unavailable: %s", exc)
        return _unavailable(str(exc))
    except Exception as exc:  # an overlay must never take the snapshot down
        logger.warning("Map queue overlay unavailable: %s", exc)
        return _unavailable(f"Could not read the queue: {exc}")

    verdicts = {str(row.get("queue_id")): row for row in snapshot_queue}
    live_ids = {str(row["id"]) for row in live}
    items: List[dict] = []
    for row in live:
        queue_id = str(row["id"])
        known = verdicts.get(queue_id)
        items.append(
            {
                "queue_id": queue_id,
                "paper_id": str(row.get("paper_id") or ""),
                "title": row.get("title") or "",
                "processed": bool(row.get("processed")),
                "in_snapshot": known is not None,
                "node": known.get("node") if known else None,
                "verdict": known.get("verdict") if known else None,
                "rank_fraction": known.get("rank_fraction") if known else None,
                # processed at build time vs now: it was read after the build
                "processed_since_build": bool(
                    known is not None
                    and bool(row.get("processed")) != bool(known.get("processed"))
                ),
            }
        )
    removed = [
        {"queue_id": queue_id, "title": row.get("title") or ""}
        for queue_id, row in sorted(verdicts.items())
        if queue_id not in live_ids
    ]
    return {
        "available": True,
        "items": items,
        "removed_since_build": removed,
        "counts": {
            "queued": len(items),
            "pending": sum(1 for item in items if not item["processed"]),
            "not_in_snapshot": sum(1 for item in items if not item["in_snapshot"]),
            "removed_since_build": len(removed),
        },
    }


# ------------------------------------------------------------------- mastery


def _float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def mastery_overlay(scope: str, nodes: Sequence[dict], user_id: str) -> dict:
    """Mean concept retrievability per library paper and per library node."""
    paper_ids = sorted(
        {pid for node in nodes for pid in (node.get("paper_ids") or []) if pid}
    )
    if not paper_ids:
        return {"available": True, "papers": {}, "nodes": {}}
    try:
        mastery = {
            str(row["concept_id"]): row
            for row in load_concept_mastery(
                user_id=user_id, pillar=None if scope == SCOPE_ALL else scope
            )
            if row.get("concept_id")
        }
        links = map_store._read_in(
            "paper_concepts", "paper_id,concept_id", "paper_id,concept_id",
            "paper_id", paper_ids,
        )
    except (GraphStoreError, MapStoreError) as exc:
        logger.warning("Map mastery overlay unavailable: %s", exc)
        return _unavailable(str(exc))
    except Exception as exc:
        logger.warning("Map mastery overlay unavailable: %s", exc)
        return _unavailable(f"Could not read concept mastery: {exc}")

    by_paper: Dict[str, List[dict]] = {}
    for link in links:
        row = mastery.get(str(link.get("concept_id")))
        if row is not None:
            by_paper.setdefault(str(link["paper_id"]), []).append(row)

    papers: Dict[str, dict] = {}
    for paper_id in paper_ids:
        rows = by_paper.get(paper_id, [])
        values = [v for v in (_float(r.get("retrievability")) for r in rows) if v is not None]
        papers[paper_id] = {
            "retrievability": round(sum(values) / len(values), 4) if values else None,
            "concepts": len(rows),
            "cards": sum(int(r.get("card_count") or 0) for r in rows),
            "reviewed_cards": sum(int(r.get("reviewed_card_count") or 0) for r in rows),
        }

    node_rollup: Dict[str, dict] = {}
    for node in nodes:
        ids = [pid for pid in (node.get("paper_ids") or []) if pid in papers]
        if not ids:
            continue
        values = [
            papers[pid]["retrievability"]
            for pid in ids
            if papers[pid]["retrievability"] is not None
        ]
        node_rollup[str(node["id"])] = {
            "retrievability": round(sum(values) / len(values), 4) if values else None,
            "concepts": sum(papers[pid]["concepts"] for pid in ids),
            "cards": sum(papers[pid]["cards"] for pid in ids),
        }
    return {"available": True, "papers": papers, "nodes": node_rollup}


# ---------------------------------------------------------------------- view


def map_view(scope: str, *, user_id: str = "default_user") -> dict:
    """The map for ``scope`` (a pillar id or ``all``) with live overlays.

    Raises ``MapStoreError`` when the snapshot table is missing or unreadable,
    and ``UnknownScopeError`` for a pillar that does not exist.
    """
    row = read_snapshot(scope)
    if row is None:
        if scope != SCOPE_ALL and not pillar_exists(scope):
            raise UnknownScopeError(f"No pillar with id {scope!r}")
        return {
            "scope": scope,
            "status": STATUS_NOT_BUILT,
            "message": (
                "The map for this scope has not been built yet. Build it with "
                f"`python -m nlp_pillars.cli map build --pillar {scope}`."
                if scope != SCOPE_ALL
                else "No map has been built for all pillars. Maps are built per pillar."
            ),
            "built_at": None,
            "snapshot": None,
            "overlays": None,
        }

    payload = row.get("payload") or {}
    nodes = payload.get("nodes") or []
    return {
        "scope": scope,
        "status": STATUS_READY if nodes else STATUS_EMPTY,
        "message": None if nodes else "The map was built and holds no works yet.",
        "built_at": row.get("built_at"),
        "payload_hash": row.get("payload_hash"),
        "params_hash": row.get("params_hash"),
        "model_id": row.get("model_id"),
        "snapshot": payload,
        "overlays": {
            "queue": queue_overlay(scope, payload.get("queue") or []),
            "mastery": mastery_overlay(scope, nodes, user_id),
        },
    }
