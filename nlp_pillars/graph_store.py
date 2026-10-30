"""Read the library and write the derived graph through PostgREST.

The planner in ``graph.py`` does the work. This module only loads rows and
calls ``replace_derived_graph``, which is one transaction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import db
from .graph import (
    GraphPlan,
    GraphSnapshot,
    plan_payload,
    run_graph_build,
    slice_graph,
)
from .paper_vectors import MIGRATION as PAPER_VECTORS_MIGRATION

logger = logging.getLogger(__name__)

MIGRATION = "docs/migrations/020_knowledge_graph.sql"
MASTERY_MIGRATION = "docs/migrations/024_vocabulary_concepts.sql"
_PAGE = 200


class GraphStoreError(Exception):
    """The graph tables could not be read or written."""


def _store_error(error: Any) -> GraphStoreError:
    text = str(error).lower()
    missing = any(
        token in text
        for token in (
            "replace_derived_graph",
            "graph_edges",
            "paper_concepts",
            "pgrst202",
            "pgrst205",
        )
    )
    # ``concepts`` alone is too common a word. Require the migration's table
    # phrase or PostgREST's schema-cache error next to it.
    if "concepts" in text and (
        "schema cache" in text or "could not find" in text or "pgrst" in text
    ):
        missing = True
    if missing:
        return GraphStoreError(
            f"Knowledge graph is not available ({error}). Apply {MIGRATION} by hand."
        )
    return GraphStoreError(f"Could not update the knowledge graph: {error}")


def _missing_table(error: Any, table: str) -> bool:
    text = str(error).lower()
    return table.lower() in text and (
        "pgrst205" in text
        or "schema cache" in text
        or "does not exist" in text
        or "could not find" in text
    )


def _marker(table: str, row: dict) -> tuple:
    """Identity of the first row on a page.

    ``graph_edges`` and ``concepts`` have none of the id columns the old
    marker read, so every page looked like ``("None", "", "", "")`` and the
    guard rejected a real second page. The tuple is the row's order key.
    """
    if table == "graph_edges":
        return (
            str(row.get("src_kind") or ""),
            str(row.get("src_id") or ""),
            str(row.get("dst_kind") or ""),
            str(row.get("dst_id") or ""),
            str(row.get("kind") or ""),
        )
    if table == "concepts":
        return (str(row.get("name") or ""), str(row.get("id") or ""))
    return (
        str(
            row.get("id")
            or row.get("paper_id")
            or row.get("chunk_id")
            or row.get("ref_id")
            or ""
        ),
        str(row.get("ref_kind") or ""),
        str(row.get("model_id") or ""),
        str(row.get("source") or ""),
    )


def _read_table(table: str, columns: str, order: str) -> List[dict]:
    client = db.get_client()
    rows: List[dict] = []
    offset = 0
    previous = None
    while True:
        response = (
            client.table(table)
            .select(columns)
            .order(order)
            .limit(_PAGE)
            .offset(offset)
            .execute()
        )
        if response["error"]:
            raise _store_error(response["error"])
        page = response["data"] or []
        if not page:
            return rows
        marker = _marker(table, page[0])
        if marker == previous:
            raise GraphStoreError(
                f"Reading {table} did not advance past offset {offset}"
            )
        previous = marker
        rows.extend(page)
        if len(page) < _PAGE:
            return rows
        offset += len(page)


def _read_optional(
    table: str, columns: str, order: str
) -> Tuple[List[dict], Optional[str]]:
    try:
        return _read_table(table, columns, order), None
    except GraphStoreError as exc:
        if _missing_table(exc, table):
            return [], (
                f"{table} is not available ({exc}). "
                "Related edges from it are skipped."
            )
        raise


def load_snapshot() -> Tuple[GraphSnapshot, List[str]]:
    """Library rows the free builders read. Does not create a graph row."""
    warnings: List[str] = []
    papers = _read_table("papers", "id,pillar_id,title,authors", "id")
    pillars = _read_table("pillars", "id,name", "id")
    notes = _read_table("notes", "id,paper_id,key_terms", "id")
    lessons = _read_table("lessons", "id,paper_id,pillar_id,tl_dr", "id")
    cards = _read_table("quiz_cards", "id,paper_id,pillar_id,question", "id")
    episodes = _read_table(
        "podcast_scripts", "id,paper_id,pillar_id,title", "id"
    )
    citations = _read_table(
        "paper_citations",
        "id,paper_id,cited_paper_id,citation_direction,source",
        "id",
    )
    stored, vector_warning = _read_optional(
        "paper_vectors",
        "ref_kind,ref_id,model_id,source,vec",
        "ref_kind,ref_id,model_id,source",
    )
    if vector_warning:
        warnings.append(
            "paper_vectors is not available, so related_to edges are skipped. "
            f"Apply {PAPER_VECTORS_MIGRATION} by hand."
        )
    snapshot = GraphSnapshot(
        papers=papers,
        pillars=pillars,
        notes=notes,
        lessons=lessons,
        cards=cards,
        episodes=episodes,
        paper_vectors=[
            {
                "ref_kind": row.get("ref_kind"),
                "ref_id": row.get("ref_id"),
                "model_id": row.get("model_id"),
                "source": row.get("source"),
                "vector": row.get("vec"),
            }
            for row in stored
        ],
        citations=[
            {
                "paper_id": row.get("paper_id"),
                "cited_paper_id": row.get("cited_paper_id"),
                "direction": row.get("citation_direction"),
                "source": row.get("source"),
            }
            for row in citations
        ],
    )
    return snapshot, warnings


def apply_plan(plan: GraphPlan) -> Any:
    """Replace derived graph rows in one PostgREST function call."""
    try:
        client = db.get_client()
        response = client.rpc("replace_derived_graph", plan_payload(plan))
    except GraphStoreError:
        raise
    except Exception as exc:
        raise _store_error(exc) from exc
    if response["error"]:
        raise _store_error(response["error"])
    return response["data"]


def _citation_tool():
    try:
        from .tools.semantic_scholar_tool import BULK, SemanticScholarTool

        # A graph rebuild fetches every library paper; it queues behind
        # interactive Semantic Scholar calls.
        return SemanticScholarTool(priority=BULK)
    except Exception as exc:
        logger.warning("Citation fetch disabled: %s", exc)
        return None


def _concept_embedder():
    """Local name embedder. Construction does not download weights."""
    try:
        from .embeddings import get_embedding_provider

        provider = get_embedding_provider()
    except Exception as exc:
        logger.warning("Concept embedding merge disabled: %s", exc)
        return None

    def embed(names: Sequence[str]):
        return provider.embed_documents(list(names))

    return embed


def execute_graph_build(dry_run: bool):
    """CLI entry.

    Dry-run loads the library and prints the plan. It does not construct an
    embedding provider and does not call Semantic Scholar. The tool object is
    still built so the report can count papers that would be queried; building
    it does not open a request.
    """
    snapshot, warnings = load_snapshot()
    tool = _citation_tool()
    embed = None if dry_run else _concept_embedder()
    report = run_graph_build(
        snapshot,
        dry_run=dry_run,
        citation_tool=tool,
        embed_names=embed,
        store_citations=None if dry_run else db.add_paper_citations,
        apply=None if dry_run else apply_plan,
    )
    report.warnings = warnings + report.warnings
    return report


def load_graph_rows() -> Dict[str, list]:
    """Rows ``GET /api/graph`` turns into a neighbourhood."""
    return {
        "papers": _read_table("papers", "id,pillar_id,title", "id"),
        "pillars": _read_table("pillars", "id,name", "id"),
        "concepts": _read_table("concepts", "name,aliases", "name"),
        "lessons": _read_table("lessons", "id,paper_id,pillar_id,tl_dr", "id"),
        "cards": _read_table(
            "quiz_cards", "id,paper_id,pillar_id,question", "id"
        ),
        "episodes": _read_table(
            "podcast_scripts", "id,paper_id,pillar_id,title", "id"
        ),
        "edges": _read_table(
            "graph_edges",
            "src_kind,src_id,dst_kind,dst_id,kind,weight,provenance",
            "src_kind,src_id,dst_kind,dst_id,kind",
        ),
    }


def load_concept_mastery(
    *, user_id: str = "default_user", pillar: Optional[str] = None
) -> List[dict]:
    """Current FSRS retrievability aggregated by the Postgres query."""
    response = db.get_client().rpc(
        "concept_mastery",
        {"p_user_id": user_id or "default_user", "p_pillar_id": pillar},
    )
    if response["error"]:
        text = str(response["error"]).lower()
        if "concept_mastery" in text or "pgrst202" in text or "schema cache" in text:
            raise GraphStoreError(
                f"Concept mastery is not available ({response['error']}). "
                f"Apply {MASTERY_MIGRATION} by hand."
            )
        raise GraphStoreError(f"Could not read concept mastery: {response['error']}")
    return response["data"] or []


def graph_view(
    *,
    pillar: Optional[str] = None,
    depth: int = 1,
    center: Optional[str] = None,
    user_id: str = "default_user",
) -> dict:
    """Load the stored graph and return one bounded neighbourhood."""
    rows = load_graph_rows()
    view = slice_graph(
        **rows,
        pillar=pillar or None,
        depth=depth,
        center=center or None,
    )
    mastery = {
        str(row.get("name")): row
        for row in load_concept_mastery(user_id=user_id, pillar=pillar or None)
        if row.get("name")
    }
    for node in view["nodes"]:
        if node.get("kind") != "concept":
            continue
        row = mastery.get(str(node.get("id")), {})
        value = row.get("retrievability")
        node["retrievability"] = None if value is None else float(value)
        node["card_count"] = int(row.get("card_count") or 0)
        node["reviewed_card_count"] = int(row.get("reviewed_card_count") or 0)
    return view
