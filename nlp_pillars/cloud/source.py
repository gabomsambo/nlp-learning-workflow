"""Read the rows the cloud reader publishes.

Every table is read with an explicit column list, so a column the reader must
not carry (``ground_pack``, ``options``, chunk text, embeddings) never leaves
Postgres. ``export.py`` then applies a second allow-list over what arrived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import db

_PAGE = 200

# table -> (columns, order). The order must be a total order or paging skips rows.
TABLES: Dict[str, tuple] = {
    "pillars": ("id,name,goal,focus_areas", "id"),
    "papers": (
        "id,pillar_id,title,authors,venue,year,url_pdf,abstract,citation_count,"
        "added_at",
        "id",
    ),
    "notes": (
        "id,paper_id,problem,method,findings,limitations,future_work,key_terms,"
        "related_papers,confidence_score,created_at",
        "id",
    ),
    "lessons": (
        "id,paper_id,tl_dr,takeaways,practice_ideas,connections,difficulty,"
        "estimated_time,created_at",
        "id",
    ),
    "quiz_cards": (
        "id,paper_id,question,answer,difficulty,question_type,state,lapses,"
        "stability,difficulty_fsrs,last_review_date,next_review_date",
        "id",
    ),
    "podcast_scripts": (
        "id,paper_id,pillar_id,title,script,word_count,key_points,"
        "source_material,audio_metadata,created_at",
        "id",
    ),
    "podcast_fact_checks": (
        "id,script_id,script_sha256,status,counts,prompt_version,created_at",
        "id",
    ),
    "concepts": ("id,name,aliases,description", "id"),
    "paper_concepts": ("paper_id,concept_id,weight", "paper_id,concept_id"),
}

# Only these edges are read: the rest (authored_by, in_pillar, of, covers) restate
# columns already published, and cites points at papers outside the library.
EDGE_COLUMNS = "src_kind,src_id,dst_kind,dst_id,kind,weight"
EDGE_ORDER = "src_kind,src_id,dst_kind,dst_id,kind"

CLAIM_COLUMNS = (
    "id,fact_check_id,line_index,span_start,span_end,span_text,claim,scope,"
    "category,verify_marker,verdict,error_type,severity,rationale,source_says,"
    "evidence,evidence_status,dismissed_at"
)

# Tables that exist only after a hand-applied migration.
_MIGRATIONS = {
    "concepts": "docs/migrations/020_knowledge_graph.sql",
    "paper_concepts": "docs/migrations/020_knowledge_graph.sql",
    "graph_edges": "docs/migrations/020_knowledge_graph.sql",
    "podcast_fact_checks": "docs/migrations/022_podcast_fact_checks.sql",
}


class SourceError(Exception):
    """A table could not be read. An empty table is not an error."""


@dataclass
class CloudSource:
    pillars: List[dict] = field(default_factory=list)
    papers: List[dict] = field(default_factory=list)
    notes: List[dict] = field(default_factory=list)
    lessons: List[dict] = field(default_factory=list)
    cards: List[dict] = field(default_factory=list)
    scripts: List[dict] = field(default_factory=list)
    fact_checks: List[dict] = field(default_factory=list)
    claims: List[dict] = field(default_factory=list)
    concepts: List[dict] = field(default_factory=list)
    paper_concepts: List[dict] = field(default_factory=list)
    edges: List[dict] = field(default_factory=list)


def _raise_for(table: str, error) -> None:
    text = str(error).lower()
    migration = _MIGRATIONS.get(table)
    if migration and (
        table in text
        and ("pgrst205" in text or "schema cache" in text or "does not exist" in text)
    ):
        raise SourceError(
            f"{table} is not available ({error}). Apply {migration} by hand."
        )
    raise SourceError(f"Could not read {table}: {error}")


def _read(client, table: str, columns: str, order: str, **eq) -> List[dict]:
    rows: List[dict] = []
    offset = 0
    while True:
        query = client.table(table).select(columns)
        for column, value in eq.items():
            query = query.eq(column, value)
        try:
            response = query.order(order).limit(_PAGE).offset(offset).execute()
        except Exception as exc:  # noqa: BLE001 — name the table, keep the cause
            raise SourceError(f"Could not read {table}: {exc}") from exc
        if response["error"]:
            _raise_for(table, response["error"])
        page = response["data"] or []
        rows.extend(page)
        if len(page) < _PAGE:
            return rows
        offset += len(page)


def latest_fact_checks(rows: List[dict]) -> List[dict]:
    """The newest check per script, as the Listen page shows it."""
    newest: Dict[str, dict] = {}
    for row in rows:
        key = str(row.get("script_id"))
        held = newest.get(key)
        if held is None or (str(row.get("created_at") or ""), str(row["id"])) > (
            str(held.get("created_at") or ""),
            str(held["id"]),
        ):
            newest[key] = row
    return [newest[key] for key in sorted(newest)]


def load_source(client: Optional[object] = None) -> CloudSource:
    """Read everything the reader publishes. Writes nothing."""
    client = client or db.get_client()
    loaded = {
        table: _read(client, table, columns, order)
        for table, (columns, order) in TABLES.items()
    }
    # Only the claims of the check each episode shows. Older checks of the same
    # script are superseded and would only add bytes the reader never renders.
    checks = latest_fact_checks(
        [row for row in loaded["podcast_fact_checks"] if row.get("status") == "succeeded"]
    )
    claims: List[dict] = []
    for check in checks:
        claims.extend(
            _read(client, "podcast_claims", CLAIM_COLUMNS, "id", fact_check_id=check["id"])
        )
    edges = _read(client, "graph_edges", EDGE_COLUMNS, EDGE_ORDER, kind="related_to")
    return CloudSource(
        pillars=loaded["pillars"],
        papers=loaded["papers"],
        notes=loaded["notes"],
        lessons=loaded["lessons"],
        cards=loaded["quiz_cards"],
        scripts=loaded["podcast_scripts"],
        fact_checks=checks,
        claims=claims,
        concepts=loaded["concepts"],
        paper_concepts=loaded["paper_concepts"],
        edges=edges,
    )
