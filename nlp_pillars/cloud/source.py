"""Read the rows the cloud reader publishes.

Every table is read with an explicit column list, so a column the reader must
not carry (``ground_pack``, ``options``, chunk text, embeddings) never leaves
Postgres. A playground's notebook is the one exception that is read whole: its
saved outputs are reduced to the sanitized view in ``export.py``. ``export.py`` then applies a second allow-list over what arrived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import db
from .review_state import APPLY_LOCK, PostgrestSyncState, ReviewCursor, SyncStateError, SyncStateMissing

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

# Playgrounds the reader shows (docs/PLAYGROUND.md). Only a study guide that was
# written is published; the raw notebook is read so its saved outputs can be
# reduced to the sanitized view, and nothing else of it leaves export.py.
PLAYGROUND_COLUMNS = "id,paper_id,title,source_kind,notebook,steps,status"
PLAYGROUND_STATUSES = ("ready", "partial")
# A notebook can be megabytes of saved outputs, so these pages are short.
_PLAYGROUND_PAGE = 5
LINK_COLUMNS = "id,script_id,playground_id,mapping,status,script_sha256,updated_at"

# Tables that exist only after a hand-applied migration.
_MIGRATIONS = {
    "concepts": "docs/migrations/020_knowledge_graph.sql",
    "paper_concepts": "docs/migrations/020_knowledge_graph.sql",
    "graph_edges": "docs/migrations/020_knowledge_graph.sql",
    "podcast_fact_checks": "docs/migrations/022_podcast_fact_checks.sql",
    "playgrounds": "docs/migrations/028_playgrounds.sql",
    "playground_links": "docs/migrations/029_playground_links.sql",
}


class SourceError(Exception):
    """A table could not be read. An empty table is not an error."""


class MissingTable(SourceError):
    """The table needs a hand-applied migration. The message names it."""


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
    # Every review event, for folding each card's schedule (id, card_id, rating, review_timestamp).
    review_events: List[dict] = field(default_factory=list)
    # Review events from the cloud that the cards above already include (cloud_sync_state).
    through_seq: int = 0
    # Playgrounds with a study guide (ready or partial), notebook included.
    playgrounds: List[dict] = field(default_factory=list)
    # Ready follow-along links (playground_links).
    playground_links: List[dict] = field(default_factory=list)
    # What could not be read and was left out, for the export report.
    warnings: List[str] = field(default_factory=list)


def _raise_for(table: str, error) -> None:
    text = str(error).lower()
    migration = _MIGRATIONS.get(table)
    if migration and (
        table in text
        and ("pgrst205" in text or "schema cache" in text or "does not exist" in text)
    ):
        raise MissingTable(
            f"{table} is not available ({error}). Apply {migration} by hand."
        )
    raise SourceError(f"Could not read {table}: {error}")


def _read(client, table: str, columns: str, order: str, *, page_size: int = 0, **eq) -> List[dict]:
    page_size = page_size or _PAGE
    rows: List[dict] = []
    offset = 0
    while True:
        query = client.table(table).select(columns)
        for column, value in eq.items():
            query = query.eq(column, value)
        try:
            response = query.order(order).limit(page_size).offset(offset).execute()
        except Exception as exc:  # noqa: BLE001 — name the table, keep the cause
            raise SourceError(f"Could not read {table}: {exc}") from exc
        if response["error"]:
            _raise_for(table, response["error"])
        page = response["data"] or []
        rows.extend(page)
        if len(page) < page_size:
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


def _read_playgrounds(client, warnings: List[str]) -> tuple:
    """Playgrounds with a study guide and the ready links.

    Before migration 028 (or 029) is applied their part of the reader is empty
    and the report says so: the rest of the library is worth sending without it.
    """
    playgrounds: List[dict] = []
    links: List[dict] = []
    try:
        for status in PLAYGROUND_STATUSES:
            playgrounds.extend(_read(
                client, "playgrounds", PLAYGROUND_COLUMNS, "id",
                page_size=_PLAYGROUND_PAGE, status=status,
            ))
    except MissingTable as exc:
        warnings.append(f"playgrounds not published: {exc}")
        return [], []
    try:
        links = _read(client, "playground_links", LINK_COLUMNS, "id", status="ready")
    except MissingTable as exc:
        warnings.append(f"follow-along not published: {exc}")
    playgrounds.sort(key=lambda r: str(r["id"]))
    return playgrounds, links


REVIEW_LOG_COLUMNS = "id,card_id,rating,review_timestamp"
_CONSISTENCY_TRIES = 3


def _cursor(client) -> ReviewCursor:
    """The review cursor; all zeros before migration 027, so export works without it."""
    try:
        return PostgrestSyncState(client).read()
    except SyncStateMissing:
        return ReviewCursor()
    except SyncStateError as exc:
        raise SourceError(str(exc)) from exc


def _read_cards_and_reviews(client):
    """The cards, their review events and the cursor, from one consistent moment.

    ``through_seq`` tells the phone which cloud events the schedule already holds,
    so it must describe exactly the cards read here. A pull applies events and then
    moves the cursor, so a read that straddles one would pair a cursor with a card
    set it does not match. In this process the lock rules that out; a pull in another
    process (or one that crashed half-way) is caught by the cursor's two numbers and by
    reading it again after the cards.
    """
    for _ in range(_CONSISTENCY_TRIES):
        with APPLY_LOCK:
            before = _cursor(client)
            if before.unfinished:
                raise SourceError(
                    f"A review pull stopped half-way (applying through seq {before.pending}, "
                    f"folded through {before.seq}). Run `cloud pull-reviews` to finish it; "
                    "publishing waits so the phone is not told the wrong schedule."
                )
            cards = _read(client, "quiz_cards", *TABLES["quiz_cards"])
            events = _read(client, "review_logs", REVIEW_LOG_COLUMNS, "id")
            after = _cursor(client)
        if after == before:
            return cards, events, before.seq
    raise SourceError("Reviews were being applied while the schedule was read, three times running; try again.")


def load_source(client: Optional[object] = None) -> CloudSource:
    """Read everything the reader publishes. Writes nothing."""
    client = client or db.get_client()
    loaded = {
        table: _read(client, table, columns, order)
        for table, (columns, order) in TABLES.items()
        if table != "quiz_cards"
    }
    cards, review_events, through_seq = _read_cards_and_reviews(client)
    loaded["quiz_cards"] = cards
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
    warnings: List[str] = []
    playgrounds, links = _read_playgrounds(client, warnings)
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
        review_events=review_events,
        through_seq=through_seq,
        playgrounds=playgrounds,
        playground_links=links,
        warnings=warnings,
    )
