"""Read and write ``playground_links`` rows through PostgREST.

A link says which playground an episode follows, and stores the alignment
pass's answer: which script lines are about which code cell. The shape is
docs/PLAYGROUND.md, "Follow-along".

Synchronous, on the ``nlp_pillars.db`` client, like ``store``. A missing
table raises :class:`LinkTableMissing`, whose message names
``docs/migrations/029_playground_links.sql``; every other failure raises
:class:`store.PlaygroundStoreError`. ``None`` and ``[]`` mean "absent",
never "the read failed".
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .. import db
from . import FOLLOW_MIGRATION, store

TABLE = "playground_links"

STATUS_PENDING = "pending"
STATUS_READY = "ready"
STATUS_FAILED = "failed"
STATUSES = (STATUS_PENDING, STATUS_READY, STATUS_FAILED)

KIND_PLAYGROUND_ALIGN = "playground_align"

COLUMNS = (
    "id,script_id,playground_id,mapping,model,status,error,script_sha256,"
    "created_at,updated_at"
)


class LinkTableMissing(store.PlaygroundStoreError):
    """Migration 029 has not been applied to this database."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_id(value: Any) -> Optional[str]:
    """The UUID in canonical form, or None when it is not one."""
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return None


def _is_missing_table(error: Any) -> bool:
    text = str(error).lower()
    return TABLE in text and any(
        token in text
        for token in ("pgrst205", "schema cache", "does not exist", "could not find")
    )


def link_error(error: Any, action: str) -> store.PlaygroundStoreError:
    """Classify a PostgREST error. A missing table names migration 029."""
    if _is_missing_table(error):
        return LinkTableMissing(
            f"Follow-along is not set up in this database yet ({error}). "
            f"Apply {FOLLOW_MIGRATION} by hand."
        )
    return store.PlaygroundStoreError(f"{action}: {error}")


def _http_error_text(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        return f"HTTP {response.status_code} {response.text[:300]}"
    return str(exc)


def _rows(query, action: str) -> List[Dict[str, Any]]:
    try:
        response = query.execute()
    except Exception as exc:  # noqa: BLE001 - transport or config
        raise link_error(exc, action) from exc
    if response["error"]:
        raise link_error(response["error"], action)
    return response["data"] or []


def links_for_script(script_id: str) -> List[Dict[str, Any]]:
    """Every link of one episode, newest first. A malformed id has none."""
    script_id = canonical_id(script_id)
    if script_id is None:
        return []
    query = (
        db.get_client().table(TABLE).select(COLUMNS)
        .eq("script_id", script_id).order("updated_at", desc=True).limit(50)
    )
    return _rows(query, "Could not read the episode's code links")


def get_link(script_id: str, playground_id: str) -> Optional[Dict[str, Any]]:
    script_id, playground_id = canonical_id(script_id), canonical_id(playground_id)
    if script_id is None or playground_id is None:
        return None
    query = (
        db.get_client().table(TABLE).select(COLUMNS)
        .eq("script_id", script_id).eq("playground_id", playground_id).limit(1)
    )
    rows = _rows(query, "Could not read the code link")
    return rows[0] if rows else None


def ensure_link(script_id: str, playground_id: str) -> Dict[str, Any]:
    """The link row for this pair, created ``pending`` when there is none.

    An existing link is returned unchanged, so a retry keeps the mapping it
    already has until a new pass replaces it.
    """
    script_id, playground_id = canonical_id(script_id), canonical_id(playground_id)
    if script_id is None or playground_id is None:
        raise ValueError("script_id and playground_id must be UUIDs")
    existing = get_link(script_id, playground_id)
    if existing is not None:
        return existing
    client = db.get_client()
    row = {
        "script_id": script_id,
        "playground_id": playground_id,
        "status": STATUS_PENDING,
        "mapping": [],
    }
    try:
        response = client.client.post(
            f"{client.base_url}/{TABLE}",
            params={"on_conflict": "script_id,playground_id", "select": COLUMNS},
            json=[row],
            headers={
                **client.headers,
                # A second start that raced this one is not an error.
                "Prefer": "resolution=ignore-duplicates,return=representation",
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise link_error(exc, "Could not create the code link") from exc
    if response.status_code >= 400:
        raise link_error(
            f"HTTP {response.status_code} {response.text[:300]}",
            "Could not create the code link",
        )
    data = response.json() if response.text else []
    if data:
        return data[0]
    raced = get_link(script_id, playground_id)
    if raced is None:
        raise store.PlaygroundStoreError("The code link was not saved.")
    return raced


def _patch(link_id: str, fields: Dict[str, Any], action: str) -> None:
    fields = {**fields, "updated_at": _now()}
    try:
        db.get_client().table(TABLE).eq("id", link_id).update(fields)
    except Exception as exc:  # noqa: BLE001 - update() raises on HTTP errors
        raise link_error(_http_error_text(exc), action) from exc


def save_mapping(
    link_id: str,
    *,
    mapping: List[Dict[str, Any]],
    model: Optional[str],
    error: Optional[str],
    script_sha256: str,
) -> None:
    """Store a pass's validated spans and mark the link ready."""
    _patch(
        link_id,
        {
            "mapping": mapping,
            "model": model,
            "status": STATUS_READY,
            "error": error,
            "script_sha256": script_sha256,
        },
        "Could not save the alignment",
    )


def save_failure(link: Dict[str, Any], error: str) -> None:
    """Record a failed pass. A link that already has a mapping keeps it."""
    if link.get("status") == STATUS_READY:
        fields = {"error": f"The last attempt failed: {error}"}
    else:
        fields = {"status": STATUS_FAILED, "error": error}
    _patch(link["id"], fields, "Could not record the failure")


def latest_align_run(script_id: str) -> Optional[Dict[str, Any]]:
    """The newest alignment run for this episode (any playground, any status)."""
    try:
        query = (
            db.get_client().table("pipeline_runs")
            .select("id,status,error,current_stage,result,created_at,finished_at")
            .eq("kind", KIND_PLAYGROUND_ALIGN)
            .order("created_at", desc=True)
            .limit(1)
        )
        query.params["result->>script_id"] = f"eq.{script_id}"
        response = query.execute()
    except Exception as exc:  # noqa: BLE001
        raise store.PlaygroundStoreError(f"Could not read alignment runs: {exc}") from exc
    if response["error"]:
        raise store.PlaygroundStoreError(f"Could not read alignment runs: {response['error']}")
    rows = response["data"] or []
    return rows[0] if rows else None


def _quoted_in(values: List[str]) -> str:
    # Paper ids carry ':' and may carry ',' or '/'; quote each one.
    return "in.(" + ",".join('"' + str(v).replace('"', '\\"') + '"' for v in values) + ")"


def paper_titles(paper_ids: List[str]) -> Dict[str, str]:
    """Titles for these papers. Missing papers are left out."""
    ids = sorted({p for p in paper_ids if p})
    if not ids:
        return {}
    query = db.get_client().table("papers").select("id,title")
    query.params["id"] = _quoted_in(ids)
    try:
        response = query.execute()
    except Exception as exc:  # noqa: BLE001
        raise store.PlaygroundStoreError(f"Could not read paper titles: {exc}") from exc
    if response["error"]:
        raise store.PlaygroundStoreError(f"Could not read paper titles: {response['error']}")
    return {row["id"]: row.get("title") or "" for row in response["data"] or []}


def list_playgrounds(limit: int = 200) -> List[Dict[str, Any]]:
    """Every playground, newest first, with its paper's title. No notebooks."""
    query = (
        db.get_client().table(store.TABLE)
        .select("id,paper_id,title,status,updated_at")
        .order("updated_at", desc=True).limit(limit)
    )
    try:
        response = query.execute()
    except Exception as exc:  # noqa: BLE001
        raise store.store_error(exc, "Could not list playgrounds") from exc
    if response["error"]:
        raise store.store_error(response["error"], "Could not list playgrounds")
    rows = response["data"] or []
    titles = paper_titles([r.get("paper_id") for r in rows])
    for row in rows:
        row["paper_title"] = titles.get(row.get("paper_id"), "")
    return rows
