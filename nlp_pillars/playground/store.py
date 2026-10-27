"""Read and write ``playgrounds`` rows through PostgREST.

Synchronous, on the ``nlp_pillars.db`` client: the CLI and the run worker
call it directly, and async routes call it through ``asyncio.to_thread``.

A missing table raises :class:`PlaygroundTableMissing`, whose message names
``docs/migrations/028_playgrounds.sql``; every other failure raises
:class:`PlaygroundStoreError`. ``None`` means "no playground for this paper",
never "the read failed".
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .. import db
from . import MIGRATION

TABLE = "playgrounds"

STATUS_IMPORTED = "imported"
STATUS_READY = "ready"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUSES = (STATUS_IMPORTED, STATUS_READY, STATUS_PARTIAL, STATUS_FAILED)

SOURCE_NOTEBOOK = "notebook"

#: Columns for a page view. The raw notebook is large and the page renders
#: from the cell view built out of it, so a summary read leaves it out.
SUMMARY_COLUMNS = (
    "id,paper_id,title,source_kind,notebook_name,steps,datasets,status,"
    "model,error,created_at,updated_at"
)


class PlaygroundStoreError(Exception):
    """The playgrounds table could not be read or written."""


class PlaygroundTableMissing(PlaygroundStoreError):
    """Migration 028 has not been applied to this database."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_missing_table(error: Any) -> bool:
    text = str(error).lower()
    return TABLE in text and any(
        token in text
        for token in ("pgrst205", "schema cache", "does not exist", "could not find")
    )


def store_error(error: Any, action: str) -> PlaygroundStoreError:
    """Classify a PostgREST error. A missing table names the migration."""
    if _is_missing_table(error):
        return PlaygroundTableMissing(
            f"Playgrounds are not set up in this database yet ({error}). "
            f"Apply {MIGRATION} by hand."
        )
    return PlaygroundStoreError(f"{action}: {error}")


def _http_error_text(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        return f"HTTP {response.status_code} {response.text[:300]}"
    return str(exc)


def _select_one(column: str, value: str, columns: str) -> Optional[Dict[str, Any]]:
    try:
        response = (
            db.get_client().table(TABLE).select(columns).eq(column, value).limit(1).execute()
        )
    except Exception as exc:  # noqa: BLE001 - transport or config
        raise store_error(exc, "Could not read the playground") from exc
    if response["error"]:
        raise store_error(response["error"], "Could not read the playground")
    rows = response["data"] or []
    return rows[0] if rows else None


def paper_row(paper_id: str) -> Optional[Dict[str, Any]]:
    """The paper's id and pillar, or None. Raises when the read fails, so a
    database outage is not reported as "no such paper"."""
    try:
        response = (
            db.get_client().table("papers").select("id,pillar_id,title")
            .eq("id", paper_id).limit(1).execute()
        )
    except Exception as exc:  # noqa: BLE001
        raise PlaygroundStoreError(f"Could not read the paper: {exc}") from exc
    if response["error"]:
        raise PlaygroundStoreError(f"Could not read the paper: {response['error']}")
    rows = response["data"] or []
    return rows[0] if rows else None


def get_for_paper(paper_id: str, *, with_notebook: bool = False) -> Optional[Dict[str, Any]]:
    """The paper's playground, or None when it has none."""
    columns = "*" if with_notebook else SUMMARY_COLUMNS
    return _select_one("paper_id", paper_id, columns)


def get(playground_id: str, *, with_notebook: bool = True) -> Optional[Dict[str, Any]]:
    """One playground by id, or None. A malformed id is None, not a 400."""
    try:
        canonical = str(uuid.UUID(str(playground_id)))
    except ValueError:
        return None
    columns = "*" if with_notebook else SUMMARY_COLUMNS
    return _select_one("id", canonical, columns)


def latest_explain_run(playground_id: str) -> Optional[Dict[str, Any]]:
    """The newest explain run for this playground (any status), or None.

    The run row stores ``result.playground_id`` from the moment it is created,
    which is what this filters on (and what migration 028's one-active index
    covers).
    """
    try:
        query = (
            db.get_client().table("pipeline_runs")
            .select("id,status,error,current_stage,created_at,finished_at")
            .eq("kind", "playground_explain")
            .order("created_at", desc=True)
            .limit(1)
        )
        query.params["result->>playground_id"] = f"eq.{playground_id}"
        response = query.execute()
    except Exception as exc:  # noqa: BLE001
        raise PlaygroundStoreError(f"Could not read explain runs: {exc}") from exc
    if response["error"]:
        raise PlaygroundStoreError(f"Could not read explain runs: {response['error']}")
    rows = response["data"] or []
    return rows[0] if rows else None


def save_import(
    *,
    playground_id: str,
    paper_id: str,
    title: str,
    notebook: Dict[str, Any],
    notebook_name: str,
    datasets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Insert or replace the paper's playground with a freshly imported notebook.

    One current playground per paper (``UNIQUE (paper_id)``). A re-import
    keeps the row id and clears the study guide, because the guide described
    the notebook being replaced.
    """
    client = db.get_client()
    now = _now()
    row = {
        "id": playground_id,
        "paper_id": paper_id,
        "title": title,
        "source_kind": SOURCE_NOTEBOOK,
        "notebook": notebook,
        "notebook_name": notebook_name,
        "steps": [],
        "datasets": datasets,
        "status": STATUS_IMPORTED,
        "model": None,
        "error": None,
        "updated_at": now,
    }
    try:
        response = client.client.post(
            f"{client.base_url}/{TABLE}",
            params={"on_conflict": "paper_id", "select": SUMMARY_COLUMNS},
            json=[row],
            headers={
                **client.headers,
                "Prefer": "resolution=merge-duplicates,return=representation",
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise store_error(exc, "Could not save the notebook") from exc
    if response.status_code >= 400:
        raise store_error(
            f"HTTP {response.status_code} {response.text[:300]}",
            "Could not save the notebook",
        )
    data = response.json() if response.text else []
    return data[0] if data else row


def _patch(playground_id: str, fields: Dict[str, Any], action: str) -> None:
    fields = {**fields, "updated_at": _now()}
    try:
        db.get_client().table(TABLE).eq("id", playground_id).update(fields)
    except Exception as exc:  # noqa: BLE001 - update() raises on HTTP errors
        raise store_error(_http_error_text(exc), action) from exc


def save_datasets(playground_id: str, datasets: List[Dict[str, Any]]) -> None:
    _patch(playground_id, {"datasets": datasets}, "Could not record the datasets")


def save_explain(
    playground_id: str,
    *,
    steps: List[Dict[str, Any]],
    status: str,
    model: Optional[str],
    error: Optional[str],
) -> None:
    """Store an explain pass. ``steps`` may be empty when it failed outright."""
    if status not in STATUSES:
        raise ValueError(f"unknown playground status {status!r}")
    _patch(
        playground_id,
        {"steps": steps, "status": status, "model": model, "error": error},
        "Could not save the study guide",
    )


def save_explain_error(playground_id: str, error: str) -> None:
    """Record why a pass failed, keeping whatever guide is already there."""
    _patch(playground_id, {"error": error}, "Could not record the error")
