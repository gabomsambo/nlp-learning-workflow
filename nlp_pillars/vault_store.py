"""Load the library and write the Obsidian vault.

The renderer in ``vault.py`` does the work. This module only pages the
tables through PostgREST. It does not embed text and does not call a model.
"""

from __future__ import annotations

from pathlib import Path

from . import db
from .vault import ExportReport, VaultExportError, VaultSnapshot, apply_export

MIGRATION = "docs/migrations/020_knowledge_graph.sql"
_PAGE = 200

_TABLES = (
    (
        "papers",
        "id,pillar_id,title,authors,year,abstract",
        "id",
        False,
    ),
    ("pillars", "id,name,goal", "id", False),
    (
        "notes",
        "id,paper_id,problem,method,findings,limitations,future_work",
        "id",
        False,
    ),
    ("lessons", "id,paper_id,tl_dr,takeaways", "id", False),
    ("quiz_cards", "id,paper_id,question,answer", "id", False),
    ("podcast_scripts", "id,paper_id,pillar_id,title", "id", False),
    ("concepts", "id,name,aliases,description", "name", True),
    (
        "graph_edges",
        "src_kind,src_id,dst_kind,dst_id,kind,weight,provenance",
        "src_kind,src_id,dst_kind,dst_id,kind",
        True,
    ),
)


class VaultStoreError(Exception):
    """The vault could not be read from Postgres or written to disk."""


def _missing(error, table: str) -> bool:
    text = str(error).lower()
    return table.lower() in text and (
        "pgrst205" in text
        or "schema cache" in text
        or "does not exist" in text
        or "could not find" in text
    )


def _raise_for(table: str, error) -> None:
    if table in {"concepts", "graph_edges"} and _missing(error, table):
        raise VaultStoreError(
            f"{table} is not available ({error}). Apply {MIGRATION} by hand."
        )
    raise VaultStoreError(f"Could not read {table}: {error}")


def _marker(table: str, row: dict) -> tuple:
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
    return (str(row.get("id") or row.get("paper_id") or ""),)


def _read_table(table: str, columns: str, order: str) -> list:
    client = db.get_client()
    rows = []
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
            _raise_for(table, response["error"])
        page = response["data"] or []
        if not page:
            return rows
        marker = _marker(table, page[0])
        if marker == previous:
            raise VaultStoreError(
                f"Reading {table} did not advance past offset {offset}"
            )
        previous = marker
        rows.extend(page)
        if len(page) < _PAGE:
            return rows
        offset += len(page)


def load_vault_snapshot() -> VaultSnapshot:
    """Library rows plus the graph. Does not write a note."""
    loaded = {}
    for table, columns, order, _graph in _TABLES:
        loaded[table] = _read_table(table, columns, order)
    return VaultSnapshot(
        papers=loaded["papers"],
        pillars=loaded["pillars"],
        notes=loaded["notes"],
        lessons=loaded["lessons"],
        cards=loaded["quiz_cards"],
        episodes=loaded["podcast_scripts"],
        concepts=loaded["concepts"],
        edges=loaded["graph_edges"],
    )


def export_library(out: Path, *, dry_run: bool = False) -> ExportReport:
    """Read Postgres and write the vault at ``out``.

    ``--dry-run`` still reads. It does not create the directory, replace a
    note, or move one into ``_archive``.
    """
    snapshot = load_vault_snapshot()
    try:
        report = apply_export(Path(out), snapshot, dry_run=dry_run)
    except VaultExportError as exc:
        raise VaultStoreError(str(exc)) from exc
    except OSError as exc:
        raise VaultStoreError(f"Could not write the vault at {out}: {exc}") from exc
    # The CLI prints this report, and the scheduler logs it. Logging here too
    # made `vault export` show the same counts twice.
    return report
