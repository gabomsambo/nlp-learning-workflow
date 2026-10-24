"""Fact-check scripts that have no stored check yet.

``--dry-run`` lists them and prints the expected cost. It does not call a
model and does not write. A real run calls ``execute_factcheck`` once per
script and stops at the first failure. Checks saved before that failure
stay. The script row is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .. import db
from .constants import BACKFILL_COST
from .engine import FactcheckResult, execute_factcheck


@dataclass
class BackfillReport:
    lines: list[str] = field(default_factory=list)
    checked: int = 0
    failed: int = 0


def run_backfill(
    *,
    dry_run: bool,
    execute: Callable[..., FactcheckResult] | None = None,
) -> BackfillReport:
    if execute is None:
        execute = execute_factcheck
    rows = db.list_scripts_missing_factcheck()
    lines = [
        BACKFILL_COST,
        f"{len(rows)} script(s) without a fact check",
    ]
    for row in rows:
        title = row.get("title") or ""
        lines.append(f"{row['id']}  {title}".rstrip())
    if dry_run:
        lines.append("No model call. Nothing written.")
        return BackfillReport(lines=lines)
    checked = 0
    for row in rows:
        try:
            result = execute(row["id"])
        except Exception as exc:
            lines.append(f"failed {row['id']}: {exc}")
            lines.append(
                "Stopped. Fact checks saved earlier in this run stay saved. "
                "No script was changed."
            )
            return BackfillReport(lines=lines, checked=checked, failed=1)
        checked += 1
        lines.append(
            f"saved {result.fact_check_id} script {row['id']} "
            f"cost_usd {result.cost_usd:.6f}"
        )
    return BackfillReport(lines=lines, checked=checked)
