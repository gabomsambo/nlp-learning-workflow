"""Choose a real timestamp for a lesson or note the model dated itself.

The SQL in ``docs/repairs/repair_llm_created_at.sql`` applies the same order.
This module does not connect to a database.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

# First present value wins. The stored created_at is not a candidate.
REAL_TIMESTAMP_SOURCES = (
    "stage_finished_at",
    "quiz_created_at",
    "paper_added_at",
    "paper_created_at",
)

_SQL_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "repairs"
    / "repair_llm_created_at.sql"
)


def choose_real_timestamp(
    *,
    stage_finished_at: Optional[datetime] = None,
    quiz_created_at: Optional[datetime] = None,
    paper_added_at: Optional[datetime] = None,
    paper_created_at: Optional[datetime] = None,
) -> Optional[datetime]:
    """Return the best real timestamp, or None when every source is missing."""
    for value in (stage_finished_at, quiz_created_at, paper_added_at, paper_created_at):
        if value is not None:
            return value
    return None


def repair_sql_text() -> str:
    """Return the one-off repair SQL. Printing it does not apply it."""
    return _SQL_PATH.read_text(encoding="utf-8")
