"""Failures of a fact-check run.

A failure never rewrites the script. Callers persist only after ``run_factcheck``
returns, so an exception here leaves ``podcast_scripts`` untouched.
"""

from __future__ import annotations

from typing import Any


class FactcheckError(Exception):
    """The check did not finish. Nothing should be written."""

    def __init__(
        self,
        message: str,
        *,
        cost_usd: float = 0.0,
        calls: list[dict[str, Any]] | None = None,
    ):
        super().__init__(message)
        self.cost_usd = cost_usd
        self.calls = list(calls or [])


class FactcheckBudgetError(FactcheckError):
    """The run crossed its cost cap and stopped before another model call."""
