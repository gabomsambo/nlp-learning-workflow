"""Import a notebook into a paper's playground, and run the explain pass.

Both are synchronous. The web import calls :func:`import_notebook` through
``asyncio.to_thread``; the explain pass is a ``pipeline_runs`` job of kind
``playground_explain`` (webui/services/run_service.py) and the CLI calls
:func:`run_explain_job` directly.
"""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config import get_settings
from ..llm_client import structured
from ..llm_routes import PLAYGROUND_ROUTE
from ..orchestrator import RunCancelledError, _first_line
from ..schemas import PLAYGROUND_EXPLAIN_STAGES, StageName, StageStatus
from . import store
from .datasets import DatasetCache, find_dataset_urls
from .explain import (
    MAX_TOKENS,
    SYSTEM_PROMPT,
    ExplainDraft,
    make_batches,
    missing_cells,
    number_steps,
    validate_draft,
)
from .notebook import NotebookError, cells_view, notebook_title, parse_notebook
from .setup import suggest_setup

logger = logging.getLogger(__name__)

KIND_PLAYGROUND_EXPLAIN = "playground_explain"
TRIGGER_UI_PLAYGROUND = "ui_playground"

#: Run-level error text cap, inside finish_pipeline_run's 2000.
_MAX_ERROR_CHARS = 1800

StageCallback = Callable[[str, str, Optional[str]], None]


class PaperNotFound(LookupError):
    """There is no paper with this id."""


class PlaygroundNotFound(LookupError):
    """There is no playground with this id."""


class PlaygroundExplainError(RuntimeError):
    """The explain pass produced nothing usable. The message says why."""


# -- import ----------------------------------------------------------------------


def _file_name(filename: str) -> str:
    name = Path(filename or "notebook.ipynb").name
    return name[:200] or "notebook.ipynb"


def import_notebook(
    paper_id: str,
    data: bytes,
    filename: str,
    *,
    cache: Optional[DatasetCache] = None,
) -> Dict[str, Any]:
    """Attach a notebook to a paper, replacing any notebook it had.

    Order matters: the file is validated first (no database, no network),
    then the row is written, then datasets are fetched. A missing table
    therefore fails before anything is downloaded, and a failed download
    never loses the notebook.

    Raises:
        NotebookError: not a notebook, or over the size cap.
        PaperNotFound: no such paper.
        store.PlaygroundStoreError: the database refused (a missing table
            names migration 028).
    """
    settings = get_settings()
    name = _file_name(filename)
    if not name.lower().endswith(".ipynb"):
        raise NotebookError("Only .ipynb notebooks can be imported.")
    nb = parse_notebook(data, max_bytes=settings.playground_notebook_max_bytes)

    if store.paper_row(paper_id) is None:
        raise PaperNotFound(f"Paper not found: {paper_id}")
    existing = store.get_for_paper(paper_id)
    playground_id = existing["id"] if existing else str(uuid.uuid4())
    previous = (existing or {}).get("datasets") or []

    row = store.save_import(
        playground_id=playground_id,
        paper_id=paper_id,
        title=notebook_title(nb, Path(name).stem),
        notebook=nb,
        notebook_name=name,
        datasets=previous,
        setup_code=suggest_setup(nb),
    )
    datasets = cache_datasets(playground_id, find_dataset_urls(nb), previous, cache=cache)
    store.save_datasets(playground_id, datasets)
    row["datasets"] = datasets
    logger.warning(
        "Playground %s imported for %s: %d cells, %d of %d datasets cached",
        playground_id, paper_id, len(nb["cells"]),
        sum(1 for d in datasets if not d.get("error")), len(datasets),
    )
    return row


def cache_datasets(
    playground_id: str,
    urls: List[str],
    previous: List[Dict[str, Any]],
    *,
    cache: Optional[DatasetCache] = None,
) -> List[Dict[str, Any]]:
    """Fetch the notebook's datasets. Never raises for a download problem."""
    if cache is None:
        settings = get_settings()
        cache = DatasetCache(settings.playground_data_dir, settings.playground_data_max_bytes)
    try:
        return cache.cache(playground_id, urls, previous)
    except OSError as exc:
        reason = f"The playground data directory is not writable ({exc})."
        logger.error("Playground %s datasets not cached: %s", playground_id, reason)
        return [{"url": url, "error": reason} for url in urls]


# -- explain pass ----------------------------------------------------------------


class _Stages:
    """Report stage transitions; close the ones that will not run."""

    def __init__(self, on_stage: Optional[StageCallback], cancel: Optional[threading.Event]):
        self._on_stage = on_stage
        self._cancel = cancel
        self._closed: set = set()

    def start(self, stage: StageName, detail: Optional[str] = None) -> None:
        if self._cancel is not None and self._cancel.is_set():
            raise RunCancelledError(f"cancelled before {stage.value}")
        self._emit(stage, StageStatus.RUNNING, detail)

    def progress(self, stage: StageName, detail: str) -> None:
        self._emit(stage, StageStatus.RUNNING, detail)

    def done(self, stage: StageName, detail: Optional[str] = None) -> None:
        self._closed.add(stage)
        self._emit(stage, StageStatus.COMPLETED, detail)

    def failed(self, stage: StageName, detail: str) -> None:
        self._closed.add(stage)
        self._emit(stage, StageStatus.FAILED, detail)

    def stop_after_current(self, reason: str) -> None:
        for stage in PLAYGROUND_EXPLAIN_STAGES:
            if stage not in self._closed:
                self._closed.add(stage)
                self._emit(stage, StageStatus.SKIPPED, reason)

    def _emit(self, stage: StageName, status: StageStatus, detail: Optional[str]) -> None:
        if self._on_stage is None:
            return
        try:
            self._on_stage(stage.value, status.value, detail)
        except Exception as exc:  # noqa: BLE001 - losing the display must not lose the run
            logger.warning("playground on_stage(%s, %s) raised: %s", stage, status, exc)


class _Spend:
    """Running totals across the batches of one pass."""

    def __init__(self) -> None:
        self.cost = 0.0
        self.n_in = 0
        self.n_out = 0
        self.models: List[str] = []

    def add(self, call) -> None:
        self.cost += float(getattr(call, "cost_usd", 0.0) or 0.0)
        self.n_in += int(getattr(call, "input_tokens", 0) or 0)
        self.n_out += int(getattr(call, "output_tokens", 0) or 0)
        model = getattr(call, "model", None)
        if isinstance(model, str) and model not in self.models:
            self.models.append(model)

    def detail(self, prefix: str) -> str:
        # The "$" figure is what nlp_pillars/spend.py adds up for the daily cap.
        # It is the running total, so a pass that dies half-way still counts.
        bits = [prefix]
        if self.models:
            bits.append(", ".join(self.models))
        if self.n_in:
            bits.append(f"{self.n_in:,} in")
        if self.n_out:
            bits.append(f"{self.n_out:,} out")
        bits.append(f"${self.cost:.6f}")
        return " · ".join(bits)


def _join(problems: List[str]) -> Optional[str]:
    if not problems:
        return None
    text = " | ".join(problems)
    if len(text) > _MAX_ERROR_CHARS:
        text = text[:_MAX_ERROR_CHARS].rstrip() + " …(truncated, see logs)"
    return text


def run_explain_job(
    playground_id: str,
    on_stage: Optional[StageCallback] = None,
    cancel: Optional[threading.Event] = None,
    *,
    client: Any = None,
) -> Dict[str, Any]:
    """Write the study guide for one playground. Returns the run's result payload.

    A guide that covers only some cells is saved as ``partial`` and returned
    (the caller records the run as failed, with this payload, so the page can
    say what was saved). A pass that produced nothing raises
    :class:`PlaygroundExplainError` after recording the error on the row; an
    earlier guide, if there was one, is kept.
    """
    stages = _Stages(on_stage, cancel)
    try:
        return _explain(playground_id, stages, cancel, client)
    except RunCancelledError:
        stages.stop_after_current("cancelled")
        raise
    except BaseException as exc:
        stages.stop_after_current(_first_line(exc) if isinstance(exc, Exception) else "stopped")
        raise


def _explain(playground_id, stages: _Stages, cancel, client) -> Dict[str, Any]:
    stages.start(StageName.PLAYGROUND_PREPARE, "Reading the notebook")
    row = store.get(playground_id)
    if row is None:
        stages.failed(StageName.PLAYGROUND_PREPARE, "No such playground")
        raise PlaygroundNotFound(f"Playground not found: {playground_id}")
    nb = row.get("notebook") or {}
    view = cells_view(nb)
    batches = make_batches(view)
    total = sum(len(b.code_indices) for b in batches)
    if not batches:
        stages.failed(StageName.PLAYGROUND_PREPARE, "The notebook has no code to explain")
        raise PlaygroundExplainError("The notebook has no code cells to explain.")
    stages.done(
        StageName.PLAYGROUND_PREPARE,
        f"{total} code cells in {len(batches)} part{'s' if len(batches) != 1 else ''}",
    )

    stages.start(StageName.PLAYGROUND_EXPLAIN, f"Part 1 of {len(batches)}")
    spend = _Spend()
    sources = {c["index"]: c["source"] for c in view if c["type"] == "code"}
    steps: List[Dict[str, Any]] = []
    explained: List[int] = []
    problems: List[str] = []
    incomplete: List[int] = []
    title = row.get("title") or "Notebook"
    for n, batch in enumerate(batches, 1):
        if cancel is not None and cancel.is_set():
            stages.failed(StageName.PLAYGROUND_EXPLAIN, spend.detail("Cancelled"))
            raise RunCancelledError(f"cancelled before part {n} of {len(batches)}")
        try:
            call = structured(
                PLAYGROUND_ROUTE,
                SYSTEM_PROMPT,
                batch.prompt(title, n, len(batches)),
                ExplainDraft,
                client=client,
                max_tokens=MAX_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001 - one part failing keeps the others
            problems.append(f"Part {n} of {len(batches)} failed: {_first_line(exc)}")
            logger.error("Playground %s part %d failed: %s", playground_id, n, exc)
            continue
        spend.add(call)
        checked = validate_draft(call.parsed, wanted=batch.code_indices, sources=sources)
        steps.extend(checked.steps)
        explained.extend(checked.explained)
        problems.extend(checked.problems)
        incomplete.extend(checked.incomplete)
        for note in checked.notes:
            logger.warning("Playground %s: %s", playground_id, note)
        stages.progress(
            StageName.PLAYGROUND_EXPLAIN,
            spend.detail(f"Part {n} of {len(batches)} done"),
        )

    missing = missing_cells(nb, explained)
    if missing and steps:
        shown = ", ".join(str(i) for i in missing[:20]) + (" …" if len(missing) > 20 else "")
        problems.insert(0, f"{len(missing)} of {total} code cells were not explained (cells {shown}).")
    if not steps:
        status = store.STATUS_FAILED
    elif missing or incomplete:
        status = store.STATUS_PARTIAL
    else:
        status = store.STATUS_READY
    error = _join(problems)
    summary = spend.detail(f"{len(explained)} of {total} cells explained")
    if status == store.STATUS_FAILED:
        error = error or "The model returned no usable steps."
        stages.failed(StageName.PLAYGROUND_EXPLAIN, f"{summary} · {error[:200]}")
    else:
        stages.done(StageName.PLAYGROUND_EXPLAIN, summary)

    stages.start(StageName.PLAYGROUND_PERSIST, "Saving the study guide")
    numbered = number_steps(steps)
    try:
        if status == store.STATUS_FAILED and row.get("steps"):
            # Keep the guide the learner already has; say the retry failed.
            store.save_explain_error(playground_id, f"The last attempt failed: {error}")
        else:
            store.save_explain(
                playground_id,
                steps=numbered,
                status=status,
                model=", ".join(spend.models) or PLAYGROUND_ROUTE.resolved_model(),
                error=error,
            )
    except store.PlaygroundStoreError as exc:
        stages.failed(StageName.PLAYGROUND_PERSIST, _first_line(exc))
        raise
    if status == store.STATUS_FAILED:
        stages.done(StageName.PLAYGROUND_PERSIST, "Recorded the error; the notebook is unchanged")
        raise PlaygroundExplainError(error)
    stages.done(StageName.PLAYGROUND_PERSIST, f"{len(numbered)} steps saved ({status})")

    return {
        "playground_id": playground_id,
        "paper_id": row.get("paper_id"),
        "status": status,
        "steps": len(numbered),
        "cells_explained": len(explained),
        "cells_total": total,
        "cost_usd": round(spend.cost, 6),
        "error": error,
    }
