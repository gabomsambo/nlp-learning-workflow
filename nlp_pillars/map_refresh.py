"""Keeping the paper map's evidence fresh: ingest hooks and the weekly refresh.

Two things feed ``map build`` (see ``map_build.py``): the typed citations
``works sync`` stores, and the library itself. This module is the glue that
runs them at the right times:

* ``sync_new_paper`` is the hook a paper's ingest path (upload, select, daily
  run) calls once the paper is in the library. It runs the same sync ``works
  sync`` runs, narrowed to that one paper.
* ``run_weekly_refresh`` is the scheduler's weekly job: the library-wide sync
  (metadata and counts refreshed, citers paged until a known id appears), then
  one ``map_build`` run per pillar.

Both honour the rule the VECTORS stage set: the paper is already saved, so a
Semantic Scholar outage, a missing migration or a bad id is reported and never
raised. Every Semantic Scholar request goes through ``SemanticScholarTool``,
and so through the shared limiter; the hook asks at ``INTERACTIVE`` priority,
the weekly job at ``BULK`` so it queues behind anything a person is waiting on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from . import db
from .citations import is_file_upload
from .schemas import MAP_BUILD_STAGES, RunStatus
from .work_ids import best_lookup_key, s2_lookup_id
from .works import LibraryPaper, WorksSyncReport, run_works_sync

logger = logging.getLogger(__name__)

#: ``pipeline_runs.trigger_source`` of a build the scheduler runs itself
#: (allowed since migration 034; ``ui_map`` is the webui's).
TRIGGER_SCHEDULER = "scheduler"


@dataclass
class HookOutcome:
    """What a hook did. ``ok`` is False only for a failure; a skip is ok."""

    ok: bool
    detail: str = ""
    edges: int = 0
    #: The works tables are missing (migration 031 not applied). Callers that
    #: still have the old ``paper_citations`` path use it instead.
    legacy: bool = False

    def __bool__(self) -> bool:
        return self.ok


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return (text.splitlines()[0] if text else type(exc).__name__)[:180]


def sync_new_paper(
    paper_id: str,
    *,
    tool=None,
    load_library: Optional[Callable[[], Sequence[LibraryPaper]]] = None,
    load_index: Optional[Callable[[], object]] = None,
    store=None,
    sync: Callable[..., WorksSyncReport] = run_works_sync,
) -> HookOutcome:
    """Identity and typed citations for one paper that just entered the library.

    Never raises. ``tool`` is a ``SemanticScholarTool`` (built here, at
    ``INTERACTIVE`` priority, when omitted). The other arguments are the seams
    the tests use; production leaves them ``None``.
    """
    try:
        if not paper_id or is_file_upload(paper_id):
            return HookOutcome(True, "a personal document has no Semantic Scholar record")

        from . import works_store

        library = (load_library or works_store.load_library)()
        paper = next((p for p in library if p.id == paper_id), None)
        if paper is None:
            return HookOutcome(False, f"{paper_id} is not in the library")
        key = best_lookup_key(paper.keys())
        if not (key and s2_lookup_id(key)):
            return HookOutcome(True, "no id Semantic Scholar can look up")

        if tool is None:
            from .tools.semantic_scholar_tool import SemanticScholarTool

            tool = SemanticScholarTool()
        report = sync(
            [paper],
            (load_index or works_store.load_index)(),
            tool,
            store if store is not None else works_store.PostgrestWorks(),
            dry_run=False,
        )
    except Exception as exc:  # noqa: BLE001 - a hook must not fail the paper
        from .works_store import WorksUnavailable

        logger.warning("Citation sync failed for %s: %s", paper_id, _first_line(exc))
        return HookOutcome(
            False, _first_line(exc), legacy=isinstance(exc, WorksUnavailable)
        )

    problems = [*report.failures, *([report.stopped_early] if report.stopped_early else [])]
    if problems:
        logger.warning("Citation sync for %s was incomplete: %s", paper_id, problems[0])
        return HookOutcome(False, problems[0], report.reference_edges + report.citer_edges)
    if paper_id in report.not_in_s2:
        return HookOutcome(True, "Semantic Scholar does not know this paper")
    return HookOutcome(True, "", report.reference_edges + report.citer_edges)


@dataclass
class MapBuildsReport:
    """One line per pillar: the run id, or why none was made."""

    started: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)


def build_maps_inline(
    pillar_ids: Sequence[str],
    *,
    trigger_source: str = TRIGGER_SCHEDULER,
    run_job: Optional[Callable[..., dict]] = None,
) -> MapBuildsReport:
    """Run a ``map_build`` for each pillar in this process, one at a time.

    For callers with no web app to dispatch through (the scheduler container).
    Each build is a ``pipeline_runs`` row, so the Activity page shows it like
    any other. Never raises: a pillar whose build fails is recorded on its run
    row and the next pillar still runs.
    """
    report = MapBuildsReport()
    if run_job is None:
        from . import map_store

        run_job = map_store.run_map_build_job
    stage_names = [stage.value for stage in MAP_BUILD_STAGES]
    for pillar_id in pillar_ids:
        run_id: Optional[str] = None
        try:
            run = db.create_pipeline_run(pillar_id, trigger_source, "map_build", stage_names)
            if run is None:
                report.failed.append(f"{pillar_id}: a map build is already running")
                continue
            run_id = run.id
            db.start_pipeline_run(run_id)
            payload = run_job(
                pillar_id, _stage_writer(run_id), lambda: None, run_id=run_id
            )
            db.finish_pipeline_run(
                run_id, RunStatus.SUCCEEDED.value, result=payload
            )
            report.started.append(run_id)
        except Exception as exc:  # noqa: BLE001 - one pillar must not stop the rest
            logger.warning("Map build for %s failed: %s", pillar_id, _first_line(exc))
            report.failed.append(f"{pillar_id}: {_first_line(exc)}")
            if run_id:
                try:
                    db.finish_pipeline_run(
                        run_id, RunStatus.FAILED.value, error=_first_line(exc)
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Could not record map build %s as failed", run_id)
    return report


def _stage_writer(run_id: str):
    def on_stage(name: str, status: str, detail: Optional[str] = None) -> None:
        if status == "dropped":
            db.delete_pipeline_run_stage(run_id, name)
        else:
            db.update_pipeline_run_stage(run_id, name, status, detail)

    return on_stage


def run_weekly_refresh(
    *,
    sync: Optional[Callable[[], WorksSyncReport]] = None,
    list_pillars: Optional[Callable[[], Sequence[str]]] = None,
    build: Callable[..., MapBuildsReport] = build_maps_inline,
) -> Optional[WorksSyncReport]:
    """The scheduler's weekly job: refresh the evidence, then rebuild every map.

    The sync is the library-wide one (``works sync``): it re-reads each library
    paper's counts and metadata and pages new citers only until it reaches a
    citer it already holds. If it stops early (Semantic Scholar unavailable)
    the maps are still rebuilt from what is stored, since the library may have
    changed since the last build. Never raises.
    """
    report: Optional[WorksSyncReport] = None
    try:
        if sync is None:
            from .works_store import execute_works_sync

            sync = lambda: execute_works_sync(dry_run=False)  # noqa: E731
        report = sync()
        logger.warning("Weekly refresh: %s", report.render().replace("\n", "; "))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Weekly refresh: works sync failed: %s", _first_line(exc))

    try:
        if list_pillars is None:
            list_pillars = lambda: [p.id for p in db.get_pillars(limit=50)]  # noqa: E731
        pillar_ids = list(list_pillars())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Weekly refresh: could not list pillars: %s", _first_line(exc))
        return report
    result = build(pillar_ids)
    logger.warning(
        "Weekly refresh: %d map build(s) finished, %d did not",
        len(result.started), len(result.failed),
    )
    for line in result.failed:
        logger.warning("Weekly refresh: %s", line)
    return report
