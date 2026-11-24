"""How a finished daily or selected-papers run is reported on its run row.

Shared by the webui's background runs (``webui/services/run_service.py``) and the
scheduler container's daily run (``nlp_pillars/scheduler.py``), which record the
same ``pipeline_runs`` rows and must agree on what "succeeded" means.
"""

from typing import Optional

from .schemas import RunStatus


#: The `step` the orchestrator stamps on a per-paper failure. Everything else in
#: `errors` is a failure of the shared plumbing (search, queue, database), which must
#: not be counted as a dead paper.
_PAPER_FAILURE_STEP = "process_paper"


def count_failed_papers(result) -> int:
    """How many PAPERS failed, as opposed to how many things went wrong."""
    return sum(
        1 for e in result.errors
        if isinstance(e, dict) and e.get("step") == _PAPER_FAILURE_STEP
    )


def terminal_status(result) -> str:
    """Map a PipelineResult onto a run status a user can act on.

    ``PipelineResult.success`` is ``len(papers_processed) > 0``, so a run that found
    nothing new — the ordinary outcome once a pillar has caught up — arrives here as
    ``success=False`` with an empty ``errors`` list. That used to be recorded as
    ``failed``, and since there was no error to report the browser rendered a bare red
    "Failed" above eleven green completed stages. Nothing had gone wrong; there was
    simply nothing to do, and the run said so on every stage.

    ``success`` itself is deliberately left alone: the CLI, the scheduler and fourteen
    existing tests read it, and "did this run produce anything" is the right meaning
    for all of them. Only the user-facing run status needed to learn the difference.
    """
    if result.success or not result.errors:
        return RunStatus.SUCCEEDED.value
    return RunStatus.FAILED.value


#: Room for several messages inside the 2000-char cap finish_pipeline_run applies.
_MAX_ERROR_CHARS = 1800


def summarise_errors(result) -> Optional[str]:
    """Put every paper's failure on the run row, not just the first.

    Stage rows cannot carry this: they are per-run, not per-paper, so when paper 2
    re-enters a stage that paper 1 failed in it overwrites the failure back to
    completed. The run-level error string is the only place a multi-paper run can
    report more than one casualty, and reporting only ``errors[0]`` meant a run with
    three failures looked like it had one.
    """
    if not result.errors:
        return None

    messages = [
        e.get("message") if isinstance(e, dict) else str(e) for e in result.errors
    ]
    messages = [m for m in messages if m]
    if not messages:
        return None
    if len(messages) == 1:
        return messages[0]

    joined = f"{len(messages)} papers failed: " + " | ".join(messages)
    if len(joined) > _MAX_ERROR_CHARS:
        joined = joined[:_MAX_ERROR_CHARS].rstrip() + " …(truncated, see logs)"
    return joined
