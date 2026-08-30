"""Background worker for podcast script generation with per-call stages.

Mirrors ``podcast_audio_service`` / upload: the route answers 202 with a run id,
APScheduler runs this job, and ``pipeline_run_stages`` carries the five model
calls plus prepare/persist so the browser can poll through ``run-progress.js``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

from nlp_pillars import db
from nlp_pillars.agents.podcast_agent import (
    GroundPackExtractionError,
    InsufficientSourceMaterialError,
    PodcastAgent,
)
from nlp_pillars.db import PodcastScriptSaveError
from nlp_pillars.orchestrator import RunCancelledError, _first_line
from nlp_pillars.podcast_options import PodcastOptions
from nlp_pillars.schemas import (
    PODCAST_SCRIPT_STAGES,
    PodcastScript,
    StageName,
    StageStatus,
)

logger = logging.getLogger(__name__)

KIND_PODCAST_SCRIPT = "podcast_script"
TRIGGER_UI_PODCAST_SCRIPT = "ui_podcast_script"

StageCallback = Callable[[str, str, Optional[str]], None]


class _Stages:
    """Report stage transitions and close out stages that will never run."""

    def __init__(
        self,
        on_stage: Optional[StageCallback] = None,
        cancel: Optional[threading.Event] = None,
    ) -> None:
        self._on_stage = on_stage
        self._cancel = cancel
        self._closed: set = set()

    def start(self, stage: StageName, detail: Optional[str] = None) -> None:
        if self._cancel is not None and self._cancel.is_set():
            raise RunCancelledError(f"cancelled before {stage.value}")
        self._emit(stage, StageStatus.RUNNING, detail)

    def done(self, stage: StageName, detail: Optional[str] = None) -> None:
        self._closed.add(stage)
        self._emit(stage, StageStatus.COMPLETED, detail)

    def failed(self, stage: StageName, detail: str) -> None:
        self._closed.add(stage)
        self._emit(stage, StageStatus.FAILED, detail)

    def skipped(self, stage: StageName, detail: str) -> None:
        if stage in self._closed:
            return
        self._closed.add(stage)
        self._emit(stage, StageStatus.SKIPPED, detail)

    def stop_after_current(self, reason: str) -> None:
        for stage in PODCAST_SCRIPT_STAGES:
            if stage not in self._closed:
                self.skipped(stage, reason)

    def _emit(
        self, stage: StageName, status: StageStatus, detail: Optional[str]
    ) -> None:
        if self._on_stage is None:
            return
        try:
            self._on_stage(stage.value, status.value, detail)
        except Exception as e:  # noqa: BLE001 — never lose the run for UI
            logger.warning(
                "podcast_script on_stage(%s, %s) raised: %s", stage, status, e
            )


def _script_result_payload(
    script: PodcastScript,
    *,
    script_id: Optional[str],
    saved: bool,
    extra_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    warnings = list(script.source_material.warnings)
    if extra_warnings:
        warnings.extend(extra_warnings)
    payload: Dict[str, Any] = {
        "saved": saved,
        "script_id": script_id,
        "title": script.title,
        "word_count": script.word_count,
        "key_points": list(script.key_points or []),
        "source_material_level": script.source_material.level,
        "warnings": warnings,
        "options": script.options.model_dump(),
        "paper_id": script.paper_id,
        "pillar_id": script.pillar_id,
    }
    if not saved:
        # Only copy when there is no other home for the artifact.
        payload["script"] = script.script
    return payload


def run_podcast_script_job(
    run_id: str,
    paper_id: str,
    pillar_id: str,
    options: PodcastOptions,
    on_stage: StageCallback,
    cancel: threading.Event,
) -> Dict[str, Any]:
    """Generate and optionally save one podcast script on a worker thread.

    Returns the ``result`` payload for ``pipeline_runs.result``. Raises
    ``RunCancelledError`` on cancel; other failures raise after marking stages.
    """
    stages = _Stages(on_stage, cancel)

    def agent_on_stage(name: str, status: str, detail: Optional[str] = None) -> None:
        # Agent emits once; we only track terminal stages so stop_after_current
        # does not overwrite completed/failed rows with skipped.
        try:
            stage = StageName(name)
        except ValueError:
            stage = None
        if stage is not None and status in (
            StageStatus.COMPLETED.value,
            StageStatus.FAILED.value,
            StageStatus.SKIPPED.value,
        ):
            stages._closed.add(stage)
        if on_stage is not None:
            try:
                on_stage(name, status, detail)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "podcast_script on_stage(%s, %s) raised: %s", name, status, e
                )

    try:
        agent = PodcastAgent(options=options)
        script = asyncio.run(
            agent.generate(
                paper_id,
                pillar_id,
                on_stage=agent_on_stage,
                cancel=cancel,
            )
        )
    except RunCancelledError:
        stages.stop_after_current("cancelled")
        raise
    except InsufficientSourceMaterialError as e:
        # generate() already marks prepare failed; close the rest.
        stages.stop_after_current(_first_line(e))
        raise
    except GroundPackExtractionError as e:
        stages.stop_after_current(_first_line(e))
        raise
    except BaseException as e:
        stages.stop_after_current(_first_line(e))
        raise

    stages.start(StageName.PODCAST_PERSIST, "Saving the script to the database")
    try:
        script_id = db.add_podcast_script(script)
    except PodcastScriptSaveError as e:
        reason = _first_line(e)
        stages.failed(StageName.PODCAST_PERSIST, reason)
        # Generation succeeded; the artifact lives only in the result payload.
        return _script_result_payload(
            script,
            script_id=None,
            saved=False,
            extra_warnings=[
                f"This script was NOT saved to the database ({e}). It is shown "
                f"below and nowhere else — download or copy it now, or it is lost."
            ],
        )

    stages.done(
        StageName.PODCAST_PERSIST,
        f"Saved as {script_id} · {script.word_count:,} words",
    )
    return _script_result_payload(script, script_id=script_id, saved=True)
