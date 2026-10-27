"""Follow-along: map an episode's script lines to a playground's code cells.

One cheap Claude call (``PLAYGROUND_ALIGN_ROUTE``, Haiku by default) sees the
script's spoken lines with their line numbers and the playground's study
steps (step ids, cell numbers, titles, explanations and each cell's code,
truncated). It answers with spans ``{line_start, line_end, cell_index,
step_id}``. Python then keeps only spans that check out, so nothing the
Listen page shows can point at a line or a cell that is not there:

- both ends are spoken script lines (``script.splitlines()`` indexes, the
  numbering the stored read-along timeline uses), and start <= end;
- ``cell_index`` is a cell of the study guide and ``step_id`` is the step
  that lists it;
- spans do not overlap; they are stored in script order.

An answer that is not an :class:`AlignDraft` is malformed and fails the pass.
An answer whose every span was dropped fails it too. Lines about no
particular code have no span, and an answer with no spans at all is a valid,
empty mapping.

The stored shape is docs/PLAYGROUND.md, "Follow-along". The prompt is copied
into docs/PROMPTS.md and a test keeps the two equal.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from .. import db
from ..llm_client import StructuredCall, stage_detail, structured
from ..llm_routes import PLAYGROUND_ALIGN_ROUTE
from ..orchestrator import RunCancelledError, _first_line
from ..schemas import PLAYGROUND_ALIGN_STAGES, StageName, StageStatus
from ..tts.cue_parser import spoken_line_text
from . import links, store
from .notebook import cells_view

logger = logging.getLogger(__name__)

KIND_PLAYGROUND_ALIGN = links.KIND_PLAYGROUND_ALIGN

#: Code shown to the model per cell. Enough to recognise the cell from what
#: the narrator says about it; the explanation carries the rest.
CODE_LINES_PER_CELL = 30
CODE_CHARS_PER_CELL = 1_200

#: Output budget. A span is about forty tokens; a long episode has a few dozen.
MAX_TOKENS = 8_000

_MAX_ERROR_CHARS = 1_800

StageCallback = Callable[[str, str, Optional[str]], None]


SYSTEM_PROMPT = """You match the lines of a podcast episode's script to the code cells of a study notebook, so that a listener sees the code the narrator is talking about while it is being talked about.

You receive the script's spoken lines, each with its line number (for example "L12: ..."), and the notebook's study steps. Each step has an id, a title and a goal, and lists its code cells by CELL number with a one-line explanation and the code.

Return spans. A span is a run of consecutive script lines, line_start to line_end inclusive, that is clearly about one code cell: it describes what that cell does, its code, its parameters, its output or its result. Give that cell's CELL number as cell_index and the id of the step that lists it as step_id. When the lines are about a step as a whole rather than one of its cells, use the first cell of that step.

Rules, which are not negotiable:
- Make a span only when the link is clear. Lines that are about no particular code (the introduction, background, the paper's motivation, recaps, transitions, the sign-off) get no span. It is normal for many lines to have no span.
- Use the L numbers exactly as given. line_start and line_end are both numbers from the list.
- Spans do not overlap, and they are listed in script order.
- Use only CELL numbers and step ids from the list. Never invent one.
- Prefer one span over several lines about the same cell to many one-line spans, but end the span where the script moves on to something else."""


# -- model-facing schema -----------------------------------------------------------


class SpanDraft(BaseModel):
    line_start: int = Field(description="First script line (the L number) of the span")
    line_end: int = Field(description="Last script line (the L number), inclusive")
    cell_index: int = Field(description="The CELL number the lines are about")
    step_id: str = Field(description="The id of the step that lists that cell")


class AlignDraft(BaseModel):
    spans: List[SpanDraft]


# -- inputs ------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    """The script hash the fact check and the read-along timeline store."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PlaygroundAlignError(RuntimeError):
    """The pass produced nothing usable, or could not start. Says why."""


class AlignTargetMissing(PlaygroundAlignError):
    """There is no such episode or no such playground."""


def spoken_lines(script_text: str) -> List[Tuple[int, str]]:
    """``(line index, spoken text)`` for every line the narrator says.

    Indexes are ``script.splitlines()`` positions, the numbering
    ``tts/alignment.py`` stores in the read-along timeline.
    """
    out = []
    for index, raw in enumerate((script_text or "").splitlines()):
        spoken = spoken_line_text(raw)
        if spoken:
            out.append((index, " ".join(spoken.split())))
    return out


def cell_steps(steps: Sequence[Dict[str, Any]]) -> Dict[int, str]:
    """Cell index -> the id of the step that lists it."""
    out: Dict[int, str] = {}
    for step in steps or []:
        step_id = step.get("id")
        for cell in step.get("cells") or []:
            index = cell.get("cell_index")
            if isinstance(index, int) and isinstance(step_id, str):
                out.setdefault(index, step_id)
    return out


def _clip_code(source: str) -> str:
    lines = (source or "").rstrip().split("\n")
    clipped = "\n".join(lines[:CODE_LINES_PER_CELL])
    cut = len(lines) > CODE_LINES_PER_CELL
    if len(clipped) > CODE_CHARS_PER_CELL:
        clipped = clipped[:CODE_CHARS_PER_CELL].rstrip()
        cut = True
    return clipped + ("\n[code truncated]" if cut else "")


def build_prompt(
    *,
    episode_title: str,
    playground_title: str,
    lines: Sequence[Tuple[int, str]],
    steps: Sequence[Dict[str, Any]],
    sources: Dict[int, str],
) -> str:
    """The user message: the spoken lines, then the steps and their cells."""
    parts = [
        f"Episode: {episode_title or 'Untitled episode'}",
        f"Notebook: {playground_title or 'Notebook'}",
        "",
        "=== SCRIPT (spoken lines) ===",
    ]
    parts.extend(f"L{index}: {text}" for index, text in lines)
    parts += ["", "=== STUDY STEPS ==="]
    for step in steps:
        parts.append("")
        parts.append(f"STEP {step.get('id')}: {step.get('title') or ''}")
        if step.get("goal"):
            parts.append(f"Goal: {step['goal']}")
        for cell in step.get("cells") or []:
            index = cell.get("cell_index")
            parts.append(f"--- CELL {index} ---")
            if cell.get("explanation"):
                parts.append(f"Explanation: {cell['explanation']}")
            parts.append("Code:")
            parts.append(_clip_code(sources.get(index, "")))
    return "\n".join(parts)


# -- validation --------------------------------------------------------------------


@dataclass
class Checked:
    """Spans that passed, in script order, and why any others did not."""

    spans: List[Dict[str, Any]] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    offered: int = 0


def validate_spans(
    draft: Any,
    *,
    spoken: Sequence[int],
    cells: Dict[int, str],
) -> Checked:
    """Keep the spans that check out.

    ``spoken`` are the script line indexes the model was shown; ``cells``
    maps each study-guide cell to its step id.

    Raises:
        PlaygroundAlignError: the answer is not an ``AlignDraft`` (malformed).
    """
    if not isinstance(draft, AlignDraft):
        raise PlaygroundAlignError(
            f"The model's answer was malformed ({type(draft).__name__}, not a list of spans)."
        )
    allowed = set(spoken)
    out = Checked(offered=len(draft.spans))
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    for n, span in enumerate(draft.spans, 1):
        a, b = span.line_start, span.line_end
        where = f"Span {n} (lines {a}-{b})"
        if a not in allowed or b not in allowed:
            out.problems.append(f"{where} does not start and end on spoken lines; dropped.")
            continue
        if a > b:
            out.problems.append(f"{where} ends before it starts; dropped.")
            continue
        if span.cell_index not in cells:
            out.problems.append(f"{where} names cell {span.cell_index}, which is not in the study guide; dropped.")
            continue
        step_id = (span.step_id or "").strip()
        if cells[span.cell_index] != step_id:
            out.problems.append(
                f"{where} names step {step_id!r}, but cell {span.cell_index} is in "
                f"{cells[span.cell_index]}; dropped."
            )
            continue
        candidates.append((n, {
            "line_start": a, "line_end": b,
            "cell_index": span.cell_index, "step_id": step_id,
        }))
    candidates.sort(key=lambda item: (item[1]["line_start"], item[1]["line_end"], item[0]))
    for n, span in candidates:
        last = out.spans[-1] if out.spans else None
        if last is not None and span["line_start"] <= last["line_end"]:
            out.problems.append(
                f"Span {n} (lines {span['line_start']}-{span['line_end']}) overlaps "
                f"lines {last['line_start']}-{last['line_end']}; dropped."
            )
            continue
        out.spans.append(span)
    return out


# -- the job -----------------------------------------------------------------------


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

    def done(self, stage: StageName, detail: Optional[str] = None) -> None:
        self._closed.add(stage)
        self._emit(stage, StageStatus.COMPLETED, detail)

    def failed(self, stage: StageName, detail: str) -> None:
        self._closed.add(stage)
        self._emit(stage, StageStatus.FAILED, detail)

    def stop_after_current(self, reason: str) -> None:
        for stage in PLAYGROUND_ALIGN_STAGES:
            if stage not in self._closed:
                self._closed.add(stage)
                self._emit(stage, StageStatus.SKIPPED, reason)

    def _emit(self, stage: StageName, status: StageStatus, detail: Optional[str]) -> None:
        if self._on_stage is None:
            return
        try:
            self._on_stage(stage.value, status.value, detail)
        except Exception as exc:  # noqa: BLE001 - losing the display must not lose the run
            logger.warning("follow-along on_stage(%s, %s) raised: %s", stage, status, exc)


def _join(problems: List[str]) -> Optional[str]:
    if not problems:
        return None
    text = " | ".join(problems)
    if len(text) > _MAX_ERROR_CHARS:
        text = text[:_MAX_ERROR_CHARS].rstrip() + " …(truncated, see logs)"
    return text


@dataclass
class Prepared:
    """Everything the model call needs, read before any money is spent."""

    script: Any
    playground: Dict[str, Any]
    link: Dict[str, Any]
    lines: List[Tuple[int, str]]
    cells: Dict[int, str]
    prompt: str


def prepare(script_id: str, playground_id: str) -> Prepared:
    """Read the episode and the playground; refuse what cannot be aligned.

    Raises:
        PlaygroundAlignError: no such episode or playground, no spoken lines,
            or a playground without a study guide.
        store.PlaygroundStoreError: a read or the link insert failed (a
            missing table names its migration).
    """
    script_id = links.canonical_id(script_id) or script_id
    try:
        script = db.get_podcast_script_by_id(script_id)
    except db.PodcastScriptLookupError as exc:
        raise store.PlaygroundStoreError(str(exc)) from exc
    if script is None:
        raise AlignTargetMissing(f"No episode with id {script_id}.")
    playground = store.get(playground_id, with_notebook=True)
    if playground is None:
        raise AlignTargetMissing(f"No playground with id {playground_id}.")
    steps = playground.get("steps") or []
    cells = cell_steps(steps)
    if not cells:
        raise PlaygroundAlignError(
            "This playground has no study guide yet, so there is nothing to follow. "
            "Write it on the paper's Playground tab first."
        )
    lines = spoken_lines(script.script or "")
    if not lines:
        raise PlaygroundAlignError("The episode's script has no spoken lines.")
    link = links.ensure_link(script_id, playground["id"])
    sources = {
        c["index"]: c.get("source") or ""
        for c in cells_view(playground.get("notebook") or {})
        if c["type"] == "code"
    }
    prompt = build_prompt(
        episode_title=script.title or "",
        playground_title=playground.get("title") or "",
        lines=lines,
        steps=steps,
        sources=sources,
    )
    return Prepared(script, playground, link, lines, cells, prompt)


def run_align_job(
    script_id: str,
    playground_id: str,
    on_stage: Optional[StageCallback] = None,
    cancel: Optional[threading.Event] = None,
    *,
    client: Any = None,
) -> Dict[str, Any]:
    """Align one episode to one playground and store the mapping.

    Returns the run's result payload. A pass with nothing usable raises
    :class:`PlaygroundAlignError` after recording why on the link; a link
    that already had a mapping keeps it.
    """
    stages = _Stages(on_stage, cancel)
    try:
        return _align(script_id, playground_id, stages, cancel, client)
    except RunCancelledError:
        stages.stop_after_current("cancelled")
        raise
    except BaseException as exc:
        stages.stop_after_current(_first_line(exc) if isinstance(exc, Exception) else "stopped")
        raise


def _record_failure(link: Dict[str, Any], reason: str) -> None:
    try:
        links.save_failure(link, reason)
    except store.PlaygroundStoreError as exc:
        logger.error("Could not record the follow-along failure on %s: %s", link.get("id"), exc)


def _align(script_id, playground_id, stages: _Stages, cancel, client) -> Dict[str, Any]:
    stages.start(StageName.PLAYGROUND_ALIGN_PREPARE, "Reading the episode and the study guide")
    try:
        prep = prepare(script_id, playground_id)
    except (PlaygroundAlignError, store.PlaygroundStoreError, ValueError) as exc:
        stages.failed(StageName.PLAYGROUND_ALIGN_PREPARE, _first_line(exc))
        raise
    n_steps = len(prep.playground.get("steps") or [])
    stages.done(
        StageName.PLAYGROUND_ALIGN_PREPARE,
        f"{len(prep.lines)} spoken lines; {len(prep.cells)} cells in {n_steps} step"
        f"{'s' if n_steps != 1 else ''}",
    )

    stages.start(StageName.PLAYGROUND_ALIGN, "Matching script lines to code cells")
    try:
        call: StructuredCall = structured(
            PLAYGROUND_ALIGN_ROUTE,
            SYSTEM_PROMPT,
            prep.prompt,
            AlignDraft,
            client=client,
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001 - the reason is recorded, then raised
        reason = f"The alignment call failed: {_first_line(exc)}"
        stages.failed(StageName.PLAYGROUND_ALIGN, reason)
        _record_failure(prep.link, reason)
        raise PlaygroundAlignError(reason) from exc

    try:
        checked = validate_spans(
            call.parsed, spoken=[i for i, _ in prep.lines], cells=prep.cells,
        )
    except PlaygroundAlignError as exc:
        # The cost is in the detail: the call was paid for either way.
        stages.failed(StageName.PLAYGROUND_ALIGN, stage_detail(call, str(exc)))
        _record_failure(prep.link, str(exc))
        raise
    for problem in checked.problems:
        logger.warning("Follow-along %s/%s: %s", script_id, playground_id, problem)
    if checked.offered and not checked.spans:
        reason = "None of the model's spans checked out: " + (_join(checked.problems) or "")
        stages.failed(StageName.PLAYGROUND_ALIGN, stage_detail(call, reason[:300]))
        _record_failure(prep.link, reason)
        raise PlaygroundAlignError(reason)
    summary = f"{len(checked.spans)} span{'s' if len(checked.spans) != 1 else ''}"
    if checked.problems:
        summary += f", {len(checked.problems)} dropped"
    stages.done(StageName.PLAYGROUND_ALIGN, stage_detail(call, summary))

    stages.start(StageName.PLAYGROUND_ALIGN_PERSIST, "Saving the alignment")
    error = _join(checked.problems)
    if not checked.spans:
        error = "No line of this episode is clearly about a cell of this notebook."
    try:
        links.save_mapping(
            prep.link["id"],
            mapping=checked.spans,
            model=call.model,
            error=error,
            script_sha256=sha256_text(prep.script.script or ""),
        )
    except store.PlaygroundStoreError as exc:
        stages.failed(StageName.PLAYGROUND_ALIGN_PERSIST, _first_line(exc))
        raise
    stages.done(StageName.PLAYGROUND_ALIGN_PERSIST, f"{summary} saved")

    covered = sum(
        1 for index, _ in prep.lines
        if any(s["line_start"] <= index <= s["line_end"] for s in checked.spans)
    )
    return {
        "script_id": str(prep.script.id),
        "playground_id": prep.playground["id"],
        "paper_id": prep.playground.get("paper_id"),
        "link_id": prep.link["id"],
        "status": links.STATUS_READY,
        "spans": len(checked.spans),
        "lines_covered": covered,
        "lines_total": len(prep.lines),
        "dropped": len(checked.problems),
        "model": call.model,
        "cost_usd": round(float(call.cost_usd or 0.0), 6),
        "error": error,
    }
