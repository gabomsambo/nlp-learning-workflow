"""The explain pass: turn a notebook's cells into study steps.

One Claude call per batch of cells (``llm_client.structured`` with the small
``ExplainDraft`` schema), then strict validation in Python. The stored shape
is documented in docs/PLAYGROUND.md; the prompt is copied verbatim into
docs/PROMPTS.md and a test keeps the two equal.

Grounding: the model sees only the notebook — its markdown, its code with
line numbers, and its saved outputs. It is told to explain from those and to
invent no results. Validation then enforces the parts that can be checked:
every cell index is a real code cell, cells are in notebook order and appear
once, line notes point at real lines, at most three tries per cell, each try
actually changes the code, and the explanation is one short sentence.

A batch whose answer cannot be used drops only that batch; what the other
batches produced is kept and the guide is stored as ``partial`` with the
reasons, so the notebook is never hidden behind a failed pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from .notebook import cells_view, code_cell_indices, tree_text

#: Code cells per model call. Bounded by the answer, not the prompt: each
#: cell's tries repeat the whole cell, and the answer must fit MAX_TOKENS.
BATCH_CODE_CELLS = 12

#: Prompt characters per batch (cells plus outputs) before a split is forced.
BATCH_CHARS = 60_000

#: Saved output shown to the model per output. Enough for a describe() table.
PROMPT_OUTPUT_CHARS = 3_000

#: Output budget per call. Thinking shares it. 20,000 is the most the
#: Anthropic SDK allows without streaming (it refuses a call whose
#: max_tokens implies more than ten minutes: 3600 * n / 128000 > 600 above
#: 21,333), and llm_client.structured() does not stream.
MAX_TOKENS = 20_000

MAX_TRIES = 3
#: "At most about 25 words": a little slack, then the cell is rejected.
MAX_EXPLANATION_WORDS = 40
MAX_TITLE_CHARS = 80
MAX_LABEL_CHARS = 80


SYSTEM_PROMPT = """You turn a Jupyter notebook into a study guide for someone who must be able to explain every cell out loud, for example in a job interview.

You receive the notebook's cells in order. Markdown cells are context. Code cells carry line numbers ("12 | code") and their SAVED OUTPUT exactly as it was when the notebook last ran.

Group the code cells you are asked to explain into study steps that follow the notebook's own story: consecutive cells that serve one purpose (load the data, check it, build a feature, fit a model, test it) form one step. Every code cell you are asked to explain appears in exactly one step, in notebook order. Do not skip a cell; a trivial cell still gets a short explanation.

For each code cell write:
- explanation: ONE plain sentence, at most 25 words, saying what the cell does and why it matters here. No jargon without a reason.
- line_notes: notes for the NON-OBVIOUS lines only, by line number. Short. Skip lines whose meaning is plain. An empty list is fine.
- tries: zero to three "try this" tweaks, only where changing something teaches something (a parameter, a threshold, a column, a seed). Each has a short imperative label (for example "Use 256 rows per tree"), the FULL modified cell as code (not a fragment, not a diff), and expect: one sentence on what should change and why. Do not offer a try that only renames things or reformats.
- quiz: one question the learner should be able to answer about this cell.

Grounding rules, which are not negotiable:
- Use only what is in the notebook: its code, its text and its saved outputs.
- Never state a number, result or behaviour that the saved outputs do not show. If an output is missing, say what the cell computes, not what it printed.
- An "expect" line predicts the direction or kind of change from the code's logic. Do not invent the new numbers.
- Do not describe libraries, datasets or methods beyond what the code makes evident.
- Do not guess at reasons the notebook does not give. If the why is not in the notebook, say what the code does.

Each step has a short title (a few words) and a goal: one plain sentence saying what the step achieves."""


# -- model-facing schema (small on purpose: Claude rejects large schemas) -------


class LineNoteDraft(BaseModel):
    line: int = Field(description="1-based line number within the cell")
    note: str


class TryDraft(BaseModel):
    label: str
    code: str = Field(description="The full modified cell")
    expect: str


class CellDraft(BaseModel):
    cell_index: int = Field(description="The CELL number from the prompt")
    explanation: str
    line_notes: List[LineNoteDraft]
    tries: List[TryDraft]
    quiz: str


class StepDraft(BaseModel):
    title: str
    goal: str
    cells: List[CellDraft]


class ExplainDraft(BaseModel):
    steps: List[StepDraft]


# -- prompt ----------------------------------------------------------------------


def _clip(text: str, limit: int = PROMPT_OUTPUT_CHARS) -> str:
    text = text.rstrip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[output truncated]"


def output_for_prompt(output: Dict[str, Any]) -> str:
    kind = output.get("kind")
    if kind in ("stream", "text", "error"):
        prefix = "error: " if kind == "error" else ""
        return prefix + _clip(output.get("text", ""))
    if kind == "html":
        plain = output.get("plain")
        return _clip(plain if plain else tree_text(output.get("tree", [])))
    if kind == "png":
        return "[a saved figure (image); its pixels are not shown to you]"
    return "[an output of a kind this guide does not show: " + ", ".join(output.get("mime", [])) + "]"


def render_cell(cell: Dict[str, Any]) -> str:
    """One cell as the model sees it."""
    idx, ctype = cell["index"], cell["type"]
    if ctype != "code":
        return f"=== CELL {idx} ({ctype}) ===\n{cell.get('source', '').strip()}"
    lines = cell.get("source", "").split("\n")
    width = len(str(len(lines)))
    code = "\n".join(f"{n:>{width}} | {line}" for n, line in enumerate(lines, 1))
    parts = [f"=== CELL {idx} (code) ===", code]
    outputs = cell.get("outputs") or []
    if outputs:
        parts.append("--- saved output ---")
        parts.extend(output_for_prompt(o) for o in outputs)
    else:
        parts.append("--- no saved output ---")
    return "\n".join(parts)


@dataclass
class Batch:
    """A run of consecutive cells, and the code cells in it to explain."""

    cells: List[Dict[str, Any]]
    code_indices: List[int]

    def prompt(self, title: str, number: int, total: int) -> str:
        wanted = ", ".join(str(i) for i in self.code_indices)
        head = (
            f"Notebook: {title}\n"
            f"Part {number} of {total}. Explain these code cells, and only these: {wanted}.\n"
            "Cells from other parts are not shown; do not refer to their outputs.\n"
        )
        return head + "\n\n" + "\n\n".join(render_cell(c) for c in self.cells)


def make_batches(
    cells: Sequence[Dict[str, Any]],
    *,
    max_code_cells: int = BATCH_CODE_CELLS,
    max_chars: int = BATCH_CHARS,
) -> List[Batch]:
    """Split the notebook into consecutive batches. Markdown travels with the
    code it precedes; a batch closes before it would exceed either limit."""
    explainable = set()
    for cell in cells:
        if cell["type"] == "code" and cell.get("source", "").strip():
            explainable.add(cell["index"])
    batches: List[Batch] = []
    current: List[Dict[str, Any]] = []
    codes: List[int] = []
    size = 0
    for cell in cells:
        rendered = len(render_cell(cell))
        is_code = cell["index"] in explainable
        if codes and is_code and (len(codes) >= max_code_cells or size + rendered > max_chars):
            # Close the batch, carrying trailing markdown forward as the next one's context.
            tail: List[Dict[str, Any]] = []
            while current and current[-1]["index"] not in explainable:
                tail.insert(0, current.pop())
            batches.append(Batch(cells=current, code_indices=codes))
            current, codes = tail, []
            size = sum(len(render_cell(c)) for c in tail)
        current.append(cell)
        size += rendered
        if is_code:
            codes.append(cell["index"])
    if codes:
        batches.append(Batch(cells=current, code_indices=codes))
    return batches


# -- validation ----------------------------------------------------------------


@dataclass
class Validated:
    """Steps that passed, and every reason something did not."""

    steps: List[Dict[str, Any]] = field(default_factory=list)
    explained: List[int] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)  # dropped details, not gaps


def _words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _one_line(text: Any) -> str:
    return " ".join(str(text or "").split())


def validate_draft(
    draft: ExplainDraft,
    *,
    wanted: Sequence[int],
    sources: Dict[int, str],
) -> Validated:
    """Keep what is usable from one batch's answer.

    ``wanted`` are the code cell indices this batch had to explain, in order;
    ``sources`` maps a code cell index to its source.
    """
    out = Validated()
    allowed = set(wanted)
    last = -1
    for s_num, step in enumerate(draft.steps, 1):
        title = _one_line(step.title)[:MAX_TITLE_CHARS]
        goal = _one_line(step.goal)
        if not title or not goal:
            out.problems.append(f"Step {s_num} had no title or goal and was dropped.")
            continue
        kept_cells = []
        for cell in step.cells:
            idx = cell.cell_index
            if idx not in allowed:
                out.problems.append(f"Cell {idx} is not a code cell in this part; dropped.")
                continue
            if idx in out.explained:
                out.problems.append(f"Cell {idx} was explained twice; the repeat was dropped.")
                continue
            if idx < last:
                out.problems.append(f"Cell {idx} was out of notebook order; dropped.")
                continue
            checked = _validate_cell(cell, sources.get(idx, ""), out)
            if checked is None:
                continue
            kept_cells.append(checked)
            out.explained.append(idx)
            last = idx
        if kept_cells:
            out.steps.append({"title": title, "goal": goal, "cells": kept_cells})
    return out


def _validate_cell(cell: CellDraft, source: str, out: Validated) -> Optional[Dict[str, Any]]:
    idx = cell.cell_index
    explanation = _one_line(cell.explanation)
    quiz = _one_line(cell.quiz)
    if not explanation:
        out.problems.append(f"Cell {idx} had no explanation; dropped.")
        return None
    if _words(explanation) > MAX_EXPLANATION_WORDS:
        out.problems.append(
            f"Cell {idx}'s explanation was {_words(explanation)} words "
            f"(limit {MAX_EXPLANATION_WORDS}); dropped."
        )
        return None
    if not quiz:
        out.problems.append(f"Cell {idx} had no quiz question; dropped.")
        return None

    n_lines = len(source.split("\n"))
    notes: Dict[int, str] = {}
    for note in cell.line_notes:
        text = _one_line(note.note)
        if not (1 <= note.line <= n_lines) or not text:
            out.notes.append(f"Cell {idx}: line note for line {note.line} dropped.")
            continue
        notes.setdefault(note.line, text)

    tries = []
    for t in cell.tries:
        label = _one_line(t.label)[:MAX_LABEL_CHARS]
        expect = _one_line(t.expect)
        code = (t.code or "").rstrip()
        if not label or not expect or not code.strip():
            out.notes.append(f"Cell {idx}: an incomplete try was dropped.")
            continue
        if code.strip() == source.strip():
            out.notes.append(f"Cell {idx}: try “{label}” did not change the code; dropped.")
            continue
        if len(tries) == MAX_TRIES:
            out.notes.append(f"Cell {idx}: more than {MAX_TRIES} tries; extras dropped.")
            break
        tries.append({"label": label, "code": code, "expect": expect})

    return {
        "cell_index": idx,
        "explanation": explanation,
        "line_notes": [{"line": line, "note": notes[line]} for line in sorted(notes)],
        "tries": tries,
        "quiz": quiz,
    }


def number_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Give the final step list stable ids in order: step-1, step-2, …"""
    return [{"id": f"step-{n}", **step} for n, step in enumerate(steps, 1)]


def sources_by_index(nb: Dict[str, Any]) -> Dict[int, str]:
    view = cells_view(nb)
    return {c["index"]: c["source"] for c in view if c["type"] == "code"}


def missing_cells(nb: Dict[str, Any], explained: Sequence[int]) -> List[int]:
    done = set(explained)
    return [i for i in code_cell_indices(nb) if i not in done]
