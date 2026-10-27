"""The "Explain the difference" button: two or three plain sentences on a Before / After.

On the Playground tab a "try this" tweak is run next to the cell's own output.
This asks a cheap Claude model (``LLM_PLAYGROUND_DIFF_MODEL``, default the
Haiku the app already uses) what changed between the two outputs and why,
given only the two pieces of code, the two outputs and the cell's one-line
explanation. The answer is shown inline and not stored.

The prompt is copied in ``docs/PROMPTS.md``; ``tests/test_playground_difference.py``
fails if the copy drifts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..llm_client import structured
from ..llm_routes import PLAYGROUND_DIFF_ROUTE

logger = logging.getLogger(__name__)

#: Characters of each output the model sees. Long outputs are cut in the
#: middle so both the head (shapes, first rows) and the tail (final numbers)
#: survive.
OUTPUT_CHARS = 3_000
#: Characters of each piece of code the model sees.
CODE_CHARS = 6_000
#: The answer is two or three sentences; this only stops a runaway call.
MAX_TOKENS = 1_000
#: What the page shows at most, if the model ignores the length rule.
MAX_ANSWER_CHARS = 800

SYSTEM_PROMPT = """You help someone studying a Jupyter notebook. They ran one code cell two ways: BEFORE (the cell as it was) and AFTER (a changed version). You get both pieces of code, both outputs exactly as printed, and a one-line explanation of what the cell does.

Say in two or three short, plain sentences what changed in the output and why the code change caused it.

Rules, which are not negotiable:
- Use only the code and the two outputs you are given. Quote numbers only as they appear in the outputs.
- If the outputs do not show why, say what changed and that the outputs do not show the reason. Do not guess.
- If the outputs are the same, say so plainly.
- An output marked [cut] is incomplete; do not reason about what is missing.
- No preamble, no lists, no markdown. Plain sentences only."""


class DifferenceDraft(BaseModel):
    explanation: str = Field(description="Two or three plain sentences.")


class DifferenceError(RuntimeError):
    """The model could not answer. The message says why."""


class DifferenceUnavailable(DifferenceError):
    """The call could not be made at all: no API key, or no price for the model."""


@dataclass(frozen=True)
class Difference:
    explanation: str
    model: str
    cost_usd: float


def clip(text: Optional[str], limit: int) -> str:
    """Cut ``text`` to ``limit`` characters in the middle, and say so."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n[cut: {len(text) - limit:,} characters left out]\n{text[-tail:]}"


def build_prompt(
    *,
    before_code: str,
    after_code: str,
    before_output: str,
    after_output: str,
    explanation: str = "",
    expect: str = "",
) -> str:
    """The user message. Every part is delimited so code cannot pose as a rule."""
    parts = []
    if explanation.strip():
        parts.append(f"What the cell does: {' '.join(explanation.split())}")
    if expect.strip():
        parts.append(f"What the change was expected to do: {' '.join(expect.split())}")
    parts += [
        "=== BEFORE: code ===", clip(before_code, CODE_CHARS) or "(empty)",
        "=== BEFORE: output ===", clip(before_output, OUTPUT_CHARS) or "(no output)",
        "=== AFTER: code ===", clip(after_code, CODE_CHARS) or "(empty)",
        "=== AFTER: output ===", clip(after_output, OUTPUT_CHARS) or "(no output)",
        "=== END ===",
    ]
    return "\n".join(parts)


def tidy(answer: Any) -> str:
    """One paragraph, no markdown bullets, capped."""
    text = " ".join(str(answer or "").split())
    text = re.sub(r"^[-*•]\s+", "", text)
    if len(text) > MAX_ANSWER_CHARS:
        cut = text[:MAX_ANSWER_CHARS]
        stop = cut.rfind(". ")
        text = cut[: stop + 1] if stop > 200 else cut.rstrip() + "…"
    return text


def explain_difference(*, client: Any = None, **fields: str) -> Difference:
    """One cheap call. Raises :class:`DifferenceError` when it fails."""
    try:
        call = structured(
            PLAYGROUND_DIFF_ROUTE,
            SYSTEM_PROMPT,
            build_prompt(**fields),
            DifferenceDraft,
            client=client,
            max_tokens=MAX_TOKENS,
        )
    except ValueError as exc:  # no API key, no price for the model
        raise DifferenceUnavailable(str(exc)) from exc
    except Exception as exc:
        raise DifferenceError(f"The model could not explain this difference: {exc}") from exc
    text = tidy(getattr(call.parsed, "explanation", ""))
    if not text:
        raise DifferenceError("The model returned an empty answer.")
    return Difference(explanation=text, model=call.model, cost_usd=call.cost_usd)
