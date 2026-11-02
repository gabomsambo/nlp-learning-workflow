"""Import practice drill sheets as paper-less podcast scripts.

A drill sheet is the Markdown format used by the ThetaRay answer drills::

    # Day 5 drill: Graph methods for money laundering

    Intro: Day five, part two. ...

    ## Q1. How would you detect round-tripping?

    **Answer.** Round-tripping is ...

    **Key points.** Cycles in ...

The sheet is turned into the same ``[HOST]:`` script shape every other episode
uses, with ``[PAUSE 5s]`` after each question and ``[PAUSE 8s]`` after each
answer. No model is called. The row is a normal ``podcast_scripts`` row with
``source_material.kind = "drill"``, so audio, read-along and cloud publish work
without a second path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .schemas import PodcastScript, SourceMaterial
from .tts.cue_parser import spoken_line_text

DRILL_SOURCE_KIND = "drill"
TRY_PAUSE_SECONDS = 5
REPEAT_PAUSE_SECONDS = 8
GAP_PAUSE_SECONDS = 1.5

# Spoken forms for the voice only; the sheets keep the written forms. Kept in
# step with data/thetaray-answer-drills/tools/build_drills.py.
_SPOKEN = [
    (r"\bTP\s*=\s*40, FP\s*=\s*60, FN\s*=\s*20, TN\s*=\s*9,880", "forty true positives, sixty false positives, twenty false negatives, and nine thousand eight hundred and eighty true negatives"),
    (r"\bPR-AUC\b", "P R A U C"),
    (r"\bROC-AUC\b", "R O C A U C"),
    (r"\bROC\b", "R O C"),
    (r"\bAUC\b", "A U C"),
    (r"\bPR\b", "P R"),
    (r"\bPSI\b", "P S I"),
    (r"\bKS\b", "K S"),
    (r"\bLOF\b", "L O F"),
    (r"\bOCSVM\b", "one class S V M"),
    (r"\bSVM\b", "S V M"),
    (r"\bSQL\b", "S Q L"),
    (r"\bGNNs\b", "G N Ns"),
    (r"\bGNN\b", "G N N"),
    (r"\bGCN\b", "G C N"),
    (r"\bLightGBM\b", "Light G B M"),
    (r"\bGBM\b", "G B M"),
    (r"\bLLM\b", "L L M"),
    (r"\bAML\b", "A M L"),
    (r"\bML\b", "M L"),
    (r"\bCV\b", "C V"),
    (r"\bPU\b", "P U"),
    (r"\bIBM\b", "I B M"),
    (r"\bMSE\b", "M S E"),
    (r"\bTP\b", "T P"),
    (r"\bFP\b", "F P"),
    (r"\bFN\b", "F N"),
    (r"\bTN\b", "T N"),
    (r"\bF1\b", "F one"),
    (r"\bL1\b", "L one"),
    (r"\bL2\b", "L two"),
    (r"\bID-like\b", "I D like"),
    (r"\bIDs\b", "identifiers"),
    (r"\bID\b", "I D"),
    (r"\bSR 11-7\b", "S R eleven seven"),
    (r"\bW11\b", "W eleven"),
    (r"\bsection 3\.3\b", "section three point three"),
    (r"\bz-score(s?)\b", r"z score\1"),
    (r"\bp-value(s?)\b", r"p value\1"),
    (r"\btop-k\b", "top k"),
    (r"\bk-fold\b", "k fold"),
    (r"\bmax_samples\b", "max samples"),
    (r"\bscore_samples\b", "score samples"),
    (r"\bTreeSHAP\b", "Tree shap"),
    (r"\bSHAP\b", "shap"),
    (r"\bSMOTE\b", "smote"),
    (r"\bXGBoost\b", "X G boost"),
    (r"\bTESS\b", "tess"),
]

_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


class DrillSheetError(ValueError):
    """The sheet is missing a heading, an intro, or part of a question."""


def _decimal_words(match: re.Match) -> str:
    whole, fraction = match.group(1), match.group(2)
    whole_spoken = "zero" if whole == "0" else whole
    return whole_spoken + " point " + " ".join(_DIGIT_WORDS[digit] for digit in fraction)


def spoken_form(text: str) -> str:
    """The words the narrator says for one sheet fragment."""
    out = text
    for pattern, replacement in _SPOKEN:
        out = re.sub(pattern, replacement, out)
    out = out.replace("**", "")
    out = re.sub(r"(?<=\d),(?=\d{3}\b)", "", out)
    out = re.sub(r"\b(\d+)\.(\d+)\b", _decimal_words, out)
    return re.sub(r"\s+", " ", out).strip()


@dataclass
class DrillQuestion:
    number: int
    question: str
    answer: str = ""
    key_points: str = ""


@dataclass
class DrillSheet:
    slug: str
    title: str
    intro: str
    questions: List[DrillQuestion] = field(default_factory=list)


def parse_drill_sheet(text: str, *, slug: str = "drill") -> DrillSheet:
    """Parse the Markdown drill format into a title, intro and questions."""
    lines = text.splitlines()
    title = ""
    intro = ""
    questions: List[DrillQuestion] = []
    current: Optional[DrillQuestion] = None
    mode: Optional[str] = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer, mode
        if current is None or mode is None:
            buffer = []
            return
        joined = " ".join(part.strip() for part in buffer if part.strip())
        if mode == "answer":
            current.answer = (current.answer + " " + joined).strip()
        elif mode == "key":
            current.key_points = (current.key_points + " " + joined).strip()
        buffer = []

    for line in lines:
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.startswith("Intro:"):
            intro = line[len("Intro:"):].strip()
            continue
        heading = re.match(r"^## Q(\d+)\.\s*(.+)$", line)
        if heading:
            flush()
            current = DrillQuestion(int(heading.group(1)), heading.group(2).strip())
            questions.append(current)
            mode = None
            continue
        if line.startswith("**Answer.**"):
            flush()
            mode = "answer"
            buffer.append(line[len("**Answer.**"):])
            continue
        if line.startswith("**Key points.**"):
            flush()
            mode = "key"
            buffer.append(line[len("**Key points.**"):])
            continue
        buffer.append(line)
    flush()

    if not title or not intro or not questions:
        raise DrillSheetError("sheet is missing a title, an Intro: line, or questions")
    for question in questions:
        if not question.answer or not question.key_points:
            raise DrillSheetError(
                f"Q{question.number} is missing an answer or key points"
            )
    return DrillSheet(slug=slug, title=title, intro=intro, questions=questions)


def _question_label(number: int) -> str:
    return _NUMBER_WORDS.get(number, str(number))


def sheet_to_script(sheet: DrillSheet) -> str:
    """Turn a parsed sheet into the stored ``[HOST]:`` script with timed pauses."""
    lines = [
        f"[HOST]: {spoken_form(sheet.intro)}",
        f"[PAUSE {GAP_PAUSE_SECONDS:g}s]",
    ]
    for question in sheet.questions:
        lines.append(
            f"[HOST]: {spoken_form(f'Question {_question_label(question.number)}. {question.question}')}"
        )
        lines.append(f"[PAUSE {TRY_PAUSE_SECONDS}s]")
        lines.append(f"[HOST]: {spoken_form(question.answer)}")
        lines.append(f"[PAUSE {REPEAT_PAUSE_SECONDS}s]")
        lines.append(f"[HOST]: {spoken_form(f'Key points. {question.key_points}')}")
        lines.append(f"[PAUSE {GAP_PAUSE_SECONDS:g}s]")
    lines.append("[HOST]: End of drill.")
    return "\n".join(lines)


def spoken_word_count(script: str) -> int:
    """Words the narrator actually says, matching the Listen page's count."""
    total = 0
    for line in script.splitlines():
        spoken = spoken_line_text(line)
        if spoken:
            total += len(spoken.split())
    return total


def build_drill_row(
    *,
    title: str,
    pillar_id: str,
    script: str,
    paper_id: Optional[str] = None,
    key_points: Optional[Sequence[str]] = None,
) -> PodcastScript:
    """A ``podcast_scripts`` row for imported drill text, without a model call."""
    clean_title = (title or "").strip()
    clean_pillar = (pillar_id or "").strip()
    clean_script = (script or "").strip()
    if not clean_title:
        raise ValueError("title is required")
    if not clean_pillar:
        raise ValueError("pillar_id is required")
    if not clean_script:
        raise ValueError("script text is required")
    return PodcastScript(
        paper_id=(paper_id or "").strip() or None,
        pillar_id=clean_pillar,
        title=clean_title,
        script=clean_script,
        word_count=spoken_word_count(clean_script),
        key_points=[point for point in (key_points or []) if point],
        source_material=SourceMaterial(kind=DRILL_SOURCE_KIND),
    )


def drill_row_from_sheet(
    sheet: DrillSheet,
    *,
    pillar_id: str,
    title: Optional[str] = None,
) -> PodcastScript:
    """Build the importable row for one parsed drill sheet."""
    return build_drill_row(
        title=title or sheet.title,
        pillar_id=pillar_id,
        script=sheet_to_script(sheet),
        key_points=[question.question for question in sheet.questions],
    )
