"""Which script lines must yield a claim, and what each extraction window did.

A host line that states a non-trivial number is a line the fact-check must not
silently skip. The extractor is a model, so it sometimes returns nothing for a
whole window; this module is the deterministic side that notices.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HOST_LINE = re.compile(r"^\s*\[HOST")

# A digit run that is a bare integer of 12 or less is structural ("three
# stages", "Table 9"). Anything with a decimal, a percent sign, a scale word
# or suffix, a thousands comma, or a value over 12 is a figure worth checking.
_DIGITS = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(\s?(?:%|percent\b|[KMB]\b|"
    r"(?:thousand|million|billion|trillion)\b))?",
)
_SPOKEN = re.compile(
    r"\b(?:thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"million|billion|trillion)\b",
    re.IGNORECASE,
)

# "Question fourteen", "Step 3": a label for the script's own structure, not
# a figure about the paper.
_LABEL = re.compile(
    r"\b(?:question|step|part|round|chapter|section|episode|number|item)\s+"
    r"(?:\d+|[a-z]+(?:-[a-z]+)?)\b",
    re.IGNORECASE,
)

# A window that came back empty is only worth a second call when it held a
# host line with some substance. Greetings and cue-only stretches are empty
# for a good reason.
SUBSTANTIVE_WORDS = 12


def is_host_line(line: str) -> bool:
    return bool(HOST_LINE.match(line))


def has_nontrivial_number(line: str) -> bool:
    line = _LABEL.sub(" ", line)
    for match in _DIGITS.finditer(line):
        if match.group(2) or match.group(3) or "," in match.group(1):
            return True
        if int(match.group(1)) > 12:
            return True
    return bool(_SPOKEN.search(line))


def number_bearing_lines(lines: list[str], start: int, end: int) -> list[int]:
    return [
        index
        for index in range(start, min(end, len(lines)))
        if is_host_line(lines[index]) and has_nontrivial_number(lines[index])
    ]


def substantive_host_lines(lines: list[str], start: int, end: int) -> list[int]:
    return [
        index
        for index in range(start, min(end, len(lines)))
        if is_host_line(lines[index])
        and len(lines[index].split()) >= SUBSTANTIVE_WORDS
    ]


@dataclass
class WindowStat:
    """What one extraction window produced, before and after any retry."""

    start: int
    end: int
    first_pass: int = 0
    claims: int = 0
    dropped: int = 0
    retried: bool = False
    uncovered: list[int] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"L{self.start}-{self.end - 1}"

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "first_pass": self.first_pass,
            "claims": self.claims,
            "dropped": self.dropped,
            "retried": self.retried,
            "uncovered": list(self.uncovered),
        }


def describe(windows: list[WindowStat], claims: int, dropped: int) -> str:
    """One stage-detail line: totals, then a count per window.

    A retried window reads ``L120-149 0>43``: nothing the first time, 43 after.
    Number-bearing lines still without a claim are named at the end.
    """
    text = f"{claims} claims"
    if dropped:
        text += f", {dropped} spans dropped"
    parts = []
    for window in windows:
        count = str(window.claims)
        if window.retried:
            count = f"{window.first_pass}>{window.claims}"
        parts.append(f"{window.label} {count}")
    text += "; windows " + ", ".join(parts)
    uncovered = [line for window in windows for line in window.uncovered]
    if uncovered:
        text += "; numbers with no claim on " + ", ".join(f"L{n}" for n in uncovered)
    return text
