"""Deterministic number lint for a podcast script.

Every quantity a host line states should occur in the paper. A number the
paper does not contain was either derived by the writer ("six of eight",
"eighty thousand times") or came from somewhere else, and those are the lines
worth a second look. No model call, no network, no database: the inputs are
the script text and the paper text, and the output is a list of flags, each
with a reason a person can read.

This is a triage signal, not a gate. It cannot tell a wrong derived number
from a right one, only that the paper never states it, so it warns and never
blocks audio. It also cannot see a *true* number bound to the wrong thing;
that is the fact-check engine's job.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

WARNING_PREFIX = "Number check"
# One warning line per flagged script line, and no more than this many, so a
# badly off script cannot bury the other warnings.
MAX_WARNINGS = 10

KIND_UNMATCHED = "unmatched"
KIND_COUNT = "count"
KIND_RATIO = "ratio"

_ONES = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_TENS = {
    w: 10 * i
    for i, w in enumerate(
        "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
    )
    if w != "_"
}
_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 10**12,
}
_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
# Only upper-case K/M/B are scale suffixes: "51m" is minutes, "7B" is a model.
_DIGITS = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:\s?(%|percent|[KMB](?![a-zA-Z])))?"
)
_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9}
_A_SCALE = re.compile(r"\b(half a|a)\s+(hundred|thousand|million|billion|trillion)\b", re.I)
_MARKDOWN = re.compile(r"[*_`]")
# Words that hold a number in a script but are not a quantity.
_YEAR_HEAD = {"nineteen": 1900, "twenty": 2000}
_YEAR_TAIL_WORDS = {w for w, v in _ONES.items() if 10 <= v <= 19} | set(_TENS)

# "six of eight", "six out of the eight": two numbers joined by of / out of.
_COUNT_JOIN = re.compile(r"^\s+(?:out of|of)\s+(?:the\s+)?$", re.I)
_TIMES_TAIL = re.compile(r"^\s*times\b", re.I)

# "175 M - 180 M" / "one seventy-five to one eighty million": the scale word
# is written once, on the last number, and belongs to both.
_RANGE_JOIN = re.compile(r"^\s*(?:-|–|−|—|to|or|and)\s*\$?\s*$", re.I)

# A count over a list this long is a fraction of a population, not a recount.
_MAX_COUNTABLE = 100


@dataclass(frozen=True)
class Span:
    value: float
    raw: str
    start: int
    end: int


@dataclass(frozen=True)
class NumberFlag:
    line: int  # 1-based, over the whole script
    raw: str
    kind: str
    reason: str


@dataclass
class NumberLintResult:
    host_lines: int = 0
    numbers_checked: int = 0
    flags: list[NumberFlag] = field(default_factory=list)

    @property
    def flagged_lines(self) -> int:
        return len({f.line for f in self.flags})

    @property
    def per_ten_host_lines(self) -> float:
        if not self.host_lines:
            return 0.0
        return 10 * self.flagged_lines / self.host_lines


def _clean(text: str) -> str:
    """Drop markdown emphasis so ``175 _M_`` reads as ``175 M``."""
    return _MARKDOWN.sub("", text)


def _digit_spans(text: str) -> list[Span]:
    out: list[Span] = []
    for m in _DIGITS.finditer(text):
        value = float((m.group(1) + (m.group(2) or "")).replace(",", ""))
        suffix = m.group(3)
        end = m.end()
        raw = m.group(0).strip()
        if suffix in _SUFFIX:
            value *= _SUFFIX[suffix]
        elif not suffix:
            word = _WORD.match(text, end + 1) if text[end:end + 1] == " " else None
            if word and word.group(0).lower() in _SCALES:
                value *= _SCALES[word.group(0).lower()]
                end = word.end()
                raw = text[m.start():end]
        out.append(Span(value, raw, m.start(), end))
    return out


def _all_number_words(parts: list[str]) -> bool:
    return all(p in _ONES or p in _TENS or p in _SCALES for p in parts)


def _word_spans(text: str) -> list[Span]:
    """Spoken numbers: 'eighty thousand', 'twenty-four', 'two point five'."""
    toks = [(m.group(0).lower(), m.start(), m.end()) for m in _WORD.finditer(text)]
    out: list[Span] = []
    i = 0
    while i < len(toks):
        word, start, _ = toks[i]
        if not _all_number_words(word.split("-")):
            i += 1
            continue
        year = _spoken_year(toks, i)
        if year:
            out.append(year[0])
            i = year[1]
            continue
        total = cur = 0
        j = i
        used = False
        while j < len(toks):
            parts = toks[j][0].split("-")
            if parts == ["and"] and not (j > i and toks[j - 1][0] == "hundred"):
                break
            if not all(_is_part(p) for p in parts):
                break
            for p in parts:
                if p in _ONES:
                    cur += _ONES[p]
                    used = True
                elif p in _TENS:
                    cur += _TENS[p]
                    used = True
                elif p == "hundred":
                    cur = max(cur, 1) * 100
                elif p in _SCALES:
                    total += max(cur, 1) * _SCALES[p]
                    cur = 0
                    used = True
            j += 1
        has_digit_word = any(
            p in _ONES or p in _TENS for t in toks[i:j] for p in t[0].split("-")
        )
        if used and has_digit_word:
            value = float(total + cur)
            if j + 1 < len(toks) and toks[j][0] == "point" and toks[j + 1][0] in _ONES:
                k, frac = j + 1, ""
                while k < len(toks) and toks[k][0] in _ONES and _ONES[toks[k][0]] < 10:
                    frac += str(_ONES[toks[k][0]])
                    k += 1
                if frac:
                    value = float(f"{int(value)}.{frac}")
                    j = k
            out.append(Span(value, " ".join(t[0] for t in toks[i:j]), start, toks[j - 1][2]))
        i = max(j, i + 1)
    return out


def _is_part(p: str) -> bool:
    return p == "and" or p in _ONES or p in _TENS or p in _SCALES


def _spoken_year(toks, i):
    """'twenty twenty-four' is 2024, not 44. Returns (Span, next index) or None."""
    head = toks[i][0]
    if head not in _YEAR_HEAD or i + 1 >= len(toks):
        return None
    tail = toks[i + 1][0].split("-")
    if tail[0] in _SCALES or tail[0] not in _YEAR_TAIL_WORDS:
        return None
    value = _YEAR_HEAD[head] + sum(_ONES.get(p, _TENS.get(p, 0)) for p in tail)
    return Span(float(value), f"{head} {toks[i + 1][0]}", toks[i][1], toks[i + 1][2]), i + 2


def _a_scale_spans(text: str) -> list[Span]:
    """'a million', 'half a million'. 'a hundred' is a number; 'a' alone is not."""
    out = []
    for m in _A_SCALE.finditer(text):
        scale = _SCALES[m.group(2).lower()]
        half = m.group(1).lower().startswith("half")
        out.append(Span(scale / 2 if half else float(scale), m.group(0), m.start(), m.end()))
    return out


def number_spans(text: str) -> list[Span]:
    """Every quantity in ``text``, in order, with the span it came from."""
    text = _clean(text)
    spans = _digit_spans(text) + _word_spans(text) + _a_scale_spans(text)
    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    kept: list[Span] = []
    for s in spans:
        if kept and s.start < kept[-1].end:
            continue  # inside a longer span already taken
        kept.append(s)
    return _share_range_scale(text, kept)


def _scale_word(raw: str) -> float | None:
    tail = re.split(r"[\s-]+", raw.strip().lower())[-1]
    return _SCALES[tail] if tail in _SCALES and tail != "hundred" else None


def _share_range_scale(text: str, spans: list[Span]) -> list[Span]:
    out = list(spans)
    for i in range(len(out) - 2, -1, -1):
        a, b = out[i], out[i + 1]
        scale = _scale_word(b.raw)
        if (
            scale
            and _scale_word(a.raw) is None
            and a.value < scale
            and _RANGE_JOIN.match(text[a.end:b.start])
        ):
            out[i] = Span(a.value * scale, a.raw, a.start, a.end)
    return out


def _decimals(v: float) -> int:
    s = repr(float(v))
    return len(s.split(".")[1].rstrip("0")) if "." in s else 0


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def supported(v: float, paper: set[float]) -> bool:
    """``v`` is a paper number, a rounding of one, or a 2-significant-figure one."""
    if v in paper or any(_close(v, p) for p in paper):
        return True
    d = _decimals(v)
    half = 0.5 * 10 ** (-d) + 1e-9
    for p in paper:
        if abs(p - v) <= half and (d > 0 or abs(p) >= 10):
            return True  # "about 62" for 62.4
        if v >= 1000 and p >= 1000:  # "about 87 thousand" for 87,109
            q = 10 ** (int(math.floor(math.log10(p))) - 1)
            if round(p / q) * q == v:
                return True
    return False


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:g}"


def _is_year(s: Span) -> bool:
    return float(s.value).is_integer() and 1900 <= s.value <= 2099 and "," not in s.raw


def _count_pairs(text: str, spans: list[Span]) -> list[tuple[Span, Span]]:
    pairs = []
    for a, b in zip(spans, spans[1:]):
        if _COUNT_JOIN.match(text[a.end:b.start]):
            pairs.append((a, b))
    return pairs


def _is_count(a: Span, b: Span) -> bool:
    return (
        float(a.value).is_integer()
        and float(b.value).is_integer()
        and 2 <= a.value <= b.value <= _MAX_COUNTABLE
    )


def lint_script(script: str, paper_text: str) -> NumberLintResult:
    """Check every ``[HOST]`` line of ``script`` against ``paper_text``."""
    paper = _clean(paper_text)
    paper_spans = number_spans(paper)
    paper_values = {s.value for s in paper_spans}
    paper_counts = {
        (a.value, b.value) for a, b in _count_pairs(_clean(paper_text), paper_spans)
    }
    result = NumberLintResult()
    for ln, raw_line in enumerate(script.splitlines(), start=1):
        if not raw_line.startswith("[HOST"):
            continue
        result.host_lines += 1
        line = _clean(raw_line)
        spans = number_spans(line)
        for s in spans:
            result.numbers_checked += 1
            if float(s.value).is_integer() and s.value <= 12:
                continue  # structural counts: "three stages"
            if _is_year(s):
                continue  # dates are the fact-check engine's, and rarely derived
            if supported(s.value, paper_values):
                continue
            result.flags.append(NumberFlag(
                ln, s.raw, KIND_UNMATCHED,
                f'"{s.raw}" ({_fmt(s.value)}) is not a number the paper states, '
                "even rounded. It may be derived, converted or from outside the paper.",
            ))
        for a, b in _count_pairs(line, spans):
            if _is_count(a, b) and (a.value, b.value) not in paper_counts:
                phrase = line[a.start:b.end]
                result.flags.append(NumberFlag(
                    ln, phrase, KIND_COUNT,
                    f'"{phrase}" is a count the paper never states. '
                    "Recount it against the paper's table or list.",
                ))
        for s in spans:
            if (
                s.value >= 2
                and _TIMES_TAIL.match(line[s.end:])
                and not supported(s.value, paper_values)
                and not any(f.line == ln and f.raw == s.raw and f.kind == KIND_UNMATCHED
                            for f in result.flags)
            ):
                result.flags.append(NumberFlag(
                    ln, f"{s.raw} times", KIND_RATIO,
                    f'"{s.raw} times" is a ratio the paper never states. '
                    "Recompute it from the numbers it divides.",
                ))
    return result


def warnings_for(result: NumberLintResult) -> list[str]:
    """One human-readable warning per flagged line, for ``source_material.warnings``."""
    by_line: dict[int, list[NumberFlag]] = {}
    for f in result.flags:
        by_line.setdefault(f.line, []).append(f)
    out = []
    for ln in sorted(by_line):
        flags = by_line[ln]
        quoted = "; ".join(f'"{f.raw}"' for f in flags)
        out.append(f"{WARNING_PREFIX}, script line {ln}: {quoted}. {flags[0].reason}")
    if len(out) > MAX_WARNINGS:
        extra = len(out) - MAX_WARNINGS
        out = out[:MAX_WARNINGS] + [
            f"{WARNING_PREFIX}: {extra} more flagged line(s) not listed."
        ]
    return out


def merge_warnings(existing: list[str], new: list[str]) -> list[str]:
    """Replace an earlier lint's warnings and keep everything else."""
    kept = [w for w in existing if not w.startswith(WARNING_PREFIX)]
    return kept + new
