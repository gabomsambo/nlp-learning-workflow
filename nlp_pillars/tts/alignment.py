"""Time every word of an episode script against the audio that was made from it.

The script is known exactly, so this is alignment and not transcription: a
speech recogniser listens to the finished audio and says when each word it
heard began and ended, and those words are matched back to the script's own
words. The recogniser is allowed to be wrong. It hears "three point five" where
the script says ``3.5``, spells "LightGBM" as two words, and drops a word now and
then; the script stays the source of truth for the text and the recogniser only
supplies clock times.

Three levels degrade in order, so a weak stretch never loses its place:

* a word the recogniser heard is timed by the recogniser (``matched``);
* a word it did not hear is timed from the characters it shares with the words
  around it, or spread across the gap by length (``matched`` is false);
* every spoken line therefore has a start and an end even when no word in it
  matched, which is all line-level sync needs.

``nlp_pillars/tts/aligner.py`` owns the recogniser. This module is pure so the
matching can be tested without a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

from nlp_pillars.tts.cue_parser import spoken_line_text

TIMELINE_VERSION = 1

# Below this share of matched words the audio and the script are probably not
# the same episode, and a timeline would be a confident lie.
MIN_MATCHED_RATIO = 0.5
# Below this the per-word times are not worth highlighting one word at a time;
# the timeline keeps its line times and says it is line-level only.
WORD_LEVEL_MATCHED_RATIO = 0.8

# A character-level gap fill is quadratic in the gap. Past this many cells the
# gap is spread by length instead; real gaps are a handful of words.
_MAX_GAP_CELLS = 4_000_000
# Shared runs shorter than this are coincidence ("e", "th"), not evidence.
_MIN_CHAR_RUN = 2
# A word counts as heard when this share of its characters sit in a shared run.
_GAP_MATCH_SHARE = 0.6


class AlignmentError(RuntimeError):
    """The audio and the script do not line up well enough to time."""


@dataclass(frozen=True)
class ScriptWord:
    """One whitespace-separated word of a spoken script line."""

    line: int
    index: int
    text: str
    units: Tuple[str, ...]


@dataclass(frozen=True)
class AsrWord:
    """One word the recogniser heard, with its clock times in seconds."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class WordTiming:
    start_ms: int
    end_ms: int
    matched: bool


# ---------------------------------------------------------------------------
# Spoken form of a token
# ---------------------------------------------------------------------------

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()

_TOKEN_PARTS = re.compile(r"\d[\d,]*(?:\.\d+)?|[^\W\d_]+(?:'[^\W\d_]+)*|[%&$+=]", re.UNICODE)
_SYMBOLS = {"%": "percent", "&": "and", "$": "dollars", "+": "plus", "=": "equals"}


def _below_hundred(n: int) -> List[str]:
    if n < 20:
        return [_ONES[n]]
    tens, ones = divmod(n, 10)
    return [_TENS[tens]] + ([_ONES[ones]] if ones else [])


def _below_thousand(n: int) -> List[str]:
    hundreds, rest = divmod(n, 100)
    out: List[str] = []
    if hundreds:
        out += [_ONES[hundreds], "hundred"]
    if rest or not hundreds:
        out += _below_hundred(rest)
    return out


def cardinal_words(n: int) -> List[str]:
    """``1234`` -> one thousand two hundred thirty four. Up to the billions."""
    if n < 1000:
        return _below_thousand(n)
    out: List[str] = []
    for size, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
        chunk, n = divmod(n, size)
        if chunk:
            out += _below_thousand(chunk) + [name]
    if n:
        out += _below_thousand(n)
    return out


def year_words(n: int) -> List[str]:
    """A four-digit year the way it is said: 2024 -> twenty twenty four."""
    if n % 1000 == 0 or 2000 <= n <= 2009:
        return cardinal_words(n)
    head, tail = divmod(n, 100)
    if tail == 0:
        return _below_hundred(head) + ["hundred"]
    if tail < 10:
        return _below_hundred(head) + ["oh", _ONES[tail]]
    return _below_hundred(head) + _below_hundred(tail)


def number_words(text: str) -> List[str]:
    """Spoken words for a digit run such as ``3.5``, ``1,000`` or ``2024``."""
    cleaned = text.replace(",", "")
    whole, _, fraction = cleaned.partition(".")
    if len(whole) > 12:
        # Too long to be a quantity (an id, a hash): said digit by digit.
        digits = list(whole)
        words = [_ONES[int(d)] for d in digits]
    elif len(whole) == 4 and not fraction and 1000 <= int(whole) <= 2099 and "," not in text:
        words = year_words(int(whole))
    else:
        words = cardinal_words(int(whole))
    if fraction:
        words = words + ["point"] + [_ONES[int(d)] for d in fraction]
    return words


def spoken_units(token: str) -> Tuple[str, ...]:
    """The lower-case words a token is read as: ``GPT-4`` -> (gpt, four).

    Punctuation is not spoken, an apostrophe disappears (``that's`` -> thats),
    and a hyphen or slash separates two words. Digits and a few symbols are
    spelled out the way a narrator says them. The recogniser's words go through
    the same function, so both sides agree whenever they heard the same thing.
    """
    units: List[str] = []
    text = token.replace("’", "'").lower()
    parts = _TOKEN_PARTS.findall(text)
    for position, part in enumerate(parts):
        if part[0].isdigit():
            number = number_words(part)
            # "$5" is "five dollars": the symbol follows the number it prefixes.
            if position and parts[position - 1] == "$":
                units.extend(number + ["dollars"])
                continue
            units.extend(number)
        elif part in _SYMBOLS:
            if part == "$" and position + 1 < len(parts) and parts[position + 1][0].isdigit():
                continue
            units.append(_SYMBOLS[part])
        else:
            units.append(part.replace("'", ""))
    return tuple(unit for unit in units if unit)


# ---------------------------------------------------------------------------
# The script's words
# ---------------------------------------------------------------------------


def script_words(script: str) -> List[ScriptWord]:
    """Every spoken word of a script, numbered by line and position in the line.

    Lines are ``script.splitlines()`` — the numbering the fact check and the
    Listen page use — and the words are the whitespace-separated pieces of
    what the narrator says for that line, which the page splits the same way.
    """
    out: List[ScriptWord] = []
    for line_no, raw in enumerate(script.splitlines()):
        spoken = spoken_line_text(raw)
        if not spoken:
            continue
        for index, token in enumerate(spoken.split()):
            out.append(ScriptWord(line_no, index, token, spoken_units(token)))
    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass
class _Unit:
    text: str
    start: float = 0.0
    end: float = 0.0
    matched: bool = False


def _asr_units(asr: Sequence[AsrWord]) -> List[_Unit]:
    """One unit per spoken word; a word that expands splits its time by length."""
    units: List[_Unit] = []
    for word in asr:
        pieces = spoken_units(word.text)
        if not pieces:
            continue
        total = sum(len(p) for p in pieces) or 1
        cursor = float(word.start)
        span = max(float(word.end) - float(word.start), 0.0)
        for piece in pieces:
            width = span * len(piece) / total
            units.append(_Unit(piece, cursor, cursor + width, True))
            cursor += width
    return units


def _spread_by_length(units: List[_Unit], lo: float, hi: float) -> None:
    """Give units that nothing matched a share of [lo, hi] by their length."""
    if not units:
        return
    hi = max(hi, lo)
    total = sum(len(u.text) for u in units) or 1
    cursor = lo
    for unit in units:
        width = (hi - lo) * len(unit.text) / total
        unit.start, unit.end, unit.matched = cursor, cursor + width, False
        cursor += width


def _char_runs(a: str, b: str) -> List[Tuple[int, int, int]]:
    blocks = SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
    return [(i, j, n) for i, j, n in blocks if n >= _MIN_CHAR_RUN]


def _piecewise(points: List[Tuple[int, int]], x: float) -> float:
    """Map x through monotone (from, to) anchor points, straight between them."""
    if x <= points[0][0]:
        return float(points[0][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            if x1 == x0:
                return float(y1)
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return float(points[-1][1])


def _fill_gap(
    script_units: List[_Unit],
    heard: List[_Unit],
    lo: float,
    hi: float,
) -> None:
    """Time script units the word-level match left over.

    ``heard`` are the recogniser's units that fall in the same gap. The two
    sides are laid out as one string of letters each and matched at the
    character level, which copes with a word heard as two ("LightGBM"),
    two heard as one, and a spelling that differs by a letter.
    """
    if not script_units:
        return
    if not heard:
        _spread_by_length(script_units, lo, hi)
        return
    s_text = "".join(u.text for u in script_units)
    a_text = "".join(u.text for u in heard)
    if len(s_text) * len(a_text) > _MAX_GAP_CELLS:
        _spread_by_length(script_units, heard[0].start, heard[-1].end)
        return

    runs = _char_runs(s_text, a_text)
    points: List[Tuple[int, int]] = [(0, 0)]
    covered = [False] * len(s_text)
    for i, j, n in runs:
        points.append((i, j))
        points.append((i + n, j + n))
        for k in range(i, i + n):
            covered[k] = True
    points.append((len(s_text), len(a_text)))
    # Anchors can touch but never cross; sorting guards a degenerate tie.
    points = sorted(set(points))

    a_offsets: List[Tuple[int, int, _Unit]] = []
    cursor = 0
    for unit in heard:
        a_offsets.append((cursor, cursor + len(unit.text), unit))
        cursor += len(unit.text)

    def time_at(position: float) -> float:
        for c0, c1, unit in a_offsets:
            if position <= c1:
                if c1 == c0:
                    return unit.start
                share = min(max((position - c0) / (c1 - c0), 0.0), 1.0)
                return unit.start + (unit.end - unit.start) * share
        return heard[-1].end

    cursor = 0
    for unit in script_units:
        c0, c1 = cursor, cursor + len(unit.text)
        cursor = c1
        unit.start = time_at(_piecewise(points, c0))
        unit.end = time_at(_piecewise(points, c1))
        share = sum(covered[c0:c1]) / max(c1 - c0, 1)
        unit.matched = share >= _GAP_MATCH_SHARE


def align_words(
    words: Sequence[ScriptWord],
    asr: Sequence[AsrWord],
    duration_s: float,
) -> List[WordTiming]:
    """Clock times for every script word, in script order.

    Always returns one timing per word, monotone and inside the audio. How much
    of it was actually heard is in each timing's ``matched`` flag.
    """
    script_units: List[_Unit] = []
    owner: List[int] = []
    for position, word in enumerate(words):
        for text in word.units:
            script_units.append(_Unit(text))
            owner.append(position)
    heard = _asr_units(asr)
    duration = max(float(duration_s), 0.0)

    blocks = SequenceMatcher(
        None,
        [u.text for u in script_units],
        [u.text for u in heard],
        autojunk=False,
    ).get_matching_blocks()

    s_cursor = 0
    h_cursor = 0
    prev_end = 0.0
    for i, j, n in blocks:
        gap_s = script_units[s_cursor:i]
        gap_h = heard[h_cursor:j]
        next_start = heard[j].start if n else duration
        _fill_gap(gap_s, gap_h, prev_end, next_start)
        for k in range(n):
            target, source = script_units[i + k], heard[j + k]
            target.start, target.end, target.matched = source.start, source.end, True
        s_cursor, h_cursor = i + n, j + n
        if n:
            prev_end = heard[j + n - 1].end
        elif script_units[:i]:
            prev_end = script_units[i - 1].end

    # Group units back into the words they came from.
    starts: List[Optional[float]] = [None] * len(words)
    ends: List[float] = [0.0] * len(words)
    matched: List[bool] = [True] * len(words)
    for unit, position in zip(script_units, owner):
        if starts[position] is None:
            starts[position] = unit.start
        ends[position] = unit.end
        matched[position] = matched[position] and unit.matched

    timings: List[WordTiming] = []
    previous_start = 0.0
    previous_end = 0.0
    raw: List[Tuple[float, float, bool]] = []
    for position, word in enumerate(words):
        start = starts[position]
        if start is None:
            # A word with nothing to say ("—", "...") has no sound to find. It sits
            # where the word before it ended and does not count as unheard.
            start = previous_end
            end = start
            was_matched = True
        else:
            end = ends[position]
            was_matched = matched[position]
        start = min(max(start, previous_start), duration)
        end = min(max(end, start), duration)
        raw.append((start, end, was_matched))
        previous_start, previous_end = start, end
    # A word never ends after the next one starts.
    for position, (start, end, was_matched) in enumerate(raw):
        if position + 1 < len(raw):
            end = min(end, max(raw[position + 1][0], start))
        timings.append(WordTiming(round(start * 1000), round(end * 1000), was_matched))
    return timings


# ---------------------------------------------------------------------------
# The stored timeline
# ---------------------------------------------------------------------------


def build_timeline(
    script: str,
    asr: Sequence[AsrWord],
    duration_s: float,
    *,
    engine: str,
    script_sha256: str,
    aligned_at: Optional[datetime] = None,
) -> Dict[str, object]:
    """The ``audio_metadata.timeline`` payload for one episode.

    ``lines`` holds one entry per *spoken* line: its start and end in
    milliseconds and, when word-level timing is trustworthy, ``words`` as
    ``[start_ms, end_ms, heard]`` triples in the line's word order. Raises
    ``AlignmentError`` when too little of the script was heard to time at all.
    """
    words = script_words(script)
    if not words:
        raise AlignmentError("The script has no spoken words to time.")
    if not asr:
        raise AlignmentError("The recogniser heard no speech in the audio.")
    timings = align_words(words, asr, duration_s)
    heard = sum(1 for t in timings if t.matched)
    ratio = heard / len(timings)
    if ratio < MIN_MATCHED_RATIO:
        raise AlignmentError(
            f"Only {ratio:.0%} of the script's words were found in the audio, "
            "so it does not look like the same episode."
        )
    word_level = ratio >= WORD_LEVEL_MATCHED_RATIO

    lines: List[Dict[str, object]] = []
    current: Optional[int] = None
    group: List[WordTiming] = []

    def flush() -> None:
        if current is None or not group:
            return
        entry: Dict[str, object] = {
            "line": current,
            "start_ms": group[0].start_ms,
            "end_ms": max(t.end_ms for t in group),
        }
        if word_level:
            entry["words"] = [[t.start_ms, t.end_ms, 1 if t.matched else 0] for t in group]
        lines.append(entry)

    for word, timing in zip(words, timings):
        if word.line != current:
            flush()
            current, group = word.line, []
        group.append(timing)
    flush()

    stamp = aligned_at or datetime.now(timezone.utc)
    return {
        "version": TIMELINE_VERSION,
        "engine": engine,
        "script_sha256": script_sha256,
        "aligned_at": stamp.isoformat(),
        "duration_ms": round(float(duration_s) * 1000),
        "word_count": len(words),
        "matched_ratio": round(ratio, 4),
        "word_level": word_level,
        "lines": lines,
    }
