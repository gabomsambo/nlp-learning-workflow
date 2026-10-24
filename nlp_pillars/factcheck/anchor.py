"""Locate a claim's span in its script line, and a quote in the paper.

Spans that are not in the line are re-anchored when the window contains them
exactly once, and dropped otherwise. A quote is shown only when it is found
in ``paper_texts``: normalised exact match, or a fuzzy match at 90 or better
that is then snapped to the paper's own characters. An unlocated quote is
omitted. It does not change the verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

FUZZY_MIN_SCORE = 90.0
_FUZZY_MIN_CHARS = 24
_FOLD = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
})
_TOKEN = re.compile(r"[a-z0-9]{5,}")


@dataclass(frozen=True)
class SpanHit:
    line_index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class QuoteHit:
    quote: str
    char_start: int
    char_end: int
    match_score: float
    chunk_id: str | None
    chunk_index: int | None


def _fold(text: str) -> str:
    # One character in, one character out, so span indexes stay on the original.
    return text.translate(_FOLD)


def _normalize(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Lowercase and collapse whitespace.

    Each normalised character maps to an original ``[start, end)`` span so a
    match can be snapped back onto the paper or the script line.
    """
    folded = _fold(text)
    out: list[str] = []
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(folded)
    started = False
    while i < n:
        if folded[i].isspace():
            j = i + 1
            while j < n and folded[j].isspace():
                j += 1
            if started and j < n:
                out.append(" ")
                spans.append((i, j))
            i = j
            continue
        lower = folded[i].lower()
        out.append(lower)
        spans.append((i, i + 1))
        started = True
        i += 1
    return "".join(out), spans


def _snap(text: str, spans: list[tuple[int, int]], start: int, length: int) -> tuple[int, int, str]:
    orig_start = spans[start][0]
    orig_end = spans[start + length - 1][1]
    return orig_start, orig_end, text[orig_start:orig_end]


def find_in_text(text: str, snippet: str) -> tuple[int, int, str, float] | None:
    """Return ``(start, end, snapped, score)`` or None.

    Score is 100 for a raw or normalised exact match. Fuzzy search runs only
    for snippets long enough that a 90 score is not a short accidental overlap,
    and the returned text is the paper's, not the model's.
    """
    snippet = snippet.strip()
    if not snippet or not text:
        return None
    raw = text.find(snippet)
    if raw >= 0:
        end = raw + len(snippet)
        return raw, end, text[raw:end], 100.0

    needle, _needle_spans = _normalize(snippet)
    hay, hay_spans = _normalize(text)
    if not needle or not hay:
        return None
    pos = hay.find(needle)
    if pos >= 0:
        start, end, snapped = _snap(text, hay_spans, pos, len(needle))
        return start, end, snapped, 100.0
    if len(needle) < _FUZZY_MIN_CHARS:
        return None
    return _fuzzy(text, hay, hay_spans, needle)


def _fuzzy(
    text: str,
    hay: str,
    hay_spans: list[tuple[int, int]],
    needle: str,
) -> tuple[int, int, str, float] | None:
    tokens = _TOKEN.findall(needle)
    if not tokens:
        return None
    ranked = sorted(tokens, key=lambda token: (hay.count(token), -len(token)))
    seed = next((token for token in ranked if hay.count(token) > 0), None)
    if seed is None:
        return None
    occurrences: list[int] = []
    cursor = 0
    while len(occurrences) < 30:
        found = hay.find(seed, cursor)
        if found < 0:
            break
        occurrences.append(found)
        cursor = found + 1
    seed_at = needle.find(seed)
    best: tuple[float, int, int] | None = None
    for origin in occurrences:
        window_start = origin - seed_at
        for delta in range(-2, 3):
            length = len(needle) + delta
            if length < _FUZZY_MIN_CHARS:
                continue
            start = window_start
            end = start + length
            if start < 0:
                end -= start
                start = 0
            if end > len(hay):
                end = len(hay)
            if end - start < _FUZZY_MIN_CHARS:
                continue
            window = hay[start:end]
            score = SequenceMatcher(None, needle, window, autojunk=False).ratio() * 100
            if score < FUZZY_MIN_SCORE:
                continue
            if best is None or score > best[0] or (score == best[0] and start < best[1]):
                best = (score, start, end - start)
    if best is None:
        return None
    score, start, length = best
    orig_start, orig_end, snapped = _snap(text, hay_spans, start, length)
    return orig_start, orig_end, snapped, round(score, 1)


def find_in_line(line: str, span: str) -> tuple[int, int, str] | None:
    found = find_in_text(line, span)
    if found is None:
        return None
    # A fuzzy snap against a single line can attach a span to the wrong words.
    # Spans are machine-checked verbatim, so only exact matches count.
    if found[3] < 100:
        return None
    return found[0], found[1], found[2]


def place_span(
    lines: list[str],
    line: int,
    span: str,
    window_start: int,
    window_end: int,
) -> SpanHit | None:
    """Place ``span`` on ``line``, or on the one window line that contains it."""
    if window_start <= line < window_end and 0 <= line < len(lines):
        hit = find_in_line(lines[line], span)
        if hit is not None:
            start, end, text = hit
            return SpanHit(line, start, end, text)
    hits: list[SpanHit] = []
    start_i = max(0, window_start)
    end_i = min(len(lines), window_end)
    for index in range(start_i, end_i):
        hit = find_in_line(lines[index], span)
        if hit is None:
            continue
        start, end, text = hit
        hits.append(SpanHit(index, start, end, text))
    if len(hits) == 1:
        return hits[0]
    return None


def choose_chunk_rows(rows: list[dict[str, Any]], current_version: str) -> list[dict[str, Any]]:
    """Keep one chunker version: the current one, else the newest."""
    if not rows:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("chunker_version") or ""), []).append(row)
    if current_version in grouped:
        chosen = grouped[current_version]
    else:
        def rank(item: tuple[str, list[dict[str, Any]]]) -> tuple[str, int]:
            _version, group = item
            newest = max(str(row.get("created_at") or "") for row in group)
            return newest, len(group)

        chosen = max(grouped.items(), key=rank)[1]
    chosen = sorted(chosen, key=lambda row: (int(row["chunk_index"]), str(row["id"])))
    return [
        {
            "id": str(row["id"]),
            "chunk_index": int(row["chunk_index"]),
            "text": row.get("text") or "",
        }
        for row in chosen
    ]


def _chunk_for_quote(quote: str, chunks: list[dict[str, Any]]) -> tuple[str | None, int | None]:
    if not quote or not chunks:
        return None, None
    needle, _spans = _normalize(quote)
    if not needle:
        return None, None
    containing = []
    for chunk in chunks:
        hay, _hay_spans = _normalize(chunk.get("text") or "")
        if needle in hay:
            containing.append(chunk)
    if containing:
        containing.sort(key=lambda chunk: int(chunk["chunk_index"]))
        winner = containing[0]
        return str(winner["id"]), int(winner["chunk_index"])
    best: tuple[int, int, dict[str, Any]] | None = None
    for chunk in chunks:
        hay, _hay_spans = _normalize(chunk.get("text") or "")
        if not hay:
            continue
        match = SequenceMatcher(None, needle, hay, autojunk=False).find_longest_match(
            0, len(needle), 0, len(hay)
        )
        size = match.size
        if size <= 0:
            continue
        index = int(chunk["chunk_index"])
        if best is None or size > best[0] or (size == best[0] and index < best[1]):
            best = (size, index, chunk)
    if best is None:
        return None, None
    size, _index, chunk = best
    if size < max(24, len(needle) // 2):
        return None, None
    return str(chunk["id"]), int(chunk["chunk_index"])


def locate_quote(
    paper: str,
    chunks: list[dict[str, Any]],
    quote: str,
) -> QuoteHit | None:
    found = find_in_text(paper, quote)
    if found is None:
        return None
    start, end, snapped, score = found
    chunk_id, chunk_index = _chunk_for_quote(snapped, chunks)
    return QuoteHit(
        quote=snapped,
        char_start=start,
        char_end=end,
        match_score=score,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
    )


def evidence_for_quotes(
    paper: str,
    chunks: list[dict[str, Any]],
    quotes: list[str],
    *,
    verdict: str,
) -> tuple[list[dict[str, Any]], str]:
    """Located quotes, and ``located`` / ``not_located`` / ``not_required``."""
    located: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for quote in quotes[:3]:
        hit = locate_quote(paper, chunks, quote)
        if hit is None:
            continue
        key = (hit.char_start, hit.char_end)
        if key in seen:
            continue
        seen.add(key)
        located.append({
            "quote": hit.quote,
            "chunk_id": hit.chunk_id,
            "chunk_index": hit.chunk_index,
            "char_start": hit.char_start,
            "char_end": hit.char_end,
            "match_score": hit.match_score,
        })
    if verdict == "not_checkable":
        return located, "not_required"
    if located:
        return located, "located"
    return [], "not_located"
