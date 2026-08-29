"""Normalize podcast speech text before IndexTTS synthesis.

IndexTTS splits internally on hyphen tokens after normalizing em-dashes to ``-``.
That can isolate a trailing quote or punctuation into its own one-token segment,
which crashes the semantic codec with "input size (0)" (measured on chunk 9,
2026-08-29).
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

# IndexTTS uses max_text_tokens_per_segment=120 on /gen_single (see indextts_client).
DEFAULT_MAX_TOKENS_PER_SEGMENT = 120

# Unicode dashes IndexTTS normalizes to ASCII hyphen, triggering split_segments.
_EM_EN_DASHES = (
    "\u2014",  # em dash — the chunk-9 trigger
    "\u2013",  # en dash
    "\u2212",  # minus sign
)

# Curly/smart quotes → straight so closing quotes are less likely to split alone.
_CURLY_QUOTE_MAP = str.maketrans({
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
})

# SentencePiece-style tokens that carry no speakable content on their own.
_PUNCTUATION_ONLY_RE = re.compile(
    r"^[\s\.\,\!\?\;\:\-\—\–\''\"\"…]+$",
    re.UNICODE,
)


def prepare_text_for_indextts(text: str) -> str:
    """Return text safe for IndexTTS segment splitting."""
    prepared = text.translate(_CURLY_QUOTE_MAP)
    for dash in _EM_EN_DASHES:
        prepared = prepared.replace(dash, ", ")
    # Collapse any double spaces introduced by dash replacement.
    prepared = re.sub(r"  +", " ", prepared)
    return prepared.strip()


def is_degenerate_segment(segment_text: str, token_count: int) -> bool:
    """True when a segment would crash IndexTTS (punctuation-only micro-segment)."""
    if token_count <= 0:
        return True
    # Strip SentencePiece word-boundary marker for the punctuation check.
    visible = segment_text.replace("▁", "").strip()
    if not visible:
        return True
    if token_count <= 2 and _PUNCTUATION_ONLY_RE.match(visible):
        return True
    return False


def find_degenerate_segments(
    segments: Sequence[Tuple[int, str, int]],
) -> List[Tuple[int, str, int]]:
    """Return segments that would fail IndexTTS synthesis."""
    bad: List[Tuple[int, str, int]] = []
    for index, content, token_count in segments:
        if is_degenerate_segment(content, token_count):
            bad.append((index, content, token_count))
    return bad


def describe_degenerate_segment(index: int, content: str, token_count: int) -> str:
    visible = content.replace("▁", "").strip() or content
    if len(visible) > 60:
        visible = visible[:57] + "..."
    return (
        f"segment {index + 1} would be punctuation-only after IndexTTS splitting "
        f"({token_count} token(s): {visible!r})"
    )
