"""Citation tags: ``[[F12,T3]]`` at the end of a script line.

The evidence writer ends every spoken line with the ledger ids it relies on, or
``[[-]]`` when the line states no fact about the paper. The tags are
bookkeeping. They are parsed once, right after the model answers, and the
script that is stored, fact-checked, narrated and aligned is the stripped one;
the ids live in ``line_citations``, keyed by that stripped script's line index.

ASCII brackets on purpose. The first run used Unicode brackets and the model
put a stray character in front of the closing one.

Stripping is strict: only text that matches the id grammar is removed, so a
double-bracket that is part of what the host says is never eaten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

TAG = re.compile(r"\[\[([^\[\]\n]*)\]\]")
ID = re.compile(r"^[FT]\d+$")
_TAG_BODY = re.compile(r"^\s*(?:-|[FT]\d+(?:\s*[,;\s]\s*[FT]\d+)*)\s*$", re.I)
_SPLIT = re.compile(r"[,;\s]+")
_CUES = ("PAUSE", "MUSIC", "SFX", "TRANSITION")
_STANDALONE_CUE = re.compile(r"^\s*\[(PAUSE|MUSIC|SFX|TRANSITION)\]\s*$", re.I)
_SPACES = re.compile(r"[ \t]{2,}")

NO_FACT = "-"


@dataclass
class ParsedScript:
    """A tagged script, after parsing."""

    script: str
    #: line index (0-based, in ``script``) -> cited ids. ``[]`` is ``[[-]]``.
    citations: dict[int, list[str]] = field(default_factory=dict)
    #: spoken lines with no tag at all.
    untagged: list[int] = field(default_factory=list)
    #: ``(line, text)`` for tags that were not valid ids and were dropped.
    invalid: list[tuple[int, str]] = field(default_factory=list)
    spoken_lines: int = 0

    def as_storage(self) -> dict[str, list[str]]:
        """JSON-safe form: object keys are strings."""
        return {str(i): ids for i, ids in sorted(self.citations.items())}


def is_spoken(line: str) -> bool:
    """A line the narrator reads: not blank, not a standalone cue."""
    stripped = line.strip()
    return bool(stripped) and not _STANDALONE_CUE.match(stripped)


def _tidy(line: str) -> str:
    return _SPACES.sub(" ", line).rstrip()


def parse_citations(text: str) -> ParsedScript:
    """Split a tagged script into the clean script and its citations.

    Every spoken line should carry one tag group. A line with none is recorded
    in ``untagged``; a tag that is not an id list (``[[PAUSE]]``, ``[[note]]``)
    is dropped and recorded in ``invalid``. A tag group that is a cue name on
    a line of its own becomes that cue, because the model has been seen to
    write ``[[PAUSE]]``.
    """
    parsed = ParsedScript(script="")
    out: list[str] = []
    for index, raw in enumerate(text.splitlines()):
        groups = TAG.findall(raw)
        if not groups:
            out.append(raw.rstrip())
            if is_spoken(raw):
                parsed.spoken_lines += 1
                parsed.untagged.append(index)
            continue

        ids: list[str] = []
        valid_group = False
        for group in groups:
            body = group.strip()
            if body == NO_FACT:
                valid_group = True
                continue
            if body.upper() in _CUES:
                continue
            tokens = [t.upper() for t in _SPLIT.split(body) if t]
            good = [t for t in tokens if ID.match(t)]
            bad = [t for t in tokens if not ID.match(t)]
            if good:
                valid_group = True
                ids.extend(i for i in good if i not in ids)
            if bad or not tokens:
                parsed.invalid.append((index, group))

        stripped = TAG.sub("", raw)
        cue_only = [g.strip().upper() for g in groups if g.strip().upper() in _CUES]
        if not stripped.strip() and cue_only:
            out.append(f"[{cue_only[0]}]")
            continue
        out.append(_tidy(stripped))
        if is_spoken(stripped):
            parsed.spoken_lines += 1
            if valid_group:
                parsed.citations[index] = ids
            else:
                parsed.untagged.append(index)
    parsed.script = "\n".join(out)
    if text.endswith("\n"):
        parsed.script += "\n"
    return parsed


def strip_citation_tags(text: str) -> str:
    """Remove id-list tags from ``text``; leave every other double bracket.

    Defence in depth for the narrator: the stored script is already clean, but
    a script that predates stripping, or one edited by hand, must never have
    ``F12`` read aloud.
    """
    if "[[" not in text:
        return text

    def drop(match: re.Match) -> str:
        return "" if _TAG_BODY.match(match.group(1)) else match.group(0)

    lines = []
    for line in text.split("\n"):
        cleaned = TAG.sub(drop, line)
        lines.append(_tidy(cleaned) if cleaned != line else line)
    return "\n".join(lines)


def resolve_ids(
    citations: dict[int, list[str]], known: Iterable[str]
) -> tuple[dict[int, list[str]], list[tuple[int, str]], list[int]]:
    """Check cited ids against the ledger.

    Returns ``(citations, unknown, unresolved)``: the citations with unknown
    ids removed; ``(line, id)`` for each id the ledger does not have; and the
    lines whose every id was unknown, which cite nothing real and are not
    ``[[-]]``.
    """
    have = set(known)
    kept: dict[int, list[str]] = {}
    unknown: list[tuple[int, str]] = []
    unresolved: list[int] = []
    for line, ids in citations.items():
        good = [i for i in ids if i in have]
        for i in ids:
            if i not in have:
                unknown.append((line, i))
        if ids and not good:
            unresolved.append(line)
        kept[line] = good
    return kept, unknown, unresolved
