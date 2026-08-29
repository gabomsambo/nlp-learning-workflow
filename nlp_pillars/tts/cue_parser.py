"""Parse podcast script markup into speech chunks and silence gaps.

IndexTTS normalizes `[MUSIC]` into speakable tokens — cues must never reach the model.
See the IndexTTS integration contract (firstmate research report, 2026-08-29).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Literal, Union

# Standalone cue lines and the anchored [HOST]: prefix (case-insensitive).
_HOST_PREFIX = re.compile(r"^\[HOST\]:\s*", re.IGNORECASE)
_STANDALONE_CUE = re.compile(
    r"^\[(PAUSE|MUSIC|SFX|TRANSITION)\]\s*$", re.IGNORECASE
)

# Editorial silence durations (ms). IndexTTS defines none for cues.
PAUSE_SILENCE_MS = 400
BOUNDARY_SILENCE_MS = 0  # segment boundary only for MUSIC/SFX/TRANSITION

# Split synthesis calls at paragraph boundaries for retry granularity.
MAX_SPEECH_CHARS = 1800


@dataclass(frozen=True)
class SpeechChunk:
    """Text to send to TTS after cue stripping."""

    text: str
    line_start: int


@dataclass(frozen=True)
class SilenceGap:
    """PCM silence inserted during assembly, not synthesized."""

    duration_ms: int
    cue: Literal["pause", "music", "sfx", "transition", "paragraph"]


TimelineItem = Union[SpeechChunk, SilenceGap]


@dataclass
class ParsedScript:
    """Ordered timeline of speech and silence for one episode."""

    timeline: List[TimelineItem] = field(default_factory=list)
    speech_chunks: List[SpeechChunk] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        return len(self.speech_chunks)


def _flush_speech(buffer: List[str], line_start: int, out: ParsedScript) -> None:
    text = " ".join(part.strip() for part in buffer if part.strip())
    if not text:
        return
    chunk = SpeechChunk(text=text, line_start=line_start)
    out.speech_chunks.append(chunk)
    out.timeline.append(chunk)
    buffer.clear()


def parse_podcast_script(script: str) -> ParsedScript:
    """Turn a [HOST]: script into speech chunks and real silence markers."""
    result = ParsedScript()
    buffer: List[str] = []
    buffer_start = 0

    for line_no, raw_line in enumerate(script.splitlines()):
        line = raw_line.strip()
        if not line:
            if buffer:
                _flush_speech(buffer, buffer_start, result)
                result.timeline.append(SilenceGap(BOUNDARY_SILENCE_MS, "paragraph"))
            continue

        host_match = _HOST_PREFIX.match(line)
        if host_match:
            spoken = line[host_match.end():].strip()
            if spoken:
                if not buffer:
                    buffer_start = line_no
                buffer.append(spoken)
            continue

        cue_match = _STANDALONE_CUE.match(line)
        if cue_match:
            if buffer:
                _flush_speech(buffer, buffer_start, result)
            cue_name = cue_match.group(1).lower()
            if cue_name == "pause":
                result.timeline.append(SilenceGap(PAUSE_SILENCE_MS, "pause"))
            else:
                result.timeline.append(SilenceGap(BOUNDARY_SILENCE_MS, cue_name))  # type: ignore[arg-type]
            continue

        # Non-[HOST] prose without a cue prefix: treat as narration (rare).
        if not buffer:
            buffer_start = line_no
        buffer.append(line)

    if buffer:
        _flush_speech(buffer, buffer_start, result)

    return result


@dataclass(frozen=True)
class SynthesisPlan:
    """One IndexTTS call plus silence to insert after it during assembly."""

    text: str
    pause_after_ms: int = 0


def plan_synthesis_chunks(parsed: ParsedScript) -> List[SynthesisPlan]:
    """Merge timeline speech into bounded synthesis calls.

    Flushes on paragraph boundaries, standalone [PAUSE] markers, and the char
    budget so a failed chunk has a smaller retry blast radius than one whole-script
    Gradio job.
    """
    plans: List[SynthesisPlan] = []
    current: List[str] = []
    current_len = 0
    pending_pause_ms = 0

    def flush(pause_ms: int = 0) -> None:
        nonlocal pending_pause_ms, current_len
        text = " ".join(part.strip() for part in current if part.strip())
        trailing = pause_ms or pending_pause_ms
        pending_pause_ms = 0
        if text:
            plans.append(SynthesisPlan(text=text, pause_after_ms=trailing))
        current.clear()
        current_len = 0

    for item in parsed.timeline:
        if isinstance(item, SilenceGap):
            if item.cue == "pause":
                if current:
                    flush(pause_ms=item.duration_ms)
                elif plans:
                    plans[-1] = SynthesisPlan(
                        text=plans[-1].text,
                        pause_after_ms=plans[-1].pause_after_ms + item.duration_ms,
                    )
                else:
                    pending_pause_ms += item.duration_ms
            elif item.cue == "paragraph" and current:
                flush()
            continue

        if current_len + len(item.text) + 1 > MAX_SPEECH_CHARS and current:
            flush()
        current.append(item.text)
        current_len = sum(len(part) + 1 for part in current)

    if current:
        flush()

    return [plan for plan in plans if plan.text.strip()]
