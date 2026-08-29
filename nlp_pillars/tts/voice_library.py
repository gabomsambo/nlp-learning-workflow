"""Scan and validate the captain's voice reference library."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".mp4", ".ogg"}
SKIP_NAMES = {".DS_Store"}
SKIP_PREFIXES = ("._",)

# IndexTTS uses the first 15 seconds only; longer files decode fully first.
MAX_REFERENCE_SECONDS = 15.0
MIN_REFERENCE_SECONDS = 2.0
RECOMMENDED_MAX_SECONDS = 15.0


class VoiceUsability(str, Enum):
    USABLE = "usable"
    WARNING = "warning"
    UNUSABLE = "unusable"


@dataclass(frozen=True)
class VoiceEntry:
    """One audio file in the nested voice tree."""

    relative_path: str
    absolute_path: str
    label: str
    duration_seconds: Optional[float]
    usability: VoiceUsability
    reason: str


def _is_audio_file(path: Path) -> bool:
    if path.name in SKIP_NAMES:
        return False
    if any(path.name.startswith(p) for p in SKIP_PREFIXES):
        return False
    return path.suffix.lower() in AUDIO_EXTENSIONS


def iter_voice_files(root: Path) -> Iterator[Path]:
    """Walk the nested voice folder recursively."""
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and _is_audio_file(path):
            yield path


def _wav_duration_seconds(path: Path) -> Optional[float]:
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return wf.getnframes() / float(rate)
    except wave.Error:
        return None


def _ffprobe_duration_seconds(path: Path) -> Optional[float]:
    if not shutil.which("ffprobe"):
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            return float(data["format"]["duration"])
    except (subprocess.SubprocessError, KeyError, ValueError, json.JSONDecodeError, FileNotFoundError):
        return None
    return None


def _probe_duration_seconds(path: Path) -> Optional[float]:
    """Duration from the WAV header when possible; ffprobe only for other formats."""
    if path.suffix.lower() == ".wav":
        return _wav_duration_seconds(path)
    return _ffprobe_duration_seconds(path)


def _classify(path: Path, duration: Optional[float]) -> tuple[VoiceUsability, str]:
    if duration is None:
        if path.suffix.lower() == ".wav":
            return VoiceUsability.UNUSABLE, "Could not read WAV header — file may be corrupt."
        if not shutil.which("ffprobe"):
            ext = path.suffix.lower() or "this format"
            return (
                VoiceUsability.UNUSABLE,
                f"Cannot inspect {ext} without ffprobe — use a WAV reference or install ffmpeg.",
            )
        return VoiceUsability.UNUSABLE, "Could not read audio — file may be corrupt."

    if duration < MIN_REFERENCE_SECONDS:
        return (
            VoiceUsability.UNUSABLE,
            f"Too short ({duration:.1f}s). Need at least {MIN_REFERENCE_SECONDS:.0f}s of speech.",
        )

    if duration > 120:
        return (
            VoiceUsability.UNUSABLE,
            (
                f"Very long reference ({duration / 60:.1f} min). IndexTTS decodes the "
                f"whole file but uses only the first {MAX_REFERENCE_SECONDS:.0f}s — "
                "extract a 5–15s clip first."
            ),
        )

    if duration > RECOMMENDED_MAX_SECONDS:
        return (
            VoiceUsability.WARNING,
            (
                f"Long reference ({duration:.1f}s). Only the first "
                f"{MAX_REFERENCE_SECONDS:.0f}s affects the voice; a shorter clip is safer."
            ),
        )

    return VoiceUsability.USABLE, f"OK ({duration:.1f}s)"


def scan_voice_library(root: Path) -> List[VoiceEntry]:
    """Enumerate voices with usability preflight."""
    entries: List[VoiceEntry] = []
    for path in iter_voice_files(root):
        rel = path.relative_to(root).as_posix()
        duration = _probe_duration_seconds(path)
        usability, reason = _classify(path, duration)
        entries.append(
            VoiceEntry(
                relative_path=rel,
                absolute_path=str(path),
                label=rel,
                duration_seconds=duration,
                usability=usability,
                reason=reason,
            )
        )
    return entries


def resolve_voice_path(root: Path, relative_path: str) -> Path:
    """Resolve a library-relative path and reject traversal."""
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    if not str(candidate).startswith(str(root)):
        raise ValueError("Voice path escapes the library root")
    if not candidate.is_file():
        raise FileNotFoundError(f"Voice file not found: {relative_path}")
    return candidate


def extract_preview_clip(
    source: Path,
    dest: Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float = 10.0,
) -> Path:
    """Write a mono 22.05 kHz WAV preview of the reference clip."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-t",
        str(min(duration_seconds, MAX_REFERENCE_SECONDS)),
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "22050",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg preview failed: {proc.stderr.strip() or proc.stdout}")
    return dest
