"""The cloud copy of an episode's audio: 64 kbps mono MP3.

Read-along timings are milliseconds into the file, so the transcode must not
move them. ``transcode`` therefore probes both files and refuses a copy whose
duration differs from the original by more than ``MAX_DURATION_DELTA_S``.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

BITRATE_KBPS = 64
# LAME's encoder delay is ~1105 samples (50 ms at 22.05 kHz) and the Xing header
# makes ffprobe subtract it; measured delta on a real 10 MB episode was 0.000 s.
MAX_DURATION_DELTA_S = 0.1
_TIMEOUT_S = 900


class AudioError(Exception):
    """An MP3 could not be probed, encoded, or did not keep its duration."""


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise AudioError(f"{tool} is not installed. The image installs ffmpeg.")
    return path


def probe_duration(path: Path) -> float:
    """Container duration in seconds, as ffprobe reports it."""
    proc = subprocess.run(
        [
            _require("ffprobe"), "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, timeout=60, check=False,
    )
    try:
        seconds = float(proc.stdout.strip())
    except ValueError:
        raise AudioError(
            f"ffprobe could not read the duration of {path.name}: "
            f"{proc.stderr.strip() or proc.stdout.strip() or 'no output'}"
        ) from None
    if proc.returncode != 0 or seconds <= 0:
        raise AudioError(f"ffprobe reported duration {seconds} for {path.name}")
    return seconds


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def transcode(
    source: Path,
    dest: Path,
    *,
    bitrate_kbps: int = BITRATE_KBPS,
    max_delta_s: float = MAX_DURATION_DELTA_S,
    probe: Callable[[Path], float] = probe_duration,
) -> float:
    """Write ``dest`` as mono MP3 and return its duration in seconds.

    Metadata is dropped (``-map_metadata -1``) and ``bitexact`` keeps the encoder
    from stamping its version, so the same input always yields the same bytes
    and the content hash is stable. ``dest`` appears only if the duration check
    passes.
    """
    ffmpeg = _require("ffmpeg")
    original = probe(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=dest.parent, prefix=".transcode-", suffix=".mp3", delete=False
    ) as handle:
        scratch = Path(handle.name)
    try:
        proc = subprocess.run(
            [
                ffmpeg, "-y", "-v", "error", "-i", str(source), "-vn",
                "-map_metadata", "-1", "-ac", "1", "-codec:a", "libmp3lame",
                "-b:a", f"{bitrate_kbps}k", "-fflags", "+bitexact",
                "-flags:a", "+bitexact", str(scratch),
            ],
            capture_output=True, text=True, timeout=_TIMEOUT_S, check=False,
        )
        if proc.returncode != 0:
            raise AudioError(
                f"ffmpeg could not encode {source.name}: "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        copy = probe(scratch)
        delta = abs(copy - original)
        if delta > max_delta_s:
            raise AudioError(
                f"{source.name}: the {bitrate_kbps} kbps copy is {copy:.3f}s long, "
                f"the original {original:.3f}s (delta {delta:.3f}s, limit "
                f"{max_delta_s:.3f}s). Read-along timings would drift."
            )
        scratch.replace(dest)
        return copy
    finally:
        scratch.unlink(missing_ok=True)
