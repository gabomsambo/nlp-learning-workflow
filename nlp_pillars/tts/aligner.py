"""The speech recogniser behind read-along timing.

faster-whisper ``base.en`` on the CPU. It listens to the finished MP3 and
reports when each word it heard began and ended; ``alignment.py`` matches
those words to the script. Nothing here is a paid API and nothing needs the
GPU, so the webui container can time an episode right after recording it.

Measured 2026-10-25 on the 654-second episode ``arxiv:2402.08593`` against the
38 synthesis WAVs it was assembled from (the chunk boundaries are exact):

==========  ============  =========  ==============  ==================
model       recognise     heard      chunk start err  notes
==========  ============  =========  ==============  ==================
base.en     8.0 s         96.1 %     median 56 ms     p90 157 ms
small.en    29.0 s        96.4 %     median 70 ms     p90 232 ms
base.en b5  12.8 s        95.7 %     median 37 ms     one chunk 4.25 s out
==========  ============  =========  ==============  ==================

``small.en`` heard no more of the script and cost 3.6x the time, and beam
search 5 produced a worst case far worse than greedy, so greedy ``base.en`` it is.

The weights are not in the image. They are downloaded once, at the pinned
revision, into ``ALIGNER_CACHE_DIR`` on the uploads volume (about 145 MB).
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from nlp_pillars.tts.alignment import AsrWord

logger = logging.getLogger(__name__)

MODEL_REPO = "Systran/faster-whisper-base.en"
MODEL_REVISION = "3d3d5dee26484f91867d81cb899cfcf72b96be6c"
ENGINE_ID = f"faster-whisper base.en@{MODEL_REVISION[:8]}"
# huggingface.co API usedStorage for that revision, 2026-10-25.
FIRST_RUN_DOWNLOAD_BYTES = 145_000_000

SAMPLE_RATE = 16000
_WEIGHT_FILES = ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")


class AlignerError(RuntimeError):
    """The recogniser could not run: missing dependency, weights or audio."""


def cache_dir() -> Path:
    """Where the weights live. ``ALIGNER_CACHE_DIR``, else a user cache dir."""
    raw = os.getenv("ALIGNER_CACHE_DIR", "").strip()
    path = Path(raw) if raw else Path.home() / ".cache" / "nlp-aligner"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_dir(root: Path) -> Path:
    return root / f"faster-whisper-base.en-{MODEL_REVISION}"


def _weights_ready(dest: Path) -> bool:
    return all((dest / name).is_file() and (dest / name).stat().st_size > 0 for name in _WEIGHT_FILES)


def _snapshot_download(**kwargs):
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def ensure_model(root: Path) -> Path:
    """Download the pinned weights once; later calls stay offline."""
    dest = model_dir(root)
    if _weights_ready(dest):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    logger.warning(
        "Downloading %s@%s into %s (about %s MB, first run only)",
        MODEL_REPO,
        MODEL_REVISION,
        dest,
        FIRST_RUN_DOWNLOAD_BYTES // 1_000_000,
    )
    _snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, local_dir=str(dest))
    if not _weights_ready(dest):
        raise AlignerError(
            f"Download of {MODEL_REPO}@{MODEL_REVISION} into {dest} "
            f"did not produce {', '.join(_WEIGHT_FILES)}."
        )
    return dest


def decode_audio(path: Path) -> np.ndarray:
    """Mono float32 at 16 kHz, decoded by ffmpeg (already in the image)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
    except FileNotFoundError as exc:
        raise AlignerError("ffmpeg is not installed, so the audio cannot be decoded.") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AlignerError(f"ffmpeg could not decode {path.name}: {detail[-1] if detail else proc.returncode}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


_MODEL = None
_MODEL_LOCK = threading.Lock()


def _load_model(root: Path):
    with _MODEL_LOCK:
        return _load_model_locked(root)


def _load_model_locked(root: Path):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise AlignerError(
            "faster-whisper is not installed. It is in requirements.txt; rebuild the image "
            "(or `pip install faster-whisper`) to time episodes."
        ) from exc
    weights = ensure_model(root)
    threads = max(1, min(os.cpu_count() or 1, 8))
    _MODEL = WhisperModel(str(weights), device="cpu", compute_type="int8", cpu_threads=threads)
    return _MODEL


def transcribe_words(audio_path: Path, *, root: Optional[Path] = None) -> Tuple[List[AsrWord], float]:
    """Every word heard in the audio with its times, and the audio's length in seconds."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise AlignerError(f"Audio file not found: {audio_path}")
    samples = decode_audio(audio_path)
    if samples.size == 0:
        raise AlignerError(f"{audio_path.name} decoded to no audio.")
    model = _load_model(root or cache_dir())
    # Greedy, no VAD, no carry-over between windows: see the measurements above.
    segments, _info = model.transcribe(
        samples,
        language="en",
        beam_size=1,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    words: List[AsrWord] = []
    for segment in segments:
        for word in segment.words or []:
            words.append(AsrWord(word.word.strip(), float(word.start), float(word.end)))
    return words, samples.size / SAMPLE_RATE
