"""Join TTS WAV segments and export compressed audio."""

from __future__ import annotations

import subprocess
import wave
from array import array
from pathlib import Path
from typing import Sequence

from nlp_pillars.tts.cue_parser import ParsedScript, SilenceGap, SpeechChunk

SAMPLE_RATE = 22050
SAMPLE_WIDTH = 2  # int16


def _read_wav_pcm(path: Path) -> array:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != SAMPLE_WIDTH:
            raise ValueError(f"Expected mono int16 WAV: {path}")
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    samples = array("h")
    samples.frombytes(frames)
    if rate != SAMPLE_RATE:
        samples = _resample_linear(samples, rate, SAMPLE_RATE)
    return samples


def _resample_linear(samples: array, from_rate: int, to_rate: int) -> array:
    if from_rate == to_rate:
        return samples
    ratio = to_rate / from_rate
    out_len = int(len(samples) * ratio)
    out = array("h", [0] * out_len)
    for i in range(out_len):
        src = i / ratio
        idx = int(src)
        frac = src - idx
        if idx + 1 < len(samples):
            value = samples[idx] * (1 - frac) + samples[idx + 1] * frac
        elif idx < len(samples):
            value = samples[idx]
        else:
            value = 0
        out[i] = int(max(-32768, min(32767, value)))
    return out


def _silence_samples(duration_ms: int) -> array:
    count = int(SAMPLE_RATE * duration_ms / 1000)
    return array("h", [0] * count)


def assemble_timeline_wav(
    parsed: ParsedScript,
    chunk_wav_paths: Sequence[str],
    output_wav: Path,
) -> float:
    """Merge per-line synthesis WAVs and cue silence into one PCM WAV."""
    if len(chunk_wav_paths) != len(parsed.speech_chunks):
        raise ValueError(
            f"Expected {len(parsed.speech_chunks)} WAV files, got {len(chunk_wav_paths)}"
        )

    chunk_iter = iter(chunk_wav_paths)
    pcm: array = array("h")

    for item in parsed.timeline:
        if isinstance(item, SpeechChunk):
            wav_path = next(chunk_iter)
            pcm.extend(_read_wav_pcm(Path(wav_path)))
        elif isinstance(item, SilenceGap) and item.duration_ms > 0:
            pcm.extend(_silence_samples(item.duration_ms))

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    return len(pcm) / SAMPLE_RATE


def assemble_synthesis_chunks_wav(
    wav_paths: Sequence[str],
    output_wav: Path,
    *,
    pause_ms_between: int = 0,
) -> float:
    """Concatenate paragraph-level chunk WAVs with optional gap silence."""
    pcm: array = array("h")
    gap = _silence_samples(pause_ms_between) if pause_ms_between else None
    for index, wav_path in enumerate(wav_paths):
        if index and gap:
            pcm.extend(gap)
        pcm.extend(_read_wav_pcm(Path(wav_path)))

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return len(pcm) / SAMPLE_RATE


def encode_mp3(wav_path: Path, mp3_path: Path, *, bitrate: str = "128k") -> Path:
    """Compress assembled WAV to MP3 via ffmpeg."""
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(mp3_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg MP3 encode failed: {proc.stderr.strip() or proc.stdout}")
    return mp3_path


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())
