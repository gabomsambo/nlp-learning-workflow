"""The cloud copy of an episode's audio: 64 kbps mono, same length.

Runs the real ffmpeg on a synthetic tone. CI installs ffmpeg (tests.yml); a
machine without it skips these rather than pretending the transcode works.
"""

import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from nlp_pillars.cloud import audio

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed"
)


def _wav(path: Path, seconds: float, *, rate: int = 22050, channels: int = 1) -> Path:
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * i / rate))) * channels
        for i in range(int(seconds * rate))
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return path


def _episode(tmp_path: Path, seconds: float = 6.0, channels: int = 1, rate: int = 22050) -> Path:
    """What the app stores today: 128 kbps MP3 at the TTS sample rate."""
    source = tmp_path / "episode.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(_wav(tmp_path / "t.wav", seconds, rate=rate, channels=channels)),
         "-codec:a", "libmp3lame", "-b:a", "128k", str(source)],
        check=True,
    )
    return source


def _stream(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=channels,sample_rate,codec_name:format=bit_rate",
         "-of", "default=nw=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return dict(line.split("=") for line in out.split())


def test_transcode_is_64kbps_mono_and_keeps_the_duration(tmp_path):
    source = _episode(tmp_path)
    dest = tmp_path / "out" / "copy.mp3"
    seconds = audio.transcode(source, dest)
    assert abs(seconds - audio.probe_duration(source)) <= audio.MAX_DURATION_DELTA_S
    assert abs(audio.probe_duration(dest) - audio.probe_duration(source)) <= audio.MAX_DURATION_DELTA_S
    info = _stream(dest)
    assert info["codec_name"] == "mp3" and info["channels"] == "1" and info["sample_rate"] == "22050"
    assert 60_000 <= int(info["bit_rate"]) <= 68_000
    assert dest.stat().st_size < source.stat().st_size * 0.6


def test_a_stereo_44k_source_comes_out_mono_and_still_the_same_length(tmp_path):
    source = _episode(tmp_path, seconds=4.0, channels=2, rate=44100)
    dest = tmp_path / "copy.mp3"
    audio.transcode(source, dest)
    assert _stream(dest)["channels"] == "1"
    assert abs(audio.probe_duration(dest) - audio.probe_duration(source)) <= audio.MAX_DURATION_DELTA_S


def test_the_same_input_gives_the_same_bytes(tmp_path):
    source = _episode(tmp_path)
    audio.transcode(source, tmp_path / "a.mp3")
    audio.transcode(source, tmp_path / "b.mp3")
    assert audio.sha256_file(tmp_path / "a.mp3") == audio.sha256_file(tmp_path / "b.mp3")


def test_a_copy_that_drifts_is_refused_and_leaves_no_file(tmp_path):
    source = _episode(tmp_path)
    dest = tmp_path / "copy.mp3"
    real = audio.probe_duration

    def drifting(path):  # the copy "measures" a second longer than the original
        return real(path) + (1.0 if path.name != source.name else 0.0)

    with pytest.raises(audio.AudioError, match=r"delta 1\.000s.*Read-along"):
        audio.transcode(source, dest, probe=drifting)
    assert not dest.exists() and not list(tmp_path.glob(".transcode-*"))


def test_the_tolerance_is_what_decides(tmp_path):
    source = _episode(tmp_path)
    real = audio.probe_duration
    probe = lambda p: real(p) + (0.5 if p.name != source.name else 0.0)  # noqa: E731
    audio.transcode(source, tmp_path / "ok.mp3", max_delta_s=0.6, probe=probe)
    with pytest.raises(audio.AudioError):
        audio.transcode(source, tmp_path / "no.mp3", max_delta_s=0.4, probe=probe)


def test_garbage_input_is_an_audio_error_not_a_crash(tmp_path):
    source = tmp_path / "bad.mp3"
    source.write_bytes(b"this is not audio")
    with pytest.raises(audio.AudioError):
        audio.transcode(source, tmp_path / "copy.mp3")
    assert not (tmp_path / "copy.mp3").exists()


def test_a_missing_tool_is_named(monkeypatch, tmp_path):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    with pytest.raises(audio.AudioError, match="ffprobe is not installed"):
        audio.probe_duration(tmp_path / "x.mp3")
