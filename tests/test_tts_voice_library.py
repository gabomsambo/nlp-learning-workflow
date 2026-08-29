"""Tests for voice library scanning and validation."""

from pathlib import Path

import pytest

from nlp_pillars.tts.voice_library import VoiceUsability, scan_voice_library


@pytest.fixture
def voice_tree(tmp_path: Path) -> Path:
    short = tmp_path / "good.wav"
    _write_silent_wav(short, duration_seconds=8.0)

    long = tmp_path / "Podcasts" / "long.wav"
    long.parent.mkdir(parents=True)
    _write_silent_wav(long, duration_seconds=200.0)

    tiny = tmp_path / "tiny.wav"
    _write_silent_wav(tiny, duration_seconds=0.5)

    return tmp_path


def _write_silent_wav(path: Path, *, duration_seconds: float) -> None:
    import struct
    import wave

    rate = 22050
    frames = int(rate * duration_seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<h", 0) * frames)


def test_scan_finds_nested_voices(voice_tree: Path):
    entries = scan_voice_library(voice_tree)
    paths = {e.relative_path for e in entries}
    assert "good.wav" in paths
    assert "Podcasts/long.wav" in paths


def test_long_reference_marked_unusable(voice_tree: Path):
    entries = {e.relative_path: e for e in scan_voice_library(voice_tree)}
    assert entries["Podcasts/long.wav"].usability == VoiceUsability.UNUSABLE
    assert "15" in entries["Podcasts/long.wav"].reason


def test_short_reference_marked_unusable(voice_tree: Path):
    entries = {e.relative_path: e for e in scan_voice_library(voice_tree)}
    assert entries["tiny.wav"].usability == VoiceUsability.UNUSABLE


def test_non_wav_without_ffprobe_is_unusable(tmp_path: Path, monkeypatch):
    fake_mp3 = tmp_path / "clip.mp3"
    fake_mp3.write_bytes(b"not really mp3")
    monkeypatch.setattr("nlp_pillars.tts.voice_library.shutil.which", lambda name: None)

    entries = scan_voice_library(tmp_path)
    assert len(entries) == 1
    assert entries[0].usability == VoiceUsability.UNUSABLE
    assert "ffprobe" in entries[0].reason
