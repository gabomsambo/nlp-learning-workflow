"""Tests for podcast audio chunk failure surfacing."""

from __future__ import annotations

import threading
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nlp_pillars.schemas import StageStatus
from nlp_pillars.tts.engine import ChunkSynthesisError, TtsEngineStatus
from webui.services import podcast_audio_service


def _write_wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(22050)
        output.writeframes(array("h", [0] * 220).tobytes())
    return path


class FakeIndexTTS:
    def __init__(self):
        self.download_dir = None
        self.texts = []
        self.fail_text_once = None
        self.failed = False

    def check_status(self):
        return SimpleNamespace(status=TtsEngineStatus.READY, message="ready")

    def synthesize(self, text, voice_path, *, on_progress=None, **kwargs):
        self.texts.append(text)
        if text == self.fail_text_once and not self.failed:
            self.failed = True
            raise RuntimeError("fake synthesis failure")
        path = Path(self.download_dir) / f"chunk-{len(self.texts)}.wav"
        return str(_write_wav(path))


def _configure_job(tmp_path, monkeypatch, script_text, fake_client):
    voice_path = tmp_path / "voices" / "nested" / "voice.wav"
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    voice_path.write_bytes(b"fake voice reference")
    audio_dir = tmp_path / "audio"
    download_root = tmp_path / "downloads"
    preview_dir = tmp_path / "previews"
    monkeypatch.setattr(
        podcast_audio_service.db,
        "get_podcast_script_by_id",
        lambda _script_id: SimpleNamespace(
            script=script_text,
            paper_id="paper-1",
            title="A Useful Paper",
        ),
    )
    monkeypatch.setattr(
        podcast_audio_service,
        "validate_voice_for_generation",
        lambda _voice: voice_path,
    )

    def client_for_run(download_dir):
        fake_client.download_dir = Path(download_dir)
        fake_client.download_dir.mkdir(parents=True, exist_ok=True)
        return fake_client

    monkeypatch.setattr(podcast_audio_service, "get_tts_client", client_for_run)

    def write_fake_mp3(_source, target):
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake mp3")
        return path

    monkeypatch.setattr(
        podcast_audio_service,
        "_settings_paths",
        lambda: (
            "http://index-tts.invalid",
            voice_path.parent,
            audio_dir,
            download_root,
            preview_dir,
            "start-index-tts",
        ),
    )
    monkeypatch.setattr(
        podcast_audio_service,
        "encode_mp3",
        write_fake_mp3,
    )
    monkeypatch.setattr(
        podcast_audio_service,
        "compute_timeline",
        lambda _text, _path: {
            "word_count": 4, "matched_ratio": 1.0, "word_level": True,
        },
    )
    return voice_path, audio_dir, download_root


def test_run_podcast_audio_job_reports_chunk_error_and_cleans_run_downloads(
    tmp_path, monkeypatch,
):
    stages: list[tuple[str, str, str | None]] = []

    def on_stage(name: str, status: str, detail: str | None = None) -> None:
        stages.append((name, status, detail))

    fake_client = FakeIndexTTS()
    fake_client.fail_text_once = "Hello."
    _voice, _audio_dir, download_root = _configure_job(
        tmp_path, monkeypatch, "[HOST]: Hello.\n", fake_client,
    )

    with pytest.raises(ChunkSynthesisError) as exc_info:
        podcast_audio_service.run_podcast_audio_job(
            "run-1", "script-1", "nested/voice.wav", on_stage, threading.Event(),
        )

    assert exc_info.value.chunk_index == 1
    failed = [(n, s, d) for n, s, d in stages if s == StageStatus.FAILED.value]
    assert any(n == "tts_synthesize" for n, _, _ in failed)
    assert any(d and "chunk 1/1 failed" in d for _, _, d in failed)
    assert not (download_root / "run-run-1").exists()


SCRIPT_WITH_CUES = (
    "[HOST]: Welcome to the paper. [[F12]]\n"
    "[PAUSE]\n[MUSIC]\n"
    "[HOST]: The result is surprising.\n"
)


def test_retry_reuses_matching_chunk_checkpoints_and_cleans_scratch(
    tmp_path, monkeypatch,
):
    fake_client = FakeIndexTTS()
    fake_client.fail_text_once = "The result is surprising."
    _voice, _audio_dir, download_root = _configure_job(
        tmp_path, monkeypatch, SCRIPT_WITH_CUES, fake_client,
    )
    monkeypatch.setattr(
        podcast_audio_service.db, "update_podcast_audio_metadata", MagicMock(),
    )
    metadata_rows = []
    stages = []
    with pytest.raises(ChunkSynthesisError) as exc_info:
        podcast_audio_service.run_podcast_audio_job(
            "run-first", "script-1", "nested/voice.wav",
            lambda *args: stages.append(args), threading.Event(),
        )
    assert exc_info.value.chunk_index == 2
    assert not (download_root / "run-run-first").exists()
    cached = list((download_root / "checkpoints").glob("*.wav"))
    assert len(cached) == 1

    monkeypatch.setattr(
        podcast_audio_service.db,
        "update_podcast_audio_metadata",
        lambda _script_id, metadata: metadata_rows.append(metadata),
    )
    podcast_audio_service.run_podcast_audio_job(
        "run-second", "script-1", "nested/voice.wav",
        lambda *args: stages.append(args), threading.Event(),
    )

    assert fake_client.texts == [
        "Welcome to the paper.",
        "The result is surprising.",
        "The result is surprising.",
    ]
    assert all(
        not any(token in text.upper() for token in ("HOST", "PAUSE", "MUSIC", "F12"))
        for text in fake_client.texts
    )
    assert not cached[0].exists()
    assert not (download_root / "run-run-second").exists()
    assert Path(metadata_rows[0].file_path).is_file()
    assert metadata_rows[0].voice_sha256
    assert metadata_rows[0].chunk_fingerprints


def test_synthesis_fingerprint_changes_with_request_inputs():
    base = dict(
        text="A planned chunk.", script_sha256="script-a", voice_sha256="voice-a",
        engine_revision="revision-a", params={"temperature": 0.2},
    )
    fingerprint = podcast_audio_service._chunk_fingerprint(**base)
    for key, changed in (
        ("script_sha256", "script-b"),
        ("voice_sha256", "voice-b"),
        ("engine_revision", "revision-b"),
        ("text", "A changed chunk."),
        ("params", {"temperature": 0.3}),
    ):
        variant = dict(base)
        variant[key] = changed
        assert podcast_audio_service._chunk_fingerprint(**variant) != fingerprint


def test_output_filename_uses_paper_title_short_id_voice_and_utc_timestamp():
    from datetime import datetime, timezone

    script = SimpleNamespace(title="Deep Dive: Attention / Is All You Need")
    filename = podcast_audio_service._output_filename(
        script,
        "a1234567-89ab-cdef-0123-456789abcdef",
        "Group/Voice Host.wav",
        datetime(2026, 10, 31, 12, 13, 14, 1500, tzinfo=timezone.utc),
    )

    assert filename == (
        "attention-is-all-you-need-a1234567_voice-host_20261031T121314001500Z.mp3"
    )
    assert "/" not in filename and "\\" not in filename and ".." not in filename


def test_zero_speakable_content_fails_before_tts_and_marks_prepare_failed(
    tmp_path, monkeypatch,
):
    fake_client = FakeIndexTTS()
    _configure_job(tmp_path, monkeypatch, "[MUSIC]\n[PAUSE]\n", fake_client)
    stages = []
    with pytest.raises(ValueError, match="no speakable content"):
        podcast_audio_service.run_podcast_audio_job(
            "run-empty", "script-1", "nested/voice.wav",
            lambda *args: stages.append(args), threading.Event(),
        )
    assert fake_client.texts == []
    assert any(
        name == "tts_prepare" and status == StageStatus.FAILED.value
        for name, status, _ in stages
    )


def test_metadata_failure_removes_the_unreferenced_mp3_but_keeps_retry_wavs(
    tmp_path, monkeypatch,
):
    fake_client = FakeIndexTTS()
    _voice, audio_dir, download_root = _configure_job(
        tmp_path, monkeypatch, "[HOST]: Hello.\n", fake_client,
    )
    monkeypatch.setattr(
        podcast_audio_service.db,
        "update_podcast_audio_metadata",
        MagicMock(side_effect=RuntimeError("database unavailable")),
    )
    with pytest.raises(RuntimeError, match="database unavailable"):
        podcast_audio_service.run_podcast_audio_job(
            "run-save-failed", "script-1", "nested/voice.wav",
            lambda *args: None, threading.Event(),
        )
    assert not (download_root / "run-run-save-failed").exists()
    assert not list(audio_dir.glob("*.mp3"))
    assert list((download_root / "checkpoints").glob("*.wav"))
