"""Tests for podcast audio chunk failure surfacing."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from nlp_pillars.schemas import StageStatus
from nlp_pillars.tts.engine import ChunkSynthesisError, TtsEngineStatus
from webui.services import podcast_audio_service


def test_run_podcast_audio_job_marks_stage_failed_on_chunk_error():
    stages: list[tuple[str, str, str | None]] = []

    def on_stage(name: str, status: str, detail: str | None = None) -> None:
        stages.append((name, status, detail))

    mock_script = MagicMock()
    mock_script.script = "[HOST]: Hello.\n"
    mock_script.paper_id = "paper-1"

    mock_client = MagicMock()
    mock_client.check_status.return_value = MagicMock(
        status=TtsEngineStatus.READY,
        message="ready",
    )
    mock_client.synthesize.side_effect = RuntimeError("codec input size zero")

    with patch.object(
        podcast_audio_service.db, "get_podcast_script_by_id", return_value=mock_script
    ), patch.object(
        podcast_audio_service, "validate_voice_for_generation", return_value=MagicMock()
    ), patch.object(
        podcast_audio_service, "get_tts_client", return_value=mock_client
    ), patch.object(
        podcast_audio_service,
        "_settings_paths",
        return_value=("http://x", MagicMock(), MagicMock(), MagicMock(), MagicMock(), ""),
    ):
        with pytest.raises(ChunkSynthesisError) as exc_info:
            podcast_audio_service.run_podcast_audio_job(
                "run-1",
                "script-1",
                "voice.wav",
                on_stage,
                threading.Event(),
            )

    assert exc_info.value.chunk_index == 1
    failed = [(n, s, d) for n, s, d in stages if s == StageStatus.FAILED.value]
    assert any(n == "tts_synthesize" for n, _, _ in failed)
    assert any(d and "chunk 1/1 failed" in d for _, _, d in failed)
