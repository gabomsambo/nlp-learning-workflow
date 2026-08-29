"""Background worker for podcast script → MP3 via IndexTTS."""

from __future__ import annotations

import logging
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from nlp_pillars import db
from nlp_pillars.config import get_settings
from nlp_pillars.schemas import AudioMetadata, StageStatus
from nlp_pillars.orchestrator import RunCancelledError
from nlp_pillars.tts.audio_assembler import encode_mp3, wav_duration_seconds
from nlp_pillars.tts.cue_parser import parse_podcast_script, plan_synthesis_chunks
from nlp_pillars.tts.engine import TtsEngineStatus
from nlp_pillars.tts.indextts_client import IndexTtsClient
from nlp_pillars.tts.voice_library import (
    VoiceUsability,
    extract_preview_clip,
    resolve_voice_path,
    scan_voice_library,
)

logger = logging.getLogger(__name__)

KIND_PODCAST_AUDIO = "podcast_audio"
TRIGGER_UI_PODCAST_AUDIO = "ui_podcast_audio"

StageCallback = Callable[[str, str, Optional[str]], None]


def _slugify(value: str, *, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return (slug[:max_len] or "voice")


def _settings_paths() -> tuple[str, Path, Path, Path, Path, str]:
    settings = get_settings()
    return (
        settings.indextts_url,
        Path(settings.voices_dir),
        Path(settings.podcast_audio_dir),
        Path(settings.tts_download_dir),
        Path(settings.tts_preview_dir),
        settings.indextts_start_command,
    )


def get_tts_client() -> IndexTtsClient:
    base_url, _, _, download_dir, _, start_command = _settings_paths()
    return IndexTtsClient(
        base_url,
        download_dir,
        start_command=start_command,
    )


def tts_status_payload() -> Dict[str, Any]:
    client = get_tts_client()
    info = client.check_status()
    return {
        "status": info.status.value,
        "message": info.message,
        "start_command": info.start_command,
        "base_url": info.base_url,
        "ready": info.status == TtsEngineStatus.READY,
    }


def list_voices_payload() -> Dict[str, Any]:
    _, voices_root, _, _, _, _ = _settings_paths()
    entries = scan_voice_library(voices_root)
    return {
        "voices": [
            {
                "relative_path": entry.relative_path,
                "label": entry.label,
                "duration_seconds": entry.duration_seconds,
                "usability": entry.usability.value,
                "reason": entry.reason,
            }
            for entry in entries
        ],
        "root": str(voices_root),
    }


def preview_voice(relative_path: str) -> Path:
    _, voices_root, _, _, preview_dir, _ = _settings_paths()
    source = resolve_voice_path(voices_root, relative_path)
    dest = preview_dir / f"{_slugify(relative_path)}.preview.wav"
    return extract_preview_clip(source, dest)


def validate_voice_for_generation(relative_path: str) -> Path:
    _, voices_root, _, _, _, _ = _settings_paths()
    source = resolve_voice_path(voices_root, relative_path)
    entry = next(
        (e for e in scan_voice_library(voices_root) if e.relative_path == relative_path),
        None,
    )
    if entry is None:
        raise ValueError(f"Unknown voice: {relative_path}")
    if entry.usability == VoiceUsability.UNUSABLE:
        raise ValueError(entry.reason)
    return source


def run_podcast_audio_job(
    run_id: str,
    script_id: str,
    voice_relative_path: str,
    on_stage: StageCallback,
    cancel: threading.Event,
) -> None:
    """Synthesize one podcast script to MP3 on a worker thread."""
    if cancel.is_set():
        raise RunCancelledError("cancelled before start")

    script = db.get_podcast_script_by_id(script_id)
    if not script:
        raise ValueError(f"Script not found: {script_id}")

    voice_source = validate_voice_for_generation(voice_relative_path)
    client = get_tts_client()
    status = client.check_status()
    if status.status != TtsEngineStatus.READY:
        raise RuntimeError(status.message)

    on_stage("tts_prepare", StageStatus.RUNNING.value, "Parsing script and planning chunks")
    parsed = parse_podcast_script(script.script)
    plans = plan_synthesis_chunks(parsed)
    if not plans:
        raise ValueError("Script has no speakable content after cue stripping")
    on_stage(
        "tts_prepare",
        StageStatus.COMPLETED.value,
        f"{len(plans)} chunk(s) planned",
    )

    if cancel.is_set():
        raise RunCancelledError("cancelled")

    _, _, audio_dir, download_dir, _, _ = _settings_paths()
    work_dir = download_dir / f"run-{run_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[str] = []

    on_stage("tts_synthesize", StageStatus.RUNNING.value, f"chunk 0/{len(plans)}")
    for index, plan in enumerate(plans, start=1):
        if cancel.is_set():
            raise RunCancelledError("cancelled")

        def _progress_cb(progress, idx=index, total=len(plans)) -> None:
            on_stage(
                "tts_synthesize",
                StageStatus.RUNNING.value,
                f"chunk {idx}/{total} — {progress.description}",
            )

        wav_path = client.synthesize(
            plan.text,
            str(voice_source),
            on_progress=_progress_cb,
        )
        chunk_paths.append(wav_path)
        on_stage(
            "tts_synthesize",
            StageStatus.RUNNING.value,
            f"chunk {index}/{len(plans)} complete",
        )

    on_stage(
        "tts_synthesize",
        StageStatus.COMPLETED.value,
        f"{len(plans)} chunk(s) synthesized",
    )

    if cancel.is_set():
        raise RunCancelledError("cancelled")

    on_stage("tts_assemble", StageStatus.RUNNING.value, "Joining audio and silences")
    assembled_wav = work_dir / "assembled.wav"
    _assemble_with_variable_pauses(
        [Path(w) for w in chunk_paths],
        [plan.pause_after_ms for plan in plans],
        assembled_wav,
    )
    duration = wav_duration_seconds(assembled_wav)
    on_stage(
        "tts_assemble",
        StageStatus.COMPLETED.value,
        f"{duration:.1f}s assembled",
    )

    on_stage("tts_encode", StageStatus.RUNNING.value, "Encoding MP3")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    voice_slug = _slugify(Path(voice_relative_path).stem)
    paper_slug = _slugify(script.paper_id)
    filename = f"{paper_slug}_{voice_slug}_{timestamp}.mp3"
    mp3_path = Path(audio_dir) / filename
    encode_mp3(assembled_wav, mp3_path)
    on_stage("tts_encode", StageStatus.COMPLETED.value, filename)

    on_stage("tts_save", StageStatus.RUNNING.value, "Saving metadata")
    metadata = AudioMetadata(
        engine="indextts",
        voice_path=voice_relative_path,
        voice_label=voice_relative_path,
        file_name=filename,
        file_path=str(mp3_path),
        duration_seconds=duration,
        generated_at=datetime.now(timezone.utc),
        chunk_count=len(plans),
    )
    db.update_podcast_audio_metadata(script_id, metadata)
    on_stage("tts_save", StageStatus.COMPLETED.value, "Saved")

    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except OSError:
        logger.warning("Could not clean up TTS work dir %s", work_dir)


def _assemble_with_variable_pauses(
    chunk_wavs: list[Path],
    pause_ms_after: list[int],
    output_wav: Path,
) -> None:
    """Concatenate chunk WAVs, inserting per-chunk trailing silence."""
    from array import array

    from nlp_pillars.tts.audio_assembler import SAMPLE_RATE, SAMPLE_WIDTH, _read_wav_pcm, _silence_samples

    pcm: array = array("h")
    for index, wav in enumerate(chunk_wavs):
        pcm.extend(_read_wav_pcm(wav))
        pause = pause_ms_after[index] if index < len(pause_ms_after) else 0
        if pause > 0:
            pcm.extend(_silence_samples(pause))

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    import wave

    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
