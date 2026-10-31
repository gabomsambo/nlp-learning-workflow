"""Background worker for podcast script → MP3 via IndexTTS."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from nlp_pillars import db
from nlp_pillars.config import get_settings
from nlp_pillars.schemas import AudioMetadata, StageStatus
from nlp_pillars.orchestrator import RunCancelledError
from nlp_pillars.tts.alignment import AlignmentError
from nlp_pillars.tts.aligner import AlignerError
from nlp_pillars.tts.audio_assembler import encode_mp3, wav_duration_seconds
from nlp_pillars.tts.cue_parser import parse_podcast_script, plan_synthesis_chunks
from nlp_pillars.tts.engine import TtsEngineStatus, ChunkSynthesisError
from nlp_pillars.tts.indextts_client import (
    IndexTtsClient,
    contract_revision,
    synthesis_parameters,
)
from nlp_pillars.tts.text_prep import (
    DEFAULT_MAX_TOKENS_PER_SEGMENT,
    prepare_text_for_indextts,
)
from nlp_pillars.tts.timeline import compute_timeline
from nlp_pillars.tts.voice_library import (
    VoiceUsability,
    extract_preview_clip,
    resolve_voice_path,
    scan_voice_library,
)

logger = logging.getLogger(__name__)

CHUNK_CHECKPOINT_TTL_SECONDS = 24 * 60 * 60

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


def get_tts_client(per_run_download_dir: Optional[Path] = None) -> IndexTtsClient:
    base_url, _, _, settings_download_dir, _, start_command = _settings_paths()
    return IndexTtsClient(
        base_url,
        (
            Path(per_run_download_dir)
            if per_run_download_dir is not None
            else settings_download_dir
        ),
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_fingerprint(
    *,
    text: str,
    script_sha256: str,
    voice_sha256: str,
    engine_revision: str,
    params: Dict[str, Any],
) -> str:
    payload = {
        "text": prepare_text_for_indextts(text),
        "script_sha256": script_sha256,
        "voice_sha256": voice_sha256,
        "engine_revision": engine_revision,
        "synthesis_parameters": params,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _prune_chunk_checkpoints(checkpoint_dir: Path) -> None:
    """Remove completed chunk files outside the bounded retry window."""
    if not checkpoint_dir.is_dir():
        return
    cutoff = time.time() - CHUNK_CHECKPOINT_TTL_SECONDS
    for path in checkpoint_dir.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError as exc:
            logger.warning("Could not prune TTS checkpoint %s: %s", path, exc)


def _prune_abandoned_run_dirs(download_root: Path) -> None:
    """Bound leftover download dirs from a process that died before its finally."""
    cutoff = time.time() - CHUNK_CHECKPOINT_TTL_SECONDS
    for path in download_root.glob("run-*"):
        try:
            if path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError as exc:
            logger.warning("Could not prune abandoned TTS run directory %s: %s", path, exc)


def _valid_checkpoint(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return wav_duration_seconds(path) > 0
    except Exception:
        return False


def _store_chunk_checkpoint(source_path: str, checkpoint_path: Path, run_id: str) -> None:
    """Install a complete WAV atomically so a killed worker cannot cache a partial."""
    source = Path(source_path)
    if not source.is_file() or wav_duration_seconds(source) <= 0:
        raise ValueError(f"IndexTTS returned an invalid WAV file: {source_path}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_name(
        f".{checkpoint_path.name}.{run_id}.tmp"
    )
    try:
        shutil.copyfile(source, temp_path)
        if wav_duration_seconds(temp_path) <= 0:
            raise ValueError("checkpoint copy is not a valid WAV file")
        temp_path.replace(checkpoint_path)
    finally:
        temp_path.unlink(missing_ok=True)


def run_podcast_audio_job(
    run_id: str,
    script_id: str,
    voice_relative_path: str,
    on_stage: StageCallback,
    cancel: threading.Event,
    expected_script_sha256: Optional[str] = None,
) -> None:
    """Synthesize one script, resuming matching WAV chunks after a failed attempt."""
    if cancel.is_set():
        raise RunCancelledError("cancelled before start")

    script = db.get_podcast_script_by_id(script_id)
    if not script:
        raise ValueError(f"Script not found: {script_id}")

    voice_source = validate_voice_for_generation(voice_relative_path)
    actual_script_sha256 = hashlib.sha256(script.script.encode("utf-8")).hexdigest()
    if expected_script_sha256 and actual_script_sha256 != expected_script_sha256:
        raise ValueError(
            "Saved script changed after this audio request was queued; "
            "request a new audio run."
        )
    voice_sha256 = _sha256_file(voice_source)
    params = synthesis_parameters(DEFAULT_MAX_TOKENS_PER_SEGMENT)
    engine_revision = contract_revision(DEFAULT_MAX_TOKENS_PER_SEGMENT)

    on_stage("tts_prepare", StageStatus.RUNNING.value, "Parsing script and planning chunks")
    parsed = parse_podcast_script(script.script)
    plans = plan_synthesis_chunks(parsed)
    if not plans:
        on_stage(
            "tts_prepare",
            StageStatus.FAILED.value,
            "Script has no speakable content after cue stripping",
        )
        raise ValueError("Script has no speakable content after cue stripping")
    on_stage(
        "tts_prepare",
        StageStatus.COMPLETED.value,
        f"{len(plans)} chunk(s) planned",
    )

    if cancel.is_set():
        raise RunCancelledError("cancelled")

    _, _, audio_dir, download_root, _, _ = _settings_paths()
    download_root = Path(download_root)
    work_dir = download_root / f"run-{run_id}"
    per_run_download_dir = work_dir / "downloads"
    checkpoint_dir = download_root / "checkpoints"

    chunk_paths: list[Path] = []
    chunk_fingerprints: list[str] = []
    used_checkpoints: set[Path] = set()
    mp3_path: Optional[Path] = None
    metadata_saved = False
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _prune_abandoned_run_dirs(download_root)
        _prune_chunk_checkpoints(checkpoint_dir)
        client = get_tts_client(per_run_download_dir)
        try:
            status = client.check_status()
        except Exception as exc:
            detail = f"Could not verify IndexTTS contract: {exc}"
            on_stage("tts_synthesize", StageStatus.FAILED.value, detail)
            raise RuntimeError(detail) from exc
        if status.status != TtsEngineStatus.READY:
            on_stage("tts_synthesize", StageStatus.FAILED.value, status.message)
            raise RuntimeError(status.message)

        on_stage("tts_synthesize", StageStatus.RUNNING.value, f"chunk 0/{len(plans)}")
        for index, plan in enumerate(plans, start=1):
            if cancel.is_set():
                raise RunCancelledError("cancelled")

            fingerprint = _chunk_fingerprint(
                text=plan.text,
                script_sha256=actual_script_sha256,
                voice_sha256=voice_sha256,
                engine_revision=engine_revision,
                params=params,
            )
            checkpoint_path = checkpoint_dir / f"chunk-{fingerprint}.wav"
            chunk_fingerprints.append(fingerprint)
            if _valid_checkpoint(checkpoint_path):
                chunk_paths.append(checkpoint_path)
                used_checkpoints.add(checkpoint_path)
                on_stage(
                    "tts_synthesize",
                    StageStatus.RUNNING.value,
                    f"chunk {index}/{len(plans)} reused from checkpoint",
                )
                continue
            def _progress_cb(progress, idx=index, total=len(plans)) -> None:
                on_stage(
                    "tts_synthesize",
                    StageStatus.RUNNING.value,
                    f"chunk {idx}/{total} — {progress.description}",
                )

            try:
                if checkpoint_path.exists():
                    checkpoint_path.unlink(missing_ok=True)
                wav_path = client.synthesize(
                    plan.text,
                    str(voice_source),
                    on_progress=_progress_cb,
                    max_text_tokens_per_segment=DEFAULT_MAX_TOKENS_PER_SEGMENT,
                )
                _store_chunk_checkpoint(wav_path, checkpoint_path, run_id)
            except Exception as exc:
                reason = str(exc).strip() or type(exc).__name__
                detail = f"chunk {index}/{len(plans)} failed: {reason}"
                on_stage("tts_synthesize", StageStatus.FAILED.value, detail)
                raise ChunkSynthesisError(index, len(plans), reason) from exc

            chunk_paths.append(checkpoint_path)
            used_checkpoints.add(checkpoint_path)
            on_stage(
                "tts_synthesize",
                StageStatus.RUNNING.value,
                f"chunk {index}/{len(plans)} complete",
            )

        on_stage(
            "tts_synthesize",
            StageStatus.COMPLETED.value,
            f"{len(plans)} chunk(s) synthesized or resumed",
        )

        if cancel.is_set():
            raise RunCancelledError("cancelled")

        on_stage("tts_assemble", StageStatus.RUNNING.value, "Joining audio and silences")
        assembled_wav = work_dir / "assembled.wav"
        _assemble_with_variable_pauses(
            chunk_paths,
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

        # Read-along timing is non-fatal: a recogniser problem costs highlighting,
        # not the recording. The stage carries the reason and metadata keeps the
        # successful synthesis.
        on_stage("tts_align", StageStatus.RUNNING.value, "Timing every word for read-along")
        timeline: Optional[dict] = None
        try:
            timeline = compute_timeline(script.script, mp3_path)
            detail = (
                f"{timeline['word_count']} words, {timeline['matched_ratio']:.0%} heard"
                if timeline["word_level"]
                else f"Line timing only, {timeline['matched_ratio']:.0%} of words heard"
            )
            on_stage("tts_align", StageStatus.COMPLETED.value, detail)
        except (AlignerError, AlignmentError) as exc:
            reason = str(exc).strip() or type(exc).__name__
            logger.warning("Read-along timing failed for script %s: %s", script_id, reason)
            on_stage(
                "tts_align",
                StageStatus.FAILED.value,
                f"No read-along timing: {reason} The audio is saved; "
                "`python -m nlp_pillars.cli timeline backfill` retries it.",
            )
        except Exception as exc:  # noqa: BLE001 - the audio must still be saved
            logger.exception("Read-along timing crashed for script %s", script_id)
            on_stage(
                "tts_align",
                StageStatus.FAILED.value,
                f"No read-along timing: {type(exc).__name__}: {exc}. The audio is saved.",
            )

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
            script_sha256=actual_script_sha256,
            voice_sha256=voice_sha256,
            engine_revision=engine_revision,
            synthesis_parameters=params,
            chunk_fingerprints=chunk_fingerprints,
            timeline=timeline,
        )
        db.update_podcast_audio_metadata(script_id, metadata)
        metadata_saved = True
        on_stage("tts_save", StageStatus.COMPLETED.value, "Saved")
    except BaseException:
        if mp3_path is not None and mp3_path.exists() and not metadata_saved:
            mp3_path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        if metadata_saved:
            for checkpoint in used_checkpoints:
                checkpoint.unlink(missing_ok=True)
            try:
                checkpoint_dir.rmdir()
            except OSError:
                pass


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
