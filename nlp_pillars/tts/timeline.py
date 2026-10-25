"""Read-along timing for a recorded episode: when it is needed and how it is made.

``audio_metadata.timeline`` is computed from the finished MP3 and the script
text it was recorded from (``alignment.py``), after the recording is saved and
by ``python -m nlp_pillars.cli timeline backfill`` for episodes recorded before
this existed. A failure here never takes the audio with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence

from nlp_pillars.factcheck.engine import sha256_text
from nlp_pillars.tts import aligner
from nlp_pillars.tts.alignment import TIMELINE_VERSION, build_timeline


def compute_timeline(script_text: str, audio_path: Path) -> dict:
    """Time ``script_text`` against the MP3 at ``audio_path``.

    Raises ``AlignerError`` when the recogniser cannot run and
    ``AlignmentError`` when the audio does not match the script.
    """
    words, duration_s = aligner.transcribe_words(audio_path)
    return build_timeline(
        script_text,
        words,
        duration_s,
        engine=aligner.ENGINE_ID,
        script_sha256=sha256_text(script_text),
    )


def timeline_state(timeline: Optional[Mapping[str, Any]], script_text: str) -> str:
    """``current``, ``stale`` (the script changed since) or ``missing``."""
    if not timeline or timeline.get("version") != TIMELINE_VERSION:
        return "missing"
    if timeline.get("script_sha256") != sha256_text(script_text):
        return "stale"
    return "current"


def audio_file(file_name: Optional[str], file_path: Optional[str], audio_dir: Path) -> Optional[Path]:
    """Where an episode's MP3 is on disk, or None when the metadata names none.

    ``file_name`` under the configured audio directory wins over the recorded
    absolute path: that path is the one the recording container saw, and the
    same volume can be mounted somewhere else.
    """
    if file_name:
        return Path(audio_dir) / Path(file_name).name
    return Path(file_path) if file_path else None


@dataclass
class BackfillRow:
    """What the backfill did, or would do, for one episode."""

    script_id: str
    title: str
    state: str  # current | stale | missing | no_audio | no_file
    action: str  # skip | align | aligned | failed
    detail: str = ""


def backfill_timelines(
    scripts: Sequence[Any],
    audio_dir: Path,
    *,
    dry_run: bool,
    force: bool = False,
    compute: Callable[[str, Path], dict] = compute_timeline,
    save: Optional[Callable[[str, Any], None]] = None,
) -> List[BackfillRow]:
    """Time every episode that has audio and no current timeline.

    ``dry_run`` only reads: it loads no model and writes nothing. ``force``
    re-times episodes whose timeline is already current. A failure on one
    episode is reported on its row and the rest still run.
    """
    rows: List[BackfillRow] = []
    for script in scripts:
        meta = script.audio_metadata
        script_id = str(script.id)
        title = script.title or script_id
        if not meta or not meta.file_name:
            rows.append(BackfillRow(script_id, title, "no_audio", "skip", "no recorded audio"))
            continue
        path = audio_file(meta.file_name, meta.file_path, audio_dir)
        state = timeline_state(meta.timeline, script.script)
        if path is None or not path.is_file():
            rows.append(BackfillRow(script_id, title, "no_file", "skip", f"{meta.file_name} is not on disk"))
            continue
        if state == "current" and not force:
            rows.append(BackfillRow(script_id, title, state, "skip", "timeline is current"))
            continue
        if dry_run:
            rows.append(BackfillRow(script_id, title, state, "align", f"would time {path.name}"))
            continue
        try:
            meta.timeline = compute(script.script, path)
            (save or _save_metadata)(script_id, meta)
        except Exception as exc:  # noqa: BLE001 - one bad episode must not stop the rest
            rows.append(BackfillRow(script_id, title, state, "failed", str(exc).strip() or type(exc).__name__))
            continue
        tl = meta.timeline
        rows.append(
            BackfillRow(
                script_id,
                title,
                state,
                "aligned",
                f"{tl['word_count']} words, {tl['matched_ratio']:.1%} heard"
                + ("" if tl["word_level"] else ", line timing only"),
            )
        )
    return rows


def _save_metadata(script_id: str, meta: Any) -> None:
    from nlp_pillars import db

    db.update_podcast_audio_metadata(script_id, meta)
