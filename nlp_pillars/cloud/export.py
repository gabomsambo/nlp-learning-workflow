"""``cloud export``: write the offline bundle the cloud reader serves.

The bundle is a directory::

    manifest.json              written last; lists every other file with its hash
    library.json               pillars, paper and episode index, concepts
    schedule.json              one scheduling snapshot per quiz card
    papers/<name>.json         paper, note, lesson, quiz cards, concept links
    episodes/<name>.json       script, source warnings, fact check, timeline, follow-along
    playgrounds/<id>.json      study steps, cells and saved outputs, read-only
    audio/<sha256>.mp3         64 kbps mono copy, named by the hash of its bytes

Nothing here talks to Cloudflare. A document is built field by field from an
explicit list, never by copying a row, and every document is checked once more
before it is written (``_guard``). Output is canonical JSON, so the same
database produces the same bytes and a second run changes nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .. import fsrs_fold
from ..playground import notebook as nb_view
from ..services import quiz_reviews
from . import audio as audio_tools
from .source import CloudSource, load_source

BUNDLE_VERSION = 1
# Default FSRS-6 weights; bump when user_fsrs_parameters ever feeds the scheduler.
FSRS_PARAMS_VERSION = 1
TOMBSTONE_DAYS = 30
MANAGED_DIRS = ("papers", "episodes", "playgrounds", "audio")
# The Worker refuses a larger document (cloud/reader/src/ingest.js MAX_DOC_BYTES).
MAX_DOC_BYTES = 8 * 1024 * 1024
# Per playground, what of its saved outputs travels (CLOUD_PLAYGROUND_MAX_*):
# base64 figure data, and every other output as encoded JSON. What does not fit
# is left out in notebook order and marked where it was.
DEFAULT_PLAYGROUND_MAX_IMAGE_BYTES = 3_000_000
DEFAULT_PLAYGROUND_MAX_OUTPUT_BYTES = 1_500_000
_TEMP_PREFIXES = (".transcode-", ".pending-")

# ---------------------------------------------------------------------------
# The allow-list. Adding a field to the reader means adding it here.
# ---------------------------------------------------------------------------

PILLAR_FIELDS = ("id", "name", "goal", "focus_areas")
PAPER_FIELDS = (
    "id", "pillar_id", "title", "authors", "venue", "year", "abstract",
    "citation_count", "added_at",
)
NOTE_FIELDS = (
    "id", "problem", "method", "findings", "limitations", "future_work",
    "key_terms", "related_papers", "confidence_score", "created_at",
)
LESSON_FIELDS = (
    "id", "tl_dr", "takeaways", "practice_ideas", "connections", "difficulty",
    "estimated_time", "created_at",
)
CARD_FIELDS = ("id", "question", "answer", "difficulty", "question_type")
SOURCE_MATERIAL_FIELDS = (
    "level", "full_text_chars", "has_abstract", "has_notes", "warnings",
)
CLAIM_FIELDS = (
    "id", "line_index", "span_start", "span_end", "span_text", "claim", "scope",
    "category", "verify_marker", "verdict", "error_type", "severity", "rationale",
    "source_says", "evidence_status", "dismissed_at",
)
EVIDENCE_FIELDS = ("quote", "chunk_index")
FACT_CHECK_FIELDS = ("id", "script_sha256", "status", "counts", "prompt_version", "created_at")
TIMELINE_FIELDS = (
    "version", "duration_ms", "word_count", "word_level", "matched_ratio",
    "script_sha256",
)
TIMELINE_LINE_FIELDS = ("line", "start_ms", "end_ms", "words")
CONCEPT_FIELDS = ("id", "name", "aliases", "description")
PLAYGROUND_FIELDS = ("id", "paper_id", "title", "status", "source_kind")
STEP_FIELDS = ("id", "title", "goal")
GUIDE_FIELDS = ("cell_index", "explanation", "quiz", "quiz_answer")
LINE_NOTE_FIELDS = ("line", "note")
TRY_FIELDS = ("label", "code", "expect")
SPAN_FIELDS = ("line_start", "line_end", "cell_index", "step_id")

# Keys that must never appear anywhere in a document, at any depth.
FORBIDDEN_KEYS = frozenset({
    "ground_pack", "ground_pack_calls", "options", "file_path", "voice_path",
    "voice_label", "chunk_id", "chunks", "text", "embedding", "embeddings",
    "judge_trail", "dismissed_note", "models", "cost_usd", "user_id",
})
_SECRET_KEY = re.compile(
    r"(^|_)(api[_-]?key|secret|token|password|passwd|authorization|credential|bearer)s?($|_)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"sk-ant-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9]{32,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."
    r"|file:///app/|/app/data/|BEGIN [A-Z ]*PRIVATE KEY"
)
_SECRET_ENV_NAME = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)


class ExportError(Exception):
    """The bundle could not be built. Nothing is written, the manifest stays."""


# ---------------------------------------------------------------------------
# Canonical JSON, names, hashes
# ---------------------------------------------------------------------------


def canonical_bytes(value: Any) -> bytes:
    """Sorted keys, no whitespace, UTF-8, one trailing newline."""
    try:
        text = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ExportError(f"Not JSON-serialisable: {exc}") from exc
    return (text + "\n").encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_NAME_KEEP = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"
)


def safe_name(identifier: str) -> str:
    """A file and URL safe form of an id. Injective: ``_`` starts every escape.

    ``arxiv:1706.03762`` is ``arxiv_3a1706.03762``; ``cs/0301`` is ``cs_2f0301``.
    """
    out = []
    for char in str(identifier):
        if char in _NAME_KEEP:
            out.append(char)
        else:
            out.extend(f"_{byte:02x}" for byte in char.encode("utf-8"))
    name = "".join(out)
    if not name:
        raise ExportError("An empty id cannot name a document")
    return name


def paper_key(paper_id: str) -> str:
    return f"papers/{safe_name(paper_id)}.json"


def episode_key(script_id: str) -> str:
    return f"episodes/{safe_name(script_id)}.json"


def audio_key(sha256: str) -> str:
    return f"audio/{sha256}.mp3"


def playground_key(playground_id: str) -> str:
    return f"playgrounds/{safe_name(playground_id)}.json"


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def _pick(row: Mapping[str, Any], fields: Iterable[str]) -> Dict[str, Any]:
    return {name: row.get(name) for name in fields}


def public_pdf_url(url: Optional[str]) -> Optional[str]:
    """Web links only. An upload's ``file:///app/data/uploads/..`` is the only copy."""
    if isinstance(url, str) and url.lower().startswith(("http://", "https://")):
        return url
    return None


def serialize_pillar(row: Mapping[str, Any]) -> Dict[str, Any]:
    return _pick(row, PILLAR_FIELDS)


def serialize_note(row: Mapping[str, Any]) -> Dict[str, Any]:
    return _pick(row, NOTE_FIELDS)


def serialize_lesson(row: Mapping[str, Any]) -> Dict[str, Any]:
    return _pick(row, LESSON_FIELDS)


def serialize_card(row: Mapping[str, Any]) -> Dict[str, Any]:
    return _pick(row, CARD_FIELDS)


def serialize_schedule_card(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The card row's scheduling cache. ``due`` is null until the first review."""
    return {
        "card_id": row.get("id"),
        "paper_id": row.get("paper_id"),
        "state": row.get("state") or "new",
        "due": row.get("next_review_date"),
        "stability": row.get("stability"),
        "difficulty": row.get("difficulty_fsrs"),
        "last_review": row.get("last_review_date"),
        "lapses": row.get("lapses") or 0,
    }


def schedule_cards(source: CloudSource) -> List[Dict[str, Any]]:
    """One scheduling snapshot per quiz card, folded from its review events.

    A card with events is the py-fsrs card the fold produces: ``state``, ``due``,
    ``stability``, ``difficulty`` and ``step`` are everything a phone needs to carry
    on from it with ts-fsrs. A card with no events is ``new``. A card that has a
    schedule in its row and no events (the cache is all that is left of it) is
    published as its row says, without a ``step``.
    """
    by_card: Dict[str, List[Mapping[str, Any]]] = {}
    for row in source.review_events:
        by_card.setdefault(str(row["card_id"]), []).append(row)
    out = []
    for row in sorted(source.cards, key=lambda r: str(r["id"])):
        folded = fsrs_fold.fold_events(quiz_reviews._events(by_card.get(str(row["id"]), [])))
        if folded is None:
            entry = serialize_schedule_card(row)
            entry.update({"step": None, "reps": 0 if entry["state"] == "new" else None})
        else:
            card = folded.card
            entry = {
                "card_id": row.get("id"),
                "paper_id": row.get("paper_id"),
                "state": fsrs_fold.cache_fields(folded)["state"],
                "due": card.due.astimezone(timezone.utc).isoformat(),
                "stability": card.stability,
                "difficulty": card.difficulty,
                "step": card.step,
                "last_review": card.last_review.astimezone(timezone.utc).isoformat(),
                "reps": folded.review_count,
                "lapses": folded.lapses,
            }
        out.append(entry)
    return out


def serialize_source_material(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    return _pick(value, SOURCE_MATERIAL_FIELDS)


def serialize_timeline(value: Any) -> Optional[Dict[str, Any]]:
    """The read-along timeline, or None for an episode that has none."""
    if not isinstance(value, Mapping) or not isinstance(value.get("lines"), list):
        return None
    lines = [
        {k: line[k] for k in TIMELINE_LINE_FIELDS if k in line}
        for line in value["lines"]
        if isinstance(line, Mapping)
    ]
    timeline = _pick(value, TIMELINE_FIELDS)
    timeline["lines"] = lines
    return timeline


def serialize_claim(row: Mapping[str, Any]) -> Dict[str, Any]:
    claim = _pick(row, CLAIM_FIELDS)
    claim["evidence"] = [
        _pick(item, EVIDENCE_FIELDS)
        for item in (row.get("evidence") or [])
        if isinstance(item, Mapping)
    ]
    return claim


def serialize_fact_check(
    check: Mapping[str, Any], claims: List[Mapping[str, Any]], script: str,
) -> Dict[str, Any]:
    out = _pick(check, FACT_CHECK_FIELDS)
    # A check grades the exact script text it was run on; the reader needs to
    # know whether that text is still the one it shows.
    out["stale"] = check.get("script_sha256") != sha256_hex(script.encode("utf-8"))
    out["claims"] = [
        serialize_claim(row)
        for row in sorted(
            claims,
            key=lambda r: (r.get("line_index") or 0, r.get("span_start") or 0, str(r.get("id"))),
        )
    ]
    return out


def serialize_concept(row: Mapping[str, Any], paper_ids: List[str]) -> Dict[str, Any]:
    out = _pick(row, CONCEPT_FIELDS)
    out["paper_ids"] = paper_ids
    return out


# ---------------------------------------------------------------------------
# Playgrounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaygroundCaps:
    """Saved output kept per playground; see DEFAULT_PLAYGROUND_MAX_*."""

    image_bytes: int = DEFAULT_PLAYGROUND_MAX_IMAGE_BYTES
    output_bytes: int = DEFAULT_PLAYGROUND_MAX_OUTPUT_BYTES

    @classmethod
    def from_settings(cls, settings: Any) -> "PlaygroundCaps":
        return cls(
            image_bytes=int(getattr(settings, "cloud_playground_max_image_bytes", DEFAULT_PLAYGROUND_MAX_IMAGE_BYTES)),
            output_bytes=int(getattr(settings, "cloud_playground_max_output_bytes", DEFAULT_PLAYGROUND_MAX_OUTPUT_BYTES)),
        )


def serialize_steps(steps: Any) -> List[Dict[str, Any]]:
    """The study guide, field by field (docs/PLAYGROUND.md, "Stored shape")."""
    out = []
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, Mapping):
            continue
        cells = []
        for cell in step.get("cells") or []:
            if not isinstance(cell, Mapping) or not isinstance(cell.get("cell_index"), int):
                continue
            guide = _pick(cell, GUIDE_FIELDS)
            guide["line_notes"] = [
                _pick(n, LINE_NOTE_FIELDS) for n in cell.get("line_notes") or [] if isinstance(n, Mapping)
            ]
            guide["tries"] = [
                _pick(t, TRY_FIELDS) for t in cell.get("tries") or [] if isinstance(t, Mapping)
            ]
            cells.append(guide)
        entry = _pick(step, STEP_FIELDS)
        entry["cells"] = cells
        out.append(entry)
    return out


def serialize_output(view: Mapping[str, Any]) -> Dict[str, Any]:
    """One output of ``notebook.cells_view``, already sanitized. Text is ``body``:
    ``text`` is a forbidden key, kept for the paper's own text."""
    kind = view.get("kind")
    if kind in ("stream", "text", "error"):
        out: Dict[str, Any] = {"kind": kind, "body": view.get("text") or ""}
        if kind == "stream":
            out["name"] = view.get("name") or "stdout"
    elif kind == "html":
        out = {"kind": "html", "tree": view.get("tree") or []}
    elif kind == "png":
        return {"kind": "png", "data": view.get("data"), "alt": view.get("alt")}
    else:
        return {"kind": "unsupported", "mime": [str(m) for m in view.get("mime") or []]}
    if view.get("truncated"):
        out["truncated"] = True
    return out


class _Budget:
    """What is left of one playground's caps, spent in notebook order."""

    def __init__(self, caps: PlaygroundCaps) -> None:
        self.image_left = caps.image_bytes
        self.output_left = caps.output_bytes
        self.figures_dropped = 0
        self.outputs_dropped = 0

    def keep(self, out: Dict[str, Any]) -> Dict[str, Any]:
        if out["kind"] == "png":
            size = len(out.get("data") or "")
            if size <= self.image_left:
                self.image_left -= size
                return out
            self.figures_dropped += 1
            return {"kind": "dropped", "what": "figure", "bytes": size}
        size = len(canonical_bytes(out))
        if size <= self.output_left:
            self.output_left -= size
            return out
        self.outputs_dropped += 1
        return {"kind": "dropped", "what": "output", "bytes": size}


def serialize_playground(row: Mapping[str, Any], caps: PlaygroundCaps) -> Dict[str, Any]:
    """A playground as the reader shows it: no notebook JSON, no datasets, no model.

    Cells come from the same ``cells_view`` the Playground tab renders, so HTML
    output is the sanitized tag tree and only PNG figures survive.
    """
    notebook = row.get("notebook") if isinstance(row.get("notebook"), Mapping) else {}
    budget = _Budget(caps)
    cells = []
    for view in nb_view.cells_view(dict(notebook)):
        cell: Dict[str, Any] = {"index": view["index"], "type": view["type"], "source": view["source"]}
        if view["type"] == "code":
            cell["execution_count"] = view.get("execution_count")
            cell["outputs"] = [budget.keep(serialize_output(o)) for o in view.get("outputs") or []]
        cells.append(cell)
    doc = _pick(row, PLAYGROUND_FIELDS)
    doc.update({
        "steps": serialize_steps(row.get("steps")),
        "cells": cells,
        "limits": {"image_bytes": caps.image_bytes, "output_bytes": caps.output_bytes},
        "dropped": {"figures": budget.figures_dropped, "outputs": budget.outputs_dropped},
    })
    return doc


def build_playgrounds(
    source: CloudSource,
    caps: PlaygroundCaps,
    secrets: Iterable[str] = (),
    warnings: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Each publishable playground's document, keyed by playground id.

    A notebook is content someone uploaded, so one that trips the guard or will
    not fit in a document is left out (and named in ``warnings``) instead of
    stopping the whole publish. Only playgrounds of published papers are kept.
    """
    secrets = list(secrets)
    paper_ids = {str(p["id"]) for p in source.papers}
    out: Dict[str, Dict[str, Any]] = {}
    for row in sorted(source.playgrounds, key=lambda r: str(r["id"])):
        pid = str(row["id"])
        if str(row.get("paper_id")) not in paper_ids or row.get("status") not in ("ready", "partial"):
            continue
        doc = serialize_playground(row, caps)
        try:
            _guard(playground_key(pid), doc, secrets)
            size = len(canonical_bytes(doc))
        except ExportError as exc:
            if warnings is not None:
                warnings.append(f"playground {pid} not published: {exc}")
            continue
        if size > MAX_DOC_BYTES:
            if warnings is not None:
                warnings.append(
                    f"playground {pid} not published: {size / 1_000_000:.1f} MB is over the "
                    f"{MAX_DOC_BYTES / 1_000_000:.1f} MB document limit even with its outputs capped"
                )
            continue
        if warnings is not None and (doc["dropped"]["figures"] or doc["dropped"]["outputs"]):
            warnings.append(
                f"playground {pid}: {doc['dropped']['figures']} figure(s) and "
                f"{doc['dropped']['outputs']} other output(s) left out by CLOUD_PLAYGROUND_MAX_*"
            )
        out[pid] = doc
    return out


def playground_summary(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """What the paper page needs before the playground itself is fetched."""
    return {
        "id": doc["id"], "doc": playground_key(str(doc["id"])), "title": doc["title"],
        "status": doc["status"], "step_count": len(doc["steps"]),
        "cell_count": sum(1 for c in doc["cells"] if c["type"] == "code"),
    }


def serialize_follow_along(
    links: List[Mapping[str, Any]],
    playgrounds: Mapping[str, Mapping[str, Any]],
    script: str,
) -> List[Dict[str, Any]]:
    """An episode's ready links to published playgrounds, freshest first.

    ``stale`` says the mapping indexes another version of the script, as the
    desktop panel decides it (``FollowAlong.chooseLink``).
    """
    sha = sha256_hex(script.encode("utf-8"))
    rows = [r for r in links if r.get("status") == "ready" and str(r.get("playground_id")) in playgrounds]
    rows.sort(key=lambda r: str(r["id"]))
    rows.sort(key=lambda r: str(r.get("updated_at") or ""), reverse=True)
    out = []
    for row in rows:
        pg = playgrounds[str(row["playground_id"])]
        out.append({
            "playground_id": pg["id"],
            "doc": playground_key(str(pg["id"])),
            "paper_id": pg["paper_id"],
            "title": pg["title"],
            "status": "ready",
            "script_sha256": row.get("script_sha256"),
            "stale": bool(row.get("script_sha256")) and row.get("script_sha256") != sha,
            "mapping": [
                _pick(span, SPAN_FIELDS)
                for span in (row.get("mapping") if isinstance(row.get("mapping"), list) else [])
                if isinstance(span, Mapping)
            ],
        })
    out.sort(key=lambda link: link["stale"])
    return out


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


@dataclass
class AudioEntry:
    """One cloud MP3, as listed in the manifest."""

    key: str
    sha256: str
    bytes: int
    duration_seconds: float
    source_sha256: str
    source_bytes: int
    bitrate_kbps: int = audio_tools.BITRATE_KBPS

    def manifest(self) -> Dict[str, Any]:
        return {
            "key": self.key, "sha256": self.sha256, "bytes": self.bytes,
            "duration_seconds": self.duration_seconds,
            "bitrate_kbps": self.bitrate_kbps,
            "source_sha256": self.source_sha256, "source_bytes": self.source_bytes,
        }


@dataclass
class AudioPlan:
    """What happens to one episode's MP3."""

    script_id: str
    file_name: str
    entry: Optional[AudioEntry] = None  # set once the cloud copy exists
    status: str = "pending"  # pending | reused | transcoded | missing
    reason: str = ""


Transcoder = Callable[[Path, Path], float]


def _episode_file_name(meta: Any) -> Optional[str]:
    if not isinstance(meta, Mapping):
        return None
    name = meta.get("file_name")
    if not isinstance(name, str) or not name:
        return None
    if name != Path(name).name or name in {".", ".."}:
        raise ExportError(f"audio file name {name!r} is not a bare file name")
    return name


def plan_audio(
    scripts: List[Mapping[str, Any]],
    audio_dir: Path,
    out: Path,
    previous: Optional[Mapping[str, Any]],
    *,
    transcode: bool,
    transcoder: Transcoder,
) -> Dict[str, AudioPlan]:
    """Find, reuse or transcode each episode's MP3, keyed by script id.

    A cached copy is reused when the manifest already records the original's
    hash, the file is still there and its bytes still hash to the recorded
    value. Otherwise, with ``transcode`` set, the copy is made now.
    """
    cached: Dict[Tuple[Any, Any], Mapping[str, Any]] = {}
    if previous:
        for item in previous.get("audio", []):
            cached[(item.get("source_sha256"), item.get("bitrate_kbps"))] = item
    plans: Dict[str, AudioPlan] = {}
    for script in scripts:
        file_name = _episode_file_name(script.get("audio_metadata"))
        if not file_name:
            continue
        plan = AudioPlan(script_id=str(script["id"]), file_name=file_name)
        plans[plan.script_id] = plan
        source = audio_dir / file_name
        if not source.is_file():
            plan.status = "missing"
            plan.reason = f"{file_name} is not in {audio_dir}"
            continue
        source_sha = audio_tools.sha256_file(source)
        hit = cached.get((source_sha, audio_tools.BITRATE_KBPS))
        if hit is not None:
            held = out / hit["key"]
            if held.is_file() and audio_tools.sha256_file(held) == hit["sha256"]:
                plan.entry = AudioEntry(
                    key=hit["key"], sha256=hit["sha256"], bytes=held.stat().st_size,
                    duration_seconds=hit["duration_seconds"], source_sha256=source_sha,
                    source_bytes=source.stat().st_size,
                )
                plan.status = "reused"
                continue
        if not transcode:
            plan.reason = "would be transcoded"
            continue
        pending = out / "audio" / f"{_TEMP_PREFIXES[1]}{safe_name(plan.script_id)}.mp3"
        try:
            duration = transcoder(source, pending)
            digest = audio_tools.sha256_file(pending)
            final = out / audio_key(digest)
            pending.replace(final)
        except audio_tools.AudioError as exc:
            raise ExportError(f"Episode {plan.script_id}: {exc}") from exc
        finally:
            pending.unlink(missing_ok=True)
        plan.entry = AudioEntry(
            key=audio_key(digest), sha256=digest, bytes=final.stat().st_size,
            duration_seconds=round(duration, 3), source_sha256=source_sha,
            source_bytes=source.stat().st_size,
        )
        plan.status = "transcoded"
    return plans


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def _latest(rows: List[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not rows:
        return None
    return max(rows, key=lambda r: (str(r.get("created_at") or ""), str(r.get("id"))))


def _group(rows: Iterable[Mapping[str, Any]], key: str) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key)), []).append(row)
    return grouped


def serialize_episode(
    script: Mapping[str, Any],
    check: Optional[Mapping[str, Any]],
    claims: List[Mapping[str, Any]],
    plan: Optional[AudioPlan],
    follow_along: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    meta = script.get("audio_metadata") if isinstance(script.get("audio_metadata"), Mapping) else {}
    entry = plan.entry if plan else None
    audio = None
    if entry:
        audio = {
            "key": entry.key, "sha256": entry.sha256, "bytes": entry.bytes,
            "duration_seconds": entry.duration_seconds,
            "bitrate_kbps": entry.bitrate_kbps,
            "engine": meta.get("engine"), "generated_at": meta.get("generated_at"),
        }
    text = script.get("script") or ""
    return {
        "id": script.get("id"),
        "paper_id": script.get("paper_id"),
        "pillar_id": script.get("pillar_id"),
        "title": script.get("title"),
        "script": text,
        "word_count": script.get("word_count"),
        "key_points": script.get("key_points") or [],
        "source_material": serialize_source_material(script.get("source_material")),
        "created_at": script.get("created_at"),
        "audio": audio,
        "timeline": serialize_timeline(meta.get("timeline")) if entry else None,
        "fact_check": serialize_fact_check(check, claims, text) if check else None,
        "follow_along": follow_along or [],
    }


def build_documents(
    source: CloudSource,
    plans: Mapping[str, AudioPlan],
    playgrounds: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Every JSON document, keyed by its path in the bundle. Pure.

    ``playgrounds`` is ``build_playgrounds``'s result; without it they are built
    here with the default caps.
    """
    if playgrounds is None:
        playgrounds = build_playgrounds(source, PlaygroundCaps())
    playground_by_paper = {str(doc["paper_id"]): doc for doc in playgrounds.values()}
    links_by_script = _group(source.playground_links, "script_id")
    papers = sorted(source.papers, key=lambda r: str(r["id"]))
    paper_ids = {str(p["id"]) for p in papers}
    notes = _group(source.notes, "paper_id")
    lessons = _group(source.lessons, "paper_id")
    cards = _group(source.cards, "paper_id")
    scripts_by_paper = _group(source.scripts, "paper_id")
    checks = {str(c["script_id"]): c for c in source.fact_checks}
    claims_by_check = _group(source.claims, "fact_check_id")
    concepts = {str(c["id"]): c for c in source.concepts}

    links_by_paper = _group(
        (r for r in source.paper_concepts if str(r.get("concept_id")) in concepts),
        "paper_id",
    )
    papers_by_concept: Dict[str, List[str]] = {}
    for row in source.paper_concepts:
        if str(row["paper_id"]) in paper_ids and str(row["concept_id"]) in concepts:
            papers_by_concept.setdefault(str(row["concept_id"]), []).append(str(row["paper_id"]))
    related_by_paper = _group(
        (
            e for e in source.edges
            if e.get("kind") == "related_to"
            and e.get("src_kind") == "paper" and e.get("dst_kind") == "paper"
            and str(e.get("dst_id")) in paper_ids
        ),
        "src_id",
    )

    docs: Dict[str, Any] = {}
    paper_index = []
    for paper in papers:
        pid = str(paper["id"])
        note = _latest(notes.get(pid, []))
        lesson = _latest(lessons.get(pid, []))
        paper_cards = sorted(cards.get(pid, []), key=lambda r: str(r["id"]))
        episodes = sorted(
            scripts_by_paper.get(pid, []),
            key=lambda r: (str(r.get("created_at") or ""), str(r["id"])),
        )
        links = sorted(
            links_by_paper.get(pid, []),
            key=lambda r: (str(concepts[str(r["concept_id"])]["name"]), str(r["concept_id"])),
        )
        related = sorted(
            related_by_paper.get(pid, []),
            key=lambda r: (-(r.get("weight") or 0), str(r["dst_id"])),
        )
        doc = _pick(paper, PAPER_FIELDS)
        doc.update({
            "url_pdf": public_pdf_url(paper.get("url_pdf")),
            "note": serialize_note(note) if note else None,
            "lesson": serialize_lesson(lesson) if lesson else None,
            "quiz_cards": [serialize_card(c) for c in paper_cards],
            "episode_ids": [str(e["id"]) for e in episodes],
            "concepts": [
                {"id": str(r["concept_id"]), "weight": r.get("weight")} for r in links
            ],
            "related": [
                {"paper_id": str(r["dst_id"]), "weight": r.get("weight")} for r in related
            ],
            "playground": (
                playground_summary(playground_by_paper[pid]) if pid in playground_by_paper else None
            ),
        })
        docs[paper_key(pid)] = doc
        index = _pick(paper, ("id", "pillar_id", "title", "authors", "venue", "year", "added_at"))
        index.update({
            "doc": paper_key(pid),
            "has_note": note is not None,
            "has_lesson": lesson is not None,
            "card_count": len(paper_cards),
            "episode_ids": doc["episode_ids"],
            "has_playground": doc["playground"] is not None,
        })
        paper_index.append(index)

    for pg_id, pg_doc in playgrounds.items():
        docs[playground_key(pg_id)] = pg_doc

    episode_index = []
    for script in sorted(source.scripts, key=lambda r: str(r["id"])):
        sid = str(script["id"])
        check = checks.get(sid)
        claims = claims_by_check.get(str(check["id"]), []) if check else []
        follow = serialize_follow_along(links_by_script.get(sid, []), playgrounds, script.get("script") or "")
        doc = serialize_episode(script, check, claims, plans.get(sid), follow)
        docs[episode_key(sid)] = doc
        episode_index.append({
            "id": sid, "paper_id": doc["paper_id"], "pillar_id": doc["pillar_id"],
            "title": doc["title"], "created_at": doc["created_at"], "doc": episode_key(sid),
            "audio_key": doc["audio"]["key"] if doc["audio"] else None,
            "duration_seconds": doc["audio"]["duration_seconds"] if doc["audio"] else None,
            "has_timeline": doc["timeline"] is not None,
            "has_follow_along": any(not link["stale"] for link in doc["follow_along"]),
        })
    episode_index.sort(key=lambda e: (str(e["created_at"] or ""), e["id"]), reverse=True)

    docs["library.json"] = {
        "pillars": [serialize_pillar(p) for p in sorted(source.pillars, key=lambda r: str(r["id"]))],
        "papers": paper_index,
        "episodes": episode_index,
        "concepts": [
            serialize_concept(c, sorted(papers_by_concept.get(cid, [])))
            for cid, c in sorted(concepts.items(), key=lambda kv: (str(kv[1]["name"]), kv[0]))
            if cid in papers_by_concept
        ],
    }
    docs["schedule.json"] = {
        # The last cloud review event every card below already includes. A phone folds
        # the events after this onto these cards.
        "through_seq": source.through_seq,
        "params_version": FSRS_PARAMS_VERSION,
        # Folded from review_logs, the source of truth, so each card carries the
        # learning `step` the quiz_cards row has no column for.
        "source": "review_log",
        "cards": schedule_cards(source),
    }
    return docs


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def env_secret_values(environ: Mapping[str, str]) -> List[str]:
    """Values of secret-looking environment variables, long enough to not collide."""
    return sorted({
        value for name, value in environ.items()
        if _SECRET_ENV_NAME.search(name) and isinstance(value, str) and len(value) >= 12
    })


def _walk(value: Any, path: str) -> Iterable[Tuple[str, Any, bool]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield f"{path}.{key}", key, True
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")
    else:
        yield path, value, False


def _guard(key: str, doc: Any, secrets: Iterable[str]) -> None:
    """Refuse a document that carries a forbidden key, a secret or a local path."""
    for path, item, is_key in _walk(doc, key):
        if is_key:
            if item in FORBIDDEN_KEYS or _SECRET_KEY.search(str(item)):
                raise ExportError(f"{path}: field {item!r} must not be published")
        elif isinstance(item, str):
            match = _SECRET_VALUE.search(item)
            if match:
                raise ExportError(
                    f"{path}: value contains {match.group(0)[:12]!r}, which must not be published"
                )
    text = canonical_bytes(doc).decode("utf-8")
    for secret in secrets:
        if secret in text:
            raise ExportError(f"{key}: contains the value of a secret environment variable")


# ---------------------------------------------------------------------------
# The bundle on disk
# ---------------------------------------------------------------------------


@dataclass
class ExportReport:
    dry_run: bool
    out: Path
    docs_total: int = 0
    docs_added: int = 0
    docs_changed: int = 0
    docs_unchanged: int = 0
    docs_removed: int = 0
    doc_bytes: int = 0
    audio_files: int = 0
    audio_transcoded: int = 0
    audio_reused: int = 0
    audio_pending: int = 0
    audio_source_bytes: int = 0
    audio_cloud_bytes: int = 0
    audio_seconds: float = 0.0
    playgrounds: int = 0
    manifest_written: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def docs_written(self) -> int:
        return self.docs_added + self.docs_changed

    def render(self) -> str:
        mb = lambda n: f"{n / 1_000_000:.1f} MB"  # noqa: E731
        verb = "would write" if self.dry_run else "wrote"
        lines = [
            f"cloud export {'(dry run) ' if self.dry_run else ''}-> {self.out}",
            f"  documents: {self.docs_total} ({self.doc_bytes / 1000:.0f} KB)  "
            f"added {self.docs_added}, changed {self.docs_changed}, "
            f"unchanged {self.docs_unchanged}, removed {self.docs_removed}",
            f"  audio: {self.audio_files} files, {mb(self.audio_cloud_bytes)} "
            f"(originals {mb(self.audio_source_bytes)}, {self.audio_seconds / 60:.0f} min)  "
            f"transcoded {self.audio_transcoded}, reused {self.audio_reused}, "
            f"still to transcode {self.audio_pending}",
            f"  playgrounds: {self.playgrounds}",
            f"  {verb} {self.docs_written} document(s); manifest "
            + (
                "unchanged"
                if not self.manifest_written
                else ("would be written last" if self.dry_run else "written last")
            ),
        ]
        lines.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(lines)


def read_manifest(out: Path) -> Optional[Dict[str, Any]]:
    path = out / "manifest.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise ExportError(f"{path} is not readable JSON ({exc}); move it aside or pick a new --out") from exc
    if not isinstance(data, dict) or data.get("version") != BUNDLE_VERSION:
        raise ExportError(f"{path} is not a version {BUNDLE_VERSION} manifest")
    return data


def _check_out_dir(out: Path) -> None:
    """Refuse to scatter a bundle (and delete orphans) in a directory that is not one."""
    if out.exists():
        if not out.is_dir():
            raise ExportError(f"{out} is not a directory")
        # A run that failed part-way leaves audio/ (and no manifest). That is still ours.
        ours = set(MANAGED_DIRS) | {"manifest.json", "library.json", "schedule.json"}
        foreign = [p.name for p in out.iterdir() if p.name not in ours]
        if foreign and not (out / "manifest.json").is_file():
            raise ExportError(
                f"{out} is not empty and is not a bundle (found {foreign[0]!r}); "
                "refusing to write into it"
            )


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f"{_TEMP_PREFIXES[0]}{path.name}")
    scratch.write_bytes(data)
    scratch.replace(path)


def _tombstones(
    previous: Optional[Mapping[str, Any]], live: Iterable[str], now: datetime,
) -> List[Dict[str, str]]:
    live = set(live)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    kept: Dict[str, str] = {}
    if previous:
        for item in previous.get("tombstones", []):
            try:
                deleted = datetime.strptime(item["deleted_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
            except (KeyError, ValueError):
                continue
            if item["key"] not in live and now - deleted <= timedelta(days=TOMBSTONE_DAYS):
                kept[item["key"]] = item["deleted_at"]
        for item in list(previous.get("docs", [])) + list(previous.get("audio", [])):
            if item["key"] not in live:
                kept.setdefault(item["key"], stamp)
    return [{"key": key, "deleted_at": kept[key]} for key in sorted(kept)]


def run_export(
    out: Path,
    *,
    dry_run: bool = False,
    source: Optional[CloudSource] = None,
    audio_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
    secrets: Optional[Iterable[str]] = None,
    transcoder: Optional[Transcoder] = None,
    playground_caps: Optional[PlaygroundCaps] = None,
) -> ExportReport:
    """Build the bundle in ``out``. With ``dry_run``: read, hash, report, write nothing.

    Raises ``ExportError`` before anything is written when a document fails the
    guard, a transcode fails, or its duration drifts. The manifest is the last
    file written, so a reader never sees one that points at documents that are
    not there yet.
    """
    out = Path(out)
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)
    secrets = list(env_secret_values(os.environ) if secrets is None else secrets)
    if audio_dir is None or playground_caps is None:
        from ..config import get_settings

        settings = get_settings()
        if audio_dir is None:
            audio_dir = Path(settings.podcast_audio_dir)
        if playground_caps is None:
            playground_caps = PlaygroundCaps.from_settings(settings)
    _check_out_dir(out)
    previous = read_manifest(out)
    source = source if source is not None else load_source()

    playground_warnings: List[str] = []
    playgrounds = build_playgrounds(source, playground_caps, secrets, playground_warnings)
    # Fail on a leak before spending minutes in ffmpeg. The audio entry only adds
    # a key and a hash to an episode, so the guard sees the same content here.
    for key, doc in build_documents(source, {}, playgrounds).items():
        _guard(key, doc, secrets)
    plans = plan_audio(
        source.scripts, Path(audio_dir), out, previous,
        transcode=not dry_run, transcoder=transcoder or audio_tools.transcode,
    )
    encoded: Dict[str, bytes] = {}
    for key, doc in build_documents(source, plans, playgrounds).items():
        _guard(key, doc, secrets)
        encoded[key] = canonical_bytes(doc)

    report = ExportReport(dry_run=dry_run, out=out, playgrounds=len(playgrounds))
    report.warnings.extend(source.warnings)
    report.warnings.extend(playground_warnings)
    prev_docs = {item["key"]: item["sha256"] for item in (previous or {}).get("docs", [])}
    doc_entries = []
    for key in sorted(encoded):
        digest = sha256_hex(encoded[key])
        doc_entries.append({"key": key, "sha256": digest, "bytes": len(encoded[key])})
        report.doc_bytes += len(encoded[key])
        if key not in prev_docs:
            report.docs_added += 1
        elif prev_docs[key] != digest:
            report.docs_changed += 1
        else:
            report.docs_unchanged += 1
    report.docs_total = len(doc_entries)
    report.docs_removed = len(set(prev_docs) - set(encoded))

    entries: Dict[str, AudioEntry] = {}
    for plan in plans.values():
        if plan.status == "missing":
            report.warnings.append(f"episode {plan.script_id}: {plan.reason}; published without audio")
        elif plan.entry is None:
            report.audio_pending += 1
        else:
            entries[plan.entry.key] = plan.entry
            report.audio_transcoded += plan.status == "transcoded"
            report.audio_reused += plan.status == "reused"
    audio_entries = [entries[key].manifest() for key in sorted(entries)]
    report.audio_files = len(audio_entries)
    report.audio_source_bytes = sum(e.source_bytes for e in entries.values())
    report.audio_cloud_bytes = sum(e.bytes for e in entries.values())
    report.audio_seconds = sum(e.duration_seconds for e in entries.values())

    live = set(encoded) | set(entries)
    core = {
        "version": BUNDLE_VERSION,
        "fsrs": {"params_version": FSRS_PARAMS_VERSION, "through_seq": source.through_seq},
        "docs": doc_entries,
        "audio": audio_entries,
        "tombstones": _tombstones(previous, live, now),
    }
    unchanged = previous is not None and {k: previous.get(k) for k in core} == core
    manifest = dict(previous) if unchanged else {**core, "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    report.manifest_written = not unchanged

    if dry_run:
        if report.audio_pending:
            report.warnings.append(
                f"{report.audio_pending} episode(s) have no cloud audio yet; their documents "
                "are hashed without it and will change once transcoded"
            )
        return report

    for key, data in encoded.items():
        path = out / key
        if not path.is_file() or path.read_bytes() != data:
            _write_atomic(path, data)
    if report.manifest_written:
        _write_atomic(out / "manifest.json", canonical_bytes(manifest))
    # After the manifest: a file the old manifest listed and the new one does not.
    _remove_orphans(out, live)
    return report


def _remove_orphans(out: Path, live: set) -> None:
    for name in MANAGED_DIRS:
        folder = out / name
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            key = f"{name}/{path.name}"
            if path.is_file() and key not in live:
                path.unlink()
