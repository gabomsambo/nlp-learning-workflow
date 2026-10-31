"""Durable podcast audio queue behavior without an IndexTTS server or GPU."""

import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nlp_pillars import db
from nlp_pillars.schemas import PODCAST_AUDIO_STAGES, RunStatus, StageStatus
from webui.services import run_service

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (ROOT / "docs/migrations/037_podcast_audio_queue.sql").read_text()
PILLAR = "models-architectures"
SCRIPT_ID = "11111111-2222-3333-4444-555555555555"
RUN_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
VOICE = "nested/voice.wav"
SCRIPT = "[HOST] A short research lesson."
SCRIPT_HASH = hashlib.sha256(SCRIPT.encode()).hexdigest()


def _pending_run(run_id=RUN_ID, script_id=SCRIPT_ID, created_at="2026-10-31T10:00:00Z"):
    return SimpleNamespace(
        id=run_id,
        pillar_id=PILLAR,
        kind="podcast_audio",
        status=RunStatus.PENDING.value,
        created_at=created_at,
        result={
            "script_id": script_id,
            "voice_path": VOICE,
            "script_sha256": SCRIPT_HASH,
            "consent": True,
            "queue_state": "waiting_for_indextts",
        },
    )


def _index_definition(name):
    match = re.search(
        rf"CREATE UNIQUE INDEX(?: IF NOT EXISTS)?[^;]*{name}[^;]*;",
        MIGRATION,
        re.I | re.S,
    )
    assert match, f"migration 037 no longer defines {name}"
    return match.group(0)


def _guarded_db():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE pipeline_runs (id INTEGER PRIMARY KEY, pillar_id TEXT NOT NULL, "
        "kind TEXT NOT NULL, status TEXT NOT NULL, result TEXT NOT NULL)"
    )
    conn.execute(_index_definition("pipeline_runs_one_active_per_pillar"))
    conn.execute(_index_definition("pipeline_runs_one_active_podcast_audio_per_script"))
    return conn


def _insert(conn, kind, script_id=None, status="pending", pillar=PILLAR):
    payload = json.dumps({"script_id": script_id})
    conn.execute(
        "INSERT INTO pipeline_runs (pillar_id, kind, status, result) VALUES (?, ?, ?, ?)",
        (pillar, kind, status, payload),
    )


def test_migration_indexes_allow_audio_queue_without_pillar_conflicts():
    conn = _guarded_db()
    _insert(conn, "discover")
    _insert(conn, "podcast_audio", "script-a")
    _insert(conn, "podcast_audio", "script-b")
    assert conn.execute("SELECT count(*) FROM pipeline_runs").fetchone()[0] == 3

    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, "run_daily")
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, "podcast_audio", "script-a")


def test_migration_037_is_independent_of_evidence_migration_036():
    assert "ALTER TABLE podcast_scripts" not in MIGRATION
    assert "CREATE TABLE IF NOT EXISTS podcast_settings" in MIGRATION
    assert "pipeline_runs_one_active_podcast_audio_per_script" in MIGRATION
    assert "NOTIFY pgrst, 'reload schema';" in MIGRATION


def test_queue_request_is_stored_before_any_tts_probe():
    stored = {}
    fake_run = SimpleNamespace(id=RUN_ID)

    def create(pillar, trigger, kind, stages, result=None):
        stored.update(
            pillar=pillar, trigger=trigger, kind=kind, stages=stages, result=result,
        )
        return fake_run

    with patch.object(run_service.db, "create_pipeline_run", side_effect=create), \
         patch.object(run_service.podcast_audio_service, "tts_status_payload") as probe:
        run_id, created = run_service.queue_podcast_audio_run(
            PILLAR, SCRIPT_ID, VOICE, SCRIPT_HASH, consent=True,
        )

    assert (run_id, created) == (RUN_ID, True)
    assert stored["kind"] == "podcast_audio"
    assert stored["stages"] == [stage.value for stage in PODCAST_AUDIO_STAGES]
    assert stored["result"] == {
        "script_id": SCRIPT_ID,
        "voice_path": VOICE,
        "script_sha256": SCRIPT_HASH,
        "consent": True,
        "queue_state": "waiting_for_indextts",
        "queue_message": "Waiting for IndexTTS readiness check.",
    }
    probe.assert_not_called()


def test_duplicate_queue_request_attaches_to_existing_script_run():
    with (
        patch.object(run_service.db, "create_pipeline_run", return_value=None),
        patch.object(
            run_service.db,
            "get_active_podcast_audio_run",
            return_value=SimpleNamespace(id="existing-run"),
        ),
    ):
        assert run_service.queue_podcast_audio_run(
            PILLAR, SCRIPT_ID, VOICE, SCRIPT_HASH, True,
        ) == ("existing-run", False)


@pytest.mark.parametrize(
    ("engine_status", "state", "message"),
    [
        ("not_running", "waiting_for_indextts", "IndexTTS is not running"),
        ("wrong_service", "wrong_service", "Dots TTS answered"),
        ("contract_mismatch", "contract_mismatch", "missing /gen_single"),
    ],
)
def test_pump_keeps_unready_audio_visible_without_scheduling(
    engine_status, state, message,
):
    run = _pending_run()
    updated = []
    scheduler = MagicMock()
    with (
        patch.object(run_service.db, "list_active_podcast_audio_runs", return_value=[run]),
        patch.object(
            run_service.podcast_audio_service,
            "tts_status_payload",
            return_value={
                "status": engine_status,
                "ready": False,
                "message": message,
                "start_command": "./start-indextts.sh",
            },
        ),
        patch.object(
            run_service.db,
            "update_pending_podcast_audio_result",
            side_effect=lambda run_id, result: updated.append((run_id, result)) or True,
        ),
    ):
        run_service.pump_podcast_audio_queue(scheduler, {})

    scheduler.add_job.assert_not_called()
    saved = updated[0][1]
    assert saved["queue_state"] == state
    assert saved["queue_message"] == message
    assert saved.get("start_command") == (
        "./start-indextts.sh" if engine_status == "not_running" else None
    )


def test_ready_pump_claims_and_schedules_only_the_oldest_pending_audio():
    oldest = _pending_run(run_id="old", script_id="script-old")
    newer = _pending_run(
        run_id="new", script_id="script-new", created_at="2026-10-31T10:01:00Z"
    )
    scheduler = MagicMock()
    cancel_events = {}
    claimed = []

    def claim(run_id, result):
        claimed.append((run_id, result))
        return True

    with (
        patch.object(
            run_service.db,
            "list_active_podcast_audio_runs",
            return_value=[oldest, newer],
        ),
        patch.object(
            run_service.podcast_audio_service,
            "tts_status_payload",
            return_value={"status": "ready", "ready": True},
        ),
        patch.object(run_service.db, "claim_pending_podcast_audio_run", side_effect=claim),
    ):
        run_service.pump_podcast_audio_queue(scheduler, cancel_events)

    assert [run_id for run_id, _ in claimed] == ["old"]
    assert scheduler.add_job.call_count == 1
    call = scheduler.add_job.call_args
    assert call.kwargs["id"] == "run:old"
    assert call.kwargs["kwargs"]["script_id"] == "script-old"
    assert call.kwargs["kwargs"]["voice_path"] == VOICE
    assert "old" in cancel_events


def test_pump_does_not_claim_another_request_while_audio_is_running():
    running = _pending_run(run_id="running")
    running.status = RunStatus.RUNNING.value
    pending = _pending_run(run_id="pending", script_id="script-pending")
    scheduler = MagicMock()
    with (
        patch.object(
            run_service.db,
            "list_active_podcast_audio_runs",
            return_value=[running, pending],
        ),
        patch.object(run_service.db, "claim_pending_podcast_audio_run") as claim,
    ):
        run_service.pump_podcast_audio_queue(scheduler, {})
    claim.assert_not_called()
    scheduler.add_job.assert_not_called()


def test_pending_audio_cancel_closes_stages_and_removes_cancel_event():
    cancel_events = {RUN_ID: threading.Event()}
    with (
        patch.object(run_service.db, "cancel_pending_podcast_audio_run", return_value=True),
        patch.object(run_service.db, "update_pipeline_run_stage") as update_stage,
    ):
        accepted, message = run_service.cancel_run(cancel_events, RUN_ID)

    assert accepted is True
    assert "waiting for IndexTTS" in message
    assert RUN_ID not in cancel_events
    assert update_stage.call_count == len(PODCAST_AUDIO_STAGES)
    assert all(call.args[2] == StageStatus.SKIPPED.value for call in update_stage.call_args_list)


def test_execute_audio_job_never_constructs_the_discovery_orchestrator():
    with (
        patch.object(run_service.db, "start_pipeline_run"),
        patch.object(run_service, "_finish_podcast_audio") as finish_audio,
        patch.object(run_service, "Orchestrator", side_effect=AssertionError("Qdrant path")) as orchestrator,
        patch.object(run_service.cloud_publish_service, "notify"),
    ):
        run_service.execute_run(
            RUN_ID,
            "podcast_audio",
            PILLAR,
            threading.Event(),
            {RUN_ID: threading.Event()},
            script_id=SCRIPT_ID,
            voice_path=VOICE,
            audio_request={"script_sha256": SCRIPT_HASH},
        )

    orchestrator.assert_not_called()
    finish_audio.assert_called_once()
    assert finish_audio.call_args.args[3]["audio_request"]["script_sha256"] == SCRIPT_HASH


def test_auto_audio_is_queued_only_after_a_script_is_saved():
    saved_result = {"saved": True, "script_id": SCRIPT_ID, "script": SCRIPT}
    script = SimpleNamespace(script=SCRIPT)
    queue = MagicMock(return_value=("audio-run", True))
    with (
        patch.object(
            run_service.podcast_script_service,
            "run_podcast_script_job",
            return_value=saved_result.copy(),
        ),
        patch.object(run_service.db, "finish_pipeline_run") as finish,
        patch.object(run_service.db, "get_podcast_script_by_id", return_value=script),
        patch.object(run_service, "queue_podcast_audio_run", queue),
        patch.object(run_service, "_maybe_factcheck_after_script"),
    ):
        run_service._finish_podcast_script(
            "script-run", PILLAR, threading.Event(), {},
            {"paper_id": "paper-1", "options": {}, "voice_path": VOICE,
             "auto_audio": True},
            MagicMock(),
        )

    queue.assert_called_once_with(
        pillar_id=PILLAR,
        script_id=SCRIPT_ID,
        voice_path=VOICE,
        script_sha256=SCRIPT_HASH,
        consent=True,
    )
    assert finish.call_args.kwargs["result"]["audio_run_id"] == "audio-run"


def test_unsaved_script_never_queues_audio():
    with (
        patch.object(
            run_service.podcast_script_service,
            "run_podcast_script_job",
            return_value={"saved": False, "script": "NOT saved"},
        ),
        patch.object(run_service.db, "finish_pipeline_run"),
        patch.object(run_service, "queue_podcast_audio_run") as queue,
        patch.object(run_service, "_maybe_factcheck_after_script"),
    ):
        run_service._finish_podcast_script(
            "script-run", PILLAR, threading.Event(), {},
            {"paper_id": "paper-1", "options": {}, "voice_path": VOICE,
             "auto_audio": True},
            MagicMock(),
        )
    queue.assert_not_called()
