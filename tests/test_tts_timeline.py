"""Read-along timeline: staleness, and the backfill that fills in old episodes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


from nlp_pillars.schemas import AudioMetadata
from nlp_pillars.tts import aligner
from nlp_pillars.tts.alignment import AlignmentError, TIMELINE_VERSION
from nlp_pillars.tts.timeline import audio_file, backfill_timelines, timeline_state
from nlp_pillars.factcheck.engine import sha256_text

SCRIPT = "[HOST]: One two three."


def episode(tmp_path, *, name="ep.mp3", on_disk=True, timeline=None, script=SCRIPT, sid="s1"):
    if on_disk and name:
        (tmp_path / name).write_bytes(b"mp3")
    return SimpleNamespace(
        id=sid,
        title=f"Episode {sid}",
        script=script,
        audio_metadata=AudioMetadata(file_name=name, timeline=timeline),
    )


def current(script=SCRIPT):
    return {"version": TIMELINE_VERSION, "script_sha256": sha256_text(script), "lines": []}


class TestState:
    def test_missing_current_and_stale(self):
        assert timeline_state(None, SCRIPT) == "missing"
        assert timeline_state({}, SCRIPT) == "missing"
        assert timeline_state(current(), SCRIPT) == "current"
        assert timeline_state(current(), SCRIPT + " Edited.") == "stale"
        assert timeline_state({**current(), "version": 0}, SCRIPT) == "missing"


def test_audio_file_prefers_the_configured_directory(tmp_path):
    assert audio_file("a.mp3", "/elsewhere/a.mp3", tmp_path) == tmp_path / "a.mp3"
    assert audio_file("../a.mp3", None, tmp_path) == tmp_path / "a.mp3"
    assert audio_file(None, "/x/y.mp3", tmp_path) == Path("/x/y.mp3")
    assert audio_file(None, None, tmp_path) is None


class TestBackfill:
    def fake_timeline(self):
        return {"version": TIMELINE_VERSION, "word_count": 3, "matched_ratio": 1.0, "word_level": True}

    def test_dry_run_writes_nothing_and_never_listens(self, tmp_path):
        compute, save = MagicMock(), MagicMock()
        rows = backfill_timelines(
            [episode(tmp_path)], tmp_path, dry_run=True, compute=compute, save=save
        )
        assert [(r.action, r.state) for r in rows] == [("align", "missing")]
        compute.assert_not_called()
        save.assert_not_called()

    def test_dry_run_does_not_load_the_recogniser(self, tmp_path, monkeypatch):
        monkeypatch.setattr(aligner, "_load_model", MagicMock(side_effect=AssertionError("model loaded")))
        backfill_timelines([episode(tmp_path)], tmp_path, dry_run=True)

    def test_times_an_episode_and_saves_it_with_its_timeline(self, tmp_path):
        ep = episode(tmp_path)
        compute = MagicMock(return_value=self.fake_timeline())
        save = MagicMock()
        rows = backfill_timelines([ep], tmp_path, dry_run=False, compute=compute, save=save)
        assert [r.action for r in rows] == ["aligned"]
        compute.assert_called_once_with(SCRIPT, tmp_path / "ep.mp3")
        save.assert_called_once()
        saved_id, saved_meta = save.call_args.args
        assert saved_id == "s1"
        assert saved_meta.timeline["word_count"] == 3
        assert saved_meta.file_name == "ep.mp3"

    def test_skips_a_current_timeline_unless_forced(self, tmp_path):
        ep = episode(tmp_path, timeline=current())
        compute = MagicMock(return_value=self.fake_timeline())
        assert backfill_timelines([ep], tmp_path, dry_run=False, compute=compute, save=MagicMock())[0].action == "skip"
        compute.assert_not_called()
        forced = backfill_timelines([ep], tmp_path, dry_run=False, force=True, compute=compute, save=MagicMock())
        assert forced[0].action == "aligned"

    def test_a_stale_timeline_is_redone(self, tmp_path):
        ep = episode(tmp_path, timeline=current("[HOST]: Old text."))
        rows = backfill_timelines(
            [ep], tmp_path, dry_run=False, compute=MagicMock(return_value=self.fake_timeline()), save=MagicMock()
        )
        assert (rows[0].state, rows[0].action) == ("stale", "aligned")

    def test_episodes_without_audio_are_left_alone(self, tmp_path):
        no_audio = episode(tmp_path, name=None, sid="a")
        gone = episode(tmp_path, name="missing.mp3", on_disk=False, sid="b")
        compute = MagicMock()
        rows = backfill_timelines([no_audio, gone], tmp_path, dry_run=False, compute=compute, save=MagicMock())
        assert [(r.state, r.action) for r in rows] == [("no_audio", "skip"), ("no_file", "skip")]
        compute.assert_not_called()

    def test_one_failure_does_not_stop_the_rest(self, tmp_path):
        bad = episode(tmp_path, name="bad.mp3", sid="bad")
        good = episode(tmp_path, name="good.mp3", sid="good")

        def compute(script, path):
            if path.name == "bad.mp3":
                raise AlignmentError("does not look like the same episode")
            return self.fake_timeline()

        save = MagicMock()
        rows = backfill_timelines([bad, good], tmp_path, dry_run=False, compute=compute, save=save)
        assert [(r.script_id, r.action) for r in rows] == [("bad", "failed"), ("good", "aligned")]
        assert "same episode" in rows[0].detail
        save.assert_called_once()
        assert save.call_args.args[0] == "good"

    def test_a_save_failure_is_reported_not_raised(self, tmp_path):
        rows = backfill_timelines(
            [episode(tmp_path)],
            tmp_path,
            dry_run=False,
            compute=MagicMock(return_value=self.fake_timeline()),
            save=MagicMock(side_effect=RuntimeError("PGRST204")),
        )
        assert rows[0].action == "failed" and "PGRST204" in rows[0].detail


class TestCliDryRun:
    def test_dry_run_through_the_cli_writes_nothing(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from nlp_pillars import cli, db

        scripts = [episode(tmp_path)]
        monkeypatch.setattr(db, "get_podcast_scripts", lambda limit=10, pillar_id=None: scripts)
        save = MagicMock()
        monkeypatch.setattr(db, "update_podcast_audio_metadata", save)
        monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(podcast_audio_dir=str(tmp_path), log_level="INFO"))
        monkeypatch.setattr(aligner, "_load_model", MagicMock(side_effect=AssertionError("model loaded")))
        result = CliRunner().invoke(cli.app, ["timeline", "backfill", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "would be timed" in result.output
        save.assert_not_called()
        assert scripts[0].audio_metadata.timeline is None


class TestAudioJob:
    """The recording job times the episode, and survives not being able to."""

    def run_job(self, tmp_path, compute):
        import threading
        from unittest.mock import patch

        from webui.services import podcast_audio_service as service

        stages: list[tuple[str, str, str | None]] = []
        script = MagicMock()
        script.script = SCRIPT
        script.paper_id = "paper-1"
        (tmp_path / "v.wav").write_bytes(b"fake voice reference")
        (tmp_path / "chunk.wav").write_bytes(b"fake synthesized chunk")
        client = MagicMock()
        client.check_status.return_value = MagicMock(status=service.TtsEngineStatus.READY, message="ready")
        client.synthesize.return_value = str(tmp_path / "chunk.wav")
        saved = MagicMock()
        paths = ("http://x", tmp_path, tmp_path / "audio", tmp_path / "dl", tmp_path, "")
        with patch.object(service.db, "get_podcast_script_by_id", return_value=script), \
                patch.object(service, "validate_voice_for_generation", return_value=tmp_path / "v.wav"), \
                patch.object(service, "get_tts_client", return_value=client), \
                patch.object(service, "_settings_paths", return_value=paths), \
                patch.object(service, "_assemble_with_variable_pauses"), \
                patch.object(service, "wav_duration_seconds", return_value=3.0), \
                patch.object(service, "encode_mp3"), \
                patch.object(service, "compute_timeline", compute), \
                patch.object(service.db, "update_podcast_audio_metadata", saved):
            service.run_podcast_audio_job(
                "run-1", "script-1", "voice.wav",
                lambda name, status, detail=None: stages.append((name, status, detail)),
                threading.Event(),
            )
        return stages, saved

    def test_the_timeline_is_saved_with_the_audio(self, tmp_path):
        timeline = {"version": TIMELINE_VERSION, "word_count": 3, "matched_ratio": 0.97, "word_level": True}
        stages, saved = self.run_job(tmp_path, MagicMock(return_value=timeline))
        meta = saved.call_args.args[1]
        assert meta.timeline == timeline
        names = [n for n, _, _ in stages]
        assert names.index("tts_encode") < names.index("tts_align") < names.index("tts_save")
        assert ("tts_align", "completed", "3 words, 97% heard") in stages

    def test_a_failed_timing_still_saves_the_audio_and_says_why(self, tmp_path):
        stages, saved = self.run_job(
            tmp_path, MagicMock(side_effect=aligner.AlignerError("faster-whisper is not installed."))
        )
        meta = saved.call_args.args[1]
        assert meta.timeline is None
        assert meta.file_name and meta.file_name.endswith(".mp3")
        failed = [d for n, st, d in stages if n == "tts_align" and st == "failed"]
        assert failed and "faster-whisper is not installed." in failed[0]
        assert ("tts_save", "completed", "Saved") in stages

    def test_an_unexpected_crash_in_timing_also_keeps_the_audio(self, tmp_path):
        stages, saved = self.run_job(tmp_path, MagicMock(side_effect=ValueError("boom")))
        assert saved.call_args.args[1].timeline is None
        assert any(n == "tts_align" and st == "failed" and "boom" in d for n, st, d in stages)
