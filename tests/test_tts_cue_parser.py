"""Tests for podcast script cue stripping before TTS."""

from nlp_pillars.tts.cue_parser import (
    PAUSE_SILENCE_MS,
    parse_podcast_script,
    plan_synthesis_chunks,
)


def test_host_prefix_stripped_from_speech():
    parsed = parse_podcast_script("[HOST]: Welcome to the show.\n")
    assert len(parsed.speech_chunks) == 1
    assert parsed.speech_chunks[0].text == "Welcome to the show."
    assert "[HOST]" not in parsed.speech_chunks[0].text
    assert "HOST" not in parsed.speech_chunks[0].text.upper() or "Welcome" in parsed.speech_chunks[0].text


def test_music_never_in_synthesis_text():
    script = (
        "[HOST]: Before the break.\n"
        "[MUSIC]\n"
        "[HOST]: After the break.\n"
    )
    parsed = parse_podcast_script(script)
    plans = plan_synthesis_chunks(parsed)
    joined = " ".join(plan.text for plan in plans)
    assert "[MUSIC]" not in joined
    assert "MUSIC" not in joined
    assert "Before the break." in joined
    assert "After the break." in joined


def test_pause_becomes_silence_not_speech():
    script = "[HOST]: Line one.\n[PAUSE]\n[HOST]: Line two.\n"
    parsed = parse_podcast_script(script)
    silences = [item for item in parsed.timeline if hasattr(item, "duration_ms")]
    assert any(s.duration_ms == PAUSE_SILENCE_MS for s in silences)
    plans = plan_synthesis_chunks(parsed)
    assert any(plan.pause_after_ms == PAUSE_SILENCE_MS for plan in plans)


def test_all_cue_types_stripped():
    script = (
        "[HOST]: Welcome.\n"
        "[PAUSE]\n"
        "[MUSIC]\n"
        "[SFX]\n"
        "[TRANSITION]\n"
        "[HOST]: Today we discuss transformers.\n"
    )
    plans = plan_synthesis_chunks(parse_podcast_script(script))
    blob = " ".join(plan.text for plan in plans).upper()
    for token in ("PAUSE", "MUSIC", "SFX", "TRANSITION", "HOST"):
        assert token not in blob
    assert "Welcome." in " ".join(plan.text for plan in plans)
    assert "transformers." in " ".join(plan.text for plan in plans)
