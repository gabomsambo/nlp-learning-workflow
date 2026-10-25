"""Read-along alignment: script words against what a recogniser heard."""

from __future__ import annotations

import json

import pytest

from nlp_pillars.tts import alignment as A
from nlp_pillars.tts.cue_parser import parse_podcast_script, spoken_line_text

SCRIPT = (
    "[HOST]: Welcome back. Today we cover GPT-4 and LightGBM.\n"
    "\n"
    "[PAUSE]\n"
    "[HOST]: It scored 3.5 points in 2024, up 50% — a big jump.\n"
    "[MUSIC]\n"
    "[HOST]: That's the whole story."
)


def hear(script: str, *, per_word: float = 0.4, drop=(), rewrite=None) -> list[A.AsrWord]:
    """A recogniser that heard the script, word by word, at a steady pace."""
    rewrite = rewrite or {}
    out: list[A.AsrWord] = []
    clock = 1.0
    for number, word in enumerate(A.script_words(script)):
        if number in drop:
            clock += per_word
            continue
        for piece in rewrite.get(word.text, [word.text]):
            out.append(A.AsrWord(piece, clock, clock + per_word * 0.9))
            clock += per_word
    return out


class TestSpokenUnits:
    @pytest.mark.parametrize(
        "token, expected",
        [
            ("Welcome", ("welcome",)),
            ("back.", ("back",)),
            ("that's", ("thats",)),
            ("that’s", ("thats",)),
            ("GPT-4", ("gpt", "four")),
            ("LightGBM,", ("lightgbm",)),
            ("3.5", ("three", "point", "five")),
            ("2024", ("twenty", "twenty", "four")),
            ("2005", ("two", "thousand", "five")),
            ("1999", ("nineteen", "ninety", "nine")),
            ("1,000", ("one", "thousand")),
            ("50%", ("fifty", "percent")),
            ("$5", ("five", "dollars")),
            ("transaction-monitoring", ("transaction", "monitoring")),
            ("—", ()),
            ("...", ()),
        ],
    )
    def test_tokens(self, token, expected):
        assert A.spoken_units(token) == expected

    def test_long_digit_runs_are_read_digit_by_digit(self):
        assert A.spoken_units("1234567890123") == tuple(
            "one two three four five six seven eight nine zero one two three".split()
        )

    def test_the_recogniser_and_the_script_agree_on_digits(self):
        # Whisper writes "3.5"; the script may say it in words. Both land on one form.
        assert A.spoken_units("3.5") == ("three", "point", "five")
        assert A.spoken_units("three point five") == ("three", "point", "five")


class TestScriptWords:
    def test_numbers_lines_the_way_splitlines_does(self):
        words = A.script_words(SCRIPT)
        assert sorted({w.line for w in words}) == [0, 3, 5]
        assert [w.index for w in words if w.line == 5] == [0, 1, 2, 3]
        assert words[0].text == "Welcome"

    def test_cues_and_blank_lines_have_no_words(self):
        lines = {w.line for w in A.script_words(SCRIPT)}
        assert not lines & {1, 2, 4}

    def test_agrees_with_what_the_tts_chunker_speaks(self):
        parsed = parse_podcast_script(SCRIPT)
        spoken = " ".join(chunk.text for chunk in parsed.speech_chunks).split()
        assert [w.text for w in A.script_words(SCRIPT)] == spoken

    def test_spoken_line_text(self):
        assert spoken_line_text("[HOST]: Hi there.") == "Hi there."
        assert spoken_line_text("[host]:   Hi") == "Hi"
        assert spoken_line_text("[PAUSE]") is None
        assert spoken_line_text("   ") is None
        assert spoken_line_text("[HOST]:") is None
        assert spoken_line_text("Plain narration.") == "Plain narration."


class TestAlignWords:
    def test_every_word_heard_is_matched_and_timed_by_the_recogniser(self):
        asr = hear(SCRIPT)
        words = A.script_words(SCRIPT)
        timings = A.align_words(words, asr, 60.0)
        assert len(timings) == len(words)
        assert all(t.matched for t in timings)
        assert timings[0].start_ms == 1000
        assert timings[1].start_ms == 1400

    def test_digits_the_recogniser_wrote_out_still_match(self):
        # The script has "3.5"; the recogniser heard it as three words.
        asr = hear(SCRIPT, rewrite={"3.5": ["three", "point", "five"], "2024": ["twenty", "twenty", "four"]})
        words = A.script_words(SCRIPT)
        timings = A.align_words(words, asr, 60.0)
        assert all(t.matched for t in timings)
        number = next(i for i, w in enumerate(words) if w.text == "3.5")
        # The word spans all three heard words.
        assert timings[number].end_ms - timings[number].start_ms > 700

    def test_an_acronym_the_recogniser_split_is_timed_across_both_pieces(self):
        asr = hear(SCRIPT, rewrite={"LightGBM.": ["light", "gbm"]})
        words = A.script_words(SCRIPT)
        timings = A.align_words(words, asr, 60.0)
        index = next(i for i, w in enumerate(words) if w.text == "LightGBM.")
        assert timings[index].matched
        assert timings[index].end_ms - timings[index].start_ms > 600

    def test_a_misspelt_word_keeps_its_place(self):
        asr = hear(SCRIPT, rewrite={"Welcome": ["wellcome"]})
        words = A.script_words(SCRIPT)
        timings = A.align_words(words, asr, 60.0)
        assert timings[0].start_ms == 1000

    def test_a_word_the_recogniser_missed_is_timed_between_its_neighbours(self):
        words = A.script_words(SCRIPT)
        dropped = next(i for i, w in enumerate(words) if w.text == "cover")
        timings = A.align_words(words, hear(SCRIPT, drop={dropped}), 60.0)
        assert not timings[dropped].matched
        before, after = timings[dropped - 1], timings[dropped + 1]
        assert before.end_ms <= timings[dropped].start_ms
        assert timings[dropped].end_ms <= after.start_ms
        assert all(t.matched for i, t in enumerate(timings) if i != dropped)

    def test_a_long_stretch_nobody_heard_is_spread_across_the_gap(self):
        words = A.script_words(SCRIPT)
        gone = set(range(8, 14))
        timings = A.align_words(words, hear(SCRIPT, drop=gone), 60.0)
        starts = [timings[i].start_ms for i in sorted(gone)]
        assert starts == sorted(starts)
        assert starts[0] >= timings[7].end_ms
        assert timings[14].start_ms >= max(timings[i].end_ms for i in gone)
        assert not any(timings[i].matched for i in gone)

    def test_the_recogniser_hearing_extra_words_does_not_shift_the_script(self):
        asr = hear(SCRIPT)
        asr.insert(3, A.AsrWord("um", asr[2].end, asr[2].end + 0.05))
        timings = A.align_words(A.script_words(SCRIPT), asr, 60.0)
        assert all(t.matched for t in timings)

    def test_timings_never_go_backwards_or_past_the_audio(self):
        words = A.script_words(SCRIPT)
        timings = A.align_words(words, hear(SCRIPT, drop={2, 5, 9}), 14.0)
        for a, b in zip(timings, timings[1:]):
            assert a.start_ms <= b.start_ms
            assert a.end_ms <= b.start_ms or a.end_ms == a.start_ms
        assert all(0 <= t.start_ms <= t.end_ms <= 14000 for t in timings)

    def test_nothing_heard_still_times_every_word(self):
        words = A.script_words(SCRIPT)
        timings = A.align_words(words, [], 30.0)
        assert len(timings) == len(words)
        # A bare dash has no sound to find, so it never counts as unheard.
        assert not any(t.matched for t, w in zip(timings, words) if w.units)
        assert timings[0].start_ms == 0
        assert timings[-1].end_ms <= 30000


class TestBuildTimeline:
    def build(self, asr, **kwargs):
        return A.build_timeline(
            SCRIPT, asr, 60.0, engine="test", script_sha256="abc", **kwargs
        )

    def test_one_entry_per_spoken_line_keyed_by_splitlines_index(self):
        timeline = self.build(hear(SCRIPT))
        assert [entry["line"] for entry in timeline["lines"]] == [0, 3, 5]
        assert timeline["version"] == A.TIMELINE_VERSION
        assert timeline["script_sha256"] == "abc"
        assert timeline["word_level"] is True
        first = timeline["lines"][0]
        assert first["start_ms"] == first["words"][0][0]
        assert first["end_ms"] == max(w[1] for w in first["words"])
        assert len(first["words"]) == len(SCRIPT.splitlines()[0].replace("[HOST]:", "").split())

    def test_is_plain_json(self):
        assert json.loads(json.dumps(self.build(hear(SCRIPT))))["matched_ratio"] == 1.0

    def test_flags_the_words_nobody_heard(self):
        timeline = self.build(hear(SCRIPT, drop={2}))
        flags = [w[2] for w in timeline["lines"][0]["words"]]
        assert flags[2] == 0 and sum(flags) == len(flags) - 1

    def test_falls_back_to_line_timing_when_few_words_were_heard(self):
        words = A.script_words(SCRIPT)
        drop = set(range(0, len(words), 3))
        timeline = self.build(hear(SCRIPT, drop=drop))
        assert A.MIN_MATCHED_RATIO <= timeline["matched_ratio"] < A.WORD_LEVEL_MATCHED_RATIO
        assert timeline["word_level"] is False
        assert all("words" not in entry for entry in timeline["lines"])
        assert all(entry["start_ms"] <= entry["end_ms"] for entry in timeline["lines"])

    def test_refuses_audio_that_is_not_this_script(self):
        stranger = [A.AsrWord(w, i * 0.4, i * 0.4 + 0.3) for i, w in enumerate("completely unrelated words about cooking pasta tonight".split())]
        with pytest.raises(A.AlignmentError, match="same episode"):
            self.build(stranger)

    def test_refuses_silence_and_empty_scripts(self):
        with pytest.raises(A.AlignmentError, match="no speech"):
            self.build([])
        with pytest.raises(A.AlignmentError, match="no spoken words"):
            A.build_timeline("[PAUSE]\n", hear(SCRIPT), 10.0, engine="t", script_sha256="x")
