"""Practice drill sheets become paper-less podcast scripts."""

import pytest

from nlp_pillars.drills import (
    DRILL_SOURCE_KIND,
    DrillSheetError,
    build_drill_row,
    drill_row_from_sheet,
    parse_drill_sheet,
    sheet_to_script,
    spoken_form,
    spoken_word_count,
)
from nlp_pillars.tts.cue_parser import (
    SilenceGap,
    parse_podcast_script,
    plan_synthesis_chunks,
)

SHEET = """# Day 5 drill: Graph methods for money laundering

Intro: Day five, part two. Graph methods for money laundering. Two questions.

## Q1. How would you detect round-tripping?

**Answer.** Round-tripping is a cycle. The PR-AUC went from 0.42 to 1.0
in my workshop. The second paragraph is still part of the answer.

**Key points.** Cycles. Strongly connected components. Amount similarity.

## Q2. Would you use a graph neural network?

**Answer.** Start with boosted trees, then try a GNN when the data shows a gap.

**Key points.** Trees first. GNNs for multi-hop structure.
"""


def test_parse_reads_the_title_intro_and_every_question():
    sheet = parse_drill_sheet(SHEET, slug="d5")
    assert sheet.title == "Day 5 drill: Graph methods for money laundering"
    assert sheet.intro.startswith("Day five, part two.")
    assert [q.number for q in sheet.questions] == [1, 2]
    assert "second paragraph" in sheet.questions[0].answer
    assert sheet.questions[1].key_points == "Trees first. GNNs for multi-hop structure."


def test_spoken_form_keeps_the_build_tool_substitutions():
    assert spoken_form("PR-AUC") == "P R A U C"
    assert spoken_form("LOF") == "L O F"
    assert spoken_form("0.905") == "zero point nine zero five"
    assert spoken_form("9,880") == "9880"


def test_script_carries_the_five_and_eight_second_pauses():
    script = sheet_to_script(parse_drill_sheet(SHEET))
    lines = script.splitlines()
    assert any(line.startswith("[HOST]: Day five") for line in lines)
    assert "[HOST]: Question one. How would you detect round-tripping?" in lines
    assert "[HOST]: Question two. Would you use a graph neural network?" in lines
    assert "[PAUSE 5s]" in lines
    assert "[PAUSE 8s]" in lines
    assert "[PAUSE 1.5s]" in lines
    assert lines[-1] == "[HOST]: End of drill."


def test_the_timed_pauses_reach_the_audio_timeline():
    parsed = parse_podcast_script(sheet_to_script(parse_drill_sheet(SHEET)))
    pauses = [
        item.duration_ms
        for item in parsed.timeline
        if isinstance(item, SilenceGap) and item.cue == "pause"
    ]
    assert pauses.count(5000) == 2
    assert pauses.count(8000) == 2
    assert pauses.count(1500) == 3
    plans = plan_synthesis_chunks(parsed)
    assert any(plan.pause_after_ms == 5000 for plan in plans)
    assert any(plan.pause_after_ms == 8000 for plan in plans)
    assert all("PAUSE" not in plan.text.upper() for plan in plans)


def test_build_drill_row_marks_it_as_a_drill_and_counts_spoken_words():
    sheet = parse_drill_sheet(SHEET)
    row = drill_row_from_sheet(sheet, pillar_id="ml-systems-production")
    assert row.paper_id is None
    assert row.pillar_id == "ml-systems-production"
    assert row.title == sheet.title
    assert row.source_material.kind == DRILL_SOURCE_KIND
    assert row.word_count == spoken_word_count(row.script)
    assert row.word_count > 20
    assert row.key_points == [q.question for q in sheet.questions]


def test_build_drill_row_rejects_blank_required_fields():
    with pytest.raises(ValueError):
        build_drill_row(title=" ", pillar_id="p", script="[HOST]: hello")
    with pytest.raises(ValueError):
        build_drill_row(title="T", pillar_id=" ", script="[HOST]: hello")
    with pytest.raises(ValueError):
        build_drill_row(title="T", pillar_id="p", script="  \n ")


def test_parse_rejects_a_sheet_with_no_questions():
    with pytest.raises(DrillSheetError):
        parse_drill_sheet("# Title\n\nIntro: hello.\n")
