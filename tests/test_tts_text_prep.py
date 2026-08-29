"""Regression tests for IndexTTS text preparation and chunk-9 crash (2026-08-29)."""

from __future__ import annotations

import pytest

from nlp_pillars.tts.engine import ChunkSynthesisError
from nlp_pillars.tts.text_prep import (
    describe_degenerate_segment,
    find_degenerate_segments,
    is_degenerate_segment,
    prepare_text_for_indextts,
)

# Captain's chunk 9 from script 57637815-dddf-420e-abe7-b10d0f2eb86a (78-chunk run).
CAPTAIN_CHUNK_9 = (
    'So the natural question is: do MG and MCG actually generate the same languages? '
    "Are they equivalent in their generative power? That's the question this paper "
    'begins to address. And I emphasize "begins"—because, as the author explicitly '
    'states, this work represents "first steps towards giving a proof of inclusion of '
    'their generated languages."'
)

# Live /on_input_text_change output on 2026-08-29 before prepare_text.
CAPTAIN_CHUNK_9_RAW_SEGMENTS = [
    (
        0,
        "▁SO▁THE▁NATURAL▁QUESTION▁IS,▁DO▁MG▁AND▁MCG▁ACTUALLY▁GENERATE▁THE▁SAME▁LANGUAGES?"
        "▁ARE▁THEY▁EQUIVALENT▁IN▁THEIR▁GENERATIVE▁POWER?▁THAT▁IS▁THE▁QUESTION▁THIS▁PAPER▁"
        "BEGINS▁TO▁ADDRESS.▁AND▁I▁EMPHASIZE▁'BEGINS'-BECAUSE,▁AS▁THE▁AUTHOR▁EXPLICITLY▁"
        "STATES,▁THIS▁WORK▁REPRESENTS▁'FIRST▁STEPS▁TOWARDS▁GIVING▁A▁PROOF▁OF▁INCLUSION▁"
        "OF▁THEIR▁GENERATED▁LANGUAGES.'",
        120,
    ),
    (1, "'", 1),
]


def test_prepare_text_replaces_em_dash_that_triggers_indextts_split():
    prepared = prepare_text_for_indextts(CAPTAIN_CHUNK_9)
    assert "\u2014" not in prepared
    assert '"begins", because' in prepared or '"begins", because,' in prepared


def test_captain_chunk_9_raw_segments_include_degenerate_quote():
    bad = find_degenerate_segments(CAPTAIN_CHUNK_9_RAW_SEGMENTS)
    assert len(bad) == 1
    assert bad[0][0] == 1
    assert bad[0][2] == 1


def test_is_degenerate_segment_rejects_lone_punctuation_token():
    assert is_degenerate_segment("'", 1)
    assert not is_degenerate_segment(
        CAPTAIN_CHUNK_9_RAW_SEGMENTS[0][1],
        CAPTAIN_CHUNK_9_RAW_SEGMENTS[0][2],
    )


def test_describe_degenerate_segment_names_segment_index():
    msg = describe_degenerate_segment(1, "'", 1)
    assert "segment 2" in msg
    assert "punctuation-only" in msg


def test_chunk_synthesis_error_carries_index_and_reason():
    err = ChunkSynthesisError(9, 78, "codec input size zero")
    assert err.chunk_index == 9
    assert err.chunk_total == 78
    assert "chunk 9/78 failed" in str(err)
    assert "codec input size zero" in str(err)


@pytest.mark.integration
def test_captain_chunk_9_succeeds_after_prepare_on_live_indextts():
    """Live gate: requires IndexTTS on localhost:7860."""
    from pathlib import Path

    from nlp_pillars.tts.indextts_client import IndexTtsClient

    client = IndexTtsClient("http://127.0.0.1:7860", Path("/tmp/tts_test"))
    if client.check_status().status.value != "ready":
        pytest.skip("IndexTTS not running")

    prepared = prepare_text_for_indextts(CAPTAIN_CHUNK_9)
    segments = client.preview_segments(prepared)
    assert find_degenerate_segments(segments) == []
    assert len(segments) == 1

    voice = "/home/gabo/Desktop/indexttsvoices/FernandaRamirez.L.wav"
    if not Path(voice).is_file():
        pytest.skip("captain voice reference not mounted")

    wav = client.synthesize(CAPTAIN_CHUNK_9, voice, job_timeout=120.0)
    assert wav.endswith(".wav")
