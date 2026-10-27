"""Output conversion, ANSI stripping and the output size cap. No kernel needed."""

from playground_runner.outputs import (
    ERROR_TRACEBACK_MAX_BYTES,
    OutputCollector,
    strip_ansi,
)


def test_strip_ansi_removes_colour_cursor_and_osc_sequences():
    coloured = "\x1b[0;31mZeroDivisionError\x1b[0m: \x1b[1mdivision\x1b[22m by zero\x1b[2K\x1b]8;;http://x\x07link\x1b]8;;\x07"
    assert strip_ansi(coloured) == "ZeroDivisionError: division by zero" + "link"
    assert strip_ansi("plain\ntext") == "plain\ntext"
    assert strip_ansi("") == ""


def test_error_traceback_is_plain_text_without_ansi():
    c = OutputCollector(10_000)
    c.add_message(
        "error",
        {
            "ename": "ValueError",
            "evalue": "\x1b[31mbad\x1b[0m",
            "traceback": ["\x1b[0;31m-----\x1b[0m", "\x1b[0;32mCell In[3], line 1\x1b[0m", "ValueError: bad"],
        },
    )
    (out,) = c.result()["outputs"]
    assert out == {
        "type": "error",
        "ename": "ValueError",
        "evalue": "bad",
        "traceback": "-----\nCell In[3], line 1\nValueError: bad",
    }
    assert "\x1b" not in out["traceback"]


def test_streams_merge_per_name_and_strip_ansi():
    c = OutputCollector(10_000)
    c.add_message("stream", {"name": "stdout", "text": "a\n"})
    c.add_message("stream", {"name": "stdout", "text": "\x1b[32mb\x1b[0m\n"})
    c.add_message("stream", {"name": "stderr", "text": "warn\n"})
    c.add_message("stream", {"name": "stdout", "text": "c\n"})
    assert c.result()["outputs"] == [
        {"type": "stream", "name": "stdout", "text": "a\nb\n"},
        {"type": "stream", "name": "stderr", "text": "warn\n"},
        {"type": "stream", "name": "stdout", "text": "c\n"},
    ]


def test_rich_outputs_prefer_png_then_html_then_plain():
    c = OutputCollector(1_000_000)
    c.add_message("display_data", {"data": {"image/png": "iVBOR\nw0K", "text/plain": "<Figure>"}})
    c.add_message("execute_result", {"data": {"text/html": ["<table>", "</table>"], "text/plain": "df"}})
    c.add_message("execute_result", {"data": {"text/plain": "42"}})
    c.add_message("display_data", {"data": {"application/json": {}}})  # nothing we render
    assert c.result()["outputs"] == [
        {"type": "image", "mime": "image/png", "data": "iVBORw0K", "text": "<Figure>"},
        {"type": "html", "html": "<table></table>", "text": "df", "untrusted": True},
        {"type": "text", "text": "42"},
    ]


def test_clear_output_with_and_without_wait():
    c = OutputCollector(10_000)
    c.add_message("stream", {"name": "stdout", "text": "old"})
    c.add_message("clear_output", {"wait": False})
    assert c.result()["outputs"] == []
    c.add_message("stream", {"name": "stdout", "text": "kept until next"})
    c.add_message("clear_output", {"wait": True})
    assert len(c.result()["outputs"]) == 1
    c.add_message("stream", {"name": "stdout", "text": "new"})
    assert c.result()["outputs"] == [{"type": "stream", "name": "stdout", "text": "new"}]


def test_stream_past_the_cap_is_cut_and_marked():
    c = OutputCollector(100)
    c.add_message("stream", {"name": "stdout", "text": "x" * 60})
    c.add_message("stream", {"name": "stdout", "text": "y" * 60})
    c.add_message("stream", {"name": "stdout", "text": "z" * 10})
    result = c.result()
    (out,) = result["outputs"]
    assert out["truncated"] is True
    assert out["text"].startswith("x" * 60 + "y" * 40)
    assert "output truncated" in out["text"]
    assert "z" not in out["text"]
    assert result["truncated"] is True
    assert result["dropped_outputs"] == 1  # the z chunk arrived after the cap was reached


def test_outputs_after_the_cap_are_dropped_and_counted():
    c = OutputCollector(50)
    c.add_message("execute_result", {"data": {"text/plain": "a" * 50}})
    c.add_message("execute_result", {"data": {"text/plain": "b"}})
    c.add_message("display_data", {"data": {"image/png": "A" * 10}})
    result = c.result()
    assert [o["text"] for o in result["outputs"]] == ["a" * 50]
    assert result["truncated"] is True
    assert result["dropped_outputs"] == 2


def test_an_image_that_does_not_fit_falls_back_to_its_text():
    c = OutputCollector(1_000)
    c.add_message("display_data", {"data": {"image/png": "A" * 5_000, "text/plain": "<Figure size 640x480>"}})
    (out,) = c.result()["outputs"]
    assert out["type"] == "text"
    assert out["text"].startswith("<Figure size 640x480>")
    assert out["truncated"] is True


def test_truncation_never_splits_a_utf8_character():
    c = OutputCollector(5)
    c.add_message("stream", {"name": "stdout", "text": "ééé"})  # 6 bytes
    text = c.result()["outputs"][0]["text"]
    assert text.startswith("éé")
    text.encode("utf-8")  # still valid


def test_errors_are_kept_past_the_cap_with_a_bounded_traceback():
    c = OutputCollector(10)
    c.add_message("stream", {"name": "stdout", "text": "x" * 10})
    huge = ["line " + "y" * 100] * 2_000
    c.add_message("error", {"ename": "RecursionError", "evalue": "too deep", "traceback": huge})
    outputs = c.result()["outputs"]
    assert outputs[-1]["ename"] == "RecursionError"
    assert outputs[-1]["truncated"] is True
    assert len(outputs[-1]["traceback"].encode()) <= ERROR_TRACEBACK_MAX_BYTES + 50


def test_remove_errors_drops_only_the_named_kind():
    c = OutputCollector(1_000)
    c.add_error("KeyboardInterrupt", "")
    c.add_error("ValueError", "bad")
    c.remove_errors("KeyboardInterrupt")
    assert [o["ename"] for o in c.result()["outputs"]] == ["ValueError"]
    assert not c.has_error("KeyboardInterrupt")
