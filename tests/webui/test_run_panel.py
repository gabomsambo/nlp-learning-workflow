"""The shared run card keeps the DOM contract the pages already poll through.

Colour strings in run-progress.js became class names; the card itself is markup
and CSS. Page sources pass ids into the macro, and rendering the macro is what
proves those ids land on the elements RunProgress looks up.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "webui" / "templates"
ENV = Environment(loader=FileSystemLoader(TEMPLATES))


def _text(name):
    return (TEMPLATES / name).read_text()


def _render(source):
    return ENV.from_string(source).render()


def test_partial_renders_inline_and_floating_variants():
    inline = _render(
        "{% from '_run_panel.html' import run_panel %}"
        "{{ run_panel(status_id='run-status', stages_id='run-progress',"
        " progress_id='run-bar', cancel_id='cancel-btn', panel_id='run-panel',"
        " title='Run', title_id='run-title', hidden=True) }}"
    )
    assert 'id="run-panel"' in inline
    assert 'id="run-title"' in inline
    assert 'id="run-status"' in inline
    assert 'role="status"' in inline
    assert 'aria-live="polite"' in inline
    assert 'id="run-bar"' in inline
    assert 'id="run-progress"' in inline
    assert 'aria-live="off"' in inline
    assert 'id="cancel-btn"' in inline
    assert "hidden" in inline
    assert "is-floating" not in inline
    assert "Show steps" in inline
    assert "You can keep working" in inline

    floating = _render(
        "{% from '_run_panel.html' import run_panel %}"
        "{{ run_panel(status_id='s', stages_id='g', progress_id='p',"
        " cancel_id='c', floating=True) }}"
    )
    assert "is-floating" in floating

    with_queries = _render(
        "{% from '_run_panel.html' import run_panel %}"
        "{% call run_panel(status_id='s', stages_id='g', progress_id='p', cancel_id='c') %}"
        "<div id='queries-panel'></div>{% endcall %}"
    )
    assert 'id="queries-panel"' in with_queries or "id='queries-panel'" in with_queries


def test_pages_that_show_runs_pass_their_element_ids():
    expected = {
        "discovery.html": [
            "run_panel",
            "panel_id='run-panel'",
            "title_id='run-title'",
            "status_id='run-status'",
            "progress_id='run-bar'",
            "stages_id='run-progress'",
            "cancel_id='cancel-btn'",
            'id="queries-panel"',
        ],
        "pipeline.html": [
            "run_panel",
            "status_id='status'",
            "stages_id='progress'",
            "progress_id='run-bar'",
            "cancel_id='cancel-btn'",
            "floating=True",
        ],
        "podcast.html": [
            "run_panel",
            "stages_id='script-stages'",
            "progress_id='progress-bar'",
            "cancel_id='cancel-script-btn'",
            "status_id='audio-run-status'",
            "stages_id='audio-stages'",
            "progress_id='audio-progress-bar'",
            "cancel_id='cancel-audio-btn'",
        ],
        "pillar_detail.html": [
            "run_panel",
            "status_id='upload-run-status'",
            "progress_id='upload-run-bar'",
            "stages_id='upload-run-stages'",
            "cancel_id='upload-cancel-btn'",
            'id="upload-result"',
        ],
    }
    for name, needles in expected.items():
        body = _text(name)
        for needle in needles:
            assert needle in body, f"{name} lost {needle}"


def test_run_card_styles_use_tokens():
    css = (ROOT / "webui" / "static" / "app.css").read_text()
    start = css.index("/* Run card")
    block = css[start:]
    assert ".run-card.is-floating" in block
    assert ".run-segs" in block
    assert ".run-stage-failed" in block
    assert 'html[data-player="on"] .run-card.is-floating' in block
    assert "overflow: auto" in block
    assert "100dvh - 32px" not in block
    assert "#" not in block


def test_renderer_no_longer_paints_inline_colours():
    js = (ROOT / "webui" / "static" / "run-progress.js").read_text()
    assert "STAGE_COLOR" not in js
    assert "--pico-progress-color" not in js
    assert "darkorange" not in js
    assert "crimson" not in js
    assert ".style.color" not in js
