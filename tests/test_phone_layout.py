"""Phone layout contracts for the Today streak and the mini player.

Geometry is checked with screenshots. These assertions lock the rules those
shots depend on: seven streak days stay on one line, and the phone player is
a single row until it is expanded.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "webui/static/app.css").read_text(encoding="utf-8")
HTML = (ROOT / "webui/templates/base.html").read_text(encoding="utf-8")


def test_streak_days_do_not_wrap_on_a_phone():
    assert ".streak-dots" in CSS
    dots = CSS.split(".streak-dots {", 1)[1].split("}", 1)[0]
    assert "flex-wrap: nowrap" in dots
    assert "flex-wrap: wrap" not in dots
    phone = CSS.split("@media (max-width: 859px) {", 1)[-1]
    # The streak block is its own media query, after the base rule.
    streak_phone = CSS.split(".streak { min-width: 0; }", 1)[1]
    assert "flex: 1 1 0" in streak_phone
    assert phone  # the phone breakpoint still exists


def test_phone_mini_player_is_one_row_until_expanded():
    assert 'id="mini-player-expand"' in HTML
    assert "mini-skip" in HTML
    player = CSS.split("@media (max-width: 859px) {\n  .mini-player:not([hidden]) {", 1)[1]
    assert "flex-wrap: nowrap" in player.split("}", 1)[0]
    assert ".mini-player:not(.is-expanded) .mini-skip" in CSS
    assert ".mini-player.is-expanded" in CSS
    assert "bottom: calc(59px + env(safe-area-inset-bottom))" in player


def test_playground_cards_scroll_inside_themselves_on_a_phone():
    # Code, outputs and diffs scroll sideways inside their box; the page never
    # does. min-width: 0 lets the grid shrink below the longest code line.
    for rule in (".pg-code {", ".pg-diff {", ".pg-out {"):
        block = CSS.split(rule, 1)[1].split("}", 1)[0]
        assert "overflow-x: auto" in block or "overflow: auto" in block
    for rule in (".pg-root {", ".pg-card {", ".pg-step {", ".pg-outputs {"):
        block = CSS.split(rule, 1)[1].split("}", 1)[0]
        # An auto grid track grows to the longest code line; this one cannot.
        assert "grid-template-columns: minmax(0, 1fr)" in block
        assert "min-width: 0" in block
    assert "max-width: 100%" in CSS.split(".pg-out-img img {", 1)[1].split("}", 1)[0]
    phone = CSS.split("@media (max-width: 859px) {\n  .pg-card", 1)[1].split("\n}", 1)[0]
    assert ".pg-import-row .btn { min-height: 40px; }" in phone
    # The tab is highlighted like the others when it is the target.
    assert '.paper-page:has(.panel-playground:target) .tabs a[data-panel="playground"]' in CSS


def test_live_playground_controls_fit_a_phone():
    # Before / After sits side by side on a desktop and stacks on a phone; both
    # columns can shrink below their longest output line.
    ba = CSS.split(".pg-ba {", 1)[1].split("}", 1)[0]
    assert "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)" in ba
    assert "min-width: 0" in ba
    for rule in (".pg-ba-col {", ".pg-try-result {", ".pg-compare {", ".pg-saved,\n.pg-live {"):
        block = CSS.split(rule, 1)[1].split("}", 1)[0]
        assert "grid-template-columns: minmax(0, 1fr)" in block
        assert "min-width: 0" in block
    # The editor scrolls inside itself; its text column may shrink.
    editor = CSS.split(".pg-editor {", 1)[1].split("}", 1)[0]
    assert "grid-template-columns: auto minmax(0, 1fr)" in editor
    assert "overflow" in CSS.split(".pg-ta {", 1)[1].split("}", 1)[0]
    # Toolbars and the session bar wrap instead of pushing the page sideways.
    for rule in (".pg-session {", ".pg-toolbar {", ".pg-run-status {"):
        assert "flex-wrap: wrap" in CSS.split(rule, 1)[1].split("}", 1)[0]
    phone = CSS.split("@media (max-width: 859px) {\n  .pg-card", 1)[1].split("\n}", 1)[0]
    assert ".pg-ba { grid-template-columns: minmax(0, 1fr); }" in phone
    assert ".pg-editor { font-size: 16px; }" in phone  # no focus zoom
    for selector in (".pg-session .btn", ".pg-toolbar .btn", ".pg-run-status .btn", ".pg-try-notice .btn"):
        assert selector in phone
    assert "min-height: 40px" in phone
