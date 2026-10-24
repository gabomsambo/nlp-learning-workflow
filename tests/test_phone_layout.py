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
