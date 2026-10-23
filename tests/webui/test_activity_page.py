"""Activity page and the shell chip: markup contracts, not a live stack."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "webui" / "templates"
STATIC = ROOT / "webui" / "static"


def _text(name: str) -> str:
    return (TEMPLATES / name).read_text()


def test_activity_page_keeps_filters_status_and_leaves_errors_closed():
    page = _text("activity.html")
    assert 'id="kind-filters"' in page
    assert 'data-kind="upload"' in page
    assert 'id="activity-pillar"' in page
    assert 'id="status"' in page
    assert 'id="activity-list"' in page
    assert "Technical detail" not in page  # the control is created when a run has an error
    script = (STATIC / "activity.js").read_text()
    assert "No runs recorded yet." in script
    assert "Could not load run history:" in script
    assert "Technical detail" in script
    assert ".attach(" not in script
    assert "localStorage" not in script
    chip = (STATIC / "status-chip.js").read_text()
    assert ".attach(" not in chip
    assert "localStorage" not in chip
    assert "/api/system/status" in chip


def test_pipeline_keeps_the_run_form_and_points_history_at_activity():
    page = _text("pipeline.html")
    assert "Run Pipeline" in page
    assert 'href="/activity/"' in page
    assert "history-body" not in page
    assert "Recent runs" not in page
    assert "run_panel" in page
    assert "floating=True" in page


def test_shell_activity_link_and_chip_do_not_follow_a_run():
    base = _text("base.html")
    assert "nav_link('/activity/', 'Activity'" in base
    assert "nav_link('/pipeline/', 'Activity'" not in base
    assert 'class="status-chip status-chip-shell"' in base
    assert "/static/status-chip.js" in base
    chip = (STATIC / "status-chip.js").read_text()
    assert "querySelectorAll('.status-chip-shell')" in chip
    today = _text("dashboard.html")
    assert 'id="service-status-slot"' in today
    assert 'data-status-endpoint="/api/system/status"' in today
    assert 'href="/activity/#status"' in today
    assert 'href="/activity/">Activity</a>' in today
    assert 'href="/pipeline/">Activity</a>' not in today
    shell = (STATIC / "shell.js").read_text()
    assert "href: '/activity/'" in shell
    assert "href: '/pipeline/'" not in shell
