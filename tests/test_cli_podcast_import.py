"""`nlp_pillars.cli podcast import` turns a drill sheet into a stored row."""

from unittest.mock import patch

from typer.testing import CliRunner

from nlp_pillars.cli import app

SHEET = """# Day 2 drill: Metrics

Intro: Day two. Metrics. One question.

## Q1. What does precision mean?

**Answer.** Of the alerts you raised, the share that were right.

**Key points.** Precision is about the alerts. Recall is about the cases.
"""


def test_podcast_import_reads_the_sheet_and_saves_a_drill(tmp_path):
    sheet = tmp_path / "d2-metrics.md"
    sheet.write_text(SHEET, encoding="utf-8")
    saved = {}

    def add(row):
        saved["row"] = row
        return "22222222-3333-4444-8555-666666666666"

    with (
        patch(
            "nlp_pillars.cli.get_valid_pillars",
            return_value=["ml-systems-production"],
        ),
        patch("nlp_pillars.db.add_podcast_script", side_effect=add),
    ):
        result = CliRunner().invoke(
            app,
            ["podcast", "import", str(sheet), "--pillar", "ml-systems-production"],
        )

    assert result.exit_code == 0, result.output
    assert "Imported practice drill" in result.output
    assert "22222222-3333-4444-8555-666666666666" in result.output
    row = saved["row"]
    assert row.title == "Day 2 drill: Metrics"
    assert row.paper_id is None
    assert row.source_material.kind == "drill"
    assert "[PAUSE 5s]" in row.script
    assert "[PAUSE 8s]" in row.script


def test_podcast_import_title_override_and_bad_sheet(tmp_path):
    sheet = tmp_path / "d2-metrics.md"
    sheet.write_text(SHEET, encoding="utf-8")
    with (
        patch(
            "nlp_pillars.cli.get_valid_pillars",
            return_value=["ml-systems-production"],
        ),
        patch("nlp_pillars.db.add_podcast_script", return_value="id"),
    ):
        result = CliRunner().invoke(
            app,
            [
                "podcast", "import", str(sheet),
                "--pillar", "ml-systems-production",
                "--title", "Interview drill",
            ],
        )
    assert result.exit_code == 0
    assert "Interview drill" in result.output

    broken = tmp_path / "broken.md"
    broken.write_text("# Only a title\n", encoding="utf-8")
    result = CliRunner().invoke(
        app, ["podcast", "import", str(broken), "--pillar", "ml-systems-production"]
    )
    assert result.exit_code == 1
    assert "missing a title" in result.output
