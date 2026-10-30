"""Table repair at ingestion (nlp_pillars/table_repair.py).

The fixtures are recorded from a live run on 2026-10-30: the text layer's
numerals for the table pages, the Haiku 5.5 structured response for each page,
and an excerpt of what pymupdf4llm extracted around the captions. No test calls
a model or opens a PDF.

  dsvl  arXiv:2403.05525 (DeepSeek-VL) page 20, Tables 8 and 9
  aml   arXiv:2306.16424 (AMLworld) pages 9 and 20, Tables 2, 3, 12, 13, 14
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nlp_pillars import table_repair as tr
from nlp_pillars.llm_client import LLMCallError, StructuredCall

FIXTURES = Path(__file__).parent / "fixtures" / "tables"


def _pages(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}_pages.json").read_text())["pages"]


def _excerpt(name: str) -> str:
    return (FIXTURES / f"{name}_excerpt.md").read_text()


class FakePages:
    """A PageSource over a recorded fixture. ``render`` returns the page index."""

    def __init__(self, pages: dict):
        self._pages = pages
        self.rendered: list[int] = []

    def captions(self):
        return {int(k): v["captions"] for k, v in self._pages.items()}

    def render(self, page: int) -> bytes:
        self.rendered.append(page)
        return str(page).encode()

    def page_numerals(self, page: int):
        return set(self._pages[str(page)]["numerals"])


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the model call with the recorded response for the rendered page."""

    def install(pages: dict):
        def fake(route, system, user, out_model, **kwargs):
            assert user[0]["type"] == "image"
            page = base64.b64decode(user[0]["source"]["data"]).decode()
            rec = pages[page]
            return StructuredCall(
                stage="table_repair",
                provider="anthropic",
                model=rec["model"],
                parsed=out_model(**rec["response"]),
                cost_usd=rec["cost_usd"],
            )

        monkeypatch.setattr(tr, "structured", fake)

    return install


def _run(name: str, fake_model, pages=None):
    pages = pages or _pages(name)
    fake_model(pages)
    return tr.repair_tables(FakePages(pages), _excerpt(name))


# -- the gate ------------------------------------------------------------------


def test_deepseek_vl_table_9_is_recovered_exactly(fake_model):
    before = _excerpt("dsvl")
    assert "✓ ✓ 59.4 54.2 61.4 82.5 29.2 57.4" in before  # the garbled form

    repair = _run("dsvl", fake_model)

    assert "| Stage 1 | Stage 2 | Stage 3 | MMB | MMC | SEED | POPE | MMMU | Average |\n" \
           "|---|---|---|---|---|---|---|---|---|\n" \
           "| yes |  | yes | 59.4 | 54.2 | 61.4 | 82.5 | 29.2 | 57.4 |\n" \
           "|  | yes | yes | 63.4 | 60.5 | 65.9 | 87.1 | 31.8 | 61.7 |\n" \
           "| yes | yes | yes | 64.3 | 61.3 | 66.7 | 87.6 | 32.2 | 62.4 |" in repair.text
    assert "✓" not in repair.text
    # the prose around the table is untouched, and the caption stays
    assert "Table 9 | Analysis of model performance across training stages." in repair.text
    assert "demonstrate that expanding the data scale" in repair.text
    by_label = {t.label: t for t in repair.tables}
    assert by_label["9"].status == tr.STATUS_SPLICED
    assert by_label["9"].agreement == 1.0


def test_aml_result_tables_get_their_cells_back(fake_model):
    before = _excerpt("aml")
    repair = _run("aml", fake_model)

    for label in ("2", "3", "13", "14"):
        table = next(t for t in repair.tables if t.label == label)
        cells = tr.cell_numerals(table.columns, table.rows)
        assert table.status == tr.STATUS_SPLICED
        assert tr.coverage(before, cells) < 0.5, f"Table {label} was not lost before"
        assert tr.coverage(repair.text, cells) >= 0.98, f"Table {label} after repair"
    assert "| GIN [65, 25] | 28.70 ± 1.13 |" in repair.text
    # captions and the surrounding prose survive
    assert "Table 13: Minority class recall rate (%)" in repair.text
    assert "XGBoost models trained on the respective LI datasets" in repair.text


def test_cost_and_agreement_are_reported(fake_model):
    repair = _run("aml", fake_model)
    assert repair.pages == 2 and repair.calls == 2 and repair.failed_pages == 0
    assert repair.cost_usd == pytest.approx(sum(p["cost_usd"] for p in _pages("aml").values()))
    assert repair.agreement == 1.0
    assert "spliced" in repair.detail() and "100.0% agreement" in repair.detail()


# -- below the agreement threshold: flagged, not spliced -----------------------


def _corrupt(pages: dict, fraction: float) -> dict:
    """Replace a share of the numeric cells with a number that is not on the page."""
    pages = copy.deepcopy(pages)
    for rec in pages.values():
        for table in rec["response"]["tables"]:
            for row in table["rows"]:
                for i, cell in enumerate(row):
                    if tr.numerals(cell) and (i % 10) < fraction * 10:
                        row[i] = "987.65"
    return pages


def test_a_paper_under_95_percent_is_flagged_and_its_text_is_untouched(fake_model):
    repair = _run("aml", fake_model, _corrupt(_pages("aml"), 0.3))

    assert repair.agreement < tr.AGREEMENT_THRESHOLD
    assert repair.text == _excerpt("aml")
    assert repair.spliced == 0
    assert {t.status for t in repair.tables} == {tr.STATUS_FLAGGED}
    assert all("under 95%" in t.reason for t in repair.tables)


def test_one_bad_table_in_a_good_paper_is_flagged_alone(fake_model):
    pages = copy.deepcopy(_pages("aml"))
    table13 = next(t for t in pages["19"]["response"]["tables"] if t["label"] == "13")
    # two wrong cells: 93% for the table, still 99% for the paper
    table13["rows"][0][1] = table13["rows"][1][1] = "987.65"
    repair = _run("aml", fake_model, pages)

    status = {t.label: t.status for t in repair.tables}
    assert status["13"] == tr.STATUS_FLAGGED
    assert status["2"] == status["14"] == tr.STATUS_SPLICED
    assert "987.65" not in repair.text


def test_a_table_with_too_few_numbers_is_not_trusted(fake_model):
    pages = copy.deepcopy(_pages("dsvl"))
    pages["19"]["response"]["tables"][1]["rows"] = [["a", "1"]]
    repair = _run("dsvl", fake_model, pages)
    t9 = next(t for t in repair.tables if t.label == "9")
    assert t9.status == tr.STATUS_UNVERIFIED
    assert "✓ ✓ 59.4" in repair.text


def test_a_table_whose_caption_is_not_in_the_text_is_unplaced(fake_model):
    fake_model(_pages("dsvl"))
    repair = tr.repair_tables(FakePages(_pages("dsvl")), "A paper with no table captions at all.")
    assert {t.status for t in repair.tables} == {tr.STATUS_UNPLACED}
    assert repair.text == "A paper with no table captions at all."


def test_a_splice_that_would_pass_the_cap_is_refused(fake_model):
    pages = _pages("aml")
    fake_model(pages)
    text = _excerpt("aml")
    repair = tr.repair_tables(FakePages(pages), text, refuse_over=len(text) + 50)
    assert any(t.status == tr.STATUS_TOO_LARGE for t in repair.tables)
    assert len(repair.text) <= len(text) + 50


def test_a_failed_page_call_is_counted_and_the_rest_carries_on(monkeypatch, fake_model):
    pages = _pages("aml")
    fake_model(pages)
    good = tr.structured

    def flaky(route, system, user, out_model, **kw):
        if base64.b64decode(user[0]["source"]["data"]) == b"8":
            raise LLMCallError("overloaded")
        return good(route, system, user, out_model, **kw)

    monkeypatch.setattr(tr, "structured", flaky)
    repair = tr.repair_tables(FakePages(pages), _excerpt("aml"))
    assert repair.failed_pages == 1 and repair.calls == 1
    assert {t.label for t in repair.tables} == {"12", "13", "14"}


# -- text surgery --------------------------------------------------------------


def test_filled_col1_tables_are_kept_and_empty_plot_tables_are_stripped():
    plot = "|Col1|Col2|\n|---|---|\n|||\n|||\n"
    real = "|Col1|A|\n|---|---|\n|x|1|\n|y|2|\n"
    text, removed = tr.strip_junk_tables(f"before\n{plot}\nmiddle\n{real}after")
    assert removed == 1
    assert "|||" not in text and "|x|1|" in text and "middle" in text


def test_pipes_in_cells_are_escaped_and_short_rows_padded():
    block = tr.render_markdown(["a", "b"], [["x|y"], ["1", "2", "3"]])
    assert block[2] == "| x\\|y |  |  |"
    assert block[3] == "| 1 | 2 | 3 |"


def test_caption_markup_does_not_hide_a_caption():
    record = tr.TableRecord(
        label="1", caption="c", page=1, columns=["A", "B"], rows=[["1", "2"]], model="m"
    )
    out = tr.splice_table("intro\n\n<u>Table 1:</u> <u>Comparison.</u>\n\nnext", record)
    assert "| A | B |" in out and "<u>Table 1:</u>" in out


def test_prose_next_to_a_caption_is_never_removed():
    record = tr.TableRecord(
        label="2", caption="c", page=1, columns=["Model", "F1"],
        rows=[["GIN", "28.70"]], model="m",
    )
    text = "Table 2: scores.\n\nGIN 28.70\nThe model improves F1 on every dataset we tried.\n"
    out = tr.splice_table(text, record)
    assert "The model improves F1 on every dataset we tried." in out
    assert "\nGIN 28.70\n" not in out


# -- the ingestion stage never raises ------------------------------------------


def test_stage_is_skipped_without_a_pdf_a_key_or_when_disabled(monkeypatch):
    settings = MagicMock(table_repair_enabled=True, anthropic_api_key="k")
    monkeypatch.setattr(tr, "get_settings", lambda: settings)
    assert tr.stage_outcome(None, "t").status == "skipped"
    settings.anthropic_api_key = None
    assert "ANTHROPIC_API_KEY" in tr.stage_outcome("/x.pdf", "t").detail
    settings.anthropic_api_key, settings.table_repair_enabled = "k", False
    assert "TABLE_REPAIR_ENABLED" in tr.stage_outcome("/x.pdf", "t").detail


def test_stage_failure_is_a_failed_stage_not_an_exception(monkeypatch):
    monkeypatch.setattr(
        tr, "get_settings", lambda: MagicMock(table_repair_enabled=True, anthropic_api_key="k")
    )

    def boom(path):
        raise RuntimeError("not a PDF")

    monkeypatch.setattr(tr, "PymupdfPages", boom)
    outcome = tr.stage_outcome("/x.pdf", "text")
    assert outcome.status == "failed" and "not a PDF" in outcome.detail
    assert not outcome.text_changed


def test_stage_completes_and_reports_a_changed_text(monkeypatch, fake_model):
    pages = _pages("dsvl")
    fake_model(pages)
    monkeypatch.setattr(
        tr, "get_settings", lambda: MagicMock(table_repair_enabled=True, anthropic_api_key="k")
    )
    fake = FakePages(pages)
    fake.close = lambda: None
    monkeypatch.setattr(tr, "PymupdfPages", lambda path: fake)
    outcome = tr.stage_outcome("/x.pdf", _excerpt("dsvl"))
    assert outcome.status == "completed" and outcome.text_changed
    assert tr.tagged_extractor("pymupdf4llm", True) == "pymupdf4llm+tables"
    assert tr.tagged_extractor("pymupdf4llm+tables", True) == "pymupdf4llm+tables"


# -- stored papers: re-chunk, re-embed, re-project -------------------------------


def _paper():
    return MagicMock(id="2306.16424", url_pdf="https://arxiv.org/pdf/2306.16424", pillar_id="p")


@pytest.fixture
def stored_paper_env(monkeypatch, fake_model):
    from nlp_pillars import db, vectors

    pages = _pages("aml")
    fake_model(pages)
    fake = FakePages(pages)
    fake.close = lambda: None
    calls: list[str] = []
    monkeypatch.setattr("nlp_pillars.tools.pdf_loader.download_pdf", lambda url, **kw: "/x.pdf")
    monkeypatch.setattr(db, "get_stored_paper_text", lambda pid: _excerpt("aml"))
    monkeypatch.setattr(db, "get_stored_paper_extractor", lambda pid: "pymupdf4llm")

    def replace(pid, text, extractor):
        calls.append(f"text:{extractor}")
        replace.text = text
        return 4

    monkeypatch.setattr(db, "replace_paper_text", replace)
    monkeypatch.setattr(db, "embed_paper_chunks", lambda pid: calls.append("embed") or 4)
    monkeypatch.setattr(db, "replace_paper_tables", lambda pid, rows: calls.append(f"tables:{len(rows)}"))
    monkeypatch.setattr(vectors, "ensure_collections", lambda: None)
    monkeypatch.setattr(vectors, "upsert_text", lambda *a: calls.append("project") or 4)
    return fake, calls, replace


def test_a_repair_rechunks_reembeds_and_reprojects_in_that_order(stored_paper_env):
    fake, calls, replace = stored_paper_env
    report = tr.repair_stored_paper(_paper(), dry_run=False, open_pages=lambda p: fake)

    assert report.status == "repaired"
    assert calls == ["text:pymupdf4llm+tables", "embed", "project", "tables:5"]
    assert report.chunks == 4 and report.embeddings == 4
    assert "28.70 ± 1.13" in replace.text


def test_the_repaired_text_chunks_with_its_table_numbers_in_it(stored_paper_env):
    from nlp_pillars.paper_text import build_chunk_rows

    fake, _, replace = stored_paper_env
    tr.repair_stored_paper(_paper(), dry_run=False, open_pages=lambda p: fake)
    rows = build_chunk_rows(replace.text)
    joined = "\n".join(r["text"] for r in rows)
    assert "28.70 ± 1.13" in joined and "36.60 ± 0.74" in joined


def test_dry_run_reads_the_pages_and_writes_nothing(stored_paper_env):
    fake, calls, _ = stored_paper_env
    report = tr.repair_stored_paper(_paper(), dry_run=True, open_pages=lambda p: fake)
    assert report.status == "repaired" and calls == []
    out = report.render(dry_run=True)
    assert "table 2 (page 9): spliced" in out and "dry_run: true" in out


def test_a_flagged_paper_stores_its_tables_but_not_its_text(stored_paper_env, fake_model):
    fake, calls, _ = stored_paper_env
    bad = _corrupt(_pages("aml"), 0.3)
    fake_model(bad)
    fake._pages = bad
    report = tr.repair_stored_paper(_paper(), dry_run=False, open_pages=lambda p: fake)
    assert report.status == "flagged"
    assert calls == ["tables:5"]


def test_a_paper_without_a_pdf_is_skipped():
    paper = MagicMock(id="x", url_pdf=None)
    assert tr.repair_stored_paper(paper, dry_run=True).status == "skipped"


# -- the command ----------------------------------------------------------------


def test_repair_tables_needs_exactly_one_of_paper_or_all():
    from typer.testing import CliRunner

    from nlp_pillars.cli import app

    runner = CliRunner()
    assert runner.invoke(app, ["text", "repair-tables"]).exit_code == 2
    assert runner.invoke(app, ["text", "repair-tables", "--paper", "x", "--all"]).exit_code == 2


def test_all_dry_run_counts_and_makes_no_model_call(monkeypatch):
    from typer.testing import CliRunner

    from nlp_pillars import db
    from nlp_pillars.cli import app

    monkeypatch.setattr(db, "list_all_papers", lambda: [MagicMock(id="a"), MagicMock(id="b")])
    monkeypatch.setattr(db, "list_repaired_table_paper_ids", lambda: ["a"])
    monkeypatch.setattr(
        tr, "repair_stored_paper", lambda *a, **k: pytest.fail("dry run must not read pages")
    )
    result = CliRunner().invoke(app, ["text", "repair-tables", "--all", "--dry-run"])
    assert result.exit_code == 0
    assert "already_repaired: 1" in result.output and "would_read: 1" in result.output
