"""Paper Playground: notebook import, output safety, dataset cache, explain pass.

Hermetic. The fixture is a small synthetic notebook written for these tests;
no network, database or model is reached.
"""

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from nlp_pillars.llm_client import StructuredCall
from nlp_pillars.orchestrator import RunCancelledError
from nlp_pillars.playground import MIGRATION, store
from nlp_pillars.playground import service as pg_service
from nlp_pillars.playground.datasets import (
    MANIFEST_NAME,
    DatasetCache,
    DatasetError,
    file_name_for,
    find_dataset_urls,
    read_manifest,
)
from nlp_pillars.playground.explain import (
    SYSTEM_PROMPT,
    CellDraft,
    ExplainDraft,
    LineNoteDraft,
    StepDraft,
    TryDraft,
    make_batches,
    render_cell,
    validate_draft,
)
from nlp_pillars.playground.notebook import (
    MAX_HTML_NODES,
    NotebookError,
    cells_view,
    code_cell_indices,
    notebook_title,
    parse_notebook,
    sanitize_html,
    tree_text,
)
from nlp_pillars.schemas import PLAYGROUND_EXPLAIN_STAGES, RunStatus
from webui.services import run_service

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/playground/toy_regression.ipynb"
MIGRATION_SQL = (ROOT / MIGRATION).read_text()
SCHEMA = (ROOT / "schema.sql").read_text()
PG_ID = "6a1f6c1e-1111-4111-8111-111111111111"
CAP = 25_000_000


def _nb():
    return parse_notebook(FIXTURE.read_bytes(), max_bytes=CAP)


def _tags(nodes):
    found = []

    def walk(node):
        if isinstance(node, dict):
            found.append(node["t"])
            assert set(node) <= {"t", "a", "c"}
            for attr in node.get("a", {}):
                assert attr in ("colspan", "rowspan")
            for child in node.get("c", []):
                walk(child)

    for n in nodes:
        walk(n)
    return found


# -- notebook parsing -------------------------------------------------------------


def test_fixture_parses_and_keeps_every_cell():
    nb = _nb()
    assert len(nb["cells"]) == 8
    assert notebook_title(nb, "x") == "Fitting a line by gradient descent"
    # The whitespace-only code cell is not something to study.
    assert code_cell_indices(nb) == [1, 2, 4, 5, 7]


@pytest.mark.parametrize(
    "payload,needle",
    [
        (b"", "empty"),
        (b"not json at all", "not JSON"),
        (b"\xff\xfe\x00garbage", "not JSON"),
        (b'{"hello": "world"}', "no cell list"),
        (b'{"nbformat": 3, "cells": []}', "nbformat 4"),
        (b'{"nbformat": 4, "cells": [{"cell_type": "markdown", "source": "hi"}]}', "no code cells"),
        (b'{"nbformat": 4, "cells": [{"cell_type": "widget", "source": ""}]}', "Cell 0"),
        (b'{"nbformat": 4, "cells": [{"cell_type": "code", "source": 5}]}', "not text"),
    ],
)
def test_non_notebooks_are_rejected_with_a_reason(payload, needle):
    with pytest.raises(NotebookError, match=needle):
        parse_notebook(payload, max_bytes=CAP)


def test_oversized_notebook_is_rejected_and_names_the_setting():
    with pytest.raises(NotebookError, match="PLAYGROUND_NOTEBOOK_MAX_BYTES"):
        parse_notebook(FIXTURE.read_bytes(), max_bytes=100)


def test_saved_outputs_are_kept_in_the_four_shown_kinds():
    view = {c["index"]: c for c in cells_view(_nb())}
    assert view[0]["type"] == "markdown"
    assert "[image]" in view[0]["source"] and "<img" not in view[0]["source"]

    html = view[2]["outputs"][0]
    assert html["kind"] == "html"
    assert "0.1" in tree_text(html["tree"])
    assert "x     y" in html["plain"]

    stdout, stderr = view[4]["outputs"]
    assert stdout == {"kind": "stream", "name": "stdout", "text": "slope=2.013\n"}
    assert stderr["name"] == "stderr" and "\x1b" not in stderr["text"]

    png = view[5]["outputs"][0]
    assert png["kind"] == "png" and png["data"].startswith("iVBORw0KGgo")
    assert png["alt"].startswith("<Figure")

    error = view[7]["outputs"][0]
    assert error["kind"] == "error" and "ZeroDivisionError" in error["text"]
    assert "\x1b" not in error["text"]


def test_markdown_drops_table_rules_and_tags():
    from nlp_pillars.playground.notebook import markdown_text

    text = markdown_text("| a | b |\n|:---|---:|\n| 1 | 2 |\n<b>bold</b> ![plot](x.png)")
    assert text == "| a | b |\n| 1 | 2 |\nbold [image: plot]"


def test_html_output_loses_scripts_styles_handlers_links_and_images():
    tree = cells_view(_nb())[2]["outputs"][0]["tree"]
    tags = _tags(tree)
    for bad in ("script", "style", "a", "img"):
        assert bad not in tags
    flat = json.dumps(tree)
    for bad in ("alert", "onclick", "onmouseover", "javascript", "color:red", "dataframe"):
        assert bad not in flat
    # The table survives, with only the numeric span attribute.
    assert "table" in tags and "td" in tags
    assert '"a": {"colspan": "2"}' in flat
    # Link text is kept as plain text: content, not markup.
    assert "link" in tree_text(tree)


@pytest.mark.parametrize(
    "markup,forbidden",
    [
        ("<SCRIPT>alert(1)</SCRIPT>ok", "alert"),
        # Leaves the literal text "ipt>alert(1)", which is harmless; no tag survives.
        ("<scr<script>ipt>alert(1)</script>ok", '"t": "script"'),
        ("<svg><script>alert(1)</script><text>x</text></svg>ok", "alert"),
        ('<iframe srcdoc="<script>alert(1)</script>"></iframe>ok', "alert"),
        ("<style>body{background:url(http://evil)}</style>ok", "evil"),
        ('<td colspan="2 onclick=alert(1)">ok</td>', "onclick"),
        ("<math><mi>x</mi></math><!-- <script>alert(1)</script> -->ok", "alert"),
        ('<div style="background:url(http://evil)">ok</div>', "evil"),
        ('<table><tr><td background="http://evil">ok</td></tr></table>', "evil"),
    ],
)
def test_sanitizer_holds_against_hostile_markup(markup, forbidden):
    tree, _ = sanitize_html(markup)
    assert forbidden not in json.dumps(tree)
    assert "ok" in tree_text(tree)


def test_sanitizer_keeps_uppercase_spans_and_drops_absurd_ones():
    tree, _ = sanitize_html('<table><tr><TD COLSPAN=3>a</TD><td rowspan="99999">b</td></tr></table>')
    flat = json.dumps(tree)
    assert '"colspan": "3"' in flat
    assert "rowspan" not in flat


def test_sanitizer_caps_the_node_count_and_says_so():
    markup = "<table>" + "<tr><td>1</td></tr>" * (MAX_HTML_NODES // 2) + "</table>"
    tree, truncated = sanitize_html(markup)
    assert truncated is True
    assert json.dumps(tree).count('"t"') <= MAX_HTML_NODES


def test_style_only_html_falls_back_to_plain_text():
    nb = {"nbformat": 4, "cells": [{"cell_type": "code", "source": "x", "outputs": [{
        "output_type": "execute_result",
        "data": {"text/html": "<style>.a{}</style>", "text/plain": "42"},
    }]}]}
    out = cells_view(nb)[0]["outputs"][0]
    assert out == {"kind": "text", "text": "42"}


def test_a_png_that_is_not_a_png_is_not_shown_as_one():
    nb = {"nbformat": 4, "cells": [{"cell_type": "code", "source": "x", "outputs": [{
        "output_type": "display_data",
        "data": {"image/png": "PHN2ZyBvbmxvYWQ9YWxlcnQoMSk+", "text/plain": "fig"},
    }]}]}
    assert cells_view(nb)[0]["outputs"][0]["kind"] == "text"


# -- dataset URLs and cache ---------------------------------------------------------


def test_dataset_urls_are_quoted_urls_in_code_cells_only():
    urls = find_dataset_urls(_nb())
    assert urls == [
        "https://example.org/data/points.csv",
        "https://example.org/data/extra.csv",  # commented out, still cached
    ]


def _public(host, port):
    return ["93.184.215.14"]


def _cache(tmp_path, handler, *, max_bytes=10_000, resolver=_public):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return DatasetCache(tmp_path, max_bytes, client=client, resolver=resolver)


def test_cache_downloads_records_and_writes_a_flat_manifest(tmp_path):
    bodies = {"/data/points.csv": b"x,y\n0,0.1\n", "/data/extra.csv": b"a\n1\n"}
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=bodies[request.url.path])

    records = _cache(tmp_path, handler).cache(PG_ID, find_dataset_urls(_nb()))
    assert [r["url"] for r in records] == find_dataset_urls(_nb())
    first = records[0]
    assert first["path"] == "files/" + file_name_for(first["url"])
    assert first["path"].endswith("-points.csv")
    assert first["bytes"] == len(bodies["/data/points.csv"])
    assert first["sha256"] == hashlib.sha256(bodies["/data/points.csv"]).hexdigest()
    assert (tmp_path / PG_ID / first["path"]).read_bytes() == bodies["/data/points.csv"]

    manifest = json.loads((tmp_path / PG_ID / MANIFEST_NAME).read_text())
    assert manifest == {r["url"]: r["path"] for r in records}
    assert all(isinstance(v, str) for v in manifest.values())
    assert read_manifest(tmp_path, PG_ID) == manifest
    assert len(seen) == 2


def test_a_failed_download_is_recorded_and_left_out_of_the_manifest(tmp_path):
    def handler(request):
        if request.url.path.endswith("extra.csv"):
            return httpx.Response(404)
        return httpx.Response(200, content=b"ok")

    records = _cache(tmp_path, handler).cache(PG_ID, find_dataset_urls(_nb()))
    assert "error" not in records[0]
    assert records[1]["error"].startswith("HTTP 404")
    manifest = read_manifest(tmp_path, PG_ID)
    assert list(manifest) == ["https://example.org/data/points.csv"]
    assert not list((tmp_path / PG_ID / "files").glob("*.part"))


def test_the_size_cap_is_enforced_by_header_and_by_stream(tmp_path):
    def handler(request):
        if request.url.path.endswith("points.csv"):
            return httpx.Response(200, headers={"content-length": "5000"}, content=b"x" * 5000)
        # No length header: caught while streaming.
        return httpx.Response(200, content=iter([b"y" * 4000, b"y" * 4000]))

    records = _cache(tmp_path, handler, max_bytes=6000).cache(PG_ID, find_dataset_urls(_nb()))
    assert "error" not in records[0]
    assert "PLAYGROUND_DATA_MAX_BYTES" in records[1]["error"]
    assert list(read_manifest(tmp_path, PG_ID)) == ["https://example.org/data/points.csv"]
    assert len(list((tmp_path / PG_ID / "files").iterdir())) == 1


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "172.18.0.3", "169.254.169.254", "::1"])
def test_private_and_local_hosts_are_refused(tmp_path, address):
    handler = MagicMock(side_effect=AssertionError("must not be requested"))
    records = _cache(tmp_path, handler, resolver=lambda h, p: [address]).cache(
        PG_ID, ["http://postgrest:3000/papers"],
    )
    assert "private or local" in records[0]["error"]
    assert read_manifest(tmp_path, PG_ID) == {}


def test_a_redirect_to_a_private_host_is_refused(tmp_path):
    def handler(request):
        return httpx.Response(302, headers={"location": "http://internal.example/secret"})

    def resolver(host, port):
        return ["10.0.0.9"] if host == "internal.example" else ["93.184.215.14"]

    records = _cache(tmp_path, handler, resolver=resolver).cache(PG_ID, ["https://example.org/a.csv"])
    assert "private or local" in records[0]["error"]


def test_a_redirect_to_a_public_host_is_followed(tmp_path):
    def handler(request):
        if request.url.host == "example.org":
            return httpx.Response(302, headers={"location": "https://cdn.example.net/a.csv"})
        return httpx.Response(200, content=b"data")

    records = _cache(tmp_path, handler).cache(PG_ID, ["https://example.org/a.csv"])
    assert records[0]["bytes"] == 4
    # The manifest keys the URL the notebook names, not where it redirected.
    assert list(read_manifest(tmp_path, PG_ID)) == ["https://example.org/a.csv"]


def test_reimport_reuses_cached_files_and_removes_stale_ones(tmp_path):
    def handler(request):
        return httpx.Response(200, content=b"x,y\n")

    first = _cache(tmp_path, handler).cache(PG_ID, ["https://example.org/a.csv", "https://example.org/b.csv"])
    untouched = MagicMock(side_effect=AssertionError("cached file must be reused"))
    second = _cache(tmp_path, untouched).cache(PG_ID, ["https://example.org/a.csv"], first)
    assert second == [first[0]]
    files = sorted(p.name for p in (tmp_path / PG_ID / "files").iterdir())
    assert files == [file_name_for("https://example.org/a.csv")]


def test_a_playground_id_must_be_a_uuid(tmp_path):
    with pytest.raises(DatasetError):
        _cache(tmp_path, MagicMock()).cache("../../etc", ["https://example.org/a.csv"])


def test_an_unwritable_data_directory_does_not_fail_the_import(tmp_path):
    cache = MagicMock()
    cache.cache.side_effect = PermissionError("read-only file system")
    records = pg_service.cache_datasets(PG_ID, ["https://example.org/a.csv"], [], cache=cache)
    assert records == [{
        "url": "https://example.org/a.csv",
        "error": "The playground data directory is not writable (read-only file system).",
    }]


# -- import ------------------------------------------------------------------------


def test_import_validates_before_touching_the_database():
    with patch.object(store, "paper_row") as paper_row:
        with pytest.raises(NotebookError, match=".ipynb"):
            pg_service.import_notebook("arxiv:1", b"{}", "notes.txt")
        with pytest.raises(NotebookError):
            pg_service.import_notebook("arxiv:1", b"nope", "x.ipynb")
    paper_row.assert_not_called()


def test_import_of_an_unknown_paper_is_refused():
    with patch.object(store, "paper_row", return_value=None):
        with pytest.raises(pg_service.PaperNotFound):
            pg_service.import_notebook("arxiv:missing", FIXTURE.read_bytes(), "toy.ipynb")


def test_import_stores_the_raw_notebook_then_caches_datasets(tmp_path):
    calls = []
    cache = MagicMock()
    cache.cache.side_effect = lambda pid, urls, previous: calls.append((pid, urls, previous)) or [
        {"url": u, "path": "files/x", "bytes": 1, "sha256": "0" * 64} for u in urls
    ]
    existing = {"id": PG_ID, "datasets": [{"url": "https://example.org/old.csv", "path": "files/o", "bytes": 3, "sha256": "1" * 64}]}
    with patch.object(store, "paper_row", return_value={"id": "arxiv:1", "pillar_id": "p"}), \
         patch.object(store, "get_for_paper", return_value=existing), \
         patch.object(store, "save_import", side_effect=lambda **kw: dict(kw, id=kw["playground_id"])) as save, \
         patch.object(store, "save_datasets") as save_datasets:
        row = pg_service.import_notebook("arxiv:1", FIXTURE.read_bytes(), "/tmp/x/toy.ipynb", cache=cache)
    kw = save.call_args.kwargs
    assert kw["playground_id"] == PG_ID  # a re-import keeps the id
    assert kw["notebook"] == json.loads(FIXTURE.read_text())  # raw, outputs included
    assert kw["notebook_name"] == "toy.ipynb"
    assert kw["title"] == "Fitting a line by gradient descent"
    assert calls == [(PG_ID, find_dataset_urls(_nb()), existing["datasets"])]
    save_datasets.assert_called_once_with(PG_ID, row["datasets"])
    assert len(row["datasets"]) == 2


def test_a_missing_table_stops_the_import_before_any_download():
    cache = MagicMock()
    missing = store.PlaygroundTableMissing(f"Apply {MIGRATION} by hand.")
    with patch.object(store, "paper_row", return_value={"id": "arxiv:1"}), \
         patch.object(store, "get_for_paper", side_effect=missing):
        with pytest.raises(store.PlaygroundTableMissing, match="028_playgrounds.sql"):
            pg_service.import_notebook("arxiv:1", FIXTURE.read_bytes(), "toy.ipynb", cache=cache)
    cache.cache.assert_not_called()


def _client_returning(error):
    client = MagicMock()
    query = client.table.return_value
    query.select.return_value = query
    query.eq.return_value = query
    query.limit.return_value = query
    query.order.return_value = query
    query.params = {}
    query.execute.return_value = {"data": None, "error": error}
    return client


def test_store_names_the_migration_when_the_table_is_missing():
    error = {"code": "PGRST205", "message": "Could not find the table 'public.playgrounds' in the schema cache"}
    with patch("nlp_pillars.db.get_client", return_value=_client_returning(error)):
        with pytest.raises(store.PlaygroundTableMissing) as exc:
            store.get_for_paper("arxiv:1")
    assert "docs/migrations/028_playgrounds.sql" in str(exc.value)


def test_store_reports_other_failures_without_blaming_the_migration():
    with patch("nlp_pillars.db.get_client", return_value=_client_returning({"message": "connection refused"})):
        with pytest.raises(store.PlaygroundStoreError) as exc:
            store.get_for_paper("arxiv:1")
    assert not isinstance(exc.value, store.PlaygroundTableMissing)
    assert "028" not in str(exc.value)


def test_store_get_treats_a_malformed_id_as_absent():
    with patch("nlp_pillars.db.get_client") as get_client:
        assert store.get("not-a-uuid") is None
    get_client.assert_not_called()


# -- explain pass: prompt and batching ------------------------------------------------


def test_prompt_shows_code_with_line_numbers_and_saved_output():
    view = cells_view(_nb())
    text = render_cell(view[4])
    assert "=== CELL 4 (code) ===" in text
    assert "1 | w = 0.0" in text
    assert "slope=2.013" in text
    assert "[a saved figure" in render_cell(view[5])
    assert "--- no saved output ---" in render_cell(view[1])


def test_prompt_in_docs_is_the_prompt_in_code():
    assert SYSTEM_PROMPT in (ROOT / "docs/PROMPTS.md").read_text()


def test_one_call_stays_under_the_sdk_non_streaming_limit():
    from nlp_pillars.playground.explain import MAX_TOKENS

    # anthropic refuses a non-streaming call when 3600 * max_tokens / 128000 > 600.
    assert 3600 * MAX_TOKENS / 128_000 <= 600


def test_prompt_states_the_grounding_rules():
    for phrase in ("Use only what is in the notebook", "Never state a number", "Do not invent the new numbers"):
        assert phrase in SYSTEM_PROMPT


def _many_cells(n):
    cells = []
    for i in range(n):
        cells.append({"index": 2 * i, "type": "markdown", "source": f"part {i}"})
        cells.append({"index": 2 * i + 1, "type": "code", "source": f"x = {i}", "outputs": []})
    return cells


def test_batches_split_long_notebooks_and_keep_markdown_with_its_code():
    batches = make_batches(_many_cells(45), max_code_cells=20)
    assert [len(b.code_indices) for b in batches] == [20, 20, 5]
    # Batch two starts with the markdown that introduces its first code cell.
    assert batches[1].cells[0] == {"index": 40, "type": "markdown", "source": "part 20"}
    covered = [i for b in batches for i in b.code_indices]
    assert covered == [2 * i + 1 for i in range(45)]


def test_batches_split_on_size_too():
    cells = _many_cells(3)
    for cell in cells:
        if cell["type"] == "code":
            cell["source"] = "y = 1\n" * 2000
    assert len(make_batches(cells, max_chars=15_000)) == 3


def test_fixture_is_one_batch_without_the_blank_cell():
    batches = make_batches(cells_view(_nb()))
    assert len(batches) == 1 and batches[0].code_indices == [1, 2, 4, 5, 7]


# -- explain pass: validation ----------------------------------------------------------

SOURCES = {c["index"]: c["source"] for c in cells_view(_nb()) if c["type"] == "code"}


def _cell(idx, **over):
    base = dict(
        cell_index=idx,
        explanation="Loads the libraries the rest of the notebook needs.",
        line_notes=[],
        tries=[],
        quiz="Which library reads the CSV?",
    )
    base.update(over)
    return CellDraft(**base)


def _good_draft():
    return ExplainDraft(steps=[
        StepDraft(title="Load", goal="Read the points.", cells=[
            _cell(1),
            _cell(2, line_notes=[LineNoteDraft(line=3, note="Shows five rows.")],
                  tries=[TryDraft(label="Show ten rows", code=SOURCES[2].replace("head()", "head(10)"),
                                  expect="The table grows to ten rows.")]),
        ]),
        StepDraft(title="Fit", goal="Recover the slope.", cells=[_cell(4), _cell(5), _cell(7)]),
    ])


def test_a_good_answer_validates_completely():
    out = validate_draft(_good_draft(), wanted=[1, 2, 4, 5, 7], sources=SOURCES)
    assert out.problems == [] and out.notes == []
    assert out.explained == [1, 2, 4, 5, 7]
    assert [s["title"] for s in out.steps] == ["Load", "Fit"]
    cell = out.steps[0]["cells"][1]
    assert cell["line_notes"] == [{"line": 3, "note": "Shows five rows."}]
    assert cell["tries"][0]["code"].endswith("head(10)")
    assert set(cell) == {"cell_index", "explanation", "line_notes", "tries", "quiz"}


def test_a_malformed_answer_keeps_only_what_checks_out():
    long = " ".join(["word"] * 60)
    draft = ExplainDraft(steps=[
        StepDraft(title="  ", goal="no title", cells=[_cell(1)]),
        StepDraft(title="Mixed", goal="Some good, some bad.", cells=[
            _cell(0),                                   # markdown cell
            _cell(6),                                   # blank code cell
            _cell(99),                                  # does not exist
            _cell(2, explanation=long),                 # too long
            _cell(4, quiz=""),                          # no quiz
            _cell(5, line_notes=[LineNoteDraft(line=0, note="x"), LineNoteDraft(line=40, note="y"),
                                 LineNoteDraft(line=2, note="kept"), LineNoteDraft(line=2, note="dup")],
                  tries=[TryDraft(label="Same", code=SOURCES[5], expect="nothing"),
                         TryDraft(label="", code="x", expect="y")]
                  + [TryDraft(label=f"T{i}", code=f"plt.plot([{i}])", expect="A line.") for i in range(5)]),
            _cell(5),                                   # repeated
            _cell(1),                                   # out of order
        ]),
    ])
    out = validate_draft(draft, wanted=[1, 2, 4, 5, 7], sources=SOURCES)
    assert out.explained == [5]
    kept = out.steps[0]["cells"][0]
    assert kept["line_notes"] == [{"line": 2, "note": "kept"}]
    assert [t["label"] for t in kept["tries"]] == ["T0", "T1", "T2"]
    joined = " ".join(out.problems)
    for needle in ("no title", "Cell 0 is not", "Cell 6 is not", "Cell 99", "60 words", "no quiz", "explained twice", "out of notebook order"):
        assert needle in joined
    assert any("did not change the code" in n for n in out.notes)


# -- explain pass: the job ---------------------------------------------------------------


def _row(**over):
    row = {
        "id": PG_ID, "paper_id": "arxiv:1", "title": "Toy", "notebook": _nb(),
        "steps": [], "status": "imported",
    }
    row.update(over)
    return row


def _call(parsed, cost=0.012):
    return StructuredCall(
        stage="playground_explain", provider="anthropic", model="claude-sonnet-5",
        parsed=parsed, input_tokens=3000, output_tokens=1500, cost_usd=cost,
    )


def _run(row, answers, cancel=None):
    stages = []
    saved = {}
    answers = list(answers)

    def fake_structured(*args, **kwargs):
        item = answers.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    with patch.object(store, "get", return_value=row), \
         patch.object(store, "save_explain", side_effect=lambda pid, **kw: saved.update(kw, how="explain")), \
         patch.object(store, "save_explain_error", side_effect=lambda pid, err: saved.update(error=err, how="error")), \
         patch.object(pg_service, "structured", side_effect=fake_structured):
        try:
            result = pg_service.run_explain_job(
                PG_ID, lambda n, s, d=None: stages.append((n, s, d)), cancel,
            )
        except Exception as exc:  # noqa: BLE001
            result = exc
    return result, stages, saved


def test_a_good_pass_is_saved_ready_with_numbered_steps_and_its_cost():
    result, stages, saved = _run(_row(), [_call(_good_draft())])
    assert result["status"] == "ready"
    assert result["cells_explained"] == result["cells_total"] == 5
    assert saved["how"] == "explain" and saved["status"] == "ready" and saved["error"] is None
    assert [s["id"] for s in saved["steps"]] == ["step-1", "step-2"]
    assert saved["model"] == "claude-sonnet-5"
    final = [s for s in stages if s[1] in ("completed", "failed", "skipped")]
    assert [s[0] for s in final] == [s.value for s in PLAYGROUND_EXPLAIN_STAGES]
    explain_detail = [d for n, s, d in stages if n == "playground_explain" and s == "completed"][0]
    assert "$0.012000" in explain_detail  # what the daily spend cap adds up


def test_a_pass_that_misses_cells_is_saved_partial_with_the_gaps_named():
    draft = ExplainDraft(steps=[StepDraft(title="Load", goal="Read.", cells=[_cell(1), _cell(2)])])
    result, _, saved = _run(_row(), [_call(draft)])
    assert result["status"] == "partial"
    assert saved["status"] == "partial"
    assert "3 of 5 code cells were not explained (cells 4, 5, 7)" in saved["error"]


def test_a_pass_with_nothing_usable_fails_and_keeps_the_notebook():
    draft = ExplainDraft(steps=[StepDraft(title="Bad", goal="Bad.", cells=[_cell(42)])])
    result, stages, saved = _run(_row(), [_call(draft)])
    assert isinstance(result, pg_service.PlaygroundExplainError)
    assert saved["status"] == "failed" and saved["steps"] == []
    assert "Cell 42" in saved["error"]
    assert ("playground_explain", "failed") in [(n, s) for n, s, _ in stages]


def test_a_failed_retry_keeps_the_guide_that_was_already_there():
    previous = [{"id": "step-1", "title": "Old", "goal": "g", "cells": []}]
    result, _, saved = _run(_row(steps=previous, status="ready"), [RuntimeError("overloaded")])
    assert isinstance(result, pg_service.PlaygroundExplainError)
    assert saved["how"] == "error"
    assert saved["error"].startswith("The last attempt failed:")
    assert "overloaded" in saved["error"]


def test_one_failed_part_keeps_the_other_parts():
    big = _row(notebook={"nbformat": 4, "cells": [
        {"cell_type": "code", "source": f"x = {i}", "outputs": []} for i in range(15)
    ]})
    first = ExplainDraft(steps=[StepDraft(title="A", goal="g.", cells=[
        _cell(i, explanation="Sets x.") for i in range(12)
    ])])
    result, stages, saved = _run(big, [_call(first, cost=0.5), RuntimeError("rate limited")])
    assert result["status"] == "partial"
    assert result["cells_explained"] == 12 and result["cells_total"] == 15
    assert "Part 2 of 2 failed: rate limited" in saved["error"]
    assert result["cost_usd"] == 0.5


def test_cancel_stops_before_the_next_paid_call():
    cancel = threading.Event()
    cancel.set()
    result, stages, saved = _run(_row(), [], cancel=cancel)
    assert isinstance(result, RunCancelledError)
    assert saved == {}
    assert all(s != "completed" for n, s, d in stages if n == "playground_persist")


def test_a_missing_playground_fails_in_prepare():
    with patch.object(store, "get", return_value=None):
        with pytest.raises(pg_service.PlaygroundNotFound):
            pg_service.run_explain_job(PG_ID)


# -- run kind, dispatch and migration -------------------------------------------------


def test_dispatch_seeds_the_three_stages_and_stores_the_playground_id():
    scheduler = MagicMock()
    with patch.object(run_service.db, "create_pipeline_run", return_value=MagicMock(id="run-pg")) as create:
        run_id = run_service.dispatch_run(
            scheduler, {}, pillar_id="neural-architectures-language",
            trigger_source=pg_service.TRIGGER_UI_PLAYGROUND,
            kind=run_service.KIND_PLAYGROUND_EXPLAIN,
            playground_id=PG_ID,
            initial_result={"playground_id": PG_ID, "paper_id": "arxiv:1"},
        )
    assert run_id == "run-pg"
    assert create.call_args.args[3] == ["playground_prepare", "playground_explain", "playground_persist"]
    assert create.call_args.kwargs["result"]["playground_id"] == PG_ID
    assert scheduler.add_job.call_args.kwargs["kwargs"]["playground_id"] == PG_ID
    assert pg_service.TRIGGER_UI_PLAYGROUND in run_service.WEBUI_TRIGGER_SOURCES


@pytest.mark.parametrize("status,run_status", [("ready", "succeeded"), ("partial", "failed")])
def test_execute_run_records_ready_as_succeeded_and_partial_as_failed(status, run_status):
    payload = {"playground_id": PG_ID, "status": status, "error": None if status == "ready" else "gap"}
    with patch.object(run_service.db, "start_pipeline_run"), \
         patch.object(run_service.db, "finish_pipeline_run") as finish, \
         patch.object(run_service.playground_service, "run_explain_job", return_value=payload), \
         patch.object(run_service.cloud_publish_service, "notify") as notify:
        run_service.execute_run(
            "run-1", run_service.KIND_PLAYGROUND_EXPLAIN, "p", threading.Event(), {},
            playground_id=PG_ID,
        )
    args, kwargs = finish.call_args
    assert args[1] == run_status
    assert kwargs["result"] == payload
    notify.assert_not_called()  # the cloud reader does not carry playgrounds yet


def test_execute_run_records_a_failed_pass_as_failed():
    with patch.object(run_service.db, "start_pipeline_run"), \
         patch.object(run_service.db, "finish_pipeline_run") as finish, \
         patch.object(run_service.playground_service, "run_explain_job",
                      side_effect=pg_service.PlaygroundExplainError("nothing usable")):
        with pytest.raises(pg_service.PlaygroundExplainError):
            run_service.execute_run(
                "run-1", run_service.KIND_PLAYGROUND_EXPLAIN, "p", threading.Event(), {},
                playground_id=PG_ID,
            )
    assert finish.call_args.args[1] == RunStatus.FAILED.value
    assert finish.call_args.kwargs["error"] == "nothing usable"


def _statement(pattern):
    match = re.search(pattern, MIGRATION_SQL, re.S | re.I)
    assert match, pattern
    return match.group(0)


def test_migration_keeps_every_kind_and_trigger_and_adds_the_playground():
    kind = _statement(r"ADD CONSTRAINT pipeline_runs_kind_check.*?;")
    for value in ("run_daily", "process_selected", "discover", "podcast_audio", "upload",
                  "podcast_script", "podcast_factcheck", "playground_explain"):
        assert f"'{value}'" in kind
        assert f"'{value}'" in SCHEMA
    trigger = _statement(r"ADD CONSTRAINT pipeline_runs_trigger_source_check.*?;")
    for value in ("ui_pipeline", "ui_select", "ui_discover", "scheduler", "ui_podcast_audio",
                  "ui_upload", "ui_podcast_script", "ui_podcast_factcheck",
                  "auto_podcast_factcheck", "ui_playground"):
        assert f"'{value}'" in trigger
    assert "CREATE TABLE IF NOT EXISTS playgrounds" in MIGRATION_SQL
    assert "CREATE TABLE playgrounds" in SCHEMA
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON playgrounds TO web_anon" in MIGRATION_SQL
    assert MIGRATION_SQL.rstrip().endswith("NOTIFY pgrst, 'reload schema';")


def test_migration_and_schema_agree_on_the_table_columns():
    def columns(sql, head):
        body = sql.split(head, 1)[1].split("\n);", 1)[0]
        return [line.split()[0] for line in body.splitlines()
                if re.match(r"\s{4}[a-z_]+\s", line)]
    # 030 adds setup_code to the table 028 created.
    later = re.findall(
        r"ALTER TABLE playgrounds ADD COLUMN IF NOT EXISTS ([a-z_]+)",
        (ROOT / "docs/migrations/030_playground_setup_code.sql").read_text(),
    )
    assert later == ["setup_code"]
    assert sorted(columns(MIGRATION_SQL, "CREATE TABLE IF NOT EXISTS playgrounds (") + later) == \
        sorted(columns(SCHEMA, "CREATE TABLE playgrounds ("))


def _runs_table(index_pattern):
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE pipeline_runs (id INTEGER PRIMARY KEY, pillar_id TEXT,"
        " kind TEXT, status TEXT, result TEXT)"
    )
    conn.execute(_statement(index_pattern))
    return conn


def _insert(conn, kind, status="running", result=None):
    conn.execute(
        "INSERT INTO pipeline_runs (pillar_id, kind, status, result) VALUES ('p', ?, ?, ?)",
        (kind, status, result),
    )


def test_an_explain_pass_may_overlap_discovery_on_the_same_pillar():
    conn = _runs_table(r"CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_per_pillar[^;]*;")
    _insert(conn, "discover")
    _insert(conn, "playground_explain")
    _insert(conn, "playground_explain")
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, "discover")


def test_one_active_explain_pass_per_playground():
    conn = _runs_table(
        r"CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_explain_per_playground[^;]*;"
    )
    _insert(conn, "playground_explain", result='{"playground_id": "a"}')
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, "playground_explain", "pending", '{"playground_id": "a"}')
    _insert(conn, "playground_explain", "succeeded", '{"playground_id": "a"}')
    _insert(conn, "playground_explain", result='{"playground_id": "b"}')


def test_route_and_settings_defaults():
    from nlp_pillars.config import Settings
    from nlp_pillars.llm_routes import PLAYGROUND_ROUTE, PODCAST_SYNTHESIS_ROUTE, route_for

    assert route_for("playground_explain") is PLAYGROUND_ROUTE
    assert PLAYGROUND_ROUTE.default_model == PODCAST_SYNTHESIS_ROUTE.default_model
    assert PLAYGROUND_ROUTE.model_attrs == ("llm_playground_model",)
    assert Settings.model_fields["playground_data_max_bytes"].default == 1_000_000_000
    assert Settings.model_fields["playground_data_dir"].default == "/app/playground-data"


def test_compose_mounts_the_playground_volume_without_touching_the_others():
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "- playground_data:/app/playground-data" in compose
    assert "playground_data:\n    name: nlp_playground_data" in compose
    for existing in ("name: nlp_pg_data", "name: nlp_uploads", "name: nlp_qdrant_data"):
        assert existing in compose
    assert "mkdir -p /app/playground-data" in (ROOT / "Dockerfile").read_text()
