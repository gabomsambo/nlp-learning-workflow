"""map build's I/O, run kind, migration and CLI.

The store reads through a real ``PostgRESTClient`` whose HTTP layer is a
``Mock(spec=httpx.Client)`` serving the pure build's fixture as table rows,
so the filters and the RPC body are what PostgREST would see. The run kind
is checked through ``run_service`` with ``db`` patched, and migration 034 by
executing its own index DDL against sqlite, as the earlier run-kind
migrations are.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest
from test_map_build import PILLAR, fixture_inputs
from typer.testing import CliRunner

from nlp_pillars import db, map_build, map_store
from nlp_pillars.db import PostgRESTClient
from nlp_pillars.map_store import MIGRATION, MapStoreError
from nlp_pillars.orchestrator import RunCancelledError
from nlp_pillars.schemas import MAP_BUILD_STAGES, RunStatus
from nlp_pillars.work_ids import parse_paper_id
from webui.services import run_service

ROOT = Path(__file__).resolve().parents[1]
MIGRATION_SQL = (ROOT / MIGRATION).read_text()
SCHEMA = (ROOT / "schema.sql").read_text()
MODEL = "BAAI/bge-base-en-v1.5@test"


def reply(status, body):
    return httpx.Response(status, json=body, request=httpx.Request("GET", "http://x/"))


def _matches(row, key, condition):
    value = row.get(key)
    if condition.startswith("eq."):
        return str(value) == condition[3:] or (
            isinstance(value, bool) and str(value).lower() == condition[3:]
        )
    if condition.startswith("in.("):
        items = [
            item.strip('"')
            for item in re.findall(r'"(?:[^"\\]|\\.)*"|[^,]+', condition[4:-1])
        ]
        return str(value) in items
    raise AssertionError(f"unexpected filter {key}={condition}")


class FakeHttp:
    """PostgREST over in-memory tables: eq / in filters, limit and offset."""

    def __init__(self, tables):
        self.tables = tables
        self.gets = []
        self.posts = []
        self.rpc_answer = {"scope": PILLAR, "changed": True}
        self.missing = set()
        self.client = Mock(spec=httpx.Client)
        self.client.get.side_effect = self._get
        self.client.post.side_effect = self._post

    def _get(self, url, params=None):
        params = dict(params or {})
        table = url.rsplit("/", 1)[-1]
        self.gets.append((table, params))
        if table in self.missing:
            return reply(
                404,
                {
                    "code": "PGRST205",
                    "message": f"Could not find the table 'public.{table}' in the schema cache",
                },
            )
        rows = [
            row
            for row in self.tables.get(table, [])
            if all(
                _matches(row, key, value)
                for key, value in params.items()
                if key not in ("select", "order", "limit", "offset")
            )
        ]
        offset = int(params.get("offset", 0))
        limit = int(params.get("limit", len(rows)))
        return reply(200, rows[offset : offset + limit])

    def _post(self, url, json=None):
        name = url.rsplit("/", 1)[-1]
        self.posts.append((name, json))
        if name in self.missing:
            return reply(
                404,
                {
                    "code": "PGRST202",
                    "message": f"Could not find the function public.{name} in the schema cache",
                },
            )
        return reply(200, self.rpc_answer)


def tables_from_fixture():
    """The pure build's fixture as the rows migration 021 / 031 / 034 tables hold."""
    inputs = fixture_inputs()
    papers = [
        {
            "id": p.paper_id,
            "pillar_id": p.pillar_id,
            "work_id": p.work_id,
            "title": p.title,
            "year": p.year,
            "authors": list(p.authors),
        }
        for p in inputs.library
    ]
    works = [
        {
            "id": w.id,
            "title": w.title,
            "year": w.year,
            "publication_date": w.publication_date,
            "citation_count": w.citation_count,
            "influential_citation_count": w.influential_citation_count,
            "authors": list(w.authors),
            "venue": w.venue,
        }
        for w in inputs.works.values()
    ]
    citations = [
        {
            "citing": c.citing,
            "cited": c.cited,
            "is_influential": c.is_influential,
            "intents": list(c.intents),
            "contexts": list(c.contexts),
        }
        for c in inputs.citations
    ]
    vectors = []
    node_of = {p.node_id: p for p in inputs.library}
    for node, vec in inputs.vectors.items():
        if node.startswith("queue:"):
            continue  # queued titles are embedded at build time
        if node in node_of:
            vectors.append(
                {
                    "ref_kind": "paper",
                    "ref_id": node_of[node].paper_id,
                    "model_id": MODEL,
                    "source": "title_abstract",
                    "vec": vec,
                }
            )
        else:
            vectors.append(
                {
                    "ref_kind": "external_paper",
                    "ref_id": node,
                    "model_id": MODEL,
                    "source": "title_abstract",
                    "vec": vec,
                }
            )
    # Queue rows carry the ids discovery stores (bare arXiv ids); the two that
    # name a stored work resolve through work_ids.
    arxiv = {"q-hit": "2401.00001", "q-mine": "2401.00002v2"}
    queue = [
        {
            "id": q.queue_id,
            "pillar_id": PILLAR,
            "paper_id": arxiv.get(q.queue_id, q.paper_id),
            "title": q.title,
            "processed": q.processed,
        }
        for q in inputs.queue
    ]
    work_ids = []
    for q in inputs.queue:
        key = parse_paper_id(arxiv.get(q.queue_id, ""))
        if q.work_id and key:
            work_ids.append({"kind": key[0], "value": key[1], "work_id": q.work_id})
    pillars = [
        {
            "id": PILLAR,
            "name": "Model interpretability",
            "goal": "Learn to analyse models",
            "focus_areas": ["Mechanistic interpretability"],
        }
    ]
    return {
        "pillars": pillars,
        "papers": papers,
        "works": works,
        "work_citations": citations,
        "paper_vectors": vectors,
        "paper_queue": queue,
        "work_ids": work_ids,
        "map_snapshots": [],
    }, inputs


@pytest.fixture
def http():
    tables, inputs = tables_from_fixture()
    fake = FakeHttp(tables)
    fake.inputs = inputs
    client = PostgRESTClient("http://postgrest.test", "token")
    client.client = fake.client
    previous = db._client
    db.set_client(client)
    yield fake
    db._client = previous


def fake_embedder(inputs):
    """Returns the fixture's own vectors for the goal and the queued titles."""
    by_title = {q.title: inputs.vectors.get(q.node_id) for q in inputs.queue}
    calls = []

    def embed(texts):
        calls.append(list(texts))
        out = []
        for position, text in enumerate(texts):
            if text in by_title and by_title[text] is not None:
                out.append(by_title[text])
            else:
                out.append(inputs.anchors[position % len(inputs.anchors)])
        return out

    return (embed, MODEL), calls


# --------------------------------------------------------------- store


def test_dry_run_reads_computes_and_neither_embeds_nor_writes(http):
    (embed, model), calls = fake_embedder(http.inputs)
    report = map_store.execute_map_build(
        pillar=PILLAR, dry_run=True, embedder=(embed, model)
    )
    assert calls == []
    assert http.posts == []
    build = report.pillars[0]
    assert build.result.stats["library_works"] == 4  # three works and one document
    assert build.written is False
    assert report.model_id == MODEL
    text = report.render()
    assert "map build (dry-run)" in text and PILLAR in text and "dry-run" in text


def test_a_write_embeds_the_goal_and_unvectored_titles_and_upserts_once(http):
    (embed, model), calls = fake_embedder(http.inputs)
    stages = []
    report = map_store.execute_map_build(
        pillar=PILLAR,
        embedder=(embed, model),
        on_stage=lambda name, status, detail: stages.append((name, status)),
        run_id="run-1",
    )
    assert calls[0] == ["Learn to analyse models", "Mechanistic interpretability"]
    assert sorted(calls[1]) == sorted(
        [
            "mechanistic circuits in transformers",
            "bottlenose dolphin acoustics",
            "probing chain wA-seed2",
            "a title nothing embedded",
        ]
    )
    rpc = [body for name, body in http.posts if name == "put_map_snapshot"]
    assert len(rpc) == 1
    body = rpc[0]
    result = report.pillars[0].result
    assert body["p_scope"] == PILLAR
    assert (
        body["p_payload_hash"]
        == result.payload_hash
        == map_build.sha256_of(body["p_payload"])
    )
    assert body["p_params_hash"] == map_build.params_hash()
    assert body["p_model_id"] == MODEL and body["p_run_id"] == "run-1"
    assert body["p_layout_key"] == result.payload["layout"]["key"]
    assert set(body["p_positions"]) == {
        node["id"] for node in body["p_payload"]["nodes"]
    }
    completed = [name for name, status in stages if status == "completed"]
    assert completed == [stage.value for stage in MAP_BUILD_STAGES]
    assert report.pillars[0].written is True


def test_the_queue_is_resolved_to_works_through_work_ids(http):
    (embed, model), _ = fake_embedder(http.inputs)
    report = map_store.execute_map_build(pillar=PILLAR, embedder=(embed, model))
    verdicts = {
        row["queue_id"]: row["verdict"]
        for row in report.pillars[0].result.payload["queue"]
    }
    assert verdicts["q-mine"] == map_build.VERDICT_ALREADY_YOURS
    assert verdicts["q-hit"] == map_build.VERDICT_STRONG
    assert verdicts["q-off"] == map_build.VERDICT_OFF_TOPIC


def test_a_rebuild_starts_from_the_stored_layout(http):
    http.tables["map_snapshots"] = [
        {"scope": PILLAR, "positions": {"wA00": [0.5, -0.5]}, "layout_key": "k"}
    ]
    inputs, warnings = map_store.assemble_inputs(map_store.load_evidence(), PILLAR)
    assert inputs.previous_positions == {"wA00": (0.5, -0.5)}
    assert inputs.previous_layout_key == "k"
    assert warnings == []


def test_before_034_a_dry_run_still_works_and_a_write_names_the_migration(http):
    http.missing = {"map_snapshots", "put_map_snapshot"}
    report = map_store.execute_map_build(pillar=PILLAR, dry_run=True)
    assert any("No previous layout" in w for w in report.pillars[0].result.warnings)
    (embed, model), _ = fake_embedder(http.inputs)
    with pytest.raises(MapStoreError, match="034_map_snapshots.sql"):
        map_store.execute_map_build(pillar=PILLAR, embedder=(embed, model))


def test_before_031_the_build_names_the_works_migration(http):
    http.missing = {"works"}
    with pytest.raises(MapStoreError, match="031_works_identity.sql"):
        map_store.execute_map_build(pillar=PILLAR, dry_run=True)


def test_vectors_of_another_model_are_not_mixed_with_a_new_embedding(http):
    (embed, _), calls = fake_embedder(http.inputs)
    inputs, warnings = map_store.assemble_inputs(
        map_store.load_evidence(), PILLAR, embed=embed, embed_model_id="other-model"
    )
    assert calls == [] and inputs.anchors == []
    assert any("not embedded" in w for w in warnings)


def test_a_local_model_failure_costs_only_the_goal_anchor(http):
    from nlp_pillars.embeddings import EmbeddingError

    def broken(texts):
        raise EmbeddingError("fastembed is not installed")

    report = map_store.execute_map_build(pillar=PILLAR, embedder=(broken, MODEL))
    build = report.pillars[0]
    assert build.written is True
    assert any("Could not embed" in w for w in build.result.warnings)
    verdicts = {
        row["queue_id"]: row["verdict"] for row in build.result.payload["queue"]
    }
    assert verdicts["q-new"] == map_build.VERDICT_UNSCORED  # no vector, no work


def test_one_failing_pillar_does_not_stop_the_others(http):
    evidence = map_store.load_evidence()
    with (
        patch.object(map_store, "load_evidence", return_value=evidence),
        patch.object(map_store, "build_pillar", side_effect=MapStoreError("boom")),
    ):
        report = map_store.execute_map_build(dry_run=True)
    assert report.failed and report.pillars[0].error == "boom"


def test_cancellation_is_checked_as_each_stage_starts(http):
    class Stop(Exception):
        pass

    seen = []

    def check():
        if len(seen) >= 2:
            raise Stop

    with pytest.raises(Stop):
        map_store.execute_map_build(
            pillar=PILLAR,
            dry_run=True,
            check_cancel=check,
            on_stage=lambda name, status, detail: seen.append(name),
        )
    assert http.posts == []


# ------------------------------------------------------------ run kind


def test_dispatch_seeds_the_seven_map_stages():
    scheduler = MagicMock()
    with patch.object(
        run_service.db, "create_pipeline_run", return_value=MagicMock(id="run-map")
    ) as create:
        run_id = run_service.dispatch_run(
            scheduler,
            {},
            pillar_id=PILLAR,
            trigger_source=map_store.TRIGGER_UI_MAP,
            kind=run_service.KIND_MAP_BUILD,
        )
    assert run_id == "run-map"
    assert create.call_args.args[2] == "map_build"
    assert create.call_args.args[3] == [
        "map_load",
        "map_graph",
        "map_communities",
        "map_lineage",
        "map_layout",
        "map_suggest",
        "map_persist",
    ]
    assert map_store.TRIGGER_UI_MAP in run_service.WEBUI_TRIGGER_SOURCES


def test_execute_run_records_a_written_map_as_succeeded():
    payload = {
        "pillar_id": PILLAR,
        "written": True,
        "payload_hash": "abc",
        "stats": {"works": 3},
    }
    with (
        patch.object(run_service.db, "start_pipeline_run"),
        patch.object(run_service.db, "finish_pipeline_run") as finish,
        patch.object(
            run_service.map_store, "run_map_build_job", return_value=payload
        ) as job,
        patch.object(run_service.cloud_publish_service, "notify") as notify,
    ):
        run_service.execute_run(
            "run-1", run_service.KIND_MAP_BUILD, PILLAR, threading.Event(), {}
        )
    assert job.call_args.args[0] == PILLAR
    assert finish.call_args.args[1] == RunStatus.SUCCEEDED.value
    assert finish.call_args.kwargs["result"] == payload
    notify.assert_not_called()  # the cloud reader does not carry the map


def test_a_cancelled_map_build_is_recorded_cancelled():
    cancel = threading.Event()
    cancel.set()

    def job(pillar_id, on_stage, check_cancel, run_id=None):
        check_cancel()

    with (
        patch.object(run_service.db, "start_pipeline_run"),
        patch.object(run_service.db, "finish_pipeline_run") as finish,
        patch.object(run_service.map_store, "run_map_build_job", side_effect=job),
    ):
        run_service.execute_run("run-1", run_service.KIND_MAP_BUILD, PILLAR, cancel, {})
    assert finish.call_args.args[1] == RunStatus.CANCELLED.value


def test_a_failed_map_build_is_recorded_failed_with_the_reason():
    with (
        patch.object(run_service.db, "start_pipeline_run"),
        patch.object(run_service.db, "finish_pipeline_run") as finish,
        patch.object(
            run_service.map_store,
            "run_map_build_job",
            side_effect=MapStoreError(
                "Apply docs/migrations/034_map_snapshots.sql by hand."
            ),
        ),
    ):
        with pytest.raises(MapStoreError):
            run_service.execute_run(
                "run-1", run_service.KIND_MAP_BUILD, PILLAR, threading.Event(), {}
            )
    assert finish.call_args.args[1] == RunStatus.FAILED.value
    assert "034_map_snapshots" in finish.call_args.kwargs["error"]


def test_run_cancelled_error_is_the_orchestrators():
    assert run_service.RunCancelledError is RunCancelledError


# ----------------------------------------------------------- migration


def _statement(sql, pattern):
    match = re.search(pattern, sql, re.S | re.I)
    assert match, pattern
    return match.group(0)


def test_migration_keeps_every_kind_and_trigger_and_adds_the_map():
    kind = _statement(MIGRATION_SQL, r"ADD CONSTRAINT pipeline_runs_kind_check.*?;")
    for value in (
        "run_daily",
        "process_selected",
        "discover",
        "podcast_audio",
        "upload",
        "podcast_script",
        "podcast_factcheck",
        "playground_explain",
        "playground_align",
        "map_build",
    ):
        assert f"'{value}'" in kind
        assert f"'{value}'" in _statement(
            SCHEMA, r"CONSTRAINT pipeline_runs_kind_check.*?\)\)"
        )
    trigger = _statement(
        MIGRATION_SQL, r"ADD CONSTRAINT pipeline_runs_trigger_source_check.*?;"
    )
    for value in (
        "ui_pipeline",
        "ui_select",
        "ui_discover",
        "scheduler",
        "ui_podcast_audio",
        "ui_upload",
        "ui_podcast_script",
        "ui_podcast_factcheck",
        "auto_podcast_factcheck",
        "ui_playground",
        "ui_map",
    ):
        assert f"'{value}'" in trigger
        assert f"'{value}'" in _statement(
            SCHEMA, r"CONSTRAINT pipeline_runs_trigger_source_check.*?\)\)"
        )
    assert (
        "GRANT SELECT, INSERT, UPDATE, DELETE ON map_snapshots TO web_anon"
        in MIGRATION_SQL
    )
    assert "GRANT EXECUTE ON FUNCTION put_map_snapshot" in MIGRATION_SQL
    assert MIGRATION_SQL.rstrip().endswith("NOTIFY pgrst, 'reload schema';")


def test_migration_and_schema_agree_on_the_table_and_function():
    def block(sql, head, end):
        return sql.split(head, 1)[1].split(end, 1)[0]

    assert block(
        MIGRATION_SQL, "CREATE TABLE IF NOT EXISTS map_snapshots (", "\n);"
    ) == block(SCHEMA, "CREATE TABLE map_snapshots (", "\n);")
    assert block(
        MIGRATION_SQL, "CREATE OR REPLACE FUNCTION put_map_snapshot(", "$$;"
    ) == block(SCHEMA, "CREATE OR REPLACE FUNCTION put_map_snapshot(", "$$;")


def _runs_table(sql, pattern):
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE pipeline_runs (id INTEGER PRIMARY KEY, pillar_id TEXT, kind TEXT, status TEXT, result TEXT)"
    )
    conn.execute(_statement(sql, pattern).replace("IF NOT EXISTS ", ""))
    return conn


def _insert(conn, kind, status="running", pillar="p"):
    conn.execute(
        "INSERT INTO pipeline_runs (pillar_id, kind, status) VALUES (?, ?, ?)",
        (pillar, kind, status),
    )


@pytest.mark.parametrize("sql", [MIGRATION_SQL, SCHEMA], ids=["migration", "schema"])
def test_a_map_build_may_overlap_a_daily_run_on_the_same_pillar(sql):
    conn = _runs_table(
        sql,
        r"CREATE UNIQUE INDEX (IF NOT EXISTS )?pipeline_runs_one_active_per_pillar[^;]*;",
    )
    _insert(conn, "run_daily")
    _insert(conn, "map_build")
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, "discover")


@pytest.mark.parametrize("sql", [MIGRATION_SQL, SCHEMA], ids=["migration", "schema"])
def test_one_active_map_build_per_pillar(sql):
    conn = _runs_table(
        sql,
        r"CREATE UNIQUE INDEX (IF NOT EXISTS )?pipeline_runs_one_active_map_build[^;]*;",
    )
    _insert(conn, "map_build")
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, "map_build", "pending")
    _insert(conn, "map_build", "succeeded")
    _insert(conn, "map_build", pillar="other")


# ----------------------------------------------------------------- CLI


def test_cli_map_build_dry_run_prints_the_report():
    from nlp_pillars.cli import app

    report = map_store.MapBuildReport(dry_run=True, model_id=MODEL)
    with (
        patch("nlp_pillars.map_store.execute_map_build", return_value=report) as run,
        patch("nlp_pillars.cli._validate_pillar", return_value=PILLAR),
    ):
        result = CliRunner().invoke(
            app, ["map", "build", "--pillar", PILLAR, "--dry-run"]
        )
    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs == {"pillar": PILLAR, "dry_run": True}
    assert "map build (dry-run)" in result.output


def test_cli_map_build_exits_1_and_names_the_migration():
    from nlp_pillars.cli import app

    with patch(
        "nlp_pillars.map_store.execute_map_build",
        side_effect=MapStoreError(
            "Apply docs/migrations/034_map_snapshots.sql by hand."
        ),
    ):
        result = CliRunner().invoke(app, ["map", "build"])
    assert result.exit_code == 1
    assert "034_map_snapshots.sql" in result.output
