"""cloud export: the offline bundle. Hermetic: no database, network or ffmpeg.

The fixture database is ``tests/fixtures/cloud/db.json``. It carries the columns
the reader must not publish (ground packs, options, voice paths, an upload's
``file://`` path, judge trails, chunk ids) so the allow-list is tested against
rows that try to leak.
"""

import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nlp_pillars.cli import app
from nlp_pillars.cloud import export as cloud_export
from nlp_pillars.cloud import source as cloud_source
from nlp_pillars.cloud.export import (
    ExportError,
    build_documents,
    canonical_bytes,
    env_secret_values,
    run_export,
    safe_name,
)
from nlp_pillars.cloud.source import CloudSource, load_source

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "cloud"
GOLDEN = FIXTURES / "golden"
NOW = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)
SCRIPT_WITH_AUDIO = "11111111-1111-4111-8111-111111111111"
POISON = [
    "POISON-GROUND-PACK", "POISON-OPTIONS", "POISON-GP-CALLS", "POISON-VOICE",
    "POISON-CHUNK-ID", "POISON-JUDGE-MODEL", "POISON-DISMISSED-NOTE", "POISON-MODEL",
    "/app/data", "file://", "Podcasts/voices", "attention_20261022.mp3", "uploads/",
]
AUDIO_BYTES = b"original-mp3-bytes"


def fixture_rows() -> dict:
    return json.loads((FIXTURES / "db.json").read_text("utf-8"))


class FakeQuery:
    """The slice of ``db.TableQuery`` the loader uses, over fixture rows.

    Unlike a real fixture dump it *projects*: a column that is not in the
    ``select`` list does not come back, and a column the table does not have is
    an error, as PostgREST makes it.
    """

    def __init__(self, rows, columns_of, table):
        self.rows, self.columns_of, self.table = rows, columns_of, table
        self.params = {}
        self.filters = {}

    def select(self, columns="*"):
        self.params["select"] = columns
        return self

    def eq(self, column, value):
        self.filters[column] = value
        return self

    def order(self, column, desc=False):
        self.params["order"] = column
        return self

    def limit(self, count):
        self.params["limit"] = int(count)
        return self

    def offset(self, count):
        self.params["offset"] = int(count)
        return self

    def execute(self):
        known = self.columns_of[self.table]
        wanted = self.params["select"].split(",")
        unknown = [c for c in wanted + list(self.filters) if c not in known]
        if unknown:
            return {"data": None, "error": {"code": "42703", "message": f"{self.table}: no column {unknown}"}}
        rows = [r for r in self.rows[self.table] if all(r.get(k) == v for k, v in self.filters.items())]
        order = self.params["order"].split(",")
        rows.sort(key=lambda r: tuple(str(r.get(c)) for c in order))
        start = self.params.get("offset", 0)
        page = rows[start:start + self.params["limit"]]
        self.log.append((self.table, self.params["select"]))
        return {"data": [{c: r.get(c) for c in wanted} for r in page], "error": None}

    log: list = []


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.columns_of = schema_columns()
        self.selects = []

    def table(self, name):
        query = FakeQuery(self.rows, self.columns_of, name)
        query.log = self.selects
        return query


def schema_columns() -> dict:
    """table -> columns, read off the CREATE TABLE blocks in schema.sql."""
    text = (ROOT / "schema.sql").read_text("utf-8")
    tables = {}
    for match in re.finditer(r"^CREATE TABLE (\w+) \((.*?)^\);", text, re.S | re.M):
        columns = set()
        for line in match.group(2).splitlines():
            m = re.match(r"    ([a-z_][a-z0-9_]*)\s+[A-Za-z]", line)
            if m and m.group(1) not in {"constraint", "primary", "foreign", "unique", "check"}:
                columns.add(m.group(1))
        tables[match.group(1)] = columns
    return tables


def raw_source(rows=None) -> CloudSource:
    """Full rows, poison included, handed straight to the serializers."""
    rows = rows or fixture_rows()
    return CloudSource(
        pillars=rows["pillars"], papers=rows["papers"], notes=rows["notes"],
        lessons=rows["lessons"], cards=rows["quiz_cards"], scripts=rows["podcast_scripts"],
        fact_checks=[r for r in rows["podcast_fact_checks"] if r["id"].endswith("2")],
        claims=[r for r in rows["podcast_claims"] if r["fact_check_id"].endswith("2")],
        concepts=rows["concepts"], paper_concepts=rows["paper_concepts"],
        edges=[r for r in rows["graph_edges"] if r["kind"] == "related_to"],
    )


class Transcoder:
    def __init__(self):
        self.calls = []

    def __call__(self, source: Path, dest: Path) -> float:
        self.calls.append(source.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"64k:" + source.read_bytes())
        return 12.5


@pytest.fixture
def audio_dir(tmp_path):
    folder = tmp_path / "podcast_audio"
    folder.mkdir()
    (folder / "attention_20261022.mp3").write_bytes(AUDIO_BYTES)
    return folder


def export(out, audio_dir, source=None, transcoder=None, **kw):
    transcoder = transcoder or Transcoder()
    kw.setdefault("now", NOW)
    kw.setdefault("secrets", [])
    report = run_export(
        out, source=source or load_source(FakeClient(fixture_rows())),
        audio_dir=audio_dir, transcoder=transcoder, **kw,
    )
    return report, transcoder


def tree(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


# --- the loader ---------------------------------------------------------------


def test_loader_selects_only_existing_columns_and_nothing_secret():
    client = FakeClient(fixture_rows())
    load_source(client)  # a column that is not in schema.sql would be an error here
    selected = {col for _, cols in client.selects for col in cols.split(",")}
    assert not selected & {
        "ground_pack", "ground_pack_calls", "options", "text", "embedding", "chunk_id",
        "judge_trail", "dismissed_note", "models", "cost_usd", "user_id", "paper_text_sha256",
    }
    assert {t for t, _ in client.selects} >= {"papers", "podcast_claims", "graph_edges"}


def test_loader_pages_every_table(monkeypatch):
    monkeypatch.setattr(cloud_source, "_PAGE", 1)
    source = load_source(FakeClient(fixture_rows()))
    assert len(source.papers) == 3 and len(source.cards) == 3 and len(source.notes) == 3


def test_loader_reads_only_the_newest_succeeded_check_and_its_claims():
    source = load_source(FakeClient(fixture_rows()))
    assert [c["id"][-1] for c in source.fact_checks] == ["2"]
    assert len(source.claims) == 3  # the superseded check's claim is not read
    assert [e["kind"] for e in source.edges] == ["related_to", "related_to"]


def test_loader_names_the_migration_for_a_missing_table():
    class Missing(FakeClient):
        def table(self, name):
            query = super().table(name)
            if name == "concepts":
                query.execute = lambda: {
                    "data": None,
                    "error": {"code": "PGRST205", "message": "Could not find the table 'public.concepts' in the schema cache"},
                }
            return query

    with pytest.raises(cloud_source.SourceError, match="020_knowledge_graph"):
        load_source(Missing(fixture_rows()))


# --- the allow-list ------------------------------------------------------------


def test_no_secret_or_internal_field_reaches_any_document():
    docs = build_documents(raw_source(), {})
    blob = b"".join(canonical_bytes(d) for d in docs.values()).decode("utf-8")
    for marker in POISON:
        assert marker not in blob, marker
    keys = set(re.findall(r'"([a-z_0-9]+)":', blob))
    assert not keys & cloud_export.FORBIDDEN_KEYS
    assert not [k for k in keys if cloud_export._SECRET_KEY.search(k)]


def test_upload_path_is_dropped_and_web_links_kept():
    docs = build_documents(raw_source(), {})
    assert docs["papers/file_3aabc123def456.json"]["url_pdf"] is None
    assert docs["papers/arxiv_3a1706.03762.json"]["url_pdf"] == "https://arxiv.org/pdf/1706.03762"


def test_evidence_keeps_the_quote_not_the_chunk():
    docs = build_documents(raw_source(), {})
    claims = docs[f"episodes/{SCRIPT_WITH_AUDIO}.json"]["fact_check"]["claims"]
    assert [c["id"][-1] for c in claims] == ["1", "2", "3"]  # by line, then span
    assert claims[0]["evidence"] == [
        {"quote": "Recurrent models factor computation.", "chunk_index": 1},
        {"quote": "Second quote.", "chunk_index": 2},
    ]
    assert claims[1]["dismissed_at"] and "dismissed_note" not in claims[1]


def test_newest_note_and_lesson_win_and_related_is_library_only():
    paper = build_documents(raw_source(), {})["papers/arxiv_3a1706.03762.json"]
    assert paper["note"]["problem"] == "Recurrence is sequential."
    assert paper["related"] == [{"paper_id": "file:abc123def456", "weight": 0.81}]
    assert [c["id"][-1] for c in paper["concepts"]] == ["1", "2"]
    assert len(paper["quiz_cards"]) == 2 and "user_id" not in paper["quiz_cards"][0]


def test_schedule_snapshot_is_the_card_cache():
    schedule = build_documents(raw_source(), {})["schedule.json"]
    assert schedule["through_seq"] == 0 and schedule["source"] == "quiz_cards"
    new, reviewed, _ = schedule["cards"]
    assert new["state"] == "new" and new["due"] is None
    assert reviewed == {
        "card_id": "a0000000-0000-4000-8000-000000000002", "paper_id": "arxiv:1706.03762",
        "state": "review", "due": "2026-10-24T12:00:00+00:00", "stability": 3.25,
        "difficulty": 5.5, "last_review": "2026-10-20T12:00:00+00:00", "lapses": 1,
    }


def test_guard_refuses_forbidden_keys_secrets_and_local_paths():
    guard = cloud_export._guard
    with pytest.raises(ExportError, match="api_key"):
        guard("x.json", {"nested": [{"api_key": "x"}]}, [])
    with pytest.raises(ExportError, match="ground_pack"):
        guard("x.json", {"ground_pack": {}}, [])
    with pytest.raises(ExportError, match="file:///app/"):
        guard("x.json", {"url": "file:///app/data/uploads/a.pdf"}, [])
    with pytest.raises(ExportError, match="sk-ant-"):
        guard("x.json", {"note": "sk-ant-api03-" + "a" * 30}, [])
    with pytest.raises(ExportError, match="secret environment"):
        guard("x.json", {"note": "hello s3cr3t-value-123 world"}, ["s3cr3t-value-123"])
    guard("x.json", {"key_terms": ["token"], "key_points": ["a key point"]}, ["s3cr3t-value-123"])


def test_env_secret_values_picks_secret_names_only():
    values = env_secret_values({
        "ANTHROPIC_API_KEY": "sk-ant-abcdefghijkl", "DEEPSEEK_API_KEY": "short",
        "SUPABASE_KEY": "eyJabcdefghijklmnop", "HOME": "/home/someone-long-enough",
        "QDRANT_API_KEY": "",
    })
    assert values == ["eyJabcdefghijklmnop", "sk-ant-abcdefghijkl"]


def test_a_secret_in_content_fails_the_export_and_writes_nothing(tmp_path, audio_dir):
    rows = fixture_rows()
    rows["papers"][0]["abstract"] = "leaked sk-ant-api03-" + "z" * 30
    out = tmp_path / "bundle"
    with pytest.raises(ExportError, match="sk-ant-"):
        export(out, audio_dir, source=load_source(FakeClient(rows)))
    assert not out.exists()  # the guard runs before any transcode

    rows = fixture_rows()
    rows["papers"][0]["abstract"] = "token is hunter2-hunter2-hunter2"
    with pytest.raises(ExportError, match="secret environment"):
        export(out, audio_dir, source=load_source(FakeClient(rows)),
               secrets=["hunter2-hunter2-hunter2"])
    assert not out.exists()


# --- canonical JSON and names -------------------------------------------------


def test_canonical_json_is_sorted_compact_utf8():
    assert canonical_bytes({"b": [1, 2], "a": "—"}) == '{"a":"—","b":[1,2]}\n'.encode("utf-8")
    with pytest.raises(ExportError):
        canonical_bytes({"x": float("nan")})
    with pytest.raises(ExportError):
        canonical_bytes({"x": datetime.now()})


def test_safe_name_is_injective_and_url_safe():
    ids = ["arxiv:1706.03762", "cs/0301001", "cs_2f0301001", "a_b", "a b", "é", "file:abc"]
    names = [safe_name(i) for i in ids]
    assert len(set(names)) == len(ids)
    assert all(re.fullmatch(r"[A-Za-z0-9._-]+", n) for n in names)
    assert safe_name("cs/0301001") == "cs_2f0301001" and safe_name("cs_2f0301001") == "cs_5f2f0301001"


# --- the bundle ------------------------------------------------------------------


def test_bundle_matches_the_golden_files(tmp_path, audio_dir):
    out = tmp_path / "bundle"
    export(out, audio_dir)
    if os.environ.get("UPDATE_GOLDEN"):
        shutil.rmtree(GOLDEN, ignore_errors=True)
        shutil.copytree(out, GOLDEN)
    assert tree(out) == tree(GOLDEN)


def test_manifest_lists_every_file_with_its_hash_and_is_written_last(tmp_path, audio_dir):
    out = tmp_path / "bundle"
    order = []
    original = cloud_export._write_atomic
    cloud_export._write_atomic = lambda path, data: (order.append(path.name), original(path, data))[1]
    try:
        report, _ = export(out, audio_dir)
    finally:
        cloud_export._write_atomic = original
    manifest = json.loads((out / "manifest.json").read_text())
    assert order[-1] == "manifest.json" and order.count("manifest.json") == 1
    assert manifest["version"] == 1 and manifest["generated_at"] == "2026-10-25T12:00:00Z"
    assert manifest["fsrs"] == {"params_version": 1, "through_seq": 0}
    for item in manifest["docs"] + manifest["audio"]:
        data = (out / item["key"]).read_bytes()
        assert cloud_export.sha256_hex(data) == item["sha256"] and len(data) == item["bytes"]
    assert [d["key"] for d in manifest["docs"]] == sorted(d["key"] for d in manifest["docs"])
    assert report.docs_total == len(manifest["docs"]) == 7  # 3 papers, 2 episodes, library, schedule
    assert [a["key"] for a in manifest["audio"]] == [f"audio/{manifest['audio'][0]['sha256']}.mp3"]
    assert manifest["audio"][0]["duration_seconds"] == 12.5 and manifest["audio"][0]["bitrate_kbps"] == 64


def test_episode_with_audio_carries_the_timeline_and_one_without_does_not(tmp_path, audio_dir):
    out = tmp_path / "bundle"
    export(out, audio_dir)
    with_audio = json.loads((out / f"episodes/{SCRIPT_WITH_AUDIO}.json").read_text())
    assert with_audio["audio"]["key"].startswith("audio/") and with_audio["audio"]["engine"] == "indextts"
    assert with_audio["timeline"]["lines"][0]["words"][0] == [0, 500, 1]
    assert "engine" not in with_audio["timeline"] and "aligned_at" not in with_audio["timeline"]
    assert with_audio["fact_check"]["stale"] is False
    plain = json.loads((out / "episodes/22222222-2222-4222-8222-222222222222.json").read_text())
    assert plain["audio"] is None and plain["timeline"] is None and plain["fact_check"] is None
    assert plain["source_material"]["warnings"] == ["Written from the abstract only."]


def test_an_episode_without_a_timeline_is_tolerated(tmp_path, audio_dir):
    rows = fixture_rows()
    del rows["podcast_scripts"][0]["audio_metadata"]["timeline"]
    out = tmp_path / "bundle"
    export(out, audio_dir, source=load_source(FakeClient(rows)))
    doc = json.loads((out / f"episodes/{SCRIPT_WITH_AUDIO}.json").read_text())
    assert doc["audio"] and doc["timeline"] is None
    rows["podcast_scripts"][0]["audio_metadata"]["timeline"] = "not a timeline"
    export(out, audio_dir, source=load_source(FakeClient(rows)))
    assert json.loads((out / f"episodes/{SCRIPT_WITH_AUDIO}.json").read_text())["timeline"] is None


def test_a_stale_fact_check_is_marked(tmp_path, audio_dir):
    rows = fixture_rows()
    rows["podcast_scripts"][0]["script"] += "\n[HOST]: Edited afterwards."
    out = tmp_path / "bundle"
    export(out, audio_dir, source=load_source(FakeClient(rows)))
    doc = json.loads((out / f"episodes/{SCRIPT_WITH_AUDIO}.json").read_text())
    assert doc["fact_check"]["stale"] is True


def test_a_second_run_changes_nothing(tmp_path, audio_dir):
    out = tmp_path / "bundle"
    first, first_tc = export(out, audio_dir)
    stamps = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    before = tree(out)
    second, second_tc = export(out, audio_dir, now=NOW + timedelta(days=3))
    assert (second.docs_added, second.docs_changed, second.docs_removed) == (0, 0, 0)
    assert second.docs_unchanged == first.docs_total
    assert second.manifest_written is False and second.audio_reused == 1 and second.audio_transcoded == 0
    assert first_tc.calls and not second_tc.calls
    assert tree(out) == before
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()} == stamps


def test_one_changed_row_changes_one_document_and_the_manifest(tmp_path, audio_dir):
    out = tmp_path / "bundle"
    export(out, audio_dir)
    before = tree(out)
    rows = fixture_rows()
    rows["lessons"][0]["tl_dr"] = "Changed."
    report, tc = export(out, audio_dir, source=load_source(FakeClient(rows)), now=NOW + timedelta(days=1))
    changed = {k for k, v in tree(out).items() if before.get(k) != v}
    assert changed == {"papers/arxiv_3a1706.03762.json", "manifest.json"}
    assert report.docs_changed == 1 and report.manifest_written and not tc.calls
    assert json.loads((out / "manifest.json").read_text())["generated_at"] == "2026-10-26T12:00:00Z"


def test_a_deleted_paper_leaves_a_tombstone_that_expires(tmp_path, audio_dir):
    out = tmp_path / "bundle"
    export(out, audio_dir)
    rows = fixture_rows()
    rows["papers"] = [p for p in rows["papers"] if p["id"] != "cs/0301001"]
    report, _ = export(out, audio_dir, source=load_source(FakeClient(rows)), now=NOW + timedelta(days=1))
    assert report.docs_removed == 1 and not (out / "papers/cs_2f0301001.json").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["tombstones"] == [{"key": "papers/cs_2f0301001.json", "deleted_at": "2026-10-26T12:00:00Z"}]
    assert "papers/cs_2f0301001.json" not in [d["key"] for d in manifest["docs"]]

    export(out, audio_dir, source=load_source(FakeClient(rows)), now=NOW + timedelta(days=20))
    assert json.loads((out / "manifest.json").read_text())["tombstones"] == manifest["tombstones"]
    export(out, audio_dir, source=load_source(FakeClient(rows)), now=NOW + timedelta(days=40))
    assert json.loads((out / "manifest.json").read_text())["tombstones"] == []


def test_regenerated_audio_replaces_the_old_file_and_tombstones_it(tmp_path, audio_dir):
    out = tmp_path / "bundle"
    export(out, audio_dir)
    old = json.loads((out / "manifest.json").read_text())["audio"][0]["key"]
    (audio_dir / "attention_20261022.mp3").write_bytes(b"a new recording")
    _, tc = export(out, audio_dir, now=NOW + timedelta(days=1))
    manifest = json.loads((out / "manifest.json").read_text())
    assert tc.calls and manifest["audio"][0]["key"] != old
    assert not (out / old).exists() and old in [t["key"] for t in manifest["tombstones"]]
    assert sorted(p.name for p in (out / "audio").iterdir()) == [manifest["audio"][0]["key"][6:]]


def test_a_missing_mp3_publishes_the_episode_without_audio_and_says_so(tmp_path):
    empty = tmp_path / "no-audio"
    empty.mkdir()
    out = tmp_path / "bundle"
    report, tc = export(out, empty)
    assert not tc.calls and report.audio_files == 0
    assert any("attention_20261022.mp3 is not in" in w for w in report.warnings)
    assert json.loads((out / f"episodes/{SCRIPT_WITH_AUDIO}.json").read_text())["audio"] is None


def test_a_failed_transcode_stops_before_the_manifest(tmp_path, audio_dir):
    from nlp_pillars.cloud.audio import AudioError

    def failing(source, dest):
        raise AudioError("delta 3.0s")

    out = tmp_path / "bundle"
    with pytest.raises(ExportError, match="delta 3.0s"):
        export(out, audio_dir, transcoder=failing)
    assert not (out / "manifest.json").exists()
    assert not list(out.rglob("*.json"))
    export(out, audio_dir)  # the half-written directory is still ours; a retry works
    assert (out / "manifest.json").is_file()


def test_dry_run_writes_nothing_and_does_not_transcode(tmp_path, audio_dir):
    out = tmp_path / "bundle"
    report, tc = export(out, audio_dir, dry_run=True)
    assert not out.exists() and not tc.calls
    assert report.docs_added == report.docs_total == 7 and report.audio_pending == 1
    assert any("no cloud audio yet" in w for w in report.warnings)

    export(out, audio_dir)
    before = tree(out)
    again, tc = export(out, audio_dir, dry_run=True)
    assert tree(out) == before and not tc.calls
    assert (again.docs_added, again.docs_changed, again.audio_pending) == (0, 0, 0)


def test_a_directory_that_is_not_a_bundle_is_refused(tmp_path, audio_dir):
    out = tmp_path / "documents"
    out.mkdir()
    (out / "thesis.docx").write_bytes(b"x")
    with pytest.raises(ExportError, match="not a bundle"):
        export(out, audio_dir)
    assert (out / "thesis.docx").exists() and sorted(p.name for p in out.iterdir()) == ["thesis.docx"]


def test_orphans_in_managed_folders_go_and_foreign_files_stay(tmp_path, audio_dir):
    out = tmp_path / "bundle"
    export(out, audio_dir)
    (out / "papers" / "stale.json").write_text("{}")
    (out / "audio" / ".pending-x.mp3").write_bytes(b"x")
    (out / "README.txt").write_text("mine")
    export(out, audio_dir)
    assert not (out / "papers/stale.json").exists() and not (out / "audio/.pending-x.mp3").exists()
    assert (out / "README.txt").read_text() == "mine"


def test_bundle_text_is_far_smaller_than_the_database():
    docs = build_documents(raw_source(), {})
    assert sum(len(canonical_bytes(d)) for d in docs.values()) < 40_000


# --- the command ----------------------------------------------------------------


def test_cli_exports_and_reports(tmp_path, audio_dir, monkeypatch):
    monkeypatch.setattr(cloud_export, "load_source", lambda: load_source(FakeClient(fixture_rows())))
    monkeypatch.setenv("PODCAST_AUDIO_DIR", str(tmp_path / "none"))
    monkeypatch.setattr("nlp_pillars.config.get_settings", lambda: type("S", (), {"podcast_audio_dir": str(tmp_path / "none")})())
    out = tmp_path / "bundle"
    result = CliRunner().invoke(app, ["cloud", "export", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "documents: 7" in result.output and "written last" in result.output
    assert (out / "manifest.json").is_file()

    dry = CliRunner().invoke(app, ["cloud", "export", "--out", str(tmp_path / "other"), "--dry-run"])
    assert dry.exit_code == 0 and "(dry run)" in dry.output and not (tmp_path / "other").exists()


def test_cli_exits_one_with_the_reason_when_the_guard_trips(tmp_path, monkeypatch):
    rows = fixture_rows()
    rows["papers"][0]["abstract"] = "see file:///app/data/uploads/x.pdf"
    monkeypatch.setattr(cloud_export, "load_source", lambda: load_source(FakeClient(rows)))
    monkeypatch.setattr("nlp_pillars.config.get_settings", lambda: type("S", (), {"podcast_audio_dir": str(tmp_path)})())
    result = CliRunner().invoke(app, ["cloud", "export", "--out", str(tmp_path / "bundle")])
    assert result.exit_code == 1 and "must not be published" in result.output
