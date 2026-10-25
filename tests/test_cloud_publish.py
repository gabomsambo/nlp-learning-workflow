"""cloud publish: hermetic. A fake ingest server stands in for the Worker.

The fake follows the Worker's rules (content-hash check, manifest refused while
an object it names is missing, inventory with hashes) so the publisher is tested
against the contract, not against its own assumptions. ``test_cloud_ingest_worker``
runs the same publisher against the real Worker code.
"""

import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import unquote

import httpx
import pytest
from typer.testing import CliRunner

from nlp_pillars.cli import app
from nlp_pillars.cloud import publish as cloud_publish
from nlp_pillars.cloud.export import paper_key
from nlp_pillars.cloud.publish import (
    IngestClient, PostgrestState, PublishConfig, PublishError, publish_bundle, run_publish,
)
from nlp_pillars.cloud.source import load_source
# Shared fixtures: the same fixture database `cloud export` is tested with.
from test_cloud_export import (  # noqa: F401  (audio_dir is a fixture)
    NOW, FakeClient, Transcoder, audio_dir, export, fixture_rows,
)

SECRET = "s3cret-service-token-value"
CONFIG = PublishConfig(
    url="https://reader.example.test", client_id="cid.access", client_secret=SECRET, max_bytes=10_000_000,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeIngest:
    """The Worker's /ingest contract over an httpx MockTransport."""

    def __init__(self):
        self.objects = {}      # key -> (bytes, sha256)
        self.log = []          # (method, key) in arrival order
        self.fail = None       # callable(method, key, n) -> response | Exception | None
        self.puts = 0
        self.seen_headers = []
        self.gc_deleted = 0

    def transport(self):
        return httpx.MockTransport(self.handle)

    def client(self, sleeps=None):
        http = httpx.Client(transport=self.transport())
        return IngestClient(CONFIG, client=http, sleep=(sleeps.append if sleeps is not None else lambda s: None))

    def json(self, status, body):
        return httpx.Response(status, json=body)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.seen_headers.append(dict(request.headers))
        path = request.url.path
        if request.headers.get("cf-access-client-secret") != SECRET:
            return httpx.Response(403, text="<html>Access denied</html>")
        if path == "/ingest/inventory":
            objects = [{"key": k, "size": len(v[0]), "sha256": v[1]} for k, v in sorted(self.objects.items())]
            return self.json(200, {"objects": objects, "bytes": sum(len(v[0]) for v in self.objects.values())})
        if path == "/ingest/manifest":
            key = "manifest.json"
        elif path.startswith("/ingest/object/"):
            key = unquote(path[len("/ingest/object/"):])
        else:
            return self.json(404, {"error": "not found"})
        self.log.append((request.method, key))
        self.puts += 1
        if self.fail:
            injected = self.fail(request.method, key, self.puts)
            if isinstance(injected, Exception):
                raise injected
            if injected is not None:
                return injected
        body = request.content
        if sha(body) != request.headers.get("x-content-sha256", ""):
            return self.json(400, {"error": "body does not match x-content-sha256"})
        if key == "manifest.json":
            manifest = json.loads(body)
            wanted = {e["key"]: e for e in manifest["docs"] + manifest["audio"]}
            missing = [k for k, e in wanted.items() if k not in self.objects or self.objects[k][1] != e["sha256"]]
            if missing:
                return self.json(409, {"error": "manifest names objects that are not stored yet", "missing": missing})
            for k in [k for k in self.objects if k != "manifest.json" and k not in wanted]:
                if any(t["key"] == k for t in manifest.get("tombstones", [])):
                    del self.objects[k]
                    self.gc_deleted += 1
            self.objects[key] = (body, sha(body))
            return self.json(200, {"ok": True, "deleted": self.gc_deleted})
        self.objects[key] = (body, sha(body))
        return self.json(201, {"ok": True, "key": key})

    def served_manifest_is_complete(self) -> bool:
        manifest = json.loads(self.objects["manifest.json"][0])
        return all(
            e["key"] in self.objects and self.objects[e["key"]][1] == e["sha256"]
            for e in manifest["docs"] + manifest["audio"]
        )


class MemoryState:
    def __init__(self):
        self.rows = {}
        self.fail_record_after = None
        self.recorded = 0

    def load(self):
        return dict(self.rows)

    def record(self, key, sha256, size):
        self.rows[key] = (sha256, size)
        self.recorded += 1

    def forget(self, keys):
        for key in keys:
            self.rows.pop(key, None)


@pytest.fixture
def bundle(tmp_path, audio_dir):
    out = tmp_path / "bundle"
    export(out, audio_dir)
    return out


def rebuild(out, audio_dir, rows=None):
    export(out, audio_dir, source=load_source(FakeClient(rows or fixture_rows())))


def methods_in_order(server):
    return [f"{m} {k}" for m, k in server.log]


# --- the happy path -------------------------------------------------------------


def test_first_publish_uploads_everything_and_the_manifest_last(bundle):
    server, state = FakeIngest(), MemoryState()
    report = publish_bundle(bundle, server.client(), state)

    manifest = json.loads((bundle / "manifest.json").read_text())
    expected = len(manifest["docs"]) + len(manifest["audio"])
    assert len(report.uploaded) == expected and expected > 3
    assert report.manifest_sent and server.log[-1] == ("PUT", "manifest.json")
    assert [k for _, k in server.log].count("manifest.json") == 1
    # audio before documents
    kinds = ["audio" if k.startswith("audio/") else "doc" for _, k in server.log[:-1]]
    assert kinds == sorted(kinds, key=lambda k: k != "audio")
    assert server.served_manifest_is_complete()
    assert set(state.rows) == {e["key"] for e in manifest["docs"] + manifest["audio"]} | {"manifest.json"}
    assert report.total_bytes == sum(e["bytes"] for e in manifest["docs"] + manifest["audio"])


def test_a_second_publish_with_nothing_changed_sends_nothing(bundle):
    server, state = FakeIngest(), MemoryState()
    publish_bundle(bundle, server.client(), state)
    server.log.clear()
    report = publish_bundle(bundle, server.client(), state)
    assert server.log == [] and report.uploaded == [] and not report.manifest_sent
    assert report.unchanged > 3


def test_only_a_changed_document_and_the_manifest_go_up(bundle, audio_dir):
    server, state = FakeIngest(), MemoryState()
    publish_bundle(bundle, server.client(), state)
    rows = fixture_rows()
    rows["lessons"][0]["tl_dr"] = "A different summary."
    rebuild(bundle, audio_dir, rows)
    server.log.clear()
    report = publish_bundle(bundle, server.client(), state)
    sent = methods_in_order(server)
    assert len(sent) == 2, sent
    assert sent[0].startswith("PUT papers/") and sent[1] == "PUT manifest.json"
    assert report.manifest_sent
    assert server.served_manifest_is_complete()


def test_credentials_travel_as_headers_and_never_in_a_url_or_report(bundle, caplog):
    server, state = FakeIngest(), MemoryState()
    with caplog.at_level(logging.DEBUG):
        report = publish_bundle(bundle, server.client(), state)
    assert all(h.get("cf-access-client-id") == "cid.access" for h in server.seen_headers)
    assert SECRET not in report.render() and SECRET not in caplog.text


# --- interruption, retry, and the manifest-last guarantee ------------------------


def test_an_interrupted_publish_resumes_where_it_stopped(bundle):
    server, state = FakeIngest(), MemoryState()
    manifest = json.loads((bundle / "manifest.json").read_text())
    total = len(manifest["docs"]) + len(manifest["audio"])

    # The connection dies for good on the fourth object (every retry fails).
    server.fail = lambda method, key, n: httpx.ConnectError("link down") if n >= 4 else None
    with pytest.raises(PublishError, match="gave up after 3 tries"):
        publish_bundle(bundle, server.client(), state)
    assert "manifest.json" not in server.objects, "the manifest must not go up after a failed upload"
    stored = {k for k in server.objects}
    assert len(stored) == 3 and set(state.rows) == stored

    server.fail = None
    server.log.clear()
    report = publish_bundle(bundle, server.client(), state)
    resent = {k for _, k in server.log}
    assert not resent & stored, "objects that already landed are not sent again"
    assert len(report.uploaded) == total - 3
    assert server.served_manifest_is_complete()


def test_a_transient_server_error_is_retried_with_backoff(bundle):
    server, state, sleeps = FakeIngest(), MemoryState(), []
    flaky = {"left": 2}

    def fail(method, key, n):
        if key == "library.json" and flaky["left"]:
            flaky["left"] -= 1
            return httpx.Response(503, json={"error": "try later"})

    server.fail = fail
    publish_bundle(bundle, server.client(sleeps), state)
    assert sleeps == [2.0, 4.0]
    assert server.served_manifest_is_complete()


def test_a_client_error_is_not_retried_and_says_why(bundle):
    server, state, sleeps = FakeIngest(), MemoryState(), []
    server.fail = lambda m, k, n: httpx.Response(400, json={"error": "a document must be a JSON object"})
    with pytest.raises(PublishError, match="HTTP 400 a document must be a JSON object"):
        publish_bundle(bundle, server.client(sleeps), state)
    assert sleeps == [] and server.puts == 1


def test_a_refused_token_names_access_not_the_secret(bundle):
    server, state = FakeIngest(), MemoryState()
    wrong = PublishConfig(url=CONFIG.url, client_id="cid.access", client_secret="not-the-secret-at-all")
    client = IngestClient(wrong, client=httpx.Client(transport=server.transport()), sleep=lambda s: None)
    with pytest.raises(PublishError) as raised:
        publish_bundle(bundle, client, state)
    assert "HTTP 403" in str(raised.value) and "not the secret" not in str(raised.value)


def test_the_manifest_is_refused_while_an_object_is_missing_and_the_error_says_so(bundle):
    server, state = FakeIngest(), MemoryState()
    publish_bundle(bundle, server.client(), state)
    # The bucket loses a document behind the publisher's back; its state row still says it is there.
    lost = next(k for k in server.objects if k.startswith("papers/"))
    del server.objects[lost]
    manifest = (bundle / "manifest.json").read_bytes()
    with pytest.raises(PublishError, match="HTTP 409"):
        server.client().put_manifest(manifest, sha(manifest))


def test_a_wiped_bucket_is_refilled_even_though_state_says_published(bundle):
    server, state = FakeIngest(), MemoryState()
    publish_bundle(bundle, server.client(), state)
    server.objects.clear()
    report = publish_bundle(bundle, server.client(), state)
    assert len(report.uploaded) > 3 and report.manifest_sent
    assert server.served_manifest_is_complete()


def test_a_lost_state_table_is_repaired_without_uploading_again(bundle):
    server, state = FakeIngest(), MemoryState()
    publish_bundle(bundle, server.client(), state)
    state.rows.clear()
    server.log.clear()
    report = publish_bundle(bundle, server.client(), state)
    assert report.uploaded == [] and report.repaired > 3 and len(state.rows) > 3


def test_full_sends_everything_again(bundle):
    server, state = FakeIngest(), MemoryState()
    publish_bundle(bundle, server.client(), state)
    server.log.clear()
    report = publish_bundle(bundle, server.client(), state, full=True)
    assert len(report.uploaded) > 3 and report.manifest_sent


def test_a_file_changed_after_export_is_caught_before_it_is_sent(bundle):
    server, state = FakeIngest(), MemoryState()
    target = next((bundle / "papers").iterdir())
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(PublishError, match="changed on disk"):
        publish_bundle(bundle, server.client(), state)
    assert "manifest.json" not in server.objects


# --- deletion, size cap ---------------------------------------------------------


def test_a_deleted_paper_is_dropped_from_state_and_swept_by_the_cloud(bundle, audio_dir):
    server, state = FakeIngest(), MemoryState()
    publish_bundle(bundle, server.client(), state)
    rows = fixture_rows()
    gone_id = rows["papers"][-1]["id"]
    for table in ("papers", "notes", "lessons", "quiz_cards", "podcast_scripts"):
        rows[table] = [r for r in rows[table] if r.get("id") != gone_id and r.get("paper_id") != gone_id]
    rows["paper_concepts"] = [r for r in rows["paper_concepts"] if r["paper_id"] != gone_id]
    rows["graph_edges"] = [r for r in rows["graph_edges"] if gone_id not in (str(r["src_id"]), str(r["dst_id"]))]
    gone_key = paper_key(gone_id)
    assert gone_key in server.objects
    rebuild(bundle, audio_dir, rows)

    report = publish_bundle(bundle, server.client(), state)
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert gone_key in {t["key"] for t in manifest["tombstones"]}
    assert gone_key not in state.rows and report.removed >= 1
    assert gone_key not in server.objects, "the cloud sweeps a tombstoned object"
    assert server.served_manifest_is_complete()


def test_a_bundle_over_the_cap_uploads_nothing(bundle):
    server, state = FakeIngest(), MemoryState()
    with pytest.raises(PublishError, match="over the .* cap"):
        publish_bundle(bundle, server.client(), state, max_bytes=1000)
    assert server.log == [] and state.rows == {}


def test_what_the_cloud_already_holds_counts_against_the_cap(bundle):
    server, state = FakeIngest(), MemoryState()
    server.objects["papers/old.json"] = (b"x" * 5000, sha(b"x" * 5000))
    total = sum(e["bytes"] for e in json.loads((bundle / "manifest.json").read_text())["docs"])
    with pytest.raises(PublishError, match="already holds"):
        publish_bundle(bundle, server.client(), state, max_bytes=total + 2000)
    assert server.log == []


def test_a_dry_run_reads_but_writes_nothing(bundle):
    server, state = FakeIngest(), MemoryState()
    report = publish_bundle(bundle, server.client(), state, dry_run=True)
    assert report.dry_run and len(report.uploaded) > 3
    assert server.log == [] and state.rows == {}
    assert "would upload" in report.render()


# --- configuration --------------------------------------------------------------


class Settings:
    cloud_publish_url = "https://read.example.test/"
    cf_access_client_id = "cid.access"
    cf_access_client_secret = SECRET
    cloud_publish_max_bytes = 123


def test_config_names_exactly_what_is_missing():
    class Empty(Settings):
        cloud_publish_url = None
        cf_access_client_secret = " "

    with pytest.raises(PublishError, match="CLOUD_PUBLISH_URL, CF_ACCESS_CLIENT_SECRET"):
        PublishConfig.from_settings(Empty())
    config = PublishConfig.from_settings(Settings())
    assert config.url == "https://read.example.test" and config.max_bytes == 123


def test_config_refuses_plain_http_to_the_internet():
    class Plain(Settings):
        cloud_publish_url = "http://read.example.test"

    with pytest.raises(PublishError, match="https"):
        PublishConfig.from_settings(Plain())


# --- cloud_publish_state over PostgREST ------------------------------------------


class FakeResponse:
    def __init__(self, status=201, text=""):
        self.status_code, self.text = status, text


class FakePostgrest:
    base_url = "http://pg.test"
    headers = {"Authorization": "Bearer x"}

    def __init__(self, post=None, delete=None, select=None):
        self.client = self
        self._post, self._delete, self._select = post, delete, select
        self.calls = []

    def post(self, url, **kw):
        self.calls.append(("post", url, kw))
        return self._post or FakeResponse()

    def delete(self, url, **kw):
        self.calls.append(("delete", url, kw))
        return self._delete or FakeResponse(204)

    def table(self, name):
        return self

    def select(self, *_):
        return self

    order = limit = offset = select

    def execute(self):
        return self._select or {"data": [{"doc_key": "library.json", "content_hash": "a" * 64, "byte_size": 5}], "error": None}


def test_state_upserts_on_the_key_and_forgets_in_batches():
    pg = FakePostgrest()
    state = PostgrestState(pg)
    assert state.load() == {"library.json": ("a" * 64, 5)}
    state.record("papers/a.json", "b" * 64, 9)
    _, url, kw = pg.calls[0]
    assert url.endswith("/cloud_publish_state") and kw["params"] == {"on_conflict": "doc_key"}
    assert kw["headers"]["Prefer"] == "resolution=merge-duplicates,return=minimal"
    state.forget([f"papers/{i}.json" for i in range(120)])
    assert [c[0] for c in pg.calls].count("delete") == 3


def test_a_missing_state_table_names_the_migration():
    pg = FakePostgrest(select={"data": None, "error": {"code": "PGRST205", "message": "Could not find the table 'public.cloud_publish_state' in the schema cache"}})
    with pytest.raises(PublishError, match="026_cloud_publish_state.sql"):
        PostgrestState(pg).load()
    pg = FakePostgrest(post=FakeResponse(404, '{"message":"relation \\"cloud_publish_state\\" does not exist"}'))
    with pytest.raises(PublishError, match="026_cloud_publish_state.sql"):
        PostgrestState(pg).record("library.json", "a" * 64, 1)


# --- export + publish together, and the CLI ---------------------------------------


def test_run_publish_exports_first_and_a_failed_export_sends_nothing(tmp_path):
    server, state = FakeIngest(), MemoryState()
    from nlp_pillars.cloud.export import ExportError

    def broken(_out):
        raise ExportError("a document failed the guard")

    with pytest.raises(PublishError, match="Export failed: a document failed the guard"):
        run_publish(CONFIG, tmp_path / "b", ingest=server.client(), state=state, export=broken)
    assert server.log == []


def test_cli_publish_prints_the_report_and_exits_nonzero_on_failure(monkeypatch, bundle):
    server, state = FakeIngest(), MemoryState()

    class Cfg:
        cloud_publish_url = CONFIG.url
        cf_access_client_id = "cid.access"
        cf_access_client_secret = SECRET
        cloud_publish_max_bytes = 10_000_000
        cloud_bundle_dir = str(bundle)

    monkeypatch.setattr("nlp_pillars.config.get_settings", lambda: Cfg())
    monkeypatch.setattr(cloud_publish, "PostgrestState", lambda: state)
    monkeypatch.setattr(cloud_publish, "run_export", lambda out: None)
    monkeypatch.setattr(
        cloud_publish, "IngestClient",
        lambda config: IngestClient(config, client=httpx.Client(transport=server.transport()), sleep=lambda s: None),
    )
    result = CliRunner().invoke(app, ["cloud", "publish"])
    assert result.exit_code == 0, result.output
    assert "uploaded" in result.output and SECRET not in result.output

    server.fail = lambda m, k, n: httpx.Response(400, json={"error": "nope"})
    server.objects.clear()
    state.rows.clear()
    failed = CliRunner().invoke(app, ["cloud", "publish"])
    assert failed.exit_code == 1 and "HTTP 400 nope" in failed.output and SECRET not in failed.output


def test_cli_publish_without_credentials_says_which_are_missing(monkeypatch):
    class Cfg:
        cloud_publish_url = None
        cf_access_client_id = None
        cf_access_client_secret = None
        cloud_bundle_dir = "/nonexistent"

    monkeypatch.setattr("nlp_pillars.config.get_settings", lambda: Cfg())
    result = CliRunner().invoke(app, ["cloud", "publish"])
    assert result.exit_code == 1
    assert "CLOUD_PUBLISH_URL, CF_ACCESS_CLIENT_ID, CF_ACCESS_CLIENT_SECRET" in result.output
