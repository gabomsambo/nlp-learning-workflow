"""The Python publisher against the real Worker code (cloud/reader/src), over an in-memory bucket.

``test_cloud_publish`` tests the publisher against a Python fake of the Worker's
contract; this runs the actual JavaScript, so the two cannot drift apart
unnoticed. Needs ``node`` (the CI runner has it); skipped without it. Binds
127.0.0.1 only.
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from nlp_pillars.cloud.publish import IngestClient, PublishConfig, PublishError, publish_bundle
from test_cloud_export import Transcoder, audio_dir, export, fixture_rows  # noqa: F401
from test_cloud_publish import MemoryState

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "cloud" / "reader" / "scripts" / "dev-ingest-server.mjs"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


class Mp3Transcoder(Transcoder):
    """The Worker checks audio for an MP3 frame header; the plain fake's bytes would be refused."""

    def __call__(self, source, dest):
        self.calls.append(source.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\xff\xfb\x90\x44" + source.read_bytes())
        return 12.5


@pytest.fixture
def worker():
    proc = subprocess.Popen(
        ["node", str(SERVER), "0"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    try:
        port = json.loads(proc.stdout.readline())["port"]
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.wait(5)


@pytest.fixture
def bundle(tmp_path, audio_dir):
    out = tmp_path / "bundle"
    export(out, audio_dir, transcoder=Mp3Transcoder())
    return out


def client_for(url, secret="anything"):
    config = PublishConfig(url=url, client_id="x", client_secret=secret, max_bytes=10_000_000)
    return IngestClient(config, sleep=lambda s: None)


def reader_get(url, path):
    return httpx.get(f"{url}{path}")


def test_a_bundle_published_to_the_real_worker_is_what_the_reader_serves(worker, bundle):
    state = MemoryState()
    report = publish_bundle(bundle, client_for(worker), state)
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert report.manifest_sent and len(report.uploaded) == len(manifest["docs"]) + len(manifest["audio"])

    served = reader_get(worker, "/api/manifest")
    assert served.status_code == 200 and served.content == (bundle / "manifest.json").read_bytes()
    for entry in manifest["docs"]:
        body = reader_get(worker, f"/api/doc/{entry['key']}").content
        assert hashlib.sha256(body).hexdigest() == entry["sha256"], entry["key"]

    inventory = client_for(worker).inventory()
    audio = [o for o in inventory["objects"] if o["key"].startswith("audio/")]
    assert audio and all(o["sha256"] == o["key"][6:-4] for o in audio)


def test_publishing_again_changes_nothing_and_lost_state_is_repaired_not_resent(worker, bundle):
    state = MemoryState()
    publish_bundle(bundle, client_for(worker), state)
    again = publish_bundle(bundle, client_for(worker), state)
    assert again.uploaded == [] and not again.manifest_sent
    # The inventory, not the state table, says what the cloud holds.
    fresh = publish_bundle(bundle, client_for(worker), MemoryState())
    assert fresh.uploaded == [] and fresh.repaired > 3


def test_the_worker_refuses_what_the_publisher_would_never_send(worker, bundle):
    client = client_for(worker)
    manifest = (bundle / "manifest.json").read_bytes()
    with pytest.raises(PublishError, match="HTTP 409"):
        client.put_manifest(manifest, hashlib.sha256(manifest).hexdigest())  # nothing uploaded yet
    with pytest.raises(PublishError, match="HTTP 400"):
        client.put_object("library.json", b'{"a":1}', "0" * 64)  # hash does not match
    with pytest.raises(PublishError, match="HTTP 404"):
        client.put_object("../escape.json", b"{}", hashlib.sha256(b"{}").hexdigest())
    with pytest.raises(PublishError, match="HTTP 400"):
        data = b"not audio"
        client.put_object(f"audio/{hashlib.sha256(data).hexdigest()}.mp3", data, hashlib.sha256(data).hexdigest())


def test_a_changed_episode_replaces_the_old_audio_in_the_manifest(worker, bundle, audio_dir):
    state = MemoryState()
    publish_bundle(bundle, client_for(worker), state)
    (audio_dir / "attention_20261022.mp3").write_bytes(b"a re-recorded episode")
    export(bundle, audio_dir, transcoder=Mp3Transcoder())  # a changed original is transcoded again
    report = publish_bundle(bundle, client_for(worker), state)
    new_audio = [k for k in report.uploaded if k.startswith("audio/")]
    assert len(new_audio) == 1
    manifest = json.loads((bundle / "manifest.json").read_text())
    served = json.loads(reader_get(worker, "/api/manifest").content)
    assert served == manifest
