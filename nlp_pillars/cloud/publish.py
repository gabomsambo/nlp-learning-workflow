"""``cloud publish``: send the bundle ``cloud export`` wrote to the cloud reader.

The cloud is a rebuildable projection of Postgres, so this is one-way and
idempotent. A run:

1. exports the bundle into a directory that survives restarts (the transcoded
   audio is cached there, so only a new episode costs ffmpeg time);
2. asks the Worker what it already holds (``GET /ingest/inventory``);
3. uploads each object that is new or changed, audio first and documents next,
   recording each one in ``cloud_publish_state`` as soon as it lands, so a run
   that dies half-way resumes instead of starting again;
4. writes the manifest **last**. The Worker refuses a manifest that names an
   object it does not hold, so the order is enforced on both sides.

An object counts as changed when ``cloud_publish_state`` has a different hash
for it, or when the Worker's inventory does not hold it with the bundle's hash
(a wiped bucket heals on the next run). Deletions need no call: a key missing
from the new manifest is a tombstone, and the Worker's garbage collection
removes the object after its grace period.

Nothing here raises into a caller that is running a pipeline: the webui's
``CloudPublisher`` catches everything. The CLI exits non-zero instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import quote

import httpx

from .export import ExportError, ExportReport, read_manifest, run_export

logger = logging.getLogger(__name__)

STATE_MIGRATION = "docs/migrations/026_cloud_publish_state.sql"
MANIFEST_KEY = "manifest.json"
DEFAULT_MAX_BYTES = 5_000_000_000
ATTEMPTS = 3
BACKOFF_SECONDS = 2.0
REQUEST_TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=300.0, pool=15.0)


class PublishError(Exception):
    """The publish stopped. What was uploaded stays uploaded; the next run resumes."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishConfig:
    url: str
    client_id: str
    client_secret: str
    max_bytes: int = DEFAULT_MAX_BYTES

    @classmethod
    def missing(cls, settings: Any) -> List[str]:
        """Names of the settings that are still empty."""
        wanted = (
            ("CLOUD_PUBLISH_URL", "cloud_publish_url"),
            ("CF_ACCESS_CLIENT_ID", "cf_access_client_id"),
            ("CF_ACCESS_CLIENT_SECRET", "cf_access_client_secret"),
        )
        return [env for env, attr in wanted if not (getattr(settings, attr, None) or "").strip()]

    @classmethod
    def from_settings(cls, settings: Any) -> "PublishConfig":
        missing = cls.missing(settings)
        if missing:
            raise PublishError(f"Set {', '.join(missing)} in .env to publish to the cloud reader.")
        url = settings.cloud_publish_url.strip().rstrip("/")
        if not url.startswith("https://") and not url.startswith("http://localhost") and not url.startswith("http://127.0.0.1"):
            raise PublishError("CLOUD_PUBLISH_URL must be an https:// address.")
        return cls(
            url=url,
            client_id=settings.cf_access_client_id.strip(),
            client_secret=settings.cf_access_client_secret.strip(),
            max_bytes=int(getattr(settings, "cloud_publish_max_bytes", DEFAULT_MAX_BYTES) or DEFAULT_MAX_BYTES),
        )

    def redact(self, text: str) -> str:
        for secret in (self.client_secret,):
            if secret and len(secret) >= 6:
                text = text.replace(secret, "••••")
        return text


# ---------------------------------------------------------------------------
# The Worker's side door
# ---------------------------------------------------------------------------


class IngestClient:
    """The three calls the Worker's ``/ingest`` answers. Retries 5xx and transport errors."""

    def __init__(
        self,
        config: PublishConfig,
        *,
        client: Optional[httpx.Client] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._sleep = sleep
        self._client = client or httpx.Client(timeout=REQUEST_TIMEOUT)
        self._headers = {
            "CF-Access-Client-Id": config.client_id,
            "CF-Access-Client-Secret": config.client_secret,
        }

    def close(self) -> None:
        self._client.close()

    def _send(self, method: str, path: str, *, body: Optional[bytes] = None, sha256: Optional[str] = None) -> Dict[str, Any]:
        headers = dict(self._headers)
        if sha256:
            headers["x-content-sha256"] = sha256
        if body is not None:
            headers["content-type"] = "application/octet-stream"
        last = ""
        for attempt in range(1, ATTEMPTS + 1):
            try:
                response = self._client.request(
                    method, f"{self.config.url}{path}", content=body, headers=headers,
                )
            except httpx.TransportError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code < 400:
                    return self._json(response)
                detail = self._detail(response)
                if response.status_code < 500 and response.status_code != 429:
                    raise PublishError(self.config.redact(f"{method} {path}: HTTP {response.status_code} {detail}"))
                last = f"HTTP {response.status_code} {detail}"
            if attempt < ATTEMPTS:
                self._sleep(BACKOFF_SECONDS * attempt)
        raise PublishError(self.config.redact(f"{method} {path}: gave up after {ATTEMPTS} tries ({last})"))

    @staticmethod
    def _json(response: httpx.Response) -> Dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            raise PublishError(f"The cloud reader answered something that is not JSON (HTTP {response.status_code}).") from None
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _detail(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            # Cloudflare Access answers its own pages when the token is refused.
            return "(not from the Worker)" if response.status_code in (401, 403) else response.text[:120].strip()
        if isinstance(data, dict):
            text = str(data.get("error") or "")
            if data.get("missing"):
                text += f" ({', '.join(map(str, data['missing'][:3]))}...)"
            return text
        return ""

    def inventory(self) -> Dict[str, Any]:
        return self._send("GET", "/ingest/inventory")

    def put_object(self, key: str, data: bytes, sha256: str) -> Dict[str, Any]:
        return self._send("PUT", f"/ingest/object/{quote(key, safe='')}", body=data, sha256=sha256)

    def put_manifest(self, data: bytes, sha256: str) -> Dict[str, Any]:
        return self._send("PUT", "/ingest/manifest", body=data, sha256=sha256)


# ---------------------------------------------------------------------------
# cloud_publish_state
# ---------------------------------------------------------------------------


class PostgrestState:
    """``cloud_publish_state`` over the project's PostgREST client."""

    _PAGE = 1000

    def __init__(self, client: Optional[Any] = None) -> None:
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from .. import db

            self._client = db.get_client()
        return self._client

    @staticmethod
    def _fail(action: str, error: Any) -> PublishError:
        text = str(error)
        lowered = text.lower()
        hint = ""
        if "cloud_publish_state" in lowered and any(
            token in lowered for token in ("pgrst205", "pgrst204", "does not exist", "schema cache", "404")
        ):
            hint = f" Apply {STATE_MIGRATION} by hand."
        return PublishError(f"{action}: {text}.{hint}")

    def load(self) -> Dict[str, Tuple[str, int]]:
        held: Dict[str, Tuple[str, int]] = {}
        offset = 0
        while True:
            try:
                response = (
                    self.client.table("cloud_publish_state")
                    .select("doc_key,content_hash,byte_size")
                    .order("doc_key").limit(self._PAGE).offset(offset).execute()
                )
            except Exception as exc:  # noqa: BLE001
                raise self._fail("Could not read cloud_publish_state", exc) from exc
            if response["error"]:
                raise self._fail("Could not read cloud_publish_state", response["error"])
            page = response["data"] or []
            for row in page:
                held[row["doc_key"]] = (row["content_hash"], int(row.get("byte_size") or 0))
            if len(page) < self._PAGE:
                return held
            offset += len(page)

    def record(self, key: str, sha256: str, size: int) -> None:
        client = self.client
        row = {
            "doc_key": key, "content_hash": sha256, "byte_size": size,
            "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        try:
            response = client.client.post(
                f"{client.base_url}/cloud_publish_state",
                params={"on_conflict": "doc_key"},
                json=[row],
                headers={**client.headers, "Prefer": "resolution=merge-duplicates,return=minimal"},
            )
        except Exception as exc:  # noqa: BLE001
            raise self._fail(f"Could not record {key}", exc) from exc
        if response.status_code >= 400:
            raise self._fail(f"Could not record {key}", response.text[:300])

    def forget(self, keys: List[str]) -> None:
        client = self.client
        for start in range(0, len(keys), 50):
            batch = keys[start:start + 50]
            quoted = ",".join('"' + key.replace('"', '\\"') + '"' for key in batch)
            try:
                response = client.client.delete(
                    f"{client.base_url}/cloud_publish_state", params={"doc_key": f"in.({quoted})"},
                )
            except Exception as exc:  # noqa: BLE001
                raise self._fail("Could not forget removed documents", exc) from exc
            if response.status_code >= 400:
                raise self._fail("Could not forget removed documents", response.text[:300])


# ---------------------------------------------------------------------------
# A publish
# ---------------------------------------------------------------------------


@dataclass
class PublishReport:
    dry_run: bool = False
    uploaded: List[str] = field(default_factory=list)
    uploaded_bytes: int = 0
    unchanged: int = 0
    repaired: int = 0
    removed: int = 0
    manifest_sent: bool = False
    total_objects: int = 0
    total_bytes: int = 0
    gc_deleted: int = 0
    export: Optional[ExportReport] = None
    seconds: float = 0.0

    def render(self) -> str:
        mb = lambda n: f"{n / 1_000_000:.1f} MB"  # noqa: E731
        verb = "would upload" if self.dry_run else "uploaded"
        lines = [
            f"cloud publish {'(dry run) ' if self.dry_run else ''}-> {self.total_objects} objects, {mb(self.total_bytes)}",
            f"  {verb} {len(self.uploaded)} ({mb(self.uploaded_bytes)}), unchanged {self.unchanged}, "
            f"state repaired {self.repaired}, dropped from state {self.removed}",
            "  manifest: "
            + ("not needed" if not self.manifest_sent and not self.dry_run else ("would be written last" if self.dry_run else "written last"))
            + (f"; the cloud removed {self.gc_deleted} old object(s)" if self.gc_deleted else ""),
        ]
        lines.extend(f"    {key}" for key in self.uploaded[:12])
        if len(self.uploaded) > 12:
            lines.append(f"    ... and {len(self.uploaded) - 12} more")
        if self.export is not None:
            lines.extend(f"  warning: {w}" for w in self.export.warnings)
        return "\n".join(lines)

    def summary(self) -> Dict[str, Any]:
        return {
            "uploaded": len(self.uploaded), "uploaded_bytes": self.uploaded_bytes,
            "unchanged": self.unchanged, "objects": self.total_objects, "bytes": self.total_bytes,
            "manifest_sent": self.manifest_sent, "seconds": round(self.seconds, 1),
        }


def _items(manifest: Mapping[str, Any]) -> List[Tuple[str, str, int]]:
    """(key, sha256, bytes) for audio first, then documents: the upload order."""
    out = [(e["key"], e["sha256"], int(e["bytes"])) for e in manifest.get("audio", [])]
    out += [(e["key"], e["sha256"], int(e["bytes"])) for e in manifest.get("docs", [])]
    return out


def _read_verified(bundle: Path, key: str, sha256: str, size: int) -> bytes:
    path = bundle / key
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PublishError(f"{key} is in the manifest but not readable in the bundle ({exc}).") from exc
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
        raise PublishError(f"{key} changed on disk after the manifest was written; run the publish again.")
    return data


def publish_bundle(
    bundle: Path,
    ingest: Optional[IngestClient],
    state: Any,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    full: bool = False,
    dry_run: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
) -> PublishReport:
    """Send an already-exported bundle. ``ingest`` may be None only for a dry run."""
    started = time.monotonic()
    report = PublishReport(dry_run=dry_run)
    note = on_progress or (lambda _msg: None)

    manifest = read_manifest(Path(bundle))
    if manifest is None:
        raise PublishError(f"{bundle} has no manifest.json; export first.")
    items = _items(manifest)
    report.total_objects = len(items)
    report.total_bytes = sum(size for _, _, size in items)
    if report.total_bytes > max_bytes:
        raise PublishError(
            f"The bundle is {report.total_bytes / 1e9:.2f} GB, over the {max_bytes / 1e9:.2f} GB cap "
            "(CLOUD_PUBLISH_MAX_BYTES). Nothing was uploaded."
        )

    held = state.load()
    remote: Dict[str, Optional[str]] = {}
    remote_bytes = 0
    if ingest is not None:
        inv = ingest.inventory()
        remote = {o["key"]: o.get("sha256") for o in inv.get("objects", [])}
        remote_bytes = int(inv.get("bytes") or 0)

    pending: List[Tuple[str, str, int]] = []
    for key, sha, size in items:
        in_state = held.get(key, (None, 0))[0] == sha
        in_cloud = ingest is not None and remote.get(key) == sha
        if not full and in_cloud:
            report.unchanged += 1
            if not in_state:
                report.repaired += 1
                if not dry_run:
                    state.record(key, sha, size)
        elif not full and ingest is None and in_state:
            report.unchanged += 1
        else:
            pending.append((key, sha, size))

    growth = sum(size for _, _, size in pending)
    if remote_bytes + growth > max_bytes:
        raise PublishError(
            f"The cloud already holds {remote_bytes / 1e9:.2f} GB and this publish adds {growth / 1e9:.2f} GB, "
            f"over the {max_bytes / 1e9:.2f} GB cap (CLOUD_PUBLISH_MAX_BYTES). Nothing was uploaded."
        )

    if dry_run:
        report.uploaded = [key for key, _, _ in pending]
        report.uploaded_bytes = growth
        report.seconds = time.monotonic() - started
        return report

    for index, (key, sha, size) in enumerate(pending, 1):
        data = _read_verified(Path(bundle), key, sha, size)
        note(f"uploading {key} ({index}/{len(pending)})")
        ingest.put_object(key, data, sha)
        state.record(key, sha, size)
        report.uploaded.append(key)
        report.uploaded_bytes += size

    manifest_bytes = (Path(bundle) / MANIFEST_KEY).read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_current = held.get(MANIFEST_KEY, (None, 0))[0] == manifest_sha and remote.get(MANIFEST_KEY) == manifest_sha
    if pending or full or not manifest_current:
        note("writing the manifest")
        answer = ingest.put_manifest(manifest_bytes, manifest_sha)
        state.record(MANIFEST_KEY, manifest_sha, len(manifest_bytes))
        report.manifest_sent = True
        report.gc_deleted = int(answer.get("deleted") or 0)

    live = {key for key, _, _ in items} | {MANIFEST_KEY}
    stale = sorted(key for key in held if key not in live)
    if stale:
        state.forget(stale)
        report.removed = len(stale)
    report.seconds = time.monotonic() - started
    return report


def run_publish(
    config: Optional[PublishConfig],
    bundle_dir: Path,
    *,
    ingest: Optional[IngestClient] = None,
    state: Optional[Any] = None,
    full: bool = False,
    dry_run: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
    export: Optional[Callable[..., ExportReport]] = None,
) -> PublishReport:
    """Export, then publish. Local export errors surface as ``PublishError``."""
    state = state if state is not None else PostgrestState()
    note = on_progress or (lambda _msg: None)
    note("exporting the bundle")
    try:
        export_report = (export or run_export)(Path(bundle_dir))
    except ExportError as exc:
        raise PublishError(f"Export failed: {exc}") from exc
    owned = False
    if ingest is None and config is not None:
        # A dry run still asks what the cloud holds (a read), so its plan is the real one.
        ingest = IngestClient(config)
        owned = True
    if ingest is None and not dry_run:
        raise PublishError("No cloud reader is configured.")
    try:
        report = publish_bundle(
            Path(bundle_dir), ingest, state,
            max_bytes=config.max_bytes if config else DEFAULT_MAX_BYTES,
            full=full, dry_run=dry_run, on_progress=note,
        )
    finally:
        if owned and ingest is not None:
            ingest.close()
    report.export = export_report
    return report
