"""Download the datasets a notebook reads, so the sandbox can run it offline.

At import, every quoted ``http(s)`` URL in a code cell (``pd.read_csv("…")``
and the like, commented-out lines included, because uncommenting one is an
obvious thing to try) is downloaded into the playground's data directory:

    <PLAYGROUND_DATA_DIR>/<playground_id>/manifest.json
    <PLAYGROUND_DATA_DIR>/<playground_id>/files/<sha256(url)[:12]>-<name>

``manifest.json`` is a flat ``{"<url>": "<path relative to the playground
directory>"}`` object listing only the files that downloaded completely. The
sandbox (docs/PLAYGROUND_RUNNER.md) mounts the same volume read-only at
``/data`` and redirects those URLs to ``/data/<playground_id>/<path>``. The
file name and shape are a contract with it; do not change either alone.

A download that fails is recorded with its reason and left out of the
manifest. It never fails the import: the notebook and its saved outputs are
still worth studying without the data.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlsplit

import httpx

from .notebook import source_text

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
FILES_DIR = "files"

#: More than this many distinct URLs in one notebook is not a dataset list.
MAX_URLS = 50
MAX_REDIRECTS = 5
_TIMEOUT = httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=15.0)
_CHUNK = 1 << 20

_URL_IN_QUOTES = re.compile(r"""(?P<q>['"])(?P<url>https?://[^'"\s]+)(?P=q)""")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

Resolver = Callable[[str, int], List[str]]


class DatasetError(Exception):
    """One dataset could not be cached. The message is shown to the user."""


def find_dataset_urls(nb: Dict[str, Any]) -> List[str]:
    """Quoted http(s) URLs in code cells, first appearance order, no repeats."""
    seen: List[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for match in _URL_IN_QUOTES.finditer(source_text(cell)):
            url = match.group("url")
            if url not in seen:
                seen.append(url)
    return seen[:MAX_URLS]


def playground_dir(root: Path, playground_id: str) -> Path:
    """The playground's directory. The id must be a UUID, never a path."""
    try:
        canonical = str(uuid.UUID(str(playground_id)))
    except ValueError:
        raise DatasetError(f"{playground_id!r} is not a playground id") from None
    return Path(root) / canonical


def file_name_for(url: str) -> str:
    """``<12 hex of sha256(url)>-<last path segment>``: stable and collision-free."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    tail = urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]
    name = _SAFE_NAME.sub("_", tail).strip("._")[:80] or "data"
    return f"{digest}-{name}"


def _system_resolver(host: str, port: int) -> List[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


def _human_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f} GB"
    return f"{n / 1_000_000:.1f} MB"


@dataclass
class DatasetRecord:
    url: str
    path: Optional[str] = None
    bytes: int = 0
    sha256: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"url": self.url}
        if self.error:
            out["error"] = self.error
        else:
            out.update(path=self.path, bytes=self.bytes, sha256=self.sha256)
        return out


class DatasetCache:
    """Cache one playground's datasets under ``root``.

    ``client`` and ``resolver`` are injectable so the tests never dial out.
    """

    def __init__(
        self,
        root: Path | str,
        max_bytes: int,
        *,
        client: Optional[httpx.Client] = None,
        resolver: Optional[Resolver] = None,
    ) -> None:
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self._client = client
        self._resolver = resolver or _system_resolver

    # -- public ------------------------------------------------------------

    def cache(
        self,
        playground_id: str,
        urls: Iterable[str],
        previous: Iterable[Dict[str, Any]] = (),
    ) -> List[Dict[str, Any]]:
        """Download ``urls`` and write the manifest. Returns one record per URL.

        A URL already cached for this playground (same file, same size) is
        kept rather than downloaded again, so re-importing an edited notebook
        costs nothing for data that did not change. Files no longer named by
        the notebook are removed; this directory belongs to this playground.
        """
        base = playground_dir(self.root, playground_id)
        files = base / FILES_DIR
        files.mkdir(parents=True, exist_ok=True)
        known = {
            r.get("url"): r for r in previous
            if isinstance(r, dict) and r.get("path") and not r.get("error")
        }

        records: List[DatasetRecord] = []
        used = 0
        client = self._client or httpx.Client(timeout=_TIMEOUT)
        try:
            for url in urls:
                record = self._reuse(base, known.get(url), self.max_bytes - used)
                if record is None:
                    record = self._fetch(client, base, url, self.max_bytes - used)
                used += record.bytes
                records.append(record)
        finally:
            if self._client is None:
                client.close()

        self._remove_unreferenced(base, records)
        self._write_manifest(base, records)
        return [r.as_dict() for r in records]

    # -- internals ---------------------------------------------------------

    def _reuse(
        self, base: Path, prior: Optional[Dict[str, Any]], room: int,
    ) -> Optional[DatasetRecord]:
        if not prior:
            return None
        path = base / str(prior["path"])
        try:
            size = path.stat().st_size
        except OSError:
            return None
        if size != prior.get("bytes") or size > room or not prior.get("sha256"):
            return None
        return DatasetRecord(
            url=prior["url"], path=prior["path"], bytes=size, sha256=prior["sha256"],
        )

    def _fetch(
        self, client: httpx.Client, base: Path, url: str, room: int,
    ) -> DatasetRecord:
        rel = f"{FILES_DIR}/{file_name_for(url)}"
        dest = base / rel
        part = dest.with_name(dest.name + ".part")
        try:
            size, digest = self._download(client, url, part, room)
            os.replace(part, dest)
        except DatasetError as exc:
            part.unlink(missing_ok=True)
            logger.warning("Playground dataset %s not cached: %s", url, exc)
            return DatasetRecord(url=url, error=str(exc))
        except httpx.HTTPError as exc:
            part.unlink(missing_ok=True)
            reason = f"{type(exc).__name__}: {exc}".splitlines()[0][:300]
            logger.warning("Playground dataset %s not cached: %s", url, reason)
            return DatasetRecord(url=url, error=reason)
        except OSError as exc:
            part.unlink(missing_ok=True)
            return DatasetRecord(url=url, error=f"Could not write the file: {exc}")
        return DatasetRecord(url=url, path=rel, bytes=size, sha256=digest)

    def _check_target(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise DatasetError("Only http and https URLs are downloaded.")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        try:
            addresses = self._resolver(parts.hostname, port)
        except OSError as exc:
            raise DatasetError(f"Could not resolve {parts.hostname}: {exc}") from None
        if not addresses:
            raise DatasetError(f"Could not resolve {parts.hostname}.")
        for raw in addresses:
            try:
                address = ipaddress.ip_address(raw.split("%", 1)[0])
            except ValueError:
                raise DatasetError(f"{parts.hostname} resolved to {raw!r}.") from None
            if not address.is_global:
                # The import runs inside the app's network. A notebook must not
                # be able to read the database or anything else on it.
                raise DatasetError(
                    f"Refused: {parts.hostname} is a private or local address."
                )

    def _download(
        self, client: httpx.Client, url: str, part: Path, room: int,
    ) -> tuple[int, str]:
        cap_text = (
            f"the {_human_bytes(self.max_bytes)} playground limit "
            "(PLAYGROUND_DATA_MAX_BYTES)"
        )
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            self._check_target(current)
            with client.stream("GET", current, follow_redirects=False) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise DatasetError(f"HTTP {response.status_code} with no Location.")
                    current = urljoin(current, location)
                    continue
                if response.status_code >= 400:
                    raise DatasetError(f"HTTP {response.status_code} from {urlsplit(current).hostname}.")
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > room:
                    raise DatasetError(
                        f"{_human_bytes(int(declared))} would go over {cap_text}."
                    )
                digest = hashlib.sha256()
                size = 0
                with open(part, "wb") as handle:
                    for chunk in response.iter_bytes(_CHUNK):
                        size += len(chunk)
                        if size > room:
                            raise DatasetError(f"The download went over {cap_text}.")
                        digest.update(chunk)
                        handle.write(chunk)
                return size, digest.hexdigest()
        raise DatasetError(f"More than {MAX_REDIRECTS} redirects.")

    def _remove_unreferenced(self, base: Path, records: List[DatasetRecord]) -> None:
        keep = {base / r.path for r in records if r.path}
        for path in (base / FILES_DIR).iterdir():
            if path.is_file() and path not in keep:
                path.unlink(missing_ok=True)

    def _write_manifest(self, base: Path, records: List[DatasetRecord]) -> None:
        manifest = {r.url: r.path for r in records if r.path and not r.error}
        tmp = base / (MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, base / MANIFEST_NAME)


def read_manifest(root: Path | str, playground_id: str) -> Dict[str, str]:
    """The manifest as written, or ``{}`` when there is none yet."""
    path = playground_dir(Path(root), playground_id) / MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
