"""Kernel startup hook: serve a playground's cached datasets for their URLs.

Runs inside each kernel. The runner has no network, so a cell such as

    df = pd.read_csv("https://raw.githubusercontent.com/.../transactions.csv")

works only because this hook maps that URL to the file PR 1 cached under
``/data/<playground_id>/``. The map is ``/data/<playground_id>/manifest.json``,
a flat ``{"<url>": "<path relative to that directory>"}`` object.

Patched: pandas ``read_csv``, ``read_json``, ``read_excel``, ``read_parquet``,
``read_table``, and ``urllib.request.urlopen`` (which also covers
``urllib.request.urlretrieve``, since it calls ``urlopen`` by module lookup).
A URL that is not in the manifest raises :class:`OfflineDatasetError` naming
the URL. Local paths and file objects pass through untouched.
"""

from __future__ import annotations

import email.message
import functools
import json
import os
import urllib.error
import urllib.request
import urllib.response
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urldefrag

PANDAS_READERS = ("read_csv", "read_json", "read_excel", "read_parquet", "read_table")
_URL_SCHEMES = ("http://", "https://", "ftp://", "s3://", "gs://", "gcs://", "az://", "abfs://")
_MARK = "__playground_offline__"


class OfflineDatasetError(urllib.error.URLError):
    """A dataset URL that this playground has no cached copy of.

    Subclasses URLError so code that already handles download failures still
    catches it. ``str()`` is the full explanation.
    """

    def __init__(self, url: str, playground_id: Optional[str]):
        self.url = url
        where = f"playground '{playground_id}'" if playground_id else "this session"
        message = (
            f"This playground runs offline, and {url} is not in the dataset cache for {where}. "
            "Only URLs listed in the playground's manifest.json can be read. "
            "Re-import the playground so the dataset is cached, or read a file that is already cached."
        )
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:  # URLError's default is "<urlopen error ...>"
        return self.message


def _looks_like_url(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower().startswith(_URL_SCHEMES)


def _normalise(url: str) -> str:
    return urldefrag(url.strip())[0]


class DatasetMap:
    def __init__(self, root: Path, playground_id: Optional[str], mapping: Dict[str, Path]):
        self.root = root
        self.playground_id = playground_id
        self.mapping = mapping

    @classmethod
    def load(cls, data_dir: Path, playground_id: Optional[str]) -> "DatasetMap":
        if not playground_id:
            return cls(data_dir, None, {})
        root = (data_dir / playground_id).resolve()
        manifest = root / "manifest.json"
        mapping: Dict[str, Path] = {}
        if manifest.is_file():
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"{manifest} must be a JSON object of url -> relative path")
            for url, rel in raw.items():
                if not isinstance(url, str) or not isinstance(rel, str):
                    continue
                target = (root / rel).resolve()
                # A manifest entry must not point outside its playground directory.
                if target != root and root not in target.parents:
                    continue
                mapping[_normalise(url)] = target
        return cls(root, playground_id, mapping)

    def resolve(self, url: str) -> Path:
        path = self.mapping.get(_normalise(url))
        if path is None:
            raise OfflineDatasetError(url, self.playground_id)
        if not path.is_file():
            raise OfflineDatasetError(url, self.playground_id) from FileNotFoundError(str(path))
        return path


def _wrap_reader(original: Callable, datasets: DatasetMap) -> Callable:
    @functools.wraps(original)
    def reader(*args, **kwargs):
        if args and _looks_like_url(args[0]):
            args = (str(datasets.resolve(args[0])),) + tuple(args[1:])
            kwargs.pop("storage_options", None)
        else:
            for key in ("filepath_or_buffer", "path", "io", "path_or_buf"):
                if _looks_like_url(kwargs.get(key)):
                    kwargs[key] = str(datasets.resolve(kwargs[key]))
                    kwargs.pop("storage_options", None)
                    break
        return original(*args, **kwargs)

    setattr(reader, _MARK, original)
    return reader


def _wrap_urlopen(original: Callable, datasets: DatasetMap) -> Callable:
    @functools.wraps(original)
    def urlopen(url, *args, **kwargs):
        full = url.full_url if isinstance(url, urllib.request.Request) else url
        if isinstance(full, str) and full.lower().startswith("file:"):
            return original(url, *args, **kwargs)
        path = datasets.resolve(str(full))
        headers = email.message.Message()
        headers["Content-Length"] = str(path.stat().st_size)
        headers["Content-Type"] = "application/octet-stream"
        return urllib.response.addinfourl(open(path, "rb"), headers, str(full), code=200)

    setattr(urlopen, _MARK, original)
    return urlopen


def install(data_dir: str = "/data", playground_id: Optional[str] = None) -> DatasetMap:
    """Patch pandas and urllib in this process. Safe to call more than once."""
    playground_id = playground_id or os.environ.get("PLAYGROUND_ID") or None
    datasets = DatasetMap.load(Path(data_dir), playground_id)

    current = urllib.request.urlopen
    urllib.request.urlopen = _wrap_urlopen(getattr(current, _MARK, current), datasets)

    try:
        import pandas
    except ImportError:  # pragma: no cover - pandas is in the image
        return datasets
    for name in PANDAS_READERS:
        current = getattr(pandas, name, None)
        if current is None:
            continue
        setattr(pandas, name, _wrap_reader(getattr(current, _MARK, current), datasets))
    return datasets
