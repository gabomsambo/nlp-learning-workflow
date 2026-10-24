"""Read-only health for the embedding and vector projection pipeline."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import httpx
from qdrant_client import QdrantClient

EMBEDDER_FAILED = "embedder_failed"
INDEX_UNREACHABLE = "index_unreachable"
CLOUD_SUSPENDED_OR_DELETED = "cloud_suspended_or_deleted"
COLLECTION_OR_ALIAS_MISSING = "collection_or_alias_missing"
PROJECTION_LAG = "projection_lag"
PROJECTION_STATE_UNAVAILABLE = "projection_state_unavailable"

SUSPENDED_BODY = "404 page not found"
HEALTH_CACHE_SECONDS = 60.0
PROBE_TEXT = "vector health check"

_cache_lock = threading.Lock()
_cache: Dict[str, Any] = {"at": 0.0, "value": None}


class ProjectionOutcome(int):
    """An integer chunk count carrying non-fatal projection notices."""

    notices: tuple[str, ...]

    def __new__(cls, chunks: int, notices: Sequence[str] = ()):
        value = int.__new__(cls, chunks)
        value.notices = tuple(str(item) for item in notices if item)
        return value


def failure_detail(kind: str, target: str, detail: str = "") -> str:
    """Stable, plain-language detail for health rows and run stages."""
    suffix = f": {detail.strip()}" if detail and detail.strip() else ""
    if kind == EMBEDDER_FAILED:
        return f"embedder failed{suffix}"
    if kind == INDEX_UNREACHABLE:
        return f"{target} vector index unreachable{suffix}"
    if kind == CLOUD_SUSPENDED_OR_DELETED:
        return f"cloud vector index suspended or deleted (plain-text 404){suffix}"
    if kind == COLLECTION_OR_ALIAS_MISSING:
        return f"{target} vector collection or alias missing{suffix}"
    if kind == PROJECTION_LAG:
        return f"{target} vector projection is behind{suffix}"
    return f"{target} vector projection state unavailable{suffix}"


def clear_health_cache() -> None:
    with _cache_lock:
        _cache.update(at=0.0, value=None)


def _default_http_get(url: str, *, headers: Dict[str, str], timeout: float):
    return httpx.get(url, headers=headers, timeout=timeout)


def _default_client(url: str, api_key: str) -> QdrantClient:
    kwargs: Dict[str, Any] = {"url": url, "timeout": 3}
    if api_key:
        kwargs["api_key"] = api_key
    return QdrantClient(**kwargs)


def _target_row(
    target: str, *, state: str, fatal: bool, **extra: Any
) -> Dict[str, Any]:
    row = {"target": target, "state": state, "fatal": fatal}
    row.update(extra)
    return row


def _root_failure(target: str, response: Any) -> Optional[tuple[str, str]]:
    text = str(getattr(response, "text", "") or "")
    status = getattr(response, "status_code", None)
    if status == 404 and SUSPENDED_BODY in text.lower() and target == "cloud":
        return (
            CLOUD_SUSPENDED_OR_DELETED,
            "the Qdrant root returned plain text instead of a JSON version banner",
        )
    if status != 200:
        return INDEX_UNREACHABLE, f"the Qdrant root answered HTTP {status}"
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return INDEX_UNREACHABLE, "the Qdrant root did not return JSON"
    if not isinstance(payload, dict) or not payload.get("version"):
        return INDEX_UNREACHABLE, "the Qdrant root did not return a version banner"
    return None


def _inspect_target(
    target: str,
    url: str,
    api_key: str,
    *,
    expected_collection: str,
    embedded_ids: set[str],
    state_loader: Callable[[str, str], Iterable[str]],
    model_id: str,
    http_get: Callable[..., Any],
    client_factory: Callable[[str, str], QdrantClient],
) -> Dict[str, Any]:
    fatal = target == "local"
    headers = {"api-key": api_key} if api_key else {}
    try:
        response = http_get(url.rstrip("/") + "/", headers=headers, timeout=2.5)
    except Exception as exc:
        kind = INDEX_UNREACHABLE
        return _target_row(
            target,
            state="down",
            fatal=fatal,
            failure=kind,
            detail=failure_detail(kind, target, str(exc)),
        )

    failed = _root_failure(target, response)
    if failed:
        kind, reason = failed
        return _target_row(
            target,
            state="down",
            fatal=fatal,
            failure=kind,
            detail=failure_detail(kind, target, reason),
        )

    try:
        client = client_factory(url, api_key)
        collections = getattr(client.get_collections(), "collections", None) or []
        names = {str(item.name) for item in collections}
        aliases = getattr(client.get_aliases(), "aliases", None) or []
    except Exception as exc:
        kind = INDEX_UNREACHABLE
        return _target_row(
            target,
            state="down",
            fatal=fatal,
            failure=kind,
            detail=failure_detail(kind, target, str(exc)),
        )

    if expected_collection not in names:
        kind = COLLECTION_OR_ALIAS_MISSING
        return _target_row(
            target,
            state="down",
            fatal=fatal,
            failure=kind,
            detail=failure_detail(
                kind, target, f"collection {expected_collection} is absent"
            ),
        )
    alias_target = next(
        (
            getattr(item, "collection_name", None)
            for item in aliases
            if getattr(item, "alias_name", None) == "nlp_chunks"
        ),
        None,
    )
    if alias_target != expected_collection:
        kind = COLLECTION_OR_ALIAS_MISSING
        reason = (
            "alias nlp_chunks is absent"
            if alias_target is None
            else f"alias nlp_chunks points at {alias_target}, not {expected_collection}"
        )
        return _target_row(
            target,
            state="down",
            fatal=fatal,
            failure=kind,
            detail=failure_detail(kind, target, reason),
        )

    try:
        projected_ids = {str(item) for item in state_loader(target, model_id)}
    except Exception as exc:
        kind = PROJECTION_STATE_UNAVAILABLE
        return _target_row(
            target,
            state="down",
            fatal=fatal,
            failure=kind,
            detail=failure_detail(kind, target, str(exc)),
        )
    pending = embedded_ids - projected_ids
    counts = {
        "embedded_chunks": len(embedded_ids),
        "projected_chunks": len(embedded_ids & projected_ids),
        "pending_chunks": len(pending),
    }
    if pending:
        kind = PROJECTION_LAG
        return _target_row(
            target,
            state="behind",
            fatal=fatal,
            failure=kind,
            detail=failure_detail(kind, target, f"{len(pending)} chunk(s) pending"),
            **counts,
        )
    return _target_row(
        target,
        state="ok",
        fatal=fatal,
        detail="ready and fully projected",
        **counts,
    )


def collect_vector_health(
    *,
    provider_factory: Optional[Callable[[], Any]] = None,
    embedded_ids_loader: Optional[Callable[[str], Iterable[str]]] = None,
    state_loader: Optional[Callable[[str, str], Iterable[str]]] = None,
    http_get: Optional[Callable[..., Any]] = None,
    client_factory: Optional[Callable[[str, str], QdrantClient]] = None,
    environ: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Collect vector health with injectable I/O for hermetic tests."""
    from . import db
    from .embeddings import get_embedding_provider
    from .projector import collection_name

    env = os.environ if environ is None else environ
    provider_factory = provider_factory or get_embedding_provider
    embedded_ids_loader = embedded_ids_loader or db.list_embedding_chunk_ids
    state_loader = state_loader or db.list_projection_state
    http_get = http_get or _default_http_get
    client_factory = client_factory or _default_client

    try:
        provider = provider_factory()
        model_id = str(provider.model_id)
        dims = int(provider.dims)
    except Exception as exc:
        kind = EMBEDDER_FAILED
        embedder = {
            "state": "down",
            "failure": kind,
            "detail": failure_detail(kind, "embedder", str(exc)),
        }
        return {"ok": False, "embedder": embedder, "targets": []}

    # Health must never become a periodic paid API call. The approved local
    # provider is actively loaded and encoded; legacy OpenAI configuration is
    # reported as configured and proves itself on the next real request.
    paid_provider = model_id.startswith("text-embedding-")
    try:
        if not paid_provider:
            vector = provider.embed_query(PROBE_TEXT)
            if len(vector) != dims:
                raise ValueError(
                    f"returned {len(vector)} dims, expected {dims}"
                )
    except Exception as exc:
        kind = EMBEDDER_FAILED
        embedder = {
            "state": "down",
            "failure": kind,
            "detail": failure_detail(kind, "embedder", str(exc)),
        }
    else:
        embedder = {
            "state": "ok",
            "model_id": model_id,
            "dims": dims,
            "detail": (
                "configured; paid encode probe skipped"
                if paid_provider
                else "loaded and encoded a probe"
            ),
        }
    try:
        embedded_ids = {str(item) for item in embedded_ids_loader(model_id)}
    except Exception as exc:
        kind = PROJECTION_STATE_UNAVAILABLE
        return {
            "ok": False,
            "embedder": embedder,
            "targets": [],
            "failure": kind,
            "detail": failure_detail(kind, "Postgres", str(exc)),
        }

    expected = collection_name(model_id, dims)
    configured = [
        (
            "local",
            env.get("QDRANT_URL", "").strip(),
            env.get("QDRANT_API_KEY", "").strip(),
        ),
        (
            "cloud",
            env.get("QDRANT_CLOUD_URL", "").strip(),
            env.get("QDRANT_CLOUD_API_KEY", "").strip(),
        ),
    ]
    targets = []
    for target, url, api_key in configured:
        if not url:
            if target == "cloud":
                continue
            kind = INDEX_UNREACHABLE
            targets.append(
                _target_row(
                    target,
                    state="down",
                    fatal=True,
                    failure=kind,
                    detail=failure_detail(
                        kind, target, "QDRANT_URL is not configured"
                    ),
                )
            )
            continue
        targets.append(
            _inspect_target(
                target,
                url,
                api_key,
                expected_collection=expected,
                embedded_ids=embedded_ids,
                state_loader=state_loader,
                model_id=model_id,
                http_get=http_get,
                client_factory=client_factory,
            )
        )
    ok = embedder["state"] == "ok" and all(
        row["state"] == "ok" for row in targets if row["fatal"]
    )
    return {"ok": ok, "embedder": embedder, "targets": targets}


def cached_vector_health(*, force: bool = False) -> Dict[str, Any]:
    """Cache the production report across frequent Docker health probes."""
    now = time.monotonic()
    with _cache_lock:
        value = _cache["value"]
        if (
            not force
            and value is not None
            and now - _cache["at"] < HEALTH_CACHE_SECONDS
        ):
            return value
        value = collect_vector_health()
        _cache.update(at=now, value=value)
        return value
