"""Project chunk embeddings from Postgres into Qdrant.

Postgres is the source of truth (``paper_chunks`` + ``chunk_embeddings``).
Each Qdrant — the local server and, when configured, Qdrant Cloud — is a
copy that this module rebuilds. The two copies never sync with each other.

Point id is ``uuid5(POINT_NAMESPACE, chunk_id)``, stable across targets and
independent of pillar. The payload carries the chunk text plus
``pillar_id``, ``paper_id``, ``chunk_id`` and ``chunk_index``. Each model
has its own collection (``chunks__<model>__<dims>``) behind the alias
``nlp_chunks``, which is what search reads.

A dead or suspended cloud is not a failure of the local index. That target
gets no ``vector_index_state`` rows, so the next sync retries the same
chunks. Local failure raises ``ProjectionError``; callers put that on the
VECTORS stage and still keep the paper.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AbstractSet, Callable, Iterable, List, Optional, Sequence

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from .vector_health import (
    CLOUD_SUSPENDED_OR_DELETED,
    COLLECTION_OR_ALIAS_MISSING,
    INDEX_UNREACHABLE,
    PROJECTION_STATE_UNAVAILABLE,
    ProjectionOutcome,
    failure_detail,
)

logger = logging.getLogger(__name__)

MIGRATION = "docs/migrations/019_vector_index_state.sql"

# Frozen. uuid5(NAMESPACE_URL, this name). Do not edit the name: every point
# id already written was derived from it.
_POINT_NAMESPACE_NAME = "https://nlp-learning-workflow.local/chunk-point"
POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, _POINT_NAMESPACE_NAME)

READ_ALIAS = "nlp_chunks"
PILLAR_ID_FIELD = "pillar_id"
PAPER_ID_FIELD = "paper_id"
PAYLOAD_INDEX_FIELDS = (PILLAR_ID_FIELD, PAPER_ID_FIELD)

TARGETS = ("local", "cloud")
UPSERT_BATCH = 64
SCROLL_PAGE = 128
_STATE_BATCH = 200

_UNHEALTHY_404 = "404 page not found"


class ProjectionError(Exception):
    """The local index was not updated, or the projection tables are missing.

    The message is the VECTORS stage detail. A cloud outage is not this
    error: that target is left behind and retried.
    """


@dataclass
class ChunkVector:
    """One stored chunk and the embedding that will be the Qdrant vector."""

    chunk_id: str
    paper_id: str
    pillar_id: str
    chunk_index: int
    text: str
    embedding: List[float]
    dims: int

    @classmethod
    def from_row(cls, row: dict) -> "ChunkVector":
        chunk_id = str(row.get("chunk_id") or "")
        embedding = _coerce_vector(row.get("embedding"), chunk_id)
        try:
            dims = int(row["dims"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectionError(f"chunk {chunk_id} has no dims") from exc
        if len(embedding) != dims:
            raise ProjectionError(
                f"chunk {chunk_id} embedding length {len(embedding)} != dims {dims}"
            )
        pillar_id = row.get("pillar_id")
        if not pillar_id or not str(pillar_id).strip():
            raise ProjectionError(f"chunk {chunk_id} has no pillar_id")
        text = row.get("chunk_text")
        if text is None:
            text = row.get("text") or ""
        try:
            chunk_index = int(row["chunk_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectionError(f"chunk {chunk_id} has no chunk_index") from exc
        paper_id = row.get("paper_id")
        if not paper_id:
            raise ProjectionError(f"chunk {chunk_id} has no paper_id")
        return cls(
            chunk_id=chunk_id,
            paper_id=str(paper_id),
            pillar_id=str(pillar_id),
            chunk_index=chunk_index,
            text=str(text),
            embedding=embedding,
            dims=dims,
        )


@dataclass
class SyncResult:
    """What one target did, or would do under ``--dry-run``."""

    target: str
    model_id: str
    dims: int
    collection: str
    pending: int = 0
    stale: int = 0
    upserted_ids: List[str] = field(default_factory=list)
    deleted_ids: List[str] = field(default_factory=list)
    dry_run: bool = False
    unhealthy: bool = False
    failure: Optional[str] = None
    error: Optional[str] = None

    def render(self) -> str:
        lines = [
            f"target: {self.target}",
            f"model_id: {self.model_id}",
            f"dims: {self.dims}",
            f"collection: {self.collection}",
            f"alias: {READ_ALIAS}",
            f"dry_run: {'true' if self.dry_run else 'false'}",
            f"pending: {self.pending}",
            f"stale: {self.stale}",
        ]
        if self.dry_run:
            lines.append(f"would_upsert: {self.pending}")
            lines.append(f"would_delete: {self.stale}")
        else:
            lines.append(f"upserted: {len(self.upserted_ids)}")
            lines.append(f"deleted: {len(self.deleted_ids)}")
        if self.unhealthy:
            lines.append("unhealthy: true")
        if self.failure:
            lines.append(f"failure: {self.failure}")
        if self.error:
            lines.append(f"error: {self.error}")
        return "\n".join(lines)


def point_id(chunk_id: str) -> str:
    """Stable Qdrant point id for a chunk. Same value on every target."""
    return str(uuid.uuid5(POINT_NAMESPACE, str(chunk_id)))


def collection_name(model_id: str, dims: int) -> str:
    """Per-model collection. The read path uses ``READ_ALIAS``, not this name."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", model_id).strip("_")
    if not safe:
        raise ProjectionError(f"model id {model_id!r} is not a collection name")
    return f"chunks__{safe}__{int(dims)}"


def probe(client: QdrantClient, target: str) -> Optional[tuple[str, str]]:
    """Return a classified failure when the server does not answer.

    A suspended Qdrant Cloud cluster returns plain text ``404 page not found``
    on every path, including collection listing. The same ingress shape is
    observed after deletion, so the app deliberately does not claim which
    lifecycle state applies.
    """
    try:
        client.get_collections()
    except UnexpectedResponse as exc:
        body = exc.content
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8", "replace")
        text = body or ""
        if (
            target == "cloud"
            and exc.status_code == 404
            and _UNHEALTHY_404 in text
        ):
            kind = CLOUD_SUSPENDED_OR_DELETED
            return kind, failure_detail(
                kind,
                target,
                "this target stays behind until a cloud cluster answers",
            )
        kind = INDEX_UNREACHABLE
        return kind, failure_detail(
            kind, target, f"Qdrant answered HTTP {exc.status_code}"
        )
    except Exception as exc:
        kind = INDEX_UNREACHABLE
        return kind, failure_detail(kind, target, str(exc))
    return None


def ensure_index(client: QdrantClient, model_id: str, dims: int) -> str:
    """Create the model's collection, both payload indexes, and the alias.

    ``pillar_id`` and ``paper_id`` are keyword indexes. Cloud strict mode
    rejects a filter on a key that has no index, and deleting a paper's old
    points filters on ``paper_id``.

    Returns the collection name (not the alias).
    """
    name = collection_name(model_id, dims)
    names = _collection_names(client)
    if name in names:
        logger.info("Collection '%s' already exists", name)
    else:
        try:
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=int(dims),
                    distance=models.Distance.COSINE,
                ),
            )
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise ProjectionError(
                    f"could not create collection {name}: {exc}"
                ) from exc
            logger.info("Collection '%s' already exists", name)
        else:
            logger.info(
                "Created collection '%s' with vector size %s and cosine distance",
                name,
                dims,
            )
    _ensure_payload_indexes(client, name)
    _ensure_alias(client, name)
    return name


def apply_projection(
    client: QdrantClient,
    *,
    target: str,
    model_id: str,
    dims: int,
    chunks: Sequence[ChunkVector],
    indexed_ids: AbstractSet[str],
    stale_ids: AbstractSet[str],
    rebuild: bool,
    dry_run: bool,
    scope_complete: bool,
    scroll_paper_ids: Optional[AbstractSet[str]],
) -> SyncResult:
    """Upsert missing chunks and delete points whose chunks are gone.

    ``scope_complete`` is the full library. The stale set is then every
    indexed chunk that is not in ``chunks``. A single-paper call leaves
    that false and passes ``stale_ids`` from ``projection_stale_chunks``
    (chunks deleted out from under the state row) plus a scroll limited
    to ``scroll_paper_ids``.

    Dry-run does not create, upsert, or delete. It still probes, so a
    suspended cloud shows up as unhealthy with the pending count beside it.
    """
    if target not in TARGETS:
        raise ProjectionError(f"unknown projection target {target!r}")

    live = {chunk.chunk_id for chunk in chunks}
    indexed = {str(item) for item in indexed_ids}
    stale = {str(item) for item in stale_ids}
    if scope_complete:
        stale |= indexed - live
    pending = list(chunks) if rebuild else [c for c in chunks if c.chunk_id not in indexed]

    result = SyncResult(
        target=target,
        model_id=model_id,
        dims=dims,
        collection=collection_name(model_id, dims),
        pending=len(pending),
        stale=len(stale),
        dry_run=dry_run,
    )

    failed = probe(client, target)
    if failed:
        kind, reason = failed
        result.unhealthy = True
        result.failure = kind
        result.error = reason
        return result

    mismatch = _first_dim_mismatch(chunks, dims)
    if mismatch is not None:
        result.error = (
            f"chunk {mismatch.chunk_id} is {len(mismatch.embedding)}-d, "
            f"collection is {dims}-d"
        )
        return result

    if dry_run:
        return result

    try:
        result.collection = ensure_index(client, model_id, dims)
    except ProjectionError as exc:
        result.failure = COLLECTION_OR_ALIAS_MISSING
        result.error = failure_detail(result.failure, target, str(exc))
        return result
    except Exception as exc:
        result.failure = COLLECTION_OR_ALIAS_MISSING
        result.error = failure_detail(
            result.failure,
            target,
            f"could not prepare collection {result.collection}: {exc}",
        )
        return result

    try:
        result.upserted_ids = _upsert_chunks(client, result.collection, pending)
    except Exception as exc:
        result.failure = INDEX_UNREACHABLE
        result.error = failure_detail(
            result.failure,
            target,
            f"upsert failed after {len(result.upserted_ids)} point(s): {exc}",
        )
        return result

    try:
        deleted = _delete_stale(
            client,
            result.collection,
            stale_chunk_ids=stale,
            live_ids=live,
            scroll_paper_ids=scroll_paper_ids,
            scope_complete=scope_complete,
        )
    except Exception as exc:
        result.failure = INDEX_UNREACHABLE
        result.error = failure_detail(
            result.failure, target, f"delete of stale points failed: {exc}"
        )
        return result
    result.deleted_ids = sorted(deleted)
    return result


def commit_projection(
    result: SyncResult,
    mark: Callable[[str, str, Sequence[str]], None],
    forget: Callable[[str, str, Sequence[str]], None],
) -> None:
    """Record a successful projection. A down target records nothing.

    Points that were upserted before a later batch failed are still
    recorded, so the retry does not rewrite them. Deletes are recorded
    only when the delete step finished.
    """
    if result.dry_run or result.unhealthy:
        return
    if result.upserted_ids:
        mark(result.target, result.model_id, result.upserted_ids)
    if result.error is None and result.deleted_ids:
        forget(result.target, result.model_id, result.deleted_ids)


def open_client(target: str) -> Optional[QdrantClient]:
    """A client for ``target``, or None when that target has no URL.

    Local reads ``QDRANT_URL`` (compose points this at the local server).
    Cloud reads ``QDRANT_CLOUD_URL``, which is optional.
    """
    url, api_key = _target_endpoint(target)
    if not url:
        return None
    if api_key:
        return QdrantClient(url=url, api_key=api_key, timeout=30)
    return QdrantClient(url=url, timeout=30)


def project_paper(pillar_id: str, paper_id: str) -> ProjectionOutcome:
    """Project one paper's stored embeddings to local, then cloud if configured.

    Returns how many of the paper's chunks the local index now represents
    (already-projected chunks count: a second run is not "zero chunks").

    Raises:
        ProjectionError: the local target failed or the paper has no stored
            chunks/embeddings. Cloud failures are logged and do not raise.
    """
    from .embeddings import active_model_id, get_embedding_provider

    model_id = active_model_id()
    dims = get_embedding_provider().dims
    chunks = _chunks_for(model_id, paper_id)
    if not chunks:
        raise ProjectionError(
            f"no {model_id} embeddings stored for {paper_id}; "
            f"the persist stage did not store chunks to project"
        )
    wrong_pillar = [c.chunk_id for c in chunks if c.pillar_id != pillar_id]
    if wrong_pillar:
        logger.warning(
            "Paper %s is pillar %s in the caller and %s on the stored chunks",
            paper_id,
            pillar_id,
            chunks[0].pillar_id,
        )

    from . import vectors

    local = vectors.get_client()
    if local is None:
        kind = INDEX_UNREACHABLE
        local_result = SyncResult(
            target="local",
            model_id=model_id,
            dims=dims,
            collection=collection_name(model_id, dims),
            failure=kind,
            error=failure_detail(
                kind, "local", "QDRANT_URL is not configured"
            ),
        )
    else:
        # loud=False so a local failure still lets the cloud copy catch up.
        # The raise below is what fails the VECTORS stage.
        local_result = _sync_loaded(
            "local",
            local,
            chunks,
            model_id=model_id,
            dims=dims,
            rebuild=False,
            dry_run=False,
            scope_complete=False,
            scroll_paper_ids={paper_id},
            loud=False,
        )

    notices: List[str] = []
    if _cloud_configured():
        try:
            cloud = open_client("cloud")
            cloud_result = _sync_loaded(
                "cloud",
                cloud,
                chunks,
                model_id=model_id,
                dims=dims,
                rebuild=False,
                dry_run=False,
                scope_complete=False,
                scroll_paper_ids={paper_id},
                loud=False,
            )
        except Exception as exc:
            notice = failure_detail(INDEX_UNREACHABLE, "cloud", str(exc))
            notices.append(notice)
            logger.warning(
                "cloud projection left behind for %s: %s", paper_id, notice
            )
        else:
            if cloud_result.unhealthy or cloud_result.error:
                notice = cloud_result.error or "cloud vector projection is behind"
                notices.append(notice)
                logger.warning(
                    "cloud projection left behind for %s: %s", paper_id, notice
                )

    if local is None or local_result.unhealthy or local_result.error:
        raise ProjectionError(
            local_result.error or "local vector index failed"
        )
    return ProjectionOutcome(len(chunks), notices)


def sync_targets(
    targets: Sequence[str],
    *,
    rebuild: bool,
    dry_run: bool,
) -> List[SyncResult]:
    """Project the active model's embeddings. Used by the CLI.

    ``targets`` is ``local``, ``cloud``, and/or both. Dry-run reads Postgres
    and probes Qdrant, and writes neither.
    """
    from .embeddings import active_model_id, get_embedding_provider

    chosen = _expand_targets(targets)
    model_id = active_model_id()
    provider = get_embedding_provider()
    if provider.model_id != model_id:
        raise ProjectionError(
            f"provider model {provider.model_id} does not match {model_id}"
        )
    chunks = _chunks_for(model_id, None)
    results: List[SyncResult] = []
    for target in chosen:
        client = open_client(target)
        if client is None:
            pending = 0
            message = "not configured"
            if target == "local":
                from . import db

                indexed = set(db.list_projection_state(target, model_id))
                pending = len(chunks) if rebuild else sum(
                    1 for chunk in chunks if chunk.chunk_id not in indexed
                )
                message = "local vector index is not configured (QDRANT_URL)"
            results.append(SyncResult(
                target=target,
                model_id=model_id,
                dims=provider.dims,
                collection=collection_name(model_id, provider.dims),
                pending=pending,
                dry_run=dry_run,
                failure=INDEX_UNREACHABLE,
                error=message,
            ))
            continue
        results.append(_sync_loaded(
            target,
            client,
            chunks,
            model_id=model_id,
            dims=provider.dims,
            rebuild=rebuild,
            dry_run=dry_run,
            scope_complete=True,
            scroll_paper_ids=None,
            loud=False,
        ))
    return results


def _sync_loaded(
    target: str,
    client: Optional[QdrantClient],
    chunks: Sequence[ChunkVector],
    *,
    model_id: str,
    dims: int,
    rebuild: bool,
    dry_run: bool,
    scope_complete: bool,
    scroll_paper_ids: Optional[AbstractSet[str]],
    loud: bool,
) -> SyncResult:
    from . import db

    if client is None:
        kind = INDEX_UNREACHABLE
        result = SyncResult(
            target=target,
            model_id=model_id,
            dims=dims,
            collection=collection_name(model_id, dims),
            dry_run=dry_run,
            failure=kind,
            error=failure_detail(kind, target, "not configured"),
        )
        if loud and not dry_run:
            raise ProjectionError(f"{target} vector index is not configured")
        logger.warning("%s projection left behind: not configured", target)
        return result

    try:
        indexed = set(db.list_projection_state(target, model_id))
        stale = set(db.list_projection_stale(target, model_id))
    except ProjectionError as exc:
        kind = PROJECTION_STATE_UNAVAILABLE
        result = SyncResult(
            target=target,
            model_id=model_id,
            dims=dims,
            collection=collection_name(model_id, dims),
            dry_run=dry_run,
            failure=kind,
            error=failure_detail(kind, target, str(exc)),
        )
        if loud and not dry_run:
            raise ProjectionError(result.error) from exc
        logger.warning("%s projection left behind: %s", target, result.error)
        return result
    result = apply_projection(
        client,
        target=target,
        model_id=model_id,
        dims=dims,
        chunks=chunks,
        indexed_ids=indexed,
        stale_ids=stale,
        rebuild=rebuild,
        dry_run=dry_run,
        scope_complete=scope_complete,
        scroll_paper_ids=scroll_paper_ids,
    )
    try:
        commit_projection(
            result,
            db.mark_projection_state,
            db.forget_projection_state,
        )
    except ProjectionError as exc:
        result.failure = result.failure or PROJECTION_STATE_UNAVAILABLE
        result.error = result.error or str(exc)
        if loud and not dry_run:
            raise
        logger.warning("%s projection state was not recorded: %s", target, result.error)
        return result
    if not dry_run and (result.unhealthy or result.error):
        if loud:
            raise ProjectionError(result.error or f"{target} vector index failed")
        logger.warning("%s projection left behind: %s", target, result.error)
    return result


def _chunks_for(model_id: str, paper_id: Optional[str]) -> List[ChunkVector]:
    from . import db

    rows = db.list_chunks_with_embeddings(model_id, paper_id)
    return [ChunkVector.from_row(row) for row in rows]


def _expand_targets(targets: Sequence[str]) -> List[str]:
    chosen: List[str] = []
    for name in targets:
        if name == "all":
            for item in TARGETS:
                if item not in chosen:
                    chosen.append(item)
            continue
        if name not in TARGETS:
            raise ProjectionError(
                f"unknown projection target {name!r}. Use local, cloud, or all."
            )
        if name not in chosen:
            chosen.append(name)
    if not chosen:
        raise ProjectionError("no projection target. Use local, cloud, or all.")
    return chosen


def _cloud_configured() -> bool:
    return bool(os.getenv("QDRANT_CLOUD_URL", "").strip())


def _target_endpoint(target: str) -> tuple[str, str]:
    if target == "local":
        return (
            os.getenv("QDRANT_URL", "").strip(),
            os.getenv("QDRANT_API_KEY", "").strip(),
        )
    if target == "cloud":
        return (
            os.getenv("QDRANT_CLOUD_URL", "").strip(),
            os.getenv("QDRANT_CLOUD_API_KEY", "").strip(),
        )
    raise ProjectionError(f"unknown projection target {target!r}")


def _first_dim_mismatch(chunks: Sequence[ChunkVector], dims: int) -> Optional[ChunkVector]:
    for chunk in chunks:
        if chunk.dims != dims or len(chunk.embedding) != dims:
            return chunk
    return None


def _collection_names(client: QdrantClient) -> List[str]:
    listed = client.get_collections()
    collections = getattr(listed, "collections", None) or []
    return [col.name for col in collections]


def _ensure_payload_indexes(client: QdrantClient, name: str) -> None:
    try:
        info = client.get_collection(name)
        schema = getattr(info, "payload_schema", None) or {}
    except Exception as exc:
        raise ProjectionError(
            f"could not read payload schema for {name}: {exc}"
        ) from exc
    for field_name in PAYLOAD_INDEX_FIELDS:
        if field_name in schema:
            continue
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            if "already exists" in str(exc).lower():
                continue
            raise ProjectionError(
                f"could not index {field_name} on {name}: {exc}"
            ) from exc
        logger.info("Created keyword payload index on '%s'", field_name)


def _ensure_alias(client: QdrantClient, name: str) -> None:
    try:
        current = client.get_aliases()
    except Exception as exc:
        raise ProjectionError(f"could not read collection aliases: {exc}") from exc
    aliases = getattr(current, "aliases", None) or []
    for item in aliases:
        if getattr(item, "alias_name", None) == READ_ALIAS and getattr(item, "collection_name", None) == name:
            return
    try:
        client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=name,
                        alias_name=READ_ALIAS,
                    )
                )
            ]
        )
    except Exception as exc:
        raise ProjectionError(
            f"could not point {READ_ALIAS} at {name}: {exc}"
        ) from exc
    logger.info("Alias '%s' points at '%s'", READ_ALIAS, name)


def _upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: Sequence[ChunkVector],
) -> List[str]:
    written: List[str] = []
    for start in range(0, len(chunks), UPSERT_BATCH):
        batch = list(chunks[start:start + UPSERT_BATCH])
        points = [_point(chunk) for chunk in batch]
        client.upsert(collection_name=collection, points=points)
        written.extend(chunk.chunk_id for chunk in batch)
    return written


def _point(chunk: ChunkVector) -> models.PointStruct:
    return models.PointStruct(
        id=point_id(chunk.chunk_id),
        vector=chunk.embedding,
        payload={
            "pillar_id": chunk.pillar_id,
            "paper_id": chunk.paper_id,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
        },
    )


def _delete_stale(
    client: QdrantClient,
    collection: str,
    *,
    stale_chunk_ids: AbstractSet[str],
    live_ids: AbstractSet[str],
    scroll_paper_ids: Optional[AbstractSet[str]],
    scope_complete: bool,
) -> set[str]:
    """Delete points for gone chunks, and points the scroll shows as orphans.

    ``scroll_paper_ids is None`` with ``scope_complete`` scrolls the whole
    collection. A paper-scoped call scrolls only that paper, which needs
    the ``paper_id`` payload index under strict mode.
    """
    forget: set[str] = set(stale_chunk_ids)
    point_ids: List[str] = [point_id(chunk_id) for chunk_id in sorted(forget)]

    if scope_complete:
        papers: Optional[Iterable[Optional[str]]] = [None]
    elif scroll_paper_ids:
        papers = list(scroll_paper_ids)
    else:
        papers = []

    for paper_id in papers:
        for record in _scroll(client, collection, paper_id):
            payload = getattr(record, "payload", None) or {}
            chunk_id = payload.get("chunk_id")
            chunk_key = str(chunk_id) if chunk_id else ""
            if chunk_key and chunk_key in live_ids:
                continue
            record_id = getattr(record, "id", None)
            if record_id is not None:
                point_ids.append(str(record_id))
            if chunk_key:
                forget.add(chunk_key)

    unique_points = list(dict.fromkeys(point_ids))
    for start in range(0, len(unique_points), UPSERT_BATCH):
        batch = unique_points[start:start + UPSERT_BATCH]
        if batch:
            client.delete(collection_name=collection, points_selector=batch)
    return forget


def _scroll(client: QdrantClient, collection: str, paper_id: Optional[str]):
    scroll_filter = None
    if paper_id is not None:
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=PAPER_ID_FIELD,
                    match=models.MatchValue(value=paper_id),
                )
            ]
        )
    offset = None
    seen_offsets: set[str] = set()
    while True:
        records, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=SCROLL_PAGE,
            offset=offset,
            with_payload=["chunk_id", "paper_id"],
            with_vectors=False,
        )
        for record in records or []:
            yield record
        if not records or offset is None:
            return
        key = str(offset)
        if key in seen_offsets:
            return
        seen_offsets.add(key)


def _coerce_vector(value, chunk_id: str) -> List[float]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            text = "[" + text[1:-1] + "]"
        import json
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProjectionError(
                f"chunk {chunk_id} embedding is not a JSON array"
            ) from exc
    if not isinstance(value, (list, tuple)):
        raise ProjectionError(f"chunk {chunk_id} embedding is not an array")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ProjectionError(f"chunk {chunk_id} embedding is not floats") from exc


def indexed_at_now() -> str:
    """Timestamp stored on a projection row. UTC, second resolution is enough."""
    return datetime.now(timezone.utc).isoformat()
