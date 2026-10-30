"""Load the map's evidence through PostgREST, and write snapshots.

``map_build.py`` does the computing. This module reads the rows it needs
(``papers``, ``pillars``, ``works``, ``work_citations``, ``paper_vectors``,
``paper_queue``, the previous ``map_snapshots`` positions), embeds the
pillar's goal and any queued title that has no stored vector with the local
bge model (free, CPU), and writes one ``map_snapshots`` row per pillar through
``put_map_snapshot`` (docs/migrations/034_map_snapshots.sql).

No paid model is called and Semantic Scholar is not contacted: the build is
pure computation over what M1 and M2 stored. ``--dry-run`` reads, computes and
prints; it does not load the embedding model and does not write.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import db
from .embeddings import EmbeddingError
from .map_build import (
    STATUS_COMPLETED,
    STATUS_RUNNING,
    VERDICT_ORDER,
    Citation,
    CitationIndex,
    LibraryPaper,
    MapInputs,
    MapResult,
    Pillar,
    QueueItem,
    Work,
    build_map,
    select_scope,
)
from .paper_vectors import MIGRATION as PAPER_VECTORS_MIGRATION
from .paper_vectors import (
    REF_KIND_EXTERNAL,
    REF_KIND_PAPER,
    SOURCE_TITLE_ABSTRACT,
    title_abstract_text,
)
from .work_ids import parse_paper_id
from .works_store import MIGRATION as WORKS_MIGRATION

logger = logging.getLogger(__name__)

MIGRATION = "docs/migrations/034_map_snapshots.sql"
KIND_MAP_BUILD = "map_build"
#: Webui-started builds (the map page's rebuild, M4/M5, and ingest hooks, M7).
#: The scheduler's weekly rebuild (M7) uses the existing 'scheduler' source.
TRIGGER_UI_MAP = "ui_map"

STAGE_LOAD = "map_load"
STAGE_PERSIST = "map_persist"

_PAGE = 1000
_IN_CHUNK = 100

StageCallback = Callable[[str, str, Optional[str]], None]
Embed = Callable[[Sequence[str]], List[List[float]]]


class MapStoreError(Exception):
    """The map's tables could not be read or written."""


def _error(error: Any) -> MapStoreError:
    text = str(error).lower()
    missing = any(
        token in text
        for token in ("pgrst202", "pgrst205", "does not exist", "schema cache")
    )
    if missing and any(
        token in text for token in ("map_snapshots", "put_map_snapshot")
    ):
        return MapStoreError(
            f"Map snapshots are not available ({error}). Apply {MIGRATION} by hand."
        )
    if missing and any(
        token in text for token in ("works", "work_citations", "work_ids", "work_id")
    ):
        return MapStoreError(
            f"The works tables are not available ({error}). "
            f"Apply {WORKS_MIGRATION} by hand "
            "and run `python -m nlp_pillars.cli works sync`."
        )
    if missing and "paper_vectors" in text:
        return MapStoreError(
            f"paper_vectors is not available ({error}). "
            f"Apply {PAPER_VECTORS_MIGRATION} by hand."
        )
    return MapStoreError(f"Could not read or write the map: {error}")


def _quote(value: str) -> str:
    """Quote a PostgREST ``in.(...)`` item: ids can hold commas and parentheses."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _chunks(items: Sequence[str], size: int = _IN_CHUNK) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _read(
    table: str,
    columns: str,
    order: str,
    narrow: Optional[Callable[[Any], Any]] = None,
) -> List[dict]:
    client = db.get_client()
    rows: List[dict] = []
    offset = 0
    while True:
        query = (
            client.table(table).select(columns).order(order).limit(_PAGE).offset(offset)
        )
        if narrow is not None:
            query = narrow(query)
        response = query.execute()
        if response["error"]:
            raise _error(response["error"])
        page = response["data"] or []
        rows.extend(page)
        if len(page) < _PAGE:
            return rows
        offset += len(page)


def _read_in(
    table: str, columns: str, order: str, column: str, values: Sequence[str], extra=None
) -> List[dict]:
    rows: List[dict] = []
    for chunk in _chunks(sorted(set(values))):

        def narrow(query, chunk=chunk):
            query = query.in_(column, [_quote(value) for value in chunk])
            return extra(query) if extra else query

        rows.extend(_read(table, columns, order, narrow))
    return rows


def _names(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out = []
    for item in value:
        name = item.get("name") if isinstance(item, dict) else item
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return tuple(out)


def _int(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------- reads


def load_pillars() -> Dict[str, Pillar]:
    """Every pillar, with its goal and focus areas."""
    rows = _read("pillars", "id,name,goal,focus_areas", "id")
    return {
        str(row["id"]): Pillar(
            id=str(row["id"]),
            name=row.get("name") or "",
            goal=row.get("goal") or "",
            focus_areas=tuple(
                str(area) for area in (row.get("focus_areas") or []) if area
            ),
        )
        for row in rows
    }


def load_library() -> List[LibraryPaper]:
    """Every library paper of every pillar."""
    rows = _read("papers", "id,pillar_id,work_id,title,year,authors", "id")
    return [
        LibraryPaper(
            paper_id=str(row["id"]),
            pillar_id=str(row.get("pillar_id") or ""),
            work_id=str(row["work_id"]) if row.get("work_id") else None,
            title=row.get("title") or "",
            year=_int(row.get("year")),
            authors=_names(row.get("authors")),
        )
        for row in rows
        if row.get("pillar_id")
    ]


def load_works() -> Dict[str, Work]:
    """Every work's display and count columns (no abstracts)."""
    rows = _read(
        "works",
        "id,title,year,publication_date,citation_count,influential_citation_count,authors,venue",
        "id",
    )
    return {
        str(row["id"]): Work(
            id=str(row["id"]),
            title=row.get("title") or "",
            year=_int(row.get("year")),
            publication_date=str(row["publication_date"])
            if row.get("publication_date")
            else None,
            citation_count=_int(row.get("citation_count")),
            influential_citation_count=_int(row.get("influential_citation_count")),
            authors=_names(row.get("authors")),
            venue=row.get("venue") or None,
        )
        for row in rows
    }


def load_citations() -> List[Citation]:
    """Every stored citation with its intents, sentences and influence flag."""
    rows = _read(
        "work_citations", "citing,cited,is_influential,intents,contexts", "citing,cited"
    )
    return [
        Citation(
            citing=str(row["citing"]),
            cited=str(row["cited"]),
            is_influential=bool(row.get("is_influential")),
            intents=tuple(row.get("intents") or ()),
            contexts=tuple(row.get("contexts") or ()),
        )
        for row in rows
    ]


def load_queue(pillar_id: str) -> List[dict]:
    """Read the pillar's ``paper_queue`` rows, processed or not."""
    return _read(
        "paper_queue",
        "id,paper_id,title,processed",
        "id",
        lambda query: query.eq("pillar_id", pillar_id),
    )


def resolve_queue(rows: Sequence[dict]) -> List[QueueItem]:
    """Queue rows with the work their paper id names, when one is stored."""
    from .works_store import _lookup_keys

    keys = {
        str(row["id"]): parse_paper_id(str(row.get("paper_id") or "")) for row in rows
    }
    wanted = [key for key in keys.values() if key]
    index = _lookup_keys(wanted) if wanted else None
    items = []
    for row in rows:
        key = keys[str(row["id"])]
        work_id = index.work_for(key) if (index is not None and key) else None
        items.append(
            QueueItem(
                queue_id=str(row["id"]),
                paper_id=str(row.get("paper_id") or ""),
                title=row.get("title") or "",
                processed=bool(row.get("processed")),
                work_id=str(work_id) if work_id else None,
            )
        )
    return items


def choose_model() -> Optional[str]:
    """Pick the title+abstract model with the most library vectors (ties by id)."""
    rows = _read(
        "paper_vectors",
        "ref_kind,ref_id,model_id",
        "ref_kind,ref_id,model_id",
        lambda query: query.eq("source", SOURCE_TITLE_ABSTRACT),
    )
    library: Dict[str, int] = {}
    for row in rows:
        model = str(row.get("model_id") or "")
        if model:
            library[model] = library.get(model, 0) + (
                1 if row.get("ref_kind") == REF_KIND_PAPER else 0
            )
    if not library:
        return None
    return sorted(library, key=lambda model: (-library[model], model))[0]


def load_vectors(
    model_id: str, library: Sequence[LibraryPaper], work_ids: Iterable[str]
) -> Dict[str, List[float]]:
    """Vectors keyed by map node id: library rows by paper id, externals by work id."""

    def same_model(query):
        return query.eq("model_id", model_id).eq("source", SOURCE_TITLE_ABSTRACT)

    out: Dict[str, List[float]] = {}
    papers = _read_in(
        "paper_vectors",
        "ref_id,vec",
        "ref_id",
        "ref_id",
        [paper.paper_id for paper in library],
        lambda query: same_model(query).eq("ref_kind", REF_KIND_PAPER),
    )
    by_paper = {str(row["ref_id"]): row.get("vec") for row in papers}
    node_of_paper = {paper.paper_id: paper.node_id for paper in library}
    wanted = sorted(set(work_ids))
    external = _read_in(
        "paper_vectors",
        "ref_id,vec",
        "ref_id",
        "ref_id",
        wanted,
        lambda query: same_model(query).eq("ref_kind", REF_KIND_EXTERNAL),
    )
    for row in external:
        if row.get("vec"):
            out[str(row["ref_id"])] = [float(value) for value in row["vec"]]
    # The library row wins over an external row for the same work.
    for paper_id in sorted(by_paper):
        vector = by_paper[paper_id]
        if vector and paper_id in node_of_paper:
            out[node_of_paper[paper_id]] = [float(value) for value in vector]
    return out


def load_previous_layout(
    scope: str,
) -> Tuple[Dict[str, Tuple[float, float]], Optional[str], Optional[str]]:
    """Read the last snapshot's positions and layout key (or why it could not)."""
    try:
        rows = _read(
            "map_snapshots",
            "scope,positions,layout_key",
            "scope",
            lambda query: query.eq("scope", scope),
        )
    except MapStoreError as exc:
        return {}, None, f"No previous layout to start from ({exc})"
    if not rows:
        return {}, None, None
    positions = rows[0].get("positions") or {}
    return (
        {
            str(node): (float(value[0]), float(value[1]))
            for node, value in positions.items()
            if isinstance(value, list) and len(value) == 2
        },
        rows[0].get("layout_key"),
        None,
    )


# --------------------------------------------------------------------- write


def put_snapshot(
    result: MapResult, model_id: Optional[str], run_id: Optional[str] = None
) -> dict:
    """Upsert one ``map_snapshots`` row. Returns ``{scope, built_at, changed}``."""
    positions = {
        node["id"]: [node["x"], node["y"]]
        for node in result.payload["nodes"]
        if node.get("x") is not None
    }
    response = db.get_client().rpc(
        "put_map_snapshot",
        {
            "p_scope": result.scope,
            "p_params_hash": result.params_hash,
            "p_payload_hash": result.payload_hash,
            "p_model_id": model_id,
            "p_stats": result.stats,
            "p_positions": positions,
            "p_layout_key": result.payload["layout"]["key"],
            "p_payload": result.payload,
            "p_run_id": run_id,
        },
    )
    if response["error"]:
        raise _error(response["error"])
    data = response["data"]
    if isinstance(data, list):
        data = data[0] if data else {}
    return data or {}


# -------------------------------------------------------------------- report


@dataclass
class PillarBuild:
    """One pillar's outcome, for the CLI and the run result."""

    pillar_id: str
    result: Optional[MapResult] = None
    written: bool = False
    changed: Optional[bool] = None
    error: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        """Return the run-result payload for this pillar."""
        out: Dict[str, Any] = {"pillar_id": self.pillar_id, "written": self.written}
        if self.error:
            out["error"] = self.error
        if self.result is not None:
            out["payload_hash"] = self.result.payload_hash
            out["stats"] = self.result.stats
            out["queue"] = queue_counts(self.result)
            out["warnings"] = self.result.warnings
        if self.changed is not None:
            out["changed"] = self.changed
        return out


def queue_counts(result: MapResult) -> Dict[str, int]:
    """Queue papers per verdict, in badge order."""
    counts: Dict[str, int] = {}
    for row in result.payload.get("queue") or []:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    return {verdict: counts[verdict] for verdict in VERDICT_ORDER if verdict in counts}


@dataclass
class MapBuildReport:
    """What ``map build`` did. ``render`` is the CLI output."""

    dry_run: bool
    model_id: Optional[str]
    pillars: List[PillarBuild] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        """Whether any pillar failed."""
        return any(build.error for build in self.pillars)

    def render(self) -> str:
        """Plain-text summary for the CLI."""
        lines = [
            f"map build ({'dry-run' if self.dry_run else 'write'})",
            f"model: {self.model_id or 'none'}",
        ]
        for build in self.pillars:
            if build.error:
                lines.append(f"{build.pillar_id}: failed: {build.error}")
                continue
            result = build.result
            assert result is not None
            stats = result.stats
            state = (
                "dry-run"
                if self.dry_run
                else ("written" if build.written else "not written")
            )
            if build.changed is False:
                state += ", unchanged"
            lines.append(
                f"{build.pillar_id}: {stats['works']} works "
                f"({stats['library_works']} yours), {stats['hybrid_edges']} links, "
                f"{stats['topics']} topics / {stats['regions']} regions, "
                f"{stats['lineage_paths']} lineage path(s), "
                f"{stats['suggestions']} suggestions, {stats['insights']} insights; "
                f"{state}; hash {result.payload_hash[:12]}"
            )
            counts = queue_counts(result)
            if counts:
                lines.append(
                    "  queue: "
                    + ", ".join(
                        f"{verdict} {count}" for verdict, count in counts.items()
                    )
                )
            lines.extend(f"  warning: {warning}" for warning in result.warnings)
        lines.extend(f"warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


# ---------------------------------------------------------------------- build


@dataclass
class Evidence:
    """The tables every pillar's build shares, read once."""

    pillars: Dict[str, Pillar]
    library: List[LibraryPaper]
    works: Dict[str, Work]
    citations: List[Citation]
    model_id: Optional[str]


def load_evidence() -> Evidence:
    """Read the tables every pillar's build shares."""
    return Evidence(
        pillars=load_pillars(),
        library=load_library(),
        works=load_works(),
        citations=load_citations(),
        model_id=choose_model(),
    )


def assemble_inputs(
    evidence: Evidence,
    pillar_id: str,
    *,
    embed: Optional[Embed] = None,
    embed_model_id: Optional[str] = None,
    with_queue: bool = True,
    with_previous: bool = True,
) -> Tuple[MapInputs, List[str]]:
    """One pillar's ``MapInputs``. Embeds only when ``embed`` is given.

    ``embed`` must produce vectors in the stored model's space; it is used only
    when ``embed_model_id`` equals the stored model.
    """
    warnings: List[str] = []
    pillar = evidence.pillars.get(pillar_id) or Pillar(id=pillar_id, name=pillar_id)
    index = CitationIndex(evidence.citations)
    scope = select_scope(pillar_id, evidence.library, evidence.works, index)
    queue = resolve_queue(load_queue(pillar_id)) if with_queue else []
    vectors: Dict[str, List[float]] = {}
    if evidence.model_id:
        work_ids = set(scope.nodes) | {item.work_id for item in queue if item.work_id}
        vectors = load_vectors(
            evidence.model_id, evidence.library, (node for node in work_ids if node)
        )
    else:
        warnings.append(
            "No title+abstract vectors are stored; "
            "run `python -m nlp_pillars.cli embeddings papers`."
        )

    anchors: List[List[float]] = []
    can_embed = (
        embed is not None and evidence.model_id and embed_model_id == evidence.model_id
    )
    if embed is not None and not can_embed:
        warnings.append(
            f"Stored vectors are {evidence.model_id or 'missing'}, "
            f"not {embed_model_id}: "
            "the pillar goal and unvectored queue titles are not embedded."
        )
    if can_embed:
        texts = [
            text for text in [pillar.goal, *pillar.focus_areas] if text and text.strip()
        ]
        missing = [
            item for item in queue if item.node_id not in vectors and item.title.strip()
        ]
        try:
            if texts:
                anchors = [list(vector) for vector in embed(texts)]
            if missing:
                embedded = embed(
                    [title_abstract_text(item.title, None) for item in missing]
                )
                for item, vector in zip(missing, embedded, strict=True):
                    vectors[item.node_id] = list(vector)
        except EmbeddingError as exc:
            # The local model is a convenience here, not the map: without it
            # the queue is scored against your papers only.
            anchors = []
            warnings.append(f"Could not embed the pillar goal or queued titles: {exc}")

    previous: Dict[str, Tuple[float, float]] = {}
    previous_key: Optional[str] = None
    if with_previous:
        previous, previous_key, warning = load_previous_layout(pillar_id)
        if warning:
            warnings.append(warning)
    return (
        MapInputs(
            pillar=pillar,
            library=evidence.library,
            works=evidence.works,
            citations=evidence.citations,
            vectors=vectors,
            model_id=evidence.model_id,
            queue=queue,
            anchors=anchors,
            previous_positions=previous,
            previous_layout_key=previous_key,
        ),
        warnings,
    )


def _local_embedder() -> Tuple[Embed, str]:
    from .embeddings import LOCAL_MODEL_ID, LocalBgeEmbeddingProvider

    provider = LocalBgeEmbeddingProvider()
    return provider.embed_documents, LOCAL_MODEL_ID


def build_pillar(
    evidence: Evidence,
    pillar_id: str,
    *,
    dry_run: bool,
    on_stage: Optional[StageCallback] = None,
    check_cancel: Optional[Callable[[], None]] = None,
    embedder: Optional[Tuple[Embed, str]] = None,
    run_id: Optional[str] = None,
) -> PillarBuild:
    """Assemble, compute and (unless ``dry_run``) write one pillar's snapshot."""

    def stage(name: str, status: str, detail: Optional[str] = None) -> None:
        if status == STATUS_RUNNING and check_cancel is not None:
            check_cancel()
        if on_stage is not None:
            on_stage(name, status, detail)

    build = PillarBuild(pillar_id)
    stage(STAGE_LOAD, STATUS_RUNNING, "Reading works, citations and vectors")
    embed, embed_model = embedder if embedder else (None, None)
    inputs, warnings = assemble_inputs(
        evidence,
        pillar_id,
        embed=None if dry_run else embed,
        embed_model_id=embed_model,
    )
    stage(
        STAGE_LOAD,
        STATUS_COMPLETED,
        f"{len(evidence.works)} works, {len(evidence.citations)} citations, "
        f"{len(inputs.vectors)} vectors, {len(inputs.queue)} queued",
    )
    result = build_map(inputs, progress=stage, notes=warnings)
    build.result = result
    if dry_run:
        return build
    stage(STAGE_PERSIST, STATUS_RUNNING, "Saving the map")
    saved = put_snapshot(result, inputs.model_id, run_id)
    build.written = True
    build.changed = saved.get("changed") if isinstance(saved, dict) else None
    stage(
        STAGE_PERSIST,
        STATUS_COMPLETED,
        "Saved"
        + (" (unchanged)" if build.changed is False else "")
        + f", hash {result.payload_hash[:12]}",
    )
    return build


def execute_map_build(
    *,
    pillar: Optional[str] = None,
    dry_run: bool = False,
    on_stage: Optional[StageCallback] = None,
    check_cancel: Optional[Callable[[], None]] = None,
    embedder: Optional[Tuple[Embed, str]] = None,
    run_id: Optional[str] = None,
) -> MapBuildReport:
    """Build one pillar's map, or every pillar that has library papers.

    The shared tables are read once. One pillar failing does not stop the
    next; the report says which failed and why. ``--dry-run`` never loads the
    embedding model and never writes.
    """
    evidence = load_evidence()
    report = MapBuildReport(dry_run=dry_run, model_id=evidence.model_id)
    with_papers = sorted({paper.pillar_id for paper in evidence.library})
    targets = [pillar] if pillar else with_papers
    if pillar and pillar not in with_papers:
        report.warnings.append(
            f"{pillar} has no library papers; its map holds only queued papers."
        )
    if not dry_run and embedder is None:
        embedder = _local_embedder()
    for pillar_id in targets:
        try:
            report.pillars.append(
                build_pillar(
                    evidence,
                    pillar_id,
                    dry_run=dry_run,
                    on_stage=on_stage,
                    check_cancel=check_cancel,
                    embedder=embedder,
                    run_id=run_id,
                )
            )
        except MapStoreError as exc:
            if pillar:
                raise
            report.pillars.append(PillarBuild(pillar_id, error=str(exc)))
    return report


def run_map_build_job(
    pillar_id: str,
    on_stage: StageCallback,
    check_cancel: Callable[[], None],
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one pillar's ``map_build`` pipeline run and return its result payload."""
    report = execute_map_build(
        pillar=pillar_id, on_stage=on_stage, check_cancel=check_cancel, run_id=run_id
    )
    return report.pillars[0].summary()
