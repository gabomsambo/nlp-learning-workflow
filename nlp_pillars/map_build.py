"""The paper map's connections layer: one pillar's evidence in, one snapshot out.

Pure: rows in, a JSON-ready payload out. No database, no network, no model.
``map_store.py`` loads the rows and writes the result to ``map_snapshots``.

What one build computes, in order (paper-map scout report, sections 3-5):

1. **Scope.** The pillar's library works plus their neighbourhood: every
   reference, the most-cited and the newest citers of each, and any work linked
   to at least two of them (extended direct citation).
2. **Balanced hybrid graph.** bge title+abstract kNN, direct citations, and
   bibliographic coupling capped at the top 10 per work. Uncapped coupling
   swamps the text edges (report section 4.4).
3. **Communities.** Leiden (graspologic-native, modularity, fixed seed) at two
   resolutions: regions and topics. Below ``MIN_CLUSTER_SCOPE`` works the
   pillar itself is the only community.
4. **Lineage.** Search Path Count weights on the citation DAG, then key-node
   paths: the source-to-sink path through the most library works, highest
   SPC first. Labelled a sketch: the newest end moves when citers are refetched.
5. **Importance.** Local citation score (citations from inside the map),
   year-cohort citation percentile, sampled betweenness. No global PageRank on
   an ego-sample: it measures how the neighbourhood was sampled.
6. **Layout.** networkx spring layout on the hybrid graph, warm-started from
   the previous snapshot so the map does not reshuffle on every build.
7. **Suggestions.** Per community of the pillar's own works: reciprocal-rank
   fusion of a personalised PageRank over the citation graph and bge
   max-cosine to the seeds. RRF beat every single scorer and the earlier
   hand-weighted blend in the report's leave-one-out test (median rank 40.5
   vs 221.5).
8. **Queue verdicts.** The same scorer over ``paper_queue``, with the pillar's
   goal and focus areas as a second anchor, so a young pillar does not call
   everything outside its first few papers off-topic (report section 4.9).
9. **Insights and evidence.** Template sentences over stored evidence only.

Every number that shapes the result is a named constant below, and all of
them go into ``params_hash``. The payload is canonical JSON: the same inputs
in any order give the same ``payload_hash``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import networkx as nx
import numpy as np

# --------------------------------------------------------------------- version

#: Bumped when the payload shape changes, so a reader can refuse an old one.
MAP_SNAPSHOT_VERSION = 1

# ----------------------------------------------------------------------- scope

#: Citers of each library work kept by citation count. The report's neighbour
#: cap (section 5.3): ~93 neighbours per library paper on the real data.
CITERS_PER_SEED = 60
#: Newest citers of each library work kept as well, for the Frontier end.
FRONTIER_CITERS_PER_SEED = 20
#: A work linked (citing or cited) to this many library works joins the scope
#: whatever its rank: extended direct citation (Waltman et al. 2020).
MIN_SEEDS_LINKED = 2
#: Hard cap on one pillar's scope. Library works always stay.
MAX_SCOPE_WORKS = 3000

# -------------------------------------------------------------- hybrid graph

#: Text neighbours per work (report section 3.2: top-k 10-15 for layout).
TEXT_KNN_K = 10
#: Text edge weights are min-max scaled from this percentile of the kNN
#: cosines up to their maximum, and never below ``TEXT_WEIGHT_MIN``, so a
#: text edge is worth at most one direct citation.
TEXT_WEIGHT_LOW_PERCENTILE = 5.0
TEXT_WEIGHT_MIN = 0.05
#: Coupled partners kept per work. The cap is what balances the graph:
#: every coupled pair outnumbered text edges 9:1 and swamped them.
COUPLING_TOP_K = 10
#: Two works are coupled when they share at least this many references.
MIN_SHARED_REFERENCES = 2
#: A reference cited by more works than this inside the scope says little about
#: any pair of them (a hub discount, Klavans & Boyack 2017). Skipped when
#: counting shared references and co-citations.
COUPLING_MAX_CITING = 300
#: Relative weights of the three layers in the hybrid graph.
TEXT_LAYER_WEIGHT = 1.0
DIRECT_LAYER_WEIGHT = 1.0
COUPLING_LAYER_WEIGHT = 1.0

# ---------------------------------------------------------------- communities

#: Below this many works a computed clustering is noise; the pillar is shown
#: as one community instead (report section 3.2).
MIN_CLUSTER_SCOPE = 30
#: Leiden resolutions for the two levels. On the 1,227-work neighbourhood,
#: 0.3 gave 4 regions and 1.0 gave 10 topics (report section 4.4).
REGION_RESOLUTION = 0.3
TOPIC_RESOLUTION = 1.0
LEIDEN_SEED = 0
#: Full Leiden cycles, each starting from the last result.
LEIDEN_ITERATIONS = 10
#: c-TF-IDF title terms kept per community. Free names until M8 adds Haiku's.
COMMUNITY_TERMS = 5
COMMUNITY_TERM_MIN_LENGTH = 3

# -------------------------------------------------------------------- lineage

#: Key-node paths per pillar. Each further path covers library works the
#: earlier ones missed.
MAX_LINEAGE_PATHS = 3
#: Highest-SPC works kept between two consecutive library works in the
#: condensed path. The full path stays in the payload.
LINEAGE_CONDENSED_PER_GAP = 2

# ----------------------------------------------------------------- importance

#: A year cohort smaller than this widens by a year on each side, up to
#: ``COHORT_MAX_WINDOW`` years.
COHORT_MIN_SIZE = 5
COHORT_MAX_WINDOW = 3
#: Source nodes sampled for betweenness. A scope this small or smaller is exact.
BETWEENNESS_SAMPLES = 256
BETWEENNESS_SEED = 0
#: Works named as bridges in the insights.
BRIDGE_COUNT = 3

# --------------------------------------------------------------------- layout

LAYOUT_SEED = 0
LAYOUT_ITERATIONS = 50
#: Fewer moves from a warm start, so an unchanged map stays where it was.
WARM_LAYOUT_ITERATIONS = 15
#: How far a new work is placed from the mean of its placed neighbours.
LAYOUT_JITTER = 0.05

# ---------------------------------------------------------------- suggestions

#: Restart probability of the personalised PageRank (alpha = 1 - this).
PPR_RESTART = 0.15
PPR_TOLERANCE = 1e-10
PPR_MAX_ITERATIONS = 500
#: Reciprocal-rank-fusion constant. k=10 gave median rank 40.5 against 54.0
#: for the customary k=60 (report section 4.5; chosen on the same 12 trials).
RRF_K = 10
SUGGESTIONS_PER_COMMUNITY = 10
PILLAR_SUGGESTIONS = 20
#: A candidate is a duplicate of a library work when their cosine reaches this
#: percentile of the scope's pairwise cosines and their titles match. Both
#: scorers surfaced the journal version of a library paper (cosine 0.99).
DUPLICATE_PERCENTILE = 99.9
#: The cut for a scope too small for a percentile.
DUPLICATE_FALLBACK_COSINE = 0.95
#: Title token Jaccard at or above which two titles "match".
DUPLICATE_TITLE_JACCARD = 0.6
#: Below this many vectors a percentile of pairwise cosines is noise
#: (same reasoning and value as ``similarity.MIN_PERCENTILE_SCOPE``).
MIN_PERCENTILE_SCOPE = 30
#: Pairwise cosines kept for percentiles. Larger scopes are sampled.
EXACT_PAIRS_MAX_VECTORS = 3000
SAMPLED_PAIRS = 2_000_000
SAMPLE_SEED = 0

# ---------------------------------------------------------------------- queue

#: Queue badges (report section 4.9). Rank fractions are of the scored pool.
STRONG_RANK_FRACTION = 0.05
RELATED_RANK_FRACTION = 0.25
#: "On the pillar's topic" when the cosine to the pillar's goal or a focus area
#: reaches this percentile of the pool's. p75 was 0.677 on the real queue.
ANCHOR_PERCENTILE = 75.0

VERDICT_STRONG = "close_strong"
VERDICT_RELATED = "close_related"
VERDICT_NEW_DIRECTION = "on_topic_new_direction"
VERDICT_FAR = "far_unanchored"
VERDICT_OFF_TOPIC = "off_topic"
VERDICT_DUPLICATE = "duplicate"
VERDICT_ALREADY_YOURS = "already_yours"
VERDICT_UNSCORED = "unscored"

#: Sort order of the queue, best fit first. Nothing is hidden.
VERDICT_ORDER = (
    VERDICT_STRONG,
    VERDICT_RELATED,
    VERDICT_NEW_DIRECTION,
    VERDICT_FAR,
    VERDICT_OFF_TOPIC,
    VERDICT_DUPLICATE,
    VERDICT_ALREADY_YOURS,
    VERDICT_UNSCORED,
)

# --------------------------------------------------------- insights, evidence

#: A community with no library work is a gap when it has at least this many works.
GAP_MIN_COMMUNITY_SIZE = 10
GAP_COUNT = 3
#: Hybrid neighbours explained per focus work (library, suggestion, queue).
EVIDENCE_NEIGHBOURS = 8
#: Citing sentences are already capped at 300 characters by M1.
CONTEXT_CHARS = 300
#: S2 intents in the words the panel uses (report section 3.3).
INTENT_WORDS = {
    "methodology": "builds on",
    "result": "compares with",
    "background": "background",
}

# ------------------------------------------------------------------ rounding

SCORE_DECIMALS = 6
LAYOUT_DECIMALS = 4

# --------------------------------------------------------------- node kinds

KIND_LIBRARY = "library"
KIND_DOCUMENT = "document"  # a library paper Semantic Scholar does not know
KIND_EXTERNAL = "external"
KIND_QUEUE = "queue"  # a queued paper that is not a known work

DOCUMENT_PREFIX = "paper:"
QUEUE_PREFIX = "queue:"

Pair = Tuple[str, str]
Progress = Callable[[str, str, Optional[str]], None]

STAGE_GRAPH = "map_graph"
STAGE_COMMUNITIES = "map_communities"
STAGE_LINEAGE = "map_lineage"
STAGE_LAYOUT = "map_layout"
STAGE_SUGGEST = "map_suggest"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"


# ===================================================================== inputs


@dataclass(frozen=True)
class Work:
    """One row of ``works`` (migration 031), the columns the map reads."""

    id: str
    title: str = ""
    year: Optional[int] = None
    publication_date: Optional[str] = None
    citation_count: Optional[int] = None
    influential_citation_count: Optional[int] = None
    authors: Tuple[str, ...] = ()
    venue: Optional[str] = None


@dataclass(frozen=True)
class LibraryPaper:
    """One ``papers`` row. ``work_id`` is NULL for the captain's own documents."""

    paper_id: str
    pillar_id: str
    work_id: Optional[str] = None
    title: str = ""
    year: Optional[int] = None
    authors: Tuple[str, ...] = ()

    @property
    def node_id(self) -> str:
        """The work id, or ``paper:<id>`` for a document S2 cannot know."""
        return self.work_id or f"{DOCUMENT_PREFIX}{self.paper_id}"


@dataclass(frozen=True)
class Citation:
    """One ``work_citations`` row: ``citing`` cites ``cited``."""

    citing: str
    cited: str
    is_influential: bool = False
    intents: Tuple[str, ...] = ()
    contexts: Tuple[str, ...] = ()


@dataclass(frozen=True)
class QueueItem:
    """One ``paper_queue`` row. ``work_id`` when its id names a known work."""

    queue_id: str
    paper_id: str
    title: str = ""
    processed: bool = False
    work_id: Optional[str] = None

    @property
    def node_id(self) -> str:
        """The work id, or ``queue:<id>`` for a paper no work names yet."""
        return self.work_id or f"{QUEUE_PREFIX}{self.queue_id}"


@dataclass(frozen=True)
class Pillar:
    """The pillar the map is for. ``goal`` and focus areas are the second anchor."""

    id: str
    name: str = ""
    goal: str = ""
    focus_areas: Tuple[str, ...] = ()


@dataclass
class MapInputs:
    """Everything one pillar's build reads.

    ``library`` holds every pillar's papers: another pillar's work in this
    scope is still yours (never suggested) and is where bridges come from.
    ``vectors`` is keyed by node id (``LibraryPaper.node_id`` /
    ``QueueItem.node_id`` / work id), one bge model. ``anchors`` are the
    vectors of the pillar's goal and focus areas, in the same space.
    """

    pillar: Pillar
    library: List[LibraryPaper] = field(default_factory=list)
    works: Dict[str, Work] = field(default_factory=dict)
    citations: List[Citation] = field(default_factory=list)
    vectors: Dict[str, List[float]] = field(default_factory=dict)
    model_id: Optional[str] = None
    queue: List[QueueItem] = field(default_factory=list)
    anchors: List[List[float]] = field(default_factory=list)
    previous_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    previous_layout_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready copy. ``from_dict`` reads it back."""
        return {
            "pillar": asdict(self.pillar),
            "library": [asdict(paper) for paper in self.library],
            "works": [asdict(work) for _, work in sorted(self.works.items())],
            "citations": [asdict(citation) for citation in self.citations],
            "vectors": {
                key: list(map(float, value))
                for key, value in sorted(self.vectors.items())
            },
            "model_id": self.model_id,
            "queue": [asdict(item) for item in self.queue],
            "anchors": [list(map(float, value)) for value in self.anchors],
            "previous_positions": {
                key: list(value)
                for key, value in sorted(self.previous_positions.items())
            },
            "previous_layout_key": self.previous_layout_key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MapInputs":
        """Inverse of ``to_dict``."""

        def tup(value: Any) -> Tuple[str, ...]:
            return tuple(str(item) for item in (value or ()))

        pillar = data["pillar"]
        return cls(
            pillar=Pillar(
                id=pillar["id"],
                name=pillar.get("name") or "",
                goal=pillar.get("goal") or "",
                focus_areas=tup(pillar.get("focus_areas")),
            ),
            library=[
                LibraryPaper(**{**row, "authors": tup(row.get("authors"))})
                for row in data.get("library") or []
            ],
            works={
                row["id"]: Work(**{**row, "authors": tup(row.get("authors"))})
                for row in data.get("works") or []
            },
            citations=[
                Citation(
                    citing=row["citing"],
                    cited=row["cited"],
                    is_influential=bool(row.get("is_influential")),
                    intents=tup(row.get("intents")),
                    contexts=tup(row.get("contexts")),
                )
                for row in data.get("citations") or []
            ],
            vectors={
                key: list(value) for key, value in (data.get("vectors") or {}).items()
            },
            model_id=data.get("model_id"),
            queue=[QueueItem(**row) for row in data.get("queue") or []],
            anchors=[list(value) for value in data.get("anchors") or []],
            previous_positions={
                key: (float(value[0]), float(value[1]))
                for key, value in (data.get("previous_positions") or {}).items()
            },
            previous_layout_key=data.get("previous_layout_key"),
        )


@dataclass
class MapResult:
    """One pillar's snapshot. ``payload_hash`` is over the canonical payload JSON."""

    scope: str
    payload: Dict[str, Any]
    payload_hash: str
    params_hash: str
    stats: Dict[str, Any]
    warnings: List[str]
    #: How the layout was made: ``{"warm_started": n, "reused": bool}``. Not hashed.
    layout: Dict[str, Any] = field(default_factory=dict)


def map_params() -> Dict[str, Any]:
    """Every constant that shapes a snapshot. Its hash is ``params_hash``."""
    return {
        name: value
        for name, value in sorted(globals().items())
        if name.isupper()
        and not name.startswith(("VERDICT_", "KIND_", "STAGE_", "STATUS_"))
        and isinstance(value, (int, float, str, tuple, dict))
        and not name.endswith("_PREFIX")
    }


def canonical_json(value: Any) -> str:
    """Sorted keys, no whitespace: the bytes the hashes are taken over."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_of(value: Any) -> str:
    """sha256 hex of ``canonical_json(value)``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def params_hash() -> str:
    """Hash of ``map_params()``. A constant change makes every snapshot stale."""
    return sha256_of(map_params())


# ============================================================= citation index


class CitationIndex:
    """References and citers of every work, and the stored edge between two works."""

    def __init__(self, citations: Iterable[Citation]):
        """Index ``citations``; self-citations and blank ids are dropped."""
        self.refs_of: Dict[str, Set[str]] = defaultdict(set)
        self.cited_by: Dict[str, Set[str]] = defaultdict(set)
        self.edges: Dict[Pair, Citation] = {}
        for citation in citations:
            if (
                not citation.citing
                or not citation.cited
                or citation.citing == citation.cited
            ):
                continue
            key = (citation.citing, citation.cited)
            stored = self.edges.get(key)
            # Two rows for one edge cannot happen in Postgres; in a fixture keep
            # the richer one, so the result does not depend on input order.
            if stored is None or _edge_rank(citation) > _edge_rank(stored):
                self.edges[key] = citation
            self.refs_of[citation.citing].add(citation.cited)
            self.cited_by[citation.cited].add(citation.citing)

    def linked(self, left: str, right: str) -> bool:
        """Tell whether either work cites the other."""
        return (left, right) in self.edges or (right, left) in self.edges


def _edge_rank(citation: Citation) -> tuple:
    return (
        citation.is_influential,
        len(citation.intents),
        len(citation.contexts),
        tuple(sorted(citation.intents)),
        tuple(citation.contexts),
    )


def _pair(left: str, right: str) -> Pair:
    return (left, right) if left <= right else (right, left)


# ====================================================================== scope


@dataclass
class Scope:
    """The works on one pillar's map."""

    nodes: List[str]
    seeds: List[str]
    linked: Dict[str, int]
    unknown_ids: int
    capped: int


def _date_key(work: Optional[Work]) -> tuple:
    if work is None:
        return (-1, "")
    return (work.year if work.year is not None else -1, work.publication_date or "")


def select_scope(
    pillar_id: str,
    library: Sequence[LibraryPaper],
    works: Mapping[str, Work],
    index: CitationIndex,
) -> Scope:
    """Select the pillar's library works and their extended neighbourhood.

    A neighbour must be a known work (a ``works`` row). Library works of other
    pillars are kept when they fall in the neighbourhood. Above
    ``MAX_SCOPE_WORKS`` the works linked to most seeds, then the most cited,
    are kept.
    """
    seeds = sorted({paper.node_id for paper in library if paper.pillar_id == pillar_id})
    linked: Counter = Counter()
    chosen: Set[str] = set(seeds)
    unknown: Set[str] = set()
    for seed in seeds:
        refs = index.refs_of.get(seed, set())
        citers = index.cited_by.get(seed, set())
        for work_id in refs | citers:
            if work_id in works:
                linked[work_id] += 1
            else:
                unknown.add(work_id)
        chosen.update(work_id for work_id in refs if work_id in works)
        known_citers = sorted(work_id for work_id in citers if work_id in works)
        by_count = sorted(
            known_citers,
            key=lambda work_id: (-(works[work_id].citation_count or 0), work_id),
        )
        chosen.update(by_count[:CITERS_PER_SEED])
        by_date = sorted(
            known_citers,
            key=lambda work_id: (_neg_date(works.get(work_id)), work_id),
        )
        chosen.update(by_date[:FRONTIER_CITERS_PER_SEED])
    chosen.update(
        work_id for work_id, count in linked.items() if count >= MIN_SEEDS_LINKED
    )

    capped = 0
    if len(chosen) > MAX_SCOPE_WORKS:
        others = sorted(
            (work_id for work_id in chosen if work_id not in seeds),
            key=lambda work_id: (
                -linked.get(work_id, 0),
                -((works[work_id].citation_count or 0) if work_id in works else 0),
                work_id,
            ),
        )
        keep = max(MAX_SCOPE_WORKS - len(seeds), 0)
        capped = len(others) - keep
        chosen = set(seeds) | set(others[:keep])
    return Scope(
        nodes=sorted(chosen),
        seeds=seeds,
        linked={key: linked[key] for key in sorted(linked) if key in chosen},
        unknown_ids=len(unknown),
        capped=capped,
    )


def _neg_date(work: Optional[Work]) -> tuple:
    year, date = _date_key(work)
    # Newest first: negate the year, invert the date string's order.
    return (-year, tuple(-ord(char) for char in date))


# ===================================================================== vectors


class VectorSpace:
    """Unit vectors of one model, in sorted node order."""

    def __init__(self, vectors: Mapping[str, Sequence[float]]):
        """Normalise the most common width's vectors; others are dropped."""
        usable = sorted(
            key for key, value in vectors.items() if value is not None and len(value)
        )
        width = Counter(len(vectors[key]) for key in usable).most_common(1)
        self.dims = width[0][0] if width else 0
        self.ids = [key for key in usable if len(vectors[key]) == self.dims]
        self.row = {key: position for position, key in enumerate(self.ids)}
        matrix = np.asarray([vectors[key] for key in self.ids], dtype=np.float64)
        if matrix.size:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            matrix = matrix / norms
        self.unit = (
            matrix.reshape(len(self.ids), self.dims) if self.ids else np.zeros((0, 0))
        )

    def has(self, node: str) -> bool:
        """Whether ``node`` has a vector."""
        return node in self.row

    def vector(self, node: str) -> np.ndarray:
        """Return the unit vector of ``node``."""
        return self.unit[self.row[node]]

    def matrix(self, nodes: Sequence[str]) -> np.ndarray:
        """Return the unit vectors of ``nodes``, one row each."""
        return (
            self.unit[[self.row[node] for node in nodes]]
            if nodes
            else np.zeros((0, self.dims))
        )


def pairwise_cosines(unit: np.ndarray) -> np.ndarray:
    """Sorted upper-triangle cosines of ``unit``; sampled (fixed seed) above the cap."""
    size = int(unit.shape[0])
    if size < 2:
        return np.zeros(0)
    if size <= EXACT_PAIRS_MAX_VECTORS:
        values = (unit @ unit.T)[np.triu_indices(size, k=1)]
    else:
        rng = np.random.default_rng(SAMPLE_SEED)
        left = rng.integers(0, size, SAMPLED_PAIRS)
        right = rng.integers(0, size - 1, SAMPLED_PAIRS)
        right = right + (right >= left)
        values = np.einsum("ij,ij->i", unit[left], unit[right])
    return np.sort(values)


def cosine_percentile(sorted_cosines: np.ndarray, cosine: float) -> Optional[float]:
    """Share of the scope's pairs (0-100) with a cosine at or below ``cosine``."""
    if sorted_cosines.size < 1:
        return None
    below = int(np.searchsorted(sorted_cosines, cosine, side="right"))
    return 100.0 * below / sorted_cosines.size


# ================================================================ hybrid graph


def text_knn_edges(
    space: VectorSpace, nodes: Sequence[str], k: int = TEXT_KNN_K
) -> Dict[Pair, float]:
    """Each node's top-``k`` text neighbours, as scaled edge weights in (0, 1].

    Ties break by node id (stable sort over sorted ids).
    """
    with_vec = [node for node in sorted(nodes) if space.has(node)]
    if len(with_vec) < 2:
        return {}
    unit = space.matrix(with_vec)
    scores = unit @ unit.T
    raw: Dict[Pair, float] = {}
    for position, node in enumerate(with_vec):
        row = scores[position].copy()
        row[position] = -np.inf
        for other in np.argsort(-row, kind="stable")[: min(k, len(with_vec) - 1)]:
            key = _pair(node, with_vec[int(other)])
            raw[key] = max(raw.get(key, -1.0), float(scores[position, int(other)]))
    values = np.asarray(list(raw.values()))
    low, high = (
        float(np.percentile(values, TEXT_WEIGHT_LOW_PERCENTILE)),
        float(values.max()),
    )
    span = high - low
    out: Dict[Pair, float] = {}
    for key in sorted(raw):
        scaled = (raw[key] - low) / span if span > 0 else 1.0
        out[key] = float(min(1.0, max(TEXT_WEIGHT_MIN, scaled)))
    return out


def direct_edges(nodes: Iterable[str], index: CitationIndex) -> Dict[Pair, float]:
    """One undirected edge per citation between two nodes."""
    members = set(nodes)
    return {
        _pair(citing, cited): 1.0
        for (citing, cited) in sorted(index.edges)
        if citing in members and cited in members
    }


def shared_reference_counts(
    nodes: Iterable[str], index: CitationIndex
) -> Dict[Pair, int]:
    """Pairs of nodes sharing at least ``MIN_SHARED_REFERENCES`` references.

    A reference cited by more than ``COUPLING_MAX_CITING`` of the nodes is
    skipped (hub discount).
    """
    members = sorted(set(nodes))
    citing_of: Dict[str, List[str]] = defaultdict(list)
    for node in members:
        for ref in index.refs_of.get(node, ()):
            citing_of[ref].append(node)
    counts: Counter = Counter()
    for ref in sorted(citing_of):
        group = citing_of[ref]
        if len(group) < 2 or len(group) > COUPLING_MAX_CITING:
            continue
        for left_position, left in enumerate(group):
            for right in group[left_position + 1 :]:
                counts[(left, right)] += 1
    return {
        key: count for key, count in counts.items() if count >= MIN_SHARED_REFERENCES
    }


def cocitation_counts(nodes: Iterable[str], index: CitationIndex) -> Dict[Pair, int]:
    """Pairs of nodes cited together by at least ``MIN_SHARED_REFERENCES`` nodes."""
    members = set(nodes)
    counts: Counter = Counter()
    for node in sorted(members):
        cited = sorted(ref for ref in index.refs_of.get(node, ()) if ref in members)
        if len(cited) < 2 or len(cited) > COUPLING_MAX_CITING:
            continue
        for left_position, left in enumerate(cited):
            for right in cited[left_position + 1 :]:
                counts[(left, right)] += 1
    return {
        key: count for key, count in counts.items() if count >= MIN_SHARED_REFERENCES
    }


def salton(shared: int, left_size: int, right_size: int) -> float:
    """Cosine of two sets from their overlap and sizes."""
    if not shared or left_size <= 0 or right_size <= 0:
        return 0.0
    return shared / math.sqrt(left_size * right_size)


def coupling_edges(
    shared: Mapping[Pair, int],
    index: CitationIndex,
    nodes: Iterable[str],
    k: int = COUPLING_TOP_K,
) -> Dict[Pair, float]:
    """Each node's top-``k`` coupled partners (Salton), scaled so the strongest is 1.

    Salton, not association strength: it is what the report's balanced hybrid
    measured (70% of citation links inside clusters, seed NMI 0.83).
    """
    members = set(nodes)
    partners: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    for (left, right), count in shared.items():
        if left not in members or right not in members:
            continue
        score = salton(
            count, len(index.refs_of.get(left, ())), len(index.refs_of.get(right, ()))
        )
        partners[left].append((score, right))
        partners[right].append((score, left))
    kept: Dict[Pair, float] = {}
    for node in sorted(partners):
        for score, other in sorted(
            partners[node], key=lambda item: (-item[0], item[1])
        )[:k]:
            kept[_pair(node, other)] = score
    if not kept:
        return {}
    top = max(kept.values())
    return {key: kept[key] / top for key in sorted(kept)}


def hybrid_graph(
    nodes: Sequence[str],
    text: Mapping[Pair, float],
    direct: Mapping[Pair, float],
    coupling: Mapping[Pair, float],
) -> nx.Graph:
    """Build the balanced hybrid graph, nodes and edges inserted in sorted order."""
    weights: Dict[Pair, float] = defaultdict(float)
    for layer_weight, layer in (
        (TEXT_LAYER_WEIGHT, text),
        (DIRECT_LAYER_WEIGHT, direct),
        (COUPLING_LAYER_WEIGHT, coupling),
    ):
        for key, value in layer.items():
            weights[key] += layer_weight * value
    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes))
    for left, right in sorted(weights):
        if left in graph and right in graph:
            graph.add_edge(left, right, weight=float(weights[(left, right)]))
    return graph


def walk_graph(nodes: Sequence[str], index: CitationIndex) -> nx.Graph:
    """Build the citation graph the personalised PageRank walks.

    Direct citations (1.0), plus Salton bibliographic coupling and Salton
    co-citation for pairs with at least two shared works, capped at 1. The
    report's leave-one-out test measured this graph (section 4.5).
    """
    members = sorted(set(nodes))
    weights: Dict[Pair, float] = defaultdict(float)
    for key in direct_edges(members, index):
        weights[key] += 1.0
    for (left, right), count in shared_reference_counts(members, index).items():
        weights[(left, right)] += salton(
            count, len(index.refs_of.get(left, ())), len(index.refs_of.get(right, ()))
        )
    member_set = set(members)
    for (left, right), count in cocitation_counts(members, index).items():
        left_citers = len(index.cited_by.get(left, set()) & member_set)
        right_citers = len(index.cited_by.get(right, set()) & member_set)
        weights[(left, right)] += salton(count, left_citers, right_citers)
    graph = nx.Graph()
    graph.add_nodes_from(members)
    for key in sorted(weights):
        graph.add_edge(key[0], key[1], weight=float(min(1.0, weights[key])))
    return graph


# ================================================================ communities


def leiden_partition(graph: nx.Graph, resolution: float) -> Dict[str, int]:
    """Leiden (modularity, fixed seed) via graspologic-native.

    Community ids are canonical: largest first, ties by smallest member id.
    Nodes with no edge are left out (unclustered).
    """
    import graspologic_native as gn

    edges = sorted(
        (left, right, float(data.get("weight", 1.0)))
        for left, right, data in graph.edges(data=True)
    )
    edges = [(min(a, b), max(a, b), w) for a, b, w in edges]
    edges.sort()
    if not edges:
        return {}
    _, membership = gn.leiden(
        edges,
        resolution=resolution,
        seed=LEIDEN_SEED,
        use_modularity=True,
        iterations=LEIDEN_ITERATIONS,
    )
    return canonical_communities(membership)


def canonical_communities(membership: Mapping[str, int]) -> Dict[str, int]:
    """Relabel: biggest community 0, ties broken by the smallest member id."""
    groups: Dict[int, List[str]] = defaultdict(list)
    for node, community in membership.items():
        groups[community].append(node)
    ordered = sorted(groups.values(), key=lambda members: (-len(members), min(members)))
    return {
        node: position for position, members in enumerate(ordered) for node in members
    }


_TOKEN = re.compile(r"[a-z][a-z0-9\-]+")
_STOPWORDS = frozenset(
    """a about above after again against all also an and any are as at based be been
    being between both but by can could do does for from further has have how however
    in into is it its may more most new not of on or our over such than that the their
    them then there these they this those through to towards toward under using via
    was we what when where which while who why will with within without you your
    approach approaches paper study analysis method methods""".split()
)


def title_tokens(title: str) -> List[str]:
    """Lowercase content words of a title."""
    return [
        token
        for token in _TOKEN.findall((title or "").casefold())
        if len(token) >= COMMUNITY_TERM_MIN_LENGTH and token not in _STOPWORDS
    ]


def community_terms(
    members_by_community: Mapping[int, Sequence[str]], titles: Mapping[str, str]
) -> Dict[int, List[str]]:
    """c-TF-IDF over titles: terms frequent in a community and rare elsewhere."""
    term_counts: Dict[int, Counter] = {}
    total = Counter()
    for community, members in members_by_community.items():
        counts = Counter(
            token for node in members for token in title_tokens(titles.get(node, ""))
        )
        term_counts[community] = counts
        total.update(counts)
    if not term_counts:
        return {}
    average_words = sum(sum(counts.values()) for counts in term_counts.values()) / len(
        term_counts
    )
    out: Dict[int, List[str]] = {}
    for community, counts in term_counts.items():
        size = sum(counts.values()) or 1
        scored = [
            (counts[term] / size * math.log(1 + average_words / total[term]), term)
            for term in counts
        ]
        out[community] = [
            term
            for _, term in sorted(scored, key=lambda item: (-item[0], item[1]))[
                :COMMUNITY_TERMS
            ]
        ]
    return out


# ==================================================================== lineage


@dataclass
class Lineage:
    """SPC weights and the key-node paths through the pillar's library works."""

    kind: str  # "key_node", "main_path" or "none"
    arc_spc: Dict[Pair, float]  # (cited, citing) -> SPC / max SPC
    paths: List[Dict[str, Any]]
    dag_nodes: int
    dag_arcs: int
    dropped_backwards: int
    dropped_cycles: int


def citation_dag(
    nodes: Sequence[str], index: CitationIndex, years: Mapping[str, Optional[int]]
) -> Tuple[nx.DiGraph, int, int]:
    """Knowledge-flow DAG (cited -> citing) over the scope.

    Arcs whose citing work is dated before the cited one are dropped; any cycle
    left (same-year mutual citations) loses its closing arc. Isolated works are
    removed. Insertion is sorted, so the result does not depend on input order.
    """
    members = set(nodes)
    dag = nx.DiGraph()
    dag.add_nodes_from(sorted(members))
    backwards = 0
    for citing, cited in sorted(index.edges):
        if citing not in members or cited not in members:
            continue
        cited_year, citing_year = years.get(cited), years.get(citing)
        if (
            cited_year is not None
            and citing_year is not None
            and citing_year < cited_year
        ):
            backwards += 1
            continue
        dag.add_edge(cited, citing)
    cycles = 0
    while True:
        try:
            cycle = nx.find_cycle(dag)
        except nx.NetworkXNoCycle:
            break
        dag.remove_edge(*cycle[-1][:2])
        cycles += 1
    dag.remove_nodes_from([node for node in sorted(dag) if dag.degree(node) == 0])
    return dag, backwards, cycles


def search_path_counts(dag: nx.DiGraph) -> Dict[Pair, int]:
    """Search Path Count of every arc.

    Sources that reach its tail times sinks its head reaches (Batagelj 2003).
    """
    order = list(nx.lexicographical_topological_sort(dag))
    from_sources = {node: (1 if dag.in_degree(node) == 0 else 0) for node in order}
    for node in order:
        for pred in sorted(dag.predecessors(node)):
            from_sources[node] += from_sources[pred]
    to_sinks = {node: (1 if dag.out_degree(node) == 0 else 0) for node in order}
    for node in reversed(order):
        for succ in sorted(dag.successors(node)):
            to_sinks[node] += to_sinks[succ]
    return {
        (tail, head): from_sources[tail] * to_sinks[head]
        for tail, head in sorted(dag.edges)
    }


def _best_paths(
    dag: nx.DiGraph, weight: Mapping[Pair, float], marked: Set[str]
) -> List[str]:
    """Source-to-sink path through the most ``marked`` nodes, then the most weight.

    Forward and backward dynamic programmes over the topological order; the
    path through the best node is their join. Ties break by node id.
    """
    order = list(nx.lexicographical_topological_sort(dag))
    forward: Dict[str, Tuple[int, float]] = {}
    back_ptr: Dict[str, Optional[str]] = {}
    for node in order:
        mark = 1 if node in marked else 0
        best: Tuple[int, float] = (mark, 0.0)
        best_pred: Optional[str] = None
        for pred in sorted(dag.predecessors(node)):
            count, total = forward[pred]
            candidate = (count + mark, round(total + weight[(pred, node)], 12))
            if candidate > best:
                best, best_pred = candidate, pred
        forward[node], back_ptr[node] = best, best_pred
    backward: Dict[str, Tuple[int, float]] = {}
    next_ptr: Dict[str, Optional[str]] = {}
    for node in reversed(order):
        mark = 1 if node in marked else 0
        best = (mark, 0.0)
        best_next: Optional[str] = None
        for succ in sorted(dag.successors(node)):
            count, total = backward[succ]
            candidate = (count + mark, round(total + weight[(node, succ)], 12))
            if candidate > best:
                best, best_next = candidate, succ
        backward[node], next_ptr[node] = best, best_next

    def through(node: str) -> Tuple[int, float]:
        mark = 1 if node in marked else 0
        return (
            forward[node][0] + backward[node][0] - mark,
            round(forward[node][1] + backward[node][1], 12),
        )

    pivot = min(
        order, key=lambda node: (tuple(-value for value in through(node)), node)
    )
    head: List[str] = []
    cursor: Optional[str] = pivot
    while cursor is not None:
        head.append(cursor)
        cursor = back_ptr[cursor]
    head.reverse()
    cursor = next_ptr[pivot]
    while cursor is not None:
        head.append(cursor)
        cursor = next_ptr[cursor]
    return head


def lineage(
    nodes: Sequence[str],
    index: CitationIndex,
    years: Mapping[str, Optional[int]],
    library: Set[str],
) -> Lineage:
    """SPC weights for every arc, and up to ``MAX_LINEAGE_PATHS`` key-node paths.

    With no library work in the DAG the single global main path is returned
    instead (``kind="main_path"``).
    """
    dag, backwards, cycles = citation_dag(nodes, index, years)
    if dag.number_of_edges() == 0:
        return Lineage("none", {}, [], dag.number_of_nodes(), 0, backwards, cycles)
    spc = search_path_counts(dag)
    top = max(spc.values()) or 1
    weight = {key: value / top for key, value in spc.items()}
    remaining = {node for node in library if node in dag}
    kind = "key_node" if remaining else "main_path"
    paths: List[Dict[str, Any]] = []
    while len(paths) < MAX_LINEAGE_PATHS:
        path = _best_paths(dag, weight, remaining)
        covered = [node for node in path if node in remaining]
        if kind == "key_node" and not covered:
            break
        paths.append(_path_record(path, weight, library))
        remaining -= set(covered)
        if kind == "main_path" or not remaining:
            break
    return Lineage(
        kind,
        {key: round(value, SCORE_DECIMALS) for key, value in weight.items()},
        paths,
        dag.number_of_nodes(),
        dag.number_of_edges(),
        backwards,
        cycles,
    )


def _path_record(
    path: Sequence[str], weight: Mapping[Pair, float], library: Set[str]
) -> Dict[str, Any]:
    arcs = [
        weight[(path[position], path[position + 1])]
        for position in range(len(path) - 1)
    ]
    node_weight = {}
    for position, node in enumerate(path):
        incident = []
        if position > 0:
            incident.append(arcs[position - 1])
        if position < len(arcs):
            incident.append(arcs[position])
        node_weight[node] = max(incident) if incident else 0.0
    return {
        "nodes": list(path),
        "library": [node for node in path if node in library],
        "condensed": condense_path(path, node_weight, library),
        "spc_sum": round(sum(arcs), SCORE_DECIMALS),
    }


def condense_path(
    path: Sequence[str], node_weight: Mapping[str, float], library: Set[str]
) -> List[str]:
    """Keep library works and the highest-SPC works in each gap, in path order."""
    anchors = [position for position, node in enumerate(path) if node in library]
    keep: Set[int] = set(anchors)
    bounds = [-1] + anchors + [len(path)]
    for start, end in zip(bounds, bounds[1:], strict=False):
        gap = list(range(start + 1, end))
        best = sorted(
            gap, key=lambda position: (-node_weight[path[position]], position)
        )
        keep.update(best[:LINEAGE_CONDENSED_PER_GAP])
    return [path[position] for position in sorted(keep)]


# ================================================================= importance


def local_citation_scores(nodes: Sequence[str], index: CitationIndex) -> Dict[str, int]:
    """Citations each work receives from inside the map (HistCite's LCS)."""
    members = set(nodes)
    return {
        node: len(index.cited_by.get(node, set()) & members) for node in sorted(members)
    }


def cohort_percentiles(
    nodes: Sequence[str], works: Mapping[str, Work]
) -> Dict[str, Optional[float]]:
    """Citation-count percentile of each work among works of its year in the map.

    Ties count half. A cohort smaller than ``COHORT_MIN_SIZE`` widens a year at
    a time up to ``COHORT_MAX_WINDOW``. No year or no count: None.
    """
    dated = {
        node: (works[node].year, works[node].citation_count)
        for node in nodes
        if node in works
        and works[node].year is not None
        and works[node].citation_count is not None
    }
    by_year: Dict[int, List[int]] = defaultdict(list)
    for year, count in dated.values():
        by_year[year].append(count)
    out: Dict[str, Optional[float]] = {node: None for node in sorted(nodes)}
    for node, (year, count) in dated.items():
        cohort: List[int] = []
        for window in range(COHORT_MAX_WINDOW + 1):
            cohort = [
                value
                for cohort_year in range(year - window, year + window + 1)
                for value in by_year.get(cohort_year, [])
            ]
            if len(cohort) >= COHORT_MIN_SIZE:
                break
        less = sum(1 for value in cohort if value < count)
        equal = sum(1 for value in cohort if value == count)
        out[node] = round(100.0 * (less + 0.5 * equal) / len(cohort), 1)
    return out


def betweenness(graph: nx.Graph) -> Dict[str, float]:
    """Unweighted betweenness, sampled above ``BETWEENNESS_SAMPLES`` nodes."""
    if graph.number_of_nodes() < 3 or graph.number_of_edges() == 0:
        return {node: 0.0 for node in graph}
    sample = (
        None if graph.number_of_nodes() <= BETWEENNESS_SAMPLES else BETWEENNESS_SAMPLES
    )
    scores = nx.betweenness_centrality(
        graph, k=sample, seed=BETWEENNESS_SEED, normalized=True
    )
    return {node: round(float(scores[node]), SCORE_DECIMALS) for node in sorted(scores)}


# ===================================================================== layout


def layout_key(graph: nx.Graph) -> str:
    """Hash of what the layout depends on: the graph, its weights, the constants."""
    edges = sorted(
        (left, right, round(float(data.get("weight", 1.0)), SCORE_DECIMALS))
        for left, right, data in graph.edges(data=True)
    )
    edges = sorted((min(a, b), max(a, b), w) for a, b, w in edges)
    return sha256_of({"nodes": sorted(graph), "edges": edges, "params": params_hash()})


def force_layout(
    graph: nx.Graph,
    previous: Mapping[str, Tuple[float, float]],
    *,
    reuse: bool = False,
) -> Dict[str, Tuple[float, float]]:
    """Spring layout in [-1, 1], warm-started from ``previous`` when it overlaps.

    ``reuse`` (the graph is the one ``previous`` was laid out for) returns the
    previous positions unchanged, so an unchanged map keeps its hash: a warm
    start alone still moves every point a little on each rebuild. A new work
    starts at the mean of its already-placed neighbours plus a small seeded
    jitter, so it lands near what it is connected to.
    """
    nodes = sorted(graph)
    if not nodes:
        return {}
    if reuse and all(node in previous for node in nodes):
        return {
            node: (
                round(float(previous[node][0]), LAYOUT_DECIMALS),
                round(float(previous[node][1]), LAYOUT_DECIMALS),
            )
            for node in nodes
        }
    if len(nodes) == 1:
        return {nodes[0]: (0.0, 0.0)}
    placed = {node: previous[node] for node in nodes if node in previous}
    rng = np.random.default_rng(LAYOUT_SEED)
    initial: Optional[Dict[str, np.ndarray]] = None
    iterations = LAYOUT_ITERATIONS
    if placed:
        iterations = WARM_LAYOUT_ITERATIONS
        initial = {
            node: np.asarray(placed[node], dtype=np.float64) for node in sorted(placed)
        }
        for node in nodes:
            jitter = rng.uniform(-LAYOUT_JITTER, LAYOUT_JITTER, 2)
            if node in initial:
                continue
            neighbours = [
                initial[other]
                for other in sorted(graph.neighbors(node))
                if other in initial
            ]
            if neighbours:
                initial[node] = np.mean(neighbours, axis=0) + jitter
            else:
                initial[node] = rng.uniform(-1.0, 1.0, 2)
    positions = nx.spring_layout(
        graph,
        pos=initial,
        iterations=iterations,
        seed=LAYOUT_SEED,
        weight="weight",
    )
    return {
        node: (
            round(float(positions[node][0]), LAYOUT_DECIMALS),
            round(float(positions[node][1]), LAYOUT_DECIMALS),
        )
        for node in nodes
    }


# ================================================================ suggestions


def personalized_pagerank(graph: nx.Graph, seeds: Iterable[str]) -> Dict[str, float]:
    """Random walk with restart to ``seeds``. Empty when no seed is in the graph."""
    present = sorted(seed for seed in set(seeds) if seed in graph)
    if not present or graph.number_of_edges() == 0:
        return {}
    restart = {seed: 1.0 / len(present) for seed in present}
    scores = nx.pagerank(
        graph,
        alpha=1.0 - PPR_RESTART,
        personalization=restart,
        dangling=restart,
        weight="weight",
        tol=PPR_TOLERANCE,
        max_iter=PPR_MAX_ITERATIONS,
    )
    return {node: float(scores[node]) for node in sorted(scores)}


def max_cosine(
    space: VectorSpace, candidates: Sequence[str], seeds: Sequence[str]
) -> Dict[str, float]:
    """Best cosine from each candidate to any seed.

    Candidates without a vector are absent from the result.
    """
    seed_rows = [seed for seed in sorted(set(seeds)) if space.has(seed)]
    rows = [node for node in candidates if space.has(node)]
    if not seed_rows or not rows:
        return {}
    scores = space.matrix(rows) @ space.matrix(seed_rows).T
    best = scores.max(axis=1)
    return {node: float(best[position]) for position, node in enumerate(rows)}


def rank_positions(scores: Mapping[str, float], pool: Sequence[str]) -> Dict[str, int]:
    """1-based rank of every pool member by score, highest first.

    A member without a score ranks after every scored one. Ties break by id.
    """
    ordered = sorted(
        pool,
        key=lambda node: (0 if node in scores else 1, -scores.get(node, 0.0), node),
    )
    return {node: position + 1 for position, node in enumerate(ordered)}


def rrf_fuse(
    *rankings: Mapping[str, int],
    k: int = RRF_K,
    only: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Reciprocal-rank fusion: sum of 1 / (k + rank) over the rankings.

    ``only`` limits the result to those nodes (the ones some scorer scored).
    """
    pool = set().union(*[set(ranking) for ranking in rankings])
    if only is not None:
        pool &= only
    return {
        node: sum(1.0 / (k + ranking[node]) for ranking in rankings if node in ranking)
        for node in sorted(pool)
    }


@dataclass
class Ranking:
    """One scorer's view of a pool."""

    rrf: Dict[str, float]
    rank: Dict[str, int]
    ppr_rank: Dict[str, int]
    text_rank: Dict[str, int]
    text: Dict[str, float]


def rank_pool(
    pool: Sequence[str],
    seeds: Sequence[str],
    walk: nx.Graph,
    space: VectorSpace,
) -> Ranking:
    """RRF of PPR-from-seeds over ``walk`` and bge max-cosine to the seeds."""
    ppr = personalized_pagerank(walk, seeds)
    ppr_scores = {node: ppr[node] for node in pool if node in ppr}
    text = max_cosine(space, pool, seeds)
    ppr_rank = rank_positions(ppr_scores, pool)
    text_rank = rank_positions(text, pool)
    # A node neither scorer could score (no vector, not on the citation graph)
    # is left unranked rather than given two last places.
    scored = set(ppr_scores) | set(text)
    if ppr_scores and text:
        fused = rrf_fuse(ppr_rank, text_rank, only=scored)
    elif ppr_scores:
        fused = rrf_fuse(ppr_rank, only=scored)
    elif text:
        fused = rrf_fuse(text_rank, only=scored)
    else:
        fused = {}
    return Ranking(
        rrf=fused,
        rank=rank_positions(fused, pool),
        ppr_rank=ppr_rank,
        text_rank=text_rank,
        text=text,
    )


def normalised_title(title: str) -> Set[str]:
    """Title as a set of lowercase word tokens (stopwords kept)."""
    return set(re.findall(r"[a-z0-9]+", (title or "").casefold()))


def titles_match(left: str, right: str) -> bool:
    """Token Jaccard of the two titles reaches ``DUPLICATE_TITLE_JACCARD``."""
    a, b = normalised_title(left), normalised_title(right)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= DUPLICATE_TITLE_JACCARD


# ===================================================================== build


@dataclass
class _Context:
    inputs: MapInputs
    index: CitationIndex
    scope: Scope
    library_nodes: Dict[str, List[LibraryPaper]]
    pillar_seeds: List[str]
    titles: Dict[str, str]
    years: Dict[str, Optional[int]]
    space: VectorSpace
    sorted_cosines: np.ndarray
    warnings: List[str]


def _node_title(
    node: str,
    inputs: MapInputs,
    library_nodes: Mapping[str, List[LibraryPaper]],
    queue: Mapping[str, QueueItem],
) -> str:
    if node in inputs.works and inputs.works[node].title:
        return inputs.works[node].title
    for paper in library_nodes.get(node, []):
        if paper.title:
            return paper.title
    if node in queue:
        return queue[node].title
    return ""


def short_label(
    node: str,
    inputs: MapInputs,
    library_nodes: Mapping[str, List[LibraryPaper]],
    titles: Mapping[str, str],
) -> str:
    """Label a work "Hewitt 2019" from its first author and year, else by title."""
    work = inputs.works.get(node)
    authors: Tuple[str, ...] = work.authors if work else ()
    year = work.year if work else None
    for paper in library_nodes.get(node, []):
        authors = authors or paper.authors
        year = year if year is not None else paper.year
    if authors:
        surname = authors[0].split()[-1] if authors[0].split() else authors[0]
        return f"{surname} {year}" if year else surname
    title = titles.get(node, "") or node
    title = title if len(title) <= 60 else title[:57].rstrip() + "..."
    return f"{title} ({year})" if year else title


def build_map(  # noqa: C901
    inputs: MapInputs,
    *,
    progress: Optional[Progress] = None,
    notes: Sequence[str] = (),
) -> MapResult:
    """Compute one pillar's snapshot.

    ``progress(stage, status, detail)`` is optional and may raise to stop the
    build between stages (cancellation). ``notes`` are the loader's warnings;
    they lead the snapshot's own.
    """

    def report(stage: str, status: str, detail: Optional[str] = None) -> None:
        if progress is not None:
            progress(stage, status, detail)

    warnings: List[str] = list(notes)
    pillar_id = inputs.pillar.id
    index = CitationIndex(inputs.citations)
    library_nodes: Dict[str, List[LibraryPaper]] = defaultdict(list)
    for paper in sorted(
        inputs.library, key=lambda row: (row.node_id, row.pillar_id, row.paper_id)
    ):
        library_nodes[paper.node_id].append(paper)
    all_library = set(library_nodes)
    queue_by_node = {
        item.node_id: item
        for item in sorted(inputs.queue, key=lambda row: row.queue_id)
    }

    # -- graph ---------------------------------------------------------------
    report(STAGE_GRAPH, STATUS_RUNNING, "Choosing the works on this map")
    scope = select_scope(pillar_id, inputs.library, inputs.works, index)
    pillar_seeds = scope.seeds
    nodes = scope.nodes
    members = set(nodes)
    space = VectorSpace(inputs.vectors)
    if not space.ids:
        warnings.append(
            "No title+abstract vectors: text edges and text scores are skipped."
        )
    scope_vectors = [node for node in nodes if space.has(node)]
    sorted_cosines = pairwise_cosines(space.matrix(scope_vectors))
    titles = {
        node: _node_title(node, inputs, library_nodes, queue_by_node)
        for node in sorted(set(nodes) | set(queue_by_node))
    }
    years: Dict[str, Optional[int]] = {}
    for node in nodes:
        work = inputs.works.get(node)
        year = work.year if work else None
        if year is None:
            year = next(
                (
                    paper.year
                    for paper in library_nodes.get(node, [])
                    if paper.year is not None
                ),
                None,
            )
        years[node] = year
    walk_nodes = sorted(
        set(nodes) | {node for node in queue_by_node if node in inputs.works}
    )
    shared = shared_reference_counts(walk_nodes, index)
    text = text_knn_edges(space, nodes)
    direct = direct_edges(nodes, index)
    coupling = coupling_edges(shared, index, nodes)
    graph = hybrid_graph(nodes, text, direct, coupling)
    walk = walk_graph(walk_nodes, index)
    report(
        STAGE_GRAPH,
        STATUS_COMPLETED,
        f"{len(nodes)} works ({len(pillar_seeds)} yours), "
        f"{graph.number_of_edges()} links: "
        f"{len(text)} text, {len(direct)} citation, {len(coupling)} shared-reference",
    )
    ctx = _Context(
        inputs,
        index,
        scope,
        library_nodes,
        pillar_seeds,
        titles,
        years,
        space,
        sorted_cosines,
        warnings,
    )

    # -- communities ---------------------------------------------------------
    report(STAGE_COMMUNITIES, STATUS_RUNNING, "Finding communities")
    clustered = len(nodes) >= MIN_CLUSTER_SCOPE and graph.number_of_edges() > 0
    if clustered:
        topics = leiden_partition(graph, TOPIC_RESOLUTION)
        regions = leiden_partition(graph, REGION_RESOLUTION)
    else:
        topics = {node: 0 for node in nodes}
        regions = {node: 0 for node in nodes}
        warnings.append(
            f"{len(nodes)} works is under {MIN_CLUSTER_SCOPE}: "
            "the pillar is shown as one community."
        )
    local = local_citation_scores(nodes, index)
    communities = _communities_payload(ctx, topics, regions, local)
    report(
        STAGE_COMMUNITIES,
        STATUS_COMPLETED,
        f"{len(communities['topics'])} topics in {len(communities['regions'])} regions"
        + ("" if clustered else " (too few works to cluster)"),
    )

    # -- lineage -------------------------------------------------------------
    report(STAGE_LINEAGE, STATUS_RUNNING, "Tracing the main line of development")
    line = lineage(nodes, index, years, set(pillar_seeds))
    report(STAGE_LINEAGE, STATUS_COMPLETED, _lineage_detail(line))

    # -- layout --------------------------------------------------------------
    report(STAGE_LAYOUT, STATUS_RUNNING, "Placing the works")
    previous = {node: tuple(value) for node, value in inputs.previous_positions.items()}
    key = layout_key(graph)
    reused = key == inputs.previous_layout_key and all(
        node in previous for node in nodes
    )
    positions = force_layout(graph, previous, reuse=reused)
    warm = sum(1 for node in nodes if node in previous)
    central = betweenness(graph)
    if reused:
        layout_detail = f"{len(positions)} placed, unchanged since the last map"
    else:
        layout_detail = f"{len(positions)} placed" + (
            f", {warm} from the last map" if warm else ""
        )
    report(STAGE_LAYOUT, STATUS_COMPLETED, layout_detail)

    # -- suggestions, queue, insights, evidence -----------------------------
    report(STAGE_SUGGEST, STATUS_RUNNING, "Ranking suggestions and the queue")
    duplicates = _duplicates(
        ctx,
        [
            node
            for node in sorted(set(nodes) | set(queue_by_node))
            if node not in all_library
        ],
    )
    pool = [
        node for node in nodes if node not in all_library and node not in duplicates
    ]
    suggestions = _suggestions(ctx, pool, topics, walk, clustered)
    queue_rows = _queue_verdicts(
        ctx, pool, walk, duplicates, queue_by_node, all_library
    )
    insights = _insights(ctx, graph, topics, communities, central, local)
    focus = set(pillar_seeds)
    focus.update(item["id"] for item in suggestions["pillar"])
    for group in suggestions["communities"]:
        focus.update(item["id"] for item in group["items"])
    focus.update(row["node"] for row in queue_rows if row["node"] in graph)
    evidence = _evidence(ctx, graph, topics, sorted(focus))
    unscored = sum(1 for row in queue_rows if row["verdict"] == VERDICT_UNSCORED)
    if unscored:
        warnings.append(
            f"{unscored} queued paper(s) have no vector and no known work, "
            "so they are unscored."
        )
    if inputs.queue and not inputs.anchors:
        warnings.append(
            "The pillar's goal is not embedded: queue papers far from yours "
            "are not split into on-topic and off-topic."
        )
    report(
        STAGE_SUGGEST,
        STATUS_COMPLETED,
        f"{len(suggestions['pillar'])} suggestions, "
        f"{len(queue_rows)} queued papers checked, "
        f"{len(insights)} insights",
    )

    nodes_payload = _nodes_payload(
        ctx, topics, regions, positions, central, local, line, duplicates, queue_by_node
    )
    citations_payload = [
        {
            "citing": citing,
            "cited": cited,
            "spc": line.arc_spc.get((cited, citing), 0.0),
            "influential": bool(edge.is_influential),
            "intents": sorted(edge.intents),
        }
        for (citing, cited), edge in sorted(index.edges.items())
        if citing in members and cited in members
    ]
    stats = {
        "works": len(nodes),
        "library_works": len(pillar_seeds),
        "external_works": len([node for node in nodes if node not in all_library]),
        "citations": len(citations_payload),
        "text_edges": len(text),
        "coupling_edges": len(coupling),
        "hybrid_edges": graph.number_of_edges(),
        "topics": len(communities["topics"]),
        "regions": len(communities["regions"]),
        "lineage_paths": len(line.paths),
        "suggestions": len(suggestions["pillar"]),
        "queue": len(queue_rows),
        "insights": len(insights),
        "evidence": len(evidence),
        "vectors": len(scope_vectors),
        "unknown_ids": scope.unknown_ids,
        "capped": scope.capped,
    }
    phash = params_hash()
    payload = {
        "version": MAP_SNAPSHOT_VERSION,
        "scope": pillar_id,
        "pillar": {"id": pillar_id, "name": inputs.pillar.name},
        "model_id": inputs.model_id,
        "params_hash": phash,
        "clustered": clustered,
        # Only the key: whether this build reused or warm-started the layout
        # is about the build, not the map, and must not change the hash.
        "layout": {"key": key},
        "stats": stats,
        "nodes": nodes_payload,
        "citations": citations_payload,
        "communities": communities,
        "lineage": {
            "kind": line.kind,
            "paths": line.paths,
            "dag_nodes": line.dag_nodes,
            "dag_arcs": line.dag_arcs,
            "note": "A likely line of development through what this map shows. "
            "The newest end moves when citers are refetched.",
        },
        "suggestions": suggestions,
        "queue": queue_rows,
        "insights": insights,
        "evidence": evidence,
        "warnings": list(warnings),
    }
    return MapResult(
        scope=pillar_id,
        payload=payload,
        payload_hash=sha256_of(payload),
        params_hash=phash,
        stats=stats,
        warnings=list(warnings),
        layout={"warm_started": warm, "reused": reused},
    )


def _lineage_detail(line: Lineage) -> str:
    if line.kind == "none":
        return "No citation links between dated works, so no line of development"
    longest = max((len(path["nodes"]) for path in line.paths), default=0)
    covered = sum(len(path["library"]) for path in line.paths)
    if line.kind == "main_path":
        return f"Main path of {longest} works (none of yours are on the citation graph)"
    return (
        f"{len(line.paths)} path(s) through {covered} of your works, "
        f"longest {longest} works"
    )


# -------------------------------------------------------------- communities


def _communities_payload(
    ctx: _Context,
    topics: Mapping[str, int],
    regions: Mapping[str, int],
    local: Mapping[str, int],
) -> Dict[str, List[Dict[str, Any]]]:
    library = set(ctx.library_nodes)

    def level(
        membership: Mapping[str, int],
    ) -> Tuple[Dict[int, List[str]], List[Dict[str, Any]]]:
        members: Dict[int, List[str]] = defaultdict(list)
        for node in sorted(membership):
            members[membership[node]].append(node)
        terms = community_terms(members, ctx.titles)
        rows = []
        for community in sorted(members):
            group = members[community]
            group_set = set(group)
            inside = {
                node: len(ctx.index.cited_by.get(node, set()) & group_set)
                for node in group
            }
            representative = min(
                group,
                key=lambda node: (
                    -inside[node],
                    -local.get(node, 0),
                    -(
                        (ctx.inputs.works[node].citation_count or 0)
                        if node in ctx.inputs.works
                        else 0
                    ),
                    node,
                ),
            )
            rows.append(
                {
                    "id": community,
                    "size": len(group),
                    "library": [node for node in group if node in library],
                    "pillar_library": [
                        node for node in group if node in set(ctx.pillar_seeds)
                    ],
                    "representative": representative,
                    "terms": terms.get(community, []),
                    "member_hash": sha256_of(group)[:16],
                }
            )
        return members, rows

    topic_members, topic_rows = level(topics)
    _, region_rows = level(regions)
    for row in topic_rows:
        votes = Counter(
            regions[node] for node in topic_members[row["id"]] if node in regions
        )
        row["region"] = (
            min(votes, key=lambda region: (-votes[region], region)) if votes else None
        )
    return {"regions": region_rows, "topics": topic_rows}


# --------------------------------------------------------------- duplicates


def _duplicate_cut(ctx: _Context) -> float:
    if ctx.sorted_cosines.size and len(ctx.space.ids) >= MIN_PERCENTILE_SCOPE:
        return float(np.percentile(ctx.sorted_cosines, DUPLICATE_PERCENTILE))
    return DUPLICATE_FALLBACK_COSINE


def _duplicates(ctx: _Context, candidates: Sequence[str]) -> Dict[str, str]:
    """Candidate -> the library work it duplicates (vector and title both agree)."""
    library = [node for node in sorted(ctx.library_nodes) if ctx.space.has(node)]
    rows = [node for node in candidates if ctx.space.has(node)]
    if not library or not rows:
        return {}
    cut = _duplicate_cut(ctx)
    scores = ctx.space.matrix(rows) @ ctx.space.matrix(library).T
    out: Dict[str, str] = {}
    for position, node in enumerate(rows):
        order = np.argsort(-scores[position], kind="stable")
        for column in order:
            if scores[position, int(column)] < cut:
                break
            other = library[int(column)]
            if titles_match(
                ctx.titles.get(node, ""),
                ctx.titles.get(other, "") or _library_title(ctx, other),
            ):
                out[node] = other
                break
    return out


def _library_title(ctx: _Context, node: str) -> str:
    for paper in ctx.library_nodes.get(node, []):
        if paper.title:
            return paper.title
    return ""


# -------------------------------------------------------------- suggestions


def _direction(ctx: _Context, node: str, seeds: Sequence[str]) -> str:
    """Classify a suggestion as foundation, frontier or related to the seeds.

    Foundation: cited by your works, or older. Frontier: cites them, or newer.
    """
    if any(node in ctx.index.refs_of.get(seed, ()) for seed in seeds):
        return "foundation"
    if any(seed in ctx.index.refs_of.get(node, ()) for seed in seeds):
        return "frontier"
    seed_years = [
        ctx.years.get(seed) for seed in seeds if ctx.years.get(seed) is not None
    ]
    year = ctx.years.get(node)
    if year is None or not seed_years:
        return "related"
    middle = statistics.median(seed_years)
    if year < middle:
        return "foundation"
    if year > middle:
        return "frontier"
    return "related"


def _reasons(ctx: _Context, node: str, seeds: Sequence[str]) -> Dict[str, Any]:
    """Count what links a suggestion to the seeds (Local Citation Network style)."""
    cited_by_yours = sorted(
        seed for seed in seeds if node in ctx.index.refs_of.get(seed, ())
    )
    cites_yours = sorted(
        seed for seed in seeds if seed in ctx.index.refs_of.get(node, ())
    )
    best_shared, best_seed = 0, None
    for seed in sorted(seeds):
        count = len(
            ctx.index.refs_of.get(node, set()) & ctx.index.refs_of.get(seed, set())
        )
        if count > best_shared:
            best_shared, best_seed = count, seed
    return {
        "cited_by_yours": cited_by_yours,
        "cites_yours": cites_yours,
        "shared_references": best_shared,
        "shared_with": best_seed,
    }


def _ranked_items(
    ctx: _Context,
    ranking: Ranking,
    pool: Sequence[str],
    seeds: Sequence[str],
    limit: int,
    topics: Mapping[str, int],
) -> List[Dict[str, Any]]:
    ordered = sorted(pool, key=lambda node: ranking.rank[node])[:limit]
    items = []
    for node in ordered:
        if node not in ranking.rrf:
            break
        text = ranking.text.get(node)
        items.append(
            {
                "id": node,
                "rank": ranking.rank[node],
                "score": round(ranking.rrf[node], SCORE_DECIMALS),
                "ppr_rank": ranking.ppr_rank[node],
                "text_rank": ranking.text_rank[node],
                "text_percentile": _round_or_none(
                    cosine_percentile(ctx.sorted_cosines, text)
                    if text is not None
                    else None
                ),
                "direction": _direction(ctx, node, seeds),
                "topic": topics.get(node),
                **_reasons(ctx, node, seeds),
            }
        )
    return items


def _round_or_none(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(value, digits)


def _suggestions(
    ctx: _Context,
    pool: Sequence[str],
    topics: Mapping[str, int],
    walk: nx.Graph,
    clustered: bool,
) -> Dict[str, Any]:
    seeds = ctx.pillar_seeds
    pillar_ranking = rank_pool(pool, seeds, walk, ctx.space)
    out: Dict[str, Any] = {
        "pillar": _ranked_items(
            ctx, pillar_ranking, pool, seeds, PILLAR_SUGGESTIONS, topics
        ),
        "communities": [],
    }
    groups: Dict[int, List[str]] = defaultdict(list)
    for seed in seeds:
        if seed in topics:
            groups[topics[seed]].append(seed)
    for community in sorted(groups):
        group_seeds = sorted(groups[community])
        ranking = rank_pool(pool, group_seeds, walk, ctx.space)
        out["communities"].append(
            {
                "topic": community,
                "seeds": group_seeds,
                "items": _ranked_items(
                    ctx, ranking, pool, group_seeds, SUGGESTIONS_PER_COMMUNITY, topics
                ),
            }
        )
    return out


# -------------------------------------------------------------------- queue


def anchor_similarity(
    space: VectorSpace, nodes: Sequence[str], anchors: Sequence[Sequence[float]]
) -> Dict[str, float]:
    """Best cosine from each node to the pillar's goal or one of its focus areas."""
    usable = [
        anchor for anchor in anchors if anchor is not None and len(anchor) == space.dims
    ]
    rows = [node for node in nodes if space.has(node)]
    if not usable or not rows:
        return {}
    matrix = np.asarray(usable, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    scores = space.matrix(rows) @ (matrix / norms).T
    best = scores.max(axis=1)
    return {node: float(best[position]) for position, node in enumerate(rows)}


def queue_verdict(
    *,
    in_library: bool,
    duplicate_of: Optional[str],
    scored: bool,
    rank_fraction: Optional[float],
    anchor: Optional[float],
    anchor_cut: Optional[float],
    has_seeds: bool,
) -> str:
    """Choose the badge for one queued paper (report section 4.9)."""
    if in_library:
        return VERDICT_ALREADY_YOURS
    if duplicate_of:
        return VERDICT_DUPLICATE
    if not scored:
        return VERDICT_UNSCORED
    if has_seeds and rank_fraction is not None:
        if rank_fraction <= STRONG_RANK_FRACTION:
            return VERDICT_STRONG
        if rank_fraction <= RELATED_RANK_FRACTION:
            return VERDICT_RELATED
    if anchor is None or anchor_cut is None:
        return VERDICT_FAR
    return VERDICT_NEW_DIRECTION if anchor >= anchor_cut else VERDICT_OFF_TOPIC


def _queue_verdicts(
    ctx: _Context,
    pool: Sequence[str],
    walk: nx.Graph,
    duplicates: Mapping[str, str],
    queue_by_node: Mapping[str, QueueItem],
    all_library: Set[str],
) -> List[Dict[str, Any]]:
    if not queue_by_node:
        return []
    seeds = ctx.pillar_seeds
    queued = [
        node
        for node in sorted(queue_by_node)
        if node not in all_library and node not in duplicates
    ]
    scored_pool = sorted(set(pool) | set(queued))
    ranking = rank_pool(scored_pool, seeds, walk, ctx.space)
    anchors = anchor_similarity(ctx.space, scored_pool, ctx.inputs.anchors)
    anchor_cut = (
        float(np.percentile(list(anchors.values()), ANCHOR_PERCENTILE))
        if anchors
        else None
    )
    rows = []
    for node in sorted(queue_by_node):
        item = queue_by_node[node]
        ranked = node in ranking.rrf
        # With no library works yet, the pillar's goal is the only anchor.
        scored = ranked or node in anchors
        fraction = ranking.rank[node] / len(scored_pool) if ranked else None
        verdict = queue_verdict(
            in_library=node in all_library,
            duplicate_of=duplicates.get(node),
            scored=scored,
            rank_fraction=fraction,
            anchor=anchors.get(node),
            anchor_cut=anchor_cut,
            has_seeds=bool(seeds),
        )
        rows.append(
            {
                "queue_id": item.queue_id,
                "paper_id": item.paper_id,
                "node": node,
                "title": item.title or ctx.titles.get(node, ""),
                "processed": item.processed,
                "verdict": verdict,
                "rank": ranking.rank.get(node) if ranked else None,
                "pool": len(scored_pool),
                "rank_fraction": _round_or_none(fraction, 4),
                "ppr_rank": ranking.ppr_rank.get(node) if node in walk else None,
                "text_rank": ranking.text_rank.get(node)
                if node in ranking.text
                else None,
                "anchor": _round_or_none(anchors.get(node), 4),
                "anchor_cut": _round_or_none(anchor_cut, 4),
                "duplicate_of": duplicates.get(node),
                **(_reasons(ctx, node, seeds) if node not in all_library else {}),
            }
        )
    order = {verdict: position for position, verdict in enumerate(VERDICT_ORDER)}
    rows.sort(
        key=lambda row: (order[row["verdict"]], row["rank"] or 10**9, row["queue_id"])
    )
    return rows


# ----------------------------------------------------------------- insights


def _insights(  # noqa: C901
    ctx: _Context,
    graph: nx.Graph,
    topics: Mapping[str, int],
    communities: Mapping[str, List[Dict[str, Any]]],
    central: Mapping[str, float],
    local: Mapping[str, int],
) -> List[Dict[str, Any]]:
    seeds = [seed for seed in ctx.pillar_seeds if seed in graph]
    label = lambda node: short_label(node, ctx.inputs, ctx.library_nodes, ctx.titles)  # noqa: E731
    out: List[Dict[str, Any]] = []

    pairs = [
        (data["weight"], left, right)
        for left, right, data in graph.edges(data=True)
        if left in set(seeds) and right in set(seeds)
    ]
    if pairs:
        weight, left, right = min(
            ((weight, *sorted((left, right))) for weight, left, right in pairs),
            key=lambda item: (-item[0], item[1], item[2]),
        )
        out.append(
            {
                "kind": "most_aligned",
                "text": (
                    "Your most closely connected pair: "
                    f"{label(left)} and {label(right)}."
                ),
                "nodes": [left, right],
                "weight": round(weight, SCORE_DECIMALS),
            }
        )

    if len(seeds) >= 3:
        mass = {}
        for seed in seeds:
            others = [other for other in seeds if other != seed]
            mass[seed] = personalized_pagerank(graph, others).get(seed, 0.0)
        furthest = min(seeds, key=lambda node: (mass[node], node))
        out.append(
            {
                "kind": "furthest",
                "text": (
                    f"{label(furthest)} is the furthest of your works "
                    "from the others on this map."
                ),
                "nodes": [furthest],
                "weight": round(mass[furthest], SCORE_DECIMALS),
            }
        )

    pillar_of = {
        node: sorted({paper.pillar_id for paper in papers})
        for node, papers in ctx.library_nodes.items()
    }
    bridges = []
    for seed in seeds:
        for other in sorted(graph.neighbors(seed)):
            other_pillars = [
                pillar
                for pillar in pillar_of.get(other, [])
                if pillar != ctx.inputs.pillar.id
            ]
            if other_pillars and ctx.inputs.pillar.id not in pillar_of.get(other, []):
                bridges.append(
                    (graph[seed][other]["weight"], seed, other, other_pillars[0])
                )
    for weight, seed, other, pillar in sorted(
        bridges, key=lambda item: (-item[0], item[1], item[2])
    )[:BRIDGE_COUNT]:
        out.append(
            {
                "kind": "cross_pillar",
                "text": (
                    f"{label(seed)} connects this pillar to {label(other)} in {pillar}."
                ),
                "nodes": [seed, other],
                "pillar": pillar,
                "weight": round(weight, SCORE_DECIMALS),
            }
        )

    ranked = sorted(
        (node for node in central if central[node] > 0),
        key=lambda node: (-central[node], node),
    )
    for node in ranked[:BRIDGE_COUNT]:
        neighbour_topics = sorted(
            {topics[other] for other in graph.neighbors(node) if other in topics}
        )
        if len(neighbour_topics) < 2:
            continue
        out.append(
            {
                "kind": "bridge",
                "text": (
                    f"{label(node)} links {len(neighbour_topics)} "
                    "communities of this map."
                ),
                "nodes": [node],
                "topics": neighbour_topics,
                "weight": central[node],
            }
        )

    gaps = [
        row
        for row in communities["topics"]
        if not row["library"] and row["size"] >= GAP_MIN_COMMUNITY_SIZE
    ]
    for row in sorted(gaps, key=lambda row: (-row["size"], row["id"]))[:GAP_COUNT]:
        terms = ", ".join(row["terms"][:3]) or "no common title terms"
        out.append(
            {
                "kind": "gap",
                "text": (
                    f"{row['size']} works gather around {label(row['representative'])} "
                    f"({terms}); none of them are yours."
                ),
                "nodes": [row["representative"]],
                "topic": row["id"],
            }
        )

    if len(ctx.scope.nodes) < MIN_CLUSTER_SCOPE:
        members = set(ctx.scope.nodes)
        links = sum(
            1
            for (citing, cited) in ctx.index.edges
            if citing in members and cited in members
        )
        out.append(
            {
                "kind": "sparse",
                "text": (
                    f"This map is sparse: {len(ctx.scope.nodes)} works "
                    f"and {links} citation links."
                ),
                "nodes": [],
            }
        )
    return out


# ----------------------------------------------------------------- evidence


def pair_evidence(
    ctx: _Context, left: str, right: str, topics: Mapping[str, int]
) -> Dict[str, Any]:
    """Everything stored that links two works, as counts and quotes."""
    out: Dict[str, Any] = {}
    for citing, cited in ((left, right), (right, left)):
        edge = ctx.index.edges.get((citing, cited))
        if edge is not None:
            out["cites"] = {
                "citing": citing,
                "cited": cited,
                "influential": bool(edge.is_influential),
                "intents": sorted(edge.intents),
                "context": (
                    edge.contexts[0][:CONTEXT_CHARS] if edge.contexts else None
                ),
            }
            break
    shared = len(
        ctx.index.refs_of.get(left, set()) & ctx.index.refs_of.get(right, set())
    )
    if shared:
        out["shared_references"] = shared
        out["shared_salton"] = round(
            salton(
                shared,
                len(ctx.index.refs_of.get(left, ())),
                len(ctx.index.refs_of.get(right, ())),
            ),
            SCORE_DECIMALS,
        )
    cocited = len(
        ctx.index.cited_by.get(left, set()) & ctx.index.cited_by.get(right, set())
    )
    if cocited:
        out["cocited"] = cocited
    if ctx.space.has(left) and ctx.space.has(right):
        cosine = float(ctx.space.vector(left) @ ctx.space.vector(right))
        percentile = cosine_percentile(ctx.sorted_cosines, cosine)
        if percentile is not None:
            out["text_percentile"] = round(percentile, 2)
    if left in topics and right in topics and topics[left] == topics[right]:
        out["same_topic"] = topics[left]
    return out


def evidence_sentence(evidence: Mapping[str, Any], label: Callable[[str], str]) -> str:
    """One plain sentence from ``pair_evidence``. No model: templates only."""
    parts: List[str] = []
    cites = evidence.get("cites")
    if cites:
        how = [INTENT_WORDS.get(intent, intent) for intent in cites["intents"]]
        clause = f"{label(cites['citing'])} cites {label(cites['cited'])}"
        tags = list(how)
        if cites["influential"]:
            tags.append("influential")
        if tags:
            clause += f" ({', '.join(tags)})"
        if cites.get("context"):
            clause += f": “{cites['context'].strip()}”"
        parts.append(clause)
    if evidence.get("shared_references"):
        count = evidence["shared_references"]
        parts.append(f"they share {count} reference{'s' if count != 1 else ''}")
    if evidence.get("cocited"):
        count = evidence["cocited"]
        parts.append(f"{count} paper{'s' if count != 1 else ''} cite both")
    percentile = evidence.get("text_percentile")
    if percentile is not None and percentile >= 90.0:
        top = max(100.0 - percentile, 0.01)
        parts.append(
            f"their abstracts are in the top {_percent(top)} "
            "most similar pairs on this map"
        )
    if "same_topic" in evidence:
        parts.append(f"both are in topic {evidence['same_topic']}")
    if not parts:
        return ""
    sentence = "; ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."


def _percent(value: float) -> str:
    return f"{value:.0f}%" if value >= 1 else f"{value:.1f}%"


def _evidence(
    ctx: _Context, graph: nx.Graph, topics: Mapping[str, int], focus: Sequence[str]
) -> List[Dict[str, Any]]:
    seeds = set(ctx.pillar_seeds)
    label = lambda node: short_label(node, ctx.inputs, ctx.library_nodes, ctx.titles)  # noqa: E731
    pairs: Set[Pair] = set()
    for node in focus:
        if node not in graph:
            continue
        neighbours = sorted(
            graph.neighbors(node),
            key=lambda other: (-graph[node][other]["weight"], other),
        )
        for other in neighbours[:EVIDENCE_NEIGHBOURS]:
            pairs.add(_pair(node, other))
        for other in seeds:
            if other != node and (
                ctx.index.linked(node, other) or graph.has_edge(node, other)
            ):
                pairs.add(_pair(node, other))
    out = []
    for left, right in sorted(pairs):
        evidence = pair_evidence(ctx, left, right, topics)
        sentence = evidence_sentence(evidence, label)
        if not sentence:
            continue
        weight = graph[left][right]["weight"] if graph.has_edge(left, right) else 0.0
        out.append(
            {
                "a": left,
                "b": right,
                "weight": round(float(weight), SCORE_DECIMALS),
                "evidence": evidence,
                "sentence": sentence,
            }
        )
    return out


# -------------------------------------------------------------------- nodes


def _nodes_payload(
    ctx: _Context,
    topics: Mapping[str, int],
    regions: Mapping[str, int],
    positions: Mapping[str, Tuple[float, float]],
    central: Mapping[str, float],
    local: Mapping[str, int],
    line: Lineage,
    duplicates: Mapping[str, str],
    queue_by_node: Mapping[str, QueueItem],
) -> List[Dict[str, Any]]:
    cohort = cohort_percentiles(ctx.scope.nodes, ctx.inputs.works)
    on_line = {node for path in line.paths for node in path["nodes"]}
    out = []
    for node in ctx.scope.nodes:
        work = ctx.inputs.works.get(node)
        papers = ctx.library_nodes.get(node, [])
        if papers:
            kind = (
                KIND_LIBRARY
                if work is not None or not node.startswith(DOCUMENT_PREFIX)
                else KIND_DOCUMENT
            )
        else:
            kind = KIND_EXTERNAL
        authors = list(
            (work.authors if work else ()) or (papers[0].authors if papers else ())
        )[:3]
        position = positions.get(node)
        out.append(
            {
                "id": node,
                "kind": kind,
                "label": short_label(node, ctx.inputs, ctx.library_nodes, ctx.titles),
                "title": ctx.titles.get(node, ""),
                "year": ctx.years.get(node),
                "authors": authors,
                "venue": work.venue if work else None,
                "paper_ids": [paper.paper_id for paper in papers],
                "pillars": sorted({paper.pillar_id for paper in papers}),
                "citation_count": work.citation_count if work else None,
                "influential_citation_count": work.influential_citation_count
                if work
                else None,
                "cohort_percentile": cohort.get(node),
                "local_citations": local.get(node, 0),
                "seeds_linked": ctx.scope.linked.get(node, 0),
                "topic": topics.get(node),
                "region": regions.get(node),
                "x": position[0] if position else None,
                "y": position[1] if position else None,
                "betweenness": central.get(node, 0.0),
                "on_lineage": node in on_line,
                "duplicate_of": duplicates.get(node),
                "queue_ids": [queue_by_node[node].queue_id]
                if node in queue_by_node
                else [],
            }
        )
    return out
