"""Learner knowledge graph: plan it, store the derived rows, read a neighbourhood.

Nothing here calls a model or the network. Concept-name embedding and Semantic
Scholar are passed in by the caller, and a dry run uses neither. Related-paper
edges read one stored title+abstract vector per paper; the database is not
asked to do arithmetic on ``real[]``.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .citations import fetch_library_citations
from .paper_vectors import SOURCE_TITLE_ABSTRACT
from .schemas import PaperCitation

# Surface variants of one concept ("Self-Attention" / "self attention mechanism")
# merge when their name embeddings are at least this similar. High on purpose:
# bge cosine between NLP phrases sits high, and a low cut collapses the vocabulary.
CONCEPT_MERGE_COSINE = 0.92

# Neighbours kept per paper after the cosine floor. Five was already the rank
# cut; without a floor it linked every paper to five others.
RELATED_TOP_K = 5

# Title+abstract bge cosines on the measured 12-paper library span 0.43–0.84
# (std 0.093). Mean-pooled chunk centroids span 0.79–0.97 (std 0.042), so
# top-k alone emitted 60 edges, 39 of them cross-pillar, all weighted
# 0.84–0.97. Minimalist Grammars' five neighbours score 0.50–0.62 on
# title+abstract; on-topic pairs in that measurement sit at 0.76–0.84
# (DeepSeek-V4 ↔ DeepSeek-VL 0.84, the AML trio 0.76–0.81, Attention ↔ BPE
# 0.77). 0.70 is in that gap, and it is the suggestion-eligibility floor
# from the graph-page report (section 6). Pairs below it are dropped before
# the rank cut. The floor is for this bge title+abstract space only.
RELATED_MIN_COSINE = 0.70

# One 319-author paper would otherwise flood the graph. First authors win.
AUTHORS_PER_PAPER = 10

MAX_NODES = 400
MAX_EDGES = 800
MAX_DEPTH = 3
LABEL_LIMIT = 160

PROVENANCE_STRUCTURE = "derived:structure"
PROVENANCE_KEY_TERMS = "derived:key_terms"
PROVENANCE_S2 = "semantic_scholar"
SOURCE_KEY_TERMS = "key_terms"

EdgeKey = Tuple[str, str, str, str, str]

_ACRONYM = re.compile(r"\s*\([A-Z0-9][A-Z0-9-]{1,}\)\s*")
_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9\s]")


def canonicalize_term(raw: str) -> str:
    """Lowercase, drop a trailing ``(ACRONYM)``, and singularise the last word.

    ``Self-Attention (SA)`` and ``self attentions`` both become ``self attention``.
    Words that are already singular (``analysis``, ``loss``) are left alone.
    """
    if not isinstance(raw, str):
        return ""
    text = _ACRONYM.sub(" ", raw)
    text = text.replace("-", " ").replace("_", " ")
    text = text.casefold()
    text = _PUNCT.sub(" ", text)
    text = _SPACE.sub(" ", text).strip()
    if not text:
        return ""
    parts = text.split(" ")
    parts[-1] = _singular_token(parts[-1])
    return " ".join(part for part in parts if part)


def _singular_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("sses") and len(token) > 5:
        return token[:-2]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sis", "ss", "us", "is")):
        return token
    if token.endswith("s"):
        return token[:-1]
    return token


def author_key(name: str) -> str:
    """Case-folded author identity. The displayed spelling is chosen separately."""
    if not isinstance(name, str):
        return ""
    return _SPACE.sub(" ", name.strip()).casefold()


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, or 0 when either vector is empty or a different width."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for x_value, y_value in zip(left, right, strict=True):
        dot += x_value * y_value
        left_norm += x_value * x_value
        right_norm += y_value * y_value
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def _as_vector(value: Any) -> List[float]:
    if isinstance(value, str):
        text = value.strip().strip("{}[]")
        if not text:
            return []
        try:
            return [float(part) for part in text.split(",")]
        except ValueError:
            return []
    if isinstance(value, (list, tuple)):
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return []
    return []


def _terms(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]
    return []


def _authors(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    names: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            names.append(_SPACE.sub(" ", item.strip()))
        elif isinstance(item, dict):
            raw = item.get("name") or item.get("author")
            if isinstance(raw, str) and raw.strip():
                names.append(_SPACE.sub(" ", raw.strip()))
    return names


def _owned_provenance(provenance: str) -> bool:
    return provenance.startswith("derived:") or provenance == PROVENANCE_S2


@dataclass
class GraphSnapshot:
    """Rows the planner reads. Dicts, so a test can build one without the database."""

    papers: List[dict] = field(default_factory=list)
    pillars: List[dict] = field(default_factory=list)
    notes: List[dict] = field(default_factory=list)
    lessons: List[dict] = field(default_factory=list)
    cards: List[dict] = field(default_factory=list)
    episodes: List[dict] = field(default_factory=list)
    paper_vectors: List[dict] = field(default_factory=list)
    citations: List[dict] = field(default_factory=list)


@dataclass(frozen=True)
class GraphEdge:
    """One directed edge. ``key`` is the unique identity stored in Postgres."""

    src_kind: str
    src_id: str
    dst_kind: str
    dst_id: str
    kind: str
    weight: float
    provenance: str

    @property
    def key(self) -> EdgeKey:
        """``(src_kind, src_id, dst_kind, dst_id, kind)``."""
        return (self.src_kind, self.src_id, self.dst_kind, self.dst_id, self.kind)


@dataclass(frozen=True)
class ConceptRow:
    """A canonical concept and the surface forms that mapped to it."""

    name: str
    aliases: List[str]


@dataclass(frozen=True)
class PaperConceptLink:
    """One paper covers one concept. ``name`` is the canonical concept name."""

    paper_id: str
    name: str
    source: str
    weight: float


@dataclass
class GraphPlan:
    """Everything a rebuild writes. Edge keys are unique."""

    concepts: List[ConceptRow]
    paper_concepts: List[PaperConceptLink]
    edges: List[GraphEdge]
    embedding_model: Optional[str]
    concept_merge: str


@dataclass
class GraphBuildReport:
    """Counts printed by ``graph build``. ``edges`` is a count, not the rows."""

    dry_run: bool
    library_papers: int
    concepts: int
    paper_concepts: int
    edges: int
    edges_by_kind: Dict[str, int]
    embedding_model: Optional[str]
    concept_merge: str
    citations_to_fetch: int
    citations_fetched: int
    citations_stored: int
    warnings: List[str]
    citations_failed: int = 0
    citations_failure_reason: str = ""

    def render(self) -> str:
        """Plain-text summary. Markup is left off so ids are not styled."""
        mode = "dry-run" if self.dry_run else "write"
        kinds = ", ".join(
            f"{kind} {count}" for kind, count in sorted(self.edges_by_kind.items())
        ) or "none"
        lines = [
            f"graph build ({mode})",
            f"library papers: {self.library_papers}",
            f"concepts: {self.concepts} ({self.concept_merge})",
            f"paper_concepts: {self.paper_concepts}",
            f"edges: {self.edges} ({kinds})",
            f"embedding model: {self.embedding_model or 'none'}",
            f"citations to fetch: {self.citations_to_fetch}",
            f"citations fetched: {self.citations_fetched}",
            f"citations newly stored: {self.citations_stored}",
        ]
        if self.citations_failed:
            reason = self.citations_failure_reason or "request failed"
            lines.append(f"citations failed: {self.citations_failed} ({reason})")
        else:
            lines.append("citations failed: 0")
        lines.extend(f"warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def _citation_failure_reason(reasons: Sequence[str]) -> str:
    """One short reason, or a few when the failures differ."""
    unique: List[str] = []
    for reason in reasons:
        if reason and reason not in unique:
            unique.append(reason)
    return "; ".join(unique[:3])


def merge_names(
    names: Sequence[str],
    vectors: Sequence[Sequence[float]],
    threshold: float = CONCEPT_MERGE_COSINE,
) -> Dict[str, str]:
    """Map each name to the lexicographically smallest name in its component.

    Pairs at or above ``threshold`` are unioned. The representative is the
    smallest name so a rebuild does not flip which spelling survives.
    """
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for index, left in enumerate(names):
        left_vector = vectors[index]
        for offset, right in enumerate(names[index + 1 :], start=index + 1):
            if cosine(left_vector, vectors[offset]) >= threshold:
                union(left, right)
    return {name: find(name) for name in names}


def _edge(
    src_kind: str,
    src_id: str,
    dst_kind: str,
    dst_id: str,
    kind: str,
    provenance: str,
    weight: float = 1.0,
) -> GraphEdge:
    return GraphEdge(
        src_kind=src_kind,
        src_id=str(src_id),
        dst_kind=dst_kind,
        dst_id=str(dst_id),
        kind=kind,
        weight=weight,
        provenance=provenance,
    )


def _dedupe_edges(edges: Iterable[GraphEdge]) -> List[GraphEdge]:
    best: Dict[EdgeKey, GraphEdge] = {}
    for edge in edges:
        current = best.get(edge.key)
        if current is None or edge.weight > current.weight:
            best[edge.key] = edge
    return [best[key] for key in sorted(best)]


def _structural_edges(snapshot: GraphSnapshot) -> List[GraphEdge]:  # noqa: C901
    papers = sorted(
        (paper for paper in snapshot.papers if paper.get("id")),
        key=lambda paper: str(paper["id"]),
    )
    displays: Dict[str, str] = {}
    for paper in papers:
        for author in _authors(paper.get("authors")):
            key = author_key(author)
            if key and key not in displays:
                displays[key] = author

    edges: List[GraphEdge] = []
    for paper in papers:
        paper_id = str(paper["id"])
        pillar_id = paper.get("pillar_id")
        if pillar_id:
            edges.append(
                _edge(
                    "paper",
                    paper_id,
                    "pillar",
                    pillar_id,
                    "in_pillar",
                    PROVENANCE_STRUCTURE,
                )
            )
        seen = set()
        for author in _authors(paper.get("authors")):
            key = author_key(author)
            if not key or key in seen:
                continue
            seen.add(key)
            if len(seen) > AUTHORS_PER_PAPER:
                break
            edges.append(
                _edge(
                    "paper",
                    paper_id,
                    "author",
                    displays[key],
                    "authored_by",
                    PROVENANCE_STRUCTURE,
                )
            )

    def _of(rows: Sequence[dict], kind: str) -> None:
        for row in rows:
            row_id = row.get("id")
            paper_id = row.get("paper_id")
            if row_id and paper_id:
                edges.append(
                    _edge(
                        kind,
                        row_id,
                        "paper",
                        paper_id,
                        "of",
                        PROVENANCE_STRUCTURE,
                    )
                )

    _of(snapshot.lessons, "lesson")
    _of(snapshot.cards, "quiz_card")
    _of(snapshot.episodes, "episode")
    return edges


def _concept_plan(  # noqa: C901
    snapshot: GraphSnapshot,
    embed_names: Optional[
        Callable[[Sequence[str]], Optional[Sequence[Sequence[float]]]]
    ],
) -> Tuple[List[ConceptRow], List[PaperConceptLink], List[GraphEdge], str]:
    library = {str(paper["id"]) for paper in snapshot.papers if paper.get("id")}
    surfaces: Dict[str, set] = defaultdict(set)
    papers_for: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for note in snapshot.notes:
        paper_id = note.get("paper_id")
        if not paper_id or str(paper_id) not in library:
            continue
        for term in _terms(note.get("key_terms")):
            canonical = canonicalize_term(term)
            if not canonical:
                continue
            surfaces[canonical].add(term.strip())
            papers_for[canonical][str(paper_id)] += 1

    names = sorted(surfaces)
    mapping = {name: name for name in names}
    concept_merge = "string"
    if embed_names is not None and len(names) >= 2:
        vectors = embed_names(names)
        if vectors and len(vectors) == len(names):
            mapping = merge_names(names, vectors)
            concept_merge = "string+embedding"

    grouped_aliases: Dict[str, set] = defaultdict(set)
    grouped_papers: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name in names:
        root = mapping[name]
        grouped_aliases[root].update(surfaces[name])
        for paper_id, count in papers_for[name].items():
            grouped_papers[root][paper_id] += count

    concepts = [
        ConceptRow(name=name, aliases=sorted(grouped_aliases[name]))
        for name in sorted(grouped_aliases)
    ]
    links: List[PaperConceptLink] = []
    edges: List[GraphEdge] = []
    for name in sorted(grouped_papers):
        for paper_id in sorted(grouped_papers[name]):
            weight = float(grouped_papers[name][paper_id])
            links.append(
                PaperConceptLink(
                    paper_id=paper_id,
                    name=name,
                    source=SOURCE_KEY_TERMS,
                    weight=weight,
                )
            )
            edges.append(
                _edge(
                    "paper",
                    paper_id,
                    "concept",
                    name,
                    "covers",
                    PROVENANCE_KEY_TERMS,
                    weight,
                )
            )
    return concepts, links, edges, concept_merge


def _title_abstract_by_model(snapshot: GraphSnapshot) -> Dict[str, List[dict]]:
    """Library title+abstract rows grouped by model. Other sources are skipped."""
    library = {str(paper["id"]) for paper in snapshot.papers if paper.get("id")}
    by_model: Dict[str, List[dict]] = defaultdict(list)
    for row in snapshot.paper_vectors:
        if str(row.get("source") or "") != SOURCE_TITLE_ABSTRACT:
            continue
        if str(row.get("ref_kind") or "") != "paper":
            continue
        paper_id = str(row.get("ref_id") or "")
        model_id = row.get("model_id")
        vector = _as_vector(row.get("vector"))
        if paper_id not in library or not model_id or not vector:
            continue
        by_model[str(model_id)].append(
            {"paper_id": paper_id, "vector": vector}
        )
    return by_model


def _related_edges(
    snapshot: GraphSnapshot,
) -> Tuple[Optional[str], List[GraphEdge]]:
    """Top-k title+abstract neighbours at or above ``RELATED_MIN_COSINE``.

    One vector per library paper. Chunk means are not read. When several
    models are stored, the one with the most rows wins (ties break by model
    id) so two spaces are never mixed.
    """
    by_model = _title_abstract_by_model(snapshot)
    if not by_model:
        return None, []

    model_id = sorted(by_model, key=lambda name: (-len(by_model[name]), name))[0]
    vectors: Dict[str, Sequence[float]] = {}
    for row in by_model[model_id]:
        vectors.setdefault(row["paper_id"], row["vector"])
    provenance = f"derived:embedding:{model_id}"
    edges: List[GraphEdge] = []
    for paper_id in sorted(vectors):
        scored = []
        for other_id in sorted(vectors):
            if other_id == paper_id:
                continue
            score = cosine(vectors[paper_id], vectors[other_id])
            if score < RELATED_MIN_COSINE:
                continue
            scored.append((score, other_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        for score, other_id in scored[:RELATED_TOP_K]:
            edges.append(
                _edge(
                    "paper",
                    paper_id,
                    "paper",
                    other_id,
                    "related_to",
                    provenance,
                    round(score, 6),
                )
            )
    return model_id, edges


def _citation_edges(snapshot: GraphSnapshot) -> List[GraphEdge]:
    library = {str(paper["id"]) for paper in snapshot.papers if paper.get("id")}
    edges: List[GraphEdge] = []
    for row in snapshot.citations:
        library_id = str(row.get("paper_id") or "")
        other_id = str(row.get("cited_paper_id") or "")
        if library_id not in library or not other_id or library_id == other_id:
            continue
        direction = row.get("direction") or row.get("citation_direction")
        if direction == "outgoing":
            src_id, dst_id = library_id, other_id
        elif direction == "incoming":
            src_id, dst_id = other_id, library_id
        else:
            continue
        if src_id == dst_id:
            continue

        def _kind(paper_id: str) -> str:
            return "paper" if paper_id in library else "external_paper"

        edges.append(
            _edge(
                _kind(src_id),
                src_id,
                _kind(dst_id),
                dst_id,
                "cites",
                PROVENANCE_S2,
            )
        )
    return edges


def build_plan(
    snapshot: GraphSnapshot,
    embed_names: Optional[
        Callable[[Sequence[str]], Optional[Sequence[Sequence[float]]]]
    ] = None,
) -> GraphPlan:
    """Build the full derived graph for this snapshot. No I/O."""
    concepts, links, concept_edges, concept_merge = _concept_plan(
        snapshot, embed_names
    )
    model_id, related = _related_edges(snapshot)
    edges = _dedupe_edges(
        [
            *_structural_edges(snapshot),
            *concept_edges,
            *related,
            *_citation_edges(snapshot),
        ]
    )
    link_best: Dict[Tuple[str, str], PaperConceptLink] = {}
    for link in links:
        key = (link.paper_id, link.name)
        current = link_best.get(key)
        if current is None or link.weight > current.weight:
            link_best[key] = link
    return GraphPlan(
        concepts=concepts,
        paper_concepts=[link_best[key] for key in sorted(link_best)],
        edges=edges,
        embedding_model=model_id,
        concept_merge=concept_merge,
    )


def plan_payload(plan: GraphPlan) -> Dict[str, list]:
    """JSON arguments for ``replace_derived_graph``."""
    return {
        "p_concepts": [
            {"name": concept.name, "aliases": list(concept.aliases)}
            for concept in plan.concepts
        ],
        "p_links": [
            {
                "paper_id": link.paper_id,
                "name": link.name,
                "source": link.source,
                "weight": link.weight,
            }
            for link in plan.paper_concepts
        ],
        "p_edges": [
            {
                "src_kind": edge.src_kind,
                "src_id": edge.src_id,
                "dst_kind": edge.dst_kind,
                "dst_id": edge.dst_id,
                "kind": edge.kind,
                "weight": edge.weight,
                "provenance": edge.provenance,
            }
            for edge in plan.edges
        ],
    }


class MemoryGraphStore:
    """In-memory stand-in for ``replace_derived_graph``.

    Same ownership rules: derived and Semantic Scholar edges are replaced,
    ``user`` and ``llm:`` edges stay, and a second apply does not grow the
    unique edge set.
    """

    def __init__(self) -> None:
        """Start empty. ``apply`` is the only writer."""
        self.concepts: Dict[str, ConceptRow] = {}
        self.links: Dict[Tuple[str, str], PaperConceptLink] = {}
        self.edges: Dict[EdgeKey, GraphEdge] = {}

    def apply(self, plan: GraphPlan) -> None:
        """Replace derived rows with ``plan``. A second call does not grow the set."""
        for concept in plan.concepts:
            self.concepts[concept.name] = concept
        self.links = {
            key: link
            for key, link in self.links.items()
            if link.source != SOURCE_KEY_TERMS
        }
        for link in plan.paper_concepts:
            self.links[(link.paper_id, link.name)] = link
        referenced = {link.name for link in self.links.values()}
        self.concepts = {
            name: concept
            for name, concept in self.concepts.items()
            if name in referenced
        }
        self.edges = {
            key: edge
            for key, edge in self.edges.items()
            if not _owned_provenance(edge.provenance)
        }
        for edge in plan.edges:
            if edge.key in self.edges and not _owned_provenance(
                self.edges[edge.key].provenance
            ):
                # A user edge already occupies this key. The rebuild's row
                # replaces it, matching ON CONFLICT DO UPDATE.
                pass
            self.edges[edge.key] = edge


def _citation_dict(row: PaperCitation) -> dict:
    return {
        "paper_id": row.paper_id,
        "cited_paper_id": row.cited_paper_id,
        "direction": row.citation_direction,
        "source": row.source,
    }


def _merge_citations(
    existing: Sequence[dict], fetched: Sequence[PaperCitation]
) -> List[dict]:
    merged: Dict[Tuple[str, str, str], dict] = {}
    for row in list(existing) + [_citation_dict(item) for item in fetched]:
        key = (
            str(row.get("paper_id") or ""),
            str(row.get("cited_paper_id") or ""),
            str(row.get("direction") or row.get("citation_direction") or ""),
        )
        if key[0] and key[1] and key[2]:
            merged[key] = {
                "paper_id": key[0],
                "cited_paper_id": key[1],
                "direction": key[2],
                "source": row.get("source") or PROVENANCE_S2,
            }
    return [merged[key] for key in sorted(merged)]


def run_graph_build(
    snapshot: GraphSnapshot,
    *,
    dry_run: bool,
    citation_tool=None,
    embed_names: Optional[
        Callable[[Sequence[str]], Sequence[Sequence[float]]]
    ] = None,
    store_citations: Optional[Callable[[Sequence[PaperCitation]], int]] = None,
    apply: Optional[Callable[[GraphPlan], None]] = None,
) -> GraphBuildReport:
    """Plan the graph and, unless ``dry_run``, fetch, embed, and write.

    Dry-run does not call ``citation_tool``, ``embed_names``, ``store_citations``,
    or ``apply``.
    """
    warnings: List[str] = []
    paper_ids = [
        str(paper["id"]) for paper in snapshot.papers if paper.get("id")
    ]
    # file: uploads 404 on Semantic Scholar. Counting them as "to fetch"
    # made a string of failures look like a plan.
    query_ids = [paper_id for paper_id in paper_ids if not paper_id.startswith("file:")]
    citations_to_fetch = len(query_ids) if citation_tool is not None else 0
    fetched: List[PaperCitation] = []
    failures: List[str] = []
    stored = 0
    if not dry_run and citation_tool is not None:
        fetched, failures = fetch_library_citations(query_ids, citation_tool)
        if fetched and store_citations is not None:
            added = store_citations(fetched)
            stored = added if isinstance(added, int) else 0

    citations = _merge_citations(snapshot.citations, fetched if not dry_run else [])

    def _embed(names: Sequence[str]):
        if embed_names is None:
            return None
        try:
            return embed_names(names)
        except Exception as exc:
            warnings.append(f"concept embedding merge skipped: {exc}")
            return None

    plan = build_plan(
        replace(snapshot, citations=citations),
        embed_names=None if dry_run else _embed,
    )
    if not dry_run:
        if apply is None:
            raise RuntimeError("graph build has no writer")
        apply(plan)

    by_kind: Dict[str, int] = defaultdict(int)
    for edge in plan.edges:
        by_kind[edge.kind] += 1
    return GraphBuildReport(
        dry_run=dry_run,
        library_papers=len(paper_ids),
        concepts=len(plan.concepts),
        paper_concepts=len(plan.paper_concepts),
        edges=len(plan.edges),
        edges_by_kind=dict(by_kind),
        embedding_model=plan.embedding_model,
        concept_merge=plan.concept_merge,
        citations_to_fetch=citations_to_fetch,
        citations_fetched=0 if dry_run else len(fetched),
        citations_stored=0 if dry_run else stored,
        warnings=warnings,
        citations_failed=0 if dry_run else len(failures),
        citations_failure_reason=(
            "" if dry_run or not failures else _citation_failure_reason(failures)
        ),
    )


def _short_label(text: Any) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= LABEL_LIMIT:
        return collapsed
    return collapsed[: LABEL_LIMIT - 1].rstrip() + "…"


def slice_graph(  # noqa: C901
    *,
    papers: Sequence[dict],
    pillars: Sequence[dict],
    concepts: Sequence[dict],
    lessons: Sequence[dict],
    cards: Sequence[dict],
    episodes: Sequence[dict],
    edges: Sequence[dict],
    pillar: Optional[str] = None,
    depth: int = 1,
    center: Optional[str] = None,
    max_nodes: int = MAX_NODES,
    max_edges: int = MAX_EDGES,
) -> dict:
    """Nodes and edges within ``depth`` hops of the seed, capped.

    Traversal walks edges in either direction. Library papers outside
    ``pillar`` are not entered, except the requested ``center`` itself.
    """
    depth = max(0, min(int(depth), MAX_DEPTH))
    pillar = pillar or None
    center = center or None
    nodes: Dict[Tuple[str, str], dict] = {}

    def add_node(
        kind: str,
        node_id: Any,
        label: Any,
        *,
        pillar_id: Optional[str] = None,
        aliases: Optional[Sequence[str]] = None,
    ) -> None:
        if node_id is None or str(node_id) == "":
            return
        key = (kind, str(node_id))
        if key in nodes:
            return
        node = {
            "id": str(node_id),
            "kind": kind,
            "label": _short_label(label if label else node_id),
        }
        if pillar_id:
            node["pillar_id"] = str(pillar_id)
        if aliases:
            node["aliases"] = [str(alias) for alias in aliases if str(alias).strip()]
        nodes[key] = node

    for row in pillars:
        if row.get("id"):
            add_node(
                "pillar",
                row["id"],
                row.get("name") or row["id"],
                pillar_id=str(row["id"]),
            )
    for row in papers:
        if row.get("id"):
            add_node(
                "paper",
                row["id"],
                row.get("title") or row["id"],
                pillar_id=row.get("pillar_id"),
            )
    for row in concepts:
        name = row.get("name") or row.get("id")
        add_node("concept", name, name, aliases=row.get("aliases") or [])
    for row in lessons:
        add_node(
            "lesson",
            row.get("id"),
            row.get("tl_dr") or row.get("title") or row.get("id"),
            pillar_id=row.get("pillar_id"),
        )
    for row in cards:
        add_node(
            "quiz_card",
            row.get("id"),
            row.get("question") or row.get("id"),
            pillar_id=row.get("pillar_id"),
        )
    for row in episodes:
        add_node(
            "episode",
            row.get("id"),
            row.get("title") or row.get("id"),
            pillar_id=row.get("pillar_id"),
        )

    clean_edges = []
    for edge in edges:
        src_kind = str(edge.get("src_kind") or "")
        src_id = str(edge.get("src_id") or "")
        dst_kind = str(edge.get("dst_kind") or "")
        dst_id = str(edge.get("dst_id") or "")
        kind = str(edge.get("kind") or "")
        if not (src_kind and src_id and dst_kind and dst_id and kind):
            continue
        add_node(src_kind, src_id, src_id)
        add_node(dst_kind, dst_id, dst_id)
        weight = edge.get("weight")
        try:
            weight_value = None if weight is None else float(weight)
        except (TypeError, ValueError):
            weight_value = None
        clean_edges.append(
            {
                "src_kind": src_kind,
                "src_id": src_id,
                "dst_kind": dst_kind,
                "dst_id": dst_id,
                "kind": kind,
                "weight": weight_value,
                "provenance": str(edge.get("provenance") or ""),
            }
        )

    def allowed(kind: str, node_id: str) -> bool:
        if not pillar:
            return True
        if kind == "pillar":
            return node_id == pillar
        if kind == "paper":
            node = nodes.get((kind, node_id))
            if node is None:
                return False
            return node.get("pillar_id") == pillar
        return True

    if center:
        seeds = [key for key, node in nodes.items() if node["id"] == center]
        found = bool(seeds)
    elif pillar:
        seeds = [
            key
            for key, node in nodes.items()
            if (node["kind"] == "paper" and node.get("pillar_id") == pillar)
            or (node["kind"] == "pillar" and node["id"] == pillar)
        ]
        found = True
    else:
        seeds = [key for key, node in nodes.items() if node["kind"] == "paper"]
        found = True
    seeds = sorted(seeds)

    adjacency: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    for edge in clean_edges:
        src = (edge["src_kind"], edge["src_id"])
        dst = (edge["dst_kind"], edge["dst_id"])
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    for key in adjacency:
        adjacency[key] = sorted(set(adjacency[key]))

    seen = set()
    queue: deque = deque()
    truncated = False
    for key in seeds:
        if len(seen) >= max_nodes:
            truncated = True
            break
        if key not in seen:
            seen.add(key)
            queue.append((key, 0))

    while queue:
        key, distance = queue.popleft()
        if distance >= depth:
            continue
        for neighbour in adjacency.get(key, []):
            if neighbour in seen:
                continue
            if neighbour not in seeds and not allowed(neighbour[0], neighbour[1]):
                continue
            if len(seen) >= max_nodes:
                truncated = True
                continue
            seen.add(neighbour)
            queue.append((neighbour, distance + 1))

    kept_nodes = [nodes[key] for key in sorted(seen) if key in nodes]
    kept_edges = [
        edge
        for edge in clean_edges
        if (edge["src_kind"], edge["src_id"]) in seen
        and (edge["dst_kind"], edge["dst_id"]) in seen
    ]
    kept_edges.sort(
        key=lambda edge: (
            edge["kind"],
            edge["src_kind"],
            edge["src_id"],
            edge["dst_kind"],
            edge["dst_id"],
        )
    )
    if len(kept_edges) > max_edges:
        kept_edges = kept_edges[:max_edges]
        truncated = True

    return {
        "nodes": kept_nodes,
        "edges": kept_edges,
        "truncated": truncated,
        "found": found,
        "depth": depth,
        "pillar": pillar,
        "center": center,
    }
