"""Anchor concepts to a controlled vocabulary, merge duplicates, link parents.

``concepts`` holds free-text names. Measured on the captain's library, the
vocabulary-aware extraction (#62) still leaves near-duplicates ("conditional
mutual information" / "mutual information", "minimalist categorial grammar" /
"minimalist grammar") and no hierarchy. Anchoring ties each concept to a field
of the NLP-KG vocabulary (``nlp_pillars/vocabulary.py``) and uses that to:

- **merge** concepts that are the same field, so "bleu" and "bleu score" are
  one concept, the other name kept as an alias;
- **link parents**: "multi-head attention" is a kind of "attention", and the
  hierarchy of the field ("Language Model Mechanisms" above it) comes with it.

How a concept gets an anchor, cheapest first:

1. **Exact.** The canonicalised name equals a field's label or synonym. Free.
2. **Retrieval, then a pick.** The five nearest fields by bge cosine, and one
   Haiku call that answers with a candidate's *position* (or none) and whether
   the concept IS that field or is a kind of it. The model never writes an id:
   grounded retrieval picked the right entry 98 times in 100 where generated
   ids were right 3 in 100 (SPIRES, report section 3.3). Costs about $0.0001.
3. **None.** Under ``CANDIDATE_FLOOR`` nothing is close enough to ask about,
   and the concept stays unanchored. That is an answer, not an error.

With ``MAP_ENRICHMENT_ENABLED`` off, only step 1 runs.

``plan_anchoring`` is pure: concepts in, a plan out. ``apply_plan`` writes it
in one transaction (``apply_concept_anchoring``, migration 035).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from . import db, enrichment
from .enrichment import CallLog
from .graph import canonicalize_term
from .llm_client import LLMCallError, StructuredCall, structured
from .llm_routes import CONCEPT_ANCHOR_ROUTE
from .vocabulary import CandidateIndex, VocabField, VocabularySet

logger = logging.getLogger(__name__)

MIGRATION = "docs/migrations/035_knowledge_layer.sql"

#: Fields retrieved per concept for the model to choose among.
CANDIDATES = 5
#: Below this cosine to the nearest field nothing is worth asking about.
#: Measured with the local bge model on 26 concept names from the library
#: against the field label + description: genuine matches (multi-head
#: attention 0.72, probing classifier 0.75, linear probe 0.65, subword
#: tokenization 0.72) sit at 0.65 and above; the nearest field for a concept the
#: vocabulary lacks (dolphin acoustics, bleu score, beam search) is 0.60 or
#: less. The pick is the model's; this only spares it the obvious misses.
CANDIDATE_FLOOR = 0.62
#: Characters of a field's description shown to the model.
DESCRIPTION_CHARS = 220
#: Concepts anchored in one run at most, so a surprise vocabulary of thousands
#: of unanchored concepts cannot become a surprise bill.
MAX_MODEL_CALLS = 500

METHOD_EXACT = "exact"
METHOD_MODEL = "model"
METHOD_NONE = "none"

_PAGE = 1000


class AnchorChoice(BaseModel):
    """What the model fills in: a position among the candidates, never an id."""

    choice: int = Field(
        ...,
        description="Position of the matching candidate (1 to 5), or 0 if none fits",
    )
    relation: Literal["same", "narrower", "none"] = Field(
        ...,
        description="same: the concept IS that field. narrower: it is a kind of it. none: neither",
    )


SYSTEM_PROMPT = (
    "You match a concept name taken from an NLP research paper to entries of a controlled "
    "vocabulary of NLP fields of study. Candidates are numbered. "
    "Answer relation 'same' only when the concept and the candidate are the same thing "
    "(synonym, abbreviation, plural, a wording variant). "
    "Answer 'narrower' when the concept is a specific kind, instance or variant of the candidate. "
    "Otherwise answer choice 0 and relation 'none'. "
    "A candidate that merely shares words with the concept is not a match. "
    "Choose only from the numbered candidates."
)


def candidates_prompt(name: str, candidates: Sequence[VocabField]) -> str:
    lines = [f"Concept: {name}", "", "Candidates:"]
    for position, field_ in enumerate(candidates, start=1):
        description = (field_.description or "").strip()
        if len(description) > DESCRIPTION_CHARS:
            description = description[: DESCRIPTION_CHARS - 1].rstrip() + "…"
        synonyms = f" (also: {', '.join(field_.synonyms[:4])})" if field_.synonyms else ""
        lines.append(f"{position}. {field_.label}{synonyms}" + (f" - {description}" if description else ""))
    return "\n".join(lines)


# ------------------------------------------------------------------ model


@dataclass(frozen=True)
class ConceptRow:
    """One ``concepts`` row, with how many papers link to it."""

    id: str
    name: str
    aliases: Tuple[str, ...] = ()
    anchor: Optional[str] = None
    anchor_relation: Optional[str] = None
    papers: int = 0


@dataclass(frozen=True)
class Decision:
    """How one concept was (or was not) anchored."""

    name: str
    anchor: Optional[str]
    relation: Optional[str]
    method: str
    note: str = ""
    candidates: Tuple[str, ...] = ()


@dataclass
class AnchorPlan:
    """Everything one pass decided. ``anchors`` are only the new ones."""

    decisions: List[Decision] = field(default_factory=list)
    merges: List[Dict[str, Any]] = field(default_factory=list)
    parents: List[Dict[str, str]] = field(default_factory=list)
    calls: CallLog = field(default_factory=CallLog)
    skipped: List[str] = field(default_factory=list)
    estimated_calls: int = 0

    def anchors(self) -> List[Dict[str, str]]:
        return [
            {"name": d.name, "anchor": d.anchor, "relation": d.relation}
            for d in self.decisions
            if d.anchor and d.relation
        ]

    def counts(self) -> Dict[str, int]:
        return {
            "anchored_exact": sum(1 for d in self.decisions if d.anchor and d.method == METHOD_EXACT),
            "anchored_model": sum(1 for d in self.decisions if d.anchor and d.method == METHOD_MODEL),
            "unanchored": sum(1 for d in self.decisions if not d.anchor),
            "merged_away": sum(len(m["drop"]) for m in self.merges),
            "parents": len(self.parents),
            "model_calls": len(self.calls.calls),
        }


# ------------------------------------------------------------------- plan

Pick = Callable[[str, Sequence[VocabField]], Optional[Tuple[StructuredCall, AnchorChoice]]]


def model_pick(name: str, candidates: Sequence[VocabField], *, client: Any = None) -> Tuple[StructuredCall, AnchorChoice]:
    """One priced call: which of ``candidates`` is ``name``, if any."""
    call = structured(
        CONCEPT_ANCHOR_ROUTE,
        system=SYSTEM_PROMPT,
        user=candidates_prompt(name, candidates),
        out_model=AnchorChoice,
        client=client,
        max_tokens=200,
    )
    return call, call.parsed


def _valid(choice: AnchorChoice, count: int) -> bool:
    if choice.relation == "none" or choice.choice == 0:
        return choice.relation == "none" and choice.choice == 0
    return 1 <= choice.choice <= count and choice.relation in ("same", "narrower")


def plan_anchoring(
    concepts: Sequence[ConceptRow],
    vocabularies: VocabularySet,
    *,
    index: Optional[CandidateIndex] = None,
    pick: Optional[Pick] = None,
    use_model: Optional[bool] = None,
    max_calls: int = MAX_MODEL_CALLS,
    reanchor: bool = False,
) -> AnchorPlan:
    """Decide an anchor for every unanchored concept, then merges and parents.

    ``pick`` is the model step (tests pass a stub); ``index`` retrieves the
    candidates. With ``use_model`` false (default: the config switch) only exact
    matches are made and ``index``/``pick`` are never touched.
    """
    allow_model = enrichment.enabled() if use_model is None else use_model
    plan = AnchorPlan()
    pick = pick or model_pick

    todo = [c for c in concepts if reanchor or not c.anchor]
    for concept in sorted(todo, key=lambda c: c.name):
        exact = _exact(concept, vocabularies)
        if exact is not None:
            plan.decisions.append(Decision(concept.name, exact.anchor, "same", METHOD_EXACT))
            continue
        if not allow_model or index is None:
            plan.decisions.append(
                Decision(concept.name, None, None, METHOD_NONE, "no exact match; model step off")
            )
            continue
        near = index.top(concept.name, CANDIDATES)
        if not near or near[0][1] < CANDIDATE_FLOOR:
            plan.decisions.append(
                Decision(concept.name, None, None, METHOD_NONE, "nothing near enough in the vocabulary")
            )
            continue
        plan.estimated_calls += 1
        if len(plan.calls.calls) >= max_calls:
            plan.skipped.append(concept.name)
            continue
        candidates = [field_ for field_, _ in near]
        try:
            call, choice = pick(concept.name, candidates)
        except LLMCallError as exc:
            plan.decisions.append(Decision(concept.name, None, None, METHOD_NONE, f"model call failed: {exc}"))
            if exc.call is not None:
                plan.calls.add(exc.call, "anchor")
            continue
        plan.calls.add(call, "anchor")
        if not _valid(choice, len(candidates)):
            plan.decisions.append(
                Decision(
                    concept.name, None, None, METHOD_NONE,
                    f"unusable answer (choice {choice.choice}, {choice.relation})",
                    tuple(c.anchor for c in candidates),
                )
            )
            continue
        if choice.choice == 0:
            plan.decisions.append(
                Decision(concept.name, None, None, METHOD_NONE, "model: none fit", tuple(c.anchor for c in candidates))
            )
            continue
        picked = candidates[choice.choice - 1]
        plan.decisions.append(
            Decision(concept.name, picked.anchor, choice.relation, METHOD_MODEL, "", tuple(c.anchor for c in candidates))
        )

    _plan_merges_and_parents(plan, concepts, vocabularies)
    return plan


def _exact(concept: ConceptRow, vocabularies: VocabularySet) -> Optional[VocabField]:
    for name in (concept.name, *concept.aliases):
        found = vocabularies.lookup(name)
        if found is not None:
            return found
    return None


def _plan_merges_and_parents(
    plan: AnchorPlan, concepts: Sequence[ConceptRow], vocabularies: VocabularySet
) -> None:
    """Merge concepts anchored 'same' to one field, and link parents, over old and new anchors."""
    by_name = {c.name: c for c in concepts}
    state: Dict[str, Tuple[str, str]] = {
        c.name: (c.anchor, c.anchor_relation) for c in concepts if c.anchor and c.anchor_relation
    }
    for decision in plan.decisions:
        if decision.anchor and decision.relation:
            state[decision.name] = (decision.anchor, decision.relation)

    same: Dict[str, List[str]] = {}
    for name, (anchor, relation) in state.items():
        if relation == "same":
            same.setdefault(anchor, []).append(name)

    dropped: set = set()
    for anchor, names in sorted(same.items()):
        if len(names) < 2:
            continue
        field_ = vocabularies.get(anchor)
        label = canonicalize_term(field_.label) if field_ else ""
        # Keep the concept already named like the field, else the one most papers
        # link to, else the shortest name, then alphabetical: a stable choice.
        keep = min(
            names,
            key=lambda n: (
                0 if canonicalize_term(n) == label else 1,
                -(by_name[n].papers if n in by_name else 0),
                len(n),
                n,
            ),
        )
        drop = sorted(n for n in names if n != keep)
        plan.merges.append({"anchor": anchor, "keep": keep, "drop": drop})
        dropped.update(drop)

    live = {name: value for name, value in state.items() if name not in dropped}
    holder: Dict[str, str] = {}  # anchor -> the concept that IS that field
    for name, (anchor, relation) in sorted(live.items()):
        if relation == "same":
            holder.setdefault(anchor, name)
    for merge in plan.merges:
        holder[merge["anchor"]] = merge["keep"]

    for name, (anchor, relation) in sorted(live.items()):
        chain: List[str] = [anchor] if relation == "narrower" else []
        chain += [f"{field_.vocab}:{field_.id}" for field_ in vocabularies.ancestors(anchor)]
        for ancestor in chain:
            parent = holder.get(ancestor)
            if parent and parent != name:
                plan.parents.append({"name": name, "parent": parent})
                break


# ------------------------------------------------------------------ store


class ConceptStoreError(Exception):
    """Concepts could not be read or the plan could not be written."""


def _store_error(error: Any) -> ConceptStoreError:
    text = str(error).lower()
    if any(t in text for t in ("pgrst202", "pgrst204", "pgrst205", "does not exist", "schema cache", "anchor")):
        return ConceptStoreError(
            f"Concept anchoring is not available ({error}). Apply {MIGRATION} by hand."
        )
    return ConceptStoreError(f"Could not read or write concepts: {error}")


def _read(table: str, columns: str, order: str) -> List[dict]:
    client = db.get_client()
    rows: List[dict] = []
    offset = 0
    while True:
        response = client.table(table).select(columns).order(order).limit(_PAGE).offset(offset).execute()
        if response["error"]:
            raise _store_error(response["error"])
        page = response["data"] or []
        rows.extend(page)
        if len(page) < _PAGE:
            return rows
        offset += len(page)


def load_concepts() -> List[ConceptRow]:
    """Every concept with its anchor and the number of papers that link to it."""
    rows = _read("concepts", "id,name,aliases,anchor,anchor_relation", "name")
    links: Dict[str, int] = {}
    for row in _read("paper_concepts", "concept_id,paper_id", "concept_id,paper_id"):
        links[str(row["concept_id"])] = links.get(str(row["concept_id"]), 0) + 1
    return [
        ConceptRow(
            id=str(row["id"]),
            name=str(row["name"]),
            aliases=tuple(str(a) for a in (row.get("aliases") or []) if isinstance(a, str)),
            anchor=row.get("anchor") or None,
            anchor_relation=row.get("anchor_relation") or None,
            papers=links.get(str(row["id"]), 0),
        )
        for row in rows
    ]


def apply_plan(plan: AnchorPlan) -> Dict[str, Any]:
    """Write anchors, merges and parents in one transaction."""
    response = db.get_client().rpc(
        "apply_concept_anchoring",
        {"p_anchors": plan.anchors(), "p_merges": [{"keep": m["keep"], "drop": m["drop"]} for m in plan.merges], "p_parents": plan.parents},
    )
    if response["error"]:
        raise _store_error(response["error"])
    data = response["data"]
    if isinstance(data, list):
        data = data[0] if data else {}
    return data or {}


# ----------------------------------------------------------------- report


@dataclass
class AnchorReport:
    """What ``concepts anchor`` did, for the CLI."""

    dry_run: bool
    concepts: int
    plan: AnchorPlan
    stored: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def render(self) -> str:
        counts = self.plan.counts()
        lines = [
            f"concepts anchor ({'dry-run' if self.dry_run else 'write'})",
            f"vocabulary: {', '.join(sorted({d.anchor.split(':')[0] for d in self.plan.decisions if d.anchor})) or 'none matched'}",
            f"concepts {self.concepts}; anchored exact {counts['anchored_exact']}, "
            f"by model {counts['anchored_model']}, unanchored {counts['unanchored']}",
            f"merges {len(self.plan.merges)} (drops {counts['merged_away']}), parent links {counts['parents']}",
        ]
        if self.note:
            lines.append(self.note)
        if self.plan.skipped:
            lines.append(f"{len(self.plan.skipped)} concept(s) left for the next run (call limit).")
        for merge in self.plan.merges:
            lines.append(f"  merge {merge['anchor']}: keep '{merge['keep']}', drop {merge['drop']}")
        for decision in self.plan.decisions:
            if decision.anchor:
                lines.append(f"  {decision.name} -> {decision.anchor} ({decision.relation}, {decision.method})")
        for call in self.plan.calls.calls:
            lines.append(f"  call {call.render()}")
        lines.append(self.plan.calls.detail("model spend"))
        if not self.dry_run and self.stored:
            lines.append(f"stored {self.stored}")
        return "\n".join(lines)


def run_anchoring(
    *,
    dry_run: bool = False,
    reanchor: bool = False,
    max_calls: int = MAX_MODEL_CALLS,
    embed: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
    pick: Optional[Pick] = None,
    vocabularies: Optional[VocabularySet] = None,
) -> AnchorReport:
    """Anchor the library's concepts. ``--dry-run`` makes no model call and writes nothing."""
    from .vocabulary import load_vocabularies

    vocabularies = vocabularies or load_vocabularies()
    concepts = load_concepts()
    use_model = enrichment.enabled() and not dry_run
    note = ""
    if enrichment.enabled() and dry_run:
        note = "dry-run: exact matches only; the model step is not run and nothing is billed."
    if not enrichment.enabled():
        note = "MAP_ENRICHMENT_ENABLED is off: exact matches only; no model call."

    index = None
    if use_model:
        try:
            enrichment.require_budget(0.01)
        except enrichment.EnrichmentRefused as exc:
            use_model = False
            note = str(exc)
    if use_model:
        if embed is None:
            from .embeddings import LocalBgeEmbeddingProvider

            embed = LocalBgeEmbeddingProvider().embed_documents
        index = CandidateIndex(vocabularies, embed)

    plan = plan_anchoring(
        concepts, vocabularies, index=index, pick=pick, use_model=use_model, max_calls=max_calls, reanchor=reanchor
    )
    stored: Dict[str, Any] = {}
    if not dry_run and (plan.anchors() or plan.merges or plan.parents):
        stored = apply_plan(plan)
    return AnchorReport(dry_run=dry_run, concepts=len(concepts), plan=plan, stored=stored, note=note)
