"""Typed connections between papers: what a lesson says, and what the learner decided.

Two sources feed the map's typed edges, and neither calls a model here:

1. **The lesson.** The synthesis call writes each connection as
   ``{work, title, relation, why}`` (``schemas.ConnectionDraft``). ``work`` is
   how the paper itself writes the citation ("Hewitt and Liang 2019").
   ``resolve_mention`` finds that work in the paper's own reference list: first
   author's surname, year within one (a preprint year is not the venue year),
   and a title match when a title is given. It refuses to guess: a mention that
   fits two references, or none, stays unresolved and is reported as such.
   Nothing resolved is stored. The edge is recomputed on every map build from
   the lesson and the reference list, so it cannot disagree with either.

2. **The learner.** ``learner_connections`` (migration 035) holds a confirm,
   reject or edit of a suggested connection, ``provenance = 'user'``. The map
   build reads it and never writes it.

Lessons written before the typed form exist only as prose. ``legacy_links``
reads author-year mentions out of that prose and guesses the relation from the
words around it, or says ``background``. Those links are marked ``legacy`` so
a reader can tell a guess from the model's own typing.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import db
from .schemas import CONNECTION_RELATIONS

logger = logging.getLogger(__name__)

MIGRATION = "docs/migrations/035_knowledge_layer.sql"

#: The relations a learner may set. ``related`` is the generic one.
RELATIONS: Tuple[str, ...] = (*CONNECTION_RELATIONS, "related")
STATUS_CONFIRMED = "confirmed"
STATUS_REJECTED = "rejected"
STATUSES: Tuple[str, ...] = (STATUS_CONFIRMED, STATUS_REJECTED)

#: Longest phrase a learner can write (the column's CHECK).
PHRASE_MAX_CHARS = 400

#: A reference year may differ from the mention's by this much and still match
#: (arXiv year against venue year).
YEAR_TOLERANCE = 1
#: Token overlap at which a stated title names a reference.
TITLE_MATCH = 0.6
#: Resolution scores: an author-year match is worth this, plus a bonus for the
#: exact year and for a second surname that agrees.
AUTHOR_YEAR_BASE = 0.6
EXACT_YEAR_BONUS = 0.2
SECOND_AUTHOR_BONUS = 0.2


# ------------------------------------------------------------------ text


def fold(text: str) -> str:
    """Lowercase, accent-free, letters and digits only."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", plain.lower()).strip()


_TITLE_STOP = frozenset(
    "a an and are as at be by for from in into is of on or the to with via using use".split()
)


def title_tokens(title: str) -> frozenset:
    return frozenset(t for t in fold(title).split() if t not in _TITLE_STOP and len(t) > 1)


def title_similarity(left: str, right: str) -> float:
    """Share of the shorter title's content words found in the other."""
    a, b = title_tokens(left), title_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def surname(name: str) -> str:
    """The family name of "Alec Radford" or "Radford, Alec", folded."""
    name = (name or "").strip()
    if "," in name:
        name = name.split(",", 1)[0]
    parts = fold(name).split()
    return parts[-1] if parts else ""


# --------------------------------------------------------------- mentions


@dataclass(frozen=True)
class Mention:
    """What a lesson said about another work."""

    raw: str
    surnames: Tuple[str, ...] = ()
    year: Optional[int] = None
    title: str = ""


_YEAR = re.compile(r"\b((?:19|20)\d{2})[a-z]?\b")
_NAME = r"[A-Z][\w'’\-]+"
_AUTHORS = re.compile(
    rf"({_NAME})(?:\s*,?\s*(?:and|&)\s+({_NAME})|\s*,?\s+et\s+al\.?)?"
)
_MENTION_IN_PROSE = re.compile(
    rf"({_NAME}(?:\s*,?\s*(?:and|&)\s+{_NAME}|\s*,?\s+et\s+al\.?)?)\s*[,(]?\s*\(?((?:19|20)\d{{2}})[a-z]?\)?"
)


def parse_mention(work: str, title: str = "") -> Mention:
    """Author surnames and year from "Hewitt and Liang 2019" / "Voita et al. (2020)"."""
    text = (work or "").strip()
    year_match = _YEAR.search(text)
    year = int(year_match.group(1)) if year_match else None
    head = text[: year_match.start()] if year_match else text
    found = _AUTHORS.match(head.strip(" (,"))
    surnames: Tuple[str, ...] = ()
    if found:
        surnames = tuple(fold(g) for g in found.groups() if g)
        surnames = tuple(s for s in surnames if s)
    return Mention(raw=text, surnames=surnames, year=year, title=(title or "").strip())


@dataclass(frozen=True)
class Reference:
    """One work a paper cites: the columns resolution reads."""

    work_id: str
    title: str = ""
    authors: Tuple[str, ...] = ()
    year: Optional[int] = None


@dataclass(frozen=True)
class Resolution:
    """The outcome of looking one mention up. ``work_id`` is None when it did not resolve."""

    work_id: Optional[str]
    how: str  # "title", "author_year", "ambiguous", "no_match", "unparsed"
    score: float = 0.0
    candidates: Tuple[str, ...] = ()


def _score(mention: Mention, ref: Reference) -> Tuple[float, str]:
    """(score, how) for one reference; score 0 means it does not match."""
    if mention.title:
        similarity = title_similarity(mention.title, ref.title)
        if similarity >= TITLE_MATCH:
            return similarity, "title"
    if not mention.surnames or mention.year is None or ref.year is None:
        return 0.0, ""
    first = surname(ref.authors[0]) if ref.authors else ""
    if not first or first != mention.surnames[0]:
        return 0.0, ""
    gap = abs(ref.year - mention.year)
    if gap > YEAR_TOLERANCE:
        return 0.0, ""
    score = AUTHOR_YEAR_BASE
    if gap == 0:
        score += EXACT_YEAR_BONUS
    if len(mention.surnames) > 1 and len(ref.authors) > 1:
        if surname(ref.authors[1]) == mention.surnames[1]:
            score += SECOND_AUTHOR_BONUS
    return score, "author_year"


def resolve_mention(mention: Mention, references: Iterable[Reference]) -> Resolution:
    """Find ``mention`` among ``references``, or say why it cannot be found.

    The best-scoring reference wins only when it is the only one at that score:
    two references by "Zhang 2020" are an ambiguity, not a coin toss.
    """
    if not mention.surnames and not mention.title:
        return Resolution(None, "unparsed")
    scored = []
    for ref in references:
        score, how = _score(mention, ref)
        if score > 0:
            scored.append((score, how, ref.work_id))
    if not scored:
        return Resolution(None, "no_match")
    best = max(score for score, _, _ in scored)
    top = sorted((work_id, how) for score, how, work_id in scored if score == best)
    if len(top) > 1:
        return Resolution(None, "ambiguous", best, tuple(work for work, _ in top))
    work_id, how = top[0]
    return Resolution(work_id, how, round(best, 4), (work_id,))


# ----------------------------------------------------------------- edges


@dataclass(frozen=True)
class TypedEdge:
    """A connection a lesson states, resolved to a work: ``src relation dst``."""

    src: str
    dst: str
    relation: str
    why: str
    source_paper_id: str
    mention: str
    how: str
    score: float
    legacy: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "relation": self.relation,
            "why": self.why,
            "paper_id": self.source_paper_id,
            "mention": self.mention,
            "how": self.how,
            "score": self.score,
            "legacy": self.legacy,
        }


@dataclass
class ResolveReport:
    """What resolving a set of lessons did, for the CLI and the snapshot stats."""

    edges: List[TypedEdge] = field(default_factory=list)
    unresolved: List[Dict[str, Any]] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        return {
            "resolved": len(self.edges),
            "unresolved": len(self.unresolved),
            "legacy": sum(1 for edge in self.edges if edge.legacy),
        }


_RELATION_CUES: Tuple[Tuple[str, re.Pattern], ...] = (
    ("critiques", re.compile(r"critiqu|disagree|challeng|contradict|question|refut|argues against", re.I)),
    ("alternative", re.compile(r"alternative|instead of|in contrast|rival|competing", re.I)),
    ("compares", re.compile(r"compar|versus|\bvs\b|outperform|baseline|benchmark", re.I)),
    ("builds_on", re.compile(r"builds? on|extend|based on|follow|inspired|generali[sz]|adapts?\b|refines?", re.I)),
    ("uses", re.compile(r"\buses?\b|\busing\b|appl(?:y|ies)|adopts?|employs?|relies", re.I)),
)


def guess_relation(text: str) -> str:
    """The first relation whose cue words appear in ``text``, else ``background``."""
    for relation, pattern in _RELATION_CUES:
        if pattern.search(text or ""):
            return relation
    return "background"


#: Capitalised words that precede a year without being an author ("In 2019 ...").
_NOT_AN_AUTHOR = frozenset(
    "in since after before during by until around from circa early late mid as of on at for the".split()
)


def legacy_links(prose: Sequence[str]) -> List[Dict[str, Any]]:
    """Typed links from a lesson that only has prose connections.

    Every author-year mention in each line becomes one link whose ``why`` is
    the whole line. The relation is a guess from the line's words, which is why
    ``legacy`` is set: "which this paper critiques" is read correctly, "Voita
    2020 shows otherwise" is read as background.
    """
    links: List[Dict[str, Any]] = []
    for line in prose:
        if not isinstance(line, str):
            continue
        for found in _MENTION_IN_PROSE.finditer(line):
            if fold(found.group(1).split()[0]) in _NOT_AN_AUTHOR:
                continue
            links.append(
                {
                    "work": f"{found.group(1).strip()} {found.group(2)}",
                    "title": "",
                    "relation": guess_relation(line),
                    "why": line.strip(),
                    "legacy": True,
                }
            )
    return links


def lesson_links(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The typed links of one ``lessons`` row: stored ones, else read from the prose."""
    stored = row.get("connection_links")
    if isinstance(stored, list) and stored:
        return [dict(item) for item in stored if isinstance(item, dict)]
    prose = row.get("connections")
    return legacy_links(prose) if isinstance(prose, list) else []


def resolve_lessons(
    lessons: Iterable[Mapping[str, Any]],
    node_of_paper: Mapping[str, str],
    references_of: Mapping[str, Sequence[Reference]],
) -> ResolveReport:
    """Resolve every lesson's connections against its paper's references.

    ``node_of_paper`` maps a library paper id to its map node (a work id, or
    ``paper:<id>`` for a document Semantic Scholar does not know).
    ``references_of`` maps a node to the works it cites. A paper that has no
    work has no reference list, so its mentions are reported unresolved with
    the reason rather than silently dropped.
    """
    report = ResolveReport()
    seen = set()
    for row in sorted(lessons, key=lambda item: str(item.get("paper_id") or "")):
        paper_id = str(row.get("paper_id") or "")
        node = node_of_paper.get(paper_id)
        if not node:
            continue
        references = references_of.get(node, ())
        for link in lesson_links(row):
            relation = link.get("relation") if link.get("relation") in RELATIONS else "background"
            mention = parse_mention(str(link.get("work") or ""), str(link.get("title") or ""))
            if not references:
                resolution = Resolution(None, "no_references")
            else:
                resolution = resolve_mention(mention, references)
            if resolution.work_id is None or resolution.work_id == node:
                report.unresolved.append(
                    {
                        "paper_id": paper_id,
                        "mention": mention.raw or mention.title,
                        "reason": resolution.how if resolution.work_id is None else "self",
                    }
                )
                continue
            key = (node, resolution.work_id, relation)
            if key in seen:
                continue
            seen.add(key)
            report.edges.append(
                TypedEdge(
                    src=node,
                    dst=resolution.work_id,
                    relation=relation,
                    why=str(link.get("why") or "").strip(),
                    source_paper_id=paper_id,
                    mention=mention.raw or mention.title,
                    how=resolution.how,
                    score=resolution.score,
                    legacy=bool(link.get("legacy")),
                )
            )
    return report


# ------------------------------------------------------- learner decisions


class ConnectionStoreError(Exception):
    """The learner's decisions could not be read or written."""


@dataclass(frozen=True)
class Decision:
    """One confirm or reject (``learner_connections`` row)."""

    src: str
    dst: str
    relation: str
    status: str
    phrase: Optional[str] = None
    provenance: str = "user"
    updated_at: Optional[str] = None

    @property
    def pair(self) -> Tuple[str, str]:
        return pair_key(self.src, self.dst)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "relation": self.relation,
            "status": self.status,
            "phrase": self.phrase,
            "provenance": self.provenance,
            "updated_at": self.updated_at,
        }


def pair_key(left: str, right: str) -> Tuple[str, str]:
    """An unordered pair as a sorted tuple."""
    return (left, right) if left <= right else (right, left)


def _store_error(error: Any) -> ConnectionStoreError:
    text = str(error).lower()
    if any(t in text for t in ("pgrst202", "pgrst205", "does not exist", "schema cache")):
        return ConnectionStoreError(
            f"The learner's connections are not available ({error}). Apply {MIGRATION} by hand."
        )
    return ConnectionStoreError(f"Could not read or write the learner's connections: {error}")


def clean_phrase(phrase: Optional[str]) -> Optional[str]:
    """One line, no control characters, at most ``PHRASE_MAX_CHARS``; ``None`` when blank."""
    if phrase is None:
        return None
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(phrase))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    if len(text) > PHRASE_MAX_CHARS:
        raise ValueError(f"The phrase is {len(text)} characters; the limit is {PHRASE_MAX_CHARS}.")
    return text


def _decision(row: Mapping[str, Any]) -> Decision:
    return Decision(
        src=str(row["src"]),
        dst=str(row["dst"]),
        relation=str(row.get("relation") or "related"),
        status=str(row["status"]),
        phrase=row.get("phrase"),
        provenance=str(row.get("provenance") or "user"),
        updated_at=str(row["updated_at"]) if row.get("updated_at") else None,
    )


def load_decisions() -> List[Decision]:
    """Every decision the learner has made."""
    client = db.get_client()
    out: List[Decision] = []
    offset = 0
    while True:
        response = (
            client.table("learner_connections")
            .select("src,dst,relation,status,phrase,provenance,updated_at")
            .order("src,dst")
            .limit(1000)
            .offset(offset)
            .execute()
        )
        if response["error"]:
            raise _store_error(response["error"])
        page = response["data"] or []
        out.extend(_decision(row) for row in page)
        if len(page) < 1000:
            return out
        offset += len(page)


def put_decision(
    src: str,
    dst: str,
    *,
    status: str,
    relation: str = "related",
    phrase: Optional[str] = None,
) -> Decision:
    """Confirm (optionally with an edited relation and phrase) or reject a pair."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")
    if relation not in RELATIONS:
        raise ValueError(f"relation must be one of {', '.join(RELATIONS)}")
    if not src.strip() or not dst.strip() or src.strip() == dst.strip():
        raise ValueError("A connection needs two different works.")
    response = db.get_client().rpc(
        "put_learner_connection",
        {
            "p_src": src.strip(),
            "p_dst": dst.strip(),
            "p_relation": relation,
            "p_status": status,
            "p_phrase": clean_phrase(phrase),
        },
    )
    if response["error"]:
        raise _store_error(response["error"])
    data = response["data"]
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict):
        raise ConnectionStoreError("The decision was not stored: the database returned no row.")
    return _decision(data)


def delete_decision(a: str, b: str) -> int:
    """Forget the decision for a pair, returning it to "suggested". Returns rows removed."""
    response = db.get_client().rpc("delete_learner_connection", {"p_a": a.strip(), "p_b": b.strip()})
    if response["error"]:
        raise _store_error(response["error"])
    data = response["data"]
    return int(data[0] if isinstance(data, list) and data else data or 0)
