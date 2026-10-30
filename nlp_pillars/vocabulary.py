"""Controlled vocabularies concepts are anchored to.

Concepts (``concepts`` table, migrations 020 and 024) are free text, so the
same idea arrives under several names ("bleu", "bleu score") and nothing says
that "multi-head attention" is a kind of attention. Anchoring ties a concept
to a field of a published vocabulary, which gives it a stable id, merges
surface variants, and brings the vocabulary's hierarchy with it.

What ships: the NLP-KG fields of study (Schopf & Matthes, ACL 2024 demo; MIT
licence), 421 fields and 583 synonyms, in ``data/nlp_kg_fields.json``. It is
regenerated from the project's OWL file by ``scripts/build_nlp_kg_vocabulary.py``.

What does not: the Papers with Code archive (methods, tasks, datasets). Its
licence is CC BY-SA 4.0 and its 8,725 methods would make this repository's
data share-alike. ``VOCABULARY_FILES`` accepts further files in the same shape
(``{"vocabulary": key, "fields": [{id, label, synonyms, description, level,
parents}]}``), so a local conversion of that archive anchors exactly like
NLP-KG does, under its own key.

This module reads files and answers lookups. It calls no model and no network.
Retrieval of candidates by embedding takes the embedder as an argument.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import get_settings
from .graph import canonicalize_term

logger = logging.getLogger(__name__)

BUNDLED_PATH = Path(__file__).resolve().parent / "vocab" / "nlp_kg_fields.json"
BUNDLED_KEY = "nlp-kg"

Embed = Callable[[Sequence[str]], List[List[float]]]


class VocabularyError(Exception):
    """A vocabulary file is missing or is not the shape this module reads."""


@dataclass(frozen=True)
class VocabField:
    """One field of one vocabulary."""

    vocab: str
    id: str
    label: str
    synonyms: Tuple[str, ...] = ()
    description: str = ""
    level: Optional[int] = None
    parents: Tuple[str, ...] = ()

    @property
    def anchor(self) -> str:
        """``vocab:id``, the single string a concept row's anchor is kept as."""
        return f"{self.vocab}:{self.id}"

    def text(self) -> str:
        """What is embedded for retrieval: the label, then what it means."""
        return f"{self.label}. {self.description}".strip() if self.description else self.label


class Vocabulary:
    """A vocabulary's fields, with exact lookup and ancestry."""

    def __init__(
        self,
        key: str,
        name: str,
        fields: Iterable[VocabField],
        *,
        licence: str = "",
        source: str = "",
    ):
        self.key = key
        self.name = name
        self.licence = licence
        self.source = source
        self.fields: Dict[str, VocabField] = {field.id: field for field in fields}
        self._by_name: Dict[str, str] = {}
        # Shallower fields win a name clash: "Attention" the field of study
        # over a deeper field that lists "attention" as a synonym.
        for field in sorted(
            self.fields.values(),
            key=lambda f: (f.level if f.level is not None else 99, f.id),
        ):
            for name_ in (field.label, *field.synonyms):
                canonical = canonicalize_term(name_)
                if canonical:
                    self._by_name.setdefault(canonical, field.id)

    # ------------------------------------------------------------------ read

    @classmethod
    def from_file(cls, path: Path) -> "Vocabulary":
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise VocabularyError(f"Cannot read vocabulary {path}: {exc}") from exc
        key = str(document.get("vocabulary") or "").strip()
        rows = document.get("fields")
        if not key or not isinstance(rows, list):
            raise VocabularyError(
                f"{path} is not a vocabulary file: it needs a 'vocabulary' key and a 'fields' list."
            )
        fields = []
        for row in rows:
            identifier = str(row.get("id") or "").strip()
            label = str(row.get("label") or "").strip()
            if not identifier or not label:
                raise VocabularyError(f"{path} has a field without an id or a label: {row!r}")
            fields.append(
                VocabField(
                    vocab=key,
                    id=identifier,
                    label=label,
                    synonyms=tuple(str(s) for s in row.get("synonyms") or ()),
                    description=str(row.get("description") or ""),
                    level=row.get("level") if isinstance(row.get("level"), int) else None,
                    parents=tuple(str(p) for p in row.get("parents") or ()),
                )
            )
        return cls(
            key,
            str(document.get("name") or key),
            fields,
            licence=str(document.get("licence") or ""),
            source=str(document.get("source") or ""),
        )

    def __len__(self) -> int:
        return len(self.fields)

    def get(self, identifier: str) -> Optional[VocabField]:
        return self.fields.get(identifier)

    def lookup(self, name: str) -> Optional[VocabField]:
        """The field whose label or synonym is ``name`` once both are canonicalised."""
        identifier = self._by_name.get(canonicalize_term(name))
        return self.fields.get(identifier) if identifier else None

    def ancestors(self, identifier: str) -> List[VocabField]:
        """Every ancestor, nearest first. The first parent leads at each step.

        A field can have several parents (102 of NLP-KG's 421 do). Breadth-first
        from the first parent keeps the order stable, and a field is listed once.
        """
        seen = {identifier}
        out: List[VocabField] = []
        queue = [identifier]
        while queue:
            current = self.fields.get(queue.pop(0))
            if current is None:
                continue
            for parent in current.parents:
                if parent not in seen and parent in self.fields:
                    seen.add(parent)
                    out.append(self.fields[parent])
                    queue.append(parent)
        return out

    def path(self, identifier: str) -> List[str]:
        """Labels from the root down to the field, along first parents."""
        labels: List[str] = []
        current = self.fields.get(identifier)
        seen = set()
        while current is not None and current.id not in seen:
            seen.add(current.id)
            labels.append(current.label)
            current = self.fields.get(current.parents[0]) if current.parents else None
        return list(reversed(labels))


class VocabularySet:
    """The vocabularies in use, addressed by ``vocab:id`` anchors."""

    def __init__(self, vocabularies: Iterable[Vocabulary]):
        self.vocabularies: Dict[str, Vocabulary] = {v.key: v for v in vocabularies}

    def get(self, anchor: str) -> Optional[VocabField]:
        key, _, identifier = (anchor or "").partition(":")
        vocabulary = self.vocabularies.get(key)
        return vocabulary.get(identifier) if vocabulary else None

    def lookup(self, name: str) -> Optional[VocabField]:
        """Exact label/synonym match, vocabularies tried in load order."""
        for vocabulary in self.vocabularies.values():
            found = vocabulary.lookup(name)
            if found is not None:
                return found
        return None

    def ancestors(self, anchor: str) -> List[VocabField]:
        key, _, identifier = (anchor or "").partition(":")
        vocabulary = self.vocabularies.get(key)
        return vocabulary.ancestors(identifier) if vocabulary else []

    def all_fields(self) -> List[VocabField]:
        return [
            field
            for vocabulary in self.vocabularies.values()
            for field in vocabulary.fields.values()
        ]


@lru_cache(maxsize=4)
def _load(paths: Tuple[str, ...]) -> VocabularySet:
    vocabularies = [Vocabulary.from_file(Path(path)) for path in paths]
    return VocabularySet(vocabularies)


def load_vocabularies(extra_paths: Optional[Sequence[str]] = None) -> VocabularySet:
    """The bundled NLP-KG vocabulary plus any ``VOCABULARY_FILES``.

    ``extra_paths`` overrides the setting (tests). A listed file that cannot be
    read raises: a vocabulary that silently fails to load would leave every
    concept it should have anchored unanchored with nothing saying why.
    """
    if extra_paths is None:
        raw = getattr(get_settings(), "vocabulary_files", None)
        extra_paths = [p.strip() for p in raw.split(",") if p.strip()] if isinstance(raw, str) else []
    return _load((str(BUNDLED_PATH), *map(str, extra_paths)))


# ------------------------------------------------------------- retrieval


class CandidateIndex:
    """Embedded fields, for the top-k nearest to a concept name.

    Built once per process from the vocabularies. ``embed`` is the local bge
    provider's ``embed_documents`` in the app; tests pass a fake. Both the
    fields and the query are embedded as plain text: a concept name and a field
    label are the same kind of short phrase, so the query prefix bge uses for
    questions would only distort the match.
    """

    def __init__(self, vocabularies: VocabularySet, embed: Embed):
        self.fields = vocabularies.all_fields()
        self._embed = embed
        vectors = np.asarray(embed([f.text() for f in self.fields]), dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = vectors / norms

    def top(self, name: str, k: int = 5) -> List[Tuple[VocabField, float]]:
        vector = np.asarray(self._embed([name])[0], dtype=np.float64)
        norm = np.linalg.norm(vector) or 1.0
        scores = self._matrix @ (vector / norm)
        order = np.lexsort((np.arange(len(self.fields)), -scores))[:k]
        return [(self.fields[int(i)], float(scores[int(i)])) for i in order]
