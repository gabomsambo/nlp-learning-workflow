"""Work identity: one normal form for every id a scholarly work goes by.

A library paper is stored as ``arxiv:1706.03762`` while Semantic Scholar names
the same work ``1706.03762`` (or a DOI, or a 40-hex ``paperId``). Comparing
those strings is how the library came to hold zero citations between its own
papers. This module turns any of them into a ``(kind, value)`` key, so two
spellings of one id are the same key and the ``work_ids`` table can map every
key to one work.

Pure functions, no I/O. ``paper_ids`` stays the home of "can this be downloaded";
this is the home of "which work is this".
"""

from __future__ import annotations

import re
from typing import Iterable, List, Mapping, Optional, Tuple

from .paper_ids import ARXIV_ID_RE, extract_arxiv_id, extract_doi

KIND_S2 = "s2"
KIND_ARXIV = "arxiv"
KIND_DOI = "doi"
KIND_ACL = "acl"
KIND_CORPUS = "corpus"
KIND_TITLE_HASH = "title_hash"

#: Mirrors the CHECK on ``work_ids.kind`` in docs/migrations/031_works_identity.sql.
KINDS = (KIND_S2, KIND_ARXIV, KIND_DOI, KIND_ACL, KIND_CORPUS, KIND_TITLE_HASH)

#: Display preference, matching what ``SemanticScholarTool._convert_to_paper_ref``
#: has always named an external paper: arXiv id, else DOI, else S2 id.
DISPLAY_ORDER = (KIND_ARXIV, KIND_DOI, KIND_S2)

#: Kinds Semantic Scholar's API accepts as a lookup id, best first.
LOOKUP_ORDER = (KIND_S2, KIND_ARXIV, KIND_DOI, KIND_ACL, KIND_CORPUS)

WorkKey = Tuple[str, str]

_S2_RE = re.compile(r"^[0-9a-f]{40}$")
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
_VERSION_RE = re.compile(r"v\d+$")
_PREFIXES = {
    "arxiv": KIND_ARXIV,
    "doi": KIND_DOI,
    "s2": KIND_S2,
    "acl": KIND_ACL,
    "corpusid": KIND_CORPUS,
    "corpus": KIND_CORPUS,
}


def normalise_arxiv(raw: str) -> Optional[str]:
    """Bare, versionless arXiv id, or None when ``raw`` is not one.

    Old-style ids keep the shape arXiv prints (``hep-th/9901001``,
    ``math.GT/0309136``) so the same paper written two ways is one key.
    """
    text = _VERSION_RE.sub("", raw.strip())
    if not text:
        return None
    if "/" in text:
        archive, _, number = text.partition("/")
        name, dot, subject = archive.partition(".")
        text = f"{name.lower()}{dot}{subject.upper()}/{number}"
    return text if ARXIV_ID_RE.match(text) else None


def normalise_doi(raw: str) -> Optional[str]:
    """Lowercased DOI without a resolver URL or trailing punctuation."""
    text = raw.strip()
    if "doi.org/" in text.lower():
        text = extract_doi(text) or ""
    text = text.rstrip(".").rstrip("/").lower()
    return text if _DOI_RE.match(text) else None


def normalise_s2(raw: str) -> Optional[str]:
    """A Semantic Scholar ``paperId``: 40 lowercase hex characters."""
    text = raw.strip().lower()
    return text if _S2_RE.match(text) else None


def _normalise_acl(raw: str) -> Optional[str]:
    text = raw.strip().lower()
    return text or None


def _normalise_corpus(raw: object) -> Optional[str]:
    text = str(raw).strip()
    return str(int(text)) if text.isdigit() else None


def parse_paper_id(raw: object) -> Optional[WorkKey]:
    """Key for whatever id string the app holds, or None if it names no work.

    Accepts ``arxiv:1706.03762``, ``1706.03762v5``, ``arXiv.org`` and doi.org
    URLs, ``doi:10.…``, a bare DOI and a bare S2 ``paperId``. ``file:`` uploads
    and anything else return None: an id that identifies nothing must not be
    turned into a work.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or text.startswith("file:"):
        return None

    url_arxiv = extract_arxiv_id(text)
    if url_arxiv:
        return (KIND_ARXIV, normalise_arxiv(url_arxiv) or url_arxiv)
    if "doi.org/" in text.lower():
        doi = normalise_doi(text)
        return (KIND_DOI, doi) if doi else None

    prefix, sep, rest = text.partition(":")
    kind = _PREFIXES.get(prefix.strip().lower()) if sep else None
    if kind:
        return _key_of_kind(kind, rest)

    arxiv = normalise_arxiv(text)
    if arxiv:
        return (KIND_ARXIV, arxiv)
    doi = normalise_doi(text)
    if doi:
        return (KIND_DOI, doi)
    s2 = normalise_s2(text)
    if s2:
        return (KIND_S2, s2)
    return None


def _key_of_kind(kind: str, value: str) -> Optional[WorkKey]:
    normal = {
        KIND_ARXIV: normalise_arxiv,
        KIND_DOI: normalise_doi,
        KIND_S2: normalise_s2,
        KIND_ACL: _normalise_acl,
        KIND_CORPUS: _normalise_corpus,
    }[kind](value)
    return (kind, normal) if normal else None


def keys_from_s2(
    paper_id: Optional[str],
    external_ids: Optional[Mapping[str, object]] = None,
    corpus_id: object = None,
) -> List[WorkKey]:
    """Every key one Semantic Scholar record carries, S2 id first.

    ``externalIds`` also lists MAG, DBLP and PubMed ids. They are not stored:
    nothing in the app looks a work up by them.
    """
    external = external_ids or {}
    found: List[Optional[WorkKey]] = []
    if paper_id:
        found.append(_key_of_kind(KIND_S2, str(paper_id)))
    for kind, field in (
        (KIND_ARXIV, "ArXiv"),
        (KIND_DOI, "DOI"),
        (KIND_ACL, "ACL"),
    ):
        value = external.get(field)
        if value:
            found.append(_key_of_kind(kind, str(value)))
    corpus = corpus_id if corpus_id not in (None, "") else external.get("CorpusId")
    if corpus not in (None, ""):
        found.append(_key_of_kind(KIND_CORPUS, str(corpus)))
    return _unique(found)


def _unique(keys: Iterable[Optional[WorkKey]]) -> List[WorkKey]:
    out: List[WorkKey] = []
    for key in keys:
        if key and key not in out:
            out.append(key)
    return out


def s2_lookup_id(key: WorkKey) -> Optional[str]:
    """The id Semantic Scholar's API accepts for this key, or None."""
    kind, value = key
    return {
        KIND_S2: value,
        KIND_ARXIV: f"ARXIV:{value}",
        KIND_DOI: f"DOI:{value}",
        KIND_ACL: f"ACL:{value}",
        KIND_CORPUS: f"CorpusId:{value}",
    }.get(kind)


def best_lookup_key(keys: Iterable[WorkKey]) -> Optional[WorkKey]:
    """The key to ask Semantic Scholar about: S2 id, then arXiv, DOI, ACL."""
    usable = [key for key in keys if key[0] in LOOKUP_ORDER]
    usable.sort(key=lambda key: LOOKUP_ORDER.index(key[0]))
    return usable[0] if usable else None


def display_id(keys: Iterable[WorkKey]) -> Optional[str]:
    """Bare id the rest of the app shows for an external work."""
    by_kind = {}
    for kind, value in keys:
        by_kind.setdefault(kind, value)
    for kind in DISPLAY_ORDER:
        if kind in by_kind:
            return by_kind[kind]
    return None
