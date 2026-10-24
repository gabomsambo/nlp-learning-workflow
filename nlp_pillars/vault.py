"""Obsidian vault export. A projection of the library and the graph.

Postgres stays the source of truth. This module turns a snapshot into
Markdown notes and writes them only when the bytes differ. It does not
read the network, embed text, or delete a note outright.

One note per paper, concept, pillar, author and episode. Wikilinks for
cites, covers, related-by-embedding, pillar membership and authorship
follow ``graph_edges``. A paper note also carries its abstract, notes,
lesson, quiz cards and episode links.

Generated text sits between ``<!-- nlp:generated -->`` and
``<!-- /nlp:generated -->``. Front matter is above the start marker so
Obsidian still reads it, and it is regenerated with the body. Everything
after the end marker is the user's block and is copied back unchanged.
A managed note whose subject is gone moves to ``_archive/``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

START_MARK = "<!-- nlp:generated -->"
END_MARK = "<!-- /nlp:generated -->"

FOLDERS = {
    "paper": "Papers",
    "concept": "Concepts",
    "pillar": "Pillars",
    "author": "Authors",
    "episode": "Episodes",
}

_UNSAFE = re.compile(r"[\\/:*?\"<>|#^\[\]]+")
_SPACE = re.compile(r"\s+")
_UUID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")
_EMBEDDING = "derived:embedding:"


class VaultExportError(Exception):
    """The vault directory could not be written."""


@dataclass
class VaultSnapshot:
    """Rows the exporter reads. Dicts, so a test can build one without a database."""

    papers: list = field(default_factory=list)
    pillars: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    lessons: list = field(default_factory=list)
    cards: list = field(default_factory=list)
    episodes: list = field(default_factory=list)
    concepts: list = field(default_factory=list)
    edges: list = field(default_factory=list)


@dataclass(frozen=True)
class NoteDraft:
    """One note before its filename is chosen."""

    type: str
    id: str
    title: str


@dataclass
class ExportReport:
    """What one export did, or would do when ``dry_run`` is set.

    ``changed`` counts notes whose bytes differ from the file on disk.
    A dry run leaves the disk alone, so that count is "would write".
    A second run with the same snapshot and the same user blocks reports
    ``changed == 0`` and ``archived == 0``.
    """

    dry_run: bool
    notes: int
    by_type: dict
    changed: int
    unchanged: int
    archived: int
    kept: int

    def render(self) -> str:
        """One short report. Dry-run says "would", because nothing was written."""
        kind = "vault export (dry-run)" if self.dry_run else "vault export"
        write_label = "would write" if self.dry_run else "written"
        archive_label = "would archive" if self.dry_run else "archived"
        counts = ", ".join(
            f"{key} {self.by_type[key]}" for key in sorted(self.by_type)
        )
        lines = [
            f"{kind}: {self.notes} notes ({counts})",
            f"{write_label}: {self.changed}",
            f"unchanged: {self.unchanged}",
            f"{archive_label}: {self.archived}",
        ]
        if self.kept:
            label = "would keep" if self.dry_run else "kept"
            lines.append(
                f"{label}: {self.kept} (a file without the generated marker)"
            )
        return "\n".join(lines)


def resolve_vault_dir(value: Optional[str]) -> Path:
    """Directory the scheduler writes. Blank means ``~/NLP-Vault``."""
    raw = (value or "").strip() or "~/NLP-Vault"
    return Path(raw).expanduser()


def show(value: Any) -> str:
    """Text that is safe to place inside a generated section.

    A paper abstract that contains the end marker would otherwise split the
    user block on the next read. Break the marker token and leave the words.
    """
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(END_MARK, "<!-- /nlp:generated - ->")
    text = text.replace(START_MARK, "<!-- nlp:generated - ->")
    return text


def _list_items(value: list) -> list:
    items = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
        elif isinstance(item, dict):
            raw = item.get("name") or item.get("author")
            if isinstance(raw, str) and raw.strip():
                items.append(raw.strip())
    return items


def _strings_from_text(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    if not text.startswith("["):
        return [text]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return _list_items(parsed)
    return [text]


def as_list(value: Any) -> list:
    """Return strings from a JSON list, a JSON string, or one plain string."""
    if isinstance(value, str):
        return _strings_from_text(value)
    if isinstance(value, list):
        return _list_items(value)
    return []


def _yaml_value(value: Any) -> str:
    """JSON flow, which is valid YAML. Strings stay quoted.

    ``show`` runs first so a property cannot contain the end marker and
    split the user block on the next read.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return json.dumps(show(value), ensure_ascii=False)
    if isinstance(value, list):
        cleaned = [
            show(item) if isinstance(item, str) else item for item in value
        ]
        return json.dumps(cleaned, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)


def _front_matter(fields: list) -> str:
    lines = ["---"]
    for key, value in fields:
        if value is None:
            continue
        if value == [] and key == "aliases":
            continue
        lines.append(f"{key}: {_yaml_value(value)}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _heading(title: str) -> str:
    text = show(title).split("\n", 1)[0].strip().lstrip("#").strip()
    return text or "untitled"


def safe_stem(title: str) -> str:
    """Filename stem. Characters Obsidian treats specially are removed."""
    text = _heading(title)
    text = _UNSAFE.sub(" ", text)
    text = _SPACE.sub(" ", text).strip(" .")
    if not text or text in {".", ".."}:
        text = "untitled"
    if len(text) > 80:
        text = text[:80].rstrip(" .")
    return text or "untitled"


def short_id(entity_id: str) -> str:
    """Stable suffix when two notes would share a title.

    UUIDs keep their first eight hex digits. Anything else keeps a
    filesystem-safe form of the id, truncated to a fixed window so a long
    id cannot push the filename around on each run.
    """
    if _UUID.match(entity_id):
        return entity_id[:8].lower()
    cleaned = _SAFE_ID.sub("-", entity_id).strip("-._")
    if not cleaned:
        return "id"
    if len(cleaned) <= 24:
        return cleaned
    return f"{cleaned[:12]}-{cleaned[-8:]}"


def _sid(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _with_suffix(base: str, suffix: str, taken: set) -> str:
    candidate = f"{base} ({suffix})"
    number = 2
    while candidate.casefold() in taken:
        candidate = f"{base} ({suffix}-{number})"
        number += 1
    taken.add(candidate.casefold())
    return candidate


def _stems_for_group(group: list, preferred: dict) -> dict:
    if len(group) == 1:
        draft = group[0]
        return {(draft.type, draft.id): preferred[(draft.type, draft.id)]}
    taken = set()
    stems = {}
    for draft in group:
        key = (draft.type, draft.id)
        stems[key] = _with_suffix(preferred[key], short_id(draft.id), taken)
    return stems


def _break_stem_clashes(stems: dict) -> dict:
    """Rename later notes when a suffixed title matches another note."""
    for _ in range(len(stems) + 2):
        folded = defaultdict(list)
        for key, stem in stems.items():
            folded[stem.casefold()].append(key)
        clashes = [keys for keys in folded.values() if len(keys) > 1]
        if not clashes:
            return stems
        for keys in clashes:
            for key in sorted(keys)[1:]:
                taken = {stems[other].casefold() for other in stems if other != key}
                stems[key] = _with_suffix(stems[key], short_id(key[1]), taken)
    return stems


def assign_stems(drafts: Iterable[NoteDraft]) -> dict:
    """Map ``(type, id)`` to a vault-wide unique note name.

    A title used once keeps that name. A title used twice gets the id
    appended on every note that shares it, so which file keeps the bare
    title does not depend on arrival order. Input order does not matter.
    """
    ordered = sorted(drafts, key=lambda draft: (draft.type, draft.id))
    preferred = {
        (draft.type, draft.id): safe_stem(draft.title) for draft in ordered
    }
    groups = defaultdict(list)
    for draft in ordered:
        groups[preferred[(draft.type, draft.id)].casefold()].append(draft)
    stems = {}
    for group in groups.values():
        stems.update(_stems_for_group(group, preferred))
    return _break_stem_clashes(stems)


def _link(stem: Optional[str], fallback: str) -> str:
    if stem:
        return f"[[{stem}]]"
    cleaned = (
        show(fallback).replace("[", "").replace("]", "").replace("|", "-").strip()
    )
    return f"[[{cleaned or 'untitled'}]]"


def _aliases(stem: str, extras: Iterable[str]) -> list:
    seen = set()
    found = []
    for item in extras:
        text = show(item).strip()
        if not text or text == stem or text in seen:
            continue
        seen.add(text)
        found.append(text)
    found.sort(key=lambda item: (item.casefold(), item))
    return found


def _tags(*parts: str) -> list:
    seen = []
    for part in parts:
        text = show(part).strip()
        if text and text not in seen:
            seen.append(text)
    return seen


def _section(heading: str, body: str) -> str:
    body = body.strip("\n")
    if not body:
        return ""
    return f"## {heading}\n\n{body}"


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _outgoing(edges: list, src_kind: str, src_id: str, kind: str) -> list:
    rows = [
        edge
        for edge in edges
        if edge.get("kind") == kind
        and edge.get("src_kind") == src_kind
        and _sid(edge.get("src_id")) == src_id
    ]
    rows.sort(key=lambda edge: (_sid(edge.get("dst_id")), _sid(edge.get("dst_kind"))))
    return rows


def _incoming(edges: list, dst_kind: str, dst_id: str, kind: str) -> list:
    rows = [
        edge
        for edge in edges
        if edge.get("kind") == kind
        and edge.get("dst_kind") == dst_kind
        and _sid(edge.get("dst_id")) == dst_id
    ]
    rows.sort(key=lambda edge: (_sid(edge.get("src_id")), _sid(edge.get("src_kind"))))
    return rows


def _reason(edge: dict) -> str:
    provenance = _sid(edge.get("provenance"))
    model = provenance[len(_EMBEDDING) :] if provenance.startswith(_EMBEDDING) else ""
    number = ""
    weight = edge.get("weight")
    if weight is not None and weight != "":
        number = f" {float(weight):.6f}"
    if model:
        return f"related-by-embedding{number} ({model})"
    if provenance:
        return f"related-by-embedding{number} ({provenance})"
    return f"related-by-embedding{number}".strip()


def _document(fields: list, title: str, blocks: list) -> str:
    body_parts = [f"# {_heading(title)}"]
    body_parts.extend(block for block in blocks if block)
    body = "\n\n".join(body_parts).rstrip() + "\n"
    return f"{_front_matter(fields)}{START_MARK}\n{body}{END_MARK}"


def _grouped(rows: list, key: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        ident = _sid(row.get(key))
        if ident:
            grouped[ident].append(row)
    for ident in grouped:
        grouped[ident].sort(key=lambda row: _sid(row.get("id")))
    return grouped


def _index_rows(rows: list, key: str = "id") -> dict:
    chosen = {}
    ordered = sorted(
        rows,
        key=lambda row: (_sid(row.get(key)), _sid(row.get("title") or row.get("name"))),
    )
    for row in ordered:
        ident = _sid(row.get(key))
        if ident:
            chosen[ident] = row
    return chosen


def _note_body(note: dict) -> str:
    parts = []
    problem = show(note.get("problem") or "").strip()
    method = show(note.get("method") or "").strip()
    if problem:
        parts.append(f"**Problem**\n\n{problem}")
    if method:
        parts.append(f"**Method**\n\n{method}")
    for label, key in (
        ("Findings", "findings"),
        ("Limitations", "limitations"),
        ("Future work", "future_work"),
    ):
        items = [show(item) for item in as_list(note.get(key))]
        if items:
            parts.append(f"**{label}**\n\n{_bullets(items)}")
    return "\n\n".join(parts)


def _callout(question: str, answer: str) -> str:
    title = " ".join(show(question).split())
    lines = [f"> [!faq]- {title}"]
    answer_text = show(answer).strip("\n")
    if not answer_text:
        lines.append(">")
        return "\n".join(lines)
    for line in answer_text.split("\n"):
        lines.append(f"> {line}" if line else ">")
    return "\n".join(lines)


def _paper_fields(paper, stem, stems, edges) -> list:
    paper_id = _sid(paper.get("id"))
    pillar_edges = _outgoing(edges, "paper", paper_id, "in_pillar")
    author_edges = _outgoing(edges, "paper", paper_id, "authored_by")
    concept_edges = _outgoing(edges, "paper", paper_id, "covers")
    pillar_links = [
        _link(stems.get(("pillar", _sid(edge.get("dst_id")))), _sid(edge.get("dst_id")))
        for edge in pillar_edges
    ]
    author_links = [
        _link(stems.get(("author", _sid(edge.get("dst_id")))), _sid(edge.get("dst_id")))
        for edge in author_edges
    ]
    concept_links = [
        _link(
            stems.get(("concept", _sid(edge.get("dst_id")))),
            _sid(edge.get("dst_id")),
        )
        for edge in concept_edges
    ]
    slugs = [_sid(edge.get("dst_id")) for edge in pillar_edges]
    year = paper.get("year")
    year_value = year if isinstance(year, int) and not isinstance(year, bool) else None
    if year_value is None and isinstance(year, str) and year.strip().isdigit():
        year_value = int(year.strip())
    return [
        ("type", "paper"),
        ("id", paper_id),
        ("pillar", pillar_links[0] if pillar_links else None),
        ("year", year_value),
        ("authors", author_links),
        ("concepts", concept_links),
        ("aliases", _aliases(stem, [paper_id])),
        ("tags", _tags("paper", *slugs)),
    ]


def _paper_blocks(paper, stems, snapshot, edges) -> list:
    paper_id = _sid(paper.get("id"))
    blocks = []
    lessons = _grouped(snapshot.lessons, "paper_id").get(paper_id, [])
    summaries = [
        show(lesson.get("tl_dr") or "").strip()
        for lesson in lessons
        if show(lesson.get("tl_dr") or "").strip()
    ]
    if summaries:
        blocks.append(f"> **TL;DR** {summaries[0]}")

    abstract = show(paper.get("abstract") or "").strip()
    blocks.append(_section("Abstract", abstract))

    note_rows = _grouped(snapshot.notes, "paper_id").get(paper_id, [])
    note_chunks = []
    for note in note_rows:
        body = _note_body(note)
        if not body:
            continue
        if len(note_rows) == 1:
            note_chunks.append(body)
        else:
            note_chunks.append(f"### {_sid(note.get('id'))}\n\n{body}")
    blocks.append(_section("Notes", "\n\n".join(note_chunks)))

    takeaways = []
    for lesson in lessons:
        takeaways.extend(show(item) for item in as_list(lesson.get("takeaways")))
    if takeaways:
        blocks.append(_section("Takeaways", _bullets(takeaways)))

    lesson_lines = [f"`{_sid(lesson.get('id'))}`" for lesson in lessons]
    blocks.append(_section("Lesson", _bullets(lesson_lines)))

    author_edges = _outgoing(edges, "paper", paper_id, "authored_by")
    blocks.append(
        _section(
            "Authors",
            _bullets(
                _link(
                    stems.get(("author", _sid(edge.get("dst_id")))),
                    _sid(edge.get("dst_id")),
                )
                for edge in author_edges
            ),
        )
    )
    pillar_edges = _outgoing(edges, "paper", paper_id, "in_pillar")
    blocks.append(
        _section(
            "Pillar",
            _bullets(
                _link(
                    stems.get(("pillar", _sid(edge.get("dst_id")))),
                    _sid(edge.get("dst_id")),
                )
                for edge in pillar_edges
            ),
        )
    )
    concept_edges = _outgoing(edges, "paper", paper_id, "covers")
    blocks.append(
        _section(
            "Concepts",
            _bullets(
                _link(
                    stems.get(("concept", _sid(edge.get("dst_id")))),
                    _sid(edge.get("dst_id")),
                )
                for edge in concept_edges
            ),
        )
    )

    related = _outgoing(edges, "paper", paper_id, "related_to")
    related.sort(
        key=lambda edge: (
            -float(edge.get("weight") or 0),
            _sid(edge.get("dst_id")),
        )
    )
    blocks.append(
        _section(
            "Related",
            _bullets(
                "{link} - {reason}".format(
                    link=_link(
                        stems.get(("paper", _sid(edge.get("dst_id")))),
                        _sid(edge.get("dst_id")),
                    ),
                    reason=_reason(edge),
                )
                for edge in related
            ),
        )
    )
    cites = _outgoing(edges, "paper", paper_id, "cites")
    blocks.append(
        _section(
            "Cites",
            _bullets(
                _link(
                    stems.get(("paper", _sid(edge.get("dst_id")))),
                    _sid(edge.get("dst_id")),
                )
                for edge in cites
            ),
        )
    )
    cited_by = [
        edge
        for edge in _incoming(edges, "paper", paper_id, "cites")
        if edge.get("src_kind") in {"paper", "external_paper"}
    ]
    blocks.append(
        _section(
            "Cited by",
            _bullets(
                _link(
                    stems.get(("paper", _sid(edge.get("src_id")))),
                    _sid(edge.get("src_id")),
                )
                for edge in cited_by
            ),
        )
    )

    episodes = _grouped(snapshot.episodes, "paper_id").get(paper_id, [])
    blocks.append(
        _section(
            "Episodes",
            _bullets(
                _link(
                    stems.get(("episode", _sid(episode.get("id")))),
                    _sid(episode.get("id")),
                )
                for episode in episodes
            ),
        )
    )
    cards = _grouped(snapshot.cards, "paper_id").get(paper_id, [])
    if cards:
        blocks.append(
            _section(
                "Quiz cards",
                "\n\n".join(
                    _callout(
                        _sid(card.get("question")),
                        _sid(card.get("answer")),
                    )
                    for card in cards
                ),
            )
        )
    return blocks


def _concept_drafts(snapshot: VaultSnapshot) -> list:
    names = {}
    for row in snapshot.concepts:
        name = _sid(row.get("name"))
        if name:
            names[name] = name
    for edge in snapshot.edges:
        if edge.get("kind") != "covers" or edge.get("dst_kind") != "concept":
            continue
        name = _sid(edge.get("dst_id"))
        if name:
            names.setdefault(name, name)
    return [NoteDraft("concept", name, name) for name in sorted(names)]


def _author_drafts(snapshot: VaultSnapshot) -> list:
    authors = set()
    for edge in snapshot.edges:
        if edge.get("kind") != "authored_by" or edge.get("dst_kind") != "author":
            continue
        author = _sid(edge.get("dst_id"))
        if author:
            authors.add(author)
    return [NoteDraft("author", author, author) for author in sorted(authors)]


def _drafts(snapshot: VaultSnapshot) -> list:
    drafts = []
    for paper in _index_rows(snapshot.papers).values():
        drafts.append(
            NoteDraft(
                "paper",
                _sid(paper.get("id")),
                _sid(paper.get("title")) or _sid(paper.get("id")),
            )
        )
    drafts.extend(_concept_drafts(snapshot))
    for pillar in _index_rows(snapshot.pillars).values():
        drafts.append(
            NoteDraft(
                "pillar",
                _sid(pillar.get("id")),
                _sid(pillar.get("name")) or _sid(pillar.get("id")),
            )
        )
    drafts.extend(_author_drafts(snapshot))
    for episode in _index_rows(snapshot.episodes).values():
        drafts.append(
            NoteDraft(
                "episode",
                _sid(episode.get("id")),
                _sid(episode.get("title")) or _sid(episode.get("id")),
            )
        )
    return drafts


def _render_one(draft: NoteDraft, stem: str, stems: dict, snapshot, edges) -> str:
    if draft.type == "paper":
        paper = _index_rows(snapshot.papers)[draft.id]
        return _document(
            _paper_fields(paper, stem, stems, edges),
            _sid(paper.get("title")) or draft.id,
            _paper_blocks(paper, stems, snapshot, edges),
        )
    if draft.type == "concept":
        row = next(
            (
                item
                for item in snapshot.concepts
                if _sid(item.get("name")) == draft.id
            ),
            {},
        )
        papers = _incoming(edges, "concept", draft.id, "covers")
        description = show(row.get("description") or "").strip()
        blocks = []
        if description:
            blocks.append(description)
        blocks.append(
            _section(
                "Papers",
                _bullets(
                    _link(
                        stems.get(("paper", _sid(edge.get("src_id")))),
                        _sid(edge.get("src_id")),
                    )
                    for edge in papers
                ),
            )
        )
        return _document(
            [
                ("type", "concept"),
                ("id", draft.id),
                (
                    "aliases",
                    _aliases(
                        stem,
                        [_sid(row.get("id")), *as_list(row.get("aliases"))],
                    ),
                ),
                ("tags", _tags("concept")),
            ],
            draft.title,
            blocks,
        )
    if draft.type == "pillar":
        pillar = _index_rows(snapshot.pillars)[draft.id]
        papers = _incoming(edges, "pillar", draft.id, "in_pillar")
        goal = show(pillar.get("goal") or "").strip()
        blocks = [goal] if goal else []
        blocks.append(
            _section(
                "Papers",
                _bullets(
                    _link(
                        stems.get(("paper", _sid(edge.get("src_id")))),
                        _sid(edge.get("src_id")),
                    )
                    for edge in papers
                ),
            )
        )
        return _document(
            [
                ("type", "pillar"),
                ("id", draft.id),
                ("aliases", _aliases(stem, [draft.id])),
                ("tags", _tags("pillar", draft.id)),
            ],
            _sid(pillar.get("name")) or draft.id,
            blocks,
        )
    if draft.type == "author":
        papers = _incoming(edges, "author", draft.id, "authored_by")
        return _document(
            [
                ("type", "author"),
                ("id", draft.id),
                ("aliases", _aliases(stem, [draft.id])),
                ("tags", _tags("author")),
            ],
            draft.title,
            [
                _section(
                    "Papers",
                    _bullets(
                        _link(
                            stems.get(("paper", _sid(edge.get("src_id")))),
                            _sid(edge.get("src_id")),
                        )
                        for edge in papers
                    ),
                )
            ],
        )
    episode = _index_rows(snapshot.episodes)[draft.id]
    paper_id = _sid(episode.get("paper_id"))
    pillar_id = _sid(episode.get("pillar_id"))
    paper_link = _link(stems.get(("paper", paper_id)), paper_id) if paper_id else None
    pillar_link = (
        _link(stems.get(("pillar", pillar_id)), pillar_id) if pillar_id else None
    )
    blocks = []
    if paper_link:
        blocks.append(_section("Paper", _bullets([paper_link])))
    if pillar_link:
        blocks.append(_section("Pillar", _bullets([pillar_link])))
    return _document(
        [
            ("type", "episode"),
            ("id", draft.id),
            ("paper", paper_link),
            ("pillar", pillar_link),
            ("aliases", _aliases(stem, [draft.id])),
            ("tags", _tags("episode", pillar_id)),
        ],
        _sid(episode.get("title")) or draft.id,
        blocks,
    )


def render_vault(snapshot: VaultSnapshot) -> dict:
    """Relative POSIX path to generated note text, without a user block.

    The mapping is sorted by path. The same snapshot always produces the
    same bytes.
    """
    edges = list(snapshot.edges)
    drafts = _drafts(snapshot)
    stems = assign_stems(drafts)
    rendered = {}
    for draft in drafts:
        stem = stems[(draft.type, draft.id)]
        path = f"{FOLDERS[draft.type]}/{stem}.md"
        rendered[path] = _render_one(draft, stem, stems, snapshot, edges)
    return {path: rendered[path] for path in sorted(rendered)}


def extract_user_block(text: str) -> str:
    """Bytes after the end marker. Empty when the note has no marker."""
    index = text.find(END_MARK)
    if index < 0:
        return ""
    return text[index + len(END_MARK) :]


def parse_identity(text: str) -> Optional[tuple]:
    """``(type, id)`` from the YAML front matter this exporter writes."""
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    found = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        raw = raw.strip()
        try:
            found[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            found[key.strip()] = raw
    kind = found.get("type")
    ident = found.get("id")
    if isinstance(kind, str) and isinstance(ident, str) and kind and ident:
        return kind, ident
    return None


def _scan(vault: Path) -> dict:
    """Return managed notes keyed by relative POSIX path."""
    found = {}
    if not vault.is_dir():
        return found
    for folder in FOLDERS.values():
        root = vault / folder
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.md")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if START_MARK not in text:
                continue
            rel = path.relative_to(vault).as_posix()
            found[rel] = (path, text, parse_identity(text))
    return found


def _blocked(path: Path) -> bool:
    """Return whether ``path`` must be left untouched.

    A note without the generated marker is the user's. A symlink is left
    alone so a link cannot be swapped for a regular file.
    """
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() or not path.is_file():
        return True
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return True
    return START_MARK not in text and text != ""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(content.encode("utf-8"))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _archive(vault: Path, src: Path) -> Path:
    """Move ``src`` under ``_archive/``, keeping any note already archived."""
    relative = src.relative_to(vault)
    dest = vault / "_archive" / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        number = 2
        while True:
            candidate = dest.with_name(f"{dest.stem}-{number}{dest.suffix}")
            if not candidate.exists() and not candidate.is_symlink():
                dest = candidate
                break
            number += 1
    shutil.move(str(src), str(dest))
    return dest


def _user_block_for(rel: str, identity, existing: dict, by_identity: dict) -> str:
    if identity and identity in by_identity:
        matches = sorted(by_identity[identity], key=lambda item: item[0])
        same = [text for path, text in matches if path == rel]
        source = same[0] if same else matches[0][1]
        return extract_user_block(source)
    if rel in existing:
        return extract_user_block(existing[rel][1])
    return ""


def _write_notes(vault: Path, assembled: dict, existing: dict, dry_run: bool) -> tuple:
    changed = 0
    unchanged = 0
    kept = 0
    for rel, content in assembled.items():
        target = vault / rel
        if _blocked(target):
            kept += 1
            continue
        current = existing[rel][1] if rel in existing else None
        if current == content:
            unchanged += 1
            continue
        changed += 1
        if not dry_run:
            _write(target, content)
    return changed, unchanged, kept


def apply_export(
    out: Path, snapshot: VaultSnapshot, *, dry_run: bool = False
) -> ExportReport:
    """Write the vault. A second call with the same inputs rewrites nothing.

    ``dry_run`` computes the same counts and does not create, replace, or
    move any file.
    """
    vault = Path(out)
    if vault.exists() and not vault.is_dir():
        raise VaultExportError(f"Vault path is not a directory: {vault}")

    generated = render_vault(snapshot)
    existing = _scan(vault)
    by_identity = defaultdict(list)
    for rel, (_path, text, identity) in existing.items():
        if identity:
            by_identity[identity].append((rel, text))

    assembled = {}
    for rel, body in generated.items():
        identity = parse_identity(body)
        user = _user_block_for(rel, identity, existing, by_identity)
        assembled[rel] = body + user

    changed, unchanged, kept = _write_notes(vault, assembled, existing, dry_run)
    stale = [item[0] for rel, item in existing.items() if rel not in assembled]
    if not dry_run:
        for path in stale:
            _archive(vault, path)

    by_type = defaultdict(int)
    for body in generated.values():
        identity = parse_identity(body)
        if identity:
            by_type[identity[0]] += 1
    return ExportReport(
        dry_run=dry_run,
        notes=len(generated),
        by_type=dict(by_type),
        changed=changed,
        unchanged=unchanged,
        archived=len(stale),
        kept=kept,
    )
