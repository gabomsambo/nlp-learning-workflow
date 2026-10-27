"""Read a .ipynb, keep its saved outputs, and make them safe to show.

A notebook is stored as uploaded (the raw JSON, outputs included). What the
page and the explain pass see is :func:`cells_view`: one entry per cell with
its source and its outputs reduced to the four kinds the Playground shows —
stream text, ``text/plain``, ``text/html`` (sanitized into a tag tree) and
``image/png``. Errors keep their message; anything else is named, not hidden.

``text/html`` is untrusted. It is never passed through as markup: the
sanitizer below turns it into a tree of allowed tags (tables and basic
formatting) with no attributes except numeric ``colspan``/``rowspan``, and the
browser builds that tree with ``createElement``. Scripts, styles, event
handlers, links and images cannot survive the trip.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

#: Text kept per output in the view. The captain studies exact results, so
#: this is generous; a cut is marked, never silent.
MAX_OUTPUT_CHARS = 200_000

#: Nodes kept per sanitized HTML output. A DataFrame preview is a few hundred.
MAX_HTML_NODES = 20_000

ALLOWED_TAGS = frozenset({
    "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption",
    "b", "strong", "i", "em", "u", "code", "pre", "br", "hr", "p", "span",
    "div", "sub", "sup", "ul", "ol", "li", "small",
    "h1", "h2", "h3", "h4", "h5", "h6",
})

#: Dropped together with everything inside them.
DROPPED_WITH_CONTENT = frozenset({
    "script", "style", "iframe", "object", "embed", "svg", "math", "template",
    "noscript", "head", "title", "textarea", "select", "button", "form",
    "canvas", "video", "audio", "picture", "frameset", "frame", "applet",
})

VOID_TAGS = frozenset({
    "br", "hr", "img", "input", "meta", "link", "col", "area", "base", "wbr",
    "source", "track", "param", "embed",
})

_SPAN_ATTRS = frozenset({"colspan", "rowspan"})
_STRUCTURAL = frozenset({
    "#root", "table", "thead", "tbody", "tfoot", "tr", "ul", "ol", "div",
})
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class NotebookError(ValueError):
    """The file is not a notebook this app can import. The message says why."""


# -- parsing -------------------------------------------------------------------


def parse_notebook(data: bytes, *, max_bytes: int) -> Dict[str, Any]:
    """Validate an uploaded .ipynb and return its JSON.

    Raises:
        NotebookError: too large, not JSON, or not an nbformat 4 notebook.
    """
    if len(data) > max_bytes:
        raise NotebookError(
            f"The notebook is {_mb(len(data))} and the limit is {_mb(max_bytes)} "
            "(PLAYGROUND_NOTEBOOK_MAX_BYTES)."
        )
    if not data.strip():
        raise NotebookError("The file is empty.")
    try:
        nb = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NotebookError(
            "This file is not a Jupyter notebook: it is not JSON "
            f"({type(exc).__name__})."
        ) from None
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        raise NotebookError("This file is not a Jupyter notebook: it has no cell list.")
    if nb.get("nbformat") != 4:
        raise NotebookError(
            f"Only nbformat 4 notebooks are supported; this one says "
            f"nbformat {nb.get('nbformat')!r}. Re-save it from Jupyter or Colab."
        )
    for i, cell in enumerate(nb["cells"]):
        if not isinstance(cell, dict) or cell.get("cell_type") not in (
            "code", "markdown", "raw",
        ):
            raise NotebookError(f"Cell {i} is not a code, markdown or raw cell.")
        source = cell.get("source", "")
        if not isinstance(source, str) and not (
            isinstance(source, list) and all(isinstance(s, str) for s in source)
        ):
            raise NotebookError(f"Cell {i} has a source that is not text.")
        if cell["cell_type"] == "code" and not isinstance(cell.get("outputs", []), list):
            raise NotebookError(f"Cell {i} has outputs that are not a list.")
    if not any(c["cell_type"] == "code" for c in nb["cells"]):
        raise NotebookError("The notebook has no code cells, so there is nothing to study.")
    return nb


def _mb(n: int) -> str:
    return f"{n / 1_000_000:.1f} MB"


def joined(value: Any) -> str:
    """nbformat stores multi-line strings as a list of lines or as one string."""
    if isinstance(value, list):
        return "".join(str(v) for v in value)
    return "" if value is None else str(value)


def source_text(cell: Dict[str, Any]) -> str:
    return joined(cell.get("source", ""))


def notebook_title(nb: Dict[str, Any], fallback: str) -> str:
    """First markdown heading, or the file name."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        for line in source_text(cell).splitlines():
            match = re.match(r"\s*#{1,6}\s+(.+)", line)
            if match:
                return match.group(1).strip()[:200]
    return fallback[:200] or "Notebook"


def code_cell_indices(nb: Dict[str, Any]) -> List[int]:
    """Indices of code cells with any source at all (the cells a guide covers)."""
    return [
        i for i, c in enumerate(nb.get("cells", []))
        if c.get("cell_type") == "code" and source_text(c).strip()
    ]


# -- view ----------------------------------------------------------------------


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def _cap(text: str) -> Dict[str, Any]:
    if len(text) <= MAX_OUTPUT_CHARS:
        return {"text": text}
    return {"text": text[:MAX_OUTPUT_CHARS], "truncated": True}


def _png(value: Any) -> Optional[str]:
    """The base64 PNG if it decodes to one, else None."""
    raw = joined(value).replace("\n", "").strip()
    try:
        decoded = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        return None
    return raw if decoded.startswith(_PNG_MAGIC) else None


def output_view(output: Dict[str, Any]) -> Dict[str, Any]:
    """One saved output, reduced to something the page can show safely."""
    kind = output.get("output_type")
    if kind == "stream":
        return {
            "kind": "stream",
            "name": output.get("name") or "stdout",
            **_cap(strip_ansi(joined(output.get("text")))),
        }
    if kind == "error":
        trace = "\n".join(strip_ansi(joined(line)) for line in output.get("traceback") or [])
        head = f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
        return {"kind": "error", **_cap(trace or head)}
    if kind in ("execute_result", "display_data"):
        data = output.get("data") or {}
        plain = joined(data["text/plain"]) if "text/plain" in data else None
        if "image/png" in data:
            png = _png(data["image/png"])
            if png:
                return {"kind": "png", "data": png, "alt": (plain or "Saved figure")[:200]}
        if "text/html" in data:
            tree, truncated = sanitize_html(joined(data["text/html"]))
            if tree:
                view: Dict[str, Any] = {"kind": "html", "tree": tree}
                if truncated:
                    view["truncated"] = True
                if plain is not None:
                    view["plain"] = _cap(plain)["text"]
                return view
        if plain is not None:
            return {"kind": "text", **_cap(plain)}
        return {"kind": "unsupported", "mime": sorted(data.keys())}
    return {"kind": "unsupported", "mime": [str(kind)]}


_TABLE_RULE = re.compile(r"^\s*\|?(\s*:?-{3,}:?\s*\|)+\s*(:?-{3,}:?)?\s*$")


def markdown_text(source: str) -> str:
    """Markdown shown as plain text. Images and tags become words, not markup;
    a table's ``|---|---|`` rule line is dropped."""
    text = re.sub(r"<img\b[^>]*>", "[image]", source, flags=re.I)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"[image: \1]", text)
    text = re.sub(r"<[^>]+>", "", text)
    return "\n".join(line for line in text.split("\n") if not _TABLE_RULE.match(line))


def cells_view(nb: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every cell, in order, as the Playground tab and the explain pass see it."""
    cells = []
    for i, cell in enumerate(nb.get("cells", [])):
        ctype = cell.get("cell_type")
        source = source_text(cell)
        entry: Dict[str, Any] = {"index": i, "type": ctype}
        if ctype == "markdown":
            entry["source"] = markdown_text(source)
        else:
            entry["source"] = source
        if ctype == "code":
            entry["execution_count"] = cell.get("execution_count")
            entry["outputs"] = [
                output_view(o) for o in cell.get("outputs") or [] if isinstance(o, dict)
            ]
        cells.append(entry)
    return cells


# -- HTML sanitizer --------------------------------------------------------------


class _TreeBuilder(HTMLParser):
    """Allow-list HTML into ``{"t": tag, "a": {...}, "c": [...]}`` / str nodes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root: Dict[str, Any] = {"t": "#root", "c": []}
        self.stack: List[Dict[str, Any]] = [self.root]
        self.drop_depth = 0
        self.drop_tag: Optional[str] = None
        self.nodes = 0
        self.truncated = False

    def _room(self) -> bool:
        if self.nodes >= MAX_HTML_NODES:
            self.truncated = True
            return False
        self.nodes += 1
        return True

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if self.drop_tag is not None:
            if tag == self.drop_tag and tag not in VOID_TAGS:
                self.drop_depth += 1
            return
        if tag in DROPPED_WITH_CONTENT:
            if tag not in VOID_TAGS:
                self.drop_tag, self.drop_depth = tag, 1
            return
        if tag not in ALLOWED_TAGS or not self._room():
            return  # unknown tags are unwrapped: their text still arrives
        node: Dict[str, Any] = {"t": tag, "c": []}
        if tag in ("td", "th"):
            kept = {}
            for name, value in attrs:
                name = (name or "").lower()
                if name in _SPAN_ATTRS and value and value.isdigit() and 0 < int(value) <= 1000:
                    kept[name] = value
            if kept:
                node["a"] = kept
        self.stack[-1]["c"].append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)
        tag = tag.lower()
        if tag in ALLOWED_TAGS and tag not in VOID_TAGS and self.drop_tag is None:
            if len(self.stack) > 1 and self.stack[-1]["t"] == tag:
                self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.drop_tag is not None:
            if tag == self.drop_tag:
                self.drop_depth -= 1
                if self.drop_depth <= 0:
                    self.drop_tag = None
            return
        for depth in range(len(self.stack) - 1, 0, -1):
            if self.stack[depth]["t"] == tag:
                del self.stack[depth:]
                return

    def handle_data(self, data: str) -> None:
        if self.drop_tag is not None or not data:
            return
        if not data.strip() and self.stack[-1]["t"] in _STRUCTURAL:
            return  # indentation between <tr>s, not content
        siblings = self.stack[-1]["c"]
        if siblings and isinstance(siblings[-1], str):
            siblings[-1] += data
            return
        if self._room():
            siblings.append(data)


def sanitize_html(markup: str) -> tuple[List[Any], bool]:
    """Untrusted HTML to a safe node tree. Returns ``(children, truncated)``.

    Whitespace-only top-level text is dropped so an output that was only a
    ``<style>`` block comes back empty and the caller falls back to plain text.
    """
    builder = _TreeBuilder()
    builder.feed(markup or "")
    builder.close()
    children = [
        c for c in builder.root["c"]
        if not (isinstance(c, str) and not c.strip())
    ]
    return children, builder.truncated


def tree_text(nodes: List[Any]) -> str:
    """A sanitized tree as plain text: table cells tab-separated, rows on lines."""
    parts: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            parts.append(" ".join(node.split()))
            return
        tag = node.get("t")
        for child in node.get("c", []):
            walk(child)
        if tag in ("td", "th"):
            parts.append("\t")
        elif tag in ("tr", "p", "div", "br", "li", "caption", "h1", "h2", "h3", "h4", "h5", "h6"):
            parts.append("\n")

    for n in nodes:
        walk(n)
    lines = ["\t".join(seg.strip() for seg in line.split("\t")).strip() for line in "".join(parts).split("\n")]
    return "\n".join(line for line in lines if line)
