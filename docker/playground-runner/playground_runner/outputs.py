"""Turn Jupyter iopub messages into the runner's output list.

Output shapes (docs/PLAYGROUND_RUNNER.md is the contract):

    {"type": "stream", "name": "stdout"|"stderr", "text": str}
    {"type": "text",   "text": str}                         text/plain
    {"type": "html",   "html": str, "text": str|None, "untrusted": True}
    {"type": "image",  "mime": "image/png", "data": base64 str, "text": str|None}
    {"type": "error",  "ename": str, "evalue": str, "traceback": str}

Any output cut to fit carries ``"truncated": True``. The collector's result
also reports ``truncated`` and ``dropped_outputs`` for the whole execution.

Nothing here imports Jupyter, so it is testable without a kernel.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# CSI sequences (colours, cursor moves), OSC sequences (titles, hyperlinks) and
# the remaining two-byte escapes. IPython colours every traceback line.
_ANSI_RE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[@-Z\\-_]"
)

#: An error traceback is always returned, even past the output cap, so the
#: reason a cell failed is never dropped. It is cut to this many bytes instead.
ERROR_TRACEBACK_MAX_BYTES = 64_000

TRUNCATION_NOTE = "\n[output truncated: limit of {limit} bytes reached]"


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def _join_text(value: Any) -> str:
    """Jupyter allows text fields as a string or a list of lines."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value)


def _cut(text: str, max_bytes: int) -> str:
    """Cut to at most max_bytes of UTF-8 without splitting a character."""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[: max(0, max_bytes)].decode("utf-8", errors="ignore")


def _size(output: Dict[str, Any]) -> int:
    total = 0
    for key in ("text", "html", "data", "traceback", "evalue"):
        value = output.get(key)
        if isinstance(value, str):
            total += len(value.encode("utf-8"))
    return total


class OutputCollector:
    """Accumulates one execution's outputs under a byte cap."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.outputs: List[Dict[str, Any]] = []
        self.used = 0
        self.truncated = False
        self.dropped = 0
        self._clear_pending = False

    # -- public -----------------------------------------------------------

    def add_message(self, msg_type: str, content: Dict[str, Any]) -> None:
        if msg_type == "stream":
            self._add_stream(content.get("name") or "stdout", strip_ansi(_join_text(content.get("text"))))
        elif msg_type in ("display_data", "execute_result", "update_display_data"):
            output = self._from_bundle(content.get("data") or {})
            if output is not None:
                self._add(output)
        elif msg_type == "error":
            self.add_error(
                content.get("ename") or "Error",
                content.get("evalue") or "",
                content.get("traceback") or [],
            )
        elif msg_type == "clear_output":
            if content.get("wait"):
                self._clear_pending = True
            else:
                self._clear()

    def add_error(self, ename: str, evalue: str, traceback: Any = None) -> None:
        self._apply_pending_clear()
        if isinstance(traceback, list):
            tb_text = "\n".join(str(line) for line in traceback)
        else:
            tb_text = str(traceback or "")
        tb_text = strip_ansi(tb_text)
        output: Dict[str, Any] = {
            "type": "error",
            "ename": strip_ansi(str(ename)),
            "evalue": strip_ansi(str(evalue)),
            "traceback": tb_text,
        }
        if len(tb_text.encode("utf-8")) > ERROR_TRACEBACK_MAX_BYTES:
            output["traceback"] = _cut(tb_text, ERROR_TRACEBACK_MAX_BYTES) + "\n[traceback truncated]"
            output["truncated"] = True
            self.truncated = True
        self.outputs.append(output)
        self.used += _size(output)

    def remove_errors(self, ename: str) -> None:
        """Drop error outputs of one kind (the KeyboardInterrupt a timeout causes)."""
        kept = [o for o in self.outputs if not (o.get("type") == "error" and o.get("ename") == ename)]
        self.outputs = kept

    def has_error(self, ename: str) -> bool:
        return any(o.get("type") == "error" and o.get("ename") == ename for o in self.outputs)

    def result(self) -> Dict[str, Any]:
        return {
            "outputs": self.outputs,
            "truncated": self.truncated,
            "dropped_outputs": self.dropped,
        }

    # -- internals --------------------------------------------------------

    def _from_bundle(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        plain = data.get("text/plain")
        plain_text = strip_ansi(_join_text(plain)) if plain is not None else None
        if "image/png" in data:
            png = _join_text(data["image/png"]).replace("\n", "")
            return {"type": "image", "mime": "image/png", "data": png, "text": plain_text}
        if "text/html" in data:
            return {
                "type": "html",
                "html": _join_text(data["text/html"]),
                "text": plain_text,
                "untrusted": True,
            }
        if plain_text is not None:
            return {"type": "text", "text": plain_text}
        return None

    def _clear(self) -> None:
        self.outputs = []
        self.used = 0
        self._clear_pending = False

    def _apply_pending_clear(self) -> None:
        if self._clear_pending:
            self._clear()

    def _remaining(self) -> int:
        return self.max_bytes - self.used

    def _add_stream(self, name: str, text: str) -> None:
        self._apply_pending_clear()
        if not text:
            return
        last = self.outputs[-1] if self.outputs else None
        if (
            last is not None
            and last.get("type") == "stream"
            and last.get("name") == name
            and not last.get("truncated")
        ):
            fitted = self._fit_text(text)
            if fitted is None:
                return
            piece, cut = fitted
            last["text"] += piece
            self.used += len(piece.encode("utf-8"))
            if cut:
                last["text"] += TRUNCATION_NOTE.format(limit=self.max_bytes)
                last["truncated"] = True
            return
        self._add({"type": "stream", "name": name, "text": text})

    def _fit_text(self, text: str):
        """Return (text that fits, was_cut), or None when nothing fits."""
        remaining = self._remaining()
        if remaining <= 0:
            self.truncated = True
            self.dropped += 1
            return None
        size = len(text.encode("utf-8"))
        if size <= remaining:
            return text, False
        self.truncated = True
        return _cut(text, remaining), True

    def _add(self, output: Dict[str, Any]) -> None:
        self._apply_pending_clear()
        size = _size(output)
        if size <= self._remaining():
            self.outputs.append(output)
            self.used += size
            return
        self.truncated = True
        kind = output["type"]
        if kind in ("html", "image") and output.get("text"):
            # Fall back to the plain-text form the kernel sent alongside.
            output = {"type": "text", "text": output["text"]}
            kind = "text"
        if kind in ("stream", "text"):
            fitted = self._fit_text(output["text"])
            if fitted is None:
                return
            piece, _ = fitted
            output = dict(output)
            output["text"] = piece + TRUNCATION_NOTE.format(limit=self.max_bytes)
            output["truncated"] = True
            self.outputs.append(output)
            self.used += len(piece.encode("utf-8"))
            return
        self.dropped += 1
