"""Strip a cell's leading indent the way Jupyter did when the notebook was written.

IPython up to 8.x removed the first line's leading whitespace from every line
that starts with that same whitespace (``inputtransformer2.leading_indent``).
So a cell such as::

      df = load()
      df.head()
    # a note at the end

ran in Jupyter. IPython 9 replaced that with ``textwrap.dedent``, which only
removes the indent *every* line shares, comment lines included. The unindented
comment leaves the cell as it is, and it fails with "unexpected indent" at
line 1.

:func:`install` puts the older transform back in the kernel's
``input_transformer_manager``, so every execution path (a cell, an edit, a
"try this") gets it from one place. A cell whose first line is not indented is
left alone, so genuinely mis-indented code still raises.
"""

from __future__ import annotations

import re
from typing import Any, List

_INDENT = re.compile(r"^[ \t]+")


def leading_indent(lines: List[str]) -> List[str]:
    """IPython 8's transform. ``lines`` keep their line endings."""
    if not lines:
        return lines
    match = _INDENT.match(lines[0])
    if not match:
        return lines
    space = match.group(0)
    n = len(space)
    return [line[n:] if line.startswith(space) else line for line in lines]


def strip_leading_indent(code: str) -> str:
    """:func:`leading_indent` on a whole cell, after dropping leading blank lines
    as IPython's ``leading_empty_lines`` does before it."""
    lines = code.splitlines(keepends=True)
    while lines and not lines[0].strip():
        lines.pop(0)
    return "".join(leading_indent(lines))


def install(shell: Any) -> None:
    """Use :func:`leading_indent` in place of IPython's own in ``shell``."""
    from IPython.core import inputtransformer2 as it2

    manager = shell.input_transformer_manager
    transforms = list(manager.cleanup_transforms)
    current = getattr(it2, "leading_indent", None)
    if current in transforms:
        transforms[transforms.index(current)] = leading_indent
    elif leading_indent not in transforms:
        empty = getattr(it2, "leading_empty_lines", None)
        at = transforms.index(empty) + 1 if empty in transforms else 0
        transforms.insert(at, leading_indent)
    manager.cleanup_transforms = transforms
