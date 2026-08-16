#!/usr/bin/env python3
"""Fail if a pillar slug is written as bare code instead of a string.

A pillar id is a hyphenated slug, so dropping its quotes produces valid Python
that means something entirely different:

    pillar_id=models-architectures      # subtraction of two undefined names
    pillar_id="models-architectures"    # what was meant

The bad form parses, compiles, and imports. `python -m py_compile` passes, ruff
passes, and the file only dies at runtime with `NameError: name 'models' is not
defined` — from the line that *uses* the fixture, not the line that defines it.
That is why 73 of these survived a slug migration and thirteen merged PRs: the
project had no CI running the test suite, and the three files carrying them were
the ones nobody ran locally.

Detection is done with `tokenize`, not a regex, so a slug appearing inside a
string, comment, f-string literal or identifier is never flagged — only a real
NAME `-` NAME expression whose reassembled text is a known slug.

Both live and retired slugs are checked. Retired ones (config.LEGACY_TO_SLUG)
are still legitimate as *quoted* strings in tests, which is what most of the
suite uses; this script only cares whether the quotes are there.

Usage:
    python scripts/check_bare_slugs.py [path ...]     # default: repo source dirs

Exit status is 1 if anything is found, so it can gate CI.
"""

from __future__ import annotations

import io
import pathlib
import sys
import tokenize

# The eight live pillars. create_pillars.py is authoritative; this list is a
# fourth mirror of it (after config.PILLAR_CONFIGS and README) and must be
# updated with the others. Getting it stale only weakens the check — a slug
# missing here is simply not detected — so it can never cause a false failure.
LIVE_SLUGS = {
    "formal-linguistics-nlp",
    "neural-architectures-language",
    "llm-theory-practice",
    "computational-semantics",
    "model-interpretability",
    "ai-agents-tool-use",
    "ml-systems-production",
    "ai-safety-alignment",
}

# Retired pre-migration slugs, still referenced (quoted) throughout the tests.
# Mirrors config.LEGACY_TO_SLUG's values.
RETIRED_SLUGS = {
    "linguistic-cognitive-foundations",
    "models-architectures",
    "data-training-methodologies",
    "evaluation-interpretability",
    "ethics-applications",
}

SLUGS = LIVE_SLUGS | RETIRED_SLUGS
SLUG_WORDS = {word for slug in SLUGS for word in slug.split("-")}

DEFAULT_ROOTS = ("nlp_pillars", "webui", "tests", "scripts")


def find_bare_slugs(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return (line number, slug) for every unquoted slug in `path`."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"{path}: could not read ({exc})", file=sys.stderr)
        return []

    try:
        tokens = [
            tok
            for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER, tokenize.STRING)
        ]
    except (tokenize.TokenError, SyntaxError) as exc:
        # A file too broken to tokenize is a real problem, but not this one's.
        print(f"{path}: could not tokenize ({exc})", file=sys.stderr)
        return []

    found: list[tuple[int, str]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == tokenize.NAME and tok.string in SLUG_WORDS:
            # Greedily absorb `- NAME` while the words keep belonging to a slug,
            # so `a-b-c` is considered whole rather than as `a-b` plus a stray.
            j, words = i, [tok.string]
            while (
                j + 2 < len(tokens)
                and tokens[j + 1].type == tokenize.OP
                and tokens[j + 1].string == "-"
                and tokens[j + 2].type == tokenize.NAME
                and tokens[j + 2].string in SLUG_WORDS
            ):
                words.append(tokens[j + 2].string)
                j += 2
            candidate = "-".join(words)
            if len(words) > 1 and candidate in SLUGS:
                found.append((tok.start[0], candidate))
                i = j + 1
                continue
        i += 1
    return found


def iter_python_files(roots: list[str]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for root in roots:
        p = pathlib.Path(root)
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(
                f
                for f in sorted(p.rglob("*.py"))
                # Never walk into a virtualenv that happens to live in-tree.
                if not any(part in {".venv", "venv", "__pycache__"} for part in f.parts)
            )
    return files


def main(argv: list[str]) -> int:
    roots = argv[1:] or [r for r in DEFAULT_ROOTS if pathlib.Path(r).exists()]
    total = 0
    for path in iter_python_files(roots):
        for line, slug in find_bare_slugs(path):
            # GitHub Actions renders this form as an inline annotation.
            print(f"::error file={path},line={line}::bare pillar slug "
                  f'{slug} — write it as "{slug}"')
            print(f"{path}:{line}: bare pillar slug {slug} (needs quotes)")
            total += 1

    if total:
        print(
            f"\n{total} bare pillar slug(s) found. Each is a NameError waiting "
            f"to happen at runtime — add the missing quotes.",
            file=sys.stderr,
        )
        return 1

    print(f"No bare pillar slugs found (checked {len(iter_python_files(roots))} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
