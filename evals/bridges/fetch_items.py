"""Rebuild the bridge check's items file from ``pairs.json`` (stdlib only).

``pairs.json`` names 43 pairs of arXiv papers and, for 29 of them, the sentence
in the first paper that cites the second (taken from the paper's HTML on
ar5iv). This script fetches the abstracts from the arXiv API in one request
and writes ``items.json`` for ``scripts/eval_bridges.py``. The repository does
not ship abstracts.

    python evals/bridges/fetch_items.py evals/bridges/pairs.json /tmp/items.json
    ANTHROPIC_API_KEY=... python scripts/eval_bridges.py /tmp/items.json /tmp/results.json
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ATOM = "{http://www.w3.org/2005/Atom}"


def fetch_abstracts(ids: list[str]) -> dict[str, dict]:
    query = urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    request = urllib.request.Request(
        f"http://export.arxiv.org/api/query?{query}",
        headers={"User-Agent": "nlp-learning-workflow-eval/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        root = ET.fromstring(response.read())
    out = {}
    for entry in root.findall(f"{ATOM}entry"):
        identifier = entry.find(f"{ATOM}id").text.split("/abs/", 1)[-1]
        identifier = identifier.rsplit("v", 1)[0] if "v" in identifier.rsplit("/", 1)[-1] else identifier
        out[identifier] = {
            "title": " ".join(entry.find(f"{ATOM}title").text.split()),
            "abstract": " ".join(entry.find(f"{ATOM}summary").text.split()),
        }
    return out


def main(argv: list[str]) -> int:
    pairs = json.load(open(argv[1], encoding="utf-8"))
    ids = sorted({p[k] for p in pairs for k in ("a_arxiv", "b_arxiv")})
    found = fetch_abstracts(ids)
    missing = [i for i in ids if i not in found]
    if missing:
        print(f"arXiv did not return: {missing}", file=sys.stderr)
        return 1
    items = []
    for p in pairs:
        side = lambda s: {  # noqa: E731
            "label": p[f"{s}_label"],
            "arxiv": p[f"{s}_arxiv"],
            **found[p[f"{s}_arxiv"]],
        }
        items.append(
            {
                "id": p["id"],
                "kind": p["kind"],
                "a": side("a"),
                "b": side("b"),
                "citing": p["citing"],
                "context": p["context"],
                "intents": [],
            }
        )
    json.dump(items, open(argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(items)} items -> {argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
