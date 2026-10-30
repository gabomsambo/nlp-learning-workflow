"""View model for the paper map page.

The page is a shell: pillar chips, the toolbar and empty containers. Everything
on the map itself comes from ``GET /api/map`` in the browser, so this module
only decides which pillar the page opens on and what each chip says.

The map is built per pillar (there is no ``all`` snapshot), so there is no All
chip. With no ``?pillar=`` the page opens on the pillar with the most library
papers; a pillar that was asked for is kept even when unknown, so the page
reports it instead of quietly drawing another one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote

from .library_format import _pillar_pairs, pillar_css_class


def map_href(pillar_id: str | None) -> str:
    """The map page, opened on ``pillar_id`` when there is one."""
    if not pillar_id:
        return "/papers/map"
    return "/papers/map?pillar=" + quote(str(pillar_id), safe="")


@dataclass(frozen=True)
class MapChip:
    id: str
    name: str
    href: str
    pillar_class: str
    count: int | None
    current: bool
    empty: bool


@dataclass(frozen=True)
class MapPage:
    chips: list[MapChip]
    selected: str
    selected_name: str
    counts_known: bool


def build_map_page(
    pillars: Any,
    counts: Mapping[str, int] | None,
    requested: str | None,
) -> MapPage:
    """Chips for every pillar and the pillar the page opens on.

    ``counts`` maps pillar id to library papers, or is None when the counts
    could not be read. An unknown count never disables a chip: only a known
    zero does.
    """
    pairs = _pillar_pairs(pillars)
    names = dict(pairs)
    wanted = (requested or "").strip()
    if wanted:
        selected = wanted
    elif pairs:
        selected = max(
            pairs,
            key=lambda pair: (counts.get(pair[0], 0) if counts else 0, -pairs.index(pair)),
        )[0]
    else:
        selected = ""
    ordered = list(pairs)
    if selected and selected not in names:
        ordered.append((selected, selected))
    chips = []
    for pillar_id, name in ordered:
        count = None if counts is None else int(counts.get(pillar_id, 0))
        chips.append(
            MapChip(
                id=pillar_id,
                name=name,
                href=map_href(pillar_id),
                pillar_class=pillar_css_class(pillar_id),
                count=count,
                current=pillar_id == selected,
                empty=count == 0,
            )
        )
    return MapPage(
        chips=chips,
        selected=selected,
        selected_name=dict(ordered).get(selected, ""),
        counts_known=counts is not None,
    )
