"""Sources: where a search's candidates come from, as code facts without a model.

A source takes seeds and the index and reaches places. A place is a file, meaning every unit listed
in it, or an anchor, meaning the unit holding it. Each place it reaches carries its provenance: the
source, the seed it came from, a distance, and the request names it was reached by. A source makes
no Jev call.

A source never builds units and never scores them. The search resolves places into units with its
own room and reading, so the units, the anchors that named none and each seed's counts have one
owner. The frontier measures every unit's code features the same way whichever source reached it,
so two sources reaching one unit never score it differently; only the distance is a source's own.
A workflow is a composition: the sources that start it, the sources a unit that clears a target's
bar expands through, the frontier's policy and shares, and Jev judging in queue order.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from .index.code_index import CodeIndex
from .index.units import Anchor, Unit


@dataclass(frozen=True)
class Seeds:
    """What sources start from. ``names`` and ``texts`` come from the request (the texts are the
    targets' descriptions), ``files`` and ``anchors`` from the caller, and ``units`` are units a search
    already judged, such as one that cleared a target's bar."""

    names: tuple[str, ...] = ()
    texts: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    anchors: tuple[Anchor, ...] = ()
    units: tuple[Unit, ...] = ()


@dataclass(frozen=True)
class Reach:
    """One place a source reached. ``at`` is a file (every unit listed in it) or an anchor (the unit
    holding it). ``source`` is the source's name and ``seed`` what it reached the place from: a name,
    a path, an anchor as ``file:line`` or a unit id. ``distance`` is how far the place lies from what
    the caller pointed at, 0 for an anchor; the frontier subtracts it from a unit's value and keeps the
    smallest when several sources reach one unit. ``names`` are the request names the source reached
    the place by; each name's rarity counts them."""

    at: str | Anchor
    source: str
    seed: str
    distance: int
    names: frozenset[str] = frozenset()


class Source(Protocol):
    """A primitive that reaches places from seeds, without a model. ``name`` is how a result's
    provenance names it, and ``label`` how a coverage record counts the units it reached that were left
    unjudged, such as "callers"."""

    @property
    def name(self) -> str: ...

    @property
    def label(self) -> str: ...

    def reach(self, index: CodeIndex, seeds: Seeds) -> Iterable[Reach]:
        """Every place reachable from ``seeds``, in the order the source ranks them; a lazy iterable
        is read only as far as the search needs."""
        ...
