"""A static context configuration: proven call neighbours and directly named files.

Compose this with a caller's existing import queue in round-robin order. Listing,
binding, name extraction and structural excerpts reuse their canonical owners.
No judging client or generation step is involved.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import TypeVar

from ..index.code_index import CodeIndex
from ..index.languages import language_of
from ..index.units import Reading, Unit, list_units
from ..operations import files_named_by
from .excerpts import StructuralExcerpts, batch_query
from .graph import CodeGraph, GraphEdge, _bound_edges

_T = TypeVar("_T")


@dataclass(frozen=True)
class ContextConfiguration:
    """Room for unit listing and the threshold for whole structural excerpts."""

    box_chars: int = 76_800
    whole_unit_lines: int = 25

    def build(self, index: CodeIndex) -> ContextSelection:
        """Prepare the one-hop and named-file composition for this readable scope."""
        return ContextSelection.build(index, self)

    def interleave(self, queues: Sequence[Iterable[_T]]) -> Iterator[_T]:
        """One candidate per source in caller order, continuing after a source ends."""
        for turn in zip_longest(*queues):
            yield from (piece for piece in turn if piece is not None)


ONE_HOP_NAMED_CONTEXT = ContextConfiguration()


class _CallsOnly(CodeGraph):
    """Retain JVN's proven calls and reference targets without importing other routes."""

    def __init__(self, units: tuple[Unit, ...]) -> None:
        super().__init__(unit.id for unit in units)
        self.references: dict[str, set[str]] = defaultdict(set)

    def add(self, edge: GraphEdge) -> None:
        if edge.kind == "call":
            super().add(edge)
        elif edge.kind == "reference":
            self.references[edge.source].add(edge.target)


@dataclass
class ContextSelection:
    """Reusable units and one-hop queues, with no judgment calls."""

    index: CodeIndex
    units: tuple[Unit, ...]
    calls: _CallsOnly
    unlisted: dict[str, str]
    excerpts: StructuralExcerpts
    query_refused: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, index: CodeIndex, configuration: ContextConfiguration) -> ContextSelection:
        """List and bind the supplied readable scope once, without judgment calls."""
        files = index.files
        listing = list_units(index, files, box_chars=configuration.box_chars, reading=Reading.MIXED)
        units = tuple(sorted(listing.units, key=lambda unit: (unit.path, unit.start, unit.id)))
        by_file: dict[str, list[Unit]] = defaultdict(list)
        for unit in units:
            by_file[unit.path].append(unit)
        calls = _CallsOnly(units)
        for path in by_file:
            if language_of(path):
                _bound_edges(index, calls, path, by_file)
        return cls(
            index,
            units,
            calls,
            dict(listing.unlisted),
            StructuralExcerpts(index, configuration.whole_unit_lines),
        )

    def queues(self, batch_files: list[str]) -> tuple[list[Unit], list[Unit]]:
        """Call neighbours ordered by distance and connectivity, plus directly named files."""
        inside = set(batch_files)
        owned = {unit.id for unit in self.units if unit.path in inside}
        seeds = owned | {target for source in owned for target in self.calls.references.get(source, ())}
        connections: Counter[str] = Counter()
        for seed in sorted(seeds):
            connections.update(self.calls.adjacency.get(seed, {}).keys())
        # The connection tie counts edges incident to all first-hop nodes.
        first_hop = set(connections) - seeds
        for node in sorted(first_hop):
            connections.update(self.calls.adjacency.get(node, {}).keys())
        distances = {key: 0 for key in seeds}
        distances.update({key: 1 for key in first_hop})
        graph = [unit for unit in self.units if unit.path not in inside and unit.id in distances]
        graph.sort(
            key=lambda unit: (
                distances[unit.id],
                -connections[unit.id],
                unit.path,
                unit.start,
                unit.id,
            )
        )
        named = files_named_by(self.index, [], batch_files)
        paths = set(named.code) | set(named.text)
        return graph, [unit for unit in self.units if unit.path in paths and unit.path not in inside]

    def query(self, files: list[str], concerns: list[str]) -> str:
        """Concern text followed by the sorted names in the supplied source files."""
        return batch_query(self.index, files, concerns, refused=self.query_refused)
