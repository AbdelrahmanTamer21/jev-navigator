"""The frontier: the order in which a search judges the units of its population.

Under any call cap, whatever is ranked last is what gets lost, so the order is a policy, named and
compared. ``STAGE_ORDER`` is the order find_all always had: the units the anchors name, then the units
of the files, then the units holding each name's hits, rarest name first, each wave's batches sorted
by place. ``VALUE`` (B1) scores every unit by code before any call and judges the best first, batches
in that order. Its features are facts any language has: which of the request's names a unit's code
contains and how rare each is, whether the unit is named like one, whether its file is, how far its
source lies from the anchors, and whether it is a test. A test is ranked by the same score, never
dropped. Ties go to the unit's content hash, never to its path, and code that repeats a unit already in
the queue is judged once.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath

from ..index.units import Unit


class Source(StrEnum):
    """How a unit entered a population: find_all lists the units of the caller's anchors (``ANCHOR``),
    then of its files (``FILE``), then of each name's hits (``NAME``). A caller composing searches names
    the rest, such as a round of callees (``CALLEE``) it fed as anchors."""

    ANCHOR = "anchor"
    FILE = "file"
    NAME = "name"
    CALLEE = "callee"


@dataclass(frozen=True)
class Weights:
    """What each code feature adds to a unit's value. The name rarities add as they are; distance and a
    test file subtract. Fixed before the first trial (speed-bench's code-first weights, plus the test
    weight); a simulator tunes them as data."""

    defines: float = 2.0
    file_named: float = 1.0
    distance: float = 0.1
    test: float = 1.0


@dataclass(frozen=True)
class Policy:
    """``ranked`` False keeps the stage order. True orders the population by value under ``weights``,
    keeps that order in the batches, and judges repeated code once."""

    name: str
    ranked: bool
    weights: Weights = field(default_factory=Weights)


STAGE_ORDER = Policy("stage_order", ranked=False)
VALUE = Policy("value", ranked=True)


@dataclass(frozen=True)
class Features:
    """A unit's code features for one request. ``names`` are the request's names its code contains,
    ``rarity`` their sum of ``1 / log2(2 + hits)``, ``defines`` whether the unit is named like one,
    ``file_named`` whether its file is, ``distance`` how far its source lies from the anchors (see
    ``distance``), ``test`` whether its file is a test."""

    names: tuple[str, ...]
    rarity: float
    defines: bool
    file_named: bool
    distance: int
    test: bool

    def value(self, weights: Weights) -> float:
        return (
            self.rarity
            + weights.defines * self.defines
            + weights.file_named * self.file_named
            - weights.distance * self.distance
            - weights.test * self.test
        )


def name_rarities(hit_counts: Mapping[str, int]) -> dict[str, float]:
    """What finding each name adds: a name with fewer hits is rarer and adds more."""
    return {name: 1 / math.log2(2 + count) for name, count in hit_counts.items()}


def features_of(
    unit: Unit, code: str, source: Source, near_files: Collection[str], rarities: Mapping[str, float]
) -> Features:
    """``unit``'s features, from its ``code``, the ``source`` it entered by, the files near the anchors
    and each request name's rarity. A name counts only as a whole word: ``limit`` is not in ``limits``."""
    names = tuple(name for name in rarities if _holds_word(code, name))
    return Features(
        names=names,
        rarity=sum(rarities[name] for name in names),
        defines=_last_part(unit.symbol) in rarities,
        file_named=_normalised(PurePosixPath(unit.path).stem) in {_normalised(name) for name in rarities},
        distance=distance(unit, source, near_files),
        test=unit.test,
    )


def distance(unit: Unit, source: Source, near_files: Collection[str]) -> int:
    """0 for a unit an anchor names, 1 for a unit of a file near the anchors (an anchor's own file, or a
    file it imports, where the index resolves that language's imports), 2 for a unit of any other file,
    and 3 for a unit only a name's hit reached."""
    if source is Source.FILE:
        return 1 if unit.path in near_files else 2
    return _DISTANCE[source]


def value_key(unit: Unit, features: Features, weights: Weights) -> tuple[float, str, str]:
    """The queue order: best value first, then the content hash; identical code last by id."""
    return -features.value(weights), unit.content_sha256, unit.id


_DISTANCE = {Source.ANCHOR: 0, Source.NAME: 3}


def _holds_word(code: str, name: str) -> bool:
    return re.search(rf"(?<![\w$]){re.escape(name)}(?![\w$])", code) is not None


def _last_part(symbol: str) -> str:
    """``OrderService.place`` is named ``place``; a schema block ``model Website`` is named ``Website``."""
    return re.split(r"[.\s]", symbol)[-1]


def _normalised(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())
