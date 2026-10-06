"""Small search compositions and reserved call shares. No caller domain lives here."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .directives.find_all import FindAllResult, find_all_async, find_all_text_async
from .directives.frontier import STAGE_ORDER, Policy
from .index.code_index import CodeIndex
from .index.units import Anchor, RangeAnchor
from .judgments.judge import Judge
from .mentions import names_from_text
from .sources import ANCHORS, FILES, NAMES, TEXT_FILE_NAMES, TEXT_NAMED_FILES, TEXT_NAMES, Source


def reserve_calls(judge: Judge, allowances: Mapping[str, int]) -> dict[str, Judge]:
    """Reserve independent stage caps before any stage starts. Parent accounting remains shared.

    Unused calls stay reserved for their stage. A caller can explicitly allocate them to a later
    composition. An existing parent cap must have room for every reservation.
    """
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in allowances.values()
    ):
        raise ValueError("call allowances must be nonnegative integers")
    remaining = judge.max_calls
    if remaining is not None and sum(allowances.values()) > remaining - judge.calls:
        raise ValueError("reservations exceed the parent's remaining call allowance")
    stages = {name: judge.scope() for name in allowances}
    for name, stage in stages.items():
        stage.max_calls = allowances[name]
    return stages


@dataclass(frozen=True)
class SearchConfiguration:
    """Code and text searches with independent call shares. Compose continuation using reserve_calls.

    Text includes named files and name hits by default, without scanning every text file.
    Both searches use the existing unit builder, masker, frontier and judging owner.
    """

    name: str
    code_calls: int
    text_calls: int
    policy: Policy = STAGE_ORDER
    code_sources: tuple[Source, ...] = (ANCHORS, FILES, NAMES)
    text_sources: tuple[Source, ...] = (ANCHORS, FILES, TEXT_FILE_NAMES, TEXT_NAMED_FILES, TEXT_NAMES)

    async def search(
        self,
        index: CodeIndex,
        judge: Judge,
        targets: Mapping[str, str],
        *,
        files: Sequence[str] = (),
        anchors: Sequence[Anchor] = (),
        names: Sequence[str] = (),
        delivered: Sequence[RangeAnchor] = (),
    ) -> tuple[FindAllResult, FindAllResult]:
        stages = reserve_calls(judge, {"code": self.code_calls, "text": self.text_calls})
        extracted = [names_from_text(text) for text in targets.values()]
        names = tuple(dict.fromkeys([*names, *(name for mentions in extracted for name in mentions.code)]))
        options = dict(files=files, anchors=anchors, names=names, delivered=delivered, policy=self.policy)
        code, text = await asyncio.gather(
            find_all_async(index, stages["code"], targets, sources=self.code_sources, **options),
            find_all_text_async(index, stages["text"], targets, sources=self.text_sources, **options),
        )
        return code, text
