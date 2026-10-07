"""The pinned Engine build_pack entry point with a caller-supplied ranked source recipe."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import replace

from enginepy.workflows.document_analysis import evidence_pack as ep
from enginepy.workflows.document_analysis.skeptic_packet import PacketBudget
from jev_navigator.judgments.profiles import LOCAL_ROLES, ROLES_V2

from jev_navigator.directives.find_all import NOT_REACHED, REFUSED, FindAllResult
from jev_navigator.index.units import Piece, Unit, UnitKind, items_to_judge, read_ranges
from jev_navigator.judgments.judge import CallCapReachedError


class MeasuredRoom(PacketBudget):
    """The requested historical diagnostic room uses the same owner's derived shares."""

    def __post_init__(self):
        if self.tokens <= 0:
            raise ValueError("A measurement room must be positive")


def source_unit(binding):
    return Unit(
        **{
            **binding,
            "ranges": tuple(map(tuple, binding["ranges"])),
            "kind": UnitKind(binding["kind"]),
            "pieces": tuple(Piece(**piece) for piece in binding["pieces"]),
        }
    )


class RankedRecipe:
    def __init__(self, candidates):
        self.units = tuple(source_unit(candidate["unit"]) for candidate in candidates)
        self.result = None

    async def search(self, index, judge, targets, **kwargs):
        """Stream whole admitted code/text units through the real sixteen-item Judge."""
        asked = ROLES_V2.asked(targets)
        checks = [check for check, _, _ in asked.values()]
        raw, observed = defaultdict(dict), {target: [] for target in targets}
        refusals, not_judged = [], {}
        calls = judge.calls
        failure, stopped = None, "scope_examined"
        entries, places = [], []

        async def send():
            async for name, answer in judge.iter_check_every_async(
                checks,
                entries,
                {"targets": dict(targets)},
                places=places,
                refusals=refusals,
                keep_order=True,
            ):
                _, role, target = asked[name]
                key = target, answer.place.id
                raw[key][role] = answer
                if len(raw[key]) == len(ROLES_V2.templates):
                    observed[target].append(ROLES_V2.compose(raw[key], judge.thresholds))

        try:
            for unit in self.units:
                for item in items_to_judge(unit):
                    entries.append({"file": item.file, "code": read_ranges(index, item.file, item.ranges)})
                    places.append(item)
                    if len(entries) == 16:
                        await send()
                        entries.clear()
                        places.clear()
            if entries:
                await send()
        except CallCapReachedError:
            stopped = "call_cap"
        except Exception as error:
            failure, stopped = error, "failed"
        answered = {answer.place.id for answers in observed.values() for answer in answers}
        for unit in self.units:
            for item in items_to_judge(unit):
                if item.id not in answered:
                    not_judged[item.id] = NOT_REACHED
        for refusal in refusals:
            if refusal.place:
                not_judged[refusal.place.id] = REFUSED
        self.result = FindAllResult(
            dict(targets),
            76800,
            1,
            self.units,
            {target: tuple(answers) for target, answers in observed.items()},
            not_judged,
            {},
            (),
            {},
            frozenset(),
            stopped,
            judge.calls - calls,
            failure=failure,
            refusals=tuple(refusals),
            entered_by={unit.id: "ranked_selection" for unit in self.units},
            question_profile=ROLES_V2,
            required_roles=LOCAL_ROLES,
        )
        return self.result


async def build(relations, index, context, candidates, judge):
    recipe = RankedRecipe(candidates)
    request = ep.pack_request(relations, context["repository"], context["claim"], context.get("points"))
    if request is None:
        # There is no packet anchor, but this finding still receives its ranked judging workload.
        targets = context.get("points") or {"statement": context["claim"]["statement"]}
        await recipe.search(index, judge, targets)
        return None, recipe.result
    previous = ep.find_all_async
    ep.find_all_async = recipe.search
    try:
        pack = await ep.build_pack(
            relations,
            request,
            judge,
            index,
            ep.PackSettings(callee_round=False, question_profile=ROLES_V2, required_roles=LOCAL_ROLES),
        )
    finally:
        ep.find_all_async = previous
    return pack, recipe.result


def packet_in_room(pack, relations, context, tokens):
    if pack is None:
        return None
    budget = MeasuredRoom(tokens)
    request = ep.pack_request(
        relations, context["repository"], context["claim"], context.get("points"), budget=budget
    )
    return replace(pack, floor=request.floor, budget=budget, trimmed=request.trimmed).packet()


def build_sync(*args):
    return asyncio.run(build(*args))
