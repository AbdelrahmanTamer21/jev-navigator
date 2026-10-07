"""Measure saved paid observations with the frozen lab allocator and native packets.

Run with the pinned JVN, Engine and find-eval harness on PYTHONPATH. This program
does not instantiate a provider client or send a request. Missing arm results stay
explicitly incomplete rather than contributing fabricated zero observations.
"""

from __future__ import annotations

import json
import re
import statistics
from decimal import Decimal
from types import SimpleNamespace

from enginepy.workflows.document_analysis import evidence_pack as ep
from find_eval.simulate import composed_case, proposal_reducer
from paid_native import MeasuredRoom, source_unit
from paid_prepare import ranked_units
from paid_run import scopes, shown_lines
from paid_support import OUT, PROOF, ROOMS, load, resources, write

from jev_navigator.index.units import items_to_judge

COMPARE = PROOF / "runs/roles-compare-20261006"
HARD_HISTORY = PROOF / "runs/real-pack-e733ea0c-jvn0a73a59c-value-full/hard27"
ARMS = ("code", "code_scent_filename", "outline", "llm")


def cases_only(path):
    """Read the frozen container's first field without loading its source-unit pool."""
    with path.open() as stream:
        beginning = stream.read(4096)
        prefix = re.match(r'^\s*\{\s*"cases"\s*:\s*', beginning)
        if prefix is None:
            raise ValueError(f"Frozen metadata no longer starts with cases: {path}")
        buffer = beginning[prefix.end() :]
        decoder = json.JSONDecoder()
        while True:
            try:
                cases, _ = decoder.raw_decode(buffer)
                return cases
            except json.JSONDecodeError:
                part = stream.read(65536)
                if not part or len(buffer) > 8 * 1024 * 1024:
                    raise ValueError(f"Could not read bounded case geometry: {path}") from None
                buffer += part


def covered(labels, windows):
    """Use the allocator's registered-span coverage rule, including single lines."""
    return [
        len(
            {
                line
                for window in windows
                if window["file"] == label["file"]
                for line in range(
                    max(label["first_line"], window["first_line"]),
                    min(label["last_line"], window["last_line"]) + 1,
                )
            }
        )
        >= (label["last_line"] - label["first_line"] + 1) / 2
        for label in labels
    ]


def binding_windows(bindings):
    return [
        {"file": binding["path"], "first_line": start, "last_line": end}
        for binding in bindings
        for start, end in binding["ranges"]
    ]


def observed_windows(observations):
    return [
        {"file": row["place"]["file"], "first_line": start, "last_line": end}
        for row in observations
        for start, end in row["place"]["ranges"]
    ]


def invoice(folder):
    receipts = [load(path) for path in folder.glob("*/receipt.json")]
    return {
        "physical_calls": len(receipts),
        "usd": float(sum((Decimal(row.get("usd", "0")) for row in receipts), Decimal(0))),
        "seconds": sum(row["seconds"] for row in receipts),
    }


def historical_case(context, labels, dev, hard):
    cid = context["case"]
    if context["dataset"] == "dev110":
        return {**dev[cid], "labels": labels}
    prior = hard[cid]
    packet = (HARD_HISTORY / cid / "pack.txt").read_text()
    start = packet.find("\nRANKED ")
    # Exactly the existing efficiency/hard_tuning.py formula.
    overhead = start if start >= 0 else len(packet.rstrip("\n"))
    return {
        "labels": labels,
        "floor_windows": [w for w in prior["windows"] if not w["role"].startswith("ranked")],
        "pack_chars": prior["pack_chars"],
        "ranked_capacity_chars": max(0, prior["pack_chars"] - overhead),
        "historical_selected_units": sum(w["role"].startswith("ranked") for w in prior["windows"]),
    }


def lab_inputs(bindings, observations, index):
    by_place = {
        item.id: unit
        for binding in bindings
        for unit in (source_unit(binding),)
        for item in items_to_judge(unit)
    }
    pairs, units, answers = {}, {}, {}
    for order, row in enumerate(observations):
        place = row["place"]
        unit = by_place[place["id"]]
        key = place["id"]
        pair = f"{row['point']}:{key}"
        if pair in pairs:
            raise ValueError(f"Duplicate observed point/place: {pair}")
        ranked = ep.RankedUnit(
            key,
            place["file"],
            tuple(map(tuple, place["ranges"])),
            unit.ranges,
            unit.symbol,
            max(row["probabilities"][role] for role in ("decide", "guard", "value", "effect", "satisfied")),
            1,
        )
        units[key] = {
            "place": key,
            "file": place["file"],
            "runs": place["ranges"],
            "extent": unit.ranges,
            "name": unit.symbol,
            "numbered_chars": len(ep._numbered(index, ranked)),
        }
        pairs[pair] = {"unit_key": key, "point_id": row["point"], "order": order}
        answers[pair] = row["probabilities"]
    return pairs, units, answers


def lab_rooms(history, bindings, observations, index, reducer, required):
    pairs, units, answers = lab_inputs(bindings, observations, index)
    overhead = history["pack_chars"] - history["ranked_capacity_chars"]
    rooms = {}
    for room in ROOMS:
        # The first column is the exact historical per-finding allowance, whose
        # mean is about 7,200 tokens. Larger columns change only total room.
        chars = history["pack_chars"] if room == 7200 else room * 4
        case = {**history, "ranked_capacity_chars": max(0, chars - overhead)}
        result = composed_case(case, pairs, units, answers, reducer, required)
        rooms[str(room)] = {
            "delivered": [label["delivered"] for label in result["labels"]],
            "selected_units": result["selected_units"],
            "ranked_chars": result["ranked_chars"],
            "capacity_chars": result["capacity_chars"],
            "total_room_chars": chars,
            "selected": result["selected"],
        }
    return rooms


def selection(context, folder, labels):
    page = load(folder / "llm-page.json")
    receipt = load(folder / "llm-receipt.json") if (folder / "llm-receipt.json").exists() else None
    candidates = list(ranked_units(context["case"], context["weights"], 512))
    offered = set(page["offered"])
    picked = set(receipt["picked"]) if receipt else set()
    if not offered.issubset({candidate["candidate_id"] for candidate in candidates}):
        raise ValueError("Frozen LLM page contains a candidate outside its ranked preparation")
    return {
        "top": {
            str(k): covered(labels, binding_windows(c["unit"] for c in candidates[:k])) for k in (16, 32, 128)
        },
        "page": covered(
            labels, binding_windows(c["unit"] for c in candidates if c["candidate_id"] in offered)
        ),
        "picked": covered(
            labels, binding_windows(c["unit"] for c in candidates if c["candidate_id"] in picked)
        )
        if receipt
        else None,
        "offered_ids": len(offered),
        "picked_ids": len(picked) if receipt else None,
        "invalid_ids": receipt.get("invalid_ids", []) if receipt else None,
        "duplicate_ids": receipt.get("duplicate_ids", []) if receipt else None,
        "parse_error": receipt.get("parse_error") if receipt else None,
        "llm": {
            "physical_calls": 1,
            "usd": float(receipt["usd"]),
            "catalog_usd": float(receipt["catalog_usd"]),
            "seconds": receipt["seconds"],
            "input_tokens": receipt["usage"]["prompt_tokens"],
            "output_tokens": receipt["usage"]["completion_tokens"],
        }
        if receipt
        else None,
    }


def aggregate(rows, arms, dataset):
    population = [row for row in rows if row["dataset"] == dataset]
    labels = sum(len(row["labels"]) for row in population)
    selected = [row["selection"] for row in population]
    llms = [row["llm"] for row in selected if row["llm"] is not None]
    result = {
        "cases": len(population),
        "registered_deciding_lines": labels,
        "historical_lab_tokens": {
            "mean": statistics.mean(row["historical_lab_chars"] / 4 for row in population),
            "minimum": min(row["historical_lab_chars"] / 4 for row in population),
            "maximum": max(row["historical_lab_chars"] / 4 for row in population),
        },
        "llm_selection": {
            "completed_cases": len(llms),
            "top": {str(k): sum(sum(row["top"][str(k)]) for row in selected) for k in (16, 32, 128)},
            "page": sum(sum(row["page"]) for row in selected),
            "picked": sum(sum(row["picked"]) for row in selected if row["picked"] is not None),
            "invalid_ids": sum(len(row["invalid_ids"] or []) for row in selected),
            "duplicate_ids": sum(len(row["duplicate_ids"] or []) for row in selected),
            "parse_errors": sum(row["parse_error"] is not None for row in selected),
            "empty_picks": sum(row["picked_ids"] == 0 for row in selected),
            "offered_ids_mean": statistics.mean(row["offered_ids"] for row in selected),
            "picked_ids_mean": statistics.mean(
                row["picked_ids"] for row in selected if row["picked_ids"] is not None
            )
            if llms
            else None,
            "usd": sum(row["usd"] for row in llms),
            "catalog_usd": sum(row["catalog_usd"] for row in llms),
            "seconds": sum(row["seconds"] for row in llms),
            "seconds_per_finding": statistics.mean(row["seconds"] for row in llms) if llms else None,
            "input_tokens": sum(row["input_tokens"] for row in llms),
            "output_tokens": sum(row["output_tokens"] for row in llms),
            "input_tokens_mean": statistics.mean(row["input_tokens"] for row in llms) if llms else None,
            "input_tokens_minimum": min((row["input_tokens"] for row in llms), default=None),
            "input_tokens_maximum": max((row["input_tokens"] for row in llms), default=None),
        },
        "arms": {},
    }
    for name in arms:
        finished = [row for row in population if name in row["arms"]]
        values = [row["arms"][name] for row in finished]
        overhead = [
            row["outline_choice"] if name == "outline" else row["selection"]["llm"] if name == "llm" else None
            for row in finished
        ]
        role_usd = float(sum((Decimal(row["usd"]) for row in values), Decimal(0)))
        selection_usd = sum(row["usd"] for row in overhead if row)
        role_calls = sum(row["new_physical_calls"] for row in values)
        choice_calls = sum(row["physical_calls"] for row in overhead if row) if name == "outline" else 0
        result["arms"][name] = {
            "complete": len(finished) == len(population),
            "completed_cases": len(finished),
            "completed_registered_lines": sum(len(row["labels"]) for row in finished),
            "failures": sum(row["failure"] is not None for row in values),
            "stopped_by": dict(
                (key, sum(row["stopped_by"] == key for row in values))
                for key in {row["stopped_by"] for row in values}
            ),
            "judged_reach": sum(sum(row["judged_reach"]) for row in values),
            "judged_reach_with_lab_floor": sum(sum(row["judged_reach_with_lab_floor"]) for row in values),
            "judged_reach_with_native_floor": sum(
                sum(row["judged_reach_with_native_floor"]) for row in values
            ),
            "role_logical_calls": sum(row["role_logical_calls"] for row in values),
            "role_physical_calls": role_calls,
            "choice_physical_calls": choice_calls,
            "total_jev_physical_calls": role_calls + choice_calls,
            "jev_calls_per_completed_finding": (role_calls + choice_calls) / len(finished)
            if finished
            else None,
            "llm_physical_calls": len(finished) if name == "llm" else 0,
            "role_usd": role_usd,
            "selection_usd": selection_usd,
            "total_usd": role_usd + selection_usd,
            "usd_per_completed_finding": (role_usd + selection_usd) / len(finished) if finished else None,
            "role_http_seconds": sum(row["http_seconds"] for row in values),
            "judging_and_packing_wall_seconds": sum(row["wall_seconds"] for row in values),
            "selection_seconds": sum(row["seconds"] for row in overhead if row),
            "selection_plus_judging_wall_seconds_per_finding": (
                sum(row["wall_seconds"] for row in values) + sum(row["seconds"] for row in overhead if row)
            )
            / len(finished)
            if finished
            else None,
            "rooms": {
                str(room): {
                    "lab": sum(sum(row["lab_rooms"][str(room)]["delivered"]) for row in values),
                    "native": sum(sum(row["rooms"][str(room)]["delivered"]) for row in values),
                    "lab_selected_units": sum(
                        row["lab_rooms"][str(room)]["selected_units"] for row in values
                    ),
                }
                for room in ROOMS
            },
        }
    return result


def spend():
    latest = {}
    if (OUT / "spend-ledger.jsonl").exists():
        with (OUT / "spend-ledger.jsonl").open() as lines:
            for line in lines:
                event = json.loads(line)
                latest[event["id"]] = event
    categories = {}
    for category in ("planner", "agent", "guard", "jev"):
        events = [row for row in latest.values() if row["category"] == category]
        categories[category] = {
            "physical_calls": len(events),
            "settled_usd": float(
                sum((Decimal(row["usd"]) for row in events if row["status"] == "settled"), Decimal(0))
            ),
            "unresolved_reserved_usd": float(
                sum((Decimal(row["usd"]) for row in events if row["status"] != "settled"), Decimal(0))
            ),
            "unresolved_calls": sum(row["status"] != "settled" for row in events),
        }
    return {
        "cap_usd": 2.5,
        "categories": categories,
        "spent_or_reserved_usd": float(sum((Decimal(row["usd"]) for row in latest.values()), Decimal(0))),
    }


def main():
    resources()
    dev = cases_only(COMPARE / "step-3/metadata.json")
    hard = cases_only(HARD_HISTORY / "windows.json")
    reducer = proposal_reducer(COMPARE / "step-1b/roles-PROPOSAL.md")
    required = load(COMPARE / "step-2/selection-policy.json")["required_roles"]
    cached, rows = {}, []
    for folder in sorted((OUT / "cases").iterdir()):
        context, labels = load(folder / "context.json"), load(folder / "labels.json")
        if any(label["first_line"] != label["last_line"] for label in labels):
            raise ValueError("Native trial delivery receipts require registered single deciding lines")
        history = historical_case(context, labels, dev, hard)
        row = {
            "case": context["case"],
            "dataset": context["dataset"],
            "labels": labels,
            "historical_lab_chars": history["pack_chars"],
            "selection": selection(context, folder, labels),
            "outline_choice": invoice(folder / "outline-http"),
            "arms": {},
        }
        completed = [name for name in ARMS if (folder / name / "result.json").exists()]
        if completed:
            relations, index = scopes(context, cached)
            request = ep.pack_request(
                relations,
                context["repository"],
                context["claim"],
                context.get("points"),
                budget=MeasuredRoom(7200),
            )
            native_floor = (
                [
                    {"file": file, "first_line": line, "last_line": line}
                    for file, line in shown_lines(SimpleNamespace(regions=request.floor))
                ]
                if request
                else []
            )
            for name in completed:
                destination = folder / name
                result = load(destination / "result.json")
                observations = load(destination / "observations.json")
                judged = observed_windows(observations)
                result.update(
                    judged_reach=covered(labels, judged),
                    judged_reach_with_lab_floor=covered(labels, [*judged, *history["floor_windows"]]),
                    judged_reach_with_native_floor=covered(labels, [*judged, *native_floor]),
                    lab_rooms=lab_rooms(
                        history, load(destination / "units.json"), observations, index, reducer, required
                    ),
                )
                row["arms"][name] = result
        rows.append(row)
    datasets = {
        "dev110": aggregate(rows, ("code", "outline", "llm"), "dev110"),
        "hard27": aggregate(rows, ARMS, "hard27"),
    }
    assert datasets["dev110"]["cases"] == 110 and datasets["dev110"]["registered_deciding_lines"] == 201
    assert datasets["hard27"]["cases"] == 13 and datasets["hard27"]["registered_deciding_lines"] == 28
    write(OUT / "per-finding-summary.json", rows)
    summary = {
        "development_measure": True,
        "provider_calls_from_this_report": 0,
        "packing_calls": 0,
        "spend": spend(),
        "datasets": datasets,
        "limitations": [
            "Room 7200 is exact historical per-finding lab capacity and a uniform native diagnostic room.",
            "Lab larger rooms preserve floor and overhead; native rooms rebuild the owner's floor.",
            "Judged reach counts observed item spans, never merely admitted or LLM-offered candidates.",
            "Code top K and LLM retained spans exclude floors; floor-inclusive judged reach is separate.",
            "The singular native floor-inclusive reach uses the requested 7200-token diagnostic floor.",
            "Selection plus judging wall time excludes corpus preparation, guards and cold code ranking.",
            "Arm aggregates include completed results with explicit denominators and incomplete status.",
        ],
    }
    write(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
