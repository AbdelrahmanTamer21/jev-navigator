"""Frozen-pool allocator diagnostic. Changed groups do not inherit exact-context validity.

Run with the retained pinned find-eval harness and Engine interpreter. Reuses its approved
proposal reducer and real PackUnit renderer; no second pack implementation or provider.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from find_eval.simulate import composed_case, pair_probabilities, proposal_reducer


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("The frozen allocation replay permits zero provider calls")


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def location(unit):
    return unit["file"], tuple(map(tuple, unit["runs"]))


def main():
    sys.addaudithook(deny_network)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--proof", type=Path, default=Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"
    )
    args = parser.parse_args()
    base = args.proof / "runs/roles-compare-20261006"
    metadata = load(base / "step-3/metadata.json")
    reducer = proposal_reducer(base / "step-1b/roles-PROPOSAL.md")
    probabilities, models = pair_probabilities(
        base / "step-3/paid/roles16/answer-table.jsonl",
        load(base / "step-3/roles16-candidate.json")["request"]["questions"],
    )
    policy = load(base / "step-2/selection-policy.json")
    print(json.dumps({"required_policy": policy}), flush=True)
    required = policy["required_roles"]
    by_case = defaultdict(dict)
    for key, pair in metadata["pairs"].items():
        by_case[pair["case_id"]][key] = pair
    db = sqlite3.connect(args.data / "catalogue.sqlite")
    fit = load(args.out / "ranking.json")
    weights = {cid: part["weights"] for part in fit["folds"] for cid in part["held_out_cases"]}
    root = next(iter(db.execute("select root from cases where dataset='dev110'").fetchone()))
    bindings = {}
    for key, raw in db.execute("select id,binding from documents where root=?", (root,)):
        unit = json.loads(raw)
        if unit:
            bindings[key] = (unit["path"], tuple(map(tuple, unit["ranges"])))
    summaries = defaultdict(list)
    for cid, case in metadata["cases"].items():
        with np.load(args.data / (cid.replace(":", "_") + ".npz")) as arrays:
            score = arrays["features"] @ np.asarray(weights[cid])
            location_scores = {}
            for key, value in zip(arrays["ids"].tolist(), score.tolist(), strict=True):
                if key in bindings:
                    place = bindings[key]
                    location_scores[place] = max(location_scores.get(place, -math.inf), value)
        pairs = by_case[cid]
        original = sorted(pairs, key=lambda key: pairs[key]["order"])
        ranked = sorted(
            original,
            key=lambda key: -location_scores.get(location(metadata["units"][pairs[key]["unit_key"]]), 0),
        )
        configurations = {"today_frozen_full": original}
        for count in (16, 32, 64, 128, 256):
            selected_units = list(dict.fromkeys(pairs[key]["unit_key"] for key in ranked))[:count]
            configurations[f"combined_cv_top{count}"] = [
                key for key in ranked if pairs[key]["unit_key"] in selected_units
            ]
        for name, ordered in configurations.items():
            started = time.monotonic()
            selected_pairs = {key: {**pairs[key], "order": order} for order, key in enumerate(ordered)}
            result = composed_case(
                case,
                selected_pairs,
                metadata["units"],
                {key: probabilities[key] for key in ordered if key in probabilities},
                reducer,
                required,
            )
            record = {
                "case": cid,
                "configuration": name,
                "development_measure": True,
                "provider_calls": 0,
                "usd": 0,
                "seconds": time.monotonic() - started,
                "hypothetical_requests_of_16": math.ceil(
                    len({pairs[key]["unit_key"] for key in ordered}) / 16
                ),
                "fallback_unranked_units": len(
                    {
                        pairs[key]["unit_key"]
                        for key in ordered
                        if location(metadata["units"][pairs[key]["unit_key"]]) not in location_scores
                    }
                ),
                **result,
            }
            summaries[name].append(record)
        print(json.dumps({"lab": cid}), flush=True)
    write(args.out / "lab-cases.json", dict(summaries))
    write(
        args.out / "lab-summary.json",
        {
            "development_measure": True,
            "provider_calls": 0,
            "usd": 0,
            "served_models": sorted(models),
            "boundary": (
                "Frozen original role probabilities, code ranks and original historical floor and allocator. "
                "Changed batches are a development diagnostic. No new-context Jev accuracy is claimed."
            ),
            "summary": {
                name: {
                    "delivered": sum(label["delivered"] for record in records for label in record["labels"]),
                    "labels": sum(len(record["labels"]) for record in records),
                    "requests_mean": sum(record["hypothetical_requests_of_16"] for record in records)
                    / len(records),
                    "allocation_seconds_mean": sum(record["seconds"] for record in records) / len(records),
                    "estimated_usd_at_historical_mean": sum(
                        record["hypothetical_requests_of_16"] for record in records
                    )
                    / len(records)
                    * 0.000656705,
                }
                for name, records in summaries.items()
            },
        },
    )


if __name__ == "__main__":
    main()
