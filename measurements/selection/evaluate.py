"""Finding-held-out ranking and active-search diagnostics over the free #149 census.

Reference truth scores ranks and the explicitly named perfect oracle only. Stored single answers
are a development oracle, not valid answers to changed groups of 16. Exact group reuse is audited
separately. No provider can be called from this process.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from jev_navigator.selection import ActivePolicy, CodeGraph, Observation, active_search


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("The selection replay permits zero provider calls")


FIXED = {
    "scent": (1, 0, 0, 0, 0, 0),
    "filename": (0, 0, 1, 0, 0, 0),
    "walk": (0, 1, 0, 0, 0, 0),
    "scent_filename": (1, 0, 2, 0, 0, 0),
    "combined_default": (1, 1, 2, 0, -0.1, 0),
}
GRID = [
    (scent, walk, path, 0, -hub, -test)
    for scent, walk, path, hub, test in itertools.product(
        (0.5, 1), (0, 0.25, 1), (0, 1, 2), (0, 0.2), (0, 0.1)
    )
]


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fold(cid):
    return int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16) % 5


def ranks_for(case, order):
    positions = {case["ids"][int(i)]: rank for rank, i in enumerate(order, 1)}
    return [
        0
        if label.get("floor")
        else min((positions[key] for key in label["units"] if key in positions), default=None)
        for label in case["labels"]
    ]


def ordered(case, weights):
    return np.argsort(-(case["features"] @ np.asarray(weights, dtype=np.float32)), kind="stable")


def utility(ranks):
    if not ranks:
        return 0.0
    return sum(
        (float(rank <= 128) + 0.1 / math.log2(2 + rank)) if rank is not None else 0 for rank in ranks
    ) / len(ranks)


def fit(cases, candidates=GRID):
    """Case-normalized recall@128 plus a small discounted rank term; ties choose simpler weights."""
    values = []
    for weights in candidates:
        scores = [utility(ranks_for(case, ordered(case, weights))) for case in cases if case["labels"]]
        values.append(sum(scores) / max(1, len(scores)))
    winner = min(range(len(candidates)), key=lambda i: (-values[i], sum(abs(w) for w in candidates[i]), i))
    return candidates[winner], values[winner]


def summarize(ranks, cases):
    known = sorted(rank for rank in ranks if rank is not None)
    k = {}
    for target in (0.5, 0.8, 0.9):
        needed = math.ceil(len(ranks) * target)
        k[str(target)] = known[needed - 1] if needed <= len(known) else None
    return {
        "cases": cases,
        "labels": len(ranks),
        "ranked_labels": len(known),
        "K": k,
        "recall_counts": {
            str(n): sum(rank is not None and rank <= n for rank in ranks)
            for n in (16, 32, 64, 128, 256, 512, 768)
        },
        "rank_median": float(np.median(known)) if known else None,
        "rank_p90": known[math.ceil(len(known) * 0.9) - 1] if known else None,
    }


def read_cases(data):
    cases = []
    for path in sorted(data.glob("*.npz")):
        receipt = load(path.with_suffix(".json"))
        with np.load(path) as arrays:
            cases.append({**receipt, "ids": arrays["ids"].tolist(), "features": arrays["features"]})
    return cases


def apply_floors(cases, proof):
    metadata = load(proof / "runs/roles-compare-20261006/step-3/metadata.json")
    hard = load(proof / "runs/real-pack-e733ea0c-jvn0a73a59c-value-full/hard27/windows.json")["cases"]
    for case in cases:
        cid = case["case"]
        windows = (
            metadata["cases"][cid]["floor_windows"]
            if cid in metadata["cases"]
            else [window for window in hard[cid]["windows"] if not window["role"].startswith("ranked")]
        )
        for label in case["labels"]:
            label["floor"] = any(
                window["file"] == label["file"]
                and window["first_line"] <= label["line"] <= window["last_line"]
                for window in windows
            )


def cross_validate(cases, out):
    dev = [case for case in cases if case["case"].startswith("analysis-engine:")]
    hard = [case for case in cases if not case["case"].startswith("analysis-engine:")]
    if len(dev) != 110 or len(hard) != 13:
        raise ValueError(f"Incomplete population: {len(dev)} dev110 and {len(hard)} hard tuning")
    fits = []
    for held in range(5):
        training = [case for case in dev if fold(case["case"]) != held]
        testing = [case for case in dev if fold(case["case"]) == held]
        weights, score = fit(training)
        fits.append(
            {
                "fold": held,
                "weights": weights,
                "training_score": score,
                "training_cases": [case["case"] for case in training],
                "held_out_cases": [case["case"] for case in testing],
            }
        )
        print(json.dumps({"fit_fold": held, "weights": weights, "train_cases": len(training)}), flush=True)
    final_weights, _ = fit(dev)
    summaries = {}
    with (out / "deciding-ranks.jsonl").open("w") as stream:
        for population, selected in (("dev110", dev), ("hard27", hard)):
            aggregate = defaultdict(list)
            for case in selected:
                fitted = fits[fold(case["case"])]["weights"] if population == "dev110" else final_weights
                orders = {
                    "today_census": np.arange(len(case["ids"])),
                    **{name: ordered(case, weights) for name, weights in FIXED.items()},
                    "combined_cv": ordered(case, fitted),
                }
                case["weights"] = fitted
                case["combined_order"] = orders["combined_cv"]
                for name, order in orders.items():
                    ranks = ranks_for(case, order)
                    aggregate[name].extend(ranks)
                    for label, rank in zip(case["labels"], ranks, strict=True):
                        positions = {case["ids"][int(i)]: place for place, i in enumerate(order, 1)}
                        stream.write(
                            json.dumps(
                                {
                                    "case": case["case"],
                                    "population": population,
                                    "configuration": name,
                                    **label,
                                    "rank": rank,
                                    "unit_ranks": {key: positions[key] for key in label["units"]},
                                    "request_of_16": None if rank is None else math.ceil(rank / 16),
                                }
                            )
                            + "\n"
                        )
            summaries[population] = {
                name: summarize(ranks, len(selected)) for name, ranks in aggregate.items()
            }
    result = {
        "provider_calls": 0,
        "usd": 0,
        "development_measure": True,
        "method": "5 folds by sha256(case_id), whole findings held out. Hard P/U not tuned.",
        "objective": (
            "Mean per-finding recall@128 plus 0.1/log2(2+rank). Unchanged floor lines have rank zero."
        ),
        "grid": GRID,
        "folds": fits,
        "hard27_frozen_weights": final_weights,
        "summary": summaries,
    }
    write(out / "ranking.json", result)
    return dev, hard, result


def graph_for(path):
    recorded = load(Path(path))
    graph = CodeGraph(())
    graph.adjacency = recorded["adjacency"]
    return graph


def single_answers(proof, metadata):
    """Paid single-unit role observations can guide a development simulation without batch claims."""
    values = defaultdict(dict)
    for row in rows(proof / "runs/roles-compare-20261006/step-3/paid/roles/answer-table.jsonl"):
        values[row["place_id"]][row["question"]] = row["probability"]
    by_case = defaultdict(dict)
    for key, pair in metadata["pairs"].items():
        if key in values:
            probability = max(
                values[key][role] for role in ("decide", "guard", "value", "effect", "satisfied")
            )
            by_case[pair["case_id"]][pair["unit_key"]] = Observation(probability, probability >= 0.8)
    return by_case


class PerfectOracle:
    def __init__(self, labels):
        self.positive = {key for label in labels for key in label["units"]}

    def judge(self, candidates):
        return {key: Observation(float(key in self.positive), key in self.positive) for key in candidates}


class StoredOracle:
    def __init__(self, values):
        self.values = values

    def judge(self, candidates):
        return {key: self.values[key] for key in candidates if key in self.values}


def active_arms(dev, hard, proof, out):
    metadata = load(proof / "runs/roles-compare-20261006/step-3/metadata.json")
    stored = single_answers(proof, metadata)
    results = defaultdict(list)
    graph = None
    graph_path = None
    # Fixed marginal policies are sensitivity arms, never tuned on held-out outcomes.
    for case in [*dev, *hard]:
        if case["graph"] != graph_path:
            graph = graph_for(case["graph"])
            graph_path = case["graph"]
        order = case["combined_order"]
        ids = [case["ids"][int(i)] for i in order]
        scores = dict(
            zip(case["ids"], (case["features"] @ np.asarray(case["weights"])).tolist(), strict=True)
        )
        arms = {
            "perfect_oracle": (PerfectOracle(case["labels"]), ActivePolicy()),
            "perfect_no_marginal": (PerfectOracle(case["labels"]), ActivePolicy(min_expected_gain=0)),
            "stored_single_development": (StoredOracle(stored[case["case"]]), ActivePolicy()),
        }
        if not case["case"].startswith("analysis-engine:"):
            arms.pop("stored_single_development")
        for name, (oracle, policy) in arms.items():
            started = time.monotonic()
            result = active_search(ids, scores, graph, oracle, policy=policy)
            scheduled = [key for batch in result.batches for key in batch]
            positions = {key: rank for rank, key in enumerate(scheduled, 1) if key in result.observations}
            ranks = [
                0
                if label.get("floor")
                else min((positions[key] for key in label["units"] if key in positions), default=None)
                for label in case["labels"]
            ]
            # Store exact groups to make missing answers and group company reviewable.
            record = {
                "case": case["case"],
                "population": "dev110" if case["case"].startswith("analysis-engine:") else "hard27",
                "requests": len(result.batches),
                "scheduled_units": len(scheduled),
                "observed_units": len(result.observations),
                "unknown_units": len(result.unjudged),
                "pending_units": len(result.pending),
                "stopped_by": result.stopped_by,
                "expected_gain": result.expected_gain,
                "ranks": ranks,
                "groups": result.batches,
                "seconds": time.monotonic() - started,
                "estimated_usd_at_historical_roles_mean": len(result.batches) * 0.000656705,
                "provider_calls": 0,
                "usd": 0,
                "policy": asdict(policy),
            }
            results[name].append(record)
        print(
            json.dumps(
                {
                    "active": case["case"],
                    "requests": {
                        name: records[-1]["requests"]
                        for name, records in results.items()
                        if records[-1]["case"] == case["case"]
                    },
                }
            ),
            flush=True,
        )
    write(out / "active-cases.json", dict(results))
    summary = {}
    for name, records in results.items():
        summary[name] = {}
        for population in ("dev110", "hard27"):
            selected = [record for record in records if record["population"] == population]
            if not selected:
                continue
            summary[name][population] = {
                **summarize([rank for record in selected for rank in record["ranks"]], len(selected)),
                "requests_mean": sum(record["requests"] for record in selected) / len(selected),
                "offline_seconds_mean": sum(record["seconds"] for record in selected) / len(selected),
                "unknown_units": sum(record["unknown_units"] for record in selected),
                "stop_reasons": {
                    reason: sum(record["stopped_by"] == reason for record in selected)
                    for reason in {record["stopped_by"] for record in selected}
                },
            }
    write(
        out / "active-summary.json",
        {"development_measure": True, "provider_calls": 0, "usd": 0, "summary": summary},
    )


def main():
    sys.addaudithook(deny_network)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--proof", type=Path, default=Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"
    )
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = read_cases(args.data)
    apply_floors(cases, args.proof)
    dev, hard, _ = cross_validate(cases, args.out)
    if not args.static_only:
        active_arms(dev, hard, args.proof, args.out)


if __name__ == "__main__":
    main()
