"""Prepare compact real outlines and a priced trial, without invoking a provider.

Meta Builder prepare runs separately. Token counts are sensitivity estimates from recorded
physical request attribution, not tokenizer measurements or billed tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.client import JEV_INPUT_LIMITS
from jev_navigator.judgments.secrets import SecretMasker, mask_request
from jev_navigator.selection import OutlineLimits, outline, outline_choice


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("Trial preparation permits zero provider calls")


sys.addaudithook(deny_network)


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def priced(request):
    size = len(json.dumps({"state": request["state"], "questions": request["questions"]}).encode())
    tokens = [254 + density * size for density in (0.23, 0.30)]
    return {
        "body_bytes": size,
        "estimated_tokens": tokens,
        "estimated_usd": [n * 0.042 / 1e6 for n in tokens],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--proof", type=Path, default=Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"
    )
    args = parser.parse_args()
    db = sqlite3.connect(args.data / "catalogue.sqlite")
    fit = load(args.out / "ranking.json")
    weights = {cid: part["weights"] for part in fit["folds"] for cid in part["held_out_cases"]}
    inputs = {}
    for name in ("dev110", "hard27"):
        inputs.update(load(args.proof / f"runs/pack-49b78955/pack-inputs-{name}.json")["cases"])
    indexes, receipts = {}, []
    candidate_hashes = {}
    with (args.out / "outline-requests.jsonl").open("w") as stream:
        for path in sorted(args.data.glob("*.npz")):
            receipt = load(path.with_suffix(".json"))
            cid = receipt["case"]
            row = inputs[cid]
            root = row["repository"]
            if root not in indexes:
                base = CodeIndex.from_git(Path(root), ["."])
                indexes[root] = CodeIndex(
                    Path(root), [file for file in base.files if file not in row["withheld"]]
                )
            index = indexes[root]
            with np.load(path) as arrays:
                ids = arrays["ids"].tolist()
                chosen_weights = weights.get(cid, fit["hard27_frozen_weights"])
                order = np.argsort(-(arrays["features"] @ np.asarray(chosen_weights)), kind="stable")
            files = []
            for position in order:
                binding = json.loads(
                    db.execute(
                        "select binding from documents where root=? and id=?", (root, ids[int(position)])
                    ).fetchone()[0]
                )
                if binding and binding["path"] not in files:
                    files.append(binding["path"])
                if len(files) == 16:
                    break
            query = row["claim"]["statement"]
            limits = OutlineLimits()
            while True:
                offered = [outline(index, file, (query,), limits=limits) for file in files]
                request = outline_choice(query, offered)
                state, questions, _ = mask_request(request["state"], request["questions"], SecretMasker())
                request = {"model": "jev-latest", "state": state, "questions": questions}
                if not JEV_INPUT_LIMITS.exceeded_by(state, questions):
                    break
                limits = OutlineLimits(
                    max(1, limits.definitions // 2),
                    max(1, limits.imports // 2),
                    max(1, limits.matches // 2),
                    max(1, limits.line_chars // 2),
                )
                if limits == OutlineLimits(1, 1, 1, 1):
                    raise ValueError("The compact outline request cannot fit")
            candidate = {
                "case_id": cid,
                "group_id": "selection-outlines",
                "revision_id": "outline-choice-v1",
                "request": request,
                "intended_uses": {
                    "read_file": (
                        "Choose one offered file to read in full. Keep none or low confidence unresolved "
                        "and retain the code ranking. This is navigation evidence, "
                        "never a verdict about unread implementation."
                    )
                },
                "workflow": {
                    "purpose": (
                        "Choose a visible file for the next full source read. "
                        "Confidence policy needs a future measured trial."
                    ),
                    "consumer_code": (
                        "answer = answers['read_file']\nchoice = answer.choice\n"
                        "confidence = answer.confidence\n"
                        "if choice != 'none' and confidence >= min_confidence:\n"
                        "    next_file = files[choice]['path']\nelse:\n    next_file = None"
                    ),
                },
            }
            size = priced(request)
            stream.write(json.dumps({"case": cid, "request": request, **size}) + "\n")
            receipts.append({"case": cid, "population": receipt["dataset"], "files": files, **size})
            population = "dev110" if cid.startswith("analysis-engine:") else "hard27"
            if population not in candidate_hashes:
                candidate_path = args.out / f"outline-{population}-candidate.json"
                write(candidate_path, candidate)
                candidate_hashes[population] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    totals = [sum(record["estimated_usd"][i] for record in receipts) for i in (0, 1)]
    # Two frozen trial arms, each up to eight groups of 16 full bodies. This uses the recorded
    # roles16 mean as a forecast. Exact instantiated group sizes remain the admission owner.
    role_trial = len(receipts) * 2 * 8 * 0.000656705
    write(
        args.out / "trial-price.json",
        {
            "provider_calls": 0,
            "usd": 0,
            "development_measure": True,
            "cases": len(receipts),
            "outline_requests": len(receipts),
            "outline_total_estimated_usd": totals,
            "outline_tokens_mean": [
                sum(r["estimated_tokens"][i] for r in receipts) / len(receipts) for i in (0, 1)
            ],
            "historical_roles16_tokens_mean": 15635.8,
            "historical_roles16_usd_mean": 0.000656705,
            "two_arm_role_requests_max": len(receipts) * 2 * 8,
            "two_arm_role_usd_forecast": role_trial,
            "total_usd_forecast": [role_trial + total for total in totals],
            "proposed_cap_usd": 1.50,
            "cap_owner": (
                "Exact billed-token receipts stop provider admission at the approved dollar cap. "
                "Eight requests is a per-finding trial guard, never a time budget."
            ),
            "next_action": "STOP. Only a resumed go from Andre can run paid review or trial calls.",
            "candidate_sha256": candidate_hashes,
            "token_method": (
                "Exact JSON body bytes with historical sensitivity 254 + (0.23 to 0.30)*bytes. "
                "Not billed tokens. Role forecast uses the historical mean, not new group measurements."
            ),
            "receipts": receipts,
        },
    )


if __name__ == "__main__":
    main()
