"""Free preparation of frozen code queues and the bounded LLM selection page."""

from __future__ import annotations

import hashlib
import json
import sqlite3

import numpy as np
from paid_support import MODEL, OUT, PICK_SCHEMA, PROOF, ROOT, resources, write

from jev_navigator.judgments.secrets import SecretMasker


def ranked_units(cid, weights, limit=None):
    data = ROOT / "data"
    receipt = json.loads((data / (cid.replace(":", "_") + ".json")).read_text())
    with np.load(data / (cid.replace(":", "_") + ".npz")) as arrays:
        ids = arrays["ids"].tolist()
        order = np.argsort(-(arrays["features"] @ np.asarray(weights)), kind="stable")
    with sqlite3.connect(data / "catalogue.sqlite") as db:
        for position in order:
            row = db.execute(
                "select binding,code from documents where root=? and id=?",
                (receipt["root"], ids[int(position)]),
            ).fetchone()
            binding = json.loads(row[0])
            if binding:
                yield {"candidate_id": ids[int(position)], "unit": binding, "code": row[1]}
                if limit is not None:
                    limit -= 1
                    if limit == 0:
                        return


def picker_prompt(statement, candidates, budget_chars=44000):
    """One page in rank order. Labels never enter the prompt or page boundary."""
    from jev_navigator.selection import words

    prefix = (
        "Select source candidates to read in full to decide the code behavior described below. "
        "Return JSON with only candidate_ids: a ranked list of at most 16 ids copied exactly "
        "from the supplied candidates. Use an empty list if none connects. Select implementations, "
        "guards, value sources, effects or callers needed to decide the point. These are compact "
        "outlines; omitted code supplies no proof. Do not decide whether the finding is true.\n"
        + "POINT:\n"
        + statement
        + "\nCANDIDATES IN CODE RANK ORDER:\n"
    )
    terms = set(words(statement))
    page, offered = prefix, []
    masker = SecretMasker()
    for candidate in candidates:
        unit, code = candidate["unit"], candidate["code"]
        source = code.splitlines()
        matching = [line[:220] for line in source if terms.intersection(words(line))][:3]
        imports = [line[:220] for line in source if line.lstrip().startswith(("import ", "from "))][:2]
        visible = dict.fromkeys([*(source[:1]), *imports, *matching])
        text = (
            f"id={candidate['candidate_id']} file={unit['path']} "
            f"definition={unit['symbol']} ranges={unit['ranges']}\n"
            + "\n".join(visible)
            + f"\nomitted source lines={max(0, len(source) - len(visible))}\n"
        )
        if len(page) + len(text) > budget_chars:
            break
        page += text
        offered.append(candidate["candidate_id"])
    return masker.mask(page), offered


def main():
    resources()
    fit = json.loads((ROOT / "ranking.json").read_text())
    folds = {cid: fold["weights"] for fold in fit["folds"] for cid in fold["held_out_cases"]}
    inputs, labels = {}, {}
    allowed = {*(f"P{i}" for i in range(1, 8)), *(f"U{i}" for i in range(1, 7))}
    for dataset, case_name in (("dev110", "replay-dev110"), ("hard27", "hard27")):
        rows = json.loads((PROOF / f"runs/pack-49b78955/pack-inputs-{dataset}.json").read_text())["cases"]
        cases = json.loads((PROOF / f"cases/{case_name}.json").read_text())
        for case in cases:
            cid = case["case_id"]
            if dataset == "hard27" and cid not in allowed:
                continue
            inputs[cid] = {"dataset": dataset, **rows[cid]}
            labels[cid] = case["labels"]
    outline_requests = {
        row["case"]: row["request"] for row in map(json.loads, (ROOT / "outline-requests.jsonl").open())
    }
    manifest = []
    for cid, row in inputs.items():
        weights = folds.get(cid, fit["hard27_frozen_weights"])
        folder = OUT / "cases" / cid.replace(":", "_")
        units = list(ranked_units(cid, weights, 512))
        prompt, offered = picker_prompt(row["claim"]["statement"], units)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "llm-prompt.txt").write_text(prompt)
        write(folder / "context.json", {"case": cid, "weights": weights, **row})
        write(folder / "labels.json", labels[cid])
        write(
            folder / "llm-page.json",
            {
                "offered": offered,
                "schema": PICK_SCHEMA,
                "bytes": len(prompt.encode()),
                "tokens_proxy": len(prompt) / 4,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            },
        )
        write(folder / "outline-request.json", outline_requests[cid])
        manifest.append(
            {
                "case": cid,
                "dataset": row["dataset"],
                "offered": len(offered),
                "llm_tokens_proxy": len(prompt) / 4,
                "llm_bytes": len(prompt.encode()),
            }
        )
    write(
        OUT / "dry-run.json",
        {
            "provider_calls": 0,
            "development_measure": True,
            "cap_usd": 2.5,
            "model": MODEL,
            "llm_schema": PICK_SCHEMA,
            "cases": manifest,
            "llm_price_at_proxy_input_and_1024_output": sum(
                r["llm_tokens_proxy"] * 0.28 / 1e6 + 1024 * 0.56 / 1e6 for r in manifest
            ),
            "llm_conservative_reservation_total": sum(
                (r["llm_bytes"] + 1024) * 0.28 / 1e6 + 1024 * 0.56 / 1e6 for r in manifest
            ),
            "two_prepared_arms_forecast": [1.33058, 1.34180],
            "third_arm_roles_at_historical_mean": len(manifest) * 0.000656705,
            "guards_reserved_per_physical_request": (
                "serialized bytes times $0.042 per million plus 1,024 tokens"
            ),
            "choice_min_confidence": 0.70,
            "noul_yes_at": 0.80,
            "per_point": 5,
            "role_requests_guard": 8,
            "rooms": [7200, 20000, 36000],
            "resources": resources(),
            "llm_page_method": (
                "First rank-ordered 44,000-character page, about 11,000 proxy tokens. "
                "Actual usage reported separately. Remaining candidates are unshown."
            ),
        },
    )


if __name__ == "__main__":
    main()
