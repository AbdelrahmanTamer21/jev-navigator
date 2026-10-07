"""Execute the authorized frozen trial. Modes separate preparation, guard, selection and judging."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import sys
import time
import urllib.request
from decimal import Decimal
from pathlib import Path

from enginepy.workflows.document_analysis.code_relations import CodeRelations
from enginepy.workflows.document_analysis.import_neighbors import repository_import_maps
from paid_client import JevSession, pick_llm
from paid_native import build, packet_in_room
from paid_prepare import ranked_units
from paid_support import OUT, REQUESTY, ROOT, CapStopError, Ledger, load, resources, write

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.answers import JevResponse, NoulAnswer
from jev_navigator.judgments.judge import Judge


class Capture:
    model = "jev-latest"

    def __init__(self):
        self.requests = []

    def ask(self, state, questions):
        self.requests.append({"model": self.model, "state": state, "questions": questions})
        return JevResponse({name: NoulAnswer(0.5) for name in questions}, self.model)


def scopes(context, cached):
    root = context["repository"]
    if root not in cached:
        withheld = frozenset(context["withheld"])
        relations = CodeRelations(root, repository_import_maps(root, withheld), withheld=withheld)
        inventory = CodeIndex.from_git(Path(root), ["."])
        index = CodeIndex(Path(root), tuple(file for file in inventory.files if file not in withheld))
        cached[root] = relations, index
    return cached[root]


def role_candidate(request):
    return {
        "case_id": "selection-role-16",
        "revision_id": "case1-paid-roles-v2",
        "request": request,
        "intended_uses": {
            name: (
                "Retain the raw role probability. The unchanged owner composes relevance as max "
                "of decide, guard, value, effect and satisfied. Best local-role units are retained "
                "first. Delegates records forwarding. This is source selection, not a verdict."
            )
            for name in request["questions"]
        },
    }


async def shape():
    cached = {}
    for folder in sorted((OUT / "cases").iterdir()):
        context = load(folder / "context.json")
        candidates = list(ranked_units(context["case"], context["weights"], 128))
        relations, index = scopes(context, cached)
        capture = Capture()
        pack, result = await build(
            relations,
            index,
            context,
            candidates,
            Judge(capture, max_calls=8, items_per_request=16, max_concurrency=1),
        )
        write(
            folder / "code-shape.json",
            {
                "requests": capture.requests,
                "units": len(result.units),
                "judged_places": len({a.place.id for answers in result.judged.values() for a in answers}),
                "failure": str(result.failure) if result.failure else None,
                "stopped_by": result.stopped_by,
            },
        )
        if not (OUT / "roles-candidate.json").exists():
            full = next((r for r in capture.requests if len(r["state"]["items"]) == 16), None)
            if full:
                write(OUT / "roles-candidate.json", role_candidate(full))
        print(
            json.dumps(
                {
                    "dry_shape": context["case"],
                    "requests": len(capture.requests),
                    "failure": str(result.failure),
                }
            ),
            flush=True,
        )


def guards(ledger):
    sys.path.insert(0, str(Path.home() / "Projects/system-one-meta-builder/src"))
    from system_one_meta_builder import cli

    for name, path in (
        ("roles", OUT / "roles-candidate.json"),
        ("outline-dev110", ROOT / "outline-dev110-candidate.json"),
        ("outline-hard27", ROOT / "outline-hard27-candidate.json"),
    ):
        report = OUT / (name + "-guard.json")
        if report.exists():
            continue
        session = JevSession(ledger, OUT / "guards" / name, "guard")

        def call(submitted, questions, *, record_event, session=session):
            # Meta's own prepared review request and response stay in its receipts.
            raw, cached = session.ask_raw(submitted["state"], submitted["questions"])
            record_event(
                {"event": "reserved_exact_wire_receipt", "folder": str(session.folder), "cached": cached}
            )
            return raw, session.receipts[-1]["seconds"] if session.receipts else 0

        cli.call_once = call
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = cli.main(["guard", str(path), "--output", str(OUT / (name + "-guard-receipts.jsonl"))])
        report.write_text(output.getvalue())
        print(json.dumps({"guard": name, "exit": status, "balance": str(ledger.balance())}), flush=True)
        if status not in (0, 3):
            raise RuntimeError(f"Meta guard failed: {name}, exit {status}")


def planner(ledger):
    headers = {"Authorization": "Bearer " + os.environ["REQUESTY_API_KEY"]}
    with urllib.request.urlopen(urllib.request.Request(REQUESTY + "/models", headers=headers)) as response:
        catalog = json.load(response)
    model = next(row for row in catalog["data"] if row["id"] == "sference/deepseek-v4-flash-0731")
    if model.get("geolocation") != "eu":
        raise RuntimeError("The selected existing route is no longer EU")
    write(OUT / "provider-model.json", model)
    from concurrent.futures import ThreadPoolExecutor

    pending = [
        folder for folder in sorted((OUT / "cases").iterdir()) if not (folder / "llm-receipt.json").exists()
    ]

    def run(folder):
        receipt = pick_llm(ledger, folder, model)
        print(
            json.dumps(
                {
                    "llm": folder.name,
                    "picked": len(receipt["picked"]),
                    "invalid": len(receipt["invalid_ids"]),
                    "usd": receipt["usd"],
                    "seconds": round(receipt["seconds"], 2),
                    "balance": str(ledger.balance()),
                }
            ),
            flush=True,
        )

    if pending:
        run(pending[0])
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run, pending[1:]))


def observations(result):
    rows = []
    for point, answers in result.judged.items():
        for answer in answers:
            rows.append(
                {
                    "point": point,
                    "place": {
                        "id": answer.place.id,
                        "file": answer.place.file,
                        "ranges": answer.place.ranges,
                    },
                    "probabilities": {role: part.probability for role, part in answer.components.items()},
                    "request_sha256": answer.request_sha256,
                    "from_store": answer.from_store,
                }
            )
    return rows


async def judge_arms(ledger):
    cached = {}
    for folder in sorted((OUT / "cases").iterdir()):
        context = load(folder / "context.json")
        cid = context["case"]
        relations, index = scopes(context, cached)
        code = list(ranked_units(cid, context["weights"], 128))
        configurations = {"code": code}
        if context["dataset"] == "hard27":
            configurations["code_scent_filename"] = list(ranked_units(cid, [1, 0, 1, 0, 0, 0], 128))
        choice_path = folder / "outline-choice.json"
        if not choice_path.exists():
            request = load(folder / "outline-request.json")
            session = JevSession(ledger, folder / "outline-http")
            raw, _ = session.ask_raw(request["state"], request["questions"])
            answer = response_from_choice(raw)
            picked = (
                request["state"]["files"].get(answer.choice, {}).get("path")
                if answer.confidence >= 0.70
                else None
            )
            write(
                choice_path,
                {"choice": answer.choice, "confidence": answer.confidence, "picked_file": picked, "raw": raw},
            )
        choice = load(choice_path)
        if choice["picked_file"]:
            first = [
                c for c in ranked_units(cid, context["weights"]) if c["unit"]["path"] == choice["picked_file"]
            ]
            configurations["outline"] = (
                first + [c for c in code if c["unit"]["path"] != choice["picked_file"]]
            )[:128]
        else:
            configurations["outline"] = code
        picked = load(folder / "llm-receipt.json")["picked"]
        wanted = set(picked)
        pool = {
            c["candidate_id"]: c for c in ranked_units(cid, context["weights"]) if c["candidate_id"] in wanted
        }
        configurations["llm"] = [pool[key] for key in picked]
        for name, candidates in configurations.items():
            destination = folder / name
            if (destination / "result.json").exists():
                continue
            resources()
            start = time.perf_counter()
            session = JevSession(ledger, destination / "http")
            judge = Judge(session, max_calls=8, items_per_request=16, max_concurrency=1)
            pack, result = await build(relations, index, context, candidates, judge)
            observed = observations(result)
            write(destination / "observations.json", observed)
            write(destination / "units.json", [c["unit"] for c in candidates])
            write(destination / "selected-candidates.json", [c["candidate_id"] for c in candidates])
            rooms = {}
            for room in (7200, 20000, 36000):
                packet = packet_in_room(pack, relations, context, room)
                text = packet.render() if packet else ""
                (destination / f"packet-{room}.txt").write_text(text)
                shown = shown_lines(packet)
                rooms[str(room)] = {
                    "packet_chars": len(text),
                    "delivered": [
                        (label["file"], label["first_line"]) in shown
                        for label in load(folder / "labels.json")
                    ],
                }
            receipt = {
                "case": cid,
                "dataset": context["dataset"],
                "arm": name,
                "development_measure": True,
                "role_logical_calls": judge.calls,
                "new_physical_calls": len(session.receipts),
                "usd": str(sum(Decimal(r.get("usd", "0")) for r in session.receipts)),
                "http_seconds": sum(r["seconds"] for r in session.receipts),
                "wall_seconds": time.perf_counter() - start,
                "observed_pairs": len(observed),
                "rooms": rooms,
                "failure": str(result.failure) if result.failure else None,
                "stopped_by": result.stopped_by,
            }
            write(destination / "result.json", receipt)
            print(
                json.dumps(
                    {
                        "judged": cid,
                        "arm": name,
                        "new_calls": len(session.receipts),
                        "balance": str(ledger.balance()),
                        "failure": receipt["failure"],
                    }
                ),
                flush=True,
            )
            if isinstance(result.failure, CapStopError):
                raise result.failure


def response_from_choice(raw):
    from jev_navigator.judgments.answers import response_from_raw

    return response_from_raw(raw).answers["read_file"]


def shown_lines(packet):
    import re

    return (
        {
            (region.file, int(match[1]))
            for region in packet.regions
            for match in re.finditer(r"(?m)^\s*(\d+)\s*[:|]\s?", region.text)
        }
        if packet
        else set()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("shape", "guard", "planner", "judge"))
    args = parser.parse_args()
    # Read only the two existing provider credentials. Values never enter receipts or output.
    for path in (Path.home() / "Projects/analysis-engine/.env", Path.home() / ".config/jgrep/env"):
        for line in path.read_text().splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key in {"REQUESTY_API_KEY", "TYPESAFE_API_KEY", "TYPESAFE_BASE_URL"}:
                os.environ.setdefault(key, value.strip().strip("\"'"))
    ledger = Ledger(OUT / "spend-ledger.jsonl")
    if args.mode == "shape":
        asyncio.run(shape())
    elif args.mode == "guard":
        guards(ledger)
    elif args.mode == "planner":
        planner(ledger)
    else:
        asyncio.run(judge_arms(ledger))


if __name__ == "__main__":
    main()
