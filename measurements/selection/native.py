"""Run reordered candidates through the anchored Engine packet with an exact-context-only client.

Invoke with Engine #1475 and JVN #148 on PYTHONPATH, using the Engine's interpreter. The selection
blocks themselves need not be imported: ranks and source bindings are immutable replay inputs.
No worktree outside Case 1 is edited. A cache miss ends search and retains the real failure packet.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from enginepy.workflows.document_analysis import evidence_pack as ep
from enginepy.workflows.document_analysis.code_relations import CodeRelations
from enginepy.workflows.document_analysis.import_neighbors import repository_import_maps
from jev_navigator.judgments.profiles import LOCAL_ROLES, ROLES_V2
from replay_store import ExactClient

from jev_navigator.directives.find_all import find_all_async
from jev_navigator.directives.frontier import Policy
from jev_navigator.index.units import RangeAnchor
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import content_hash
from jev_navigator.sources import Reach


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("The native selection replay permits zero provider calls")


sys.addaudithook(deny_network)


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def exact_store(proof, out):
    # The real asynchronous judge dispatches the synchronous client on worker threads.
    # The populated store is read-only during replay; every worker owns its connection.
    db = sqlite3.connect(out / "exact-requests.sqlite")
    db.execute("create table if not exists requests(hash text primary key,sent_body blob,response blob)")
    if db.execute("select count(*) from requests").fetchone()[0]:
        return db
    journal = proof / "runs/roles-compare-20261006/step-3/paid/roles16/journal.jsonl"
    with journal.open() as stream, db:
        for line in stream:
            record = json.loads(line)
            if record["kind"] != "http_attempt" or record.get("status") != 200:
                continue
            sent = base64.b64decode(record["sent_body_base64"])
            body = json.loads(sent)
            key = content_hash({"state": body["state"], "questions": body["questions"]})
            db.execute(
                "insert or ignore into requests values (?,?,?)",
                (key, sent, base64.b64decode(record["body_base64"])),
            )
    return db


@dataclass(frozen=True)
class RankedSource:
    locations: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    name: str = "selection"
    label: str = "code-ranked candidates"

    def reach(self, index, seeds):
        return [
            Reach(RangeAnchor(path, start, end), self.name, "code rank", 0)
            for path, ranges in self.locations
            for start, end in ranges
        ]


def shown_lines(packet):
    """Literal numbered source lines, never a region header's enclosing range."""
    return {
        (region.file, int(match[1]))
        for region in packet.regions
        for match in re.finditer(r"(?m)^\s*(\d+)\s*[:|]\s?", region.text)
    }


async def run(args):
    cases = load(args.proof / "cases/replay-dev110.json")
    inputs = load(args.proof / "runs/pack-49b78955/pack-inputs-dev110.json")["cases"]
    catalog = sqlite3.connect(args.data / "catalogue.sqlite")
    exact = exact_store(args.proof, args.out)
    ranked = load(args.out / "ranking.json")
    by_case = {cid: fit["weights"] for fit in ranked["folds"] for cid in fit["held_out_cases"]}
    scopes = {}
    records = []
    for case in cases:
        cid = case["case_id"]
        folder = args.out / "native" / cid.replace(":", "_")
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "result.json").exists():
            records.append(load(folder / "result.json"))
            continue
        row = inputs[cid]
        root = row["repository"]
        started = time.monotonic()
        if root not in scopes:
            withheld = frozenset(row["withheld"])
            relations = CodeRelations(root, repository_import_maps(root, withheld), withheld=withheld)
            scopes[root] = relations, ep.pack_index(relations, root)
        relations, index = scopes[root]
        request = ep.pack_request(relations, root, row["claim"], row.get("points"))
        client = ExactClient(exact)
        if request is None:
            packet = None
            failure = "no readable anchor"
        else:
            with np.load(args.data / (cid.replace(":", "_") + ".npz")) as arrays:
                ids = arrays["ids"].tolist()
                order = np.argsort(-(arrays["features"] @ np.asarray(by_case[cid])), kind="stable")[:128]
            locations = []
            for position in order:
                binding = catalog.execute(
                    "select binding from documents where root=? and id=?", (root, ids[int(position)])
                ).fetchone()[0]
                unit = json.loads(binding)
                if unit:
                    locations.append((unit["path"], tuple(map(tuple, unit["ranges"]))))
            source = RankedSource(tuple(locations))

            async def ranked_search(*positional, source=source, **kwargs):
                # Recipe composition only: source, policy and no further hops. The native source
                # resolver, real groups of 16, six-role judge and packet renderer retain ownership.
                kwargs.update(sources=(source,), hops=(), policy=Policy("selection", ranked=False))
                return await find_all_async(*positional, **kwargs)

            ep.find_all_async = ranked_search
            judge = Judge(client, scanner=None, max_calls=8, items_per_request=16)
            pack = await ep.build_pack(
                relations,
                request,
                judge,
                index,
                ep.PackSettings(callee_round=False, question_profile=ROLES_V2, required_roles=LOCAL_ROLES),
            )
            packet = pack.packet()
            failure = pack.failure
            (folder / "packet.txt").write_text(packet.render())
        shown = shown_lines(packet) if packet else set()
        labels = [
            {**label, "delivered": (label["file"], label["first_line"]) in shown} for label in case["labels"]
        ]
        result = {
            "case": cid,
            "labels": labels,
            "packet_chars": len(packet.render()) if packet else 0,
            "seconds": time.monotonic() - started,
            "requests_attempted": len(client.attempts),
            "exact_requests_reused": sum(record["exact_context"] for record in client.attempts),
            "unknown_request_count": sum(not record["exact_context"] for record in client.attempts),
            "failure": failure,
            "provider_calls": 0,
            "usd": 0,
            "development_measure": True,
        }
        write(folder / "requests.json", client.attempts)
        write(folder / "result.json", result)
        records.append(result)
        print(
            json.dumps({key: result[key] for key in ("case", "seconds", "requests_attempted", "failure")}),
            flush=True,
        )
    write(
        args.out / "native-summary.json",
        {
            "provider_calls": 0,
            "usd": 0,
            "development_measure": True,
            "cases": records,
            "delivered": sum(label["delivered"] for record in records for label in record["labels"]),
            "exact_requests_reused": sum(record["exact_requests_reused"] for record in records),
            "unknown_requests": sum(record["unknown_request_count"] for record in records),
            "wall_seconds_mean": sum(record["seconds"] for record in records) / len(records),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--proof", type=Path, default=Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
