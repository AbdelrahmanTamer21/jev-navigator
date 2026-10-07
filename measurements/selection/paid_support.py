"""Shared receipts and money ownership for the authorized Case 1 measurement."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from decimal import Decimal
from pathlib import Path
from threading import Lock

ROOT = Path.home() / ".local/share/jvn-takeover/2026-10-03/search-design/case1"
PROOF = Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"
OUT = ROOT / "paid-trial-20261007"
MODEL = "sference/deepseek-v4-flash-0731"
REQUESTY = "https://router.eu.requesty.ai/v1"
ROOMS = (7200, 20000, 36000)


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def resources():
    disk = shutil.disk_usage(ROOT).free
    vm = subprocess.check_output(["vm_stat"], text=True)
    page = int(re.search(r"page size of (\d+)", vm)[1])
    counts = dict((key, int(n)) for key, n in re.findall(r"(Pages [^:]+):\s+(\d+)", vm))
    memory = sum(counts.get("Pages " + name, 0) for name in ("free", "inactive", "speculative")) * page
    if disk < 30 * 10**9 or memory < 8 * 10**9:
        raise RuntimeError(f"Resource stop: disk={disk}, reclaimable memory={memory}")
    return {"disk_free": disk, "memory_available": memory}


class CapStopError(RuntimeError):
    pass


class Ledger:
    """Reserve every physical send. Uncertain usage stays reserved across restarts."""

    def __init__(self, path, cap="2.50"):
        self.path = path
        self.cap = Decimal(cap)
        self.lock = Lock()
        self.events = [json.loads(line) for line in path.open()] if path.exists() else []

    def latest(self):
        return {event["id"]: event for event in self.events}

    def balance(self):
        return self.cap - sum((Decimal(e["usd"]) for e in self.latest().values()), Decimal(0))

    def append(self, event):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)

    def reserve(self, identity, category, amount):
        with self.lock:
            if identity in self.latest():
                raise CapStopError(f"Existing reservation {identity} needs receipt reconciliation")
            amount = Decimal(str(amount))
            if amount < 0 or amount > self.balance():
                raise CapStopError(f"Cap stop: ${amount} requested, ${self.balance()} available")
            self.append({"id": identity, "category": category, "status": "reserved", "usd": str(amount)})

    def settle(self, identity, amount, **details):
        with self.lock:
            previous = self.latest()[identity]
            if previous["status"] != "reserved":
                raise ValueError("An attempt can settle only once")
            amount = Decimal(str(amount))
            if amount < 0:
                raise ValueError("Negative billed amount")
            self.append({**previous, **details, "status": "settled", "usd": str(amount)})
            if self.balance() < 0:
                raise CapStopError("Reported bill exceeded the reservation; stop dispatch")


def validated_pick(text, offered):
    """Typed ranked ids, validated against the actual visible list, with rejected ids retained."""
    document = json.loads(text)
    if not isinstance(document, dict) or set(document) != {"candidate_ids"}:
        raise ValueError("Expected only candidate_ids")
    ids = document["candidate_ids"]
    if not isinstance(ids, list) or len(ids) > 16 or any(not isinstance(key, str) for key in ids):
        raise ValueError("Expected at most sixteen string ids")
    accepted, invalid, duplicate = [], [], []
    for key in ids:
        if key not in offered:
            invalid.append(key)
        elif key in accepted:
            duplicate.append(key)
        else:
            accepted.append(key)
    return accepted, invalid, duplicate


PICK_SCHEMA = {
    "type": "object",
    "properties": {"candidate_ids": {"type": "array", "maxItems": 16, "items": {"type": "string"}}},
    "required": ["candidate_ids"],
    "additionalProperties": False,
}
