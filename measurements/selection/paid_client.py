"""Every physical HTTP send has a durable reservation and exact wire receipt."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
import uuid
from dataclasses import replace
from decimal import Decimal
from threading import local

import httpx2
from paid_support import MODEL, PICK_SCHEMA, REQUESTY, frozen_prompt, load, resources, validated_pick, write
from typesafe_sdk import RetryPolicy, TypeSafeClient

from jev_navigator.judgments.answers import response_from_raw
from jev_navigator.judgments.questions import content_hash
from jev_navigator.judgments.secrets import SecretScanner, refuse_if_secret

JEV_RATE = Decimal("0.000000042")


class JevSession:
    model = "jev-latest"

    def __init__(self, ledger, folder, category="jev"):
        self.ledger, self.folder, self.category = ledger, folder, category
        self.receipts = []
        self.active = local()

    def before(self, request):
        resources()
        wire = request.read()
        identity = self.category + ":" + uuid.uuid4().hex
        folder = self.folder / identity.replace(":", "_")
        folder.mkdir(parents=True, exist_ok=True)
        reserve = Decimal(len(wire) + 1024) * JEV_RATE
        self.ledger.reserve(identity, self.category, reserve)
        (folder / "request.json").write_bytes(wire)
        self.active.current = identity, folder, time.perf_counter(), wire

    def after(self, response):
        response.read()
        identity, folder, start, wire = self.active.current
        (folder / "response.json").write_bytes(response.content)
        document = response.json()
        usage = document.get("usage") or {}
        receipt = {
            "id": identity,
            "category": self.category,
            "status": response.status_code,
            "seconds": time.perf_counter() - start,
            "usage": usage,
            "model": document.get("model"),
            "request_sha256": hashlib.sha256(wire).hexdigest(),
        }
        if "input_tokens" in usage:
            usd = Decimal(usage["input_tokens"]) * JEV_RATE
            self.ledger.settle(identity, usd, usage=usage, seconds=receipt["seconds"], model=receipt["model"])
            receipt["usd"] = str(usd)
        else:
            receipt["unpriced_reserved"] = True
        write(folder / "receipt.json", receipt)
        self.receipts.append(receipt)

    def ask_raw(self, state, questions):
        refuse_if_secret(state, questions, SecretScanner())
        # Exact ordered request caching is owned by this trial, not the provider.
        key = hashlib.sha256(
            json.dumps({"state": state, "questions": questions}, ensure_ascii=False).encode()
        ).hexdigest()
        cached = self.folder / "exact" / (key + ".json")
        if cached.exists():
            return load(cached), True
        with (
            httpx2.Client(
                timeout=180, event_hooks={"request": [self.before], "response": [self.after]}
            ) as transport,
            TypeSafeClient(retry=RetryPolicy(max_retries=0), http_client=transport) as sdk,
        ):
            result = sdk.system_one(state=state, questions=questions, model=self.model)
            raw = json.loads(result.raw_http_response.content)
        write(cached, raw)
        return raw, False

    def ask(self, state, questions):
        raw, cached = self.ask_raw(state, questions)
        response = response_from_raw(raw)
        return replace(
            response,
            request_sha256=content_hash({"state": state, "questions": questions}),
            input_tokens=None if cached else response.input_tokens,
            from_store=cached,
        )


def pick_llm(ledger, folder, model):
    receipt_path = folder / "llm-receipt.json"
    if receipt_path.exists():
        return load(receipt_path)
    page = load(folder / "llm-page.json")
    prompt = frozen_prompt(folder)
    identity = "planner:" + folder.name
    reserve = Decimal(len(prompt.encode()) + 1024) * Decimal(str(model["input_price"]))
    reserve += Decimal(1024) * Decimal(str(model["output_price"]))
    resources()
    ledger.reserve(identity, "planner", reserve)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1024,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "ranked_source_pick", "strict": True, "schema": PICK_SCHEMA},
        },
    }
    wire = json.dumps(body, ensure_ascii=False).encode()
    (folder / "llm-request.json").write_bytes(wire)
    started = time.perf_counter()
    request = urllib.request.Request(
        REQUESTY + "/chat/completions",
        data=wire,
        headers={
            "Authorization": "Bearer " + os.environ["REQUESTY_API_KEY"],
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = response.read()
    (folder / "llm-response.json").write_bytes(raw)
    seconds = time.perf_counter() - started
    reply = json.loads(raw)
    usage = reply["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    calculated = Decimal(usage["prompt_tokens"] - cached) * Decimal(str(model["input_price"]))
    calculated += Decimal(cached) * Decimal(str(model["cached_price"]))
    calculated += Decimal(usage["completion_tokens"]) * Decimal(str(model["output_price"]))
    usd = Decimal(str(usage.get("cost", usage.get("total_cost", calculated))))
    ledger.settle(identity, usd, usage=usage, seconds=seconds, model=reply.get("model"))
    choice = reply["choices"][0]
    receipt = {
        "usage": usage,
        "usd": str(usd),
        "catalog_usd": str(calculated),
        "seconds": seconds,
        "served_model": reply.get("model"),
        "finish_reason": choice.get("finish_reason"),
    }
    try:
        if choice.get("finish_reason") != "stop":
            raise ValueError("Partial LLM output is not a selection")
        picked, invalid, duplicates = validated_pick(choice["message"]["content"], set(page["offered"]))
        receipt.update(picked=picked, invalid_ids=invalid, duplicate_ids=duplicates)
    except (ValueError, TypeError) as error:
        receipt.update(picked=[], invalid_ids=[], duplicate_ids=[], parse_error=str(error))
    write(receipt_path, receipt)
    return receipt
