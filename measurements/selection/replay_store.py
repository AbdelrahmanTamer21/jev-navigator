"""Exact ordered-context replay with a read connection owned by the calling thread."""

from __future__ import annotations

import json
import sqlite3

from jev_navigator.judgments.answers import JevResponse, response_from_raw
from jev_navigator.judgments.questions import content_hash


class ExactClient:
    def __init__(self, db):
        self.store_path = db.execute("pragma database_list").fetchone()[2]
        self.attempts = []

    def ask(self, state, questions):
        key = content_hash({"state": state, "questions": questions})
        with sqlite3.connect(f"file:{self.store_path}?mode=ro", uri=True) as db:
            row = db.execute("select sent_body,response from requests where hash=?", (key,)).fetchone()
        matched = False
        if row:
            body = json.loads(row[0])
            # Canonical hashing locates a candidate. Serialized order authorizes reuse.
            matched = json.dumps(body["state"]) == json.dumps(state) and json.dumps(
                body["questions"]
            ) == json.dumps(questions)
        self.attempts.append(
            {
                "hash": key,
                "items": len(state["items"]),
                "exact_context": matched,
                "request": {"state": state, "questions": questions},
            }
        )
        if not matched:
            raise LookupError("No exact stored request with these ordered units, questions and target bytes")
        response = response_from_raw(json.loads(row[1]))
        return JevResponse(response.answers, response.model, request_sha256=key, from_store=True)
