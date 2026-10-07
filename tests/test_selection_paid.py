"""Protect the real dollar admission boundary and typed shortlist membership."""

import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "measurements/selection"))
from paid_support import CapStopError, Ledger, frozen_prompt, validated_pick


def test_reservations_remain_durable_and_concurrent_sends_cannot_exceed_cap(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path, cap="0.10")

    def reserve(identity):
        try:
            ledger.reserve(identity, "jev", ".06")
            return identity
        except CapStopError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        admitted = [key for key in pool.map(reserve, ("first", "second")) if key]
    assert len(admitted) == 1
    reopened = Ledger(path, cap=".10")
    assert reopened.balance() == Decimal(".04")
    with pytest.raises(CapStopError):
        reopened.reserve(admitted[0], "jev", ".01")
    reopened.settle(admitted[0], ".02", usage={"input_tokens": 12})
    reopened.reserve("next", "guard", ".07")
    assert Ledger(path, cap=".10").balance() == Decimal(".01")


def test_llm_rank_keeps_only_offered_ids_and_retains_invalid_proposals():
    accepted, invalid, duplicates = validated_pick('{"candidate_ids":["b","invented","a","b"]}', {"a", "b"})
    assert accepted == ["b", "a"]
    assert invalid == ["invented"]
    assert duplicates == ["b"]
    assert validated_pick('{"candidate_ids":[]}', {"a"}) == ([], [], [])
    for text in ('{"candidate_ids":[7]}', '{"candidate_ids":[],"reason":"x"}', '{"candidate_ids":null}'):
        with pytest.raises(ValueError):
            validated_pick(text, {"a"})


def test_frozen_prompt_preserves_crlf_and_checks_exact_bytes(tmp_path):
    import hashlib
    import json

    wire = "A point\r\nA source line with München\r\n".encode()
    (tmp_path / "llm-prompt.txt").write_bytes(wire)
    (tmp_path / "llm-page.json").write_text(json.dumps({"prompt_sha256": hashlib.sha256(wire).hexdigest()}))
    assert frozen_prompt(tmp_path).encode() == wire
    (tmp_path / "llm-prompt.txt").write_bytes(wire.replace(b"\r\n", b"\n"))
    with pytest.raises(ValueError, match="Frozen prompt bytes changed"):
        frozen_prompt(tmp_path)


def test_physical_http_requests_are_reserved_before_send_and_settle_their_own_usage(tmp_path, monkeypatch):
    pytest.importorskip("typesafe_sdk")
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Barrier, Thread

    import paid_client

    ledger = Ledger(tmp_path / "spend.jsonl", cap=".10")
    rendezvous = Barrier(2)
    reservation_seen = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reservation_seen.append(
                any(event["status"] == "reserved" for event in Ledger(ledger.path).latest().values())
            )
            rendezvous.wait(timeout=5)
            tokens = body["state"]["marker"]
            response = json.dumps(
                {
                    "model": "jev-test",
                    "usage": {"input_tokens": tokens, "output_tokens": 0},
                    "answers": {"pick": {"type": "noul", "noul": 0.9}},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(paid_client, "resources", lambda: {})
    session = paid_client.JevSession(ledger, tmp_path / "http", "guard")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(
                pool.map(
                    lambda n: session.ask(
                        {"marker": n}, {"pick": {"type": "noul", "instructions": "Is the marker present?"}}
                    ),
                    (100, 200),
                )
            )
        assert all(reservation_seen)
        assert [reply.answers["pick"].probability for reply in replies] == [0.9, 0.9]
        settled = Ledger(ledger.path).latest().values()
        assert sorted(event["usage"]["input_tokens"] for event in settled) == [100, 200]
        assert ledger.balance() == Decimal(".10") - Decimal(300) * Decimal(".000000042")
        assert len(list((tmp_path / "http").glob("guard_*/request.json"))) == 2
        assert len(list((tmp_path / "http").glob("guard_*/response.json"))) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
