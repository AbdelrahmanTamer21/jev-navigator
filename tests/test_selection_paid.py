"""Protect the real dollar admission boundary and typed shortlist membership."""

import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "measurements/selection"))
from paid_support import CapStopError, Ledger, validated_pick


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
