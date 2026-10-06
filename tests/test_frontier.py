"""The frontier's two policies over one search: today's stage order, and value order by code features."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from shop_search import FROZEN, REQUEST, SHOP, TARGETS, sent_requests, shop_index

from jev_navigator.directives.find_all import NOT_REACHED, find_all, find_all_async
from jev_navigator.directives.frontier import STAGE_ORDER, VALUE
from jev_navigator.judgments.judge import Judge
from jev_navigator.testing import AsyncScriptedJevClient, ScriptedJevClient

PRODUCTION_WITHOUT_NAMES = [
    "export function submitOrder(order) {",
    "export function cancelOrder(id) {",
    "def invoice(order):",
    "def refund(order):",
]
RENAMED = {path.replace("billing/", "zzz/"): text for path, text in SHOP.items()}
RENAMED_REQUEST = {**REQUEST, "files": [path.replace("billing/", "zzz/") for path in REQUEST["files"]]}


def first_lines(requests: list[list[list]]) -> list[str]:
    return [line for request in requests for _, line in request]


def test_the_stage_order_policy_and_the_default_send_the_requests_find_all_sent_before_the_frontier(
    tmp_path: Path,
) -> None:
    # Arrange
    index = shop_index(tmp_path)
    frozen = json.loads(FROZEN.read_text())

    # Act
    stage = {
        "uncapped": sent_requests(index, policy=STAGE_ORDER),
        "three_calls": sent_requests(index, 3, policy=STAGE_ORDER),
    }
    default = sent_requests(index)

    # Assert
    assert stage == frozen
    assert default == frozen["uncapped"]


def test_the_value_policy_judges_the_unit_defining_a_name_first_and_the_test_last(tmp_path: Path) -> None:
    # Arrange: check_limit defines a name and holds both; place_order is the anchor and holds one; the
    # limit constant and cancel_order sit in files the anchor's file is or imports; the test holds both
    # names but is a test
    index = shop_index(tmp_path)

    # Act
    judged = first_lines(sent_requests(index, policy=VALUE))

    # Assert
    assert judged[:4] == [
        "def check_limit(order):",
        "def place_order(order):",
        "MAX_ITEMS = 4",
        "def cancel_order(order):",
    ]
    assert sorted(judged[4:8]) == sorted(PRODUCTION_WITHOUT_NAMES)
    assert judged[8:] == ["def test_check_limit_refuses_one_item_over(order_factory):"]


def test_renaming_a_folder_leaves_the_value_order_and_ranking_unchanged_while_it_moves_the_stage_order(
    tmp_path: Path,
) -> None:
    # Arrange: billing/ sorts before orders/, zzz/ after it
    index = shop_index(tmp_path / "shop")
    renamed = shop_index(tmp_path / "renamed", RENAMED)

    # Act
    value = first_lines(sent_requests(index, policy=VALUE))
    value_renamed = first_lines(sent_requests(renamed, request=RENAMED_REQUEST, policy=VALUE))
    stage = first_lines(sent_requests(index, policy=STAGE_ORDER))
    stage_renamed = first_lines(sent_requests(renamed, request=RENAMED_REQUEST, policy=STAGE_ORDER))
    ranked = _ranked_first_lines(index, REQUEST)
    ranked_renamed = _ranked_first_lines(renamed, RENAMED_REQUEST)

    # Assert
    assert value == value_renamed
    assert ranked == ranked_renamed
    assert stage != stage_renamed


def test_the_value_policy_judges_repeated_code_once_and_gives_the_copy_its_answer(tmp_path: Path) -> None:
    # Arrange: orders/copy.py's cancel_order is the same code as orders/service.py's
    index = shop_index(tmp_path)
    client = ScriptedJevClient(default_noul=0.1)

    # Act
    result = find_all(index, Judge(client, items_per_request=2), TARGETS, **REQUEST, policy=VALUE)

    # Assert
    copy, original = (
        _unit_at(result, path, "cancel_order") for path in ("orders/copy.py", "orders/service.py")
    )
    assert result.repeat_of == {copy.id: original.id}
    scores = {score.unit.id: score.probability for score in result.scores("limit")}
    assert scores[copy.id] == scores[original.id]
    assert copy.id not in result.not_judged
    sent = [item["code"] for state, _ in client.requests for item in state["items"]]
    assert sum(code.startswith("def cancel_order") for code in sent) == 1


def test_a_copy_of_code_the_search_did_not_reach_is_not_reached_too(tmp_path: Path) -> None:
    # Arrange: one call judges check_limit and place_order, never cancel_order
    index = shop_index(tmp_path)

    # Act
    result = find_all(
        index, Judge(ScriptedJevClient(), items_per_request=2, max_calls=1), TARGETS, **REQUEST, policy=VALUE
    )

    # Assert
    copy = _unit_at(result, "orders/copy.py", "cancel_order")
    assert result.not_judged[copy.id] == NOT_REACHED


def test_a_value_search_resumed_from_its_answers_asks_only_the_units_it_had_not_reached(
    tmp_path: Path,
) -> None:
    # Arrange
    index = shop_index(tmp_path)
    stopped = find_all(
        index, Judge(ScriptedJevClient(), items_per_request=2, max_calls=2), TARGETS, **REQUEST, policy=VALUE
    )
    client = ScriptedJevClient()

    # Act
    find_all(
        index, Judge(client, items_per_request=2), TARGETS, **REQUEST, policy=VALUE, completed=stopped.judged
    )

    # Assert
    asked = [item["code"].splitlines()[0] for state, _ in client.requests for item in state["items"]]
    assert sorted(asked) == sorted(
        [*PRODUCTION_WITHOUT_NAMES, "def test_check_limit_refuses_one_item_over(order_factory):"]
    )


def test_the_async_search_keeps_the_value_order(tmp_path: Path) -> None:
    # Arrange
    index = shop_index(tmp_path)
    client = ScriptedJevClient(default_noul=0.1)
    judge = Judge(AsyncScriptedJevClient(client), items_per_request=2, max_concurrency=1)

    # Act
    asyncio.run(find_all_async(index, judge, TARGETS, **REQUEST, policy=VALUE))

    # Assert
    sent = [
        [[item["file"], item["code"].splitlines()[0]] for item in state["items"]]
        for state, _ in client.requests
    ]
    assert sent == sent_requests(index, policy=VALUE)


def _ranked_first_lines(index, request) -> list[str]:
    """The ranking when every unit gets the same answer, so only the tie-break orders it."""
    result = find_all(index, Judge(ScriptedJevClient(default_noul=0.5)), TARGETS, **request, policy=VALUE)
    return [_first_line(index, score.unit) for score in result.ranked("limit")]


def _first_line(index, unit) -> str:
    return index.lines(unit.path)[unit.ranges[0][0] - 1]


def _unit_at(result, path: str, symbol: str):
    return next(unit for unit in result.units if unit.path == path and unit.symbol == symbol)
