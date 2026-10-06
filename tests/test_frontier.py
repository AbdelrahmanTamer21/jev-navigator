"""The frontier's two policies over one search: today's stage order, and value order by code features."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from conftest import RefusingClient, labelled
from shop_search import FROZEN, REQUEST, SHOP, TARGETS, sent_requests, shop_index

from jev_navigator.directives.find_all import NOT_REACHED, REFUSED, find_all, find_all_async
from jev_navigator.directives.frontier import (
    STAGE_ORDER,
    VALUE,
    Features,
    Weights,
    features_of,
    name_rarities,
)
from jev_navigator.directives.search_coverage import Outcome, Round, point_results
from jev_navigator.index.units import list_units, read_ranges
from jev_navigator.judgments.client import JEV_INPUT_LIMITS
from jev_navigator.judgments.judge import Judge
from jev_navigator.sources import CALLEES, CALLERS, NAMES
from jev_navigator.testing import AsyncScriptedJevClient, ScriptedJevClient

PRODUCTION_WITHOUT_NAMES = [
    "export function submitOrder(order) {",
    "export function cancelOrder(id) {",
    "def invoice(order):",
    "def refund(order):",
]
JEV_BOX = JEV_INPUT_LIMITS.box_chars
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
    # limit constant holds one in a file the anchor's file imports; round_total and cancel_order hold
    # none but sit in files the anchor's file is or imports; submitOrder holds MAX_ITEMS_PER_PAGE, which
    # is not the name; the test holds both names but is a test
    index = shop_index(tmp_path)

    # Act
    judged = first_lines(sent_requests(index, policy=VALUE))

    # Assert
    assert judged[:3] == ["def check_limit(order):", "def place_order(order):", "MAX_ITEMS = 4"]
    assert sorted(judged[3:5]) == ["def cancel_order(order):", "def round_total(order):"]
    assert sorted(judged[5:9]) == sorted(PRODUCTION_WITHOUT_NAMES)
    assert judged[9:] == ["def test_check_limit_refuses_one_item_over(order_factory):"]


def test_a_units_features_count_whole_word_names_and_its_own_and_its_files_name(tmp_path: Path) -> None:
    # Arrange: refund calls invoice and sits in invoice.py; refunds is a longer word than refund
    index = shop_index(tmp_path)
    units = list_units(index, ["billing/invoice.py"], box_chars=JEV_BOX).units
    [refund] = [unit for unit in units if unit.symbol == "refund"]
    rarities = name_rarities({"invoice": 2, "refunds": 6})

    # Act
    features = features_of(refund, read_ranges(index, refund.path, refund.ranges), 2, rarities)

    # Assert
    assert features == Features(
        names=("invoice",), rarity=rarities["invoice"], defines=False, file_named=True, distance=2, test=False
    )


def test_by_default_a_unit_defining_a_name_outranks_one_holding_two_names() -> None:
    # Arrange: the definer holds only its own name; the other holds both names, each as rare as can be
    rarest = name_rarities({"check_limit": 0})["check_limit"]
    definer = Features(("check_limit",), rarest, defines=True, file_named=False, distance=2, test=False)
    mentioner = Features(
        ("check_limit", "MAX_ITEMS"), 2 * rarest, defines=False, file_named=False, distance=2, test=False
    )

    # Act / Assert
    assert definer.value(Weights()) > mentioner.value(Weights())


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

    # Assert: the two calls answered the first four units in value order
    asked = [item["code"].splitlines()[0] for state, _ in client.requests for item in state["items"]]
    assert asked == first_lines(sent_requests(index, policy=VALUE))[4:]


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


CHAIN = {
    "orders/checkout.py": (
        "from orders.rules import enforce\n\n\ndef checkout(order):\n    enforce(order)\n    return order\n"
    ),
    "orders/rules.py": (
        "from orders.limits import over_limit\n\n\n"
        "def enforce(order):\n    if over_limit(order):\n        raise ValueError('too many items')\n"
    ),
    "orders/limits.py": "def over_limit(order):\n    return len(order.items) > 4\n",
    "web/routes.py": (
        "from orders.checkout import checkout\n\n\n"
        "def post_order(request):\n    return checkout(request.order)\n"
    ),
    "billing/refund.py": "def refund(order):\n    return -order.total\n",
}
LIMIT_ONLY = {"limit": TARGETS["limit"]}
LIMIT_ANSWERS = {
    ("limit", "enforce(order)\n    return"): 0.85,
    ("limit", "raise ValueError"): 0.93,
    ("limit", "> 4"): 0.9,
}


def settling_search(root: Path, client=None, targets=LIMIT_ONLY, per_request=1, **settings):
    """One unit per request by default, one request at a time: checkout clears the limit at 0.85, the
    callee it pushes, enforce, scores 0.93, and enforce's own callee, over_limit, would score 0.9."""
    index = shop_index(root, CHAIN)
    judge = Judge(client or labelled(LIMIT_ANSWERS), items_per_request=per_request, max_concurrency=1)
    request = {"files": ["orders/checkout.py", "billing/refund.py"], "names": ["checkout"]}
    return index, find_all(index, judge, targets, **request, batches_per_wave=1, policy=VALUE, **settings)


def test_a_target_settles_only_after_the_callers_and_callees_its_clearing_unit_pushed_are_judged(
    tmp_path: Path,
) -> None:
    # Act
    index, result = settling_search(tmp_path)

    # Assert: the callee pushed one step deep is the best answer; its own callee is never listed.
    checkout, post_order, enforce, refund = (
        _unit_at(result, path, symbol)
        for path, symbol in [
            ("orders/checkout.py", "checkout"),
            ("web/routes.py", "post_order"),
            ("orders/rules.py", "enforce"),
            ("billing/refund.py", "refund"),
        ]
    )
    judged = [_first_line(index, score.unit) for score in result.scores("limit")]
    assert sorted(judged) == sorted(
        ["def checkout(order):", "def post_order(request):", "def enforce(order):"]
    )
    assert result.ranked("limit")[0].unit == enforce
    assert result.pushed == {checkout.id: (post_order.id, enforce.id)}
    assert (result.entered_by[post_order.id], result.entered_by[enforce.id]) == (NAMES.name, CALLEES.name)
    assert "orders/limits.py" not in {unit.path for unit in result.units}
    assert (result.stopped_by, result.settled) == ("settled", ("limit",))
    assert result.not_judged == {refund.id: NOT_REACHED}


def test_the_hop_sources_a_caller_composes_decide_what_a_clearing_unit_pushes(tmp_path: Path) -> None:
    # Act: callees only, so post_order, checkout's caller, is never pushed
    _, result = settling_search(tmp_path, hops=(CALLEES,))

    # Assert
    checkout, post_order, enforce = (
        _unit_at(result, path, symbol)
        for path, symbol in [
            ("orders/checkout.py", "checkout"),
            ("web/routes.py", "post_order"),
            ("orders/rules.py", "enforce"),
        ]
    )
    assert result.pushed == {checkout.id: (enforce.id,)}
    assert result.not_judged[post_order.id] == NOT_REACHED
    assert result.ranked("limit")[0].unit == enforce
    assert (result.stopped_by, result.settled) == ("settled", ("limit",))


def test_a_settling_target_spends_its_slots_only_on_the_units_it_pushed(tmp_path: Path) -> None:
    # Arrange: two units per request, so the request after checkout cleared has a slot to spare
    client = labelled(LIMIT_ANSWERS)

    # Act
    index, result = settling_search(tmp_path, client, per_request=2)

    # Assert: post_order was judged with checkout, so the next request holds only enforce, and refund,
    # which a slot to spare could have held, is never judged
    requests = [[item["code"].splitlines()[0] for item in state["items"]] for state, _ in client.requests]
    assert requests == [["def checkout(order):", "def post_order(request):"], ["def enforce(order):"]]
    refund = _unit_at(result, "billing/refund.py", "refund")
    assert (result.not_judged, result.stopped_by) == ({refund.id: NOT_REACHED}, "settled")


def test_a_refused_hop_does_not_hold_its_target_open(tmp_path: Path) -> None:
    # Arrange: the provider refuses the request holding enforce
    client = RefusingClient(labelled(LIMIT_ANSWERS), "raise ValueError", list_name="items")

    # Act
    _, result = settling_search(tmp_path, client)

    # Assert
    enforce = _unit_at(result, "orders/rules.py", "enforce")
    assert result.not_judged[enforce.id] == REFUSED
    assert (result.stopped_by, result.settled) == ("settled", ("limit",))


def test_the_async_search_settles_as_the_sync_search_does(tmp_path: Path) -> None:
    # Arrange
    index, sync = settling_search(tmp_path)
    judge = Judge(AsyncScriptedJevClient(labelled(LIMIT_ANSWERS)), items_per_request=1, max_concurrency=1)
    request = {"files": ["orders/checkout.py", "billing/refund.py"], "names": ["checkout"]}

    # Act
    concurrent = asyncio.run(
        find_all_async(index, judge, LIMIT_ONLY, **request, batches_per_wave=1, policy=VALUE)
    )

    # Assert
    assert concurrent.judged == sync.judged
    assert (concurrent.pushed, concurrent.settled, concurrent.stopped_by) == (
        sync.pushed,
        sync.settled,
        sync.stopped_by,
    )


def test_a_coverage_record_counts_the_callers_and_callees_a_spent_budget_left_unjudged(
    tmp_path: Path,
) -> None:
    # Arrange: both population units go in the first request; checkout clears the limit, and the call
    # cap stops the request that would judge its caller and its callee
    index = shop_index(tmp_path, CHAIN)
    judge = Judge(labelled(LIMIT_ANSWERS), items_per_request=2, max_concurrency=1, max_calls=1)

    # Act
    result = find_all(
        index,
        judge,
        TARGETS,
        files=["orders/checkout.py", "billing/refund.py"],
        batches_per_wave=1,
        policy=VALUE,
    )
    [limit, refund] = point_results([Round(result)], 0.8)

    # Assert
    assert result.stopped_by == "budget"
    assert limit.outcome is Outcome.FOUND
    assert refund.coverage.cut == {(CALLERS.name, NOT_REACHED): 1, (CALLEES.name, NOT_REACHED): 1}
    assert "2 not reached (1 callers, 1 callees)" in refund.render(0.8)


SPLIT = {
    "pkg/alpha.py": "".join(f"def alpha_{n}():\n    return alpha({n})\n\n\n" for n in range(4)),
    "pkg/beta.py": "".join(f"def beta_{n}():\n    return beta({n})\n\n\n" for n in range(4)),
}
SPLIT_TARGETS = {"a": "the code that calls `alpha`", "b": "the code that calls `beta`"}


def split_search(root: Path, client, **settings):
    """A search over four units holding each name, and each of its requests' units, as alpha or beta."""
    index = shop_index(root, SPLIT)
    judge = Judge(client, items_per_request=settings.pop("per_request", 2), max_concurrency=1)
    result = find_all(
        index, judge, SPLIT_TARGETS, names=["alpha", "beta"], batches_per_wave=1, policy=VALUE, **settings
    )
    return result, [
        [item["file"].removeprefix("pkg/").removesuffix(".py") for item in state["items"]]
        for state, _ in client.requests
    ]


def test_each_target_draws_its_equal_share_of_every_batch_from_its_own_queue(tmp_path: Path) -> None:
    # Act
    _, requests = split_search(tmp_path, ScriptedJevClient(default_noul=0.1))

    # Assert
    assert requests == [["alpha", "beta"]] * 4


def test_a_caller_sets_the_shares(tmp_path: Path) -> None:
    # Act: three slots of four for a, one for b
    _, requests = split_search(tmp_path, ScriptedJevClient(default_noul=0.1), per_request=4, shares={"a": 3})

    # Assert
    assert sorted(requests[0]) == ["alpha", "alpha", "alpha", "beta"]


def test_a_settled_targets_share_flows_to_the_target_still_open(tmp_path: Path) -> None:
    # Arrange: every unit holding alpha clears a in the first request; nothing ever clears b
    client = labelled({("a", "alpha("): 0.9})

    # Act
    result, requests = split_search(tmp_path, client)

    # Assert: after the first request every slot is b's, whose queue holds beta first
    assert requests[0] == ["alpha", "beta"]
    assert [unit for request in requests[1:] for unit in request] == ["beta"] * 3 + ["alpha"] * 3
    assert (result.settled, result.stopped_by) == (("a",), "scope_examined")


@pytest.mark.parametrize(
    ("shares", "policy", "error"),
    [
        ({"c": 1.0}, VALUE, "targets the search does not have"),
        ({"a": 0.0}, VALUE, "positive number"),
        ({"a": 2.0}, STAGE_ORDER, "ranked policy"),
    ],
)
def test_shares_name_targets_of_a_ranked_search_and_are_positive(
    tmp_path: Path, shares: dict, policy, error: str
) -> None:
    # Arrange
    index = shop_index(tmp_path, SPLIT)

    # Act and Assert
    with pytest.raises(ValueError, match=error):
        find_all(
            index, Judge(ScriptedJevClient()), SPLIT_TARGETS, names=["alpha"], policy=policy, shares=shares
        )
