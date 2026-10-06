"""Refusal markers select code, independently of repository metadata."""

import pytest
from conftest import REFUSAL, RefusingClient

from jev_navigator.judgments.client import InputBudgetExceededError
from jev_navigator.testing import ScriptedJevClient


@pytest.mark.parametrize("list_name", ["trace", "items"])
@pytest.mark.parametrize("status", [201, 422])
def test_refusal_marker_matches_only_item_code(list_name: str, status: int) -> None:
    scripted = ScriptedJevClient(default_noul=0.2)
    client = RefusingClient(scripted, marker="422", list_name=list_name)
    state = {
        list_name: [
            {"code": "def audit(value):\n    return value\n", "commit": "a422b"},
            {"code": f"return {{'status': {status}}}", "commit": "a422b"},
        ]
    }
    questions = {"relevant": {"type": "noul", "question": "Is this relevant?"}}

    if status == 422:
        with pytest.raises(InputBudgetExceededError, match=REFUSAL):
            client.ask(state, questions)
        assert scripted.requests == []
    else:
        response = client.ask(state, questions)
        assert response.answers["relevant"].probability == 0.2
        assert scripted.requests == [(state, questions)]
