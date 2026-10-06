"""Execute the proposed reducer against independent delegation and local-role answers."""

import json
from pathlib import Path

import pytest

from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.thresholds import Thresholds
from jev_navigator.testing import ScriptedJevClient


@pytest.mark.parametrize("local_role", [None, "decide", "guard", "value", "effect"])
def test_delegation_routes_only_when_every_local_role_is_low(local_role: str | None) -> None:
    proposal = Path(__file__).parents[1] / "question-templates/evidence_roles/PROPOSAL.md"
    code = proposal.read_text().split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {"__name__": __name__}
    exec(compile(code, str(proposal), "exec"), namespace)
    probabilities = dict.fromkeys(("decide", "guard", "value", "effect", "satisfied"), 0.1)
    probabilities["delegates"] = 0.9
    if local_role is not None:
        probabilities[local_role] = 0.9
    candidate = json.loads((proposal.parent / "candidate.json").read_text())["request"]
    client = ScriptedJevClient(nouls=probabilities)
    response = Judge(client).ask(candidate["state"], candidate["questions"], thresholds=Thresholds())
    raw = {name: response.noul(name).probability for name in candidate["questions"]}
    answer = namespace["EvidenceAnswers"]("wrapper", raw)
    result = namespace["compose"]([answer], [local_role] if local_role else [])
    assert result["follow_units"] == ([] if local_role else ["wrapper"])
    assert result["uncovered_roles"] == set()
    assert result["ranked"] == [answer]
