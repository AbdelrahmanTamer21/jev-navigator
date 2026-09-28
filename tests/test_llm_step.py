from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_navigator.connectors import CommandConnector, OpenAICompatibleConnector
from jev_navigator.judgments.judge import Judge, PickResult
from jev_navigator.judgments.questions import Pick
from jev_navigator.judgments.secrets import SecretInRequestError
from jev_navigator.llm_step import JsonContract, LlmGuard, LlmStep, PickFromOptions
from jev_navigator.testing import ScriptedJevClient

FAKE_LLM = Path(__file__).parent / "fake_llm" / "reply.sh"
PHRASE = Pick("phrase", "Which phrase in `comment.text` names what `slice.code` does?")
PHRASES = {"p0": "retries twice", "p1": "logs the error"}
STATE = {
    "comment": {"text": "retries twice and logs the error"},
    "slice": {"code": "for attempt in range(2): ..."},
}


def fake_connector(tmp_path: Path, *replies: str) -> CommandConnector:
    replies_file = tmp_path / "replies.txt"
    replies_file.write_text("\n".join(replies) + "\n")
    return CommandConnector(["sh", str(FAKE_LLM), str(replies_file)], name="fake", model="fake-1")


def phrase_step(connector, guard: LlmGuard | None = None) -> LlmStep[PickResult, str]:
    """The README example: ask an LLM when the phrase Choice is below 0.70."""
    return LlmStep(
        name="phrase_fallback",
        when=lambda result: result.confidence < 0.70,
        context=lambda result: {"comment": STATE["comment"], "slice": STATE["slice"], "options": PHRASES},
        answer=PickFromOptions(
            "Which phrase of the comment names what the code does?", answer_field="phrase"
        ),
        connector=connector,
        guard=guard or LlmGuard(),
    )


def unsure_phrase_pick() -> PickResult:
    client = ScriptedJevClient(choices={"phrase": {"p0": 0.55, "p1": 0.45}})
    return Judge(client).pick(PHRASE, PHRASES, STATE)


def test_the_step_runs_only_when_the_user_predicate_says_so(tmp_path: Path) -> None:
    # Arrange
    confident = Judge(ScriptedJevClient()).pick(PHRASE, PHRASES, STATE)
    step = phrase_step(fake_connector(tmp_path, '{"phrase": "p1"}'))

    # Act
    skipped = step.run(confident)
    called = step.run(unsure_phrase_pick())

    # Assert
    assert skipped is None
    assert called.answer == "p1" and called.connector == "fake"


def test_an_unparsable_reply_is_retried_once_then_reported(tmp_path: Path) -> None:
    # Arrange
    step = phrase_step(fake_connector(tmp_path, "not json", '{"phrase": "made-up"}'))

    # Act
    call = step.run(unsure_phrase_pick())

    # Assert
    assert call.answer is None
    assert "made-up" in call.parse_error


def test_a_retry_that_parses_is_used(tmp_path: Path) -> None:
    # Act
    call = phrase_step(fake_connector(tmp_path, "I pick p0", '{"phrase": "p0"}')).run(unsure_phrase_pick())

    # Assert
    assert call.answer == "p0"


def test_the_budget_stops_further_calls(tmp_path: Path) -> None:
    # Arrange
    step = phrase_step(
        fake_connector(tmp_path, '{"phrase": "p0"}', '{"phrase": "p1"}'), LlmGuard(max_calls=1)
    )
    step.run(unsure_phrase_pick())

    # Act
    second = step.run(unsure_phrase_pick())

    # Assert
    assert second is None


def test_context_is_masked_and_a_prompt_with_a_secret_is_refused(tmp_path: Path) -> None:
    # Arrange
    prompts: list[str] = []

    class Recording:
        name, model = "recording", "r"

        def complete(self, prompt: str) -> str:
            prompts.append(prompt)
            return '{"phrase": "p0"}'

    token = f"ghp_{'e5' * 18}"
    leaky = LlmStep(
        "leaky",
        lambda result: True,
        lambda result: {"code": f'TOKEN = "{token}"', "options": PHRASES},
        PickFromOptions("Pick", answer_field="phrase"),
        Recording(),
    )
    unmasked = LlmStep(
        "unmasked",
        lambda result: True,
        lambda result: {"code": f'TOKEN = "{token}"', "options": PHRASES},
        PickFromOptions("Pick", answer_field="phrase"),
        Recording(),
        LlmGuard(masker=None),
    )

    # Act
    leaky.run(None)

    # Assert
    assert token not in prompts[0]
    with pytest.raises(SecretInRequestError):
        unmasked.run(None)
    assert len(prompts) == 1


def test_every_call_is_stored_with_prompt_hash_reply_and_answer(tmp_path: Path) -> None:
    # Arrange
    store = tmp_path / "llm.jsonl"
    step = phrase_step(fake_connector(tmp_path, '{"phrase": "p1"}'), LlmGuard(store_path=store))

    # Act
    step.run(unsure_phrase_pick())

    # Assert
    line = json.loads(store.read_text())
    assert (line["kind"], line["step"], line["answer"]) == ("llm_step", "phrase_fallback", "p1")
    assert line["prompt_sha256"] and '"phrase": "p1"' in line["reply"]


def test_json_contract_checks_required_fields() -> None:
    # Arrange
    contract = JsonContract("Say whether the comment is accurate.", {"accurate": bool, "line": int})

    # Act and Assert
    assert contract.parse('{"accurate": true, "line": 4}', {}) == {"accurate": True, "line": 4}
    with pytest.raises(ValueError, match="line"):
        contract.parse('{"accurate": true}', {})


def test_openai_compatible_connector_posts_the_prompt() -> None:
    # Arrange
    posted: list[tuple[str, dict]] = []

    def post(url: str, body: dict, headers: dict, timeout: float) -> dict:
        posted.append((url, body))
        return {"choices": [{"message": {"content": "hello"}}]}

    # Act
    reply = OpenAICompatibleConnector("http://localhost:8000/v1", "glm", post=post).complete("prompt")

    # Assert
    assert reply == "hello"
    assert posted[0][0] == "http://localhost:8000/v1/chat/completions" and posted[0][1]["model"] == "glm"
