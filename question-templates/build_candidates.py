"""Writes one review file per shipped question set, using this repository's own code."""

from __future__ import annotations

from pathlib import Path

from jev_navigator.directives.find_code import SearchBudget, find_code
from jev_navigator.directives.places import place_for_line
from jev_navigator.directives.similar import SAME_BEHAVIOUR
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.review import CapturingJevClient, export_for_review

HERE = Path(__file__).parent
REVISION = "v3"
SEARCH_USES = {
    "contains_target": (
        "Yes (0.80 or more) ends the search and reports this code as found; unsure keeps it as an"
        " unsure place; no records it as searched without finding the target."
    ),
    "could_contain_target": "The probability orders this neighbour in the best-first queue. A low value only"
    " lowers its priority; the neighbour is never discarded and stays in the result as not inspected.",
    "open_first": (
        "When confidence is 0.70 or more and the answer is not 'none', that neighbour is opened first."
    ),
}


def main() -> None:
    index = CodeIndex.from_git(HERE.parent, prefixes=("src/",))
    write_search_candidate(index)
    subject = index.read_slice(index.find_definition("mask_value")[0]).text
    write_item_candidate(
        index,
        SAME_BEHAVIOUR,
        {"subject": {"code": subject}},
        "Yes lists the candidate as duplicated behaviour; unsure is listed separately; no is"
        " dropped from the duplication candidates.",
        "same_behaviour",
    )


def write_search_candidate(index: CodeIndex) -> None:
    capture = CapturingJevClient()
    start = [place_for_line(index, "src/jev_navigator/judgments/judge.py", 60, "start")]
    find_code(
        index,
        Judge(capture),
        "the check that refuses to send a request that still contains a secret",
        start,
        budget=SearchBudget(max_steps=1, beam_width=1),
    )
    state, questions = capture.requests[0]
    uses = {question_id: SEARCH_USES[question_id.split("@")[0]] for question_id in questions}
    export_for_review(
        state,
        questions,
        uses,
        HERE / "find_code" / REVISION / "candidate.json",
        case_id="find_code:secret-refusal",
        group_id="jev-navigator-find-code",
        revision_id=REVISION,
    )


def write_item_candidate(index: CodeIndex, check, shared: dict, use: str, name: str) -> None:
    capture = CapturingJevClient()
    spans = index.functions_in("src/jev_navigator/judgments/secrets.py")[:3]
    items = [
        {"file": span.file, "lines": [span.start, span.end], "code": index.read_slice(span).text}
        for span in spans
    ]
    Judge(capture).check_each(check, items, shared)
    state, questions = capture.requests[0]
    export_for_review(
        state,
        questions,
        {question_id: use for question_id in questions},
        HERE / name / REVISION / "candidate.json",
        case_id=f"{name}:secrets-module",
        group_id=f"jev-navigator-{name}",
        revision_id=REVISION,
    )


if __name__ == "__main__":
    main()
