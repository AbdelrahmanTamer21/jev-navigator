"""Jev's generated-file judgment on real git repositories: what each flagged file's entry holds, who
imports it, and how answers and refusals map back to files. Answers come from the scripted client;
everything else is the real scope, index, masker and judge."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from git_repos import commit_all, write_files

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.file_shape import shape_of
from jev_navigator.index.scope import ResolvedScope, Scope, resolve_scope
from jev_navigator.judgments.generated_files import (
    EXCERPT_CHARS,
    GENERATED_FILE,
    MAX_IMPORTERS,
    generated_file_entry,
    importers_of,
    judge_generated_files,
)
from jev_navigator.judgments.judge import Judge
from jev_navigator.testing import ScriptedJevClient

BUNDLE = "".join(f"var a{number}=function(){{return {number}}};" for number in range(800)) + "\n"
SECRET_MARK = "LOOKS-LIKE-A-KEY"


def _repository(root: Path, files: dict[str, str]) -> Path:
    root.mkdir()
    write_files(root, files)
    commit_all(root)
    return root


def _awaiting(repo: Path) -> tuple[CodeIndex, Mapping]:
    resolved = resolve_scope(
        Scope(
            repo=repo,
            with_tests=False,
            with_generated=False,
            with_vendored=False,
            with_docs=False,
            max_files=200,
        )
    )
    assert isinstance(resolved, ResolvedScope)
    return CodeIndex(repo, resolved.files), resolved.awaiting_generated_judgment


def test_a_flagged_file_reaches_jev_as_its_path_measured_facts_and_two_excerpts(tmp_path: Path) -> None:
    text = BUNDLE.removesuffix("\n")
    repo = _repository(tmp_path / "repo", {"web/bundle.js": text})
    index, awaiting = _awaiting(repo)
    shape = shape_of(repo, "web/bundle.js")
    middle_start = (len(text) - EXCERPT_CHARS) // 2

    entry = generated_file_entry(index, "web/bundle.js", awaiting["web/bundle.js"])

    assert entry == {
        "file": "web/bundle.js",
        "size_bytes": shape.size_bytes,
        "line_count": shape.line_count,
        "longest_line_bytes": shape.longest_line,
        "average_line_bytes": round(shape.chars_per_line, 1),
        "importers": [],
        "importer_count": 0,
        "opening": text[:EXCERPT_CHARS],
        "middle": text[middle_start : middle_start + EXCERPT_CHARS],
    }


def test_importers_are_the_files_whose_imports_resolve_to_it_capped_with_the_true_count(
    tmp_path: Path,
) -> None:
    importers = {f"web/use{number:02}.ts": 'import { a0 } from "./bundle";\n' for number in range(12)}
    mention_only = {"web/notes.ts": "// the bundle is rebuilt nightly\nexport const n = 1;\n"}
    repo = _repository(tmp_path / "repo", {"web/bundle.js": BUNDLE, **importers, **mention_only})
    index, _ = _awaiting(repo)

    entry = generated_file_entry(index, "web/bundle.js", shape_of(repo, "web/bundle.js"))

    assert entry["importers"] == sorted(importers)[:MAX_IMPORTERS]
    assert entry["importer_count"] == 12


def test_finding_importers_runs_no_fact_scan(tmp_path: Path) -> None:
    repo = _repository(tmp_path / "repo", {"web/bundle.js": BUNDLE, "web/use.ts": 'import "./bundle";\n'})
    scans: list[tuple[str, str, int]] = []
    fact_cache = tmp_path / "facts"
    index = CodeIndex(
        repo,
        ["web/bundle.js", "web/use.ts"],
        scan_observer=lambda *scan: scans.append(scan),
        fact_cache_dir=fact_cache,
    )

    found = importers_of(index, "web/bundle.js")

    assert found == ("web/use.ts",)
    assert scans == []
    assert not fact_cache.exists() or not any(fact_cache.rglob("*"))


def test_each_flagged_file_gets_one_question_naming_it_and_its_own_answer(tmp_path: Path) -> None:
    repo = _repository(tmp_path / "repo", {"web/a.js": BUNDLE, "web/b.js": "/*b*/" + BUNDLE})
    index, awaiting = _awaiting(repo)
    by_file = {"web/a.js": 0.93, "web/b.js": 0.12}
    client = ScriptedJevClient(nouls=lambda question_id, question, state: _scripted(question, state, by_file))

    judgments = judge_generated_files(Judge(client), index, awaiting)

    assert {path: result.probability for path, result in judgments.judged.items()} == by_file
    assert dict(judgments.not_judged) == {}
    [(state, questions)] = client.requests
    assert list(state) == ["files"]
    assert sorted(question["instructions"] for question in questions.values()) == [
        GENERATED_FILE.to_question(f"files[{slot}]")["instructions"] for slot in range(2)
    ]


def test_a_file_the_secret_scanner_refuses_stays_named_as_not_judged(tmp_path: Path) -> None:
    repo = _repository(
        tmp_path / "repo", {"web/a.js": BUNDLE, "web/keyed.js": f"/* {SECRET_MARK} */" + BUNDLE}
    )
    index, awaiting = _awaiting(repo)
    client = ScriptedJevClient(default_noul=0.9)

    judgments = judge_generated_files(Judge(client, scanner=_MarkScanner()), index, awaiting)

    assert list(judgments.judged) == ["web/a.js"]
    assert dict(judgments.not_judged) == {"web/keyed.js": "not judged: the secret scan refused its excerpts"}
    assert all(SECRET_MARK not in str(state) for state, _ in client.requests)


def test_nothing_is_sent_when_no_file_awaits_a_judgment(tmp_path: Path) -> None:
    repo = _repository(tmp_path / "repo", {"app/orders.py": "def run():\n    return 1\n"})
    index, awaiting = _awaiting(repo)
    client = ScriptedJevClient()

    judgments = judge_generated_files(Judge(client), index, awaiting)

    assert (dict(judgments.judged), dict(judgments.not_judged), client.requests) == ({}, {}, [])


class _MarkScanner:
    """A host's stronger scanner, which the judge accepts by design: it finds one marked string."""

    def findings(self, text: str) -> list[str]:
        return [SECRET_MARK] if SECRET_MARK in text else []


def _scripted(question: Mapping, state: Mapping, by_file: Mapping[str, float]) -> float:
    slot = int(question["instructions"].split("`files[")[1].split("]")[0])
    return by_file[state["files"][slot]["file"]]
