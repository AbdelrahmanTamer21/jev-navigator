from __future__ import annotations

from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.directives.find_code import Outcome, find_code
from jev_navigator.directives.places import neighbours_and_omissions, place_for_line
from jev_navigator.index import tools
from jev_navigator.index.bindings import Binding
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.judge import Judge
from jev_navigator.testing import ScriptedJevClient


@pytest.fixture
def ast_grep_runs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    runs: list[str] = []
    original_rules = tools.ast_grep_rules

    def counted_rules(rules: str, files, cwd):
        runs.append(rules.split("\n", 1)[0])
        return original_rules(rules, files, cwd)

    monkeypatch.setattr(tools, "ast_grep_rules", counted_rules)
    return runs


class CountingResolver:
    """Answers nothing, so the index decides; counts how often the index asks."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, int, str]] = []

    def resolve_call(self, file: str, line: int, name: str, receiver: str | None) -> Binding | None:
        self.asked.append((file, line, name))
        return None


def committed(root: Path, files: dict[str, str]) -> CodeIndex:
    commit_files(root, files)
    return CodeIndex.from_git(root)


FLOW_ADAPTER = """\
// @flow
const toPostgresValue = (value: any, options: ?QueryOptions): any => value;

export class PostgresAdapter {
  constructor({ uri }: { uri: string }) {
    this._client = connect(uri);
  }

  async createObject(className: string, object: Object, options: ?QueryOptions): Promise<void> {
    await this._client.none('INSERT INTO $1:name', [className, toPostgresValue(object, options)]);
  }

  find(className: string, query: Object): Promise<Array<Object>> {
    return this._client.any('SELECT * FROM $1:name', [className, query]);
  }
}

function connect(uri: string): any {
  return { uri };
}
"""

MEMORY_ADAPTER = """\
export class MemoryAdapter {
  constructor() {
    this.rows = [];
  }

  createObject(className, object) {
    this.rows.push({ className, object });
  }
}
"""

ADAPTER_CALLER = """\
import { PostgresAdapter } from './postgres';

export function store(uri) {
  const adapter = new PostgresAdapter({ uri });
  return adapter.find('users', {});
}
"""


def an_adapter_scope(tmp_path: Path) -> CodeIndex:
    """A Flow-typed file the JavaScript grammar only partly recovers, a valid sibling, and a caller."""
    return committed(
        tmp_path,
        {
            "src/adapters/postgres.js": FLOW_ADAPTER,
            "src/adapters/memory.js": MEMORY_ADAPTER,
            "src/adapters/index.js": ADAPTER_CALLER,
        },
    )


def test_each_file_is_parsed_once_and_a_new_index_reuses_its_facts(
    sample_index: CodeIndex, ast_grep_runs
) -> None:
    # Arrange
    opened = [
        sample_index.read_slice(span) for file in sample_index.files for span in sample_index.symbols_in(file)
    ]

    # Act
    for code in opened:
        neighbours_and_omissions(sample_index, code)
    for name in ("validate_order", "check_limits", "handleOrder"):
        sample_index.find_callers(name)
        sample_index.find_references(name)

    cold_runs = len(ast_grep_runs)
    warm = CodeIndex.from_git(sample_index.root, fact_cache_dir=sample_index.root.parent / "fact-cache")
    for file in warm.files:
        warm.symbols_in(file)

    # Assert
    assert len(opened) >= 6
    assert cold_runs == len(sample_index._code_files)
    assert len(ast_grep_runs) == cold_runs


def test_each_call_site_is_bound_once_however_often_it_is_looked_up(sample_repo: Path) -> None:
    # Arrange
    resolver = CountingResolver()
    index = CodeIndex.from_git(sample_repo, binding_resolver=resolver)

    # Act
    first = index.find_callers("check_limits")
    index.find_callers("check_limits")
    index.callee_edges(index.find_definition("validate_order")[0])

    # Assert
    assert len(first) == 1
    assert resolver.asked.count(("app/validation.py", 7, "check_limits")) == 1
    assert len(resolver.asked) == len(set(resolver.asked))


def test_an_external_parser_failure_is_not_relabelled_as_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "module.py").write_text("def run():\n    return 1\n")

    def fail_parser(rules, files, cwd):
        raise tools.ToolFailedError("ast-grep failed for a real tool reason")

    monkeypatch.setattr(tools, "ast_grep_rules", fail_parser)
    index = CodeIndex(tmp_path, ["module.py"], fact_cache_dir=tmp_path / "cache")

    with pytest.raises(tools.ToolFailedError, match="real tool reason"):
        index.functions_in("module.py")


def test_a_plain_call_wins_over_a_method_call_of_the_same_name_on_one_line(tmp_path: Path) -> None:
    # Arrange
    index = committed(
        tmp_path,
        {"app/orders.py": "def save(order):\n    return helpers.save(order), save(order)\n"},
    )

    # Act
    site = index.find_callers("save")[0]

    # Assert
    assert site.binding.status == "resolved"


def test_a_file_the_grammar_only_partly_recovers_counts_as_unparsed(tmp_path: Path) -> None:
    """parse-server P1: `options: ?QueryOptions` on a method of an exported class makes the JavaScript
    grammar report an ERROR in the class. The file must not count as completely indexed, while
    valid module functions and the valid sibling stay usable. Recovery inside the malformed class
    may vary across parser versions."""
    # Arrange
    index = an_adapter_scope(tmp_path)

    # Act
    methods = {span.name for span in index.functions_in("src/adapters/postgres.js")}

    # Assert
    assert index.unparsed_files == {"src/adapters/postgres.js"}
    assert {"toPostgresValue", "connect"} <= methods
    assert {"constructor", "createObject"} <= {
        span.name for span in index.functions_in("src/adapters/memory.js")
    }


def test_a_name_defined_only_where_the_grammar_errored_is_unknown_not_unresolved(tmp_path: Path) -> None:
    # Arrange
    index = an_adapter_scope(tmp_path)

    # Act
    site = index.find_callers("find")[0]

    # Assert
    assert site.binding.status == "unknown"
    assert "src/adapters/postgres.js" in site.binding.reason


def test_a_search_over_a_scope_with_grammar_errors_never_reports_nothing_left(tmp_path: Path) -> None:
    # Arrange
    index = an_adapter_scope(tmp_path)
    judge = Judge(ScriptedJevClient(nouls=lambda question_id, question, state: 0.05))
    start = [place_for_line(index, "src/adapters/index.js", 4, "start")]

    # Act
    result = find_code(index, judge, "where users are found", start)

    # Assert
    assert result.outcome == Outcome.SCOPE_INCOMPLETE
    assert result.unparsed_files == {"src/adapters/postgres.js"}
    assert result.history.steps[-1].judgments["unparsed_files"] == ["src/adapters/postgres.js"]
