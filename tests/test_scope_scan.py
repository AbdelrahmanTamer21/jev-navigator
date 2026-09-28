from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.directives.find_code import Outcome, find_code
from jev_navigator.directives.places import neighbours_and_omissions, place_for_line
from jev_navigator.index import scope_scan, tools
from jev_navigator.index.bindings import Binding
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.languages import has_flow_pragma
from jev_navigator.judgments.judge import Judge
from jev_navigator.testing import ScriptedJevClient


@pytest.fixture
def ast_grep_runs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str | None, list[str]]]:
    """One entry per ast-grep invocation: its first rule id, the sgconfig passed (None for a plain
    scan), and the files scanned."""
    runs: list[tuple[str, str | None, list[str]]] = []
    original_rules = tools.ast_grep_rules

    def counted_rules(rules: str, files, cwd, config=None):
        runs.append((rules.split("\n", 1)[0], config, list(files)))
        return original_rules(rules, files, cwd, config=config)

    monkeypatch.setattr(tools, "ast_grep_rules", counted_rules)
    return runs


@pytest.fixture
def validation_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    original_rules = tools.ast_grep_rules

    def slow_on_validation(rules: str, files, cwd, config=None):
        if "app/validation.py" in files:
            raise subprocess.TimeoutExpired(["ast-grep"], tools.COMMAND_TIMEOUT_SECONDS)
        return original_rules(rules, files, cwd)

    monkeypatch.setattr(scope_scan, "BATCH_FILES", 1)
    monkeypatch.setattr(tools, "ast_grep_rules", slow_on_validation)


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


# The evidence bundle's flow_adapter.js: a real Flow-typed storage adapter (the eval fixture).
FLOW_ADAPTER = """\
// @flow
import type { StorageAdapter, QueryOptions } from './StorageAdapter';

const toPostgresValue = (value: any, options: ?QueryOptions): any => {
  const fields: Array<string>[] = [];
  return value;
};

export class PostgresAdapter implements StorageAdapter {
  _client: any;

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

# The evidence bundle's RootTag.js (react-native v0.71.3): its `export opaque type` is a Flow
# construct no shipped grammar recovers, so the file must stay honestly incomplete.
ROOT_TAG = """\
/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @flow strict
 * @format
 */

import * as React from 'react';

export opaque type RootTag = number;

export const RootTagContext: React$Context<RootTag> =
  React.createContext<RootTag>(0);

if (__DEV__) {
  RootTagContext.displayName = 'RootTagContext';
}

/**
 * Intended to only be used by `AppContainer`.
 */
export function createRootTag(rootTag: number | RootTag): RootTag {
  return rootTag;
}
"""

ROOT_TAG_CALLER = """\
import { createRootTag } from './RootTag';

export function show(rootTag) {
  return createRootTag(rootTag);
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

# The evidence bundle's auth.js: a plain-JS control with calls and a plain function.
AUTH_MIDDLEWARE = """\
export function authenticate(req, res, next) {
  if (!req.headers.authorization) {
    return res.status(401).json({ error: 'unauthorized' });
  }
  return next();
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
    """A Flow-typed file the tsx grammar recovers, plain-JS siblings, and a caller."""
    return committed(
        tmp_path,
        {
            "src/adapters/postgres.js": FLOW_ADAPTER,
            "src/adapters/memory.js": MEMORY_ADAPTER,
            "src/adapters/index.js": ADAPTER_CALLER,
        },
    )


def a_root_tag_scope(tmp_path: Path) -> CodeIndex:
    """A real Flow file whose opaque type no shipped grammar recovers, and its caller."""
    return committed(
        tmp_path,
        {"src/native/RootTag.js": ROOT_TAG, "src/native/show.js": ROOT_TAG_CALLER},
    )


def test_the_scope_is_parsed_once_however_many_lookups_follow(sample_index: CodeIndex, ast_grep_runs) -> None:
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

    # Assert
    assert len(opened) >= 6
    assert len(ast_grep_runs) == 3


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


def test_a_file_that_timed_out_makes_its_names_unknown_not_absent(
    sample_index: CodeIndex, validation_times_out: None
) -> None:
    # Act
    site = sample_index.find_callers("validate_order")[0]

    # Assert
    assert sample_index.unparsed_files == {"app/validation.py"}
    assert sample_index.find_definition("check_limits") == ()
    assert site.binding.status == "unknown"
    assert "app/validation.py" in site.binding.reason


def test_the_unparsed_list_is_complete_before_any_lookup(
    sample_index: CodeIndex, validation_times_out: None
) -> None:
    # Act
    unparsed = sample_index.unparsed_files

    # Assert
    assert unparsed == {"app/validation.py"}


def test_a_search_over_a_scope_with_unparsed_files_never_reports_nothing_left(
    sample_index: CodeIndex, validation_times_out: None
) -> None:
    # Arrange
    judge = Judge(ScriptedJevClient(nouls=lambda question_id, question, state: 0.05))
    start = [place_for_line(sample_index, "app/orders.py", 6, "start")]

    # Act
    result = find_code(sample_index, judge, "the item limit check", start)

    # Assert
    assert result.outcome == Outcome.SCOPE_INCOMPLETE
    assert result.unparsed_files == {"app/validation.py"}
    assert result.history.steps[-1].judgments["unparsed_files"] == ["app/validation.py"]


def test_the_flow_pragma_is_taken_from_leading_comments_not_from_strings_or_the_body() -> None:
    """Only comments before the first line of code count: `@flow` inside a string, after code, or in
    a lookalike word must not route a plain JavaScript file to the Flow grammar, and a byte-order
    mark or shebang may precede the pragma."""
    # Act and assert
    assert has_flow_pragma(["// @flow", "const a = 1;"])
    assert has_flow_pragma(["\ufeff#!/usr/bin/env node", "/* @flow strict */", "const a = 1;"])
    assert has_flow_pragma(["/**", " * @flow", " */", "const a = 1;"])
    assert has_flow_pragma(["/* @flow */ const a = 1;"])
    assert not has_flow_pragma(['const note = "// @flow";', "export function later() {}"])
    assert not has_flow_pragma(["const a = 1;", "// @flow"])
    assert not has_flow_pragma(["// @flowish", "const a = 1;"])
    assert not has_flow_pragma(["const a = 1;"])


def test_the_flow_pragma_survives_a_leading_comment_longer_than_a_hundred_lines() -> None:
    """Leading comments have no length cap: a `@flow` pragma past the hundredth leading line still
    routes, whether it hides in one block comment or in a run of line comments."""
    # Arrange
    padded_block = ["/* banner"] + [f" * padding line {i}" for i in range(120)] + [" * @flow", " */"]
    padded_lines = [f"// padding line {i}" for i in range(120)] + ["// @flow"]

    # Act and assert
    assert has_flow_pragma([*padded_block, "const a = 1;"])
    assert has_flow_pragma([*padded_lines, "const a = 1;"])


def test_the_pragma_scan_stops_at_real_code_even_when_the_head_is_long() -> None:
    """Stopping at the first line of code — not at an arbitrary line count — keeps later `@flow`
    occurrences false-positive free no matter how long the leading comment was: `@flow` in a string,
    in a trailing comment, or after code never routes."""
    # Arrange
    long_head = [f"// padding line {i}" for i in range(150)]

    # Act and assert
    assert not has_flow_pragma([*long_head, 'const note = "@flow";'])
    assert not has_flow_pragma([*long_head, "const a = 1; // @flow"])
    assert not has_flow_pragma([*long_head, "const a = 1;", "// @flow"])
    assert not has_flow_pragma([*long_head, "const a = 1; /* @flow */"])


def test_a_flow_typed_class_keeps_its_methods(tmp_path: Path) -> None:
    """eval: `@flow` methods are recovered, not merely reported as omitted. The deciding spans the
    navigation needs (the class, its constructor and methods, module functions) resolve, the file
    parses without grammar errors, and the caller's binding degrades no further than a candidate."""
    # Arrange
    index = an_adapter_scope(tmp_path)

    # Act
    names = {span.name for span in index.symbols_in("src/adapters/postgres.js")}

    # Assert
    assert index.unparsed_files == set()
    assert {"PostgresAdapter", "constructor", "createObject", "find", "toPostgresValue", "connect"} <= names
    assert index.find_definition("createObject")
    site = index.find_callers("find")[0]
    assert (site.file, site.line) == ("src/adapters/index.js", 5)
    assert site.binding.status == "candidate"


def test_the_flow_partition_is_scanned_on_its_own(tmp_path: Path, ast_grep_runs) -> None:
    """The `languageGlobs` config is global per invocation, so `@flow` files are scanned in their own
    invocation and plain JavaScript keeps the JavaScript grammar byte for byte."""
    # Arrange
    index = an_adapter_scope(tmp_path)

    # Act
    assert index.find_definition("createObject")  # the structure scan
    index.find_references("connect")  # the calls and references scans

    # Assert
    flow_runs = [files for _, config, files in ast_grep_runs if config]
    assert flow_runs and all(files == ["src/adapters/postgres.js"] for files in flow_runs)
    assert all("src/adapters/memory.js" not in files for _, config, files in ast_grep_runs if config)


def test_plain_javascript_is_unchanged_whether_or_not_flow_files_share_the_scope(tmp_path: Path) -> None:
    # Arrange
    alone = committed(tmp_path / "alone", {"src/memory.js": MEMORY_ADAPTER, "src/auth.js": AUTH_MIDDLEWARE})
    mixed = committed(
        tmp_path / "mixed",
        {
            "src/memory.js": MEMORY_ADAPTER,
            "src/auth.js": AUTH_MIDDLEWARE,
            "src/postgres.js": FLOW_ADAPTER,
        },
    )

    # Act and assert
    for file in ("src/memory.js", "src/auth.js"):
        assert mixed.functions_in(file) == alone.functions_in(file)
        assert mixed.symbols_in(file) == alone.symbols_in(file)
        assert mixed.declarations_in(file) == alone.declarations_in(file)
    for name in ("authenticate", "createObject"):
        span = alone.find_definition(name)[0]
        assert mixed.find_callees(span) == alone.find_callees(span)
        assert mixed.find_references(name) == alone.find_references(name)
    assert alone.unparsed_files == set()


def test_an_unsupported_flow_construct_keeps_its_file_incomplete(tmp_path: Path) -> None:
    """`export opaque type` degrades to a grammar ERROR: the file must stay honestly unparsed while
    the methods recovery did keep still count."""
    # Arrange
    index = a_root_tag_scope(tmp_path)

    # Act
    names = {span.name for span in index.functions_in("src/native/RootTag.js")}

    # Assert
    assert "createRootTag" in names
    assert index.unparsed_files == {"src/native/RootTag.js"}


def test_a_name_defined_only_in_a_partly_recovered_file_is_unknown_not_unresolved(tmp_path: Path) -> None:
    # Arrange
    index = a_root_tag_scope(tmp_path)

    # Act
    site = index.find_callers("createRootTag")[0]

    # Assert
    assert site.binding.status == "unknown"
    assert "src/native/RootTag.js" in site.binding.reason


def test_a_malformed_flow_file_still_counts_as_unparsed(tmp_path: Path) -> None:
    # Arrange
    index = committed(tmp_path, {"src/broken.js": "// @flow\nexport class Broken {\n  find(a: string:\n"})

    # Act and assert
    assert index.unparsed_files == {"src/broken.js"}


def test_a_search_over_a_scope_with_grammar_errors_never_reports_nothing_left(tmp_path: Path) -> None:
    # Arrange
    index = a_root_tag_scope(tmp_path)
    judge = Judge(ScriptedJevClient(nouls=lambda question_id, question, state: 0.05))
    start = [place_for_line(index, "src/native/show.js", 4, "start")]

    # Act
    result = find_code(index, judge, "where the root tag is made", start)

    # Assert
    assert result.outcome == Outcome.SCOPE_INCOMPLETE
    assert result.unparsed_files == {"src/native/RootTag.js"}
    assert result.history.steps[-1].judgments["unparsed_files"] == ["src/native/RootTag.js"]
