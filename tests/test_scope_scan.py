from __future__ import annotations

import subprocess

import pytest

from jev_navigator.directives.places import neighbours_and_omissions
from jev_navigator.index import scope_scan, tools
from jev_navigator.index.code_index import CodeIndex


@pytest.fixture
def ast_grep_runs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    runs: list[str] = []
    original_rules, original_pattern = tools.ast_grep_rules, tools.ast_grep_pattern

    def counted_rules(rules: str, files, cwd):
        runs.append("rules")
        return original_rules(rules, files, cwd)

    def counted_pattern(pattern: str, files, cwd):
        runs.append("pattern")
        return original_pattern(pattern, files, cwd)

    monkeypatch.setattr(tools, "ast_grep_rules", counted_rules)
    monkeypatch.setattr(tools, "ast_grep_pattern", counted_pattern)
    return runs


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


def test_each_call_site_binding_is_computed_once(sample_index: CodeIndex) -> None:
    # Arrange
    sites = sample_index.find_callers("check_limits")

    # Act
    again = sample_index.find_callers("check_limits")

    # Assert
    assert [site.binding for site in again] == [site.binding for site in sites]
    assert all(first.binding is second.binding for first, second in zip(sites, again, strict=True))


def test_a_batch_that_times_out_is_reported_and_the_rest_still_parse(
    sample_index: CodeIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    original = tools.ast_grep_rules

    def slow_on_orders(rules: str, files, cwd):
        if "app/orders.py" in files:
            raise subprocess.TimeoutExpired(["ast-grep"], tools.COMMAND_TIMEOUT_SECONDS)
        return original(rules, files, cwd)

    monkeypatch.setattr(scope_scan, "BATCH_FILES", 1)
    monkeypatch.setattr(tools, "ast_grep_rules", slow_on_orders)

    # Act
    definitions = sample_index.find_definition("check_limits")

    # Assert
    assert [span.name for span in definitions] == ["check_limits"]
    assert sample_index.unparsed_files == {"app/orders.py"}
    assert sample_index.symbols_in("app/orders.py") == ()
