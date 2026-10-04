"""A skipped test did not run, so a run with a skip nobody declared fails, naming each skip and its reason.
A skip is declared where it happens, so the mark also reaches an xdist controller. Two skips are declared:
a test's own `skipif` whose condition held (a platform it cannot run on), and, in a run started with
`--without-typesafe`, a test that skips because a module only the TypeSafe extra brings is missing. That
run proves the core works without the extra, and refuses to start when the extra is installed.
Not covered: an imperative `pytest.xfail` reports xfailed, and a conftest `collect_ignore` reports nothing."""

from __future__ import annotations

import importlib

import pytest

__all__ = ["pytest_addoption", "pytest_configure", "pytest_runtest_makereport", "pytest_sessionfinish"]

TYPESAFE_PACKAGE = "typesafe_sdk"
TYPESAFE_IMPORTS = (TYPESAFE_PACKAGE, "httpx2")
DECLARED = ("declared_skip", True)
SKIP_PREFIX = "Skipped: "


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--without-typesafe",
        action="store_true",
        help="the TypeSafe extra is absent on purpose, so its tests may skip",
    )


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("without_typesafe") and _installed(TYPESAFE_PACKAGE):
        raise pytest.UsageError(
            f"--without-typesafe was given, but {TYPESAFE_PACKAGE} is installed, so this run would not show "
            "that JVN works without it. Run it as `uv run --no-dev --with pytest pytest --without-typesafe`."
        )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> pytest.TestReport:
    report = yield
    if report.skipped and _declared(item, call.when, _reason(report)):
        report.user_properties = [*report.user_properties, DECLARED]
    return report


def pytest_sessionfinish(session: pytest.Session) -> None:
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    undeclared = [
        report
        for report in reporter.stats.get("skipped", [])
        if DECLARED not in getattr(report, "user_properties", ())
    ]
    if not undeclared:
        return
    count = len(undeclared)
    reporter.write_line(
        f"{count} skipped test{'s' if count > 1 else ''} did not run, so this run fails. Install what it "
        "needs (plain `uv run pytest` installs every extra), or declare a platform with a skipif condition:",
        red=True,
    )
    for report in undeclared:
        reporter.write_line(f"  {report.nodeid}: {_reason(report)}", red=True)
    if session.exitstatus in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def _declared(item: pytest.Item, phase: str, reason: str) -> bool:
    return _platform_skip(item, phase, reason) or (
        item.config.getoption("without_typesafe") and _missing_typesafe_import(reason)
    )


def _platform_skip(item: pytest.Item, phase: str, reason: str) -> bool:
    """`skipif` conditions are evaluated in setup, and a held condition skips with the marker's own reason."""
    marks = item.iter_markers("skipif")
    return phase == "setup" and any(mark.kwargs.get("reason") == reason for mark in marks)


def _missing_typesafe_import(reason: str) -> bool:
    return any(reason.startswith(f"could not import {module!r}") for module in TYPESAFE_IMPORTS)


def _reason(report: pytest.TestReport) -> str:
    location = report.longrepr
    message = location[2] if isinstance(location, tuple) else str(location)
    return message.removeprefix(SKIP_PREFIX)


def _installed(package: str) -> bool:
    try:
        importlib.import_module(package)
    except ImportError:
        return False
    return True
