"""A skipped test did not run, so a run with a skip nobody declared fails, naming each skip and its reason.
Two skips are declared: a test's own `skipif` condition (a platform it cannot run on), and the TypeSafe
extra's tests in a run started with `--without-typesafe`, which proves the core works without that extra.
That run refuses to start when the extra is installed, because it would then prove nothing."""

from __future__ import annotations

import importlib

import pytest

TYPESAFE_PACKAGE = "typesafe_sdk"


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


def pytest_sessionfinish(session: pytest.Session) -> None:
    if session.config.getoption("without_typesafe"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    undeclared = _undeclared_skips(session, reporter.stats.get("skipped", []))
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


def _undeclared_skips(session: pytest.Session, skipped: list[pytest.TestReport]) -> list[pytest.TestReport]:
    declared = {item.nodeid for item in session.items if item.get_closest_marker("skipif")}
    return [report for report in skipped if report.nodeid not in declared]


def _reason(report: pytest.TestReport) -> str:
    location = report.longrepr
    return location[2] if isinstance(location, tuple) else str(location)


def _installed(package: str) -> bool:
    try:
        importlib.import_module(package)
    except ImportError:
        return False
    return True
