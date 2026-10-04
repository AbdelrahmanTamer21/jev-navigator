"""The suite needs every distribution jev-navigator's extras name in `pyproject.toml`. Without one,
the tests that use it would be skipped or would fail one by one, so the session refuses to start
and names what is missing (see `conftest.pytest_configure`)."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
from packaging.requirements import Requirement

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def require_optional_packages(pyproject: Path = PYPROJECT) -> None:
    """Raise a usage error naming each distribution the extras require that is not installed."""
    missing = missing_optional_packages(pyproject)
    if missing:
        raise pytest.UsageError(
            "the test suite needs every package jev-navigator's extras name, and these are not "
            f"installed: {', '.join(missing)}. `uv run pytest` installs them through the dev group; "
            "in another environment, run `uv sync --all-extras`."
        )


def missing_optional_packages(pyproject: Path = PYPROJECT) -> list[str]:
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    extras = project.get("optional-dependencies", {})
    names = sorted({Requirement(spec).name for specs in extras.values() for spec in specs})
    return [name for name in names if not _installed(name)]


def _installed(distribution: str) -> bool:
    try:
        version(distribution)
    except PackageNotFoundError:
        return False
    return True
