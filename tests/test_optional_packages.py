"""The session check that refuses to run the suite without the packages jev-navigator's extras name."""

from __future__ import annotations

import pytest
from optional_packages import missing_optional_packages, require_optional_packages

ABSENT = "jvn-no-such-distribution-4f2a"


def _pyproject(tmp_path, *requirements: str):
    path = tmp_path / "pyproject.toml"
    listed = ", ".join(f'"{requirement}"' for requirement in requirements)
    path.write_text(f'[project]\nname = "demo"\n\n[project.optional-dependencies]\nlive = [{listed}]\n')
    return path


def test_extras_whose_packages_are_installed_report_nothing(tmp_path):
    # Arrange: an extra naming two installed distributions, one with a version range
    pyproject = _pyproject(tmp_path, "pytest>=8", "packaging")

    # Act and assert
    assert missing_optional_packages(pyproject) == []
    require_optional_packages(pyproject)


def test_a_missing_package_stops_the_session_and_is_named(tmp_path):
    # Arrange: an extra naming an installed distribution and one that cannot exist
    pyproject = _pyproject(tmp_path, "pytest", ABSENT)

    # Act and assert: the distribution name, not a module name, is what gets reported
    assert missing_optional_packages(pyproject) == [ABSENT]
    with pytest.raises(pytest.UsageError, match=f"not installed: {ABSENT}\\."):
        require_optional_packages(pyproject)
