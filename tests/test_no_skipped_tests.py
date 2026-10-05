"""A skipped test is a test that did not run. On 4 October three fresh checkouts silently skipped 56 to 58
tests whose package was missing, and each run still looked green, so an undeclared skip fails the run."""

from __future__ import annotations

from importlib.metadata import requires

import pytest
from no_skipped_tests import TYPESAFE_IMPORTS
from packaging.requirements import Requirement

pytest_plugins = ("pytester",)

GUARD = "from no_skipped_tests import *  # noqa: F403"
JEV_ABSENT = 'import sys\nsys.modules["typesafe_sdk"] = None\n'


@pytest.fixture
def guarded(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest(GUARD)
    return pytester


def test_a_skip_inside_a_test_fails_the_run_and_names_its_reason(guarded) -> None:
    # Arrange
    guarded.makepyfile(
        test_sample="""
        import pytest

        def test_runs():
            assert True

        def test_needs_a_missing_package():
            pytest.importorskip("a_package_that_is_not_installed")
        """
    )

    # Act
    result = guarded.runpytest()

    # Assert
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        [
            "*1 skipped test did not run*",
            "*test_needs_a_missing_package*could not import 'a_package_that_is_not_installed'*",
        ]
    )


def test_a_module_that_skips_while_collecting_fails_the_run(guarded) -> None:
    # Arrange
    guarded.makepyfile(
        test_sample="""
        import pytest

        pytest.importorskip("a_package_that_is_not_installed")

        def test_never_collected():
            assert True
        """
    )

    # Act
    result = guarded.runpytest()

    # Assert
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        ["*1 skipped test did not run*", "*test_sample.py*a_package_that_is_not_installed*"]
    )


def test_a_skip_the_test_declares_with_a_platform_condition_passes(guarded) -> None:
    # Arrange
    guarded.makepyfile(
        test_sample="""
        import pytest

        @pytest.mark.skipif(True, reason="needs a platform this run does not have")
        def test_platform_only():
            assert True
        """
    )

    # Act
    result = guarded.runpytest()

    # Assert
    assert result.ret == pytest.ExitCode.OK


def test_a_skipif_whose_condition_did_not_hold_declares_nothing(guarded) -> None:
    # Arrange
    guarded.makepyfile(
        test_sample="""
        import pytest

        @pytest.mark.skipif(False, reason="windows only")
        def test_needs_a_missing_package():
            pytest.importorskip("a_package_that_is_not_installed")
        """
    )

    # Act
    result = guarded.runpytest()

    # Assert
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*1 skipped test did not run*", "*a_package_that_is_not_installed*"])


def test_a_skip_that_only_repeats_a_skipif_reason_is_not_declared(guarded) -> None:
    # Arrange
    guarded.makepyfile(
        test_sample="""
        import pytest

        @pytest.mark.skipif(False, reason="windows only")
        def test_skips_by_hand():
            pytest.skip("windows only")
        """
    )

    # Act
    result = guarded.runpytest()

    # Assert
    assert result.ret == pytest.ExitCode.TESTS_FAILED


def test_a_fixture_skip_under_a_skipif_that_did_not_hold_is_not_declared(guarded) -> None:
    # Arrange
    guarded.makepyfile(
        test_sample="""
        import pytest

        @pytest.fixture
        def database():
            pytest.skip("no database")

        @pytest.mark.skipif(False, reason="windows only")
        def test_reads(database):
            assert True
        """
    )

    # Act
    result = guarded.runpytest()

    # Assert
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*1 skipped test did not run*", "*test_reads*no database*"])


def test_a_run_not_declared_without_jev_fails_when_jev_is_missing(guarded) -> None:
    # Arrange
    guarded.makeconftest(GUARD + "\n" + JEV_ABSENT)
    guarded.makepyfile(
        test_sample="""
        import pytest

        def test_needs_jev():
            pytest.importorskip("typesafe_sdk")
        """
    )

    # Act
    result = guarded.runpytest()

    # Assert
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*1 skipped test did not run*", "*could not import 'typesafe_sdk'*"])


def test_a_class_skipif_does_not_declare_a_skip_inside_its_method(guarded) -> None:
    # Arrange
    guarded.makepyfile(
        test_sample="""
        import pytest

        @pytest.mark.skipif(False, reason="windows only")
        class TestSample:
            def test_flaky(self):
                pytest.skip("flaky today")
        """
    )

    # Act
    result = guarded.runpytest()

    # Assert
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*1 skipped test did not run*", "*test_flaky*flaky today*"])


def test_a_run_without_skips_passes(guarded) -> None:
    # Arrange
    guarded.makepyfile(test_sample="def test_runs():\n    assert True\n")

    # Act
    result = guarded.runpytest()

    # Assert
    assert result.ret == pytest.ExitCode.OK


def test_a_run_declared_without_jev_lets_the_jev_tests_skip(guarded) -> None:
    # Arrange
    guarded.makeconftest(GUARD + "\n" + JEV_ABSENT)
    guarded.makepyfile(
        test_sample="""
        import pytest

        def test_needs_jev():
            pytest.importorskip("typesafe_sdk")
        """
    )

    # Act
    result = guarded.runpytest("--without-typesafe")

    # Assert
    assert result.ret == pytest.ExitCode.OK


def test_a_run_declared_without_jev_still_fails_a_skip_for_another_package(guarded) -> None:
    # Arrange
    guarded.makeconftest(GUARD + "\n" + JEV_ABSENT)
    guarded.makepyfile(
        test_sample="""
        import pytest

        def test_needs_jev():
            pytest.importorskip("typesafe_sdk")

        def test_needs_something_else():
            pytest.importorskip("a_package_that_is_not_installed")
        """
    )

    # Act
    result = guarded.runpytest("--without-typesafe")

    # Assert
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*1 skipped test did not run*", "*test_needs_something_else*"])


def test_the_modules_a_run_without_jev_may_miss_are_the_ones_the_extra_brings() -> None:
    # Arrange
    pytest.importorskip("typesafe_sdk")

    # Act
    requirements = {Requirement(line).name.replace("-", "_") for line in requires("typesafe-sdk")}
    brought = {"typesafe_sdk"} | requirements

    # Assert
    assert set(TYPESAFE_IMPORTS) <= brought


def test_a_run_declared_without_jev_refuses_to_start_when_jev_is_installed(guarded) -> None:
    # Arrange
    pytest.importorskip("typesafe_sdk")
    guarded.makepyfile(test_sample="def test_runs():\n    assert True\n")

    # Act
    result = guarded.runpytest("--without-typesafe")

    # Assert
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--without-typesafe*typesafe_sdk is installed*"])
