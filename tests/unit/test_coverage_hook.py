"""Tests for the coverage warning hook."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ("pytester",)
pytestmark = pytest.mark.allow_no_new_coverage


def _project_conftest() -> str:
    return Path(__file__).resolve().with_name("conftest.py").read_text(encoding="utf-8")


def test_warns_when_test_does_not_add_new_coverage(pytester: pytest.Pytester) -> None:
    """Tests that only repeat covered behavior should emit a warning."""
    pytester.makeconftest(_project_conftest())
    pytester.makepyfile(
        module_under_test="""
        def branch(flag: bool) -> int:
            if flag:
                return 1
            return 0
        """,
        test_first="""
        from module_under_test import branch


        def test_covers_false_branch() -> None:
            assert branch(False) == 0
        """,
        test_second="""
        from module_under_test import branch


        def test_repeats_false_branch() -> None:
            assert branch(False) == 0
        """,
    )

    result = pytester.runpytest("--cov=module_under_test", "-W", "default")

    result.assert_outcomes(passed=2, warnings=1)
    result.stdout.fnmatch_lines(
        [
            "*test_second.py::test_repeats_false_branch",
            "*PytestWarning: test did not introduce new line coverage*",
        ],
    )


def test_marker_suppresses_warning(pytester: pytest.Pytester) -> None:
    """Tests can opt out when zero new coverage is intentional."""
    pytester.makeconftest(_project_conftest())
    pytester.makepyfile(
        module_under_test="""
        def branch(flag: bool) -> int:
            if flag:
                return 1
            return 0
        """,
        test_example="""
        import pytest

        from module_under_test import branch


        def test_initial_coverage() -> None:
            assert branch(False) == 0


        @pytest.mark.allow_no_new_coverage
        def test_intentional_repeat() -> None:
            assert branch(False) == 0
        """,
    )

    result = pytester.runpytest("--cov=module_under_test", "-W", "default")

    result.assert_outcomes(passed=2)
    result.stdout.no_fnmatch_line("*did not introduce new line coverage*")
