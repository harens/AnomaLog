"""Tests for the coverage warning hook."""

from __future__ import annotations

from typing import Any

import pytest

import tests.unit.conftest as unit_conftest

pytestmark = pytest.mark.allow_no_new_coverage


def _current_coverage() -> object:
    return object()


def test_pytest_runtest_makereport_stashes_call_report(
    request: pytest.FixtureRequest,
) -> None:
    """The hook should keep the call-phase report available to later fixtures.

    Args:
        request (pytest.FixtureRequest): Pytest request used to access the
            current test item and its stash.
    """
    item = request.node
    reports_key = unit_conftest._TEST_REPORTS_KEY  # ruff: ignore[private-member-access]
    call = pytest.CallInfo.from_call(lambda: None, "call")
    report = pytest.TestReport.from_item_and_call(item, call)

    hook = unit_conftest.pytest_runtest_makereport(item, call)
    next(hook)

    with pytest.raises(StopIteration) as exc_info:
        hook.send(report)

    assert exc_info.value.value is report
    stored_reports = item.stash[reports_key]
    assert stored_reports["call"] is report


def test_warns_when_test_does_not_add_new_coverage(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Tests that only repeat covered behaviour should emit a warning.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the hook dependencies so the
            warning branch can be exercised deterministically.
        request (pytest.FixtureRequest): Pytest request used to build the
            synthetic report and fixture hook.
    """
    warnings: list[pytest.PytestWarning] = []

    def _allow_coverage_marker(
        _self: object,
        _name: str,
    ) -> object | None:
        return None

    def _record_warning(_self: object, warning: pytest.PytestWarning) -> None:
        warnings.append(warning)

    monkeypatch.setattr(
        type(request.node),
        "get_closest_marker",
        _allow_coverage_marker,
    )
    monkeypatch.setattr(
        type(request.node),
        "warn",
        _record_warning,
    )
    call = pytest.CallInfo.from_call(lambda: None, "call")
    report = pytest.TestReport.from_item_and_call(request.node, call)

    hook = unit_conftest.pytest_runtest_makereport(request.node, call)
    next(hook)
    with pytest.raises(StopIteration):
        hook.send(report)

    snapshots = iter(
        [
            {"module_under_test.py": frozenset({1})},
            {"module_under_test.py": frozenset({1})},
        ],
    )
    monkeypatch.setattr(
        unit_conftest,
        "_coverage_snapshot",
        lambda _cov: next(snapshots),
    )
    monkeypatch.setattr(
        unit_conftest.coverage.Coverage,
        "current",
        _current_coverage,
    )

    warn_fixture: Any = unit_conftest.warn_when_test_adds_no_new_coverage
    hook = warn_fixture.__wrapped__(request)
    next(hook)

    with pytest.raises(StopIteration):
        next(hook)

    assert [str(warning) for warning in warnings] == [
        (
            "test did not introduce new line coverage; add assertions for uncovered "
            "behavior or mark with @pytest.mark.allow_no_new_coverage"
        ),
    ]


def test_marker_suppresses_warning(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Tests can opt out when zero new coverage is intentional.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the hook dependencies so the
            suppression branch can be exercised deterministically.
        request (pytest.FixtureRequest): Pytest request used to build the
            synthetic report and fixture hook.
    """
    warnings: list[pytest.PytestWarning] = []

    def _mark_suppressed(
        _self: object,
        name: str,
    ) -> object | None:
        if name == "allow_no_new_coverage":
            return object()
        return None

    def _record_warning(_self: object, warning: pytest.PytestWarning) -> None:
        warnings.append(warning)

    monkeypatch.setattr(
        type(request.node),
        "get_closest_marker",
        _mark_suppressed,
    )
    monkeypatch.setattr(
        type(request.node),
        "warn",
        _record_warning,
    )

    warn_fixture: Any = unit_conftest.warn_when_test_adds_no_new_coverage
    hook = warn_fixture.__wrapped__(request)
    next(hook)

    with pytest.raises(StopIteration):
        next(hook)

    assert warnings == []
