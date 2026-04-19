"""Unit-test-specific pytest hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import coverage
import pytest
from prefect.testing.utilities import prefect_test_harness

_TEST_REPORTS_KEY = pytest.StashKey[dict[str, pytest.TestReport]]()
PREFECT_TEST_SERVER_STARTUP_TIMEOUT_SECONDS = 60

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any


@pytest.fixture(scope="session", autouse=True)
def unit_prefect_harness() -> Generator[None]:
    """Run unit tests against Prefect's session-scoped test harness backend."""
    with prefect_test_harness(
        server_startup_timeout=PREFECT_TEST_SERVER_STARTUP_TIMEOUT_SECONDS,
    ):
        yield


def _coverage_snapshot(cov: coverage.Coverage) -> dict[str, frozenset[int]]:
    """Capture the currently executed lines for each measured file.

    Args:
        cov (coverage.Coverage): Active coverage collector to read from.

    Returns:
        dict[str, frozenset[int]]: Covered line numbers keyed by filename.
    """
    data = cov.get_data()
    return {
        filename: frozenset(data.lines(filename) or ())
        for filename in data.measured_files()
    }


def _introduces_new_coverage(
    before: dict[str, frozenset[int]],
    after: dict[str, frozenset[int]],
) -> bool:
    """Return whether any new covered lines were added."""
    return any(
        lines - before.get(filename, frozenset()) for filename, lines in after.items()
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],
) -> Generator[None, Any, pytest.TestReport]:
    """Store the phase reports for each test item.

    Args:
        item (pytest.Item): Test item whose phase reports should be stored.
        call (pytest.CallInfo[None]): Call metadata for the current phase.

    Returns:
        Generator[None, Any, pytest.TestReport]: Wrapped pytest hook generator.
    """
    del call
    report = yield
    reports = item.stash.setdefault(_TEST_REPORTS_KEY, {})
    reports[report.when] = report
    return report


@pytest.fixture(autouse=True)
def warn_when_test_adds_no_new_coverage(
    request: pytest.FixtureRequest,
) -> Generator[None]:
    """Warn when a unit test does not increase cumulative line coverage."""
    if request.node.get_closest_marker("allow_no_new_coverage") is not None:
        yield
        return

    cov = coverage.Coverage.current()
    if cov is None:
        yield
        return

    before = _coverage_snapshot(cov)
    yield

    report = request.node.stash.get(_TEST_REPORTS_KEY, {}).get("call")
    if report is None or report.skipped:
        return

    after = _coverage_snapshot(cov)
    if _introduces_new_coverage(before, after):
        return

    request.node.warn(
        pytest.PytestWarning(
            "test did not introduce new line coverage; add assertions for "
            "uncovered behavior or mark with @pytest.mark.allow_no_new_coverage",
        ),
    )
