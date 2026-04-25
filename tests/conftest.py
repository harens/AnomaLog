"""Suite-wide pytest configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers.

    Args:
        config (pytest.Config): Pytest config object to extend with suite markers.
    """
    config.addinivalue_line(
        "markers",
        "allow_no_new_coverage: suppress the warning for tests that only "
        "exercise already-covered lines",
    )


# Run integration tests last so unit tests aren't affected by
# "test did not introduce new line coverage" warnings due to the integration tests
# covering a lot of lines.
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run unit tests before integration tests for faster failure feedback.

    Args:
        items (list[pytest.Item]): Collected test items to reorder in place.
    """
    items.sort(
        key=lambda item: (
            "tests/integration/" in item.nodeid,
            item.nodeid,
        ),
    )
