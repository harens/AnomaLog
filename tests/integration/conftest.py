"""Integration-specific pytest fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from prefect.testing.utilities import prefect_test_harness

if TYPE_CHECKING:
    from collections.abc import Generator


PREFECT_TEST_SERVER_STARTUP_TIMEOUT_SECONDS = 60


@pytest.fixture(scope="session", autouse=True)
def integration_prefect_harness() -> Generator[None]:
    """Run integration tests against Prefect's test harness backend."""
    with prefect_test_harness(
        server_startup_timeout=PREFECT_TEST_SERVER_STARTUP_TIMEOUT_SECONDS,
    ):
        yield
