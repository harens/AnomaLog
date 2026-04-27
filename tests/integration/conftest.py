"""Integration-specific pytest fixtures."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from prefect.settings import PREFECT_API_URL
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


@pytest.fixture(autouse=True)
def integration_prefect_api_url_env() -> Generator[None]:
    """Mirror the active Prefect API URL into subprocess environments."""
    previous_value = os.environ.get("PREFECT_API_URL")
    os.environ["PREFECT_API_URL"] = PREFECT_API_URL.value()
    try:
        yield
    finally:
        if previous_value is None:
            os.environ.pop("PREFECT_API_URL", None)
        else:
            os.environ["PREFECT_API_URL"] = previous_value
