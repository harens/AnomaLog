"""Tests for the Thunderbird slice audit CLI entrypoint."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import experiments.runners.audit_thunderbird_slice as audit_cli

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_audit_thunderbird_slice_cli_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The CLI should pass parsed flags through to the audit helper.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the audit helper so the
            CLI can be exercised without touching parquet caches.
        capsys (pytest.CaptureFixture[str]): Captures the JSON payload written
            to standard output.
        tmp_path (Path): Temporary directory used to stage the fake cache
            root argument.
    """
    monkeypatch.setattr(
        audit_cli,
        "audit_thunderbird_slice",
        lambda **kwargs: {
            "kwargs": {
                "cache_root": str(kwargs["cache_root"]),
                "start_line_order": kwargs["start_line_order"],
                "end_line_order": kwargs["end_line_order"],
            },
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_thunderbird_slice",
            "--cache-root",
            str(tmp_path / "cache"),
            "--start-line-order",
            "4",
            "--end-line-order",
            "8",
        ],
    )

    assert audit_cli.main() == 0
    assert capsys.readouterr().out.strip() == json.dumps(
        {
            "kwargs": {
                "cache_root": (tmp_path / "cache").as_posix(),
                "end_line_order": 8,
                "start_line_order": 4,
            },
        },
        sort_keys=True,
    )
