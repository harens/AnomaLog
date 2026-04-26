"""Tests for the experiment runner CLI helpers."""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import TYPE_CHECKING

from experiments.runners import run_experiment as runner

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class _RecordingConsole:
    """Minimal console double that records printed messages."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, message: str, *, soft_wrap: bool) -> None:
        """Record one rendered log line.

        Args:
            message (str): Rendered message sent to the console.
            soft_wrap (bool): Whether Rich soft wrapping is enabled.
        """
        del soft_wrap
        self.messages.append(message)


def test_shared_console_handler_uses_shared_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Experiment console logs should be routed through the shared console.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the shared console accessor
            so the test can capture emitted output.
    """
    console = _RecordingConsole()
    monkeypatch.setattr(
        "experiments.runners.run_experiment.get_shared_console",
        lambda: console,
    )
    handler = runner.SharedConsoleHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger = logging.getLogger("tests.shared_console_handler")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)

    try:
        logger.info("progress-safe output")
    finally:
        logger.handlers.clear()

    assert console.messages == ["INFO progress-safe output"]


def test_main_does_not_print_the_run_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The CLI should let logging report the directory instead of printing twice.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the parser and runner so the
            test can exercise the CLI boundary without creating artefacts.
        capsys (pytest.CaptureFixture[str]): Captures any stdout emitted by the
            CLI entrypoint.
        tmp_path (Path): Temporary filesystem root used to fabricate a dummy
            run directory.
    """
    expected_config = object()
    seen: list[tuple[object, bool]] = []

    class _Parser:
        @staticmethod
        def parse_args() -> Namespace:
            return Namespace(config=expected_config, force=True)

    def _build_arg_parser() -> _Parser:
        return _Parser()

    monkeypatch.setattr(runner, "build_arg_parser", _build_arg_parser)
    monkeypatch.setattr(
        runner,
        "run_experiment",
        lambda config_path, *, force: (
            seen.append((config_path, force)) or tmp_path / "result-dir"
        ),
    )

    exit_code = runner.main()

    assert exit_code == 0
    assert seen == [(expected_config, True)]
    assert not capsys.readouterr().out
