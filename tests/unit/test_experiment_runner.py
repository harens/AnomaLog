"""Tests for the experiment runner CLI helpers."""

from __future__ import annotations

import logging
from argparse import Namespace
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol

from typing_extensions import Self

from experiments.runners import run_experiment as runner

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
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


class _OneShotSequenceView:
    """Replay probe that raises if the same view is iterated twice.

    Args:
        token (int): Monotonic identifier for the created sequence view.
    """

    def __init__(self, token: int) -> None:
        self.token = token
        self._consumed = False

    @staticmethod
    def split_count_hint() -> SimpleNamespace:
        return SimpleNamespace(train_count=1, test_count=1)

    @staticmethod
    def train_sequence_count_unit_hint() -> str:
        return "sequences"

    def __iter__(self) -> Iterator[SimpleNamespace]:
        if self._consumed:
            msg = f"sequence view {self.token} was reused"
            raise AssertionError(msg)
        self._consumed = True
        yield SimpleNamespace(
            split_label=SimpleNamespace(value="train"),
            label=0,
        )


class _SequenceConfig:
    """Sequence config double that returns a fresh one-shot view each time."""

    def __init__(self) -> None:
        self.apply_calls = 0

    def apply(self, templated: object) -> _OneShotSequenceView:
        return _sequence_config_apply(self, templated)


def _sequence_config_apply(
    config: _SequenceConfig,
    templated: object,
) -> _OneShotSequenceView:
    del templated
    config.apply_calls += 1
    return _OneShotSequenceView(config.apply_calls)


def _build_dataset_spec(_dataset: object, *, repo_root: Path) -> SimpleNamespace:
    del repo_root

    def _build() -> SimpleNamespace:
        return SimpleNamespace()

    return SimpleNamespace(build=_build)


def _build_sequence_split_summary(
    sequences: Iterable[object],
    *,
    sequence_summary: object,
) -> SimpleNamespace:
    del sequence_summary
    list(sequences)
    return SimpleNamespace(
        train_on_normal_entities_only=None,
        requested_train_fraction=0.2,
        realised_train_sequence_count=1,
        eligible_train_sequence_count=1,
        train_pool_sequence_count=1,
        ineligible_train_pool_count=0,
        ignored_sequence_count=0,
    )


def _build_run_metrics_report(
    *,
    bundle: object,
    sequences: Iterable[object],
    model_summary: object,
    debug_reporting: bool = False,
) -> dict[str, object]:
    del bundle, model_summary, debug_reporting
    list(sequences)
    return {
        "primary_metric_scope": None,
        "metric_blocks": {},
    }


@contextmanager
def _logger_context(
    *_args: object,
    **_kwargs: object,
) -> Iterator[logging.Logger]:
    yield logging.getLogger("tests.experiment_runner")


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
    seen: list[tuple[object, bool, bool, bool]] = []

    class _Parser:
        @staticmethod
        def parse_args() -> Namespace:
            return Namespace(
                config=expected_config,
                force=True,
                write_predictions=False,
            )

    def _build_arg_parser() -> _Parser:
        return _Parser()

    monkeypatch.setattr(runner, "build_arg_parser", _build_arg_parser)
    monkeypatch.setattr(
        runner,
        "run_experiment",
        lambda config_path, *, force, write_predictions, debug_reporting=False: (
            seen.append((config_path, force, write_predictions, debug_reporting))
            or tmp_path / "result-dir"
        ),
    )

    exit_code = runner.main()

    assert exit_code == 0
    assert seen == [(expected_config, True, False, False)]
    assert not capsys.readouterr().out


def test_run_experiment_submits_plain_worker_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Parallel sweeps should submit plain worker payloads.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces bundle loading and worker
            execution so the test can inspect submitted payloads.
        tmp_path (Path): Temporary path used to fabricate a sweep config path.
    """
    bundles = [
        SimpleNamespace(sweep=SimpleNamespace(max_workers=2)),
        SimpleNamespace(sweep=SimpleNamespace(max_workers=2)),
    ]
    submitted_payloads: list[tuple[Path, int, bool, bool, bool]] = []

    class _FakeExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object,
        ) -> None:
            del exc_type, exc, tb

        def map(
            self,
            func: object,
            payloads: list[tuple[Path, int, bool, bool, bool]],
        ) -> list[Path]:
            assert self.max_workers == len(bundles)
            del func
            submitted_payloads.extend(payloads)
            return [tmp_path / f"result-{index}" for index in range(len(payloads))]

    monkeypatch.setattr(runner, "load_experiment_bundles", lambda _path: bundles)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 8)
    result = runner.run_experiment(tmp_path / "sweep.toml", force=True)

    assert result == [tmp_path / "result-0", tmp_path / "result-1"]
    assert submitted_payloads == [
        (tmp_path / "sweep.toml", 0, True, False, False),
        (tmp_path / "sweep.toml", 1, True, False, False),
    ]


def test_run_experiment_batches_groups_sequentially(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Separate run groups should not execute in the same worker pool.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces bundle loading, the worker
            pool, and the serial runner so the test can inspect scheduling.
        tmp_path (Path): Temporary path used to fabricate result locations.
    """

    class _IndexedBundle(Protocol):
        index: int

    expected_max_workers = 2
    bundles = [
        SimpleNamespace(
            index=0,
            run_group="baselines",
            sweep=SimpleNamespace(max_workers=expected_max_workers),
        ),
        SimpleNamespace(
            index=1,
            run_group="baselines",
            sweep=SimpleNamespace(max_workers=expected_max_workers),
        ),
        SimpleNamespace(
            index=2,
            run_group="deepcase",
            sweep=SimpleNamespace(max_workers=expected_max_workers),
        ),
    ]
    submitted_payloads: list[list[tuple[Path, int, bool, bool, bool]]] = []
    serial_runs: list[int] = []

    class _FakeExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object,
        ) -> None:
            del exc_type, exc, tb

        def map(
            self,
            func: object,
            payloads: list[tuple[Path, int, bool, bool, bool]],
        ) -> list[Path]:
            assert self.max_workers == expected_max_workers
            del func
            submitted_payloads.append(payloads)
            return [tmp_path / f"parallel-{index}" for index in range(len(payloads))]

    def _run_bundle(
        bundle: _IndexedBundle,
        *,
        force: bool = False,
        write_predictions: bool = False,
        debug_reporting: bool = False,
    ) -> Path:
        del force, write_predictions, debug_reporting
        serial_runs.append(bundle.index)
        return tmp_path / f"serial-{bundle.index}"

    monkeypatch.setattr(runner, "load_experiment_bundles", lambda _path: bundles)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(runner, "_run_bundle", _run_bundle)
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 8)

    result = runner.run_experiment(tmp_path / "sweep.toml", force=True)

    assert result == [
        tmp_path / "parallel-0",
        tmp_path / "parallel-1",
        tmp_path / "serial-2",
    ]
    assert submitted_payloads == [
        [
            (tmp_path / "sweep.toml", 0, True, False, False),
            (tmp_path / "sweep.toml", 1, True, False, False),
        ],
    ]
    assert serial_runs == [2]


def test_run_bundle_rebuilds_sequence_views_for_each_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Each runner stage should receive a fresh sequence view.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces filesystem and orchestration
            helpers so the test can focus on sequence-view reuse.
        tmp_path (Path): Temporary filesystem root used for the fake run.
    """
    sequence_config = _SequenceConfig()

    bundle = SimpleNamespace(
        sweep_path=tmp_path / "sweep.toml",
        dataset_path=tmp_path / "dataset.toml",
        model_path=tmp_path / "model.toml",
        sweep=SimpleNamespace(max_workers=1),
        dataset=SimpleNamespace(
            dataset_name="demo",
            sequence=sequence_config,
        ),
        model=SimpleNamespace(detector="demo", name="demo"),
        concrete_name="demo-run",
        applied_overrides={},
        repo_root=tmp_path,
    )
    run_paths = SimpleNamespace(
        run_dir=tmp_path / "run",
        run_log_path=tmp_path / "run.log",
        predictions_path=tmp_path / "predictions.jsonl",
    )

    monkeypatch.setattr(runner, "prepare_result_paths", lambda _bundle: run_paths)
    monkeypatch.setattr(runner, "build_dataset_spec", _build_dataset_spec)
    monkeypatch.setattr(runner, "_experiment_logger", _logger_context)
    monkeypatch.setattr(
        sequence_config,
        "apply",
        lambda templated: _sequence_config_apply(sequence_config, templated),
    )
    monkeypatch.setattr(
        runner,
        "run_model",
        lambda *, sequence_factory, **_kwargs: (
            list(sequence_factory()),
            list(sequence_factory()),
            SimpleNamespace(
                sequence_summary=SimpleNamespace(
                    sequence_count=2,
                    train_sequence_count=1,
                    test_sequence_count=1,
                    ignored_sequence_count=0,
                ),
            ),
        )[2],
    )
    monkeypatch.setattr(
        runner,
        "build_sequence_split_summary",
        _build_sequence_split_summary,
    )
    monkeypatch.setattr(
        runner,
        "build_run_metrics_report",
        _build_run_metrics_report,
    )
    monkeypatch.setattr(runner, "write_run_outputs", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "load_experiment_bundles", lambda _path: [bundle])

    runner.run_experiment(tmp_path / "sweep.toml", force=True)

    expected_apply_calls = 6
    assert sequence_config.apply_calls == expected_apply_calls
