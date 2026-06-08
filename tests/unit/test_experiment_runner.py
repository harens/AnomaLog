"""Tests for the experiment runner CLI helpers."""

from __future__ import annotations

import logging
from argparse import Namespace
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, Protocol

import pytest
from typing_extensions import Self

from experiments.config import (
    DatasetVariantConfig,
    EntitySequenceConfig,
    ExperimentBundle,
)
from experiments.config_types import (
    RawEntryPrefixCountSplitConfig,
    SplitApplicationOrder,
    StraddlingGroupPolicy,
)
from experiments.models.base import decode_experiment_model_config
from experiments.models.template_frequency import TemplateFrequencyModelConfig
from experiments.results import ResultPaths
from experiments.runners import run_experiment as runner

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


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

    @property
    def sequences(self) -> _OneShotSequenceView:
        return self

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

    def iter_training_sequences(self) -> Iterator[SimpleNamespace]:
        return iter(self)


@dataclass(frozen=True)
class _FakeRunConfig:
    name: str = "demo"
    dataset: DatasetVariantConfig | None = None
    models: list[TemplateFrequencyModelConfig] = field(default_factory=list)
    results_root: Path = Path("experiments/results")
    description: str | None = None
    max_workers: int | Literal["auto"] = 1


def _make_bundle(tmp_path: Path, *, concrete_name: str = "demo") -> ExperimentBundle:
    dataset = DatasetVariantConfig(
        name="demo",
        dataset_name="demo",
        preset="demo",
        sequence=EntitySequenceConfig(),
    )
    return ExperimentBundle(
        experiments_root=tmp_path / "experiments",
        repo_root=tmp_path,
        sweep_path=tmp_path / "sweep.toml",
        dataset_path=tmp_path / "dataset.toml",
        model_path=tmp_path / "model.toml",
        sweep=_FakeRunConfig(),
        dataset=dataset,
        model=decode_experiment_model_config(
            {"name": "template_frequency"},
            config_type=TemplateFrequencyModelConfig,
        ),
        concrete_name=concrete_name,
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

    return SimpleNamespace(clear_cache=lambda: None, build=_build)


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
    seen: list[tuple[object, bool, bool, bool, bool]] = []

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
        lambda config_path, *, force, rerun, write_predictions, debug_reporting=False: (
            seen.append(
                (config_path, force, rerun, write_predictions, debug_reporting),
            )
            or tmp_path / "result-dir"
        ),
    )

    exit_code = runner.main()

    assert exit_code == 0
    assert seen == [(expected_config, True, False, False, False)]
    assert not capsys.readouterr().out


def test_build_arg_parser_exposes_config_and_registry_inputs() -> None:
    """The CLI parser should expose both config and registry modes."""
    parser = runner.build_arg_parser()
    help_text = parser.format_help()

    assert "--config" in help_text
    assert "--experiment" in help_text
    assert "--registry" in help_text
    assert "--repo-root" in help_text
    assert "--force" in help_text
    assert "--rerun" in help_text
    assert "--write-predictions" in help_text
    assert "--debug-reporting" in help_text


def test_run_registered_experiment_forwards_rerun_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Registry-backed runs should forward rerun mode into bundle execution.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces registry resolution and
            bundle execution so the test can inspect forwarded options.
        tmp_path (Path): Temporary filesystem root used to fabricate a registry
            result path.
    """
    bundle = _make_bundle(tmp_path, concrete_name="demo")
    seen: list[runner._BundleRunOptions] = []

    monkeypatch.setattr(
        runner,
        "resolve_registry_experiment",
        lambda *_args, **_kwargs: SimpleNamespace(bundles=[bundle]),
    )
    monkeypatch.setattr(
        runner,
        "_run_bundle",
        lambda _bundle, *, options: (
            seen.append(options),
            tmp_path / "result",
        )[1],
    )

    result = runner.run_registered_experiment(
        runner.RegisteredExperimentRunRequest(
            experiment_name="demo",
            repo_root=tmp_path,
            force=True,
            rerun=True,
            write_predictions=True,
            debug_reporting=True,
            console=False,
        ),
    )

    assert result == [tmp_path / "result"]
    assert seen == [
        runner._BundleRunOptions(  # noqa: SLF001
            force=True,
            rerun=True,
            write_predictions=True,
            debug_reporting=True,
            console=False,
        ),
    ]


def test_run_bundle_from_manifest_payload_forwards_rerun_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Manifest payload workers should decode rerun flags into bundle options.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces manifest loading and bundle
            execution so the test can inspect the worker payload.
        tmp_path (Path): Temporary filesystem root used to fabricate a sweep
            path and result path.
    """
    bundle = _make_bundle(tmp_path, concrete_name="demo")
    seen: list[runner._BundleRunOptions] = []

    monkeypatch.setattr(
        runner,
        "load_experiment_bundles",
        lambda _path: [bundle],
    )
    monkeypatch.setattr(
        runner,
        "_run_bundle",
        lambda _bundle, *, options: (
            seen.append(options),
            tmp_path / "result",
        )[1],
    )

    result = runner._run_bundle_from_manifest_payload(  # noqa: SLF001
        (
            tmp_path / "sweep.toml",
            0,
            True,
            True,
            False,
            True,
        ),
    )

    assert result == tmp_path / "result"
    assert seen == [
        runner._BundleRunOptions(  # noqa: SLF001
            force=True,
            rerun=True,
            write_predictions=False,
            debug_reporting=True,
        ),
    ]


def test_failure_helpers_format_bundle_exceptions(
    tmp_path: Path,
) -> None:
    """Bundle failure helpers should return stable log messages.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for the synthetic bundle
            paths used by the helper checks.
    """
    bundle = _make_bundle(tmp_path, concrete_name="demo")

    failure = runner._run_bundle_with_failure_capture(  # noqa: SLF001
        bundle,
        options=runner._BundleRunOptions(),  # noqa: SLF001
    )
    assert failure[1] is not None
    assert "Traceback (most recent call last):" in failure[1]

    future: Future[Path] = Future()
    future.set_exception(RuntimeError("boom"))
    captured = runner._capture_future_result(future, bundle)  # noqa: SLF001
    assert captured[0] is None
    assert captured[1] is not None
    assert captured[1].startswith("demo: boom")
    assert "Traceback (most recent call last):" in captured[1]
    assert runner._format_bundle_failure(bundle, RuntimeError()) == "demo: RuntimeError"  # noqa: SLF001


def test_run_bundle_logs_traceback_before_reraising(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Concrete run failures should be logged with a traceback before escaping.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the run logger and model
            execution so the test can force a controlled failure.
        tmp_path (Path): Temporary path used to fabricate the run bundle.
    """
    bundle = _make_bundle(tmp_path, concrete_name="demo")
    run_paths = SimpleNamespace(
        run_dir=tmp_path / "run",
        run_root=tmp_path / "run",
        run_log_path=tmp_path / "run.log",
        predictions_path=tmp_path / "predictions.jsonl",
        metrics_path=tmp_path / "run" / "metrics.json",
        dataset_manifest_path=tmp_path / "run" / "dataset_manifest.json",
        config_path=tmp_path / "run" / "experiment_config.json",
        environment_path=tmp_path / "run" / "environment.json",
    )
    logger_messages: list[tuple[str, str]] = []

    class _SequenceView:
        def with_split_fractions(self, *_args: object) -> _SequenceView:
            return self

        @staticmethod
        def split_count_hint() -> SimpleNamespace:
            return SimpleNamespace(train_count=1, test_count=1)

        @staticmethod
        def train_sequence_count_unit_hint() -> str:
            return "sequences"

        @staticmethod
        def iter_training_sequences() -> Iterator[SimpleNamespace]:
            return iter(())

    class _Logger:
        @staticmethod
        def info(message: str, *args: object) -> None:
            logger_messages.append(("info", message % args if args else message))

        @staticmethod
        def exception(message: str, *args: object) -> None:
            logger_messages.append(("exception", message % args if args else message))

    class _Templated:
        @staticmethod
        def group_by_entity() -> _SequenceView:
            return _SequenceView()

    class _DatasetSpec:
        @staticmethod
        def build() -> _Templated:
            return _Templated()

        @staticmethod
        def clear_cache() -> None:
            return None

    def _build_dataset_spec(*_args: object, **_kwargs: object) -> _DatasetSpec:
        return _DatasetSpec()

    @contextmanager
    def _logger_context(
        *_args: object,
        **_kwargs: object,
    ) -> Iterator[object]:
        yield _Logger()

    monkeypatch.setattr(runner, "_experiment_logger", _logger_context)
    monkeypatch.setattr(
        runner,
        "_prepare_result_paths",
        lambda _bundle, **_kwargs: run_paths,
    )
    monkeypatch.setattr(
        runner,
        "build_dataset_spec",
        _build_dataset_spec,
    )
    monkeypatch.setattr(
        runner,
        "run_model",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        runner._run_bundle(  # noqa: SLF001
            bundle,
            options=runner._BundleRunOptions(),  # noqa: SLF001
        )

    assert any(kind == "exception" for kind, _ in logger_messages)
    assert any(
        "Concrete experiment demo failed" in message for _, message in logger_messages
    )


def test_execute_bundle_skips_exact_split_count_hint_for_raw_entry_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Concrete raw-entry splits should not pay for an exact count pre-pass.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces dataset construction and
            model execution so the test can reach the split-count branch
            without doing any heavy work.
        tmp_path (Path): Temporary filesystem root used to fabricate the run
            artefact paths.
        caplog (pytest.LogCaptureFixture): Captures the runner's skip message so
            the test can verify the expensive replay is bypassed.
    """

    @dataclass
    class _SequenceView:
        split_mode: object | None = None
        split_application_order: object | None = None
        straddling_group_policy: object | None = None
        train_entry_count: int | None = None

        def with_split_fractions(
            self,
            *_args: object,
        ) -> _SequenceView:
            return self

        @staticmethod
        def split_count_hint() -> SimpleNamespace:
            msg = "exact split counting should be skipped for raw-entry splits"
            raise AssertionError(msg)

        @staticmethod
        def train_sequence_count_unit_hint() -> str:
            return "entities"

        @staticmethod
        def iter_training_sequences() -> Iterator[SimpleNamespace]:
            return iter(())

    class _Templated:
        @staticmethod
        def group_by_entity() -> _SequenceView:
            return _SequenceView()

    class _DatasetSpec:
        @staticmethod
        def build() -> _Templated:
            return _Templated()

        @staticmethod
        def clear_cache() -> None:
            return None

    bundle = ExperimentBundle(
        experiments_root=tmp_path / "experiments",
        repo_root=tmp_path,
        sweep_path=tmp_path / "sweep.toml",
        dataset_path=tmp_path / "dataset.toml",
        model_path=tmp_path / "model.toml",
        sweep=_FakeRunConfig(),
        dataset=DatasetVariantConfig(
            name="demo",
            dataset_name="demo",
            preset="demo",
            sequence=EntitySequenceConfig(
                split=RawEntryPrefixCountSplitConfig(
                    train_entry_count=1,
                    application_order=SplitApplicationOrder.BEFORE_GROUPING,
                    straddling_group_policy=(
                        StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES
                    ),
                ),
            ),
        ),
        model=decode_experiment_model_config(
            {"name": "template_frequency"},
            config_type=TemplateFrequencyModelConfig,
        ),
        applied_overrides={"dataset.train_fraction": 0.25},
        concrete_name="demo",
    )
    result_paths = ResultPaths(
        run_fingerprint="fingerprint",
        run_root=tmp_path / "run-root",
        run_dir=tmp_path / "run",
        config_path=tmp_path / "run" / "experiment_config.json",
        dataset_manifest_path=tmp_path / "run" / "dataset_manifest.json",
        metrics_path=tmp_path / "run" / "metrics.json",
        predictions_path=tmp_path / "predictions.jsonl",
        environment_path=tmp_path / "run" / "environment.json",
        run_log_path=tmp_path / "run.log",
    )
    logger = logging.getLogger("tests.execute_bundle_split_count_hint")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = True

    monkeypatch.setattr(
        runner,
        "build_dataset_spec",
        lambda *_args, **_kwargs: _DatasetSpec(),
    )
    saw_progress_plan_none = False

    def _run_model(**kwargs: object) -> SimpleNamespace:
        nonlocal saw_progress_plan_none
        progress_plan = kwargs["progress_plan"]
        assert isinstance(progress_plan, runner.RunProgressPlan)
        assert progress_plan.train is None
        assert progress_plan.score is None
        saw_progress_plan_none = True
        return SimpleNamespace()

    monkeypatch.setattr(runner, "run_model", _run_model)
    monkeypatch.setattr(runner, "_finalise_bundle_run", lambda **_kwargs: None)

    with caplog.at_level(logging.INFO):
        returned = runner._execute_bundle_run(  # noqa: SLF001
            bundle=bundle,
            options=runner._BundleRunOptions(console=False),  # noqa: SLF001
            result_paths=result_paths,
            logger=logger,
        )

    assert returned == result_paths.run_dir
    assert any(
        message.startswith(
            "Skipping exact sequence split count hint for demo because the "
            "raw-entry before-grouping split would require a full replay",
        )
        for message in caplog.messages
    )
    assert any(
        message.startswith("Applied overrides: {'dataset.train_fraction': 0.25}")
        for message in caplog.messages
    )
    assert saw_progress_plan_none


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
    submitted_payloads: list[tuple[Path, int, bool, bool, bool, bool]] = []

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
            payloads: list[tuple[Path, int, bool, bool, bool, bool]],
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
        (tmp_path / "sweep.toml", 0, True, False, False, False),
        (tmp_path / "sweep.toml", 1, True, False, False, False),
    ]


def test_prepare_result_paths_returns_base_paths_when_not_rerunning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Non-rerun calls should return the deterministic result path bundle.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the result-path builder so
            the test can observe the branch selection.
        tmp_path (Path): Temporary filesystem root used to fabricate a bundle.
    """
    bundle = _make_bundle(tmp_path, concrete_name="demo")
    base_paths = SimpleNamespace(
        run_root=tmp_path / "results",
        run_dir=tmp_path / "run",
    )
    attempts: list[int | None] = []

    def _prepare_result_paths(
        _bundle: object,
        *,
        run_attempt: int | None = None,
    ) -> SimpleNamespace:
        attempts.append(run_attempt)
        return base_paths

    monkeypatch.setattr(runner, "prepare_result_paths", _prepare_result_paths)

    result = runner._prepare_result_paths(  # noqa: SLF001
        bundle,
        rerun=False,
    )

    assert attempts == [None]
    assert result is base_paths


def test_prepare_result_paths_allocates_next_attempt_for_reruns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reruns should land in the next numbered attempt directory.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the result-path builder so
            the test can drive the attempt selection logic deterministically.
        tmp_path (Path): Temporary filesystem root used to fake existing
            attempt directories.
    """
    run_root = tmp_path / "results" / "demo" / "fingerprint"
    (run_root / "attempt-0001").mkdir(parents=True)
    (run_root / "attempt-0003").mkdir(parents=True)

    base_paths = SimpleNamespace(
        run_root=run_root,
        run_dir=run_root,
        metrics_path=run_root / "metrics.json",
    )
    attempts: list[int | None] = []

    def _prepare_result_paths(
        _bundle: object,
        *,
        run_attempt: int | None = None,
    ) -> SimpleNamespace:
        attempts.append(run_attempt)
        if run_attempt is None:
            return base_paths
        attempt_dir = run_root / f"attempt-{run_attempt:04d}"
        return SimpleNamespace(
            run_root=run_root,
            run_dir=attempt_dir,
            metrics_path=attempt_dir / "metrics.json",
        )

    monkeypatch.setattr(runner, "prepare_result_paths", _prepare_result_paths)

    fake_bundle = _make_bundle(tmp_path, concrete_name="demo")
    rerun_paths = runner._prepare_result_paths(  # noqa: SLF001
        fake_bundle,
        rerun=True,
    )

    assert attempts == [None, 4]
    assert rerun_paths.run_dir == run_root / "attempt-0004"


def test_reserve_rerun_result_paths_retries_when_directory_already_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Rerun reservation should retry when a competing attempt wins first.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the result-path builder so
            the test can simulate a collision on the first attempt.
        tmp_path (Path): Temporary filesystem root used to fabricate attempt
            directories.
    """
    bundle = _make_bundle(tmp_path, concrete_name="demo")
    first_run_dir = tmp_path / "results" / "demo" / "fingerprint" / "attempt-0001"
    second_run_dir = tmp_path / "results" / "demo" / "fingerprint" / "attempt-0002"
    first_run_dir.mkdir(parents=True)
    attempts = [
        SimpleNamespace(run_dir=first_run_dir),
        SimpleNamespace(run_dir=second_run_dir),
    ]
    seen = 0

    def _prepare_result_paths(
        _bundle: object,
        *,
        rerun: bool,
    ) -> SimpleNamespace:
        nonlocal seen
        del rerun
        seen += 1
        return attempts.pop(0)

    monkeypatch.setattr(runner, "_prepare_result_paths", _prepare_result_paths)

    result = runner._reserve_rerun_result_paths(bundle)  # noqa: SLF001

    assert result.run_dir == second_run_dir
    expected_attempt_count = 2
    assert seen == expected_attempt_count


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
    submitted_payloads: list[list[tuple[Path, int, bool, bool, bool, bool]]] = []
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
            payloads: list[tuple[Path, int, bool, bool, bool, bool]],
        ) -> list[Path]:
            assert self.max_workers == expected_max_workers
            del func
            submitted_payloads.append(payloads)
            return [tmp_path / f"parallel-{index}" for index in range(len(payloads))]

    def _run_bundle(
        bundle: _IndexedBundle,
        *,
        options: object,
    ) -> Path:
        del options
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
            (tmp_path / "sweep.toml", 0, True, False, False, False),
            (tmp_path / "sweep.toml", 1, True, False, False, False),
        ],
    ]
    assert serial_runs == [2]


def test_run_experiment_parallelises_baselines_with_nb_before_deepcase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The registry baseline group should submit together before DeepCASE.

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
            run_group="baselines_with_nb",
            sweep=SimpleNamespace(max_workers=expected_max_workers),
        ),
        SimpleNamespace(
            index=1,
            run_group="baselines_with_nb",
            sweep=SimpleNamespace(max_workers=expected_max_workers),
        ),
        SimpleNamespace(
            index=2,
            run_group="baselines_with_nb",
            sweep=SimpleNamespace(max_workers=expected_max_workers),
        ),
        SimpleNamespace(
            index=3,
            run_group="deepcase",
            sweep=SimpleNamespace(max_workers=expected_max_workers),
        ),
    ]
    submitted_payloads: list[tuple[Path, int, bool, bool, bool, bool]] = []
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

        @staticmethod
        def submit(
            func: object,
            payload: tuple[Path, int, bool, bool, bool, bool],
        ) -> Future[Path]:
            del func
            submitted_payloads.append(payload)
            future: Future[Path] = Future()
            index = payload[1]
            future.set_result(tmp_path / f"parallel-{index}")
            return future

    def _run_bundle(
        bundle: _IndexedBundle,
        *,
        options: object,
    ) -> Path:
        del options
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
        tmp_path / "parallel-2",
        tmp_path / "serial-3",
    ]
    assert submitted_payloads == [
        (tmp_path / "sweep.toml", 0, True, False, False, False),
        (tmp_path / "sweep.toml", 1, True, False, False, False),
        (tmp_path / "sweep.toml", 2, True, False, False, False),
    ]
    assert serial_runs == [3]


def test_run_experiment_logs_bundle_failures_and_keeps_running(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """One failing bundle should not stop the rest of the run group.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces bundle loading and worker
            execution so the test can inspect failure handling.
        capsys (pytest.CaptureFixture[str]): Captures the failure summary
            emitted by the runner.
        tmp_path (Path): Temporary path used to fabricate result locations.
    """
    bundles = [
        SimpleNamespace(
            concrete_name="demo-a",
            run_group="baselines",
            sweep=SimpleNamespace(max_workers=2),
        ),
        SimpleNamespace(
            concrete_name="demo-b",
            run_group="baselines",
            sweep=SimpleNamespace(max_workers=2),
        ),
        SimpleNamespace(
            concrete_name="demo-c",
            run_group="baselines",
            sweep=SimpleNamespace(max_workers=2),
        ),
    ]
    submitted_payloads: list[tuple[Path, int, bool, bool, bool, bool]] = []

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

        @staticmethod
        def submit(
            func: object,
            payload: tuple[Path, int, bool, bool, bool, bool],
        ) -> Future[Path]:
            del func
            submitted_payloads.append(payload)
            future: Future[Path] = Future()
            index = payload[1]
            if index == 1:
                future.set_exception(RuntimeError("boom"))
            else:
                future.set_result(tmp_path / f"result-{index}")
            return future

    monkeypatch.setattr(runner, "load_experiment_bundles", lambda _path: bundles)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 8)

    result = runner.run_experiment(tmp_path / "sweep.toml", force=True)

    assert result == [tmp_path / "result-0", tmp_path / "result-2"]
    assert submitted_payloads == [
        (tmp_path / "sweep.toml", 0, True, False, False, False),
        (tmp_path / "sweep.toml", 1, True, False, False, False),
        (tmp_path / "sweep.toml", 2, True, False, False, False),
    ]
    output_lines = capsys.readouterr().out.splitlines()
    assert output_lines[0] == "One or more runs in this group failed:"
    assert output_lines[1] == "  - demo-b: boom"


def test_next_run_attempt_ignores_invalid_attempt_directories(
    tmp_path: Path,
) -> None:
    """Attempt numbering should ignore stray directories and bad suffixes.

    Args:
        tmp_path (Path): Temporary filesystem root used to fabricate a mixed
            attempt directory listing.
    """
    run_root = tmp_path / "results" / "demo" / "fingerprint"
    (run_root / "attempt-0002").mkdir(parents=True)
    (run_root / "attempt-abc").mkdir()
    (run_root / "misc").mkdir()

    expected_next_attempt = 3
    assert runner._next_run_attempt(run_root) == expected_next_attempt  # noqa: SLF001


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
    bundle = _make_bundle(tmp_path, concrete_name="demo-run")
    run_paths = SimpleNamespace(
        run_dir=tmp_path / "run",
        run_root=tmp_path / "run",
        run_log_path=tmp_path / "run.log",
        predictions_path=tmp_path / "predictions.jsonl",
        metrics_path=tmp_path / "run" / "metrics.json",
        dataset_manifest_path=tmp_path / "run" / "dataset_manifest.json",
        config_path=tmp_path / "run" / "experiment_config.json",
        environment_path=tmp_path / "run" / "environment.json",
    )

    class _SequenceViewStub:
        def with_split_fractions(self, *_args: object) -> _SequenceViewStub:
            return self

        @staticmethod
        def split_count_hint() -> SimpleNamespace:
            return SimpleNamespace(train_count=1, test_count=1)

        @staticmethod
        def train_sequence_count_unit_hint() -> str:
            return "sequences"

        def __iter__(self) -> Iterator[SimpleNamespace]:
            yield SimpleNamespace(
                split_label=SimpleNamespace(value="train"),
                label=0,
            )

        def iter_training_sequences(self) -> Iterator[SimpleNamespace]:
            return iter(self)

    class _TemplatedStub:
        def __init__(self) -> None:
            self.group_by_entity_calls = 0

        def group_by_entity(self) -> _SequenceViewStub:
            self.group_by_entity_calls += 1
            return _SequenceViewStub()

    templated = _TemplatedStub()

    monkeypatch.setattr(
        runner,
        "_prepare_result_paths",
        lambda _bundle, **_kwargs: run_paths,
    )
    monkeypatch.setattr(
        runner,
        "build_dataset_spec",
        lambda *_args, **_kwargs: SimpleNamespace(
            build=lambda: templated,
            clear_cache=lambda: None,
        ),
    )
    monkeypatch.setattr(runner, "_experiment_logger", _logger_context)
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
    assert templated.group_by_entity_calls == expected_apply_calls


def test_run_bundle_replaces_stale_output_directory_without_force(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A stale run directory without metrics should be replaced without force.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces filesystem and orchestration
            helpers so the test can focus on overwrite behaviour.
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
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.log").write_text("stale", encoding="utf-8")
    run_paths = SimpleNamespace(
        run_dir=run_dir,
        metrics_path=tmp_path / "metrics.json",
        run_log_path=tmp_path / "run.log",
        predictions_path=tmp_path / "predictions.jsonl",
    )
    removed_paths: list[Path] = []

    real_rmtree = runner.shutil.rmtree

    def _recording_rmtree(path: Path) -> None:
        removed_paths.append(path)
        real_rmtree(path)

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
    monkeypatch.setattr(runner.shutil, "rmtree", _recording_rmtree)

    monkeypatch.setattr(runner, "load_experiment_bundles", lambda _path: [bundle])

    result = runner.run_experiment(tmp_path / "sweep.toml", force=False)

    assert result == [run_dir]
    assert removed_paths == [run_dir]
    assert run_dir.exists()


def test_run_bundle_rejects_non_directory_result_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A file at the run path should still raise a directory error.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces orchestration helpers.
        tmp_path (Path): Temporary filesystem root used for the fake run.
    """
    bundle = _make_bundle(tmp_path, concrete_name="demo-run")
    run_dir = tmp_path / "run"
    run_dir.write_text("occupied", encoding="utf-8")
    run_paths = SimpleNamespace(
        run_dir=run_dir,
        metrics_path=run_dir / "metrics.json",
        run_log_path=run_dir / "run.log",
        predictions_path=run_dir / "predictions.jsonl",
    )

    monkeypatch.setattr(runner, "prepare_result_paths", lambda _bundle: run_paths)

    with pytest.raises(
        FileExistsError,
        match="Result path exists but is not a directory",
    ):
        runner._run_bundle(  # noqa: SLF001
            bundle,
            options=runner._BundleRunOptions(),  # noqa: SLF001
        )


def test_run_bundle_force_clears_dataset_cache_before_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Force reruns should invalidate the dataset cache before rebuilding.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces orchestration helpers.
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
    run_dir = tmp_path / "run"
    run_paths = SimpleNamespace(
        run_dir=run_dir,
        metrics_path=run_dir / "metrics.json",
        run_log_path=run_dir / "run.log",
        predictions_path=run_dir / "predictions.jsonl",
    )
    clear_cache_calls: list[str] = []

    def _build_dataset_spec(
        _dataset: object,
        *,
        repo_root: Path,
    ) -> SimpleNamespace:
        del repo_root

        def _build() -> SimpleNamespace:
            return SimpleNamespace()

        def _clear_cache() -> None:
            clear_cache_calls.append("demo")

        return SimpleNamespace(clear_cache=_clear_cache, build=_build)

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

    result = runner.run_experiment(tmp_path / "sweep.toml", force=True)

    assert result == [run_dir]
    assert clear_cache_calls == ["demo"]


def test_run_bundle_rerun_keeps_existing_attempts_and_writes_new_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Rerun mode should keep prior artefacts and use a fresh attempt dir.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces orchestration helpers.
        tmp_path (Path): Temporary filesystem root used for the fake run.
    """
    bundle = _make_bundle(tmp_path, concrete_name="demo-run")
    run_root = tmp_path / "run"
    existing_attempt = run_root / "attempt-0001"
    existing_attempt.mkdir(parents=True)
    run_paths = SimpleNamespace(
        run_root=run_root,
        run_dir=run_root / "attempt-0002",
        metrics_path=run_root / "attempt-0002" / "metrics.json",
        run_log_path=run_root / "attempt-0002" / "run.log",
        predictions_path=run_root / "attempt-0002" / "predictions.jsonl",
        dataset_manifest_path=run_root / "attempt-0002" / "dataset_manifest.json",
        config_path=run_root / "attempt-0002" / "experiment_config.json",
        environment_path=run_root / "attempt-0002" / "environment.json",
    )

    class _SequenceViewStub:
        def with_split_fractions(self, *_args: object) -> _SequenceViewStub:
            return self

        @staticmethod
        def split_count_hint() -> SimpleNamespace:
            return SimpleNamespace(train_count=1, test_count=1)

        @staticmethod
        def train_sequence_count_unit_hint() -> str:
            return "sequences"

        def __iter__(self) -> Iterator[SimpleNamespace]:
            yield SimpleNamespace(
                split_label=SimpleNamespace(value="train"),
                label=0,
            )

        def iter_training_sequences(self) -> Iterator[SimpleNamespace]:
            return iter(self)

    class _TemplatedStub:
        @staticmethod
        def group_by_entity() -> _SequenceViewStub:
            return _SequenceViewStub()

    templated = _TemplatedStub()
    removed_paths: list[Path] = []

    real_rmtree = runner.shutil.rmtree

    def _recording_rmtree(path: Path) -> None:
        removed_paths.append(path)
        real_rmtree(path)

    monkeypatch.setattr(
        runner,
        "_prepare_result_paths",
        lambda _bundle, **_kwargs: run_paths,
    )
    monkeypatch.setattr(
        runner,
        "build_dataset_spec",
        lambda *_args, **_kwargs: SimpleNamespace(
            build=lambda: templated,
            clear_cache=lambda: None,
        ),
    )
    monkeypatch.setattr(runner, "_experiment_logger", _logger_context)
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
    monkeypatch.setattr(runner.shutil, "rmtree", _recording_rmtree)

    result = runner._run_bundle(  # noqa: SLF001
        bundle,
        options=runner._BundleRunOptions(rerun=True),  # noqa: SLF001
    )

    assert result == run_paths.run_dir
    assert removed_paths == []
    assert existing_attempt.exists()
    assert run_paths.run_dir.exists()


def test_run_experiment_skips_completed_bundle_and_rebuilds_stale_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Completed bundles should be skipped while stale ones are rebuilt.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patch helper used to stub run
            orchestration.
        tmp_path (Path): Temporary directory used to host the synthetic run
            state.
    """
    sequence_config = _SequenceConfig()
    completed_bundle = SimpleNamespace(
        sweep_path=tmp_path / "completed.toml",
        dataset_path=tmp_path / "dataset.toml",
        model_path=tmp_path / "model-completed.toml",
        sweep=SimpleNamespace(max_workers=1),
        dataset=SimpleNamespace(dataset_name="demo", sequence=sequence_config),
        model=SimpleNamespace(detector="demo", name="demo"),
        concrete_name="completed-run",
        applied_overrides={},
        repo_root=tmp_path,
    )
    stale_bundle = SimpleNamespace(
        sweep_path=tmp_path / "stale.toml",
        dataset_path=tmp_path / "dataset.toml",
        model_path=tmp_path / "model-stale.toml",
        sweep=SimpleNamespace(max_workers=1),
        dataset=SimpleNamespace(dataset_name="demo", sequence=sequence_config),
        model=SimpleNamespace(detector="demo", name="demo"),
        concrete_name="stale-run",
        applied_overrides={},
        repo_root=tmp_path,
    )
    completed_run_dir = tmp_path / "completed"
    completed_run_dir.mkdir()
    completed_metrics_path = completed_run_dir / "metrics.json"
    completed_metrics_path.write_text("{}", encoding="utf-8")
    stale_run_dir = tmp_path / "stale"
    stale_run_dir.mkdir()
    (stale_run_dir / "run.log").write_text("stale", encoding="utf-8")

    run_paths_by_name = {
        "completed-run": SimpleNamespace(
            run_dir=completed_run_dir,
            metrics_path=completed_metrics_path,
            run_log_path=tmp_path / "completed.log",
            predictions_path=tmp_path / "completed.jsonl",
        ),
        "stale-run": SimpleNamespace(
            run_dir=stale_run_dir,
            metrics_path=stale_run_dir / "metrics.json",
            run_log_path=tmp_path / "stale.log",
            predictions_path=tmp_path / "stale.jsonl",
        ),
    }
    removed_paths: list[Path] = []
    run_model_calls: list[str] = []

    class _HasConcreteName(Protocol):
        concrete_name: str

    real_rmtree = runner.shutil.rmtree

    def _recording_rmtree(path: Path) -> None:
        removed_paths.append(path)
        real_rmtree(path)

    def _prepare_result_paths(bundle: _HasConcreteName) -> SimpleNamespace:
        return run_paths_by_name[bundle.concrete_name]

    monkeypatch.setattr(runner, "prepare_result_paths", _prepare_result_paths)
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
        lambda *, sequence_factory, config, **_kwargs: (
            run_model_calls.append(config.name),
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
    monkeypatch.setattr(runner.shutil, "rmtree", _recording_rmtree)
    monkeypatch.setattr(
        runner,
        "load_experiment_bundles",
        lambda _path: [completed_bundle, stale_bundle],
    )

    result = runner.run_experiment(tmp_path / "sweep.toml", force=False)

    assert result == [completed_run_dir, stale_run_dir]
    assert removed_paths == [stale_run_dir]
    assert run_model_calls == ["demo"]
