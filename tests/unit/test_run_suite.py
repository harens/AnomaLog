"""Tests for the local registry suite runner."""

from __future__ import annotations

import shlex
from concurrent.futures import Future
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol

import pytest
from typing_extensions import Self

from experiments import ConfigError
from experiments.config import load_experiment_registry
from experiments.results import prepare_result_paths
from experiments.runners.run_suite import (
    SuiteRunRequest,
    run_suite,
)
from experiments.runners.run_suite import (
    build_arg_parser as run_suite_build_arg_parser,
)
from experiments.runners.run_suite import (
    main as run_suite_main,
)

if TYPE_CHECKING:
    from pathlib import Path


class _ExperimentRunRequest(Protocol):
    """Minimal protocol for the fake executor request object.

    Attributes:
        experiment_name (str): Experiment name used by the fake executor to
            choose the synthetic result payload.
    """

    experiment_name: str


def _write_suite_registry_tree(tmp_path: Path) -> Path:
    experiments_root = tmp_path / "experiments"
    datasets_dir = experiments_root / "configs" / "datasets"
    models_dir = experiments_root / "configs" / "models"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (datasets_dir / "demo.toml").write_text(
        (
            'name = "demo"\n'
            'dataset_name = "demo"\n'
            "\n[dataset]\n"
            'name = "demo"\n'
            'dataset_name = "demo"\n'
            'structured_parser = "bgl"\n'
            "\n[dataset.source]\n"
            'type = "local_dir"\n'
            'path = "."\n'
            "\n[dataset.sequence]\n"
            'grouping = "fixed"\n'
            "window_size = 3\n"
            "step = 2\n"
        ),
        encoding="utf-8",
    )
    (models_dir / "template_frequency_default.toml").write_text(
        'name = "template_frequency_default"\ndetector = "template_frequency"\n',
        encoding="utf-8",
    )
    (models_dir / "naive_bayes_default.toml").write_text(
        'name = "naive_bayes_default"\ndetector = "naive_bayes"\n',
        encoding="utf-8",
    )
    (models_dir / "markov_default.toml").write_text(
        'name = "markov_default"\ndetector = "markov"\n',
        encoding="utf-8",
    )
    (models_dir / "deepcase.toml").write_text(
        'name = "deepcase"\ndetector = "deepcase"\n',
        encoding="utf-8",
    )
    registry_path = experiments_root / "configs" / "registry.toml"
    registry_path.write_text(
        (
            "[model_sets.baselines_with_nb]\n"
            'models = ["template_frequency_default", '
            '"naive_bayes_default", "markov_default"]\n'
            "\n"
            "[experiments.demo]\n"
            'dataset = "demo"\n'
            'model_sets = ["baselines_with_nb"]\n'
            'models = ["deepcase"]\n'
            "\n"
            "[experiments.demo_two]\n"
            'dataset = "demo"\n'
            'model_sets = ["baselines_with_nb"]\n'
            'models = ["deepcase"]\n'
        ),
        encoding="utf-8",
    )
    return registry_path


def _write_missing_runs_suite_registry_tree(tmp_path: Path) -> Path:
    experiments_root = tmp_path / "experiments"
    datasets_dir = experiments_root / "configs" / "datasets"
    models_dir = experiments_root / "configs" / "models"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (datasets_dir / "demo.toml").write_text(
        (
            'name = "demo"\n'
            "\n[dataset]\n"
            'name = "demo"\n'
            'dataset_name = "demo"\n'
            'structured_parser = "bgl"\n'
            "\n[dataset.source]\n"
            'type = "local_dir"\n'
            'path = "."\n'
            "\n[dataset.sequence]\n"
            'grouping = "fixed"\n'
            "window_size = 3\n"
            "step = 2\n"
        ),
        encoding="utf-8",
    )
    (models_dir / "template_frequency_default.toml").write_text(
        'name = "template_frequency_default"\ndetector = "template_frequency"\n',
        encoding="utf-8",
    )
    (models_dir / "markov_default.toml").write_text(
        'name = "markov_default"\ndetector = "markov"\n',
        encoding="utf-8",
    )
    (models_dir / "deepcase.toml").write_text(
        'name = "deepcase"\ndetector = "deepcase"\n',
        encoding="utf-8",
    )
    registry_path = experiments_root / "configs" / "registry.toml"
    registry_path.write_text(
        (
            "[model_sets.complete]\n"
            'models = ["template_frequency_default"]\n'
            "\n"
            "[model_sets.missing_dir]\n"
            'models = ["markov_default"]\n'
            "\n"
            "[model_sets.missing_metrics]\n"
            'models = ["deepcase"]\n'
            "\n"
            "[experiments.complete]\n"
            'dataset = "demo"\n'
            'model_sets = ["complete"]\n'
            "\n"
            "[experiments.missing_dir]\n"
            'dataset = "demo"\n'
            'model_sets = ["missing_dir"]\n'
            "\n"
            "[experiments.missing_metrics]\n"
            'dataset = "demo"\n'
            'model_sets = ["missing_metrics"]\n'
        ),
        encoding="utf-8",
    )
    return registry_path


def test_run_suite_list_only_filters_by_group(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The suite runner should list only the selected group.

    Args:
        tmp_path (Path): Temporary directory used to build the registry tree.
        capsys (pytest.CaptureFixture[str]): Pytest capture fixture used to
            inspect the printed listing.
    """
    registry_path = _write_suite_registry_tree(tmp_path)

    run_suite(
        SuiteRunRequest(
            registry_path=registry_path,
            repo_root=tmp_path,
            groups=("baselines_with_nb",),
            list_only=True,
        ),
    )

    output = capsys.readouterr().out.strip().splitlines()
    assert output == [
        (
            "demo\tdataset=demo\tmodel_sets=baselines_with_nb\t"
            "models=template_frequency_default,naive_bayes_default,"
            "markov_default,deepcase\tgroups=demo,baselines_with_nb"
        ),
        (
            "demo_two\tdataset=demo\tmodel_sets=baselines_with_nb\t"
            "models=template_frequency_default,naive_bayes_default,"
            "markov_default,deepcase\tgroups=demo_two,baselines_with_nb"
        ),
    ]


def test_run_suite_dry_run_prints_command_for_selected_experiment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dry-run mode should print the resolved local command line.

    Args:
        tmp_path (Path): Temporary directory used to build the registry tree.
        capsys (pytest.CaptureFixture[str]): Pytest capture fixture used to
            inspect the printed command line.
    """
    registry_path = _write_suite_registry_tree(tmp_path)

    run_suite(
        SuiteRunRequest(
            registry_path=registry_path,
            repo_root=tmp_path,
            experiment_names=("demo",),
            dry_run=True,
            rerun=True,
        ),
    )

    output = capsys.readouterr().out.strip()
    assert "--experiment demo" in output
    assert "--rerun" in output
    assert "--registry" in output
    assert "experiments.runners.run_experiment" in output


def test_run_suite_check_missing_reports_incomplete_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The missing-run check should group missing bundles by registry experiment.

    Args:
        tmp_path (Path): Temporary directory used to build the registry tree.
        capsys (pytest.CaptureFixture[str]): Pytest capture fixture used to
            inspect the printed status report.
        monkeypatch (pytest.MonkeyPatch): Replaces the execution hook so the
            check cannot accidentally run experiments.
    """
    registry_path = _write_missing_runs_suite_registry_tree(tmp_path)
    registry = load_experiment_registry(registry_path, repo_root=tmp_path)

    complete_bundle = registry.resolve_experiment(
        "complete",
        registry_path=registry_path,
        repo_root=tmp_path,
    ).bundle
    complete_paths = prepare_result_paths(complete_bundle)
    complete_paths.run_dir.mkdir(parents=True, exist_ok=True)
    complete_paths.metrics_path.write_text("{}", encoding="utf-8")

    missing_dir_bundle = registry.resolve_experiment(
        "missing_dir",
        registry_path=registry_path,
        repo_root=tmp_path,
    ).bundle
    missing_metrics_bundle = registry.resolve_experiment(
        "missing_metrics",
        registry_path=registry_path,
        repo_root=tmp_path,
    ).bundle
    missing_metrics_paths = prepare_result_paths(missing_metrics_bundle)
    missing_metrics_paths.run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "experiments.runners.run_suite.run_registered_experiment",
        lambda *_args, **_kwargs: pytest.fail(
            "missing-run check should not execute runs",
        ),
    )

    run_suite(
        SuiteRunRequest(
            registry_path=registry_path,
            repo_root=tmp_path,
            check_missing=True,
        ),
    )

    output = capsys.readouterr().out.strip().splitlines()
    assert output == [
        "- missing_dir",
        (
            "  rerun: "
            + shlex.join(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "experiments.runners.run_experiment",
                    "--experiment",
                    "missing_dir",
                    "--registry",
                    registry_path.as_posix(),
                    "--repo-root",
                    tmp_path.as_posix(),
                ],
            )
        ),
        f"  - [{missing_dir_bundle.concrete_name}]",
        f"    dataset: {missing_dir_bundle.dataset_path.as_posix()}",
        f"    model: {missing_dir_bundle.model_path.as_posix()}",
        f"    output: {prepare_result_paths(missing_dir_bundle).run_dir.as_posix()}",
        "    status: missing output directory",
        "- missing_metrics",
        (
            "  rerun: "
            + shlex.join(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "experiments.runners.run_experiment",
                    "--experiment",
                    "missing_metrics",
                    "--registry",
                    registry_path.as_posix(),
                    "--repo-root",
                    tmp_path.as_posix(),
                ],
            )
        ),
        f"  - [{missing_metrics_bundle.concrete_name}]",
        f"    dataset: {missing_metrics_bundle.dataset_path.as_posix()}",
        f"    model: {missing_metrics_bundle.model_path.as_posix()}",
        f"    output: {missing_metrics_paths.run_dir.as_posix()}",
        "    status: missing metrics.json",
        "Summary: total=3 completed=1 missing=2",
    ]


def test_run_suite_sequential_and_parallel_branches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sequential and parallel execution paths should report results cleanly.

    Args:
        tmp_path (Path): Temporary directory used to build the registry tree.
        capsys (pytest.CaptureFixture[str]): Pytest capture fixture used to
            inspect the emitted suite output.
        monkeypatch (pytest.MonkeyPatch): Patch helper used to replace the
            suite execution hooks.
    """
    registry_path = _write_suite_registry_tree(tmp_path)

    monkeypatch.setattr(
        "experiments.runners.run_suite.run_registered_experiment",
        lambda run_request: [tmp_path / f"{run_request.experiment_name}.json"],
    )

    sequential_results = run_suite(
        SuiteRunRequest(
            registry_path=registry_path,
            repo_root=tmp_path,
            experiment_names=("demo",),
            force=True,
            write_predictions=True,
        ),
    )
    assert sequential_results == [tmp_path / "demo.json"]
    sequential_output = capsys.readouterr().out.strip().splitlines()
    assert sequential_output[0] == "[suite] starting demo"
    assert sequential_output[-1] == f"[suite] finished demo -> {tmp_path / 'demo.json'}"

    class _FakeExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: object,
            exc: object,
            tb: object,
        ) -> None:
            del exc_type, exc, tb

        def submit(
            self,
            fn: object,
            run_request: _ExperimentRunRequest,
        ) -> Future[list[Path]]:
            del fn
            expected_max_workers = 2
            assert self.max_workers == expected_max_workers
            future: Future[list[Path]] = Future()
            if run_request.experiment_name == "demo":
                future.set_result([tmp_path / "demo.json"])
            else:
                future.set_result([tmp_path / "demo_two.json"])
            return future

    monkeypatch.setattr(
        "experiments.runners.run_suite.ProcessPoolExecutor",
        _FakeExecutor,
    )

    parallel_results = run_suite(
        SuiteRunRequest(
            registry_path=registry_path,
            repo_root=tmp_path,
            max_parallel=2,
            force=True,
            write_predictions=True,
            experiment_names=("demo", "demo_two"),
        ),
    )
    assert parallel_results == [tmp_path / "demo.json", tmp_path / "demo_two.json"]
    parallel_output = capsys.readouterr().out.strip()
    assert "[suite] finished demo ->" in parallel_output
    assert "[suite] finished demo_two ->" in parallel_output


def test_run_suite_build_arg_parser_exposes_cli_options() -> None:
    """The suite parser should publish all documented CLI options."""
    parser = run_suite_build_arg_parser()
    help_text = parser.format_help()

    assert "--registry" in help_text
    assert "--repo-root" in help_text
    assert "--group" in help_text
    assert "--experiment" in help_text
    assert "--max-parallel" in help_text
    assert "--dry-run" in help_text
    assert "--list" in help_text
    assert "--check-missing" in help_text
    assert "--force" in help_text
    assert "--rerun" in help_text
    assert "--write-predictions" in help_text
    assert "--debug-reporting" in help_text


def test_run_suite_main_surfaces_config_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI wrapper should convert config failures into SystemExit.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patch helper used to stub the parser
            and runner entry points.
    """

    class _Parser:
        @staticmethod
        def parse_args() -> SimpleNamespace:
            return SimpleNamespace(
                registry="registry.toml",
                repo_root=".",
                group=[],
                experiment=[],
                max_parallel=1,
                dry_run=False,
                list_only=False,
                check_missing=False,
                force=False,
                rerun=False,
                write_predictions=False,
                debug_reporting=False,
            )

    def _build_arg_parser() -> _Parser:
        return _Parser()

    def _run_suite(request: object) -> None:
        del request
        message = "boom"
        raise ConfigError(message)

    monkeypatch.setattr(
        "experiments.runners.run_suite.build_arg_parser",
        _build_arg_parser,
    )
    monkeypatch.setattr(
        "experiments.runners.run_suite.run_suite",
        _run_suite,
    )

    with pytest.raises(SystemExit, match="boom"):
        run_suite_main()
