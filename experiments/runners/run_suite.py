"""Run a selected registry suite locally."""

from __future__ import annotations

import argparse
import shlex
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from experiments import ConfigError
from experiments.config import (
    ExperimentBundle,
    ExperimentRegistry,
    RegisteredExperiment,
    load_experiment_registry,
)
from experiments.results import prepare_result_paths
from experiments.runners.run_experiment import (
    RegisteredExperimentRunRequest,
    run_registered_experiment,
)


@dataclass(frozen=True, slots=True)
class SuiteRunRequest:
    """Request metadata for a local experiment suite run.

    Attributes:
        registry_path (Path): Path to the registry TOML file.
        repo_root (Path | None): Repository root used to resolve relative
            paths.
        groups (tuple[str, ...]): Registry groups to include in the suite.
        experiment_names (tuple[str, ...]): Explicit registry experiment names
            to include.
        dry_run (bool): Whether to print the resolved command lines only.
        list_only (bool): Whether to print the selected registry entries only.
        check_missing (bool): Whether to report selected concrete runs that do
            not yet have a completed `metrics.json`.
        max_parallel (int): Maximum number of experiments to run concurrently.
        force (bool): Whether to replace existing deterministic run outputs.
        write_predictions (bool): Whether to persist `predictions.jsonl`.
        debug_reporting (bool): Whether to keep verbose diagnostics.
    """

    registry_path: Path = Path("experiments/configs/registry.toml")
    repo_root: Path | None = None
    groups: tuple[str, ...] = ()
    experiment_names: tuple[str, ...] = ()
    dry_run: bool = False
    list_only: bool = False
    check_missing: bool = False
    max_parallel: int = 1
    force: bool = False
    write_predictions: bool = False
    debug_reporting: bool = False


@dataclass(frozen=True, slots=True)
class _RunStatus:
    """Concrete run status derived from a resolved registry experiment.

    Attributes:
        experiment (RegisteredExperiment): Logical registry entry owning the
            concrete bundle.
        bundle (ExperimentBundle): Resolved concrete bundle under inspection.
        run_dir (Path): Expected deterministic output directory.
        metrics_path (Path): Expected metrics output file path.
        missing_reason (str | None): Why the bundle is considered incomplete,
            or ``None`` when the run looks complete.
    """

    experiment: RegisteredExperiment
    bundle: ExperimentBundle
    run_dir: Path
    metrics_path: Path
    missing_reason: str | None


def run_suite(request: SuiteRunRequest) -> list[Path]:
    """Run a curated subset of registry experiments locally.

    Args:
        request (SuiteRunRequest): Suite selection and execution settings.

    Returns:
        list[Path]: Result directories for the selected experiments.

    Raises:
        ConfigError: If the selection is invalid or no experiments match the
            requested filters.
    """
    resolved_repo_root = Path.cwd() if request.repo_root is None else request.repo_root
    registry = load_experiment_registry(
        request.registry_path,
        repo_root=resolved_repo_root,
    )
    selected = registry.select(names=request.experiment_names, groups=request.groups)
    if request.check_missing:
        _report_missing_runs(
            registry=registry,
            selected=selected,
            registry_path=request.registry_path,
            repo_root=resolved_repo_root,
        )
        return []
    if request.list_only:
        for experiment in selected:
            _write_line(_format_experiment_listing(experiment))
        return []
    if request.dry_run:
        for experiment in selected:
            _write_line(
                shlex.join(
                    _build_experiment_command(
                        experiment.name,
                        request=request,
                        repo_root=resolved_repo_root,
                    ),
                ),
            )
        return []
    if request.max_parallel < 1:
        msg = "--max-parallel must be at least 1."
        raise ConfigError(msg)
    if request.max_parallel == 1 or len(selected) <= 1:
        return _run_sequential_suite(request=request, selected=selected)
    return _run_parallel_suite(request=request, selected=selected)


def _run_sequential_suite(
    *,
    request: SuiteRunRequest,
    selected: tuple[RegisteredExperiment, ...],
) -> list[Path]:
    results: list[Path] = []
    for experiment in selected:
        _write_line(_format_run_start(experiment))
        result_paths = run_registered_experiment(
            RegisteredExperimentRunRequest(
                experiment_name=experiment.name,
                registry_path=request.registry_path,
                repo_root=request.repo_root,
                force=request.force,
                write_predictions=request.write_predictions,
                debug_reporting=request.debug_reporting,
                console=False,
            ),
        )
        _write_line(_format_run_finish(experiment, result_paths))
        results.extend(result_paths)
    return results


def _run_parallel_suite(
    *,
    request: SuiteRunRequest,
    selected: tuple[RegisteredExperiment, ...],
) -> list[Path]:
    requests_by_name = {
        experiment.name: RegisteredExperimentRunRequest(
            experiment_name=experiment.name,
            registry_path=request.registry_path,
            repo_root=request.repo_root,
            force=request.force,
            write_predictions=request.write_predictions,
            debug_reporting=request.debug_reporting,
            console=False,
        )
        for experiment in selected
    }
    results_by_name: dict[str, list[Path]] = {}
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=request.max_parallel) as executor:
        future_to_name = {
            executor.submit(run_registered_experiment, run_request): experiment_name
            for experiment_name, run_request in requests_by_name.items()
        }
        for future in as_completed(future_to_name):
            experiment_name = future_to_name[future]
            try:
                result_paths = future.result()
                results_by_name[experiment_name] = result_paths
                _write_line(
                    f"[suite] finished {experiment_name} -> "
                    f"{', '.join(str(path) for path in result_paths)}",
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{experiment_name}: {exc}")

    if failures:
        msg = "One or more suite experiments failed:\n" + "\n".join(
            f"- {failure}" for failure in failures
        )
        raise ConfigError(msg)

    return [
        result_path
        for experiment in selected
        for result_path in results_by_name[experiment.name]
    ]


def _build_experiment_command(
    experiment_name: str,
    *,
    request: SuiteRunRequest,
    repo_root: Path,
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "experiments.runners.run_experiment",
        "--experiment",
        experiment_name,
        "--registry",
        request.registry_path.as_posix(),
        "--repo-root",
        repo_root.as_posix(),
    ]
    if request.force:
        command.append("--force")
    if request.write_predictions:
        command.append("--write-predictions")
    if request.debug_reporting:
        command.append("--debug-reporting")
    return command


def _format_experiment_listing(experiment: RegisteredExperiment) -> str:
    return (
        f"{experiment.name}\t"
        f"dataset={experiment.dataset}\t"
        f"preset={experiment.preset or '-'}\t"
        f"models={','.join(experiment.model_sets)}\t"
        f"groups={','.join(experiment.groups)}"
    )


def _format_run_start(experiment: RegisteredExperiment) -> str:
    return f"[suite] starting {experiment.name}"


def _format_run_finish(
    experiment: RegisteredExperiment,
    result_paths: list[Path],
) -> str:
    return (
        f"[suite] finished {experiment.name} -> "
        f"{', '.join(str(path) for path in result_paths)}"
    )


def _report_missing_runs(
    *,
    registry: ExperimentRegistry,
    selected: tuple[RegisteredExperiment, ...],
    registry_path: Path,
    repo_root: Path,
) -> None:
    statuses = _collect_run_statuses(
        registry=registry,
        selected=selected,
        registry_path=registry_path,
        repo_root=repo_root,
    )
    missing_statuses = [
        status for status in statuses if status.missing_reason is not None
    ]
    for status in missing_statuses:
        for line in _format_missing_run(
            status,
            registry_path=registry_path,
            repo_root=repo_root,
        ):
            _write_line(line)
    total = len(statuses)
    completed = total - len(missing_statuses)
    missing = len(missing_statuses)
    _write_line(
        f"Summary: total={total} completed={completed} missing={missing}",
    )


def _collect_run_statuses(
    *,
    registry: ExperimentRegistry,
    selected: tuple[RegisteredExperiment, ...],
    registry_path: Path,
    repo_root: Path,
) -> list[_RunStatus]:
    statuses: list[_RunStatus] = []
    for experiment in selected:
        resolved = registry.resolve_experiment(
            experiment.name,
            registry_path=registry_path,
            repo_root=repo_root,
        )
        for bundle in resolved.bundles:
            result_paths = prepare_result_paths(bundle)
            run_dir_exists = result_paths.run_dir.is_dir()
            metrics_exists = result_paths.metrics_path.is_file()
            missing_reason = None
            if not run_dir_exists:
                missing_reason = "missing output directory"
            elif not metrics_exists:
                missing_reason = "missing metrics.json"
            statuses.append(
                _RunStatus(
                    experiment=experiment,
                    bundle=bundle,
                    run_dir=result_paths.run_dir,
                    metrics_path=result_paths.metrics_path,
                    missing_reason=missing_reason,
                ),
            )
    return statuses


def _format_missing_run(
    status: _RunStatus,
    *,
    registry_path: Path,
    repo_root: Path,
) -> list[str]:
    command = _build_rerun_command(
        status.experiment.name,
        registry_path=registry_path,
        repo_root=repo_root,
    )
    return [
        f"- {status.experiment.name} [{status.bundle.concrete_name}]",
        f"  dataset: {status.bundle.dataset_path.as_posix()}",
        f"  model: {status.bundle.model_path.as_posix()}",
        f"  output: {status.run_dir.as_posix()}",
        f"  status: {status.missing_reason}",
        f"  rerun: {shlex.join(command)}",
    ]


def _build_rerun_command(
    experiment_name: str,
    *,
    registry_path: Path,
    repo_root: Path,
) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "-m",
        "experiments.runners.run_experiment",
        "--experiment",
        experiment_name,
        "--registry",
        registry_path.as_posix(),
        "--repo-root",
        repo_root.as_posix(),
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for local suite execution.

    Returns:
        argparse.ArgumentParser: Parser for the local suite CLI.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("experiments/configs/registry.toml"),
        help="Path to the named experiment registry TOML file.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve registry-relative paths.",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Registry group to include. Repeat to include multiple groups.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Explicit registry experiment name to include. Repeat as needed.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum number of experiments to execute concurrently.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved commands without executing anything.",
    )
    parser.add_argument(
        "--list",
        dest="list_only",
        action="store_true",
        help="List the selected experiments without executing them.",
    )
    parser.add_argument(
        "--check-missing",
        action="store_true",
        help=(
            "Report selected concrete runs whose output directory is missing "
            "or does not contain metrics.json."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace any existing deterministic result directories.",
    )
    parser.add_argument(
        "--write-predictions",
        action="store_true",
        help="Write predictions.jsonl for each run.",
    )
    parser.add_argument(
        "--debug-reporting",
        action="store_true",
        help="Keep verbose diagnostics in the run artefacts.",
    )
    return parser


def main() -> int:
    """Run the local suite CLI.

    Returns:
        int: Process exit status.

    Raises:
        SystemExit: Raised with a non-zero exit status on invalid input.
    """
    args = build_arg_parser().parse_args()
    try:
        run_suite(
            SuiteRunRequest(
                registry_path=args.registry,
                repo_root=args.repo_root,
                groups=tuple(args.group),
                experiment_names=tuple(args.experiment),
                dry_run=args.dry_run,
                list_only=args.list_only,
                check_missing=args.check_missing,
                max_parallel=args.max_parallel,
                force=args.force,
                write_predictions=args.write_predictions,
                debug_reporting=args.debug_reporting,
            ),
        )
    except (ConfigError, FileExistsError, ValueError) as exc:
        message = f"{exc}\n"
        raise SystemExit(message) from exc
    return 0


def _write_line(message: str) -> None:
    sys.stdout.write(message + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
