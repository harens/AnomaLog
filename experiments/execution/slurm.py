"""Submit registry experiments to Slurm."""

from __future__ import annotations

import argparse
import hashlib
import re
import shlex
import subprocess  # noqa: S404
import sys
from dataclasses import dataclass
from pathlib import Path

import msgspec

from experiments import ConfigError
from experiments.config import RegisteredExperiment, load_experiment_registry


@dataclass(frozen=True, slots=True)
class _SlurmDefaults:
    """Default Slurm submission settings."""

    time: str = "24:00:00"
    cpus_per_task: int = 3
    mem: str = "32G"
    gres: str = "gpu:1"
    partition: str | None = None


@dataclass(frozen=True, slots=True)
class SlurmSubmitRequest:
    """Request metadata for a Slurm array submission batch."""

    registry_path: Path = Path("experiments/configs/registry.toml")
    repo_root: Path | None = None
    defaults_path: Path = Path("experiments/configs/slurm.toml")
    groups: tuple[str, ...] = ()
    experiment_names: tuple[str, ...] = ()
    dry_run: bool = False
    force: bool = False
    write_predictions: bool = False
    debug_reporting: bool = False


@dataclass(frozen=True, slots=True)
class _SlurmSubmission:
    experiments: tuple[RegisteredExperiment, ...]
    defaults: _SlurmDefaults
    registry_path: Path
    repo_root: Path
    log_dir: Path
    force: bool
    write_predictions: bool
    debug_reporting: bool


def load_slurm_defaults(
    defaults_path: Path,
    *,
    repo_root: Path,
) -> _SlurmDefaults:
    """Load Slurm defaults from TOML.

    Args:
        defaults_path (Path): TOML file containing the submission defaults.
        repo_root (Path): Repository root used to resolve relative paths.

    Returns:
        _SlurmDefaults: Validated Slurm defaults.

    Raises:
        ConfigError: If the defaults file is missing or malformed.
    """
    resolved_defaults_path = _resolve_path(defaults_path, repo_root)
    try:
        return msgspec.toml.decode(
            resolved_defaults_path.read_bytes(),
            type=_SlurmDefaults,
        )
    except FileNotFoundError as exc:
        msg = f"Missing Slurm defaults file: {resolved_defaults_path}"
        raise ConfigError(msg) from exc
    except (
        msgspec.DecodeError,
        msgspec.ValidationError,
        TypeError,
        ValueError,
    ) as exc:
        msg = f"{resolved_defaults_path}: {exc}"
        raise ConfigError(msg) from exc


def submit_experiments(request: SlurmSubmitRequest) -> list[str]:
    """Submit selected registry experiments as one Slurm array job.

    Args:
        request (SlurmSubmitRequest): Slurm submission settings.

    Returns:
        list[str]: Slurm submission output lines for the array submission.

    Raises:
        ConfigError: If the registry selection is invalid or empty.
    """
    resolved_repo_root = Path.cwd() if request.repo_root is None else request.repo_root
    resolved_repo_root = resolved_repo_root.resolve()
    resolved_registry_path = _resolve_path(request.registry_path, resolved_repo_root)
    resolved_defaults_path = _resolve_path(request.defaults_path, resolved_repo_root)
    registry = load_experiment_registry(
        resolved_registry_path,
        repo_root=resolved_repo_root,
    )
    selected = registry.select(names=request.experiment_names, groups=request.groups)
    if not selected:
        msg = "No registry experiments were selected."
        raise ConfigError(msg)
    defaults = load_slurm_defaults(resolved_defaults_path, repo_root=resolved_repo_root)
    log_dir = (
        resolved_repo_root
        / "experiments"
        / "results"
        / "slurm-logs"
        / _build_submission_label(selected)
    )
    if not request.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    submission = _SlurmSubmission(
        experiments=selected,
        defaults=defaults,
        registry_path=resolved_registry_path,
        repo_root=resolved_repo_root,
        log_dir=log_dir,
        force=request.force,
        write_predictions=request.write_predictions,
        debug_reporting=request.debug_reporting,
    )
    command = build_sbatch_command(submission)
    if request.dry_run:
        _write_line(_format_sbatch_command_preview(command))
        return []
    completed = subprocess.run(  # noqa: S603
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout.strip() or completed.stderr.strip()
    if output:
        _write_line(output)
    outputs: list[str] = []
    outputs.append(output)
    return outputs


def build_sbatch_command(submission: _SlurmSubmission) -> list[str]:
    """Build the `sbatch` command for one registry experiment array.

    Args:
        submission (_SlurmSubmission): Resolved Slurm submission details.

    Returns:
        list[str]: `sbatch` command tokens.
    """
    wrap_script = _build_wrap_script(submission)
    job_name = "anomalog"
    command = [
        "sbatch",
        f"--job-name={job_name}",
        f"--array=0-{len(submission.experiments) - 1}",
        f"--chdir={submission.repo_root.as_posix()}",
        f"--output={submission.log_dir.as_posix()}/%A_%a.out",
        f"--error={submission.log_dir.as_posix()}/%A_%a.err",
        f"--time={submission.defaults.time}",
        f"--cpus-per-task={submission.defaults.cpus_per_task}",
        f"--mem={submission.defaults.mem}",
        f"--gres={submission.defaults.gres}",
    ]
    if submission.defaults.partition is not None:
        command.append(f"--partition={submission.defaults.partition}")
    command.extend(["--wrap", wrap_script])
    return command


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Slurm backend.

    Returns:
        argparse.ArgumentParser: Parser for the Slurm submission CLI.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit registry runs as one Slurm array job.",
    )
    submit_parser.add_argument(
        "--registry",
        type=Path,
        default=Path("experiments/configs/registry.toml"),
        help="Path to the named experiment registry TOML file.",
    )
    submit_parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve registry-relative paths.",
    )
    submit_parser.add_argument(
        "--defaults",
        type=Path,
        default=Path("experiments/configs/slurm.toml"),
        help="Path to the Slurm defaults TOML file.",
    )
    submit_parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Registry group to include. Repeat to include multiple groups.",
    )
    submit_parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Explicit registry experiment name to include. Repeat as needed.",
    )
    submit_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the `sbatch` array command without submitting anything.",
    )
    submit_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace any existing deterministic result directories.",
    )
    submit_parser.add_argument(
        "--write-predictions",
        action="store_true",
        help="Write predictions.jsonl for each run.",
    )
    submit_parser.add_argument(
        "--debug-reporting",
        action="store_true",
        help="Keep verbose diagnostics in the run artefacts.",
    )
    return parser


def main() -> int:
    """Run the Slurm backend CLI.

    Returns:
        int: Process exit status.

    Raises:
        ConfigError: If the user selects an unsupported subcommand.
        SystemExit: Raised with a non-zero exit status on invalid input.
    """
    args = build_arg_parser().parse_args()
    if args.command != "submit":
        msg = f"Unsupported Slurm subcommand: {args.command!r}"
        raise ConfigError(msg)
    try:
        submit_experiments(
            SlurmSubmitRequest(
                registry_path=args.registry,
                repo_root=args.repo_root,
                defaults_path=args.defaults,
                groups=tuple(args.group),
                experiment_names=tuple(args.experiment),
                dry_run=args.dry_run,
                force=args.force,
                write_predictions=args.write_predictions,
                debug_reporting=args.debug_reporting,
            ),
        )
    except (ConfigError, FileExistsError, ValueError) as exc:
        message = f"{exc}\n"
        raise SystemExit(message) from exc
    return 0


def _build_wrap_script(submission: _SlurmSubmission) -> str:
    """Build the shell wrapper executed by `sbatch`.

    Args:
        submission (_SlurmSubmission): Resolved Slurm submission details.

    Returns:
        str: Shell script executed by `sbatch`.
    """
    wrapped_command = _build_run_command(submission)
    experiment_array = "\n".join(
        f"  {shlex.quote(experiment.name)}" for experiment in submission.experiments
    )
    return "\n".join(
        [
            'export PATH="${HOME}/.local/bin:${PATH}"',
            f"export REPO_ROOT={shlex.quote(submission.repo_root.as_posix())}",
            "set -euo pipefail",
            "EXPERIMENTS=(",
            experiment_array,
            ")",
            'EXPERIMENT_NAME="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]:-}"',
            'if [ -z "$EXPERIMENT_NAME" ]; then',
            (
                "  printf 'Missing experiment entry for "
                'SLURM_ARRAY_TASK_ID=%s\\n\' "$SLURM_ARRAY_TASK_ID" >&2'
            ),
            "  exit 1",
            "fi",
            'export RUN_NAME="$EXPERIMENT_NAME"',
            'export PREFECT_ROOT="${PREFECT_ROOT:-${REPO_ROOT}/prefect}"',
            'export PREFECT_HOME="${PREFECT_ROOT}/${RUN_NAME}"',
            'export PREFECT_LOCAL_STORAGE_PATH="${PREFECT_ROOT}/storage"',
            (
                'export PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS="${'
                'PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS:-120}"'
            ),
            (
                'export UV_CACHE_DIR="${UV_CACHE_DIR:-'
                '${SLURM_TMPDIR:-${REPO_ROOT}/.cache/uv}}"'
            ),
            (
                "export PREFECT_API_DATABASE_CONNECTION_URL="
                '"sqlite+aiosqlite:///${PREFECT_HOME}/prefect.db"'
            ),
            'mkdir -p "$PREFECT_HOME" "$PREFECT_LOCAL_STORAGE_PATH" "$UV_CACHE_DIR"',
            'cd "$REPO_ROOT"',
            wrapped_command,
        ],
    )


def _build_run_command(submission: _SlurmSubmission) -> str:
    command = [
        shlex.quote("uv"),
        shlex.quote("run"),
        shlex.quote("python"),
        shlex.quote("-m"),
        shlex.quote("experiments.runners.run_experiment"),
        shlex.quote("--experiment"),
        '"$EXPERIMENT_NAME"',
        shlex.quote("--registry"),
        shlex.quote(submission.registry_path.as_posix()),
        shlex.quote("--repo-root"),
        shlex.quote(submission.repo_root.as_posix()),
    ]
    if submission.force:
        command.append(shlex.quote("--force"))
    if submission.write_predictions:
        command.append(shlex.quote("--write-predictions"))
    if submission.debug_reporting:
        command.append(shlex.quote("--debug-reporting"))
    return f"bash -lc {shlex.quote(' '.join(command))}"


def _build_submission_label(experiments: tuple[RegisteredExperiment, ...]) -> str:
    digest_source = "\n".join(experiment.name for experiment in experiments)
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "-", experiments[0].name).strip("-")
    if not prefix:
        prefix = "slurm-array"
    return f"{prefix}-x{len(experiments)}-{digest[:8]}"


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def _format_sbatch_command_preview(command: list[str]) -> str:
    """Return a readable multi-line preview of one `sbatch` invocation."""
    if "--wrap" not in command:
        return f"sbatch command:\n  {shlex.join(command)}"
    wrap_index = command.index("--wrap")
    prefix_tokens = command[:wrap_index]
    wrap_script = command[wrap_index + 1]
    lines = ["sbatch command:"]
    lines.extend(f"  {shlex.quote(token)}" for token in prefix_tokens)
    lines.append("  --wrap <<'EOF'")
    lines.extend(f"  {line}" for line in wrap_script.splitlines())
    lines.append("  EOF")
    return "\n".join(lines)


def _write_line(message: str) -> None:
    sys.stdout.write(message + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
