"""Render Slurm wrappers for experiment sweep configs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec

from experiments import ConfigError


class SlurmJobConfig(msgspec.Struct, frozen=True):
    """Configuration for one generated Slurm wrapper.

    Attributes:
        sweep (Path): Sweep config path relative to the repository root.
        force (bool): Whether the generated wrapper should pass `--force`.
        time (str): Slurm wall-clock limit.
        cpus_per_task (int): CPU reservation for the job.
        mem (str): Memory reservation for the job.
        gres (str): Generic resources request, typically the GPU allocation.
    """

    sweep: Path
    force: bool = True
    time: str = "24:00:00"
    cpus_per_task: int = 3
    mem: str = "32G"
    gres: str = "gpu:1"


class SlurmJobsManifest(msgspec.Struct, frozen=True):
    """Manifest listing the Slurm sweep configs that should be wrapped.

    Attributes:
        sweeps (list[Path]): Ordered sweep config paths to render.
    """

    sweeps: list[Path]


@dataclass(frozen=True, slots=True)
class RenderedSlurmJob:
    """A rendered Slurm wrapper and its destination filename.

    Attributes:
        path (Path): Relative destination filename for the generated script.
        content (str): Rendered script body.
    """

    path: Path
    content: str


def load_slurm_jobs(manifest_path: Path, *, repo_root: Path) -> list[SlurmJobConfig]:
    """Load the Slurm job manifest.

    Args:
        manifest_path (Path): TOML manifest describing the generated wrappers.
        repo_root (Path): Repository root used to resolve relative sweep paths.

    Returns:
        list[SlurmJobConfig]: Manifest entries in their declared order.

    Raises:
        ConfigError: If the manifest cannot be decoded, references a missing
            sweep config, or repeats the same sweep stem more than once.
    """
    resolved_manifest_path = _resolve_path(manifest_path, repo_root)
    try:
        manifest = msgspec.toml.decode(
            resolved_manifest_path.read_bytes(),
            type=SlurmJobsManifest,
            dec_hook=_path_dec_hook,
        )
    except (
        msgspec.DecodeError,
        msgspec.ValidationError,
        TypeError,
        ValueError,
    ) as exc:
        msg = f"{resolved_manifest_path}: {exc}"
        raise ConfigError(msg) from exc
    jobs = []
    for sweep in manifest.sweeps:
        sweep_path = _resolve_path(sweep, repo_root)
        if not sweep_path.exists():
            msg = f"Missing sweep config for Slurm job: {sweep_path}"
            raise ConfigError(msg)
        jobs.append(SlurmJobConfig(sweep=sweep))
    _validate_unique_stems(jobs)
    return jobs


def render_slurm_script(
    job: SlurmJobConfig,
    *,
    repo_root: Path,
    manifest_path: Path,
) -> RenderedSlurmJob:
    """Render one Slurm wrapper from a job specification.

    Args:
        job (SlurmJobConfig): Manifest entry to render.
        repo_root (Path): Repository root used to resolve the sweep config path.
        manifest_path (Path): Manifest path used in the generated header comment.

    Returns:
        RenderedSlurmJob: Destination path and rendered script content.
    """
    sweep_path = _resolve_path(job.sweep, repo_root)
    sweep_relpath = _repo_relative_path(sweep_path, repo_root)
    script_name = f"{sweep_path.stem}.sbatch"
    run_name = sweep_path.stem
    force_flag = " \\\n  --force" if job.force else ""
    content = "\n".join(
        [
            "#!/bin/bash",
            (
                "# Generated from "
                f"{manifest_path.as_posix()}; edit that manifest and regenerate."
            ),
            f"#SBATCH --job-name={run_name}",
            f"#SBATCH --output=slurm-{run_name}-%j.out",
            f"#SBATCH --error=slurm-{run_name}-%j.err",
            f"#SBATCH --time={job.time}",
            f"#SBATCH --cpus-per-task={job.cpus_per_task}",
            f"#SBATCH --mem={job.mem}",
            f"#SBATCH --gres={job.gres}",
            "",
            'export PATH="${HOME}/.local/bin:${PATH}"',
            f'export RUN_NAME="{run_name}"',
            "",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"',
            "",
            'export PREFECT_ROOT="${PREFECT_ROOT:-${REPO_ROOT}/prefect}"',
            'export PREFECT_HOME="${PREFECT_ROOT}/${RUN_NAME}"',
            'export PREFECT_LOCAL_STORAGE_PATH="${PREFECT_HOME}/storage"',
            (
                "export PREFECT_API_DATABASE_CONNECTION_URL="
                '"sqlite+aiosqlite:///${PREFECT_HOME}/prefect.db"'
            ),
            "# export PREFECT_TASKS_REFRESH_CACHE=true",
            "",
            'mkdir -p "$PREFECT_HOME" "$PREFECT_LOCAL_STORAGE_PATH"',
            "",
            'cd "$REPO_ROOT"',
            "",
            "uv run python -m experiments.runners.run_experiment \\",
            f'  --config "{sweep_relpath}"{force_flag}',
            "",
        ],
    )
    return RenderedSlurmJob(path=Path(script_name), content=content)


def write_slurm_scripts(
    manifest_path: Path,
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> list[Path]:
    """Write the generated Slurm wrappers to disk.

    Args:
        manifest_path (Path): Manifest describing the wrappers to generate.
        repo_root (Path): Repository root used to resolve sweep paths.
        output_dir (Path | None): Directory that should receive the generated
            `.sbatch` files. Defaults to the manifest directory.

    Returns:
        list[Path]: Paths written to disk.
    """
    resolved_manifest_path = _resolve_path(manifest_path, repo_root)
    jobs = load_slurm_jobs(resolved_manifest_path, repo_root=repo_root)
    destination_root = (
        _resolve_path(output_dir, repo_root)
        if output_dir is not None
        else resolved_manifest_path.parent
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for job in jobs:
        rendered = render_slurm_script(
            job,
            repo_root=repo_root,
            manifest_path=manifest_path,
        )
        destination_path = destination_root / rendered.path
        destination_path.write_text(rendered.content, encoding="utf-8")
        written_paths.append(destination_path)
    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Slurm wrapper generation.

    Returns:
        argparse.ArgumentParser: Parser for the Slurm wrapper generator CLI.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("slurm/jobs.toml"),
        help="TOML manifest describing the generated wrappers.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve sweep paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory that should receive the generated wrappers.",
    )
    return parser


def main() -> int:
    """Render the checked-in Slurm wrappers from their manifest.

    Returns:
        int: Process exit status.
    """
    args = build_arg_parser().parse_args()
    write_slurm_scripts(
        args.manifest,
        repo_root=args.repo_root,
        output_dir=args.output_dir,
    )
    return 0


def _path_dec_hook(type_: type[Any], obj: object) -> object:
    if type_ is not Path or not isinstance(obj, str):
        msg = f"Unsupported decoded type: {type_!r}"
        raise NotImplementedError(msg)
    return Path(obj)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def _repo_relative_path(path: Path, repo_root: Path) -> str:
    resolved_repo_root = repo_root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_repo_root).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def _validate_unique_stems(jobs: list[SlurmJobConfig]) -> None:
    stems = [job.sweep.stem for job in jobs]
    if len(stems) != len(set(stems)):
        msg = "Slurm job manifest must not repeat the same sweep stem."
        raise ConfigError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
