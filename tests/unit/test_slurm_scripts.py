"""Tests for generated Slurm wrapper rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import ConfigError
from experiments.slurm_scripts import (
    SlurmJobConfig,
    load_slurm_jobs,
    render_slurm_script,
    write_slurm_scripts,
)


def test_render_slurm_script_uses_sweep_stem_and_relative_config_path(
    tmp_path: Path,
) -> None:
    """Generated wrappers should be derived directly from the sweep file name.

    Args:
        tmp_path (Path): Temporary repository root for the generated files.
    """
    repo_root = tmp_path
    sweep_path = repo_root / "experiments/configs/datasets/example.toml"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path.write_text("name = 'example'\n", encoding="utf-8")

    rendered = render_slurm_script(
        SlurmJobConfig(
            sweep=Path("experiments/configs/datasets/example.toml"),
            force=True,
        ),
        repo_root=repo_root,
        manifest_path=repo_root / "slurm/jobs.toml",
    )

    assert rendered.path == Path("example.sbatch")
    assert "#SBATCH --job-name=example" in rendered.content
    assert 'export RUN_NAME="example"' in rendered.content
    assert (
        'REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"'
        in rendered.content
    )
    assert (
        'export UV_CACHE_DIR="${UV_CACHE_DIR:-'
        '${SLURM_TMPDIR:-${REPO_ROOT}/.cache/uv}}"' in rendered.content
    )
    assert (
        'mkdir -p "$PREFECT_HOME" "$PREFECT_LOCAL_STORAGE_PATH" '
        '"$UV_CACHE_DIR"' in rendered.content
    )
    assert (
        'export PREFECT_LOCAL_STORAGE_PATH="${PREFECT_ROOT}/storage"'
        in rendered.content
    )
    assert (
        '  --config "experiments/configs/datasets/example.toml" \\' in rendered.content
    )
    assert "  --force" in rendered.content


def test_write_slurm_scripts_materialises_all_manifest_entries(tmp_path: Path) -> None:
    """The manifest should control the generated Slurm file set.

    Args:
        tmp_path (Path): Temporary repository root for the generated files.
    """
    repo_root = tmp_path
    sweep_root = repo_root / "experiments/configs/datasets"
    sweep_root.mkdir(parents=True, exist_ok=True)
    for stem in ("first", "second"):
        (sweep_root / f"{stem}.toml").write_text(f'name = "{stem}"\n', encoding="utf-8")

    manifest_path = repo_root / "slurm/jobs.toml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        (
            "sweeps = [\n"
            '  "experiments/configs/datasets/first.toml",\n'
            '  "experiments/configs/datasets/second.toml",\n'
            "]\n"
        ),
        encoding="utf-8",
    )

    written_paths = write_slurm_scripts(
        manifest_path,
        repo_root=repo_root,
        output_dir=repo_root / "slurm",
    )

    assert written_paths == [
        repo_root / "slurm/first.sbatch",
        repo_root / "slurm/second.sbatch",
    ]
    assert "  --force" in (repo_root / "slurm/first.sbatch").read_text(
        encoding="utf-8",
    )
    assert "  --force" in (repo_root / "slurm/second.sbatch").read_text(
        encoding="utf-8",
    )


def test_load_slurm_jobs_rejects_duplicate_stems(tmp_path: Path) -> None:
    """Generated wrappers should retain a one-to-one mapping by sweep stem.

    Args:
        tmp_path (Path): Temporary repository root for the generated files.
    """
    repo_root = tmp_path
    sweep_root = repo_root / "experiments/configs/datasets"
    sweep_root.mkdir(parents=True, exist_ok=True)
    (sweep_root / "a.toml").write_text('name = "a"\n', encoding="utf-8")
    nested_root = sweep_root / "nested"
    nested_root.mkdir(parents=True, exist_ok=True)
    (nested_root / "a.toml").write_text('name = "a"\n', encoding="utf-8")

    manifest_path = repo_root / "slurm/jobs.toml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        (
            "sweeps = [\n"
            '  "experiments/configs/datasets/a.toml",\n'
            '  "experiments/configs/datasets/nested/a.toml",\n'
            "]\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="repeat the same manifest stem"):
        load_slurm_jobs(manifest_path, repo_root=repo_root)
