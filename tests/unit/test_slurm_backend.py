"""Tests for the Slurm submission backend."""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from typing import TYPE_CHECKING

from experiments.config import RegisteredExperiment
from experiments.execution import slurm
from experiments.execution.slurm import SlurmSubmitRequest

if TYPE_CHECKING:
    import pytest


def _write_slurm_tree(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    experiments_root = tmp_path / "experiments"
    datasets_dir = experiments_root / "configs" / "datasets"
    models_dir = experiments_root / "configs" / "models"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (datasets_dir / "demo.toml").write_text(
        (
            'name = "demo"\ndataset_name = "demo"\n\n[[models]]\n'
            'ref = "template_frequency_default"\n'
        ),
        encoding="utf-8",
    )
    (models_dir / "template_frequency_default.toml").write_text(
        'name = "template_frequency_default"\ndetector = "template_frequency"\n',
        encoding="utf-8",
    )
    registry_path = experiments_root / "configs" / "registry.toml"
    registry_path.write_text(
        (
            "[model_sets.baselines]\n"
            'models = ["template_frequency_default"]\n'
            "\n"
            "[experiment_presets.entity_with_deepcase]\n"
            'models = ["baselines"]\n'
            "\n"
            "[experiments.demo]\n"
            'dataset = "demo"\n'
            'preset = "entity_with_deepcase"\n'
            "\n"
            "[experiments.demo_two]\n"
            'dataset = "demo"\n'
            'preset = "entity_with_deepcase"\n'
        ),
        encoding="utf-8",
    )
    defaults_path = experiments_root / "configs" / "slurm.toml"
    defaults_path.write_text(
        ('time = "01:00:00"\ncpus_per_task = 2\nmem = "8G"\ngres = "gpu:1"\n'),
        encoding="utf-8",
    )
    return registry_path, defaults_path


def _make_submission(
    *,
    repo_root: pathlib.Path,
    force: bool = False,
    write_predictions: bool = False,
    debug_reporting: bool = False,
) -> slurm._SlurmSubmission:
    experiments = (
        RegisteredExperiment(
            name="demo",
            dataset="demo",
            model_sets=("baselines",),
            groups=("baselines",),
            preset="entity_with_deepcase",
        ),
        RegisteredExperiment(
            name="demo_two",
            dataset="demo",
            model_sets=("baselines",),
            groups=("baselines",),
            preset="entity_with_deepcase",
        ),
    )
    return slurm._SlurmSubmission(  # noqa: SLF001
        experiments=experiments,
        defaults=slurm._SlurmDefaults(  # noqa: SLF001
            time="01:00:00",
            cpus_per_task=2,
            mem="8G",
            gres="gpu:1",
        ),
        registry_path=pathlib.Path("experiments/configs/registry.toml"),
        repo_root=repo_root,
        log_dir=repo_root
        / "experiments"
        / "results"
        / "slurm-logs"
        / slurm._build_submission_label(experiments),  # noqa: SLF001
        force=force,
        write_predictions=write_predictions,
        debug_reporting=debug_reporting,
    )


def test_build_sbatch_command_uses_job_array_without_throttle() -> None:
    """Slurm submissions should expand selected experiments into one array."""
    submission = _make_submission(
        repo_root=pathlib.Path("/repo"),
    )

    command = slurm.build_sbatch_command(submission)

    assert command[0] == "sbatch"
    assert "--job-name=anomalog" in command
    assert "--array=0-1" in command
    assert "--chdir=/repo" in command
    assert not any(token.startswith("--array=0-1%") for token in command)


def test_build_sbatch_command_uses_array_output_names() -> None:
    """Array jobs should write task logs with both array identifiers."""
    submission = _make_submission(
        repo_root=pathlib.Path("/repo"),
    )

    command = slurm.build_sbatch_command(submission)

    assert f"--output={submission.log_dir.as_posix()}/%A_%a.out" in command
    assert f"--error={submission.log_dir.as_posix()}/%A_%a.err" in command


def test_build_wrap_script_indexes_embedded_array_by_task_id() -> None:
    """Each array task should load its experiment name from the embedded array."""
    submission = _make_submission(
        repo_root=pathlib.Path("/repo with spaces"),
    )

    wrap_script = slurm._build_wrap_script(submission)  # noqa: SLF001

    assert "EXPERIMENTS=(" in wrap_script
    assert "  demo" in wrap_script
    assert "  demo_two" in wrap_script
    assert 'EXPERIMENT_NAME="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]:-}"' in wrap_script
    assert 'if [ -z "$EXPERIMENT_NAME" ]; then' in wrap_script
    assert 'export RUN_NAME="$EXPERIMENT_NAME"' in wrap_script
    assert "export REPO_ROOT='/repo with spaces'" in wrap_script
    assert "set -euo pipefail" in wrap_script
    assert "bash -lc" in wrap_script


def test_build_wrap_script_propagates_run_flags() -> None:
    """Array tasks should pass execution flags through to the run command."""
    submission = _make_submission(
        repo_root=pathlib.Path("/repo"),
        force=True,
        write_predictions=True,
        debug_reporting=True,
    )

    wrap_script = slurm._build_wrap_script(submission)  # noqa: SLF001

    assert "--force" in wrap_script
    assert "--write-predictions" in wrap_script
    assert "--debug-reporting" in wrap_script


def test_submit_experiments_submits_one_array_job(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submitting multiple registry entries should call `sbatch` once.

    Args:
        tmp_path (pathlib.Path): Temporary directory used to build the registry
            fixtures.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture used to
            replace `subprocess.run`.
    """
    registry_path, defaults_path = _write_slurm_tree(tmp_path)
    completed = SimpleNamespace(
        stdout="Submitted batch job 123\n",
        stderr="",
    )
    calls: list[list[str]] = []

    def _record_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> SimpleNamespace:
        del check, capture_output, text
        calls.append(command)
        return completed

    monkeypatch.setattr(slurm.subprocess, "run", _record_run)

    outputs = slurm.submit_experiments(
        SlurmSubmitRequest(
            registry_path=registry_path,
            repo_root=tmp_path,
            defaults_path=defaults_path,
            groups=("baselines",),
        ),
    )

    assert outputs == ["Submitted batch job 123"]
    assert len(calls) == 1
    command = calls[0]
    assert "--array=0-1" in command
    assert "--job-name=anomalog" in command
    wrap_script = command[command.index("--wrap") + 1]
    assert "EXPERIMENTS=(" in wrap_script
    assert "  demo" in wrap_script
    assert "  demo_two" in wrap_script
    output_arg = next(token for token in command if token.startswith("--output="))
    log_dir = pathlib.Path(output_arg.removeprefix("--output=").rsplit("/", 1)[0])
    assert log_dir.exists()


def test_submit_experiments_dry_run_prints_command_and_skips_subprocess(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry-run mode should print the command and avoid `sbatch`.

    Args:
        tmp_path (pathlib.Path): Temporary directory used to build the registry
            fixtures.
        capsys (pytest.CaptureFixture[str]): Pytest capture fixture used to
            inspect dry-run output.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture used to
            replace `subprocess.run`.
    """
    registry_path, defaults_path = _write_slurm_tree(tmp_path)

    def _unexpected_run(*_args: object, **_kwargs: object) -> None:
        msg = "subprocess.run should not be called in dry-run mode"
        raise AssertionError(msg)

    monkeypatch.setattr(slurm.subprocess, "run", _unexpected_run)

    outputs = slurm.submit_experiments(
        SlurmSubmitRequest(
            registry_path=registry_path,
            repo_root=tmp_path,
            defaults_path=defaults_path,
            groups=("baselines",),
            dry_run=True,
        ),
    )

    assert outputs == []
    output_lines = capsys.readouterr().out.splitlines()
    assert output_lines[0] == "sbatch command:"
    assert any(line.strip() == "--wrap <<'EOF'" for line in output_lines)
    assert any(line.strip() == "EXPERIMENTS=(" for line in output_lines)
    assert not any(line.startswith("Manifest") for line in output_lines)
    expected_log_dir = (
        tmp_path
        / "experiments"
        / "results"
        / "slurm-logs"
        / slurm._build_submission_label(  # noqa: SLF001
            (
                RegisteredExperiment(
                    name="demo",
                    dataset="demo",
                    model_sets=("baselines",),
                    groups=("baselines",),
                    preset="entity_with_deepcase",
                ),
                RegisteredExperiment(
                    name="demo_two",
                    dataset="demo",
                    model_sets=("baselines",),
                    groups=("baselines",),
                    preset="entity_with_deepcase",
                ),
            ),
        )
    )
    assert not expected_log_dir.exists()
