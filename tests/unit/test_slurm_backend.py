"""Tests for the Slurm submission backend."""

from __future__ import annotations

import os
import pathlib
from types import SimpleNamespace

import pytest

from experiments.config import RegisteredExperiment
from experiments.execution import slurm
from experiments.execution.slurm import SlurmSubmitRequest


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
    cluster_roots: tuple[pathlib.Path | None, pathlib.Path | None] | None = None,
) -> slurm._SlurmSubmission:
    data_root: pathlib.Path | None
    cache_root: pathlib.Path | None
    if cluster_roots is None:
        data_root = None
        cache_root = None
    else:
        data_root, cache_root = cluster_roots
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
        data_root=data_root,
        cache_root=cache_root,
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

    assert "set -- \\" in wrap_script
    assert "  demo" in wrap_script
    assert "  demo_two" in wrap_script
    assert 'for EXPERIMENT in "$@"; do' in wrap_script
    assert "EXPERIMENT_INDEX=0" in wrap_script
    assert 'if [ "$EXPERIMENT_INDEX" -eq "$SLURM_ARRAY_TASK_ID" ]; then' in wrap_script
    assert "EXPERIMENT_NAME=$EXPERIMENT" in wrap_script
    assert 'if [ -z "$EXPERIMENT_NAME" ]; then' in wrap_script
    assert 'export RUN_NAME="$EXPERIMENT_NAME"' in wrap_script
    assert "export REPO_ROOT='/repo with spaces'" in wrap_script
    assert "set -eu" in wrap_script
    assert "pipefail" not in wrap_script
    assert "bash -lc" in wrap_script


def test_build_wrap_script_runs_under_posix_shell_and_exports_experiment(
    tmp_path: pathlib.Path,
) -> None:
    """The generated wrapper should execute under `/bin/sh`.

    It must keep the selected experiment visible to the nested command.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    submission = _make_submission(repo_root=repo_root)
    wrap_script = slurm._build_wrap_script(submission)  # noqa: SLF001

    home_dir = tmp_path / "home"
    bin_dir = home_dir / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    capture_path = tmp_path / "uv-args.txt"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" > "$UV_CAPTURE_PATH"\n'
        'printf "EXPERIMENT_NAME=%s\\n" "$EXPERIMENT_NAME" >> "$UV_CAPTURE_PATH"\n'
        'printf "RUN_NAME=%s\\n" "$RUN_NAME" >> "$UV_CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    env = dict(os.environ)
    env["HOME"] = home_dir.as_posix()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["SLURM_ARRAY_TASK_ID"] = "1"
    env["UV_CAPTURE_PATH"] = capture_path.as_posix()

    result = slurm.subprocess.run(
        ["/bin/sh", "-c", wrap_script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert not result.stdout
    assert not result.stderr
    captured_lines = capture_path.read_text(encoding="utf-8").splitlines()
    assert captured_lines[:2] == ["run", "python"]
    assert "--experiment" in captured_lines
    assert "demo_two" in captured_lines
    assert "EXPERIMENT_NAME=demo_two" in captured_lines
    assert "RUN_NAME=demo_two" in captured_lines


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


def test_build_wrap_script_exports_cluster_cache_roots() -> None:
    """Configured SLURM roots should propagate into the wrapped job environment."""
    submission = _make_submission(
        repo_root=pathlib.Path("/repo"),
        cluster_roots=(
            pathlib.Path("/data/hs1822"),
            pathlib.Path("/data/hs1822/.cache"),
        ),
    )

    wrap_script = slurm._build_wrap_script(submission)  # noqa: SLF001

    assert "export ANOMALOG_DATA_ROOT=/data/hs1822" in wrap_script
    assert "export ANOMALOG_CACHE_ROOT=/data/hs1822/.cache" in wrap_script
    assert (
        'export PREFECT_ROOT="${PREFECT_ROOT:-${ANOMALOG_CACHE_ROOT}/prefect}"'
        in wrap_script
    )
    assert (
        'export PREFECT_SERVER_ANALYTICS_ENABLED="${'
        'PREFECT_SERVER_ANALYTICS_ENABLED:-false}"' in wrap_script
    )
    assert 'export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"' in wrap_script
    assert (
        'export UV_CACHE_DIR="${UV_CACHE_DIR:-'
        '${SLURM_TMPDIR:-${ANOMALOG_CACHE_ROOT:-${REPO_ROOT}}/uv}}"' in wrap_script
    )
    assert (
        'mkdir -p "$PREFECT_HOME" "$PREFECT_LOCAL_STORAGE_PATH" "$UV_CACHE_DIR"'
        in wrap_script
    )
    assert 'mkdir -p "$ANOMALOG_DATA_ROOT"' in wrap_script
    assert 'mkdir -p "$ANOMALOG_CACHE_ROOT"' in wrap_script


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
    assert "for EXPERIMENT in" in wrap_script
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
    assert any(line.strip() == "set -- \\" for line in output_lines)
    assert any(line.strip() == 'for EXPERIMENT in "$@"; do' for line in output_lines)
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


def test_submit_experiments_uses_relative_cluster_roots_from_request(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request-level SLURM roots should resolve relative to the repo root."""
    registry_path, defaults_path = _write_slurm_tree(tmp_path)
    completed = SimpleNamespace(stdout="Submitted batch job 456\n", stderr="")
    captured: dict[str, list[str]] = {}

    def _record_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> SimpleNamespace:
        del check, capture_output, text
        captured["command"] = command
        return completed

    monkeypatch.setattr(slurm.subprocess, "run", _record_run)

    slurm.submit_experiments(
        SlurmSubmitRequest(
            registry_path=registry_path,
            repo_root=tmp_path,
            defaults_path=defaults_path,
            groups=("baselines",),
            data_root=pathlib.Path("cluster/data"),
            cache_root=pathlib.Path("cluster/cache"),
        ),
    )

    wrap_script = captured["command"][captured["command"].index("--wrap") + 1]
    assert f"export ANOMALOG_DATA_ROOT={tmp_path / 'cluster/data'}" in wrap_script
    assert f"export ANOMALOG_CACHE_ROOT={tmp_path / 'cluster/cache'}" in wrap_script


def test_submit_experiments_surfaces_sbatch_failure_message(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed Slurm submissions should raise a concise scheduler error."""
    registry_path, defaults_path = _write_slurm_tree(tmp_path)

    def _raise(
        *_args: object,
        **_kwargs: object,
    ) -> None:
        raise slurm.subprocess.CalledProcessError(
            1,
            ["sbatch"],
            output="",
            stderr="sbatch: error: invalid partition",
        )

    monkeypatch.setattr(slurm.subprocess, "run", _raise)

    with pytest.raises(SystemExit, match="sbatch: error: invalid partition"):
        slurm.submit_experiments(
            SlurmSubmitRequest(
                registry_path=registry_path,
                repo_root=tmp_path,
                defaults_path=defaults_path,
                groups=("baselines",),
            ),
        )
