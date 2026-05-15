"""Tests for the local registry suite runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiments.runners.run_suite import SuiteRunRequest, run_suite

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


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
    (models_dir / "markov_default.toml").write_text(
        'name = "markov_default"\ndetector = "markov"\n',
        encoding="utf-8",
    )
    registry_path = experiments_root / "configs" / "registry.toml"
    registry_path.write_text(
        (
            "[model_sets.baselines]\n"
            'models = ["template_frequency_default", "markov_default"]\n'
            "\n"
            "[experiment_presets.entity_with_deepcase]\n"
            'models = ["baselines"]\n'
            "\n"
            "[experiments.demo]\n"
            'dataset = "demo"\n'
            'preset = "entity_with_deepcase"\n'
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
            groups=("baselines",),
            list_only=True,
        ),
    )

    output = capsys.readouterr().out.strip().splitlines()
    assert output == [
        "demo\tdataset=demo\tpreset=entity_with_deepcase\tmodels=baselines\tgroups=entity_with_deepcase,baselines",
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
        ),
    )

    output = capsys.readouterr().out.strip()
    assert "--experiment demo" in output
    assert "--registry" in output
    assert "experiments.runners.run_experiment" in output
