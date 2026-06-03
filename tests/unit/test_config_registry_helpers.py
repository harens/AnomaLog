"""Tests for public experiment config and registry loading APIs."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import ConfigError
from experiments.config import load_experiment_bundles, load_experiment_registry


def _write_minimal_dataset_manifest(datasets_dir: Path, name: str) -> Path:
    path = datasets_dir / f"{name}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            f'name = "{name}"\n'
            'description = "Demo dataset"\n'
            "\n[dataset]\n"
            f'name = "{name}"\n'
            f'dataset_name = "{name.upper()}"\n'
            'preset = "demo"\n'
            'structured_parser = "bgl"\n'
            'template_parser = "identity"\n'
            "\n[dataset.source]\n"
            'type = "local_dir"\n'
            'path = "."\n'
            "\n[dataset.sequence]\n"
            'grouping = "entity"\n'
        ),
        encoding="utf-8",
    )
    return path


def _write_minimal_model_files(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "template_frequency_default.toml").write_text(
        (
            'name = "template_frequency_default"\n'
            'detector = "template_frequency"\n'
            'primary_metric_scope = "event_level_detection"\n'
        ),
        encoding="utf-8",
    )
    (models_dir / "naive_bayes_default.toml").write_text(
        ('name = "naive_bayes_default"\ndetector = "naive_bayes"\n'),
        encoding="utf-8",
    )


def test_load_experiment_registry_expands_model_sets_and_experiments(
    tmp_path: Path,
) -> None:
    """Registry loading should expand named sets and concrete experiments.

    Args:
        tmp_path (Path): Temporary directory used to build the synthetic
            registry tree.
    """
    experiments_root = tmp_path / "experiments"
    datasets_dir = experiments_root / "configs" / "datasets"
    models_dir = experiments_root / "configs" / "models"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    _write_minimal_dataset_manifest(datasets_dir, "demo")
    _write_minimal_model_files(models_dir)
    registry_path = experiments_root / "configs" / "registry.toml"
    registry_path.write_text(
        (
            "[model_sets.baselines]\n"
            'models = ["template_frequency_default", "naive_bayes_default"]\n'
            "\n"
            "[experiments.demo]\n"
            'dataset = "demo"\n'
            'model_sets = ["baselines"]\n'
            'models = ["template_frequency_default"]\n'
        ),
        encoding="utf-8",
    )

    registry = load_experiment_registry(registry_path, repo_root=tmp_path)

    assert registry.model_set("baselines").models == (
        "template_frequency_default",
        "naive_bayes_default",
    )
    assert registry.require("demo").dataset == "demo"
    assert tuple(
        experiment.name for experiment in registry.select(names=("demo",))
    ) == ("demo",)
    resolved = registry.resolve_experiment(
        "demo",
        registry_path=registry_path,
        repo_root=tmp_path,
    )
    expected_bundle_count = 3
    assert len(resolved.bundles) == expected_bundle_count
    assert all(bundle.dataset_path.name == "demo.toml" for bundle in resolved.bundles)


def test_load_experiment_registry_rejects_missing_model_reference(
    tmp_path: Path,
) -> None:
    """Registry loading should fail fast for unresolved model references.

    Args:
        tmp_path (Path): Temporary directory used to build the synthetic
            registry tree.
    """
    experiments_root = tmp_path / "experiments"
    datasets_dir = experiments_root / "configs" / "datasets"
    models_dir = experiments_root / "configs" / "models"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    _write_minimal_dataset_manifest(datasets_dir, "demo")
    _write_minimal_model_files(models_dir)
    registry_path = experiments_root / "configs" / "registry.toml"
    registry_path.write_text(
        ('[experiments.demo]\ndataset = "demo"\nmodels = ["missing_model"]\n'),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Missing named config"):
        load_experiment_registry(registry_path, repo_root=tmp_path)


def test_load_experiment_bundles_expands_existing_dataset_config() -> None:
    """Dataset experiment loading should expand a checked-in public config."""
    bundles = load_experiment_bundles(
        Path("experiments/configs/datasets/bgl/entity_chronological.toml"),
    )

    assert bundles
    assert bundles[0].dataset.preset == "bgl"
