"""Tests for named experiment registry loading and selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from experiments import ConfigError
from experiments.config import (
    ExperimentRegistry,
    RegisteredExperiment,
    load_experiment_registry,
    resolve_registry_experiment,
)
from experiments.models.deeplog import DeepLogModelConfig

if TYPE_CHECKING:
    from pathlib import Path


def _write_dataset_manifest(datasets_dir: Path, relative_path: str) -> None:
    dataset_path = datasets_dir / f"{relative_path}.toml"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        (
            f'name = "{relative_path.rsplit("/", maxsplit=1)[-1]}"\n'
            "\n[dataset]\n"
            f'name = "{relative_path.rsplit("/", maxsplit=1)[-1]}"\n'
            f'dataset_name = "{relative_path.rsplit("/", maxsplit=1)[-1]}"\n'
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


def _write_registry_tree(tmp_path: Path) -> Path:
    experiments_root = tmp_path / "experiments"
    datasets_dir = experiments_root / "configs" / "datasets"
    models_dir = experiments_root / "configs" / "models"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    for relative_path in (
        "demo",
        "ait_ads/base",
    ):
        _write_dataset_manifest(datasets_dir, relative_path)

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
    (models_dir / "deeplog_default.toml").write_text(
        'name = "deeplog_default"\ndetector = "deeplog"\n',
        encoding="utf-8",
    )

    registry_path = experiments_root / "configs" / "registry.toml"
    registry_path.write_text(
        (
            "[model_sets.baselines]\n"
            'models = ["template_frequency_default", "markov_default"]\n'
            "\n"
            "[model_sets.deepcase]\n"
            'models = ["deepcase"]\n'
            "\n"
            "[model_sets.deeplog]\n"
            'models = ["deeplog_default"]\n'
            "\n"
            "[model_sets.deeplog.overrides]\n"
            '"model.parameter_detection_enabled" = false\n'
            "\n"
            "[experiment_presets.entity_with_deepcase]\n"
            'models = ["baselines", "deepcase"]\n'
            "\n"
            "[experiment_presets.paper_deeplog]\n"
            'models = ["deeplog", "baselines"]\n'
            "\n"
            "[experiment_presets.paper_deeplog.overrides.deeplog]\n"
            '"model.top_g_values" = [1, 3, 5]\n'
            '"model.parameter_detection_enabled" = false\n'
            "\n"
            "[experiments.demo]\n"
            'dataset = "demo"\n'
            'preset = "entity_with_deepcase"\n'
            "\n"
            "[experiments.demo_deeplog]\n"
            'dataset = "demo"\n'
            'preset = "paper_deeplog"\n'
            "\n"
            "[experiments.ait_ads]\n"
            'dataset = "ait_ads/base"\n'
            'preset = "entity_with_deepcase"\n'
        ),
        encoding="utf-8",
    )
    return registry_path


def test_load_experiment_registry_exposes_metadata(
    tmp_path: Path,
) -> None:
    """Registry entries should decode into stable logical experiment metadata.

    Args:
        tmp_path (Path): Temporary directory used to build the registry tree.
    """
    registry_path = _write_registry_tree(tmp_path)
    registry = load_experiment_registry(registry_path, repo_root=tmp_path)

    assert registry.names() == (
        "demo",
        "demo_deeplog",
        "ait_ads",
    )
    entry = registry.require("demo")
    assert entry.dataset == "demo"
    assert entry.model_sets == ("baselines", "deepcase")
    assert entry.preset == "entity_with_deepcase"
    assert entry.groups == ("entity_with_deepcase", "baselines", "deepcase")


def test_registry_select_combines_names_and_groups() -> None:
    """Explicit names and group filters should both contribute to selection."""
    registry = ExperimentRegistry(
        model_sets=(),
        experiment_presets=(),
        experiments=(
            RegisteredExperiment(
                name="alpha",
                dataset="demo",
                model_sets=("deeplog",),
                groups=("paper", "deeplog"),
                preset="paper_deeplog",
            ),
            RegisteredExperiment(
                name="beta",
                dataset="demo",
                model_sets=("baselines",),
                groups=("entity_with_deepcase", "baselines"),
                preset="entity_with_deepcase",
            ),
            RegisteredExperiment(
                name="gamma",
                dataset="other",
                model_sets=("deepcase",),
                groups=("entity_with_deepcase", "deepcase"),
                preset="entity_with_deepcase",
            ),
        ),
    )

    selected = registry.select(names=("beta",), groups=("deeplog",))

    assert [experiment.name for experiment in selected] == [
        "beta",
        "alpha",
    ]


def test_load_experiment_registry_rejects_missing_model_config(
    tmp_path: Path,
) -> None:
    """Registry loading should fail when a referenced model config is absent.

    Args:
        tmp_path (Path): Temporary directory used to build the registry tree.
    """
    experiments_root = tmp_path / "experiments"
    datasets_dir = experiments_root / "configs" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset_manifest(datasets_dir, "demo")
    registry_path = experiments_root / "configs" / "registry.toml"
    registry_path.write_text(
        (
            "[model_sets.missing]\n"
            'models = ["missing_model"]\n'
            "\n"
            "[experiment_presets.single]\n"
            'models = ["missing"]\n'
            "\n"
            "[experiments.demo]\n"
            'dataset = "demo"\n'
            'preset = "single"\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Missing named config"):
        load_experiment_registry(registry_path, repo_root=tmp_path)


def test_load_experiment_registry_rejects_missing_dataset_config(
    tmp_path: Path,
) -> None:
    """Registry loading should fail when a referenced dataset config is absent.

    Args:
        tmp_path (Path): Temporary directory used to build the registry tree.
    """
    experiments_root = tmp_path / "experiments"
    models_dir = experiments_root / "configs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
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
            'dataset = "missing_dataset"\n'
            'preset = "entity_with_deepcase"\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Missing dataset config"):
        load_experiment_registry(registry_path, repo_root=tmp_path)


def test_load_experiment_registry_rejects_malformed_entries(
    tmp_path: Path,
) -> None:
    """Malformed registry fields should fail during decoding.

    Args:
        tmp_path (Path): Temporary directory used to build the registry tree.
    """
    experiments_root = tmp_path / "experiments"
    models_dir = experiments_root / "configs" / "models"
    datasets_dir = experiments_root / "configs" / "datasets"
    models_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset_manifest(datasets_dir, "demo")
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
            'models = "baselines"\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="models"):
        load_experiment_registry(registry_path, repo_root=tmp_path)


def test_resolve_registry_experiment_attaches_metadata(
    tmp_path: Path,
) -> None:
    """Named experiments should resolve to metadata-annotated bundles.

    Args:
        tmp_path (Path): Temporary directory used to build the registry tree.
    """
    registry_path = _write_registry_tree(tmp_path)
    resolved = resolve_registry_experiment(
        "demo",
        registry_path=registry_path,
        repo_root=tmp_path,
    )

    assert resolved.experiment.name == "demo"
    assert [bundle.concrete_name for bundle in resolved.bundles] == [
        "demo_template_frequency",
        "demo_markov",
        "demo_deepcase",
    ]
    assert resolved.bundles[0].experiment_name == "demo"
    assert resolved.bundles[0].experiment_groups == (
        "entity_with_deepcase",
        "baselines",
        "deepcase",
    )
    assert resolved.bundles[0].normalized_config()["experiment"] == {
        "name": "demo",
        "groups": ["entity_with_deepcase", "baselines", "deepcase"],
    }
    assert resolved.bundles[2].run_group == "deepcase"


def test_resolve_registry_experiment_applies_preset_overrides(
    tmp_path: Path,
) -> None:
    """Preset overrides should flow into the resolved DeepLog bundle.

    Args:
        tmp_path (Path): Temporary directory used to build the registry tree.
    """
    registry_path = _write_registry_tree(tmp_path)
    resolved = resolve_registry_experiment(
        "demo_deeplog",
        registry_path=registry_path,
        repo_root=tmp_path,
    )

    deeplog_bundle = next(
        bundle for bundle in resolved.bundles if bundle.model.name == "deeplog_default"
    )
    assert isinstance(deeplog_bundle.model, DeepLogModelConfig)
    assert deeplog_bundle.model.parameter_detection_enabled is False
    assert deeplog_bundle.model.top_g_values == (1, 3, 5)
