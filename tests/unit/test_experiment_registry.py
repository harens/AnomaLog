"""Tests for named experiment registry loading and selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import ConfigError
from experiments.config import (
    ExperimentRegistry,
    RegisteredExperiment,
    load_experiment_registry,
    resolve_registry_experiment,
)
from experiments.models.deeplog import DeepLogModelConfig


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
        "alt",
        "paper_demo",
        "paper_alt",
        "ait_ads/base",
    ):
        _write_dataset_manifest(datasets_dir, relative_path)

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
    (models_dir / "deeplog_default.toml").write_text(
        'name = "deeplog_default"\ndetector = "deeplog"\n',
        encoding="utf-8",
    )

    registry_path = experiments_root / "configs" / "registry.toml"
    registry_path.write_text(
        (
            "[model_sets.baselines_with_nb]\n"
            'models = ["template_frequency_default", '
            '"naive_bayes_default", "markov_default"]\n'
            "\n"
            "[model_sets.baselines_no_nb]\n"
            'models = ["template_frequency_default", "markov_default"]\n'
            "\n"
            "[experiments.demo]\n"
            'dataset = "demo"\n'
            'model_sets = ["baselines_with_nb"]\n'
            'models = ["deepcase", "deeplog_default"]\n'
            "\n"
            "[experiments.demo_deeplog]\n"
            'dataset = "demo"\n'
            'model_sets = ["baselines_no_nb"]\n'
            'models = ["deeplog_default"]\n'
            "\n"
            "[experiments.demo_deeplog.overrides.deeplog_default]\n"
            '"model.top_g_values" = [1, 3, 5]\n'
            '"model.parameter_detection_enabled" = false\n'
            "\n"
            "[experiments.ait_ads]\n"
            'dataset = "ait_ads/base"\n'
            'model_sets = ["baselines_with_nb"]\n'
            'models = ["deepcase"]\n'
            "\n"
            "[experiment_sets.paper_group]\n"
            'model_sets = ["baselines_no_nb"]\n'
            'models = ["deepcase"]\n'
            'datasets = ["paper_demo", "paper_alt"]\n'
        ),
        encoding="utf-8",
    )
    return registry_path


def test_load_experiment_registry_exposes_metadata(
    tmp_path: Path,
) -> None:
    """Registry entries should decode into stable logical experiment metadata."""
    registry_path = _write_registry_tree(tmp_path)
    registry = load_experiment_registry(registry_path, repo_root=tmp_path)

    assert registry.names() == (
        "demo",
        "demo_deeplog",
        "ait_ads",
        "paper_demo",
        "paper_alt",
    )
    entry = registry.require("demo")
    assert entry.dataset == "demo"
    assert entry.models == ("deepcase", "deeplog_default")
    assert entry.model_sets == ("baselines_with_nb",)
    assert entry.groups == ("demo", "baselines_with_nb")


def test_load_experiment_registry_registers_thunderbird_runs() -> None:
    """Thunderbird experiment names should resolve through the checked-in registry."""
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_experiment_registry(
        repo_root / "experiments" / "configs" / "registry.toml",
        repo_root=repo_root,
    )

    thunderbird = registry.require("thunderbird")
    assert thunderbird.dataset == "thunderbird"
    assert thunderbird.models == ("deeplog_default",)
    assert thunderbird.model_sets == ("baselines_with_nb",)


def test_load_experiment_registry_registers_thunderbird_entity_runs() -> None:
    """Thunderbird entity-grouped runs should resolve through the registry."""
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_experiment_registry(
        repo_root / "experiments" / "configs" / "registry.toml",
        repo_root=repo_root,
    )

    thunderbird_entity = registry.require("thunderbird_entity_chronological")
    assert thunderbird_entity.dataset == "thunderbird/entity_chronological"
    assert thunderbird_entity.models == ("deepcase", "deeplog_default")
    assert thunderbird_entity.model_sets == ("baselines_with_nb",)


def test_registry_select_combines_names_and_groups() -> None:
    """Explicit names and group filters should both contribute to selection."""
    registry = ExperimentRegistry(
        model_sets=(),
        experiments=(
            RegisteredExperiment(
                name="alpha",
                dataset="demo",
                models=("deeplog_default",),
                model_sets=("deeplog",),
                groups=("paper", "deeplog"),
            ),
            RegisteredExperiment(
                name="beta",
                dataset="demo",
                models=(),
                model_sets=("baselines",),
                groups=("entity_with_deepcase", "baselines"),
            ),
            RegisteredExperiment(
                name="gamma",
                dataset="other",
                models=("deepcase",),
                model_sets=(),
                groups=("entity_with_deepcase", "deepcase"),
            ),
        ),
    )

    selected = registry.select(names=("beta",), groups=("deeplog",))

    assert [experiment.name for experiment in selected] == [
        "beta",
        "alpha",
    ]


def test_experiment_set_uses_set_name_as_group(
    tmp_path: Path,
) -> None:
    """Experiment sets should stay selectable even without a wrapper layer."""
    experiments_root = tmp_path / "experiments"
    datasets_dir = experiments_root / "configs" / "datasets"
    models_dir = experiments_root / "configs" / "models"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset_manifest(datasets_dir, "demo")
    _write_dataset_manifest(datasets_dir, "alt")
    (models_dir / "template_frequency_default.toml").write_text(
        'name = "template_frequency_default"\ndetector = "template_frequency"\n',
        encoding="utf-8",
    )
    (models_dir / "deepcase.toml").write_text(
        'name = "deepcase"\ndetector = "deepcase"\n',
        encoding="utf-8",
    )
    registry_path = experiments_root / "configs" / "registry.toml"
    registry_path.write_text(
        (
            "[model_sets.baselines]\n"
            'models = ["template_frequency_default"]\n'
            "\n"
            "[experiment_sets.paper_group]\n"
            'model_sets = ["baselines"]\n'
            'models = ["deepcase"]\n'
            'datasets = ["paper_demo", "paper_alt"]\n'
        ),
        encoding="utf-8",
    )

    registry = load_experiment_registry(registry_path, repo_root=tmp_path)
    selected = registry.select(groups=("paper_group",))

    assert [experiment.name for experiment in selected] == [
        "paper_demo",
        "paper_alt",
    ]
    assert all("paper_group" in experiment.groups for experiment in selected)


def test_load_experiment_registry_rejects_missing_model_config(
    tmp_path: Path,
) -> None:
    """Registry loading should fail when a referenced model config is absent."""
    experiments_root = tmp_path / "experiments"
    datasets_dir = experiments_root / "configs" / "datasets"
    models_dir = experiments_root / "configs" / "models"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset_manifest(datasets_dir, "demo")
    (models_dir / "deepcase.toml").write_text(
        'name = "deepcase"\ndetector = "deepcase"\n',
        encoding="utf-8",
    )
    registry_path = experiments_root / "configs" / "registry.toml"
    registry_path.write_text(
        (
            "[model_sets.missing]\n"
            'models = ["missing_model"]\n'
            "\n"
            "[experiments.demo]\n"
            'dataset = "demo"\n'
            'model_sets = ["missing"]\n'
            'models = ["deepcase"]\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Missing named config"):
        load_experiment_registry(registry_path, repo_root=tmp_path)


def test_load_experiment_registry_rejects_missing_dataset_config(
    tmp_path: Path,
) -> None:
    """Registry loading should fail when a referenced dataset config is absent."""
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
            "[experiments.demo]\n"
            'dataset = "missing_dataset"\n'
            'model_sets = ["baselines"]\n'
            'models = ["deepcase"]\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Missing dataset config"):
        load_experiment_registry(registry_path, repo_root=tmp_path)


def test_load_experiment_registry_rejects_unknown_model_set(
    tmp_path: Path,
) -> None:
    """Registry loading should fail when a referenced model set is absent."""
    experiments_root = tmp_path / "experiments"
    datasets_dir = experiments_root / "configs" / "datasets"
    models_dir = experiments_root / "configs" / "models"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset_manifest(datasets_dir, "demo")
    (models_dir / "deepcase.toml").write_text(
        'name = "deepcase"\ndetector = "deepcase"\n',
        encoding="utf-8",
    )
    registry_path = experiments_root / "configs" / "registry.toml"
    registry_path.write_text(
        (
            "[experiments.demo]\n"
            'dataset = "demo"\n'
            'model_sets = ["missing"]\n'
            'models = ["deepcase"]\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Unknown model set"):
        load_experiment_registry(registry_path, repo_root=tmp_path)


def test_load_experiment_registry_rejects_malformed_entries(
    tmp_path: Path,
) -> None:
    """Malformed registry fields should fail during decoding."""
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
            "[experiments.demo]\n"
            'dataset = "demo"\n'
            'model_sets = "baselines"\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="model_sets"):
        load_experiment_registry(registry_path, repo_root=tmp_path)


def test_resolve_registry_experiment_attaches_metadata(
    tmp_path: Path,
) -> None:
    """Named experiments should resolve to metadata-annotated bundles."""
    registry_path = _write_registry_tree(tmp_path)
    resolved = resolve_registry_experiment(
        "demo",
        registry_path=registry_path,
        repo_root=tmp_path,
    )

    assert resolved.experiment.name == "demo"
    assert [bundle.concrete_name for bundle in resolved.bundles] == [
        "demo_template_frequency",
        "demo_naive_bayes",
        "demo_markov",
        "demo_deepcase",
        "demo_deeplog",
    ]
    assert resolved.bundles[0].experiment_name == "demo"
    assert resolved.bundles[0].experiment_groups == (
        "demo",
        "baselines_with_nb",
    )
    assert resolved.bundles[0].normalized_config()["experiment"] == {
        "name": "demo",
        "groups": ["demo", "baselines_with_nb"],
    }
    assert resolved.bundles[3].run_group == "deepcase"
    assert resolved.bundles[4].run_group == "deeplog_default"


def test_resolve_registry_experiment_applies_model_overrides(
    tmp_path: Path,
) -> None:
    """Model overrides should flow into the resolved DeepLog bundle."""
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
