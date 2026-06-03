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
from experiments.models.deepcase.detector import DeepCaseModelConfig
from experiments.models.deepcase.shared import DeepCaseClusterScoreStrategy
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
    """Registry entries should decode into stable logical experiment metadata.

    Args:
        tmp_path (Path): Temporary directory used to build a synthetic registry
            tree.
    """
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
    assert thunderbird_entity.model_sets == (
        "baselines_with_nb",
        "deepcase_majority_vote",
        "deepcase_threshold_fraction",
        "deepcase_abstain_mixed",
    )


def test_load_experiment_registry_splits_bgl_protocol_targets() -> None:
    """BGL paper and benchmark targets should resolve as distinct registry sets."""
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_experiment_registry(
        repo_root / "experiments" / "configs" / "registry.toml",
        repo_root=repo_root,
    )

    ccs2017 = registry.select(groups=("bgl_deeplog_ccs2017_paper",))
    benchmark_2022 = registry.select(groups=("bgl_how_far_are_we_2022",))

    expected_ccs2017_count = 2
    assert len(ccs2017) == expected_ccs2017_count
    assert [experiment.dataset for experiment in ccs2017] == [
        "bgl/bgl_deeplog_ccs2017_paper_1pct_normal_entry_stream_no_online",
        "bgl/bgl_deeplog_ccs2017_paper_10pct_entry_stream_no_online",
    ]
    assert {experiment.model_sets for experiment in ccs2017} == {("baselines_no_nb",)}
    assert {experiment.models for experiment in ccs2017} == {("deeplog_default",)}
    assert len(benchmark_2022) == 1
    assert [experiment.dataset for experiment in benchmark_2022] == [
        "bgl/how_far_are_we_2022",
    ]
    assert {experiment.model_sets for experiment in benchmark_2022} == {
        ("baselines_no_nb",),
    }
    assert {experiment.models for experiment in benchmark_2022} == {
        ("deeplog_default",),
    }


def test_load_experiment_registry_registers_bgl_cfdr_reuse() -> None:
    """The CFDR BGL variant should reuse the regular BGL experiment coverage."""
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_experiment_registry(
        repo_root / "experiments" / "configs" / "registry.toml",
        repo_root=repo_root,
    )

    selected = registry.select(groups=("bgl_cfdr_deeplog_ccs2017_paper",))

    assert [experiment.dataset for experiment in selected] == [
        "bgl/cfdr_deeplog_ccs2017_paper_1pct_normal_entry_stream_no_online",
        "bgl/cfdr_deeplog_ccs2017_paper_10pct_entry_stream_no_online",
    ]
    assert {experiment.model_sets for experiment in selected} == {("baselines_no_nb",)}
    assert {experiment.models for experiment in selected} == {("deeplog_default",)}

    bgl_cfdr_entity = registry.require("bgl_cfdr_entity_chronological")
    assert bgl_cfdr_entity.dataset == "bgl/cfdr_entity_chronological"
    assert bgl_cfdr_entity.models == ("deepcase", "deeplog_default")
    assert bgl_cfdr_entity.model_sets == (
        "baselines_with_nb",
        "deepcase_majority_vote",
        "deepcase_threshold_fraction",
        "deepcase_abstain_mixed",
    )


def test_hdfs_deeplog_paper_registry_includes_short_session_padding_variant() -> None:
    """The HDFS DeepLog paper registry should expose the legacy padding variant."""
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_experiment_registry(
        repo_root / "experiments" / "configs" / "registry.toml",
        repo_root=repo_root,
    )

    selected = registry.select(groups=("hdfs_deeplog_paper",))
    hdfs_wuyifan18 = next(
        experiment
        for experiment in selected
        if experiment.name == "hdfs_wuyifan18_deeplog_preprocessed"
    )

    assert hdfs_wuyifan18.models == (
        "deeplog_default",
        "deepcase",
    )
    assert hdfs_wuyifan18.model_sets == (
        "baselines_no_nb",
        "deeplog_short_session_padding_fidelity",
    )


def test_hdfs_deeplog_paper_registry_adds_drain3_ablation_group() -> None:
    """The HDFS DeepLog paper registry should expose a Drain3 ablation group."""
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_experiment_registry(
        repo_root / "experiments" / "configs" / "registry.toml",
        repo_root=repo_root,
    )

    selected = registry.select(groups=("hdfs_deeplog_paper_drain3_ablation",))

    assert [experiment.name for experiment in selected] == [
        "hdfs_v1_deeplog_paper_entry100k_split_partial_drain3",
        "hdfs_v1_deeplog_paper_entry100k_assign_first_drain3",
    ]
    assert {experiment.dataset for experiment in selected} == {
        "hdfs/v1_deeplog_paper_entry100k_split_partial_drain3",
        "hdfs/v1_deeplog_paper_entry100k_assign_first_drain3",
    }
    assert {experiment.models for experiment in selected} == {
        (
            "deeplog_default",
            "deepcase",
        ),
    }


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
    """Experiment sets should stay selectable even without a wrapper layer.

    Args:
        tmp_path (Path): Temporary directory used to build a synthetic registry
            tree.
    """
    experiments_root = tmp_path / "experiments"
    datasets_dir = experiments_root / "configs" / "datasets"
    models_dir = experiments_root / "configs" / "models"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset_manifest(datasets_dir, "demo")
    _write_dataset_manifest(datasets_dir, "alt")
    _write_dataset_manifest(datasets_dir, "paper_demo")
    _write_dataset_manifest(datasets_dir, "paper_alt")
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

    assert {experiment.name for experiment in selected} == {
        "paper_demo",
        "paper_alt",
    }
    assert all("paper_group" in experiment.groups for experiment in selected)


def test_load_experiment_registry_rejects_missing_model_config(
    tmp_path: Path,
) -> None:
    """Registry loading should fail when a referenced model config is absent.

    Args:
        tmp_path (Path): Temporary directory used to build a synthetic registry
            tree.
    """
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
    """Registry loading should fail when a referenced dataset config is absent.

    Args:
        tmp_path (Path): Temporary directory used to build a synthetic registry
            tree.
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
    """Registry loading should fail when a referenced model set is absent.

    Args:
        tmp_path (Path): Temporary directory used to build a synthetic registry
            tree.
    """
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
    """Malformed registry fields should fail during decoding.

    Args:
        tmp_path (Path): Temporary directory used to build a synthetic registry
            tree.
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
    """Named experiments should resolve to metadata-annotated bundles.

    Args:
        tmp_path (Path): Temporary directory used to build a synthetic registry
            tree.
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
    """Model overrides should flow into the resolved DeepLog bundle.

    Args:
        tmp_path (Path): Temporary directory used to build a synthetic registry
            tree.
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


def test_resolve_registry_experiment_applies_deepcase_model_set_overrides() -> None:
    """DeepCASE model-set overrides should reach the resolved model config."""
    repo_root = Path(__file__).resolve().parents[2]
    resolved = resolve_registry_experiment(
        "bgl_entity_chronological",
        registry_path=repo_root / "experiments" / "configs" / "registry.toml",
        repo_root=repo_root,
    )

    deepcase_bundles = {
        bundle.run_group: bundle
        for bundle in resolved.bundles
        if bundle.model.detector == "deepcase"
    }

    deepcase_majority_vote = deepcase_bundles["deepcase_majority_vote"]
    deepcase_threshold_fraction = deepcase_bundles["deepcase_threshold_fraction"]
    deepcase_abstain_mixed = deepcase_bundles["deepcase_abstain_mixed"]

    assert isinstance(deepcase_majority_vote.model, DeepCaseModelConfig)
    assert isinstance(deepcase_threshold_fraction.model, DeepCaseModelConfig)
    assert isinstance(deepcase_abstain_mixed.model, DeepCaseModelConfig)
    assert (
        deepcase_majority_vote.model.cluster_score_strategy
        is DeepCaseClusterScoreStrategy.MAJORITY_VOTE
    )
    assert (
        deepcase_threshold_fraction.model.cluster_score_strategy
        is DeepCaseClusterScoreStrategy.THRESHOLD_FRACTION
    )
    assert (
        deepcase_abstain_mixed.model.cluster_score_strategy
        is DeepCaseClusterScoreStrategy.ABSTAIN_MIXED
    )
    assert deepcase_majority_vote.applied_overrides == {
        "model.name": "deepcase_majority_vote",
        "model.cluster_score_strategy": "majority_vote",
    }


def test_resolve_registry_experiment_applies_deeplog_model_set_overrides() -> None:
    """DeepLog model-set overrides should reach the resolved compatibility bundle."""
    repo_root = Path(__file__).resolve().parents[2]
    resolved = resolve_registry_experiment(
        "hdfs_wuyifan18_deeplog_preprocessed",
        registry_path=repo_root / "experiments" / "configs" / "registry.toml",
        repo_root=repo_root,
    )

    deeplog_compat = next(
        bundle
        for bundle in resolved.bundles
        if bundle.run_group == "deeplog_short_session_padding_fidelity"
    )

    assert isinstance(deeplog_compat.model, DeepLogModelConfig)
    assert deeplog_compat.model.name == "deeplog_default"
    assert deeplog_compat.model.short_session_padding_fidelity is True
    assert (
        deeplog_compat.model_path
        == repo_root / "experiments" / "configs" / "models" / "deeplog_default.toml"
    )
    assert deeplog_compat.applied_overrides == {
        "model.short_session_padding_fidelity": True,
    }


def test_load_dataset_experiment_config_is_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated dataset resolution should not re-read the dataset TOML file.

    Args:
        tmp_path (Path): Temporary directory used to build a synthetic registry
            tree.
        monkeypatch (pytest.MonkeyPatch): File-read spy used to confirm the
            cache is reused.
    """
    registry_path = _write_registry_tree(tmp_path)
    registry = load_experiment_registry(registry_path, repo_root=tmp_path)
    dataset_path = tmp_path / "experiments" / "configs" / "datasets" / "demo.toml"
    read_calls = 0
    original_read_bytes = Path.read_bytes

    def _counting_read_bytes(self: Path) -> bytes:
        nonlocal read_calls
        if self == dataset_path:
            read_calls += 1
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _counting_read_bytes)

    registry.resolve_experiment(
        "demo",
        registry_path=registry_path,
        repo_root=tmp_path,
    )
    registry.resolve_experiment(
        "demo",
        registry_path=registry_path,
        repo_root=tmp_path,
    )

    assert read_calls == 1


def test_load_model_config_reference_is_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated model references should not re-read the model TOML file.

    Args:
        tmp_path (Path): Temporary directory used to build a synthetic registry
            tree.
        monkeypatch (pytest.MonkeyPatch): File-read spy used to confirm the
            cache is reused.
    """
    registry_path = _write_registry_tree(tmp_path)
    registry = load_experiment_registry(registry_path, repo_root=tmp_path)
    model_path = tmp_path / "experiments" / "configs" / "models" / "deepcase.toml"
    read_calls = 0
    original_read_bytes = Path.read_bytes

    def _counting_read_bytes(self: Path) -> bytes:
        nonlocal read_calls
        if self == model_path:
            read_calls += 1
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _counting_read_bytes)

    registry.resolve_experiment(
        "demo",
        registry_path=registry_path,
        repo_root=tmp_path,
    )
    registry.resolve_experiment(
        "demo",
        registry_path=registry_path,
        repo_root=tmp_path,
    )

    assert read_calls == 1
