"""Tests for experiment config loading and validation."""

from pathlib import Path

import pytest

from experiments import ConfigError
from experiments.config import (
    CSVLabelReaderConfig,
    DatasetVariantConfig,
    EntitySequenceConfig,
    ExperimentBundle,
    LocalDirSourceConfig,
    LocalZipSourceConfig,
    RemoteZipSourceConfig,
    load_experiment_bundles,
)
from experiments.datasets import build_dataset_spec, dataset_source_summary


def _write_config_tree(
    tmp_path: Path,
    *,
    sweep_name: str,
    dataset: tuple[str, str],
    model: tuple[str, str],
    sweep_body_suffix: str = "",
) -> Path:
    experiments_root = tmp_path / "experiments"
    sweeps_dir = experiments_root / "configs" / "sweeps"
    datasets_dir = experiments_root / "configs" / "datasets"
    models_dir = experiments_root / "configs" / "models"
    sweeps_dir.mkdir(parents=True)
    datasets_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    dataset_name, dataset_body = dataset
    model_name, model_body = model
    sweep_path = sweeps_dir / f"{sweep_name}.toml"
    sweep_path.write_text(
        (
            f'name = "{sweep_name}"\n'
            f'dataset = "{dataset_name}"\n'
            f'model = "{model_name}"\n'
            f"{sweep_body_suffix}"
        ),
        encoding="utf-8",
    )
    (datasets_dir / f"{dataset_name}.toml").write_text(dataset_body, encoding="utf-8")
    (models_dir / f"{model_name}.toml").write_text(model_body, encoding="utf-8")
    return sweep_path


def _load_one_bundle(sweep_path: Path) -> ExperimentBundle:
    bundles = load_experiment_bundles(sweep_path)
    assert len(bundles) == 1
    return bundles[0]


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundles_resolve_dataset_and_model_configs(
    tmp_path: Path,
) -> None:
    """Sweep configs resolve dataset/model references under experiments/configs.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    # This protects experiment config resolution without depending on mutable
    # checked-in experiment files.
    # The experiment framework lives outside `--cov=anomalog`, so this test
    # cannot contribute line coverage to the configured coverage target.
    sweep_path = _write_config_tree(
        tmp_path,
        sweep_name="bgl_template_frequency",
        dataset=(
            "bgl_entity",
            (
                'name = "bgl_entity"\n'
                'dataset_name = "BGL"\n'
                'preset = "bgl"\n'
                "\n[sequence]\n"
                'grouping = "entity"\n'
                "train_fraction = 0.8\n"
                "train_on_normal_entities_only = true\n"
            ),
        ),
        model=(
            "template_frequency_default",
            'name = "template_frequency_default"\ndetector = "template_frequency"\n',
        ),
    )
    bundle = _load_one_bundle(sweep_path)

    assert bundle.sweep.name == "bgl_template_frequency"
    assert bundle.concrete_name == "bgl_template_frequency"
    assert bundle.dataset.name == "bgl_entity"
    assert bundle.model.name == "template_frequency_default"
    assert bundle.dataset.preset == "bgl"
    assert bundle.dataset.cache_paths is None
    assert isinstance(bundle.dataset.sequence, EntitySequenceConfig)
    assert bundle.dataset_path.name == "bgl_entity.toml"
    assert bundle.model_path.name == "template_frequency_default.toml"


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundles_support_naive_bayes_model_configs(
    tmp_path: Path,
) -> None:
    """Naive Bayes configs should resolve through the same model loader.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    # This protects model config decoding outside the
    # `anomalog` coverage target.
    sweep_path = _write_config_tree(
        tmp_path,
        sweep_name="hdfs_v1_naive_bayes",
        dataset=(
            "hdfs_v1_entity_supervised",
            (
                'name = "hdfs_v1_entity_supervised"\n'
                'dataset_name = "HDFS_v1"\n'
                'preset = "hdfs_v1"\n'
            ),
        ),
        model=(
            "naive_bayes_default",
            'name = "naive_bayes_default"\ndetector = "naive_bayes"\n',
        ),
    )
    bundle = _load_one_bundle(sweep_path)

    assert bundle.sweep.name == "hdfs_v1_naive_bayes"
    assert bundle.dataset.name == "hdfs_v1_entity_supervised"
    assert bundle.model.name == "naive_bayes_default"
    assert bundle.model.detector == "naive_bayes"
    assert bundle.dataset.preset == "hdfs_v1"
    assert bundle.dataset.cache_paths is None


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundles_support_river_multinomial_nb_model_configs(
    tmp_path: Path,
) -> None:
    """River model configs should resolve through the same model loader.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    # This protects model config decoding outside the
    # `anomalog` coverage target.
    sweep_path = _write_config_tree(
        tmp_path,
        sweep_name="bgl_river_multinomial_nb",
        dataset=(
            "bgl_entity",
            (
                'name = "bgl_entity"\n'
                'dataset_name = "BGL"\n'
                'preset = "bgl"\n'
                "\n[sequence]\n"
                'grouping = "entity"\n'
                "train_on_normal_entities_only = true\n"
            ),
        ),
        model=(
            "river_multinomial_nb_default",
            'name = "river_multinomial_nb_default"\ndetector = "river"\n',
        ),
        sweep_body_suffix=(
            '\n[overrides]\n"dataset.sequence.train_on_normal_entities_only" = false\n'
        ),
    )
    bundle = _load_one_bundle(sweep_path)

    assert bundle.sweep.name == "bgl_river_multinomial_nb"
    assert bundle.dataset.name == "bgl_entity"
    assert bundle.model.name == "river_multinomial_nb_default"
    assert bundle.model.detector == "river"
    assert bundle.dataset.preset == "bgl"
    assert bundle.dataset.cache_paths is None
    assert isinstance(bundle.dataset.sequence, EntitySequenceConfig)
    assert bundle.dataset.sequence.train_on_normal_entities_only is False


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundles_support_deeplog_model_configs(
    tmp_path: Path,
) -> None:
    """DeepLog model configs should resolve through the same model loader.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    # This protects experiment model config decoding outside the configured
    # `anomalog` coverage target.
    sweep_path = _write_config_tree(
        tmp_path,
        sweep_name="bgl_deeplog",
        dataset=(
            "bgl_entity",
            'name = "bgl_entity"\ndataset_name = "BGL"\npreset = "bgl"\n',
        ),
        model=(
            "deeplog_default",
            'name = "deeplog_default"\ndetector = "deeplog"\n',
        ),
    )
    bundle = _load_one_bundle(sweep_path)

    assert bundle.sweep.name == "bgl_deeplog"
    assert bundle.model.name == "deeplog_default"
    assert bundle.model.detector == "deeplog"
    assert bundle.dataset.preset == "bgl"


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundles_expands_model_and_dataset_axes(
    tmp_path: Path,
) -> None:
    """Sweep axes should expand into concrete bundles across model choices.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    # This protects sweep expansion outside the configured `anomalog` coverage
    # target.
    sweep_path = _write_config_tree(
        tmp_path,
        sweep_name="bgl_model_matrix",
        dataset=(
            "bgl_entity",
            (
                'name = "bgl_entity"\n'
                'dataset_name = "BGL"\n'
                'preset = "bgl"\n'
                "\n[sequence]\n"
                'grouping = "entity"\n'
                "train_fraction = 0.2\n"
                "train_on_normal_entities_only = true\n"
            ),
        ),
        model=(
            "template_frequency_default",
            'name = "template_frequency_default"\ndetector = "template_frequency"\n',
        ),
        sweep_body_suffix=(
            '\n[[axes]]\npath = "sweep.model"\n'
            'values = ["template_frequency_default", "deeplog_default"]\n'
            '\n[[axes]]\npath = "dataset.sequence.train_fraction"\n'
            "values = [0.2, 0.4]\n"
        ),
    )
    models_dir = sweep_path.parent.parent / "models"
    (models_dir / "deeplog_default.toml").write_text(
        'name = "deeplog_default"\ndetector = "deeplog"\n',
        encoding="utf-8",
    )

    bundles = load_experiment_bundles(sweep_path)

    assert [bundle.concrete_name for bundle in bundles] == [
        "bgl_template_frequency_train_fraction_0p2",
        "bgl_template_frequency_train_fraction_0p4",
        "bgl_deeplog_train_fraction_0p2",
        "bgl_deeplog_train_fraction_0p4",
    ]
    assert {
        (bundle.model.name, bundle.dataset.sequence.train_fraction)
        for bundle in bundles
    } == {
        ("template_frequency_default", 0.2),
        ("template_frequency_default", 0.4),
        ("deeplog_default", 0.2),
        ("deeplog_default", 0.4),
    }


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundles_defaults_max_workers_to_auto(
    tmp_path: Path,
) -> None:
    """Sweep configs should use auto worker selection unless overridden.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    # This protects experiment config defaults outside the configured
    # `anomalog` coverage target.
    sweep_path = _write_config_tree(
        tmp_path,
        sweep_name="bgl_template_frequency",
        dataset=(
            "bgl_entity",
            'name = "bgl_entity"\ndataset_name = "BGL"\npreset = "bgl"\n',
        ),
        model=(
            "template_frequency_default",
            'name = "template_frequency_default"\ndetector = "template_frequency"\n',
        ),
    )

    [bundle] = load_experiment_bundles(sweep_path)

    assert bundle.sweep.max_workers == "auto"


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundles_reject_missing_model_config(tmp_path: Path) -> None:
    """Missing referenced config files should fail fast with a clear error.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    # This regression check exercises experiment config validation outside the
    # `anomalog` coverage target, so the warning is intentionally suppressed.
    experiments_root = tmp_path / "experiments"
    sweeps_dir = experiments_root / "configs" / "sweeps"
    sweeps_dir.mkdir(parents=True)
    sweep_path = sweeps_dir / "missing_model.toml"
    sweep_path.write_text(
        """name = "broken"
dataset = "bgl_entity"
model = "does_not_exist"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Config file not found"):
        _load_one_bundle(sweep_path)


@pytest.mark.allow_no_new_coverage
def test_build_dataset_spec_applies_label_reader_for_custom_datasets() -> None:
    """Custom dataset variants should not drop an explicit label reader."""
    # This protects experiment-layer dataset assembly outside the configured
    # `anomalog` coverage target.
    spec = build_dataset_spec(
        DatasetVariantConfig(
            name="custom_demo",
            dataset_name="custom-demo",
            source=LocalDirSourceConfig(path=Path()),
            structured_parser="bgl",
            label_reader=CSVLabelReaderConfig(relative_path=Path("labels.csv")),
        ),
        repo_root=Path("/repo"),
    )

    assert spec.anomaly_label_reader is not None


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundles_reject_normal_only_training_for_fixed_grouping(
    tmp_path: Path,
) -> None:
    """Fixed grouping configs should reject the normal-only training flag.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    # This protects experiment config typing outside the configured
    # `anomalog` coverage target.
    sweep_path = _write_config_tree(
        tmp_path,
        sweep_name="fixed_invalid",
        dataset=(
            "fixed_invalid",
            (
                'name = "fixed_invalid"\n'
                'dataset_name = "demo"\n'
                'preset = "bgl"\n'
                "\n[sequence]\n"
                'grouping = "fixed"\n'
                "window_size = 4\n"
                "train_on_normal_entities_only = true\n"
            ),
        ),
        model=(
            "template_frequency_default",
            'name = "template_frequency_default"\ndetector = "template_frequency"\n',
        ),
    )

    with pytest.raises(ConfigError, match="Object contains unknown field"):
        _load_one_bundle(sweep_path)


@pytest.mark.allow_no_new_coverage
def test_dataset_source_summary_uses_config_layer_manifest_entries() -> None:
    """Dataset manifests should come from config-layer source metadata."""
    # This protects experiment-layer manifest shaping outside the configured
    # `anomalog` coverage target.
    repo_root = Path("/repo")

    assert dataset_source_summary(
        DatasetVariantConfig(
            name="local-dir",
            dataset_name="demo",
            source=LocalDirSourceConfig(
                path=Path("datasets/demo"),
                raw_logs_relpath=Path("BGL.log"),
            ),
            structured_parser="bgl",
        ),
        repo_root=repo_root,
    ) == {
        "type": "local_dir",
        "path": "/repo/datasets/demo",
        "raw_logs_relpath": "BGL.log",
    }
    assert dataset_source_summary(
        DatasetVariantConfig(
            name="local-zip",
            dataset_name="demo",
            source=LocalZipSourceConfig(
                zip_path=Path("archives/demo.zip"),
                raw_logs_relpath=Path("BGL.log"),
                md5_checksum="abc123",
            ),
            structured_parser="bgl",
        ),
        repo_root=repo_root,
    ) == {
        "type": "local_zip",
        "zip_path": "/repo/archives/demo.zip",
        "raw_logs_relpath": "BGL.log",
        "md5_checksum": "abc123",
    }
    assert dataset_source_summary(
        DatasetVariantConfig(
            name="remote-zip",
            dataset_name="demo",
            source=RemoteZipSourceConfig(
                url="https://example.com/demo.zip",
                md5_checksum="abc123",
                raw_logs_relpath=Path("BGL.log"),
            ),
            structured_parser="bgl",
        ),
        repo_root=repo_root,
    ) == {
        "type": "remote_zip",
        "url": "https://example.com/demo.zip",
        "raw_logs_relpath": "BGL.log",
        "md5_checksum": "abc123",
    }
