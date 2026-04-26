"""Tests for experiment config loading and validation."""

from pathlib import Path

import pytest

from experiments import ConfigError
from experiments.config import (
    CSVLabelReaderConfig,
    DatasetVariantConfig,
    EntitySequenceConfig,
    LocalDirSourceConfig,
    LocalZipSourceConfig,
    RemoteZipSourceConfig,
    load_experiment_bundle,
)
from experiments.datasets import build_dataset_spec, dataset_source_summary


def _write_config_tree(
    tmp_path: Path,
    *,
    run_name: str,
    dataset: tuple[str, str],
    model: tuple[str, str],
) -> Path:
    experiments_root = tmp_path / "experiments"
    runs_dir = experiments_root / "configs" / "runs"
    datasets_dir = experiments_root / "configs" / "datasets"
    models_dir = experiments_root / "configs" / "models"
    runs_dir.mkdir(parents=True)
    datasets_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    dataset_name, dataset_body = dataset
    model_name, model_body = model
    run_path = runs_dir / f"{run_name}.toml"
    run_path.write_text(
        f'name = "{run_name}"\ndataset = "{dataset_name}"\nmodel = "{model_name}"\n',
        encoding="utf-8",
    )
    (datasets_dir / f"{dataset_name}.toml").write_text(dataset_body, encoding="utf-8")
    (models_dir / f"{model_name}.toml").write_text(model_body, encoding="utf-8")
    return run_path


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundle_resolves_dataset_and_model_configs(
    tmp_path: Path,
) -> None:
    """Run configs resolve dataset/model references under experiments/configs.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    # This protects experiment config resolution without depending on mutable
    # checked-in experiment files.
    # The experiment framework lives outside `--cov=anomalog`, so this test
    # cannot contribute line coverage to the configured coverage target.
    run_path = _write_config_tree(
        tmp_path,
        run_name="bgl_template_frequency",
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
    bundle = load_experiment_bundle(run_path)

    assert bundle.run.name == "bgl_template_frequency"
    assert bundle.dataset.name == "bgl_entity"
    assert bundle.model.name == "template_frequency_default"
    assert bundle.dataset.preset == "bgl"
    assert bundle.dataset.cache_paths is None
    assert isinstance(bundle.dataset.sequence, EntitySequenceConfig)
    assert bundle.dataset_path.name == "bgl_entity.toml"
    assert bundle.model_path.name == "template_frequency_default.toml"


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundle_supports_naive_bayes_model_configs(
    tmp_path: Path,
) -> None:
    """Naive Bayes configs should resolve through the same model loader.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    # This protects model config decoding outside the
    # `anomalog` coverage target.
    run_path = _write_config_tree(
        tmp_path,
        run_name="hdfs_v1_naive_bayes",
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
    bundle = load_experiment_bundle(run_path)

    assert bundle.run.name == "hdfs_v1_naive_bayes"
    assert bundle.dataset.name == "hdfs_v1_entity_supervised"
    assert bundle.model.name == "naive_bayes_default"
    assert bundle.model.detector == "naive_bayes"
    assert bundle.dataset.preset == "hdfs_v1"
    assert bundle.dataset.cache_paths is None


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundle_supports_river_multinomial_nb_model_configs(
    tmp_path: Path,
) -> None:
    """River model configs should resolve through the same model loader.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    # This protects model config decoding outside the
    # `anomalog` coverage target.
    run_path = _write_config_tree(
        tmp_path,
        run_name="bgl_river_multinomial_nb",
        dataset=(
            "bgl_entity_supervised",
            'name = "bgl_entity_supervised"\ndataset_name = "BGL"\npreset = "bgl"\n',
        ),
        model=(
            "river_multinomial_nb_default",
            'name = "river_multinomial_nb_default"\ndetector = "river"\n',
        ),
    )
    bundle = load_experiment_bundle(run_path)

    assert bundle.run.name == "bgl_river_multinomial_nb"
    assert bundle.dataset.name == "bgl_entity_supervised"
    assert bundle.model.name == "river_multinomial_nb_default"
    assert bundle.model.detector == "river"
    assert bundle.dataset.preset == "bgl"
    assert bundle.dataset.cache_paths is None


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundle_supports_deeplog_model_configs(
    tmp_path: Path,
) -> None:
    """DeepLog model configs should resolve through the same model loader.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    # This protects experiment model config decoding outside the configured
    # `anomalog` coverage target.
    run_path = _write_config_tree(
        tmp_path,
        run_name="bgl_deeplog",
        dataset=(
            "bgl_entity_supervised",
            'name = "bgl_entity_supervised"\ndataset_name = "BGL"\npreset = "bgl"\n',
        ),
        model=(
            "deeplog_default",
            'name = "deeplog_default"\ndetector = "deeplog"\n',
        ),
    )
    bundle = load_experiment_bundle(run_path)

    assert bundle.run.name == "bgl_deeplog"
    assert bundle.model.name == "deeplog_default"
    assert bundle.model.detector == "deeplog"
    assert bundle.dataset.preset == "bgl"


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundle_rejects_missing_model_config(tmp_path: Path) -> None:
    """Missing referenced config files should fail fast with a clear error.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    # This regression check exercises experiment config validation outside the
    # `anomalog` coverage target, so the warning is intentionally suppressed.
    experiments_root = tmp_path / "experiments"
    runs_dir = experiments_root / "configs" / "runs"
    runs_dir.mkdir(parents=True)
    run_path = runs_dir / "missing_model.toml"
    run_path.write_text(
        """name = "broken"
dataset = "bgl_entity"
model = "does_not_exist"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Config file not found"):
        load_experiment_bundle(run_path)


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
def test_load_experiment_bundle_rejects_normal_only_training_for_fixed_grouping(
    tmp_path: Path,
) -> None:
    """Fixed grouping configs should reject the normal-only training flag.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    # This protects experiment config typing outside the configured
    # `anomalog` coverage target.
    run_path = _write_config_tree(
        tmp_path,
        run_name="fixed_invalid",
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
        load_experiment_bundle(run_path)


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
