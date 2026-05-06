# ruff: noqa: PLR2004
"""Tests for experiment config loading and validation."""

from pathlib import Path

import pytest

from experiments import ConfigError
from experiments.audit import (
    validate_deepcase_bgl_extension_config,
    validate_deepcase_hdfs_table_iv_config,
    validate_deepcase_hdfs_table_x_config,
    validate_deeplog_paper_config,
)
from experiments.config import (
    ChronologicalStreamSequenceConfig,
    CSVLabelReaderConfig,
    DatasetVariantConfig,
    EntitySequenceConfig,
    ExperimentBundle,
    LocalDirSourceConfig,
    LocalZipSourceConfig,
    RawEntryPrefixCountSplitConfig,
    RawEntryPrefixFractionSplitConfig,
    RawEntryPrefixNormalFractionSplitConfig,
    RemoteZipSourceConfig,
    load_experiment_bundles,
)
from experiments.datasets import build_dataset_spec, dataset_source_summary
from experiments.models.deepcase.detector import DeepCaseModelConfig
from experiments.models.deeplog.detector import DeepLogModelConfig


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
                "test_fraction = 0.2\n"
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
                "test_fraction = 0.5\n"
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
        (
            bundle.model.name,
            bundle.dataset.sequence.train_fraction,
            bundle.dataset.sequence.test_fraction,
        )
        for bundle in bundles
    } == {
        ("template_frequency_default", 0.2, 0.5),
        ("template_frequency_default", 0.4, 0.5),
        ("deeplog_default", 0.2, 0.5),
        ("deeplog_default", 0.4, 0.5),
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


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundles_supports_chronological_stream_grouping(
    tmp_path: Path,
) -> None:
    """Chronological-stream configs should decode through the shared loader.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    sweep_path = _write_config_tree(
        tmp_path,
        sweep_name="stream_grouping",
        dataset=(
            "stream_grouping",
            (
                'name = "stream_grouping"\n'
                'dataset_name = "demo"\n'
                'preset = "bgl"\n'
                "\n[sequence]\n"
                'grouping = "chronological_stream"\n'
                "chunk_size = 7\n"
            ),
        ),
        model=(
            "template_frequency_default",
            'name = "template_frequency_default"\ndetector = "template_frequency"\n',
        ),
    )

    bundle = _load_one_bundle(sweep_path)
    expected_chunk_size = 7

    assert isinstance(bundle.dataset.sequence, ChronologicalStreamSequenceConfig)
    assert bundle.dataset.sequence.chunk_size == expected_chunk_size


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundles_supports_raw_entry_prefix_splits(
    tmp_path: Path,
) -> None:
    """Raw-entry split configs should decode into the shared sequence model.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    sweep_path = _write_config_tree(
        tmp_path,
        sweep_name="raw_entry_prefix",
        dataset=(
            "raw_entry_prefix",
            (
                'name = "raw_entry_prefix"\n'
                'dataset_name = "demo"\n'
                'preset = "bgl"\n'
                "\n[sequence]\n"
                'grouping = "entity"\n'
                "\n[sequence.split]\n"
                'mode = "raw_entry_prefix_count"\n'
                "train_entry_count = 100\n"
                'application_order = "before_grouping"\n'
                'straddling_group_policy = "split_partial_sequences"\n'
            ),
        ),
        model=(
            "template_frequency_default",
            'name = "template_frequency_default"\ndetector = "template_frequency"\n',
        ),
    )

    bundle = _load_one_bundle(sweep_path)
    expected_train_entry_count = 100

    assert bundle.dataset.sequence.split is not None
    assert isinstance(bundle.dataset.sequence.split, RawEntryPrefixCountSplitConfig)
    assert bundle.dataset.sequence.split.train_entry_count == expected_train_entry_count


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundles_supports_raw_entry_prefix_normal_fraction_splits(
    tmp_path: Path,
) -> None:
    """Normal-entry raw prefix splits should decode through the shared loader.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    sweep_path = _write_config_tree(
        tmp_path,
        sweep_name="raw_entry_prefix_normal_fraction",
        dataset=(
            "raw_entry_prefix_normal_fraction",
            (
                'name = "raw_entry_prefix_normal_fraction"\n'
                'dataset_name = "demo"\n'
                'preset = "bgl"\n'
                "\n[sequence]\n"
                'grouping = "chronological_stream"\n'
                "\n[sequence.split]\n"
                'mode = "raw_entry_prefix_normal_fraction"\n'
                "train_normal_entry_fraction = 0.01\n"
                'application_order = "before_grouping"\n'
            ),
        ),
        model=(
            "template_frequency_default",
            'name = "template_frequency_default"\ndetector = "template_frequency"\n',
        ),
    )

    bundle = _load_one_bundle(sweep_path)
    expected_train_normal_entry_fraction = pytest.approx(0.01)

    assert bundle.dataset.sequence.split is not None
    assert isinstance(
        bundle.dataset.sequence.split,
        RawEntryPrefixNormalFractionSplitConfig,
    )
    assert (
        bundle.dataset.sequence.split.train_normal_entry_fraction
        == expected_train_normal_entry_fraction
    )


@pytest.mark.allow_no_new_coverage
def test_deeplog_paper_configs_pin_expected_protocols() -> None:
    """Paper reproduction configs should keep their declared split semantics."""
    repo_root = Path(__file__).resolve().parents[2]

    bgl_1pct_bundle = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "sweeps"
        / "bgl_deeplog_paper_1pct_normal_entry_stream_no_online.toml",
    )[0]
    bgl_10pct_bundle = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "sweeps"
        / "bgl_deeplog_paper_10pct_entry_stream_no_online.toml",
    )[0]
    hdfs_bundle = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "sweeps"
        / "hdfs_v1_deeplog_paper_entry100k_split_partial.toml",
    )[0]

    validate_deeplog_paper_config(
        dataset_config=bgl_1pct_bundle.dataset,
        model_config=bgl_1pct_bundle.model,
    )
    validate_deeplog_paper_config(
        dataset_config=bgl_10pct_bundle.dataset,
        model_config=bgl_10pct_bundle.model,
    )
    validate_deeplog_paper_config(
        dataset_config=hdfs_bundle.dataset,
        model_config=hdfs_bundle.model,
    )

    assert isinstance(
        bgl_1pct_bundle.dataset.sequence,
        ChronologicalStreamSequenceConfig,
    )
    assert bgl_1pct_bundle.dataset.sequence.chunk_size == 100_000
    assert isinstance(
        bgl_1pct_bundle.dataset.sequence.split,
        RawEntryPrefixNormalFractionSplitConfig,
    )
    assert (
        bgl_1pct_bundle.dataset.sequence.split.application_order.value
        == "before_grouping"
    )
    assert (
        bgl_1pct_bundle.dataset.sequence.split.train_normal_entry_fraction
        == pytest.approx(0.01)
    )
    assert isinstance(bgl_1pct_bundle.model, DeepLogModelConfig)
    assert bgl_1pct_bundle.model.history_size == 3
    assert bgl_1pct_bundle.model.top_g == 6
    assert bgl_1pct_bundle.model.num_layers == 1
    assert bgl_1pct_bundle.model.hidden_size == 256

    assert isinstance(
        bgl_10pct_bundle.dataset.sequence.split,
        RawEntryPrefixFractionSplitConfig,
    )
    assert (
        bgl_10pct_bundle.dataset.sequence.split.application_order.value
        == "before_grouping"
    )
    assert (
        bgl_10pct_bundle.dataset.sequence.split.train_entry_fraction
        == pytest.approx(0.10)
    )
    assert isinstance(bgl_10pct_bundle.model, DeepLogModelConfig)
    assert bgl_10pct_bundle.model.history_size == 3
    assert bgl_10pct_bundle.model.top_g == 6
    assert bgl_10pct_bundle.model.num_layers == 1
    assert bgl_10pct_bundle.model.hidden_size == 256

    assert isinstance(hdfs_bundle.dataset.sequence, EntitySequenceConfig)
    assert isinstance(
        hdfs_bundle.dataset.sequence.split,
        RawEntryPrefixCountSplitConfig,
    )
    assert (
        hdfs_bundle.dataset.sequence.split.application_order.value == "before_grouping"
    )
    assert hdfs_bundle.dataset.sequence.split.train_entry_count == 100_000
    assert hdfs_bundle.dataset.sequence.train_fraction == pytest.approx(0.01)
    assert hdfs_bundle.dataset.sequence.test_fraction == pytest.approx(0.99)
    assert isinstance(hdfs_bundle.model, DeepLogModelConfig)
    assert hdfs_bundle.model.history_size == 10
    assert hdfs_bundle.model.top_g == 9
    assert hdfs_bundle.model.num_layers == 2
    assert hdfs_bundle.model.hidden_size == 64


@pytest.mark.allow_no_new_coverage
def test_deepcase_paper_configs_pin_expected_protocols() -> None:
    """DeepCASE paper and extension configs should keep their declared contracts."""
    repo_root = Path(__file__).resolve().parents[2]

    hdfs_table_iv_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "sweeps"
        / "hdfs_v1_deepcase_table_iv_prediction.toml",
    )
    hdfs_table_x_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "sweeps"
        / "hdfs_v1_deepcase_table_x_workload.toml",
    )
    bgl_extension_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "sweeps"
        / "bgl_deepcase_event_level_extension.toml",
    )

    assert len(hdfs_table_iv_bundles) == 10
    assert len(hdfs_table_x_bundles) == 10
    assert len(bgl_extension_bundles) == 10

    for expected_seed, bundle in enumerate(hdfs_table_iv_bundles):
        assert isinstance(bundle.model, DeepCaseModelConfig)
        assert bundle.model.random_seed == expected_seed
    for expected_seed, bundle in enumerate(hdfs_table_x_bundles):
        assert isinstance(bundle.model, DeepCaseModelConfig)
        assert bundle.model.random_seed == expected_seed
    for expected_seed, bundle in enumerate(bgl_extension_bundles):
        assert isinstance(bundle.model, DeepCaseModelConfig)
        assert bundle.model.random_seed == expected_seed

    for bundle in hdfs_table_iv_bundles:
        assert isinstance(bundle.model, DeepCaseModelConfig)
        assert isinstance(bundle.dataset.sequence, EntitySequenceConfig)
        validate_deepcase_hdfs_table_iv_config(
            dataset_config=bundle.dataset,
            model_config=bundle.model,
        )
        assert bundle.model.epochs == 100
        assert bundle.model.context_length == 10
        assert bundle.model.timeout_seconds == 86_400
        assert bundle.model.hidden_size == 128
        assert bundle.model.label_smoothing_delta == pytest.approx(0.1)
        assert bundle.model.confidence_threshold == pytest.approx(0.2)
        assert bundle.model.eps == pytest.approx(0.1)
        assert bundle.model.min_samples == 5
        assert bundle.dataset.sequence.train_fraction == pytest.approx(0.2)
        assert bundle.dataset.sequence.test_fraction == pytest.approx(0.8)
        assert bundle.dataset.sequence.train_on_normal_entities_only is False

    for bundle in hdfs_table_x_bundles:
        assert isinstance(bundle.model, DeepCaseModelConfig)
        assert isinstance(bundle.dataset.sequence, EntitySequenceConfig)
        validate_deepcase_hdfs_table_x_config(
            dataset_config=bundle.dataset,
            model_config=bundle.model,
        )
        assert bundle.model.epochs == 100

    for bundle in bgl_extension_bundles:
        assert isinstance(bundle.model, DeepCaseModelConfig)
        validate_deepcase_bgl_extension_config(
            dataset_config=bundle.dataset,
            model_config=bundle.model,
        )
        assert bundle.model.epochs == 100
        assert isinstance(bundle.dataset.sequence, EntitySequenceConfig)
        assert bundle.dataset.sequence.train_fraction == pytest.approx(0.2)
        assert bundle.dataset.sequence.test_fraction == pytest.approx(0.8)
        assert bundle.dataset.sequence.train_on_normal_entities_only is False
