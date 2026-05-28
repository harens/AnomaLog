# ruff: noqa: PLR2004
"""Tests for experiment config loading and validation."""

import zipfile
from pathlib import Path

import msgspec
import pytest

from anomalog.parsers.template import IdentityTemplateParser
from anomalog.sources.deeplog_preprocessed import (
    materialise_labelled_raw_stream,
    materialise_labelled_session_stream,
)
from experiments import ConfigError
from experiments.audit import (
    validate_bgl_how_far_are_we_2022_config,
    validate_deepcase_bgl_extension_config,
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
    TimeSequenceConfig,
    load_experiment_bundles,
)
from experiments.config_types import CachePathsConfigModel
from experiments.datasets import build_dataset_spec, dataset_source_summary
from experiments.models.deepcase.detector import DeepCaseModelConfig
from experiments.models.deeplog.detector import DeepLogModelConfig
from experiments.models.markov import MarkovModelConfig
from experiments.models.metric_schema import EvaluationUnit
from experiments.models.template_frequency import TemplateFrequencyModelConfig


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
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    dataset_name, dataset_body = dataset
    model_name, model_body = model
    sweep_path = sweeps_dir / f"{sweep_name}.toml"
    dataset_lines = []
    for line in dataset_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            header = stripped[1:-1]
            if header.startswith("dataset."):
                dataset_lines.append(line)
            else:
                dataset_lines.append(line.replace(f"[{header}]", f"[dataset.{header}]"))
        else:
            dataset_lines.append(line)
    dataset_block = "\n".join(dataset_lines)
    sweep_path.write_text(
        (
            f'name = "{sweep_name}"\n'
            "[dataset]\n"
            f"{dataset_block}\n"
            f"\n[[models]]\n"
            f"{model_body}"
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


def _assert_template_frequency_baseline_bundle(bundle: ExperimentBundle) -> None:
    assert isinstance(bundle.model, TemplateFrequencyModelConfig)


def _assert_markov_baseline_bundle(bundle: ExperimentBundle) -> None:
    assert isinstance(bundle.model, MarkovModelConfig)


def _assert_naive_bayes_baseline_bundle(bundle: ExperimentBundle) -> None:
    assert bundle.model.name == "naive_bayes_default"


def _assert_bgl_1pct_deeplog_bundle(bundle: ExperimentBundle) -> None:
    assert isinstance(bundle.dataset.sequence, ChronologicalStreamSequenceConfig)
    assert isinstance(
        bundle.dataset.sequence.split,
        RawEntryPrefixNormalFractionSplitConfig,
    )
    assert bundle.dataset.sequence.split.application_order.value == "before_grouping"
    assert isinstance(bundle.model, DeepLogModelConfig)
    assert bundle.run_group == "deeplog_default"


def _assert_bgl_10pct_deeplog_bundle(bundle: ExperimentBundle) -> None:
    assert isinstance(
        bundle.dataset.sequence.split,
        RawEntryPrefixNormalFractionSplitConfig,
    )
    assert bundle.dataset.sequence.split.application_order.value == "before_grouping"
    assert isinstance(bundle.model, DeepLogModelConfig)
    assert bundle.run_group == "deeplog_default"


def _assert_hdfs_deeplog_bundle(bundle: ExperimentBundle) -> None:
    assert isinstance(bundle.dataset.sequence, EntitySequenceConfig)
    assert isinstance(
        bundle.dataset.sequence.split,
        RawEntryPrefixCountSplitConfig,
    )
    assert bundle.dataset.sequence.split.application_order.value == "before_grouping"
    assert isinstance(bundle.model, DeepLogModelConfig)
    assert bundle.run_group == "deeplog_default"


def _write_preprocessed_hdfs_archive(
    tmp_path: Path,
    *,
    train_text: str,
    normal_text: str,
    abnormal_text: str,
) -> tuple[Path, Path, int]:
    """Create a tiny archive containing the already preprocessed HDFS stream.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for the archive fixture.
        train_text (str): Raw session lines written to `hdfs_train`.
        normal_text (str): Raw session lines written to `hdfs_test_normal`.
        abnormal_text (str): Raw session lines written to `hdfs_test_abnormal`.

    Returns:
        tuple[Path, Path, int]: Source root, synthetic archive, and raw-entry
            count contributed by `hdfs_train`.
    """
    source_root = tmp_path / "hdfs_preprocessed_source"
    source_root.mkdir()
    (source_root / "hdfs_train").write_text(train_text, encoding="utf-8")
    (source_root / "hdfs_test_normal").write_text(normal_text, encoding="utf-8")
    (source_root / "hdfs_test_abnormal").write_text(
        abnormal_text,
        encoding="utf-8",
    )

    raw_logs_path = tmp_path / "preprocessed" / "hdfs_events.log"
    raw_logs_path.parent.mkdir(parents=True, exist_ok=True)
    materialise_labelled_session_stream(
        source_root=source_root,
        raw_logs_path=raw_logs_path,
        split_files=(
            ("hdfs_train", 0),
            ("hdfs_test_normal", 0),
            ("hdfs_test_abnormal", 1),
        ),
    )

    archive_path = tmp_path / "hdfs_preprocessed.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(raw_logs_path, arcname="preprocessed/hdfs_events.log")

    train_entry_count = sum(
        len(line.split()) for line in train_text.splitlines() if line
    )
    return source_root, archive_path, train_entry_count


def _write_openstack_labelled_raw_archive(
    tmp_path: Path,
    *,
    train_text: str,
    normal_text: str,
    abnormal_text: str,
) -> tuple[Path, Path, int]:
    """Create a tiny archive containing a labelled raw OpenStack stream.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for the archive fixture.
        train_text (str): Raw rows written to `openstack_normal1.log`.
        normal_text (str): Raw rows written to `openstack_normal2.log`.
        abnormal_text (str): Raw rows written to `openstack_abnormal.log`.

    Returns:
        tuple[Path, Path, int]: Source root, synthetic archive, and train row
            count contributed by `openstack_normal1.log`.
    """
    source_root = tmp_path / "openstack_source"
    source_root.mkdir()
    (source_root / "openstack_normal1.log").write_text(train_text, encoding="utf-8")
    (source_root / "openstack_normal2.log").write_text(normal_text, encoding="utf-8")
    (source_root / "openstack_abnormal.log").write_text(abnormal_text, encoding="utf-8")

    raw_logs_path = tmp_path / "preprocessed" / "openstack_labelled_raw.log"
    raw_logs_path.parent.mkdir(parents=True, exist_ok=True)
    materialise_labelled_raw_stream(
        source_root=source_root,
        raw_logs_path=raw_logs_path,
        split_files=(
            ("openstack_normal1.log", "openstack_train", 0),
            ("openstack_normal2.log", "openstack_test_normal", 0),
            ("openstack_abnormal.log", "openstack_test_abnormal", 1),
        ),
    )

    archive_path = tmp_path / "openstack_preprocessed.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(
            raw_logs_path,
            arcname="preprocessed/openstack_labelled_raw.log",
        )

    train_entry_count = len([line for line in train_text.splitlines() if line])
    return source_root, archive_path, train_entry_count


def _read_token_sessions(path: Path) -> list[list[str]]:
    return [
        line.strip().split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _model_input_signature(
    *,
    bundle: ExperimentBundle,
    repo_root: Path,
) -> list[tuple[str, str, int, list[str]]]:
    sequences = list(
        bundle.dataset.sequence.apply(
            build_dataset_spec(bundle.dataset, repo_root=repo_root).build(),
        ),
    )
    return [
        (
            sequence.entity_ids[0],
            sequence.split_label.value,
            sequence.label,
            sequence.templates,
        )
        for sequence in sequences
    ]


def _expected_hdfs_split_signature(
    *,
    sessions_by_split: dict[str, list[list[str]]],
) -> list[tuple[str, str, int, list[str]]]:
    return (
        [
            (f"hdfs_train:{index}", "train", 0, session)
            for index, session in enumerate(sessions_by_split["hdfs_train"])
        ]
        + [
            (f"hdfs_test_normal:{index}", "test", 0, session)
            for index, session in enumerate(sessions_by_split["hdfs_test_normal"])
        ]
        + [
            (f"hdfs_test_abnormal:{index}", "test", 1, session)
            for index, session in enumerate(sessions_by_split["hdfs_test_abnormal"])
        ]
    )


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
        sweep_name="bgl_template_frequency_chronological",
        dataset=(
            "bgl_entity_chronological",
            (
                'name = "bgl_entity_chronological"\n'
                'dataset_name = "BGL"\n'
                'preset = "bgl"\n'
                "\n[sequence]\n"
                'grouping = "entity"\n'
                "train_fraction = 0.8\n"
                "test_fraction = 0.2\n"
            ),
        ),
        model=(
            "template_frequency_default",
            'name = "template_frequency_default"\ndetector = "template_frequency"\n',
        ),
    )
    bundle = _load_one_bundle(sweep_path)

    assert bundle.sweep.name == "bgl_template_frequency_chronological"
    assert bundle.concrete_name == "bgl_template_frequency_chronological"
    assert bundle.dataset.name == "bgl_entity_chronological"
    assert bundle.model.name == "template_frequency_default"
    assert bundle.dataset.preset == "bgl"
    assert bundle.dataset.cache_paths is None
    assert isinstance(bundle.dataset.sequence, EntitySequenceConfig)
    assert bundle.dataset_path == sweep_path
    assert bundle.model_path == sweep_path


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
        sweep_name="hdfs_v1_naive_bayes_chronological",
        dataset=(
            "hdfs_v1_entity_chronological",
            (
                'name = "hdfs_v1_entity_chronological"\n'
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

    assert bundle.sweep.name == "hdfs_v1_naive_bayes_chronological"
    assert bundle.dataset.name == "hdfs_v1_entity_chronological"
    assert bundle.model.name == "naive_bayes_default"
    assert bundle.model.detector == "naive_bayes"
    assert bundle.dataset.preset == "hdfs_v1"
    assert bundle.dataset.cache_paths is None


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
            "bgl_entity_normal_only",
            ('name = "bgl_entity_normal_only"\ndataset_name = "BGL"\npreset = "bgl"\n'),
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


def test_load_experiment_bundles_preserves_model_run_group(
    tmp_path: Path,
) -> None:
    """Inline model entries should keep their scheduling group metadata.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    sweep_path = _write_config_tree(
        tmp_path,
        sweep_name="bgl_grouped_models",
        dataset=(
            "bgl_entity_grouped",
            ('name = "bgl_entity_grouped"\ndataset_name = "BGL"\npreset = "bgl"\n'),
        ),
        model=(
            "template_frequency_default",
            'name = "template_frequency_default"\ndetector = "template_frequency"\n',
        ),
        sweep_body_suffix='run_group = "baselines"\n',
    )
    bundle = _load_one_bundle(sweep_path)

    assert bundle.run_group == "baselines"
    assert bundle.model.name == "template_frequency_default"


def test_load_experiment_bundles_supports_extends(
    tmp_path: Path,
) -> None:
    """Child manifests should inherit shared dataset and model boilerplate.

    Args:
        tmp_path (Path): Temporary directory used to build the manifest tree.
    """
    experiments_root = tmp_path / "experiments"
    datasets_dir = experiments_root / "configs" / "datasets"
    models_dir = experiments_root / "configs" / "models"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (datasets_dir / "base.toml").write_text(
        (
            'name = "base"\n'
            'description = "base"\n'
            "\n[dataset]\n"
            'template_parser = "drain3"\n'
            'description = "base dataset"\n'
            "\n[dataset.sequence]\n"
            'grouping = "entity"\n'
            "train_fraction = 0.2\n"
            "train_on_normal_entities_only = false\n"
            "\n[[models]]\n"
            'ref = "template_frequency_default"\n'
            'run_group = "baselines"\n'
        ),
        encoding="utf-8",
    )
    (datasets_dir / "child.toml").write_text(
        (
            'extends = "base.toml"\n'
            'name = "child"\n'
            "\n[dataset]\n"
            'name = "child_dataset"\n'
            'dataset_name = "CHILD"\n'
            'preset = "bgl"\n'
            'description = "child dataset"\n'
            "\n[dataset.cache_paths]\n"
            'namespace = "child"\n'
        ),
        encoding="utf-8",
    )
    (models_dir / "template_frequency_default.toml").write_text(
        'name = "template_frequency_default"\ndetector = "template_frequency"\n',
        encoding="utf-8",
    )

    bundle = _load_one_bundle(datasets_dir / "child.toml")

    assert bundle.sweep.name == "child"
    assert bundle.dataset.name == "child_dataset"
    assert bundle.dataset.dataset_name == "CHILD"
    assert bundle.dataset.preset == "bgl"
    assert bundle.dataset.cache_paths is not None
    assert bundle.dataset.cache_paths.namespace == "child"
    assert bundle.model.name == "template_frequency_default"
    assert bundle.run_group == "baselines"


def test_cache_paths_namespace_uses_cluster_base_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Namespace expansion should honour the configured cluster base roots."""
    monkeypatch.setenv("ANOMALOG_DATA_ROOT", (tmp_path / "data-base").as_posix())
    monkeypatch.setenv("ANOMALOG_CACHE_ROOT", (tmp_path / "cache-base").as_posix())

    resolved = CachePathsConfigModel(namespace="child").resolve(repo_root=tmp_path)

    assert resolved.data_root == tmp_path / "data-base" / "child"
    assert resolved.cache_root == tmp_path / "cache-base" / "child"


def test_load_experiment_bundles_rejects_missing_extends_target(
    tmp_path: Path,
) -> None:
    """Missing inherited manifests should fail with a clear config error.

    Args:
        tmp_path (Path): Temporary directory used to build the manifest tree.
    """
    experiments_root = tmp_path / "experiments"
    datasets_dir = experiments_root / "configs" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    child_path = datasets_dir / "child.toml"
    child_path.write_text(
        'extends = "missing.toml"\nname = "child"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Missing"):
        load_experiment_bundles(child_path)


@pytest.mark.allow_no_new_coverage
def test_ait_ads_base_manifest_uses_combined_chronological_stream() -> None:
    """The combined AIT-ADS base manifest should resolve the paper chronology."""
    paper_bundles = load_experiment_bundles(
        Path("experiments/configs/datasets") / "ait_ads/base.toml",
    )

    assert {bundle.model.detector for bundle in paper_bundles} >= {
        "template_frequency",
        "markov",
    }
    assert all(
        isinstance(bundle.dataset.sequence, ChronologicalStreamSequenceConfig)
        for bundle in paper_bundles
    )
    assert all(
        bundle.dataset.sequence.train_fraction == pytest.approx(0.5)
        for bundle in paper_bundles
    )
    assert all(
        bundle.dataset.sequence.test_fraction == pytest.approx(0.5)
        for bundle in paper_bundles
    )
    assert all(
        bundle.dataset.evaluation_unit is EvaluationUnit.CONTINUOUS_EVENT_STREAM
        for bundle in paper_bundles
    )
    assert {bundle.dataset.preset for bundle in paper_bundles} == {"ait_ads"}


@pytest.mark.allow_no_new_coverage
def test_ait_ads_entity_manifest_uses_entity_grouping_with_chronological_split() -> (
    None
):
    """AIT-ADS entity-local runs should split chronologically before grouping."""
    paper_bundles = load_experiment_bundles(
        Path("experiments/configs/datasets") / "ait_ads/entity_chronological.toml",
    )

    assert {bundle.model.detector for bundle in paper_bundles} >= {
        "deeplog",
        "deepcase",
    }
    assert all(
        isinstance(bundle.dataset.sequence, EntitySequenceConfig)
        for bundle in paper_bundles
    )
    for bundle in paper_bundles:
        split = bundle.dataset.sequence.split
        assert isinstance(split, RawEntryPrefixFractionSplitConfig)
        assert bundle.dataset.sequence.train_fraction == pytest.approx(0.5)
        assert bundle.dataset.sequence.test_fraction == pytest.approx(0.5)
        assert split.application_order.value == "before_grouping"
        assert split.straddling_group_policy.value == "split_partial_sequences"
        assert split.train_entry_fraction == pytest.approx(0.5)
    assert all(
        bundle.dataset.evaluation_unit is EvaluationUnit.SEQUENCE
        for bundle in paper_bundles
    )
    assert {bundle.dataset.preset for bundle in paper_bundles} == {"ait_ads"}


@pytest.mark.allow_no_new_coverage
def test_load_experiment_bundles_expands_model_and_dataset_axes(
    tmp_path: Path,
) -> None:
    """Embedded model entries should expand into concrete bundles.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    sweep_path = (
        tmp_path / "experiments" / "configs" / "datasets" / "bgl_model_matrix.toml"
    )
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path.write_text(
        (
            'name = "bgl_model_matrix"\n'
            "\n[dataset]\n"
            'name = "bgl_entity_chronological"\n'
            'dataset_name = "BGL"\n'
            'preset = "bgl"\n'
            "\n[dataset.sequence]\n"
            'grouping = "entity"\n'
            "train_fraction = 0.01\n"
            "test_fraction = 0.5\n"
            "\n[[models]]\n"
            'name = "template_frequency_default"\n'
            'detector = "template_frequency"\n'
            "\n[models.overrides]\n"
            '"dataset.sequence.train_fraction" = 0.01\n'
            "\n[[models]]\n"
            'name = "template_frequency_default"\n'
            'detector = "template_frequency"\n'
            "\n[models.overrides]\n"
            '"dataset.sequence.train_fraction" = 0.1\n'
            "\n[[models]]\n"
            'name = "deeplog_default"\n'
            'detector = "deeplog"\n'
            "\n[models.overrides]\n"
            '"dataset.sequence.train_fraction" = 0.01\n'
            "\n[[models]]\n"
            'name = "deeplog_default"\n'
            'detector = "deeplog"\n'
            "\n[models.overrides]\n"
            '"dataset.sequence.train_fraction" = 0.1\n'
        ),
        encoding="utf-8",
    )

    bundles = load_experiment_bundles(sweep_path)

    assert [bundle.concrete_name for bundle in bundles] == [
        "bgl_entity_chronological_template_frequency_train_fraction_0p01",
        "bgl_entity_chronological_template_frequency_train_fraction_0p1",
        "bgl_entity_chronological_deeplog_train_fraction_0p01",
        "bgl_entity_chronological_deeplog_train_fraction_0p1",
    ]
    assert {
        (
            bundle.model.name,
            bundle.dataset.sequence.train_fraction,
            bundle.dataset.sequence.test_fraction,
        )
        for bundle in bundles
    } == {
        ("template_frequency_default", 0.01, 0.5),
        ("template_frequency_default", 0.1, 0.5),
        ("deeplog_default", 0.01, 0.5),
        ("deeplog_default", 0.1, 0.5),
    }


@pytest.mark.parametrize(
    "case",
    [
        (
            "experiments/configs/datasets/bgl/entity_chronological.toml",
            "bgl_entity_chronological",
            {
                "template_frequency_default",
                "naive_bayes_default",
                "markov_default",
                "deepcase",
            },
        ),
        (
            "experiments/configs/datasets/hdfs_v1_entity_chronological.toml",
            "hdfs_v1_entity_chronological",
            {
                "template_frequency_default",
                "naive_bayes_default",
                "markov_default",
                "deepcase",
            },
        ),
    ],
)
def test_entity_chronological_manifests_target_the_expected_dataset_and_models(
    case: tuple[str, str, set[str]],
) -> None:
    """Merged entity-chronological manifests should reflect their dataset family.

    Args:
        case (tuple[str, str, set[str]]): Manifest path, dataset name, and the
            set of expected model names.

    Raises:
        TypeError: If a manifest resolves to a non-entity sequence config.
    """
    sweep_relpath, expected_dataset, expected_models = case
    repo_root = Path(__file__).resolve().parents[2]
    bundles = load_experiment_bundles(repo_root / sweep_relpath)

    assert {bundle.dataset.name for bundle in bundles} == {expected_dataset}
    assert {bundle.model.name for bundle in bundles} >= expected_models
    assert {bundle.dataset.sequence.train_fraction for bundle in bundles} == {0.2}
    for bundle in bundles:
        if not isinstance(bundle.dataset.sequence, EntitySequenceConfig):
            msg = "expected an entity sequence configuration"
            raise TypeError(msg)
        assert bundle.dataset.sequence.train_on_normal_entities_only is False


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
        sweep_name="bgl_template_frequency_chronological",
        dataset=(
            "bgl_entity_chronological",
            (
                'name = "bgl_entity_chronological"\n'
                'dataset_name = "BGL"\n'
                'preset = "bgl"\n'
                "\n[sequence]\n"
                'grouping = "entity"\n'
                "train_fraction = 0.8\n"
                "test_fraction = 0.2\n"
            ),
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
    """Missing model entries should fail fast with a clear error.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    # This regression check exercises experiment config validation outside the
    # `anomalog` coverage target, so the warning is intentionally suppressed.
    experiments_root = tmp_path / "experiments"
    sweeps_dir = experiments_root / "configs" / "datasets"
    sweeps_dir.mkdir(parents=True)
    sweep_path = sweeps_dir / "missing_model.toml"
    sweep_path.write_text(
        (
            'name = "broken"\n'
            "\n[dataset]\n"
            'name = "bgl_entity_normal_only"\n'
            'dataset_name = "BGL"\n'
            'preset = "bgl"\n'
        ),
        encoding="utf-8",
    )
    (experiments_root / "configs" / "registry.toml").write_text(
        (
            "[model_sets.missing]\n"
            'models = ["missing_model_default"]\n'
            "\n"
            "[experiments.missing_model]\n"
            'dataset = "missing_model"\n'
            'models = ["missing"]\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigError,
        match="Missing named config",
    ):
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

    with pytest.raises(msgspec.ValidationError, match="train_on_normal_entities_only"):
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
    assert dataset_source_summary(
        DatasetVariantConfig(
            name="openstack-preset",
            dataset_name="demo",
            preset="openstack_deeplog_preprocessed",
        ),
        repo_root=repo_root,
    ) == {
        "type": "preset",
        "preset": "openstack_deeplog_preprocessed",
        "split_source": "predefined_file_boundary",
        "train_source_files": ["openstack_normal1.log"],
        "test_normal_source_files": ["openstack_normal2.log"],
        "test_anomalous_source_files": ["openstack_abnormal.log"],
        "source_file_labels": [
            {
                "source_file": "openstack_normal1.log",
                "label": 0,
                "split": "train",
            },
            {
                "source_file": "openstack_normal2.log",
                "label": 0,
                "split": "test_normal",
            },
            {
                "source_file": "openstack_abnormal.log",
                "label": 1,
                "split": "test_anomalous",
            },
        ],
    }
    assert dataset_source_summary(
        DatasetVariantConfig(
            name="hdfs-preset",
            dataset_name="demo",
            preset="hdfs_wuyifan18_deeplog_preprocessed",
        ),
        repo_root=repo_root,
    ) == {
        "type": "preset",
        "preset": "hdfs_wuyifan18_deeplog_preprocessed",
        "split_source": "predefined_file_boundary",
        "train_source_files": ["hdfs_train"],
        "test_normal_source_files": ["hdfs_test_normal"],
        "test_anomalous_source_files": ["hdfs_test_abnormal"],
        "source_file_labels": [
            {"source_file": "hdfs_train", "label": 0, "split": "train"},
            {
                "source_file": "hdfs_test_normal",
                "label": 0,
                "split": "test_normal",
            },
            {
                "source_file": "hdfs_test_abnormal",
                "label": 1,
                "split": "test_anomalous",
            },
        ],
    }
    assert dataset_source_summary(
        DatasetVariantConfig(
            name="hdfs-compat-preset",
            dataset_name="demo",
            preset="hdfs_wuyifan18_deepcase_table_iv_compat",
        ),
        repo_root=repo_root,
    ) == {
        "type": "preset",
        "preset": "hdfs_wuyifan18_deepcase_table_iv_compat",
        "split_source": "normal_only_event_prefix",
        "included_source_files": ["hdfs_test_normal"],
        "excluded_source_files": ["hdfs_train", "hdfs_test_abnormal"],
        "excluded_anomalous_source_files": ["hdfs_test_abnormal"],
        "source_file_labels": [
            {
                "source_file": "hdfs_test_normal",
                "label": 0,
                "split": "normal_only",
            },
        ],
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
def test_load_experiment_bundles_supports_time_window_grouping(
    tmp_path: Path,
) -> None:
    """Time-window configs should decode through the shared loader.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for a synthetic config tree.
    """
    sweep_path = _write_config_tree(
        tmp_path,
        sweep_name="time_window_grouping",
        dataset=(
            "time_window_grouping",
            (
                'name = "time_window_grouping"\n'
                'dataset_name = "demo"\n'
                'preset = "bgl"\n'
                'evaluation_unit = "window"\n'
                "\n[sequence]\n"
                'grouping = "time"\n'
                "time_span_ms = 3600000\n"
                "step = 3600000\n"
                "train_fraction = 0.8\n"
                "test_fraction = 0.2\n"
            ),
        ),
        model=(
            "template_frequency_default",
            'name = "template_frequency_default"\ndetector = "template_frequency"\n',
        ),
    )

    bundle = _load_one_bundle(sweep_path)

    assert isinstance(bundle.dataset.sequence, TimeSequenceConfig)
    assert bundle.dataset.sequence.time_span_ms == 3_600_000
    assert bundle.dataset.sequence.step == 3_600_000
    assert bundle.dataset.sequence.train_fraction == pytest.approx(0.8)
    assert bundle.dataset.sequence.test_fraction == pytest.approx(0.2)
    assert bundle.dataset.evaluation_unit is EvaluationUnit.WINDOW


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

    bgl_1pct_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "bgl"
        / "bgl_deeplog_ccs2017_paper_1pct_normal_entry_stream_no_online.toml",
    )
    bgl_10pct_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "bgl"
        / "bgl_deeplog_ccs2017_paper_10pct_entry_stream_no_online.toml",
    )
    bgl_2022_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "bgl"
        / "how_far_are_we_2022.toml",
    )
    hdfs_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "hdfs/v1_deeplog_paper_entry100k_split_partial.toml",
    )
    hdfs_assign_first_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "hdfs/v1_deeplog_paper_entry100k_assign_first.toml",
    )
    hdfs_drain3_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "hdfs/v1_deeplog_paper_entry100k_split_partial_drain3.toml",
    )
    hdfs_assign_first_drain3_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "hdfs/v1_deeplog_paper_entry100k_assign_first_drain3.toml",
    )

    def bundle_named(
        bundles: list[ExperimentBundle],
        detector: str,
        *,
        run_group: str | None = None,
    ) -> ExperimentBundle:
        return next(
            bundle
            for bundle in bundles
            if bundle.model.detector == detector
            and (run_group is None or bundle.run_group == run_group)
        )

    validate_deeplog_paper_config(
        dataset_config=bundle_named(bgl_1pct_bundles, "deeplog").dataset,
        model_config=bundle_named(bgl_1pct_bundles, "deeplog").model,
    )
    validate_deeplog_paper_config(
        dataset_config=bundle_named(bgl_10pct_bundles, "deeplog").dataset,
        model_config=bundle_named(bgl_10pct_bundles, "deeplog").model,
    )
    validate_deeplog_paper_config(
        dataset_config=bundle_named(
            hdfs_bundles,
            "deeplog",
            run_group="deeplog_default",
        ).dataset,
        model_config=bundle_named(
            hdfs_bundles,
            "deeplog",
            run_group="deeplog_default",
        ).model,
    )
    validate_bgl_how_far_are_we_2022_config(
        dataset_config=bundle_named(bgl_2022_bundles, "deeplog").dataset,
        model_config=bundle_named(bgl_2022_bundles, "deeplog").model,
    )
    assert (
        bundle_named(
            hdfs_bundles,
            "deeplog",
            run_group="deeplog_default",
        ).dataset.template_parser
        == "spell"
    )
    assert (
        bundle_named(
            hdfs_drain3_bundles,
            "deeplog",
            run_group="deeplog_default",
        ).dataset.template_parser
        == "drain3"
    )
    assert (
        bundle_named(
            hdfs_assign_first_bundles,
            "deeplog",
            run_group="deeplog_default",
        ).dataset.template_parser
        == "spell"
    )
    assert (
        bundle_named(
            hdfs_assign_first_drain3_bundles,
            "deeplog",
            run_group="deeplog_default",
        ).dataset.template_parser
        == "drain3"
    )
    assert bundle_named(bgl_2022_bundles, "deeplog").dataset.template_parser == "drain3"
    assert isinstance(
        bundle_named(bgl_2022_bundles, "deeplog").dataset.sequence.split,
        RawEntryPrefixFractionSplitConfig,
    )
    split = bundle_named(bgl_2022_bundles, "deeplog").dataset.sequence.split
    assert split is not None
    assert split.application_order.value == "before_grouping"
    assert split.straddling_group_policy.value == "drop_straddlers"

    assert {bundle.model.detector for bundle in bgl_1pct_bundles} >= {
        "deeplog",
        "template_frequency",
        "markov",
    }
    assert {bundle.model.detector for bundle in bgl_10pct_bundles} >= {
        "deeplog",
        "template_frequency",
        "markov",
    }
    assert {bundle.model.detector for bundle in hdfs_bundles} >= {
        "deeplog",
        "deepcase",
        "template_frequency",
        "markov",
    }
    assert {bundle.model.detector for bundle in hdfs_assign_first_bundles} >= {
        "deeplog",
        "deepcase",
        "template_frequency",
        "markov",
    }

    _assert_template_frequency_baseline_bundle(
        bundle_named(hdfs_bundles, "template_frequency"),
    )
    _assert_template_frequency_baseline_bundle(
        bundle_named(bgl_1pct_bundles, "template_frequency"),
    )
    _assert_template_frequency_baseline_bundle(
        bundle_named(bgl_10pct_bundles, "template_frequency"),
    )
    _assert_template_frequency_baseline_bundle(
        bundle_named(hdfs_assign_first_bundles, "template_frequency"),
    )
    _assert_markov_baseline_bundle(bundle_named(hdfs_bundles, "markov"))
    _assert_markov_baseline_bundle(bundle_named(hdfs_assign_first_bundles, "markov"))

    _assert_bgl_1pct_deeplog_bundle(
        bundle_named(bgl_1pct_bundles, "deeplog"),
    )
    _assert_bgl_10pct_deeplog_bundle(
        bundle_named(bgl_10pct_bundles, "deeplog"),
    )
    _assert_hdfs_deeplog_bundle(
        bundle_named(
            hdfs_bundles,
            "deeplog",
            run_group="deeplog_default",
        ),
    )


def test_wuyifan18_deeplog_preprocessed_config_uses_exact_session_boundary() -> None:
    """The wuyifan18 DeepLog config should pin the exact file boundary."""
    repo_root = Path(__file__).resolve().parents[2]
    deeplog_sweep_path = (
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "hdfs/wuyifan18_deeplog_preprocessed.toml"
    )

    bundle = next(
        bundle
        for bundle in load_experiment_bundles(
            deeplog_sweep_path,
        )
        if bundle.model.detector == "deeplog"
    )

    assert bundle.dataset.preset == "hdfs_wuyifan18_deeplog_preprocessed"
    assert bundle.dataset.name == "hdfs_wuyifan18_preprocessed_exact_boundary"
    assert bundle.dataset.template_parser == "identity"
    assert bundle.dataset_path == deeplog_sweep_path
    assert bundle.model_path == (
        repo_root / "experiments" / "configs" / "models" / "deeplog_default.toml"
    )
    assert isinstance(bundle.dataset.sequence, EntitySequenceConfig)
    assert bundle.dataset.sequence.split is None
    assert bundle.dataset.sequence.train_on_normal_entities_only is False
    assert isinstance(bundle.model, DeepLogModelConfig)
    assert bundle.model.detector == "deeplog"
    assert bundle.model.parameter_detection_enabled is False
    spec = build_dataset_spec(bundle.dataset, repo_root=repo_root)
    assert spec.template_parser is IdentityTemplateParser


def test_wuyifan18_preprocessed_manifest_exposes_the_deeplog_bundle() -> None:
    """The checked-in HDFS manifest should expose the DeepLog bundle."""
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "hdfs/wuyifan18_deeplog_preprocessed.toml"
    )

    bundles = load_experiment_bundles(sweep_path)

    assert {bundle.model.detector for bundle in bundles} == {
        "deeplog",
        "deepcase",
        "template_frequency",
        "markov",
    }
    bundle = next(
        bundle
        for bundle in bundles
        if bundle.model.detector == "deeplog" and bundle.run_group == "deeplog_default"
    )
    assert bundle.dataset.preset == "hdfs_wuyifan18_deeplog_preprocessed"
    assert bundle.dataset.name == "hdfs_wuyifan18_preprocessed_exact_boundary"
    assert bundle.dataset.template_parser == "identity"
    assert bundle.dataset_path == sweep_path
    assert (
        bundle.model_path
        == repo_root / "experiments" / "configs" / "models" / "deeplog_default.toml"
    )
    assert isinstance(bundle.model, DeepLogModelConfig)
    assert bundle.model.detector == "deeplog"
    assert bundle.run_group == "deeplog_default"

    spec = build_dataset_spec(bundle.dataset, repo_root=repo_root)
    assert spec.template_parser is IdentityTemplateParser


def test_deepcase_table_iv_compat_manifest_prediction_only() -> None:
    """The DeepCASE compatibility manifest should stay prediction-only."""
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "hdfs/wuyifan18_deepcase_table_iv_compat.toml"
    )

    bundles = load_experiment_bundles(sweep_path)

    assert {bundle.model.detector for bundle in bundles} == {
        "deepcase",
        "deeplog",
        "markov",
        "template_frequency",
    }
    bundle = next(bundle for bundle in bundles if bundle.model.detector == "deepcase")
    assert bundle.dataset.preset == "hdfs_wuyifan18_deepcase_table_iv_compat"
    assert bundle.dataset.name == "hdfs_wuyifan18_deepcase_table_iv_compat"
    assert bundle.dataset.evaluation_unit is EvaluationUnit.SEQUENCE
    assert isinstance(bundle.dataset.sequence, EntitySequenceConfig)
    assert bundle.dataset.sequence.train_fraction == pytest.approx(0.2)
    assert bundle.dataset.sequence.test_fraction == pytest.approx(0.8)
    assert bundle.dataset.sequence.split is not None
    assert bundle.dataset.sequence.split.application_order.value == "before_grouping"
    assert bundle.dataset.sequence.split.straddling_group_policy.value == (
        "split_partial_sequences"
    )
    assert bundle.dataset.sequence.train_on_normal_entities_only is False
    assert bundle.model.primary_metric_scope is not None
    assert bundle.model.primary_metric_scope.value == "next_event_prediction"
    assert bundle.run_group == "deepcase"

    spec = build_dataset_spec(bundle.dataset, repo_root=repo_root)
    assert spec.template_parser is IdentityTemplateParser


def test_wuyifan18_deeplog_preprocessed_config_keeps_split_labels_stable(
    tmp_path: Path,
) -> None:
    """Raw-entry split settings should leave the file-level boundary unchanged.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for the synthetic config tree.
    """
    train_text = "5 5 5 22 11 9 11 9 11 9 26 26 26 23 23 23 21 21 21\n1 2 3\n"
    normal_text = "4 4 4\n"
    abnormal_text = "6\n"
    source_root, archive_path, train_entry_count = _write_preprocessed_hdfs_archive(
        tmp_path,
        train_text=train_text,
        normal_text=normal_text,
        abnormal_text=abnormal_text,
    )

    def load_split_sequences(
        *,
        dataset_name: str,
        train_fraction: float,
        test_fraction: float,
    ) -> list[tuple[str, str, int, list[str]]]:
        sweep_path = _write_config_tree(
            tmp_path,
            sweep_name=f"{dataset_name}_sweep",
            dataset=(
                dataset_name,
                (
                    f'name = "{dataset_name}"\n'
                    'dataset_name = "HDFS_WUYIFAN18_DEEPLOG_PREPROCESSED"\n'
                    'structured_parser = "delimited_labelled_event"\n'
                    'template_parser = "identity"\n'
                    "\n[source]\n"
                    'type = "local_zip"\n'
                    f'zip_path = "{archive_path.as_posix()}"\n'
                    'raw_logs_relpath = "preprocessed/hdfs_events.log"\n'
                    "\n[sequence]\n"
                    'grouping = "entity"\n'
                    f"train_fraction = {train_fraction}\n"
                    f"test_fraction = {test_fraction}\n"
                    "\n[sequence.split]\n"
                    'mode = "raw_entry_prefix_count"\n'
                    f"train_entry_count = {train_entry_count}\n"
                    'application_order = "before_grouping"\n'
                    'straddling_group_policy = "split_partial_sequences"\n'
                    "\n[cache_paths]\n"
                    'data_root = "data/hdfs_preprocessed"\n'
                    'cache_root = ".cache/hdfs_preprocessed"\n'
                ),
            ),
            model=(
                "template_frequency_default",
                (
                    'name = "template_frequency_default"\n'
                    'detector = "template_frequency"\n'
                ),
            ),
        )
        bundle = _load_one_bundle(sweep_path)
        return _model_input_signature(bundle=bundle, repo_root=tmp_path)

    expected = _expected_hdfs_split_signature(
        sessions_by_split={
            "hdfs_train": _read_token_sessions(source_root / "hdfs_train"),
            "hdfs_test_normal": _read_token_sessions(
                source_root / "hdfs_test_normal",
            ),
            "hdfs_test_abnormal": _read_token_sessions(
                source_root / "hdfs_test_abnormal",
            ),
        },
    )
    split_signatures = [
        load_split_sequences(
            dataset_name="hdfs_wuyifan18_deeplog_preprocessed_local_low",
            train_fraction=0.2,
            test_fraction=0.8,
        ),
        load_split_sequences(
            dataset_name="hdfs_wuyifan18_deeplog_preprocessed_local_high",
            train_fraction=1.0,
            test_fraction=0.0,
        ),
    ]
    assert split_signatures[0] == expected
    assert split_signatures[1] == expected


@pytest.mark.allow_no_new_coverage
def test_wuyifan18_preprocessed_config_uses_real_split_files_for_model_input() -> None:
    """Real wuyifan18 config should keep the checked-in exact-boundary wiring."""
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "hdfs/wuyifan18_deeplog_preprocessed.toml"
    )
    bundles = load_experiment_bundles(sweep_path)
    assert {bundle.model.name for bundle in bundles} == {
        "deeplog_default",
        "deepcase",
        "template_frequency_default",
        "markov_default",
    }
    assert {bundle.run_group for bundle in bundles} >= {
        "deeplog_short_session_padding_fidelity",
        "deeplog_default",
    }
    assert any(
        bundle.run_group == "deeplog_short_session_padding_fidelity"
        and isinstance(bundle.model, DeepLogModelConfig)
        and bundle.model.short_session_padding_fidelity
        for bundle in bundles
    )
    bundle = next(
        bundle for bundle in bundles if bundle.model.name == "deeplog_default"
    )
    assert bundle.dataset.name == "hdfs_wuyifan18_preprocessed_exact_boundary"
    assert bundle.dataset.preset == "hdfs_wuyifan18_deeplog_preprocessed"
    assert bundle.dataset.template_parser == "identity"
    assert isinstance(bundle.dataset.sequence, EntitySequenceConfig)
    assert isinstance(bundle.dataset.sequence, EntitySequenceConfig)
    assert bundle.dataset.sequence.split is None
    assert isinstance(bundle.model, DeepLogModelConfig)
    assert bundle.model.name == "deeplog_default"
    assert bundle.model.parameter_detection_enabled is False
    assert bundle.model_path == (
        repo_root / "experiments" / "configs" / "models" / "deeplog_default.toml"
    )
    assert bundle.dataset_path == sweep_path


def test_openstack_deeplog_config_keeps_model_input_stable_across_train_fractions(
    tmp_path: Path,
) -> None:
    """OpenStack split settings should keep model-facing train/test data stable.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for the synthetic config tree.
    """
    train_text = (
        "100 2017-01-01 00:00:10.000 1 INFO nova.compute "
        "[instance: instance-a] Build start\n"
        "101 2017-01-01 00:00:40.000 1 INFO nova.compute "
        "[instance: instance-a] Build done\n"
        "102 2017-01-01 00:01:05.000 1 INFO nova.compute "
        "[instance: instance-b] Delete start\n"
    )
    normal_text = (
        "200 2017-01-01 00:02:05.000 2 INFO nova.compute "
        "[instance: instance-c] Build start\n"
    )
    abnormal_text = (
        "300 2017-01-01 00:03:05.000 3 INFO nova.compute "
        "[instance: instance-d] Libvirt error\n"
    )
    _source_root, archive_path, train_entry_count = (
        _write_openstack_labelled_raw_archive(
            tmp_path,
            train_text=train_text,
            normal_text=normal_text,
            abnormal_text=abnormal_text,
        )
    )

    def load_split_sequences(
        *,
        dataset_name: str,
        train_fraction: float,
        test_fraction: float,
    ) -> list[tuple[str, str, int, list[str]]]:
        sweep_path = _write_config_tree(
            tmp_path,
            sweep_name=f"{dataset_name}_sweep",
            dataset=(
                dataset_name,
                (
                    f'name = "{dataset_name}"\n'
                    'dataset_name = "OPENSTACK_DEEPLOG_PREPROCESSED"\n'
                    'structured_parser = "openstack_deeplog"\n'
                    'template_parser = "identity"\n'
                    "\n[source]\n"
                    'type = "local_zip"\n'
                    f'zip_path = "{archive_path.as_posix()}"\n'
                    'raw_logs_relpath = "preprocessed/openstack_labelled_raw.log"\n'
                    "\n[sequence]\n"
                    'grouping = "entity"\n'
                    f"train_fraction = {train_fraction}\n"
                    f"test_fraction = {test_fraction}\n"
                    "\n[sequence.split]\n"
                    'mode = "raw_entry_prefix_count"\n'
                    f"train_entry_count = {train_entry_count}\n"
                    'application_order = "before_grouping"\n'
                    'straddling_group_policy = "split_partial_sequences"\n'
                    "\n[cache_paths]\n"
                    'data_root = "data/openstack_preprocessed"\n'
                    'cache_root = ".cache/openstack_preprocessed"\n'
                ),
            ),
            model=(
                "template_frequency_default",
                (
                    'name = "template_frequency_default"\n'
                    'detector = "template_frequency"\n'
                ),
            ),
        )
        bundle = _load_one_bundle(sweep_path)
        spec = build_dataset_spec(bundle.dataset, repo_root=tmp_path)
        sequences = list(bundle.dataset.sequence.apply(spec.build()))
        return [
            (
                sequence.entity_ids[0],
                sequence.split_label.value,
                sequence.label,
                sequence.templates,
            )
            for sequence in sequences
        ]

    expected = [
        (
            "openstack_train:instance-a",
            "train",
            0,
            ["Build start", "Build done"],
        ),
        (
            "openstack_train:instance-b",
            "train",
            0,
            ["Delete start"],
        ),
        (
            "openstack_test_normal:instance-c",
            "test",
            0,
            ["Build start"],
        ),
        (
            "openstack_test_abnormal:instance-d",
            "test",
            1,
            ["Libvirt error"],
        ),
    ]
    split_signatures = [
        load_split_sequences(
            dataset_name="openstack_deeplog_preprocessed_local_low",
            train_fraction=0.2,
            test_fraction=0.8,
        ),
        load_split_sequences(
            dataset_name="openstack_deeplog_preprocessed_local_high",
            train_fraction=1.0,
            test_fraction=0.0,
        ),
    ]
    assert split_signatures[0] == expected
    assert split_signatures[1] == expected


def test_thunderbird_smoke_dataset_builds_chronological_stream(
    tmp_path: Path,
) -> None:
    """Thunderbird smoke configs should build the raw-entry stream successfully.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for the local fixture.
    """
    source_root = tmp_path / "thunderbird_source"
    source_root.mkdir()
    (source_root / "Thunderbird.log").write_text(
        (
            "- 1131566461 2005.11.09 dn228 Nov 9 12:01:01 dn228/dn228 "
            "crond(pam_unix)[2915]: session closed for user root\n"
            "+ 1131566522 2005.11.09 dn228 Nov 9 12:02:02 dn228/dn228 "
            "sshd[1234]: disk failure on /dev/sda\n"
        ),
        encoding="utf-8",
    )

    spec = build_dataset_spec(
        DatasetVariantConfig(
            name="thunderbird_smoke_local",
            dataset_name="THUNDERBIRD_SMOKE_LOCAL",
            source=LocalDirSourceConfig(
                path=source_root,
                raw_logs_relpath=Path("Thunderbird.log"),
            ),
            structured_parser="thunderbird",
            template_parser="drain3",
            cache_paths=CachePathsConfigModel(namespace="thunderbird_smoke_test"),
            sequence=ChronologicalStreamSequenceConfig(
                chunk_size=2,
                train_fraction=0.5,
                test_fraction=0.5,
                split=RawEntryPrefixNormalFractionSplitConfig(
                    train_normal_entry_fraction=0.5,
                ),
            ),
        ),
        repo_root=tmp_path,
    )
    templated = spec.build()
    sequences = list(templated.group_by_chronological_stream(chunk_size=2))

    assert templated.sink.count_rows() == 2
    assert len(sequences) == 1
    assert len(sequences[0].events) == 2
    assert sequences[0].label == 1


@pytest.mark.allow_no_new_coverage
def test_deepcase_configs_pin_expected_protocols() -> None:
    """DeepCASE manifests should keep their declared dataset and model contracts."""
    repo_root = Path(__file__).resolve().parents[2]

    hdfs_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "hdfs/wuyifan18_deeplog_preprocessed.toml",
    )
    bgl_extension_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "bgl/entity_chronological.toml",
    )

    hdfs_model_names = {bundle.model.name for bundle in hdfs_bundles}
    assert hdfs_model_names >= {
        "deeplog_default",
        "deepcase",
        "template_frequency_default",
        "markov_default",
    }
    bgl_model_names = {bundle.model.name for bundle in bgl_extension_bundles}
    assert bgl_model_names >= {
        "template_frequency_default",
        "naive_bayes_default",
        "markov_default",
        "deepcase",
        "deeplog_default",
    }
    bgl_run_groups = {bundle.run_group for bundle in bgl_extension_bundles}
    assert bgl_run_groups >= {
        "baselines_with_nb",
        "deepcase",
        "deeplog_default",
        "deepcase_majority_vote",
        "deepcase_threshold_fraction",
        "deepcase_abstain_mixed",
    }

    bgl_deepcase_bundle = next(
        bundle for bundle in bgl_extension_bundles if bundle.model.name == "deepcase"
    )
    assert isinstance(bgl_deepcase_bundle.model, DeepCaseModelConfig)
    assert bgl_deepcase_bundle.model.random_seed == 0
    assert bgl_deepcase_bundle.model.attention_query_iterations == 100
    validate_deepcase_bgl_extension_config(
        dataset_config=bgl_deepcase_bundle.dataset,
        model_config=bgl_deepcase_bundle.model,
    )
    assert bgl_deepcase_bundle.model.epochs == 100
    assert isinstance(bgl_deepcase_bundle.dataset.sequence, EntitySequenceConfig)
    assert bgl_deepcase_bundle.dataset.sequence.train_fraction == pytest.approx(0.2)
    assert bgl_deepcase_bundle.dataset.sequence.test_fraction == pytest.approx(0.8)
    assert bgl_deepcase_bundle.dataset.sequence.train_on_normal_entities_only is False


@pytest.mark.allow_no_new_coverage
def test_thunderbird_entity_manifest_uses_entity_grouping() -> None:
    """Thunderbird DeepCASE runs should use an entity-local sequence view."""
    paper_bundles = load_experiment_bundles(
        Path("experiments/configs/datasets") / "thunderbird/entity_chronological.toml",
    )

    assert {bundle.model.name for bundle in paper_bundles} >= {
        "template_frequency_default",
        "naive_bayes_default",
        "markov_default",
        "deepcase",
    }
    assert {bundle.run_group for bundle in paper_bundles} >= {
        "baselines_with_nb",
        "deepcase",
        "deeplog_default",
        "deepcase_majority_vote",
        "deepcase_threshold_fraction",
        "deepcase_abstain_mixed",
    }
    assert all(
        isinstance(bundle.dataset.sequence, EntitySequenceConfig)
        for bundle in paper_bundles
    )
    assert all(
        bundle.dataset.sequence.train_fraction == pytest.approx(0.2)
        for bundle in paper_bundles
    )
    assert all(
        bundle.dataset.sequence.test_fraction == pytest.approx(0.8)
        for bundle in paper_bundles
    )
    assert all(
        bundle.dataset.evaluation_unit is EvaluationUnit.SEQUENCE
        for bundle in paper_bundles
    )
    assert {bundle.dataset.preset for bundle in paper_bundles} == {"thunderbird"}


def test_mixed_model_manifests_assign_run_groups_for_runner_batching() -> None:
    """Mixed manifests should separate heavyweight models into their own groups."""
    repo_root = Path(__file__).resolve().parents[2]

    bgl_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "bgl/entity_chronological.toml",
    )
    hdfs_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "hdfs_v1_entity_chronological.toml",
    )
    thunderbird_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "thunderbird/entity_chronological.toml",
    )
    openstack_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "openstack/deeplog_preprocessed.toml",
    )
    openstack_parameter_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "openstack/deeplog_parameter_ci_approx.toml",
    )
    ait_ads_bundles = load_experiment_bundles(
        repo_root / "experiments" / "configs" / "datasets" / "ait_ads/base.toml",
    )
    ait_ads_entity_bundles = load_experiment_bundles(
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "ait_ads/entity_chronological.toml",
    )

    def run_group_for(bundle_set: list[ExperimentBundle], detector: str) -> str:
        return next(
            bundle.run_group
            for bundle in bundle_set
            if bundle.model.detector == detector
        )

    assert run_group_for(bgl_bundles, "template_frequency") == "baselines_with_nb"
    assert run_group_for(bgl_bundles, "naive_bayes") == "baselines_with_nb"
    assert run_group_for(bgl_bundles, "markov") == "baselines_with_nb"
    assert run_group_for(bgl_bundles, "deeplog") == "deeplog_default"
    assert {bundle.run_group for bundle in bgl_bundles} >= {
        "baselines_with_nb",
        "deepcase",
        "deeplog_default",
        "deepcase_majority_vote",
        "deepcase_threshold_fraction",
        "deepcase_abstain_mixed",
    }

    assert run_group_for(hdfs_bundles, "template_frequency") == "baselines_with_nb"
    assert run_group_for(hdfs_bundles, "naive_bayes") == "baselines_with_nb"
    assert run_group_for(hdfs_bundles, "markov") == "baselines_with_nb"
    assert run_group_for(hdfs_bundles, "deepcase") == "deepcase"
    openstack_parameter_sequence = openstack_parameter_bundles[0].dataset.sequence
    assert isinstance(openstack_parameter_sequence, EntitySequenceConfig)
    assert openstack_parameter_sequence.continuous_context is True
    openstack_parameter_model = openstack_parameter_bundles[0].model
    assert isinstance(openstack_parameter_model, DeepLogModelConfig)
    assert openstack_parameter_model.parameter_ci_highlight_templates == (
        "VM Started (Lifecycle Event)",
        "VM Paused (Lifecycle Event)",
        "During sync_power_state the instance has a pending task (spawning). Skip.",
        "Took NUM seconds to build instance.",
    )
    assert run_group_for(hdfs_bundles, "deeplog") == "deeplog_default"

    assert run_group_for(thunderbird_bundles, "template_frequency") == (
        "baselines_with_nb"
    )
    assert run_group_for(thunderbird_bundles, "naive_bayes") == "baselines_with_nb"
    assert run_group_for(thunderbird_bundles, "markov") == "baselines_with_nb"
    assert run_group_for(thunderbird_bundles, "deeplog") == "deeplog_default"
    assert {bundle.run_group for bundle in thunderbird_bundles} >= {
        "baselines_with_nb",
        "deepcase",
        "deeplog_default",
        "deepcase_majority_vote",
        "deepcase_threshold_fraction",
        "deepcase_abstain_mixed",
    }

    assert run_group_for(openstack_bundles, "template_frequency") == "baselines"
    assert run_group_for(openstack_bundles, "markov") == "baselines"
    assert run_group_for(openstack_bundles, "deeplog") == "deeplog"
    assert run_group_for(openstack_bundles, "deepcase") == "deepcase"
    assert run_group_for(openstack_parameter_bundles, "deeplog") == (
        "deeplog_parameter"
    )

    assert run_group_for(ait_ads_bundles, "template_frequency") == "baselines_no_nb"
    assert run_group_for(ait_ads_bundles, "markov") == "baselines_no_nb"
    assert run_group_for(ait_ads_bundles, "deeplog") == "deeplog_default"

    assert (
        run_group_for(ait_ads_entity_bundles, "template_frequency") == "baselines_no_nb"
    )
    assert run_group_for(ait_ads_entity_bundles, "markov") == "baselines_no_nb"
    assert run_group_for(ait_ads_entity_bundles, "deeplog") == "deeplog_default"
    assert run_group_for(ait_ads_entity_bundles, "deepcase") == "deepcase"
