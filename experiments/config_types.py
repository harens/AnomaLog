"""Typed experiment configuration models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, TypeVar

import msgspec

from anomalog.cache import CachePathsConfig
from anomalog.labels import CSVReader
from anomalog.sequences import (
    RawEntrySplitMode,
    SplitApplicationOrder,
    StraddlingGroupPolicy,
)
from anomalog.sources import (
    DatasetSource,
    LocalDirSource,
    LocalZipSource,
    RemoteZipSource,
)
from anomalog.split_validation import validate_split_fractions
from experiments import ConfigError

if TYPE_CHECKING:
    from anomalog.labels import AnomalyLabelReader
    from anomalog.parsers.template import TemplatedDataset
    from anomalog.sequences import EntitySequenceBuilder, SequenceBuilder
    from experiments.models import ExperimentModelConfig


class DatasetSourceConfig(msgspec.Struct, frozen=True, tag_field="type"):
    """Tagged config base for materialising a dataset source."""

    def build(self, *, repo_root: Path) -> DatasetSource:
        """Build the runtime dataset source.

        Args:
            repo_root (Path): Repository root used to resolve relative paths.

        Raises:
            NotImplementedError: Always, until implemented by a concrete source config.
        """  # noqa: DOC201, DOC203 - No return doc since base method always raises.
        del repo_root
        msg = f"{type(self).__name__} must implement build()."
        raise NotImplementedError(msg)

    def manifest_entry(
        self,
        *,
        repo_root: Path,
    ) -> dict[str, str | None]:
        """Return a stable source manifest entry.

        Args:
            repo_root (Path): Repository root used to resolve relative paths.

        Raises:
            NotImplementedError: Always, until implemented by a concrete source config.
        """  # noqa: DOC201, DOC203 - No return doc since base method always raises.
        del repo_root
        msg = f"{type(self).__name__} must implement manifest_entry()."
        raise NotImplementedError(msg)


class LocalDirSourceConfig(
    DatasetSourceConfig,
    tag="local_dir",
    frozen=True,
):
    """Use an existing local directory as the dataset root.

    Attributes:
        path (Path): Source directory, relative to the repo when not absolute.
        raw_logs_relpath (Path | None): Optional raw-log path relative to the
            source directory.
    """

    path: Path
    raw_logs_relpath: Path | None = None

    def build(self, *, repo_root: Path) -> LocalDirSource:
        """Build a local-directory dataset source.

        Args:
            repo_root (Path): Repository root used to resolve relative source paths.

        Returns:
            LocalDirSource: Runtime local-directory source.
        """
        return LocalDirSource(
            path=_resolve_path(self.path, repo_root),
            raw_logs_relpath=self.raw_logs_relpath,
        )

    def manifest_entry(self, *, repo_root: Path) -> dict[str, str | None]:
        """Return a stable source manifest entry.

        Args:
            repo_root (Path): Repository root used to resolve relative source paths.

        Returns:
            dict[str, str | None]: Manifest entry for the local directory source.
        """
        return {
            "type": "local_dir",
            "path": _resolve_path(self.path, repo_root).as_posix(),
            "raw_logs_relpath": _optional_posix_path(self.raw_logs_relpath),
        }


class LocalZipSourceConfig(
    DatasetSourceConfig,
    tag="local_zip",
    frozen=True,
):
    """Use a local zip archive as the dataset source.

    Attributes:
        zip_path (Path): Archive path, relative to the repo when not absolute.
        raw_logs_relpath (Path | None): Optional raw-log path relative to the
            extracted dataset root.
        md5_checksum (str | None): Optional checksum used to verify the archive.
    """

    zip_path: Path
    raw_logs_relpath: Path | None = None
    md5_checksum: str | None = None

    def build(self, *, repo_root: Path) -> LocalZipSource:
        """Build a local-zip dataset source.

        Args:
            repo_root (Path): Repository root used to resolve relative source paths.

        Returns:
            LocalZipSource: Runtime local-zip source.
        """
        return LocalZipSource(
            zip_path=_resolve_path(self.zip_path, repo_root),
            raw_logs_relpath=self.raw_logs_relpath,
            md5_checksum=self.md5_checksum,
        )

    def manifest_entry(self, *, repo_root: Path) -> dict[str, str | None]:
        """Return a stable source manifest entry.

        Args:
            repo_root (Path): Repository root used to resolve relative source paths.

        Returns:
            dict[str, str | None]: Manifest entry for the local zip source.
        """
        return {
            "type": "local_zip",
            "zip_path": _resolve_path(self.zip_path, repo_root).as_posix(),
            "raw_logs_relpath": _optional_posix_path(self.raw_logs_relpath),
            "md5_checksum": self.md5_checksum,
        }


class RemoteZipSourceConfig(
    DatasetSourceConfig,
    tag="remote_zip",
    frozen=True,
):
    """Download a remote zip archive for the dataset.

    Attributes:
        url (str): Absolute URL of the dataset archive.
        md5_checksum (str): Expected checksum for the archive.
        raw_logs_relpath (Path | None): Optional raw-log path relative to the
            extracted dataset root.
    """

    url: str
    md5_checksum: str
    raw_logs_relpath: Path | None = None

    def build(self, *, repo_root: Path) -> RemoteZipSource:
        """Build a remote-zip dataset source.

        Args:
            repo_root (Path): Repository root. Unused for remote zip sources.

        Returns:
            RemoteZipSource: Runtime remote-zip source.
        """
        del repo_root
        return RemoteZipSource(
            url=self.url,
            md5_checksum=self.md5_checksum,
            raw_logs_relpath=self.raw_logs_relpath,
        )

    def manifest_entry(self, *, repo_root: Path) -> dict[str, str | None]:
        """Return a stable source manifest entry.

        Args:
            repo_root (Path): Repository root. Unused for remote zip sources.

        Returns:
            dict[str, str | None]: Manifest entry for the remote zip source.
        """
        del repo_root
        return {
            "type": "remote_zip",
            "url": self.url,
            "raw_logs_relpath": _optional_posix_path(self.raw_logs_relpath),
            "md5_checksum": self.md5_checksum,
        }


class LabelReaderConfig(msgspec.Struct, frozen=True, tag_field="type"):
    """Tagged config base for anomaly-label readers."""

    def build(self) -> AnomalyLabelReader:
        """Build the runtime anomaly-label reader.

        Raises:
            NotImplementedError: Always, until implemented by a concrete
                label-reader config.
        """  # noqa: DOC201, DOC203 - No return doc since base method always raises.
        msg = f"{type(self).__name__} must implement build()."
        raise NotImplementedError(msg)


class CSVLabelReaderConfig(
    LabelReaderConfig,
    tag="csv",
    frozen=True,
):
    """Read anomaly labels from a CSV file.

    Attributes:
        relative_path (Path): CSV path relative to the materialised dataset root.
        entity_column (str): CSV column containing the entity/group id.
        label_column (str): CSV column containing the integer anomaly label.
    """

    relative_path: Path
    entity_column: str = "entity_id"
    label_column: str = "anomalous"

    def build(self) -> CSVReader:
        """Build a CSV-backed anomaly label reader.

        Returns:
            CSVReader: Runtime CSV-backed label reader.
        """
        return CSVReader(
            relative_path=self.relative_path,
            entity_column=self.entity_column,
            label_column=self.label_column,
        )


_DATASET_SOURCE_CONFIG_TYPES: dict[str, type[DatasetSourceConfig]] = {
    "local_dir": LocalDirSourceConfig,
    "local_zip": LocalZipSourceConfig,
    "remote_zip": RemoteZipSourceConfig,
}

_LABEL_READER_CONFIG_TYPES: dict[str, type[LabelReaderConfig]] = {
    "csv": CSVLabelReaderConfig,
}


class CachePathsConfigModel(msgspec.Struct, frozen=True):
    """Cache/data root paths for dataset materialsation.

    Attributes:
        data_root (Path): Root for materialised raw datasets.
        cache_root (Path): Root for derived artifacts and cached outputs.
    """

    data_root: Path
    cache_root: Path

    def resolve(self, *, repo_root: Path) -> CachePathsConfig:
        """Resolve cache/data roots relative to the repository root.

        Args:
            repo_root (Path): Repository root used to resolve relative cache paths.

        Returns:
            CachePathsConfig: Concrete cache paths resolved against the repo root.
        """
        return CachePathsConfig(
            data_root=_resolve_path(self.data_root, repo_root),
            cache_root=_resolve_path(self.cache_root, repo_root),
        )


class RawEntrySplitConfigBase(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
    tag_field="mode",
):
    """Shared configuration for raw-entry chronological split modes.

    Attributes:
        application_order (SplitApplicationOrder): When to apply the split
            relative to grouping.
        straddling_group_policy (StraddlingGroupPolicy): How to handle groups
            that cross the raw-entry split boundary.
    """

    application_order: SplitApplicationOrder = SplitApplicationOrder.BEFORE_GROUPING
    straddling_group_policy: StraddlingGroupPolicy = (
        StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES
    )


class RawEntryPrefixCountSplitConfig(
    RawEntrySplitConfigBase,
    tag="raw_entry_prefix_count",
    frozen=True,
):
    """Split by the first N raw entries in chronological order.

    Attributes:
        train_entry_count (int): Number of raw entries to keep in the train
            prefix.
    """

    train_entry_count: int


class RawEntryPrefixFractionSplitConfig(
    RawEntrySplitConfigBase,
    tag="raw_entry_prefix_fraction",
    frozen=True,
):
    """Split by the first p fraction of raw entries in chronological order.

    Attributes:
        train_entry_fraction (Annotated[float, msgspec.Meta(gt=0.0, le=1.0)]):
            Fraction of raw entries to keep in the train prefix.
    """

    train_entry_fraction: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)]


class RawEntryPrefixNormalFractionSplitConfig(
    RawEntrySplitConfigBase,
    tag="raw_entry_prefix_normal_fraction",
    frozen=True,
):
    """Split by the first p fraction of normal raw entries in chronological order.

    Attributes:
        train_normal_entry_fraction (Annotated[float, msgspec.Meta(gt=0.0, le=1.0)]):
            Fraction of normal raw entries to keep in the train prefix.
    """

    train_normal_entry_fraction: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)]


RawEntrySplitConfig: TypeAlias = (
    RawEntryPrefixCountSplitConfig
    | RawEntryPrefixFractionSplitConfig
    | RawEntryPrefixNormalFractionSplitConfig
)


SweepOverrideValues = Annotated[list[Any], msgspec.Meta(min_length=1)]
TrainFraction = Annotated[float, msgspec.Meta(ge=0.0, le=1.0)]
TestFraction = Annotated[float, msgspec.Meta(ge=0.0, le=1.0)]
PositiveWorkerCount = Annotated[int, msgspec.Meta(gt=0)]
WorkerCount = Literal["auto"] | PositiveWorkerCount
TSequenceBuilder = TypeVar("TSequenceBuilder", bound="SequenceBuilder")


class SequenceConfigBase(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
    tag_field="grouping",
):
    """Shared sequence-generation settings for a dataset variant.

    Attributes:
        split (RawEntrySplitConfig | None): Optional raw-entry split mode to
            apply before grouping.
        step (int | None): Grouping-specific step between windows. `None`
            delegates to the grouping mode's default.
        train_fraction (TrainFraction): Requested training fraction for the
            total sequence population.
        test_fraction (TestFraction): Fixed test suffix fraction.
    """

    split: RawEntrySplitConfig | None = None
    step: int | None = None
    train_fraction: TrainFraction = 0.2
    test_fraction: TestFraction = 0.8

    def __post_init__(self) -> None:
        """Validate cross-field split constraints.

        Raises:
            ConfigError: If the requested test suffix is invalid or leaves no
                room for the train prefix.
        """
        try:
            if self.split is None:
                validate_split_fractions(
                    train_frac=self.train_fraction,
                    test_frac=self.test_fraction,
                )
            elif self.split.application_order == SplitApplicationOrder.AFTER_GROUPING:
                msg = (
                    "raw-entry split modes must use "
                    'split.application_order = "before_grouping".'
                )
                raise ConfigError(msg)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

    def apply(self, templated: TemplatedDataset) -> SequenceBuilder:
        """Build a configured sequence view from a templated dataset.

        Args:
            templated (TemplatedDataset): Built templated dataset to group into
                sequences.

        Returns:
            SequenceBuilder: Sequence builder with grouping and split settings applied.
        """
        return self._apply_split_settings(self._group_sequences(templated))

    def _group_sequences(
        self,
        templated: TemplatedDataset,
    ) -> SequenceBuilder:
        """Apply the grouping-specific builder transformation.

        Args:
            templated (TemplatedDataset): Built dataset to group into sequences.

        Raises:
            NotImplementedError: Always, until implemented by a concrete
                grouping config.
        """  # noqa: DOC201, DOC203 - No return doc since base method always raises.
        cls_name = type(self).__name__
        del self, templated
        msg = f"{cls_name} must implement _group_sequences()."
        raise NotImplementedError(msg)

    def _apply_split_settings(
        self,
        sequences: TSequenceBuilder,
    ) -> TSequenceBuilder:
        """Build a configured sequence view from a templated dataset.

        Args:
            sequences (TSequenceBuilder): Grouped sequence builder to apply
                shared split settings to.

        Returns:
            TSequenceBuilder: Grouped sequence builder with shared split
                settings applied.

        Raises:
            ConfigError: If the configured raw-entry split is unsupported.
        """
        sequences = sequences.with_split_fractions(
            self.train_fraction,
            self.test_fraction,
        )
        if self.split is None:
            return sequences
        split = self.split
        if isinstance(split, RawEntryPrefixCountSplitConfig):
            split_mode = RawEntrySplitMode.PREFIX_COUNT
            split_application_order = split.application_order
            straddling_group_policy = split.straddling_group_policy
            split_kwargs: dict[str, object] = {
                "split_mode": split_mode,
                "split_application_order": split_application_order,
                "straddling_group_policy": straddling_group_policy,
                "train_entry_count": split.train_entry_count,
            }
        elif isinstance(split, RawEntryPrefixFractionSplitConfig):
            split_mode = RawEntrySplitMode.PREFIX_FRACTION
            split_application_order = split.application_order
            straddling_group_policy = split.straddling_group_policy
            split_kwargs = {
                "split_mode": split_mode,
                "split_application_order": split_application_order,
                "straddling_group_policy": straddling_group_policy,
                "train_entry_fraction": split.train_entry_fraction,
            }
        elif isinstance(split, RawEntryPrefixNormalFractionSplitConfig):
            split_mode = RawEntrySplitMode.PREFIX_NORMAL_FRACTION
            split_application_order = split.application_order
            straddling_group_policy = StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES
            split_kwargs = {
                "split_mode": split_mode,
                "split_application_order": split_application_order,
                "straddling_group_policy": straddling_group_policy,
                "train_normal_entry_fraction": split.train_normal_entry_fraction,
            }
        else:
            msg = f"Unsupported raw-entry split config: {type(split).__name__}"
            raise ConfigError(msg)
        return replace(sequences, **split_kwargs)


class EntitySequenceConfig(
    SequenceConfigBase,
    tag="entity",
    frozen=True,
    kw_only=True,
):
    """Entity-based sequence configuration.

    Attributes:
        train_on_normal_entities_only (bool): Whether anomalous entities are
            excluded from the training split budget.
    """

    train_on_normal_entities_only: bool = False

    def apply(self, templated: TemplatedDataset) -> SequenceBuilder:
        """Build a configured entity-grouped sequence view.

        Args:
            templated (TemplatedDataset): Built templated dataset to group by entity.

        Returns:
            SequenceBuilder: Entity-grouped builder with split settings applied.
        """
        sequences = self._apply_split_settings(self._group_sequences(templated))
        if self.train_on_normal_entities_only:
            return sequences.with_train_on_normal_entities_only()
        return sequences

    def _group_sequences(
        self,
        templated: TemplatedDataset,
    ) -> EntitySequenceBuilder:
        """Apply entity grouping.

        Args:
            templated (TemplatedDataset): Built templated dataset to group by entity.

        Returns:
            EntitySequenceBuilder: Entity-grouped sequence builder.
        """
        del self
        return templated.group_by_entity()


class FixedSequenceConfig(
    SequenceConfigBase,
    tag="fixed",
    frozen=True,
    kw_only=True,
):
    """Fixed-window sequence configuration.

    Attributes:
        window_size (int): Number of rows per fixed window.
    """

    window_size: int

    def _group_sequences(self, templated: TemplatedDataset) -> SequenceBuilder:
        """Apply fixed-window grouping.

        Args:
            templated (TemplatedDataset): Built templated dataset to group into
                fixed windows.

        Returns:
            SequenceBuilder: Fixed-window sequence builder.
        """
        return templated.group_by_fixed_window(self.window_size, step_size=self.step)


class TimeSequenceConfig(
    SequenceConfigBase,
    tag="time",
    frozen=True,
    kw_only=True,
):
    """Time-window sequence configuration.

    Attributes:
        time_span_ms (int): Duration of each emitted time window in milliseconds.
    """

    time_span_ms: int

    def _group_sequences(self, templated: TemplatedDataset) -> SequenceBuilder:
        """Apply time-window grouping.

        Args:
            templated (TemplatedDataset): Built templated dataset to group into
                time windows.

        Returns:
            SequenceBuilder: Time-window sequence builder.
        """
        return templated.group_by_time_window(self.time_span_ms, step_span_ms=self.step)


class ChronologicalStreamSequenceConfig(
    SequenceConfigBase,
    tag="chronological_stream",
    frozen=True,
    kw_only=True,
):
    """Chronological raw-entry stream grouping configuration.

    Attributes:
        chunk_size (int): Maximum number of raw entries per emitted chunk.
    """

    chunk_size: int = 100_000

    def __post_init__(self) -> None:
        """Validate the chunk size and shared split settings.

        Raises:
            ConfigError: If the chunk size is not positive or the shared split
                settings are invalid.
        """
        if self.chunk_size <= 0:
            msg = "chunk_size must be a positive integer."
            raise ConfigError(msg)
        super().__post_init__()

    def _group_sequences(self, templated: TemplatedDataset) -> SequenceBuilder:
        """Apply chronological stream grouping.

        Args:
            templated (TemplatedDataset): Built dataset to group into
                chronological stream chunks.

        Returns:
            SequenceBuilder: Chronological stream sequence builder.
        """
        return templated.group_by_chronological_stream(chunk_size=self.chunk_size)


SequenceConfig: TypeAlias = (
    EntitySequenceConfig
    | FixedSequenceConfig
    | TimeSequenceConfig
    | ChronologicalStreamSequenceConfig
)


class DatasetVariantConfig(msgspec.Struct, frozen=True):
    """Dataset preprocessing and sequence-generation configuration.

    Attributes:
        name (str): Human-readable dataset variant name.
        dataset_name (str): Dataset identifier used for runtime caches/artifacts.
        preset (str | None): Optional built-in dataset preset name.
        source (DatasetSourceConfig | None): Source config for custom datasets.
        structured_parser (str | None): Structured parser name for custom datasets.
        template_parser (str): Template parser name.
        label_reader (LabelReaderConfig | None): Optional anomaly label reader config.
        cache_paths (CachePathsConfigModel | None): Optional cache/data root override.
        sequence (SequenceConfigBase): Sequence grouping and split config.
        description (str | None): Optional free-text dataset description.
    """

    name: str
    dataset_name: str
    preset: str | None = None
    source: DatasetSourceConfig | None = None
    structured_parser: str | None = None
    template_parser: str = "drain3"
    label_reader: LabelReaderConfig | None = None
    cache_paths: CachePathsConfigModel | None = None
    sequence: SequenceConfigBase = EntitySequenceConfig()
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate the minimum dataset config required to build a spec.

        Raises:
            ConfigError: If the dataset config omits required source or parser data.
        """
        if self.preset is None and self.source is None:
            msg = "dataset config must define either `preset` or `source`."
            raise ConfigError(msg)
        if self.preset is None and self.structured_parser is None:
            msg = (
                "dataset config must define `structured_parser` when no preset is used."
            )
            raise ConfigError(msg)

    def custom_dataset_components(self) -> tuple[DatasetSourceConfig, str]:
        """Return the validated source/parser pair for non-preset datasets.

        Returns:
            tuple[DatasetSourceConfig, str]: Source config and structured parser name.

        Raises:
            ConfigError: If the config is not a valid custom dataset definition.
        """
        if self.source is None or self.structured_parser is None:
            msg = (
                "dataset config invariant violated: custom datasets need "
                "source and structured_parser."
            )
            raise ConfigError(msg)
        return self.source, self.structured_parser

    def source_summary(self, *, repo_root: Path) -> dict[str, str | None]:
        """Return a stable source summary for manifests.

        Args:
            repo_root (Path): Repository root used to resolve relative source paths.

        Returns:
            dict[str, str | None]: Stable JSON-serialisable source summary.
        """
        if self.preset is not None:
            return {"preset": self.preset, "type": "preset"}
        source, _ = self.custom_dataset_components()
        return source.manifest_entry(repo_root=repo_root)


class SweepAxisConfig(msgspec.Struct, frozen=True):
    """One Cartesian-product axis for a sweep.

    Attributes:
        path (str): Dot-separated override path rooted at `sweep`, `dataset`,
            or `model`.
        values (SweepOverrideValues): Concrete values to apply at that path.
    """

    path: str
    values: SweepOverrideValues

    def __post_init__(self) -> None:
        """Validate the override axis shape."""
        _validate_override_path(self.path)


class SweepConfig(msgspec.Struct, frozen=True):
    """Top-level experiment sweep configuration.

    A sweep is now the authoritative experiment entrypoint. A config with no
    axes still represents one concrete run; axes expand that base definition
    into multiple concrete runs that differ only by validated overrides.

    Attributes:
        name (str): Human-readable sweep name.
        dataset (str): Referenced base dataset config name.
        model (str): Referenced base model config name.
        results_root (Path): Root directory for run outputs.
        description (str | None): Optional free-text sweep description.
        overrides (dict[str, Any]): Fixed overrides applied to every concrete
            run generated from the sweep.
        axes (list[SweepAxisConfig]): Cartesian-product axes for generating
            multiple concrete runs.
        max_workers (WorkerCount): Maximum number of concrete runs to execute
            in parallel. `"auto"` caps parallelism to the concrete run count
            and the machine CPU count.
    """

    name: str
    dataset: str
    model: str
    results_root: Path = Path("experiments/results")
    description: str | None = None
    overrides: dict[str, Any] = msgspec.field(default_factory=dict)
    axes: list[SweepAxisConfig] = msgspec.field(default_factory=list)
    max_workers: WorkerCount = "auto"

    def __post_init__(self) -> None:
        """Validate override and execution settings.

        Raises:
            ConfigError: If override paths are malformed or execution settings
                are invalid.
        """
        for path in self.overrides:
            _validate_override_path(path)
        axis_paths = [axis.path for axis in self.axes]
        if len(axis_paths) != len(set(axis_paths)):
            msg = "sweep axes must not repeat the same override path."
            raise ConfigError(msg)
        overlapping_paths = set(axis_paths).intersection(self.overrides)
        if overlapping_paths:
            joined_paths = ", ".join(sorted(overlapping_paths))
            msg = (
                "sweep fixed overrides and axes must not target the same path: "
                f"{joined_paths}."
            )
            raise ConfigError(msg)


class ExperimentBundle(msgspec.Struct, frozen=True):
    """Resolved concrete run config derived from a sweep.

    Attributes:
        experiments_root (Path): Root directory containing experiment configs.
        repo_root (Path): Repository root used for path resolution.
        sweep_path (Path): Resolved sweep config path.
        dataset_path (Path): Resolved dataset config path.
        model_path (Path): Resolved model config path.
        sweep (SweepConfig): Decoded sweep config.
        dataset (DatasetVariantConfig): Decoded dataset config.
        model (ExperimentModelConfig): Decoded model config.
        concrete_name (str): Deterministic label for the concrete run within the
            sweep.
        applied_overrides (dict[str, Any]): Fixed and axis overrides applied to
            derive the concrete run.
    """

    experiments_root: Path
    repo_root: Path
    sweep_path: Path
    dataset_path: Path
    model_path: Path
    sweep: SweepConfig
    dataset: DatasetVariantConfig
    model: ExperimentModelConfig
    concrete_name: str
    applied_overrides: dict[str, Any] = msgspec.field(default_factory=dict)

    def normalized_config(self) -> dict[str, object]:
        """Return a JSON-like normalised config payload for manifests.

        Returns:
            dict[str, object]: Normalised config payload for hashing and manifests.

        Raises:
            TypeError: If msgspec returns a non-dict payload unexpectedly.
        """
        payload = msgspec.to_builtins(
            {
                "sweep": self.sweep,
                "dataset": self.dataset,
                "model": self.model,
                "concrete": {
                    "name": self.concrete_name,
                    "overrides": self.applied_overrides,
                },
                "paths": {
                    "sweep": self.sweep_path.relative_to(self.repo_root).as_posix(),
                    "dataset": self.dataset_path.relative_to(self.repo_root).as_posix(),
                    "model": self.model_path.relative_to(self.repo_root).as_posix(),
                },
            },
            enc_hook=_path_to_string,
        )
        if not isinstance(payload, dict):
            msg = f"Expected dict payload, got {type(payload).__name__}."
            raise TypeError(msg)
        return payload


def _path_to_string(obj: object) -> str:
    if isinstance(obj, Path):
        return obj.as_posix()
    msg = f"Unsupported encoded type: {type(obj)!r}"
    raise NotImplementedError(msg)


def serialise_config(value: object) -> dict[str, object]:
    """Convert config structs into builtins for hashing and manifests.

    Args:
        value (object): Config object or struct to serialise.

    Returns:
        dict[str, object]: JSON-like builtins representation of the config.

    Raises:
        TypeError: If msgspec returns a non-dict payload unexpectedly.
    """
    builtins = msgspec.to_builtins(value, enc_hook=_path_to_string)
    if not isinstance(builtins, dict):
        msg = f"Expected dict payload, got {type(builtins).__name__}."
        raise TypeError(msg)
    return builtins


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def _optional_posix_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.as_posix()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _validate_override_path(path: str) -> None:
    root, *segments = path.split(".")
    if root not in {"sweep", "dataset", "model"} or not segments:
        msg = (
            "override paths must start with `sweep.`, `dataset.`, or `model.` "
            f"and target a nested field: {path!r}."
        )
        raise ConfigError(msg)
    if any(not segment for segment in segments):
        msg = f"override path contains an empty segment: {path!r}."
        raise ConfigError(msg)
