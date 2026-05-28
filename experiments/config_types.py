"""Typed experiment configuration models."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import msgspec

from anomalog.cache import CachePathsConfig
from anomalog.labels import CSVReader
from anomalog.presets import resolve_preset
from anomalog.sequences import (
    RawEntrySplitMode,
    SplitApplicationOrder,
    StraddlingGroupPolicy,
)
from anomalog.sources import (
    DatasetSource,
    LocalDirSource,
    LocalZipSource,
    PostProcessedSource,
    RemoteZipSource,
)
from anomalog.split_validation import validate_split_fractions
from experiments import ConfigError
from experiments.models.metric_schema import EvaluationUnit

if TYPE_CHECKING:
    from anomalog.labels import AnomalyLabelReader
    from anomalog.parsers.template import TemplatedDataset
    from anomalog.sequences import EntitySequenceBuilder, SequenceBuilder
    from experiments.models import ExperimentModelConfig
    from experiments.models.metric_schema import EvaluationUnit


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
        md5_checksum (str | None): Optional checksum for the archive.
        raw_logs_relpath (Path | None): Optional raw-log path relative to the
            extracted dataset root.
    """

    url: str
    md5_checksum: str | None = None
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
    """Cache/data root paths for dataset materialisation.

    The configuration supports either an explicit `data_root`/`cache_root`
    pair or a shorthand `namespace`. The shorthand expands to
    `data/<namespace>` and `.cache/<namespace>` relative to the repository
    root, which keeps the common case short while still allowing manual
    overrides when needed.

    Attributes:
        namespace (str | None): Optional shared suffix used for both roots.
        data_root (Path | None): Root for materialised raw datasets.
        cache_root (Path | None): Root for derived artifacts and cached outputs.
    """

    namespace: str | None = None
    data_root: Path | None = None
    cache_root: Path | None = None

    def __post_init__(self) -> None:
        """Validate the shorthand and explicit cache-path forms.

        Raises:
            ConfigError: If the configuration mixes shorthand and explicit roots
                or omits one half of the explicit root pair.
        """
        if self.namespace is not None:
            if self.data_root is not None or self.cache_root is not None:
                msg = (
                    "cache_paths may define either `namespace` or explicit "
                    "`data_root`/`cache_root` values, not both."
                )
                raise ConfigError(msg)
            if not self.namespace.strip():
                msg = "cache_paths namespace must not be empty."
                raise ConfigError(msg)
            return
        if self.data_root is None or self.cache_root is None:
            msg = (
                "cache_paths must define either `namespace` or both "
                "`data_root` and `cache_root`."
            )
            raise ConfigError(msg)

    def resolve(self, *, repo_root: Path) -> CachePathsConfig:
        """Resolve cache/data roots relative to the repository root.

        Args:
            repo_root (Path): Repository root used to resolve relative cache paths.

        Returns:
            CachePathsConfig: Concrete cache paths resolved against the repo root.

        Raises:
            ConfigError: If the config is invalid or incomplete.
        """
        if self.namespace is not None:
            namespace_root = _namespace_root(
                self.namespace,
                repo_root=repo_root,
                env_var="ANOMALOG_DATA_ROOT",
                fallback_prefix=Path("data"),
            )
            cache_namespace_root = _namespace_root(
                self.namespace,
                repo_root=repo_root,
                env_var="ANOMALOG_CACHE_ROOT",
                fallback_prefix=Path(".cache"),
            )
            return CachePathsConfig(
                data_root=namespace_root,
                cache_root=cache_namespace_root,
            )
        if self.data_root is None or self.cache_root is None:
            msg = (
                "cache_paths must define either `namespace` or both "
                "`data_root` and `cache_root`."
            )
            raise ConfigError(msg)
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
        continuous_context (bool): Whether adjacent entity windows should
            carry state across sequence boundaries.
    """

    train_on_normal_entities_only: bool = False
    continuous_context: bool = False

    def apply(self, templated: TemplatedDataset) -> SequenceBuilder:
        """Build a configured entity-grouped sequence view.

        Args:
            templated (TemplatedDataset): Built templated dataset to group by entity.

        Returns:
            SequenceBuilder: Entity-grouped builder with split settings applied.
        """
        sequences = self._apply_split_settings(self._group_sequences(templated))
        if self.continuous_context:
            sequences = sequences.with_continuous_context()
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
        SequenceConfigBase.__post_init__(self)

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
        evaluation_unit (EvaluationUnit | None): Optional primary evaluation
            abstraction for the run's headline metrics.
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
    evaluation_unit: EvaluationUnit | None = None
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

    def source_summary(self, *, repo_root: Path) -> dict[str, object]:
        """Return a stable source summary for manifests.

        Args:
            repo_root (Path): Repository root used to resolve relative source paths.

        Returns:
            dict[str, object]: Stable JSON-serialisable source summary.
        """
        if self.preset is not None:
            summary: dict[str, object] = {
                "preset": self.preset,
                "type": "preset",
            }
            preset_spec = resolve_preset(self.preset)
            preset_source = preset_spec.source
            if (
                isinstance(preset_source, PostProcessedSource)
                and preset_source.split_provenance is not None
            ):
                summary.update(preset_source.split_provenance.as_dict())
            return summary
        source, _ = self.custom_dataset_components()
        return dict(source.manifest_entry(repo_root=repo_root))


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


@runtime_checkable
class ExperimentRunConfig(Protocol):
    """Shared runtime contract for dataset-owned experiment matrices.

    Attributes:
        name (str): Human-readable run name.
        dataset (Any): Decoded dataset config.
        models (list[Any]): Concrete model run entries embedded in the file.
        results_root (Path): Root directory for run outputs.
        description (str | None): Optional free-text run description.
        max_workers (WorkerCount): Maximum concurrent concrete runs.
    """

    name: str
    dataset: Any
    models: list[Any]
    results_root: Path
    description: str | None
    max_workers: WorkerCount


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
    """Resolved concrete run config derived from a sweep or inline scenario.

    Attributes:
        experiments_root (Path): Root directory containing experiment configs.
        repo_root (Path): Repository root used for path resolution.
        sweep_path (Path): Resolved sweep config path.
        dataset_path (Path): Resolved dataset config path.
        model_path (Path): Resolved model config path.
        sweep (ExperimentRunConfig): Decoded sweep or inline scenario config.
        dataset (DatasetVariantConfig): Decoded dataset config.
        model (ExperimentModelConfig): Decoded model config.
        concrete_name (str): Deterministic label for the concrete run within the
            sweep.
        run_group (str): Scheduling group used to batch compatible model runs
            together inside one manifest.
        applied_overrides (dict[str, Any]): Fixed and axis overrides applied to
            derive the concrete run.
        experiment_name (str | None): Registry experiment name when the bundle
            was resolved from the named registry.
        experiment_groups (tuple[str, ...]): Registry groups attached to the
            bundle when it came from the named registry.
    """

    experiments_root: Path
    repo_root: Path
    sweep_path: Path
    dataset_path: Path
    model_path: Path
    sweep: ExperimentRunConfig
    dataset: DatasetVariantConfig
    model: ExperimentModelConfig
    concrete_name: str
    run_group: str = "default"
    applied_overrides: dict[str, Any] = msgspec.field(default_factory=dict)
    experiment_name: str | None = None
    experiment_groups: tuple[str, ...] = msgspec.field(default_factory=tuple)

    def normalized_config(self) -> dict[str, object]:
        """Return a JSON-like normalised config payload for manifests.

        Returns:
            dict[str, object]: Normalised config payload for hashing and manifests.

        Raises:
            TypeError: If msgspec returns a non-dict payload unexpectedly.
        """
        sweep_path = _resolve_path(self.sweep_path, self.repo_root)
        dataset_path = _resolve_path(self.dataset_path, self.repo_root)
        model_path = _resolve_path(self.model_path, self.repo_root)
        payload = msgspec.to_builtins(
            {
                "sweep": self.sweep,
                "dataset": self.dataset,
                "model": self.model,
                "concrete": {
                    "name": self.concrete_name,
                    "overrides": self.applied_overrides,
                },
                **(
                    {
                        "experiment": {
                            "name": self.experiment_name,
                            "groups": list(self.experiment_groups),
                        },
                    }
                    if self.experiment_name is not None
                    else {}
                ),
                "paths": {
                    "sweep": sweep_path.relative_to(self.repo_root).as_posix(),
                    "dataset": dataset_path.relative_to(self.repo_root).as_posix(),
                    "model": model_path.relative_to(self.repo_root).as_posix(),
                },
            },
            enc_hook=_path_to_string,
        )
        if not isinstance(payload, dict):
            msg = f"Expected dict payload, got {type(payload).__name__}."
            raise TypeError(msg)
        return payload

    def with_experiment_metadata(
        self,
        *,
        experiment_name: str,
        experiment_groups: tuple[str, ...],
    ) -> ExperimentBundle:
        """Return a copy annotated with registry metadata.

        Args:
            experiment_name (str): Registry entry name for the selected run.
            experiment_groups (tuple[str, ...]): Registry groups attached to the
                selected run.

        Returns:
            ExperimentBundle: Bundle annotated with registry provenance.
        """
        return ExperimentBundle(
            experiments_root=self.experiments_root,
            repo_root=self.repo_root,
            sweep_path=self.sweep_path,
            dataset_path=self.dataset_path,
            model_path=self.model_path,
            sweep=self.sweep,
            dataset=self.dataset,
            model=self.model,
            concrete_name=self.concrete_name,
            run_group=self.run_group,
            applied_overrides=dict(self.applied_overrides),
            experiment_name=experiment_name,
            experiment_groups=experiment_groups,
        )


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


def _namespace_root(
    namespace: str,
    *,
    repo_root: Path,
    env_var: str,
    fallback_prefix: Path,
) -> Path:
    base_root = os.environ.get(env_var)
    if base_root:
        resolved_base_root = Path(base_root).expanduser()
        if not resolved_base_root.is_absolute():
            resolved_base_root = _resolve_path(resolved_base_root, repo_root)
        return resolved_base_root / namespace
    return _resolve_path(fallback_prefix / namespace, repo_root)


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
