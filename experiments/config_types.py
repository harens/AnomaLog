"""Typed experiment configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import msgspec

from anomalog.cache import CachePathsConfig
from anomalog.labels import CSVReader
from anomalog.sources import (
    DatasetSource,
    LocalDirSource,
    LocalZipSource,
    RemoteZipSource,
)
from experiments import ConfigError

if TYPE_CHECKING:
    from anomalog.labels import AnomalyLabelReader
    from anomalog.parsers.template import TemplatedDataset
    from anomalog.sequences import SequenceBuilder
    from experiments.models import ExperimentModelConfig


class DatasetSourceConfig(msgspec.Struct, frozen=True, tag_field="type"):
    """Tagged config base for materializing a dataset source."""

    def build(self, *, repo_root: Path) -> DatasetSource:
        """Build the runtime dataset source."""
        del repo_root
        msg = f"{type(self).__name__} must implement build()."
        raise NotImplementedError(msg)

    def manifest_entry(self, *, repo_root: Path) -> dict[str, str | None]:
        """Return a stable source manifest entry."""
        del repo_root
        msg = f"{type(self).__name__} must implement manifest_entry()."
        raise NotImplementedError(msg)


class LocalDirSourceConfig(
    DatasetSourceConfig,
    tag="local_dir",
    frozen=True,
):
    """Use an existing local directory as the dataset root."""

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
    """Use a local zip archive as the dataset source."""

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
    """Download a remote zip archive for the dataset."""

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
        """Build the runtime anomaly-label reader."""
        msg = f"{type(self).__name__} must implement build()."
        raise NotImplementedError(msg)


class CSVLabelReaderConfig(
    LabelReaderConfig,
    tag="csv",
    frozen=True,
):
    """Read anomaly labels from a CSV file."""

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
    """Cache/data root paths for dataset materialization."""

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


class SequenceConfigBase(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
    tag_field="grouping",
):
    """Shared sequence-generation settings for a dataset variant."""

    step: int | None = None
    train_fraction: float = 0.8

    def __post_init__(self) -> None:
        """Validate grouping-specific sequence settings.

        Raises:
            ConfigError: If `train_fraction` is outside `[0.0, 1.0]`.
        """
        if not 0.0 <= self.train_fraction <= 1.0:
            msg = "sequence.train_fraction must be between 0.0 and 1.0."
            raise ConfigError(msg)

    def apply(self, templated: TemplatedDataset) -> SequenceBuilder:
        """Build a configured sequence view from a templated dataset.

        Args:
            templated (TemplatedDataset): Built templated dataset to group into
                sequences.

        Returns:
            SequenceBuilder: Sequence builder with grouping and split settings applied.
        """
        return self._apply_split_settings(self._group_sequences(templated))

    def _group_sequences(self, templated: TemplatedDataset) -> SequenceBuilder:
        """Apply the grouping-specific builder transformation."""
        cls_name = type(self).__name__
        del self, templated
        msg = f"{cls_name} must implement _group_sequences()."
        raise NotImplementedError(msg)

    def _apply_split_settings(self, sequences: SequenceBuilder) -> SequenceBuilder:
        """Build a configured sequence view from a templated dataset.

        Args:
            sequences (SequenceBuilder): Grouped sequence builder to apply shared
                split settings to.

        Returns:
            SequenceBuilder: Sequence builder with shared split settings applied.
        """
        return sequences.with_train_fraction(self.train_fraction)


class EntitySequenceConfig(
    SequenceConfigBase,
    tag="entity",
    frozen=True,
    kw_only=True,
):
    """Entity-based sequence configuration."""

    train_on_normal_entities_only: bool = False

    def apply(self, templated: TemplatedDataset) -> SequenceBuilder:
        """Build a configured entity-grouped sequence view.

        Args:
            templated (TemplatedDataset): Built templated dataset to group by entity.

        Returns:
            SequenceBuilder: Entity-grouped builder with split settings applied.
        """
        sequences = templated.group_by_entity().with_train_fraction(self.train_fraction)
        if self.train_on_normal_entities_only:
            return sequences.with_train_on_normal_entities_only()
        return sequences

    def _group_sequences(self, templated: TemplatedDataset) -> SequenceBuilder:
        """Apply entity grouping.

        Args:
            templated (TemplatedDataset): Built templated dataset to group by entity.

        Returns:
            SequenceBuilder: Entity-grouped sequence builder.
        """
        del self
        return templated.group_by_entity()


class FixedSequenceConfig(
    SequenceConfigBase,
    tag="fixed",
    frozen=True,
    kw_only=True,
):
    """Fixed-window sequence configuration."""

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
    """Time-window sequence configuration."""

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


SequenceConfig: TypeAlias = (
    EntitySequenceConfig | FixedSequenceConfig | TimeSequenceConfig
)


class DatasetVariantConfig(msgspec.Struct, frozen=True):
    """Dataset preprocessing and sequence-generation configuration."""

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
            dict[str, str | None]: Stable JSON-serializable source summary.
        """
        if self.preset is not None:
            return {"preset": self.preset, "type": "preset"}
        source, _ = self.custom_dataset_components()
        return source.manifest_entry(repo_root=repo_root)


class RunConfig(msgspec.Struct, frozen=True):
    """Top-level experiment run configuration."""

    name: str
    dataset: str
    model: str
    results_root: Path = Path("experiments/results")
    description: str | None = None


class ExperimentBundle(msgspec.Struct, frozen=True):
    """Resolved run, dataset, and model configs for an experiment."""

    experiments_root: Path
    repo_root: Path
    run_path: Path
    dataset_path: Path
    model_path: Path
    run: RunConfig
    dataset: DatasetVariantConfig
    model: ExperimentModelConfig

    def normalized_config(self) -> dict[str, object]:
        """Return a JSON-like normalized config payload for manifests.

        Returns:
            dict[str, object]: Normalized config payload for hashing and manifests.

        Raises:
            TypeError: If msgspec returns a non-dict payload unexpectedly.
        """
        payload = msgspec.to_builtins(
            {
                "run": self.run,
                "dataset": self.dataset,
                "model": self.model,
                "paths": {
                    "run": self.run_path.relative_to(self.repo_root).as_posix(),
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


def serialize_config(value: object) -> dict[str, object]:
    """Convert config structs into builtins for hashing and manifests.

    Args:
        value (object): Config object or struct to serialize.

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
