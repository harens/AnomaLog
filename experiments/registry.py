"""Named experiment registry loading and expansion.

The registry is the central catalogue for experiment composition. It keeps the
user-facing TOML compact by letting experiments combine shared model groups
with inline model references, while the runtime still expands those logical
entries into concrete bundles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import msgspec

from experiments import ConfigError
from experiments.config_types import (
    _DATASET_SOURCE_CONFIG_TYPES,
    _LABEL_READER_CONFIG_TYPES,
    CachePathsConfigModel,
    DatasetSourceConfig,
    DatasetVariantConfig,
    EntitySequenceConfig,
    EvaluationUnit,
    ExperimentBundle,
    LabelReaderConfig,
    SequenceConfig,
    SweepAxisConfig,
    WorkerCount,
    _optional_str,
    serialise_config,
)
from experiments.models import resolve_model_config_type
from experiments.models.base import decode_experiment_model_config

if TYPE_CHECKING:
    from collections.abc import Callable

    from experiments.models import ExperimentModelConfig

TDecoded = TypeVar("TDecoded")


@dataclass(frozen=True, slots=True)
class ModelSetDefinition:
    """Concrete model configs that should always run together.

    Attributes:
        name (str): Registry name of the model set.
        models (tuple[str, ...]): Named model configs included in the set.
        overrides (dict[str, object]): Fixed overrides applied to each model.
        description (str | None): Optional human-readable description.
    """

    name: str
    models: tuple[str, ...]
    overrides: dict[str, object] = field(default_factory=dict)
    description: str | None = None


@dataclass(frozen=True, slots=True)
class RegisteredExperiment:
    """Logical experiment entry resolved from the registry.

    Attributes:
        name (str): Registry experiment name.
        dataset (str): Dataset manifest name resolved by the registry.
        models (tuple[str, ...]): Inline model references defined directly on
            the experiment.
        model_sets (tuple[str, ...]): Shared model sets used by the
            experiment.
        groups (tuple[str, ...]): Derived reporting and scheduling groups.
        overrides (dict[str, dict[str, object]]): Experiment-specific model
            overrides keyed by model or model-set name.
        description (str | None): Optional human-readable description.
    """

    name: str
    dataset: str
    models: tuple[str, ...]
    model_sets: tuple[str, ...]
    groups: tuple[str, ...]
    overrides: dict[str, dict[str, object]] = field(default_factory=dict)
    description: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedRegistryExperiment:
    """Resolved registry experiment and its concrete bundles.

    Attributes:
        experiment (RegisteredExperiment): Logical registry entry.
        bundles (tuple[ExperimentBundle, ...]): Concrete bundle expansion.
    """

    experiment: RegisteredExperiment
    bundles: tuple[ExperimentBundle, ...]

    @property
    def bundle(self) -> ExperimentBundle:
        """Return the only bundle when a logical experiment expands to one.

        Raises:
            ConfigError: If the logical experiment expands to more than one
                concrete bundle.
        """
        if len(self.bundles) != 1:
            msg = "Registry experiment expands to multiple bundles."
            raise ConfigError(msg)
        return self.bundles[0]


@dataclass(frozen=True, slots=True)
class _ResolvedDatasetExperiment:
    """Dataset manifest context used while expanding registry runs.

    Attributes:
        experiments_root (Path): Root directory containing experiment configs.
        config_path (Path): Dataset manifest path being expanded.
        config (_DatasetExperimentConfig): Decoded dataset manifest config.
    """

    experiments_root: Path
    config_path: Path
    config: _DatasetExperimentConfig


@dataclass(frozen=True, slots=True)
class _ConcreteBundleRequest:
    """Inputs required to build one concrete registry-backed bundle.

    Attributes:
        context (_ResolvedDatasetExperiment): Dataset manifest context.
        model_ref (str): Named model config being expanded.
        run_group (str): Scheduling group assigned to the bundle.
        model_overrides (dict[str, object]): Concrete model overrides.
        experiment (RegisteredExperiment): Logical registry experiment.
        repo_root (Path): Repository root used for path resolution.
    """

    context: _ResolvedDatasetExperiment
    model_ref: str
    run_group: str
    model_overrides: dict[str, object]
    experiment: RegisteredExperiment
    repo_root: Path


@dataclass(frozen=True, slots=True)
class _ExperimentSetBuildRequest:
    """Inputs required to expand one experiment-set table.

    Attributes:
        datasets (tuple[str, ...]): Dataset names included in the set.
        model_sets (tuple[str, ...]): Shared model sets associated with the
            set.
        models (tuple[str, ...]): Inline model references associated with the
            set.
        groups (tuple[str, ...]): Derived groups assigned to the experiments.
        overrides (dict[str, dict[str, object]]): Override map for the set.
        description (str | None): Optional human-readable description.
    """

    datasets: tuple[str, ...]
    model_sets: tuple[str, ...]
    models: tuple[str, ...]
    groups: tuple[str, ...]
    overrides: dict[str, dict[str, object]]
    description: str | None


class _DatasetExperimentConfig(msgspec.Struct, frozen=True):
    """Dataset manifest loaded from one TOML file.

    Attributes:
        name (str): Canonical manifest name.
        dataset (DatasetVariantConfig): Decoded dataset variant definition.
        models (list[dict[str, object]]): Concrete model entries from the
            manifest.
        model (dict[str, object] | None): Inline single-model entry, if used.
        overrides (dict[str, object]): Fixed experiment-level overrides.
        axes (list[SweepAxisConfig]): Sweep axes for concrete bundle expansion.
        results_root (Path): Root directory for deterministic outputs.
        description (str | None): Optional human-readable description.
        max_workers (WorkerCount): Parallelism policy for bundle expansion.
    """

    name: str
    dataset: DatasetVariantConfig
    models: list[dict[str, object]] = msgspec.field(default_factory=list)
    model: dict[str, object] | None = None
    overrides: dict[str, object] = msgspec.field(default_factory=dict)
    axes: list[SweepAxisConfig] = msgspec.field(default_factory=list)
    results_root: Path = Path("experiments/results")
    description: str | None = None
    max_workers: WorkerCount = "auto"


class _RegistryFile(msgspec.Struct, frozen=True):
    """Top-level registry TOML structure.

    Attributes:
        model_sets (dict[str, object]): Raw model-set tables.
        experiments (dict[str, object]): Raw named experiment tables.
        experiment_sets (dict[str, object]): Raw experiment-set tables.
    """

    model_sets: dict[str, object] = msgspec.field(default_factory=dict)
    experiments: dict[str, object] = msgspec.field(default_factory=dict)
    experiment_sets: dict[str, object] = msgspec.field(default_factory=dict)


def _path_hook(type_: type[Path], obj: str) -> Path:
    del type_
    return Path(obj)


def _path_dec_hook(type_: type, obj: object) -> object:
    if type_ is object:
        return obj
    if type_ is not Path or not isinstance(obj, str):
        msg = f"Unsupported decoded type: {type_!r}"
        raise NotImplementedError(msg)
    return _path_hook(type_, obj)


def _normalize_toml_table(
    obj: object,
    *,
    expected_type: str,
) -> dict[str, object]:
    if not isinstance(obj, dict):
        msg = f"{expected_type} config must decode from a TOML table."
        raise TypeError(msg)
    return {str(key): value for key, value in obj.items()}


def _load_toml(path: Path, *, expected_type: type[TDecoded]) -> TDecoded:
    try:
        return msgspec.toml.decode(
            path.read_bytes(),
            type=expected_type,
            dec_hook=_path_dec_hook,
        )
    except (
        msgspec.ValidationError,
        msgspec.DecodeError,
        TypeError,
        ValueError,
    ) as exc:
        msg = f"{path}: {exc}"
        raise ConfigError(msg) from exc


def _decode_toml_file(
    path: Path,
    *,
    decode: Callable[[object], TDecoded],
) -> TDecoded:
    try:
        raw = msgspec.toml.decode(path.read_bytes())
        return decode(raw)
    except FileNotFoundError as exc:
        msg = f"Missing config file: {path}"
        raise ConfigError(msg) from exc
    except (
        msgspec.ValidationError,
        msgspec.DecodeError,
        TypeError,
        ValueError,
    ) as exc:
        msg = f"{path}: {exc}"
        raise ConfigError(msg) from exc


def _decode_sequence_config(obj: object | None) -> SequenceConfig:
    if obj is None:
        return EntitySequenceConfig()
    return msgspec.convert(obj, type=SequenceConfig, dec_hook=_path_dec_hook)


def _decode_dataset_source_config(obj: object) -> DatasetSourceConfig:
    raw_config = _normalize_toml_table(obj, expected_type="dataset source")
    tag_value = raw_config.get("type")
    if not isinstance(tag_value, str):
        msg = "dataset source config must define `type`."
        raise TypeError(msg)
    config_type = _DATASET_SOURCE_CONFIG_TYPES.get(tag_value)
    if config_type is None:
        msg = f"Unsupported dataset source: {tag_value!r}"
        raise ValueError(msg)
    return msgspec.convert(
        raw_config,
        type=config_type,
        dec_hook=_path_dec_hook,
    )


def _decode_label_reader_config(obj: object) -> LabelReaderConfig:
    raw_config = _normalize_toml_table(obj, expected_type="label reader")
    tag_value = raw_config.get("type")
    if not isinstance(tag_value, str):
        msg = "label reader config must define `type`."
        raise TypeError(msg)
    config_type = _LABEL_READER_CONFIG_TYPES.get(tag_value)
    if config_type is None:
        msg = f"Unsupported label reader: {tag_value!r}"
        raise ValueError(msg)
    return msgspec.convert(raw_config, type=config_type, dec_hook=_path_dec_hook)


def _decode_model_config(obj: object) -> ExperimentModelConfig:
    raw_config = _normalize_toml_table(obj, expected_type="model")
    detector_name = raw_config.get("detector")
    if not isinstance(detector_name, str):
        msg = "model config must define `detector`."
        raise TypeError(msg)
    return decode_experiment_model_config(
        raw_config,
        config_type=resolve_model_config_type(detector_name),
        dec_hook=_path_dec_hook,
    )


def _decode_dataset_config(obj: object) -> DatasetVariantConfig:
    raw_config = _normalize_toml_table(obj, expected_type="dataset")
    return DatasetVariantConfig(
        name=str(raw_config["name"]),
        dataset_name=str(raw_config["dataset_name"]),
        preset=_optional_str(raw_config.get("preset")),
        source=(
            None
            if raw_config.get("source") is None
            else _decode_dataset_source_config(raw_config["source"])
        ),
        structured_parser=_optional_str(raw_config.get("structured_parser")),
        template_parser=str(raw_config.get("template_parser", "drain3")),
        label_reader=(
            None
            if raw_config.get("label_reader") is None
            else _decode_label_reader_config(raw_config["label_reader"])
        ),
        cache_paths=(
            None
            if raw_config.get("cache_paths") is None
            else msgspec.convert(
                raw_config["cache_paths"],
                type=CachePathsConfigModel,
                dec_hook=_path_dec_hook,
            )
        ),
        evaluation_unit=(
            None
            if raw_config.get("evaluation_unit") is None
            else msgspec.convert(raw_config["evaluation_unit"], type=EvaluationUnit)
        ),
        sequence=_decode_sequence_config(raw_config.get("sequence")),
        description=_optional_str(raw_config.get("description")),
    )


def _decode_dataset_experiment_config(obj: object) -> _DatasetExperimentConfig:
    raw_config = _normalize_toml_table(obj, expected_type="dataset experiment")
    dataset_obj = raw_config.get("dataset")
    models_obj = raw_config.get("models")
    model_obj = raw_config.get("model")
    if not isinstance(dataset_obj, dict):
        msg = "dataset experiment config must define a `[dataset]` table."
        raise TypeError(msg)
    if models_obj is None:
        models_obj = [] if model_obj is None else [model_obj]
    if not isinstance(models_obj, list):
        msg = "dataset experiment config `models` must be a TOML array."
        raise TypeError(msg)
    results_root_obj = raw_config.get("results_root", "experiments/results")
    if not isinstance(results_root_obj, (str, Path)):
        msg = "dataset experiment config `results_root` must be a path string."
        raise TypeError(msg)
    results_root = (
        Path(results_root_obj)
        if isinstance(results_root_obj, str)
        else results_root_obj
    )
    max_workers_obj = raw_config.get("max_workers", "auto")
    if max_workers_obj != "auto" and (
        not isinstance(max_workers_obj, int) or isinstance(max_workers_obj, bool)
    ):
        msg = "dataset experiment max_workers must be a positive integer or `auto`."
        raise TypeError(msg)
    if max_workers_obj == "auto":
        max_workers: WorkerCount = "auto"
    elif isinstance(max_workers_obj, int):
        max_workers = max_workers_obj
    else:
        msg = "dataset experiment max_workers must be a positive integer or `auto`."
        raise TypeError(msg)
    return _DatasetExperimentConfig(
        name=str(raw_config["name"]),
        dataset=_decode_dataset_config(dataset_obj),
        models=[_normalize_model_entry(model) for model in models_obj],
        model=(None if model_obj is None else _normalize_model_entry(model_obj)),
        overrides=msgspec.convert(
            raw_config.get("overrides", {}),
            type=dict[str, object],
            dec_hook=_path_dec_hook,
        ),
        axes=msgspec.convert(
            raw_config.get("axes", []),
            type=list[SweepAxisConfig],
            dec_hook=_path_dec_hook,
        ),
        results_root=results_root,
        description=_optional_str(raw_config.get("description")),
        max_workers=max_workers,
    )


def _normalize_model_entry(model: object) -> dict[str, object]:
    if isinstance(model, str):
        return {"ref": model}
    return _normalize_toml_table(model, expected_type="model")


def _resolve_named_config(config_dir: Path, config_name: str) -> Path:
    candidate = config_dir / f"{config_name}.toml"
    if candidate.exists():
        return candidate
    msg = f"Missing named config: {candidate}"
    raise ConfigError(msg)


def _find_experiments_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.name == "experiments":
            return candidate
    msg = f"Could not locate experiments root for {path}."
    raise ConfigError(msg)


def _find_datasets_root(path: Path) -> Path:
    experiments_root = _find_experiments_root(path.resolve())
    return experiments_root / "configs" / "datasets"


def _find_models_root(path: Path) -> Path:
    experiments_root = _find_experiments_root(path.resolve())
    return experiments_root / "configs" / "models"


def _resolve_dataset_manifest_path(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    datasets_root = _find_datasets_root(path)
    candidates = [
        datasets_root / "bgl" / path.name.removeprefix("bgl_"),
        datasets_root / "hdfs" / path.name.removeprefix("hdfs_"),
        datasets_root / "openstack" / path.name.removeprefix("openstack_"),
        datasets_root / "shared" / path.name,
    ]
    if path.name == "entity_chronological_base.toml":
        candidates.insert(0, datasets_root / "shared" / path.name)
    matches = [candidate.resolve() for candidate in candidates if candidate.exists()]
    if not matches:
        matches = [
            candidate.resolve()
            for candidate in datasets_root.rglob(path.name)
            if candidate.is_file()
        ]
    if len(matches) == 1:
        return matches[0]
    return path.resolve()


def _load_merged_dataset_experiment_config(
    path: Path,
    *,
    seen_paths: tuple[Path, ...],
) -> dict[str, object]:
    raw_config = _normalize_toml_table(
        _decode_toml_file(path, decode=lambda raw: raw),
        expected_type="dataset experiment",
    )
    extends = raw_config.pop("extends", None)
    if extends is None:
        return raw_config
    if not isinstance(extends, str) or not extends.strip():
        msg = f"{path}: dataset experiment `extends` must be a non-empty string."
        raise ConfigError(msg)
    parent_path = _resolve_dataset_experiment_path(path, extends)
    if parent_path in seen_paths:
        chain = " -> ".join(
            str(candidate) for candidate in (*seen_paths, path, parent_path)
        )
        msg = f"Detected cyclic dataset experiment inheritance: {chain}"
        raise ConfigError(msg)
    parent_config = _load_merged_dataset_experiment_config(
        parent_path,
        seen_paths=(*seen_paths, path),
    )
    return _merge_toml_tables(parent_config, raw_config)


def _resolve_dataset_experiment_path(path: Path, extends: str) -> Path:
    candidate = Path(extends)
    if not candidate.is_absolute():
        candidate = (path.parent / candidate).resolve()
    if candidate.exists():
        return candidate
    if candidate.suffix != ".toml":
        candidate_with_suffix = candidate.with_suffix(".toml")
        if candidate_with_suffix.exists():
            return candidate_with_suffix
    return candidate


def _merge_toml_tables(
    parent: dict[str, object],
    child: dict[str, object],
) -> dict[str, object]:
    merged = dict(parent)
    for key, value in child.items():
        parent_value = merged.get(key)
        if isinstance(parent_value, dict) and isinstance(value, dict):
            merged[key] = _merge_toml_tables(
                _normalize_toml_table(
                    parent_value,
                    expected_type="nested dataset experiment",
                ),
                _normalize_toml_table(
                    value,
                    expected_type="nested dataset experiment",
                ),
            )
        else:
            merged[key] = value
    return merged


@cache
def _load_dataset_experiment_config(path: Path) -> _DatasetExperimentConfig:
    resolved_config_path = _resolve_dataset_manifest_path(path)
    raw_config = _load_merged_dataset_experiment_config(
        resolved_config_path,
        seen_paths=(),
    )
    if not isinstance(raw_config, dict):
        msg = (
            f"{resolved_config_path}: dataset experiment config must decode "
            "from a TOML table."
        )
        raise ConfigError(msg)
    return _decode_dataset_experiment_config(raw_config)


def _dataset_config_key(path: Path, *, experiments_root: Path) -> str:
    datasets_root = experiments_root / "configs" / "datasets"
    try:
        relative_path = path.relative_to(datasets_root)
    except ValueError as exc:
        msg = f"{path}: dataset manifest must live under {datasets_root}."
        raise ConfigError(msg) from exc
    if relative_path.suffix == ".toml":
        relative_path = relative_path.with_suffix("")
    return relative_path.as_posix()


def _registry_name_to_dataset_path(dataset: str, *, experiments_root: Path) -> Path:
    return (
        experiments_root / "configs" / "datasets" / Path(dataset).with_suffix(".toml")
    )


def _decode_model_set(name: str, obj: object) -> ModelSetDefinition:
    raw = _normalize_toml_table(obj, expected_type=f"model set {name}")
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        msg = f"model set {name!r} must define a non-empty `models` array."
        raise TypeError(msg)
    model_refs: list[str] = []
    for model in models:
        if not isinstance(model, str) or not model.strip():
            msg = f"model set {name!r} contains an invalid model reference."
            raise TypeError(msg)
        model_refs.append(model)
    overrides = msgspec.convert(
        raw.get("overrides", {}),
        type=dict[str, object],
        dec_hook=_path_dec_hook,
    )
    description = _optional_str(raw.get("description"))
    _reject_unknown_keys(
        raw,
        allowed={"models", "overrides", "description"},
        context=f"model set {name!r}",
    )
    return ModelSetDefinition(
        name=name,
        models=tuple(model_refs),
        overrides=overrides,
        description=description,
    )


def _decode_experiment_definition(
    name: str,
    obj: object,
    *,
    experiment_kind: str,
) -> RegisteredExperiment:
    raw = _normalize_toml_table(obj, expected_type=experiment_kind)
    dataset = raw.get("dataset")
    inline_models, model_sets = _decode_model_ref_lists(
        raw,
        context=f"{experiment_kind} {name!r}",
    )
    overrides = _decode_overrides_map(
        raw.get("overrides", {}),
        context=f"{experiment_kind} {name!r}",
    )
    description = _optional_str(raw.get("description"))

    if not isinstance(dataset, str) or not dataset.strip():
        msg = f"{experiment_kind} {name!r} must define a non-empty `dataset`."
        raise TypeError(msg)

    groups = _derive_groups(entry_name=name, model_sets=model_sets)

    _reject_unknown_keys(
        raw,
        allowed={
            "dataset",
            "models",
            "model_sets",
            "overrides",
            "description",
        },
        context=f"{experiment_kind} {name!r}",
    )

    return RegisteredExperiment(
        name=name,
        dataset=dataset,
        models=inline_models,
        model_sets=model_sets,
        groups=groups,
        overrides=overrides,
        description=description,
    )


def _decode_experiment_set_definition(
    name: str,
    obj: object,
) -> list[RegisteredExperiment]:
    raw = _normalize_toml_table(obj, expected_type=f"experiment set {name}")
    inline_models, model_sets = _decode_model_ref_lists(
        raw,
        context=f"experiment set {name!r}",
    )
    datasets = _require_string_list(
        raw.get("datasets"),
        context=f"experiment set {name!r}",
        field_name="datasets",
    )
    overrides = _decode_overrides_map(
        raw.get("overrides", {}),
        context=f"experiment set {name!r}",
    )
    description = _optional_str(raw.get("description"))
    groups = _derive_groups(entry_name=name, model_sets=model_sets)

    _reject_unknown_keys(
        raw,
        allowed={
            "models",
            "model_sets",
            "datasets",
            "overrides",
            "description",
        },
        context=f"experiment set {name!r}",
    )

    return _build_experiment_set_entries(
        _ExperimentSetBuildRequest(
            datasets=datasets,
            model_sets=model_sets,
            models=inline_models,
            groups=groups,
            overrides=overrides,
            description=description,
        ),
    )


def _decode_overrides_map(
    obj: object,
    *,
    context: str,
) -> dict[str, dict[str, object]]:
    raw = _normalize_toml_table(obj, expected_type=f"{context} overrides")
    overrides: dict[str, dict[str, object]] = {}
    for model_set_name, payload in raw.items():
        if not isinstance(payload, dict):
            msg = f"{context} overrides for {model_set_name!r} must be a TOML table."
            raise TypeError(msg)
        overrides[str(model_set_name)] = {
            str(key): value for key, value in payload.items()
        }
    return overrides


def _decode_model_ref_lists(
    raw: dict[str, object],
    *,
    context: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    inline_models = raw.get("models")
    model_sets = raw.get("model_sets")
    decoded_inline_models = (
        _require_string_list(
            inline_models,
            context=context,
            field_name="models",
        )
        if inline_models is not None
        else ()
    )
    decoded_model_sets = (
        _require_string_list(
            model_sets,
            context=context,
            field_name="model_sets",
        )
        if model_sets is not None
        else ()
    )
    if not decoded_inline_models and not decoded_model_sets:
        msg = f"{context} must define `models` or `model_sets`."
        raise TypeError(msg)
    return decoded_inline_models, decoded_model_sets


def _require_string_list(
    value: object,
    *,
    context: str,
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        msg = f"{context} must define a non-empty `{field_name}` array."
        raise TypeError(msg)
    values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            msg = f"{context} contains an invalid {field_name[:-1]}."
            raise TypeError(msg)
        values.append(item)
    return tuple(values)


def _build_experiment_set_entries(
    request: _ExperimentSetBuildRequest,
) -> list[RegisteredExperiment]:
    return [
        RegisteredExperiment(
            name=_slugify_label(dataset),
            dataset=dataset,
            models=request.models,
            model_sets=request.model_sets,
            groups=request.groups,
            overrides=request.overrides,
            description=request.description,
        )
        for dataset in request.datasets
    ]


def _reject_unknown_keys(
    raw: dict[str, object],
    *,
    allowed: set[str],
    context: str,
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        msg = f"{context} contains unsupported fields: {', '.join(unknown)}."
        raise ConfigError(msg)


def _derive_groups(
    *,
    entry_name: str | None,
    model_sets: tuple[str, ...],
) -> tuple[str, ...]:
    groups: list[str] = []
    if entry_name is not None:
        groups.append(entry_name)
    groups.extend(model_sets)
    deduped: list[str] = []
    for group in groups:
        if group not in deduped:
            deduped.append(group)
    return tuple(deduped)


def _slugify_label(value: str) -> str:
    normalised = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return re.sub(r"_+", "_", normalised)


@dataclass(frozen=True, slots=True)
class ExperimentRegistry:
    """Validated registry of logical experiments and reusable model sets.

    Attributes:
        model_sets (tuple[ModelSetDefinition, ...]): Loaded model-set
            definitions.
        experiments (tuple[RegisteredExperiment, ...]): Loaded logical
            experiments.
    """

    model_sets: tuple[ModelSetDefinition, ...]
    experiments: tuple[RegisteredExperiment, ...]
    _model_sets_by_name: dict[str, ModelSetDefinition] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    _experiments_by_name: dict[str, RegisteredExperiment] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        """Build quick lookup tables for validated registry objects."""
        object.__setattr__(
            self,
            "_model_sets_by_name",
            {model_set.name: model_set for model_set in self.model_sets},
        )
        object.__setattr__(
            self,
            "_experiments_by_name",
            {experiment.name: experiment for experiment in self.experiments},
        )

    def names(self) -> tuple[str, ...]:
        """Return experiment names in registry order.

        Returns:
            tuple[str, ...]: Experiment names in registry order.
        """
        return tuple(experiment.name for experiment in self.experiments)

    def require(self, name: str) -> RegisteredExperiment:
        """Return one named logical experiment.

        Args:
            name (str): Registry experiment name to resolve.

        Returns:
            RegisteredExperiment: Loaded logical experiment definition.

        Raises:
            ConfigError: If the experiment name is not present in the registry.
        """
        try:
            return self._experiments_by_name[name]
        except KeyError as exc:
            msg = f"Unknown registry experiment: {name!r}"
            raise ConfigError(msg) from exc

    def select(
        self,
        *,
        names: tuple[str, ...] = (),
        groups: tuple[str, ...] = (),
    ) -> tuple[RegisteredExperiment, ...]:
        """Select named experiments and/or group-filtered experiments.

        Args:
            names (tuple[str, ...]): Explicit registry experiment names to
                include.
            groups (tuple[str, ...]): Registry groups to include.

        Returns:
            tuple[RegisteredExperiment, ...]: Registry experiments in their
                original order.

        Raises:
            ConfigError: If a requested name or group is unknown.
        """
        if not names and not groups:
            return self.experiments
        selected: list[RegisteredExperiment] = []
        seen: set[str] = set()
        for experiment_name in (*names, *groups):
            if experiment_name in self._experiments_by_name:
                experiment = self._experiments_by_name[experiment_name]
                if experiment.name not in seen:
                    selected.append(experiment)
                    seen.add(experiment.name)
                continue
            group_matches = [
                experiment
                for experiment in self.experiments
                if experiment_name in experiment.groups
            ]
            if not group_matches and experiment_name not in self._experiments_by_name:
                msg = f"Unknown registry experiment or group: {experiment_name!r}"
                raise ConfigError(msg)
            for experiment in group_matches:
                if experiment.name not in seen:
                    selected.append(experiment)
                    seen.add(experiment.name)
        return tuple(selected)

    def model_set(self, name: str) -> ModelSetDefinition:
        """Return one named model set.

        Args:
            name (str): Model-set name to resolve.

        Returns:
            ModelSetDefinition: Loaded model-set definition.

        Raises:
            ConfigError: If the model set name is not present in the registry.
        """
        try:
            return self._model_sets_by_name[name]
        except KeyError as exc:
            msg = f"Unknown model set: {name!r}"
            raise ConfigError(msg) from exc

    def resolve_experiment(
        self,
        name: str,
        *,
        registry_path: Path,
        repo_root: Path,
    ) -> ResolvedRegistryExperiment:
        """Resolve one logical experiment into concrete bundles.

        Args:
            name (str): Registry experiment name to resolve.
            registry_path (Path): Path to the registry TOML file.
            repo_root (Path): Repository root used to resolve relative paths.

        Returns:
            ResolvedRegistryExperiment: Logical registry entry plus concrete
                bundle expansion.
        """
        experiment = self.require(name)
        return ResolvedRegistryExperiment(
            experiment=experiment,
            bundles=tuple(
                _build_experiment_bundles(
                    registry=self,
                    registry_path=registry_path,
                    repo_root=repo_root,
                    experiment=experiment,
                ),
            ),
        )


def load_experiment_registry(
    registry_path: Path,
    *,
    repo_root: Path | None = None,
) -> ExperimentRegistry:
    """Load and validate the named experiment registry.

    Args:
        registry_path (Path): Path to the registry TOML file.
        repo_root (Path | None): Repository root used to resolve relative
            paths.

    Returns:
        ExperimentRegistry: Validated registry with resolved model sets,
            and logical experiments.

    Raises:
        ConfigError: If the registry file is missing or malformed, or if any
            referenced dataset/model config cannot be resolved.
    """
    resolved_repo_root = Path.cwd() if repo_root is None else repo_root
    resolved_registry_path = _resolve_path(registry_path, resolved_repo_root)
    raw_registry = _load_toml(resolved_registry_path, expected_type=_RegistryFile)
    if not isinstance(raw_registry, _RegistryFile):
        msg = (
            f"{resolved_registry_path}: registry config must decode from a TOML table."
        )
        raise ConfigError(msg)
    model_sets = _load_model_sets(raw_registry.model_sets)
    experiments = _load_registered_experiments(
        raw_registry.experiments,
        raw_registry.experiment_sets,
    )
    registry = ExperimentRegistry(
        model_sets=model_sets,
        experiments=experiments,
    )
    _validate_registry_references(
        registry,
        registry_path=resolved_registry_path,
    )
    return registry


def resolve_registry_experiment(
    name: str,
    *,
    registry_path: Path,
    repo_root: Path | None = None,
) -> ResolvedRegistryExperiment:
    """Resolve one registry experiment into concrete bundles.

    Args:
        name (str): Registry experiment name to resolve.
        registry_path (Path): Path to the registry TOML file.
        repo_root (Path | None): Repository root used to resolve relative
            paths.

    Returns:
        ResolvedRegistryExperiment: Logical registry entry plus concrete
            bundle expansion.
    """
    registry = load_experiment_registry(
        registry_path,
        repo_root=repo_root,
    )
    resolved_repo_root = Path.cwd() if repo_root is None else repo_root
    return registry.resolve_experiment(
        name,
        registry_path=_resolve_path(registry_path, resolved_repo_root),
        repo_root=resolved_repo_root,
    )


def _load_model_sets(
    raw_model_sets: dict[str, object],
) -> tuple[ModelSetDefinition, ...]:
    if not raw_model_sets:
        return ()
    model_sets: list[ModelSetDefinition] = []
    seen: set[str] = set()
    for name, obj in raw_model_sets.items():
        if name in seen:
            msg = f"Duplicate model set definition: {name!r}"
            raise ConfigError(msg)
        seen.add(name)
        try:
            model_sets.append(_decode_model_set(name, obj))
        except (TypeError, ValueError) as exc:
            msg = f"model set {name!r}: {exc}"
            raise ConfigError(msg) from exc
    return tuple(model_sets)


def _load_registered_experiments(
    raw_experiments: dict[str, object],
    raw_experiment_sets: dict[str, object],
) -> tuple[RegisteredExperiment, ...]:
    experiments: list[RegisteredExperiment] = []
    seen: set[str] = set()
    for name, obj in raw_experiments.items():
        try:
            resolved = _decode_experiment_definition(
                name,
                obj,
                experiment_kind="experiment",
            )
        except (TypeError, ValueError) as exc:
            msg = f"experiment {name!r}: {exc}"
            raise ConfigError(msg) from exc
        if resolved.name in seen:
            msg = f"Duplicate experiment name: {resolved.name!r}"
            raise ConfigError(msg)
        seen.add(resolved.name)
        experiments.append(resolved)
    for name, obj in raw_experiment_sets.items():
        try:
            resolved = _decode_experiment_set_definition(
                name,
                obj,
            )
        except (TypeError, ValueError) as exc:
            msg = f"experiment set {name!r}: {exc}"
            raise ConfigError(msg) from exc
        for experiment in resolved:
            if experiment.name in seen:
                msg = f"Duplicate experiment name: {experiment.name!r}"
                raise ConfigError(msg)
            seen.add(experiment.name)
            experiments.append(experiment)
    if not experiments:
        msg = "Registry must define at least one experiment."
        raise ConfigError(msg)
    return tuple(experiments)


def _validate_registry_references(
    registry: ExperimentRegistry,
    *,
    registry_path: Path,
) -> None:
    experiments_root = _find_experiments_root(registry_path)
    models_root = experiments_root / "configs" / "models"
    if registry.model_sets:
        _validate_model_set_configs(registry, models_root=models_root)
    _validate_experiment_definitions(
        registry,
        experiments_root=experiments_root,
        models_root=models_root,
    )


def _validate_model_set_configs(
    registry: ExperimentRegistry,
    *,
    models_root: Path,
) -> None:
    for model_set in registry.model_sets:
        for model_ref in model_set.models:
            _resolve_named_config(models_root, model_ref)


def _validate_experiment_definitions(
    registry: ExperimentRegistry,
    *,
    experiments_root: Path,
    models_root: Path,
) -> None:
    for experiment in registry.experiments:
        dataset_path = _registry_name_to_dataset_path(
            experiment.dataset,
            experiments_root=experiments_root,
        )
        if not dataset_path.exists():
            msg = (
                f"Missing dataset config for registry experiment "
                f"{experiment.name!r}: {dataset_path}"
            )
            raise ConfigError(msg)
        for model_ref in experiment.models:
            _resolve_named_config(models_root, model_ref)
        for model_set_name in experiment.model_sets:
            model_set = registry.model_set(model_set_name)
            for model_ref in model_set.models:
                _resolve_named_config(models_root, model_ref)
        for override_name in experiment.overrides:
            if override_name not in {*experiment.models, *experiment.model_sets}:
                msg = (
                    f"Registry experiment {experiment.name!r} overrides "
                    f"unknown model or model set {override_name!r}."
                )
                raise ConfigError(msg)


def _build_experiment_bundles(
    *,
    registry: ExperimentRegistry,
    registry_path: Path,
    repo_root: Path,
    experiment: RegisteredExperiment,
) -> list[ExperimentBundle]:
    experiments_root = _find_experiments_root(registry_path)
    dataset_path = _registry_name_to_dataset_path(
        experiment.dataset,
        experiments_root=experiments_root,
    )
    context = _ResolvedDatasetExperiment(
        experiments_root=experiments_root,
        config_path=dataset_path,
        config=_load_dataset_experiment_config(dataset_path),
    )
    bundles: list[ExperimentBundle] = []
    for model_set_name in experiment.model_sets:
        model_set = registry.model_set(model_set_name)
        for model_ref in model_set.models:
            model_overrides = _merge_model_overrides(
                _select_model_overrides(
                    model_set.overrides,
                    model_ref=model_ref,
                ),
                experiment.overrides.get(model_set_name, {}),
            )
            bundles.append(
                _build_concrete_bundle(
                    request=_ConcreteBundleRequest(
                        context=context,
                        model_ref=model_ref,
                        run_group=model_set_name,
                        model_overrides=model_overrides,
                        experiment=experiment,
                        repo_root=repo_root,
                    ),
                ),
            )
    bundles.extend(
        _build_concrete_bundle(
            request=_ConcreteBundleRequest(
                context=context,
                model_ref=model_ref,
                run_group=model_ref,
                model_overrides=experiment.overrides.get(model_ref, {}),
                experiment=experiment,
                repo_root=repo_root,
            ),
        )
        for model_ref in experiment.models
    )
    return bundles


def _merge_model_overrides(*sources: dict[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for source in sources:
        merged.update(source)
    return merged


def _select_model_overrides(
    overrides: dict[str, object],
    *,
    model_ref: str,
) -> dict[str, object]:
    """Return the concrete override table for one model reference.

    Args:
        overrides (dict[str, object]): Model-set override mapping from the
            registry entry.
        model_ref (str): Model reference name used to select a nested override
            table.

    Returns:
        dict[str, object]: Concrete override table for the requested model.

    Model-set overrides may be declared as a flat table or as a nested table
    keyed by model reference. The latter is the shape used by the checked-in
    DeepCASE ablations, so the builder needs to select the correct per-model
    sub-table before applying the overrides to the decoded config.

    Raises:
        ConfigError: If the selected nested override entry is not a TOML
            table.
    """
    if not overrides:
        return {}

    nested_override_tables = any(
        isinstance(value, dict) for value in overrides.values()
    )
    if not nested_override_tables:
        return dict(overrides)

    selected = overrides.get(model_ref)
    if selected is None:
        return {}
    if not isinstance(selected, dict):
        msg = f"Model overrides for {model_ref!r} must be a TOML table."
        raise ConfigError(msg)
    return {str(key): value for key, value in selected.items()}


def _build_concrete_bundle(
    *,
    request: _ConcreteBundleRequest,
) -> ExperimentBundle:
    context = request.context
    model, model_path = _resolve_model_config(
        {"ref": request.model_ref},
        repo_root=request.repo_root,
        fallback_path=context.config_path,
    )
    applied_overrides = dict(context.config.overrides)
    applied_overrides.update(request.model_overrides)
    dataset = _apply_config_overrides(
        config=context.config.dataset,
        overrides=applied_overrides,
        prefix="dataset",
        decode=_decode_dataset_config,
    )
    model = _apply_config_overrides(
        config=model,
        overrides=applied_overrides,
        prefix="model",
        decode=_decode_model_config,
    )
    concrete_name = _build_concrete_name(
        default_name=None,
        dataset=dataset,
        model=model,
        applied_overrides=applied_overrides,
    )
    return ExperimentBundle(
        experiments_root=context.experiments_root,
        repo_root=request.repo_root,
        sweep_path=context.config_path,
        dataset_path=context.config_path,
        model_path=model_path,
        sweep=context.config,
        dataset=dataset,
        model=model,
        concrete_name=concrete_name,
        run_group=request.run_group,
        applied_overrides=applied_overrides,
    ).with_experiment_metadata(
        experiment_name=request.experiment.name,
        experiment_groups=request.experiment.groups,
    )


def _resolve_model_config(
    model_config: dict[str, object],
    *,
    repo_root: Path,
    fallback_path: Path,
) -> tuple[ExperimentModelConfig, Path]:
    ref = model_config.get("ref")
    if ref is not None:
        if not isinstance(ref, str):
            msg = "model `ref` must be a string."
            raise TypeError(msg)
        return _load_model_config_reference(ref, repo_root=repo_root)
    model = _decode_model_config(
        {
            key: value
            for key, value in model_config.items()
            if key not in {"overrides", "axes", "run_group"}
        },
    )
    return model, fallback_path


@cache
def _load_model_config_reference(
    ref: str,
    *,
    repo_root: Path,
) -> tuple[ExperimentModelConfig, Path]:
    model_path = repo_root / "experiments" / "configs" / "models" / f"{ref}.toml"
    if not model_path.exists():
        msg = f"Missing model config for reference: {model_path}"
        raise ConfigError(msg)
    model = _decode_toml_file(model_path, decode=_decode_model_config)
    return model, model_path


def _apply_config_overrides(
    *,
    config: TDecoded,
    overrides: dict[str, object],
    prefix: str,
    decode: Callable[[object], TDecoded],
) -> TDecoded:
    updated = serialise_config(config)
    applied = False
    for path, value in overrides.items():
        if not path.startswith(f"{prefix}."):
            continue
        applied = True
        _set_nested_value(updated, path.split(".")[1:], value, root_name=prefix)
    if not applied:
        return config
    return decode(updated)


def _set_nested_value(
    payload: dict[str, object],
    segments: list[str],
    value: object,
    *,
    root_name: str,
) -> None:
    current = payload
    traversed = [root_name]
    for segment in segments[:-1]:
        if segment not in current:
            msg = f"Unknown override path: {'.'.join([*traversed, segment])!r}."
            raise ConfigError(msg)
        next_table = _require_object_dict(current[segment], path=".".join(traversed))
        current[segment] = next_table
        current = next_table
        traversed.append(segment)
    final_segment = segments[-1]
    if final_segment not in current:
        msg = f"Unknown override path: {'.'.join([*traversed, final_segment])!r}."
        raise ConfigError(msg)
    current[final_segment] = value


def _require_object_dict(value: object, *, path: str) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    msg = f"Override path {path!r} is not a table."
    raise ConfigError(msg)


def _build_concrete_name(
    *,
    default_name: str | None,
    dataset: DatasetVariantConfig,
    model: ExperimentModelConfig,
    applied_overrides: dict[str, object],
) -> str:
    if default_name is not None and not applied_overrides:
        return _slugify_label(default_name)
    dataset_label = _slugify_label(_trim_known_suffixes(dataset.name))
    model_label = _slugify_label(_trim_known_suffixes(model.name))
    override_labels = [
        _override_label(path, value)
        for path, value in sorted(applied_overrides.items())
        if path not in {"dataset", "model"}
    ]
    return "_".join([dataset_label, model_label, *override_labels])


def _trim_known_suffixes(value: str) -> str:
    for suffix in ("_entity_supervised", "_entity", "_default"):
        if value.endswith(suffix):
            return value.removesuffix(suffix)
    return value


def _override_label(path: str, value: object) -> str:
    field_name = path.rsplit(".", maxsplit=1)[-1]
    return f"{_slugify_label(field_name)}_{_slugify_value(value)}"


def _slugify_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, "g").replace(".", "p")
    return _slugify_label(str(value))


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()
