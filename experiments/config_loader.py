"""TOML decoding and bundle loading for dataset-owned experiment matrices."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
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
from experiments.registry import load_experiment_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from experiments.models import ExperimentModelConfig

TDecoded = TypeVar("TDecoded")


@dataclass(frozen=True, slots=True)
class _ResolvedDatasetExperiment:
    """Resolved dataset-owned experiment matrix loaded from one TOML file.

    Attributes:
        experiments_root (Path): Root directory containing experiment configs.
        config_path (Path): Resolved dataset experiment path.
        config (_DatasetExperimentConfig): Decoded dataset experiment config.
    """

    experiments_root: Path
    config_path: Path
    config: _DatasetExperimentConfig


class _DatasetExperimentConfig(msgspec.Struct, frozen=True):
    """Dataset-owned experiment matrix loaded from one TOML file.

    Attributes:
        name (str): Human-readable matrix name.
        dataset (DatasetVariantConfig): Decoded dataset config.
        models (list[dict[str, object]]): Embedded model run entries.
        model (dict[str, object] | None): Backwards-compatible single model.
        overrides (dict[str, object]): Fixed overrides applied to every run.
        axes (list[SweepAxisConfig]): Cartesian-product axes for every run.
        results_root (Path): Root directory for run outputs.
        description (str | None): Optional free-text matrix description.
        max_workers (WorkerCount): Maximum concurrent concrete runs.
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
    if isinstance(results_root_obj, str):
        results_root = Path(results_root_obj)
    else:
        results_root = results_root_obj
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


def load_experiment_bundles(sweep_config_path: Path) -> list[ExperimentBundle]:
    """Load a dataset-owned experiment matrix and expand it into bundles.

    Args:
        sweep_config_path (Path): Dataset manifest TOML path to resolve.

    Returns:
        list[ExperimentBundle]: Fully resolved concrete runs derived from the
            manifest or inline scenario.

    Raises:
        ConfigError: If the manifest does not decode or is missing its root
            `experiments` directory.
    """
    resolved_config_path = _resolve_dataset_manifest_path(sweep_config_path)
    experiments_root = _find_experiments_root(resolved_config_path)
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
    config = _decode_dataset_experiment_config(raw_config)
    if not config.models and config.model is None:
        return _expand_registry_runs(
            context=_ResolvedDatasetExperiment(
                experiments_root=experiments_root,
                config_path=resolved_config_path,
                config=config,
            ),
        )
    return _expand_model_runs(
        context=_ResolvedDatasetExperiment(
            experiments_root=experiments_root,
            config_path=resolved_config_path,
            config=config,
        ),
    )


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


def _find_experiments_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.name == "experiments":
            return candidate
    msg = f"Could not locate experiments root for {path}."
    raise ConfigError(msg)


def _find_datasets_root(path: Path) -> Path:
    experiments_root = _find_experiments_root(path.resolve())
    return experiments_root / "configs" / "datasets"


def _resolve_named_config(config_dir: Path, config_name: str) -> Path:
    candidate = config_dir / f"{config_name}.toml"
    if candidate.exists():
        return candidate
    msg = f"Missing named config: {candidate}"
    raise ConfigError(msg)


def _expand_model_runs(
    *,
    context: _ResolvedDatasetExperiment,
) -> list[ExperimentBundle]:
    bundles: list[ExperimentBundle] = []
    model_entries = context.config.models or (
        [] if context.config.model is None else [context.config.model]
    )
    axis_combinations = list(product(*(axis.values for axis in context.config.axes)))
    if not axis_combinations:
        axis_combinations = [()]
    for model_config in model_entries:
        for current_axis_values in axis_combinations:
            axis_overrides = {
                axis.path: value
                for axis, value in zip(
                    context.config.axes,
                    current_axis_values,
                    strict=True,
                )
            }
            bundles.append(
                _build_concrete_bundle(
                    context=context,
                    model_config=model_config,
                    axis_overrides=axis_overrides,
                    default_name=(
                        context.config.name
                        if len(model_entries) == 1 and len(context.config.axes) == 0
                        else None
                    ),
                ),
            )
    return bundles


def _expand_registry_runs(
    *,
    context: _ResolvedDatasetExperiment,
) -> list[ExperimentBundle]:
    registry_path = context.experiments_root / "configs" / "registry.toml"
    registry = load_experiment_registry(
        registry_path,
        repo_root=context.experiments_root.parent,
    )
    dataset_key = _dataset_config_key(
        context.config_path,
        experiments_root=context.experiments_root,
    )
    selected_experiments = [
        experiment
        for experiment in registry.experiments
        if experiment.dataset == dataset_key
    ]
    if not selected_experiments:
        msg = f"No registry experiments match dataset manifest {dataset_key!r}."
        raise ConfigError(msg)
    bundles: list[ExperimentBundle] = []
    for experiment in selected_experiments:
        resolved = registry.resolve_experiment(
            experiment.name,
            registry_path=registry_path,
            repo_root=context.experiments_root.parent,
        )
        bundles.extend(resolved.bundles)
    return bundles


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


def _build_concrete_bundle(
    *,
    context: _ResolvedDatasetExperiment,
    model_config: dict[str, object],
    axis_overrides: dict[str, object],
    default_name: str | None,
) -> ExperimentBundle:
    applied_overrides = dict(context.config.overrides)
    model, model_path = _resolve_model_config(
        model_config,
        repo_root=context.experiments_root.parent,
        fallback_path=context.config_path,
    )
    run_group = _resolve_run_group(model_config)
    model_overrides_obj = model_config.get("overrides", {})
    if not isinstance(model_overrides_obj, dict):
        msg = "model overrides must be a TOML table."
        raise TypeError(msg)
    applied_overrides.update(
        {str(key): value for key, value in model_overrides_obj.items()},
    )
    applied_overrides.update(axis_overrides)
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
        default_name=default_name,
        dataset=dataset,
        model=model,
        applied_overrides=applied_overrides,
    )
    return ExperimentBundle(
        experiments_root=context.experiments_root,
        repo_root=context.experiments_root.parent,
        sweep_path=context.config_path,
        dataset_path=context.config_path,
        model_path=model_path,
        sweep=context.config,
        dataset=dataset,
        model=model,
        concrete_name=concrete_name,
        run_group=run_group,
        applied_overrides=applied_overrides,
    )


def _normalize_model_entry(model: object) -> dict[str, object]:
    if isinstance(model, str):
        return {"ref": model}
    return _normalize_toml_table(model, expected_type="model")


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
        model_path = repo_root / "experiments" / "configs" / "models" / f"{ref}.toml"
        if not model_path.exists():
            msg = f"Missing model config for reference: {model_path}"
            raise ConfigError(msg)
        model = _decode_toml_file(model_path, decode=_decode_model_config)
    else:
        model = _decode_model_config(
            {
                key: value
                for key, value in model_config.items()
                if key not in {"overrides", "axes", "run_group"}
            },
        )
        model_path = fallback_path
    return model, model_path


def _resolve_run_group(model_config: dict[str, object]) -> str:
    run_group = model_config.get("run_group", "default")
    if not isinstance(run_group, str):
        msg = "model `run_group` must be a string."
        raise ConfigError(msg)
    if not run_group:
        msg = "model `run_group` must not be empty."
        raise ConfigError(msg)
    return run_group


TConfig = TypeVar("TConfig")


def _apply_config_overrides(
    *,
    config: TConfig,
    overrides: dict[str, object],
    prefix: str,
    decode: Callable[[object], TConfig],
) -> TConfig:
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


def _slugify_label(value: str) -> str:
    normalised = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return re.sub(r"_+", "_", normalised)
