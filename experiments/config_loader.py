"""TOML decoding and bundle loading for experiment sweeps."""

from __future__ import annotations

import re
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
    ExperimentBundle,
    LabelReaderConfig,
    SequenceConfig,
    SweepConfig,
    _optional_str,
    serialise_config,
)
from experiments.models import resolve_model_config_type
from experiments.models.base import decode_experiment_model_config

if TYPE_CHECKING:
    from collections.abc import Callable

    from experiments.models import ExperimentModelConfig

TDecoded = TypeVar("TDecoded")


def _path_hook(type_: type[Path], obj: str) -> Path:
    del type_
    return Path(obj)


def _path_dec_hook(type_: type, obj: object) -> object:
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
        sequence=_decode_sequence_config(raw_config.get("sequence")),
        description=_optional_str(raw_config.get("description")),
    )


def load_experiment_bundles(sweep_config_path: Path) -> list[ExperimentBundle]:
    """Load a sweep config and expand it into concrete experiment bundles.

    Args:
        sweep_config_path (Path): Sweep config TOML path to resolve.

    Returns:
        list[ExperimentBundle]: Fully resolved concrete runs derived from the sweep.
    """
    resolved_sweep_path = sweep_config_path.resolve()
    experiments_root = _find_experiments_root(resolved_sweep_path)
    sweep = _load_toml(resolved_sweep_path, expected_type=SweepConfig)
    _resolve_config_refs(
        experiments_root=experiments_root,
        dataset=sweep.dataset,
        model=sweep.model,
    )
    return _expand_sweep(
        experiments_root=experiments_root,
        sweep_path=resolved_sweep_path,
        sweep=sweep,
    )


def _find_experiments_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.name == "experiments":
            return candidate
    msg = f"Could not locate experiments root for {path}."
    raise ConfigError(msg)


def _resolve_named_config(base_dir: Path, config_ref: str) -> Path:
    ref_path = Path(config_ref)
    candidate = (
        ref_path
        if ref_path.suffix == ".toml" and ref_path.is_absolute()
        else base_dir
        / (ref_path if ref_path.suffix == ".toml" else f"{config_ref}.toml")
    )
    resolved = candidate.resolve()
    if not resolved.exists():
        msg = f"Config file not found: {resolved}"
        raise ConfigError(msg)
    return resolved


def _resolve_config_refs(
    *,
    experiments_root: Path,
    dataset: str,
    model: str,
) -> tuple[Path, Path]:
    return (
        _resolve_named_config(experiments_root / "configs" / "datasets", dataset),
        _resolve_named_config(experiments_root / "configs" / "models", model),
    )


def _expand_sweep(
    *,
    experiments_root: Path,
    sweep_path: Path,
    sweep: SweepConfig,
) -> list[ExperimentBundle]:
    axis_combinations = list(product(*(axis.values for axis in sweep.axes)))
    if not axis_combinations:
        axis_combinations = [()]
    bundles: list[ExperimentBundle] = []
    for axis_values in axis_combinations:
        axis_overrides = {
            axis.path: value
            for axis, value in zip(sweep.axes, axis_values, strict=True)
        }
        bundles.append(
            _build_concrete_bundle(
                experiments_root=experiments_root,
                sweep_path=sweep_path,
                sweep=sweep,
                default_name=sweep.name if len(axis_combinations) == 1 else None,
                axis_overrides=axis_overrides,
            ),
        )
    return bundles


def _build_concrete_bundle(
    *,
    experiments_root: Path,
    sweep_path: Path,
    sweep: SweepConfig,
    default_name: str | None,
    axis_overrides: dict[str, object],
) -> ExperimentBundle:
    applied_overrides = dict(sweep.overrides)
    applied_overrides.update(axis_overrides)
    concrete_sweep = _apply_sweep_overrides(sweep, applied_overrides)
    dataset_path, model_path = _resolve_config_refs(
        experiments_root=experiments_root,
        dataset=concrete_sweep.dataset,
        model=concrete_sweep.model,
    )
    dataset = _apply_config_overrides(
        config=_decode_toml_file(dataset_path, decode=_decode_dataset_config),
        overrides=applied_overrides,
        prefix="dataset",
        decode=_decode_dataset_config,
    )
    model = _apply_config_overrides(
        config=_decode_toml_file(model_path, decode=_decode_model_config),
        overrides=applied_overrides,
        prefix="model",
        decode=_decode_model_config,
    )
    concrete_name = _build_concrete_name(
        default_name=default_name,
        concrete_sweep=concrete_sweep,
        applied_overrides=applied_overrides,
    )
    return ExperimentBundle(
        experiments_root=experiments_root,
        repo_root=experiments_root.parent,
        sweep_path=sweep_path,
        dataset_path=dataset_path,
        model_path=model_path,
        sweep=concrete_sweep,
        dataset=dataset,
        model=model,
        concrete_name=concrete_name,
        applied_overrides=applied_overrides,
    )


def _apply_sweep_overrides(
    sweep: SweepConfig,
    overrides: dict[str, object],
) -> SweepConfig:
    updated = serialise_config(sweep)
    for path, value in overrides.items():
        if not path.startswith("sweep."):
            continue
        leaf_segments = path.split(".")[1:]
        if leaf_segments and leaf_segments[0] in {"axes", "max_workers", "overrides"}:
            msg = f"Sweep overrides may not target {path!r}."
            raise ConfigError(msg)
        _set_nested_value(updated, leaf_segments, value, root_name="sweep")
    return msgspec.convert(updated, type=SweepConfig, dec_hook=_path_dec_hook)


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
    concrete_sweep: SweepConfig,
    applied_overrides: dict[str, object],
) -> str:
    if default_name is not None and not applied_overrides:
        return _slugify_label(default_name)
    dataset_label = _slugify_label(_trim_known_suffixes(concrete_sweep.dataset))
    model_label = _slugify_label(_trim_known_suffixes(concrete_sweep.model))
    override_labels = [
        _override_label(path, value)
        for path, value in sorted(applied_overrides.items())
        if path != "sweep.model"
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
