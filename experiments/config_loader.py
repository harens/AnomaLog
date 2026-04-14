"""TOML decoding and bundle loading for experiment configs."""

from __future__ import annotations

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
    RunConfig,
    SequenceConfig,
    _optional_str,
)
from experiments.models import resolve_model_config_type

if TYPE_CHECKING:
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


def _load_dataset_config(path: Path) -> DatasetVariantConfig:
    try:
        raw = msgspec.toml.decode(path.read_bytes())
        raw_config = _normalize_toml_table(raw, expected_type="dataset")
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


def _load_model_config(path: Path) -> ExperimentModelConfig:
    try:
        raw = msgspec.toml.decode(path.read_bytes())
        return _decode_model_config(raw)
    except (
        msgspec.ValidationError,
        msgspec.DecodeError,
        TypeError,
        ValueError,
    ) as exc:
        msg = f"{path}: {exc}"
        raise ConfigError(msg) from exc


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
    return msgspec.convert(
        raw_config,
        type=resolve_model_config_type(detector_name),
        dec_hook=_path_dec_hook,
    )


def load_experiment_bundle(run_config_path: Path) -> ExperimentBundle:
    """Load a run config and its referenced dataset/model configs.

    Returns:
        ExperimentBundle: Fully resolved run, dataset, and model configuration.
    """
    resolved_run_path = run_config_path.resolve()
    experiments_root = _find_experiments_root(resolved_run_path)
    repo_root = experiments_root.parent
    run = _load_toml(resolved_run_path, expected_type=RunConfig)
    dataset_path = _resolve_named_config(
        experiments_root / "configs" / "datasets",
        run.dataset,
    )
    model_path = _resolve_named_config(
        experiments_root / "configs" / "models",
        run.model,
    )
    return ExperimentBundle(
        experiments_root=experiments_root,
        repo_root=repo_root,
        run_path=resolved_run_path,
        dataset_path=dataset_path,
        model_path=model_path,
        run=run,
        dataset=_load_dataset_config(dataset_path),
        model=_load_model_config(model_path),
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
