"""Dataset/spec resolution for config-driven experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anomalog import DatasetSpec
from anomalog.parsers.structured import resolve_structured_parser
from anomalog.parsers.template import resolve_template_parser
from anomalog.presets import resolve_preset
from experiments import ConfigError

if TYPE_CHECKING:
    from pathlib import Path

    from experiments.config import DatasetVariantConfig


def build_dataset_spec(config: DatasetVariantConfig, *, repo_root: Path) -> DatasetSpec:
    """Construct a `DatasetSpec` from a dataset variant config."""
    try:
        template_parser = resolve_template_parser(config.template_parser)
    except KeyError as exc:
        raise ConfigError(exc.args[0]) from exc

    spec = _base_dataset_spec(config, repo_root=repo_root).template_with(
        template_parser,
    )
    if config.cache_paths is not None:
        spec = spec.with_cache_paths(config.cache_paths.resolve(repo_root=repo_root))
    if config.label_reader is not None:
        spec = spec.label_with(config.label_reader.build())
    return spec


def dataset_source_summary(
    config: DatasetVariantConfig,
    *,
    repo_root: Path,
) -> dict[str, str | None]:
    """Return a stable source summary for the dataset manifest."""
    return config.source_summary(repo_root=repo_root)


def _base_dataset_spec(config: DatasetVariantConfig, *, repo_root: Path) -> DatasetSpec:
    if config.preset is not None:
        try:
            return resolve_preset(config.preset)
        except KeyError as exc:
            raise ConfigError(exc.args[0]) from exc
    source, structured_parser_name = config.custom_dataset_components()
    try:
        structured_parser = resolve_structured_parser(structured_parser_name)
    except KeyError as exc:
        raise ConfigError(exc.args[0]) from exc
    return (
        DatasetSpec(config.dataset_name)
        .from_source(source.build(repo_root=repo_root))
        .parse_with(structured_parser())
    )
