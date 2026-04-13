"""Tests for template parser implementations."""

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import pytest
from prefect.logging import disable_run_logger

from anomalog.cache import CachePathsConfig
from anomalog.parsers.template import (
    resolve_template_parser,
    template_parser_names,
)
from anomalog.parsers.template.parsers import Drain3Parser, IdentityTemplateParser

ZeroArgFn: TypeAlias = Callable[[], None]
MaterializeDecorator: TypeAlias = Callable[[ZeroArgFn], ZeroArgFn]


def _direct_materialize(
    *_args: str,
    **_kwargs: str,
) -> MaterializeDecorator:
    def _decorate(func: ZeroArgFn) -> ZeroArgFn:
        return func

    return _decorate


def _skip_materialize(
    *_args: str,
    **_kwargs: str,
) -> MaterializeDecorator:
    def _decorate(_func: ZeroArgFn) -> ZeroArgFn:
        def _skip() -> None:
            return None

        return _skip

    return _decorate


def test_drain3_parser_inference_requires_training(tmp_path: Path) -> None:
    """Drain3Parser refuses inference until training has produced a model."""
    parser = Drain3Parser(dataset_name="demo", cache_path=tmp_path / "cache")

    with pytest.raises(ValueError, match="not been trained"):
        parser.inference("User alice logged in")


# Protects the basic Drain3 training/inference contract.
# The nearby uncovered lines belong to cache-recovery behavior covered elsewhere.
@pytest.mark.allow_no_new_coverage
def test_drain3_parser_trains_and_extracts_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drain3Parser learns a template and returns extracted parameters."""
    monkeypatch.setattr(
        "anomalog.cache.materialize",
        _direct_materialize,
    )

    parser = Drain3Parser(dataset_name="demo", cache_path=tmp_path / "cache")

    with disable_run_logger():
        parser.train(
            lambda: iter(
                [
                    "User alice logged in",
                    "User bob logged in",
                ],
            ),
        )

    template, parameters = parser.inference("User charlie logged in")
    assert template == "User <:*:> logged in"
    assert list(parameters) == ["charlie"]


def test_drain3_parser_uses_bound_dataset_name_for_cache_paths(tmp_path: Path) -> None:
    """Bound Drain3Parser instances resolve both explicit and default cache paths."""
    with pytest.raises(ValueError, match="requires a dataset name"):
        _ = Drain3Parser().cache_file_path

    parser = Drain3Parser(dataset_name="demo", cache_path=tmp_path / "cache")
    default_cache_parser = Drain3Parser(dataset_name="demo")

    assert parser.dataset_name == "demo"
    assert parser.cache_file_path.parent == tmp_path / "cache"
    assert (
        default_cache_parser.resolved_cache_path
        == CachePathsConfig().cache_root / "demo" / "drain3"
    )


def test_drain3_parser_recovers_when_prefect_skips_and_local_cache_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drain3Parser still trains if Prefect skips execution without a cache file."""
    monkeypatch.setattr(
        "anomalog.cache.materialize",
        _skip_materialize,
    )

    parser = Drain3Parser(dataset_name="demo", cache_path=tmp_path / "cache")

    with disable_run_logger():
        parser.train(
            lambda: iter(
                [
                    "User alice logged in",
                    "User bob logged in",
                ],
            ),
        )

    template, parameters = parser.inference("User charlie logged in")
    assert template == "User <:*:> logged in"
    assert list(parameters) == ["charlie"]


def test_template_parser_registry_resolves_builtins() -> None:
    """Built-in template parsers register themselves by config name."""
    assert resolve_template_parser("drain3") is Drain3Parser
    assert resolve_template_parser("identity") is IdentityTemplateParser
    assert set(template_parser_names()) >= {"drain3", "identity"}
