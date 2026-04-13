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


@pytest.mark.allow_no_new_coverage
def test_drain3_parser_recovers_when_prefect_skips_and_local_cache_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drain3Parser still trains if Prefect skips execution without a cache file."""
    # This keeps the public "train then infer" contract covered for the skip
    # scenario even though the cache-reload branch is exercised explicitly below.
    # There is no distinct nearby uncovered branch left for this behavior.
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


def test_template_parser_registry_rejects_unknown_names() -> None:
    """Unknown template parser names raise a descriptive KeyError."""
    with pytest.raises(KeyError, match="Unsupported template parser: 'missing'"):
        resolve_template_parser("missing")


def test_drain3_parser_loads_inference_from_existing_cache_when_training_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train should recover inference from a persisted cache when Prefect skips work."""

    def _skip_materialize_at_parser(
        *_args: object,
        **_kwargs: object,
    ) -> MaterializeDecorator:
        return _skip_materialize()

    trained = Drain3Parser(dataset_name="demo", cache_path=tmp_path / "cache")
    with disable_run_logger():
        trained.train(
            lambda: iter(
                [
                    "User alice logged in",
                    "User bob logged in",
                ],
            ),
        )

    parser = Drain3Parser(dataset_name="demo", cache_path=tmp_path / "cache")
    monkeypatch.setattr(
        "anomalog.parsers.template.parsers.materialize",
        _skip_materialize_at_parser,
    )

    with disable_run_logger():
        parser.train(lambda: iter(["ignored because prefect cache hit"]))

    template, parameters = parser.inference("User charlie logged in")
    assert template == "User <:*:> logged in"
    assert list(parameters) == ["charlie"]


def test_drain3_parser_train_deletes_stale_cache_and_handles_empty_training_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training should remove stale cache files and allow empty iterator runs."""
    monkeypatch.setattr(
        "anomalog.parsers.template.parsers.materialize",
        _direct_materialize,
    )

    class _FakeMiner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def add_log_message(self, _log_line: str) -> dict[str, int]:
            return {"cluster_count": 0}

        def match(self, _log_line: str) -> None:
            return None

        def get_parameter_list(
            self,
            _template: str,
            _log_line: str,
        ) -> list[str]:
            return []

    monkeypatch.setattr(
        "anomalog.parsers.template.parsers.TemplateMiner",
        _FakeMiner,
    )

    parser = Drain3Parser(dataset_name="demo", cache_path=tmp_path / "cache")
    parser.cache_file_path.write_text("stale", encoding="utf-8")

    with disable_run_logger():
        parser.train(lambda: iter(()))

    assert not parser.cache_file_path.exists()
    assert parser.inference_func is not None


def test_identity_template_parser_is_a_no_op_for_train_and_inference() -> None:
    """IdentityTemplateParser should echo input text and ignore training."""
    parser = IdentityTemplateParser(dataset_name="demo")

    parser.train(lambda: iter(["hello"]))

    assert parser.inference("hello") == ("hello", [])
