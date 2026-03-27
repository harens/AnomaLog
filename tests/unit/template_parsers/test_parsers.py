"""Tests for template parser implementations."""

from collections.abc import Callable
from pathlib import Path

import pytest
from prefect.logging import disable_run_logger

from anomalog.template_parsers.parsers import Drain3Parser


def _direct_materialize(
    *_args: object,
    **_kwargs: object,
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    def _decorate(func: Callable[..., object]) -> Callable[..., object]:
        return func

    return _decorate


def test_drain3_parser_inference_requires_training(tmp_path: Path) -> None:
    """Drain3Parser refuses inference until training has produced a model."""
    parser = Drain3Parser(dataset_name="demo", cache_path=tmp_path / "cache")

    with pytest.raises(ValueError, match="not been trained"):
        parser.inference("User alice logged in")


def test_drain3_parser_trains_and_extracts_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drain3Parser learns a template and returns extracted parameters."""
    monkeypatch.setattr(
        "anomalog.template_parsers.parsers.materialize",
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
