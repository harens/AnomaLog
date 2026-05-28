"""Tests for template parser implementations."""

import math
import types
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
from anomalog.parsers.template.parsers import (
    Drain3Parser,
    IdentityTemplateParser,
    SpellTemplateParser,
)

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
    """Drain3Parser refuses inference until training has produced a model.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for parser cache files.
    """
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
    """Drain3Parser learns a template and returns extracted parameters.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for parser cache files.
        monkeypatch (pytest.MonkeyPatch): Patches Prefect materialization so the
            test can exercise training logic directly.
    """
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
    """Bound Drain3Parser instances resolve both explicit and default cache paths.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for explicit cache roots.
    """
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
    """Drain3Parser still trains if Prefect skips execution without a cache file.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for parser cache files.
        monkeypatch (pytest.MonkeyPatch): Patches Prefect materialization to
            simulate a cached Prefect state with no local cache file.
    """
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
    assert resolve_template_parser("spell") is SpellTemplateParser
    assert set(template_parser_names()) >= {"drain3", "identity", "spell"}


def test_template_parser_registry_rejects_unknown_names() -> None:
    """Unknown template parser names raise a descriptive KeyError."""
    with pytest.raises(KeyError, match="Unsupported template parser: 'missing'"):
        resolve_template_parser("missing")


def test_drain3_parser_loads_inference_from_existing_cache_when_training_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train should recover inference from a persisted cache when Prefect skips work.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for parser cache files.
        monkeypatch (pytest.MonkeyPatch): Patches Prefect materialization at the
            parser module boundary to simulate a cache hit.
    """

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
    """Training should remove stale cache files and allow empty iterator runs.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for parser cache files.
        monkeypatch (pytest.MonkeyPatch): Replaces Drain3's miner with a minimal
            fake so the test can isolate cache cleanup behavior.
    """
    monkeypatch.setattr(
        "anomalog.parsers.template.parsers.materialize",
        _direct_materialize,
    )

    class _FakeMiner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def add_log_message(self, _log_line: str) -> dict[str, int]:
            del self
            return {"cluster_count": 0}

        def match(self, _log_line: str) -> None:
            del self

        def get_parameter_list(
            self,
            _template: str,
            _log_line: str,
        ) -> list[str]:
            del self
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


def test_spell_template_parser_trains_and_infers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Spell parser should expose trained templates through inference.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patch the working directory so
            Spell artefacts stay inside the per-test sandbox.
        tmp_path (Path): Per-test filesystem sandbox for spell artefacts.
    """
    monkeypatch.chdir(tmp_path)

    parser = SpellTemplateParser(dataset_name="demo")
    parser.train(
        lambda: iter(
            [
                "Build vm-1 complete",
                "Build vm-2 complete",
                "Delete vm-1 complete",
            ],
        ),
    )

    template, parameters = parser.inference("Build vm-3 complete")
    assert template == "Build <*> complete"
    assert parameters == ["vm-3"]


def test_spell_template_parser_trains_without_materialising_legacy_spellpy_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Spell parser training should mine templates without legacy raw artefacts.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patches the working directory and
            guards against accidental `list(...)` materialisation.
        tmp_path (Path): Per-test filesystem sandbox for spell artefacts.
    """

    def _forbid_list_materialisation(*_args: object, **_kwargs: object) -> None:
        msg = "Spell training should not materialise the full input stream."
        raise AssertionError(msg)

    monkeypatch.chdir(tmp_path)
    stale_output_dir = tmp_path / ".cache" / "spell" / "demo_spell_output"
    stale_output_dir.mkdir(parents=True, exist_ok=True)
    (stale_output_dir / "stale.txt").write_text("stale", encoding="utf-8")
    parser = SpellTemplateParser(dataset_name="demo")
    with monkeypatch.context() as m:
        m.setattr(
            "anomalog.parsers.template.parsers.list",
            _forbid_list_materialisation,
            raising=False,
        )
        parser.train(
            lambda: iter(
                [
                    "Build vm-1 complete",
                    "Build vm-2 complete",
                ],
            ),
        )

    raw_path = tmp_path / ".cache" / "spell" / "demo_spell_input.log"
    output_dir = tmp_path / ".cache" / "spell" / "demo_spell_output"
    assert not raw_path.exists()
    assert output_dir.exists()
    assert not (output_dir / "stale.txt").exists()
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "demo.log_structured.csv",
        "demo.log_templates.csv",
    ]
    assert parser.inference("Build vm-3 complete") == (
        "Build <*> complete",
        ["vm-3"],
    )


def test_spell_template_parser_omits_removed_persist_state_argument(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Spell parser training should match the installed LogParser signature.

    Args:
        monkeypatch (pytest.MonkeyPatch): Redirect the `spellpy` import to a
            strict fake module that rejects removed keyword arguments.
        tmp_path (Path): Per-test filesystem sandbox for spell artefacts.
        caplog (pytest.LogCaptureFixture): Captures the training summary so the
            occurrence-count regression stays observable through the public log
            output.
    """
    monkeypatch.chdir(tmp_path)

    class _FakeLogParser:
        def __init__(self, **kwargs: object) -> None:
            expected_tau = 0.5
            expected_max_lcs_comparisons_per_line = 10_000
            assert "persist_state" not in kwargs
            assert set(kwargs) == {
                "indir",
                "outdir",
                "log_format",
                "tau",
                "keep_para",
                "max_lcs_comparisons_per_line",
                "resume_state",
            }
            indir = kwargs["indir"]
            assert isinstance(indir, str)
            assert Path(indir).parent == tmp_path / ".cache" / "spell"
            outdir = kwargs["outdir"]
            assert isinstance(outdir, str)
            assert outdir == str(
                tmp_path / ".cache" / "spell" / "demo_spell_output",
            )
            log_format = kwargs["log_format"]
            assert isinstance(log_format, str)
            assert log_format == "<Content>"
            tau = kwargs["tau"]
            assert isinstance(tau, float)
            assert math.isclose(tau, expected_tau)
            keep_para = kwargs["keep_para"]
            assert isinstance(keep_para, bool)
            assert keep_para is False
            max_lcs_comparisons_per_line = kwargs["max_lcs_comparisons_per_line"]
            assert isinstance(max_lcs_comparisons_per_line, int)
            assert max_lcs_comparisons_per_line == expected_max_lcs_comparisons_per_line
            resume_state = kwargs["resume_state"]
            assert isinstance(resume_state, bool)
            assert resume_state is False
            self.logCluL = [
                types.SimpleNamespace(
                    logTemplate=["Build", "<*>", "complete"],
                    logIDL=[],
                    occurrence_count=2,
                ),
            ]

        @staticmethod
        def parse(_filename: str) -> None:
            return None

    fake_spell_module = types.SimpleNamespace(
        spell=types.SimpleNamespace(LogParser=_FakeLogParser),
    )
    monkeypatch.setattr(
        "anomalog.parsers.template.parsers.importlib.import_module",
        lambda _name: fake_spell_module,
    )

    parser = SpellTemplateParser(dataset_name="demo")
    with caplog.at_level("INFO"):
        parser.train(lambda: iter(["Build vm-1 complete", "Build vm-2 complete"]))

    assert "Spell parser training finished: templates=1 occurrences=2" in caplog.text
    assert parser.inference("Build vm-3 complete") == (
        "Build <*> complete",
        ["vm-3"],
    )
