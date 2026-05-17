"""Tests for template parser implementations."""

import io
import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TypeAlias

import pytest
from prefect.logging import disable_run_logger

import anomalog.parsers.template.parsers as template_parsers
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


def test_spell_template_parser_streams_input_without_materialising_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Spell parser training should consume the input stream incrementally.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patches the working directory and
            guards against accidental `list(...)` materialisation.
        tmp_path (Path): Per-test filesystem sandbox for spell artefacts.
    """

    def _forbid_list_materialisation(*_args: object, **_kwargs: object) -> None:
        msg = "Spell training should not materialise the full input stream."
        raise AssertionError(msg)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "anomalog.parsers.template.parsers.list",
        _forbid_list_materialisation,
        raising=False,
    )

    parser = SpellTemplateParser(dataset_name="demo")
    parser.train(
        lambda: iter(
            [
                "Build vm-1 complete",
                "Build vm-2 complete",
            ],
        ),
    )

    raw_path = tmp_path / ".cache" / "spell" / "demo_spell_input.log"
    assert raw_path.read_text(encoding="utf-8") == (
        "Build vm-1 complete\nBuild vm-2 complete\n"
    )
    templates_path = (
        tmp_path
        / ".cache"
        / "spell"
        / "demo_spell_output"
        / "demo_spell_input.log_templates.csv"
    )
    assert templates_path.exists()


def test_spell_template_parser_uses_the_lightweight_spellpy_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Spell parser should only request the template-mining artefacts it uses.

    Args:
        monkeypatch (pytest.MonkeyPatch): Redirects imports and the working
            directory so the parser exercises a fake spellpy backend locally.
        tmp_path (Path): Per-test filesystem sandbox for spell cache artefacts.
    """
    captured: dict[str, object] = {}

    class _FakeLogParser:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            outdir = captured["outdir"]
            assert isinstance(outdir, str)
            self._outdir = Path(outdir)
            self.parse_metrics = {"input_lines_processed": 2}

        def parse(self, logname: str) -> None:
            templates_path = self._outdir / f"{logname}_templates.csv"
            templates_path.write_text(
                "EventTemplate,Occurrences\nBuild <*> complete,2\n",
                encoding="utf-8",
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "anomalog.parsers.template.parsers.importlib.import_module",
        lambda _name: SimpleNamespace(
            spell=SimpleNamespace(LogParser=_FakeLogParser),
        ),
    )

    parser = SpellTemplateParser(dataset_name="demo", tau=0.75)
    parser.train(lambda: iter(["Build vm-1 complete", "Build vm-2 complete"]))

    assert captured == {
        "indir": str(tmp_path / ".cache" / "spell"),
        "outdir": str(tmp_path / ".cache" / "spell" / "demo_spell_output"),
        "log_format": "<Content>",
        "tau": 0.75,
        "progress_interval": 1000,
        "max_lcs_comparisons_per_line": 10000,
        "keep_para": False,
    }
    assert parser.inference("Build vm-3 complete") == (
        "Build <*> complete",
        ["vm-3"],
    )


def test_spell_template_parser_forwards_spellpy_module_logs() -> None:
    """Spellpy module logs should flow through the active experiment logger."""
    stream = io.StringIO()
    run_logger = logging.getLogger("tests.spellpy_forwarding")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(name)s:%(message)s"))
    run_logger.handlers.clear()
    run_logger.addHandler(handler)
    run_logger.setLevel(logging.INFO)
    run_logger.propagate = False

    spell_logger = logging.getLogger("spellpy.spell")
    previous_level = spell_logger.level
    previous_propagate = spell_logger.propagate

    with template_parsers.spellpy_logger_context(run_logger):
        spell_logger.info("Parsing file: demo_spell_input.log")

    assert "spellpy.spell:Parsing file: demo_spell_input.log" in stream.getvalue()
    assert spell_logger.level == previous_level
    assert spell_logger.propagate == previous_propagate


def test_spellpy_logger_context_uses_inherited_experiment_handlers() -> None:
    """Spellpy logs should still flow when the run logger inherits handlers."""
    stream = io.StringIO()
    parent_logger = logging.getLogger("tests.spellpy_forwarding.parent")
    parent_handler = logging.StreamHandler(stream)
    parent_handler.setFormatter(logging.Formatter("%(name)s:%(message)s"))
    parent_logger.handlers.clear()
    parent_logger.addHandler(parent_handler)
    parent_logger.setLevel(logging.INFO)
    parent_logger.propagate = False

    run_logger = logging.getLogger("tests.spellpy_forwarding.parent.child")
    run_logger.handlers.clear()
    run_logger.setLevel(logging.INFO)
    run_logger.propagate = True

    with template_parsers.spellpy_logger_context(run_logger):
        logging.getLogger("spellpy.spell").info("checkpoint")

    assert "spellpy.spell:checkpoint" in stream.getvalue()
