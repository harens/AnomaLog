"""Tests for config-loader and registry helper branches."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest

from experiments import ConfigError
from experiments.config import (
    CSVLabelReaderConfig,
    DatasetVariantConfig,
    EntitySequenceConfig,
    LocalDirSourceConfig,
    load_experiment_registry,
)
from experiments.config_loader import (
    _apply_config_overrides as loader_apply_config_overrides,
    _dataset_config_key as loader_dataset_config_key,
    _decode_dataset_config as loader_decode_dataset_config,
    _decode_dataset_experiment_config as loader_decode_dataset_experiment_config,
    _decode_dataset_source_config as loader_decode_dataset_source_config,
    _decode_label_reader_config as loader_decode_label_reader_config,
    _decode_model_config as loader_decode_model_config,
    _decode_sequence_config as loader_decode_sequence_config,
    _decode_toml_file as loader_decode_toml_file,
    _find_experiments_root as loader_find_experiments_root,
    _load_merged_dataset_experiment_config as loader_load_merged_dataset_experiment_config,
    _merge_toml_tables as loader_merge_toml_tables,
    _normalize_toml_table as loader_normalize_toml_table,
    _path_dec_hook as loader_path_dec_hook,
    _resolve_dataset_experiment_path as loader_resolve_dataset_experiment_path,
    _resolve_dataset_manifest_path as loader_resolve_dataset_manifest_path,
    _resolve_named_config as loader_resolve_named_config,
    _resolve_model_config as loader_resolve_model_config,
    _resolve_run_group as loader_resolve_run_group,
    _set_nested_value as loader_set_nested_value,
    _slugify_value as loader_slugify_value,
)
from experiments.registry import (
    ModelSetDefinition,
    _build_experiment_set_entries as registry_build_experiment_set_entries,
    _build_concrete_name as registry_build_concrete_name,
    _decode_experiment_definition as registry_decode_experiment_definition,
    _decode_experiment_set_definition as registry_decode_experiment_set_definition,
    _decode_model_ref_lists as registry_decode_model_ref_lists,
    _decode_model_set as registry_decode_model_set,
    _decode_overrides_map as registry_decode_overrides_map,
    _load_registered_experiments as registry_load_registered_experiments,
    _derive_groups as registry_derive_groups,
    _normalize_toml_table as registry_normalize_toml_table,
    _path_dec_hook as registry_path_dec_hook,
    _require_string_list as registry_require_string_list,
    _reject_unknown_keys as registry_reject_unknown_keys,
    _resolve_path as registry_resolve_path,
    _select_model_overrides as registry_select_model_overrides,
    _slugify_label as registry_slugify_label,
    _slugify_value as registry_slugify_value,
    _trim_known_suffixes as registry_trim_known_suffixes,
    _load_model_config_reference as registry_load_model_config_reference,
)


def _write_minimal_dataset_manifest(datasets_dir: Path, name: str) -> Path:
    path = datasets_dir / f"{name}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            f'name = "{name}"\n'
            "preset = \"demo\"\n"
            "\n[dataset]\n"
            f'name = "{name}"\n'
            f'dataset_name = "{name.upper()}"\n'
            'preset = "demo"\n'
            'structured_parser = "bgl"\n'
            "\n[dataset.source]\n"
            'type = "local_dir"\n'
            'path = "."\n'
            "\n[dataset.label_reader]\n"
            'type = "csv"\n'
            'relative_path = "labels.csv"\n'
            "\n[dataset.sequence]\n"
            'grouping = "entity"\n'
        ),
        encoding="utf-8",
    )
    return path


def _write_minimal_model_files(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "template_frequency_default.toml").write_text(
        (
            'name = "template_frequency_default"\n'
            'detector = "template_frequency"\n'
            'primary_metric_scope = "event_level_detection"\n'
            'secondary_metric_scopes = ["sequence_level_detection"]\n'
            "smoothing = 1.0\n"
        ),
        encoding="utf-8",
    )
    (models_dir / "deeplog_default.toml").write_text(
        (
            'name = "deeplog_default"\n'
            'detector = "deeplog"\n'
            "top_g_values = [1, 3, 5]\n"
            "parameter_detection_enabled = false\n"
        ),
        encoding="utf-8",
    )


def test_config_loader_helper_branches(tmp_path: Path) -> None:
    """Config-loader helpers should keep their decode and override contracts."""
    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / "experiments" / "configs" / "models" / "template_frequency_default.toml"
    model_raw = msgspec.toml.decode(model_path.read_bytes())

    assert loader_path_dec_hook(object, "demo") == "demo"
    assert loader_path_dec_hook(Path, "demo") == Path("demo")
    with pytest.raises(NotImplementedError, match="Unsupported decoded type"):
        loader_path_dec_hook(int, "demo")
    assert loader_normalize_toml_table({"a": 1, 2: 3}, expected_type="demo") == {
        "a": 1,
        "2": 3,
    }
    with pytest.raises(TypeError, match="must decode from a TOML table"):
        loader_normalize_toml_table([], expected_type="demo")

    missing = tmp_path / "missing.toml"
    with pytest.raises(ConfigError, match="Missing config file"):
        loader_decode_toml_file(missing, decode=lambda raw: raw)
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("this is not toml = [", encoding="utf-8")
    with pytest.raises(ConfigError):
        loader_decode_toml_file(bad_toml, decode=lambda raw: raw)

    assert isinstance(loader_decode_sequence_config(None), EntitySequenceConfig)
    assert isinstance(
        loader_decode_dataset_source_config({"type": "local_dir", "path": "."}),
        LocalDirSourceConfig,
    )
    with pytest.raises(TypeError, match="must define `type`"):
        loader_decode_dataset_source_config({})
    with pytest.raises(ValueError, match="Unsupported dataset source"):
        loader_decode_dataset_source_config({"type": "missing"})
    assert isinstance(
        loader_decode_label_reader_config(
            {"type": "csv", "relative_path": "labels.csv"},
        ),
        CSVLabelReaderConfig,
    )
    with pytest.raises(TypeError, match="must define `type`"):
        loader_decode_label_reader_config({})
    with pytest.raises(ValueError, match="Unsupported label reader"):
        loader_decode_label_reader_config({"type": "missing"})
    assert loader_decode_model_config(model_raw).detector == "template_frequency"
    with pytest.raises(TypeError, match="must define `detector`"):
        loader_decode_model_config({"name": "demo"})

    dataset_config = loader_decode_dataset_config(
        {
            "name": "demo",
            "dataset_name": "DEMO",
            "preset": "demo",
            "source": {"type": "local_dir", "path": "."},
            "structured_parser": "bgl",
            "template_parser": "identity",
            "label_reader": {"type": "csv", "relative_path": "labels.csv"},
            "sequence": {"grouping": "entity"},
        },
    )
    assert dataset_config.sequence.__class__ is EntitySequenceConfig

    with pytest.raises(TypeError, match="dataset experiment config must define"):
        loader_decode_dataset_experiment_config({"name": "demo"})
    with pytest.raises(TypeError, match="`models` must be a TOML array"):
        loader_decode_dataset_experiment_config(
            {
                "name": "demo",
                "dataset": {
                    "name": "demo",
                    "dataset_name": "demo",
                    "preset": "demo",
                    "source": {"type": "local_dir", "path": "."},
                    "structured_parser": "bgl",
                    "template_parser": "identity",
                },
                "models": "bad",
            },
        )
    experiment_config = loader_decode_dataset_experiment_config(
        {
            "name": "demo",
            "dataset": {
                "name": "demo",
                "dataset_name": "demo",
                "preset": "demo",
                "source": {"type": "local_dir", "path": "."},
                "structured_parser": "bgl",
                "template_parser": "identity",
            },
            "model": model_raw,
            "results_root": tmp_path / "results",
            "max_workers": 1,
        },
    )
    assert experiment_config.model is not None

    experiments_root = tmp_path / "experiments"
    datasets_dir = experiments_root / "configs" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    _write_minimal_dataset_manifest(datasets_dir, "demo")
    assert loader_resolve_dataset_manifest_path(
        datasets_dir / "demo.toml",
    ) == (datasets_dir / "demo.toml").resolve()
    assert loader_resolve_dataset_manifest_path(
        experiments_root / "configs" / "datasets" / "entity_chronological_base.toml",
    ).name == "entity_chronological_base.toml"

    parent = tmp_path / "parent.toml"
    child = tmp_path / "child.toml"
    parent.write_text("name = 'parent'\n", encoding="utf-8")
    child.write_text('extends = "parent.toml"\nname = "child"\n', encoding="utf-8")
    assert loader_load_merged_dataset_experiment_config(
        child,
        seen_paths=(),
    )["name"] == "child"
    blank_extends = tmp_path / "blank.toml"
    blank_extends.write_text('extends = ""\nname = "blank"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="extends"):
        loader_load_merged_dataset_experiment_config(blank_extends, seen_paths=())
    cycle_a = tmp_path / "cycle_a.toml"
    cycle_b = tmp_path / "cycle_b.toml"
    cycle_a.write_text('extends = "cycle_b.toml"\nname = "a"\n', encoding="utf-8")
    cycle_b.write_text('extends = "cycle_a.toml"\nname = "b"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="cyclic dataset experiment inheritance"):
        loader_load_merged_dataset_experiment_config(cycle_a, seen_paths=())

    assert loader_resolve_dataset_experiment_path(
        tmp_path / "demo.toml",
        "child.toml",
    ).name == "child.toml"
    assert loader_merge_toml_tables(
        {"outer": {"value": 1}, "keep": 1},
        {"outer": {"value": 2}, "replace": 3},
    ) == {"outer": {"value": 2}, "keep": 1, "replace": 3}
    with pytest.raises(ConfigError, match="Could not locate experiments root"):
        loader_find_experiments_root(tmp_path / "missing")
    with pytest.raises(ConfigError, match="Missing named config"):
        loader_resolve_named_config(tmp_path / "empty", "missing")
    assert loader_dataset_config_key(
        experiments_root / "configs" / "datasets" / "demo.toml",
        experiments_root=experiments_root,
    ) == "demo"
    with pytest.raises(ConfigError, match="must live under"):
        loader_dataset_config_key(tmp_path / "demo.toml", experiments_root=experiments_root)

    model_dir = experiments_root / "configs" / "models"
    _write_minimal_model_files(model_dir)
    resolved_model, model_path = loader_resolve_model_config(
        {"ref": "template_frequency_default"},
        repo_root=tmp_path,
        fallback_path=tmp_path / "fallback.toml",
    )
    assert model_path.name == "template_frequency_default.toml"
    assert resolved_model.detector == "template_frequency"
    inline_model, inline_path = loader_resolve_model_config(
        {
            "name": "inline_model",
            "detector": "deeplog",
            "top_g_values": [1, 3, 5],
            "parameter_detection_enabled": False,
        },
        repo_root=tmp_path,
        fallback_path=tmp_path / "fallback.toml",
    )
    assert inline_path.name == "fallback.toml"
    assert inline_model.detector == "deeplog"
    with pytest.raises(TypeError, match="model `ref` must be a string"):
        loader_resolve_model_config({"ref": 1}, repo_root=tmp_path, fallback_path=tmp_path)
    with pytest.raises(ConfigError, match="Missing model config for reference"):
        loader_resolve_model_config({"ref": "missing"}, repo_root=tmp_path, fallback_path=tmp_path)
    with pytest.raises(ConfigError, match="must be a string"):
        loader_resolve_run_group({"run_group": 1})
    with pytest.raises(ConfigError, match="must not be empty"):
        loader_resolve_run_group({"run_group": ""})
    assert loader_resolve_run_group({}) == "default"
    with pytest.raises(TypeError, match="results_root"):
        loader_decode_dataset_experiment_config(
            {
                "name": "demo",
                "dataset": {
                    "name": "demo",
                    "dataset_name": "demo",
                    "preset": "demo",
                    "source": {"type": "local_dir", "path": "."},
                    "structured_parser": "bgl",
                    "template_parser": "identity",
                },
                "results_root": 1,
            },
        )
    with pytest.raises(TypeError, match="max_workers"):
        loader_decode_dataset_experiment_config(
            {
                "name": "demo",
                "dataset": {
                    "name": "demo",
                    "dataset_name": "demo",
                    "preset": "demo",
                    "source": {"type": "local_dir", "path": "."},
                    "structured_parser": "bgl",
                    "template_parser": "identity",
                },
                "max_workers": True,
            },
        )
    with pytest.raises(TypeError, match="max_workers"):
        loader_decode_dataset_experiment_config(
            {
                "name": "demo",
                "dataset": {
                    "name": "demo",
                    "dataset_name": "demo",
                    "preset": "demo",
                    "source": {"type": "local_dir", "path": "."},
                    "structured_parser": "bgl",
                    "template_parser": "identity",
                },
                "max_workers": "many",
            },
        )
    auto_experiment_config = loader_decode_dataset_experiment_config(
        {
            "name": "auto",
            "dataset": {
                "name": "auto",
                "dataset_name": "auto",
                "preset": "demo",
                "source": {"type": "local_dir", "path": "."},
                "structured_parser": "bgl",
                "template_parser": "identity",
            },
            "model": model_raw,
        },
    )
    assert auto_experiment_config.max_workers == "auto"

    config = loader_decode_dataset_config(
        {
            "name": "override",
            "dataset_name": "override",
            "preset": "demo",
            "source": {"type": "local_dir", "path": "."},
            "structured_parser": "bgl",
            "template_parser": "identity",
            "sequence": {"grouping": "entity"},
        },
    )
    assert loader_apply_config_overrides(
        config=config,
        overrides={},
        prefix="dataset",
        decode=loader_decode_dataset_config,
    ) is config
    overridden = loader_apply_config_overrides(
        config=config,
        overrides={"dataset.name": "overridden"},
        prefix="dataset",
        decode=loader_decode_dataset_config,
    )
    assert overridden.name == "overridden"
    payload = {"nested": {"value": 1}}
    loader_set_nested_value(
        payload,
        ["nested", "value"],
        2,
        root_name="dataset",
    )
    assert payload["nested"]["value"] == 2
    with pytest.raises(ConfigError, match="Unknown override path"):
        loader_set_nested_value({}, ["missing"], 1, root_name="dataset")
    with pytest.raises(ConfigError, match="is not a table"):
        loader_set_nested_value({"nested": 1}, ["nested", "value"], 1, root_name="dataset")
    assert loader_resolve_dataset_experiment_path(
        tmp_path / "child",
        str(parent),
    ).name == "parent.toml"
    assert loader_resolve_dataset_experiment_path(
        tmp_path / "child.toml",
        str(parent),
    ).name == "parent.toml"

    assert loader_slugify_value(True) == "true"
    assert loader_slugify_value(1.5) == "1p5"


def test_registry_helper_branches(tmp_path: Path) -> None:
    """Registry helpers should preserve selection and override behaviour."""
    repo_root = tmp_path
    experiments_root = repo_root / "experiments"
    datasets_dir = experiments_root / "configs" / "datasets"
    models_dir = experiments_root / "configs" / "models"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    _write_minimal_dataset_manifest(datasets_dir, "demo")
    _write_minimal_model_files(models_dir)
    registry_path = experiments_root / "configs" / "registry.toml"
    registry_path.write_text(
        (
            "[model_sets.baselines]\n"
            'models = ["template_frequency_default"]\n'
            "\n"
            "[experiments.demo]\n"
            'dataset = "demo"\n'
            'model_sets = ["baselines"]\n'
            'models = ["deeplog_default"]\n'
        ),
        encoding="utf-8",
    )

    registry = load_experiment_registry(registry_path, repo_root=repo_root)
    assert registry.names() == ("demo",)
    assert registry.require("demo").dataset == "demo"
    assert registry.model_set("baselines").models == ("template_frequency_default",)
    assert registry.select(names=("demo",))[0].name == "demo"
    with pytest.raises(ConfigError, match="Unknown registry experiment"):
        registry.require("missing")
    with pytest.raises(ConfigError, match="Unknown model set"):
        registry.model_set("missing")
    with pytest.raises(ConfigError, match="Unknown registry experiment or group"):
        registry.select(names=("missing",))

    assert registry_path.exists()
    assert registry_normalize_toml_table({"a": 1, 2: 3}, expected_type="demo") == {
        "a": 1,
        "2": 3,
    }
    with pytest.raises(NotImplementedError, match="Unsupported decoded type"):
        registry_path_dec_hook(int, "demo")
    assert registry_path_dec_hook(object, "demo") == "demo"

    model_set = registry_decode_model_set(
        "baselines",
        {"models": ["template_frequency_default"], "description": "demo"},
    )
    assert isinstance(model_set, ModelSetDefinition)
    with pytest.raises(TypeError, match="non-empty `models` array"):
        registry_decode_model_set("broken", {})
    with pytest.raises(TypeError, match="invalid model reference"):
        registry_decode_model_set("broken", {"models": [1]})

    experiment = registry_decode_experiment_definition(
        "demo",
        {"dataset": "demo", "models": ["deeplog_default"], "model_sets": ["baselines"]},
        experiment_kind="experiment",
    )
    assert experiment.groups == ("demo", "baselines")
    with pytest.raises(TypeError, match="must define a non-empty `dataset`"):
        registry_decode_experiment_definition(
            "demo",
            {"models": ["deeplog_default"]},
            experiment_kind="experiment",
        )

    experiment_set = registry_decode_experiment_set_definition(
        "paper_group",
        {
            "datasets": ["demo"],
            "model_sets": ["baselines"],
            "models": ["deeplog_default"],
        },
    )
    assert [item.name for item in experiment_set] == ["demo"]
    with pytest.raises(TypeError, match="must define `models` or `model_sets`"):
        registry_decode_experiment_set_definition("paper_group", {"datasets": ["demo"]})

    assert registry_decode_model_ref_lists(
        {"models": ["a"], "model_sets": ["b"]},
        context="demo",
    ) == (("a",), ("b",))
    with pytest.raises(TypeError, match="must define `models` or `model_sets`"):
        registry_decode_model_ref_lists({}, context="demo")
    assert registry_decode_overrides_map(
        {"baselines": {"model.name": "demo"}},
        context="demo",
    ) == {"baselines": {"model.name": "demo"}}
    with pytest.raises(TypeError, match="must be a TOML table"):
        registry_decode_overrides_map({"baselines": 1}, context="demo")
    with pytest.raises(ConfigError, match="contains unsupported fields"):
        registry_reject_unknown_keys(
            {"a": 1, "b": 2},
            allowed={"a"},
            context="demo",
        )
    with pytest.raises(TypeError, match="non-empty `models` array"):
        registry_require_string_list([], context="demo", field_name="models")
    with pytest.raises(TypeError, match="invalid model"):
        registry_require_string_list(["", "a"], context="demo", field_name="models")
    assert registry_build_experiment_set_entries(
        SimpleNamespace(
            datasets=("demo", "other"),
            model_sets=("baselines",),
            models=("deeplog_default",),
            groups=("group",),
            overrides={"baselines": {"model.name": "demo"}},
            description="demo",
        ),
    )[0].name == "demo"
    assert registry_build_experiment_set_entries(
        SimpleNamespace(
            datasets=("demo",),
            model_sets=(),
            models=("deeplog_default",),
            groups=(),
            overrides={},
            description=None,
        ),
    )[0].description is None

    assert registry_derive_groups(entry_name="demo", model_sets=("baselines", "demo")) == (
        "demo",
        "baselines",
    )
    assert registry_derive_groups(entry_name=None, model_sets=("baselines", "baselines")) == (
        "baselines",
    )
    assert registry_slugify_label("demo model") == "demo_model"
    assert registry_slugify_value(True) == "true"
    assert registry_slugify_value(1.5) == "1p5"
    assert registry_slugify_value("Template Frequency") == "template_frequency"
    assert registry_trim_known_suffixes("demo_default") == "demo"
    assert registry_trim_known_suffixes("demo_entity") == "demo"
    assert registry_trim_known_suffixes("demo_entity_supervised") == "demo"
    assert registry_select_model_overrides({"model.name": "demo"}, model_ref="x") == {
        "model.name": "demo"
    }
    assert registry_select_model_overrides(
        {"x": {"model.name": "demo"}, "y": {"model.name": "other"}},
        model_ref="missing",
    ) == {}
    assert registry_select_model_overrides(
        {"x": {"model.name": "demo"}},
        model_ref="x",
    ) == {"model.name": "demo"}
    with pytest.raises(ConfigError, match="must be a TOML table"):
        registry_select_model_overrides({"x": {"model.name": "demo"}, "y": 1}, model_ref="y")
    assert registry_resolve_path(Path("experiments/configs/registry.toml"), repo_root) == (
        repo_root / "experiments" / "configs" / "registry.toml"
    ).resolve()
    assert registry_resolve_path(
        (repo_root / "experiments" / "configs" / "registry.toml").resolve(),
        repo_root,
    ) == (repo_root / "experiments" / "configs" / "registry.toml").resolve()

    model = loader_decode_model_config(
        msgspec.toml.decode(
            (models_dir / "deeplog_default.toml").read_bytes(),
        ),
    )
    assert model.detector == "deeplog"
    assert registry_build_concrete_name(
        default_name="demo",
        dataset=loader_decode_dataset_config(
            {
                "name": "demo",
                "dataset_name": "demo",
                "preset": "demo",
                "source": {"type": "local_dir", "path": "."},
                "structured_parser": "bgl",
                "template_parser": "identity",
                "sequence": {"grouping": "entity"},
            },
        ),
        model=model,
        applied_overrides={},
    ) == "demo"
    assert registry_build_concrete_name(
        default_name=None,
        dataset=loader_decode_dataset_config(
            {
                "name": "demo_entity_supervised",
                "dataset_name": "demo",
                "preset": "demo",
                "source": {"type": "local_dir", "path": "."},
                "structured_parser": "bgl",
                "template_parser": "identity",
                "sequence": {"grouping": "entity"},
            },
        ),
        model=model,
        applied_overrides={"model.dropout": 0.25, "dataset": "ignored"},
    ).startswith("demo_deeplog_dropout_0p25")
    with pytest.raises(ConfigError, match="must define at least one experiment"):
        registry_load_registered_experiments({}, {})
    with pytest.raises(ConfigError, match="Duplicate experiment name"):
        registry_load_registered_experiments(
            {"demo": {"dataset": "demo", "models": ["deeplog_default"]}},
            {"demo_group": {"datasets": ["demo"], "models": ["deeplog_default"]}},
        )
    with pytest.raises(ConfigError, match="Missing model config for reference"):
        registry_load_model_config_reference("missing", repo_root=repo_root)
    empty_registry = experiments_root / "configs" / "empty_registry.toml"
    empty_registry.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="Registry must define at least one experiment"):
        load_experiment_registry(empty_registry, repo_root=repo_root)
