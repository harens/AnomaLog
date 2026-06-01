# ruff: noqa: PLC2701, PLR2004
"""Tests for DeepLog dataset-audit warm-up accounting helpers."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import types

import msgspec
import pytest

from anomalog.sequences import (
    SplitApplicationOrder,
    SplitLabel,
    StraddlingGroupPolicy,
    TemplateSequence,
)
from experiments.audit import deeplog_data_audit as audit_module
from experiments.audit.deeplog_data_audit import (
    BGLChunkSensitivitySummary,
    DeepLogDatasetAudit,
    EvaluationWarmupSummary,
    HDFSSessionObservation,
    HDFSFirst100kPolicySummary,
    NoEligibleSummary,
    SequenceLengthSummary,
    SplitAuditSummary,
    TrainingTargetSummary,
    WarmupAccounting,
    _build_split_strategy,
    _collect_hdfs_session_observations,
    _dataset_config_with_chunk_size,
    _evaluation_warmup_from_sequences,
    _hdfs_segments_for_policy,
    _model_config_top_g_values,
    _model_config_value,
    _no_eligible_summary_to_dict,
    _percentile,
    _require_close,
    _require_equal,
    _sequence_length_summary,
    _sequence_length_summary_to_dict,
    _split_audit_summary_to_dict,
    _summarise_hdfs_first_100k_policies,
    _structured_line_order,
    _training_target_summary_to_dict,
    _warmup_accounting_to_dict,
    aggregate_warmup_accounting,
    audit_bgl_chunk_size_sensitivity,
    audit_bgl_continuous_stream_warmup,
    audit_dataset_for_deeplog,
    audit_hdfs_first_100k_policies,
    warmup_counts_for_sequence_length,
)
from experiments.config import (
    ChronologicalStreamSequenceConfig,
    DatasetVariantConfig,
    EntitySequenceConfig,
    RawEntryPrefixCountSplitConfig,
    RawEntryPrefixFractionSplitConfig,
    RawEntryPrefixNormalFractionSplitConfig,
    TimeSequenceConfig,
)
from experiments.runners import audit_deeplog_data as audit_cli
from experiments.runners.audit_deeplog_data import (
    _decode_dataset_variant_config,
    _parse_dataset_item,
    _render_markdown,
    _require_object_dict,
    _require_object_list,
    _resolve_dataset_config_path,
)
from tests.unit.helpers import InMemoryStructuredSink, label_lookup, structured_line


@pytest.mark.parametrize(
    ("sequence_length", "history_size", "expected_insufficient", "expected_eligible"),
    [
        (0, 10, 0, 0),
        (1, 10, 1, 0),
        (10, 10, 10, 0),
        (11, 10, 10, 1),
        (13, 10, 10, 3),
    ],
)
def test_warmup_counts_for_sequence_length_matches_deeplog_contract(
    sequence_length: int,
    history_size: int,
    expected_insufficient: int,
    expected_eligible: int,
) -> None:
    """Per-sequence warm-up accounting should follow min/max DeepLog rules.

    Args:
        sequence_length (int): Sequence length under test.
        history_size (int): DeepLog history size under test.
        expected_insufficient (int): Expected warm-up exclusion count.
        expected_eligible (int): Expected eligible event count.
    """
    insufficient, eligible = warmup_counts_for_sequence_length(
        sequence_length=sequence_length,
        history_size=history_size,
    )

    assert insufficient == expected_insufficient
    assert eligible == expected_eligible


def test_aggregate_warmup_accounting_mixed_lengths() -> None:
    """Mixed sequence lengths should aggregate warm-up counts consistently."""
    expected_insufficient = 31
    expected_eligible = 4
    expected_seen = 35
    summary = aggregate_warmup_accounting(
        sequence_lengths=[0, 1, 10, 11, 13],
        history_size=10,
    )

    assert summary.insufficient_history == expected_insufficient
    assert summary.events_eligible == expected_eligible
    assert summary.events_seen == expected_seen
    assert summary.insufficient_history_rate == pytest.approx(
        expected_insufficient / expected_seen,
    )


def test_aggregate_warmup_accounting_includes_additional_exclusions() -> None:
    """Extra exclusion counts should be reflected in events_seen totals."""
    expected_insufficient = 10
    expected_eligible = 1
    expected_seen = 16
    additional_exclusions = 5
    summary = aggregate_warmup_accounting(
        sequence_lengths=[11],
        history_size=10,
        additional_excluded_events=additional_exclusions,
    )

    assert summary.insufficient_history == expected_insufficient
    assert summary.events_eligible == expected_eligible
    assert summary.events_seen == expected_seen
    assert summary.events_seen == (
        summary.insufficient_history + summary.events_eligible + additional_exclusions
    )


def test_chunk_boundary_warmup_accounting_adds_extra_warmup_loss() -> None:
    """Chunked streams should report the additional warm-up cost explicitly."""
    contiguous = aggregate_warmup_accounting(
        sequence_lengths=[6],
        history_size=2,
    )
    chunked = aggregate_warmup_accounting(
        sequence_lengths=[3, 3],
        history_size=2,
    )

    assert contiguous.insufficient_history == 2
    assert chunked.insufficient_history == 4
    assert chunked.insufficient_history - contiguous.insufficient_history == 2


def test_hdfs_first_100k_policy_summary_distinguishes_straddlers() -> None:
    """The HDFS policy audit should separate partial, first, and last policies."""
    sessions = [
        HDFSSessionObservation(
            entity_id="a",
            first_line_order=0,
            last_line_order=1,
            label=0,
            event_count=2,
            pre_cutoff_event_count=2,
            post_cutoff_event_count=0,
        ),
        HDFSSessionObservation(
            entity_id="b",
            first_line_order=2,
            last_line_order=6,
            label=1,
            event_count=5,
            pre_cutoff_event_count=3,
            post_cutoff_event_count=2,
        ),
        HDFSSessionObservation(
            entity_id="c",
            first_line_order=7,
            last_line_order=8,
            label=0,
            event_count=2,
            pre_cutoff_event_count=0,
            post_cutoff_event_count=2,
        ),
    ]

    summaries = _summarise_hdfs_first_100k_policies(
        sessions=sessions,
        cutoff=5,
        history_size=2,
        template_count=29,
    )
    by_name = {summary.policy_name: summary for summary in summaries}

    assert by_name["split_partial_sequences"].train_normal_sessions == 1
    assert by_name["split_partial_sequences"].ignored_sessions == 1
    assert by_name["split_partial_sequences"].test_anomalous_sessions == 1
    assert by_name["assign_by_first_event"].ignored_sessions == 1
    assert by_name["assign_by_last_event"].test_anomalous_sessions == 1
    assert by_name["normal_complete_sessions"].ignored_sessions == 0
    assert by_name["normal_complete_sessions"].test_anomalous_sessions == 1


def test_decode_dataset_variant_config_accepts_nested_dataset_tables() -> None:
    """The audit runner should accept the experiment-matrix DeepLog configs."""
    repo_root = Path(__file__).resolve().parents[2]
    matrix_path = (
        repo_root
        / "experiments"
        / "configs"
        / "datasets"
        / "openstack"
        / "deeplog_preprocessed.toml"
    )
    raw_matrix = msgspec.toml.decode(matrix_path.read_bytes())

    matrix_config = _decode_dataset_variant_config(raw_matrix)
    bare_config = _decode_dataset_variant_config(
        {
            "name": "bare_example",
            "dataset_name": "BARE_EXAMPLE",
            "preset": "hdfs_v1",
            "template_parser": "identity",
        },
    )

    assert matrix_config.dataset_name == "OPENSTACK_DEEPLOG_PREPROCESSED"
    assert matrix_config.template_parser == "identity"
    assert bare_config.dataset_name == "BARE_EXAMPLE"
    assert bare_config.template_parser == "identity"


def test_parse_dataset_item_validates_structure_and_values() -> None:
    """Dataset CLI items should parse and validate their history sizes."""
    assert _parse_dataset_item("hdfs_v1_entity_normal_only:10") == (
        "hdfs_v1_entity_normal_only",
        10,
    )
    with pytest.raises(ValueError, match="must use '<dataset-config-ref>"):
        _parse_dataset_item("missing")
    with pytest.raises(ValueError, match="reference is empty"):
        _parse_dataset_item(":10")
    with pytest.raises(ValueError, match="must be an integer"):
        _parse_dataset_item("demo:not-an-int")
    with pytest.raises(ValueError, match="must be non-negative"):
        _parse_dataset_item("demo:-1")


def test_resolve_dataset_config_path_supports_named_and_explicit_paths(
    tmp_path: Path,
) -> None:
    """Dataset config resolution should accept registry names and TOML files."""
    repo_root = Path(__file__).resolve().parents[2]
    experiments_root = repo_root / "experiments"
    named_path = _resolve_dataset_config_path(
        experiments_root=experiments_root,
        dataset_ref="thunderbird",
    )
    explicit_path = tmp_path / "explicit.toml"
    explicit_path.write_text("name = 'demo'\n", encoding="utf-8")

    assert named_path.exists()
    assert (
        _resolve_dataset_config_path(
            experiments_root=experiments_root,
            dataset_ref=explicit_path.as_posix(),
        )
        == explicit_path.resolve()
    )
    with pytest.raises(FileNotFoundError, match="Dataset config file not found"):
        _resolve_dataset_config_path(
            experiments_root=experiments_root,
            dataset_ref="missing.toml",
        )


def test_render_markdown_and_object_rails_cover_expected_branching() -> None:
    """Markdown rendering should serialise the nested audit payload safely."""
    payload = {
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "datasets": [
            {
                "dataset_variant": "demo",
                "dataset_name": "DEMO",
                "grouping_key": "fixed",
                "split_strategy": {"grouping": "fixed"},
                "raw_log_entry_count": 4,
                "parsed_event_count": 4,
                "parsed_template_count": 2,
                "sequence_count": 3,
                "train_sequence_count": 2,
                "test_sequence_count": 1,
                "ignored_sequence_count": 0,
                "sequence_length_summary": {
                    "min": 1,
                    "p25": 1.0,
                    "median": 2.0,
                    "p75": 3.0,
                    "max": 4,
                    "mean": 2.5,
                    "count_lte_history_size": 2,
                    "count_gt_history_size": 1,
                },
                "warmup_overall": {
                    "events_seen": 4,
                    "insufficient_history": 1,
                    "events_eligible": 3,
                    "insufficient_history_rate": 0.25,
                },
                "no_eligible_predictions": {"sequence_count": 0, "label_counts": {}},
                "training_target_summary": {
                    "eligible_normal_event_count": 2,
                    "excluded_anomalous_event_count": 1,
                    "excluded_context_event_count": 1,
                    "will_train": True,
                },
                "split_summaries": {
                    "train": {
                        "sequence_count": 2,
                        "event_count": 2,
                        "normal_sequence_count": 2,
                        "anomalous_sequence_count": 0,
                    },
                    "ignored": {
                        "sequence_count": 0,
                        "event_count": 0,
                        "normal_sequence_count": 0,
                        "anomalous_sequence_count": 0,
                    },
                    "test": {
                        "sequence_count": 1,
                        "event_count": 2,
                        "normal_sequence_count": 1,
                        "anomalous_sequence_count": 0,
                    },
                },
                "raw_entry_split_summary": {
                    "split_mode": "raw_entry_prefix_count",
                    "application_order": "before_grouping",
                    "cutoff_entry_index": 2,
                    "train_raw_entry_count": 2,
                    "train_normal_entry_count": 2,
                    "train_anomalous_entry_count": 0,
                    "test_raw_entry_count": 2,
                    "test_normal_entry_count": 2,
                    "test_anomalous_entry_count": 0,
                    "ignored_raw_entry_count": 0,
                    "ignored_normal_entry_count": 0,
                    "ignored_anomalous_entry_count": 0,
                    "straddling_group_count": 0,
                    "straddling_group_policy": "split_partial_sequences",
                },
            },
        ],
    }

    markdown = _render_markdown(payload=payload)
    assert "# DeepLog Dataset Audit" in markdown
    assert "### Raw Entry Split" in markdown
    assert "| train | 2 | 2 | 2 | 0 |" in markdown
    assert _require_object_dict({"a": 1}) == {"a": 1}
    assert _require_object_list([1, 2]) == [1, 2]
    with pytest.raises(TypeError, match="Expected dict payload"):
        _require_object_dict([1])
    with pytest.raises(TypeError, match="Expected list payload"):
        _require_object_list({"a": 1})


def test_audit_cli_main_writes_reports_and_uses_resolved_dataset_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI entry point should resolve one dataset and write both outputs."""

    class _Parser:
        @staticmethod
        def parse_args() -> types.SimpleNamespace:
            return types.SimpleNamespace(
                dataset=["demo:3"],
                output_json=tmp_path / "audit.json",
                output_markdown=tmp_path / "audit.md",
            )

    @contextmanager
    def _noop_harness() -> object:
        yield

    monkeypatch.setattr(audit_cli, "build_arg_parser", lambda: _Parser())
    monkeypatch.setattr(audit_cli, "prefect_test_harness", _noop_harness)
    monkeypatch.setattr(
        audit_cli,
        "_resolve_dataset_config_path",
        lambda *, experiments_root, dataset_ref: (
            experiments_root / f"{dataset_ref.replace(':', '_')}.toml"
        ),
    )
    monkeypatch.setattr(
        audit_cli,
        "_decode_toml_file",
        lambda path, decode: decode(
            {"name": "demo", "dataset_name": "demo", "preset": "demo"},
        ),
    )

    class _Report:
        def to_dict(self) -> dict[str, object]:
            return {
                "dataset_variant": "demo",
                "dataset_name": "demo",
                "grouping_key": "fixed",
                "split_strategy": {},
                "raw_log_entry_count": 1,
                "parsed_event_count": 1,
                "parsed_template_count": 1,
                "sequence_count": 1,
                "train_sequence_count": 1,
                "test_sequence_count": 0,
                "ignored_sequence_count": 0,
                "sequence_length_summary": {
                    "min": 1,
                    "p25": 1.0,
                    "median": 1.0,
                    "p75": 1.0,
                    "max": 1,
                    "mean": 1.0,
                    "count_lte_history_size": 1,
                    "count_gt_history_size": 0,
                },
                "warmup_overall": {
                    "events_seen": 1,
                    "insufficient_history": 0,
                    "events_eligible": 1,
                    "insufficient_history_rate": 0.0,
                },
                "no_eligible_predictions": {"sequence_count": 0, "label_counts": {}},
                "training_target_summary": {
                    "eligible_normal_event_count": 1,
                    "excluded_anomalous_event_count": 0,
                    "excluded_context_event_count": 0,
                    "will_train": True,
                },
                "split_summaries": {},
                "raw_entry_split_summary": None,
            }

    monkeypatch.setattr(audit_cli, "audit_dataset_for_deeplog", lambda **kwargs: _Report())

    assert audit_cli.main() == 0
    assert json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))["datasets"][
        0
    ]["dataset_variant"] == "demo"
    assert "# DeepLog Dataset Audit" in (tmp_path / "audit.md").read_text(
        encoding="utf-8",
    )


def test_audit_dataset_for_deeplog_builds_report_and_serialises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dataset audit should build a stable report from synthetic sequences."""
    raw_path = tmp_path / "raw.log"
    raw_path.write_text("a\nb\nc\n", encoding="utf-8")
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=raw_path,
        parser=types.SimpleNamespace(name="null"),
        rows=[
            structured_line(
                line_order=0,
                timestamp_unix_ms=0,
                entity_id="a",
                untemplated_message_text="one",
                anomalous=0,
            ),
            structured_line(
                line_order=1,
                timestamp_unix_ms=1,
                entity_id="b",
                untemplated_message_text="two",
                anomalous=1,
            ),
            structured_line(
                line_order=2,
                timestamp_unix_ms=2,
                entity_id=None,
                untemplated_message_text="three",
                anomalous=None,
            ),
        ],
    )
    templated = types.SimpleNamespace(sink=sink, anomaly_labels=label_lookup())
    sequences = [
        TemplateSequence(
            events=[("A", [], None), ("B", [], None)],
            label=0,
            entity_ids=["a"],
            window_id=1,
            split_label=SplitLabel.TRAIN,
            event_labels=(0, 1),
            training_event_mask=(True, False),
        ),
        TemplateSequence(
            events=[("C", [], None), ("D", [], None)],
            label=1,
            entity_ids=["b"],
            window_id=2,
            split_label=SplitLabel.TEST,
            event_labels=(1, 0),
        ),
        TemplateSequence(
            events=[("E", [], None)],
            label=1,
            entity_ids=["c"],
            window_id=3,
            split_label=SplitLabel.IGNORED,
            event_labels=(1,),
            training_event_mask=(False,),
        ),
    ]

    class _FakeSequenceBuilder:
        def __init__(self, items: list[TemplateSequence]) -> None:
            self._items = items

        def __iter__(self):
            yield from self._items

        def build_raw_entry_split_summary(self) -> None:
            return None

    monkeypatch.setattr(
        EntitySequenceConfig,
        "apply",
        lambda self, templated_dataset: _FakeSequenceBuilder(sequences),
    )
    monkeypatch.setattr(
        audit_module,
        "build_dataset_spec",
        lambda config, repo_root: types.SimpleNamespace(build=lambda: templated),
    )

    report = audit_dataset_for_deeplog(
        config=DatasetVariantConfig(
            name="demo",
            dataset_name="DEMO",
            preset="demo",
            sequence=EntitySequenceConfig(),
        ),
        repo_root=tmp_path,
        history_size=1,
        validate_paper_config=False,
    )

    assert isinstance(report, DeepLogDatasetAudit)
    assert report.raw_log_entry_count == 3
    assert report.parsed_event_count == 3
    assert report.sequence_count == 3
    assert report.train_sequence_count == 1
    assert report.test_sequence_count == 1
    assert report.ignored_sequence_count == 1
    assert report.training_target_summary.eligible_normal_event_count == 1
    assert report.training_target_summary.excluded_anomalous_event_count == 1
    assert report.training_target_summary.excluded_context_event_count == 0
    assert report.no_eligible_predictions.sequence_count == 1
    assert report.event_label_distribution == {"0": 1, "1": 1, "none": 1}
    assert report.sequence_label_distribution == {"0": 1, "1": 2}

    payload = report.to_dict()
    assert payload["raw_entry_split_summary"] is None
    assert payload["sequence_length_summary"]["count_gt_history_size"] == 2
    assert payload["warmup_overall"]["events_seen"] == 5
    assert payload["training_target_summary"]["will_train"] is True
    assert payload["split_summaries"]["train"]["warmup"]["events_eligible"] == 1


def test_audit_helper_primitives_and_summary_serialisation() -> None:
    """Low-level audit helpers should keep their conversion rules stable."""
    _require_equal(1, 1, "equal")
    with pytest.raises(ValueError, match="not equal"):
        _require_equal(1, 2, "not equal")

    _require_close(1.0, 1.0, "close")
    with pytest.raises(ValueError, match="not close"):
        _require_close(1.0, 1.1, "not close")

    model_config = types.SimpleNamespace(
        history_size=10,
        top_g_values=(1, 2.7, 5),
        num_layers=2,
        hidden_size=64,
    )
    assert _model_config_value(model_config, "history_size") == 10
    assert _model_config_top_g_values(model_config) == (1, 2, 5)
    assert _structured_line_order(types.SimpleNamespace(line_order=7)) == 7
    assert _percentile([1], fraction=0.25) == 1.0
    assert _percentile([1, 5], fraction=0.5) == 3.0

    empty_summary = _sequence_length_summary(sequence_lengths=[], history_size=3)
    assert empty_summary == SequenceLengthSummary(
        min=0,
        p25=0.0,
        median=0.0,
        p75=0.0,
        max=0,
        mean=0.0,
        count_lte_history_size=0,
        count_gt_history_size=0,
    )
    non_empty_summary = _sequence_length_summary(
        sequence_lengths=[1, 3, 5],
        history_size=3,
    )
    assert non_empty_summary.count_lte_history_size == 2
    assert _sequence_length_summary_to_dict(non_empty_summary)["max"] == 5

    warmup = WarmupAccounting(
        events_seen=4,
        insufficient_history=1,
        events_eligible=3,
        insufficient_history_rate=0.25,
    )
    assert _warmup_accounting_to_dict(warmup)["events_seen"] == 4
    no_eligible = NoEligibleSummary(sequence_count=2, label_counts={0: 1, 1: 1})
    assert _no_eligible_summary_to_dict(no_eligible)["label_counts"] == {0: 1, 1: 1}
    split_summary = SplitAuditSummary(
        sequence_count=1,
        event_count=2,
        normal_sequence_count=1,
        anomalous_sequence_count=0,
        warmup=warmup,
    )
    assert _split_audit_summary_to_dict(split_summary)["warmup"]["events_eligible"] == 3
    training_target_summary = TrainingTargetSummary(
        eligible_normal_event_count=1,
        excluded_anomalous_event_count=1,
        excluded_context_event_count=0,
        will_train=True,
    )
    assert _training_target_summary_to_dict(training_target_summary)["will_train"] is True
    policy_summary = HDFSFirst100kPolicySummary(
        policy_name="split_partial_sequences",
        train_normal_sessions=1,
        train_anomalous_sessions=0,
        ignored_sessions=0,
        test_normal_sessions=1,
        test_anomalous_sessions=0,
        total_sessions=2,
        emitted_segment_count=2,
        template_count=3,
        no_eligible_sessions=1,
        train_normal_delta=-1,
        test_normal_delta=-2,
        test_anomalous_delta=-3,
    )
    assert policy_summary.to_dict()["policy_name"] == "split_partial_sequences"
    assert _build_split_strategy(
        config=DatasetVariantConfig(
            name="demo",
            dataset_name="demo",
            preset="demo",
            sequence=EntitySequenceConfig(),
        ),
    )[0] == "entity"


def test_audit_evaluation_and_hdfs_helpers_cover_edge_cases(
    tmp_path: Path,
) -> None:
    """Evaluation and HDFS helpers should handle mixed chronological cases."""
    sequences = [
        TemplateSequence(
            events=[("A", [], None), ("B", [], None), ("C", [], None)],
            label=0,
            entity_ids=["a"],
            window_id=1,
            split_label=SplitLabel.TRAIN,
            evaluation_event_mask=(False, True, True),
        ),
        TemplateSequence(
            events=[("D", [], None), ("E", [], None)],
            label=1,
            entity_ids=["b"],
            window_id=2,
            split_label=SplitLabel.TEST,
            evaluation_event_mask=(True, True),
        ),
    ]
    evaluation_without_context = _evaluation_warmup_from_sequences(
        sequences=sequences,
        history_size=1,
        carry_context=False,
    )
    evaluation_with_context = _evaluation_warmup_from_sequences(
        sequences=sequences,
        history_size=1,
        carry_context=True,
    )

    assert isinstance(evaluation_without_context, EvaluationWarmupSummary)
    assert evaluation_without_context.events_eligible == 3
    assert evaluation_without_context.insufficient_history == 1
    assert evaluation_with_context.events_eligible == 4
    assert evaluation_with_context.lost_event_line_orders == []

    chrono_config = DatasetVariantConfig(
        name="demo",
        dataset_name="DEMO",
        preset="demo",
        sequence=ChronologicalStreamSequenceConfig(chunk_size=5),
    )
    chunked = _dataset_config_with_chunk_size(chrono_config, chunk_size=9)
    assert chunked.sequence.chunk_size == 9

    rows = [
        structured_line(
            line_order=0,
            timestamp_unix_ms=0,
            entity_id="a",
            untemplated_message_text="one",
            anomalous=0,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=1,
            entity_id="a",
            untemplated_message_text="two",
            anomalous=0,
        ),
        structured_line(
            line_order=2,
            timestamp_unix_ms=2,
            entity_id="b",
            untemplated_message_text="three",
            anomalous=1,
        ),
        structured_line(
            line_order=4,
            timestamp_unix_ms=4,
            entity_id="b",
            untemplated_message_text="four",
            anomalous=1,
        ),
        structured_line(
            line_order=5,
            timestamp_unix_ms=5,
            entity_id="c",
            untemplated_message_text="five",
            anomalous=0,
        ),
    ]
    observations = _collect_hdfs_session_observations(
        rows=rows,
        label_for_group=lambda entity_id: 1 if entity_id == "b" else 0,
        cutoff=3,
    )
    assert [observation.entity_id for observation in observations] == ["a", "b", "c"]
    assert observations[1].pre_cutoff_event_count == 1
    assert observations[1].post_cutoff_event_count == 1

    split_partial_segments = _hdfs_segments_for_policy(
        policy_name="split_partial_sequences",
        session=observations[1],
        cutoff=3,
    )
    assert split_partial_segments == [
        (SplitLabel.IGNORED, 1, 1),
        (SplitLabel.TEST, 1, 1),
    ]
    assert _hdfs_segments_for_policy(
        policy_name="assign_by_first_event",
        session=observations[1],
        cutoff=3,
    ) == [(SplitLabel.IGNORED, 2, 1)]
    assert _hdfs_segments_for_policy(
        policy_name="assign_by_last_event",
        session=observations[1],
        cutoff=3,
    ) == [(SplitLabel.TEST, 2, 1)]
    assert _hdfs_segments_for_policy(
        policy_name="first_100k_block_ids",
        session=observations[0],
        cutoff=3,
    ) == [(SplitLabel.TRAIN, 2, 0)]
    assert _hdfs_segments_for_policy(
        policy_name="normal_complete_sessions",
        session=observations[1],
        cutoff=3,
    ) == [(SplitLabel.TEST, 2, 1)]
    with pytest.raises(ValueError, match="Unsupported HDFS policy"):
        _hdfs_segments_for_policy(
            policy_name="missing",
            session=observations[0],
            cutoff=3,
        )

    summaries = _summarise_hdfs_first_100k_policies(
        sessions=observations,
        cutoff=3,
        history_size=1,
        template_count=2,
    )
    assert len(summaries) == 5
    assert summaries[0].policy_name == "split_partial_sequences"
    assert summaries[0].total_sessions == 3
    assert summaries[0].to_dict()["template_count"] == 2


def test_audit_bgl_sensitivity_helpers_use_chunk_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The BGL audit wrappers should honour chunk size and context policies."""
    raw_path = tmp_path / "raw.log"
    raw_path.write_text("a\nb\nc\n", encoding="utf-8")
    sink = InMemoryStructuredSink(
        dataset_name="bgl",
        raw_dataset_path=raw_path,
        parser=types.SimpleNamespace(name="null"),
        rows=[
            structured_line(
                line_order=0,
                timestamp_unix_ms=0,
                entity_id="a",
                untemplated_message_text="one",
                anomalous=0,
            ),
            structured_line(
                line_order=1,
                timestamp_unix_ms=1,
                entity_id="a",
                untemplated_message_text="two",
                anomalous=0,
            ),
        ],
    )
    sequences = [
        TemplateSequence(
            events=[("A", [], None), ("B", [], None)],
            label=0,
            entity_ids=["a"],
            window_id=1,
            split_label=SplitLabel.TRAIN,
            evaluation_event_mask=(True, True),
            training_event_mask=(True, True),
        ),
    ]
    templated = types.SimpleNamespace(sink=sink, anomaly_labels=label_lookup())
    report = DeepLogDatasetAudit(
        dataset_variant="demo",
        dataset_name="BGL",
        raw_log_entry_count=3,
        parsed_event_count=2,
        parsed_template_count=2,
        event_label_distribution={"0": 2},
        sequence_label_distribution={"0": 1},
        grouping_key="chronological_stream",
        split_strategy={"grouping": "chronological_stream"},
        raw_entry_split_summary=None,
        sequence_count=1,
        train_sequence_count=1,
        train_event_count=2,
        train_normal_sequence_count=1,
        train_anomalous_sequence_count=0,
        test_sequence_count=0,
        test_event_count=0,
        test_normal_sequence_count=0,
        test_anomalous_sequence_count=0,
        ignored_sequence_count=0,
        ignored_event_count=0,
        ignored_normal_sequence_count=0,
        ignored_anomalous_sequence_count=0,
        sequence_length_summary=SequenceLengthSummary(
            min=2,
            p25=2.0,
            median=2.0,
            p75=2.0,
            max=2,
            mean=2.0,
            count_lte_history_size=0,
            count_gt_history_size=1,
        ),
        warmup_overall=WarmupAccounting(
            events_seen=2,
            insufficient_history=1,
            events_eligible=1,
            insufficient_history_rate=0.5,
        ),
        warmup_by_split={},
        no_eligible_predictions=NoEligibleSummary(sequence_count=0, label_counts={}),
        no_eligible_predictions_by_split={},
        training_target_summary=TrainingTargetSummary(
            eligible_normal_event_count=2,
            excluded_anomalous_event_count=0,
            excluded_context_event_count=0,
            will_train=True,
        ),
        split_summaries={},
    )

    monkeypatch.setattr(
        audit_module,
        "validate_deeplog_paper_config",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        audit_module,
        "build_dataset_spec",
        lambda config, repo_root: types.SimpleNamespace(build=lambda: templated),
    )
    monkeypatch.setattr(
        audit_module,
        "audit_dataset_for_deeplog",
        lambda **kwargs: report,
    )
    monkeypatch.setattr(
        ChronologicalStreamSequenceConfig,
        "apply",
        lambda self, templated_dataset: sequences,
    )

    summaries = audit_bgl_chunk_size_sensitivity(
        config=DatasetVariantConfig(
            name="bgl_deeplog_ccs2017_paper_1pct_normal_entry_stream_no_online",
            dataset_name="BGL",
            preset="bgl",
            sequence=ChronologicalStreamSequenceConfig(
                chunk_size=3,
                split=RawEntryPrefixFractionSplitConfig(train_entry_fraction=0.8),
                train_fraction=0.8,
                test_fraction=0.2,
            ),
        ),
        repo_root=tmp_path,
        history_size=1,
        chunk_sizes=(3, 5),
    )
    assert [summary.chunk_size for summary in summaries] == [3, 5]
    assert all(summary.warmup_loss == summary.insufficient_history for summary in summaries)

    warmup = audit_bgl_continuous_stream_warmup(
        config=DatasetVariantConfig(
            name="bgl_deeplog_ccs2017_paper_1pct_normal_entry_stream_no_online",
            dataset_name="BGL",
            preset="bgl",
            sequence=ChronologicalStreamSequenceConfig(
                chunk_size=3,
                split=RawEntryPrefixFractionSplitConfig(train_entry_fraction=0.8),
                train_fraction=0.8,
                test_fraction=0.2,
            ),
        ),
        repo_root=tmp_path,
        history_size=1,
    )
    assert warmup.events_eligible == 1


def test_audit_validation_helpers_reject_protocol_drifts() -> None:
    """Paper-protocol guards should fail fast on mismatched configs."""
    bgl_sequence = ChronologicalStreamSequenceConfig(
        chunk_size=100_000,
        split=RawEntryPrefixNormalFractionSplitConfig(
            train_normal_entry_fraction=0.01,
        ),
        train_fraction=0.01,
        test_fraction=0.99,
    )
    bgl_config = DatasetVariantConfig(
        name="bgl_deeplog_ccs2017_paper_1pct_normal_entry_stream_no_online",
        dataset_name="BGL",
        preset="bgl",
        template_parser="drain3",
        sequence=bgl_sequence,
    )
    bgl_model = types.SimpleNamespace(
        history_size=3,
        top_g_values=(1, 3, 6),
        num_layers=1,
        hidden_size=256,
    )
    audit_module.validate_deeplog_paper_config(
        dataset_config=bgl_config,
        model_config=bgl_model,
    )

    hdfs_config = DatasetVariantConfig(
        name="hdfs_v1_deeplog_paper_entry100k_split_partial",
        dataset_name="HDFS",
        preset="hdfs_v1",
        sequence=EntitySequenceConfig(
            train_on_normal_entities_only=True,
            split=RawEntryPrefixCountSplitConfig(train_entry_count=100_000),
            train_fraction=0.01,
            test_fraction=0.99,
        ),
    )
    audit_module.validate_deeplog_paper_config(dataset_config=hdfs_config)

    bgl_2022_config = DatasetVariantConfig(
        name="bgl_how_far_are_we_2022_demo",
        dataset_name="BGL",
        preset="bgl",
        template_parser="drain3",
        sequence=TimeSequenceConfig(
            time_span_ms=3_600_000,
            split=RawEntryPrefixFractionSplitConfig(
                application_order=SplitApplicationOrder.BEFORE_GROUPING,
                straddling_group_policy=StraddlingGroupPolicy.DROP_STRADDLERS,
                train_entry_fraction=0.8,
            ),
            step=3_600_000,
            train_fraction=0.8,
            test_fraction=0.2,
        ),
    )
    audit_module.validate_bgl_how_far_are_we_2022_config(
        dataset_config=bgl_2022_config,
    )
