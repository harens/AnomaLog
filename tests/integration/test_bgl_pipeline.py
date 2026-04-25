"""Integration test for a small BGL archive flowing into entity sequences."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from anomalog import DatasetSpec
from anomalog.cache import CachePathsConfig
from anomalog.parsers import BGLParser, Drain3Parser
from anomalog.sequences import SplitLabel
from anomalog.sources.local import LocalZipSource

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anomalog.parsers.structured.contracts import StructuredLine
    from anomalog.parsers.template.dataset import TemplatedDataset
    from anomalog.sequences import TemplateSequence

FIXTURE_LOG = Path(__file__).parent / "logs" / "tiny_bgl_happy_path.log"
RAW_LOGS_RELPATH = Path("BGL.log")
ANOMALOUS_ENTITY = "R04-M1-N4-I:J18-U11"
NORMAL_ENTITY_A = "R23-M1-N8-I:J18-U11"
NORMAL_ENTITY_B = "R35-M1-N0-I:J18-U01"
EXPECTED_ROW_COUNT = 8
EXPECTED_TEMPLATE_COUNT = 3


def _build_local_bgl_archive(zip_path: Path) -> list[str]:
    """Package the checked-in BGL fixture into the local archive shape users fetch.

    Args:
        zip_path (Path): Destination archive path to create.

    Returns:
        list[str]: Raw log lines stored in the fixture archive.
    """
    raw_lines = FIXTURE_LOG.read_text(encoding="utf-8").splitlines()
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(FIXTURE_LOG, arcname=RAW_LOGS_RELPATH.as_posix())
    return raw_lines


def _dataset_from_local_archive(tmp_path: Path, archive_path: Path) -> DatasetSpec:
    """Create a realistic fluent dataset spec for a locally supplied BGL bundle.

    Args:
        tmp_path (Path): Temporary directory whose cache roots should be used.
        archive_path (Path): Local fixture archive to build the dataset from.

    Returns:
        DatasetSpec: Fluent dataset spec pointing at the local archive fixture.
    """
    return (
        DatasetSpec("tiny-bgl-happy-path")
        .from_source(LocalZipSource(archive_path, raw_logs_relpath=RAW_LOGS_RELPATH))
        .parse_with(BGLParser())
        .with_cache_paths(
            CachePathsConfig(
                data_root=tmp_path / "data",
                cache_root=tmp_path / "cache",
            ),
        )
        .template_with(Drain3Parser)
    )


def _rows_sorted_by_line_order(dataset: TemplatedDataset) -> list[StructuredLine]:
    """Read persisted rows back in file order.

    The parquet sink scans partitioned fragments, so the default full-dataset
    iterator is deterministic but not guaranteed to match original file order.
    `line_order` is the persisted source-of-truth for reconstructing that order.

    Args:
        dataset (TemplatedDataset): Built dataset whose rows should be re-sorted.

    Returns:
        list[StructuredLine]: Persisted rows sorted back into source file order.
    """
    sink = dataset.sink
    return sorted(sink.iter_structured_lines()(), key=lambda row: row.line_order)


def _templates_by_message(
    templated: TemplatedDataset,
    rows: Iterable[StructuredLine],
) -> dict[str, str]:
    """Infer one template per raw message text for readable grouping assertions.

    Args:
        templated (TemplatedDataset): Built dataset providing template inference.
        rows (Iterable[StructuredLine]): Structured rows to index by message text.

    Returns:
        dict[str, str]: Template text keyed by original raw message.
    """
    return {
        row.untemplated_message_text: templated.template_parser.inference(
            row.untemplated_message_text,
        )[0]
        for row in rows
    }


def _sequences_by_entity(
    sequences: Iterable[TemplateSequence],
) -> dict[str, TemplateSequence]:
    """Index entity sequences by entity id for direct assertions.

    Args:
        sequences (Iterable[TemplateSequence]): Entity-grouped sequences to index.

    Returns:
        dict[str, TemplateSequence]: Entity-id keyed sequence mapping.
    """
    return {
        sequence.sole_entity_id: sequence
        for sequence in sequences
        if sequence.sole_entity_id is not None
    }


def _assert_inline_labels(
    rows: list[StructuredLine],
    dataset: TemplatedDataset,
) -> None:
    """Assert the parser and materialization preserve BGL anomaly flags.

    Args:
        rows (list[StructuredLine]): Persisted structured rows in source order.
        dataset (TemplatedDataset): Built dataset exposing inline anomaly labels.
    """
    assert [row.line_order for row in rows] == list(range(EXPECTED_ROW_COUNT))
    assert [row.anomalous for row in rows] == [0, 0, 0, 1, 1, 0, 0, 0]
    assert dataset.anomaly_labels.label_for_line(3) == 1
    assert dataset.anomaly_labels.label_for_line(4) == 1
    assert dataset.anomaly_labels.label_for_line(7) is None
    assert dataset.anomaly_labels.label_for_group(ANOMALOUS_ENTITY) == 1
    assert dataset.anomaly_labels.label_for_group(NORMAL_ENTITY_A) is None


def _assert_template_groups(templates_by_message: dict[str, str]) -> None:
    """Assert repeated message families collapse into stable templates.

    Args:
        templates_by_message (dict[str, str]): Template text keyed by original
            raw message text.
    """
    link_templates = {
        template
        for message, template in templates_by_message.items()
        if "link card" in message
    }
    normal_service_templates = {
        template
        for message, template in templates_by_message.items()
        if "service action code" in message
    }

    assert len(set(templates_by_message.values())) == EXPECTED_TEMPLATE_COUNT
    assert len(link_templates) == 1
    assert len(normal_service_templates) == 1
    assert link_templates != normal_service_templates


def _assert_entity_sequences_are_reproducible(
    first_pass: list[TemplateSequence],
    second_pass: list[TemplateSequence],
    templates_by_message: dict[str, str],
    rows: list[StructuredLine],
) -> None:
    """Assert entity grouping and train/test assignment stay deterministic.

    Args:
        first_pass (list[TemplateSequence]): First iteration over the builder.
        second_pass (list[TemplateSequence]): Second iteration over the same
            builder to check determinism.
        templates_by_message (dict[str, str]): Template text keyed by message.
        rows (list[StructuredLine]): Persisted structured rows in source order.
    """
    first_by_entity = _sequences_by_entity(first_pass)
    second_by_entity = _sequences_by_entity(second_pass)
    normal_entities = {NORMAL_ENTITY_A, NORMAL_ENTITY_B}
    expected_templates_by_entity = {
        entity_id: [
            templates_by_message[row.untemplated_message_text]
            for row in rows
            if row.entity_id == entity_id
        ]
        for entity_id in normal_entities | {ANOMALOUS_ENTITY}
    }

    assert [sequence.sole_entity_id for sequence in second_pass] == [
        sequence.sole_entity_id for sequence in first_pass
    ]
    assert set(first_by_entity) == normal_entities | {ANOMALOUS_ENTITY}
    assert second_by_entity == first_by_entity

    anomalous_sequence = first_by_entity[ANOMALOUS_ENTITY]
    assert anomalous_sequence.split_label is SplitLabel.TEST
    assert anomalous_sequence.label == 1
    assert (
        anomalous_sequence.templates == expected_templates_by_entity[ANOMALOUS_ENTITY]
    )

    train_entities = {
        entity_id
        for entity_id, sequence in first_by_entity.items()
        if sequence.split_label is SplitLabel.TRAIN
    }
    assert train_entities == normal_entities

    for entity_id in normal_entities:
        assert first_by_entity[entity_id].label == 0

    assert (
        first_by_entity[NORMAL_ENTITY_A].templates
        == expected_templates_by_entity[NORMAL_ENTITY_A]
    )
    assert (
        first_by_entity[NORMAL_ENTITY_B].templates
        == expected_templates_by_entity[NORMAL_ENTITY_B]
    )


def test_user_can_turn_a_small_bgl_archive_into_reproducible_entity_sequences(
    tmp_path: Path,
) -> None:
    """A user can preprocess a tiny BGL archive into stable train/test sequences.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for archive and cache roots.
    """
    archive_path = tmp_path / "tiny-bgl.zip"
    raw_lines = _build_local_bgl_archive(archive_path)
    dataset = _dataset_from_local_archive(tmp_path, archive_path)

    templated = dataset.build()
    rows = _rows_sorted_by_line_order(templated)
    templates_by_message = _templates_by_message(templated, rows)

    raw_logs_path = (
        dataset.cache_paths.data_root / dataset.dataset_name / RAW_LOGS_RELPATH
    )
    assert raw_logs_path.exists()
    assert templated.sink.count_rows() == EXPECTED_ROW_COUNT == len(raw_lines)
    _assert_inline_labels(rows, templated)
    _assert_template_groups(templates_by_message)

    builder = (
        templated.group_by_entity()
        .with_train_fraction(0.5)
        .with_train_on_normal_entities_only()
    )
    first_pass = list(builder)
    second_pass = list(builder)
    _assert_entity_sequences_are_reproducible(
        first_pass,
        second_pass,
        templates_by_message,
        rows,
    )

    rebuilt = dataset.build()
    rebuilt_rows = _rows_sorted_by_line_order(rebuilt)
    rebuilt_templates_by_message = _templates_by_message(rebuilt, rebuilt_rows)
    rebuilt_sequences = list(
        rebuilt.group_by_entity()
        .with_train_fraction(0.5)
        .with_train_on_normal_entities_only(),
    )

    assert rebuilt_rows == rows
    assert rebuilt_templates_by_message == templates_by_message
    assert rebuilt_sequences == first_pass
