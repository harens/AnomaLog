"""Tests for the DeepLog preprocessed dataset source helpers."""

from functools import partial
from pathlib import Path
from typing import ClassVar

import pytest
from typing_extensions import override

from anomalog.sources import PostProcessedSource
from anomalog.sources import openstack as openstack_source
from anomalog.sources.contracts import DatasetSource
from anomalog.sources.deeplog_preprocessed import (
    materialise_labelled_raw_stream,
    materialise_labelled_session_stream,
)
from anomalog.sources.openstack import (
    materialise_openstack_deeplog_parameter_ci_subset,
)


def test_post_processed_source_invokes_keyword_only_post_processor(
    tmp_path: Path,
) -> None:
    """Preset-style partials should materialise the derived raw log stream.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for the synthetic source.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "hdfs_train").write_text("a b\n", encoding="utf-8")
    (source_root / "hdfs_test_normal").write_text("c\n", encoding="utf-8")
    (source_root / "hdfs_test_abnormal").write_text("d e\n", encoding="utf-8")

    dataset_root = tmp_path / "dataset"
    source = PostProcessedSource(
        base_source=_StubSource(source_root),
        post_process=partial(
            materialise_labelled_session_stream,
            split_files=(
                ("hdfs_train", 0),
                ("hdfs_test_normal", 0),
                ("hdfs_test_abnormal", 1),
            ),
        ),
        raw_logs_relpath=Path("preprocessed/hdfs_events.log"),
    )

    materialised_root = source.materialise(dst_dir=dataset_root)
    raw_logs_path = materialised_root / "preprocessed/hdfs_events.log"

    assert materialised_root == source_root
    assert raw_logs_path.read_text(encoding="utf-8") == (
        "hdfs_train:0\t0\ta\n"
        "hdfs_train:0\t0\tb\n"
        "hdfs_test_normal:0\t0\tc\n"
        "hdfs_test_abnormal:0\t1\td\n"
        "hdfs_test_abnormal:0\t1\te\n"
    )


class _StubSource(DatasetSource):
    """Minimal dataset source stub for post-processing tests.

    Args:
        dataset_root (Path): Materialised dataset root returned by the stub.

    Attributes:
        name (ClassVar[str]): Stable registry name required by the protocol.
        raw_logs_relpath (Path | None): Canonical derived log path for the test.
    """

    name: ClassVar[str] = "stub"
    raw_logs_relpath: Path | None = Path("preprocessed/hdfs_events.log")

    def __init__(self, dataset_root: Path) -> None:
        self._dataset_root = dataset_root

    @override
    def materialise(self, *, dst_dir: Path) -> Path:
        del dst_dir
        return self._dataset_root

    @override
    def raw_logs_path(self, *, dataset_name: str, dataset_root: Path) -> Path:
        del dataset_name
        return dataset_root / "preprocessed" / "hdfs_events.log"


def test_materialise_labelled_raw_stream_preserves_split_order_and_labels(
    tmp_path: Path,
) -> None:
    """Raw split materialisation should keep raw lines and explicit split labels.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for the synthetic source.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    for filename in (
        "openstack_normal1.log",
        "openstack_normal2.log",
        "openstack_abnormal.log",
    ):
        (source_root / filename).write_text("placeholder\n", encoding="utf-8")

    (source_root / "openstack_normal1.log").write_text(
        "n1 line 1\nn1 line 2\n",
        encoding="utf-8",
    )
    (source_root / "openstack_normal2.log").write_text("n2 line 1\n", encoding="utf-8")
    (source_root / "openstack_abnormal.log").write_text("ab line 1\n", encoding="utf-8")

    raw_logs_path = tmp_path / "preprocessed" / "openstack_labelled_raw.log"
    raw_logs_path.parent.mkdir(parents=True, exist_ok=True)
    materialise_labelled_raw_stream(
        source_root,
        raw_logs_path,
        (
            ("openstack_normal1.log", "openstack_train", 0),
            ("openstack_normal2.log", "openstack_test_normal", 0),
            ("openstack_abnormal.log", "openstack_test_abnormal", 1),
        ),
    )
    assert raw_logs_path.read_text(encoding="utf-8") == (
        "openstack_train\t0\tn1 line 1\n"
        "openstack_train\t0\tn1 line 2\n"
        "openstack_test_normal\t0\tn2 line 1\n"
        "openstack_test_abnormal\t1\tab line 1\n"
    )


def test_materialise_openstack_deeplog_parameter_ci_subset_injects_two_shared_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Figure 9-style OpenStack materialisation should inject two shared points.

    Args:
        tmp_path (Path): Temporary directory used to stage the synthetic source
            tree.
        monkeypatch (pytest.MonkeyPatch): Patch helper used to tune the
            OpenStack fixture size.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    sync_power_state_content = (
        "During sync_power_state the instance has a pending task (spawning). Skip."
    )
    build_instance_content = "Took 1 seconds to build instance."
    normal_lines: list[str] = []
    for instance_index in range(6):
        instance_id = f"instance-{instance_index:02d}"
        base_event_index = instance_index * 3
        normal_lines.extend(
            (
                _openstack_parameter_line(
                    base_event_index,
                    instance_id=instance_id,
                    content="VM Started (Lifecycle Event)",
                ),
                _openstack_parameter_line(
                    base_event_index + 1,
                    instance_id=instance_id,
                    content=sync_power_state_content,
                ),
                _openstack_parameter_line(
                    base_event_index + 2,
                    instance_id=instance_id,
                    content=build_instance_content,
                ),
            ),
        )
    (source_root / "openstack_normal1.log").write_text(
        "\n".join(normal_lines) + "\n",
        encoding="utf-8",
    )
    (source_root / "openstack_normal2.log").write_text("", encoding="utf-8")
    (source_root / "openstack_abnormal.log").write_text("", encoding="utf-8")

    monkeypatch.setattr(
        openstack_source,
        "_OPENSTACK_PARAMETER_SUBSET_INSTANCES",
        6,
    )
    monkeypatch.setattr(
        openstack_source,
        "_OPENSTACK_PARAMETER_TRAIN_INSTANCES",
        1,
    )
    monkeypatch.setattr(
        openstack_source,
        "_OPENSTACK_PARAMETER_VALIDATION_INSTANCES",
        1,
    )
    monkeypatch.setattr(
        openstack_source,
        "_OPENSTACK_PARAMETER_PERFORMANCE_OFFSET",
        1,
    )
    monkeypatch.setattr(
        openstack_source,
        "_OPENSTACK_PARAMETER_ANOMALY_INSTANCE_OFFSETS",
        (1, 2),
    )
    monkeypatch.setattr(
        openstack_source,
        "_OPENSTACK_PARAMETER_PERFORMANCE_DELAY_MS",
        1_000,
    )
    monkeypatch.setattr(
        openstack_source,
        "_OPENSTACK_PARAMETER_DURATION_MULTIPLIER",
        2.0,
    )

    raw_logs_path = tmp_path / "preprocessed" / "openstack_parameter_subset.log"
    raw_logs_path.parent.mkdir(parents=True, exist_ok=True)
    materialise_openstack_deeplog_parameter_ci_subset(source_root, raw_logs_path)

    rows = raw_logs_path.read_text(encoding="utf-8").splitlines()
    labelled_rows = [row for row in rows if row.split("\t", 2)[1] == "1"]
    expected_shared_throttle_points = 2
    expected_anomalous_row_count = expected_shared_throttle_points * 2
    assert len(labelled_rows) == expected_anomalous_row_count
    assert (
        sum(sync_power_state_content in row for row in labelled_rows)
        == expected_shared_throttle_points
    )
    assert (
        sum("Took 2 seconds to build instance." in row for row in labelled_rows)
        == expected_shared_throttle_points
    )


def _openstack_parameter_line(
    event_index: int,
    *,
    instance_id: str,
    content: str,
) -> str:
    return (
        f"{event_index:08d} 2020-01-01 00:00:{event_index:02d}.000000 123 INFO "
        f"nova.compute [addr] [instance: {instance_id}] {content}"
    )
