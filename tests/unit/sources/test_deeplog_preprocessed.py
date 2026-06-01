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
    FileBoundarySplitProvenance,
    NormalOnlySessionPrefixProvenance,
    build_labelled_raw_file_boundary_provenance,
    build_normal_only_session_prefix_provenance,
    build_session_file_boundary_provenance,
    materialise_labelled_raw_stream,
    materialise_labelled_session_stream,
)
from anomalog.sources.openstack import materialise_openstack_deeplog_parameter_ci_subset

_OPENSTACK_DATETIME_TO_UNIX_MS = vars(openstack_source)[
    "_openstack_datetime_to_unix_ms"
]
_OPENSTACK_EXTRACT_INSTANCE_ID = vars(openstack_source)[
    "_extract_openstack_instance_id"
]
_OPENSTACK_MULTIPLY_BUILD_SECONDS = vars(openstack_source)["_multiply_build_seconds"]
_OPENSTACK_PARAMETER_EVENT = vars(openstack_source)["_OpenStackParameterEvent"]
_OPENSTACK_REBUILD_LINE = vars(openstack_source)["_rebuild_openstack_line"]
_OPENSTACK_DATETIME_WITH_MICROS_MS = 1_577_836_830_123
_OPENSTACK_DATETIME_SECONDS_ONLY_MS = 1_577_836_830_000


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


def test_post_processed_source_reports_known_split_provenance(
    tmp_path: Path,
) -> None:
    """Recognised post-processors should surface their split provenance.

    Args:
        tmp_path (Path): Temporary directory used to stage the synthetic
            source tree.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()

    labelled_raw_source = PostProcessedSource(
        base_source=_StubSource(source_root),
        post_process=partial(
            materialise_labelled_raw_stream,
            split_files=(
                ("train.log", "train", 0),
                ("normal.log", "test_normal", 0),
                ("abnormal.log", "test_abnormal", 1),
            ),
        ),
    )
    assert labelled_raw_source.split_provenance == FileBoundarySplitProvenance(
        split_source="predefined_file_boundary",
        train_source_files=("train.log",),
        test_normal_source_files=("normal.log",),
        test_anomalous_source_files=("abnormal.log",),
    )

    labelled_session_source = PostProcessedSource(
        base_source=_StubSource(source_root),
        post_process=partial(
            materialise_labelled_session_stream,
            split_files=(
                ("train.log", 0),
                ("normal.log", 0),
                ("abnormal.log", 1),
            ),
        ),
    )
    assert labelled_session_source.split_provenance == FileBoundarySplitProvenance(
        split_source="predefined_file_boundary",
        train_source_files=("train.log",),
        test_normal_source_files=("normal.log",),
        test_anomalous_source_files=("abnormal.log",),
    )

    normal_only_source = PostProcessedSource(
        base_source=_StubSource(source_root),
        post_process=partial(
            materialise_labelled_session_stream,
            split_files=(("train.log", 0),),
            excluded_source_files=("normal.log",),
            excluded_anomalous_source_files=("abnormal.log",),
        ),
    )
    assert normal_only_source.split_provenance == NormalOnlySessionPrefixProvenance(
        split_source="normal_only_event_prefix",
        included_source_files=("train.log",),
        excluded_source_files=("normal.log",),
        excluded_anomalous_source_files=("abnormal.log",),
    )

    unrelated_source = PostProcessedSource(
        base_source=_StubSource(source_root),
        post_process=lambda _source_root, _raw_logs_path: None,
    )
    assert unrelated_source.split_provenance is None


def test_post_processed_source_materialise_reports_missing_output_file(
    tmp_path: Path,
) -> None:
    """Post-processing should fail if the derived raw log was not written.

    Args:
        tmp_path (Path): Temporary directory used to stage the synthetic
            source tree.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = PostProcessedSource(
        base_source=_StubSource(source_root),
        post_process=lambda _source_root, _raw_logs_path: None,
        raw_logs_relpath=Path("preprocessed/hdfs_events.log"),
    )

    with pytest.raises(FileNotFoundError, match=r"preprocessed/hdfs_events\.log"):
        source.materialise(dst_dir=tmp_path / "dataset")


def test_post_processed_source_materialise_uses_default_raw_log_name(
    tmp_path: Path,
) -> None:
    """Sources without an explicit raw-log path should use `<dataset>.log`.

    Args:
        tmp_path (Path): Temporary directory used to stage the synthetic
            source tree.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()

    def _write_default_raw_log(dataset_root: Path, raw_logs_path: Path) -> None:
        assert raw_logs_path == dataset_root / "dataset.log"
        raw_logs_path.write_text("hello\n", encoding="utf-8")

    source = PostProcessedSource(
        base_source=_StubSource(source_root),
        post_process=_write_default_raw_log,
    )

    dataset_root = source.materialise(dst_dir=tmp_path / "dataset")

    assert dataset_root == source_root
    assert (source_root / "dataset.log").read_text(encoding="utf-8") == "hello\n"


@pytest.mark.parametrize(
    ("raw_logs_relpath", "expected"),
    [
        (Path("outside.log").resolve(), "must be relative to the dataset root"),
        (Path("../outside.log"), "must stay within the dataset root"),
    ],
)
def test_post_processed_source_rejects_invalid_raw_log_paths(
    tmp_path: Path,
    raw_logs_relpath: Path,
    expected: str,
) -> None:
    """Derived raw-log paths should stay inside the materialised dataset root.

    Args:
        tmp_path (Path): Temporary directory used to stage the synthetic
            source tree.
        raw_logs_relpath (Path): Invalid raw-log path under test.
        expected (str): Expected validation message for the invalid path.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = PostProcessedSource(
        base_source=_StubSource(source_root),
        post_process=lambda _source_root, _raw_logs_path: None,
        raw_logs_relpath=raw_logs_relpath,
    )

    with pytest.raises(ValueError, match=expected):
        source.materialise(dst_dir=tmp_path / "dataset")


def test_build_provenance_helpers_reject_incomplete_split_specs() -> None:
    """Split provenance builders should validate the expected split arity."""
    with pytest.raises(ValueError, match="three source files"):
        build_session_file_boundary_provenance((("train.log", 0),))
    with pytest.raises(ValueError, match="at least one source file"):
        build_normal_only_session_prefix_provenance(
            (),
            excluded_source_files=(),
            excluded_anomalous_source_files=(),
        )
    with pytest.raises(ValueError, match="three source files"):
        build_labelled_raw_file_boundary_provenance((("train.log", "train", 0),))


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


def test_openstack_helpers_preserve_numeric_values_when_requested() -> None:
    """OpenStack normalisation should keep numeric path segments when asked."""
    assert (
        openstack_source.normalise_openstack_message(
            "[instance: vm-0001] keep /tmp/42/cache",
            preserve_numeric_values=True,
        )
        == "keep /tmp/42/cache"
    )
    assert (
        openstack_source.normalise_openstack_message(
            "[instance: vm-0001] keep /tmp/42/cache",
            preserve_numeric_values=False,
        )
        == "keep /tmp/NUM/cache"
    )


def test_parse_openstack_payload_accepts_for_instance_syntax() -> None:
    """OpenStack payload parsing should recognise alternate instance syntax."""
    parsed = openstack_source.parse_openstack_payload(
        "nova.compute 2017-01-01 00:00:30.000 1 INFO nova.compute "
        "[addr] Build complete for instance vm-alpha",
    )

    assert parsed is not None
    assert parsed.instance_id == "vm-alpha"
    assert parsed.content == "Build complete for instance vm-alpha"
    assert parsed.raw_parameters == []


def test_openstack_helpers_cover_timestamp_and_message_edge_cases() -> None:
    """OpenStack helpers should handle alternate timestamps and path tokens."""
    assert (
        _OPENSTACK_DATETIME_TO_UNIX_MS(
            "2020-01-01",
            "00:00:30.123456",
        )
        == _OPENSTACK_DATETIME_WITH_MICROS_MS
    )
    assert (
        _OPENSTACK_DATETIME_TO_UNIX_MS(
            "2020-01-01",
            "00:00:30",
        )
        == _OPENSTACK_DATETIME_SECONDS_ONLY_MS
    )
    assert (
        _OPENSTACK_DATETIME_TO_UNIX_MS(
            "2020-99-01",
            "00:00:30",
        )
        is None
    )
    assert (
        _OPENSTACK_EXTRACT_INSTANCE_ID(
            "[instance: vm-alpha] build complete",
        )
        == "vm-alpha"
    )
    assert (
        _OPENSTACK_EXTRACT_INSTANCE_ID(
            "build complete without an instance id",
        )
        is None
    )
    assert (
        openstack_source.normalise_openstack_message(
            "[instance: vm-0001] "
            "/var/lib/nova/instances/_base/"
            "a489c868f0c37da93b76227c91bb03908ac0e742 "
            "10.0.0.1 /tmp/42/cache, 2048",
            preserve_numeric_values=False,
        )
        == "INSTANCE_PATH IP /tmp/NUM/cache, NUM"
    )
    assert (
        openstack_source.normalise_openstack_message(
            "[instance: vm-0001] "
            "/var/lib/nova/instances/_base/"
            "a489c868f0c37da93b76227c91bb03908ac0e742 "
            "10.0.0.1 /tmp/42/cache, 2048",
            preserve_numeric_values=True,
        )
        == "INSTANCE_PATH IP /tmp/42/cache, 2048"
    )


def test_parse_openstack_payload_rejects_invalid_timestamp_and_missing_instance() -> (
    None
):
    """OpenStack payload parsing should reject malformed or unscoped rows."""
    assert (
        openstack_source.parse_openstack_payload(
            "nova.compute 2017-99-01 00:00:30.000 1 INFO nova.compute "
            "[instance: vm-alpha] Build complete",
        )
        is None
    )
    assert (
        openstack_source.parse_openstack_payload(
            "nova.compute 2017-01-01 00:00:30.000 1 INFO nova.compute "
            "[addr] Build complete",
        )
        is None
    )


def test_openstack_parameter_helpers_cover_rebuild_and_duration_scaling() -> None:
    """OpenStack parameter helpers should keep fallbacks and scaling stable."""
    event = _OPENSTACK_PARAMETER_EVENT(
        raw_payload="not-an-openstack-row",
        timestamp_ms=123,
        content="Took 1.5 seconds to build instance.",
    )
    assert (
        _OPENSTACK_REBUILD_LINE(
            event,
            timestamp_ms=456,
            content="Took 3 seconds to build instance.",
        )
        == "not-an-openstack-row"
    )
    assert (
        _OPENSTACK_MULTIPLY_BUILD_SECONDS(
            "[instance: vm-alpha] Took 1.5 seconds to build instance.",
            2.0,
        )
        == "[instance: vm-alpha] Took 3 seconds to build instance."
    )
    assert (
        _OPENSTACK_MULTIPLY_BUILD_SECONDS(
            "Took 1 seconds to build instance.",
            2.0,
        )
        == "Took 2 seconds to build instance."
    )
    assert (
        _OPENSTACK_MULTIPLY_BUILD_SECONDS(
            "unrelated",
            2.0,
        )
        == "unrelated"
    )


def test_materialise_openstack_deeplog_parameter_ci_subset_requires_duration_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenStack Figure 9 materialisation should fail without the duration row.

    Args:
        tmp_path (Path): Temporary directory used to stage the synthetic source
            tree.
        monkeypatch (pytest.MonkeyPatch): Patch helper used to shrink the
            OpenStack fixture to the smallest failing slice.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    sync_power_state_content = (
        "During sync_power_state the instance has a pending task (spawning). Skip."
    )
    normal_lines: list[str] = []
    for instance_index in range(5):
        instance_id = f"instance-{instance_index:02d}"
        base_event_index = instance_index * 2
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
        5,
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

    raw_logs_path = tmp_path / "preprocessed" / "openstack_parameter_subset.log"
    raw_logs_path.parent.mkdir(parents=True, exist_ok=True)

    with pytest.raises(
        ValueError,
        match="Could not find the build-duration template on held-out OpenStack",
    ):
        openstack_source.materialise_openstack_deeplog_parameter_ci_subset(
            source_root,
            raw_logs_path,
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
