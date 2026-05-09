"""Tests for the DeepLog preprocessed dataset source helpers."""

from functools import partial
from pathlib import Path
from typing import ClassVar

from typing_extensions import override

from anomalog.sources import PostProcessedSource
from anomalog.sources.contracts import DatasetSource
from anomalog.sources.deeplog_preprocessed import materialise_labelled_session_stream


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
