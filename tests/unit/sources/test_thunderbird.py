"""Tests for Thunderbird-specific raw archive helpers."""

from pathlib import Path

import pytest

from anomalog.sources.raw_prefix import (
    materialise_raw_log_prefix,
    materialise_raw_log_segment,
)
from anomalog.sources.remote_zip import RemoteZipSource


def test_thunderbird_source_uses_the_canonical_raw_log_path(
    tmp_path: Path,
) -> None:
    """Thunderbird should target the exact archive member declared in config.

    Args:
        tmp_path (Path): Temporary directory used to stage the synthetic
            archive.
    """
    dataset_root = tmp_path / "Thunderbird"
    dataset_root.mkdir()
    preferred = dataset_root / "Thunderbird.log"
    preferred.write_text("alpha\n", encoding="utf-8")
    (dataset_root / "Thunderbird_2k.log").write_text("beta\n", encoding="utf-8")

    source = RemoteZipSource(
        url="https://example.com/Thunderbird.tar.gz",
        raw_logs_relpath=Path("Thunderbird.log"),
    )

    assert (
        source.raw_logs_path(dataset_name="Thunderbird", dataset_root=dataset_root)
        == preferred
    )


def test_materialise_raw_log_prefix_copies_the_explicit_archive_log(
    tmp_path: Path,
) -> None:
    """The Thunderbird smoke helper should retain the first N raw lines only.

    Args:
        tmp_path (Path): Temporary directory used to stage the synthetic
            archive.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_log = source_root / "Thunderbird.log"
    source_log.write_text("one\n\nthree\nfour\n", encoding="utf-8")
    (source_root / "Thunderbird_2k.log").write_text("ignored\n", encoding="utf-8")

    raw_logs_path = tmp_path / "preprocessed" / "thunderbird_prefix.log"
    materialise_raw_log_prefix(
        source_root,
        raw_logs_path,
        source_log_relpath=Path("Thunderbird.log"),
        line_limit=2,
    )

    assert raw_logs_path.read_text(encoding="utf-8") == "one\n\n"


def test_materialise_raw_log_segment_copies_the_requested_line_range(
    tmp_path: Path,
) -> None:
    """The Thunderbird benchmark helper should copy the configured slice.

    Args:
        tmp_path (Path): Temporary directory used to stage the synthetic
            archive.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_log = source_root / "Thunderbird.log"
    source_log.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")

    raw_logs_path = tmp_path / "preprocessed" / "thunderbird_slice.log"
    materialise_raw_log_segment(
        source_root,
        raw_logs_path,
        source_log_relpath=Path("Thunderbird.log"),
        start_line=3,
        line_limit=2,
    )

    assert raw_logs_path.read_text(encoding="utf-8") == "three\nfour\n"


@pytest.mark.parametrize(
    ("source_log_relpath", "start_line", "line_limit", "expected"),
    [
        (Path("Thunderbird.log"), 0, 1, "start_line must be a positive integer."),
        (Path("Thunderbird.log"), 1, 0, "line_limit must be a positive integer."),
        (
            Path("/Thunderbird.log"),
            1,
            1,
            "source_log_relpath must be relative to the source root.",
        ),
    ],
)
def test_materialise_raw_log_segment_rejects_invalid_arguments(
    tmp_path: Path,
    source_log_relpath: Path,
    start_line: int,
    line_limit: int,
    expected: str,
) -> None:
    """The raw-prefix helper should reject invalid line-range configuration.

    Args:
        tmp_path (Path): Temporary directory used to stage the synthetic
            archive.
        source_log_relpath (Path): Relative path passed to the raw-prefix
            helper.
        start_line (int): First line number requested from the source log.
        line_limit (int): Maximum number of lines requested from the source
            log.
        expected (str): Expected validation message for the failing case.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "Thunderbird.log").write_text("one\n", encoding="utf-8")
    raw_logs_path = tmp_path / "preprocessed" / "thunderbird_invalid.log"

    with pytest.raises(ValueError, match=expected):
        materialise_raw_log_segment(
            source_root,
            raw_logs_path,
            source_log_relpath=source_log_relpath,
            start_line=start_line,
            line_limit=line_limit,
        )


def test_materialise_raw_log_segment_rejects_missing_or_directory_sources(
    tmp_path: Path,
) -> None:
    """The raw-prefix helper should fail fast for missing or directory inputs.

    Args:
        tmp_path (Path): Temporary directory used to stage the synthetic
            archive.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    raw_logs_path = tmp_path / "preprocessed" / "thunderbird_invalid.log"

    with pytest.raises(FileNotFoundError, match=r"Missing\.log"):
        materialise_raw_log_segment(
            source_root,
            raw_logs_path,
            source_log_relpath=Path("Missing.log"),
            start_line=1,
            line_limit=1,
        )

    directory = source_root / "Thunderbird.log"
    directory.mkdir()
    with pytest.raises(IsADirectoryError, match=r"Thunderbird\.log"):
        materialise_raw_log_segment(
            source_root,
            raw_logs_path,
            source_log_relpath=Path("Thunderbird.log"),
            start_line=1,
            line_limit=1,
        )
