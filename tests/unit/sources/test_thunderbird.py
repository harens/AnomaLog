"""Tests for Thunderbird-specific raw archive helpers."""

from pathlib import Path

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
