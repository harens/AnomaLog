"""Tests for local dataset sources."""

import zipfile
from pathlib import Path

import pytest
from prefect.logging import disable_run_logger

from anomalog.sources.local import LocalDirSource, LocalZipSource


def test_local_dir_source_returns_existing_directory(tmp_path: Path) -> None:
    """LocalDirSource.materialise returns the configured raw log path.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local source fixtures.
    """
    source_dir = tmp_path / "dataset"
    source_dir.mkdir()
    log_path = source_dir / "demo.log"
    log_path.write_text("line\n", encoding="utf-8")

    source = LocalDirSource(source_dir)
    dataset_root = source.materialise(
        dst_dir=tmp_path / "unused",
    )

    assert dataset_root == source_dir
    assert (
        source.raw_logs_path(dataset_name="demo", dataset_root=dataset_root) == log_path
    )


def test_local_dir_source_rejects_missing_and_non_directory_paths(
    tmp_path: Path,
) -> None:
    """LocalDirSource raises the appropriate filesystem exceptions.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local source fixtures.
    """
    missing = tmp_path / "missing"
    file_path = tmp_path / "file.txt"
    file_path.write_text("data", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        LocalDirSource(missing).materialise(
            dst_dir=tmp_path / "unused",
        )

    with pytest.raises(NotADirectoryError):
        LocalDirSource(file_path).materialise(
            dst_dir=tmp_path / "unused",
        )


def test_local_zip_source_extracts_archive_contents(
    tmp_path: Path,
) -> None:
    """LocalZipSource extracts a valid zip archive into the destination.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for archive fixtures.
    """
    zip_path = tmp_path / "dataset.zip"
    dst_dir = tmp_path / "out"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("logs/demo.log", "hello\n")
    with disable_run_logger():
        source = LocalZipSource(
            zip_path,
            raw_logs_relpath=Path("logs/demo.log"),
        )
        dataset_root = source.materialise(dst_dir=dst_dir)

    assert dataset_root == dst_dir
    assert source.raw_logs_path(dataset_name="demo", dataset_root=dataset_root) == (
        dst_dir / "logs" / "demo.log"
    )
    assert (dst_dir / "logs" / "demo.log").read_text(encoding="utf-8") == "hello\n"


def test_local_zip_source_short_circuits_when_destination_is_non_empty(
    tmp_path: Path,
) -> None:
    """Existing extracted content is reused before zip validation.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for archive fixtures.
    """
    dst_dir = tmp_path / "out"
    dst_dir.mkdir()
    (dst_dir / "already.txt").write_text("ready", encoding="utf-8")
    (dst_dir / "demo.log").write_text("line\n", encoding="utf-8")

    source = LocalZipSource(
        tmp_path / "missing.zip",
        raw_logs_relpath=Path("demo.log"),
    )
    dataset_root = source.materialise(dst_dir=dst_dir)

    assert dataset_root == dst_dir
    assert source.raw_logs_path(dataset_name="demo", dataset_root=dataset_root) == (
        dst_dir / "demo.log"
    )


def test_local_dir_source_rejects_invalid_raw_logs_relpath(tmp_path: Path) -> None:
    """Raw log paths must remain within the dataset root and point to a file.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local source fixtures.
    """
    source_dir = tmp_path / "dataset"
    source_dir.mkdir()
    (tmp_path / "outside.log").write_text("line\n", encoding="utf-8")
    outside_source = LocalDirSource(source_dir, raw_logs_relpath=Path("../outside.log"))
    outside_root = outside_source.materialise(
        dst_dir=tmp_path / "unused",
    )
    missing_source = LocalDirSource(source_dir, raw_logs_relpath=Path("missing.log"))
    missing_root = missing_source.materialise(
        dst_dir=tmp_path / "unused",
    )

    with pytest.raises(ValueError, match="stay within the dataset root"):
        outside_source.raw_logs_path(
            dataset_name="demo",
            dataset_root=outside_root,
        )

    with pytest.raises(FileNotFoundError):
        missing_source.raw_logs_path(
            dataset_name="demo",
            dataset_root=missing_root,
        )
