"""Tests for local dataset sources."""

import zipfile
from pathlib import Path

import pytest
from prefect.logging import disable_run_logger

from anomalog.sources.local import LocalDirSource, LocalZipSource


def test_local_dir_source_returns_existing_directory(tmp_path: Path) -> None:
    """LocalDirSource.materialise returns the configured directory."""
    source_dir = tmp_path / "dataset"
    source_dir.mkdir()

    assert LocalDirSource(source_dir).materialise(tmp_path / "unused") == source_dir


def test_local_dir_source_rejects_missing_and_non_directory_paths(
    tmp_path: Path,
) -> None:
    """LocalDirSource raises the appropriate filesystem exceptions."""
    missing = tmp_path / "missing"
    file_path = tmp_path / "file.txt"
    file_path.write_text("data", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        LocalDirSource(missing).materialise(tmp_path / "unused")

    with pytest.raises(NotADirectoryError):
        LocalDirSource(file_path).materialise(tmp_path / "unused")


def test_local_zip_source_extracts_archive_contents(
    tmp_path: Path,
) -> None:
    """LocalZipSource extracts a valid zip archive into the destination."""
    zip_path = tmp_path / "dataset.zip"
    dst_dir = tmp_path / "out"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("logs/demo.log", "hello\n")
    with disable_run_logger():
        result = LocalZipSource(zip_path).materialise(dst_dir)

    assert result == dst_dir
    assert (dst_dir / "logs" / "demo.log").read_text(encoding="utf-8") == "hello\n"


def test_local_zip_source_short_circuits_when_destination_is_non_empty(
    tmp_path: Path,
) -> None:
    """Existing extracted content is reused before zip validation."""
    dst_dir = tmp_path / "out"
    dst_dir.mkdir()
    (dst_dir / "already.txt").write_text("ready", encoding="utf-8")

    result = LocalZipSource(tmp_path / "missing.zip").materialise(dst_dir)

    assert result == dst_dir
