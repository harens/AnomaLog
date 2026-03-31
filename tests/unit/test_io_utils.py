"""Tests for IO utility helpers."""

import zipfile
from pathlib import Path
from types import TracebackType

import pytest
from prefect.logging import disable_run_logger
from typing_extensions import Self

from anomalog.io_utils import extract_zip, verify_md5


class _TrackingProgress:
    def __init__(self) -> None:
        self.task_total: int | None = None
        self.advanced = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc, traceback
        return False

    def add_task(self, description: str, total: int, unit: str | None = None) -> int:
        del description, unit
        self.task_total = total
        return 1

    def update(self, _task: int, *, advance: int) -> None:
        self.advanced += advance


def test_verify_md5_reads_entire_file_before_raising_on_mismatch(
    tmp_path: Path,
) -> None:
    """Checksum verification should advance through the full file on failure."""
    progress = _TrackingProgress()
    file_path = tmp_path / "payload.bin"
    payload = b"abcdef"
    file_path.write_bytes(payload)

    with pytest.raises(ValueError, match="MD5 checksum mismatch"):
        verify_md5(
            file_path,
            expected_hex="00000000000000000000000000000000",
            progress_factory=lambda: progress,
        )

    assert progress.task_total == len(payload)
    assert progress.advanced == len(payload)


@pytest.mark.allow_no_new_coverage
def test_extract_zip_extracts_files_into_destination(
    tmp_path: Path,
) -> None:
    """Zip extraction should create the destination and reject corrupt members."""
    zip_path = tmp_path / "archive.zip"
    destination = tmp_path / "out"
    source_file = tmp_path / "source.txt"
    source_file.write_text("hello", encoding="utf-8")

    with zipfile.ZipFile(zip_path, mode="w") as archive:
        archive.write(source_file, arcname="nested/source.txt")

    with disable_run_logger():
        extract_zip(zip_path, destination)

    assert (destination / "nested/source.txt").read_text(encoding="utf-8") == "hello"


# Keep this success-path regression because a dedicated corrupt-archive test
# would not catch accidental breakage in the normal extraction flow.
@pytest.mark.allow_no_new_coverage
def test_extract_zip_raises_for_corrupt_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zip extraction should fail before extracting when integrity checks fail."""
    zip_path = tmp_path / "archive.zip"
    destination = tmp_path / "out"
    zip_path.write_bytes(b"placeholder")

    class _CorruptArchive:
        def __init__(self, opened_path: Path) -> None:
            assert opened_path == zip_path

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool:
            del exc_type, exc, traceback
            return False

        def testzip(self) -> str:
            return "nested/source.txt"

        def extractall(self, _dst_dir: Path) -> None:
            msg = "extractall should not run for corrupt archives"
            raise AssertionError(msg)

    monkeypatch.setattr("anomalog.io_utils.zipfile.ZipFile", _CorruptArchive)

    # Keep this failure-path regression because it proves extraction aborts
    # before `extractall`, even though the coverage hook does not credit it.
    with (
        disable_run_logger(),
        pytest.raises(
            zipfile.BadZipFile,
            match=r"Corrupt file in zip: nested/source\.txt",
        ),
    ):
        extract_zip(zip_path, destination)

    assert not destination.exists()
