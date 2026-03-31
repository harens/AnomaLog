"""Tests for non-network `RemoteZipSource` branches."""

from collections.abc import Callable
from pathlib import Path

import pytest
from prefect.logging import disable_run_logger
from rich.progress import Progress

from anomalog.sources.remote_zip import RemoteZipSource


def test_remote_zip_source_materialise_short_circuits_existing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing extracted datasets are reused without attempting a download."""
    dst_dir = tmp_path / "dataset"
    dst_dir.mkdir()
    (dst_dir / "demo.log").write_text("hello\n", encoding="utf-8")
    source = RemoteZipSource(
        url="https://example.com/data.zip",
        md5_checksum="d41d8cd98f00b204e9800998ecf8427e",
        raw_logs_relpath=Path("demo.log"),
    )
    msg = "download should not be scheduled when dst_dir exists"

    def _fail_if_called() -> None:
        raise AssertionError(msg)

    monkeypatch.setattr(
        "anomalog.sources.remote_zip.materialize",
        _fail_if_called,
    )

    with disable_run_logger():
        dataset_root = source.materialise(dst_dir=dst_dir)

    assert dataset_root == dst_dir
    assert source.raw_logs_path(dataset_name="demo", dataset_root=dataset_root) == (
        dst_dir / "demo.log"
    )


def test_remote_zip_source_materialise_downloads_and_extracts_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download path verifies the archive, extracts it, and removes the zip file."""
    dst_dir = tmp_path / "dataset"
    zip_path = dst_dir.with_suffix(".zip")
    source = RemoteZipSource(
        url="https://example.com/data.zip",
        md5_checksum="expected-md5",
        raw_logs_relpath=Path("demo.log"),
    )
    extracted: list[tuple[Path, Path]] = []
    verified: list[tuple[Path, str]] = []

    def _fake_urlretrieve(
        url: str,
        target: Path,
        reporthook: Callable[[int, int, int], None] | None = None,
    ) -> None:
        assert url == source.url
        target.write_text("zip-bytes", encoding="utf-8")
        if reporthook is not None:
            reporthook(1, 4, 8)
            reporthook(2, 4, 8)

    def _fake_verify_md5(path: Path, checksum: str) -> None:
        verified.append((path, checksum))

    def _fake_extract_zip(path: Path, output_dir: Path) -> None:
        extracted.append((path, output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "demo.log").write_text("hello\n", encoding="utf-8")

    monkeypatch.setattr("anomalog.sources.remote_zip.urlretrieve", _fake_urlretrieve)
    monkeypatch.setattr("anomalog.sources.remote_zip.verify_md5", _fake_verify_md5)
    monkeypatch.setattr("anomalog.sources.remote_zip.extract_zip", _fake_extract_zip)

    def _progress_factory() -> Progress:
        return Progress()

    with disable_run_logger():
        source._download_dataset(  # noqa: SLF001 - exercising the download side effect directly
            zip_path,
            progress_factory=_progress_factory,
        )

    assert verified == [(zip_path, "expected-md5")]
    assert extracted == [(zip_path, dst_dir)]
    assert dst_dir.is_dir()
    assert not zip_path.exists()


def test_remote_zip_source_rejects_missing_resolved_raw_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved raw log paths are validated after extraction."""
    dst_dir = tmp_path / "dataset"
    source = RemoteZipSource(
        url="https://example.com/data.zip",
        md5_checksum="expected-md5",
        raw_logs_relpath=Path("missing.log"),
    )

    def _fake_urlretrieve(
        _url: str,
        target: Path,
        reporthook: Callable[[int, int, int], None] | None = None,
    ) -> None:
        del reporthook
        target.write_text("zip-bytes", encoding="utf-8")

    monkeypatch.setattr("anomalog.sources.remote_zip.urlretrieve", _fake_urlretrieve)
    monkeypatch.setattr("anomalog.sources.remote_zip.verify_md5", lambda *_args: None)
    monkeypatch.setattr(
        "anomalog.sources.remote_zip.extract_zip",
        lambda _path, output_dir: output_dir.mkdir(parents=True, exist_ok=True),
    )

    with disable_run_logger():
        dataset_root = source.materialise(dst_dir=dst_dir)

    with pytest.raises(FileNotFoundError):
        source.raw_logs_path(dataset_name="demo", dataset_root=dataset_root)
