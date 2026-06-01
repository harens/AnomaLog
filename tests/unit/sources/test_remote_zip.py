"""Tests for non-network `RemoteZipSource` branches."""

import tarfile
import zipfile
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest
from prefect.logging import disable_run_logger
from rich.progress import Progress

from anomalog.sources.remote_zip import RemoteZipSource

_REMOTE_ZIP_ARCHIVE_IS_TARBALL = vars(RemoteZipSource)["_archive_is_tarball"]
_REMOTE_ZIP_VALIDATE_REMOTE_URL = vars(RemoteZipSource)["_validate_remote_url"]
_REMOTE_ZIP_DOWNLOAD_DATASET = vars(RemoteZipSource)["_download_dataset"]


def test_remote_zip_source_materialise_short_circuits_existing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing extracted datasets are reused without attempting a download.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local dataset fixtures.
        monkeypatch (pytest.MonkeyPatch): Replaces runtime hooks so the download
            path would fail loudly if invoked.
    """
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


def test_remote_zip_source_validates_remote_urls_and_tarball_suffix() -> None:
    """Remote ZIP sources should reject unsupported URLs and detect tarballs."""
    zip_source = RemoteZipSource(url="https://example.com/data.zip")
    tar_source = RemoteZipSource(url="https://example.com/OpenStack.tar.gz")

    assert _REMOTE_ZIP_ARCHIVE_IS_TARBALL(zip_source) is False
    assert _REMOTE_ZIP_ARCHIVE_IS_TARBALL(tar_source) is True

    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        _REMOTE_ZIP_VALIDATE_REMOTE_URL("ftp://example.com/data.zip")

    with pytest.raises(ValueError, match="URL must be absolute"):
        _REMOTE_ZIP_VALIDATE_REMOTE_URL("https:///data.zip")


def test_remote_zip_source_materialise_downloads_and_extracts_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download path verifies the archive, extracts it, and removes the zip file.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local dataset fixtures.
        monkeypatch (pytest.MonkeyPatch): Replaces network and archive helpers so
            the non-network download path can be exercised deterministically.
    """
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


def test_remote_zip_source_materialise_downloads_and_extracts_tarball(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tarball-based remote sources should extract through the tar path.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local dataset fixtures.
        monkeypatch (pytest.MonkeyPatch): Replaces network and archive helpers so
            the tarball download path can be exercised deterministically.
    """
    dst_dir = tmp_path / "dataset"
    zip_path = dst_dir.with_suffix(".zip")
    source = RemoteZipSource(url="https://example.com/OpenStack.tar.gz")
    archive_path = tmp_path / "OpenStack.tar.gz"
    extracted: list[tuple[Path, Path]] = []

    with tarfile.open(archive_path, "w:gz") as archive:
        payload = tmp_path / "preprocessed" / "openstack_labelled_raw.log"
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_text("hello\n", encoding="utf-8")
        archive.add(
            payload,
            arcname="preprocessed/openstack_labelled_raw.log",
        )

    def _fake_urlretrieve(
        url: str,
        target: Path,
        reporthook: Callable[[int, int, int], None] | None = None,
    ) -> None:
        assert url == source.url
        target.write_bytes(archive_path.read_bytes())
        if reporthook is not None:
            reporthook(1, 4, 8)

    def _fail_extract_zip(_path: Path, _output_dir: Path) -> None:
        message = "tarball archive"
        raise zipfile.BadZipFile(message)

    def _record_extract_tarball(path: Path) -> None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        (dst_dir / "preprocessed" / "openstack_labelled_raw.log").parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        (dst_dir / "preprocessed" / "openstack_labelled_raw.log").write_text(
            "hello\n",
            encoding="utf-8",
        )
        extracted.append((path, dst_dir))

    monkeypatch.setattr("anomalog.sources.remote_zip.urlretrieve", _fake_urlretrieve)
    monkeypatch.setattr("anomalog.sources.remote_zip.extract_zip", _fail_extract_zip)
    monkeypatch.setattr(
        "anomalog.sources.remote_zip.RemoteZipSource._extract_tarball",
        staticmethod(_record_extract_tarball),
    )

    def _progress_factory() -> Progress:
        return Progress()

    with disable_run_logger():
        source._download_dataset(  # noqa: SLF001 - exercising the download side effect directly
            zip_path,
            progress_factory=_progress_factory,
        )

    assert extracted == [(zip_path, dst_dir)]
    assert dst_dir.is_dir()
    assert (dst_dir / "preprocessed/openstack_labelled_raw.log").is_file()
    assert not zip_path.exists()


def test_remote_zip_source_extracts_real_tarball_members(
    tmp_path: Path,
) -> None:
    """Tarball extraction should unpack real archive members to disk."""
    archive_path = tmp_path / "OpenStack.tar.gz"
    extracted_payload = tmp_path / "payload" / "preprocessed" / "demo.log"
    extracted_payload.parent.mkdir(parents=True, exist_ok=True)
    extracted_payload.write_text("hello\n", encoding="utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(extracted_payload, arcname="preprocessed/demo.log")

    source = RemoteZipSource(url="https://example.com/OpenStack.tar.gz")
    with disable_run_logger():
        source._extract_tarball(archive_path)  # noqa: SLF001 - regression coverage

    assert (archive_path.with_suffix("") / "preprocessed/demo.log").read_text(
        encoding="utf-8",
    ) == "hello\n"


def test_remote_zip_source_download_cleans_up_partial_archive_on_http_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP failures should not leave a partial archive behind.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local dataset fixtures.
        monkeypatch (pytest.MonkeyPatch): Replaces network and archive helpers so
            the failure path can be exercised deterministically.
    """
    dst_dir = tmp_path / "dataset"
    zip_path = dst_dir.with_suffix(".zip")
    source = RemoteZipSource(url="https://example.com/data.zip")

    def _fake_urlretrieve(
        _url: str,
        target: Path,
        reporthook: Callable[[int, int, int], None] | None = None,
    ) -> None:
        del reporthook
        target.write_text("partial-zip-bytes", encoding="utf-8")
        headers = Message()
        raise HTTPError(
            url=source.url,
            code=504,
            msg="Gateway Time-out",
            hdrs=headers,
            fp=None,
        )

    monkeypatch.setattr("anomalog.sources.remote_zip.urlretrieve", _fake_urlretrieve)

    with disable_run_logger(), pytest.raises(HTTPError):
        _REMOTE_ZIP_DOWNLOAD_DATASET(source, zip_path, progress_factory=Progress)

    assert not zip_path.exists()
    assert not zip_path.with_name(f"{zip_path.name}.part").exists()


def test_remote_zip_source_finalise_download_removes_bad_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corrupt archives should be removed if extraction fails.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local dataset fixtures.
        monkeypatch (pytest.MonkeyPatch): Replaces archive helpers so the
            corrupt-archive path can be exercised deterministically.
    """
    zip_path = tmp_path / "dataset.zip"
    zip_path.write_text("not-a-zip", encoding="utf-8")
    source = RemoteZipSource(url="https://example.com/data.zip")

    monkeypatch.setattr("anomalog.sources.remote_zip.verify_md5", lambda *_args: None)

    def _fake_extract_zip(_path: Path, _output_dir: Path) -> None:
        message = "bad archive"
        raise zipfile.BadZipFile(message)

    monkeypatch.setattr("anomalog.sources.remote_zip.extract_zip", _fake_extract_zip)

    with disable_run_logger(), pytest.raises(zipfile.BadZipFile):
        source._finalise_download(zip_path)  # noqa: SLF001 - regression coverage

    assert not zip_path.exists()


def test_remote_zip_source_rejects_missing_resolved_raw_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved raw log paths are validated after extraction.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local dataset fixtures.
        monkeypatch (pytest.MonkeyPatch): Replaces network and archive helpers so
            the missing-raw-log validation can be isolated.
    """
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


def test_remote_zip_source_warns_and_retries_on_service_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 503 failures should be treated as retryable download failures.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local dataset fixtures.
        monkeypatch (pytest.MonkeyPatch): Replaces network helpers so the
            retryable failure path can be exercised deterministically.
    """
    dst_dir = tmp_path / "dataset"
    zip_path = dst_dir.with_suffix(".zip")
    source = RemoteZipSource(url="https://example.com/data.zip")

    def _fake_urlretrieve(
        _url: str,
        target: Path,
        reporthook: Callable[[int, int, int], None] | None = None,
    ) -> None:
        del reporthook
        target.write_text("partial-zip-bytes", encoding="utf-8")
        headers = Message()
        raise HTTPError(
            url=source.url,
            code=503,
            msg="Service Unavailable",
            hdrs=headers,
            fp=None,
        )

    monkeypatch.setattr("anomalog.sources.remote_zip.urlretrieve", _fake_urlretrieve)

    with disable_run_logger(), pytest.raises(HTTPError):
        _REMOTE_ZIP_DOWNLOAD_DATASET(source, zip_path, progress_factory=Progress)

    assert not zip_path.exists()
    assert not zip_path.with_name(f"{zip_path.name}.part").exists()


def test_remote_zip_source_cleans_up_on_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupted downloads should not leave partial archives behind.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local dataset fixtures.
        monkeypatch (pytest.MonkeyPatch): Replaces network helpers so the
            cancellation path can be exercised deterministically.
    """
    dst_dir = tmp_path / "dataset"
    zip_path = dst_dir.with_suffix(".zip")
    source = RemoteZipSource(url="https://example.com/data.zip")

    def _fake_urlretrieve(
        _url: str,
        target: Path,
        reporthook: Callable[[int, int, int], None] | None = None,
    ) -> None:
        del reporthook
        target.write_text("partial-zip-bytes", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr("anomalog.sources.remote_zip.urlretrieve", _fake_urlretrieve)

    with disable_run_logger(), pytest.raises(KeyboardInterrupt):
        _REMOTE_ZIP_DOWNLOAD_DATASET(source, zip_path, progress_factory=Progress)

    assert not zip_path.exists()
    assert not zip_path.with_name(f"{zip_path.name}.part").exists()


def test_remote_zip_source_cleans_up_on_archive_promotion_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed archive promotion should remove the temporary download.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local dataset fixtures.
        monkeypatch (pytest.MonkeyPatch): Replaces network helpers so the
            promotion-failure path can be exercised deterministically.
    """
    dst_dir = tmp_path / "dataset"
    zip_path = dst_dir.with_suffix(".zip")
    source = RemoteZipSource(url="https://example.com/data.zip")

    def _fake_urlretrieve(
        _url: str,
        target: Path,
        reporthook: Callable[[int, int, int], None] | None = None,
    ) -> None:
        del reporthook
        target.write_text("zip-bytes", encoding="utf-8")

    monkeypatch.setattr("anomalog.sources.remote_zip.urlretrieve", _fake_urlretrieve)
    monkeypatch.setattr(
        Path,
        "replace",
        lambda _self, _target: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with disable_run_logger(), pytest.raises(OSError, match="replace failed"):
        _REMOTE_ZIP_DOWNLOAD_DATASET(source, zip_path, progress_factory=Progress)

    assert not zip_path.exists()
    assert not zip_path.with_name(f"{zip_path.name}.part").exists()
