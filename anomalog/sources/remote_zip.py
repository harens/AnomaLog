"""Remote ZIP dataset source with progress reporting and checksum verification."""

import tarfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import ClassVar
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import urlretrieve

from prefect.logging import get_run_logger
from rich.progress import (
    Progress,
    TaskID,
)

from anomalog.cache import asset_from_local_path, materialize
from anomalog.io_utils import (
    extract_zip,
    make_bounded_progress,
    verify_md5,
)
from anomalog.sources.contracts import DatasetSource

_ALLOWED_SCHEMES = {"https", "http"}


@dataclass
class _DownloadProgress:
    """Track download progress for callback-based reporting.

    Attributes:
        task_id (TaskID | None): Rich progress task id once the transfer starts.
        last_downloaded (int): Last byte count reported to the progress bar.
        total (int | None): Expected total download size in bytes when known.
    """

    task_id: TaskID | None = None
    last_downloaded: int = 0
    total: int | None = None


@dataclass(frozen=True)
class RemoteZipSource(DatasetSource):
    """Download a dataset zip from a remote URL and extract it locally.

    Attributes:
        name (ClassVar[str]): Registry/config name for the source.
        url (str): Absolute HTTP(S) URL of the dataset archive.
        md5_checksum (str | None): Optional checksum for the downloaded archive.
        raw_logs_relpath (Path | None): Optional raw-log path relative to the
            extracted dataset root.
    """

    name: ClassVar[str] = "remote_zip"
    url: str
    md5_checksum: str | None = None
    raw_logs_relpath: Path | None = None

    @staticmethod
    def _validate_remote_url(url: str) -> None:
        """Validate URL scheme and presence of network location.

        Args:
            url (str): Remote URL to validate.

        Raises:
            ValueError: If the URL is not an absolute HTTP(S) URL.
        """
        p = urlparse(url)
        if p.scheme not in _ALLOWED_SCHEMES:
            msg = f"Unsupported URL scheme: {p.scheme!r}"
            raise ValueError(msg)
        if not p.netloc:
            # rejects relative URLs like "example.com/file.zip" (no scheme/netloc)
            msg = "URL must be absolute (include scheme and host)"
            raise ValueError(msg)

    def _archive_is_tarball(self) -> bool:
        """Return whether the remote archive should be unpacked as tar.gz.

        Returns:
            bool: True when the remote URL path ends with a tarball suffix.
        """
        return urlparse(self.url).path.endswith((".tar.gz", ".tgz"))

    def materialise(
        self,
        *,
        dst_dir: Path,
    ) -> Path:
        """Fetch, checksum, and extract the dataset into dst_dir.

        Args:
            dst_dir (Path): Destination directory for the extracted dataset.

        Returns:
            Path: Extracted dataset root directory.
        """
        dataset_name = dst_dir.name
        root_dir = dst_dir.parent
        zip_path = dst_dir.with_suffix(".zip")

        if dst_dir.exists():
            logger = get_run_logger()
            logger.info("%s dataset already available at %s", dataset_name, dst_dir)
            return dst_dir

        root_dir.mkdir(parents=True, exist_ok=True)

        download_dataset_task = materialize(
            dst_dir,
            asset_deps=[asset_from_local_path(dst_dir)],
            retries=3,
            retry_delay_seconds=[2, 5, 15],
        )(self._download_dataset)

        download_dataset_task(zip_path)
        return dst_dir

    def _download_dataset(
        self,
        zip_path: Path,
        progress_factory: Callable[[], Progress] = make_bounded_progress,
    ) -> None:
        """Download the dataset archive with a progress bar and verify checksum.

        Args:
            zip_path (Path): Local temporary path for the downloaded zip archive.
            progress_factory (Callable[[], Progress]): Factory for the download
                progress bar implementation.
        """
        logger = get_run_logger()
        dataset_name = zip_path.stem
        logger.info("Starting download of %s dataset from %s", dataset_name, self.url)

        self._validate_remote_url(self.url)
        self._download_zip(zip_path, progress_factory)
        self._finalise_download(zip_path)

    def _download_zip(
        self,
        zip_path: Path,
        progress_factory: Callable[[], Progress],
    ) -> None:
        """Download the archive and clean up partial files on failure.

        Args:
            zip_path (Path): Local temporary path for the downloaded zip archive.
            progress_factory (Callable[[], Progress]): Factory for the download
                progress bar implementation.

        Raises:
            HTTPError: If the download fails with an HTTP error.
            OSError: If the temporary download cannot be promoted or removed.
            KeyboardInterrupt: If the download is interrupted by the user.
        """
        logger = get_run_logger()
        dataset_name = zip_path.stem
        state = _DownloadProgress()
        temp_zip_path = zip_path.with_name(f"{zip_path.name}.part")

        def _remove_partial(path: Path) -> None:
            if path.exists():
                path.unlink()

        try:
            with progress_factory() as pbar:
                urlretrieve(  # noqa: S310 - Validation is done in _validate_remote_url
                    self.url,
                    temp_zip_path,
                    reporthook=self._make_progress_hook(
                        dataset_name=dataset_name,
                        progress=pbar,
                        state=state,
                    ),
                )
        except HTTPError as exc:
            if exc.code == HTTPStatus.SERVICE_UNAVAILABLE:
                logger.warning(
                    "Download failed with %s %s for %s; will retry",
                    HTTPStatus.SERVICE_UNAVAILABLE.value,
                    HTTPStatus.SERVICE_UNAVAILABLE.phrase,
                    dataset_name,
                )
            else:
                logger.exception(
                    "HTTP error %s while downloading %s: %s",
                    exc.code,
                    dataset_name,
                    exc.reason,
                )
            _remove_partial(temp_zip_path)
            _remove_partial(zip_path)
            raise

        except KeyboardInterrupt:
            logger.warning("Download cancelled by user")
            _remove_partial(temp_zip_path)
            _remove_partial(zip_path)
            raise

        try:
            temp_zip_path.replace(zip_path)
        except OSError:
            _remove_partial(temp_zip_path)
            raise

    @staticmethod
    def _make_progress_hook(
        *,
        dataset_name: str,
        progress: Progress,
        state: _DownloadProgress,
    ) -> Callable[[int, int, int], None]:
        """Create a `urlretrieve` callback that advances the download bar.

        Args:
            dataset_name (str): Dataset name used in the progress description.
            progress (Progress): Active Rich progress display.
            state (_DownloadProgress): Mutable download bookkeeping shared
                across progress callbacks.

        Returns:
            Callable[[int, int, int], None]: Callback passed to `urlretrieve`
                for reporting bytes transferred.
        """

        def show_progress(
            block_num: int,
            block_size: int,
            total_size: int,
        ) -> None:
            if state.task_id is None:
                state.total = total_size if total_size and total_size > 0 else None
                state.task_id = progress.add_task(
                    f"Downloading {dataset_name} dataset",
                    total=state.total,
                )

            downloaded = block_num * block_size
            if state.total is not None:
                downloaded = min(downloaded, state.total)

            advance = downloaded - state.last_downloaded
            if advance > 0:
                progress.update(state.task_id, advance=advance)
                state.last_downloaded = downloaded

        return show_progress

    def _finalise_download(self, zip_path: Path) -> None:
        """Verify, extract, and remove a successfully downloaded archive.

        Args:
            zip_path (Path): Local path of the successfully downloaded archive.

        Raises:
            OSError: If the archive cannot be cleaned up.
            ValueError: If checksum verification fails.
            zipfile.BadZipFile: If the archive is not a valid zip file and the
                remote URL does not identify a tarball.
            tarfile.TarError: If tarball extraction fails.
        """
        try:
            if self.md5_checksum is not None:
                verify_md5(zip_path, self.md5_checksum)
            self._extract_archive(zip_path)
        except (OSError, ValueError, zipfile.BadZipFile, tarfile.TarError):
            if zip_path.exists():
                zip_path.unlink()
            raise

        logger = get_run_logger()
        logger.info("Removing zip file %s", zip_path)
        zip_path.unlink()

    def _extract_archive(self, zip_path: Path) -> None:
        """Extract a downloaded archive, falling back to tarball handling.

        Args:
            zip_path (Path): Local path of the downloaded archive.

        Raises:
            zipfile.BadZipFile: If the archive is neither a zip nor a tarball.
        """
        try:
            extract_zip(zip_path, zip_path.with_suffix(""))
        except zipfile.BadZipFile:
            if not self._archive_is_tarball():
                raise
            self._extract_tarball(zip_path)

    @staticmethod
    def _extract_tarball(archive_path: Path) -> None:
        """Extract a tarball archive into the matching dataset directory.

        Args:
            archive_path (Path): Local path of the downloaded tarball.
        """
        dst_dir = archive_path.with_suffix("")
        dst_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                archive.extract(member, dst_dir)
