from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
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
from anomalog.sources import DatasetSource
from anomalog.type_hints import URL, MD5Hex

_ALLOWED_SCHEMES = {"https", "http"}


@dataclass
class _DownloadProgress:
    task_id: TaskID | None = None
    last_downloaded: int = 0
    total: int | None = None


@dataclass(frozen=True)
class RemoteZipSource(DatasetSource):
    url: URL
    md5_checksum: MD5Hex

    @staticmethod
    def _validate_remote_url(url: URL) -> None:
        p = urlparse(url)
        if p.scheme not in _ALLOWED_SCHEMES:
            msg = f"Unsupported URL scheme: {p.scheme!r}"
            raise ValueError(msg)
        if not p.netloc:
            # rejects relative URLs like "example.com/file.zip" (no scheme/netloc)
            msg = "URL must be absolute (include scheme and host)"
            raise ValueError(msg)

    def materialise(self, dst_dir: Path) -> Path:
        dataset_name = dst_dir.name
        root_dir = dst_dir.parent
        zip_path = dst_dir.with_suffix(".zip")

        if dst_dir.exists():
            logger = get_run_logger()
            logger.info("%s dataset already available at %s", dataset_name, dst_dir)
            return dst_dir

        root_dir.mkdir(parents=True, exist_ok=True)

        download_dataset_task = materialize(
            asset_from_local_path(dst_dir),
            retries=3,
            retry_delay_seconds=[2, 5, 15],
        )(self._download_dataset)

        download_dataset_task(dataset_name, zip_path)

        return dst_dir

    def _download_dataset(
        self,
        dataset_name: str,
        zip_path: Path,
        progress_factory: Callable[[], Progress] = make_bounded_progress,
    ) -> None:
        logger = get_run_logger()
        logger.info("Starting download of %s dataset from %s", dataset_name, self.url)

        state = _DownloadProgress()
        self._validate_remote_url(self.url)

        try:
            with progress_factory() as pbar:

                def show_progress(
                    block_num: int,
                    block_size: int,
                    total_size: int,
                ) -> None:
                    if state.task_id is None:
                        state.total = (
                            total_size if total_size and total_size > 0 else None
                        )
                        state.task_id = pbar.add_task(
                            f"Downloading {dataset_name} dataset",
                            total=state.total,
                        )

                    downloaded = block_num * block_size
                    if state.total is not None:
                        downloaded = min(downloaded, state.total)

                    advance = downloaded - state.last_downloaded
                    if advance > 0:
                        pbar.update(state.task_id, advance=advance)
                        state.last_downloaded = downloaded

                urlretrieve(self.url, zip_path, reporthook=show_progress)  # noqa: S310 - Validation is done in _validate_remote_url

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
            if zip_path.exists():
                zip_path.unlink()  # remove partial file
            raise

        except KeyboardInterrupt:
            logger.warning("Download cancelled by user")
            if zip_path.exists():
                zip_path.unlink()  # remove partial file
            raise

        verify_md5(zip_path, self.md5_checksum)
        extract_zip(zip_path, zip_path.parent)
        logger.info("Removing zip file %s", zip_path)
        zip_path.unlink()
