import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

from rich.progress import (
    Progress,
    TaskID,
)

from anomalog.datasets.sources.base import DatasetSource
from anomalog.datasets.sources.io_utils import (
    extract_zip,
    make_progress,
    verify_md5,
)
from anomalog.type_hints import URL, MD5Hex

logger = logging.getLogger(__name__)


@dataclass
class _DownloadProgress:
    task_id: TaskID | None = None
    last_downloaded: int = 0
    total: int | None = None


@dataclass(frozen=True)
class RemoteZipSource(DatasetSource):
    url: URL
    md5_checksum: MD5Hex

    def materialise(self, dst_dir: Path) -> Path:
        dataset_name = dst_dir.name
        root_dir = dst_dir.parent
        zip_path = dst_dir.with_suffix(".zip")

        if dst_dir.exists():
            logger.info("%s dataset already available at %s", dataset_name, dst_dir)
            return dst_dir

        root_dir.mkdir(parents=True, exist_ok=True)

        self._download_dataset(dataset_name, zip_path)

        verify_md5(zip_path, self.md5_checksum)

        extract_zip(zip_path, dst_dir)

        logger.info("Removing zip file %s", zip_path)
        zip_path.unlink()

        return dst_dir

    # TODO: Neatly handle urllib.error.HTTPError: HTTP Error 503: Service Unavailable
    def _download_dataset(
        self,
        dataset_name: str,
        zip_path: Path,
        progress_factory: Callable[[], Progress] = make_progress,
    ) -> None:
        logger.info("Downloading %s from %s", dataset_name, self.url)

        state = _DownloadProgress()

        try:
            with progress_factory() as pbar:

                def show_progress(
                    block_num: int, block_size: int, total_size: int
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

                urlretrieve(self.url, zip_path, reporthook=show_progress)

        except KeyboardInterrupt:
            logger.warning("Download cancelled by user")
            if zip_path.exists():
                zip_path.unlink()  # remove partial file
            raise
