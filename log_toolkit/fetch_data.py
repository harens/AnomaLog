import hashlib
import logging
import zipfile
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from urllib.request import urlretrieve

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from log_toolkit.type_hints import URL, MD5Hex


def make_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        transient=True,
    )


logger = logging.getLogger(__name__)


def verify_md5(
    file_path: Path,
    expected_hex: MD5Hex,
    progress_factory: Callable[[], Progress] = make_progress,
) -> None:
    logger.info("Verifying MD5 checksum for %s", file_path)
    file_size = file_path.stat().st_size

    hash_md5 = hashlib.md5()
    CHUNK_SIZE = 4 * 1024 * 1024

    with progress_factory() as progress:
        task = progress.add_task(
            f"Verifying {file_path.name}",
            total=file_size,
        )

        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                hash_md5.update(chunk)
                progress.update(task, advance=len(chunk))

    file_md5 = hash_md5.hexdigest()
    if file_md5 != expected_hex:
        raise ValueError(
            f"MD5 checksum mismatch for {file_path}: "
            f"expected {expected_hex}, got {file_md5}"
        )


def extract_zip(zip_path: Path, dst_dir: Path) -> None:
    logger.info("Extracting %s to %s", zip_path, dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            raise zipfile.BadZipFile(f"Corrupt file in zip: {bad}")
        z.extractall(dst_dir)


@dataclass
class _DownloadProgress:
    task_id: TaskID | None = None
    last_downloaded: int = 0
    total: int | None = None


@dataclass(frozen=True)
class Dataset(ABC):
    url: ClassVar[URL]
    md5_checksum: ClassVar[MD5Hex]
    extracted_anomaly_labels_path: ClassVar[Path]
    extracted_raw_logs_path: ClassVar[Path]
    root_dir: Path = Path("data")

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def zip_path(self) -> Path:
        return self.root_dir / f"{self.name}.zip"

    @property
    def extracted_path(self) -> Path:
        return self.root_dir / self.name

    def fetch(self) -> Path:
        if self.extracted_path.exists():
            logger.info(
                "%s dataset already available at %s", self.name, self.extracted_path
            )
            return self.extracted_path

        self.root_dir.mkdir(parents=True, exist_ok=True)

        self._download_dataset()

        verify_md5(self.zip_path, self.md5_checksum)

        extract_zip(self.zip_path, self.extracted_path)

        logger.info("Removing zip file %s", self.zip_path)
        self.zip_path.unlink()

        return self.extracted_path

    # TODO: Neatly handle urllib.error.HTTPError: HTTP Error 503: Service Unavailable
    def _download_dataset(
        self, progress_factory: Callable[[], Progress] = make_progress
    ) -> None:
        logger.info("Downloading %s from %s", self.name, self.url)

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
                            f"Downloading {self.name} dataset",
                            total=state.total,
                        )

                    downloaded = block_num * block_size
                    if state.total is not None:
                        downloaded = min(downloaded, state.total)

                    advance = downloaded - state.last_downloaded
                    if advance > 0:
                        pbar.update(state.task_id, advance=advance)
                        state.last_downloaded = downloaded

                urlretrieve(self.url, self.zip_path, reporthook=show_progress)

        except KeyboardInterrupt:
            logger.warning("Download cancelled by user")
            if self.zip_path.exists():
                self.zip_path.unlink()  # remove partial file
            raise


# See LogHub: https://zenodo.org/records/8196385
# Originally tried using LogHub-2.0 (https://zenodo.org/record/8275861),
# but HDFS doesn't seem to be annotated
class HDFS_V1(Dataset):
    url = "https://zenodo.org/records/8196385/files/HDFS_v1.zip"
    md5_checksum = "76a24b4d9a6164d543fb275f89773260"
    extracted_anomaly_labels_path = Path("preprocessed/anomaly_label.csv")
    extracted_raw_logs_path = Path("HDFS.log")
