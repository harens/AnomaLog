import hashlib
import logging
import zipfile
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
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


def verify_md5(file_path: Path, expected_hex: MD5Hex) -> None:
    logger.info("Verifying MD5 checksum for %s", file_path)
    file_size = file_path.stat().st_size

    hash_md5 = hashlib.md5()
    CHUNK_SIZE = 4 * 1024 * 1024

    with make_progress() as progress:
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
        z.extractall(dst_dir)


@dataclass
class _DownloadProgress:
    task_id: TaskID | None = None
    last_downloaded: int = 0
    total: int | None = None


@dataclass(frozen=True)
class Dataset(ABC):
    url: URL
    md5_checksum: MD5Hex
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

        extract_zip(self.zip_path, self.root_dir)

        logger.info("Removing zip file %s", self.zip_path)
        self.zip_path.unlink()

        return self.extracted_path

    def _download_dataset(self) -> None:
        logger.info("Downloading %s from %s", self.name, self.url)

        state = _DownloadProgress()

        try:
            with make_progress() as pbar:

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


class HDFS(Dataset):
    def __init__(self) -> None:
        super().__init__(
            url="https://zenodo.org/records/8275861/files/HDFS.zip",
            md5_checksum="f23880dd4938379a535ab71a8d27a798",
        )
