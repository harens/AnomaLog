import hashlib
import logging
import zipfile
from collections.abc import Callable
from pathlib import Path

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from anomalog.type_hints import MD5Hex

logger = logging.getLogger(__name__)


def make_bounded_progress() -> Progress:
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


def make_spinner_progress(unit: str = "lines processed") -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        SpinnerColumn(),
        TextColumn(f"{{task.completed:,}} {unit}"),
        "•",
        TimeElapsedColumn(),
        transient=True,
    )


def verify_md5(
    file_path: Path,
    expected_hex: MD5Hex,
    progress_factory: Callable[[], Progress] = make_bounded_progress,
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
