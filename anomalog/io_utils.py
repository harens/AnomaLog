"""Input/output helpers for progress reporting and dataset integrity checks."""

import hashlib
import zipfile
from collections.abc import Callable
from pathlib import Path

from prefect.exceptions import MissingContextError
from prefect.logging import get_run_logger
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


def make_bounded_progress() -> Progress:
    """Create a progress bar suitable for bounded downloads."""
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
    )


def make_spinner_progress(unit: str = "lines processed") -> Progress:
    """Create a spinner-style progress display for streaming tasks."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        SpinnerColumn(),
        TextColumn(f"{{task.completed:,}} {unit}"),
        "•",
        TimeElapsedColumn(),
    )


def verify_md5(
    file_path: Path,
    expected_hex: MD5Hex,
    progress_factory: Callable[[], Progress] = make_bounded_progress,
) -> None:
    """Validate a file's MD5 checksum, raising on mismatch.

    >>> tmp = Path("/tmp/test_md5.bin")
    >>> _ = tmp.write_bytes(b"abc")
    >>> class _Noop:
    ...     def __enter__(self): return self
    ...     def __exit__(self, *args): return False
    ...     def add_task(self, *_, **__): return 0
    ...     def update(self, *_, **__): return None
    >>> verify_md5(
    ...     tmp,
    ...     hashlib.md5(b"abc").hexdigest(),
    ...     progress_factory=_Noop,
    ... )
    >>> tmp.unlink()
    """
    try:
        logger = get_run_logger()
        logger.info(
            "Verifying MD5 checksum for %s against expected %s",
            file_path,
            expected_hex,
        )
    except MissingContextError:
        pass

    file_size = file_path.stat().st_size

    hash_md5 = hashlib.md5()  # noqa: S324 - MD5 is used by default for zenodo datasets
    chunk_size = 4 * 1024 * 1024

    with progress_factory() as progress:
        task = progress.add_task(
            f"Verifying {file_path.name}",
            total=file_size,
        )

        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
                progress.update(task, advance=len(chunk))

    file_md5 = hash_md5.hexdigest()
    if file_md5 != expected_hex:
        msg = (
            f"MD5 checksum mismatch for {file_path}: "
            f"expected {expected_hex}, got {file_md5}"
        )
        raise ValueError(
            msg,
        )


def extract_zip(zip_path: Path, dst_dir: Path) -> None:
    """Extract a zip file to a destination directory after integrity check."""
    logger = get_run_logger()
    logger.info("Extracting %s to %s", zip_path, dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            msg = f"Corrupt file in zip: {bad}"
            raise zipfile.BadZipFile(msg)
        z.extractall(dst_dir)
