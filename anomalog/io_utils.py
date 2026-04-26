"""Input/output helpers for progress reporting and dataset integrity checks."""

import hashlib
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

from prefect.exceptions import MissingContextError
from prefect.logging import get_run_logger
from prefect.logging.configuration import (
    DEFAULT_LOGGING_SETTINGS_PATH,
    load_logging_config,
)
from prefect.logging.highlighters import PrefectConsoleHighlighter
from prefect.settings import PREFECT_LOGGING_COLORS, PREFECT_LOGGING_MARKUP
from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.theme import Theme

_PREFECT_LOGGING_CONFIG = load_logging_config(DEFAULT_LOGGING_SETTINGS_PATH)


def _build_shared_console() -> Console:
    """Create the Rich console shared by progress bars and experiment logs.

    Returns:
        Console: Shared stderr-backed console using Prefect's console theme.
    """
    handler_config = _PREFECT_LOGGING_CONFIG["handlers"]["console"]
    if PREFECT_LOGGING_COLORS.value():
        highlighter = PrefectConsoleHighlighter()
        theme = Theme(handler_config.get("styles"), inherit=True)
    else:
        highlighter = NullHighlighter()
        theme = None
    return Console(
        file=sys.stderr,
        highlighter=highlighter,
        theme=theme,
        markup=PREFECT_LOGGING_MARKUP.value(),
    )


_SHARED_CONSOLE = _build_shared_console()


def get_shared_console() -> Console:
    """Return the Rich console shared by progress bars and experiment logs.

    Returns:
        Console: Shared stderr-backed Rich console.
    """
    return _SHARED_CONSOLE


def make_bounded_progress() -> Progress:
    """Create a progress bar suitable for bounded downloads.

    Returns:
        Progress: Rich progress instance configured for bounded transfers.
    """
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
        console=get_shared_console(),
    )


def make_spinner_progress(unit: str = "lines processed") -> Progress:
    """Create a spinner-style progress display for streaming tasks.

    Args:
        unit (str): Unit label shown beside the processed item count.

    Returns:
        Progress: Rich progress instance configured for streaming workloads.
    """
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        SpinnerColumn(),
        TextColumn(f"{{task.completed:,}} {unit}"),
        "•",
        TimeElapsedColumn(),
        console=get_shared_console(),
    )


def make_count_progress(unit: str | None = None) -> Progress:
    """Create a progress bar suitable for bounded count-based work.

    Args:
        unit (str | None): Optional unit label shown after the bounded item
            count.

    Returns:
        Progress: Rich progress instance configured for count-based tasks.
    """
    columns: list[str | ProgressColumn] = [
        TextColumn("[bold blue]{task.description}"),
        SpinnerColumn(),
        BarColumn(),
        TaskProgressColumn(),
        "•",
        MofNCompleteColumn(),
    ]
    if unit is not None:
        columns.append(TextColumn(unit))
    columns.extend(
        (
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
        ),
    )
    return Progress(*columns, console=get_shared_console())


def verify_md5(
    file_path: Path,
    expected_hex: str,
    progress_factory: Callable[[], Progress] = make_bounded_progress,
) -> None:
    """Validate a file's MD5 checksum, raising on mismatch.

    Args:
        file_path (Path): File whose checksum should be verified.
        expected_hex (str): Expected lowercase hexadecimal MD5 digest.
        progress_factory (Callable[[], Progress]): Factory for the progress
            display used while reading the file.

    Examples:
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

    Raises:
        ValueError: If the computed checksum does not match `expected_hex`.
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
    """Extract a zip file to a destination directory after integrity check.

    Args:
        zip_path (Path): Archive path to extract.
        dst_dir (Path): Destination directory for extracted contents.

    Raises:
        zipfile.BadZipFile: If the archive fails the integrity check.
    """
    logger = get_run_logger()
    logger.info("Extracting %s to %s", zip_path, dst_dir)

    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            msg = f"Corrupt file in zip: {bad}"
            raise zipfile.BadZipFile(msg)
        dst_dir.mkdir(parents=True, exist_ok=True)
        z.extractall(dst_dir)
