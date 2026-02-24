"""Protocols for dataset materialisation sources."""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class DatasetSource(Protocol):
    """Download or copy a dataset into the given directory and return the path."""

    def materialise(self, dst_dir: Path) -> Path:
        """Ensure dataset exists under dst_dir, returning the root path."""
