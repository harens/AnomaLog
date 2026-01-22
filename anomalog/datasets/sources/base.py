from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class DatasetSource(Protocol):
    def materialise(self, dst_dir: Path) -> Path: ...
