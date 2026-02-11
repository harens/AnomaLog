from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WriterJobCtx:
    part_id: int
    raw_input_path: Path
    start: int
    end: int
    parquet_out_dir: Path


@dataclass(frozen=True, slots=True)
class WriterResult:
    part_id: int
    out_path: Path
    n_read: int
    n_parsed: int
    start: int
    end: int
    has_anomaly: bool
