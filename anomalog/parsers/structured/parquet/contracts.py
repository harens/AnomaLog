"""Contracts for worker jobs writing structured lines to Parquet."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WriterJobCtx:
    """Context passed to a worker handling a partition of the input file."""

    part_id: int
    raw_input_path: Path
    start: int
    end: int
    parquet_out_dir: Path


@dataclass(frozen=True, slots=True)
class WriterResult:
    """Summary of a worker's Parquet write results."""

    part_id: int
    out_path: Path
    n_read: int
    n_parsed: int
    start: int
    end: int
    has_anomaly: bool
