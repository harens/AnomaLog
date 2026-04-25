"""Contracts for worker jobs writing structured lines to Parquet."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WriterJobCtx:
    """Context passed to a worker handling a partition of the input file.

    Attributes:
        part_id (int): Stable partition identifier for the worker job.
        raw_input_path (Path): Raw log file being partition-read.
        start (int): Inclusive byte offset where this partition begins.
        end (int): Exclusive byte offset where this partition stops.
        parquet_out_dir (Path): Destination directory for partition parquet
            output.
    """

    part_id: int
    raw_input_path: Path
    start: int
    end: int
    parquet_out_dir: Path


@dataclass(frozen=True, slots=True)
class WriterResult:
    """Summary of a worker's Parquet write results.

    Attributes:
        part_id (int): Partition identifier matching the originating worker job.
        out_path (Path): Parquet file written for the partition.
        n_read (int): Number of raw lines read from the partition.
        n_parsed (int): Number of structured rows successfully emitted.
        start (int): Inclusive byte offset where the partition began.
        end (int): Exclusive byte offset where the partition ended.
        has_anomaly (bool): Whether any emitted row carried an anomalous label.
    """

    part_id: int
    out_path: Path
    n_read: int
    n_parsed: int
    start: int
    end: int
    has_anomaly: bool
