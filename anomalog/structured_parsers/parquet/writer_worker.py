from __future__ import annotations

import itertools
import shutil
from dataclasses import asdict, dataclass
from hashlib import blake2s
from pathlib import Path  # noqa: TC003 - used at runtime for file IO
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from prefect.logging import get_run_logger

from anomalog.structured_parsers.contracts import ANOMALOUS_FIELD, StructuredParser

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(slots=True)
class WriterConfig:
    buckets: int = 32
    batch_rows: int = 400_000
    max_rows_per_file: int = 5_000_000
    max_rows_per_group: int = 256_000
    max_open_files: int = 128
    log_every_rows: int = 500_000
    max_partitions: int = 8_192


ENTITY_BUCKET_FIELD = "entity_bucket"


def _stable_bucket(entity_id: str, *, buckets: int) -> int:
    """Stable, deterministic hash bucket for an entity ID."""

    digest = blake2s(entity_id.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % buckets


def _iter_record_batches(
    raw_input_path: Path,
    parser: StructuredParser,
    *,
    cfg: WriterConfig,
) -> Generator[pa.RecordBatch, None, None]:
    logger = get_run_logger()
    rows: list[dict] = []
    total_rows = 0
    with raw_input_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f):
            rec = parser.parse_line(
                raw_line.rstrip("\n").rstrip("\r"),
                line_order=line_no,
            )
            if rec is None:
                continue

            row_dict = asdict(rec)
            if rec.entity_id is not None:
                row_dict[ENTITY_BUCKET_FIELD] = _stable_bucket(
                    rec.entity_id,
                    buckets=cfg.buckets,
                )
            else:
                row_dict[ENTITY_BUCKET_FIELD] = None
            rows.append(row_dict)
            total_rows += 1

            if cfg.log_every_rows > 0 and total_rows % cfg.log_every_rows == 0:
                logger.info("Parsed %d structured rows so far", total_rows)

            if len(rows) >= cfg.batch_rows:
                logger.info(
                    "Emitting batch of %d rows (total parsed: %d)",
                    len(rows),
                    total_rows,
                )
                yield pa.RecordBatch.from_pylist(rows)
                rows = []

    if rows:
        yield pa.RecordBatch.from_pylist(rows)

    logger.info("Finished parsing %d structured rows", total_rows)


def extract_structured_components(
    *,
    raw_input_path: Path,
    parser: StructuredParser,
    parquet_out_dir: Path,
    config: WriterConfig | None = None,
) -> bool:
    """Parse raw logs and write a partitioned Parquet dataset.

    - Hive partitions on entity_id for fast pruning on entity lookups.
    - Order per file follows input order; pick large row groups to keep scans fast.
    """

    logger = get_run_logger()
    cfg = config or WriterConfig()

    raw_input_path = raw_input_path.resolve()
    parquet_out_dir = parquet_out_dir.resolve()

    if not raw_input_path.exists():
        msg = f"Input file does not exist: {raw_input_path}"
        raise FileNotFoundError(msg)

    if parquet_out_dir.exists():
        logger.info(
            "Output directory %s already exists; deleting for fresh write",
            parquet_out_dir,
        )
        shutil.rmtree(parquet_out_dir)
    parquet_out_dir.mkdir(parents=True, exist_ok=True)

    batch_iter = _iter_record_batches(
        raw_input_path=raw_input_path,
        parser=parser,
        cfg=cfg,
    )

    try:
        first_batch = next(batch_iter)
    except StopIteration:
        msg = "No structured lines produced; nothing to write"
        raise ValueError(msg) from None

    has_anomaly = False
    batches_emitted = 0

    pc_any = getattr(pc, "any")  # noqa: B009 - stubs miss these functions
    pc_equal = getattr(pc, "equal")  # noqa: B009 - stubs miss these functions

    def _tracking_batches() -> Generator[pa.RecordBatch, None, None]:
        nonlocal has_anomaly
        nonlocal batches_emitted
        for batch in itertools.chain((first_batch,), batch_iter):
            if ANOMALOUS_FIELD in batch.schema.names:
                col = batch.column(ANOMALOUS_FIELD)
                if pc_any(pc_equal(col, pa.scalar(1, col.type))).as_py():
                    has_anomaly = True

            batches_emitted += 1
            yield batch

    schema = first_batch.schema

    partitioning = ds.partitioning(
        schema=pa.schema(
            [
                pa.field(ENTITY_BUCKET_FIELD, pa.int32()),
            ],
        ),
        flavor="hive",
    )

    logger.info(
        "Starting parquet write to %s (buckets=%d)",
        parquet_out_dir,
        cfg.buckets,
    )

    ds.write_dataset(
        data=ds.Scanner.from_batches(_tracking_batches(), schema=schema),
        base_dir=parquet_out_dir,
        format="parquet",
        partitioning=partitioning,
        max_partitions=max(cfg.max_partitions, cfg.buckets),
        existing_data_behavior="delete_matching",
        use_threads=True,
        max_rows_per_file=cfg.max_rows_per_file,
        max_rows_per_group=cfg.max_rows_per_group,
        max_open_files=cfg.max_open_files,
    )

    logger.info(
        "Structured extraction complete: file=%s out=%s batches_written=%d",
        raw_input_path,
        parquet_out_dir,
        batches_emitted,
    )

    return has_anomaly
