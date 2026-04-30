"""Worker routines to parse raw logs and write Parquet partitions."""

from __future__ import annotations

import itertools
import json
import shutil
from dataclasses import asdict, dataclass
from hashlib import blake2s
from pathlib import Path  # noqa: TC003 - used at runtime for file IO
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from prefect.logging import get_run_logger

from anomalog.parsers.structured.contracts import (
    ANOMALOUS_FIELD,
    ENTITY_FIELD,
    LINE_FIELD,
    TIMESTAMP_FIELD,
    UNTEMPLATED_FIELD,
    StructuredLine,
    StructuredParser,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(slots=True)
class WriterConfig:
    """Tuning parameters for converting raw logs to parquet.

    Attributes:
        buckets (int): Number of hash buckets for partitioning entity ids.
        batch_rows (int): Number of parsed rows to accumulate before emitting a
            record batch.
        max_rows_per_file (int): Maximum rows per parquet file.
        max_rows_per_group (int): Maximum rows per row group inside each file.
        max_open_files (int): Maximum parquet files kept open by the dataset
            writer at once.
        log_every_rows (int): Logging cadence while parsing raw input rows.
        max_partitions (int): Maximum distinct partition directories to create.
    """

    buckets: int = 32
    batch_rows: int = 400_000
    max_rows_per_file: int = 5_000_000
    max_rows_per_group: int = 256_000
    max_open_files: int = 128
    log_every_rows: int = 500_000
    max_partitions: int = 8_192


ENTITY_BUCKET_FIELD = "entity_bucket"
ENTITY_CHRONOLOGY_INDEX_FILENAME = "entity_chronology_index.jsonl"


@dataclass(frozen=True, slots=True, order=True)
class EntityChronologyKey:
    """Deterministic ordering metadata for one entity during materialisation.

    Attributes:
        first_timestamp_missing (int): `1` when the entity has no timestamp,
            otherwise `0`.
        first_timestamp_unix_ms (int): First timestamp observed for the entity,
            or `0` when none is present.
        first_line_order (int): Source-order tie-breaker for the entity.
        entity_id (str): Entity identifier for the chronology entry.
    """

    first_timestamp_missing: int
    first_timestamp_unix_ms: int
    first_line_order: int
    entity_id: str


STRUCTURED_BATCH_SCHEMA = pa.schema(
    [
        pa.field(TIMESTAMP_FIELD, pa.int64()),
        pa.field(ENTITY_FIELD, pa.string()),
        pa.field(UNTEMPLATED_FIELD, pa.string()),
        pa.field(ANOMALOUS_FIELD, pa.int64()),
        pa.field(LINE_FIELD, pa.int64()),
        pa.field(ENTITY_BUCKET_FIELD, pa.int32()),
    ],
)


def _stable_bucket(entity_id: str, *, buckets: int) -> int:
    """Stable, deterministic hash bucket for an entity ID.

    Args:
        entity_id (str): Entity identifier to hash.
        buckets (int): Number of hash buckets to map into.

    Examples:
        >>> _stable_bucket("foo", buckets=4) == _stable_bucket("foo", buckets=4)
        True
        >>> 0 <= _stable_bucket("bar", buckets=3) < 3
        True

    Returns:
        int: Stable hash bucket for the entity identifier.
    """
    digest = blake2s(entity_id.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % buckets


def _iter_record_batches(
    raw_input_path: Path,
    parser: StructuredParser,
    *,
    cfg: WriterConfig,
    entity_chronology: dict[str, EntityChronologyKey] | None = None,
) -> Generator[pa.RecordBatch, None, None]:
    """Stream record batches parsed from the raw log file.

    Args:
        raw_input_path (Path): Input raw log file to parse.
        parser (StructuredParser): Structured parser used for each raw line.
        cfg (WriterConfig): Batch and partitioning configuration for the writer.
        entity_chronology (dict[str, EntityChronologyKey] | None): Optional
            sidecar index to populate with each entity's first-seen order.

    Yields:
        pa.RecordBatch: Structured rows accumulated into parquet-ready batches.
    """
    logger = get_run_logger()
    rows: list[dict] = []
    total_rows = 0
    with raw_input_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f):
            base_rec = parser.parse_line(raw_line.rstrip("\n").rstrip("\r"))
            if base_rec is None:
                continue

            rec = StructuredLine.with_line_order(
                line_order=line_no,
                base=base_rec,
            )
            if (
                entity_chronology is not None
                and rec.entity_id is not None
                and rec.entity_id not in entity_chronology
            ):
                entity_chronology[rec.entity_id] = EntityChronologyKey(
                    first_timestamp_missing=1 if rec.timestamp_unix_ms is None else 0,
                    first_timestamp_unix_ms=0
                    if rec.timestamp_unix_ms is None
                    else int(rec.timestamp_unix_ms),
                    first_line_order=line_no,
                    entity_id=rec.entity_id,
                )
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
                yield pa.RecordBatch.from_pylist(rows, schema=STRUCTURED_BATCH_SCHEMA)
                rows = []

    if rows:
        yield pa.RecordBatch.from_pylist(rows, schema=STRUCTURED_BATCH_SCHEMA)

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
    - A tiny JSONL sidecar stores each entity's first-seen chronology key so
      readers do not need to re-derive entity ordering from the parquet rows.

    Args:
        raw_input_path (Path): Raw log file to parse.
        parser (StructuredParser): Structured parser used to parse each line.
        parquet_out_dir (Path): Output directory for the parquet dataset.
        config (WriterConfig | None): Optional writer configuration override.

    Returns:
        bool: `True` if at least one anomalous row is observed; otherwise `False`.

    Raises:
        FileNotFoundError: If `raw_input_path` does not exist.
        ValueError: If parsing produces no structured rows.
    """
    logger = get_run_logger()
    cfg = config or WriterConfig()
    entity_chronology: dict[str, EntityChronologyKey] = {}

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
        try:
            shutil.rmtree(parquet_out_dir)
        except FileNotFoundError:
            logger.info(
                "Output directory %s disappeared before cleanup completed",
                parquet_out_dir,
            )
    parquet_out_dir.mkdir(parents=True, exist_ok=True)

    batch_iter = _iter_record_batches(
        raw_input_path=raw_input_path,
        parser=parser,
        cfg=cfg,
        entity_chronology=entity_chronology,
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
        nonlocal has_anomaly, batches_emitted
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
        preserve_order=True,
        max_rows_per_file=cfg.max_rows_per_file,
        max_rows_per_group=cfg.max_rows_per_group,
        max_open_files=cfg.max_open_files,
    )

    _write_entity_chronology_index(
        parquet_out_dir=parquet_out_dir,
        chronology=entity_chronology,
    )

    logger.info(
        "Structured extraction complete: file=%s out=%s batches_written=%d",
        raw_input_path,
        parquet_out_dir,
        batches_emitted,
    )

    return has_anomaly


def _write_entity_chronology_index(
    *,
    parquet_out_dir: Path,
    chronology: dict[str, EntityChronologyKey],
) -> None:
    """Persist the entity chronology sidecar alongside the parquet dataset.

    Args:
        parquet_out_dir (Path): Output directory for the structured parquet
            dataset.
        chronology (dict[str, EntityChronologyKey]): First-seen chronology
            metadata keyed by entity id.
    """
    index_path = parquet_out_dir / ENTITY_CHRONOLOGY_INDEX_FILENAME
    ordered_entries = sorted(chronology.values())
    with index_path.open("w", encoding="utf-8") as handle:
        for entry in ordered_entries:
            handle.write(json.dumps(asdict(entry), separators=(",", ":")))
            handle.write("\n")
