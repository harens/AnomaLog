"""Worker routines to parse raw logs and write Parquet partitions."""

from __future__ import annotations

import itertools
import json
import shutil
from dataclasses import asdict, dataclass
from hashlib import blake2s
from pathlib import Path  # noqa: TC003 - used at runtime for file IO
from time import perf_counter
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.dataset as ds
from prefect.logging import get_run_logger

from anomalog.parsers.structured.contracts import (
    ANOMALOUS_FIELD,
    ENTITY_FIELD,
    LINE_FIELD,
    RAW_PARAMETERS_FIELD,
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
ENTITY_COUNT_FILENAME = "entity_count.json"
INLINE_LABEL_CACHE_FILENAME = "inline_label_cache.jsonl"
_STRUCTURED_PROGRESS_SILENCE_SECONDS = 60


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


@dataclass(slots=True)
class _StructuredComponentsWriteState:
    """Mutable bookkeeping for the final structured-write summary.

    Attributes:
        parquet_out_dir (Path): Output directory for the structured parquet
            dataset.
        entity_chronology (dict[str, EntityChronologyKey]): First-seen
            chronology metadata keyed by entity id.
        has_inline_labels (bool): Whether the structured rows exposed any
            inline anomaly labels.
        inline_label_entries (list[dict[str, object]]): Sparse anomaly labels
            keyed by line order and entity id.
        raw_input_path (Path): Raw log file that was parsed into the dataset.
        batches_emitted (int): Number of record batches written to parquet.
        started_at (float): Timestamp at which the structured extraction began.
    """

    parquet_out_dir: Path
    entity_chronology: dict[str, EntityChronologyKey]
    has_inline_labels: bool
    inline_label_entries: list[dict[str, object]]
    raw_input_path: Path
    batches_emitted: int
    started_at: float


STRUCTURED_BATCH_SCHEMA = pa.schema(
    [
        pa.field(TIMESTAMP_FIELD, pa.int64()),
        pa.field(ENTITY_FIELD, pa.string()),
        pa.field(UNTEMPLATED_FIELD, pa.string()),
        pa.field(ANOMALOUS_FIELD, pa.int64()),
        pa.field(LINE_FIELD, pa.int64()),
        pa.field(RAW_PARAMETERS_FIELD, pa.list_(pa.string())),
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
    started_at = perf_counter()
    last_progress_at = started_at
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

            elif (
                perf_counter() - last_progress_at
                >= _STRUCTURED_PROGRESS_SILENCE_SECONDS
            ):
                elapsed = perf_counter() - started_at
                rate = total_rows / elapsed if elapsed > 0 else 0.0
                logger.info(
                    (
                        "Parsed %d structured rows so far (raw_line=%d "
                        "elapsed=%.3fs rows_per_sec=%.1f sample=%r)"
                    ),
                    total_rows,
                    line_no + 1,
                    elapsed,
                    rate,
                    raw_line.rstrip("\n\r"),
                )
                last_progress_at = perf_counter()

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
        bool: `True` when the structured rows carry an inline anomaly label
            field, even if every label is normal (`0`). `False` means the parser
            did not populate inline anomaly labels at all.

    Raises:
        FileNotFoundError: If `raw_input_path` does not exist.
        ValueError: If parsing produces no structured rows.
    """
    logger = get_run_logger()
    cfg = config or WriterConfig()
    entity_chronology: dict[str, EntityChronologyKey] = {}

    raw_input_path = raw_input_path.resolve()
    parquet_out_dir = parquet_out_dir.resolve()
    started_at = perf_counter()
    stage_started_at = started_at

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

    has_inline_labels = False
    batches_emitted = 0
    inline_label_entries: list[dict[str, object]] = []

    def _tracking_batches() -> Generator[pa.RecordBatch, None, None]:
        nonlocal has_inline_labels, batches_emitted
        for batch in itertools.chain((first_batch,), batch_iter):
            has_inline_labels = _track_inline_label_entries(
                batch=batch,
                has_inline_labels=has_inline_labels,
                inline_label_entries=inline_label_entries,
            )

            batches_emitted += 1
            yield batch

    logger.info(
        "Starting parquet write to %s (buckets=%d)",
        parquet_out_dir,
        cfg.buckets,
    )

    ds.write_dataset(
        data=ds.Scanner.from_batches(
            _tracking_batches(),
            schema=first_batch.schema,
        ),
        base_dir=parquet_out_dir,
        format="parquet",
        partitioning=ds.partitioning(
            schema=pa.schema(
                [
                    pa.field(ENTITY_BUCKET_FIELD, pa.int32()),
                ],
            ),
            flavor="hive",
        ),
        max_partitions=max(cfg.max_partitions, cfg.buckets),
        existing_data_behavior="delete_matching",
        use_threads=True,
        preserve_order=True,
        max_rows_per_file=cfg.max_rows_per_file,
        max_rows_per_group=cfg.max_rows_per_group,
        max_open_files=cfg.max_open_files,
    )
    logger.info(
        "Finished parquet write to %s in %.3fs",
        parquet_out_dir,
        perf_counter() - stage_started_at,
    )

    _finalise_structured_components_write(
        state=_StructuredComponentsWriteState(
            parquet_out_dir=parquet_out_dir,
            entity_chronology=entity_chronology,
            has_inline_labels=has_inline_labels,
            inline_label_entries=inline_label_entries,
            raw_input_path=raw_input_path,
            batches_emitted=batches_emitted,
            started_at=started_at,
        ),
    )

    return has_inline_labels


def _track_inline_label_entries(
    *,
    batch: pa.RecordBatch,
    has_inline_labels: bool,
    inline_label_entries: list[dict[str, object]],
) -> bool:
    """Record sparse inline labels from one parsed record batch.

    Args:
        batch (pa.RecordBatch): Batch currently being written to parquet.
        has_inline_labels (bool): Whether any earlier batch exposed inline
            anomaly labels.
        inline_label_entries (list[dict[str, object]]): Sparse anomaly label
            records accumulated so far.

    Returns:
        bool: `True` when the batch exposed inline anomaly labels.
    """
    if ANOMALOUS_FIELD not in batch.schema.names:
        return has_inline_labels

    col = batch.column(ANOMALOUS_FIELD)
    if col.null_count >= len(col):
        return has_inline_labels

    line_values = batch.column(
        batch.schema.get_field_index(LINE_FIELD),
    ).to_pylist()
    entity_values = batch.column(
        batch.schema.get_field_index(ENTITY_FIELD),
    ).to_pylist()
    label_values = col.to_pylist()
    for line_order, entity_id, raw_label in zip(
        line_values,
        entity_values,
        label_values,
        strict=True,
    ):
        if raw_label not in {None, 0}:
            inline_label_entries.append(
                {
                    "line_order": int(line_order),
                    "entity_id": entity_id,
                    "anomalous": int(raw_label),
                },
            )
    return True


def _finalise_structured_components_write(
    *,
    state: _StructuredComponentsWriteState,
) -> None:
    """Write sidecars and log the final structured extraction summary.

    Args:
        state (_StructuredComponentsWriteState): Structured-write bookkeeping
            captured during parquet emission.
    """
    stage_started_at = perf_counter()
    if state.has_inline_labels:
        _write_inline_label_cache(
            parquet_out_dir=state.parquet_out_dir,
            inline_label_entries=state.inline_label_entries,
        )
        logger = get_run_logger()
        logger.info(
            "Wrote inline label sidecar for %s in %.3fs",
            state.parquet_out_dir,
            perf_counter() - stage_started_at,
        )
        stage_started_at = perf_counter()

    _write_entity_chronology_index(
        parquet_out_dir=state.parquet_out_dir,
        chronology=state.entity_chronology,
    )
    logger = get_run_logger()
    logger.info(
        "Wrote entity chronology sidecar for %s in %.3fs",
        state.parquet_out_dir,
        perf_counter() - stage_started_at,
    )
    stage_started_at = perf_counter()
    _write_entity_count(
        parquet_out_dir=state.parquet_out_dir,
        total_entities=len(state.entity_chronology),
    )
    logger.info(
        "Wrote entity count sidecar for %s in %.3fs",
        state.parquet_out_dir,
        perf_counter() - stage_started_at,
    )
    logger.info(
        (
            "Structured extraction complete: file=%s out=%s "
            "batches_written=%d elapsed=%.3fs"
        ),
        state.raw_input_path,
        state.parquet_out_dir,
        state.batches_emitted,
        perf_counter() - state.started_at,
    )


def _write_inline_label_cache(
    *,
    parquet_out_dir: Path,
    inline_label_entries: list[dict[str, object]],
) -> None:
    """Persist sparse inline labels alongside the structured parquet dataset.

    Args:
        parquet_out_dir (Path): Output directory for the structured parquet
            dataset.
        inline_label_entries (list[dict[str, object]]): Sparse anomaly labels
            keyed by line order and entity id.
    """
    cache_path = parquet_out_dir / INLINE_LABEL_CACHE_FILENAME
    with cache_path.open("w", encoding="utf-8") as handle:
        for entry in inline_label_entries:
            handle.write(json.dumps(entry, separators=(",", ":")))
            handle.write("\n")


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


def _write_entity_count(
    *,
    parquet_out_dir: Path,
    total_entities: int,
) -> None:
    """Persist the total distinct entity count alongside the parquet cache.

    Args:
        parquet_out_dir (Path): Output directory for the structured parquet
            dataset.
        total_entities (int): Total number of distinct entities seen during
            structured extraction.
    """
    cache_path = parquet_out_dir / ENTITY_COUNT_FILENAME
    with cache_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"total_entities": total_entities}, separators=(",", ":")),
        )
        handle.write("\n")
