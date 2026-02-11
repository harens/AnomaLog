import math
import os
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from prefect.logging import get_run_logger

from anomalog.structured_parsers.contracts import (
    StructuredLine,
    StructuredParser,
)
from anomalog.structured_parsers.parquet.contracts import WriterJobCtx, WriterResult


class WriterWorker:
    def __init__(self, parser: StructuredParser, writer_ctx: WriterJobCtx) -> None:
        self.parser = parser

        self.in_path = writer_ctx.raw_input_path
        self.out_dir = writer_ctx.parquet_out_dir
        self.part_id = writer_ctx.part_id
        self.out_path = (
            writer_ctx.parquet_out_dir / f"part-{writer_ctx.part_id:05d}.parquet"
        )
        self.start = writer_ctx.start
        self.end = writer_ctx.end

    def parse_byte_range_and_write(self) -> WriterResult:
        rows: list[StructuredLine] = []
        n_read = 0
        n_parsed = 0
        has_anomaly = False

        # Read only this slice; align to newline boundaries
        with Path(self.in_path).open("rb") as f:
            f.seek(self.start)

            # If not at file start, discard partial first line
            if self.start != 0:
                _ = f.readline()

            while True:
                pos = f.tell()
                if pos > self.end:
                    break

                line = f.readline()
                if not line:
                    break

                n_read += 1
                byte_offset = pos

                # Decode one line
                raw_line = (
                    line.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
                )

                rec = self.parser.parse_line(raw_line, line_order=byte_offset)
                if rec is not None:
                    rows.append(rec)
                    n_parsed += 1
                    if rec.anomalous:
                        has_anomaly = True

        # Always write a file (even if empty) so downstream expects consistent parts
        table = pa.Table.from_pylist([asdict(r) for r in rows])
        pq.write_table(table, self.out_path)

        return WriterResult(
            part_id=self.part_id,
            out_path=self.out_path,
            n_read=n_read,
            n_parsed=n_parsed,
            start=self.start,
            end=self.end,
            has_anomaly=has_anomaly,
        )


class _WorkerState:
    parser: StructuredParser | None = None


_STATE = _WorkerState()


def _init_worker(parser: StructuredParser) -> None:
    _STATE.parser = parser


def _run_job(ctx: WriterJobCtx) -> WriterResult:
    # Top-level function so it is picklable on all platforms.
    if _STATE.parser is None:
        msg = "Parser not initialised in worker process"
        raise RuntimeError(msg)
    return WriterWorker(_STATE.parser, ctx).parse_byte_range_and_write()


# TODO: If stale parser cache exists, and we modify the parser to invalidate it
# and then ctrl+c during parser run and reset code to original, prefect would
# cache but the files would be missing as we deleted them.
def extract_structured_components(
    *,
    raw_input_path: Path,
    parser: StructuredParser,
    parquet_out_dir: Path,
    workers: int | None = None,
) -> bool:
    logger = get_run_logger()
    if workers is None:
        workers = os.cpu_count() or 8
    workers = max(1, int(workers))

    raw_input_path = raw_input_path.resolve()
    parquet_out_dir = parquet_out_dir.resolve()

    stat = raw_input_path.stat()
    size = stat.st_size
    if size == 0:
        msg = f"Empty file: {raw_input_path}"
        raise ValueError(msg)

    if parquet_out_dir.exists():
        logger.info(
            "Structured extraction output dir exists but not "
            "cached, deleting parquet files: out=%s",
            parquet_out_dir,
        )
        for f in parquet_out_dir.glob("part-*.parquet"):
            f.unlink()

    parquet_out_dir.mkdir(parents=True, exist_ok=True)

    # Don't create more parts than makes sense for the file size
    # (tiny parts waste overhead). This threshold is subjective.
    min_bytes_per_part = 8 * 1024 * 1024  # 8MB
    max_parts = max(1, size // min_bytes_per_part)
    n_parts = min(workers, max_parts) if max_parts > 0 else 1

    step = math.ceil(size / n_parts)

    jobs: list[WriterJobCtx] = []
    start = 0
    for part_id in range(n_parts):
        end = min(size - 1, start + step - 1)
        jobs.append(
            WriterJobCtx(
                part_id,
                raw_input_path,
                start,
                end,
                parquet_out_dir,
            ),
        )
        start = end + 1

    logger.info(
        "Structured extraction: file=%s size=%d bytes parts=%d workers=%d out=%s",
        raw_input_path,
        size,
        n_parts,
        workers,
        parquet_out_dir,
    )

    # Run pool: each process initialises its own parser instance
    results: list[WriterResult] = []
    with Pool(processes=n_parts, initializer=_init_worker, initargs=(parser,)) as pool:
        for res in pool.imap_unordered(_run_job, jobs, chunksize=1):
            results.append(res)
            logger.info(
                "Wrote part=%d read=%d parsed=%d range=[%d..%d] -> %s",
                res.part_id,
                res.n_read,
                res.n_parsed,
                res.start,
                res.end,
                res.out_path,
            )

    # Verify outputs exist
    results.sort(key=lambda r: r.part_id)
    missing = [r.out_path for r in results if not Path(r.out_path).exists()]
    if missing:
        msg = f"Some part files were not written: {missing[:5]}"
        raise RuntimeError(msg)

    total_read = sum(r.n_read for r in results)
    total_parsed = sum(r.n_parsed for r in results)
    logger.info(
        "Done: parts=%d total_read=%d total_parsed=%d out=%s",
        len(results),
        total_read,
        total_parsed,
        parquet_out_dir,
    )
    return any(r.has_anomaly for r in results)
