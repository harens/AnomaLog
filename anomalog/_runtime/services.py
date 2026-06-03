"""Pure-ish orchestration services used by runtime flows."""

from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import TYPE_CHECKING

from prefect import flow
from prefect.logging import get_run_logger

from anomalog.cache.core import dataset_build_lock
from anomalog.labels import InlineReader
from anomalog.parsers.structured.contracts import UNTEMPLATED_FIELD, StructuredSink
from anomalog.parsers.structured.dataset import StructuredDataset

if TYPE_CHECKING:
    from pathlib import Path

    from anomalog._runtime.models import TemplatedDatasetBuildRequest
    from anomalog.parsers.template import TemplatedDataset

_PREFECT_EPHEMERAL_STARTUP_TIMEOUT_SECONDS = 120


def _log_stage_timing(
    logger: logging.Logger | logging.LoggerAdapter[logging.Logger],
    *,
    stage: str,
    started_at: float,
) -> None:
    elapsed = perf_counter() - started_at
    logger.info("Stage complete: %s in %.3fs", stage, elapsed)


def _ensure_prefect_ephemeral_startup_timeout() -> None:
    """Set a longer startup budget for Prefect's ephemeral API server.

    Slurm nodes can take longer than Prefect's default 20 second window to
    start the temporary API server that backs a flow run. A local dataset build
    should fail for build reasons, not because the embedded server was slow to
    become ready on a busy machine.
    """
    os.environ.setdefault(
        "PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS",
        str(_PREFECT_EPHEMERAL_STARTUP_TIMEOUT_SECONDS),
    )


def _materialize_dataset(
    request: TemplatedDatasetBuildRequest,
) -> tuple[Path, Path]:
    logger = get_run_logger()
    started_at = perf_counter()
    dataset_root = request.cache_paths.data_root / request.dataset_name
    logger.info("Fetching dataset %s to %s", request.dataset_name, dataset_root)
    dataset_root = request.source.materialise(
        dst_dir=dataset_root,
    )
    raw_logs_path = request.source.raw_logs_path(
        dataset_name=request.dataset_name,
        dataset_root=dataset_root,
    )
    _log_stage_timing(
        logger,
        stage=f"dataset materialisation for {request.dataset_name}",
        started_at=started_at,
    )
    return dataset_root, raw_logs_path


def _log_example_line(dataset_name: str, sink: StructuredSink) -> None:
    logger = get_run_logger()
    examples = sink.iter_structured_lines(columns=[UNTEMPLATED_FIELD])

    try:
        example_line = next(
            row.untemplated_message_text for row in examples() if row is not None
        )
        logger.info(
            "Example unstructured line content for dataset %s: %r",
            dataset_name,
            example_line,
        )
    except StopIteration:
        logger.warning(
            "No unstructured line content found for dataset %s",
            dataset_name,
        )


def _build_structured_dataset(
    request: TemplatedDatasetBuildRequest,
) -> StructuredDataset:
    logger = get_run_logger()
    started_at = perf_counter()
    dataset_root, raw_logs_path = _materialize_dataset(request)
    logger.info(
        "Initialising structured sink for %s from %s",
        request.dataset_name,
        raw_logs_path,
    )
    sink_started_at = perf_counter()
    sink = request.structured_sink(
        dataset_name=request.dataset_name,
        raw_dataset_path=raw_logs_path,
        parser=request.structured_parser,
        cache_paths=request.cache_paths,
    )
    _log_stage_timing(
        logger,
        stage=f"structured sink initialisation for {request.dataset_name}",
        started_at=sink_started_at,
    )
    extraction_started_at = perf_counter()
    inline_labels_present = sink.write_structured_lines()
    _log_stage_timing(
        logger,
        stage=f"structured component extraction for {request.dataset_name}",
        started_at=extraction_started_at,
    )
    _log_example_line(request.dataset_name, sink)

    if inline_labels_present:
        label_reader = InlineReader(sink=sink)
    else:
        if request.anomaly_label_reader is None:
            msg = (
                "Structured data has no inline anomaly labels and no "
                "anomaly_label_reader was provided."
            )
            raise ValueError(msg)
        label_reader = request.anomaly_label_reader.with_context(
            dataset_root=dataset_root,
            sink=sink,
        )

    labels_loaded_started_at = perf_counter()
    anomaly_labels = label_reader.load()
    _log_stage_timing(
        logger,
        stage=f"anomaly label loading for {request.dataset_name}",
        started_at=labels_loaded_started_at,
    )

    _log_stage_timing(
        logger,
        stage=f"structured dataset build for {request.dataset_name}",
        started_at=started_at,
    )
    return StructuredDataset(
        sink=sink,
        cache_paths=request.cache_paths,
        anomaly_labels=anomaly_labels,
    )


def _build_templated_dataset(
    request: TemplatedDatasetBuildRequest,
) -> TemplatedDataset:
    """Build the templated dataset view from a runtime build request.

    Args:
        request (TemplatedDatasetBuildRequest): Compiled runtime request for the
            dataset build.

    Returns:
        TemplatedDataset: Structured dataset with templates mined and attached.
    """
    logger = get_run_logger()
    structured = _build_structured_dataset(request)
    template_started_at = perf_counter()
    logger.info("Training template parser for %s", request.dataset_name)
    templated = structured.mine_templates_with(
        request.template_parser(dataset_name=request.dataset_name),
    )
    _log_stage_timing(
        logger,
        stage=f"template parser training for {request.dataset_name}",
        started_at=template_started_at,
    )
    return templated


def build_templated_dataset(request: TemplatedDatasetBuildRequest) -> TemplatedDataset:
    """Run the internal templated build flow for a compiled runtime request.

    Args:
        request (TemplatedDatasetBuildRequest): Compiled runtime request for the
            dataset build.

    Returns:
        TemplatedDataset: Built dataset returned from the internal Prefect flow.
    """
    _ensure_prefect_ephemeral_startup_timeout()

    @flow
    def _build_dataset_flow() -> TemplatedDataset:
        return _build_templated_dataset(request)

    logger = logging.getLogger(__name__)
    lock_started_at = perf_counter()
    logger.info("Acquiring dataset build lock for %s", request.dataset_name)
    with dataset_build_lock(
        request.dataset_name,
        cache_paths=request.cache_paths,
    ):
        _log_stage_timing(
            logger,
            stage=f"dataset build lock acquisition for {request.dataset_name}",
            started_at=lock_started_at,
        )
        flow_started_at = perf_counter()
        try:
            return _build_dataset_flow()
        finally:
            _log_stage_timing(
                logger,
                stage=f"dataset build flow for {request.dataset_name}",
                started_at=flow_started_at,
            )
