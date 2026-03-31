"""Pure-ish orchestration services used by runtime flows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prefect import flow
from prefect.logging import get_run_logger

from anomalog.labels import InlineReader
from anomalog.parsers.structured.contracts import UNTEMPLATED_FIELD, StructuredSink
from anomalog.parsers.structured.dataset import StructuredDataset

if TYPE_CHECKING:
    from pathlib import Path

    from anomalog._runtime.models import TemplatedDatasetBuildRequest
    from anomalog.parsers.template import TemplatedDataset


def _materialize_dataset(
    request: TemplatedDatasetBuildRequest,
) -> tuple[Path, Path]:
    logger = get_run_logger()
    dataset_root = request.cache_paths.data_root / request.dataset_name
    logger.info("Fetching dataset %s to %s", request.dataset_name, dataset_root)
    dataset_root = request.source.materialise(
        dst_dir=dataset_root,
    )
    raw_logs_path = request.source.raw_logs_path(
        dataset_name=request.dataset_name,
        dataset_root=dataset_root,
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
    dataset_root, raw_logs_path = _materialize_dataset(request)
    sink = request.structured_sink(
        dataset_name=request.dataset_name,
        raw_dataset_path=raw_logs_path,
        parser=request.structured_parser,
        cache_paths=request.cache_paths,
    )
    anomalies_inline = sink.write_structured_lines()
    _log_example_line(request.dataset_name, sink)

    if anomalies_inline:
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

    return StructuredDataset(
        sink=sink,
        cache_paths=request.cache_paths,
        anomaly_labels=label_reader.load(),
    )


def _build_templated_dataset(
    request: TemplatedDatasetBuildRequest,
) -> TemplatedDataset:
    """Build the templated dataset view from a runtime build request."""
    structured = _build_structured_dataset(request)
    return structured.mine_templates_with(
        request.template_parser(dataset_name=request.dataset_name),
    )


def build_templated_dataset(request: TemplatedDatasetBuildRequest) -> TemplatedDataset:
    """Run the internal templated build flow for a compiled runtime request."""

    @flow
    def _build_dataset_flow() -> TemplatedDataset:
        return _build_templated_dataset(request)

    return _build_dataset_flow()
