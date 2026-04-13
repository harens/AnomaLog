"""Public structured parser and sink exports."""

from anomalog.parsers.structured.contracts import (
    BaseStructuredLine,
    StructuredParser,
    StructuredSink,
)
from anomalog.parsers.structured.parquet.sink import ParquetStructuredSink
from anomalog.parsers.structured.parsers import BGLParser, HDFSV1Parser

_STRUCTURED_PARSERS: dict[str, type[StructuredParser]] = {
    "bgl": BGLParser,
    "hdfs_v1": HDFSV1Parser,
}


def resolve_structured_parser(name: str) -> type[StructuredParser]:
    """Resolve a built-in structured parser by config name."""
    try:
        return _STRUCTURED_PARSERS[name]
    except KeyError as exc:
        msg = f"Unsupported structured parser: {name!r}"
        raise KeyError(msg) from exc


def structured_parser_names() -> tuple[str, ...]:
    """Return supported built-in structured parser names."""
    return tuple(_STRUCTURED_PARSERS)


__all__ = [
    "BGLParser",
    "BaseStructuredLine",
    "HDFSV1Parser",
    "ParquetStructuredSink",
    "StructuredParser",
    "StructuredSink",
    "resolve_structured_parser",
    "structured_parser_names",
]
