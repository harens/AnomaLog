"""Registry helpers for built-in structured parsers."""

from anomalog.parsers.structured.contracts import StructuredParser
from anomalog.parsers.structured.parsers import BGLParser, HDFSV1Parser

_STRUCTURED_PARSERS: dict[str, type[StructuredParser]] = {
    "bgl": BGLParser,
    "hdfs_v1": HDFSV1Parser,
}


def resolve_structured_parser(name: str) -> type[StructuredParser]:
    """Resolve a built-in structured parser by config name.

    Args:
        name (str): Registry name for the parser.

    Returns:
        type[StructuredParser]: Registered structured parser type.

    Raises:
        KeyError: If `name` does not match a built-in parser.
    """
    try:
        return _STRUCTURED_PARSERS[name]
    except KeyError as exc:
        msg = f"Unsupported structured parser: {name!r}"
        raise KeyError(msg) from exc


def structured_parser_names() -> tuple[str, ...]:
    """Return supported built-in structured parser names.

    Returns:
        tuple[str, ...]: Parser registry names in registration order.
    """
    return tuple(_STRUCTURED_PARSERS)
