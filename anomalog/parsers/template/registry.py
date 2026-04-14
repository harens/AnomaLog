"""Registry helpers for built-in template parsers."""

from anomalog.parsers.template.dataset import TemplateParser
from anomalog.parsers.template.parsers import Drain3Parser, IdentityTemplateParser

_TEMPLATE_PARSERS: dict[str, type[TemplateParser]] = {
    "drain3": Drain3Parser,
    "identity": IdentityTemplateParser,
}


def resolve_template_parser(name: str) -> type[TemplateParser]:
    """Resolve a built-in template parser by config name.

    Args:
        name (str): Registry name for the parser.

    Returns:
        type[TemplateParser]: Registered template parser type.

    Raises:
        KeyError: If `name` does not match a built-in parser.
    """
    try:
        return _TEMPLATE_PARSERS[name]
    except KeyError as exc:
        msg = f"Unsupported template parser: {name!r}"
        raise KeyError(msg) from exc


def template_parser_names() -> tuple[str, ...]:
    """Return supported built-in template parser names.

    Returns:
        tuple[str, ...]: Parser registry names in registration order.
    """
    return tuple(_TEMPLATE_PARSERS)
