"""Public template parser exports."""

from anomalog.parsers.template.dataset import (
    ExtractedParameters,
    LogTemplate,
    TemplatedDataset,
    TemplateParser,
    UntemplatedText,
)
from anomalog.parsers.template.parsers import Drain3Parser, IdentityTemplateParser

_TEMPLATE_PARSERS: dict[str, type[TemplateParser]] = {
    "drain3": Drain3Parser,
    "identity": IdentityTemplateParser,
}


def resolve_template_parser(name: str) -> type[TemplateParser]:
    """Resolve a built-in template parser by config name."""
    try:
        return _TEMPLATE_PARSERS[name]
    except KeyError as exc:
        msg = f"Unsupported template parser: {name!r}"
        raise KeyError(msg) from exc


def template_parser_names() -> tuple[str, ...]:
    """Return supported built-in template parser names."""
    return tuple(_TEMPLATE_PARSERS)


__all__ = [
    "Drain3Parser",
    "ExtractedParameters",
    "IdentityTemplateParser",
    "LogTemplate",
    "TemplateParser",
    "TemplatedDataset",
    "UntemplatedText",
    "resolve_template_parser",
    "template_parser_names",
]
