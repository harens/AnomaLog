"""Structured parser for synthetic labelled event streams."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

from typing_extensions import override

from anomalog.parsers.structured.contracts import BaseStructuredLine, StructuredParser

_EXPECTED_FIELDS = 3
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DelimitedLabelledEventParser(StructuredParser):
    """Parse a tab-separated `session_id`, `label`, `event_id` row.

    Attributes:
        name (ClassVar[str]): Registry/config name for the parser.
    """

    name: ClassVar[str] = "delimited_labelled_event"

    @override
    def parse_line(self, raw_line: str) -> BaseStructuredLine | None:
        """Parse one tab-separated labelled event row.

        Args:
            raw_line (str): One synthetic event row from a materialised
                labelled session stream.

        Returns:
            BaseStructuredLine | None: Structured event row, or `None` when the
                input is blank or malformed.
        """
        s = raw_line.rstrip("\n")
        if not s:
            return None

        parts = s.split("\t")
        if len(parts) != _EXPECTED_FIELDS:
            _LOGGER.warning("Cannot parse labelled event row: %r", s)
            return None

        session_id, label_s, event_id_s = parts
        try:
            anomalous = int(label_s)
        except ValueError:
            _LOGGER.warning("Invalid event label in row: %r", s)
            return None

        return BaseStructuredLine(
            timestamp_unix_ms=None,
            entity_id=session_id,
            untemplated_message_text=event_id_s,
            anomalous=anomalous,
        )
