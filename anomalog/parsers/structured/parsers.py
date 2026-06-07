"""Concrete StructuredParser implementations for built-in log formats."""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import ClassVar

from prefect.logging import get_logger
from typing_extensions import override

from anomalog.parsers.structured.contracts import (
    BaseStructuredLine,
    StructuredLine,
    StructuredParser,
)
from anomalog.sources.openstack import (
    extract_openstack_parameters,
    normalise_openstack_message,
    parse_openstack_payload,
)

UTC = timezone.utc


@dataclass(frozen=True, slots=True)
class HDFSV1Parser(StructuredParser):
    """Parse HDFS v1 log lines into structured fields.

    HDFS anomaly datasets are block-centric, so this parser prefers the block id
    mentioned in the log message as the `entity_id`; when no block is present it
    falls back to the logging component so entity-based grouping still works.

    Attributes:
        name (ClassVar[str]): Registry/config name for the built-in parser.
    """

    name: ClassVar[str] = "hdfs_v1"

    # Canonical HDFS v1 format:
    #   <Date> <Time> <Pid> <Level> <Component>: <Content>
    # e.g. 081109 203518 143 INFO dfs.DataNode$DataXceiver:
    # Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106
    # dest: /10.250.19.102:50010
    _HDFS_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"""
        ^\s*
        (?P<date>\d{6})\s+                 # yymmdd
        (?P<time>\d{6})\s+                 # HHMMSS
        (?P<pid>\d+)\s+                    # pid/tid-ish
        (?P<level>[A-Z]+)\s+               # INFO/WARN/ERROR/DEBUG/TRACE...
        (?P<component>\S+):\s+             # logger/component up to colon
        (?P<content>.*?)
        \s*$
        """,
        re.VERBOSE,
    )

    _BLOCK_RE: ClassVar[re.Pattern[str]] = re.compile(r"\bblk_-?\d+\b")

    @staticmethod
    def _yymmdd_hhmmss_to_unix_ms(date_s: str, time_s: str) -> int | None:
        """Convert YYMMDD and HHMMSS strings to epoch milliseconds.

        Args:
            date_s (str): Date string in `YYMMDD` format.
            time_s (str): Time string in `HHMMSS` format.

        Examples:
            >>> HDFSV1Parser._yymmdd_hhmmss_to_unix_ms("240101", "000000")
            1704067200000

        Returns:
            int | None: Parsed timestamp in milliseconds, or `None` if invalid.
        """
        try:
            dt = datetime.strptime(f"{date_s} {time_s}", "%y%m%d %H%M%S").replace(
                tzinfo=UTC,
            )
            return int(dt.timestamp() * 1000)
        except ValueError:
            return None

    @override
    def parse_line(self, raw_line: str) -> BaseStructuredLine | None:
        """Parse a single HDFS v1 line; return None for unparseable lines.

        Args:
            raw_line (str): Raw HDFS log line to parse.

        Examples:
            >>> line = (
            ...     "081109 203518 143 INFO dfs.DataNode$DataXceiver: "
            ...     "Receiving block blk_-160 src: /10.0.0.1:54106 "
            ...     "dest: /10.0.0.2:50010"
            ... )
            >>> parsed = HDFSV1Parser().parse_line(line)
            >>> parsed.entity_id, parsed.anomalous, parsed.untemplated_message_text[:13]
            ('blk_-160', None, 'INFO dfs.Data')

        Returns:
            BaseStructuredLine | None: Parsed structured record, or `None` when
                the line does not match the expected format.
        """
        s = raw_line.rstrip("\n")
        logger = get_logger()

        m = self._HDFS_RE.match(s)
        if not m:
            logger.warning("Cannot parse HDFS line: %r", s)
            return None

        d = m.groupdict()

        ts_ms = self._yymmdd_hhmmss_to_unix_ms(d["date"], d["time"])
        if ts_ms is None:
            logger.warning(
                "Failed to parse HDFS timestamp date=%r time=%r for raw line %r",
                d["date"],
                d["time"],
                s,
            )

        component = d["component"]
        content = d["content"].strip()

        # Prefer block id as entity_id when available (block-centric HDFS task).
        blk_m = self._BLOCK_RE.search(content)
        entity_id = blk_m.group(0) if blk_m else component

        untemplated = f"{d['level']} {component}: {content}".strip()

        return BaseStructuredLine(
            timestamp_unix_ms=ts_ms,
            entity_id=entity_id,
            untemplated_message_text=untemplated,
            anomalous=None,
        )


@dataclass(frozen=True, slots=True)
class BGLParser(StructuredParser):
    """Parse Blue Gene/L log lines into structured fields with anomaly flag.

    The BGL corpus encodes anomaly state in the optional leading dash, so this
    parser preserves that dataset convention directly in the shared `anomalous`
    field while keeping the original message tail for template mining.

    Attributes:
        name (ClassVar[str]): Registry/config name for the built-in parser.
    """

    name: ClassVar[str] = "bgl"

    # Matches both:
    #   - <epoch> <date> <loc> <hires_ts> <loc> <tail>
    #   <prefix> <epoch> <date> <loc> <hires_ts> <tail>
    #
    # with optional leading "-" that indicates "normal" in BGL.
    # e.g. - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779
    # R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
    _BGL_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"""
        ^\s*
        (?P<dash>-)?\s*
        (?:(?P<prefix>\d+:\S+)\s+)?(?:\S+\s+)?
        (?P<epoch>\d+)\s+
        (?P<date>\d{4}\.\d{2}\.\d{2})\s+
        (?P<entity>\S+)\s+
        (?P<hires_ts>\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+)\s+
        (?P<entity2>\S+)\s+
        (?P<tail>\S+\s+\S+\s+\S+.*)                      # FAC SUB SEV <rest...>
        \s*$
        """,
        re.VERBOSE,
    )

    @staticmethod
    def _hires_ts_to_unix_ms(ts: str) -> int | None:
        """Convert high-resolution timestamp string to epoch milliseconds.

        Args:
            ts (str): Timestamp string in BGL high-resolution format.

        Examples:
            >>> BGLParser._hires_ts_to_unix_ms("2005-06-03-15.42.50.363779")
            1117813370363
            >>> BGLParser._hires_ts_to_unix_ms("invalid") is None
            True

        Returns:
            int | None: Parsed timestamp in milliseconds, or `None` if invalid.
        """
        # BGL tooling usually treats these as UTC; adjust if you decide otherwise.
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d-%H.%M.%S.%f").replace(tzinfo=UTC)
            return int(dt.timestamp() * 1000)
        except ValueError:
            return None

    @override
    def parse_line(self, raw_line: str) -> BaseStructuredLine | None:
        """Parse a single BGL line; return None for unparseable lines.

        Args:
            raw_line (str): Raw BGL log line to parse.

        Examples:
            >>> sample = (
            ...     "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 "
            ...     "2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 "
            ...     "RAS KERNEL INFO cache parity corrected"
            ... )
            >>> parsed = BGLParser().parse_line(sample)
            >>> (parsed.entity_id, parsed.anomalous)  # dash prefix => normal
            ('R02-M1-N0-C:J12-U11', 0)

        Returns:
            BaseStructuredLine | None: Parsed structured record, or `None` when
                the line does not match the expected format.
        """
        s = raw_line.rstrip("\n")
        logger = get_logger()

        m = BGLParser._BGL_RE.match(s)
        if not m:
            logger.warning("Cannot parse BGL line: %r", s)
            return None

        d = m.groupdict()

        anomalous = 0 if d["dash"] == "-" else 1
        entity_id = d["entity"]

        ts_ms = BGLParser._hires_ts_to_unix_ms(d["hires_ts"])
        if ts_ms is None:
            # Fallback to epoch seconds if needed
            logger.warning(
                "Failed to parse hires timestamp %r for raw line %r, "
                "falling back to epoch seconds.",
                d["hires_ts"],
                s,
            )
            try:
                ts_ms = int(d["epoch"]) * 1000
            except ValueError:
                ts_ms = None

        untemplated = d["tail"].strip()

        return BaseStructuredLine(
            timestamp_unix_ms=ts_ms,
            entity_id=entity_id,
            untemplated_message_text=untemplated,
            anomalous=anomalous,
        )


@dataclass(frozen=True, slots=True)
class ThunderbirdParser(StructuredParser):
    """Parse Thunderbird supercomputer log lines into structured fields.

    Loghub's Thunderbird corpus uses a labelled raw-line format where the first
    token marks alert status (`-` for normal, any other tag for an alert) and
    the remaining header fields expose the event chronology plus the host and
    location tokens. The parser keeps the free-text tail as the message body
    for template mining, stripping an optional ``component[pid]: `` prefix
    when the raw line includes one. It also trims a trailing colon from bare
    message tails such as `mysql_install_db:` so the template miner sees the
    underlying command name rather than the punctuation artefact. The parser
    collapses the label into AnomaLog's binary anomaly flag.

    The parser deliberately stays close to the observed raw structure so the
    downstream template miner sees the message body rather than a Thunderbird-
    specific normalisation of the header fields.

    Attributes:
        name (ClassVar[str]): Registry/config name for the built-in parser.
    """

    name: ClassVar[str] = "thunderbird"

    _THUNDERBIRD_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"""
        ^\s*
        (?P<label>\S+)\s+
        (?:(?P<timestamp>\d+)\s+)?
        (?P<date>\d{4}\.\d{2}\.\d{2})\s+
        (?P<user>\S+)\s+
        (?P<month>[A-Z][a-z]{2})\s+
        (?P<day>\d{1,2})\s+
        (?P<time>\d{2}:\d{2}:\d{2})\s+
        (?P<location>\S+)
        (?:\s+(?P<tail>.*))?
        \s*$
        """,
        re.VERBOSE,
    )

    @staticmethod
    def _timestamp_seconds_to_unix_ms(timestamp_s: str | None) -> int | None:
        """Convert a Thunderbird epoch-second timestamp to milliseconds.

        Args:
            timestamp_s (str | None): Timestamp string from the raw log header.

        Returns:
            int | None: Parsed timestamp in Unix milliseconds, or `None` when
                the source omits the field or the value is malformed.
        """
        if timestamp_s is None:
            return None
        try:
            return int(timestamp_s) * 1000
        except ValueError:
            return None

    @classmethod
    def analyse_line(
        cls,
        raw_line: str,
    ) -> tuple[BaseStructuredLine | None, str | None]:
        """Parse one Thunderbird line and report the reason when skipped.

        Args:
            raw_line (str): Raw Thunderbird log line to inspect.

        Returns:
            tuple[BaseStructuredLine | None, str | None]: Parsed structured row
            and an optional skip reason.
        """
        s = raw_line.rstrip("\n")
        if not s.strip():
            return None, "blank"

        m = cls._THUNDERBIRD_RE.match(s)
        if m is None:
            return None, "malformed"

        d = m.groupdict()
        content = (d["tail"] or "").strip()
        if ": " in content:
            content = content.split(": ", maxsplit=1)[1].strip()
        if content.endswith(":"):
            content = content.rstrip(":").rstrip()
        if not content:
            return None, "empty_message"

        label = d["label"].strip()
        anomalous = 0 if label == "-" else 1
        timestamp_ms = cls._timestamp_seconds_to_unix_ms(d["timestamp"])
        entity_id = d["location"].strip() or d["user"].strip()

        return (
            BaseStructuredLine(
                timestamp_unix_ms=timestamp_ms,
                entity_id=entity_id,
                untemplated_message_text=content,
                anomalous=anomalous,
            ),
            None,
        )

    @classmethod
    def raw_label_for_line(cls, raw_line: str) -> int | None:
        """Return the raw anomaly label token for one Thunderbird line.

        Args:
            raw_line (str): Raw Thunderbird log line to inspect.

        Returns:
            int | None: `0` for normal rows, `1` for anomalous rows, or
            `None` when the line does not match the Thunderbird envelope.

        Notes:
            The helper mirrors the raw-line label token even when the parser
            later skips the row because the message body is empty. That keeps
            raw-position window labels aligned with the original line stream.
        """
        s = raw_line.rstrip("\n")
        if not s.strip():
            return None

        m = cls._THUNDERBIRD_RE.match(s)
        if m is None:
            return None

        return 0 if m.group("label").strip() == "-" else 1

    @override
    def parse_line(self, raw_line: str) -> BaseStructuredLine | None:
        """Parse a single Thunderbird line; return `None` for skipped rows.

        Args:
            raw_line (str): Raw Thunderbird log line to parse.

        Returns:
            BaseStructuredLine | None: Parsed structured record, or `None`
                when the line is blank, malformed, or has no message body.
        """
        logger = get_logger()
        parsed, reason = self.analyse_line(raw_line)
        if parsed is None:
            if reason not in {"blank", "empty_message"}:
                logger.warning(
                    "Cannot parse Thunderbird line (%s): %r",
                    reason,
                    raw_line,
                )
            return None
        return parsed


@dataclass(frozen=True, slots=True)
class AITADSParser(StructuredParser):
    """Parse the canonical JSONL alert stream derived from AIT-ADS.

    Attributes:
        name (ClassVar[str]): Registry/config name for the built-in parser.
    """

    name: ClassVar[str] = "ait_ads"

    @override
    def parse_line(self, raw_line: str) -> BaseStructuredLine | None:
        """Parse one canonical AIT-ADS alert row.

        Args:
            raw_line (str): Canonical JSONL row emitted by the AIT-ADS source.

        Returns:
            BaseStructuredLine | None: Parsed canonical alert, or `None` when
                the row is malformed.
        """
        s = raw_line.strip()
        logger = get_logger()
        if not s:
            return None

        try:
            payload = json.loads(s)
        except json.JSONDecodeError:
            logger.warning("Cannot parse AIT-ADS canonical row: %r", s)
            return None

        if not isinstance(payload, dict):
            logger.warning("AIT-ADS canonical row must be a JSON object: %r", s)
            return None

        template_key = payload.get("template_key")
        if template_key is None:
            logger.warning("AIT-ADS canonical row is missing template_key: %r", s)
            return None

        timestamp_unix_ms = _coerce_optional_int(payload.get("timestamp_unix_ms"))
        anomalous = _coerce_optional_int(payload.get("anomalous"))

        return BaseStructuredLine(
            timestamp_unix_ms=timestamp_unix_ms,
            entity_id=(
                None if payload.get("entity_id") is None else str(payload["entity_id"])
            ),
            untemplated_message_text=str(template_key),
            anomalous=anomalous,
        )


@dataclass(frozen=True, slots=True)
class _OpenStackLabelledRow:
    """Parsed labelled OpenStack row used by the DeepLog parser.

    Attributes:
        split_name (str): Split prefix attached to the row in the preprocessed
            stream.
        label (int): Inline anomaly label from the labelled OpenStack stream.
        timestamp_unix_ms (int): Parsed event timestamp in Unix milliseconds.
        instance_id (str): Instance identifier extracted from the payload.
        content (str): Raw message body after removing the OpenStack envelope.
        raw_parameters (list[str]): Numeric payload tokens preserved for
            parameter-aware downstream stages.
    """

    split_name: str
    label: int
    timestamp_unix_ms: int
    instance_id: str
    content: str
    raw_parameters: list[str]


def _parse_openstack_labelled_row(
    raw_line: str,
) -> _OpenStackLabelledRow | None:
    """Parse a labelled OpenStack row into its canonical fields.

    Args:
        raw_line (str): Raw labelled OpenStack line from the preprocessed
            stream.

    Returns:
        _OpenStackLabelledRow | None: Parsed row data, or `None` when the row
            is malformed.
    """
    if "\t" not in raw_line:
        return None
    split_name, label_text, raw_payload = raw_line.rstrip("\n").split("\t", 2)
    if label_text not in {"0", "1"}:
        return None

    parsed = parse_openstack_payload(raw_payload)
    if parsed is None:
        return None
    return _OpenStackLabelledRow(
        split_name=split_name.strip(),
        label=int(label_text),
        timestamp_unix_ms=parsed.timestamp_unix_ms,
        instance_id=parsed.instance_id,
        content=parsed.content,
        raw_parameters=extract_openstack_parameters(parsed.content or ""),
    )


@dataclass(frozen=True, slots=True)
class OpenStackDeepLogParser(StructuredParser):
    r"""Parse labelled OpenStack rows used by the DeepLog reproduction preset.

    Attributes:
        name (ClassVar[str]): Registry/config name for the built-in parser.
    """

    name: ClassVar[str] = "openstack_deeplog"

    @override
    def parse_line(self, raw_line: str) -> BaseStructuredLine | None:
        """Parse one labelled OpenStack row into the shared structured schema.

        Args:
            raw_line (str): Raw labelled OpenStack row from the preprocessed
                stream.

        Returns:
            BaseStructuredLine | None: Structured row, or `None` when the
                labelled OpenStack row is malformed.
        """
        logger = get_logger()
        parsed = _parse_openstack_labelled_row(raw_line)
        if parsed is None:
            logger.warning(
                "Cannot parse OpenStack labelled row: %r",
                raw_line.rstrip("\n"),
            )
            return None
        entity_id = f"{parsed.split_name}:{parsed.instance_id}"
        return StructuredLine(
            timestamp_unix_ms=parsed.timestamp_unix_ms,
            entity_id=entity_id,
            untemplated_message_text=normalise_openstack_message(
                parsed.content,
                preserve_numeric_values=False,
            ),
            anomalous=parsed.label,
            line_order=0,
            raw_parameters=parsed.raw_parameters,
        )


OptionalIntLike = int | float | str | None


def _coerce_optional_int(value: OptionalIntLike) -> int | None:
    """Convert a possibly-null payload field to an integer.

    Args:
        value (OptionalIntLike): Payload field to coerce.

    Returns:
        int | None: Parsed integer, or `None` when absent or malformed.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
