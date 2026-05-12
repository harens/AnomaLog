"""Concrete StructuredParser implementations for built-in log formats."""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import ClassVar

from prefect.logging import get_logger
from typing_extensions import override

from anomalog.parsers.structured.contracts import BaseStructuredLine, StructuredParser

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
    _BGL_RE = re.compile(
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
class OpenStackDeepLogParser(StructuredParser):
    r"""Parse labelled OpenStack rows used by the DeepLog reproduction preset.

    Input rows are expected in the derived-source format:
    `<split_name>\\t<label>\\t<raw_openstack_line>`.
    The parser maps each row into the canonical structured schema, keeps the
    raw message content only for template mining, and prefixes the recovered
    VM `instance_id` with the file split so the session boundary remains
    explicit. Rows that do not expose an instance identifier are skipped
    because there is no paper-faithful session key to recover from them.

    Attributes:
        name (ClassVar[str]): Registry/config name for the built-in parser.
    """

    name: ClassVar[str] = "openstack_deeplog"

    _ROW_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?P<split>[^\t]+)\t(?P<label>[01])\t(?P<raw>.*)$",
    )
    _OPENSTACK_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"""
        ^\s*
        (?P<logrecord>\S+)\s+
        (?P<date>\d{4}-\d{2}-\d{2})\s+
        (?P<time>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+
        (?P<pid>\S+)\s+
        (?P<level>[A-Z]+)\s+
        (?P<component>\S+)
        \s+\[(?P<addr>[^\]]+)\]
        \s+(?P<content>.*\S)
        \s*$
        """,
        re.VERBOSE,
    )
    _INSTANCE_ID_PATTERNS: ClassVar[tuple[re.Pattern[str], ...]] = (
        re.compile(r"\[instance:\s*(?P<instance_id>[^\]]+?)\]"),
        re.compile(r"\bfor instance (?P<instance_id>\S+)"),
    )

    @staticmethod
    def _extract_instance_id(raw_payload: str) -> str | None:
        """Return the OpenStack instance identifier, if the line exposes one.

        Args:
            raw_payload (str): Raw OpenStack payload text from the labelled
                source row.

        Returns:
            str | None: Recognised instance identifier, or `None` when the
                payload does not expose one.
        """
        for pattern in OpenStackDeepLogParser._INSTANCE_ID_PATTERNS:
            match = pattern.search(raw_payload)
            if match is None:
                continue
            instance_id = match.group("instance_id").strip()
            if instance_id:
                return instance_id
        return None

    @override
    def parse_line(self, raw_line: str) -> BaseStructuredLine | None:
        """Parse one labelled OpenStack row.

        Args:
            raw_line (str): Labelled row from the derived raw stream.

        Returns:
            BaseStructuredLine | None: Parsed row or `None` when malformed.
        """
        s = raw_line.rstrip("\n")
        logger = get_logger()

        row_match = self._ROW_RE.match(s)
        if row_match is None:
            logger.warning("Cannot parse OpenStack labelled row: %r", s)
            return None

        raw_payload = row_match.group("raw")
        split_name = row_match.group("split").strip()
        label = int(row_match.group("label"))

        match = self._OPENSTACK_RE.match(raw_payload)
        if match is None:
            logger.warning("Cannot parse OpenStack payload row: %r", raw_payload)
            return None

        data = match.groupdict()
        ts_ms = _openstack_datetime_to_unix_ms(data["date"], data["time"])
        if ts_ms is None:
            logger.warning(
                "Failed to parse OpenStack timestamp date=%r time=%r row=%r",
                data["date"],
                data["time"],
                raw_payload,
            )
            return None
        entity_id = self._extract_instance_id(raw_payload)
        if entity_id is None:
            logger.warning(
                "Skipping OpenStack row without instance identifier: %r",
                raw_payload,
            )
            return None
        entity_id = f"{split_name}:{entity_id}"

        return BaseStructuredLine(
            timestamp_unix_ms=ts_ms,
            entity_id=entity_id,
            untemplated_message_text=(data["content"] or "").strip(),
            anomalous=label,
        )


def _openstack_datetime_to_unix_ms(date_s: str, time_s: str) -> int | None:
    """Convert OpenStack date/time fragments to epoch milliseconds.

    Args:
        date_s (str): Date fragment in `YYYY-MM-DD`.
        time_s (str): Time fragment with optional fractional seconds.

    Returns:
        int | None: Unix epoch milliseconds, or `None` when parsing fails.
    """
    value = f"{date_s} {time_s}"
    try:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)
    except ValueError:
        pass
    try:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)
    except ValueError:
        pass
    return None
