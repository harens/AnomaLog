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

UTC = timezone.utc
_OPENSTACK_LABELLED_ROW_RE = re.compile(
    r"^(?P<split>[^\t]+)\t(?P<label>[01])\t(?P<raw>.*)$",
)
_OPENSTACK_RE = re.compile(
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
_INSTANCE_PREFIX_RE = re.compile(
    r"^\[instance:\s*(?P<instance_id>[^\]]+?)\]\s*",
    re.IGNORECASE,
)
_INSTANCE_ID_PATTERNS = (
    re.compile(r"\[instance:\s*(?P<instance_id>[^\]]+?)\]"),
    re.compile(r"\bfor instance (?P<instance_id>\S+)"),
)
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PATH_RE = re.compile(r"(?<!\S)/(?:[\w.-]+/)*[\w.-]+")
_INSTANCE_STORAGE_PATH_RE = re.compile(r"(?<!\S)/var/lib/nova/instances/[^)\s,.;:]+")
_HEX_RE = re.compile(r"\b[0-9a-f]{12,}\b", re.IGNORECASE)
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _openstack_datetime_to_unix_ms(date_s: str, time_s: str) -> int | None:
    """Convert OpenStack date and time fragments to Unix milliseconds.

    Args:
        date_s (str): Date fragment in `YYYY-MM-DD` format.
        time_s (str): Time fragment in `HH:MM:SS[.ffffff]` format.

    Returns:
        int | None: Parsed timestamp in milliseconds, or `None` when parsing
            fails.
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
        return None


def _extract_openstack_instance_id(raw_payload: str) -> str | None:
    """Return the OpenStack instance identifier, if present.

    Args:
        raw_payload (str): Raw OpenStack payload text to inspect.

    Returns:
        str | None: Normalised instance identifier, or `None` when the payload
            does not expose one.
    """
    for pattern in _INSTANCE_ID_PATTERNS:
        match = pattern.search(raw_payload)
        if match is None:
            continue
        instance_id = match.group("instance_id").strip()
        if instance_id:
            return instance_id
    return None


def _normalise_openstack_path_tokens(
    text: str,
    *,
    preserve_numeric_values: bool,
) -> str:
    def _normalise_path(match: re.Match[str]) -> str:
        token = match.group(0)
        suffix = ""
        while token and token[-1] in ",.;:)]":
            suffix = token[-1] + suffix
            token = token[:-1]
        segments = token.split("/")
        normalised_segments = []
        for segment in segments:
            if not segment:
                normalised_segments.append(segment)
                continue
            if _UUID_RE.fullmatch(segment):
                normalised_segments.append("UUID")
                continue
            if _IP_RE.fullmatch(segment):
                normalised_segments.append("IP")
                continue
            if _HEX_RE.fullmatch(segment):
                normalised_segments.append("HEX")
                continue
            if not preserve_numeric_values and re.fullmatch(r"\d+(?:\.\d+)?", segment):
                normalised_segments.append("NUM")
                continue
            normalised_segments.append(segment)
        return "/".join(normalised_segments) + suffix

    return _PATH_RE.sub(_normalise_path, text)


def _normalise_openstack_message(
    content: str,
    *,
    preserve_numeric_values: bool,
) -> str:
    """Canonicalise OpenStack message text before template mining.

    Args:
        content (str): Raw OpenStack message body.
        preserve_numeric_values (bool): Whether numeric tokens should be kept rather
            than replaced with `NUM`.

    Returns:
        str: Canonicalised OpenStack message text.
    """
    text = _INSTANCE_PREFIX_RE.sub("", content).strip()
    text = _UUID_RE.sub("UUID", text)
    text = _IP_RE.sub("IP", text)
    text = _normalise_openstack_path_tokens(
        text,
        preserve_numeric_values=preserve_numeric_values,
    )
    text = _INSTANCE_STORAGE_PATH_RE.sub("INSTANCE_PATH", text)
    text = _HEX_RE.sub("HEX", text)
    if preserve_numeric_values:
        return text
    return _NUM_RE.sub("NUM", text)


def _parse_openstack_labelled_row(
    raw_line: str,
) -> tuple[str, int, int, str, str, list[str]] | None:
    """Parse a labelled OpenStack row into split, label, and payload fields.

    Args:
        raw_line (str): Raw labelled OpenStack line from the preprocessed stream.

    Returns:
        tuple[str, int, int, str, str, list[str]] | None: Split name, label,
            timestamp, instance id, canonical content, and raw parameters, or
            `None` when the row is malformed.
    """
    row_match = _OPENSTACK_LABELLED_ROW_RE.match(raw_line.rstrip("\n"))
    if row_match is None:
        return None
    raw_payload = row_match.group("raw")
    payload_match = _OPENSTACK_RE.match(raw_payload)
    if payload_match is None:
        return None
    data = payload_match.groupdict()
    timestamp_ms = _openstack_datetime_to_unix_ms(data["date"], data["time"])
    if timestamp_ms is None:
        return None
    instance_id = _extract_openstack_instance_id(raw_payload)
    if instance_id is None:
        return None
    return (
        row_match.group("split").strip(),
        int(row_match.group("label")),
        timestamp_ms,
        instance_id,
        data["content"].strip(),
        _extract_openstack_parameters(data["content"] or ""),
    )


def _extract_openstack_parameters(content: str) -> list[str]:
    """Extract numeric OpenStack parameters from the message body.

    Args:
        content (str): Raw OpenStack message body.

    Returns:
        list[str]: Raw numeric tokens in encounter order.
    """
    text = _INSTANCE_PREFIX_RE.sub("", content).strip()
    return _NUM_RE.findall(text)


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
            raw_line (str): Raw labelled OpenStack row from the preprocessed stream.

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
        split_name, label, timestamp_ms, instance_id, content, raw_parameters = parsed
        entity_id = f"{split_name}:{instance_id}"
        return StructuredLine(
            timestamp_unix_ms=timestamp_ms,
            entity_id=entity_id,
            untemplated_message_text=_normalise_openstack_message(
                content,
                preserve_numeric_values=False,
            ),
            anomalous=label,
            line_order=0,
            raw_parameters=raw_parameters,
        )


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

        try:
            timestamp_unix_ms = (
                None
                if payload.get("timestamp_unix_ms") is None
                else int(payload["timestamp_unix_ms"])
            )
        except (TypeError, ValueError):
            timestamp_unix_ms = None
        try:
            anomalous = (
                None if payload.get("anomalous") is None else int(payload["anomalous"])
            )
        except (TypeError, ValueError):
            anomalous = None

        return BaseStructuredLine(
            timestamp_unix_ms=timestamp_unix_ms,
            entity_id=(
                None if payload.get("entity_id") is None else str(payload["entity_id"])
            ),
            untemplated_message_text=str(template_key),
            anomalous=anomalous,
        )
