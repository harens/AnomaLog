"""OpenStack structured parsing helpers and parser implementation."""

import re
from dataclasses import dataclass
from datetime import datetime, timezone

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


def _strip_openstack_instance_prefix(content: str) -> str:
    """Remove the leading OpenStack instance marker, if present.

    Args:
        content (str): Raw OpenStack message body.

    Returns:
        str: Message body without the leading instance prefix.
    """
    return _INSTANCE_PREFIX_RE.sub("", content).strip()


def _normalise_openstack_message(
    content: str,
    *,
    preserve_numeric_values: bool,
) -> str:
    """Canonicalise OpenStack message text before template mining.

    Args:
        content (str): Raw OpenStack message body.
        preserve_numeric_values (bool): Whether numeric tokens should be kept
            rather than replaced with `NUM`.

    Returns:
        str: Canonicalised OpenStack message text.
    """
    text = _strip_openstack_instance_prefix(content)
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


def _extract_openstack_parameters(content: str) -> list[str]:
    """Extract numeric OpenStack parameters from the message body.

    Args:
        content (str): Raw OpenStack message body.

    Returns:
        list[str]: Raw numeric tokens in encounter order.
    """
    return _NUM_RE.findall(_strip_openstack_instance_prefix(content))


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
    return _OpenStackLabelledRow(
        split_name=row_match.group("split").strip(),
        label=int(row_match.group("label")),
        timestamp_unix_ms=timestamp_ms,
        instance_id=instance_id,
        content=data["content"].strip(),
        raw_parameters=_extract_openstack_parameters(data["content"] or ""),
    )
