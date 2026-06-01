"""OpenStack-specific dataset materialisation helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

UTC = timezone.utc

OPENSTACK_RE = re.compile(
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
INSTANCE_PREFIX_RE = re.compile(
    r"^\[instance:\s*(?P<instance_id>[^\]]+?)\]\s*",
    re.IGNORECASE,
)
INSTANCE_ID_PATTERNS = (
    re.compile(r"\[instance:\s*(?P<instance_id>[^\]]+?)\]"),
    re.compile(r"\bfor instance (?P<instance_id>\S+)"),
)
UUID_RE = re.compile(
    r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PATH_RE = re.compile(r"(?<!\S)/(?:[\w.-]+/)*[\w.-]+")
INSTANCE_STORAGE_PATH_RE = re.compile(r"(?<!\S)/var/lib/nova/instances/[^)\s,.;:]+")
HEX_RE = re.compile(r"\b[0-9a-f]{12,}\b", re.IGNORECASE)
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


@dataclass(frozen=True, slots=True)
class OpenStackParsedPayload:
    """Parsed OpenStack payload fields used by the parameter materialiser."""

    timestamp_unix_ms: int
    instance_id: str
    content: str
    raw_parameters: list[str]


@dataclass(frozen=True, slots=True)
class _OpenStackParameterEvent:
    raw_payload: str
    timestamp_ms: int
    content: str


@dataclass(frozen=True, slots=True)
class _OpenStackParameterInstance:
    instance_id: str
    timestamp_ms: int
    events: list[_OpenStackParameterEvent]


_BUILD_SECONDS_RE = re.compile(
    r"^Took (?P<seconds>\d+(?:\.\d+)?) seconds to build instance\.$",
)
_PERFORMANCE_TARGET_TEMPLATE = (
    "During sync_power_state the instance has a pending task (spawning). Skip."
)
_DURATION_TARGET_TEMPLATE = "Took NUM seconds to build instance."
_OPENSTACK_PARAMETER_SUBSET_INSTANCES = 301
_OPENSTACK_PARAMETER_TRAIN_INSTANCES = 120
_OPENSTACK_PARAMETER_VALIDATION_INSTANCES = 20
_OPENSTACK_PARAMETER_PERFORMANCE_OFFSET = 7
_OPENSTACK_PARAMETER_ANOMALY_INSTANCE_OFFSETS = (
    _OPENSTACK_PARAMETER_PERFORMANCE_OFFSET,
    _OPENSTACK_PARAMETER_PERFORMANCE_OFFSET + 1,
)
_OPENSTACK_PARAMETER_PERFORMANCE_DELAY_MS = 600_000
_OPENSTACK_PARAMETER_DURATION_MULTIPLIER = 6.0


def _openstack_datetime_to_unix_ms(date_s: str, time_s: str) -> int | None:
    """Convert OpenStack date and time fragments to Unix milliseconds.

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
    """Return the OpenStack instance identifier, if present."""
    for pattern in INSTANCE_ID_PATTERNS:
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
            if UUID_RE.fullmatch(segment):
                normalised_segments.append("UUID")
                continue
            if IP_RE.fullmatch(segment):
                normalised_segments.append("IP")
                continue
            if HEX_RE.fullmatch(segment):
                normalised_segments.append("HEX")
                continue
            if not preserve_numeric_values and re.fullmatch(r"\d+(?:\.\d+)?", segment):
                normalised_segments.append("NUM")
                continue
            normalised_segments.append(segment)
        return "/".join(normalised_segments) + suffix

    return PATH_RE.sub(_normalise_path, text)


def normalise_openstack_message(
    content: str,
    *,
    preserve_numeric_values: bool,
) -> str:
    """Canonicalise OpenStack message text before template mining.

    Returns:
        str: Canonicalised OpenStack message text.
    """
    text = INSTANCE_PREFIX_RE.sub("", content).strip()
    text = UUID_RE.sub("UUID", text)
    text = IP_RE.sub("IP", text)
    text = _normalise_openstack_path_tokens(
        text,
        preserve_numeric_values=preserve_numeric_values,
    )
    text = INSTANCE_STORAGE_PATH_RE.sub("INSTANCE_PATH", text)
    text = HEX_RE.sub("HEX", text)
    if preserve_numeric_values:
        return text
    return NUM_RE.sub("NUM", text)


def extract_openstack_parameters(content: str) -> list[str]:
    """Extract raw numeric parameter tokens from an OpenStack message body.

    Returns:
        list[str]: Raw numeric tokens in encounter order.
    """
    text = INSTANCE_PREFIX_RE.sub("", content).strip()
    return NUM_RE.findall(text)


def parse_openstack_payload(raw_payload: str) -> OpenStackParsedPayload | None:
    """Parse a raw OpenStack payload into timestamp, instance, and content.

    Returns:
        OpenStackParsedPayload | None: Parsed payload fields, or `None` when
            the payload does not match the expected OpenStack shape.
    """
    match = OPENSTACK_RE.match(raw_payload)
    if match is None:
        return None
    data = match.groupdict()
    timestamp_ms = _openstack_datetime_to_unix_ms(data["date"], data["time"])
    if timestamp_ms is None:
        return None
    instance_id = _extract_openstack_instance_id(raw_payload)
    if instance_id is None:
        return None
    return OpenStackParsedPayload(
        timestamp_unix_ms=timestamp_ms,
        instance_id=instance_id,
        content=data["content"].strip(),
        raw_parameters=extract_openstack_parameters(data["content"] or ""),
    )


def _parse_openstack_parameter_line(
    raw_line: str,
) -> tuple[str, _OpenStackParameterEvent] | None:
    raw_payload = raw_line.rstrip("\n")
    parsed = parse_openstack_payload(raw_payload)
    if parsed is None:
        return None
    return (
        parsed.instance_id,
        _OpenStackParameterEvent(
            raw_payload=raw_payload,
            timestamp_ms=parsed.timestamp_unix_ms,
            content=parsed.content,
        ),
    )


def _normalised_template(content: str) -> str:
    return normalise_openstack_message(content, preserve_numeric_values=False)


def _find_event_index(
    events: list[_OpenStackParameterEvent],
    target_template: str,
) -> int:
    for event_index, event in enumerate(events):
        if _normalised_template(event.content) == target_template:
            return event_index
    return -1


def _resolve_anomaly_instance_index(
    instances: list[_OpenStackParameterInstance],
    *,
    start_index: int,
    offset: int,
    target_template: str,
) -> int:
    for instance_index in range(start_index + offset, len(instances)):
        if _find_event_index(instances[instance_index].events, target_template) >= 0:
            return instance_index
    msg = (
        f"Could not find a held-out OpenStack instance containing {target_template!r}."
    )
    raise ValueError(msg)


def _resolve_anomaly_instance_indexes(
    instances: list[_OpenStackParameterInstance],
    *,
    start_index: int,
    offsets: tuple[int, int],
    target_template: str,
) -> tuple[int, int]:
    """Return two held-out instance indexes that expose the target template.

    DeepLog Figure 9 shows two injected time points, and each affected key
    should observe both of them.

    Raises:
        ValueError: If the two requested anomaly instances collapse to the same
            held-out index.
    """
    first_offset, second_offset = offsets
    first_index = _resolve_anomaly_instance_index(
        instances,
        start_index=start_index,
        offset=first_offset,
        target_template=target_template,
    )
    second_index = _resolve_anomaly_instance_index(
        instances,
        start_index=start_index,
        offset=second_offset,
        target_template=target_template,
    )
    if second_index == first_index:
        msg = f"Expected two distinct OpenStack instances for {target_template!r}."
        raise ValueError(msg)
    return first_index, second_index


def _split_name_for_instance(
    *,
    instance_index: int,
    train_instances: int,
    validation_instances: int,
) -> str:
    if instance_index < train_instances:
        return "openstack_train"
    if instance_index < train_instances + validation_instances:
        return "openstack_validation"
    return "openstack_test"


def _multiply_build_seconds(content: str, multiplier: float) -> str:
    stripped = re.sub(
        r"^\[instance:\s*[^\]]+\]\s*",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()
    match = _BUILD_SECONDS_RE.match(stripped)
    if match is None:
        return content
    seconds = float(match.group("seconds")) * multiplier
    replacement = f"Took {seconds:g} seconds to build instance."
    prefix_match = re.match(
        r"^(?P<prefix>\[instance:\s*[^\]]+\]\s*)",
        content,
        flags=re.IGNORECASE,
    )
    if prefix_match is None:
        return replacement
    return f"{prefix_match.group('prefix')}{replacement}"


def _rebuild_openstack_line(
    event: _OpenStackParameterEvent,
    timestamp_ms: int,
    content: str,
) -> str:
    timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    timestamp_text = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")
    match = OPENSTACK_RE.match(event.raw_payload)
    if match is None:
        return event.raw_payload
    return (
        f"{match.group('logrecord')} {timestamp_text} {match.group('pid')} "
        f"{match.group('level')} {match.group('component')} "
        f"[{match.group('addr')}] {content}"
    )


def materialise_openstack_deeplog_parameter_ci_subset(
    source_root: Path,
    raw_logs_path: Path,
) -> None:
    """Materialise a Figure 9-sized OpenStack VM-creation slice with anomalies.

    Raises:
        ValueError: If the archive does not expose enough normal VM instances
            for the configured train, validation, and anomaly offsets.
    """
    if _OPENSTACK_PARAMETER_TRAIN_INSTANCES < 1:
        msg = "train_instances must be positive."
        raise ValueError(msg)
    instances = _load_openstack_parameter_instances(source_root=source_root)
    required_instance_count = (
        _OPENSTACK_PARAMETER_TRAIN_INSTANCES
        + _OPENSTACK_PARAMETER_VALIDATION_INSTANCES
        + max(_OPENSTACK_PARAMETER_ANOMALY_INSTANCE_OFFSETS)
        + 1
    )
    min_instances = max(
        required_instance_count,
        _OPENSTACK_PARAMETER_SUBSET_INSTANCES,
    )
    if len(instances) < min_instances:
        msg = f"Expected at least {min_instances} normal OpenStack instances."
        raise ValueError(msg)

    selected_instances = instances[:_OPENSTACK_PARAMETER_SUBSET_INSTANCES]
    test_start = (
        _OPENSTACK_PARAMETER_TRAIN_INSTANCES + _OPENSTACK_PARAMETER_VALIDATION_INSTANCES
    )
    anomaly_indexes = _resolve_anomaly_instance_indexes(
        selected_instances,
        start_index=test_start,
        offsets=_OPENSTACK_PARAMETER_ANOMALY_INSTANCE_OFFSETS,
        target_template=_PERFORMANCE_TARGET_TEMPLATE,
    )
    for anomaly_index in anomaly_indexes:
        if (
            _find_event_index(
                selected_instances[anomaly_index].events,
                _DURATION_TARGET_TEMPLATE,
            )
            < 0
        ):
            msg = (
                "Could not find the build-duration template on held-out OpenStack "
                f"instance index {anomaly_index}."
            )
            raise ValueError(msg)

    with raw_logs_path.open("w", encoding="utf-8") as output:
        for instance_index, instance in enumerate(selected_instances):
            split_name = _split_name_for_instance(
                instance_index=instance_index,
                train_instances=_OPENSTACK_PARAMETER_TRAIN_INSTANCES,
                validation_instances=_OPENSTACK_PARAMETER_VALIDATION_INSTANCES,
            )
            performance_target_index = (
                _find_event_index(instance.events, _PERFORMANCE_TARGET_TEMPLATE)
                if instance_index in anomaly_indexes
                else -1
            )
            duration_target_index = (
                _find_event_index(instance.events, _DURATION_TARGET_TEMPLATE)
                if instance_index in anomaly_indexes
                else -1
            )
            for event_index, event in enumerate(instance.events):
                label = 0
                timestamp_ms = event.timestamp_ms
                content = event.content
                if (
                    performance_target_index >= 0
                    and event_index >= performance_target_index
                ):
                    timestamp_ms += _OPENSTACK_PARAMETER_PERFORMANCE_DELAY_MS
                    if event_index == performance_target_index:
                        label = 1
                if event_index == duration_target_index and duration_target_index >= 0:
                    content = _multiply_build_seconds(
                        content,
                        _OPENSTACK_PARAMETER_DURATION_MULTIPLIER,
                    )
                    label = 1
                output.write(
                    f"{split_name}\t{label}\t"
                    f"{_rebuild_openstack_line(event, timestamp_ms, content)}\n",
                )


def _find_source_file(dataset_root: Path, split_name: str) -> Path | None:
    for candidate in dataset_root.rglob(split_name):
        if candidate.is_file():
            return candidate
    return None


def _load_openstack_parameter_instances(
    *,
    source_root: Path,
) -> list[_OpenStackParameterInstance]:
    normal_source_files = (
        "openstack_normal1.log",
        "openstack_normal2.log",
    )
    grouped: dict[str, list[_OpenStackParameterEvent]] = {}
    for source_name in normal_source_files:
        source_path = _find_source_file(source_root, source_name)
        if source_path is None:
            msg = f"Missing {source_name} in extracted archive at {source_root}."
            raise FileNotFoundError(msg)
        with source_path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                parsed = _parse_openstack_parameter_line(raw_line)
                if parsed is None:
                    continue
                instance_id, event = parsed
                grouped.setdefault(instance_id, []).append(event)

    ordered: list[_OpenStackParameterInstance] = []
    for instance_id, events in grouped.items():
        sorted_events = sorted(
            events,
            key=lambda event: (event.timestamp_ms, event.content),
        )
        if not sorted_events:
            continue
        ordered.append(
            _OpenStackParameterInstance(
                instance_id=instance_id,
                timestamp_ms=sorted_events[0].timestamp_ms,
                events=sorted_events,
            ),
        )
    ordered.sort(key=lambda item: (item.timestamp_ms, item.instance_id))
    return ordered
