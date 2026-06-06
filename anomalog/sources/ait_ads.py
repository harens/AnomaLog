"""AIT-ADS dataset source and canonical alert materialisation helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar
from urllib.request import urlretrieve

from anomalog.io_utils import verify_md5
from anomalog.sources.contracts import DatasetSource
from anomalog.sources.remote_zip import RemoteZipSource

AIT_ADS_ARCHIVE_URL = "https://zenodo.org/records/8263181/files/ait_ads.zip"
AIT_ADS_ARCHIVE_MD5 = "43db6b1f0996e0024befd617706c50e9"
AIT_ADS_LABELS_URL = "https://zenodo.org/records/8263181/files/labels.csv"
AIT_ADS_LABELS_MD5 = "60ff33796c77fd2136c4d1a4bc841bd9"
AIT_ADS_SCENARIOS = (
    "fox",
    "harrison",
    "russellmitchell",
    "santos",
    "shaw",
    "wardbeck",
    "wheeler",
    "wilson",
)
_AMINER_SOURCE = "aminer"
_SURICATA_SOURCE = "suricata"
_WAZUH_SOURCE = "wazuh"


@dataclass(frozen=True, slots=True)
class _LabelWindow:
    """One labelled attack interval from the AIT-ADS labels file.

    Attributes:
        attack (str): Attack phase name assigned to the interval.
        start_unix_s (float): Inclusive interval start in Unix seconds.
        end_unix_s (float): Exclusive interval end in Unix seconds.
    """

    attack: str
    start_unix_s: float
    end_unix_s: float


@dataclass(frozen=True, slots=True)
class _CanonicalAlert:
    """Normalised alert used to build the canonical AIT-ADS raw stream.

    Attributes:
        scenario (str): Scenario name the alert belongs to.
        ids_source (str): Alert family, such as `aminer`, `suricata`, or `wazuh`.
        timestamp_unix_us (int | None): Original timestamp in Unix microseconds,
            if present.
        timestamp_unix_ms (int | None): Original timestamp in Unix milliseconds,
            if present.
        source_line_order (int): Zero-based order within the source file.
        source_file (str): Source filename that contributed the alert.
        entity_id (str): Canonical entity identifier used for grouping.
        template_key (str): Canonical template key emitted into the raw stream.
        anomalous (int): Binary anomaly label derived from the label windows.
        attack_phase (str | None): Named attack phase for labelled anomaly
            intervals.
        original_timestamp (str | None): Timestamp text preserved from the source
            alert.
        alert_uid (str): Stable unique identifier for the emitted record.
        alert_signature (str | None): Source-specific signature text, when
            available.
        metadata (dict[str, object]): Additional source metadata preserved for
            auditing.
    """

    scenario: str
    ids_source: str
    timestamp_unix_us: int | None
    timestamp_unix_ms: int | None
    source_line_order: int
    source_file: str
    entity_id: str
    template_key: str
    anomalous: int
    attack_phase: str | None
    original_timestamp: str | None
    alert_uid: str
    alert_signature: str | None
    metadata: dict[str, object]

    def sort_key(self) -> tuple[int, int, str, str, int]:
        """Return a deterministic chronological ordering key.

        The dataset ships AMiner and Wazuh alerts in separate files. Both files
        are already internally timestamp-ordered. We therefore merge primarily
        by timestamp, then fall back to source file name plus per-file line
        order only when the dataset provides no finer cross-file order.

        Returns:
            tuple[int, int, str, str, int]: Ordering key used when merging the
                canonical alerts across files.
        """
        return (
            1 if self.timestamp_unix_us is None else 0,
            0 if self.timestamp_unix_us is None else self.timestamp_unix_us,
            self.scenario,
            self.source_file,
            self.source_line_order,
        )

    def as_record(self) -> dict[str, object]:
        """Return the JSON-serialisable canonical alert representation.

        Returns:
            dict[str, object]: Canonical alert payload written to the JSONL
                stream.
        """
        return {
            "scenario": self.scenario,
            "ids_source": self.ids_source,
            "timestamp_unix_ms": self.timestamp_unix_ms,
            "source_line_order": self.source_line_order,
            "source_file": self.source_file,
            "entity_id": self.entity_id,
            "template_key": self.template_key,
            "anomalous": self.anomalous,
            "attack_phase": self.attack_phase,
            "original_timestamp": self.original_timestamp,
            "alert_uid": self.alert_uid,
            "alert_signature": self.alert_signature,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class AITADSScenarioSource(DatasetSource):
    """Materialise one or more AIT-ADS scenarios into a canonical alert stream.

    Attributes:
        name (ClassVar[str]): Registry/config name for the built-in source.
        scenario_names (tuple[str, ...]): Ordered scenario names selected for
            materialisation.
        base_source (DatasetSource): Archive source that provides the extracted
            AIT-ADS files.
        labels_relpath (Path): Relative path to the published scenario label CSV.
        labels_url (str): Download URL for the label CSV when it is missing.
        labels_md5_checksum (str): Expected MD5 checksum for the label CSV.
        raw_logs_relpath (Path | None): Optional relative path for the derived
            JSONL stream.
    """

    name: ClassVar[str] = "ait_ads"
    scenario_names: tuple[str, ...] = AIT_ADS_SCENARIOS
    base_source: DatasetSource = field(
        default_factory=lambda: RemoteZipSource(
            url=AIT_ADS_ARCHIVE_URL,
            md5_checksum=AIT_ADS_ARCHIVE_MD5,
        ),
    )
    labels_relpath: Path = Path("labels.csv")
    labels_url: str = AIT_ADS_LABELS_URL
    labels_md5_checksum: str = AIT_ADS_LABELS_MD5
    raw_logs_relpath: Path | None = Path("preprocessed/ait_ads_alerts.jsonl")

    def materialise(
        self,
        *,
        dst_dir: Path,
    ) -> Path:
        """Materialise the archive, labels, and canonical alert stream.

        Args:
            dst_dir (Path): Dataset root used for archive extraction and the
                derived canonical alert stream.

        Returns:
            Path: Materialised dataset root containing the upstream files plus
                the derived canonical alert stream.
        """
        dataset_root = self.base_source.materialise(dst_dir=dst_dir)
        labels_path = dataset_root / self.labels_relpath
        self._ensure_labels_file(labels_path)
        raw_logs_path = self._derived_raw_logs_path(
            dataset_name=dst_dir.name,
            dataset_root=dataset_root,
        )
        if raw_logs_path.is_file():
            return dataset_root
        raw_logs_path.parent.mkdir(parents=True, exist_ok=True)
        materialise_ait_ads_alert_stream(
            source_root=dataset_root,
            labels_path=labels_path,
            raw_logs_path=raw_logs_path,
            scenarios=self.scenario_names,
        )
        return dataset_root

    def _ensure_labels_file(self, labels_path: Path) -> None:
        if labels_path.exists():
            verify_md5(labels_path, self.labels_md5_checksum)
            return

        labels_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(  # noqa: S310 - fixed Zenodo URL owned by the dataset preset
            self.labels_url,
            labels_path,
        )
        verify_md5(labels_path, self.labels_md5_checksum)

    def _derived_raw_logs_path(self, *, dataset_name: str, dataset_root: Path) -> Path:
        if self.raw_logs_relpath is None:
            return dataset_root / f"{dataset_name}.log"
        if self.raw_logs_relpath.is_absolute():
            msg = "raw_logs_relpath must be relative to the dataset root."
            raise ValueError(msg)
        candidate = dataset_root / self.raw_logs_relpath
        resolved_root = dataset_root.resolve()
        resolved_candidate = candidate.resolve(strict=False)
        try:
            resolved_candidate.relative_to(resolved_root)
        except ValueError as exc:
            msg = "raw_logs_relpath must stay within the dataset root."
            raise ValueError(msg) from exc
        return candidate


def materialise_ait_ads_alert_stream(
    *,
    source_root: Path,
    labels_path: Path,
    raw_logs_path: Path,
    scenarios: tuple[str, ...],
) -> None:
    """Build one canonical AIT-ADS JSONL stream from selected scenarios.

    Args:
        source_root (Path): Extracted AIT-ADS archive root.
        labels_path (Path): Scenario label CSV published beside the archive.
        raw_logs_path (Path): Output path for the canonical alert JSONL stream.
        scenarios (tuple[str, ...]): Scenario identifiers to include.

    Raises:
        ValueError: If any requested scenario is not part of AIT-ADS.
    """
    scenario_set = set(scenarios)
    invalid = sorted(scenario_set.difference(AIT_ADS_SCENARIOS))
    if invalid:
        msg = f"Unsupported AIT-ADS scenarios: {invalid!r}"
        raise ValueError(msg)

    labels_by_scenario = load_ait_ads_label_windows(labels_path)
    selected = sorted(scenario_set, key=AIT_ADS_SCENARIOS.index)

    alerts: list[_CanonicalAlert] = []
    for scenario in selected:
        scenario_windows = labels_by_scenario[scenario]
        assigner = _IntervalLabelAssigner(scenario_windows)

        for source_name in (_AMINER_SOURCE, _WAZUH_SOURCE):
            source_file = find_scenario_file(source_root, scenario, source_name)
            alerts.extend(
                iter_canonical_alerts(
                    scenario=scenario,
                    source_name=source_name,
                    source_file=source_file,
                    assigner=assigner,
                ),
            )

    alerts.sort(key=_CanonicalAlert.sort_key)
    with raw_logs_path.open("w", encoding="utf-8") as output:
        for alert in alerts:
            output.write(
                json.dumps(
                    alert.as_record(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            output.write("\n")


def load_ait_ads_label_windows(
    labels_path: Path,
) -> dict[str, tuple[_LabelWindow, ...]]:
    """Load and validate per-scenario AIT-ADS label windows.

    Args:
        labels_path (Path): Path to the published AIT-ADS label CSV.

    Returns:
        dict[str, tuple[_LabelWindow, ...]]: Sorted, non-overlapping label
            windows keyed by scenario name.

    Raises:
        ValueError: If any scenario contains overlapping label windows.
    """
    by_scenario: dict[str, list[_LabelWindow]] = {}
    with labels_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scenario = str(row["scenario"]).strip()
            by_scenario.setdefault(scenario, []).append(
                _LabelWindow(
                    attack=str(row["attack"]).strip(),
                    start_unix_s=float(row["start"]),
                    end_unix_s=float(row["end"]),
                ),
            )

    resolved: dict[str, tuple[_LabelWindow, ...]] = {}
    for scenario, windows in by_scenario.items():
        ordered = tuple(
            sorted(
                windows,
                key=lambda window: (
                    window.start_unix_s,
                    window.end_unix_s,
                    window.attack,
                ),
            ),
        )
        previous_end = None
        for window in ordered:
            if previous_end is not None and window.start_unix_s < previous_end:
                msg = f"Overlapping label windows detected for scenario {scenario!r}."
                raise ValueError(msg)
            previous_end = window.end_unix_s
        resolved[scenario] = ordered
    return resolved


def find_scenario_file(source_root: Path, scenario: str, source_name: str) -> Path:
    """Resolve the extracted upstream file for one scenario/source pair.

    Args:
        source_root (Path): Extracted archive root.
        scenario (str): Scenario name to locate.
        source_name (str): Source suffix such as `aminer` or `wazuh`.

    Returns:
        Path: Matching source file path inside the extracted archive.

    Raises:
        FileNotFoundError: If the expected source file does not exist.
    """
    expected_name = f"{scenario}_{source_name}.json"
    direct = source_root / expected_name
    if direct.is_file():
        return direct
    for candidate in source_root.rglob(expected_name):
        if candidate.is_file():
            return candidate
    msg = f"Missing AIT-ADS source file {expected_name!r} under {source_root}."
    raise FileNotFoundError(msg)


@dataclass(frozen=True, slots=True)
class _IntervalLabelAssigner:
    """Apply half-open `[start, end)` interval labels to alert timestamps.

    Attributes:
        windows (tuple[_LabelWindow, ...]): Ordered half-open label windows
            used for timestamp assignment.
    """

    windows: tuple[_LabelWindow, ...]

    def assign(self, timestamp_unix_ms: int | None) -> tuple[int, str | None]:
        if timestamp_unix_ms is None:
            return 0, None
        timestamp_unix_s = timestamp_unix_ms / 1000.0
        for window in self.windows:
            if window.start_unix_s <= timestamp_unix_s < window.end_unix_s:
                return 1, window.attack
        return 0, None


def iter_canonical_alerts(
    *,
    scenario: str,
    source_name: str,
    source_file: Path,
    assigner: _IntervalLabelAssigner,
) -> list[_CanonicalAlert]:
    """Yield canonical alerts from one upstream AIT-ADS source file.

    Args:
        scenario (str): Scenario name for the source file.
        source_name (str): Source family, either `aminer` or `wazuh`.
        source_file (Path): Source file to iterate.
        assigner (_IntervalLabelAssigner): Interval-to-label mapper.

    Returns:
        list[_CanonicalAlert]: Canonical alerts in the source file's native
            order prior to cross-file merging.
    """
    alerts: list[_CanonicalAlert] = []
    with source_file.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if source_name == _AMINER_SOURCE:
                alerts.append(
                    canonicalise_aminer_alert(
                        scenario=scenario,
                        obj=obj,
                        source_file=source_file.name,
                        source_line_order=line_number,
                        assigner=assigner,
                    ),
                )
            else:
                alerts.append(
                    canonicalise_wazuh_alert(
                        scenario=scenario,
                        obj=obj,
                        source_file=source_file.name,
                        source_line_order=line_number,
                        assigner=assigner,
                    ),
                )
    return alerts


def canonicalise_aminer_alert(
    *,
    scenario: str,
    obj: dict[str, object],
    source_file: str,
    source_line_order: int,
    assigner: _IntervalLabelAssigner,
) -> _CanonicalAlert:
    """Canonicalise one AMiner JSON alert.

    Args:
        scenario (str): Scenario name for the alert.
        obj (dict[str, object]): Parsed AMiner JSON object.
        source_file (str): Source filename the alert came from.
        source_line_order (int): Zero-based line order within `source_file`.
        assigner (_IntervalLabelAssigner): Interval-to-label mapper.

    Returns:
        _CanonicalAlert: Canonical alert representation for downstream parsing.
    """
    component = _dict(obj.get("AnalysisComponent"))
    log_data = _dict(obj.get("LogData"))
    aminer_meta = _dict(obj.get("AMiner"))
    raw_timestamps = log_data.get("Timestamps")
    timestamp_unix_us = _first_epoch_us_from_list(raw_timestamps)
    timestamp_unix_ms = None if timestamp_unix_us is None else timestamp_unix_us // 1000
    anomalous, attack_phase = assigner.assign(timestamp_unix_ms)
    component_type = _text(component.get("AnalysisComponentType"), default="unknown")
    component_name = _text(component.get("AnalysisComponentName"), default="unknown")
    persistence_name = _text(component.get("PersistenceFileName"), default="unknown")
    entity_suffix = _text(aminer_meta.get("ID"), default="unknown")

    return _CanonicalAlert(
        scenario=scenario,
        ids_source=_AMINER_SOURCE,
        timestamp_unix_us=timestamp_unix_us,
        timestamp_unix_ms=timestamp_unix_ms,
        source_line_order=source_line_order,
        source_file=source_file,
        entity_id=f"{scenario}:{_AMINER_SOURCE}:{entity_suffix}",
        template_key=(
            f"aminer|type={component_type}|name={component_name}|key={persistence_name}"
        ),
        anomalous=anomalous,
        attack_phase=attack_phase,
        original_timestamp=(
            None
            if not isinstance(raw_timestamps, list) or not raw_timestamps
            else str(raw_timestamps[0])
        ),
        alert_uid=f"{scenario}:{source_file}:{source_line_order}",
        alert_signature=persistence_name,
        metadata={
            "analysis_component_identifier": component.get(
                "AnalysisComponentIdentifier",
            ),
            "analysis_component_type": component_type,
            "analysis_component_name": component_name,
            "message": component.get("Message"),
            "persistence_file_name": persistence_name,
            "aminer_id": aminer_meta.get("ID"),
            "raw_log_resources": log_data.get("LogResources"),
            "raw_log_line_count": log_data.get("LogLinesCount"),
            "raw_timestamps": raw_timestamps,
        },
    )


def canonicalise_wazuh_alert(
    *,
    scenario: str,
    obj: dict[str, object],
    source_file: str,
    source_line_order: int,
    assigner: _IntervalLabelAssigner,
) -> _CanonicalAlert:
    """Canonicalise one Wazuh or Suricata-origin JSON alert.

    Args:
        scenario (str): Scenario name for the alert.
        obj (dict[str, object]): Parsed Wazuh JSON object.
        source_file (str): Source filename the alert came from.
        source_line_order (int): Zero-based line order within `source_file`.
        assigner (_IntervalLabelAssigner): Interval-to-label mapper.

    Returns:
        _CanonicalAlert: Canonical alert representation for downstream parsing.
    """
    rule = _dict(obj.get("rule"))
    decoder = _dict(obj.get("decoder"))
    agent = _dict(obj.get("agent"))
    data = _dict(obj.get("data"))
    timestamp_raw = _text(obj.get("@timestamp"))
    timestamp_unix_us = _parse_iso_timestamp_us(timestamp_raw)
    timestamp_unix_ms = None if timestamp_unix_us is None else timestamp_unix_us // 1000
    anomalous, attack_phase = assigner.assign(timestamp_unix_ms)
    ids_source = classify_wazuh_ids_source(obj)
    entity_suffix = _text(
        agent.get("id"),
        default=_text(agent.get("ip"), default="unknown"),
    )
    template_key, alert_signature = _wazuh_template_fields(
        obj=obj,
        rule=rule,
        decoder=decoder,
        data=data,
        ids_source=ids_source,
    )

    event_id = _text(obj.get("id"), default=str(source_line_order))

    return _CanonicalAlert(
        scenario=scenario,
        ids_source=ids_source,
        timestamp_unix_us=timestamp_unix_us,
        timestamp_unix_ms=timestamp_unix_ms,
        source_line_order=source_line_order,
        source_file=source_file,
        entity_id=f"{scenario}:{ids_source}:{entity_suffix}",
        template_key=template_key,
        anomalous=anomalous,
        attack_phase=attack_phase,
        original_timestamp=timestamp_raw,
        alert_uid=f"{scenario}:{source_file}:{event_id}",
        alert_signature=alert_signature,
        metadata={
            "agent_id": agent.get("id"),
            "agent_ip": agent.get("ip"),
            "agent_name": agent.get("name"),
            "decoder_name": decoder.get("name"),
            "location": obj.get("location"),
            "rule_id": rule.get("id"),
            "rule_level": rule.get("level"),
            "rule_description": rule.get("description"),
            "rule_groups": rule.get("groups"),
            "suricata_signature_id": _dict(data.get("alert")).get("signature_id"),
            "suricata_signature": _dict(data.get("alert")).get("signature"),
            "suricata_category": _dict(data.get("alert")).get("category"),
            "wazuh_event_id": obj.get("id"),
        },
    )


def _wazuh_template_fields(
    *,
    obj: dict[str, object],
    rule: dict[str, object],
    decoder: dict[str, object],
    data: dict[str, object],
    ids_source: str,
) -> tuple[str, str | None]:
    """Return the canonical alert key and signature for one Wazuh-family row.

    Args:
        obj (dict[str, object]): Full parsed alert object.
        rule (dict[str, object]): Normalised rule sub-document extracted from
            `obj`.
        decoder (dict[str, object]): Normalised decoder sub-document extracted
            from `obj`.
        data (dict[str, object]): Normalised data sub-document extracted from
            `obj`.
        ids_source (str): Resolved IDS source family for the alert.

    Returns:
        tuple[str, str | None]: Canonical template key and signature text.
    """
    if ids_source == _SURICATA_SOURCE:
        alert = _dict(data.get("alert"))
        signature_id = _text(
            alert.get("signature_id"),
            default=_text(
                data.get("id"),
                default=_text(rule.get("id"), default="unknown"),
            ),
        )
        signature = _text(
            alert.get("signature"),
            default=_text(rule.get("description"), default="unknown"),
        )
        category = _text(
            alert.get("category"),
            default=_text(rule.get("description"), default="unknown"),
        )
        event_type = _text(
            data.get("event_type"),
            default=_text(decoder.get("name"), default="unknown"),
        )
        return (
            f"suricata|signature_id={signature_id}|category={category}|event_type={event_type}",
            signature,
        )

    rule_id = _text(rule.get("id"), default="unknown")
    decoder_name = _text(decoder.get("name"), default="unknown")
    location_name = _path_leaf(_text(obj.get("location"), default="unknown"))
    description = _text(rule.get("description"), default="unknown")
    return (
        f"wazuh|rule_id={rule_id}|decoder={decoder_name}|location={location_name}|description={description}",
        description,
    )


def classify_wazuh_ids_source(obj: dict[str, object]) -> str:
    """Return whether a Wazuh-file alert should be treated as Wazuh or Suricata.

    Args:
        obj (dict[str, object]): Parsed JSON alert row from the Wazuh-family
            source files.

    Returns:
        str: Either `wazuh` or `suricata` depending on the alert shape.
    """
    location = _text(obj.get("location"))
    if location is not None and "/suricata/" in location:
        return _SURICATA_SOURCE
    rule = _dict(obj.get("rule"))
    group_values = rule.get("groups")
    if isinstance(group_values, list) and _SURICATA_SOURCE in group_values:
        return _SURICATA_SOURCE
    return _WAZUH_SOURCE


def _dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _text(value: object, *, default: str | None = None) -> str | None:
    if value is None:
        return default
    return str(value)


def _path_leaf(path: str | None) -> str:
    return "unknown" if path is None else Path(path).name


def _first_epoch_us_from_list(value: object) -> int | None:
    if not isinstance(value, list) or not value:
        return None
    try:
        return int(float(str(value[0])) * 1_000_000)
    except (TypeError, ValueError):
        return None


def _parse_iso_timestamp_us(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(
            datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            * 1_000_000,
        )
    except ValueError:
        return None
