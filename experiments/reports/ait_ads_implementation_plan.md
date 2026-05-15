## AIT-ADS implementation plan

- Materialise AIT-ADS through one dataset-specific source that:
  - downloads `ait_ads.zip` and `labels.csv`,
  - verifies the published MD5 checksums,
  - selects one or more scenarios,
  - converts the heterogeneous AMiner and Wazuh/Suricata JSONL files into one canonical alert JSONL stream,
  - writes a sidecar statistics JSON file for dataset-manifest reporting.
- Keep the structured parser narrow by parsing that canonical alert JSONL format rather than re-implementing every raw alert schema in the parser itself.
- Use inline labels derived during source materialisation:
  - binary `anomalous` label from `labels.csv` time windows,
  - attack phase preserved as metadata,
  - half-open interval semantics `[start, end)` to avoid double-labelling alerts that land on adjacent boundaries.
- Represent alert keys with stable detector semantics instead of volatile raw messages:
  - AMiner: analysis component type/name/persistence key,
  - Suricata: signature/rule/category style identifiers,
  - Wazuh: rule/decoder/location style identifiers.
- Default AIT-ADS experiment configs to chronological stream grouping so the alert stream stays paper-faithful and does not invent host/session boundaries across heterogeneous IDS sources.
- Add scenario-specific presets/configs for `fox`, `harrison`, `russellmitchell`, `santos`, `shaw`, `wardbeck`, `wheeler`, and `wilson`.
