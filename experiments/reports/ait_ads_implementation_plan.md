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
- Keep the combined chronology in the normal `ait_ads` preset: one dataset
  manifest should merge all eight scenarios into a single alert stream,
  sorted globally by timestamp, and expose the continuous event stream as the
  evaluation unit.
- Keep the scenario-specific and short-window presets only as diagnostics for
  local analysis and contamination checks.
- Avoid a second paper-specific preset or manifest name; the combined path
  should be the default `ait_ads` contract.

## Research notes

- The AACT paper’s AIT-ADS evaluation uses the combined dataset chronologically,
  not per-scenario evaluation. The paper explicitly says the eight tenant
  scenarios are combined and arranged in chronological order.
- The paper says it uses two-fold time-series cross-validation because the data
  is large and the authors wanted enough malicious samples in each fold.
- The AIT comparison in the paper is not an "all baselines" suite. It uses one
  simple baseline, the global category malicious/investigation rate over a
  30-day lookback, plus DeepCase.
- The dataset entry says `event_label` is the more precise label source and
  should be preferred over the coarser `time_label` intervals when available.
- The dataset entry also says Wazuh timestamps must be normalised using the ISO
  `@timestamp`, not the epoch-like `id` field.
- Local audit of the canonicalised combined stream found `2,655,821` alerts and
  `1,764,581` positives.
- Chronological bucket experiments on the combined stream showed that even
  5-fold and 8-fold splits still contain substantial numbers of positives in
  every fold. This makes the paper’s two-fold choice a conservative evaluation
  decision, not a hard data-scarcity limit.
- A temporary historical-rate baseline over the same chronology performed very
  strongly, which supports adding an AIT-specific baseline if we want a more
  appropriate non-paper comparator than the generic repository baselines.
- AIT-ADS should keep the sequence-level chunk label only as an internal
  container label. The public report should promote alert-level metrics only,
  because the paper evaluates alert triage rather than chunk classification.

## Split recommendation

- Do not change the default AIT manifest to a `20% / 80%` split if the goal is
  paper compatibility. That would move the configuration further away from the
  paper’s two-fold time-series cross-validation protocol.
- If we must keep a single holdout approximation, `50% / 50%` is the closer
  stand-in for the paper than `20% / 80%`.
- If we want the best match, the real fix is to add a CV-aware AIT experiment
  path rather than reinterpreting the paper as a fixed holdout split.
