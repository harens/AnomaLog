# Thunderbird Implementation Plan

- Reuse the shared raw-log builder path rather than adding a new pipeline.
- Add a Thunderbird structured parser that treats `-` as normal and any other
  label token as anomalous.
- Use the canonical `Thunderbird.log` archive member directly for the
  reproduction preset, but cap it at the `160,000,000`-`170,000,000` raw-line
  slice used by the public benchmark code; keep a shorter smoke prefix for
  local development.
- Keep the default Thunderbird experiment shape on 100-log chronological
  windows for the DeepLog-style reproduction, then expose a separate
  entity-grouped Thunderbird extension for DeepCASE and shared baselines.
- Extend the dataset manifest statistics with Thunderbird-specific parsing and
  split counts without changing the shared sequence/model code.
