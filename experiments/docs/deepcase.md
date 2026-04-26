# DeepCase Integration

This experiment detector integrates the official `deepcase` package with
AnomaLog's sequence-oriented experiment runner.

## Implemented Paper Components

- Event-centered analysis: every event in an entity-local `TemplateSequence`
  becomes one DeepCase `(context, event)` sample.
- Same-entity context: DeepCase rejects sequences that already contain multiple
  entity ids.
- Paper defaults: context length `10`, timeout `86400` seconds, hidden size
  `128`, confidence threshold `0.2`, DBSCAN epsilon `0.1`, and minimum cluster
  size `5`.
- Official Context Builder, attention query, total-attention vectorization,
  DBSCAN clustering, manual-mode cluster scoring, and semi-automatic prediction
  are delegated to the installed `deepcase` library.
- DeepCase special prediction codes are preserved in event findings:
  `-1` for low confidence, `-2` for unknown event, and `-3` for outside epsilon.

## AnomaLog Adaptations

AnomaLog experiment labels are sequence/entity labels. During DeepCase cluster
scoring, every event-centered sample derived from a sequence inherits that
sequence's label. This keeps the shared `TemplateSequence` interface unchanged
and avoids adding event-level labels that the rest of the experiment layer does
not currently use.

The experiment runner is non-interactive. Ground-truth labels therefore stand in
for the operator-provided labels that DeepCase would receive during manual
analysis. Predictions are still emitted as sequence records for the common
metrics contract, with detector-owned event findings preserving DeepCase's
event-level decisions.

The model should be run with entity grouping:

```toml
[sequence]
grouping = "entity"
```

The detector validates the observable invariant by rejecting sequences that span
multiple entity ids.

## Remaining Gaps

- No interactive operator labeling workflow.
- No persistent cluster database shared across experiment runs.
- No online update loop for newly inspected clusters or outliers.
