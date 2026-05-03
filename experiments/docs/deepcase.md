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
  `-1` for low confidence, `-2` for unknown event, and `-3` for outside
  epsilon.

## AnomaLog Adaptations

AnomaLog experiment labels are sequence/entity labels, but `TemplateSequence`
now preserves optional event-level labels when they exist. During DeepCase
sample construction, each event-centered sample uses the target event label
when one is present and falls back to the parent sequence label only when the
event label is missing. Run metrics record how often DeepCASE had to fall back
to the parent sequence label.

DeepCase training reports progress per context-builder epoch before moving on
to interpreter clustering. That keeps long training runs visibly alive instead
of appearing to stall once sequence preparation has finished.

Test-time DeepCase scoring now uses the configured attention-query iteration
budget. The `iterations` value still controls interpreter clustering during
fit, and it also governs the prediction-time attention query path so the
runner stays faithful to the paper's semi-automatic inference loop.

The experiment runner is non-interactive. Ground-truth labels therefore stand in
for the operator-provided labels that DeepCASE would receive during manual
analysis. Predictions are still emitted as sequence records for the common
metrics contract, with detector-owned event findings preserving DeepCASE's
event-level decisions.

DeepCASE abstentions are not treated as anomalies. Event findings now distinguish
between:

- confident benign
- confident malicious
- abstained/manual-review

The sequence-level prediction remains binary for the shared experiment contract,
but the persisted prediction record carries `sequence_decision`,
`confident_event_count`, and `abstained_event_count` so you can see how much of
the sequence was actually decisive.

### Metric Interpretation

Automated precision, recall, F1, and accuracy are computed over confident
auto-decisions only. Abstained sequences are reported separately as
manual-review workload and coverage signals:

- `auto_decision_count`: number of confident auto-decisions
- `counted_predictions`: number of automatic predictions that entered the
  shared confusion matrix
- `abstained_prediction_count`: number of deferred sequences
- `auto_coverage`: fraction of test sequences decided automatically
- `abstain_rate`: fraction of test sequences deferred for review
- `abstained_normal_label_count`: deferred normal sequences
- `abstained_anomalous_label_count`: deferred anomalous sequences

`mean_test_score` still averages all test sequences, so the score trend remains
comparable even when the abstain rate changes.

Run metrics also carry detector-owned next-event diagnostics from the Context
Builder. This is a separate, deterministic diagnostic pass that uses the
padded context windows produced by DeepCASE. The diagnostic vocabulary policy
is configurable on `DeepCaseModelConfig` and defaults to `full_dataset`, with
`train_only` still available for closed-world comparisons. The anomaly
detector itself remains unchanged.

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
- No separate threshold sweep for alternative abstain / confidence settings.
