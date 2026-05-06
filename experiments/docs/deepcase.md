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
analysis. Predictions are still emitted as sequence records for the shared
metrics contract, but the sequence-level numbers are only a parent-sequence
aggregation wrapper for comparison with detectors such as DeepLog.

DeepCASE's natural evaluation unit is the event-centred contextual sample. The
diagnostics block therefore also carries event-level automatic-decision metrics
for the latest scoring run. Those metrics reflect the model's automatic
decisions before any sequence aggregation is applied.

DeepCASE abstentions are not treated as anomalies. Event findings now distinguish
between:

- confident benign
- confident malicious
- abstained/manual-review

The persisted prediction record carries `sequence_decision`,
`confident_event_count`, and `abstained_event_count` so you can see how much of
the sequence was actually decisive. The event-level diagnostics expose the
underlying automatic decisions separately from abstentions.

### Metric Interpretation

Sequence-level precision, recall, F1, and accuracy remain the shared wrapper
view over sequence decisions. They are useful for compatibility with the
shared experiment contract, but they are not the primary paper-comparison
target for HDFS.

The paper-comparison block for HDFS Table IV uses
`next_event_prediction.classification_top1_weighted`. It mirrors the weighted
multi-class next-event metrics and is the block that should be compared to the
paper's prediction table.

Event-level automatic-decision metrics evaluate DeepCASE at its
contextual-sample level, where:

- `known_benign_cluster` maps to a predicted normal event
- `known_malicious_cluster` maps to a predicted anomalous event
- `not_confident_enough`, `closest_cluster_outside_epsilon`, unknown events,
  and other manual-review reasons are counted as abstentions

At the event level, abstentions are excluded from the confusion matrix and
tracked separately:

- `event_count`: total contextual event samples scored
- `event_auto_decision_count`: automatic event decisions
- `event_abstained_decision_count`: event samples deferred for review
- `event_auto_coverage`: automatic decision fraction
- `event_abstain_rate`: abstention fraction
- `event_tp`, `event_fp`, `event_tn`, `event_fn`: automatic confusion matrix
- `event_precision`, `event_recall`, `event_f1`, `event_accuracy`: automatic
  decision metrics
- `event_predicted_normal_count`, `event_predicted_anomalous_count`: automatic
  prediction totals
- `event_true_normal_count`, `event_true_anomalous_count`: ground-truth event
  totals

Abstained sequences are still reported separately as manual-review workload and
coverage signals:

- `auto_decision_count`: number of confident auto-decisions
- `counted_predictions`: number of automatic predictions that entered the
  shared confusion matrix
- `abstained_prediction_count`: number of deferred sequences
- `auto_coverage`: fraction of test sequences decided automatically
- `abstain_rate`: fraction of test sequences deferred for review
- `abstained_normal_label_count`: deferred normal sequences
- `abstained_anomalous_label_count`: deferred anomalous sequences

BGL can use target-event labels where they are available, which makes the event
metrics genuinely event-supervised on that dataset. HDFS often only has the
parent sequence label available, so event-level anomaly metrics there are a
weakly supervised fallback and should be interpreted in that light.

`mean_test_score` still averages all test sequences, so the score trend remains
comparable even when the abstain rate changes.

Run metrics also carry detector-owned next-event diagnostics from the Context
Builder. This is a separate, deterministic diagnostic pass that uses the
padded context windows produced by DeepCASE. The diagnostic vocabulary policy
is configurable on `DeepCaseModelConfig` and defaults to `full_dataset`, with
`train_only` still available for closed-world comparisons.

The HDFS workload-reduction formulas are surfaced as `manual_workload_reduction`
and `semi_automatic_workload_reduction`. Those summaries encode the paper's
alert, coverage, reduction, and overall calculations, and should be used for
Table X style reporting instead of the shared anomaly F1 wrapper.

The anomaly detector itself remains unchanged.

The model should be run with entity grouping:

```toml
[sequence]
grouping = "entity"
```

The detector validates the observable invariant by rejecting sequences that span
multiple entity ids.

For the BGL extension, the same DeepCASE runtime is still used, but the run is
treated as an extension rather than a paper reproduction target.

## Remaining Gaps

- No interactive operator labeling workflow.
- No persistent cluster database shared across experiment runs.
- No online update loop for newly inspected clusters or outliers.
- No separate threshold sweep for alternative abstain / confidence settings.
- No automatic importer for the public DeepCASE HDFS files referenced by the
  paper.
