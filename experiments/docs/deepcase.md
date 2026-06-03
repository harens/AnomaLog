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

Test-time DeepCase scoring uses the `attention_query_iterations` knob, which
uses 100 for the paper-faithful path. The `iterations` value still controls
interpreter clustering during fit, and `attention_query_iterations = 0` is
reserved for explicit ablation, smoke-test, or engineering runs. If a helper
calls `ContextBuilder.query()` directly, it must pass `iterations=100`
explicitly because the low-level default is zero.

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

`metrics.json` now reports scoped blocks with explicit
`status` fields, and the block map itself carries the per-scope meaning. The
shared sequence-level wrapper remains useful when it is the configured primary
scope, but it is no longer treated as the universal headline metric. The
canonical payload lives in `metric_blocks`. DeepCASE runs should foreground the
configured `primary_metric_scope` and keep sequence-level results separate from
the event-level abstention diagnostics.

The paper-comparison block for HDFS Table IV lives at
`metric_blocks.next_event_prediction.classification_top1_weighted`. It mirrors
the weighted multi-class next-event metrics and is the block that should be
compared to the paper's prediction table.

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

Manual mode uses fit-time clustering counts from the training split:

- `total_contextual_sequence_count`: event-centred training samples processed
  during fit
- `covered_contextual_sequence_count`: training samples assigned to a
  non-noise cluster
- `uncovered_contextual_sequence_count`: training samples left unclustered or
  unscored

Semi-automatic mode uses prediction-time event counts from the scored split:

- `total_contextual_sequence_count`: scored contextual event samples
- `covered_contextual_sequence_count`: confident automatic event decisions
- `uncovered_contextual_sequence_count`: abstained event samples

That split keeps the workload metrics aligned with the paper's manual and
semi-automatic definitions. It also avoids the old behaviour where the
reported coverage could accidentally mirror the sequence-level anomaly
wrapper.

Cluster labelling is now exposed as an ablation policy on top of the stable
DeepCASE clustering pipeline. `max` remains the conservative any-anomalous
baseline, while `majority_vote`, `threshold_fraction`, and `abstain_mixed`
are sensitivity checks for how mixed clusters should be treated.

For HDFS, read those runs as label-smearing sensitivity analyses rather than
paper reproduction improvements. For BGL, and for Thunderbird as a benchmark
extension, the same cluster policies are a meaningful event-label ablation
because event-level labels are available.

`mean anomaly rate` is not used as a final decision policy here, because by
itself it does not create a distinct binary or abstain decision rule under the
shared AnomaLog score thresholding contract.

The model should be run with entity grouping:

```toml
[sequence]
grouping = "entity"
```

The detector validates the observable invariant by rejecting sequences that span
multiple entity ids.

For AIT-ADS, the stream benchmark stays in `ait_ads/base`, while the
entity-local sequence view uses `ait_ads/entity_chronological` with the raw
alert chronology split before entity grouping. That keeps host-local context
for DeepCASE without turning the protocol into an entity-heldout benchmark.

For the BGL extension, the same DeepCASE runtime is still used, but the run is
treated as an extension rather than a paper reproduction target.

## Remaining Gaps

- No interactive operator labeling workflow.
- No persistent cluster database shared across experiment runs.
- No online update loop for newly inspected clusters or outliers.
- No separate threshold sweep for alternative abstain / confidence settings.
- No automatic importer for the public DeepCASE HDFS files referenced by the
  paper.
