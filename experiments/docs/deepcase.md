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

AnomaLog experiment labels are sequence/entity labels. During DeepCase cluster
scoring, every event-centered sample derived from a sequence inherits that
sequence's label. This keeps the shared `TemplateSequence` interface unchanged
and avoids adding event-level labels that the rest of the experiment layer does
not currently use.

DeepCase training reports progress per context-builder epoch before moving on
to interpreter clustering. That keeps long training runs visibly alive instead
of appearing to stall once sequence preparation has finished.

Test-time DeepCase scoring deliberately uses the upstream zero-iteration query
path. The configured `iterations` value still controls the interpreter's
clustering step during fit, but the runner does not spend time on the slow
attention-refinement loop when producing experiment predictions.

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
the sequence was actually decisive. Run metrics additionally aggregate the
DeepCASE reason histogram and confidence/abstain coverage so the evaluation does
not collapse uncertainty into anomaly.

The manifest also carries detector-owned next-event diagnostics from the
Context Builder. This is a separate, deterministic diagnostic pass that uses
the padded context windows produced by DeepCASE. The diagnostic vocabulary
policy is configurable on `DeepCaseModelConfig` and defaults to
`full_dataset`, with `train_only` still available for closed-world
comparisons. The anomaly detector itself remains unchanged.

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
