# Baseline Event-Level Coverage

- `DeepLog` supports `event_level_detection` on datasets with per-event labels, and that remains the paper-faithful headline scope for BGL-style continuous streams.
- `Markov` now supports `event_level_detection` on the same labelled event streams. It still keeps the sequence-level block for secondary diagnostics.
- `Template Frequency` now supports `event_level_detection` on labelled event streams. It still keeps the sequence-level block for secondary diagnostics.
- `Naive Bayes` remains sequence-level only. It is supervised, label-conditioned, and not a paper-faithful unsupervised anomaly detector.
- `DeepCASE` remains mixed: event-level diagnostics are still useful, but the primary public story is sequence/workload reduction rather than a pure event detector.

For BGL paper-style runs, the practical headline is event-level detection over log entries. Sequence-level metrics may still be written for completeness, but they are not a reliable headline on the chronological 100k chunk splits because those test sets can collapse to a single class.

## Modelling Notes

- `Markov` sequence scores are mean negative log transition probabilities over the eligible masked transitions in a window. They are not an aggregate over per-event anomaly labels.
- `Template Frequency` sequence scores are mean negative log template probabilities over the eligible masked events in a window. They are likewise native window scores, not a simple reduction of event predictions.
- Treating a sequence as anomalous when any scored event is anomalous is acceptable as a reporting fallback, but it changes the metric semantics. It is not the detector's native sequence score.
