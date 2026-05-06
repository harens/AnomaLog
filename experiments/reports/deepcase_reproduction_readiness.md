# DeepCASE Reproduction Readiness

This audit separates the public, reproducible DeepCASE HDFS work from the BGL
extension that AnomaLog supports for benchmarking.

## HDFS Table IV Prediction Comparison

Paper target:

- 10 runs
- first 20% chronological training
- remaining 80% testing
- next-event / context prediction metrics
- DeepLog, Tiresias, and DeepCASE compared on HDFS

Current AnomaLog equivalent:

- `metrics.next_event_prediction.table_iv_prediction_metrics`
- The block mirrors the weighted top-1 multi-class next-event metrics and is
  the right comparison target for Table IV.
- The run also records `metrics.next_event_prediction.totals`,
  `exclusions`, and `vocabulary_policy` so the sample population is auditable.

Direct comparability:

- Yes, for the next-event task itself.
- The comparison is only clean if the HDFS split matches the paper's
  chronological 20/80 split and the underlying dataset matches the DeepCASE
  public HDFS files.

What is missing:

- The repository does not currently contain the DeepCASE-Dataset HDFS files
  referenced by the paper:
  - `hdfs_train`
  - `hdfs_test_normal`
  - `hdfs_test_abnormal`
- No automatic importer has been added; the expected files and import
  requirements still need to be supplied locally if exact reproduction is
  desired.
- HDFS context-time handling is documented, but the paper notes that HDFS does
  not provide timestamps for the workload-reduction analysis, so the
  `timeout_seconds = 86400` setting is not a strong discriminator there.

Readiness:

- Best-effort reproduction.
- The next-event metric block is now auditable, but exact paper parity still
  depends on sourcing the public DeepCASE HDFS files.

## HDFS Table X Workload Reduction

Paper target:

- manual mode workload reduction
- semi-automatic mode workload reduction
- alerts based on 10 samples per cluster
- coverage = covered / total
- reduction = 1 - alerts / covered
- overall = 1 - (alerts + uncovered) / total

Current AnomaLog equivalent:

- `metrics.manual_workload_reduction`
- `metrics.semi_automatic_workload_reduction`
- These blocks carry the paper-style coverage, reduction, and overall formulas.

Direct comparability:

- Partial.
- The formulas are now encoded explicitly, but the current runtime still does
  not implement the full interactive operator workflow or a persistent
  cross-run cluster database.

What is missing:

- No persistent operator-labelled cluster store across runs.
- No live manual-review loop.
- No automatic importer for the public HDFS DeepCASE files.
- The HDFS paper's workload table is an average over runs; the current runner
  supports 10 runs through sweep axes, but no dedicated cross-run aggregation
  command is implemented.

Readiness:

- Best-effort reproduction.
- The workload metrics are implemented and auditable, but the operator workflow
  remains a simplified offline approximation.

## BGL DeepCASE Extension

Paper status:

- Not a DeepCASE paper dataset.
- BGL should be treated as an AnomaLog extension / benchmark only.

Current AnomaLog equivalent:

- Entity-grouped DeepCASE with event-level labels when available.
- `metrics.prediction_diagnostics.event_decision_metrics`
- `metrics.next_event_prediction`
- `metrics.manual_workload_reduction`
- `metrics.semi_automatic_workload_reduction`

Direct comparability:

- No, not to the DeepCASE paper.
- It is useful as an internal benchmark because BGL exposes line-level labels
  and therefore avoids the parent-sequence fallback path when the labels are
  present.

What is missing:

- Nothing required for the extension target beyond the usual offline DeepCASE
  limitations already documented in the detector docs.

Readiness:

- Extension only.
- Use it for benchmarking and regression tracking, not for paper reproduction
  claims.

## Current Local Data Check

- The repository does not contain the public DeepCASE HDFS files above.
- The only HDFS file discovered under `notes/deepcase/` is the example
  `notes/deepcase/example/data/hdfs/hdfs_test_normal` fixture used for the
  bundled documentation example.

