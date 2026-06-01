# DeepCASE Workload Reduction Audit

This note documents the paper definitions used for the DeepCASE workload
reduction blocks and compares them with the current HDFS experiment artefact.

## Paper Definitions

The DeepCASE paper defines workload reduction with three quantities:

- `coverage = covered_events / total_events`
- `reduction = 1 - alerts / covered_events`
- `overall = 1 - (alerts + uncovered_events) / total_events`

The paper also fixes the manual-mode alert sampling policy at 10 sequences per
cluster. In manual mode, `alerts = cluster_count * 10`. In semi-automatic mode,
no alert sampling is reported, so `reduction = 100%` and `overall = coverage`.

For HDFS Table X, the paper reports:

- manual DeepCASE: 393 clusters, 95.71% reduction, 96.39% coverage, 92.26%
  overall
- semi-automatic DeepCASE: 100.00% reduction, 96.43% coverage, 96.43%
  overall

## Local Interpretation

The previous AnomaLog implementation accidentally derived the workload blocks
from the test-split sequence wrapper, which made manual coverage report as
`1.0` whenever all scored test sequences received some automatic decision.
That was not the paper's intent.

The current implementation now uses the paper-aligned units:

- manual mode uses fit-time clustered training samples
- semi-automatic mode uses prediction-time scored event samples

That change fixes the denominator and the notion of coverage, so the blocks
now measure the same quantities as the paper instead of mirroring the shared
sequence-level anomaly wrapper.

## Current HDFS Artefact

The checked-in HDFS run at
`experiments/results/hdfs_v1_entity_chronological_deepcase/989072c251e8/`
still does not numerically match the paper's Table X. The local run records:

- `cluster_count = 439`
- `train_sample_count = 2,730,412`
- `clustered_sample_count = 2,688,778`
- `unknown_cluster_score_count = 41,634`
- `prediction_diagnostics.event_count = 8,445,217`
- `prediction_diagnostics.confident_event_count = 7,962,784`
- `prediction_diagnostics.abstained_event_count = 482,433`

Under the corrected definitions, that would yield:

- manual coverage `2,688,778 / 2,730,412 = 0.9848`
- manual reduction `1 - 4,390 / 2,688,778 = 0.9984`
- manual overall `0.9831`
- semi-automatic coverage `7,962,784 / 8,445,217 = 0.9429`
- semi-automatic reduction `1.0`
- semi-automatic overall `0.9429`

Those values are still different from the paper because the local artefact uses
the maintained DeepCASE integration and the local HDFS preprocessing bundle,
which do not reproduce the released paper run exactly. The important correction
is that the metric blocks now use the paper's definitions, not the shared
sequence wrapper.
