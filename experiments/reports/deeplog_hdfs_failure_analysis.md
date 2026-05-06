# DeepLog HDFS Failure Analysis

## Summary

The HDFS DeepLog reproduction is still not paper-faithful, but the failure is
now better understood.

Disabling parameter detection for the HDFS benchmark was the right protocol
fix: the paper's HDFS result is key-only. However, that was not the main source
of the remaining gap. The current corpus and split behaviour still produce a
very large number of key-level session misses compared with the paper's
reported HDFS result.

The codebase now also defends against a concrete chronology risk in the
Parquet-backed entity grouping path by sorting rows within each entity by
`line_order` before building `TemplateSequence` objects.

## Current Key-Only Result

Last completed key-only HDFS `assign_first` run:

- accuracy: `0.57893465`
- precision: `0.03783487`
- recall: `0.55055976`
- F1: `0.07080404`
- FP: `231368`
- FN: `7427`
- TN: `319228`
- TP: `9098`
- mean_test_score: `0.42400788`
- next-event top-1 accuracy: `0.78383870`
- next-event top-5 accuracy: `0.91947332`
- next-event top-9 accuracy: `0.94924544`

These numbers remain far from the DeepLog paper's HDFS targets
(`precision ≈ 0.95`, `recall ≈ 0.96`, `F1 ≈ 0.96`), but they are now understood
in the context of the corpus rather than the parameter branch.

## Ordering Audit

The cached Parquet-backed HDFS entity sequences now preserve raw source order
within each session. A direct scan over the cached Parquet dataset found:

- `0` line-order violations across entity sequences
- no evidence that the current cache reorders rows within a session

The implementation still sorts each entity's rows by `line_order` in
[`anomalog/parsers/structured/parquet/sink.py`](/Users/harensamarasinghe/Documents/Imperial/Year%204/Final%20Year%20Project/toolkits/AnomaLog/anomalog/parsers/structured/parquet/sink.py)
so that future scanner or fragment-order changes cannot break chronology.

## Transition And Coverage Diagnostics

Using the LogHub preprocessed HDFS trace corpus and the raw log's first-seen
block chronology, the exact `assign_first` split yields:

| Metric | Value |
| --- | ---: |
| total sessions | 575,061 |
| train sessions | 7,940 |
| test sessions | 567,121 |
| train normal sessions | 7,627 |
| test normal sessions | 550,596 |
| test anomalous sessions | 16,525 |
| train raw entries | 203,828 |
| test raw entries | 10,971,801 |

Transition coverage diagnostics:

| Metric | Value |
| --- | ---: |
| unique templates in train | 23 |
| unique templates in test | 29 |
| unique templates overall | 29 |
| unique `h=10` contexts in train | 4,634 |
| unique `h=10` contexts in test | 35,325 |
| fraction of test contexts seen in train | 0.10423213 |
| fraction of test targets seen after the same train context | 0.42859493 |

Event-level miss behaviour under the empirical top-9 h-gram baseline:

| Metric | Value |
| --- | ---: |
| normal event miss rate | 0.56500905 |
| anomalous event miss rate | 0.77940623 |
| normal sessions with at least one miss | 413,629 |
| anomalous sessions with at least one miss | 10,455 |

Normal session miss distribution under the same baseline:

- min: `0`
- median: `6`
- p90: `10`
- p95: `14`
- p99: `17`
- max: `187`

Anomalous session miss distribution under the same baseline:

- min: `0`
- median: `8`
- p90: `20`
- p95: `28`
- p99: `29`
- max: `117`

## Empirical H-Gram Baseline

The paper-diagnostic empirical h-gram baseline is much worse than the current
LSTM result on this corpus:

- precision: `0.02465313`
- recall: `0.99980874`
- F1: `0.04811974`
- FP: `413629`
- FN: `2`
- top-1 accuracy: `0.39577944`
- top-5 accuracy: `0.42859493`
- top-9 accuracy: `0.42859493`

That matters because it means the current model is not obviously under-trained
relative to a simple count-based next-key model. The remaining gap is more
consistent with a protocol or corpus mismatch than with a purely optimisation-
driven failure.

## Interpretation

The current evidence points to a best-effort reproduction rather than an exact
paper match:

- the parameter branch was a separate protocol mismatch, but it was not the
  dominant cause of the remaining HDFS failure;
- the current HDFS corpus is LogHub preprocessed data, not an independently
  verified official DeepLog session file;
- the empirical baseline on the current corpus is already far below the
  paper's HDFS numbers;
- the exact paper split is therefore likely blocked by a dataset/preprocessing
  mismatch rather than by a simple implementation bug.

## Bug Fixes Applied

- Added a defensive `line_order` sort to Parquet entity grouping in
  [`anomalog/parsers/structured/parquet/sink.py`](/Users/harensamarasinghe/Documents/Imperial/Year%204/Final%20Year%20Project/toolkits/AnomaLog/anomalog/parsers/structured/parquet/sink.py).
- Added a regression test that verifies grouped session rows stay in source
  order.
- Added a DeepLog target-alignment regression test that locks the first
  `h=10` target to the eleventh event.

## Status

HDFS reproduction is best-effort, not exact. The code now preserves session
chronology defensively, but the remaining gap to the paper is still explained
primarily by corpus/split mismatch rather than by a threshold choice or the
parameter branch.
