# DeepLog HDFS Failure Analysis

## Summary

The current HDFS DeepLog reproduction failure is a protocol mismatch, not a
threshold-tuning problem.

The HDFS benchmark section in the DeepLog paper reports the next-key detector
on block/session sequences. Parameter-value anomaly detection is evaluated
separately later in the paper on OpenStack logs. The previous HDFS reproduction
was combining key and parameter detection, which inflated false positives and
produced a meaningless mean test score because parameter residuals were being
folded into the sequence score.

## Current Bad Run

The previously observed HDFS result for
`hdfs_v1_deeplog_paper_entry100k_assign_first` was:

- accuracy: `0.57887118`
- precision: `0.03782921`
- recall: `0.55055976`
- F1: `0.07079412`
- FP: `231404`
- FN: `7427`
- mean_test_score: `1.2746443735516613e+37`

That behaviour is consistent with parameter scoring being applied to the HDFS
paper benchmark, which the paper does not do.

## Root Cause

The failure came from mixing two different paper sections:

- HDFS session anomaly detection, which is key-only
- OpenStack parameter anomaly detection, which uses per-template parameter
  models and Gaussian calibration

In the implementation, the HDFS reproduction sweep used the same DeepLog model
config as the OpenStack-style detector, so the parameter branch was trained and
applied on the HDFS run as well.

## Fix

The repository now has an explicit key-only HDFS reproduction config:

- [`experiments/configs/models/deeplog_hdfs_paper_key_only.toml`](/Users/harensamarasinghe/Documents/Imperial/Year%204/Final%20Year%20Project/toolkits/AnomaLog/experiments/configs/models/deeplog_hdfs_paper_key_only.toml)

The HDFS reproduction sweep now points at that config, and the detector can
also surface trigger-source breakdowns for diagnostics when the full parameter
branch is enabled elsewhere.

Additional diagnostic sweeps were added:

- [`experiments/configs/sweeps/hdfs_v1_deeplog_paper_entry100k_assign_first_full.toml`](/Users/harensamarasinghe/Documents/Imperial/Year%204/Final%20Year%20Project/toolkits/AnomaLog/experiments/configs/sweeps/hdfs_v1_deeplog_paper_entry100k_assign_first_full.toml)
- [`experiments/configs/sweeps/hdfs_v1_deeplog_paper_entry100k_split_partial_key_only.toml`](/Users/harensamarasinghe/Documents/Imperial/Year%204/Final%20Year%20Project/toolkits/AnomaLog/experiments/configs/sweeps/hdfs_v1_deeplog_paper_entry100k_split_partial_key_only.toml)

## Code Changes

- Added `parameter_detection_enabled` to `DeepLogModelConfig`.
- Disabled parameter fitting and scoring when the HDFS paper config sets that
  flag to `false`.
- Added `DeepLogSequenceTriggerBreakdown` to report key-only, parameter-only,
  both, and neither sequence triggers split by ground-truth label.
- Updated the HDFS paper-reproduction docs to make the key-only benchmark
  explicit.

## Verification Status

The key-only HDFS sweep was not rerun to completion in this session because the
user asked to defer it. The structural fix is in place, and the benchmark can
now be rerun later with the key-only config.

## Remaining Work

- Run `hdfs_v1_deeplog_paper_entry100k_assign_first` again with the new
  key-only config.
- Compare it with `hdfs_v1_deeplog_paper_entry100k_assign_first_full` if you
  want a direct key-only versus combined-model diagnostic.
