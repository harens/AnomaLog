# DeepLog Paper Reproduction Investigation

This report records the current DeepLog paper-reproduction protocol as
expressed through the generic experiment/config pipeline.

The closed-out readiness summary now lives in
[experiments/reports/deeplog_reproduction_readiness.md](deeplog_reproduction_readiness.md).

The key design choice in this pass is to keep the split and grouping logic
detector-agnostic:

- `sequence.split.mode = "raw_entry_prefix_count"` for first-`N` chronological
  raw-entry prefixes.
- `sequence.split.mode = "raw_entry_prefix_fraction"` for first-`p%`
  chronological raw-entry prefixes.
- `sequence.split.mode = "raw_entry_prefix_normal_fraction"` for first-`p%`
  chronological normal raw entries.
- `sequence.split.application_order = "before_grouping"` so the split is
  applied on raw entries before any entity/session grouping.
- `sequence.split.straddling_group_policy` makes the handling of sessions that
  cross the split boundary explicit.
- `grouping = "chronological_stream"` provides a deterministic entry-stream
  grouping mode for paper-style BGL runs.

That keeps the reproduction configs explicit without hard-coding a DeepLog-only
data path.

## Metric Semantics

`SplitLabel.IGNORED` sequences are excluded from the confusion-matrix
denominators. That is intentional for the current experiment runner, but it
means ignored anomalies do not count as false negatives.

## HDFS

The current HDFS bundle still differs from the paper's cited raw-entry and
session counts, but the split protocol is now expressible. The detailed counts
and split variants remain as recorded in the previous investigation pass.

## BGL

Current data and paper counts:

| Quantity | Paper | Reproduction config | Match? | Notes |
| --- | ---: | ---: | --- | --- |
| raw log entries | 4,747,963 | 4,747,963 | yes | Matches paper count. |
| anomalous entries | 348,460 | 348,460 | yes | Matches paper count. |

The new BGL reproduction configs use `grouping = "chronological_stream"` with a
fixed `chunk_size = 100000`. That is a deterministic memory-bound container for
the raw-entry stream, not the split unit. The emitted sequence count is chunk
dependent, but train/test membership is now driven by explicit per-event masks:

- `training_event_mask` selects the normal targets eligible for fitting;
- `evaluation_event_mask` selects the targets eligible for scoring;
- the chronological chunk is kept intact for context, but it no longer decides
  which post-cutoff events are lost.

The earlier 585-sequence result came from `split_partial_sequences` fragmenting
the 48 chronological chunks at raw-entry label boundaries. In the 1% normal
config, the first chunk was being split repeatedly as the normal quota was
reached mid-stream. The fixed policy keeps each chronological chunk intact and
attaches explicit event masks instead.

Early anomalies before the normal quota cutoff remain in the chunk context but
are excluded from training targets. Post-cutoff events inside the first chunk
are now retained for evaluation instead of being lost to chunk boundaries.

| Config | train raw | train normal | train anomalous | test raw | test normal | test anomalous | sequence count | train / ignored / test | train targets | excluded anomalies | excluded context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `1pct_normal_entry_stream_no_online` | 43,996 | 43,996 | 0 | 4,701,645 | 4,355,507 | 346,138 | 48 | 1 / 0 / 47 | 43,996 | 2,609 | 53,395 |
| `10pct_entry_stream_no_online` | 474,797 | 281,950 | 192,847 | 4,273,166 | 4,117,553 | 155,613 | 48 | 5 / 0 / 43 | 281,950 | 206,847 | 11,203 |

The 1% normal-entry split reaches the 43,996th normal raw entry after skipping
2,322 anomalous entries before the cutoff. The resulting train chunk still has
53,395 post-cutoff normal events and 2,609 anomalous events in context, but
only the 43,996 normal target events are eligible for fitting. Those
post-cutoff events are also retained for evaluation through the explicit
evaluation mask, so chunk size no longer suppresses the test population.

The 10% raw-entry split keeps the first five chronological chunks in the train
prefix. Those chunks still contain 206,847 anomalous events and 11,203
post-cutoff context events that are excluded from training targets, while
281,950 normal raw entries remain eligible for DeepLog fitting.

### Event-level chunk audit

| Chunk size | Sequence count | Eligible training targets | Event-level evaluation count | Anomalous evaluation targets | Normal evaluation targets | Insufficient-history count | Warm-up loss | Post-cutoff events excluded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50,000 | 95 | 43,996 | 4,701,363 | 346,120 | 4,355,243 | 282 | 282 | 0 |
| 100,000 | 48 | 43,996 | 4,701,504 | 346,129 | 4,355,375 | 141 | 141 | 0 |
| 200,000 | 24 | 43,996 | 4,701,576 | 346,138 | 4,355,438 | 69 | 69 | 0 |

The only remaining chunk-size effect is the expected warm-up loss at chunk
boundaries. No post-cutoff events are being dropped by chunk/context handling.

## Verdict

- HDFS paper reproduction: the split protocol is now expressible, but the
  current dataset/version mismatch means the paper counts are still not
  recovered.
- BGL paper reproduction: the split protocol is now expressible, training and
  scoring are event-level, and the only remaining gap for the fuller paper is
  the missing online update path.

## New Configs

- HDFS:
  - `experiments/configs/datasets/hdfs_v1_deeplog_paper_entry100k_split_partial.toml`
  - `experiments/configs/datasets/hdfs_v1_deeplog_paper_entry100k_assign_first.toml`
- BGL:
  - `experiments/configs/datasets/bgl_deeplog_paper_1pct_normal_entry_stream_no_online.toml`
  - `experiments/configs/datasets/bgl_deeplog_paper_10pct_entry_stream_no_online.toml`
