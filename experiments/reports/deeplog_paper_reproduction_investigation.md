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
The HDFS paper benchmark itself is now represented by a key-only DeepLog
config; parameter-value detection remains available for the OpenStack-style
diagnostic path, but the HDFS table in the paper reports the next-key detector
only.

## OpenStack

The OpenStack evidence is split between the paper and the public
reproduction code:

- the paper states that OpenStack sessions are grouped by VM `instance_id`;
- the public reproduction example in `nailo2c/deeplog` instead resamples the
  parsed log stream into 1-minute buckets after Spell parsing;
- the locally available LogHub `OpenStack.tar.gz` archive does not expose an
  instance identifier on every row, so a strict paper-faithful parser can only
  recover a subset of rows.

The current repository therefore keeps the DeepLog OpenStack preset on the
paper side: it groups by `instance_id` and skips rows that do not expose one.
That is the more defensible interpretation of the paper, but it also means the
local archive does not match the paper's quoted counts.

Observed counts on the current archive with the strict instance-id parser:

| Quantity | Count |
| --- | ---: |
| raw rows | 207,820 |
| rows with an explicit instance token | 53,618 |
| train entity groups | 557 |
| normal test entity groups | 1,315 |
| abnormal test entity groups | 198 |
| total entity groups | 2,069 |

Paper target counts:

| Quantity | Paper |
| --- | ---: |
| train normal sessions | 831 |
| test normal sessions | 5,990 |
| test abnormal sessions | 453 |
| vocabulary size | 40 |

The gap is therefore dominated by dataset/source mismatch, not by the DeepLog
scoring logic. The next faithful step would be to recover the exact paper corpus
or a reconstruction that yields the target counts before tuning any detector
behaviour.

## BGL

Current data and paper counts:

| Quantity | Paper | Reproduction config | Match? | Notes |
| --- | ---: | ---: | --- | --- |
| raw log entries | 4,747,963 | 4,747,963 | yes | Matches paper count. |
| anomalous entries | 348,460 | 348,460 | yes | Matches paper count. |

The new BGL reproduction configs use `grouping = "chronological_stream"` with a
fixed `chunk_size = 100000`. That is a deterministic memory-bound container for
the raw-entry stream, not the split unit. Chronological stream sequences are
marked as continuous by default, so DeepLog carries key and parameter context
across batch boundaries without a user-facing switch. The emitted sequence count
is batch dependent, but train/test membership is now driven by explicit
per-event masks:

- `training_event_mask` selects the normal targets eligible for fitting;
- `evaluation_event_mask` selects the targets eligible for scoring;
- the chronological batch is kept intact for context, but it no longer decides
  which post-cutoff events are lost.

The earlier 585-sequence result came from `split_partial_sequences` fragmenting
the 48 chronological batches at raw-entry label boundaries. In the 1% normal
config, the first batch was being split repeatedly as the normal quota was
reached mid-stream. The fixed policy keeps each chronological batch intact and
attaches explicit event masks instead.

Early anomalies before the normal quota cutoff remain in the batch context but
are excluded from training targets. Post-cutoff events inside the first batch
are retained for evaluation, and all later boundaries are treated as internal
batching only.

| Config | train raw | train normal | train anomalous | test raw | test normal | test anomalous | sequence count | train / ignored / test | train targets | excluded anomalies | excluded context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `1pct_normal_entry_stream_no_online` | 43,996 | 43,996 | 0 | 4,701,645 | 4,355,507 | 346,138 | 48 | 1 / 0 / 47 | 43,996 | 2,609 | 53,395 |
| `10pct_entry_stream_no_online` | 474,797 | 281,950 | 192,847 | 4,273,166 | 4,117,553 | 155,613 | 48 | 5 / 0 / 43 | 281,950 | 206,847 | 11,203 |

The 1% normal-entry split reaches the 43,996th normal raw entry after skipping
2,322 anomalous entries before the cutoff. The resulting train batch still has
53,395 post-cutoff normal events and 2,609 anomalous events in context, but
only the 43,996 normal target events are eligible for fitting. Those
post-cutoff events are also retained for evaluation through the explicit
evaluation mask, so batch size no longer suppresses the test population.

The 10% raw-entry split keeps the first five chronological batches in the train
prefix. Those batches still contain 206,847 anomalous events and 11,203
post-cutoff context events that are excluded from training targets, while
281,950 normal raw entries remain eligible for DeepLog fitting.

### Event-level boundary audit

| Mode | Sequence count | Eligible training targets | Event-level evaluation count | Anomalous evaluation targets | Normal evaluation targets | Insufficient-history count | Warm-up loss | Lost warm-up events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `batch_size = 50,000` | 95 | 43,996 | 4,701,363 | 346,120 | 4,355,243 | 282 | 282 | First three evaluation targets in each post-cutoff batch: `50,000-50,002`, `100,000-100,002`, ..., `4,700,000-4,700,002`. |
| `batch_size = 100,000` | 48 | 43,996 | 4,701,504 | 346,129 | 4,355,375 | 141 | 141 | First three evaluation targets in each post-cutoff batch: `100,000-100,002`, `200,000-200,002`, ..., `4,700,000-4,700,002`. |
| `batch_size = 200,000` | 24 | 43,996 | 4,701,576 | 346,138 | 4,355,438 | 69 | 69 | First three evaluation targets in each post-cutoff batch: `200,000-200,002`, `400,000-400,002`, ..., `4,600,000-4,600,002`. |
| continuous stream | 48 | 43,996 | 4,701,645 | 346,138 | 4,355,507 | 0 | 0 | No batch-boundary warm-up loss. Context carries across the entire stream, so the warm-up loss is eliminated rather than repeated at each artificial boundary. |

The event-level denominator now stays fixed across all batch sizes, and the
continuous-stream mode removes the artificial boundary losses entirely. The
audit helper records the raw `lost_event_line_orders` list for each mode so the
boundary losses remain inspectable in machine-readable form.

## Verdict

- HDFS paper reproduction: the split protocol is now expressible, but the
  current dataset/version mismatch means the paper counts are still not
  recovered.
- BGL paper reproduction: the split protocol is now expressible, training and
  scoring are event-level, and the only remaining gap for the fuller paper is
  the missing online update path.

## Duplicate sessions in official preprocessed HDFS

The official `wuyifan18/DeepLog/data` session files include substantial
duplicate full-session lines in both test splits.

- `hdfs_test_normal`: 553,366 rows, 14,177 unique session lines
- `hdfs_test_abnormal`: 16,838 rows, 4,123 unique session lines

For reproduction clarity, this repository does not mutate those files during
dataset materialisation. Any deduplication should be treated as an explicit
evaluation policy and reported as such.

## New Configs

- HDFS:
  - `experiments/configs/datasets/hdfs_v1_deeplog_paper_entry100k_split_partial.toml`
  - `experiments/configs/datasets/hdfs_v1_deeplog_paper_entry100k_assign_first.toml`
  - `experiments/configs/datasets/hdfs_wuyifan18_deeplog_preprocessed.toml`
  - `experiments/configs/datasets/hdfs_v1_deeplog_paper_entry100k_assign_first.toml`
  - `experiments/configs/datasets/hdfs_v1_deeplog_paper_entry100k_split_partial.toml`
  - `experiments/configs/datasets/openstack_deeplog_preprocessed.toml`
- BGL:
  - `experiments/configs/datasets/bgl_deeplog_paper_1pct_normal_entry_stream_no_online.toml`
  - `experiments/configs/datasets/bgl_deeplog_paper_10pct_entry_stream_no_online.toml`
