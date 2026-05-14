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

## 2026-05 Audit Findings

The local forensic comparison against `../LogADEmpirical` found three relevant
outcomes:

- There is no current evidence of an AnomaLog DeepLog off-by-one bug in the
  `history -> target` construction. The first target still begins immediately
  after the history window, and the same indexing is used in inference.
- The main reproduction gap is protocol and corpus mismatch, not a clear model
  implementation failure:
  - LogADEmpirical's BGL path groups by node into shuffled sessions rather than
    preserving the paper-facing chronological entry stream.
  - LogADEmpirical's HDFS path groups sessions first, shuffles them, and then
    takes a 1% session split rather than reproducing a first-100k chronological
    raw-entry protocol.
  - LogADEmpirical also replaces the requested DeepLog `g` with a validation-
    derived recommendation at runtime, so it is not holding the paper threshold
    fixed during evaluation.
- One concrete AnomaLog protocol bug was found: the HDFS paper-facing bundles
  described `g = 9` in comments, but inherited `deeplog_default.top_g_values =
  [1, 3, 5, 7, 9, 11]`, so runtime scoring was actually using `g = 11`.

This report therefore treats the remaining BGL and HDFS gaps as data/protocol
issues first. The checked-in fixes from this audit pin HDFS paper runs back to
`g = 9` and make key-only DeepLog the experiment default unless a run opts into
parameter modelling explicitly.

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
paper side: it groups by `instance_id`, namespaces each recovered instance by
the source-file split, and skips rows that do not expose one. That is the more
defensible interpretation of the paper, but it also means the local archive
does not match the paper's quoted counts.

Observed counts on the current archive with the strict instance-id parser:

| Quantity | Count |
| --- | ---: |
| raw rows | 207,820 |
| parseable rows | 207,636 |
| rows with an explicit instance token | 53,618 |
| rows without an explicit instance token | 154,018 |
| unparseable rows | 184 |
| train entity groups | 557 |
| normal test entity groups | 1,315 |
| abnormal test entity groups | 198 |
| total entity groups | 2,069 |

The main failure mode before this pass was that the `instance_id` marker was
still leaking into the Spell input. Once that happens, each VM session tends
to acquire its own template variant, and the key vocabulary explodes even
though the session grouping itself is correct.

Local audit of the current raw archive showed the problem directly:

| Quantity | Before normalisation | After normalisation |
| --- | ---: | ---: |
| distinct inferred templates in train | 1,126 | 25 |
| distinct inferred templates in test normal | 2,643 | 26 |
| distinct inferred templates in test abnormal | 410 | 24 |
| train/test template overlap | partial | complete |

The normalisation step strips the leading `[instance: ...]` tag and
canonicalises volatile UUID, IP, instance-storage filename, path-segment, hex,
and numeric tokens before Spell mining. That is a key-only heuristic for the
current OpenStack preset, not a paper-stated parameter policy. The archive
also contains three distinct `pending task (...)` states (`spawning`,
`deleting`, `networking`), which are best treated as observed task-state text
rather than noise to be merged away.

With that change, the strict instance-id view now trains on 25 stable OpenStack
keys. The same normalisation on the relaxed all-row stream still exposes 433
keys, which is a strong hint that the local corpus is not the same inventory as
the paper's CloudLab OpenStack deployment. The instance-store filename
collapse removes the obvious one-off key flood that was forcing the detector to
mark every eligible test event as anomalous, but it does not make the local
archive match the paper's 40-key target by itself.

Paper target counts:

| Quantity | Paper |
| --- | ---: |
| train normal sessions | 831 |
| test normal sessions | 5,990 |
| test abnormal sessions | 453 |
| vocabulary size | 40 |

The gap is still dominated by dataset/source mismatch and preprocessing scope.
The consolidated template inventory audit is recorded separately in
[DeepLog template inventory audit](deeplog_template_inventory_audit.md).
The released `nailo2c/deeplog` example does not use instance sessions at all:
it Spell-parses the `Content` field and then resamples the parsed event stream
into 1-minute buckets. That reproduction is useful as a reference
implementation, but it is not the paper's session protocol. The current fix
keeps the paper-faithful `instance_id` grouping while removing the accidental
session-token leakage into Spell. The next faithful step would be to recover
the exact paper corpus or a reconstruction that yields the target counts
before tuning any detector behaviour.

To rerun the local preprocessing audit, use:

`uv run python -m experiments.runners.audit_deeplog_data --dataset openstack_deeplog_preprocessed:10`

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
