# DeepLog HDFS Wuyifan18 Preprocessed Audit

This note records what can be confirmed from the saved result artefact and the
packaged `wuyifan18` preprocessed HDFS files without rerunning the full
detector.

The current sequence-level result in
[`experiments/results/hdfs_wuyifan18_preprocessed_exact_boundary_deeplog_parameter_detection_enabled_false/aca422f9dfa0/metrics.json`](/Users/harensamarasinghe/Documents/Imperial/Year%204/Final%20Year%20Project/toolkits/AnomaLog/experiments/results/hdfs_wuyifan18_preprocessed_exact_boundary_deeplog_parameter_detection_enabled_false/aca422f9dfa0/metrics.json)
is:

- sequence F1: `0.87474878`
- sequence recall: `0.8013422`
- next-event top-1 accuracy: `0.90761047`
- next-event top-5 accuracy: `0.99066909`
- next-event top-9 accuracy: `0.99255524`

## Confirmed Corpus Counts

The packaged DeepLog session archive at
[`data/hdfs_wuyifan18_deeplog_preprocessed/HDFS_WUYIFAN18_DEEPLOG_PREPROCESSED/DeepLog-master/data`](/Users/harensamarasinghe/Documents/Imperial/Year%204/Final%20Year%20Project/toolkits/AnomaLog/data/hdfs_wuyifan18_deeplog_preprocessed/HDFS_WUYIFAN18_DEEPLOG_PREPROCESSED/DeepLog-master/data)
has the expected archive shape for the common `wuyifan18` reproduction:

| Split | Sessions | Unique event keys | Notes |
| --- | ---: | ---: | --- |
| `hdfs_train` | `4,855` | `14` | Normal-only train prefix. |
| `hdfs_test_normal` | `553,366` | `17` | Mostly train-vocabulary keys, with a small unseen tail. |
| `hdfs_test_abnormal` | `16,838` | `28` | Full tail of the archive vocabulary. |
| full archive | `575,059` | `28` | Union of the three session files. |

This means `train_key_vocabulary_size = 14` is expected for this split. It is
not evidence that preprocessing lost event keys.

## Denominator Map

The different HDFS reports in this repository are not directly comparable
unless their evaluation population is stated explicitly.

| Implementation view | Source files | Next-event population | Anomalous files in next-event evaluation? | Notes |
| --- | --- | --- | --- | --- |
| AnomaLog DeepLog | `hdfs_train`, `hdfs_test_normal`, `hdfs_test_abnormal` | all test events from `hdfs_test_normal` + `hdfs_test_abnormal`, with strict `h=10` eligibility | yes | This is the all-test population used by the current experiment layer. |
| `deeplog/` example | `hdfs_train`, `hdfs_test_normal`, `hdfs_test_abnormal` | the 80% held-out tail of `hdfs_test_normal` | no | The public example keeps anomaly detection separate from the next-event classification report. |
| `deepcase/` example | `hdfs_test_normal` | the 80% held-out tail of `hdfs_test_normal` | no | The example script only loads the normal file, so it cannot be compared to a full archive denominator. |

Observed denominators from the scratch audit:

- AnomaLog test population: `11,077,032` events
- AnomaLog strict `h=10` next-event eligibility: `5,421,251` events
- `deeplog/` normal-only next-event population: `8,633,772` events
- `deepcase/` normal-only next-event population: `8,633,772` events

The important consequence is that the normal-only next-event reports and the
all-test anomaly-detection reports must not be compared as if they were the
same denominator.

For DeepCASE Table-IV-style comparisons, the repository now has a separate
compatibility dataset manifest,
`hdfs_wuyifan18_deepcase_table_iv_compat`, which uses only
`hdfs_test_normal`, groups by entity, and applies the first 20% of the
chronological raw-entry stream as train before treating the remaining 80% as
prediction-only evaluation. The existing
`hdfs_wuyifan18_preprocessed_exact_boundary` DeepLog result remains the
benchmark-archive anomaly-detection view.

## Vocabulary Coverage

Against the train vocabulary learned from `hdfs_train`:

- normal test sessions with at least one unseen key: `234`
- abnormal test sessions with at least one unseen key: `7,908`
- abnormal test sessions with only train-vocabulary keys: `8,930`
- abnormal test sessions with only train-vocabulary keys and length `<= 10`:
  `6,177`
- abnormal test sessions with only train-vocabulary keys and length `> 10`:
  `2,753`

The key point is that the corpus contains many abnormal sessions that are
perfectly in-vocabulary. Those are the only sessions that can plausibly become
false negatives under the current key-only scorer.

## Compatibility Variant

The older `0.87474878 / 0.8013422` sequence result corresponds to the legacy
DeepLog prediction-script behaviour, not to the strict fixed-history default.
That older path left-pads short standalone sessions so they still contribute
event-level key decisions at the early positions too. The
`short_session_padding_fidelity = true` model-set variant restores that
event-centred padded scoring as an explicit compatibility view and is now
exposed in the HDFS DeepLog paper registry.

For the `wuyifan18` archive, that compatibility mode matters because the
abnormal split contains `6,193` sessions of length `<= 10`, and the current
strict default leaves all of them without a key-model decision. That is why
the paper-faithful default and the legacy compatibility artefact produce
materially different session-level recall even though the next-event top-`g`
accuracy is effectively unchanged.

## What This Confirms

- The 14-key training vocabulary is the correct consequence of the first
  4,855 normal sessions, not a preprocessing bug.
- The archive is not missing its rare tail keys: the full session set spans 28
  event IDs.
- The current result gap is therefore not explained by vocabulary collapse.
- The saved result artefact already shows a substantial sequence-level gap
  despite high next-event top-9 accuracy, which points to the paper rule
  `any miss => anomalous` rather than a parser failure.
- The restored short-session padding flag explains the legacy compatibility
  artefact without changing the default paper-faithful scorer.

## What I Did Not Recompute

I did not rerun the full detector after the memory pressure issue on the local
machine.

As a result, this note does not enumerate the 3,345 sequence-level false
negative sessions individually, and it does not regenerate the `g = 5`, `g = 6`
or paper-comparison sweep. Those require a prediction stream from the detector
and would mean another full fit/scoring pass.

## Paper-Faithful Interpretation

For the current artefact, the defensible conclusion is:

- `train_key_vocabulary_size = 14` is expected for this split.
- the current HDFS result is limited by corpus/split behaviour and by the
  strict session-level `any miss` rule, not by lost event keys.
- a faithful report should separate this `wuyifan18` archive result from the
  raw LogHub reconstruction discussions in the other HDFS notes.
