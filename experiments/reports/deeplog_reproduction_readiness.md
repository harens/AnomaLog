# DeepLog Reproduction Readiness

This note closes the remaining protocol mismatches identified during the DeepLog
paper-reproduction audit.

The checks below are intentionally conservative:

- no threshold tuning to chase the paper numbers;
- no online-update implementation in this pass;
- no silent change to benchmark semantics;
- all reproduction behaviour stays config-driven and auditable.

## BGL

### Protocol verdict

- Raw dataset counts match the paper: `4,747,963` entries and `348,460`
  anomalous entries.
- The paper's BGL evaluation is entry-level. The paper text treats BGL as a
  log-entry anomaly-detection problem, and the published metrics are reported
  over log entries rather than aggregated sessions.
- The current reproduction now exposes an event-level DeepLog evaluation path
  via `DeepLogRunMetrics.event_level_detection`, while keeping the existing
  sequence/chunk metrics as diagnostics.
- The checked-in paper configs use:
  - `grouping = "chronological_stream"`
  - `split.application_order = "before_grouping"`
  - raw-entry prefix split modes
  - `h = 3`, `g = 6`, `L = 1`, `hidden = 256`
  - online update disabled
- Chunking is now a memory container, not the split unit. Event-level masks
  decide which targets train and which targets score, so post-cutoff events in
  the first chronological chunk are no longer dropped just because they share
  a container with training context.

### Current paper configs

| Config | Chunk count | Train chunk / ignored / test chunk | Eligible training targets | Event-level evaluation available? |
| --- | ---: | --- | ---: | --- |
| `bgl_deeplog_paper_1pct_normal_entry_stream_no_online` | 48 | 1 / 0 / 47 | 43,996 | yes |
| `bgl_deeplog_paper_10pct_entry_stream_no_online` | 48 | 5 / 0 / 43 | 281,950 | yes |

### 1% normal mismatch closure

The 1% normal run now distinguishes three different counts that used to be
collapsed together:

- anomalous entries before the normal cutoff: `2,322`
- anomalous target positions excluded from training: `2,609`
- non-anomalous post-cutoff context excluded from training: `53,395`

The `2,609` value is larger than the `2,322` pre-cutoff anomaly count because
the preserved first chronological chunk also contains `287` anomalous entries
that occur after the normal cutoff. Those entries are still inside the train
chunk context, but they are masked out as training targets.

### Chunk-size sensitivity

This is a data-construction audit only. It does not run the full detector.

| Chunk size | Sequence count | Eligible training targets | Event-level evaluation count | Anomalous evaluation targets | Normal evaluation targets | Insufficient-history count | Warm-up loss | Post-cutoff events excluded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50,000 | 95 | 43,996 | 4,701,363 | 346,120 | 4,355,243 | 282 | 282 | 0 |
| 100,000 | 48 | 43,996 | 4,701,504 | 346,129 | 4,355,375 | 141 | 141 | 0 |
| 200,000 | 24 | 43,996 | 4,701,576 | 346,138 | 4,355,438 | 69 | 69 | 0 |

The event-level denominator is now stable across chunk sizes apart from the
expected warm-up loss at chunk boundaries. The first chronological chunk still
contains `53,395` post-cutoff normal events and `2,609` anomalous events in
context, but those post-cutoff events are explicitly retained for evaluation
by the per-event split mask.

### Readiness

- BGL no-online paper runs are ready to execute.
- Sequence-level metrics remain available for diagnostics.
- Event-level precision/recall/F1 should be used for paper reporting.
- The 1% normal split now has a stable event-level denominator regardless of
  chunk size.

## HDFS

### Protocol verdict

- Raw/session counts do not match the cited paper counts.
- The available local HDFS data has `11,175,629` raw entries and `575,061`
  sessions, while the paper cites `11,197,954` entries and approximately
  `575,059` sessions.
- The paper protocol is expressible in the config layer, but none of the
  plausible first-100,000 interpretations recovers the paper's target counts.
- The best-effort interpretation for analysis is `assign_by_first_event`
  (equivalently `first_100k_block_ids` on this dataset), but the checked-in
  paper config remains unchanged until there is a clearer protocol recovery.

### First-100k policy audit

| Policy | Train normal | Train anomalous | Ignored | Test normal | Test anomalous | Total sessions | No-eligible sessions | Delta vs paper target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `split_partial_sequences` | 7,627 | 0 | 313 | 558,223 | 16,716 | 575,061 | 7,093 | `+2,772 / +4,857 / +1,516` |
| `assign_by_first_event` | 7,627 | 0 | 313 | 550,596 | 16,525 | 575,061 | 6,191 | `+2,772 / -2,770 / +1,325` |
| `assign_by_last_event` | 0 | 0 | 122 | 558,223 | 16,716 | 575,061 | 6,191 | `-4,855 / +4,857 / +1,516` |
| `first_100k_block_ids` | 7,627 | 0 | 313 | 550,596 | 16,525 | 575,061 | 6,191 | `+2,772 / -2,770 / +1,325` |
| `normal_complete_sessions` | 0 | 0 | 0 | 558,223 | 16,838 | 575,061 | 6,191 | `-4,855 / +4,857 / +1,638` |

Paper targets:

- train normal sessions: `4,855`
- test normal sessions: `553,366`
- test anomalous sessions: `15,200`

No candidate policy recovers those counts. The closest match by absolute delta is
`assign_by_first_event` / `first_100k_block_ids`, but it still misses the paper
counts by a wide margin.

### Dataset/version audit

- The local raw HDFS file is `data/HDFS_V1/HDFS.log`.
- The preprocessed session files in `data/HDFS_V1/preprocessed/` contain
  `575,061` labelled sessions, which is close to but not identical with the
  paper's quoted `~575,059`.
- The template file reports `29` templates/log keys, matching the paper.
- The local raw row count is lower than the cited paper count by `22,325`
  entries.
- I did not find a local explanation that fully accounts for the raw-count
  mismatch.

### Readiness

- HDFS should be described as a best-effort reproduction, not an exact one.
- The protocol is now explicit, but the available dataset/version prevents the
  paper counts from being recovered exactly.

## Final Verdict

- BGL: protocol closed enough for no-online paper runs, with event-level paper
  metrics available and chunking documented as an approximation.
- HDFS: protocol expressible, but not exact; the available data prevents exact
  recovery of the cited paper counts.
