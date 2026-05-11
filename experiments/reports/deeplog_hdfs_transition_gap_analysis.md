# DeepLog HDFS Transition Gap Analysis

This note narrows the HDFS investigation to one question:

Why does the reconstructed LogHub-derived transition space underperform the
historical `wuyifan18/DeepLog` benchmark enough to drop next-event top-9
accuracy from about `0.993` to about `0.949`?

The answer is not a chronology bug, a session alignment bug, or a DeepLog
scoring bug. It is a transition-space mismatch concentrated in the rare tail
of the HDFS template distribution, plus duplicate-weighted evaluation in the
historical benchmark archive.

## Paper Protocol

The DeepLog paper description in `notes/deeplog.md` is consistent with the
current implementation contract:

- HDFS key-only next-event detection is the primary benchmark.
- The default HDFS history window is `h = 10`.
- The top-`g` acceptance window is `g = 9`.
- Session-level evaluation uses the paper's any-miss rule: one missed key makes
  the whole session anomalous.
- Parameter-value anomaly detection is a separate path and is not part of the
  primary HDFS table result.

So the remaining gap is not explained by the scoring semantics.

## What The Current Files Show

The current LogHub HDFS corpus is close to the historical benchmark at the
head of the distribution, but it is not identical:

- `data/HDFS_V1/preprocessed/Event_traces.csv` preserves the same session
  order as raw first-seen block chronology in `data/HDFS_V1/HDFS.log`.
- The historical `wuyifan18` archive is almost, but not perfectly, a subset of
  the current LogHub session corpus.
- The historical archive exposes `28` event IDs.
- Current LogHub exposes `29` templates.

The exact extra template in current LogHub is:

- `E29`: `[*]PendingReplicationMonitor timed out block[*]`

That template is absent from the historical benchmark archive.

The more important point is that the current and historical corpora agree very
closely on the common core templates, but diverge in the rare tail:

- `E7`: `[*]writeBlock[*]received exception[*]`
- `E14`: `[*]Exception in receiveBlock for block[*]`
- `E17`: `[*]:Failed to transfer[*]to[*]got[*]`
- `E29`: `[*]PendingReplicationMonitor timed out block[*]`

Those are exception or maintenance-path templates. They are the likely parser
or corpus-version drift points, not the main block-transfer templates.

### Current template inventory

The current LogHub template frequencies are:

| EventId | Count | Template |
| --- | ---: | --- |
| `E1` | `10` | `[*]Adding an already existing block[*]` |
| `E2` | `120036` | `[*]Verification succeeded for[*]` |
| `E3` | `428726` | `[*]Served block[*]to[*]` |
| `E4` | `356207` | `[*]Got exception while serving[*]to[*]` |
| `E5` | `1723232` | `[*]Receiving block[*]src:[*]dest:[*]` |
| `E6` | `7097` | `[*]Received block[*]src:[*]dest:[*]of size[*]` |
| `E7` | `3416` | `[*]writeBlock[*]received exception[*]` |
| `E8` | `49` | `[*]PacketResponder[*]for block[*]Interrupted[*]` |
| `E9` | `1706514` | `[*]Received block[*]of size[*]from[*]` |
| `E10` | `108` | `[*]PacketResponder[*]Exception[*]` |
| `E11` | `1706679` | `[*]PacketResponder[*]for block[*]terminating[*]` |
| `E12` | `34` | `[*]:Exception writing block[*]to mirror[*]` |
| `E13` | `1464` | `[*]Receiving empty packet for block[*]` |
| `E14` | `155` | `[*]Exception in receiveBlock for block[*]` |
| `E15` | `65` | `[*]Changing block file offset of block[*]from[*]to[*]meta file offset to[*]` |
| `E16` | `6937` | `[*]:Transmitted block[*]to[*]` |
| `E17` | `9` | `[*]:Failed to transfer[*]to[*]got[*]` |
| `E18` | `7002` | `[*]Starting thread to transfer block[*]to[*]` |
| `E19` | `5` | `[*]Reopen Block[*]` |
| `E20` | `5545` | `[*]Unexpected error trying to delete block[*]BlockInfo not found in volumeMap[*]` |
| `E21` | `1402047` | `[*]Deleting block[*]file[*]` |
| `E22` | `575061` | `[*]BLOCK* NameSystem[*]allocateBlock:[*]` |
| `E23` | `1396174` | `[*]BLOCK* NameSystem[*]delete:[*]is added to invalidSet of[*]` |
| `E24` | `4` | `[*]BLOCK* Removing block[*]from neededReplications as it does not belong to any file[*]` |
| `E25` | `7002` | `[*]BLOCK* ask[*]to replicate[*]to[*]` |
| `E26` | `1719741` | `[*]BLOCK* NameSystem[*]addStoredBlock: blockMap updated:[*]is added to[*]size[*]` |
| `E27` | `975` | `[*]BLOCK* NameSystem[*]addStoredBlock: Redundant addStoredBlock request received for[*]on[*]size[*]` |
| `E28` | `1288` | `[*]BLOCK* NameSystem[*]addStoredBlock: addStoredBlock request received for[*]on[*]size[*]But it does not belong to any file[*]` |
| `E29` | `47` | `[*]PendingReplicationMonitor timed out block[*]` |

The core transfer/replication templates are stable. The mismatch lives in the
rare tail and in the split/evaluation weighting.

## Historical Benchmark Mechanics

The historical `wuyifan18` files are not a simple raw-entry prefix of the
current LogHub corpus.

Evidence:

- The historical `hdfs_train` file is not the first `4,855` normal sessions in
  current `Event_traces.csv`.
- Almost all historical train rows are present somewhere in the current LogHub
  session corpus, but the file is not a prefix slice.
- The historical test normal file is fully present in the current corpus.
- The historical abnormal file is only partially present in the current corpus.

That points to a pre-generated session archive rather than a raw-log prefix
reconstruction from the current `HDFS.log`.

## Duplicate Inflation In The Historical Archive

The historical archive contains many repeated full-session rows.

- `hdfs_train`: `4,855` rows, `839` unique session lines
- `hdfs_test_normal`: `553,366` rows, `14,177` unique session lines
- `hdfs_test_abnormal`: `16,838` rows, `4,123` unique session lines

The duplicate weighting matters a lot for transition metrics.

Using a simple `h = 10` count-based next-event baseline on the historical
archive:

| Split | Full weighted top-9 | Unique-session top-9 |
| --- | ---: | ---: |
| normal test | `0.9561512537477033` | `0.5954372844255771` |
| abnormal test | `0.4759706694038612` | `0.35158530196821913` |
| overall | `0.9417405687358877` | `0.5346059304894365` |

Two implications follow:

- The historical archive's duplicate test rows materially amplify apparent
  transition memorisation.
- The train duplicates do not change the count-based top-9 result, but the test
  duplicates do. That means the benchmark metric itself is heavily
  duplicate-weighted.

So the historical benchmark's transition space is not just "the same corpus but
with a different model". It is also an evaluation stream with strong duplicate
inflation.

## Current False-Positive Tail

The current DeepLog key-only run on the reconstructed LogHub split does not
look like an alignment bug. Its normal false positives are concentrated in a
small number of ordinary late-session transitions.

For the current run artefact in
`experiments/results/hdfs_v1_deeplog_paper_entry100k_assign_first/588e390825bd/predictions.jsonl`:

- normal false-positive sessions: `364,230`
- unique first-miss transitions: `1,325`
- top 20 first-miss transitions explain `84.5767784092469%` of all false
  positive sessions

The dominant miss pattern is the normal delete tail after a long run of
`addStoredBlock` and `PacketResponder` activity.

Representative first-miss examples from the raw log:

```text
11: 081109 212735 27 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.123.33:50010 is added to blk_7219411039777225560 size 67108864
12: 081109 212735 28 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.126.22:50010 is added to blk_7219411039777225560 size 67108864
13: 081109 212735 30 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.203.4:50010 is added to blk_7219411039777225560 size 67108864
14: 081109 213816 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.delete: blk_7219411039777225560 is added to invalidSet of 10.251.123.33:50010
```

```text
11: 081109 212611 34 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.70.5:50010 is added to blk_-4898747391262002624 size 67108864
12: 081109 212611 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.123.33:50010 is added to blk_-4898747391262002624 size 67108864
13: 081109 212612 30 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.203.246:50010 is added to blk_-4898747391262002624 size 67108864
14: 081109 213816 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.delete: blk_-4898747391262002624 is added to invalidSet of 10.251.123.33:50010
```

```text
11: 081109 213538 2449 INFO dfs.DataNode$PacketResponder: Received block blk_-5707542431860670984 of size 67108864 from /10.251.74.79
12: 081109 213538 2466 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_-5707542431860670984 terminating
13: 081109 213538 2466 INFO dfs.DataNode$PacketResponder: Received block blk_-5707542431860670984 of size 67108864 from /10.251.74.79
14: 081109 213817 33 INFO dfs.FSNamesystem: BLOCK* NameSystem.delete: blk_-5707542431860670984 is added to invalidSet of 10.250.10.100:50010
```

These are ordinary late-session tails, not unknown-template or chronology
failures.

## Direct Answers

### Can the current raw LogHub HDFS file reproduce the paper's split exactly?

No. The current `HDFS.log` cannot recover the paper's `4,855 / 553,366 /
15,200` split exactly under any of the tested first-100k interpretations.

### What is the most likely concrete mechanism behind the historical benchmark?

The historical benchmark is most likely a pre-generated session archive with:

- a slightly different parser/template map in the rare tail,
- preserved duplicate test rows,
- and a session-level split that is not a raw-prefix reconstruction from the
  current LogHub file.

It looks much more like an archive-level benchmark artefact than a live
reconstruction of the current raw file.

### What is causing the top-9 degradation?

Not the common HDFS block-transfer core. The degradation comes from:

- rare late-session delete/allocation/exception transitions,
- one extra current template with no historical counterpart (`E29`),
- and the fact that the historical benchmark weights duplicated sessions very
  heavily.

### If we had the exact historical transition distribution, would current
DeepLog likely recover paper-like F1?

Probably yes, and there is no current evidence for a remaining DeepLog
implementation bug beyond the transition-space mismatch.

The strongest evidence is that the observed false positives are almost all
plain rank misses on ordinary normal tails, not unknown-history or
alignment-related failures.

### What should be reported as the historical benchmark result?

Report the historical `wuyifan18` archive result as the benchmark result, but
label it as benchmark-archive performance rather than as a fresh raw-LogHub
reconstruction.

### What should be reported as the LogHub reconstruction result?

Report the current LogHub raw reconstruction separately as a best-effort
reproduction on the current LogHub corpus, because the corpus/version and split
semantics do not exactly match the historical archive.
