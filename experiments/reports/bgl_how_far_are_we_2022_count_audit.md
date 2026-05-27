# BGL How Far Are We? 2022 Count Audit

This note records the bounded validation work for the BGL row in
*Log-based Anomaly Detection with Deep Learning: How Far Are We?* and the
current `bgl/how_far_are_we_2022` reproduction config.

The checked-in benchmark target is:

- `1 hour (chron.)`
- `3,606` total sequences
- `2,625` train sequences
- `981` test sequences
- `496` anomalous train sequences
- `171` anomalous test sequences

The current config lives in
[`experiments/configs/datasets/bgl/how_far_are_we_2022.toml`](../configs/datasets/bgl/how_far_are_we_2022.toml)
and is intentionally separate from the CCS 2017 DeepLog BGL presets.

## What I checked

I validated the current raw LogHub BGL corpus in
[`data/BGL/BGL.log`](../../data/BGL/BGL.log) with the repository's BGL parser
and one-hour windowing logic.

I also checked that the raw corpus size still matches the paper-side event
count:

- raw log entries: `4,747,963`

That confirms we are using the same broad raw dataset family, not a different
archive. I also compared the checked-in file against a downloaded CFDR BGL
dump because the BGL dataset lineage in the literature points back to CFDR:

| Check | Result |
| --- | --- |
| raw line count | match |
| first parsed timestamp | match |
| last parsed timestamp | match |
| byte-for-byte file identity | no |
| first differing raw line | line `4,904` |

The first differing line is not a schema change. It is the same event with a
different leading alert token:

| Source | Line 4,904 |
| --- | --- |
| checked-in LogHub file | `- 1117840321 ... ddr: excessive soft failures, consider replacing the card` |
| downloaded CFDR dump | `R_DDR_EXC 1117840321 ... ddr: excessive soft failures, consider replacing the card` |

So the repository and CFDR copies are the same dataset family and have the
same event count, but they are not byte-identical files. That provenance check
is about the underlying BGL corpus lineage only; it is not a claim that the
How Far Are We? benchmark paper itself uses CFDR terminology.

## Result

The best direct reproduction I could obtain from the current corpus and
preprocessing path is:

| Quantity | Paper target | Current best match |
| --- | ---: | ---: |
| total one-hour sequences | 3,606 | 3,604 |
| train sequences | 2,625 | 2,623 |
| test sequences | 981 | 981 |
| anomalous train sequences | 496 | 496 |
| anomalous test sequences | 171 | 169 |

The same raw corpus therefore gets us very close, but not exactly to the paper
row.

For completeness, the same direct simulation under the other obvious
chronological variants gave:

| Variant | Total windows | Train | Test | Train anomalies | Test anomalies |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw split before grouping, drop straddlers | 3,604 | 2,623 | 981 | 496 | 169 |
| raw split before grouping, assign straddlers by first event | 3,605 | 2,624 | 981 | 496 | 169 |
| raw split before grouping, assign straddlers by last event | 3,605 | 2,623 | 982 | 496 | 169 |
| raw split before grouping, split partial straddlers | 3,606 | 2,624 | 982 | 496 | 169 |

## Interpretation

The remaining gap is small but real:

- the total window count is off by `2`;
- the train count is off by `2`;
- the test count matches;
- the anomalous train count matches;
- the anomalous test count is off by `2`.

I checked a few obvious alternatives while keeping the protocol family the
same:

- changing the timestamp anchoring from absolute-hour to first-observed
  timestamp did not close the gap;
- switching between the reasonable straddler policies only moved the result by
  one to two windows;
- the mismatch is not explained by timezone handling;
- the raw corpus version in `data/BGL/BGL.log` already matches the paper's raw
  event count.

That means the current state is best described as a near-paper reproduction,
not an exact replay of the published count table.

## Practical conclusion

The BGL 2022 config is now in the right protocol family:

- one-hour windows;
- chronological raw-entry split before grouping;
- straddling windows handled explicitly;
- `history_size = 10`.

But the repository should not claim exact equality with the paper's BGL row
yet. The current implementation is close enough for engineering comparison,
but the count gap should stay documented until the exact benchmark archive or
preprocessing script is available.

## Related notes

- [DeepLog paper reproduction investigation](deeplog_paper_reproduction_investigation.md)
- [Thunderbird reproduction readiness](thunderbird_reproduction_readiness.md)
