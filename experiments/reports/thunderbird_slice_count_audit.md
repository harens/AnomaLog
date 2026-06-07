# Thunderbird Slice Count Audit

This audit separates three Thunderbird contracts that were previously mixed in
the dissertation draft:

- the ICSE 2022 100-log chronology table;
- the later public/raw-position Thunderbird lineage associated with LogLLM;
- the corrected AnomaLog fixed-window contract used by the Thunderbird preset;
- the legacy compacted-window comparison run.

## Source Of Truth

- Current fixed-window Thunderbird preset:
  - raw-position window basis with no alignment offset
  - `100`-log non-overlapping windows
  - `80,000 / 20,000` train/test windows on the reconstructed cached slice
  - `846 / 29` anomalous train/test windows on the reconstructed cached slice
- Legacy compacted-window comparison run:
  - [`experiments/results-final/thunderbird_deeplog/b70c8b625684/`](../results-final/thunderbird_deeplog/b70c8b625684)
  - `environment.json` recorded at `2026-06-07T00:47:34.888154+00:00`
  - dataset manifest counts:
    - `9,999,616` structured rows
    - `384` skipped rows
    - `99,996` windows
    - `79,996 / 20,000` train/test windows
    - `376 / 51` anomalous train/test windows
    - `4,937` anomalous raw log messages
    - `4,428` templates
- Local rerunnable audit helper:
  - `uv run python -m experiments.runners.audit_thunderbird_slice --start-line-order 159999999 --end-line-order 169999998`

## Contract Table

| Contract variant | total windows | train windows | test windows | train anomalous | test anomalous | Explanation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ICSE 2022 Thunderbird table | 99,593 | 79,674 | 19,919 | 816 | 27 | Paper-faithful 100-log chronology from *How Far Are We?* |
| Corrected AnomaLog fixed-window preset | 100,000 | 80,000 | 20,000 | 846 | 29 | Natural raw-position windowing with no alignment offset |
| Legacy compacted-window fixed-window run | 99,996 | 79,996 | 20,000 | 376 | 51 | Historical compacted-window comparison point |
| Local reconstructed slice, scan-order compacted | 99,606 | 79,996 | 19,610 | 373 | 43 | Partitioned parquet scan order on the available local cache |
| Local reconstructed slice, source-order compacted | 99,606 | 79,996 | 19,610 | 822 | 29 | Same reconstructed slice, but sorted by raw `line_order` before windowing |
| Local reconstructed slice, raw positions with skipped rows padded | 100,000 | 79,996 | 20,004 | 846 | 29 | Raw-position chronology with skipped rows treated as blank positions |
| Local reconstructed slice, raw positions with offset `72` | 99,999 | 79,996 | 20,003 | 837 | 29 | Audit-only compatibility variant that reproduces the public/raw-position lineage on the available slice |

## What Changed

The corrected fixed-window preset now uses the same raw-position contract as
the public/raw-position Thunderbird lineage:

- skipped rows no longer shift later window boundaries;
- the window basis is raw-position chronology rather than compacted
  structured-event scan order;
- the main preset uses no alignment offset;
- the public lineage remains recoverable on the available reconstruction when
  the audit helper applies a `72`-row compatibility offset.

The legacy compacted-window measurement therefore no longer matches the
corrected contract. A detector rerun is required before any post-fix DeepLog
metrics are reported.

## Commands Run

- `uv run python -m experiments.runners.audit_thunderbird_slice --start-line-order 159999999 --end-line-order 169999998`
- One-off `uv run python` scripts against `~/Library/Caches/anomalog/THUNDERBIRD/structured_parquet` to compare scan-order, source-order, and raw-position windowing on the reconstructed slice.

## Recommendation

For the dissertation:

- report the corrected raw-position corpus counts from this audit;
- mention `846 / 29` as the corrected raw-position corpus counts for the main
  preset;
- mention `837 / 29` only as an audit-only compatibility variant recovered by
  the `72` offset;
- treat `376 / 51` as the legacy compacted-window comparison measurement;
- do not publish detector metrics until Thunderbird is retrained on the
  corrected contract;
- cite the audit report whenever the windowing contract needs to be explained.

## Residual Caveat

The local reconstruction was taken from the structured cache that is available
on this machine, not from the exact slice artefact used by the legacy
compacted-window comparison run. That is why the raw-position reconstruction is
used as contract evidence rather than as a byte-for-byte reproduction artefact.
