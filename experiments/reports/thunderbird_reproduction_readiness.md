# Thunderbird Reproduction Readiness

This note records the protocol we are matching for the Thunderbird dataset
and separates the paper-faithful part from the local benchmark extension.

## DeepLog-style Thunderbird protocol

The Thunderbird study used in the current AnomaLog configs is the 2022
`"How Far Are We?"` log-anomaly paper.

The Thunderbird setting in that study is described as:

- the canonical Thunderbird archive;
- 100-log chronological windows;
- a `79,674 / 19,919` train/test split;
- `99,593` total windows.

That is the protocol the checked-in Thunderbird dataset config now follows.
The full preset now reads the canonical `Thunderbird.log` member directly and
materialises the first 10 million raw lines before parsing.

## Temporary Count Check

I ran the current parser against the first 10 million raw Thunderbird lines.
That produced `9,975,211` parsed rows and `99,752` non-overlapping 100-log
windows. The paper reports `99,593` windows, so the current parser is still
`159` windows too high. The gap means the remaining mismatch is not the split
logic but the raw-line filtering or Thunderbird-specific parsing contract.

## Exact Counts

The Thunderbird row in *Log-based Anomaly Detection with Deep Learning: How Far
Are We?* reports the following chronological 100-log setting:

| Dataset | Grouping | Total seqs | Avg. len. | Train seqs | Train anomalies | Test seqs | Test anomalies |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Thunderbird | 100 logs (chron.) | 99,593 | 100 | 79,674 | 816 | 19,919 | 27 |

The counts we currently measure on the first 10 million raw lines are:

| Dataset | Grouping | Total seqs | Avg. len. | Train seqs | Train anomalies | Test seqs | Test anomalies |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Thunderbird | 100 logs (chron.) | 99,752 | 100 | 79,801 | 36,950 | 19,951 | 1,298 |

The total-sequence delta is `+159`. The anomaly deltas are much larger, which
means the remaining work is not just the 10-million-line cap; the Thunderbird
raw-line anomaly semantics still need to be aligned with the paper's
preprocessing.

The most likely explanation is that the current build is still using a
different derived Thunderbird dataset than the paper's counted sequences. In
particular, the raw archive contains many repeated `VAPI`, `ECC`, and similar
alert-category lines, and the current parser treats them as direct anomaly
signals. That choice is consistent with the LogHub/Thunderbird dataset
description, which says that the first column uses `-` for non-alert messages
and all other entries for alert messages, and later papers repeat the same
interpretation when summarising Thunderbird's alert/non-alert split. The paper
count mismatch is therefore not a case for relabelling `VAPI`/`ECC` as normal;
it is a sign that our current preprocessing contract still differs from the
paper's exact Thunderbird experiment setup, even though the same LogHub
archive is being used.

References:

- J. Oliner et al., "What Supercomputers Say: A Study of Five System Logs"
  (DSN 2007). This paper describes Thunderbird alert categories as
  administrator-supplied heuristics and discusses corrupted / inconsistent
  Thunderbird message structure.
- LogHub dataset paper: J. Zhu et al., "Loghub: A Large Collection of System
  Log Datasets for AI-driven Log Analytics" (arXiv:2008.06448).
- Thunderbird label interpretation in a later Thunderbird study:
  "LogEDL: Log Anomaly Detection via Evidential Deep Learning" (Thunderbird
  dataset section).
- Thunderbird 100-log sequence counts in the 2022 reproduction study:
  "Log-based Anomaly Detection with Deep Learning: How Far Are We?"

## DeepCASE on Thunderbird

The original DeepCASE paper does not evaluate Thunderbird. The Thunderbird
DeepCASE registry entry in this repository is therefore a benchmark extension,
not a paper reproduction target.

## Local contract

The Thunderbird preset now:

- targets the exact `Thunderbird.log` archive member directly;
- keeps the reproduction preset to the first 10 million raw lines of that
  archive member;
- keeps a smaller smoke prefix for local development;
- builds 100-log chronological windows for the experiment layer.

## Temporary Verification

What I checked locally:

- the Thunderbird preset resolves to a remote source with `raw_logs_relpath =
  "Thunderbird.log"`;
- the smoke helper still materialises a bounded prefix from the same file;
- the existing Thunderbird smoke config test passes with the current window
  contract.

What I could not run here:

- the exact paper-matching Thunderbird count run. The 10M-prefix measurement
  is close, but it still overshoots the reported window count, so the parser
  contract needs one more pass against the original preprocessing code or
  structured export.
