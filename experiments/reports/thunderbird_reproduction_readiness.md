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

That is the protocol the checked-in Thunderbird dataset config is meant to
approximate.

The later public benchmark implementations of the same Thunderbird setting use
a contiguous line slice from the canonical `Thunderbird.log` archive member.
LogLLM, for example, documents a Thunderbird slice from lines
`160,000,000` to `170,000,000` and reports `99,997` sequences with
`837 / 29` anomalous train/test windows. That is the closest public code path
we have found to the ICSE 2022 Thunderbird protocol, and it is the slice the
current preset now targets.

## Temporary Count Check

I ran the current parser against the Thunderbird benchmark slice
(`160,000,000` to `170,000,000`).

The parser now emits `9,999,616` structured rows from that slice and skips
`384` malformed or empty-message lines. Those rows produce `99,996`
non-overlapping 100-log windows with an `79,996 / 20,000` train/test split and
`837 / 29` anomalous windows in train/test. The raw slice itself contains
`4,937` anomalous log messages.

Compared with the ICSE 2022 table, the current slice is still `403` windows
high overall and `21 / 2` anomalous windows high in train/test. Compared with
the later public benchmark code, the slice matches the published `837 / 29`
anomalous-window totals exactly but is one window shorter because the parser
skips `384` malformed or empty raw lines from the 10-million-line segment.

## Exact Counts

The Thunderbird row in *Log-based Anomaly Detection with Deep Learning: How Far
Are We?* reports the following chronological 100-log setting:

| Dataset | Grouping | Total seqs | Avg. len. | Train seqs | Train anomalies | Test seqs | Test anomalies |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Thunderbird | 100 logs (chron.) | 99,593 | 100 | 79,674 | 816 | 19,919 | 27 |

The counts we currently measure on the benchmark slice are:

| Dataset | Grouping | Total seqs | Avg. len. | Train seqs | Train anomalies | Test seqs | Test anomalies |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Thunderbird | 100 logs (chron.) | 99,996 | 100 | 79,996 | 837 | 20,000 | 29 |

The total-sequence delta against the ICSE table is `+403`. The anomalous
window delta is `+21` in train and `+2` in test. That is a much smaller gap
than the full-archive prefix we started from, and it is consistent with the
benchmark slice used by later Thunderbird code bases.

The raw Thunderbird archive still contains many repeated `VAPI`, `ECC`, and
similar alert-category lines. Those are valid alert labels in the dataset
family, but the current benchmark slice is no longer the full archive prefix;
it is the later contiguous segment used by public reproduction code. The
remaining discrepancy with the ICSE 2022 table is therefore not a case for
relabeling alert categories as normal. It is a sign that the published table
and the later benchmark slice are not identical line for line, even though
they are drawing from the same LogHub archive member.

## Template Parsing Contract

The Thunderbird parser now keeps the message body as the template-mining
input, but it strips an optional `component[pid]: ` prefix when the raw line
includes one. That means the template vocabulary is built from the actual
message tail rather than from the Thunderbird header noise.

When the raw message tail is a bare command-like token with a trailing colon,
the structured parser trims that colon before the template miner sees the
text. That keeps punctuation-only tails such as `mysql_install_db:` from
forcing Drain3 to learn an artefact with a trailing separator.

Examples we now accept:

- `- 1131566461 2005.11.09 dn228 Nov 9 12:01:01 dn228/dn228 crond(pam_unix)[2915]: session closed for user root`
  becomes a structured event with message body `session closed for user root`.
- `- 1133559328 2005.12.02 #1# Dec 2 13:35:28 #1#/#1# exiting on signal 15`
  keeps `exiting on signal 15` as the message body even though there is no
  `component[pid]: ` separator.
- `- 1133563453 2005.12.02 tsqe1 Dec 2 14:44:13 tsqe1/tsqe1 ifup:`
  becomes message body `ifup` rather than retaining the trailing colon.
- `+ 2005.11.09 dn228 Nov 9 12:02:02 dn228/dn228 sshd[1234]: disk failure on /dev/sda`
  keeps `disk failure on /dev/sda` and marks the line as anomalous.

This is consistent with the Thunderbird-specific dataset descriptions in the
literature: the LogHub paper frames Thunderbird as a labelled alert stream,
the DSN 2007 study notes that Thunderbird alert labels were supplied by the
system administrators and that the raw structure can be irregular, and the
later Thunderbird dataset notes emphasise that Thunderbird is one of the cases
where custom parsing and template generation were needed to cover the corpus.

The template training cache is now keyed against the materialised raw slice
asset as well as the Drain3 config. That prevents a Thunderbird run from
silently reusing a cache trained on a different archive slice, which was the
root cause of the unmatched-template failure observed in the benchmark run.

Thunderbird also contains header-only rows whose tail is empty after the
structured parser normalises the line. Those rows are now treated as expected
skips rather than warnings, because they do not contribute any message body
for template mining and would otherwise flood the runtime logs without adding
new information.

References:

- J. Oliner et al., "What Supercomputers Say: A Study of Five System Logs"
  (DSN 2007). This paper describes Thunderbird alert categories as
  administrator-supplied heuristics and discusses corrupted / inconsistent
  Thunderbird message structure.
- LogHub dataset paper: J. Zhu et al., "Loghub: A Large Collection of System
  Log Datasets for AI-driven Log Analytics" (arXiv:2008.06448).
- The Thunderbird dataset analysis repository notes that Thunderbird is the
  exception among the common labelled datasets, because its templates were
  generated with custom clustering and parser-generation tools and may match
  some lines multiple times:
  `ait-aecid/anomaly-detection-log-datasets`.
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
- keeps the reproduction preset to the `160,000,000`-`170,000,000` raw line
  slice of that archive member;
- keeps a smaller smoke prefix for local development;
- builds 100-log chronological windows for the experiment layer.

## DeepLog Training Memory Note

On `2026-05-19` I ran a controlled temporary A/B comparison between the
current streaming key-model trainer and the pre-refactor `HEAD` version.

The comparison used:

- a fixed synthetic corpus with two short normal sequences;
- `history_size = 1`;
- `epochs = 2`;
- `batch_size = 64`, which is larger than the example count so both trainers
  execute the same full-batch update for this check;
- fixed `torch` and Python random seeds (`1234`).

Observed result:

- the old and new trainers produced the same scored findings on the held-out
  sequence;
- the largest absolute parameter difference between the two fitted models was
  `1.863e-09`, which is numerical noise rather than a behavioural change.

This confirms that the memory fix changes how the training examples are
materialised, not the observable scoring result in the controlled comparison.

## Temporary Verification

What I checked locally:

- the Thunderbird preset resolves to a remote source with `raw_logs_relpath =
  "Thunderbird.log"`;
- the smoke helper still materialises a bounded prefix from the same file;
- the Thunderbird benchmark slice now parses with `9,999,616` emitted rows,
  `99,996` windows, and `837 / 29` anomalous train/test windows;
- the existing Thunderbird smoke config test passes with the current window
  contract.

What I could not run here:

- the exact ICSE 2022 table-matching Thunderbird count run. The public
  benchmark slice is now much closer, but the ICSE table still reports a
  slightly shorter derived stream than the one used by later reproduction
  code.
