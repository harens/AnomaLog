# SPELL HDFS_V1 Findings

## Summary

The original HDFS_V1 SPELL slowdown was real, but the primary bottleneck was in
`spellpy`, not in AnomaLog's wrapper or in downstream DeepLog training.

The updated `spellpy` package now appears to fix the root cause well enough that
AnomaLog no longer needs its temporary in-memory SPELL workaround.

## Root Cause

The earlier hot path came from cluster history bookkeeping inside `spellpy`.
Large clusters stored historical line ids in `logIDL`, and membership checks on
that history became progressively more expensive on large datasets and reruns.

The pathological rerun behaviour was especially bad when prior parser state was
reused from `rootNode.pkl` and `logCluL.pkl`, because old cluster histories
could accumulate and make duplicate checks scale poorly.

## Upstream Fix Validation

The installed `spellpy` version now shows two important behaviour changes:

- cluster membership is tracked with a set-backed path rather than relying on
  linear scans over `logIDL`
- parser state reuse is disabled by default via `resume_state=False`

Bounded parser-only runs on the real `HDFS_V1_spell_input.log` no longer showed
the previous rerun collapse.

Direct `spellpy` parse timings:

- `10k` lines: `0.22s` fresh, `0.20s` rerun
- `50k` lines: `1.19s` fresh, `1.09s` rerun
- `100k` lines: `2.30s` fresh, `2.33s` rerun
- `200k` lines: `5.04s` fresh, `5.14s` rerun

These numbers are materially different from the previous failure mode, where
reruns degraded much more sharply and long-running parses could appear hung.

## AnomaLog Changes Kept

AnomaLog still keeps the following SPELL integration improvements:

- `spellpy.spell` logs are forwarded through the active experiment logger even
  when the run logger inherits its handlers
- `SpellTemplateParser` passes `progress_interval=1000`
- `SpellTemplateParser` passes `max_lcs_comparisons_per_line=10000`
- `SpellTemplateParser` logs `spellpy` parse metrics after parsing

These changes improve observability and keep the useful guardrails enabled
without duplicating upstream SPELL logic inside AnomaLog.

## AnomaLog Changes Removed

The temporary in-memory SPELL miner and the stale-output cleanup workaround were
removed after validating the upstream `spellpy` fix.

That leaves `SpellTemplateParser` on the normal direct `spellpy.LogParser`
path, which is simpler and easier to maintain.
