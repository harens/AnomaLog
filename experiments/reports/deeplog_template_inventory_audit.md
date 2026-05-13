# DeepLog Template Inventory Audit

This note consolidates the template-count checks for the datasets covered by
[`notes/deeplog.md`](../../notes/deeplog.md). It keeps paper targets separate
from corpus-specific artefacts and parser heuristics.

## HDFS

- Paper target: `29` templates/log keys.
- Current LogHub HDFS inventory: `29` templates/log keys.
- Status: matches the paper.
- Separate reproduction artefact: the `wuyifan18` preprocessed session archive
  has `28` unique event IDs. That is expected for that archive and should not
  be conflated with the paper's current LogHub inventory.

## OpenStack

- Paper target: `40` templates.
- Current local archive with the strict `instance_id` parser and the targeted
  instance-storage filename collapse: `25` inferred templates.
- Relaxed all-row audit with the same OpenStack normalisation: `433` inferred
  templates.
- Pre-normalisation strict-session view: `1,126` distinct train templates,
  `2,643` distinct normal-test templates, `410` distinct abnormal-test
  templates.
- Status: the strict session-only view is now compact, but it is still below
  the paper target; the remaining gap is likely corpus/source mismatch rather
  than detector scoring.
- The archive contains three observed `pending task (...)` states, which are
  worth recording explicitly rather than collapsing:
  - `spawning`: `4,049` occurrences
  - `deleting`: `3` occurrences
  - `networking`: `1` occurrence
- They are not treated as a paper-prescribed parameter policy, but they should
  remain visible to any future parameter model.

## BGL

- Not covered by `notes/deeplog.md`.
- There is no DeepLog paper template target to compare against, so BGL is
  excluded from the paper-alignment check here. It can still be audited as a
  corpus sanity check if needed.

## Conclusion

- HDFS: template inventory aligns with the paper.
- OpenStack: the current strict session-only corpus is now down to `25`
  templates, but it still does not reach the paper's `40`-template inventory.
- Future parameter modelling should preserve raw message values separately from
  the key-mining text.
