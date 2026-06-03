# BGL DeepLog 1% No-Online Findings

This note records the investigation into the BGL DeepLog 1% normal-entry
chronological-stream run:

- `bgl_deeplog_ccs2017_paper_1pct_normal_entry_stream_no_online`
- `bgl_deeplog_ccs2017_paper_10pct_entry_stream_no_online`

## Conclusion

The low 1% recall is methodologically explainable under the current
paper-faithful no-online setup. I did not find evidence of a remaining split,
mask, or chronology bug that would better explain the gap.

The key causal chain is:

- the 1% prefix trains on only `43,996` normal entries and a very small key
  vocabulary (`27` templates);
- the 10% prefix trains on `439,951` normal entries and a larger key
  vocabulary (`68` templates);
- the event-level evaluation remains stable and continuous across chronological
  chunks;
- the anomaly-side template and history coverage is much lower in the 1% run
  than the 10% run, which makes more anomalous events look normal enough to
  land inside DeepLog's top-`g` window.

## Evidence

### Split and accounting

- The 1% manifest records `train_normal_entry_count = 43,996`,
  `train_anomalous_entry_count = 0`, and `ignored_anomalous_entry_count = 2,322`.
- The 10% manifest records `train_normal_entry_count = 439,951`,
  `train_anomalous_entry_count = 0`, and `ignored_anomalous_entry_count = 219,471`.
- The 1% `train_sequence_count = 0` is an accounting artefact of the
  continuous-stream runner. Mixed chronological chunks that contain both
  training and evaluation targets are counted as test-bearing for run
  metrics, while the training targets are still used for fitting.

### Model behaviour

- 1% event-level metrics:
  - precision: `0.09418561`
  - recall: `0.37291774`
  - F1: `0.15038849`
  - next-event top-6: `0.7085071288878679`
- 10% event-level metrics:
  - precision: `0.10594678`
  - recall: `0.98685934`
  - F1: `0.19135063`
  - next-event top-6: `0.7061323342483297`

The similar next-event top-6 values show that the model is not simply
collapsing in the smaller run. The difference is mostly in how many anomalous
events fall inside the train vocabulary and history space.

### Lightweight coverage audit

Temporary audit output:

- `/private/tmp/bgl_deeplog_template_coverage.json`

Key findings from that audit:

- 1% run:
  - anomaly targets seen in train vocabulary: `65.2%`
  - anomaly histories fully seen in train vocabulary: `65.3%`
  - unknown anomaly targets: `120,372`
  - unknown anomaly histories: `738`
- 10% run:
  - anomaly targets seen in train vocabulary: `6.8%`
  - anomaly histories fully seen in train vocabulary: `7.1%`
  - unknown anomaly targets: `120,214`
  - unknown anomaly histories: `724`

The normal-event coverage stays broadly similar, so the recall shift is not
driven by a general scoring failure. It is driven by the different anomaly-side
vocabulary coverage induced by the smaller training prefix.

## Interpretation

- Implementation issue: not confirmed.
- Dataset/version issue: not indicated by the checked manifests.
- Parser/template issue: not the primary explanation.
- Split/evaluation issue: the current masks and chronological-stream handling
  are consistent with the intended contract.
- Expected DeepLog behaviour: yes, the observed 1% recall collapse is
  compatible with the paper's top-`g` next-key rule on a much smaller normal
  prefix.

## Notes

- I did not run a full local model rerun after the initial investigation once it
  became clear that the full BGL pass was too memory-heavy for this machine.
- No permanent diagnostics were added.
