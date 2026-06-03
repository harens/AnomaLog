# OpenStack DeepLog Recall Gap Audit

This note records the investigation into why the current OpenStack DeepLog run
achieves very high next-event top-`g` accuracy but near-zero session-level
anomaly recall.

The short answer is:

- the recorded DeepLog run is not showing an obvious implementation bug;
- the abnormal sessions are mostly key-normal under the current representation;
- the parser is aggressively collapsing the signal, and Spell becomes largely
  redundant once that structured normalisation layer has run;
- the main issue is broader dataset/protocol mismatch rather than a single
  normalisation rule;
- the Markov baseline is picking up a small amount of residual ordering noise
  that DeepLog smooths over;
- loosening the parser into raw UUID/IP/path explosion is not a defensible fix.

## Recorded Run Summary

The recorded OpenStack DeepLog run used the `OPENSTACK_DEEPLOG_PREPROCESSED`
preset with `history_size = 10`, `top_g_values = [1, 3, 5, 7, 9, 11]`, and
`parameter_detection_enabled = false`.

Observed metrics:

| Metric | Value |
| --- | ---: |
| train sequences | 557 |
| test sequences | 1,513 |
| test normal sequences | 1,315 |
| test abnormal sequences | 198 |
| next-event top-1 accuracy | 0.9837712199 |
| next-event top-9 accuracy | 0.9996679533 |
| next-event top-11 accuracy | 0.9996679533 |
| next-event unknown history | 0 |
| next-event unknown target | 0 |
| next-event insufficient history exclusions | 15,104 |
| sequence-level anomaly recall | 0.0 |
| sequence-level confusion | `tp=0`, `fp=1`, `tn=1314`, `fn=198` |

The g=9 and g=11 values are identical in the recorded run, which means raising
the cut-off from 9 to 11 does not recover any additional anomaly signal.

Source:
[DeepLog metrics](/Users/harensamarasinghe/Documents/Imperial/Year%204/Final%20Year%20Project/toolkits/AnomaLog/experiments/results/openstack_deeplog_preprocessed_deeplog/44a7a15a01cc/metrics.json)

## Bounded Reconstruction

I rebuilt the retained OpenStack corpus locally from the cached structured
rows, grouped by recovered instance id, and fit DeepLog on the normal training
sessions only. This was intentionally bounded and did not require another full
expensive end-to-end training run from the experiment runner.

The reconstruction was used only to inspect the key model behaviour at the
session level.

Key findings from the 300-epoch reconstruction on the same retained OpenStack
corpus:

| Question | Result |
| --- | --- |
| abnormal sessions with zero top-9 misses | 184 / 198 |
| abnormal sessions with zero top-11 misses | 184 / 198 |
| abnormal sessions where g=11 recovers anything beyond g=9 | 0 |
| abnormal sessions with any top-9 miss | 14 / 198 |
| abnormal sessions with any top-11 miss | 14 / 198 |

This is the main empirical result: almost every abnormal session is accepted as
normal by the DeepLog key model. The failure is therefore not primarily a
top-`g` threshold issue.

The same reconstruction also confirmed that the OpenStack vocabulary is tiny
under the current paper-facing parser:

| Quantity | Count |
| --- | ---: |
| distinct templates in the retained corpus | 14 |
| normal vocabulary | 14 |
| abnormal vocabulary | 14 |
| normal/abnormal vocabulary overlap | 14 |

In other words, the current parser representation makes normal and abnormal
sessions share the same surface templates.

## Parser Findings

The current OpenStack parser:

- recovers `instance_id` and prefixes it with the split name;
- strips the leading `[instance: ...]` tag before Spell;
- normalises UUIDs, IPs, path tokens, instance-storage filenames, hex strings,
  and standalone numbers;
- skips rows without an instance identifier.

Relevant code:
[OpenStack parser](/Users/harensamarasinghe/Documents/Imperial/Year%204/Final%20Year%20Project/toolkits/AnomaLog/anomalog/parsers/structured/parsers.py)

What this means in practice:

- the paper-facing parser variants collapse the OpenStack corpus to 14 stable
  templates;
- the normal and abnormal template vocabularies are identical;
- the parser is therefore suppressing most of the raw surface variation before
  DeepLog ever sees it.

That collapse is doing three different jobs, which should not be conflated:

- hygiene normalisation removes obvious formatting noise;
- identifier/path leakage suppression stops the model from memorising instance
  names, UUIDs, and filesystem paths;
- semantic signal collapse is the remaining effect, and it is where the current
  OpenStack representation now appears to sit.

I also compared the parser inventory against alternative cached variants:

| Variant | Templates | Normal vocab | Abnormal vocab | Overlap |
| --- | ---: | ---: | ---: | ---: |
| current paper-facing variants | 14 | 14 | 14 | 14 |
| `content_only` diagnostic variant | 4,698 | 4,305 | 402 | 9 |

The `content_only` variant shows that raw leakage of instance identifiers and
path fragments creates a huge and mostly useless vocabulary explosion. That is
not a defensible improvement.

The explicit `/var/lib/nova/instances` rule does help suppress filesystem-noise
variants, but it is not the main cause of the recall failure. The recall gap is
already present in the 14-template representation.

## Spell Ablation

I ran one more bounded comparison on the raw OpenStack split files, using the
current structured parser and then switching only the template stage.

| Representation | Train vocab | Normal-test vocab | Abnormal-test vocab | Train / normal overlap | Train / abnormal overlap | Unknown targets normal / abnormal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current Spell | 19 | 20 | 18 | 19 | 18 | 1 / 0 |
| parser-only exact normalisation | 25 | 26 | 24 | 25 | 24 | 1 / 0 |
| raw-content leakage control | 13,346 | 31,510 | 4,713 | 0 | 0 | 34,097 / 5,100 |

The key result is that the `h=10` transition picture is unchanged between
current Spell and parser-only exact normalisation:

| Representation | Normal `h=10` coverage | Abnormal `h=10` coverage | Normal unknown contexts | Abnormal unknown contexts |
| --- | ---: | ---: | ---: | ---: |
| current Spell | 20,908 / 20,964 | 3,116 / 3,129 | 56 | 13 |
| parser-only exact normalisation | 20,908 / 20,964 | 3,116 / 3,129 | 56 | 13 |

That transition-equivalence result is the strongest evidence here: once the
structured parser has removed the volatile OpenStack surface noise, Spell is
largely redundant under the current parser representation. Removing Spell does
not recover meaningful anomaly separation.

The small residual misses under parser-only exact normalisation are concentrated
in only two abnormal sessions. They look like state-progression differences
inside otherwise familiar create/resume/delete sequences rather than a new
behavioural vocabulary:

- `Creating event network-vif-plugged:UUID for instance UUID`
- `VM Resumed (Lifecycle Event)`
- `Instance spawned successfully.`
- `Took NUM seconds to spawn the instance on the hypervisor.`
- `Took NUM seconds to build instance.`

So the remaining anomaly signal is closer to parameter/state variation inside
otherwise normal key sequences than to a clean key-level regime shift.

Relevant audit:
[OpenStack `/var/lib/nova/instances` path normalisation audit](/Users/harensamarasinghe/Documents/Imperial/Year%204/Final%20Year%20Project/toolkits/AnomaLog/experiments/reports/openstack_varlib_path_normalisation_audit.md)

## Markov Comparison

The recorded Markov baseline is a weak but non-zero detector on the same test
split:

| Metric | Value |
| --- | ---: |
| true positives | 11 |
| false positives | 61 |
| false negatives | 187 |
| recall | 0.05555556 |

Source:
[Markov metrics](/Users/harensamarasinghe/Documents/Imperial/Year%204/Final%20Year%20Project/toolkits/AnomaLog/experiments/results/openstack_deeplog_preprocessed_markov/4cab5b0f91a0/metrics.json)

Interpretation:

- Markov is reacting to a small number of unstable transition patterns that
  DeepLog smooths over;
- those detections are not strong enough to suggest a better anomaly model by
  themselves;
- the gap is better explained as DeepLog over-generalising on a highly
  collapsed 14-template representation.

I did not find evidence that the Markov baseline is “finding” a separate, clean
semantic anomaly channel that DeepLog is missing. It is more likely to be
surfacing local ordering instability.

## Paper / Corpus Context

The current local OpenStack corpus should not be assumed to be identical to the
original DeepLog paper corpus.

Useful references:

- [DeepLog paper](https://www2.cs.utah.edu/~lifeifei/papers/deeplog.pdf)
- [LogHub OpenStack overview](https://deepwiki.com/logpai/loghub/2.5-openstack-datasets)
- [logpai/loghub](https://github.com/logpai/loghub)
- [logpai/deep-loglizer](https://github.com/logpai/deep-loglizer)

The important methodological point is that the paper’s OpenStack setup and the
public LogHub-style corpus are not necessarily equivalent. The current local
corpus is best treated as a reproduction target, not as a guaranteed copy of
the paper’s underlying dataset.

## Conclusion

The OpenStack DeepLog failure is best explained by a combination of:

1. aggressive parser collapse,
2. dataset/protocol mismatch relative to the paper,
3. DeepLog over-generalising over a very small template inventory.

It is not explained by:

- unknown-target failures,
- unknown-history failures,
- a g=9 vs g=11 threshold difference,
- or an obvious DeepLog implementation bug.

### Defensible parser stance

Keep the current paper-facing parser as the default. It is already close to the
right side of the trade-off for a faithful reproduction attempt.

### Not justified

- removing the `/var/lib` collapse as a blanket change;
- relaxing the parser into raw identifier explosion;
- claiming the current OpenStack corpus is paper-identical;
- chasing sequence recall by widening `g` alone.

### If more work is needed

The next defensible step would be to recover a corpus closer to the paper’s
original OpenStack protocol rather than widening the normalisation in place.
