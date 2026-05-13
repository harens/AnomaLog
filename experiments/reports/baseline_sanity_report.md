# Baseline Sanity Report

This note summarises the simple baseline runs that were already executed for the
current HDFS and BGL experiment matrices. The point is not to reproduce DeepLog
or DeepCASE here, but to record what the simplest template-statistics baselines
already explain.

## What The Baselines Are

- Naive Bayes is a supervised sequence-level classifier over template phrase
  n-grams. It uses labelled training sequences to estimate class priors and
  class-conditional phrase likelihoods.
- Template Frequency is an unsupervised sequence-level scorer over template
  counts. It learns template frequencies from the train prefix and, when no
  fixed threshold is configured, calibrates from normal training scores only.

Both are sanity baselines. They are useful for checking corpus separability and
template-statistics signal, but they are not paper-faithful DeepLog or
DeepCASE reproductions.

## Measured Results

| dataset | detector | supervision | prediction unit | label unit | F1 | precision | recall |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `hdfs_v1_entity_chronological` | `naive_bayes` | supervised | sequence | sequence | 0.92886503 | 0.86724624 | 0.99990970 |
| `hdfs_v1_entity_chronological` | `template_frequency` | unsupervised | sequence | sequence | 0.91024166 | 0.83526927 | 1.00000000 |
| `bgl_entity_chronological` | `naive_bayes` | supervised | sequence | sequence | 0.33140146 | 0.73124834 | 0.21424966 |
| `bgl_entity_chronological` | `template_frequency` | unsupervised | sequence | sequence | 0.59134225 | 0.44269607 | 0.89027373 |

## Answers

### Are the baselines exploiting obvious label artefacts?

Partly.

The HDFS Naive Bayes run is strongly label-conditioned and still reaches
sequence F1 `0.92886503` with recall `0.99990970`. That is a sign that the HDFS
corpus exposes strong phrase/label correlation. The model is not cheating in an
implementation sense, but it is learning an easy separability signal.

The HDFS template-frequency run is less directly label-conditioned, but it is
still very strong. A simple frequency model reaches sequence F1 `0.91024166`
and recall `1.0`, so template statistics alone already carry substantial signal.

The BGL Naive Bayes run does not show the same blanket separability. Its recall
is only `0.21424966`, so the learned phrase vocabulary is much narrower there.

### Are high HDFS baseline scores scientifically meaningful?

Yes, as sanity checks.

They show that the HDFS split is already highly separable under very simple
template statistics. That matters because any DeepLog or DeepCASE claim on the
same corpus has to beat a surprisingly high easy-baseline floor.

They are not evidence that the baselines discover richer temporal structure
than the deep models. They only say that the dataset itself is easy to separate
with simple statistics.

### Why is BGL Naive Bayes recall low?

The BGL phrase model seems to learn a narrow anomaly vocabulary.

It still has useful precision, so when it predicts anomaly it is often correct.
But low recall means most anomalous sequences do not contain the phrase patterns
that the model found useful during training.

### Do simple baselines already explain part of DeepLog/DeepCASE performance?

Yes.

On HDFS, the simple baselines already sit at a high sequence-level F1, so any
claimed improvement needs to be read against a strong separability floor. On
BGL, template frequency is still materially predictive, so deep models should
be compared against that frequency floor rather than against a blank slate.

### Which baseline results are acceptable to include in the paper?

Include all four only in a clearly labelled sanity-baseline section.

- HDFS Naive Bayes: acceptable as a supervised corpus-separability check
- HDFS Template Frequency: acceptable as an unsupervised frequency floor
- BGL Naive Bayes: acceptable as a lower-bound recall probe
- BGL Template Frequency: acceptable as a simple count-based sanity baseline

Do not present any of them as a DeepLog or DeepCASE reproduction. They predict
sequence labels, not the full event-level or next-event tasks used by the deep
models, so the headline numbers are not directly comparable unless the unit is
matched explicitly.

## Run Notes

These notes are taken from the current checked-in baseline experiment artefacts
under `experiments/results/`.

### HDFS Naive Bayes

- supervision: supervised
- prediction unit: sequence
- label unit: sequence
- what it learns: multinomial Naive Bayes over template phrase n-grams
- label usage: labelled training sequences drive class priors and
  class-conditional phrase likelihoods
- interpretation: the HDFS phrase model is highly label-separable
- paper guidance: include only as a sanity baseline

### HDFS Template Frequency

- supervision: unsupervised
- prediction unit: sequence
- label unit: sequence
- what it learns: sequence scorer based on simple template frequency
- label usage: counts templates from the train prefix; when no threshold is
  configured, calibrates from normal training scores only
- interpretation: simple template frequency is already highly predictive on HDFS
- paper guidance: include it as a sanity floor only

### BGL Naive Bayes

- supervision: supervised
- prediction unit: sequence
- label unit: sequence
- what it learns: multinomial Naive Bayes over template phrase n-grams
- label usage: labelled training sequences drive class priors and
  class-conditional phrase likelihoods
- interpretation: the BGL phrase model is high precision but low recall
- paper guidance: include it as a lower-bound sanity baseline only

### BGL Template Frequency

- supervision: unsupervised
- prediction unit: sequence
- label unit: sequence
- what it learns: sequence scorer based on simple template frequency
- label usage: counts templates from the train prefix; when no threshold is
  configured, calibrates from normal training scores only
- interpretation: template frequency remains useful on BGL but is noisier than
  on HDFS
- paper guidance: include it as a simple count-based sanity baseline, not as a
  direct competitor to deep sequence models
