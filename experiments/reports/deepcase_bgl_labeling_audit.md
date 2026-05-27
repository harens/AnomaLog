# BGL DeepCASE Label-Usage Audit

This note records how the current BGL DeepCASE run relates to the original
DeepCASE paper protocol and how the implementation should be described.

## Summary

The BGL DeepCASE variant in this repository is best described as:

- an oracle approximation of operator labelling, and
- supervised contextual clustering at the interpreter scoring stage.

It is not paper-faithful manual DeepCASE, because the paper expects a human
operator to inspect clusters and assign policies, whereas this run projects
ground-truth labels onto training samples and uses them to derive cluster
scores.

## What The Paper Does

The official DeepCASE protocol is:

- learn contextual representations from event contexts
- cluster those contextual samples with the interpreter
- let an operator assign a score or policy to each cluster
- use the scored clusters for semi-automatic prediction

The paper and official documentation expose a helper for reconciling labels
before cluster scoring, but that helper is still a proxy for manual cluster
labelling rather than an automated operator workflow.

Sources:

- DeepCASE paper PDF: https://thijsvane.de/static/homepage/papers/deepcase.pdf
- Official docs overview: https://deepcase.readthedocs.io/en/latest/usage/overview.html
- Official code example: https://deepcase.readthedocs.io/en/latest/usage/code.html
- Official API reference: https://deepcase.readthedocs.io/en/latest/reference/deepcase/deepcase.html

## What The BGL Implementation Does

The current BGL DeepCASE path:

- builds one event-centred `(context, event)` sample per eligible event
- uses target event labels when present
- falls back to the parent sequence label only when event labels are missing
- passes those labels into `interpreter.fit(..., scores=batch.scores, ...)`
- aggregates them with `cluster_score_strategy = "max"`

Relevant implementation points:

- [`experiments/models/deepcase/shared.py`](<file:///Users/harensamarasinghe/Documents/Imperial/Year%204/Final%20Year%20Project/toolkits/AnomaLog/experiments/models/deepcase/shared.py>)
- [`experiments/models/deepcase/detector.py`](<file:///Users/harensamarasinghe/Documents/Imperial/Year%204/Final%20Year%20Project/toolkits/AnomaLog/experiments/models/deepcase/detector.py>)
- [`experiments/configs/models/deepcase.toml`](<file:///Users/harensamarasinghe/Documents/Imperial/Year%204/Final%20Year%20Project/toolkits/AnomaLog/experiments/configs/models/deepcase.toml>)

## Classification

### Paper-faithful

- ContextBuilder training on event-centred samples
- interpreter clustering with DBSCAN
- semi-automatic prediction on scored clusters
- DeepCASE abstention codes and confidence thresholds
- paper-default hyperparameters

### Oracle approximation

- using ground-truth labels to stand in for the operator-provided cluster
  scores
- using `score_clusters`-style aggregation over those labels
- treating BGL as a benchmark extension rather than a paper reproduction

### Supervised

- event-level supervision during training when BGL event labels are available
- cluster scoring guided by labels rather than manual policy assignment

## Leakage Check

I did not find evidence that future or test labels leak into:

- fitting
- cluster formation
- thresholding

The training batch is built from the training split only, and the labels used
for cluster scoring come from that training data path. Prediction-time labels
are used for diagnostics and evaluation only.

## Safe Wording

Safe to say:

- “official DeepCASE library integration”
- “oracle cluster-scoring approximation of manual operator labelling”
- “event-label-supervised contextual clustering on BGL”
- “offline benchmark extension”

Avoid unless heavily qualified:

- “paper-faithful manual DeepCASE”
- “human-in-the-loop operator workflow”
- “full DeepCASE reproduction on BGL”
- “unsupervised DeepCASE on BGL”

## Recommendation

For paper text and internal documentation, describe the BGL DeepCASE variant
as an `oracle operator` or `oracle manual-labelling` approximation, not as a
faithful reproduction of the paper's manual protocol.
