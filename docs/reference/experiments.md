# Experiments

The `experiments` package houses the repository-local configuration, model,
and result helpers that sit on top of AnomaLog preprocessing.

Use this page when you want the module surface for the experiment layer rather
than the workflow overview in [Experiments](../experiments.md).

```pycon
>>> from experiments import ConfigError
>>> from experiments.config import load_experiment_bundle
>>> from experiments.models import model_names
>>> "naive_bayes" in model_names()
True
```

## `experiments`

::: experiments

## `experiments.config`

::: experiments.config

## `experiments.models`

::: experiments.models

## `experiments.results`

::: experiments.results
