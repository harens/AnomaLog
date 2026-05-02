# Scoped DeepLog Core v1 Design Note

This note documents the current DeepLog implementation in AnomaLog's
`experiments/` layer. The goal of this pass is a clean, honest "core DeepLog"
implementation rather than a claim of full paper fidelity.

## Scope Of This v1

This implementation covers the paper's two anomaly-detection cores:

- a stacked-LSTM next-log-key model
- a per-template parameter-value model
- Gaussian calibration over held-out parameter residuals

This implementation does not cover the paper's diagnosis and adaptation
extensions:

- workflow diagnosis / workflow model construction
- online false-positive updates

## Implemented Components

- Key-model anomaly detection with a stacked LSTM over fixed-length template
  histories and top-`g` acceptance at inference.
- One parameter LSTM per template when that template has at least one
  modelable numeric feature.
- Per-feature normalisation for each template-specific parameter model.
- Optional inclusion of `dt_prev_ms` when normal training data for that
  template contains at least one non-null elapsed-time value.
- Masked parameter vectors so missing values do not contribute to residual MSE.
- Masked parameter-model training loss so unobserved target dimensions do not
  train the regressor toward zero.
- Temporal validation splitting for parameter calibration: each per-template
  series contributes a held-out tail of history-target pairs.
- Sequence-level anomaly outputs that preserve event-level key and parameter
  findings.
- Manifest reporting for parameter-model coverage and per-template feature
  counts.
- Detector-owned next-event diagnostics derived from the key model's ranked
  predictions. These diagnostics are separate from anomaly scoring and remain
  available in the model manifest for the latest scoring run.

## Remaining Gaps Vs The Paper

- No workflow diagnosis or workflow/FSA construction.
- No online update path for false positives.
- The parameter schema policy is stricter than the paper text: a parameter
  position is modeled only if every observed value for that template-position
  pair is numeric in normal training data.
- Gaussian parameter calibration is implemented as a Normal fit over held-out
  residual MSEs. The paper motivates Gaussian modeling but does not fully pin
  down these exact repository mechanics.
- Elapsed time is derived from the existing `TemplateSequence` event payload
  (`dt_prev_ms`) instead of a separate DeepLog-specific preprocessing path.

## Inference Policies

- Unknown history templates:
  treated as immediate key anomalies rather than being scored through a
  synthetic vocabulary item. Findings report the unknown history templates
  explicitly.
- Unknown target templates:
  treated as key-model anomalies because the trained vocabulary contains no
  probability for the observed template.
- Next-event diagnostics:
  default to `full_dataset` so the diagnostic output is directly comparable
  with DeepCASE. The diagnostic vocabulary policy is configurable on
  `DeepLogModelConfig`; `train_only` remains available when you want the
  report restricted to the training vocabulary used by the key scorer.
- Missing or non-numeric parameter values:
  positions not admitted by the strict schema are never modeled. For admitted
  positions, missing values are padded with `0.0` internally but masked out of
  normalisation, training loss, and residual MSE. This keeps the deployed
  input shape stable while leaving the paper's "input is the parameter value
  vector" structure intact. Serialised findings expose masked positions as
  `None`.
- `dt_prev_ms`:
  included only when `include_elapsed_time = true` and that template has seen
  at least one non-null `dt_prev_ms` value in normal training data. If a
  scored event has no elapsed-time value, that feature is masked out for the
  event.

## Validation And Gaussian Calibration

Parameter calibration is done per template, per time series:

1. Gather the ordered raw parameter vectors for one template inside each
   normal training sequence.
2. Convert each series into `history_size -> next vector` prediction pairs.
3. Reserve the temporal tail of each series for validation, using
   `ceil(pair_count * validation_fraction)` held-out targets while keeping at
   least one train pair and one validation pair whenever possible.
4. Fit normalisation statistics on the training prefixes only.
5. Train the template's parameter LSTM on the training pairs only.
   The regression loss is masked, so only observed target dimensions
   contribute to optimisation.
6. Score the held-out validation pairs after training with the same masked
   residual policy.
7. Fit a Gaussian to those held-out residual MSE values and use the requested
   confidence interval as the acceptance region.

This means Gaussian residuals are produced from temporally held-out
history-target pairs, not from a global pooled slice across templates.

## Paper-To-Code Traceability

| Paper component | Paper behavior | Code location(s) | Status | Notes / deviations |
| --- | --- | --- | --- | --- |
| Log-key anomaly model | Stacked LSTM predicts the next log key from recent history; actual key is accepted when it appears in the top-`g` predictions | `experiments/models/deeplog/`: `DeepLogModelConfig`, `KeyLSTM`, `_fit_key_model`, `score_key_sequence`, `_score_key_event` | implemented | Uses explicit one-hot key histories to stay close to the paper's formulation. |
| Key-model OOV handling | Detect abnormal next events when the observed key is not represented by the learned model | `experiments/models/deeplog/`: `score_key_sequence`, `_score_key_event` | partial | Unknown targets are anomalous. Unknown history windows are treated as immediate anomalies instead of being scored through a synthetic token. |
| Parameter-value model | Train a separate sequence model for each log key/template | `experiments/models/deeplog/`: `ParameterLSTM`, `ParameterModelState`, `_fit_parameter_models` | implemented | Template-specific LSTMs are skipped when no modelable numeric features exist. |
| Parameter schema construction | Build per-template vectors from parameters and timing information | `experiments/models/deeplog/`: `ParameterFeatureSchema`, `build_parameter_schemas`, `raw_parameter_vector_for_event` | partial | Strict numeric-position policy; mixed numeric/string positions are dropped entirely. |
| Elapsed-time feature | Include elapsed time as part of parameter/performance anomaly modeling | `experiments/models/deeplog/`: `_DT_FEATURE_NAME`, `build_parameter_schemas`, `raw_parameter_vector_for_event` | partial | Uses `dt_prev_ms` from AnomaLog sequence events and only includes it when present in normal training data for that template. |
| Feature normalisation | Normalise parameter features before sequence modeling | `experiments/models/deeplog/`: `NormalisationStats`, `_normalisation_for_raw_series`, `_normalize_vector`, `_denormalize_vector` | implemented | Fitted on training prefixes only; masked values are excluded. |
| Missingness handling | Represent missing parameter values without training or scoring against invented zeros | `experiments/models/deeplog/`: `raw_parameter_vector_for_event`, `_normalize_vector`, `masked_regression_loss`, `_score_parameter_sequence` | implemented | Missing positions are zero-filled for shape stability but masked out of normalisation, loss, and residual MSE. |
| Residual scoring | Compare predicted and observed parameter vectors with MSE | `experiments/models/deeplog/`: `_MaskedRegressionLoss`, `_masked_mse`, `_parameter_pair_residual`, `_score_parameter_sequence` | implemented | Both training loss and calibration/inference residuals use the same target mask semantics. |
| Gaussian calibration | Model validation residuals with a Gaussian and threshold anomalies by confidence bounds | `experiments/models/deeplog/`: `GaussianThreshold`, `fit_gaussian_threshold`, `build_parameter_datasets`, `_fit_parameter_models` | implemented | Residuals come from per-series temporal validation tails; the exact calibration mechanics are repository-defined. |
| Sequence anomaly decision | Flag an event when either the key model or parameter model fires | `experiments/models/deeplog/`: `DeepLogDetector.predict`, `parameter_anomaly_score` | implemented | Follows the paper's detection order: check the key model first, then score parameters only for events whose key is accepted as normal. |
| Diagnosis output | Explain anomalies with workflow-aware diagnosis | `experiments/models/deeplog/`: `DeepLogEventFinding`, `DeepLogPredictionOutcome` | partial | The repo exposes event-level triggers, not the paper's workflow diagnosis system. |
| Workflow construction / diagnosis | Separate tasks and construct workflows or FSAs for diagnosis | not implemented | not implemented | Explicitly out of scope for this pass. |
| Online false-positive updates | Incrementally adapt the model after false positives | not implemented | not implemented | Explicitly out of scope for this pass. |
