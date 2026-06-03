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
- Spell template training now streams the raw log text into Spell's temporary
  input file instead of first buffering the full corpus in memory.
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
- Key-only experiment defaults: `deeplog_default` now keeps parameter-value
  detection disabled unless a run opts in explicitly.
- Detector-owned next-event diagnostics derived from the key model's ranked
  predictions. These diagnostics are separate from anomaly scoring and are
  exposed in run metrics for the full test scoring pass.

### Paper, Script, And Repo

The main place where the paper, the attached reproduction script, and this repo
diverge is short-session handling:

| Topic | Paper | Attached scripts | AnomaLog DeepLog |
| --- | --- | --- | --- |
| Short-session scoring | Fixed-length history windows; no explicit padding rule is stated | `LogKeyModel_predict.py` left-pads short test sessions so they still produce one decision | No padding: sessions shorter than `history_size + 1` yield no key decision, which matches the paper more closely |
| Training windows | Sliding windows from normal sequences | `LogKeyModel_train.py` also uses sliding windows only | Same core training shape, expressed through AnomaLog's sequence abstractions |
| Benchmark scope | HDFS key-model benchmark; parameter modelling is described separately | Key-model script only | Key model plus optional parameter anomaly modelling, richer diagnostics, and explicit run metrics |

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
- Short sessions:
  the default implementation does not left-pad short sessions. Sessions
  shorter than `history_size + 1` therefore do not yield a key-model decision,
  matching the fixed-window interpretation in the paper more closely. The
  optional `short_session_padding_fidelity = true` model-set variant restores
  the attached legacy prediction script's padded last-event decision for
  compatibility with historical DeepLog artefacts.
- Next-event diagnostics:
  default to `full_dataset` so the diagnostic output is directly comparable
  with DeepCASE. The diagnostic vocabulary policy is configurable on
  `DeepLogModelConfig`; `train_only` remains available when you want the
  report restricted to the training vocabulary used by the key scorer. The
  default `metrics.json` keeps the paper-facing aggregates only; run with
  `--debug-reporting` if you need the full segment bookkeeping and raw top-k
  hit counts while debugging a reproduction.
- Scoped metrics:
  `metrics.json` now carries task-aware blocks such as
  `event_level_detection`, `sequence_level_detection`, and
  `next_event_prediction` inside the canonical `metric_blocks` object plus
  top-level metadata for `primary_metric_scope`, `evaluation_unit`, and the
  active split policy. Each block is keyed by scope, so the per-block payload
  no longer repeats `metric_scope`. DeepLog BGL stream runs should foreground the
  event-level block and treat the sequence-level block as invalid or
  diagnostic-only when the sequence labels are single-class.
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
| Sequence anomaly decision | Flag an event when either the key model or parameter model fires | `experiments/models/deeplog/`: `DeepLogDetector.predict`, `parameter_anomaly_score` | implemented | Follows the paper's detection order: check the key model first, then score parameters only for events whose key is accepted as normal. The HDFS paper-reproduction config disables parameter detection entirely because the paper's HDFS benchmark reports the key model only. |
| Top-`g` replay analysis | Re-evaluate the key model against multiple configured acceptance thresholds without re-fitting | `experiments/models/deeplog/`: `DeepLogModelConfig.top_g_values`, `DeepLogKeyFinding.actual_rank`, `DeepLogDetector.run_metrics` | implemented | The detector records the exact rank of each scored key so the run summary can replay the paper rule for every configured `g` cut-off without retraining. |
| Diagnosis output | Explain anomalies with workflow-aware diagnosis | `experiments/models/deeplog/`: `DeepLogEventFinding`, `DeepLogPredictionOutcome` | partial | The repo exposes event-level triggers, not the paper's workflow diagnosis system. |
| Workflow construction / diagnosis | Separate tasks and construct workflows or FSAs for diagnosis | not implemented | not implemented | Explicitly out of scope for this pass. |
| Online false-positive updates | Incrementally adapt the model after false positives | not implemented | not implemented | Explicitly out of scope for this pass. |

## Paper Reproduction Investigation (2026-05-05)

The detailed reproduction audit now lives in
[experiments/reports/deeplog_paper_reproduction_investigation.md](../reports/deeplog_paper_reproduction_investigation.md).

This section keeps the short version in the main DeepLog note:

- the reproduction configs now use generic split modes, not a DeepLog-only
  pipeline;
- the HDFS paper-facing registry now exposes both the fixed-history DeepLog
  default and the legacy short-session padding compatibility variant;
- the experiment-layer `deeplog_default` model now defaults to key-only scoring,
  and the HDFS paper-facing bundles pin `top_g_values = [1, 3, 5, 7, 9]`;
- for the official preprocessed HDFS regime, split-file prefixes are now used
  directly to assign train/test membership for entity sequences, avoiding an
  extra raw-entry split indirection while keeping the same preset surface;
- the same direct split-prefix assignment now applies to the preprocessed
  OpenStack regime (`openstack_train`, `openstack_test_normal`,
  `openstack_test_abnormal`) for the same reason;
- the OpenStack parser now mines template content from the raw message body,
  prefixes recovered VM `instance_id` values with the split name so the file
  boundary survives grouping, strips the leading session tag before Spell, and
  canonicalises volatile UUID, IP, instance-storage filename, path-segment,
  hex, and numeric tokens in the current OpenStack preset;
  the Figure 9 approximation config now uses an explicit `h = 3`, `L = 1`,
  and `alpha = 256` approximation without claiming those values are stated as
  the Section 5.2 OpenStack defaults;
  the archive's `pending task (...)` states are kept visible for audit and
  future parameter modelling rather than being merged away;
- `sequence.split.mode` supports `raw_entry_prefix_count`,
  `raw_entry_prefix_fraction`, and `raw_entry_prefix_normal_fraction`;
- `sequence.split.application_order` makes split-before-grouping explicit;
- `sequence.split.straddling_group_policy` makes boundary handling explicit;
- `grouping = "chronological_stream"` gives BGL a deterministic entry-stream
  grouping mode while preserving the existing entity-based default for the
  benchmark configs;
- chronological stream sequences are marked continuous internally, so DeepLog
  carries history across internal batch boundaries without a user-facing switch;
- chronological stream batches remain intact as memory containers, and BGL now
  uses explicit per-event training and evaluation masks so normal targets can
  train even when a batch also contains anomalies or post-cutoff context;
- the BGL 10% paper probe now uses the same normal-entry prefix policy as the
  1% probe, so the preserved batches are mixed-context containers rather than
  wholly normal sequences.

I noticed a small ambiguity in the DeepLog BGL paper wording around the training split.

The paper says:
“first 1% normal log entries” and “first 10% log entries”.

My interpretation is that both settings are intended to be effectively normal-only training, since DeepLog is fundamentally trained on normal behaviour, and the omission of “normal” in the 10% case is probably shorthand/slightly imprecise wording.

### Summary

| Dataset | Status | Main remaining blockers |
| --- | --- | --- |
| HDFS | Split protocol now expressible, but not fully paper-faithful | Current dataset/version mismatch; paper counts still do not match the cited first-100k split. |
| BGL | Split protocol now expressible, training eligibility is explicit, and event-level evaluation is stable | Online update still absent. |

### Added Reproduction Configs

- HDFS:
  - `experiments/configs/datasets/hdfs_v1_deeplog_paper_entry100k_split_partial.toml`
  - `experiments/configs/datasets/hdfs_v1_deeplog_paper_entry100k_assign_first.toml`
- BGL:
  - `experiments/configs/datasets/bgl/bgl_deeplog_ccs2017_paper_1pct_normal_entry_stream_no_online.toml`
  - `experiments/configs/datasets/bgl/bgl_deeplog_ccs2017_paper_10pct_entry_stream_no_online.toml`

### Where To Look Next

- Audit and count details: [DeepLog paper reproduction report](../reports/deeplog_paper_reproduction_investigation.md)
  and [DeepLog template inventory audit](../reports/deeplog_template_inventory_audit.md)
- Audit command: `uv run python -m experiments.runners.audit_deeplog_data`
