# DeepLog Key-Training Stability Note

This note records the reason for the DeepLog key-model training change and the
temporary validation checks used to assess paper fidelity.

## Summary

The key-model trainer now splits large minibatches into smaller microbatches
and accumulates gradients before the optimiser step. The purpose is to reduce
peak GPU memory usage when the one-hot history tensor or LSTM activations
would otherwise exceed device memory.

## Findings

- The change does not alter the DeepLog architecture.
- The change does not alter the loss function, optimiser, or scoring rule.
- A temporary CPU equivalence check with a fixed seed produced identical
  fitted parameters with and without microbatch splitting.
- The change is therefore a training-memory implementation detail rather than
  a paper-level modelling change.

## Implementation Detail

- `_optimise_key_training_batch` now partitions each minibatch into smaller
  slices, backpropagates each slice with proportional scaling, and performs a
  single optimiser step after the full original batch has been accumulated.
- The regression test in `tests/unit/test_deeplog.py` forces the microbatch
  cap down to a tiny value and verifies that the key model is invoked on
  chunked batches.

## Validation Commands

- `uv run pytest -q tests/unit/test_deeplog.py -k 'streams_batches_without_materialising_all_examples or splits_large_training_batches_into_microbatches'`
- `uv run ruff format anomalog tests experiments`
- `uv run ruff check --fix anomalog experiments tests`
- `uv run ty check anomalog tests experiments`

