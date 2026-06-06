# DeepLog Thunderbird Training Performance Note

This note records the bounded profiling pass used to reduce Thunderbird DeepLog
training time without changing the paper-facing key model, labels, split, or
top-`g` evaluation.

## Root Cause

The DeepLog key trainer was spending most of its time repeatedly rebuilding
the same per-sequence training windows and then processing them in many tiny
microbatches. The one-hot expansion cache introduced in the refactor also
risked materialising a corpus-sized dense tensor on the real Thunderbird run.
On the bounded synthetic Thunderbird-like profile:

- the key-fit path was dominated by `torch.lstm` and the backward pass;
- rebuilding template indexes and eligible target masks every epoch added a
  sizeable Python-side tax;
- the old 64-example microbatch cap forced many more LSTM and backward calls
  than the model actually needed;
- scoring was not the main bottleneck on the same scale test.

In other words, the problem was throughput fragmentation rather than an
incorrect model or a pathological evaluation rule.

## What Changed

- The key-model microbatch cap increased from `64` to `256`.
- The training loop now caches per-sequence template indexes, history windows,
  and eligible target indexes once, then reuses that compact CPU state across
  epochs.
- Each optimiser batch now arrives as indexed tensors, and the one-hot
  expansion happens per minibatch on the target device instead of being cached
  per sequence.
- If a CUDA microbatch still exceeds available memory, the trainer halves the
  microbatch and retries the same optimiser batch rather than failing the run.
- The training order, loss, optimiser, top-`g` rule, and scoring semantics are
  unchanged.

## Bounded Timing Table

Synthetic Thunderbird-like benchmark:

- `400` normal sequences
- `100` events per sequence
- vocabulary size `4,428`
- `history_size = 10`
- `epochs = 2`
- `batch_size = 2,048`
- CPU device

| Variant | Microbatch cap | Wall time |
| --- | ---: | ---: |
| Reference trainer | `64` | `8.29 s` |
| Optimised trainer | `256` | `5.04 s` |

## Equivalence Check

A fixed-seed small-corpus comparison between the old streaming trainer and the
new batch-tensor path produced identical fitted parameters to machine
precision on CPU for the reference case used in the regression test.

## Memory Bound

The cached sequence tensors stay on CPU, and only one microbatch is moved to
the target device at a time. The one-hot expansion is transient and bounded to
the current microbatch. The microbatch retry path further reduces the chunk
size on CUDA OOM, so the trainer degrades to smaller updates instead of
crashing on a constrained card.

## Recommended Cluster Settings

For the Thunderbird fixed-window run on the A16 cluster:

- keep `history_size = 10`
- keep `top_g_values = [1, 3, 5, 7, 9]`
- keep `batch_size = 2048`
- leave `parameter_detection_enabled = false` for the paper-facing run
- use the bounded benchmark corpus only for profiling, not for headline metrics
