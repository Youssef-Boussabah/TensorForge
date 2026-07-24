# TensorForge — project summary

A two-minute overview for anyone landing on this repository.

## What TensorForge is

TensorForge is a from-scratch deep learning framework built in
Python + NumPy — a serious, Daedalus-inspired ML systems project. It
reimplements the core machinery of a framework like PyTorch —
automatic differentiation, neural network modules, optimizers,
checkpointing, CNN support — in small, readable code, plus an
experimental native C++ backend. NumPy is the only dependency. It was
built milestone by milestone (v0.1 through v3.0 and beyond), each one
tested and documented.

## What TensorForge implements

- A `Tensor` with reverse-mode autograd: arithmetic, matmul, reshape,
  reductions, and the common nonlinearities, with broadcasting-aware
  gradients verified against finite differences.
- A module system: `Parameter`, `Module`, `Linear`, `Sequential`,
  activations, `Dropout`, `BatchNorm1d`, `LayerNorm`, `Conv2d`,
  `MaxPool2d`, `Flatten`, with train/eval mode, model summaries, and
  parameter freezing.
- Losses (MSE, cross-entropy, binary cross-entropy — the latter two as
  numerically stable fused ops), metrics, and eval-safe evaluation
  helpers.
- Training tools: SGD, Adam, StepLR scheduling, gradient clipping,
  mini-batching, and train/validation splitting.
- Persistence: save/load for weights, and full checkpoints carrying
  optimizer state, scheduler state, JSON metadata, and optionally the
  RNG state — so training resumes bit-for-bit, dropout included.

## Design principles

Every design decision favors readability and verifiable correctness.
Operations carry short comments explaining their local derivative;
tricky pieces (stable losses, batchnorm statistics, max-pool gradient
routing) explain *why* they work, not just what they do. Where
possible, layers are built by composing existing autograd ops so their
gradients need no new code — and where a fused op is necessary,
finite-difference tests prove the hand-written backward correct.

## Example learning path

The six runnable examples form a progression, each introducing one
idea: linear regression (the bare training loop), XOR (why hidden
layers exist), a 3-class spiral (real classification, mini-batches,
validation), binary classification (logits and stable losses), a
dropout MLP (train mode vs eval mode), and a tiny CNN (convolution,
pooling, and flattening on synthetic images). All are seeded,
dependency-free, and finish in seconds.

## The experimental native line

The advanced branch adds a second, strictly separate framework line: a
native C++ CPU backend reached explicitly through
`tensorforge.experimental`. **Phase A (native CPU runtime) is
complete** — `NativeStorage` → `NativeTensorView` → `NativeTensorCore`
→ `NativeTensor`, with explicit ownership/lifetime, strided views,
broadcasting, sum/mean reductions, and float64/cpu metadata over
ctypes-loaded C++ kernels. **Phase B (native autograd) is complete** —
a Python-managed reverse-mode graph over autograd-unaware kernels,
covering every operation in the backend's `AUTOGRAD_OPS` registry
(elementwise math including the v3.11 `sqrt`/`reciprocal` optimizer
primitives, `matmul`, reductions, and the view ops at Phase-B
completion; Phase D then added the `conv2d` and `maxpool2d` primitives),
with view/broadcast gradients and a defined graph lifetime. **Phase C (the native training stack) is complete**, and so are
**Phase D (the native CNN stack)** and **Phase E (native classification
and stable math)** — the native line trains a
convolutional classifier end to end and resumes it exactly from a
checkpoint. Phase C provides: `NativeParameter` (value
versioning, stale-graph safety), `NativeModule` with atomic state
dictionaries, `NativeLinear`/`NativeReLU`/`NativeSequential`,
`NativeMSELoss`, `NativeSGD`, `NativeAdam` (persistent native moment
state with bias correction and an explicit state lifetime), in-memory
optimizer `state_dict`/`load_state_dict`, pickle-free native
checkpoint files with deterministic bit-identical file resume
(`save_native_checkpoint`/`load_native_checkpoint`), and a
deterministic MLP training proof
(`examples/native_mlp_training.py` — 25 native SGD steps, monotonic
99.5% loss reduction). Phase D adds the CNN layers on top:
`NativeFlatten`, the differentiable `conv2d` operation with its
trainable `NativeConv2d` module, and the `maxpool2d` operation (private
saved winners, scatter backward, no version snapshot) with its
parameter-free `NativeMaxPool2d` module — proven by
`examples/native_cnn_training.py` (40 deterministic NativeAdam steps,
98.6% loss reduction, and a checkpoint-interrupted run that reproduces
the uninterrupted one exactly) and validated under ASan/UBSan.
**Phase E (native classification and stable math) is complete**
(milestones E0–E10): differentiable native `exp` and `log` — the phase's
two backward archetypes — the fused, numerically stable `softmax` and
`log_softmax`, the fused `cross_entropy` from raw logits (its
graph-unaware Core contract and the differentiable
`NativeTensor.cross_entropy` over it, with graph-owned saved
probabilities that are never the logits), the stateless
`NativeCrossEntropyLoss` module, and the deliberately reporting-only
`native_accuracy`. It is proven by
`examples/native_classification_training.py` (a native
Conv2d→ReLU→MaxPool2d→Flatten→Linear classifier on twelve fixed 6×6
images in three classes: 40 deterministic `NativeAdam(lr=0.05)` steps,
loss 1.159638 → 0.000101, accuracy 0.3333 → 1.0000, and a run
checkpointed at step 15 that resumes into a fresh model/optimizer pair
and reproduces the uninterrupted run exactly), characterized by
`benchmarks/benchmark_native_classification.py` (seven
correctness-gated cases, no speed assertion anywhere), and validated
under Release and Debug builds with Clang ASan/UBSan and LeakSanitizer.
**Phase F (native normalization and stateful buffers) is in
progress**: the normalization module surface is complete —
`NativeLayerNorm` (F2), `NativeBatchNorm1d` (F3, the first stateful
native numerical module), and `NativeBatchNorm2d` (F4, NCHW) have all
shipped, and **F5 has proved their state/checkpoint/ownership/graph-safety
contracts by exhaustive test (tests and documentation only, no new
capability)** — while the phase's end-to-end proof, benchmark,
integration, and closure milestones have not — see below. The two
engines never mix: explicit entry via
`NativeTensor.from_array`, explicit exit via `to_numpy()`, no implicit
dispatch. The exact per-operation status lives in the
[native support matrix](native_support_matrix.md).

## Testing and reliability (both lines)

Over 2000 pytest tests cover every feature of both lines: known-value
checks against hand-computed math, finite-difference gradient
verification (stable and native), exact resume-equivalence tests for
checkpointing, NumPy-tripwire tests proving the native paths never
fall back, cross-cutting Phase C, **Phase D, and Phase E** integration
guardrails
(shared/frozen/late-active parameters, failure recovery at every
boundary, graph-version interactions, saved-winner and
saved-probability lifetime, and
lifetime discipline), and guardrail tests keeping docs, examples, and
the public API from drifting. The native C++ kernels additionally have
dependency-free CTest binaries, validated under ASan/UBSan. Native tests
skip cleanly when the backend is not built; CI builds it from source
and runs everything.

## Current limitations

Not production-ready and not a PyTorch replacement. The stable
framework is NumPy on CPU; `Conv2d` and `MaxPool2d` use deliberately
naive loops, and so do their native counterparts (direct nested loops —
no im2col, BLAS, threading, or SIMD). The native line is float64/cpu
only — no CUDA backend, no dtype promotion or casting, no native
normalized training example or normalization benchmark (Phase F is
**in progress**: all three normalization modules have shipped, its
remaining hardening/proof/benchmark milestones have not),
no dropout or native RNG, no data loaders or native
integer tensors, no scheduler or
random-state capture in native checkpoints, and
no dispatch into `tensorforge.Tensor`. Benchmarks are hardware-specific
characterizations, never universal speed claims. No real datasets, no
external ML libraries.

## What comes after v3.0

v3.0 closed the Python framework line. The advanced branch then built
the native line milestone by milestone (v1.x runtime, v2.x autograd,
v3.1–v3.9 training stack) to its first major checkpoint, v3.10, then
the optimizer math primitives (v3.11), the adaptive NativeAdam
optimizer (v3.12), the in-memory optimizer state contract (v3.13),
native checkpoint files with deterministic file resume (v3.14), and
the Phase C guardrails-and-completion milestone (v3.15) — which
**closes Phase C** — and then **Phase D, the native CNN stack
(v3.16), which is complete**: `NativeFlatten`, the differentiable
`conv2d` operation and `NativeConv2d`, the `maxpool2d` operation
(private saved winners, scatter backward) and `NativeMaxPool2d`, a
deterministic end-to-end CNN training run whose checkpoint-interrupted
resume matches it exactly, cross-cutting integration tests, honest CNN
benchmarks, and ASan/UBSan validation. **Phase E — Native
Classification and Stable Math — then completed** (E0–E10): `exp`,
`log`, the fused stable `softmax` and `log_softmax`, the fused
`cross_entropy` Core contract and the differentiable operation over it,
`NativeCrossEntropyLoss`, the reporting-only `native_accuracy`, a
deterministic classification training run with exact checkpoint resume,
an honest characterization benchmark, and full closure validation — all
still float64/CPU, with the checkpoint format unchanged at version 1.

**Phase F — Native Normalization and Stateful Buffers — is the current
phase, and it is in progress.** Milestone
**F0** is complete: it locks the architecture contract in
[native_normalization_design.md](native_normalization_design.md) —
`NativeLayerNorm`, `NativeBatchNorm1d`, and `NativeBatchNorm2d`
composed from existing native operations with **no** new kernel, C ABI
export, or Core method; persistent native running statistics; the rule
that a live mutable running buffer is never captured as a rereadable
graph operand; atomic two-buffer running-statistics updates; and
state/checkpoint integration with format version 1 unchanged. **F0
added design and documentation only — no numerical behavior.**
**F1 is complete** too — a private atomic native-buffer state
transaction, `load_state_dict` refactored onto it, and the
`persistent_buffers` capability-reporting correction, all with **no
normalization mathematics**.
**F2 is complete**: `NativeLayerNorm` — the first native normalization
module, stateless (no buffers, identical in train and eval),
differentiable through the mean and the population variance, and
**composed entirely from existing native operations** (`sqrt(var + eps)`,
no Bessel correction) with no kernel, ABI symbol, Core method, custom
backward, or `NativeTensor` normalization operation. `"NativeLayerNorm"`
has joined `NATIVE_MODULES` and `"layernorm"` has left `UNSUPPORTED`.
**F3 is complete**: `NativeBatchNorm1d` — the **first stateful native
numerical module**, `(N, C)` batch normalization with differentiable
training statistics (gradients through the batch mean *and* the
population variance), persistent native `running_mean`/`running_var`
buffers advanced graph-free by one **atomic two-buffer transaction**
(identities preserved, no parameter version moved), and evaluation from
**graph-safe immutable snapshots** of those buffers, so a later training
step, or a buffer-only state or checkpoint load, cannot change an earlier
eval graph's gradient (a full checkpoint load also replaces
`gamma`/`beta`, which correctly stales that graph through the unchanged
parameter-version rule). It too is composed from existing operations — no
kernel, ABI symbol, Core method, custom backward, or
`NativeTensor.batch_norm` operation — and the checkpoint format stays at
version 1. `"NativeBatchNorm1d"` has joined `NATIVE_MODULES`, while
`"batchnorm"` **stays** in `UNSUPPORTED` because the unqualified name is
only honest once the NCHW shape ships. **F4 is complete**:
`NativeBatchNorm2d` — NCHW `(N, C, H, W)` batch normalization reducing
over N, H, and W, over the **same** shared private implementation, which
it extends with only its rank, reduction axes, `(1, C, 1, 1)` broadcast
layout, and the channels-last permutation its rank-1 affine parameters
need. `"NativeBatchNorm2d"` has joined `NATIVE_MODULES`, and with both
shapes live `"batchnorm"` has **left** `UNSUPPORTED`, which now reads
exactly `("dropout", "float32", "cuda", "amp")`. That completes the
numerical normalization **module** surface. **F5 is complete**: the
exhaustive state/checkpoint, ownership, and graph-safety hardening — a
focused `tests/test_native_normalization_state.py` plus narrow additions
to the generic buffer and checkpoint suites — proves §7–§10 by executable
test (canonical dotted buffer keys, independent state snapshots,
strict/non-strict loads, exact never-casting metadata validation, mixed
parameter/buffer transaction atomicity, buffer identity across state and
checkpoint loads, exact eval-output reproduction, the
buffer-only-versus-full stale-graph distinction, the save/corrupt-load
failure boundaries, eval-graph snapshot safety under `retain_graph` and a
failed retryable backward, and the live-storage baselines). F5 is **tests
and documentation only** — no numerical behavior, no new public
capability, and the checkpoint format stays version 1. But Phase F is not
finished. Milestones **F6–F9 are planned and have not started**: no
deterministic normalized training run with exact resume, no normalization
benchmark, no cross-cutting integration, no closure.
Beyond Phase F
(**not started**): dropout and a native RNG, more activations/math, data
loaders, a CPU optimization phase, then the CUDA
runtime, dtype/AMP work, and Transformer/text and distributed
experiments. See [roadmap.md](roadmap.md) and
[release_history.md](release_history.md) for the full arc.
