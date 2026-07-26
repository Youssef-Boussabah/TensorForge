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

The explicit experimental native line is a second, strictly separate
framework line: a native C++ CPU backend, merged into `main` and reached
only through the `tensorforge.backends` and `tensorforge.experimental`
namespaces. **Phase A (native CPU runtime) is
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
**Phase F (native normalization and stateful buffers) is complete
(F0–F9)**: the normalization module surface is complete —
`NativeLayerNorm` (F2), `NativeBatchNorm1d` (F3, the first stateful
native numerical module), and `NativeBatchNorm2d` (F4, NCHW) have all
shipped, **F5 proved their state/checkpoint/ownership/graph-safety
contracts by exhaustive test**, **F6 shipped a deterministic
normalized training example with exact checkpoint resume** (tests and
documentation only, no new capability), **F7 shipped the honest
benchmark characterization**
(`benchmarks/benchmark_native_normalization.py` — nine correctness-gated
cases, no speed assertion anywhere), **F8 shipped the cross-cutting
integration and semantic guardrails** (`tests/test_native_phase_f.py`),
and **F9 closed the phase** under Release and Debug builds with Clang
ASan/UBSan and LeakSanitizer — validation and documentation only, adding
no numerical capability. **Phase G (native RNG and Dropout) is the
current phase and is in progress: milestones G0, G1, G2, G3, and G4 have
landed.** G0, the architecture
contract in [native_rng_dropout_design.md](native_rng_dropout_design.md),
locks Python-managed generator state, stateless native
random kernels, inverted Dropout with a graph-owned multiplier mask, one
generator call consumed per successful stochastic forward, generator
registration on `NativeModule`, and native checkpoint version 2 — as
**design, documentation, and guardrails only**. G1 shipped
`NativeGenerator` and module generator-state ownership, G2 the
stateless `dropout_forward` **Core** kernel and its C ABI, and G3 the
differentiable `NativeTensor.dropout(p, *, generator)` over that Core —
an explicit keyword-only generator, the graph-owned multiplier mask whose
`multiply` is the whole backward, and the reserve/commit/abandon call
transaction, and G4 the `NativeDropout` module over it — stochastic in
training, the input object itself in evaluation, identity at `p == 0`,
over one registered generator it owns or shares. Above the module nothing
exists: the checkpoint
format is still version 1 and does not persist generator state, so
**exact stochastic resume does not exist yet**, and
`dropout` is still listed unsupported beside `float32`,
`cuda`, and `amp`. The two
engines never mix: explicit entry via
`NativeTensor.from_array`, explicit exit via `to_numpy()`, no implicit
dispatch. The exact per-operation status lives in the
[native support matrix](native_support_matrix.md).

## Testing and reliability (both lines)

Over 3,600 pytest tests cover every feature of both lines: known-value
checks against hand-computed math, finite-difference gradient
verification (stable and native), exact resume-equivalence tests for
checkpointing, NumPy-tripwire tests proving the native paths never
fall back, cross-cutting Phase C, **Phase D, Phase E, and Phase F**
integration
guardrails
(shared/frozen/late-active parameters, failure recovery at every
boundary, graph-version interactions, saved-winner,
saved-probability, and normalization-snapshot lifetime, atomic
running-statistics transactions, and
lifetime discipline), and guardrail tests keeping docs, examples, and
the public API from drifting. The native C++ kernels additionally have
dependency-free CTest binaries, validated under ASan/UBSan. Native tests
skip cleanly when the backend is not built; CI builds it from source
and runs everything.

The completed Phase-F workflow's final full Python run reported
**3,632 passed and 5 skipped** with the Release backend active. All five
skips are the standing "the backend *is* built, so the unavailable path
cannot be forced" cases — never a missing-backend skip. Alongside it,
Phase F's closure recorded 10/10 native CTests in both Release and Debug
and 1,968 sanitized normalization-focused Python tests with zero ASan
and zero UBSan diagnostics.

## Current limitations

Not production-ready and not a PyTorch replacement. The stable
framework is NumPy on CPU; `Conv2d` and `MaxPool2d` use deliberately
naive loops, and so do their native counterparts (direct nested loops —
no im2col, BLAS, threading, or SIMD). The native line is float64/cpu
only — no CUDA backend, no dtype promotion or casting,
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

**Phase F — Native Normalization and Stateful Buffers — is the latest
phase, and it is complete (F0–F9).** Milestone
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
capability, and the checkpoint format stays version 1. **F6 is complete**:
`examples/native_normalization_training.py` trains a
`Linear → BatchNorm1d → ReLU → LayerNorm → Linear` regressor for 24
deterministic `NativeAdam` steps with `NativeMSELoss` (98.9% loss
reduction), proves two uninterrupted runs bit-identical, and resumes an
interrupted run into a fresh model/optimizer pair that reproduces the
remaining losses, every parameter, the NativeAdam state, both BatchNorm
`running_mean`/`running_var`, the final training-step prediction, and the
final evaluation-mode output exactly (format version 1 unchanged, training
flags runtime-only) — one example and its integration test, adding no
capability. **F7 is complete**:
`benchmarks/benchmark_native_normalization.py` characterizes the stack
with nine cases (both LayerNorm directions, all three BatchNorm1d paths,
all three BatchNorm2d paths, and one complete F6-style normalized
training step), each **correctness-gated before any timing**, six against
`stable_tensorforge` equivalents on identical state and three (the
BatchNorm2d shapes) labelled `native_only` because the stable line has no
public `BatchNorm2d` to time against, though those keep a rigorous NumPy
NCHW and transformed-oracle correctness gate. Medians with min, max, and
spread after warm-up; `--smoke`/`--json`; **no result file, no speed
assertion, no committed timing number, and no CI timing threshold** —
measurement only, adding no capability.
**F8 is complete**: `tests/test_native_phase_f.py` proves the
cross-cutting interactions — one integrated convolution / BatchNorm2d /
pooling / linear / BatchNorm1d / LayerNorm classifier over raw logits and
the fused loss, trained by `NativeAdam` and resumed **exactly** from one
version-1 checkpoint (all four running-statistic buffers and the
evaluation-mode output included); the three saved-resource families
coexisting safely in one eval graph; buffer versus parameter mutation
attributed to the right cause; the versioning archetypes; shared and
frozen parameters; a non-contiguous NCHW input; honest per-boundary
failure atomicity; and reality-derived capability guardrails — tests and
documentation only, adding no capability.
Milestone **F9 is complete**: the phase closure — fresh Windows Release
**and** Debug builds each passing the full existing 10-test CTest suite
with zero project warnings and the active runtime proved to stay
Release; a fresh Clang 18.1.3 ASan+UBSan build whose instrumentation is
*proved* (22 `__asan*` and 13 `__ubsan*` dynamic symbols; the library
will not load without the sanitizer runtime); 10/10 sanitized native
CTests with leak detection enabled; 1,968 sanitized
normalization-focused Python tests with zero ASan and zero UBSan
diagnostics; the F6 example and the F7 benchmark smoke path clean under
the sanitized library; and a practical LeakSanitizer lifecycle returning
native live storage **exactly** to baseline, its remaining process-exit
allocations identified honestly as CPython/NumPy shutdown retention with
no TensorForge frame and no suppression file — **validation and
documentation only, adding no numerical capability**. Closing Phase F
closes that phase only; the native line remains experimental,
float64/CPU, and explicitly scoped.

**Phase G — Native RNG and Dropout — then opened, and it is in
progress.** Milestone **G0 is complete**: the architecture contract in
[native_rng_dropout_design.md](native_rng_dropout_design.md), which locks
`NativeGenerator` (an explicit 64-bit seed and call counter with an
algorithm identifier, owning no native resource), stateless native random
kernels, inverted Dropout with a graph-owned multiplier mask, the
one-call-per-successful-forward transaction and the lock-protected,
token-validated reservation protocol that guarantees it (no two callers
can receive the same call index; state replacement is refused while a
reservation is live; parallel stochastic execution is explicitly not
claimed), generator state as a fourth `NativeModule` registration
category, native checkpoint version 2 recording the generator **alias
topology** — every registered path and its canonical target, so
shared-versus-independent identity is restored and every mismatch fails
before any live state changes — with its version-1 compatibility rule,
whole-checkpoint transaction atomicity under any ordinary synchronous
failure (external process death being the only documented exception),
and the G0–G10 ladder — **design,
documentation, and guardrails only, adding no numerical behavior**.

Milestone **G1 is complete**: `NativeGenerator` now exists as
pure-Python random *state* — the four locked fields (algorithm
identifier, algorithm version, unsigned 64-bit seed, and a counter of
committed stochastic calls) as read-only properties, atomic
`state()`/`load_state()`/`reseed()`/`reset()`, exact-`int` validation,
one OS-entropy draw through `secrets` for `seed=None`, identity rather
than value semantics with copying and pickling refused, no native
storage and no `close()`, and the lock-protected token-validated call
transaction that makes a committed call index provably unique. Generators
became a **fourth** `NativeModule` registration category beside
parameters, buffers, and children, with the same deterministic,
identity-deduplicated, cycle-safe traversal and their own
`generator_state_dict()` / `load_generator_state_dict()` surface, leaving
`state_dict()` tensor-only and unchanged. **G1 generates no random
values by itself.**

Milestone **G2 is complete**: the deterministic **stateless
Dropout-forward Core**. The exact locked `tensorforge.splitmix64`
derivation — the `mix64` finalizer, the per-call stream key
`mix64(seed + GOLDEN * (call_index + 1))`, the per-element bits
`mix64(stream + GOLDEN * (element + 1))`, the uniform
`(bits >> 11) * 2**-53`, and the strict `u < p` drop test — now lives as
hidden `namespace tf` functions in `cpp/src/random.cpp`, beside an
inverted-Dropout float64 CPU kernel that writes the output **and** the
private multiplier mask in one pass, and the self-validating guarded
export `tf_core_dropout_forward`. On the Python side that is one ctypes
declaration (the whole key as two `c_uint64` arguments), one entry in
`TENSOR_CORE_OPS` (`"dropout_forward"`), one checked kernel, and the
`NativeTensorCore.dropout_forward(p, *, seed, call_index)` /
`_dropout_forward_with_mask` pair. The Core is **stateless** — it
reserves, commits, cancels, inspects, and mutates no `NativeGenerator`,
and no C++ unit holds random state — and randomness is keyed by the
**logical** row-major element index, so a transposed, narrowed, or
nonzero-offset view gets the same mask as a contiguous tensor of the same
shape. Committed known-answer vectors are asserted identically in the
native CTest and the Python suite.

Milestone **G3 is complete**: the differentiable
`NativeTensor.dropout(p, *, generator)`, which is where the G1 state
transaction and the G2 stateless Core finally meet. Its whole registry
footprint is one name, `"dropout"`, appended to `AUTOGRAD_OPS` — no C++,
no C ABI symbol, no ctypes declaration, no Core method, no module, no
export, and no checkpoint-format change. The generator is **required and
keyword-only**: there is no default, process-global, or module-global
stream, no implicit per-call generator, and no NumPy or Python `random`
fallback, and `p` goes through the *same* validator the Core uses rather
than a second rule. The operation owns the call transaction — validate,
reserve one call, run the Core **outside** the generator's lock with the
reservation's own seed and index, build the graph, and commit as the
**last** state-changing action — so one successful stochastic forward
consumes exactly one call with or without gradients, `p == 0` returns the
caller's own tensor object having reserved and allocated nothing, and
every ordinary failure before the commit releases the result, cancels the
reservation, and leaves the same unconsumed index for the next forward.
The private multiplier mask becomes **graph-owned** state through the
unchanged `graph_resources` contract — the third member of the family
beside MaxPool2d's winners and cross-entropy's saved probabilities —
released exactly once with the graph history, retained under
`retain_graph=True`, and closed immediately by a no-grad forward. The
backward is `upstream * mask` through the existing native `multiply`, so
**no Dropout backward kernel exists**; it reads neither the input nor the
generator, records no expected parameter version, and consumes no call,
which is why mutating the input or reseeding the generator afterwards
leaves an existing graph's gradient exactly as it was.

Milestone **G4 is complete**: `NativeDropout`, the public module over
that operation, and its export — one file, one export, and one name
(`"NativeDropout"`) appended to `NATIVE_MODULES`.
`NativeDropout(p=0.5, seed=None, generator=None)` validates `p` through
the *same* shared normalizer the Core and the operation use, treats
`seed` and `generator` as **mutually exclusive** (supplying both raises
rather than silently ignoring one), and either creates and owns a
generator or registers the **exact** object supplied, so two layers can
share one interleaved stream. That generator is first-class registered
state — in `generators()`, `named_generators()`, and
`generator_state_dict()`, and deliberately absent from `state_dict()`,
which stays contractually tensor-valued — and a state load replaces it in
place, preserving identity and any sharing. Training delegates to the G3
operation, so a successful forward consumes exactly one call and a failed
one none; evaluation returns the **input object itself**, consuming
nothing and allocating nothing, so any number of eval forwards leaves no
gap in the stream; and `p == 0` is identity in both modes. The module
owns no native storage.

**G5–G10 have not started**, and the gap is persistence: the checkpoint
format is
still version 1 and does not persist generator state, so saving a model
containing a `NativeDropout` preserves its parameters and buffers and
**silently omits the random stream** — exact stochastic resume does not
exist yet, and a load fabricates nothing. That, with the unrun closure
matrix, is why `dropout` stays
listed unsupported through G9, leaving that list only at G10 after the
closure matrix.
Beyond Phase G
(**not started**): more activations/math, data
loaders, a CPU optimization phase, then the CUDA
runtime, dtype/AMP work, and Transformer/text and distributed
experiments. See [roadmap.md](roadmap.md) and
[release_history.md](release_history.md) for the full arc.
