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
no numerical capability. **Phase G (native RNG and Dropout) is complete
and is the latest *completed* phase: milestones G0 through G10 have all
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
over one registered generator it owns or shares, and G5 native checkpoint
**format version 2** — persisted generator state with its shared-generator
alias topology, strict topology validation, version-1 compatibility, and
the whole-checkpoint load transaction. G6 hardened the RNG, graph,
ownership, and checkpoint contracts, G7 **demonstrated** the end-to-end
exact stochastic training resume
(`examples/native_dropout_training.py`), G8 characterized it honestly
(`benchmarks/benchmark_native_dropout.py`), G9 added the cross-cutting
integration suite (`tests/test_native_phase_g.py`), and G10 closed the
phase — the Windows Release/Debug and Clang ASan/UBSan/LeakSanitizer
validation matrix — after which `dropout` **left** `UNSUPPORTED`. Native
Dropout is therefore supported **on the experimental native float64 CPU
line** and nowhere else; reproducibility stays exact only for the state
actually captured, and float32, CUDA, and AMP remain unsupported. The two
engines never mix: explicit entry via
`NativeTensor.from_array`, explicit exit via `to_numpy()`, no implicit
dispatch. The exact per-operation status lives in the
[native support matrix](native_support_matrix.md).

## Testing and reliability (both lines)

4,920 pytest tests cover every feature of both lines: known-value
checks against hand-computed math, finite-difference gradient
verification (stable and native), exact resume-equivalence tests for
checkpointing, NumPy-tripwire tests proving the native paths never
fall back, cross-cutting Phase C, **Phase D, Phase E, Phase F, and
Phase G** integration
guardrails
(shared/frozen/late-active parameters, failure recovery at every
boundary, graph-version interactions, saved-winner,
saved-probability, normalization-snapshot, and Dropout-mask lifetime,
atomic running-statistics transactions, generator reservation and
call-consumption discipline, whole-checkpoint transaction rollback, and
lifetime discipline), and guardrail tests keeping docs, examples, and
the public API from drifting. The native C++ kernels additionally have
dependency-free CTest binaries, validated under ASan/UBSan. CI builds the
backend from source and runs everything.

With the Release backend active the full suite currently reports
**4,920 passed, 0 skipped**. There are **no expected skips**: the
missing-backend contract — that the five Phase-E operations raise
`ImportError` with build instructions and never fall back to NumPy — used
to be skipped on every machine that could actually run the native suite,
and is now *executed* instead, simulated in fresh child processes that
repoint their own module-private library path at a nonexistent file
(`tests/test_native_backend_unavailable.py`). The real compiled library
and the parent pytest process are never touched, and each case
fingerprints the library on both sides of its subprocess to prove it.
Historical per-phase closure totals stay in
[release_history.md](release_history.md) rather than being restated here.

## Current limitations

Not production-ready and not a PyTorch replacement. The stable
framework is NumPy on CPU; `Conv2d` and `MaxPool2d` use deliberately
naive loops, and so do their native counterparts (direct nested loops —
no im2col, BLAS, threading, or SIMD). The native line is float64/cpu
only — no CUDA backend, no float32/float16 or dtype promotion and
casting, no AMP, no data loaders or native integer tensors, and no
dispatch into `tensorforge.Tensor`. Its Dropout is one deterministic
stream behind an explicit `NativeGenerator`, not a generic random-number
API, and there is no `Dropout2d`/`Dropout3d`. Native checkpoints persist
parameters, persistent buffers, optimizer state, and generator state, and
deliberately **not** data-loader position, shuffle state, epoch or
scheduler state, Python's `random`, or NumPy's global RNG — so
reproducibility is exact for the state actually captured, and
full-program determinism is not claimed. Ordinary concurrent training is
not claimed thread-safe either: the serializability guarantee covers the
participating state transactions only. Benchmarks are hardware-specific
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

**Phase F — Native Normalization and Stateful Buffers — is complete
(F0–F9).** Its architecture contract is locked in
[native_normalization_design.md](native_normalization_design.md)
(milestone **F0** — design and repository reconciliation, no numerical
behavior). **F1** shipped the private atomic native-buffer state
transaction, refactored `load_state_dict` onto it, and corrected
`STATE_SUPPORT` to report the `persistent_buffers` capability — state
management and capability reporting only, no normalization mathematics.
**F2** shipped `NativeLayerNorm`, the first native normalization module:
stateless (no buffers, identical in train and eval), differentiable
through the mean and the population variance, and **composed entirely
from existing native operations** (`sqrt(var + eps)`, no Bessel
correction) with no kernel, ABI symbol, Core method, custom backward, or
`NativeTensor` normalization operation; `"layernorm"` left `UNSUPPORTED`.
**F3** shipped `NativeBatchNorm1d`, the **first stateful native numerical
module**: `(N, C)` batch normalization with differentiable training
statistics (gradients through the batch mean *and* the population
variance), persistent native `running_mean`/`running_var` buffers
advanced graph-free by one **atomic two-buffer transaction** (both
identities preserved, no parameter version moved), and evaluation from
**graph-safe immutable snapshots** of those buffers, so a later training
step, or a buffer-only state or checkpoint load, cannot change an earlier
eval graph's gradient — while a full checkpoint load also replaces
`gamma`/`beta` and therefore stales it through the unchanged
parameter-version rule. **F4** added `NativeBatchNorm2d` — NCHW
`(N, C, H, W)` normalization reducing over N, H, and W over the **same**
shared private implementation, which it extends with only its rank,
reduction axes, `(1, C, 1, 1)` broadcast layout, and the channels-last
permutation its rank-1 affine parameters need — and with both shapes live
`"batchnorm"` left `UNSUPPORTED` too. Neither module added a kernel, ABI
symbol, Core method, custom backward, or `NativeTensor.batch_norm`
operation, and the checkpoint format stayed at version 1 for the whole
phase. **F5** then proved the state, checkpoint, ownership, and
graph-safety contracts by exhaustive test
(`tests/test_native_normalization_state.py`), **F6** shipped a
deterministic normalized training example whose interrupted run resumes
exactly (`examples/native_normalization_training.py`), **F7** the honest
nine-case benchmark characterization
(`benchmarks/benchmark_native_normalization.py` — correctness gated
before any timing, no result file, no speed assertion, no committed
timing number), **F8** the cross-cutting integration and semantic
guardrails (`tests/test_native_phase_f.py`), and **F9** the phase closure
under fresh Windows Release **and** Debug builds and a Clang
ASan/UBSan/LeakSanitizer matrix that returned native live storage exactly
to baseline with no suppression file — F5 through F9 adding **no**
numerical capability between them. Closing Phase F closed that phase
only; the native line remains experimental, float64/CPU, and explicitly
scoped.

**Phase G — Native RNG and Dropout — then opened, and it is now
complete.** Milestone **G0 is complete**: the architecture contract in
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

Milestone **G5 is complete**: native checkpoint **format version 2** —
the format *name* is unchanged — with one new manifest field,
`"generators"`. It is `null` for a model that registers none, or a
`keys`/`entries`/`aliases` object holding every canonical generator's
`algorithm`, `algorithm_version`, `seed`, and `calls` (the last two as
canonical decimal strings, since a `uint64` above `2**53` cannot survive
a JSON double) **plus the complete alias topology** — every registered
path mapped to its canonical generator, so *which layers share a stream*
is restored, not just the numbers. Generator state adds no array to the
archive; sharing is identity, never state equality. A load restores each
generator **in place**, preserving object identity and every sharing
relationship, and validates the topology strictly in both directions
against a real `named_generators()` traversal — every mismatch fails in
prevalidation, with nothing touched. A version-1 archive still loads into
a generator-free model and is **rejected**, naming them, for a model that
has generators: no seed and no counter is ever fabricated. A save or a
load is refused while any target generator has a call reservation in
flight. And the whole load is **one transaction** — model, buffers,
optimizer, and generators commit under a single rollback guard, so any
synchronous failure (a deliverable `KeyboardInterrupt` included) restores
all four together, preserving every identity and moving no parameter
version. It is **serializable** too: every participating state
replacement — the checkpoint load, `load_state_dict`,
`load_generator_state_dict`, and both optimizers' state loads — plus the
save snapshot runs under one private shared `RLock`, with generator locks
taken under it in the global `id()` order, so two concurrent loads leave
one archive's state followed by the other's rather than a mixture.
Ordinary training mutation does not take that guard, so thread-safe
concurrent training snapshots are not claimed. The registry footprint is
one reporting-only name, `"checkpoint_generator_state"` in
`STATE_SUPPORT`.

Milestone **G6 is complete**: the hardening milestone, which added **no
capability**. `tests/test_native_phase_g_hardening.py` executes the
design's ownership and failure matrices as adversarial tests — the
reservation transition matrix with five invariants per rejected
transition, the exact `uint64` boundary, forced concurrent interleavings
(no sleeps, bounded joins), the Core's structural key properties, every
pre-commit and post-commit failure position of the call transaction across
four exception classes, all four graph-owned saved-resource families in
one graph, a 76-case checkpoint corruption matrix, whole-transaction
rollback at every commit position, save-seam destination atomicity, and
repeated lifecycle loops against a real native live-storage baseline. One
runtime defect was found and fixed: a failed cleanup step could leave the
Dropout operation's `__context__` chain **cyclic**, which hangs any
ordinary chain-walking reader. Nothing else changed — no C++, ABI, ctypes,
Core method, operation, module, export, schema field, or registry value.

Milestone **G7 is complete** — the end-to-end exact stochastic resume,
and **no new capability**. `examples/native_dropout_training.py` trains
`NativeLinear(4, 8)` -> `NativeBatchNorm1d(8)` -> `NativeReLU` ->
`NativeDropout(p=0.5, seed=20240707)` -> `NativeLayerNorm(8)` ->
`NativeLinear(8, 3)` over raw logits with `NativeCrossEntropyLoss` and
`NativeAdam` on a fixed twelve-sample three-class task computed from an
explicit formula, in three fixed batches on a schedule that is a **pure
function of the training step**. It carries all four TensorForge-owned
state families at once — parameters, persistent BatchNorm running
buffers, a registered `NativeGenerator`, and NativeAdam moments with
per-parameter step counters — so an incomplete restore diverges
immediately. Two uninterrupted runs are bit-identical; an interrupted run
checkpointed after 7 **completed** steps (deliberately mid-cycle in the
batch schedule), whose model, optimizer, and generator are **released
before the resume begins**, reloads into a completely fresh set built
with a *different* Dropout seed and reproduces the uninterrupted run by
**exact equality**: the whole loss sequence, every parameter, both
running statistics, every optimizer moment and step counter, the
generator's algorithm/version/seed/calls, the final training logits, and
the final evaluation output. Two negative controls make that meaningful —
restoring all four families but restarting the batch schedule at 0
**diverges**, and restoring everything but re-seeding the generator
**diverges**. Evaluation is proved state-neutral (repeated eval passes
leave `calls` bit-identical, produce identical outputs, restore the
caller's mode, and leave a probed run's loss sequence equal to an
unprobed one's), and a separate throwaway reload matches the restored
module's next Dropout output against `NativeTensorCore.dropout_forward`
at the exact restored `(seed, call_index)`, advancing `calls` by exactly
one. **External loop progress is carried explicitly**, as validated JSON
metadata (`{"training_step": ..., "next_batch_index": ...}`), because
checkpoint v2 captures TensorForge-owned state and **not** data-loader
position, batch order, shuffle state, epoch counters, scheduler state,
Python's `random`, or NumPy's global RNG — a missing or inconsistent
field raises rather than silently restarting from step 0.
Reproducibility is exact **for the state actually captured**;
full-program determinism is not claimed. The whole milestone is one
example, one test module, and documentation: **no** C++, C ABI symbol,
ctypes declaration, Core method, autograd operation, module, export,
schema field, checkpoint version, benchmark, or registry value changed.

Milestone **G8 is complete** — the honest benchmark characterization,
and **no new capability**. `benchmarks/benchmark_native_dropout.py`
measures thirty-five cases in eight families — the stateless Core against
an **exact bit-for-bit** vectorized NumPy implementation of the same
locked derivation, scalar-to-large size scaling, four physical layouts
over one logical shape, a five-value probability sweep at three layers,
the no-grad / differentiable / backward-only / forward-plus-backward
operation layers, the module's training and identity paths, and one
complete Dropout training step. Every case is correctness-gated **before**
timing, the committed known-answer vectors pin the harness's reference
and then the native kernel, each stochastic case's generator consumption
is verified exactly, and an untimed lifecycle pass returns native live
storage to its baseline. The `NativeTensor` and `NativeDropout` cases are
`native_only` and publish no ratio: no NumPy expression owns a generator
transaction, native ownership, and an autograd graph. There is **no speed
assertion, no committed timing number, and no CI timing threshold**, no
result file unless `--json-out` names one, and **nothing was optimized to
improve a number** — no runtime file changed.

Milestone **G9 is complete** — the cross-cutting Phase-G integration
suite, and **no new capability**. `tests/test_native_phase_g.py` trains
one test-only model carrying every registered state family at once
(`NativeConv2d` -> `NativeBatchNorm2d` -> `NativeReLU` ->
`NativeMaxPool2d` -> `NativeDropout` -> `NativeFlatten` ->
`NativeLinear` -> `NativeBatchNorm1d` -> `NativeReLU` ->
`NativeLayerNorm` -> `NativeDropout` -> `NativeLinear`, raw logits into
`NativeCrossEntropyLoss`, two Dropout layers over **one** shared
generator) and proves the interactions: four saved-resource families in
one graph released exactly once, exact version-2 resume into a fresh
model/optimizer/generator set, the generator-topology matrix with every
mismatch rejected before any state changes, evaluation consuming no call,
`p == 0`, non-contiguous NCHW and strided views, whole-state rollback at
every commit position, four deterministic concurrency cases, a Phase A–F
regression matrix, and live storage returning exactly to baseline across
success and failure cycles. No runtime file changed and no defect was
found.

**G10 is complete, and with it Phase G.** The closure matrix ran with
observed results: fresh Windows Release and Debug builds, each **11/11
CTests** with zero project warnings and the active runtime proved to stay
the Release DLL; a fresh Clang 18.1.3 ASan+UBSan build in WSL2 with
instrumentation proved rather than assumed (22 `__asan*` and 14
`__ubsan*` dynamic symbols beside 51 exported `tf_*` symbols, and a
library that refuses to load without the runtime), **11/11 sanitized
CTests** with leak detection on, **3,166 sanitized Python tests** across
43 suites, the G7 example reproducing its exact resume, and the G8
benchmark passing every correctness gate — all with zero ASan and zero
UBSan diagnostics; and a LeakSanitizer lifecycle returning native live
storage exactly to baseline (0 → 0), whose remaining process-exit
allocations name **no** TensorForge frame, with **no suppression file
added**. Only then did `dropout` leave `UNSUPPORTED`, which now reads
`("float32", "cuda", "amp")`.

The claim is narrow on purpose: **native Dropout is supported in the
experimental native float64 CPU backend**. The stable framework keeps its
own separate `Dropout`, and float32, CUDA, and AMP remain unsupported.
There is no generic random-number API and no `Dropout2d`/`Dropout3d`.
Reproducibility is exact only for the state actually captured (no Python
`random`, NumPy global RNG, data-loader position, or scheduler state), and
ordinary concurrent training is not claimed thread-safe: the
serializability guarantee covers the participating state transactions
only.
**Phase H — native CPU performance and runtime efficiency — is the
current phase; it has begun, and milestones H0, H1, H2, H3, and H4 are
complete.** H0 is an
architecture, profiling, and baseline milestone and **nothing was made
faster**: it shipped the contract in
[native_cpu_performance_design.md](native_cpu_performance_design.md), the
unified measurement harness
`benchmarks/benchmark_native_cpu_performance.py`, that harness's
behavioral contract tests, and documentation reconciliation — H0 added
no numerical capability, kernel, C ABI symbol, ctypes declaration, Core
method, operation, module, export, capability-registry value, dtype, or
device, and the native checkpoint format stays version 2 with versions 1
and 2 supported.

The harness separates the layers a caller actually pays for — NumPy, the
stable line, the raw-buffer kernels, `NativeTensorCore`, `NativeTensor`
with and without a graph, backward, an optimizer step, and a whole
training step — across twelve workload families, gates correctness
**before** timing everywhere, publishes no ratio where no honest
equivalent exists, and writes no result file. The evidence it produced is
ranked honestly and is deliberately surprising in places: the largest
measured factors are an allocator behavior and a memory access pattern
rather than raw arithmetic, the Python-side per-call metadata path costs
several times the ctypes boundary it wraps, and the `NativeTensor`
wrapper and its autograd graph node are measurably **not** a bottleneck.
**Milestone H1 — the output-allocation contract — has since shipped**,
the first Phase-H change to production code. **Milestone H1 — the output-allocation contract — has now shipped.** It removed the redundant zero-fill from output storage that a kernel provably overwrites in full, behind one new C ABI symbol (`tf_storage_create_uninitialized`) that matches the zero-initializing default in size validation, allocation-failure handling, error state, ownership, destruction, and live-storage accounting, and differs only in the buffer's initial contents. The zero-initializing path remains the default; there is **no** global allocator policy, environment variable, heuristic, memory pool, scratch arena, or public empty-tensor API, and every enabled call site opts in explicitly against a per-kernel audit table. `sum`/`mean` and `narrow_backward` are explicitly **rejected** and keep a zeroed destination: the first accumulates into its output, the second writes only the narrowed region and the untouched zeros *are* the gradient. Completeness is proved by deterministic **poison** tests that are injected **exclusively by test infrastructure, around the allocator**: the suite wraps the private uninitialized allocation helper, lets the real constructor allocate, fills the returned storage with a quiet NaN or a large finite pattern through the ordinary fill primitive, and hands that same storage to the real operation — so the pattern is in place after the real allocation and before the real kernel runs. **No poison-control mechanism exists in the production runtime**: no exported hook, no thread-local flag, no environment variable, no global mode. ASan and UBSan stay separate from the initialization proof — they do not detect uninitialized-value reads — and MemorySanitizer is not available here, so neither is claimed; negative controls prove the detector can actually fail. H1 is bit-identical: every enabled operation and a full training run are compared element-wise against the zero-initializing allocator. No capability, dtype, device, registry value, checkpoint field, or checkpoint version changed, and `tf_storage_create_uninitialized` is the **only** export it added, taking the library from the pre-H1 baseline of 51 exported `tf_*` symbols to **52**.

The measured result is reported honestly rather than as a headline: isolated, the zero-fill is enormous and scales with the buffer (about 52x at 2 MB, 119x at 8 MB, 552x at 32 MB, and *negative* below roughly 16,000 elements, where it sits inside the noise). End to end it is much smaller and often inconclusive — clearly real for large memory-bound elementwise work (about 1.5-1.8x on an 8 MB output), small and variable for normalization and Adam, and with no measurable effect on Conv2d, the MLP step, or matmul, whose arithmetic dwarfs its allocation. Those inconclusive and negative rows are published as such.

**Milestone H2 — native matmul memory access — has since shipped**, the
first Phase-H milestone to change how a numerical kernel executes. It
swapped the production matmul's loop order from `i`-`j`-`k` to
`i`-`k`-`j` over four destination rows at a time, so the innermost loop
walks a *row* of the right operand and a row of the output sequentially
instead of walking a column. **Cache blocking, which the milestone title
anticipated, was measured against 22 blocked variants and rejected** — an
unblocked full-width row sweep was faster at every non-trivial size — so
H2 shipped the simpler superior design and recorded the negative blocking
result. The pre-H2 triple loop is **retained verbatim as the shipped
generic reference path**, still reachable through ordinary production
dispatch, and the choice between the two is made inside the kernel from
the stride metadata it already receives: a right operand whose column
stride is 1, with a non-empty inner dimension and at least 8 result
columns, takes the row sweep; a transposed right operand, a narrow
result, or an empty inner dimension takes the generic path — which is the
loop order that case already suits, so the fallback is a design choice
rather than a gap. Dispatch is metadata-driven, deterministic, total,
side-effect free, and independent of pointer values, alignment, timing,
environment variables, and CPU-feature probes; a failed precondition is
never an error. **H2 added no exported C ABI symbol** — the library still
exports exactly 52 `tf_*` symbols — and there is no kernel selector,
block-size setter, benchmark hook, dispatch tracer, or public dispatch
control of any kind; the two kernels and the predicate are
hidden-visibility C++ that the native test reaches only by compiling the
source in. The numerical agreement between the two paths is stated in **four
parts** rather than as a blanket claim, because a blanket claim would be
an overclaim. (1) **Accumulation order is preserved exactly** — same
starting zero, same products, same ascending `k`. (2) **Every non-NaN
result is bit-identical**, asserted as raw IEEE-754 bit patterns rather
than tolerances across shapes, layouts, signed zeros, infinities,
denormals, the largest finite magnitudes, both gradients, `NativeLinear`,
both optimizers, deterministic training, and exact checkpoint resume —
which covers every committed loss trajectory and every resume proof in
the project, since all of them run on finite data. (3) **NaN-class
equivalence holds**: NaNs appear in exactly the same positions on both
paths and are always quiet, and neither path produces a signaling NaN.
(4) **NaN payload bits are deliberately outside TensorForge's numerical
contract** and may differ between the paths. Ten source-level
formulations were measured while trying to close (4) — compound versus
explicit assignment, named locals, `__restrict`, disabled inner-loop
vectorization, and two stack-accumulator tile shapes — and all ten
`i`-`k`-`j` spellings behaved identically; the only structure that
reproduces the reference's payloads is the `i`-`j`-`k` order H2 exists to
replace, so payload parity is unavailable short of abandoning the
optimization. Measured: MSVC Release differs on 162 of 208 results in a
NaN-saturated matrix, MSVC Debug and Clang on none. H1's uninitialized-output
contract still holds on both paths, for a different reason on each — the
generic path never reads the destination, and the row sweep's `k == 0`
pass assigns every element of every row before anything accumulates into
it — proved by poison tests over both paths with both patterns plus a
negative control. The measured result is reported honestly: roughly
4.1-4.7x at 384 cubed, 4.2-4.5x at 128 cubed, about 4-6.8x on
`NativeLinear` forward, 1.7-2.5x on its backward (only one of its two
matmuls qualifies, by design), 2.0-2.4x on a 128x256 MLP Adam step, and
**no measurable effect below roughly 32 cubed or on a small MLP step**,
where a fixed ~10 microsecond per-call Python cost dominates and control
cases whose compiled code did not change at all vary by 0.50-1.44x. No
capability, dtype, device, registry value, checkpoint field, or
checkpoint version moved.

**Milestone H3 — native metadata and dispatch efficiency — has since
shipped**, and unlike H1 and H2 it is **Python-only**: no C++, no C ABI
symbol, no ctypes declaration, and no kernel changed, so the library
still exports exactly **52** `tf_*` symbols. H3 attacked the fixed
per-operation cost B3 measured at 18.6-22.6 microseconds, of which only
about 1.9 was the ctypes boundary and the rest was Python-side shape and
stride work. The measured cause was redundant *re-validation*: one
`shape_info` call ran `_as_int_tuple` **four** times over a tuple that
was fully validated after the first pass, and computed the row-major
strides **twice**, while `NativeTensorCore.zeros` validated the caller's
shape a second complete time by calling `numel(shape)` and then
constructing a view from the same raw shape. Instrumented call counts put
that at **815** `_as_int_tuple` calls per MLP training step and 604 per
`NativeAdam` step. H3 introduced **one normalization boundary** — the
private `_normalized_layout`, performing exactly the checks `shape_info`
always performed, in the same order and with the same messages, and
normalizing the shape once — with the derived quantities computed by
private `_checked` primitives that validate nothing *because there is
nothing left to validate*. Each public helper (`row_major_strides`,
`numel`, `reduce_shape`, `broadcast_shapes`) is now its own validation
followed by the matching primitive, so the two can never disagree.
`NativeTensorView` gained a private `_from_validated` constructor that
skips **only** that normalization; both constructors funnel through one
shared `_bind` that still performs the storage open check and the full
reachable-offset bounds check, and the element count and contiguity flag
are **derived inside** the private constructor rather than passed to it,
so no caller can supply an inconsistent pair — which is why H3 has a
separate private constructor rather than the misusable `validated=True`
flag. Views also memoize their `int64` shape/stride arrays for the
strided C ABI, **lazily** and **read-only**. That memoization cannot go
stale: a view's layout is assigned exactly once, in `_bind`, and every
layout-changing operation (`reshape`, `transpose`, `T`, `narrow`)
returns a *new* view, so no invalidation is ever required and none
exists. Nothing global was introduced — no shape cache, no stride
interning, no weak-reference machinery, no thread-local state — and
**no validation was removed**: every rejection still happens, with the
same exception type, the same message, and the same shape-then-strides-
then-offset ordering. Measured: `shape_info` 2.6-4.5x faster, view
construction 3.2x, `_as_int_tuple` calls per MLP step **815 -> 149** and
per CNN step **815 -> 150**; end to end, a one-element allocation 2.1x, a
`reshape` 3.1x, a view chain 2.4x, a small `add` 1.56x, `NativeAdam` on a
small MLP 1.42x, a **whole MLP training step 1.43x**, a **CNN training
step 1.29x**, and a **normalized training step 1.51x**, which cut the
Adam step's gap against the stable line from 39.8x to 31.9x. Reported
just as honestly: **large kernel-bound work shows no measurable change in
either direction** — 384-cubed, 512-cubed and 128-cubed matmul, 256-
squared elementwise, and 128-squared reduction all sit inside their own
run-to-run spread, so H2's large-matmul result is intact. The layout-
array cache is the weakest of the three changes and was kept on measured
merit, not principle: isolated, it saves 0.6-1.5 microseconds per
*strided* small operation and nothing at all on large ones or on a
contiguous training step, and even a deliberately cold-cache measurement
is no slower than pre-H3. One methodology finding is published rather
than buried: at the harness's default 11 repetitions a case appeared to
regress 35%, and at 201 repetitions the same case measured 1.19x
*faster* — so no default-repetition figure is quoted as H3 evidence.
Object footprint is unchanged for a cold view (byte-identical) and
+328 bytes for one that actually takes a strided path; in a full MLP step
only **5 of 134** views ever populate it, 1,560 bytes in total. All
instrumentation was test-local or benchmark-local monkeypatching and
subprocess A/B runs against a retained pre-H3 copy of the package — **no
production counter, environment-variable profiler, or installed tracing
mode exists**, and H3 added no public API of any kind: no cache control,
statistic, reset, profiling counter, or dispatch selector. No capability,
dtype, device, registry value, checkpoint field, or checkpoint version
moved.

**Milestone H4 — native optimizer step efficiency — has since shipped**,
also **Python-only**: no C++, no C ABI symbol, no ctypes declaration, and
no kernel changed, so the export count is still **52**. It is the first
Phase-H milestone whose subject is a *training-stack* component rather
than the tensor runtime. B4's counts were re-instrumented on the current
code and confirmed exactly — **27 native allocations per parameter per
`NativeAdam.step()`, ten of them one-element**: eight broadcast scalar
coefficients (`beta1`, `1 - beta1`, `beta2`, `1 - beta2`, both
bias-correction terms, `eps`, and `lr` — the design said six, and `eps`
and `lr` were the two it missed) plus two `reciprocal` outputs taken on
one-element tensors; `NativeSGD` allocates five per parameter. H4 shipped
three changes: the **scalar coefficients are built once per step rather
than once per parameter**, in a private per-step holder keyed by
`(dtype, device)` with the bias corrections cached per step *counter*,
which allocates nothing until the first entry asks for one, is released
before the commit begins, and is never stored on the optimizer — so no
scalar survives a step, enters `state_dict()`, reaches a checkpoint, or
must be released by `close()` (NativeSGD does the same for its single
`lr` scalar, the only change its evidence supported); the
**bias-correction reciprocal is evaluated in Python**, an *exact
substitution* rather than a reassociation, since the kernel literally is
`1.0 / x` on the same IEEE-754 binary64 value and IEEE division is
correctly rounded — proved over 20,000+ values on raw `uint64` bit
patterns; and **temporaries are released at their last use** instead of
all together at the end of the staged expression. Everything is
**bit-identical** to the pre-H4 composition, which is *retained in the
test suite* and executed natively as the reference across 60
shape/step/hyperparameter combinations, a six-step multi-shape run, and
four SGD learning rates spanning `1e-9` to `1e12`; a separate test pins
the exact operation sequence a staged entry issues. The two-phase
contract is untouched: validation is still four complete passes in the
same order with nothing moved behind a mutation, stage mutates no
parameter, moment, counter, version, or gradient, the commit is still one
`copy_value_` and exactly one version increment per updated parameter,
and the per-entry commit boundary is *tested* rather than assumed
infallible. Measured by alternating pre/post subprocess rounds (366
samples per case): `NativeAdam.step()` **1.58×** at (128, 128), **1.54×**
at (256, 256), **1.48×** on a four-parameter MLP with a 256² weight,
1.21× on a small MLP, 1.15× on a first step; a large MLP training step
1.23×, a small one 1.15×, a normalized step 1.13×, a CNN step 1.09×; and
against `tensorforge.optim.Adam` **23.8× → 19.7×**. Reported just as
honestly: **a (512, 512) parameter is neutral** (1.02×, memory-bound),
the **Dropout training step is neutral** (0.99×), and **NativeSGD is
neutral-to-slightly-positive** — with the machine's control-case noise
band stated at **0.84×–1.26×** so no single reading inside it is
mistaken for a result. **Peak live transient bytes during an Adam step
fell 2.6–3.0×** and per-parameter allocations 27 → **17**, so the time
was not bought with memory. Six alternatives were measured and
**rejected** with reasons recorded, among them scalar materialization
(faster below ~32 K elements, slower above), stride-0 same-shape views
(more NumPy layout arrays per call, not fewer), and a persistent
per-optimizer scalar cache (the hidden scratch tensor the design
forbids). H4 added **no public API of any kind**, and no capability,
dtype, device, registry value, checkpoint field, or checkpoint version
moved.

**Milestone H5 — native copy and mutation-transfer efficiency — has
since shipped**, and it is the first Phase-H milestone since H2 to change
C++ — though **not the ABI**: the library still exports exactly **52**
`tf_*` symbols. H5 replaced the native line's **value-transfer
primitive**. `_native_copy` was `zeros(shape) + core` — two allocations,
a full zero-fill pass, and a full elementwise-addition pass — and is now
the E3.1 native identity gather, `NativeTensorCore.contiguous_copy()`:
one uninitialized allocation (H1) and one pass. The composition predates
that gather and was simply never migrated to it. A complete inventory
found **ten** call sites of the one helper — `NativeParameter.copy_value_`
staging, both `state_dict()` snapshot paths, both `load_state_dict()`
staging paths, both BatchNorm running-statistic commits, and the
reshape/transpose/unbroadcast gradient materializations — and every one
of them is a **pure value transfer**: an independent contiguous
materialization of some tensor's current value, wanting no arithmetic.
All ten were enabled. `_broadcast_back`'s `zeros(x_shape) + upstream` was
**rejected** because it is not a copy at all but a genuine broadcast
expansion, which `contiguous_copy` cannot express; `sum`/`mean` and
`narrow_backward` keep their zeroed destinations for H1's unchanged
reasons.

The semantic question H4 refused to decide in passing was decided here,
by measurement over a fixed 18-pattern IEEE-754 sweep. **Exactly three**
patterns behaved differently under the two spellings: the addition
normalized `-0.0` to `+0.0` and quieted both signs of signaling NaN,
while the gather preserves all three. Everything else — `±0`, `±inf`,
quiet NaNs of either sign and **any payload**, denormals, the smallest
normal, the largest finite magnitudes — was already identical, so no NaN
payload differed at all (with one NaN operand and one zero, x86-64's
`ADDSD` returns that operand's NaN). **H2's matmul NaN-payload carve-out
does not generalize to copies**: it exists because two NaN operands meet
in an accumulation, and a copy performs no arithmetic. The pre-H5
behavior was **accidental and inconsistent**, not contracted — three
other value-copy paths (`NativeParameter(source)` construction,
`detach()`, and the `to_numpy()`/`from_array` boundary) always used the
gather and always preserved `-0.0`, while `copy_value_` documented the
same thing and did not deliver it. H5 states the narrowest coherent rule:
**a value transfer reproduces its source's bits exactly; an operation —
`zeros + x` included — follows IEEE arithmetic.** No operation's
arithmetic changed anywhere, and the whole pre-H5 suite passes unchanged
apart from the guardrails that pinned the old composition by name.

Swapping the composition alone would have **regressed** the common case,
so H5's one C++ change is a second **traversal** inside the unchanged
`tf_core_contiguous_copy` export. `zeros.add(core)` on a contiguous
source takes a flat pointer loop, while the gather always walked the
generic odometer — the only unary export without the contiguous fast path
every other one has — and a naive swap measured **0.48x** at 16,384
elements. The export now picks its traversal from the layout metadata it
already receives, exactly as H2's matmul picks its kernel:
`tf::copy_prefers_contiguous` is hidden-visibility C++ in a new internal
header, total, pure, allocation-free, and a function of metadata alone —
never of a pointer value, an alignment, a clock, an environment variable,
or a CPU-feature probe — testing exact equality against the row-major
strides implied by the shape, which is the same definition
`NativeTensorView` uses, so the two layers agree by construction. A false
answer falls back to the retained odometer and is never an error. **No
numerical carve-out is needed, and that is the difference from H2**: both
traversals evaluate `dst[out] = src[pos]` over the same logical elements
in the same destination order and differ only in how `pos` is computed,
so they are bit-identical *by construction* — proved directly at the C++
level by a new dependency-free CTest, taking the suite from 13 to 14.
There is no copy-mode selector, overlap-mode flag, traversal tracer, or
public dispatch control of any kind.

Nothing became less safe, because nothing became in-place: every call
site still **stages** an independent materialization and only then adopts
it. The overlapping arrangements the runtime can construct —
`copy_value_(self)`, a source that is a view of the destination's own
storage, a square parameter's own transpose, sibling views, duplicate
parameters across optimizers — are each tested and each correct, and no
`memcpy` is used anywhere. Parameter identity, storage replacement,
gradient retention by identity and value, the one version increment per
commit, the F1 state transaction, checkpoint atomicity, and exact resume
are all exactly what they were; gradient *accumulation* still adds rather
than assigns. H1's full-write contract is proved on both traversals by
poison injected purely by test infrastructure around the allocator, with
a negative control showing the detector can fail.

Measured by alternating pre/post **subprocess** rounds against a retained
pre-H5 composition, with a control band of **0.96x-1.05x**, and — for
the C++ half — by building a **pre-H5 library** and driving both through
identical `ctypes` calls on identical data, outputs proved bit-identical
before either was timed. The traversal alone: **2.5x-5.5x** on contiguous
sources from 16 K elements up (5.53x at 512 squared, 5.53x on 4-D NCHW,
5.46x on an offset view), 1.29-1.62x on small ones, and **0.94x-1.02x on
transposed and last-axis-narrowed sources**, which take the *unchanged*
odometer and are the design's own control. End to end: `copy_value_`
**2.14x** at (512, 512) and 1.26x at (128, 128), optimizer `state_dict()`
2.40x and `load_state_dict()` 1.69x, module `load_state_dict()` 1.37x,
`NativeSGD.step()` 1.15-1.31x. Reported just as honestly:
**`NativeAdam.step()` is neutral** (0.98x-1.06x — the commit copy is one
of about seventeen buffers and the arithmetic dominates), **every
training step is neutral** (0.95x-1.07x), the **BatchNorm running update
is neutral** (0.98x), and **copies below ~16 K elements are neutral**
(0.93x-1.01x), because a `contiguous_copy` call converts two `int64`
layout arrays at the ctypes boundary at **~1.1 us each** — a cost
measured, attributed, and left to a later dispatch milestone rather than
paid for by weakening H3's validation. Two methodology findings are
published rather than buried: at 7 alternating rounds the small copies
read 0.78x-0.94x and looked like a regression, while at 21 rounds the
same cases read 0.93x-1.01x (the same lesson H3 recorded); and the
largest single ratio, **7.9x-10.5x at 512-640 KB**, is a **512 KB
allocator cliff on this machine**, not a loop-speed result — the pre-H5
composition makes two large allocations and zero-fills one, so it crosses
that threshold at half the size and pays it twice. The durable statements
are ~2.1x at 1-2 MB and neutrality below 384 KB.

Memory moved with time, never against it: **no measured peak rose**, and
the pure-transfer paths halved. `copy_value_` at (512, 512) went 2
allocations to **1** and 4,194,304 to **2,097,152** peak bytes; module
`state_dict()` and `load_state_dict()` 4 to **2** allocations with peak
bytes halved; optimizer `state_dict()` 16 to **8**; `NativeSGD.step()`
5 to **4** with peak 393,216 to **262,152**; and `NativeAdam.step()` went
**17 to 16** allocations per parameter (H4 took it 27 to 17), removing a
whole-parameter zero-fill pass from every committed update. The harness
gained two cases, 26 to **28**: `row_major_materialization`, the
flat-traversal twin of the existing transposed-source case, so the two
traversals are separated rather than averaged; and
`parameter_value_commit`, `native_only` with **no ratio**, because the
stable line mutates a `Parameter` by rebinding `.data`, which is a
different operation. The ladder was **reordered** here — reduction
execution, drafted as H5, moved to H6 — and no public API, capability,
dtype, device, registry value, checkpoint field, or checkpoint version
moved.

The proposed H6–H11 ladder is explicitly **conditional** on that evidence,
and a memory pool, scratch allocation, SIMD, threading, and BLAS are all
currently rejected on it, with the criteria that would reopen each
recorded rather than an answer invented. Every number is a local
characterization of one machine, reported with its spread, and asserted
by no test.

Beyond Phase H
(**not started**): more activations/math, data
loaders, then the CUDA
runtime, dtype/AMP work, and Transformer/text and distributed
experiments. See [roadmap.md](roadmap.md) and
[release_history.md](release_history.md) for the full arc.
