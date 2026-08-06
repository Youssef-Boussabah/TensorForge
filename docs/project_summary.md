# TensorForge — project summary

A two-minute overview for anyone landing on this repository.

## What TensorForge is

TensorForge is a from-scratch deep learning framework built in
Python + NumPy — a serious ML systems project. It
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
no numerical capability. **Phase G (native RNG and Dropout) is complete:
milestones G0 through G10 have all landed.** (The latest *completed*
phase is now Phase H — native CPU performance — recorded below.) G0, the architecture
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

7,844 pytest tests cover every feature of both lines: known-value
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
only — no CUDA backend, no float16/bfloat16, no dtype promotion or
casting, no AMP, no data loaders or native integer tensors, and no
dispatch into `tensorforge.Tensor`. (**Native float32 is supported**, on
the CPU, beside float64, since Phase I milestone **I9** — architecture
contract
[native_dtype_float32_design.md](native_dtype_float32_design.md). float64
remains the default at every constructor, factory, module, and parameter;
the two widths never mix, because casting, promotion, and mixed-dtype
arithmetic are all absent and a mismatch raises before any allocation; the
seven handle-free raw utility kernels stay float64-only permanently; and
AMP, float16, bfloat16, and integer tensors stay outside that phase too.
Phase I itself is **complete**, closed at I11.) Its Dropout is one
deterministic
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
**Phase H — native CPU performance and runtime efficiency — is
complete.** Milestones H0 through H10 have all landed. (This paragraph read
"is the latest *completed* phase" twice, which was accurate until Phase I
closed at I11 and stale afterwards; it is repaired here rather than
rewritten away. The latest completed phase is Phase J.) H10 re-measured the whole phase against a reconstructed and verified H0 baseline (52 cases, **zero checksum mismatches** — every figure compares implementations that produced bit-identical results), resolved the acceleration gate as three documented rejections with measurements (SIMD, threading/OpenMP, BLAS), assessed `tf_core_narrow_backward` and the small-operation boundary floor and implemented neither, ran the full Release/Debug/Linux/sanitizer/lifecycle matrix, and closed the phase. **Every shipped training workload is 1.50×–3.89× faster than at H0**, matmul 4.71×, Conv2d kernels 2.59×–4.64×, reductions 3.78×–5.06×, with no allocation count or memory peak raised anywhere — and across the whole phase **no capability, dtype, device, registry value, public API, checkpoint field, or checkpoint version moved**, with exactly **one** C ABI symbol added (`tf_storage_create_uninitialized`, at H1): 51 → **52**.

Reported as honestly as the wins. The controls held — the unchanged raw-buffer matmul at 0.99×, NumPy at 1.03×, storage allocation at 0.98×, and Dropout at 1.00× — and **`to_numpy` at 0.95× is the one reproducible regression**, attributed rather than smoothed over: its compiled traversal is byte-identical source measuring 0.975×–1.008×, so what changed is that H3's and H7's much cheaper wrapper no longer hides it. The remaining limitations are stated plainly: the gap to a tuned multi-threaded BLAS is **3.6×–9.3×** and widens with size; convolution is entirely scalar (0 packed-double instructions); `tf_core_narrow_backward` still walks the odometer, deliberately, because it executes **0 times** in every shipped training workload; and a small operation still costs a few microseconds because **60 % of that is the owning allocation and 19 % is building the result's Python ownership objects, against 12 % for the ctypes crossing** — an architectural floor rather than a defect. Every number is a local characterization of one machine, reported with its spread, and asserted by no test. H0 is an
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

**Milestone H6 — native reduction execution efficiency — has since
shipped**, the third Phase-H milestone to change C++ and, like H2 and H5,
**not the ABI**: the library still exports exactly **52** `tf_*` symbols.
Reductions were the last core family in the runtime that always paid the
generic strided indexing cost.

The pre-H6 kernel was re-read and re-measured rather than taken from H0's
or H5's summaries, and the cost was **decomposed** instead of assumed. At
`(256, 256)` `axis=0` a `core.sum` costs 99.7 us of which the **raw native
call is 94.8 us — 95 %**; subtracting the three `ndpointer` conversions
leaves the C++ traversal itself at ~91.6 us, **92 %** of the operation. The
entire Python wrapper — axis normalization (0.4 us), output-shape
construction (0.6 us), write-stride construction (0.5 us), the write-stride
array (0.4 us), the H3-cached layout arrays (0.1 us), and the output
allocation (3.2 us) — is about 5 us. So this was the **opposite** of B3:
H3's subject was a fixed Python cost that dominated *small* operations,
while a reduction of any real size is dominated by the compiled loop, and
H6's only worthwhile target was the traversal.

H6 therefore reused the dispatch shape H2 and H5 each proved — one hidden
metadata predicate, inside the existing export, no new symbol, the
pre-milestone traversal retained. New `cpp/include/tf_reduction_internal.h`
declares three hidden-visibility `namespace tf` functions and
`cpp/src/reduction.cpp` implements them: `tf::sum_generic_strided`, the
**pre-H6 odometer retained as the shipped generic reference path** — the
only path that can address a transposed, narrowed, non-unit-strided, or
broadcast source at all, and the oracle every optimized result is compared
against; `tf::reduce_prefers_contiguous_blocks`, the predicate; and
`tf::sum_contiguous_blocks`, a flat walk over an `outer x mid x inner`
factorization. The predicate is total, pure, allocation-free, and a
function of layout metadata alone — never of a pointer value, an alignment,
a clock, an environment variable, or a CPU-feature probe — and a false
answer is a fallback, never an error. It accepts a reduction when (1) the
source strides are exactly the row-major strides implied by the shape (the
same definition `NativeTensorView` uses, so the two layers agree by
construction), (2) the reduced axes — those with a zero *write* stride,
which is how the kernel has always identified them — form **one contiguous
run**, and (3) the kept axes carry exactly the row-major strides of the
output formed by dropping that run. **Stride collapsing is implicit and
bounded rather than a general layout compiler**: conditions 1 and 3 *are*
the statement that adjacent axes of the same class have identical address
progressions, so each group collapses by multiplication, nothing is cached
or interned, and non-adjacent reduced axes (unreachable from Python, which
still takes one `int` or `None`) simply fall back. `keepdims` needs no
special case and the kernel cannot even observe it.

**Per-output accumulation order is preserved exactly, and the source
traversal order is not even reordered**: the `o`, `m`, `i` loop nest is the
lexicographic order of the source's own row-major index, which is precisely
what the odometer walks, and every destination cell is touched by exactly
one `(o, i)` pair, so the cells are independent. Nothing is reassociated,
no partial sums are combined, no accumulator width changes, and no FMA,
Kahan, pairwise, tree, parallel, or horizontal-vector reduction exists.
The `inner == 1` branch (a full reduction, or one whose reduced run is a
suffix) uses a local accumulator **seeded from `dst[o]`**, which is what
keeps the export's documented accumulate-into semantics identical on both
paths; the `inner > 1` branch adds a contiguous source row elementwise into
a contiguous destination row, where distinct `i` are distinct outputs, so
any vectorization is across independent cells and never a horizontal
reduction.

**The signed-zero contract is proved, not assumed.** Both paths start from
the destination's `+0.0`, and `+0.0 + -0.0` is `+0.0`, so the sum of any
number of `-0.0` values is `+0.0` on both paths and matches NumPy; seeded
with `-0.0` both keep `-0.0`. All-positive zeros, all-negative zeros,
alternating zeros, `-0.0` first, `-0.0` last, `-0.0` mixed with finite
values, a column of `-0.0`, and exactly cancelling finite values are each
compared as **raw IEEE-754 bit patterns** at every axis, both `keepdims`
values, and scalar and multi-output shapes. One case is recorded rather
than idealized: the **rank-0** export branch is a genuine
`dst[0] += src[offset]` against a zeroed destination, so a rank-0 `-0.0`
sums to `+0.0` — exactly as before H6, and now pinned by a test.

**The NaN rule is H6's own, measured rather than inherited from H2.**
Contractual: NaN positions are identical on both paths; every NaN either
path produces is quiet, and a signaling-NaN input is quieted by both with
identical bits; and with **at most one NaN per accumulation** — every case
that occurs in practice — the two paths are bit-identical, payload
included. Not contractual: when **two or more** NaNs are accumulated into
one destination cell the paths may select different payload bits, asserted
in neither direction. Why parity is unavailable at any price was measured,
not asserted: four spellings of the optimized accumulation were compared —
`acc += x`, `acc = x + acc`, a named-temporary `acc = acc + x`, and
`dst[o] += x` accumulating *through memory* exactly as the odometer does —
and **all four selected the same NaN and all four differed from the
odometer**, so the local accumulator is not the cause and removing it would
recover nothing. The divergence comes from the odometer's destination index
being a runtime-varying value, which changes which addend MSVC places in
the `ADDSD` destination register; that is an instruction-selection decision
C++ cannot express. The memory-accumulate spelling was also **1.2x-1.8x
slower** on suffix reductions, so it bought nothing. Recorded as an
observation rather than a promise: the block path keeps the **first** NaN
in accumulation order, the odometer the **last**, and the block path's
choice is the one **NumPy** makes — so where they differ, H6 moved the
answer *toward* NumPy. **H5's copy rule does not apply here either**, for
the same reason it made H5's claim strong: a value transfer performs no
arithmetic and so has no operand roles to choose between, while a reduction
is arithmetic. Three operations, three genuinely different rules.

**H1's decision stands, and H6 confirms it rather than revisiting it.** The
destination stays zero-initialized on both paths, because both *read* it —
that is what accumulation means. Outcome B was rejected on two grounds,
one measured and one semantic: the fill is 2,048 bytes against 524,288
bytes of reads at `(256, 256)` `axis=0` and **8 bytes** at `axis=None`,
under half a percent of the traffic against a traversal that was 92-95 % of
the operation; and making the fast path *assign* its first contribution
would give the two paths different behavior for a non-zero destination,
breaking the export's accumulate-into contract and stopping the generic
path from being the reference. H6 therefore adds **no poison test**,
because it introduces no uninitialized destination; the H1 poison suite is
untouched and still passes, `sum` reaching `zeros` and never
`_uninitialized` is asserted structurally, and the accumulate-into behavior
that makes the zero load-bearing has its own negative control at the ABI.

Measured by building a **pre-H6 library** from the identical sources with
only `reduction.cpp` restored, driving both through identical `ctypes`
calls on identical data with every output proved **bit-identical before
either side was timed**, over 15 alternating pre/post rounds; the machine's
control band for this measurement is **0.90x-1.03x**. Kernel level: full
reductions 1.19x at 1,024 elements rising to **3.96x** at `(512, 512)`;
2-D axis reductions 3.24x at `(128, 128)` to **6.37x** at
`(1024, 1024) axis=0`; and — the finding that was **not** predicted —
3-D and 4-D reductions **8.60x-10.94x**, because the odometer's carry loop
runs up to `ndim` iterations per element so its cost grows with rank while
the block traversal's does not. The NCHW rows matter because that is the
layout the convolution stack produces. Layer level, over 9 alternating
subprocess rounds: `TensorCore.sum(axis=0)` 4.49x and `mean(axis=0)` 4.11x
at `(256, 256)`, NCHW `sum(axis=1)` **8.56x** and `sum(axis=3)` **8.82x**,
`NativeTensor.sum` 3.88x without a graph and 3.82x with one, `sum()`
forward+backward 1.27x, `mean` forward+backward 1.23x, the **convolution
bias gradient's three chained sums 1.46x**, `_unbroadcast` 1.15x, softmax
backward 1.14x, log-softmax backward 1.10x, `NativeLayerNorm` forward
1.16x, `NativeBatchNorm2d` backward 1.10x, cross-entropy forward and
backward 1.05x. Against NumPy in the shipped harness the contiguous
reduction gap closed from roughly 8-13x to **1.67x** (4-D middle axis),
**2.43x** (axis 0), 2.90x (last axis), and 3.75x (full to scalar), while
the transposed-view control stayed at 10.33x.

Reported just as honestly. **Every training step is neutral** — MLP small
0.99x, MLP large 1.03x, normalized 1.03x, CNN 1.01x, Dropout control 1.02x,
all inside the control band — so **H6 does not make training faster**, and
no reading should be quoted as if it did; a reduction is a small share of a
step whose cost is the optimizer and the large matmuls. **Normalization is
mostly neutral** too: BatchNorm1d training forward 1.04x, eval 0.98x,
backward 1.02x, BatchNorm2d training forward 1.06x, eval 1.00x, LayerNorm
backward 1.01x, with only LayerNorm forward and BatchNorm2d backward
clearly outside the band — which **narrows H7 rather than motivating it**,
since what is left in those modules is the sheer count of broadcast
elementwise operations rather than the reductions. **Tiny reductions are
neutral** (1 element 1.00x, 16 elements 1.01x, `(8, 8)` axis 0 1.03x),
because below roughly 1,000 elements the fixed ~7 us Python-plus-ctypes
cost dominates — H3's and H5's documented boundary finding, left to a
dispatch milestone. And one **real, repeatable ~10 % regression** is
published rather than buried: a **2-D transposed source reduced over
`axis=0`** measured 0.89x-0.93x across four independent 25-round runs,
while the 3-D transposed `axis=0` fallback measured **1.04x-1.05x faster**
and every other fallback 0.96x-1.01x. Both libraries run the *identical*
odometer there, and the cause was **isolated**: in a standalone binary the
extracted-function spelling versus the inline spelling measured 0.88x-1.67x
with no stable direction, so the extracted call is not it — the remaining
attribution is whole-translation-unit code layout, which is exactly the
machine-specific tuning the design rejects chasing. It affects no shipped
path and no end-to-end case regressed. A specialized register-blocked path
for a small trailing extent (`inner=2` measured 1.75x, `inner=4` 1.77x —
the weakest wins) was **rejected on complexity**. Methodology is published
too: at 7 alternating rounds the fallback controls read 0.85x and at 21-25
rounds the same cases read 0.90x-1.02x, the same lesson H3 and H5 each
recorded, so no low-round figure is quoted as H6 evidence.

**Memory moved not at all, and that is asserted rather than assumed**: a
`sum` allocates **exactly one** native storage — its own output — on both
paths, at every axis, under both `keepdims` values, and `mean` allocates
the same one because its scale is in place. There is no scratch buffer,
workspace, arena, or pool, and the odometer's counter is unchanged and only
on the fallback path. A 10-step training run over a model carrying
parameters, BatchNorm buffers, and Adam moments produced a **bit-identical**
allocation and live-count profile before and after H6, which also confirms
that profile's oscillation is CPython's collector rather than a leak either
version introduced.

The harness gained three cases, 28 to **31**, following H5's
separate-rather-than-average precedent: `reduction_last_axis` (the suffix
form LayerNorm's mean and both softmax backwards actually reduce over),
`reduction_full_to_scalar` (every write stride 0 and a rank-0 output — the
hottest reduction in the runtime, since every mean-reduced loss ends in
it), and `reduction_middle_axis_4d` (kept axes on both sides, so all three
block extents exceed 1, plus the rank-4 reading the 2-D cases cannot give),
with `reduction_transposed_view` now explicitly the pair's control because
the predicate rejects it. One dependency-free CTest was added,
`cpp/tests/test_sum_reduction.cpp`, taking the native suite from 14 to
**15**; it drives the predicate table, both traversals in isolation, the
accumulate-into contract over a pre-filled destination, and the
special-value matrix at the layer where those properties are actually
decided. **No exported C ABI symbol, no new translation unit, no public
control of any kind** — no path selector, threshold setter, block-size
setter, dispatch tracer, profiling counter, environment variable, or
"which path ran" query — and no SIMD, threading, OpenMP, BLAS, parallel
reduction, memory pool, scratch workspace, or fast-math. Multi-axis
reduction was **not** added: the kernel can factorize a contiguous reduced
run, but the Python layer still accepts one `int` or `None`, with every
signature, default, axis rule, `keepdims` behavior, error type, and error
message exactly what they were. `tf_core_narrow_backward`, the odometer's
scatter dual, was deliberately left alone — widening H6 to it would have
made this a scatter milestone. No public API, capability, dtype, device,
registry value, checkpoint field, or checkpoint version moved.

**Milestone H7 — native Python/C ABI boundary efficiency — has since
shipped**, and it is **Python-only**: no C++, no exported symbol, no kernel,
no traversal, no arithmetic. The library still exports exactly **52** `tf_*`
symbols.

**The ladder was revised here, and the revision is recorded rather than
retrofitted.** H0's H7 slot was *composed-module cost* — the normalization
modules and the composed convolution bias gradient — explicitly conditional
on a re-measurement after H1, H3, and H6. That condition was tested and
**not met**: H6 made `mean` 3.9x-4.1x faster and moved the normalization
modules almost not at all (`NativeLayerNorm` forward 1.16x,
`NativeBatchNorm2d` backward 1.10x, everything else inside the
0.90x-1.03x control band, the normalized training step 1.03x). So the milestone was **dropped on evidence**, its
proposal and the evidence against it preserved in the design document
rather than deleted, and the slot was refilled from the *same*
measurements: H3, H5, and H6 had each ended by deferring the identical
named cost — H5 "~1.1 us per layout array at the ctypes boundary, left to a
later dispatch milestone", H6 "the fixed ~7 us Python-plus-ctypes cost,
left to a dispatch milestone". Three milestones deferred one thing to a
later dispatch milestone; H7 is that milestone. Composed-module allocation
count remains conditional future scope, and is H8's subject.

The cost was **decomposed rather than assumed**, and the claim that six
kernels were involved was checked and found wrong. All 52 exports are
configured in one file — no other module in the repository imports
`ctypes` — and **57 of their argument positions are arrays**, every one
formerly bound as `numpy.ctypeslib.ndpointer`. That binding re-verifies
array-ness, exact dtype, and contiguity at **every call**, then constructs
`obj.ctypes` and resolves it through `_as_parameter_`: two Python object
constructions and three checks, per array, per call, measured at **~2.1 us
per array position**. On real calls: `tf_core_add` on a 4x4 with three
layout arrays cost **7.6 us**, of which **6.1 us** was the binding; the
array-free `tf_core_add_contiguous` cost 0.9 us and is the control.

Then the *frequency* was counted: an MLP training step makes 245 native
calls carrying **101** array crossings, a normalized step 692 calls and
**315**, a CNN step 242 and **104** — about **20-23 %** of each step's wall
time. And the provenance was the finding that decided the architecture:
**~85 % of those crossings are operation-local broadcast strides**, not the
H3 per-view cache, so a design that only cached pointers per view would
have captured a seventh of the available work.

H7 ships **two bindings for two categories, and deliberately not one
blanket policy**. *Data* positions keep the checked `ndpointer` binding —
the seven public raw-buffer kernels (whose callers may pass anything), the
`copy_from`/`copy_to`/`materialize` host conversion boundary, and the
cross-entropy **class labels**, which are int64 like the layout metadata
but stay checked because a label array's required length comes from the
*logits*, a different object. *Layout metadata* positions — 32 of them
across 13 exports — take `ctypes.POINTER(ctypes.c_int64)`, fed by exactly
two private producers: `NativeTensorView._native_layout_pointers()`, which
memoizes `data_as` over the **unchanged** H3 read-only NumPy arrays that
remain the owning buffers, and `_layout_vector(values)`, which builds a
fresh `(c_int64 * len(values))` for metadata belonging to one operation.

**Nothing was weakened, and one thing was strengthened.** ctypes still
type-checks every call: a trusted position rejects a NumPy array of any
dtype, a differently typed pointer or vector, a `c_void_p`, a list, an
int, bytes, and a string — a NumPy array being rejected is a deliberate
consequence that makes the old binding unreachable by accident. Dtype,
byte order, and contiguity are established *by construction* rather than
re-checked. **The length/rank invariant — the one `ndpointer` never
checked, because the ABI sees only a pointer and an `ndim`** — is now
checkable for the first time: a vector carries its length in its type, and
a cached pointer carries its owning array (NumPy's `data_as` attaches it),
whose length is the view's rank. The suite asserts that per producer, per
rank 0-4, and structurally over **every strided call in a real workload**.
The one honest difference is `None`, which `ndpointer` rejected and a typed
pointer converts to NULL; it is closed structurally — both producers are
total and no public API takes a metadata pointer. (Only **three** of the
thirteen exports reject a null metadata pointer in C++ as well, so the
producers being total is the load-bearing part of the argument, not a
belt-and-braces remark.)

**Ownership is NumPy's guarantee, relied on and tested rather than
assumed.** `data_as` stores the array on the pointer, so a cached pointer
cannot outlive its buffer; `POINTER.from_address` was measured **faster**
(0.9 us against 1.6 us) and **rejected outright** because it produces a
pointer with no owner. Deriving the pointer from the array rather than
building a second vector keeps exactly **one owning description** of a
view's layout — a cached ctypes vector was measured fastest of all and
rejected, because it would duplicate that description and lose H3's
`writeable = False` protection for ~2 % of a training step. Proved with the
cyclic collector **disabled**: no reference cycle (an explicit
`gc.collect()` after dropping a view collects 0 objects), no native storage
kept alive, no pointer surviving into a usable state after close, and
operation-local vectors retained by nothing. There is no global pointer
cache, no id-keyed table, and no `from_address`, `byref`, `addressof`,
`id`, or weak-reference container **anywhere in the module's code** —
enforced by parsing it, so the docstring recording *why* `from_address` is
unused is not mistaken for using it. Binding configuration stays a
load-time act: `argtypes`/`restype`/`errcheck` are assigned only inside the
two loader functions, asserted by locating each assignment's enclosing
function, so nothing reconfigures a shared function object per call and
**no thread-safety claim is broadened**.

Measured against a **retained pre-H7 `cpp.py` driving the same Release
DLL**, over 11 alternating pre/post subprocess rounds, every case proved
**bit-identical before either side was timed**; control bands 0.95x-1.10x
(core) and 0.99x-1.05x (end to end). Core: a 1-element `sum` **1.94x**, a
16-element `sum` 1.89x, `to_numpy` 16x16 **1.83x**, `sum(axis=0)` 16x16
1.79x, a 4x4 `contiguous_copy` 1.73x, `narrow_backward` 1.73x, strided
`relu_backward` 1.71x, scalar-broadcast `add` **1.67x**, 4-D NCHW
`sum(axis=1)` 1.54x, row-broadcast `add` 1.41x, `sum(axis=0)` 256x256
1.35x, strided `exp` 1.29x, transposed materialization 1.16x. End to end —
and this is the result — the **native Dropout step 1.32x, the normalized
step 1.31x, `NativeAdam` at (32, 32) 1.31x, the CNN step 1.30x, the MLP step
1.28x**, `NativeLayerNorm` forward 1.23x, `NativeBatchNorm1d` eval
1.23x, `NativeAdam` at (128, 128) 1.14x, `NativeSGD` 1.13x, the large MLP step 1.08x. **H7 is the
first Phase-H milestone to move every training step** — H4 moved them
1.09x-1.23x and H5 and H6 were neutral on all of them — because the cost
is paid per *call* and a step makes hundreds of them.

Reported just as honestly: **large kernel-bound work is neutral**, exactly
as the attribution predicts. 256-cubed matmul **0.99x** and 8-cubed matmul
1.00x are controls that take no array at all, so **H2's result is
structurally untouched**; contiguous 16x16 `add` 1.05x is the third
array-free control; and 512x512 `copy` 1.02x, 256x256 `to_numpy` 1.04x,
512x512 full `sum` 1.06x, 256x256 broadcast `multiply` 1.08x, and the large
MLP step 1.08x are all at or inside the band. **H7 did not make matmul faster**,
and no reading should say otherwise.

A second, independent 11-round run reproduced every row: all cases again
bit-identical, every control again holding (256-cubed matmul 0.98x, 8-cubed
matmul 1.04x, contiguous 16x16 `add` 1.08x, 512x512 `copy` 1.01x), and every
training step again improving, with individual ratios moving by roughly the
control band's width in both directions (a 1-element `sum` 2.12x against
1.94x, `NativeSGD` 1.26x against 1.13x, `NativeAdam` at (128, 128) 1.08x
against 1.14x). The figures quoted are the first run's; the second is
recorded in the design so no single number is read as more precise than the
method supports.

**Milestone H8 — native elementwise traversal and composed allocation
efficiency — has since shipped**, the fourth Phase-H milestone to change
C++ and, like H2, H5, and H6, **not the ABI**: the library still exports
exactly **52** `tf_*` symbols.

H8 entered with **two** candidate tracks and an explicit instruction not to
force both into production. The measurement kept one as a large result,
kept a narrow piece of the other, and rejected the rest with reasons.
**Track A — elementwise traversal — was confirmed and is the milestone.
Track B — composed normalization allocation — was confirmed only as a
memory result and is reported as timing-neutral.**

The cost was **decomposed rather than assumed**, and the decomposition is
what decided the architecture. Driving the generic strided walker and the
flat contiguous walker through identical `ctypes` calls on identical
*contiguous* data showed the odometer costing **1.60x-6.42x** the flat
loop; a separate sweep showed that **all broadcasting is on the odometer**
(there is no broadcast fast path at all — `_binary_core_op`'s path C builds
per-operation broadcast strides and hands them to the same generic walker),
at 2.0-3.6 ns per element. A standalone binary with an anti-hoisting guard
then split the odometer's cost four ways at `(256, 256)` contiguous `add`:
the shipped odometer-plus-function-pointer at **123.5 us**, templating
alone **81.3 us (1.52x)**, collapsing alone **63.6 us (1.94x)**, and
**both together 11.5 us (10.7x)**. Neither change is worth much alone and
together they are worth an order of magnitude, because only their
combination lets the compiler emit a vector loop — the odometer's carry
chain blocks vectorization and so does an indirect call it cannot see
through. The same run showed the **existing flat contiguous kernel was
itself hobbled** by that indirect call: 21.0 us against 11.7 us for the
identical loop with the operation as a compile-time constant.

H8 therefore reused the dispatch shape H2, H5, and H6 each proved — one
hidden metadata builder, inside the existing export, no new symbol, the
pre-milestone traversal retained. New
`cpp/include/tf_elementwise_internal.h` declares `tf::build_unary_plan` and
`tf::build_binary_plan` plus the templated traversals that walk what they
build; `cpp/src/elementwise.cpp` defines the builders and dispatches to
them. A **plan is an operation-local normalized descriptor**: built on the
stack, used by one call, dropped. Nothing is cached, interned, memoized, or
shared between calls, and it applies exactly two transformations, both of
which preserve the logical element sequence — **unit axes are dropped**,
and **adjacent axes are merged** when `stride[outer] == stride[inner] *
extent(inner)` for *every* operand at once, which is the statement that the
two axes' address progressions form one arithmetic run. Axes are never
reordered, split, or transposed. **This is not a layout compiler**; the
bound is a fixed **4 axes**, which is every tensor the runtime can
construct. The builders are total, pure, allocation-free, and a function of
layout metadata alone — never a pointer value, an alignment, a clock, an
environment variable, or a CPU-feature probe — and a rejection is a
**fallback, never an error**: rank 0, an extent below 1, an element count
not representable in `int64`, a collapsed rank still above 4, or an
overflowing merge *test* all fall back to the retained odometer. (An
overflowing merge *action* simply leaves the axes unmerged, which is always
a valid description — collapsing is an optimization, never a correctness
requirement.) `core_unary` and `core_binary`, the pre-H8 odometers, are
retained **verbatim** as the shipped generic reference paths, still spelled
with the odometer's counter.

**Only operations IEEE-754 actually specifies take the templated path** —
`add`, `subtract`, `multiply`, `relu_backward`, `relu`, `sqrt`,
`reciprocal`, and the identity gather behind `tf_core_contiguous_copy`.
**`exp` and `log` keep exactly the paths they had**, because they are
library functions with no correctly-rounded guarantee and a toolchain that
vectorized them through a vector-math library would be free to return
different bits. Nothing is lost by excluding them: measured, the templated
traversal is worth **1.05x** on both, inside this machine's noise. The
pre-H8 flat *binary* kernel was removed rather than kept as a second copy
that could drift, because the templated row with both strides 1 is the
identical loop and is total, so no predicate can ever fall back to it.

**The numerical contract is H8's own, measured over every ordered pair of
14 IEEE-754 representatives x three operations x five layouts against a
pre-H8 library built from identical sources.** (1) **Every result in which
at most one operand is a NaN is bit-identical** — signed zeros, infinities,
denormals, the smallest normal, the largest finite magnitudes, and a lone
NaN of either sign with any payload included: **zero differing results**
across all 15 combinations. (2) NaN **positions** are identical and every
NaN the arithmetic produces is quiet (`relu_backward` and the identity
gather **select** an operand rather than computing, so a signaling NaN
legitimately survives them — identically on both traversals, exactly as H5
established for the copy). (3) **Subtraction is bit-identical everywhere**,
two-NaN pairs included, because it is not commutative and the compiler has
no freedom over which operand reaches the destination register. (4) For
**addition and multiplication with two NaN operands** the surviving payload
is outside the contract and asserted in neither direction. **Part 4 is not
something H8 introduced, and H8 narrows it**: measured on the pre-H8
library, its *own* flat kernel and its *own* odometer already disagreed on
**30 of 196** such pairs, while post-H8 the contiguous, same-shape strided,
and row-broadcast paths agree exactly and only a transposed operand differs
— on **5 of 196**. This is a *different* qualification from H2's and H6's,
which concerned NaNs meeting inside an accumulation; here there is no
accumulation at all, only operand order inside one commutative
instruction. Nothing is reassociated, no FMA, no fast-math, no intrinsic,
no `restrict`, and each functor spells the same expression its retained
function-pointer twin spells, character for character.

**H1's contract holds unchanged**: elementwise outputs stay uninitialized,
and the plan walk writes the destination strictly left to right over
exactly the logical element count — the same order and count the odometer
produces — proved by poison injected by test infrastructure with two
patterns over six layouts, with guard elements on both sides and a
**negative control** showing the detector can fail.

**Track B** shipped the one composed-allocation change the evidence
supported: `_NativeBatchNorm` builds its `(1 - momentum, momentum)` pair
**once per forward** instead of once per buffer — the per-step-constants
shape H4 proved on the optimizer, never stored on the module, so no scalar
survives a forward, enters `state_dict()`, or reaches a checkpoint — and
each blend **releases its temporaries at last use** rather than holding
them to the call's `finally`. Measured against a **retained pre-H8
composition executed natively**, with running statistics proved
bit-identical first: a `NativeBatchNorm1d` training forward goes **25 -> 23**
allocations with **peak live storages 25 -> 17** and constant fills
**5 -> 3**; `NativeBatchNorm2d` goes **30 -> 28**, **30 -> 22**, and **5 -> 3**.
**Its timing effect is neutral** (1.007x-1.106x over 15 alternating
rounds, only the smallest shape outside the band and only marginally):
**Track B is a memory result, not a timing result**, and no reading should
be quoted as if it were otherwise. Four alternatives were **rejected with
reasons**: releasing the normalization graph temporaries early (proved
impossible — every one is either read by a backward closure or must stay
open to accumulate a gradient); adopting the blend's core into the running-
state transaction instead of copying it (it would move numerical work
inside the staging phase, changing a failure ordering F5 and F8 prove by
test, to save two channel-sized copies); caching the eval-mode inverse
standard deviation (the hidden mutable state this design forbids); and
reshaping `gamma`/`beta` to `(1, C, 1, 1)` to skip `NativeBatchNorm2d`'s
affine transposes (F4 rejected that for a semantic reason that has not changed —
a reshaped parameter is unversioned, so the stale-parameter guard would
silently stop firing).

Measured against the pre-H8 library on identical `ctypes` calls with every
case **bit-identical before either side was timed**, 11 alternating rounds,
identical-code control band **0.97x-1.08x**: `multiply` row-broadcast
`(256,256)+(256,)` **10.58x**, `add` strided same-shape `(256,256)`
**9.67x**, `multiply` NCHW-stat `(32,16,16,16)` **7.15x**, `multiply`
col-broadcast **6.70x**, `add` scalar-broadcast **6.31x**, rank-3 broadcast
**6.18x**, NCHW same-shape 3.53x, NCHW-stat `(8,4,8,8)` 3.11x, transposed
`add` 2.63x, strided `relu` 2.51x, transposed `copy` 2.31x, contiguous
`sqrt` 2.03x, `reciprocal` 1.98x, `relu_backward` 1.86x, contiguous `relu`
1.78x, contiguous `add`/`multiply` 1.76x, contiguous `copy` 1.68x. Layer
and end to end, over 11 alternating **subprocess** rounds with all 31 cases
bit-identical first: `TensorCore` row-broadcast `multiply` **6.81x**,
`NativeTensor` broadcast multiply with a graph **6.33x**, scalar-broadcast
`add` **5.12x**, **`NativeAdam.step()` at (128,128) 2.01x**, `relu` 1.59x,
contiguous `add` 1.55x, **`NativeBatchNorm1d` eval forward 1.40x**, **`NativeBatchNorm2d`
eval forward 1.36x**, **`NativeBatchNorm2d` training forward 1.33x**, **`NativeLayerNorm`
forward 1.30x**, `NativeBatchNorm2d` fwd+bwd 1.25x, `NativeLayerNorm` non-affine 1.22x,
`NativeBatchNorm1d` training forward and `NativeLayerNorm` fwd+bwd 1.21x, **the large MLP
training step 1.19x**, `NativeBatchNorm1d` fwd+bwd 1.15x, `contiguous_copy`
`(512,512)` 1.14x, **the normalized training step 1.08x**, the native Dropout step
1.06x. **This is the milestone that finally moved the normalization
modules** — which H6 measured as almost entirely neutral, and which is
precisely why H0's composed-module H7 was dropped and this one entered.

Reported just as honestly. **Small normalization shapes are neutral**:
`NativeBatchNorm1d` training at `(32,16)` **0.98x**, `NativeBatchNorm2d`
`(8,4,8,8)` 1.02x, `NativeLayerNorm` `(32,16)` 1.06x — below roughly 1,000 elements the fixed
Python-plus-ctypes cost dominates, which is H3's, H5's, and H6's documented
boundary finding, unchanged. **The CNN step is neutral (0.99x)** and the
small MLP and SGD steps sit at the band edge (1.01-1.02x), because a
convolution step's time is in `tf_core_conv2d_*`, which H8 did not touch.
The `exp`/`log` controls read 0.97x-1.07x, exactly as the deliberate
exclusion predicts, and `sum` and 128-cubed `matmul` are inside the band.
One control is **published rather than buried**: **`matmul` 256 cubed reads
0.93x-0.96x**, and a focused 25-round run shows the effect at that one size
only — 64 cubed 1.014x, 128 cubed 1.035x, 256 cubed **0.921x**, 384 cubed
0.994x — while the identical-code twin reads 0.969x on the same case.
`matmul.cpp` is byte-identical source compiled with identical flags;
`elementwise.cpp`'s object code grew 127 KB to 188 KB, moving every
function's placement in the image. **This is the same whole-translation-
unit code-layout effect H6 documented**, it is the machine-specific tuning
the design rejects chasing, every matmul result is bit-identical, the H2
CTest passes, and no end-to-end case regressed.

**Memory: Track A moved none, and the odometer's heap-allocated counter is
now removed on every plannable layout** — a strided elementwise call makes
**one** allocation where it previously made two, which is a strict
reduction and which re-anchored one existing fault-injection test (its
assertion unchanged, its operand changed to a rank-5 reversed view the
builder declines, with a **new** test asserting the other half). The
harness gained **four** cases, 34 to **38**:
`elementwise_broadcast_column` and `elementwise_broadcast_channel_4d`
(following H5's and H6's separate-rather-than-average precedent — the row
case stretches the leading axis, the column case the trailing one, and the
NCHW case puts the stretched axis in the middle where neither side folds
into it), plus `elementwise_unary_contiguous` and
`elementwise_unary_transposed`, because every other elementwise case is
binary and the one-source traversal was only ever visible averaged into a
two-operand measurement. Native CTests 15 to **16**. **No exported C ABI
symbol, no new translation unit, and no public control of any kind** — no
path selector, plan inspector, collapse-mode flag, threshold setter,
dispatch tracer, profiling counter, environment variable, or "which path
ran" query — and no SIMD, threading, OpenMP, BLAS, memory pool, scratch
workspace, general fusion, or fast-math. No public API, capability, dtype,
device, registry value, checkpoint field, or checkpoint version moved.

**Milestone H9 — native Conv2d execution efficiency — has since shipped**,
the fifth Phase-H milestone to change C++ and, like H2, H5, H6, and H8,
**not the ABI**: the library still exports exactly **52** `tf_*` symbols.

**H9 was not in H0's ladder.** H0 pencilled that slot in for SIMD,
threading, or optional BLAS, *conditional and presumed rejected*. It was
not entered — none of the three qualified, and a larger, safer result was
available in the same slot. H6 made reductions 3.2×–10.9× faster and left
every training step neutral; H7 moved every training step but the CNN
step's share came from its many small calls; H8 moved the normalization
modules 1.21×–1.40× and the **CNN step stayed neutral at 0.99×** — because
a convolution step's time is in `tf_core_conv2d_*`, which was still the
unmodified Phase-D direct loop from D2–D5 while matmul, copy, reduction,
and elementwise had each been revisited. Convolution was the last large
compute family running its original correctness-first implementation, and
it had become the majority of the one workload Phase H had never moved.
The acceleration decision moved to H10's decision gate, where it is a
decision rather than an implementation.

The cost was **decomposed rather than assumed**, and the answer was H6's
rather than H3's: timing the complete Core wrapper against the bare
foreign call showed the Python wrapper is a fixed ≈ 8–12 µs, **66 %** of a
toy `(4,1,6,6)` convolution but **0.2 %** at `(8,3,16,16)` and
**≈ 0 %** at `(16,8,32,32)`. For any convolution with real work the
compiled traversal is essentially **100 %** of the cost, so the C++ loop
was the only target worth having. The composed **bias gradient was
measured and found immaterial** — three chained `sum` reductions H6
already made 3.9×–4.1× faster, producing an `(O,)` result beside two
full-tensor gradients — so H9 changes nothing about it, and that is a
recorded negative result rather than an oversight.

All three pre-H9 kernels shared one shape: `n, o, i, j` outer, `c, p, q`
inner, with the padded source coordinate recomputed and bounds-tested in
the inner loops. That makes the innermost loop `kernel_width` — typically
**3** — iterations long, recomputes a row bound that depends only on
`(i, p)` once per input channel, and makes both gradients read-modify-write
destinations that far-apart iterations revisit. H9 reuses the dispatch
shape H2, H5, H6, and H8 each proved: **one hidden predicate, inside the
existing export, no new symbol, the pre-milestone traversal retained**.
`tf::conv2d_forward_generic`, `tf::conv2d_input_backward_generic`, and
`tf::conv2d_weight_backward_generic` are the **Phase-D direct loops
retained verbatim** as the shipped generic reference paths — reachable
through ordinary production dispatch and the oracle every optimized result
is compared against. Beside them: `tf::conv2d_forward_row_sweep`, whose
nest becomes `n, o, i | c, p, q | j` accumulating into a bias-primed
output row; `tf::conv2d_input_backward_gather`, which walks `grad_input`
rows and gathers instead of scattering; and
`tf::conv2d_weight_backward_gather`, which owns one destination at a time
and sums it in a register, writing it **once** instead of
`batch·out_h·out_w` times. Two file-local helpers compute the half-open
run of kernel taps whose source lies inside the real input — that run is
always contiguous, which is why solving for it and testing each candidate
skip the identical taps.

**The fast-path preconditions are one shared rule plus one
direction-specific one**: `min(input_width, output_width) >= 4` for all
three, and additionally **unit stride in both axes** for the input
gradient. The minimum is measured, not tuned — at a swept extent of 1 the
optimized forms ran **0.57×–0.93×**, at 2 they ran 1.04×–1.38×, and at 4
they ran **1.91×–2.40×**. `min(input_width, output_width)` is used because
it is the honest bound on all three inner loops, and keying the input
gradient on `input_width` alone **was measured wrong** — a 5-wide input
with a 1-wide output sweeps a single element and ran **0.73×**. The
predicates are total, pure, allocation-free, and functions of the integer
geometry alone — never a pointer value, an alignment, a clock, an
environment variable, or a CPU-feature probe — and **a false answer is a
fallback, never an error**. The input gradient alone needs unit stride
because its gather walks the kernel offsets *downward* to reproduce the
reference's ascending output order, and that inversion is one-for-one only
at unit stride; the forward and weight gradient take their optimized paths
at **every** stride. **The asymmetry is deliberate**, and the strided input
gradient's 1.04× is the row that proves the fallback is really taken.

**Per-destination accumulation order is preserved exactly in all three
directions**, each with its own proof: the forward's `c, p, q` stay outer
to `j`, so a destination still receives the same seed and the same taps in
the same order; the input gradient's ascending-`o`, descending-`p`,
descending-`q` walk *is* ascending `o`, `i`, `j` at unit stride; and the
weight gradient's `n, i, j` nest is exactly the order its destination's
contributions already arrived in. Nothing is reassociated, no partial sums
are combined, no accumulator width changes, and there is no FMA,
fast-math, tree or pairwise reduction, parallel accumulation, SIMD
intrinsic, or threading anywhere.

**The numerical contract is H9's own, measured against a pre-H9 library
built from identical sources with only `conv2d.cpp` restored.**
Contractual: (1) **every non-NaN result is bit-identical** — 256 ordered
pairs of 16 IEEE-754 representatives × 3 directions, **zero non-NaN
differences**, signed zeros, ±∞, denormals, the smallest normal, and the
largest finite magnitudes included; (2) **NaN positions are identical** in
all 768 comparisons; (3) **with at most one NaN reaching a destination the
paths agree exactly, payload included** — 480 single-NaN configurations
across five geometries, zero differences; (4) **signed zeros are
bit-identical** across 80 sign-pattern configurations, with `−0.0`
surviving only while every addend is `−0.0` and one `+0.0` making the sum
`+0.0` — both halves asserted on both paths, because the sweep replaces a
register accumulator with accumulate-into-memory and the weight gather
does the reverse, exactly the rewrites that could change a zero's sign;
(5) signalling NaNs are quieted identically and every NaN either path
produces is quiet. **Not contractual**: when two or more NaNs reach one
destination the surviving payload may differ (20/256, 20/256, and 29/256
pairs), asserted in neither direction — the same qualification H2 and H6
recorded, for the same instruction-selection reason, but **measured here
rather than assumed from them**.

**H1's contract holds on all three destinations**, for a different reason
each: the forward primes every element of every output row with the bias
before accumulating; the input gradient's gather zeroes each row it visits
and visits every row; and the weight gradient's gather *assigns* every
destination from a register, so it needs no zero-fill and never reads the
destination at all. Proved by poison in two places — the C++ suite
pre-fills each optimized destination with a quiet NaN and a large finite
value across the whole geometry matrix, and the Python suite injects the
same two patterns through the real private allocation seam for both the
optimized and the fallback geometries — each with a **negative control**
proving the detector can fail.

**Layout handling is untouched**: the C ABI is contiguous-only by Policy B
and the Core layer materializes any non-contiguous operand into a private
copy, so **H9 is a geometry optimization, not a layout one** and broadened
layout support by nothing. Autograd is untouched — same parent topology,
same conditional version tracking, gradients created only for parents that
require them, and the same `retain_graph`, repeated-backward,
accumulation, and cleanup behaviour. Nothing became in-place.

**Memory did not move, and that is asserted rather than assumed**: the
same workload on both libraries reports **byte-identical** allocation
counts, peak live storages, and peak bytes — a forward is 1 allocation /
921,600 peak bytes, an input gradient 1 / 524,288, a weight gradient
1 / 9,216, a `NativeConv2d` forward+backward 8 / 3,020,160, and a CNN
training step 94 allocations / 34 peak live / 604,848 peak bytes. There is
**no scratch buffer, workspace, arena, pool, padded copy, or im2col
allocation anywhere**.

Measured against the pre-H9 library on identical `ctypes` calls, every
case **bit-identical before either side was timed**, 11 alternating rounds,
with the two fallback geometries as the identical-code control at
**1.00×–1.13×**: k1×1 `(8,16,32,32)` **6.23× / 8.37× / 5.40×** (forward /
input / weight), `(8,16,32,32) → 32` k3×3 3.64× / 5.04× / 2.60×,
`(16,8,32,32) → 16` 3.32× / 5.28× / 2.47×, padded `(8,8,32,32)` 2.91× /
4.97× / 2.54×, `(8,3,16,16) → 8` 2.87× / 3.75× / 2.45×, prime extents
2.61× / 4.57× / 2.77×, **stride 2** 2.41× / *1.04× (falls back)* / 2.41×,
rectangular k3×5 2.30× / 3.76× / 2.22×, k5×5 1.97× / 3.84× / 2.09×. End to
end over 9 alternating **subprocess** rounds, all 23 checksums identical
first: `NativeConv2d` forward+backward **3.13×** padded and **3.09×**
unpadded, forward **2.98×**, no-bias 2.46×, frozen 2.40×, stride-2 2.28×,
and — the result that matters — **a CNN training step 1.86×** at
`(8,3,32,32) → 16`, **1.38×** at `(8,3,16,16) → 8`, 1.27× with Dropout,
**1.13×** at the shipped example's shape, and 1.11× with BatchNorm2d.
**This is the first Phase-H milestone to move a CNN training step**, which
H6 and H8 both measured as neutral.

Reported just as honestly: a **small convolution is neutral** (1.06×
forward, 1.20× forward+backward at `(4,1,8,8) → 4`), because below roughly
a thousand output elements the fixed ≈ 10 µs Python-plus-ctypes cost
dominates — H3's, H5's, H6's, and H8's documented boundary finding,
unchanged; the **BatchNorm2d and shipped-example CNN steps move least**
(1.11×, 1.13×), because convolution is a smaller share of those steps; and
**no control regressed** — at 21 alternating rounds matmul 256³ **0.98×**,
the MLP training step **0.97×**, `contiguous_copy` 512² 1.01×, reduction
1.07×, broadcast elementwise 1.07×, a control band of **0.97×–1.07×**. One
methodology finding is published rather than buried: at 9 rounds the
elementwise control read **0.93×** and looked like a regression, while at
21 rounds the same case read **1.07×** — the lesson H3, H5, and H6 each
recorded, so no low-round figure is quoted as H9 evidence.

**Four candidates were rejected with reasons**: **im2col + matmul**
(changes the accumulation order, which is the whole contract, and would
allocate 8× the input at the profile shape); a **materialized padded
input** (moves cost rather than removing it, when the tap-range helpers
give the same branch-free inner loop with no allocation); **output-channel
blocking** (the sweep already produces a long unit-stride inner loop, and
blocking `o` would reintroduce a tuning constant for a second-order gain);
and a **third "hoisted" path for small extents** (measured 1.5×–3.7× and
never regressing, but it would have left the Phase-D reference unreachable
for shapes that cost microseconds).

Validation: Windows **Release and Debug**, both out-of-source with the
Debug library written outside the repository so the active runtime stayed
the Release DLL, **17/17 CTests each** with zero project compiler, linker,
and CMake warnings; Clang 18.1.3 ASan/UBSan with **instrumentation
proved** — 22 `__asan*` and 15 `__ubsan*` dynamic symbols beside the
**52** exported `tf_*` symbols, independently confirming the export count
on a second toolchain — **17/17 sanitized CTests**, **445 sanitized
convolution/CNN/Phase-D/H1 tests**, the full sanitized native suite with
**zero ASan and zero UBSan diagnostics**, and both shipped CNN examples
reproducing their exact checkpoint resumes under it. A **sanitizer
negative control** makes that absence real: handing the row sweep an input
one row shorter than its declared geometry produces a
`heap-buffer-overflow`, `READ of size 8`, inside
`conv2d_forward_row_sweep`. A LeakSanitizer lifecycle returns native live
storage **exactly to baseline (0)** at every checkpoint — core forward and
both gradients over optimized *and* fallback geometries, module cycles,
seven injected-failure cycles, abandoned graphs, and two complete CNN
training runs — with the remaining process-exit allocations containing
**no TensorForge frame** and no suppression file added.

The harness gained **three** cases, 38 to **41** —
`conv2d_forward_padded`, `conv2d_forward_strided`, and
`conv2d_forward_fallback`, following the separate-rather-than-average
precedent, because unlike H5/H6/H8 the chooser here is the *geometry*
rather than the layout; all three are `native_only` and publish **no
ratio**, and the fallback case is the family's control since its compiled
path did not change. Native CTests 16 to **17**. **No exported C ABI
symbol, no new translation unit, and no public control of any kind** — no
path selector, block-size setter, traversal control, dispatch tracer,
benchmark hook, profiling counter, environment variable, or "which path
ran" query — and no SIMD, threading, OpenMP, BLAS, oneDNN, Eigen, memory
pool, scratch workspace, im2col, or fast-math. No convolution option was
added: no dilation, no groups, no channels-last, no new padding mode. No
public API, capability, dtype, device, registry value, checkpoint field,
or checkpoint version moved.

**Memory did not move, and that is asserted**: the same boundary workload
allocates 5 native storages, peak 4 live, 584 peak bytes before and after —
identical. A view's cold footprint is byte-identical; a view that actually
takes a strided path costs **+296 bytes** for the pointer pair, and only
**9 of 98** views in an MLP step ever populate it, which is H3's laziness
argument unchanged. The harness gained three cases, 31 to **34**:
`ctypes_boundary_strided` (the array-carrying twin of the existing
array-free `ctypes_boundary`, so the two crossings are separated rather
than averaged — measured **0.8 us** array-free against **1.3 us** with
three layout arguments, where pre-H7 it would have been ~7 us), plus
`elementwise_broadcast_scalar` and `elementwise_broadcast_row`, the two
broadcast shapes the optimizer and the normalization modules actually use.
Validation added a **sanitizer negative control**: under Clang ASan,
test-only code handing `tf_core_sum` two-entry metadata with `ndim = 3`
produces a `heap-buffer-overflow`, `READ of size 8`, `0 bytes after 16-byte
region`, in `reduce_prefers_contiguous_blocks` — the exact H3 finding —
which is what makes the **zero diagnostics across 2,834 sanitized tests** a
real absence rather than a blind detector. No public API, capability,
dtype, device, registry value, checkpoint field, or checkpoint version
moved, and no C ABI symbol was added.

The ladder ran **H0–H10 and ended there**: it was reordered at H5, revised at H7 (a milestone dropped on evidence), and extended at H9 (a slot reassigned), and H0's separate H11 closure slot was **not needed** because H10 carried closure itself. A memory pool, scratch allocation, SIMD, threading/OpenMP, and BLAS were **all finally rejected at H10, with measurements** — the disassembly showed elementwise, matmul, and reduction are already auto-vectorized; a CNN step's 198 native calls have a **1.20 µs median** with only two above 1 ms; and BLAS is **not bit-identical** (3.553e-15 at 64³), which would break every exact-resume proof. The criteria that would reopen each are recorded rather than an answer invented. Every number is a local characterization of one machine, reported with its spread, and asserted by no test.

**Phase J — deterministic native data pipeline and mini-batching — is
complete: milestones J0 through J9 have all landed, and J9 closed it.**
**Phase K is the latest phase, and only K0 through K6 have landed.** **K7 through K9 are unstarted.** **Phase J is the latest completed phase**, and it remains complete. Phase J was approved *after* Phase I
closed at I11, so it is not pre-existing roadmap work. **J0 was
architecture, contract, and documentation work and added no runtime
behavior at all** — no dataset, sampler, or loader class, no helper module,
no state serializer, no public export, no C++, no C ABI symbol, no example,
no benchmark, and no checkpoint or optimizer-state change. Runtime
capability began at **J1**, which shipped `NativeTensorDataset`: the
finite host-backed dataset holding one owned, copied host snapshot of the
features and one of the class targets at an **explicitly chosen** native
feature dtype (never inferred from the input array), with a locked
SHA-256 content fingerprint, a caller-owned `NativeTensor` feature batch
and a read-only host `int64` target batch per index sequence, exact order
and duplicate preservation, and **no native storage held between calls**.
It added exactly one public experimental name and nothing else — no C++,
no C ABI symbol, no example, no benchmark, and no schema change. **J2**
added exactly one more, `NativeBatchSampler`: the deterministic order and
batch **planner**, which owns `batch_size`, `drop_last`, `shuffle`, the
`seed`, the `epoch`, and the `cursor` and emits batch-index groups through
`epoch_permutation()`, `plan()`, and `next_batch_indices()`. Every
permutation is a **pure function** of `(seed, epoch, length)`, derived by
the permanently private `_native_permutation` helper from the locked
`tensorforge.splitmix64` finalizer under one domain-separated epoch key
schedule, with unbiased rejection-based bounded integers and a downward
Fisher–Yates sweep in explicit 64-bit-masked Python integer arithmetic —
so it is bit-identical on every platform by construction, and every
committed reference vector is reproduced exactly and re-checked live
against the compiled Dropout kernel. It holds no consumable stream,
allocates nothing native, materializes no batch, and owns nothing
releasable, so it has no `close()`; its compact JSON-compatible state
carries the configuration, the position, and the dataset's four identity
fields and loads transactionally. **J3** added the last of the three,
`NativeDataLoader`: `iter(loader)` returns a private one-epoch iterator
that captures the sampler's remaining batch count and supersedes any
previous one, and every `__next__` runs an explicit five-phase
transaction — claim, construct, publish, commit-and-deliver, rollback —
under one invariant, that **the committed sampler position advances if
and only if a batch was successfully delivered to the caller**. Every
failure position closes the undelivered feature tensor, restores the
exact pre-delivery epoch and cursor through the same non-failing write
seam a state load uses, and leaves a retry returning the same indices and
the same values; each is proved by injection with its own non-vacuity
control and a native live-storage baseline. Delivered batches are the
caller's and no close path can reach one. **J4** added **no public name
at all** — the first Phase-J runtime milestone whose export delta is
zero, leaving `tensorforge.experimental.__all__` at 25 — and gave
`NativeDataLoader` exactly two methods. `state_dict()` returns a compact
tagged wrapper of **three** root keys (`format`
`"tensorforge.native_data_loader"`, `format_version` **1**, and
`sampler`) around the **unchanged** sampler state, with every container
fresh at every call, no field duplicated at the root, no permutation and
no dataset content inside, and nothing whose size grows with the number
of samples; it is JSON-compatible and accepted unchanged by the
checkpoint's existing metadata validator, is allowed between batches,
after exhaustion or supersession, with a closed dataset, and after the
loader is closed, and is **refused** while a batch transaction is in
flight because that window has no honest answer.
`load_state_dict(state)` runs a closed guard, a transaction guard, and an
active-iteration guard **before** the state is read, validates the
wrapper, **delegates** the whole nested sampler validation to the seam
that already owns it, and commits through the same non-failing write seam
the delivery uses — so a rejected load leaves the loader, sampler,
dataset, position, cache behavior, iterator slot, and native live storage
byte-identical, and a successful one adopts all six configuration and
position values while validating dataset identity without adopting it and
preserving every object identity. **Exact in-memory mid-epoch
restoration** is proved over two separate object graphs: a mid-epoch
interruption restored into a separately constructed dataset, sampler, and
loader reproduces the remaining batches exactly — identical indices,
identical raw IEEE-754 feature bits, identical `int64` targets — then the
same canonical next-epoch position and the same following epochs, at both
dtypes and with no tolerance anywhere. **J5** proved the caller-managed
checkpoint-metadata workflow end to end and **added no production code at
all** — the second consecutive Phase-J milestone with a zero export
delta, and the only one whose diff touches no file under `src/`:
`native_checkpoint.py` is unchanged. Against **real** version-3 `.npz`
archives read with pickle disabled, the format stays
`tensorforge.native_checkpoint` version **3** with `(1, 2, 3)` accepted,
the manifest keeps its same six root keys, and the array inventory is
identical whether or not loader state travels, so **the archive's own
capture set did not grow by one field**; loader state lives only inside
caller metadata, with no root field, no loader array, no serialized
permutation, and no dataset payload. `"training"`, `"data_loader"`, and
`"next_step"` are **caller conventions no production constant spells**, so
alternate nesting, alternate names, and two loaders' states side by side
all round-trip unchanged. Restoration into an entirely fresh model,
optimizer, generator set, dataset, sampler, and loader — each deliberately
built wrong first — reproduces every parameter, persistent buffer, Adam
moment and step counter, hyperparameter, generator state and **alias
topology**, and all six loader values exactly in raw IEEE-754 bit
patterns, then the exact next batch and the exact remaining sequence. All
three delivery boundaries are proved through an archive: a **failed**
delivery resumes the same candidate batch, a **successful** one resumes
the following batch, and an epoch-boundary save resumes the canonical next
epoch. The absence of cross-object atomicity is proved rather than glossed
— a checkpoint load that succeeds followed by a loader load that fails
leaves the first restored and the second untouched.

**J6** shipped `examples/native_minibatch_training.py`, the deterministic
mini-batch training example, and **added no production code and no public
name** — the third consecutive Phase-J milestone with a zero export delta.
The example inventory moved 15 → **16**; the benchmarks stayed at **8**.
It trains a `Linear → BatchNorm1d → ReLU → Dropout → Linear → LayerNorm →
Dropout → Linear` classifier over **shuffled** mini-batches with
`NativeAdam` and `NativeCrossEntropyLoss`, two Dropout layers sharing one
generator, and proves an interrupted-and-resumed run **bit-for-bit
identical** to an uninterrupted one across the whole §14.3 inventory — the
complete batch-index sequence, every feature batch's raw bits, every
`int64` target array with its flags, every loss, parameter, buffer, Adam
moment and counter, the generator state and alias topology, the final
loader `state_dict()`, and the evaluation output. It runs at float32 and
float64 **independently**, each compared only against itself; the one
cross-dtype claim is the batch-index sequence, which carries no dtype. The
interruption is genuinely mid-epoch, the resumed graph is entirely fresh
and deliberately built wrong in every family first, live storage returns
exactly to baseline, and a negative control that omits
`loader.load_state_dict` alone is proved to diverge. The example uses
**only public APIs**, asserted by an AST scan with its own negative
control, and claims and measures **no timing**.

**J7** shipped `tests/test_native_data_hardening.py`, the cross-cutting
adversarial hardening matrix, and **added no production code and no public
name** — the fourth consecutive Phase-J milestone with a zero export
delta, and one that **found no production defect**. Examples stayed at
**16** and benchmarks at **8** through J7. It asserts every §12.7, §15,
§16, and §17
row by injection: each dataset-construction row, proved to leave **no
reference alive** by reading the raised exception's own traceback; each
iteration row, with the host gather, the native allocation, the
host→native transfer, and the target copy kept as **four distinct
injections** so no failure is labelled as another; the **commit step made
to fail after the candidate position was really applied**, which
exercises the restore path with a position that genuinely moved; a
`BaseException` through the same unconditional `finally`; the reentrancy
refusal matrix at all **three** transaction phases; every abandonment
position and every close ordering; and a **checkpoint taken immediately
after a failed delivery, proved to resume the same candidate batch** at
both dtypes through a real version-3 archive into an entirely fresh
graph. Every rejection is followed by a complete before/after fingerprint
of the observable world — dataset, sampler, loader, iterator, an
unrelated `NativeParameter` with its version and gradient, a persistent
buffer, a live optimizer, a registered `NativeGenerator`, the filesystem,
both global RNGs, and every registry — and every injection and every
parser carries its own non-vacuity control. **Concurrency remains
documented as unsupported rather than tested as safe**: no lock exists in
any Phase-J module, none was added, no test starts a thread, and external
locking stays the caller's job.

**J8** shipped `benchmarks/benchmark_native_data_pipeline.py`, the
data-pipeline characterization, and **added no production code, no public
name, and no optimization** — the fifth consecutive Phase-J milestone
with a zero export delta. Examples stayed at **16** and benchmarks moved
8 → **9**. It isolates the four layers J8 undertook to answer for —
immutable host dataset indexing, deterministic batch planning,
deterministic shuffled-permutation construction, and host→native batch
materialization — and adds one clearly separate composition case for a
whole `next(iterator)` delivery, so a single end-to-end number can never
stand in for the layer that dominates. float32 and float64 are measured
**separately and never as a ratio of one to the other**; every gate is
exact (index tuples, plans, and permutations by equality; feature values
in raw IEEE-754 bits within one dtype; targets by exact `int64`
equality) and runs **before** the timing helper is reached; the length-8
configurations are known-answer checks against the design's committed
reference vectors; a case with no honest equivalent is labelled
`native_only` and publishes **no ratio at all**; cold and warm
permutation construction are separate cases and are never averaged, with
the warm case proved a genuine cache hit through the public surface
alone; medians come with an interquartile range after warm-up and every
raw sample is retained; and setup, per-repetition state reset, and every
`close()` stay outside the timer. **No threshold, no CI timing job, and
no result file** exists, and no runtime change is derived from any
measurement.

**J9 closed the phase**, adding no production code, no public name, and
no export: it shipped the permanent closure guardrails in
`tests/test_native_phase_j_closure.py`, re-ran the complete validation
matrix — Windows Release and Debug, a Linux CI-equivalent, Clang
ASan/UBSan with a detector negative control, and a LeakSanitizer lifecycle
over the whole pipeline — and reconciled every inventory. **Phase J is
complete and no milestone remains.** That sentence continued "no successor
phase is defined" for as long as it was true — Phase J closed without one,
deliberately — and **Phase K was approved afterwards**, which is recorded
here rather than rewritten away.

**No automatic loader discovery exists**, at any milestone, and none may
be added. Its architecture contract is
[native_data_pipeline_design.md](native_data_pipeline_design.md), which
locks three eventual public names (`NativeTensorDataset`,
`NativeBatchSampler`, `NativeDataLoader`), a copied-snapshot dataset whose
native feature dtype is explicitly chosen and never inferred from the input
array, a SHA-256 dataset fingerprint, a deterministic shuffle that
**reuses the locked `tensorforge.splitmix64` derivation** rather than
introducing a second RNG algorithm or coupling to a live
`NativeGenerator`, a permutation that is a pure function of
`(seed, epoch, length)` with committed reference vectors, strict
JSON-compatible state schemas carrying no payload, transactional state
loading, an explicit **caller-managed** checkpoint-metadata workflow over
the unchanged version-3 format, and an exact resume contract compared in
raw IEEE-754 bit patterns. Phase J moves no registry, no dtype, no device,
no checkpoint or optimizer-state version, and no C ABI symbol at any
milestone.

**Phase I — native dtype generalization and float32 CPU support — is
complete (I0–I11).** It was the latest completed phase until Phase J
closed after it. I11
revalidated the whole dtype-general stack across every required platform,
added the closure guardrails, and reconciled the status surfaces. Its
architecture contract is
[native_dtype_float32_design.md](native_dtype_float32_design.md).

**I0 was design and reconciliation only, and added no runtime behavior**:
the contract, its guardrail tests, and documentation.

**I1 built the dtype foundation.** The C++ dtype model now exists — frozen
ABI codes (`0 = float64`, `1 = float32`), one item-size authority, one
canonical-name authority, and a total conversion that rejects every
unknown code without producing a dtype. Native storage is dtype-tagged:
one untyped owned buffer, a logical element count whose meaning did not
move, and one dtype tag, owning a genuine runtime-selected `float[]` or
`double[]` array created with checked `numel × itemsize` and type-erased
into `void*` only after creation — so the kernels' pointer arithmetic is
valid C++17 over one array object — with the immutable dtype tag selecting
both the typed accessor and the matching central `delete[]`. The two typed creators are exported,
taking the library from 52 to **54** `tf_*` symbols — the count for the
whole phase — while the untyped pair remains unchanged as thin float64
compatibility wrappers over the same shared body. Native CTests moved
**17 → 18**.

**I2 made that foundation movable.** The three exports that carry a
storage handle *and* a raw host buffer — `tf_storage_copy_from`,
`tf_storage_copy_to`, `tf_storage_materialize` — are dtype-general, through
a **source-level retype** of their host positions from `double*` to
`void*`: same symbols, same argument slots, same calling convention, no new
export, and a previously compiled caller would link and run identically.
`tf_core_contiguous_copy` — the runtime's value-transfer primitive, and the
one compute-shaped export I2 touched — is dtype-preserving and
dtype-strict, so a float32 view of any layout materializes or copies
storage-to-storage while a mixed float32/float64 pair is rejected before
anything is written. Transfer is **bit-preserving at both widths**: signed
zeros, both infinities, subnormals, quiet NaN payloads, and signalling
NaNs all survive, proved over seventeen IEEE-754 classes per dtype as raw
`uint32`/`uint64` patterns rather than by value. `RAW_KERNEL_DTYPES`
records the other half of the ABI's division — the seven handle-free raw
utility kernels take only `double*` and an element count, so they have no
dtype to dispatch on and stay float64. Native CTests moved **18 → 19**;
exports stayed at **54**.

**I3 made that movable foundation computable, and added no export.** The
elementwise and unary Core family — `add`, `subtract`, `multiply`, `relu`,
`relu_backward`, `sqrt`, `reciprocal`, `exp`, and `log`, across their
strided and contiguous forms — validates that its operands agree,
dispatches **once** from the storage tag, and runs one instantiation of a
templated kernel, with nothing below that point branching on dtype. All
three Phase-H traversal tiers are instantiated for both element types from
the same source, so float64 runs the code Phase H measured; NumPy-style
broadcasting works at float32 for every layout it already worked at for
float64; outputs preserve the operand dtype; and mixed dtype is rejected in
the left, right, and destination positions independently, before any
allocation, at the Python layer and again in C++. float32 arithmetic is
genuinely binary32 — bit-identical to the binary32 oracle for the
IEEE-specified operations, within a measured ULP bound for `exp` and `log`
— with no widening intermediate anywhere, asserted structurally as well as
numerically. Native CTests moved **19 → 20**; exports stayed at **54**.

**I4 generalized the accumulating families and the graph built on them, and
added no export.** `sum`, `mean`, `matmul`, and `narrow_backward` dispatch
once from the storage tag into templated kernels; H6's contiguous-block
factorization, H2's `i`-`k`-`j` row sweep, and the retained generic odometer
and triple loop beside them are all instantiated for both element types from
the same source, and both metadata predicates are untouched — so the two
widths take the same path for the same layout and every optimized path keeps
its oracle per dtype. `tf_storage_scale` and `tf_storage_fill` became
dtype-general with their `(handle, double)` signatures unchanged, narrowing
the scalar **once, before the loop**, which is what makes `mean`'s `1/count`
platform-independent. Private/internal float32 `NativeTensor` graphs now run
forward and backward through every Core operation landed so far, with every
gradient, temporary, and materialized constant at the graph's dtype and
mixed-dtype accumulation refused before any allocation.

**This is the milestone where "float32 accumulates in float32" stopped being
a structural claim and became a measured one.** I3's operations each produced
their result with a single correctly-rounded IEEE operation, for which
binary64-then-round-once is *provably* indistinguishable from binary32 — so
no runtime test could separate the two, and I3 recorded that rather than
inventing one. A sum can: on `1.0` followed by eight copies of `2**-24`,
sequential binary32 stays at exactly `1.0` while binary64-then-narrow lands
four ULPs higher, and TensorForge is asserted equal to the first and unequal
to the second, on both reduction traversals and both matmul paths. Native
CTests moved **20 → 21**; exports stayed at **54**.

**I5 generalized the CNN stack, and added no export.** All three Conv2d
directions and both MaxPool2d directions dispatch once from the storage tag
into templated kernels; H9's row-sweep and gather traversals and the
retained Phase-D generic loops beside them are instantiated for both
element types from the same source, with the geometry predicates untouched,
so the two widths take the same path for the same geometry and every
optimized path keeps its oracle per dtype. Conv2d accumulates in the
element type — the binary32-versus-widened witness is proved in all three
directions, on both traversals of each. MaxPool2d's value path follows the
input dtype through the identical comparison sequence at both widths, while
the private **winner buffer stays float64 at every value dtype**, keeping
the `2**53` exact winner-plane bound instead of shrinking it to float32's
`2**24` — a float32 pool over a plane beyond `2**24` still records its
winner offsets exactly. Private float32 graphs differentiate through
convolution and pooling, with the float64 winner riding the unchanged
graph-owned saved-state contract. Native CTests moved **21 → 22**; exports
stayed at **54**.

**I6 generalized the stable-math and classification stack, and added no
export.** Softmax, log-softmax, and the fused cross-entropy forward and
backward dispatch once from the storage tag into templated kernels, with
every participating numeric handle checked for agreement first — two for
each transform, three for each cross-entropy direction. Everything
numerical happens at the element type: the maximum scan, the shift,
`std::exp`/`std::log` on the element type, the normalizing sum, the
log-normalizer, the per-row loss, the batch-loss accumulator, the mean
divisor, and every backward contribution. The batch-loss accumulator is
where that became a measured claim rather than a structural one — on a
batch whose first row contributes exactly 200 and whose remaining 199
contribute ~6.1e-6 each, binary32 stays at exactly 200 while
binary64-then-narrow lands ~1.2e-3 higher, and TensorForge is asserted
equal to the first and unequal to the second. Log-softmax is still its own
fused log-sum-exp kernel and never `softmax().log()`; the saved
probabilities carry the graph dtype and remain the only thing the backward
reads; and the class **targets stay host `int64` metadata at every width**,
so no integer tensor dtype was introduced. Private float32 graphs
differentiate through all three operations with no change to the graph
structure at all. Native CTests moved **22 → 23**; exports stayed at
**54**.

I6 is also where the float32 stability statement picked up its one honest
qualification, measured rather than assumed. The maximum shift guarantees
no *exponent* overflows; it does not make the shifted value itself
representable, so a slice whose **spread** exceeds the element type's
largest finite value overflows the shift to `-inf`. `softmax` is unaffected
and still exact; `log_softmax` reports `-inf` and `cross_entropy` `+inf`,
as values with the error slot at `TF_OK`. Those are the correctly rounded
IEEE results for quantities with no representation at that width, and the
identical thing happens at binary64 past ~1.8e308 — a dynamic-range fact,
not a float32 defect. The kernels were left alone; the counterexample is
recorded in the contract and asserted in both directions by test.

**I7 made float32 a module dtype, and added no export.** Six state-owning
constructors — `NativeParameter`, `NativeLinear`, `NativeConv2d`,
`NativeLayerNorm`, `NativeBatchNorm1d`, and `NativeBatchNorm2d` — gained a
keyword-only `dtype` accepting exactly the two widths and defaulting to
float64, all six routing through one shared private validator so no
constructor invents a dtype rule of its own. Affine parameters, both
BatchNorm running buffers, the graph-safe evaluation snapshots, and every
scalar a composed normalization forward materializes — `eps`, `momentum`,
`1 - momentum` — are built at the module's dtype through two new private
tensor constructors, because a literal float64 constant meeting a float32
operand would be a mixed-dtype request the runtime refuses. The atomic
two-buffer running-statistics transaction gained **one** dtype validation
and nothing else, and the BatchNorm forward re-proves that all four numeric
state objects still carry the module's dtype before either buffer can move.

**Initialization did not move.** The host draw is the same local
`numpy.random.default_rng(seed)` stream, in the same order, at the same
sizes, with the fan-in bound computed once in binary64 — so a float32 layer
with seed *S* holds exactly `float32(the float64 draw with seed S)`,
asserted as raw bit patterns. The seed contract therefore stays
dtype-independent, and changing one layer's dtype provably shifts no other
layer's initialization.

Dropout was the last dtype-general family, and its **randomness is
untouched**: the uniform is still the binary64 53-bit conversion at every
width, so one `(seed, call_index, element count)` key drops exactly the
same elements at both dtypes — proved at float32 against the *same*
committed Phase-G keep vectors rather than a second table. Only the two
multiplier values differ, and the kept one is the binary64 reciprocal
narrowed once, witnessed at a probability where that provably differs from
recomputing it in binary32. The generator's algorithm, version, state, and
reserve → commit/abandon call accounting are unchanged at both widths.
With Dropout, the last of the five explicit float64-only Python gates came
out. Native CTests moved **23 → 24**; exports stayed at **54**.

**No public capability moved at I1 through I8, deliberately.** Through I7
the native runtime was still declared float64 CPU only, and a float32 model
was *refused* by a version-2 save rather than written into an archive the
loader would reject, which would have been a silent, unrecoverable
checkpoint. Versions 1 and 2 remain float64-only formats permanently.

**I8 made float32 survive a step and a file, and added no export either.**
Both `NativeSGD` and `NativeAdam` execute at float32 — Adam's `m` and `v`
carry their parameter's dtype, one optimizer may hold parameters of both
widths with independent dtype-consistent state per parameter, and neither
gained a `dtype` or `device` argument, because neither owns a dtype it
could choose. No C++ changed: I3-I7 had already generalized every
operation the optimizers compose, so three constructors moving to their
private typed twins was the whole runtime change, and Phase H's
once-per-step scalar architecture is preserved whole. Design §15.3's open
exactness question was **resolved on measurement**: H4's Python
bias-correction reciprocal is an exact substitution at binary64 but not at
binary32, because the kernel divides by the *narrowed* denominator — the
two spellings differ by one ULP for a large fraction of inputs, the
default betas included — so the denominator is narrowed first, at no
allocation and no kernel call, with float64 bit-identical to before.
Native checkpoint **version 3** declares every numeric entry's dtype
explicitly, accepts versions `(1, 2, 3)`, writes 3 on every new save, and
carries Adam's moments as entry objects rather than bare archive names so
their metadata is stated rather than inferred positionally. float32 model
values, persistent buffers, and Adam moments round-trip **bit for bit**;
a dtype disagreement is rejected in either direction with no cast, no
`map_location`, and no device movement; and versions 1 and 2 remain
float64-only formats permanently that never guess a payload to be float32.
Every transactional, identity, aliasing, and rollback guarantee is
unchanged, and the in-memory optimizer state schema stayed at version 1.

**I9 made float32 public, and it is the phase's one and only public
registry change.** `SUPPORTED_DTYPES` became `("float64", "float32")` and
`UNSUPPORTED` became `("cuda", "amp")`; `SUPPORTED_DEVICES`,
`RAW_KERNEL_DTYPES`, the export count, the checkpoint version, and the
in-memory optimizer state version all stayed put. `normalize_dtype`
accepts both widths, and every public constructor — `NativeStorage`,
`NativeTensorCore.from_array`/`.zeros`/`.full`, and
`NativeTensor.from_array`/`.zeros`/`.full` — builds a float32 tensor when
asked for one explicitly, with views, operations, and gradients preserving
the dtype and `to_numpy()` never widening on the way out.

**The registry moved after the proof, not before**, which is the ordering
the rollout rule requires: the integrated example and its exact-resume
proof were written and passing first, through the already-approved private
typed route, with the registry still reading `("float64",)`; only then did
it move; then the example's one ingress helper switched to the public
constructor and the whole proof was rerun.
`examples/native_float32_training.py` is that proof — `Conv2d →
BatchNorm2d → ReLU → MaxPool2d → Dropout → Flatten → Linear → BatchNorm1d
→ ReLU → LayerNorm → Dropout → Linear` into `NativeCrossEntropyLoss` with
`NativeAdam`, **two Dropout layers sharing one registered generator** so
the model carries a real alias topology — run interrupted and uninterrupted
at each dtype and compared **only against itself** in raw IEEE-754 bit
patterns. Losses, the first resumed step's produced gradients, parameters,
buffers, Adam moments and counters, generator state, alias topology, the
next Dropout mask, final logits, predictions, and evaluation output all
match exactly; live native storage returns to baseline; and a float32 run
is never required to reproduce a float64 one. I9 changed no C++, added no
export (still 54) and no CTest (still 24), and moved no checkpoint field or
version. Phase H is untouched, remains complete, and closed at 52 exports.

**I10 hardened and characterized what I9 published, and found one real
defect while doing it.** The saver validated checkpoint metadata
recursively; the loader checked only that its root was a dict — and
because `json.loads` accepts the non-standard `NaN`/`Infinity`/`-Infinity`
literals, a hand-written archive could return a value no save could have
written. The **same** validator now runs on both sides, during archive
prevalidation, before anything is staged or mutated. That is I10's **only**
production change: no C++, no ABI or export change, no numerical runtime
change, no benchmark-path change, and no checkpoint schema, version, or
manifest field moved, so float64 and float32 numerical behavior and every
Phase-H path are unchanged. Everything else it added is evidence: the §9.2
mixed-dtype authority map exercised at every layer and
**every operand position independently**; the C ABI proved to be a second
authority rather than a restatement of Python's, by forcing a mismatch
production Python cannot emit and separately by neutering the Python guard
— each with its own negative control; the established validation orderings
recorded rather than chosen; allocation and wrapper-failure cleanup at both
widths; **all four graph-owned saved-resource families made to coexist in
one float32 graph** — BatchNorm in eval, Dropout put back into training
through the public per-module API — and driven through retained, failed,
one-shot, abandoned, and no-grad lifecycles; a **117-case**
malformed-checkpoint matrix at both dtypes — the metadata cases re-run at
v1, v2, and v3 separately — with a complete-world fingerprint after every
rejection; the concurrency contracts re-proved at exactly the width they
are claimed; and a new benchmark harness that characterizes both dtypes
**separately**, deliberately kept out of Phase H's instrument so that every
number Phase H published keeps its meaning. One finding was recorded rather
than "fixed", because it is an absence rather than a defect:
`maxpool2d_backward` has exactly one value operand, so there is no second
value position for a mixed-dtype rule to govern. Suite 7,409 → **7,629**.

**I11 closed the phase.** It changed no file under `src/` or `cpp/` and
added no capability: it revalidated Windows Release and Debug, a Linux
CI-equivalent, and Clang ASan/UBSan and LeakSanitizer builds, re-ran both
exact-resume proofs, reconciled the ABI, registry, checkpoint, CTest,
example, and benchmark inventories, and added
`tests/test_native_phase_i_closure.py` — the durable guardrail module that
keeps the closure from drifting.

The contract locked the phase before any of it was built, and the first
three items below are the ones I1 delivered: an internal dtype
model with frozen ABI codes and one central item-size authority;
dtype-tagged storage whose dtype is the single authority for every view
over it, with shapes, strides, and offsets still measured in logical
elements and checked `numel × itemsize` arithmetic at the one allocation
boundary; **exactly two** new C ABI symbols for the entire phase
(`tf_storage_create_typed`, `tf_storage_create_uninitialized_typed`,
52 → **54**), with per-operation float32 exports **rejected** because
handle-based exports already carry their operands as opaque handles and
one narrow dispatch per call is enough; templated `float`/`double`
kernels with no dtype branching below that dispatch; **no casting, no
promotion, and no mixed-dtype arithmetic**, rejected before any
allocation or mutation; **float32 accumulating in float32**, with no
hidden wider accumulator anywhere, because that would be mixed precision
and mixed precision is out of scope; the autograd, module, buffer,
RNG/Dropout, and optimizer-state dtype invariants, over an **unchanged**
generator algorithm; a dtype-aware checkpoint **version 3**, designed but
not activated, with versions 1 and 2 defined as float64-only formats that
are never guessed to be float32; exact deterministic resume proved
**separately** for float32 and float64 and never as agreement between
them; every Phase-H float64 optimization preserved and each dtype
benchmarked on its own; and the I0–I11 ladder, in which the public
support registry changes at **I9** and at no earlier milestone.

**Phase K — Native Integer Tensors and Indexing — is the newly approved
successor, and only K0 through K6 have landed.** K0 is architecture, contract, status,
and guardrails, and it **added no runtime behavior at all**: no integer
dtype or dtype code, no C++ enumerator, no kernel, no C ABI symbol, no
public export, no capability-registry movement, no checkpoint or state
version change, no example, no benchmark, and no CTest.

**K1 added the internal `int64` representation and every reachability
barrier, and no public capability at all.** The C++ dtype model gained a
third enumerator at code 2; storage allocates and destroys genuine
`std::int64_t[]` buffers; the four transfer boundaries move integer values
bit for bit at the signed extremes and beyond 2⁵³; and the 32 float-only
exports gained the hidden-visibility `tf::require_floating` guard, applied
ahead of the operand-agreement guard so a mixed float/integer call is
refused as a role error. On the Python side, nine trusted dtype paths were
narrowed to the floating registry and every barrier landed — wrapper
construction, autograd, parameters, buffers at both `persistent` values,
both optimizers, checkpoint entry validation, and every floating
operation. It added no C ABI symbol, no public Python name, and no
registry or version movement; the native CTest inventory went 24 → **25**.

**K2 made the `int64` tensor publicly constructible, atomically, and moved
no other capability.** The three Python dtype tables and the checked host
binding learned `"int64"` (code 2, 8 bytes, `numpy.int64`, reusing the
existing `int64` ndpointer object); `INDEX_DTYPES == ("int64",)` appeared
beside an **unmoved** `SUPPORTED_DTYPES` and is reported as
`backend_info()["index_dtypes"]`; the Phase-I no-drift guard was
generalized to `set(_DTYPE_CODES) == set(SUPPORTED_DTYPES) |
set(INDEX_DTYPES)` rather than deleted; the private exact ingress
`NativeStorage._from_int64_array` / `NativeTensorCore._from_int64_array`
arrived; and exactly two gates widened — `NativeTensorCore.__init__` and
`NativeTensor.__init__`, from "floating" to "floating **or** index".
**`NativeTensor.from_int64_array` is the one public API in the repository
through which an `int64` buffer can come into existence**, beside the
dtype-general `item()` and `tolist()`; it converts nothing, so a float
array, an `int32` array, a `uint64` array, a `bool` array, an `object`
array, a byte-swapped `>i8` array, a list, and a scalar are all rejected,
while a non-contiguous exact-`int64` array is copied because layout
normalization is not conversion. Views, copies, and exact host inspection
work at `int64` through the machinery that already existed. K2 added no C
ABI symbol (still 54), no experimental export (still 25), no CTest, no
example, no benchmark, and no version change.

**K3 shipped native `argmax` and K4 shipped native `index_select`, forward
only — the phase's two operations and its two C ABI symbols.** `argmax`
takes a **floating** tensor at either dtype and any rank and returns a
fresh owning contiguous **`int64`** index tensor, over `tf_core_argmax`,
with an exact value rule (lowest index on a tie, signed zeros tying, a
first-NaN rule) and no `max` beside it. `index_select` is its mirror image:
a **floating** source and a rank-1 **`int64`** index tensor in, a fresh
owning contiguous tensor of the **source's** dtype out, over
`tf_core_index_select`, preserving duplicates and order exactly, rejecting
negative and out-of-range indices rather than wrapping them, scanning every
index completely in Python *and* independently in C++ before writing
anything, and copying values by **object representation** so signed zeros,
infinities, subnormals, and NaN payloads survive bit for bit. It is
**forward only**: a source with `requires_grad=True` is rejected with a
message naming `detach()`, never silently detached. Neither operation is in
`AUTOGRAD_OPS`. Exports went 54 → **55** → **56**, the phase maximum, and
native CTests 25 → **26** → **27**; no registry, version, example, or
benchmark moved at either milestone.

**K5 is the compatibility proof, and it added zero production code** —
one new module, `tests/test_native_integer_compatibility.py`, plus the
status reconciliation a landed milestone requires. It proves against the
live tree that no checkpoint archive can declare an `int64` entry at a
parameter, persistent-buffer, optimizer-moment, or optimizer-parameter
entry, at any accepted version, and that such a load rejects **before**
publishing anything and without allocating an `int64` storage; that the
checkpoint format and version (`tensorforge.native_checkpoint`, **3**,
accepting `(1, 2, 3)`, with no version-4 constant written, reserved, or
accepted), the in-memory optimizer-state version (**1**), and the loader
and sampler state versions (**1**, accepting `(1,)`) are exactly what
Phase J left; that a version-1 archive still loads under its legacy rules
and both historical versions stay float64-only; that parameters, buffers
at both persistence values, and both optimizers still refuse a real
`int64` tensor and that a standalone index tensor beside a model is a
plain attribute rather than state; that Phase J still delivers a floating
`NativeTensor` feature batch and a read-only host `numpy.ndarray` target
batch of dtype `int64` at both widths, with no option anywhere able
to request a native label; that explicit caller conversion through
`NativeTensor.from_int64_array` works on a delivered batch and needs no
pipeline change; that `NativeCrossEntropyLoss` accepts and refuses exactly
what it did, with every accepted target form giving a bit-identical loss
and a native `int64` target refused by three routes; that
`native_accuracy` still succeeds with the native `argmax` and
`index_select` patched to raise, which is only possible if it calls
neither; and that a real classifier trains, checkpoints, and resumes
**bit-identically** at float64 and float32 while `argmax` and a detached
`index_select` run beside the training path, with an observational
control proving the indexing changes no trainable state. Exports stayed
**56**, CTests **27**, examples **16**, benchmarks **9**, and
`experimental.__all__` **25**.

**K6 is the end-to-end integration example, and it added zero production
code too** — `examples/native_integer_indexing.py` with its owner
`tests/test_native_integer_indexing_example.py`. A deterministic
`NativeLinear(5 → 8) → NativeReLU → NativeLinear(8 → 4)` classifier with
`NativeCrossEntropyLoss` and `NativeAdam` trains ten shuffled batches of
six over the Phase-J pipeline, is interrupted **strictly mid-epoch** with
three batches still owed by the active epoch, and resumes through a real
version-3 archive — loader state as ordinary caller metadata — into an
entirely fresh object graph proved different before the load. At four fixed
steps, two on each side of the interruption, the step's own logits become
native `int64` predictions through `argmax` and are then consumed by
`index_select` over a **detached** copy of those logits; the two sources
differ deliberately, because `argmax` returns a plain leaf even from a
gradient-tracking input while `index_select` **rejects** one with a message
naming `detach()`. The call is **axis selection, not a per-row gather**: a
`(6, 4)` logits batch and a `(6,)` index vector give a `(6, 6)` result
whose column *j* is the whole source column `predictions[j]` and whose
**diagonal** is each example's own predicted-class logit — both recomputed
from the recorded bit patterns by the owner test rather than read out of the
example's own booleans, with duplicate predicted classes guaranteed by
pigeonhole and proved to give identical columns in their original
positions. The uninterrupted and resumed runs agree exactly at float64 and
float32 **independently**: every prediction index by exact integer equality,
every floating value by raw IEEE-754 bits, never a tolerance and never
across widths, with the omitted-loader-state leg proved to diverge. Cross
entropy still trains on the loader's read-only host `int64` target arrays,
and no native integer tensor is ever a target, a parameter, a buffer,
optimizer state, or a checkpoint entry. The example is written entirely
against the public experimental surface (proved by an AST scan with a
planted negative control), calls no `numpy.argmax`, claims and measures no
timing, leaves no file behind, and returns live native storage exactly to
its baseline. **Examples went 16 → 17**; exports stayed **56**, CTests
**27**, benchmarks **9**, and `experimental.__all__` **25**.

**The proof found one real defect, and the chronology is part of the record.** Driving the two module registration routes with a deliberately forged `NativeParameter` — the only way to reach them, because the public constructor refuses an `int64` tensor — showed that `save_native_checkpoint` trusted whatever dtype live state reported, so the **writer** could emit an archive declaring an `int64` entry that its own loader then refused. That was a pre-existing gap in the writer, not something Phase K introduced and not reachable through any public API, and it was repaired in a **separate checkpoint-hardening change committed before K5**: a save-side persisted-dtype authority asking the same `cpp.normalize_dtype` question the loader asks, applied in `_validate_model`'s preflight and again at `_coherent_snapshot`'s serialization seam, with its own regression in `tests/test_native_checkpoint.py`. No format, field, version, capability, registry, export, CTest, example, or benchmark moved; the forged parameter is test-only and never supported public usage; and K5 itself remains the test-and-documentation compatibility milestone.

`int64` is **still not**
a supported native tensor dtype — it is an index/result dtype in its own
registry, `normalize_dtype("int64")` keeps raising, and **no generic
constructor changed what it accepts**. Every K1 barrier holds against a
real integer tensor, and no integer
arithmetic, reduction, autograd, parameter, buffer, optimizer state, or
checkpoint entry exists, nor any `max`, `argmin`, general `gather`,
`scatter`, embedding lookup, or `index_select` backward;
**K7 through K9 are unstarted**. Its contract is
[native_integer_tensors_design.md](native_integer_tensors_design.md),
which locks one extended `NativeTensor` rather than a parallel integer
class, `int64` as the only integer dtype and an exact non-differentiable
index/result one, a strict `numpy.ndarray`-only construction door with no
dtype inference and no numeric cast, integer autograd / parameter /
optimizer / buffer / checkpoint barriers enforced in Python **and**
independently at the C ABI, complete `argmax` and forward-only
`index_select` contracts, the Phase-J loader default left exactly as it
is, **no checkpoint version change**, and a C ABI maximum of **56**. The
**`SUPPORTED_DTYPES` never gains `int64`**: it remains the floating-compute registry permanently and `normalize_dtype("int64")` keeps raising, so **no generic constructor changes what it accepts at any milestone**. The one public registry movement of the phase is a separate `INDEX_DTYPES == ("int64",)` row, and it appeared at **K2**, in the same commit as the public constructor and one milestone after every reachability barrier had landed at **K1**.

Beyond Phase K (**not started**, and nothing approved): further dtypes or
devices, the CUDA runtime, AMP work, and Transformer/text and distributed
experiments. See
[roadmap.md](roadmap.md) and
[release_history.md](release_history.md) for the full arc.
