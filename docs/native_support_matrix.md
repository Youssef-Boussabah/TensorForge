# Native support matrix

The canonical, authoritative statement of what the **experimental
native C++ CPU line** supports today — **Phase E (native classification
and stable math) is complete**, closing the
Phase A (native CPU runtime) → Phase B (native autograd) → Phase C
(native training stack) → Phase D (native CNN stack) → Phase E arc in
code. **Phase F (native normalization and stateful buffers) is also
complete (F0–F9)**: the numerical normalization *module* surface is
complete — `NativeLayerNorm` (F2), `NativeBatchNorm1d` (F3), and
`NativeBatchNorm2d` (F4) are all supported — **F5 proved the
state/checkpoint/ownership/graph-safety contracts by exhaustive test**,
**F6 shipped a deterministic normalized training example with exact
checkpoint resume** (`examples/native_normalization_training.py` — tests
and documentation only, no new capability), **F7 shipped the honest
benchmark characterization**
(`benchmarks/benchmark_native_normalization.py` — measurement only, no
new capability, no committed timing number, and no speed assertion),
**F8 shipped the cross-cutting integration and semantic guardrails**
(`tests/test_native_phase_f.py` — tests and documentation only, no new
capability), and **F9 closed the phase** — Release and Debug builds each
passing the full existing CTest suite, Clang ASan/UBSan and
LeakSanitizer finding nothing attributable to TensorForge, and the
documentation reconciled — **validation and documentation only, adding
no numerical capability**. **Phase G (native RNG and Dropout) is
complete — the latest *completed* phase is now Phase H, native CPU
performance, recorded in the ladder below: milestone G0, the
architecture
contract in [native_rng_dropout_design.md](native_rng_dropout_design.md),
milestone G1, `NativeGenerator` and module generator-state
ownership, milestone G2, the stateless Dropout-forward **Core**,
milestone G3, the differentiable `NativeTensor.dropout(p, *, generator)`,
milestone G4, the `NativeDropout` module, milestone G5, native
checkpoint **format version 2** with persisted generator state and its
alias topology, milestone G6, the RNG/graph/ownership/checkpoint
hardening, milestone G7, the deterministic stochastic training example
and its exact resume, milestone G8, the honest benchmark
characterization, milestone G9, the cross-cutting integration suite, and
milestone G10, the phase closure that finally moved `dropout` out of
`UNSUPPORTED`, have all landed.** G1 added random **state**:
`NativeGenerator` exists and is exported, `NativeModule` registers
generators as a fourth state category, and `generator_state_dict()`
reports their state. G2 then added the randomness itself, at exactly one
layer: the `tensorforge.splitmix64` derivation, the inverted-Dropout
float64 CPU kernel, the guarded `tf_core_dropout_forward` export, and
`NativeTensorCore.dropout_forward` — a **stateless** Core that takes the
whole random key as explicit `seed`/`call_index` integers and touches no
generator. G3 then added the one differentiable operation over it, and
one registry name (`"dropout"` in `AUTOGRAD_OPS`): an **explicit,
keyword-only** `NativeGenerator`, the graph-owned multiplier mask whose
`multiply` is the entire backward, and the reserve → commit / abandon
call transaction that makes one successful stochastic forward consume
exactly one call and every failure consume none. G4 then added the public
**module**, and one registry name (`"NativeDropout"` in
`NATIVE_MODULES`): stochastic in training, the input object itself in
evaluation, identity at `p == 0`, over one registered generator it either
owns (the default — independent streams) or shares (an explicit one,
stored as the exact object).

G5 then closed the persistence gap, and one reporting-only registry name
(`"checkpoint_generator_state"` in `STATE_SUPPORT`): the native
checkpoint format is now **version 2** — the format *name* never moves —
and its `generators` manifest section persists every canonical
generator's `algorithm`, `algorithm_version`, `seed`, and `calls` (the
last two as canonical decimal strings, since a `uint64` above `2**53`
cannot survive a JSON double) **plus the complete alias topology**: every
registered path mapped to its canonical generator, so shared-versus-
independent streams are restored, not just the numbers. Generator state
adds no array to the archive. A load restores each generator **in
place**, preserving identity and sharing, and validates the topology
strictly in both directions against a real `named_generators()` traversal
before anything changes; a version-1 archive still loads into a
generator-free model and is **rejected** for a generator-bearing one,
naming the generators it cannot supply, fabricating nothing. The whole
load is one transaction: model, buffers, optimizer, and generators commit
under a single rollback guard.

Above that, G6 hardened the RNG, graph, ownership, and checkpoint
contracts by adversarial test, **G7 demonstrated the end-to-end exact
stochastic training resume** (`examples/native_dropout_training.py`: an
interrupted run reloaded into a completely fresh model, optimizer, and
generator set that reproduces the uninterrupted run by exact equality),
G8 characterized the stack honestly
(`benchmarks/benchmark_native_dropout.py` — correctness gated before
timing, no speed asserted), G9 added the cross-cutting integration suite
(`tests/test_native_phase_g.py`), and **G10 closed the phase** under
fresh Windows Release and Debug builds and a Clang
ASan/UBSan/LeakSanitizer matrix — after which, and not one milestone
earlier, `dropout` **left** the unsupported list. Reproducibility is
exact **only for the state actually captured** (no Python `random`, no
NumPy global RNG, no data-loader position, and no scheduler state), and
ordinary concurrent training is not claimed thread-safe. The stable Python
framework's features (see
[architecture.md](architecture.md)) are **not** listed here — a feature
appears as supported only if the native stack itself provides it.
Everything below is float64/cpu only, explicit, and experimental; see
[backend_experiments.md](backend_experiments.md) for the full story and
[native_autograd_design.md](native_autograd_design.md) for the autograd
design.

**Phase status.** Phase A — **complete** (runtime, ownership, shapes/
strides/offsets/views, broadcasting, reductions, float64/cpu metadata).
Phase B — **complete** (Python-managed reverse-mode autograd, graph
lifetime, view and broadcasting gradients, parameter-version
stale-graph safety). Phase C — **complete** (parameters, modules,
Linear/ReLU/Sequential, MSE loss, `sqrt`/`reciprocal` optimizer
primitives, SGD, Adam, optimizer state snapshots, checkpoint files,
deterministic training and in-memory/file resume, and the failure/
lifetime/ownership guardrails). Phase D — **complete** (the native CNN
stack, milestones D0–D12: the locked architecture contract
([native_cnn_design.md](native_cnn_design.md)); `NativeFlatten`; the
differentiable `conv2d` operation with input/weight/bias gradients and the
trainable `NativeConv2d` module; the `maxpool2d` operation with its private
saved winners, backward scatter, and the `NativeMaxPool2d` module; the
deterministic end-to-end CNN training + exact checkpoint-resume proof
(`examples/native_cnn_training.py`); cross-cutting Phase-D integration
tests; honest CNN benchmarks (`benchmarks/benchmark_native_cnn.py`); and
ASan/UBSan validation of the whole stack). **Phase E — Native
Classification and Stable Math — is complete** (milestones E0–E10):
its architecture contract is locked in
[native_classification_design.md](native_classification_design.md)
(milestone **E0**, complete) and milestones **E1**, **E2**, **E3**, and
**E4** have shipped the differentiable native `exp`, `log`, `softmax`,
and `log_softmax` — the phase's two backward archetypes (saved-output vs
live-input) plus both of its fused probability transforms — while
milestone **E5** shipped the fused `cross_entropy` **Core contract**
(`NativeTensorCore.cross_entropy_forward` /
`cross_entropy_backward` over the guarded
`tf_core_cross_entropy_forward` / `tf_core_cross_entropy_backward`
exports) and milestone **E6** shipped the differentiable
**`NativeTensor.cross_entropy(targets, reduction="mean")`** over it —
one autograd node whose **graph-owned private saved probabilities** drive
its backward, so the logits are never reread and no parameter version is
recorded. E6 added **no** kernel, ABI export, or numerical change.
Milestone **E7** completed the public surface: the stateless
**`NativeCrossEntropyLoss`** module (whose entire forward delegates to
that operation) and the reporting-only **`native_accuracy`** helper.
Milestone **E8** then proved the assembled stack end to end
(`examples/native_classification_training.py`): a deterministic native
Conv2d→ReLU→MaxPool2d→Flatten→Linear classifier over raw logits on twelve
fixed 6×6 images in three classes, 40 `NativeAdam(lr=0.05)` steps (loss
1.159638 → 0.000101, accuracy 0.3333 → 1.0000), checkpointed at step 15
and resumed into a **fresh** model/optimizer pair that reproduces the
remaining loss suffix, parameters, optimizer state, logits, predictions,
and accuracy **exactly** — adding **no** operation, module, optimizer,
kernel, ABI symbol, or schema change (checkpoint format stays version 1).
Milestone **E9** then characterized the stack honestly
(`benchmarks/benchmark_native_classification.py`): seven cases — `exp`,
`log`, `softmax`, `log_softmax`, cross-entropy forward, cross-entropy
backward, and one complete classification training step — each gated for
correctness **before** timing, each labelled with the reference it used
(`stable_tensorforge`, `numpy`, or `native_only`), and each reported as a
median with min/max/spread after warm-up, with `--smoke` and `--json`
modes. **No speed is asserted and no timing threshold exists anywhere.**
Milestone **E10** then closed the phase, adding no numerical capability:
cross-cutting integration tests (`tests/test_native_phase_e.py`), Release
**and** Debug native builds (10/10 CTests each, zero warnings), Clang
ASan/UBSan validation of the whole classification stack with **zero
diagnostics attributable to TensorForge**, a practical LeakSanitizer pass
with **no native leak**, the full Python regression suite, and
documentation reconciliation across every status surface.
**Phase F — Native Normalization and Stateful Buffers — is *complete*
(F0–F9).** Its architecture contract is locked in
[native_normalization_design.md](native_normalization_design.md)
(milestone **F0**, complete — design and repository reconciliation, no
numerical behavior), **F1** is complete (the private atomic
native-buffer state transaction, `load_state_dict` refactored onto it,
and the `persistent_buffers` reconciliation in `STATE_SUPPORT` — state
management and capability reporting only, **no normalization
mathematics**), and **F2** is complete (`NativeLayerNorm` — the first
native normalization module: stateless, differentiable through the mean
and the population variance, composed entirely from existing native
operations with `sqrt(var + eps)` ordering and no kernel, ABI symbol,
`NativeTensorCore` method, custom backward, or `NativeTensor`
normalization operation; now in `NATIVE_MODULES` and the exports, with
`"layernorm"` removed from `UNSUPPORTED`), and **F3** is complete
(`NativeBatchNorm1d` — the **first stateful native numerical module**:
`(N, C)` batch normalization with differentiable training statistics
(gradients flow through the batch mean and the population variance),
persistent native `running_mean`/`running_var` buffers advanced by a
graph-free **atomic two-buffer transaction** through the F1 primitive
(both identities preserved, no parameter version moved), and evaluation
from **graph-safe immutable snapshots** of those buffers rather than the
live objects; composed from the same existing operations, so again no
kernel, C ABI symbol, ctypes declaration, `NativeTensorCore` method,
custom backward, or `NativeTensor.batch_norm` operation; now in
`NATIVE_MODULES` and the exports, with the native checkpoint format
unchanged at version 1), and **F4** is complete (`NativeBatchNorm2d` —
NCHW `(N, C, H, W)` batch normalization reducing over **N, H, and W**,
so each channel gets one population mean and one population variance
over `N * H * W` values, over the **same** shared private
implementation: it supplies only its rank, its reduction axes, its
`(1, C, 1, 1)` broadcast layout, and the channels-last permutation its
rank-1 affine parameters need. Running statistics stay `(C,)` persistent
buffers and eval snapshots are owning `(1, C, 1, 1)` copies. Again no
kernel, C ABI symbol, ctypes declaration, `NativeTensorCore` method,
custom backward, or `NativeTensor.batch_norm` operation, and the
checkpoint format stays version 1; now in `NATIVE_MODULES` and the
exports, and with both BatchNorm shapes live **`batchnorm` has left
`UNSUPPORTED`**, whose remaining entries are listed in the
"Unsupported or future" section below).
**The numerical normalization module surface is complete, and F5 has
proved its state/checkpoint/ownership/graph-safety contracts.** **F5 is
complete**: an exhaustive
`tests/test_native_normalization_state.py` plus narrow additions to the
generic buffer and checkpoint suites prove §7–§10 by executable test
(canonical dotted buffer keys, independent state snapshots,
strict/non-strict loads, exact never-casting metadata validation, mixed
parameter/buffer transaction atomicity, buffer identity across state and
checkpoint loads, exact eval-output reproduction, the
buffer-only-versus-full stale-graph distinction, the save/corrupt-load
failure boundaries, eval-graph snapshot safety under `retain_graph` and a
failed retryable backward, and the live-storage baselines) — **tests and
documentation only, no new capability, and the checkpoint format stays
version 1**. **F6 is complete**:
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
with nine cases — the LayerNorm forward and backward, the BatchNorm1d
training forward, evaluation forward, and backward, the BatchNorm2d
training forward, evaluation forward, and backward, and one complete
F6-style normalized training step — each **correctness-gated before any
timing**, each labelled with the reference it actually used
(`stable_tensorforge` where a real stable counterpart exists;
`native_only` for the three BatchNorm2d cases, because the stable line
has no public `BatchNorm2d`, though those keep a rigorous NumPy
NCHW/transformed-oracle correctness gate), reporting the median with min,
max, and spread after warm-up, with `--smoke`/`--json` modes, **no result
file, no speed assertion, no committed timing number, and no CI timing
threshold** — measurement only, adding no capability. Milestones
**F8 is complete**: `tests/test_native_phase_f.py` proves the
cross-cutting interactions — one integrated `Conv2d → BatchNorm2d → ReLU
→ MaxPool2d → Flatten → Linear → BatchNorm1d → ReLU → LayerNorm →
Linear` classifier over raw logits and the fused loss, trained by
`NativeAdam` and resumed **exactly** from one version-1 checkpoint
including all four running-statistic buffers and the evaluation-mode
output; BatchNorm snapshots, MaxPool2d winners, and cross-entropy
probabilities coexisting in one eval graph and releasing exactly once;
buffer mutation leaving an earlier graph valid while parameter mutation
correctly stales it; the versioning archetypes; shared and frozen
parameters; a non-contiguous NCHW input; strict stable/native separation;
honest per-boundary failure atomicity; error-state recovery; the NumPy
boundary; live-storage baselines; and reality-derived capability, export,
and artifact guardrails — tests and documentation only, adding no
capability. Milestone **F9 is complete**: the phase closure — fresh
Windows Release **and** Debug builds each passing the full existing
10-test CTest suite with zero project warnings and the active runtime
proved to stay Release; a fresh Clang 18.1.3 ASan+UBSan build in WSL2
Ubuntu 24.04 with instrumentation *proved* by `nm -D` (22 `__asan*` and
13 `__ubsan*` symbols) rather than assumed; 10/10 sanitized native
CTests with leak detection enabled; 1,968 sanitized
normalization-focused Python tests with **zero ASan and zero UBSan
diagnostics**; the F6 example and the F7 benchmark smoke path clean under
the sanitized library; and a practical LeakSanitizer lifecycle whose
native live-storage counter returned **exactly** to baseline, its
remaining process-exit allocations identified honestly as CPython/NumPy
shutdown retention with no TensorForge frame and no suppression file —
**validation and documentation only, adding no numerical capability**.
No normalization *operation* is differentiable, and no
normalization kernel or C ABI export exists.
See also
[roadmap.md](roadmap.md). Everything Phases D and E
deliberately excluded
remains unsupported and is named in the "Unsupported or future" section
below, which stays the single place this document lists capabilities the
native line does not have.

## Runtime and metadata

| Capability | Status | Notes |
|---|---|---|
| Layered runtime | Supported | `NativeStorage` → `NativeTensorView` → `NativeTensorCore` → `NativeTensor`, each layer explicit |
| `NativeStorage` ownership | Supported | Explicit allocate/free of native memory; positive sizes only |
| Shape / strides / offsets | Supported | Full strided layouts, including non-contiguous and offset views |
| Contiguity tracking | Supported | `contiguous` reported on every tensor |
| Native contiguous copy | Supported | `contiguous_copy()` gathers any strided/offset view **storage to storage** through the native `tf_core_contiguous_copy` kernel — tensor data never round-trips through a NumPy host buffer (only shape/stride arrays cross as ctypes metadata). Shared infrastructure: it is what every Policy-B copy-then-compute path (Conv2d, MaxPool2d, softmax), `NativeFlatten`, `NativeParameter` construction, and the differentiable `contiguous_copy` operation use |
| dtype metadata | Supported | `"float64"` only — validated, never promoted or cast |
| device metadata | Supported | `"cpu"` only — validated, never transferred |
| Lifetime rules | Supported | Explicit `close()` / `with` blocks; closed tensors reject reads clearly; `owns_core` distinguishes owning from borrowing views |

## Forward operations on NativeTensor

| Operation | Forward | Differentiable | Notes |
|---|---|---|---|
| `add` | Yes | Yes | NumPy-style broadcasting; gradients un-broadcast back |
| `subtract` | Yes | Yes | Broadcasting; right operand's gradient negated |
| `multiply` | Yes | Yes | Broadcasting; each gradient reads the other operand |
| `relu` | Yes | Yes | Fused native `relu_backward` mask kernel |
| `sqrt` | Yes | Yes | v3.11 optimizer math primitive; backward `1/(2·sqrt(x))` from the **saved forward output** — IEEE: negatives → NaN, signed zeros preserved |
| `reciprocal` | Yes | Yes | v3.11 optimizer math primitive; backward `−1/x²` from the **saved forward output** — IEEE: ±0 → ±inf, ±inf → ±0, NaN propagates |
| `softmax` | Yes | Yes | Phase E, **E3**: numerically stable probability transform over any single axis (positive or negative, rank >= 1; bool/float/`None`/out-of-range axes and rank 0 rejected before any allocation). The forward is a **fused maximum-shift kernel** — `exp(x - max(x)) / sum(exp(x - max(x)))` in one pass, entirely in float64 — not a composition of public ops (E3 adds no `max`, `argmax`, or division). Its C ABI is **contiguous-only**, so the Core layer applies the existing Policy-B copy-then-compute for strided views — a fully native storage-to-storage copy, with no tensor-data NumPy round-trip; the result is always fresh owning contiguous storage. Backward is the closed-form `y * (upstream − sum(upstream · y, axis, keepdims))`, **composed from existing Core ops** (no dedicated backward kernel) and reading only the **saved output**, so it records **no** version snapshot. Exceptional values are plain IEEE: a NaN or `+inf` makes its slice NaN, `-inf` takes zero mass — values, not ABI errors |
| `log_softmax` | Yes | Yes | Phase E, **E4**: numerically stable log-probabilities over any single axis, with exactly `softmax`'s axis rules, validation order, and output contract. The forward is its **own fused log-sum-exp kernel** — `(x - max(x)) - log(sum(exp(x - max(x))))` in one pass, entirely in float64 — and is **never** `softmax().log()`: no probability buffer is formed and no division happens, so a probability too small to represent (which would round to 0, giving `log(0) = -inf`) still gets an accurate finite log-probability. Its C ABI (`tf_core_log_softmax_forward`) is **contiguous-only** and shares E3's call shape and its trust-boundary validator; the Core layer applies the same native Policy-B copy-then-compute for strided views, and the result is always fresh owning contiguous storage. Backward is the closed-form `upstream − exp(y) · sum(upstream, axis, keepdims)`, **composed from existing Core ops** (no dedicated backward kernel — `exp(y)` recovers the probabilities from the saved log probabilities) and reading only the **saved output**, so it records **no** version snapshot. E4 added no `NativeLogSoftmax` module. Exceptional values are plain IEEE: a NaN or `+inf` makes its slice NaN, an all-`-inf` slice is NaN, a `-inf` beside finite values keeps `-inf` while its neighbours stay stable — values, not ABI errors |
| `log` | Yes | Yes | Phase E, **E2**: elementwise natural logarithm over any legal shape, both execution paths; backward rereads the **live input** — `upstream × reciprocal(x)` through the existing `reciprocal` primitive (no division operation exists) — so a direct `NativeParameter` parent **is** version-guarded: mutating it after forward raises the stale-graph error before any gradient is committed anywhere in the graph. Plain IEEE `std::log` — no clamping, no epsilon, no domain rejection: `log(1)=0`, `log(±0)=-inf`, `log(negative)=NaN`, `log(+inf)=+inf`, NaN propagates, and those are **values**, not ABI errors. Reuses E1's self-validating export contract unchanged |
| `exp` | Yes | Yes | Phase E, **E1**: elementwise `e**x` over any legal shape, both the strided-odometer and contiguous execution paths; backward is `upstream × ` the **saved forward output**, so it never rereads the input and records **no** parameter-version snapshot (mutating a direct parameter after forward leaves the edge valid). Plain IEEE `std::exp` — no clamping, no inserted bound: `exp(0)=1`, overflow → `+inf`, underflow → `+0`, `-inf` → `+0`, NaN propagates. Its two guarded exports (`tf_core_exp`, `tf_core_exp_contiguous`) validate handles, layout, spans, and overflow at the ABI itself |
| `matmul` | Yes | Yes | 2-D only, no batching/broadcasting |
| `sum` | Yes | Yes | All elements or one axis; `keepdims` |
| `mean` | Yes | Yes | All elements or one axis; `keepdims` |
| `reshape` | Yes | Yes | Borrowing metadata-only view; contiguous sources only |
| `transpose` / `T` | Yes | Yes | Borrowing view; inverse-permutation backward |
| `narrow` | Yes | Yes | Borrowing view; native scatter backward |
| `contiguous_copy` | Yes | Yes | Owning materialization; pass-through backward |
| `maxpool2d` | Yes | Yes | D8–D9: fused NCHW window-maximum primitive (int/tuple `kernel_size`/`stride`/`padding`, `stride=None` ⇒ non-overlapping); backward scatters through the **private winner buffer the forward saved** — never rereading the input or recomputing a maximum — so it records **no** version snapshot and survives input mutation; the winner buffer is graph-owned state released with the graph history. The pooling ***module*** built on it, `NativeMaxPool2d`, is a separate layer and **is implemented** (D10 — see the training stack below) |
| `conv2d` | Yes | Yes | D6: fused NCHW/OIHW cross-correlation primitive (int/tuple stride & padding, optional bias); input/weight gradients via native backward kernels, bias gradient via existing `sum`; conditional stale-value versioning. The trainable convolution ***module*** built on it, `NativeConv2d`, is a separate layer and **is implemented** (D7 — see the training stack below) |
| `dropout` | Yes | Yes | Phase G, **G3**: inverted Dropout as `dropout(p, *, generator)`. The `generator` is a **required keyword-only** `NativeGenerator` — there is no default, process-global, or module-global stream, no implicit per-call generator, and no NumPy or Python `random` fallback — and `p` is validated by the *same* shared normalizer the G2 Core uses (`bool` → `TypeError`; `p == 1`, `p > 1`, `p < 0`, NaN, ±inf → `ValueError`). **`p == 0` is identity**: it returns the caller's own tensor object (`result is input`), allocating nothing, calling no kernel, building no graph node, and consuming no generator call. Otherwise it reserves exactly one call, runs the stateless G2 Core **outside** the generator's lock with that reservation's seed and index, and commits as the **last** state-changing action — so **one successful stochastic forward consumes exactly one call** with or without gradients, and every ordinary failure before the commit releases everything and cancels, leaving the same unconsumed index for the next forward. Backward consumes none, ever. The private multiplier mask is **graph-owned** state (the `graph_resources` contract, unchanged — the third member of the family beside MaxPool2d's winners and cross-entropy's probabilities), released exactly once with the graph history and closed immediately by a no-grad forward. Backward is `upstream * mask` through the existing native `multiply`, so **no Dropout backward kernel exists**; it never rereads the input, never redraws, and never touches the generator, and therefore records **no** version snapshot — a later input mutation or generator `reseed`/`reset`/`load_state` cannot change an existing graph's gradient or raise. The Dropout ***module***, `NativeDropout`, is a separate layer and shipped at **G4**; it owns the train/eval behavior, which this operation deliberately has none of. `dropout` as a *capability* stayed in `UNSUPPORTED` through G9 and left it at the **G10** closure |

This table lists **operations** on `NativeTensor`. An implemented
differentiable *operation* and an implemented *module* are different
things and are tracked separately: operations here, the modules that
compose them in the training-stack table below. For Conv2d and MaxPool2d
both halves shipped in Phase D.

**`cross_entropy` is split across two layers, and the split is
deliberate.** Phase E milestone **E5** shipped its **`NativeTensorCore`
layer** — `cross_entropy_forward` and `cross_entropy_backward`, listed
among the Core operations rather than here, because they are
graph-unaware runtime helpers:

- **Forward** takes rank-2 `(batch_size, num_classes)` logits (the class
  axis is fixed at axis 1 — there is no `axis` argument) plus a
  one-dimensional sequence of integer class labels and a `"mean"` or
  `"sum"` reduction, and returns a **scalar** loss core, the **private
  saved probabilities**, the copied labels, and the normalized
  reduction. It is one **fused** kernel — row maximum, log-sum-exp,
  probabilities, and loss in a single pass — and is never
  `-log(probability[target])`, never `softmax().log()`-then-index, and
  never `log_softmax()`-then-gather.
- **Targets** are not native tensors: classification targets remain
  **exact host-side label metadata** under the Phase-E contract, and
  Phase K milestone K2 — which gave the runtime an `int64` index/result
  dtype — did **not** widen cross-entropy to accept `NativeTensor`
  targets. They are copied into an independently owned, contiguous,
  read-only `int64` array before anything is allocated. `bool` and
  floating-point
  labels are rejected outright — including integral ones like `1.0` —
  and mutating the caller's list or array afterwards cannot affect the
  forward or the backward.
- **Backward** consumes only the saved probabilities, the copied labels,
  the reduction, and a **native one-element upstream core** (read
  straight from its storage, never through NumPy), and produces
  `upstream · (p − onehot) / N`. **It never rereads the logits** — they
  are not even an argument.
- Both C ABI exports are **contiguous-only** for tensor data (Policy-B
  copy-then-compute handles strided logits at the Core layer), validate
  every argument including every target index before writing anything,
  and leave every destination byte-for-byte unchanged when they reject.

Milestone **E6** then added the differentiable
`NativeTensor.cross_entropy(targets, reduction="mean")` on top, as a
single autograd node:

- it calls the E5 forward **once** and returns a **scalar**
  `NativeTensor`, so `loss.backward()` works with the engine's default
  seed and an explicit rank-0 upstream scales the gradient exactly;
- the private saved probabilities become **graph-owned state**: retained
  under `retain_graph=True` and after a failed retryable backward,
  released exactly once when the graph history is (a one-shot backward's
  cleanup, or `close()`), and closed immediately when no gradient is
  required. They are never a public tensor, a parameter, a buffer, a
  state-dictionary entry, or a checkpoint entry;
- the copied `int64` targets are immutable metadata held by the backward
  closure and collected with it — never a native integer tensor;
- **backward never rereads the logits**, so the node records **no
  expected parameter version**: mutating a direct `NativeParameter`
  logits parent after the forward leaves the edge valid and its gradient
  correct for the forward that ran;
- E6 added no C++ kernel, no C ABI export, and no change to any formula.

Milestone **E7** then added the public surface over that operation:

- **`NativeCrossEntropyLoss(reduction="mean")`** — a parameter-free,
  buffer-free `NativeModule` whose whole forward is
  `logits.cross_entropy(targets, reduction=self.reduction)`. It adds no
  kernel, ABI symbol, arithmetic, or target validation of its own, so it
  inherits every guarantee above unchanged. Its reduction is validated in
  the constructor by the operation's own validator, and is constructor
  configuration rather than model state: `state_dict()` is empty and it
  contributes no checkpoint keys.
- **`native_accuracy(logits, targets) -> float`** — a **reporting**
  helper, listed under metrics rather than operations because it is not
  native compute: it validates rank-2 logits and targets under the same
  strict contract, materializes the logits **once** through the explicit
  public `to_numpy()` boundary, takes `numpy.argmax(axis=1)` (ties go to
  the first maximal index), and returns a Python `float` in
  `[0.0, 1.0]`. It builds no graph, touches no gradient, parameter, or
  version, allocates no native storage, and retains nothing. There is no
  accuracy kernel, no Core method, and no native `argmax`.

## Autograd engine

| Capability | Status | Notes |
|---|---|---|
| Reverse-mode `backward()` | Supported | Python-managed graph over autograd-unaware native kernels |
| Broadcasting gradients | Supported | Native un-broadcast reduction |
| View gradients | Supported | reshape/transpose/narrow/contiguous_copy backwards |
| Narrow scatter backward | Supported | Dedicated native kernel |
| Gradient accumulation | Supported | Multiple paths sum; leaves retain `.grad` |
| One-shot graph release | Supported | Default `backward()` frees the traversed graph deterministically |
| `retain_graph=True` | Supported | Repeated passes accumulate until `zero_grad()` |
| Stale parameter-version detection | Supported | Mutated-after-forward parameters raise before any gradient changes (v3.7) — recorded only where backward reads a direct parent's current value |
| Saved-forward-result backwards | Supported | `sqrt`/`reciprocal` backward reads the recorded output, never the parent — parameter mutation after forward leaves those edges valid (v3.11) |
| Failure rollback | Supported | A failed pass commits no partial gradients and frees nothing |
| Double backward / higher-order | Not supported | No graph is built through backward math |

## Training stack

| Component | Status | Notes |
|---|---|---|
| `NativeParameter` | Supported | Graph-free trainable leaf; value versioning; controlled `copy_value_` mutation |
| `NativeModule` | Supported | Registration by assignment, recursive identity-deduplicated cycle-safe traversal, train/eval, `zero_grad()` |
| Buffers | Supported | v3.15: `register_buffer(name, tensor, persistent=True)`, `buffers()` / `named_buffers()`; NativeTensor-backed non-`Parameter` persistent state (the infrastructure `NativeBatchNorm1d`'s running statistics use as of Phase F milestone **F3**; generator state is deliberately **not** a buffer — Phase G milestone **G1** gave it its own fourth registration category, listed below); identity-deduplicated, cycle-safe traversal; persistent buffers join `state_dict`/`load_state_dict` and checkpoints, non-persistent buffers are never serialized. Reported as `persistent_buffers` in `STATE_SUPPORT` since Phase F milestone **F1** — reconciliation of an under-reported capability, not a new feature |
| `NativeGenerator` | Supported (state only) | G1 (Phase G): explicit, inspectable, serializable random **state** — **it generates no random values**. A pure-Python value holder owning **no native storage**, allocating nothing native, and having **no `close()`**, so its whole lifecycle leaves the native live-storage count untouched. Exactly four read-only fields: `algorithm` (`"tensorforge.splitmix64"`), `algorithm_version` (`1`), an unsigned 64-bit `seed`, and `calls` — the count of **committed** stochastic calls. `state()` returns an independent plain dict; `load_state()` / `reseed()` / `reset()` validate everything before assigning anything, so a rejected call leaves the generator bit-identical. Seeds are exact Python `int`s (`bool`, NumPy integer scalars, and `int` subclasses rejected; out-of-range raises rather than truncating); `seed=None` draws once through `secrets.randbits(64)`, and nothing consults the clock, the process id, an address, NumPy's global RNG, or Python's `random`. Identity is object identity — no value equality — and `copy`, `deepcopy`, and pickle are refused, because a copy would silently produce the same values in two places. Its private call transaction (`_reserve_call` → `_commit_call`/`_abandon_call`) is lock-protected and token-validated: one private `threading.RLock` covers reservation, commit, cancellation, and every state read and write with the caller's work outside it; at most one live reservation, so a concurrent **or** reentrant second caller raises *before* an index is minted; opaque single-use tokens make stale, foreign, duplicated, committed, and cancelled tokens inert; commit advances exactly once and cancel never advances; `load_state`/`reseed`/`reset` are refused mid-reservation; and exhaustion is checked under the lock at `2**64 − 1`, so the counter never wraps. Reservation creation is a **two-phase claim / construct / publish** transaction with the token built **outside** the lock: phase 1 publishes only an internal construction claim (no reservation, no counter or serial movement), phase 2 constructs the token holding no generator lock, phase 3 publishes the reservation and advances the serial exactly once, and phase 4 delivers the token. The four failure positions get **different** cleanup, because clearing the claim does nothing once a reservation is published: a failure between publication and delivery would leave an active reservation whose only token is being dropped, so it is cancelled by an **exact-match** cleanup on generator, serial, **and** index that leaves `calls` untouched and leaves a newer, foreign, committed, or already-abandoned reservation strictly alone — a failed delivery consumes an opaque serial, never a call index. Token construction is the one allocation in the path and allocation can run finalization, so keeping it outside the lock means **no user code, callback, or generator-owned allocation runs while a generator lock is held** — which is what stops a finalizer inverting the multi-generator lock order. While the claim stands, another reservation, `load_state`, `reseed`, `reset`, and `replace_generator_states` all raise and mutate nothing (inspection still works); any construction failure, `MemoryError` and `KeyboardInterrupt` included, releases the claim in `finally`, publishes nothing, and skips no serial. The lock stays an `RLock` because the multi-generator transaction re-enters through the same write seam it holds the locks around, and because CPython may collect at any remaining allocation under the lock. **Serialization for correctness — parallel stochastic execution is not claimed.** No derivation, kernel, ABI symbol, ctypes declaration, `NativeTensorCore` method, or `NativeTensor` operation exists |
| Module generator registration | Supported (state only) | G1 (Phase G): generators are a **fourth** registered state category beside parameters, buffers, and child modules. Assignment registers (a `NativeGenerator` is an unambiguous native type); `register_generator(name, generator)` is the strict explicit form (`None` unregisters, `KeyError` when absent, non-generator raises `TypeError`); `module.name = None` and `del module.name` unregister; one name stays exactly one category, evicting in **both** directions; `_generators` joined the reserved-name set. `generators()` / `named_generators(prefix="", recurse=True)` ride the same deterministic pre-order depth-first, identity-deduplicated, cycle-safe walk parameters and buffers use, so one shared generator appears **once** under its first-discovered canonical name while two generators with identical state stay two entries. Registration stores the exact object, never a copy, and unregistering one alias never resets or invalidates a generator still referenced elsewhere. `generator_state_dict()` reports `{canonical_name: state}` as independent plain dicts, and `load_generator_state_dict(state, strict=True)` restores them in place through the shared `replace_generator_states` transaction — validate → **lock every unique target in one global identity order** → recheck each for a published *or under-construction* reservation while holding them all → snapshot → non-failing integer writes, with the rollback completing before any lock is released. So no reservation can begin on a target between the recheck and the end of the commit, two concurrent loads over overlapping generators (arriving through different modules, in opposite canonical order) cannot deadlock, and no other thread can observe a partial commit. Identities are preserved, a shared generator is locked and assigned exactly once, a conflicting state supplied through an alias is rejected (strict) or reported as unexpected (non-strict) rather than applied, and a target with a live reservation blocks the whole load while leaving that reservation intact. Reported as `generator_state` in `STATE_SUPPORT` since **G1** — an **in-memory** state capability, exactly like `persistent_buffers`; it does *not* mean generator state is checkpointed. Generators are deliberately **absent** from `state_dict()`, which stays contractually `{name: NativeTensor}` and is byte-for-byte unchanged for every existing model — and **not** persisted by native checkpoints, which stay at version 1 until **G5** |
| `state_dict` / `load_state_dict` | Supported | In-memory, parameters and persistent buffers, atomic validate-then-commit with rollback (buffer identity preserved on restore). Generators are **not** included (Phase G milestone G1 gave them their own `generator_state_dict()` surface): this mapping stays contractually tensor-valued. Since **F1** the replacement half runs through the private `_native_state.replace_native_state` transaction, shared with the future normalization running-statistics update; `load_state_dict`'s public signature, validation order, error messages, key reporting, version semantics, and atomicity are unchanged |
| Atomic native state transaction | Supported (private) | **F1** (Phase F): `tensorforge.experimental._native_state.replace_native_state` — identity-preserving, exception-safe replacement of one or more registered `NativeParameter`/persistent-buffer cores as one all-or-nothing transaction. Validate → stage → commit, with the commit boundary at "every core swap **and** every parameter-version increment succeeded"; complete rollback of cores and versions before it; exactly-once closing of replaced and abandoned cores after it; destinations deduplicated by object identity (an aliased parameter is swapped, versioned, and released once; conflicting values for one destination are rejected before mutation). **Deliberately private** — absent from `tensorforge.experimental.__all__`, and *not* a public in-place mutation API for `NativeTensor` (`NativeParameter.copy_value_` remains the only public controlled-mutation primitive) |
| `NativeLinear` | Supported | Seeded deterministic init; strictly 2-D input |
| `NativeReLU` | Supported | Parameter-free activation module |
| `NativeFlatten` | Supported | D1 (Phase D): parameter-free, buffer-free batch-preserving flatten `(N, …) → (N, features)`, Python-composed from the existing `reshape`/`contiguous_copy` ops and their autograd — no new kernel, no custom backward; returns an independent owning result so it composes safely in `NativeSequential` |
| `NativeConv2d` | Supported | D7 (Phase D): the trainable convolution **module** over the differentiable `conv2d` operation — OIHW weight / optional `(O,)` bias `NativeParameter`s, deterministic uniform conv fan-in initialization, 4-D NCHW input validation; backward supplied entirely by the operation's autograd (no new kernel, ABI symbol, or custom module backward) |
| `NativeMaxPool2d` | Supported | D10 (Phase D): the pooling **module** over the differentiable `maxpool2d` operation — parameter-free, buffer-free, normalized `(h, w)` `kernel_size`/`stride`/`padding` (`stride=None` ⇒ non-overlapping); no winner state between calls and no state-dictionary or checkpoint keys |
| `NativeSequential` | Supported | Ordered container with contiguous integer-string slots |
| `NativeLayerNorm` | Supported | F2 (Phase F): the first native normalization **module** — normalizes the trailing `len(normalized_shape)` dimensions, stateless (no buffers, identical in train and eval). Population variance with `sqrt(var + eps)` ordering; multi-axis reduction as sequential single-axis `mean(keepdims=True)` calls; composed entirely from existing native operations (`mean`, `subtract`, `multiply`, `add`, `sqrt`, `reciprocal`) so backward comes from the existing autograd. `weight`/`bias` `NativeParameter`s only when `elementwise_affine=True` (registered in that order); fresh owning contiguous output. No new kernel, ABI symbol, ctypes declaration, `NativeTensorCore` method, custom backward, functional helper, or `NativeTensor.layer_norm` operation |
| `NativeBatchNorm1d` | Supported | F3 (Phase F): the **first stateful native numerical module** — `(N, C)` batch normalization. Training normalizes with this batch's own **differentiable** population statistics (gradients flow through the batch mean and the variance; `sqrt(var + eps)`, no Bessel correction) and advances the persistent native `running_mean`/`running_var` buffers by `(1 − momentum)·running + momentum·batch`, computed **graph-free** from the *same* batch statistics and committed as one **atomic two-buffer transaction** through the F1 primitive (both Python identities preserved, both old cores closed exactly once, **no parameter version moved**). Evaluation normalizes from **independent owning graph-free snapshots** of those buffers, so a later training step, or a **buffer-only** `load_state_dict()`/`load_native_checkpoint()`, can never change an already-built eval graph's gradient (§7); the snapshots are graph-owned and released exactly once with the graph history. (A *full* checkpoint load also replaces `gamma`/`beta`, whose versions then move, so the pre-existing v3.7 stale-parameter guard intentionally rejects the old graph — a parameter contract BatchNorm neither bypasses nor weakens, never a running-buffer effect.) `gamma`/`beta` `NativeParameter`s always exist (no `affine=False`, no `track_running_stats`, no `num_batches_tracked`); state order is `gamma`, `beta`, `running_mean`, `running_var`; fresh owning contiguous output. Composed entirely from existing native operations — no new kernel, ABI symbol, ctypes declaration, `NativeTensorCore` method, custom backward, functional helper, or `NativeTensor.batch_norm` operation; native checkpoint format unchanged at version 1 |
| `NativeBatchNorm2d` | Supported | F4 (Phase F): NCHW `(N, C, H, W)` batch normalization over the **same shared private implementation** as `NativeBatchNorm1d` — the public class declares only `_INPUT_NDIM = 4`, `_REDUCTION_AXES = (0, 2, 3)`, `_TRAILING_DIMS = 2`, its layout string, and `_CHANNELS_LAST = (0, 2, 3, 1)`, and inherits every method by function identity. Reduces over **N, H, and W** and never over the channel axis, so each channel gets one population mean and one population variance over `N * H * W` values, taken as three sequential single-axis `mean(keepdims=True)` calls (`(N, C, H, W)` → `(1, C, H, W)` → `(1, C, 1, W)` → `(1, C, 1, 1)`; no tuple-axis reduction was added). Batch statistics and evaluation snapshots are `(1, C, 1, 1)`; the persistent `running_mean`/`running_var` buffers stay `(C,)`. **Channelwise affine:** rank-1 `gamma`/`beta` broadcast from the *trailing* axis, so the **activation** is transposed to channels-last for the affine step and back again (then materialized contiguous) rather than the parameters being reshaped — which keeps `gamma` a **direct versioned** `multiply` operand and preserves the existing stale-parameter guard exactly, while `multiply`/`add`'s existing broadcast-aware backwards reduce the affine gradients over N, H, and W. Channels-last is an internal step, never a public layout mode. Non-contiguous NCHW input supported; output is always a fresh owning contiguous NCHW tensor. Adds **no** kernel, ABI symbol, ctypes declaration, `NativeTensorCore` method, custom backward, functional helper, or `NativeTensor.batch_norm` operation; checkpoint format unchanged at version 1 |
| `NativeDropout` | Supported | G4 (Phase G): the Dropout **module** over the differentiable `dropout` operation. `NativeDropout(p=0.5, seed=None, generator=None)`. `p` uses the **same** shared normalizer the G2 Core and the G3 operation use (`bool` → `TypeError`; `p == 1`, `p > 1`, `p < 0`, NaN, ±inf → `ValueError`; `p == 0` accepted), stored as a plain `float`. `seed` and `generator` are **mutually exclusive** — supplying both raises `TypeError` rather than silently ignoring one. Without an explicit generator the module **creates and owns** `NativeGenerator(seed)` (one OS-entropy draw at `seed=None`), so every layer gets an independent stream by default; with one it registers **that exact object**, never a copy, so several layers share one interleaved stream. All validation precedes generator creation and registration, so a rejected construction draws no entropy, registers nothing, and allocates nothing. The generator is registered under the canonical name `"generator"` as the **fourth** state category: present in `generators()` / `named_generators()` / `generator_state_dict()`, deliberately **absent** from `state_dict()` (which stays `{name: NativeTensor}`), identity-preserved across `load_generator_state_dict()`, deduplicated by identity in parent traversal, and never a parameter, buffer, or child module. The module owns **no native storage**, and dropping it never closes, resets, or mutates its generator. Forward validates the input **first** (`TypeError` for a non-`NativeTensor`, `RuntimeError` for a closed one), then dispatches: **training** is exactly `input.dropout(self.p, generator=self.generator)`, so the operation owns the whole call transaction and a successful forward consumes exactly one call while a failed one consumes none; **evaluation** returns the **input object itself** with no reservation, allocation, kernel call, or graph node, so any number of eval forwards leaves **no gap in the stream**; and **`p == 0`** is identity in both modes, delegated to the operation rather than duplicated. `train()`/`eval()` propagate normally, including through `NativeSequential`, and never reseed or reset the generator. Adds **no** kernel, ABI symbol, ctypes declaration, `NativeTensorCore` method, autograd operation, or checkpoint-format change. At G4 generator state was **not** checkpointed (format version 1): a save omitted the stream and a load fabricated nothing, so exact stochastic resume did not exist yet. **G5** moved the format to version 2 and persists every registered generator's state and alias topology, **G7** demonstrated the exact stochastic resume end to end, and the **G10** closure moved `dropout` out of `UNSUPPORTED` |
| `NativeMSELoss` | Supported | `"mean"` / `"sum"` reductions; exact shapes, no broadcasting |
| `NativeCrossEntropyLoss` | Supported | E7 (Phase E): the classification **loss module** over the differentiable `cross_entropy` operation — parameter-free, buffer-free, `"mean"`/`"sum"` only (validated in the constructor by the operation's own validator, so an invalid reduction can never reach it), targets validated and copied by the operation itself. Its forward is exactly `logits.cross_entropy(targets, reduction=self.reduction)`: no new kernel, ABI symbol, arithmetic, or custom backward, and no state-dictionary or checkpoint keys |
| `native_accuracy` | Supported (reporting only) | E7 (Phase E): **not** a native operation — a Python helper that validates rank-2 logits and strict `int64` targets, materializes once through the explicit public `to_numpy()` boundary, takes `numpy.argmax(axis=1)` (first-maximal index on ties), and returns a Python `float` in `[0.0, 1.0]`. Builds no graph, touches no gradient/parameter/version, allocates no native storage, retains nothing. Reported in the new `NATIVE_METRICS` inventory, never in the operation inventories |
| `NativeSGD` | Supported | Minimal `value ← value − lr·grad`; identity-deduplicated; two-phase mutation-atomic `step()`; `zero_grad()`; in-memory `state_dict`/`load_state_dict` (v3.13: lr + positional parameter metadata) |
| `NativeAdam` | Supported | Adaptive optimizer (v3.12): validated `lr`/`betas`/`eps`; persistent optimizer-owned native m/v moments and per-parameter step counts; bias correction via `sqrt`/`reciprocal` (no division); graph-free staged updates committed through `copy_value_`; skipped frozen/`grad=None` parameters never age state; explicit state lifetime — `close()` releases the moments; in-memory `state_dict`/`load_state_dict` (v3.13) |
| Optimizer state (in-memory) | Supported | v3.13: one versioned schema (format 1, exact optimizer type tag), ordered positional shape/dtype/device parameter metadata — no object ids, names, values, or gradients — caller-owned independent NativeTensor m/v snapshots and per-parameter step counts (NativeAdam), exact validation with no casting or device movement, staged atomic loading that never touches parameter values, versions, gradients, or retained graphs; deterministic in-memory training continuation with the module state contract |
| Checkpoint files / resume | Supported | v3.14, extended by Phase G milestone **G5**: `save_native_checkpoint` / `load_native_checkpoint` — one pickle-free NPZ archive (format `"tensorforge.native_checkpoint"`, now **version 2**; version-1 archives still load under the locked compatibility rule) holding the model state, optionally one native optimizer's v3.13 state, every registered generator's state **and alias topology** (G5), and JSON-compatible metadata; UTF-8/JSON uint8 manifest, indexed float64 array entries, strict full-archive validation before any live mutation, strict optimizer presence/type matching, atomic temporary-file replacement, `allow_pickle=False` loading, deterministic bit-identical file resume (`examples/native_checkpoint_resume.py`) and exact **stochastic** resume (`examples/native_dropout_training.py`, G7); no scheduler state, data-loader/shuffle position, Python `random`, or NumPy global-RNG capture, and no `map_location` |
| End-to-end MLP training | Proven | `examples/native_mlp_training.py`: 25 deterministic steps, monotonic 99.5% loss reduction |
| End-to-end CNN training + checkpoint resume | Proven | D11: `examples/native_cnn_training.py` — convolution → activation → pooling → flatten → linear over eight fixed 6×6 images, learning a spatial edge-strength target with `NativeMSELoss` and `NativeAdam(lr=0.05)`; 40 deterministic steps, loss 0.771306 → 0.011085 (98.6% reduction); a run interrupted at step 15, checkpointed (model **and** optimizer state) and resumed into a fresh model/optimizer pair reproduces the uninterrupted run **exactly** — losses, predictions, parameter values, and optimizer state. Adds no kernel, operation, loss, optimizer, or checkpoint schema |
| End-to-end classification training + checkpoint resume | Proven | E8: `examples/native_classification_training.py` — the same layer stack over **raw logits** into `NativeCrossEntropyLoss`, on twelve fixed 6×6 images in three classes with `NativeAdam(lr=0.05)`; 40 deterministic steps, loss 1.159638 → 0.000101 (99.99% reduction) and reporting accuracy 0.3333 → 1.0000 (`native_accuracy`, outside the training mathematics); a run interrupted at step 15, checkpointed (model **and** optimizer state, format version 1) and resumed into a fresh model/optimizer pair reproduces the uninterrupted run **exactly** — the whole remaining loss suffix, parameters, optimizer moments and step counters, logits, predictions, and accuracy. Adds no kernel, operation, module, loss, metric, optimizer, or checkpoint schema |

## Unsupported or future (native line)

None of the following exists on the native stack today. Several exist
in the stable Python framework — that does not make them native.

- `divide` as a NativeTensor operation (a raw ctypes `elementwise_divide`
  kernel exists at the kernel layer, but no tensor op and no backward;
  `reciprocal` + `multiply` compose what the training stack needs)
- `tanh` and `sigmoid` — both outside Phase E entirely. (`exp`, `log`,
  `softmax`, and `log_softmax` **are** implemented as of E1–E4 and are
  listed in the forward-operation table above; `log_softmax` shipped as
  its own fused log-sum-exp kernel, deliberately **not** composed from
  the shipped `log` and `softmax`.)
- a public `max`/`min`/`argmax` reduction or a public `divide` operation
  (the softmax and log-softmax maximum shift, normalization, and
  log-normalizer live inside their fused kernels; they are not exposed as
  operations)
- `NativeSoftmax` or `NativeLogSoftmax` **modules** — the operations
  shipped, the modules are explicitly out of scope for Phase E
- `NLLLoss` on the native line
- scheduler state, data-loader/shuffle position, epoch counters, Python
  `random`, or NumPy global-RNG capture in native checkpoints (registered
  `NativeGenerator` state and its alias topology **are** captured, as of
  Phase G milestone **G5**); `map_location`, partial or name-remapped
  loading, checkpoint merging, sharding, compression, or encryption.
  *(Phase J does not change this. Its contract carries a loader position as
  **caller-supplied metadata** through the channel that already exists, so
  the archive's own capture set stays exactly what it is. J4 gave the
  loader an in-memory `state_dict()` and J5 proved that state survives a
  real version-3 archive, but nothing discovers, registers, or archives
  one: placing it in metadata is the caller's step, and no checkpoint code
  knows a loader exists. J5 is the evidence for that too — a real save and
  a real load with the loader's two state methods patched to record any
  call, and neither fired.)*
- **automatic loader discovery, in either direction.** **Phase J**
  ([native_data_pipeline_design.md](native_data_pipeline_design.md),
  complete: milestones **J0** through **J9**) deliberately does **not**
  build it, at any milestone: no checkpoint code imports, discovers,
  registers, or validates a Phase-J object, and no pipeline module imports
  checkpoint code. Carrying a loader position through an archive is the
  **caller's** step, proved end to end at **J5** through the existing
  validated metadata channel, with the checkpoint format, its accepted
  versions, and its capture set all unmoved. What exists is
  `NativeTensorDataset` (**J1**) — a finite host-backed dataset that
  materializes a feature batch and a target batch for an index sequence —
  `NativeBatchSampler` (**J2**), which plans shuffled or sequential
  batch-index groups and holds an explicit epoch and cursor,
  `NativeDataLoader` (**J3**), which iterates one epoch at a time and
  delivers `(NativeTensor, numpy.ndarray)` batches transactionally, and
  that loader's own **in-memory** `state_dict()` / `load_state_dict()`
  with exact mid-epoch restoration (**J4**), carried through a real
  version-3 archive as caller metadata and restored exactly into fresh
  objects (**J5**), with a worked training program over the whole chain
  and its exact interrupted-versus-uninterrupted proof (**J6**). So
  **native mini-batching exists** and a loader's position can be
  serialized, checkpointed, and restored exactly — but every step of that
  is the caller's, and nothing finds a loader for them
- weight decay, AdamW, AMSGrad, parameter groups, per-parameter
  learning rates, or schedulers on the native optimizers
- *(Phase E itself is complete — E0–E10 — so nothing from it is listed
  here any more.* E1–E4 shipped `exp`, `log`, `softmax`, and
  `log_softmax`, E5 and E6 shipped the fused `cross_entropy` Core
  contract and the differentiable operation over it, E7 shipped
  `NativeCrossEntropyLoss` and `native_accuracy`, E8 proved deterministic
  classification training and exact checkpoint resume
  (`examples/native_classification_training.py`), E9 characterized
  the stack (`benchmarks/benchmark_native_classification.py`), and E10
  closed it with integration, Release/Debug, sanitizer, and leak
  validation. Note what
  E9 is **not**: no performance contract, no committed timing numbers, no
  CI speed gate, and no optimization work — the native kernels remain
  deliberately naive. See
  [native_classification_design.md](native_classification_design.md)
- `NLLLoss`, binary cross-entropy, class weights, `ignore_index`, label
  smoothing, soft/one-hot targets, and `reduction="none"` on the native
  line; top-k, per-class, confusion-matrix, streaming, or stateful
  metrics; a `NativeSoftmax`/`NativeLogSoftmax` module; and a native
  `max`, `max_with_indices`, or `argmin`.
  *(A native `argmax` **left this list at Phase K, milestone K3**, which
  shipped it as `NativeTensor.argmax` / `NativeTensorCore.argmax` over the
  `tf_core_argmax` export — the integer result dtype had arrived one
  milestone earlier at K2. `native_accuracy` still reports through its
  explicit `to_numpy()` host boundary deliberately, and `max` is declined
  **permanently** rather than deferred: a kernel that finds the position of
  a maximum necessarily knows the maximum, and Phase K does not expose
  it.)*
- `BatchNorm3d`, `InstanceNorm`, `GroupNorm`, `RMSNorm`, synchronized or
  distributed BatchNorm, a fused normalization kernel, a functional
  `batch_norm`, and a `NativeTensor.batch_norm` operation — none is in
  Phase F's scope
- a generic `rand`/`randn`/Bernoulli/sampling or distribution API, any
  global or process-wide random state, NumPy global-RNG integration,
  `Dropout2d`/`Dropout3d`, stochastic depth, and attention dropout —
  none is in Phase G's scope. *(Dropout itself **is** supported and is
  therefore no longer listed here: the G2 Core, the G3
  `NativeTensor.dropout(p, *, generator)` operation, and the G4
  `NativeDropout` module all appear in the tables above. `"dropout"`
  stayed in `UNSUPPORTED` for the whole of **G0–G9** — G4 implemented and
  exported the module and G5 persisted its stream, neither moving the
  boundary — and the name was removed at **G10**, after the full Phase-G
  closure matrix passed. See
  [native_rng_dropout_design.md](native_rng_dropout_design.md).)*
- additional native activations/math beyond
  `relu`/`sqrt`/`reciprocal`/`exp`/`log`/`softmax`/`log_softmax`
- CUDA / GPU execution
- float16 / bfloat16, dtype promotion or casting, AMP. *(**float32** is
  no longer listed here: it **is** supported, on the CPU, beside float64,
  since Phase I milestone **I9**. It stayed in `UNSUPPORTED` for the whole
  of I0–I8 while the runtime was progressively generalized underneath it —
  storage at I1, transfer at I2, elementwise at I3, reductions and matmul
  at I4, the CNN stack at I5, classification at I6, modules and Dropout at
  I7, optimizers and checkpoint v3 at I8 — and the name moved only at I9,
  after integrated float32 training and the exact float32 resume proof
  both passed. **Casting and promotion remain absent**: a tensor's dtype
  is fixed at construction and the only way to get the other one is to
  construct it. See
  [native_dtype_float32_design.md](native_dtype_float32_design.md).)*
- Transformers / text models
- distributed training
- integration or implicit dispatch into the stable `tensorforge.Tensor`

## Phase D — the native CNN stack, **complete**

Every milestone below has shipped and is validated; this section is the
per-milestone record, not a plan. The native CNN stack's **architecture
contract is locked** in
[native_cnn_design.md](native_cnn_design.md) (milestone **D0**). **D1
(`NativeFlatten`) has shipped**, **D2 shipped the internal convolution
forward compute kernel** (`tf::conv2d_forward_contiguous`, a hidden C++
symbol), and **D3 has shipped the forward-only convolution *layer***: the
exported, exception-guarded C ABI wrapper `tf_core_conv2d_forward`, its
ctypes/`errcheck` registration, and `NativeTensorCore.conv2d_forward` (a
Python-reachable, forward-only, autograd-unaware Core method). **D4/D5
shipped the internal convolution input/weight-gradient kernels**, and **D6
completed the differentiable native convolution operation**: the exported
guarded backward C ABI wrappers (`tf_core_conv2d_input_backward`,
`tf_core_conv2d_weight_backward`), the Core backward methods
(`NativeTensorCore.conv2d_input_backward`/`conv2d_weight_backward`), the
bias gradient composed from the existing native `sum` reduction (no
dedicated kernel), and the Python-managed **`NativeTensor.conv2d`** autograd
primitive (input/weight/bias gradients, conditional stale-value version
tracking, failure rollback). **D7 has shipped the trainable `NativeConv2d`
module** — an OIHW weight / optional `(O,)` bias `NativeParameter` layer
with deterministic uniform conv fan-in initialization, 4-D NCHW input
validation, and backward supplied entirely by the D6 `conv2d` autograd (no
new kernel, C ABI symbol, or custom module backward); it registers in
`NATIVE_MODULES` and exports from `tensorforge.experimental`. **D8 has
shipped the forward-only pooling *layer*:** the internal compute kernel
`tf::maxpool2d_forward_contiguous` (a hidden C++ symbol in the new
`cpp/src/pooling.cpp`), the exported guarded C ABI wrapper
`tf_core_maxpool2d_forward` with its ctypes/`errcheck` registration, and
`NativeTensorCore.maxpool2d_forward` — a Python-reachable, autograd-unaware
Core method that also produces the **private saved-winner buffer** (an
internal float64 buffer of flat plane offsets with a `-1` padding sentinel,
validated exact against `H*W ≤ 2^53`) the D9 backward consumes. **D9 has
completed the differentiable pooling operation**: the internal scatter-add
`tf::maxpool2d_backward_contiguous`, the exported guarded
`tf_core_maxpool2d_backward` wrapper (which **validates every winner value**
— `-1` or an exact in-range integer — before scattering, never rounding),
`NativeTensorCore.maxpool2d_backward`, and the Python-managed
**`NativeTensor.maxpool2d`** autograd node whose single input-gradient
callback uses only the saved winners (no input reread, no recomputed
maximum, **no version snapshot**), with the winner buffer owned by the
graph history and released exactly when it is. **D10 has shipped the
`NativeMaxPool2d` module** — a parameter-free, buffer-free layer that
normalizes `kernel_size`/`stride`/`padding` to two-element tuples
(`stride=None` ⇒ non-overlapping windows) and delegates its forward
entirely to that operation: no new kernel, C ABI symbol, ctypes
declaration, custom backward, parameter, buffer, or checkpoint schema, no
`return_indices`, and no winner state held between calls. It registers in
`NATIVE_MODULES`, exports from `tensorforge.experimental`, contributes no
state-dictionary or checkpoint keys, and composes in a `NativeSequential`
beside `NativeConv2d`/`NativeReLU`/`NativeFlatten`/`NativeLinear`. **D11
has shipped the deterministic end-to-end proof**
(`examples/native_cnn_training.py`): the full Conv→ReLU→Pool→Flatten→
Linear stack learns a genuinely spatial regression target (the strongest
bright-to-dark vertical edge of eight fixed 6×6 images) with
`NativeMSELoss` and `NativeAdam(lr=0.05)` over 40 steps — loss
0.771306 → 0.011085 — and a run interrupted at step 15, checkpointed
(model **and** optimizer state) and resumed into a completely fresh
model/optimizer pair reproduces the uninterrupted run **exactly**: loss
history, final predictions, every parameter value, and every optimizer
state entry. It adds no kernel, ABI symbol, operation, loss, optimizer, or
checkpoint schema. **D12 closed the phase**: cross-cutting Phase-D
integration tests (`tests/test_native_phase_d.py` — the full module stack,
end-to-end autograd, shared graphs, the two versioning contracts meeting in
one backward, state/checkpoint integration, cross-layer failure atomicity,
resource lifetime, and the capability boundary), honest CNN benchmarks
(`benchmarks/benchmark_native_cnn.py` — conv/pool forward, forward+backward,
end-to-end training step, and a stable-framework reference, with no speed
claims), **ASan/UBSan validation** of the whole native CNN stack under
Clang on Linux, a LeakSanitizer pass over the native CTests, documentation
reconciliation, and the replacement of the milestone-era documentation
guardrails with durable semantic ones. D12 added no kernel, C ABI symbol,
ctypes declaration, operation, module, or schema.

| Capability | Milestone | Status |
|---|---|---|
| `NativeFlatten` (batch-preserving; existing reshape/copy autograd) | D1 | **Implemented** |
| Internal convolution forward compute kernel (C++, not exposed) | D2 | **Implemented (internal)** |
| Convolution forward C ABI export (`tf_core_conv2d_forward`) — exception-guarded; self-validates handles/dims/offsets/output-shape/overflow/span-bounds; contiguous storage is a caller precondition (no stride metadata crosses the ABI, so it never inspects logical contiguity) | D3 | **Implemented (raw kernel)** |
| Convolution forward Core wrapper (`NativeTensorCore.conv2d_forward`) — ctypes, Policy-B copy, output allocation, Python forward access | D3 | **Implemented (Core, forward-only)** |
| Internal convolution input-gradient compute kernel (`tf::conv2d_input_backward_contiguous`, C++, not exposed) | D4 | **Implemented (internal)** |
| Internal convolution weight-gradient compute kernel (`tf::conv2d_weight_backward_contiguous`, C++, not exposed) | D5 | **Implemented (internal)** |
| Convolution input/weight-gradient C ABI export (`tf_core_conv2d_input_backward`, `tf_core_conv2d_weight_backward`) + Core wrappers (`NativeTensorCore.conv2d_input_backward`/`conv2d_weight_backward`) | D6 | **Implemented (raw + Core)** |
| Convolution bias-gradient numerical path (reuse of the existing native `sum` reduction: `g.sum(0).sum(1).sum(1) → (O,)`; no dedicated kernel, no C ABI symbol) | D5–D6 | **Implemented (existing-reduction composition, wired into the autograd node)** |
| Convolution `NativeTensor` autograd op — differentiable `NativeTensor.conv2d(weight, bias=None, *, stride, padding)` | D6 | **Implemented** |
| `NativeConv2d` module — trainable OIHW weight / optional `(O,)` bias layer over the D6 `conv2d` autograd (deterministic uniform conv fan-in init, 4-D NCHW validation; no new kernel/ABI/backward) | D7 | **Implemented** |
| Internal max-pooling forward compute kernel (`tf::maxpool2d_forward_contiguous`, C++, not exposed) — pooled values and winners in one pass | D8 | **Implemented (internal)** |
| Max-pooling forward C ABI export (`tf_core_maxpool2d_forward`) — exception-guarded; self-validates handles/dims/offset/output-shape/winner-exactness/overflow/span-bounds; contiguous storage is a caller precondition (no stride metadata crosses the ABI) | D8 | **Implemented (raw kernel)** |
| Max-pooling forward Core wrapper (`NativeTensorCore.maxpool2d_forward`, plus the private with-winners helper) — ctypes, Policy-B copy, output + private winner-buffer allocation, failure-atomic cleanup | D8 | **Implemented (Core, forward-only)** |
| Internal max-pooling backward compute kernel (`tf::maxpool2d_backward_contiguous`, C++, not exposed) — zero-initializing scatter-add driven only by the saved winners | D9 | **Implemented (internal)** |
| Max-pooling backward C ABI export (`tf_core_maxpool2d_backward`) — exception-guarded; validates handles/dims/offsets/spans **and every winner value** (`-1` or an exact in-range integer, never rounded); takes no kernel/stride/padding | D9 | **Implemented (raw kernel)** |
| Max-pooling backward Core wrapper (`NativeTensorCore.maxpool2d_backward`) — ctypes, Policy-B copies, fresh owning grad_input, failure-atomic cleanup | D9 | **Implemented (Core)** |
| Pooling `NativeTensor` autograd op — differentiable `NativeTensor.maxpool2d(*, kernel_size, stride, padding)`; single `(input,)` parent, graph-owned saved winners, no version snapshot | D9 | **Implemented** |
| `NativeMaxPool2d` module — parameter-free, buffer-free layer over the D8/D9 pooling operation (normalized `(h, w)` `kernel_size`/`stride`/`padding`, `stride=None` ⇒ non-overlapping); no new kernel/ABI/backward, no parameters, buffers, winner state, or state-dict keys | D10 | **Implemented** |
| Deterministic native CNN training + checkpoint-resume proof (`examples/native_cnn_training.py`: Conv→ReLU→Pool→Flatten→Linear + `NativeMSELoss` + `NativeAdam`, 40 steps, loss 0.771306 → 0.011085; interrupted-and-resumed training matches the uninterrupted run exactly — losses, predictions, parameters, and optimizer state) | D11 | **Implemented (proven)** |
| Phase-D cross-cutting integration tests (`tests/test_native_phase_d.py`), CNN benchmarks (`benchmarks/benchmark_native_cnn.py`), ASan/UBSan validation, LeakSanitizer pass over the native CTests, documentation reconciliation, and durable capability guardrails | D12 | **Implemented (phase closed)** |

Locked design decisions (see the design doc for the full contract):
**NCHW** activations, **OIHW** convolution weights, **cross-correlation**
(not flipped); floor output-shape formulas with symmetric per-axis
padding; **copy-then-compute** for non-contiguous inputs (kernels consume
contiguous storage only); convolution as a **new fused `NativeTensor`
primitive** with a Python-managed backward (input/weight kernels + bias
via existing `sum` reductions); max-pool winners saved in an **internal
float64 buffer** of flat input offsets (with a `-1` padding sentinel);
new C ABI families `tf_core_conv2d_*` / `tf_core_maxpool2d_*` under the
existing status/guard contract; and new C++ units `cpp/src/conv2d.cpp`
and `cpp/src/pooling.cpp`. Still float64/cpu only; no dilation, groups,
transposed/average/adaptive/global pooling, channels-last, float32,
CUDA, AMP, BatchNorm, Dropout, im2col, or BLAS/threaded convolution.

## Phase E — native classification and stable math, **complete**

The architecture contract is locked in
[native_classification_design.md](native_classification_design.md); the
registry above (and `tensorforge.backends.cpp`) stays the authority on
what is live. All eleven milestones (E0–E10) have landed.

| Capability | Milestone | Status |
|---|---|---|
| Phase-E architecture contract (scope, public API, stability strategy, backward/versioning matrix, `int64` target contract, saved-probability lifetime, C ABI families, inventory placements, E0–E10 ladder) | E0 | **Complete** (documentation only — no numerical behavior) |
| Native `exp`: the C++ kernel (odometer + contiguous), the self-validating guarded exports `tf_core_exp` / `tf_core_exp_contiguous`, their ctypes registration, `NativeTensorCore.exp()`, and the differentiable `NativeTensor.exp()` with its **saved-output** backward and **no** version snapshot | E1 | **Implemented** |
| Native `log`: the same four layers, reusing E1's self-validating export contract unchanged; backward is `upstream × reciprocal(live input)`, so a direct `NativeParameter` parent **is** version-checked and a stale graph fails before any gradient moves — the deliberate contrast with `exp` | E2 | **Implemented** |
| Stable `softmax`: the fused maximum-shift kernel in the new `cpp/src/classification.cpp`, the contiguous-only `tf_core_softmax_forward` export, `NativeTensorCore.softmax(axis=-1)` with Policy-B copy-then-compute, and the differentiable `NativeTensor.softmax(axis=-1)` whose saved-output backward is composed from existing Core ops | E3 | **Implemented** |
| Stable `log_softmax`: its own fused log-sum-exp kernel in `cpp/src/classification.cpp` (**never** `softmax().log()` — no probability buffer, no division), the contiguous-only `tf_core_log_softmax_forward` export sharing E3's call shape and trust-boundary validator, `NativeTensorCore.log_softmax(axis=-1)` with the same Policy-B copy-then-compute, and the differentiable `NativeTensor.log_softmax(axis=-1)` whose saved-output backward — `upstream − exp(y) · sum(upstream, axis, keepdims)` — is composed from existing Core ops with no backward kernel and no version snapshot | E4 | **Implemented** |
| Fused `cross_entropy` **Core contract**: the internal fused forward (scalar loss **and** private saved probabilities in one pass) and saved-probability backward kernels, the guarded contiguous-only `tf_core_cross_entropy_forward` / `tf_core_cross_entropy_backward` exports (which revalidate every target index themselves), their ctypes registration, and the graph-unaware `NativeTensorCore.cross_entropy_forward` / `cross_entropy_backward` wrappers with strict copied-`int64` targets, `"mean"`/`"sum"` reductions, Policy-B copy-then-compute, and deterministic multiple-output failure cleanup | E5 | **Implemented** (the Core layer; the autograd node over it is the E6 row below) |
| The differentiable `NativeTensor.cross_entropy(targets, reduction="mean")` operation over that Core contract: one scalar-output autograd node with **graph-owned** private saved probabilities (retained under `retain_graph=True` and a failed retryable backward, released exactly once with the graph history, closed immediately on a no-grad forward), closure-owned immutable `int64` target metadata, **no logits reread** and therefore **no expected parameter version**, and complete failure atomicity across E5 forward, graph construction, and backward. Adds no kernel, no ABI export, and no numerical change | E6 | **Implemented** |
| The public classification surface: the stateless **`NativeCrossEntropyLoss`** module, whose entire forward delegates to the E6 operation (no kernel, ABI symbol, arithmetic, target validation, or state of its own), and the reporting-only **`native_accuracy`** helper (strict targets, one explicit `to_numpy()`, NumPy `argmax`, Python `float`, no graph/gradient/version/storage side effects) — plus the new `NATIVE_METRICS` inventory and its `backend_info()` key. Adds no training mathematics | E7 | **Implemented** |
| Deterministic classification training + exact checkpoint resume: `examples/native_classification_training.py` — a `NativeConv2d(1, 4, 3, seed=0)` → `NativeReLU` → `NativeMaxPool2d(2)` → `NativeFlatten` → `NativeLinear(16, 3, seed=1)` classifier over **raw logits** into `NativeCrossEntropyLoss`, on twelve fixed 6×6 single-channel images in three classes (four per class, positions varying, committed as source literals, labels host integers), trained full-batch for **40** deterministic `NativeAdam(lr=0.05)` steps: loss **1.159638 → 0.000101** (99.99% reduction), reporting accuracy **0.3333 → 1.0000**, both the convolution and the linear head moving. Interrupted at step **15**, checkpointed (model **and** optimizer state, format **version 1**, no new keys) and resumed into a **fresh** model/optimizer pair, it reproduces the uninterrupted run **exactly** — remaining loss suffix, parameters, both Adam moment buffers, step counters, logits, predictions, and accuracy. `native_accuracy` is used for reporting only, never inside the training mathematics, and a tripwire proves one complete step reaches no NumPy compute or tensor-data conversion. Adds **no** operation, module, loss, metric, optimizer, kernel, ABI symbol, or schema change, and no inventory entry | E8 | **Implemented** (an integration proof on one fixed task — not a benchmark, not a generalization or speed claim) |
| Classification benchmark characterization: `benchmarks/benchmark_native_classification.py` — seven cases (`exp_forward`, `log_forward`, `softmax_forward`, `log_softmax_forward`, `cross_entropy_forward`, `cross_entropy_backward`, `classification_training_step`), each with a **correctness gate that runs before any timing** (shape, finiteness, reference parity, no input mutation; gradients for the backward case; finite loss, parameter update, optimizer-state advance, graph release, and stable parity for the training step), an honest per-case reference label (`stable_tensorforge`, `numpy` where the stable line has no direct operation — `log_softmax` — or `native_only`), warm-up plus repeated `time.perf_counter_ns` measurements with setup and cleanup outside the timer, and **median** reporting with min/max/spread and every raw sample. `--smoke` (1 warm-up / 3 repetitions) and `--json` modes; writes no result file. **No speed assertion, no committed timing number, and no CI timing threshold** — observed ratios are local characterizations only. Adds no capability of any kind | E9 | **Implemented** (measurement only) |
| Phase-E cross-cutting integration tests (`tests/test_native_phase_e.py`: the whole classification path in one graph, inventory self-consistency, stable/native separation with no implicit dispatch, saved-probability and winner lifetime, the versioning archetypes meeting in one mixed graph, Policy-B strided inputs through every fused kernel, the stateless loss module and reporting-only metric, exact resume for both native optimizers, storage baselines, failure atomicity and error-state recovery, a NumPy tripwire, and the capability boundary); **Release and Debug** native builds (10/10 CTests each, zero compiler/linker warnings); Clang **ASan/UBSan** validation of the classification stack with zero diagnostics attributable to TensorForge; a practical **LeakSanitizer** pass with no native leak; documentation reconciliation; and the conversion of milestone-era absence guardrails into durable semantic checks. Adds no numerical capability | E10 | **Implemented (phase closed)** |

Phase E adds **no** persistent state: the native checkpoint format stays
**version 1**, E1–E6 added no parameter, buffer, module, loss,
metric, optimizer, schema, benchmark, or example, E7 added a stateless
loss module and a stateless reporting helper, and E8 added only an
example and integration tests — no division operation
(`log`'s derivative composes from the existing `reciprocal`), no
public `max`/`argmax`/`gather` (the softmax, log-softmax, and
cross-entropy shifts and target lookups are internal to their kernels),
and no native integer tensor (cross-entropy targets are copied host
`int64` metadata). Neither probability transform has a backward kernel:
both gradients are composed from existing Core operations. Cross-entropy
*does* have one, because its gradient is a distinct closed form over the
saved probabilities. Those saved probabilities are **private state** at
both layers: Core-owned at E5, **graph-owned** by the E6 autograd node —
never a public tensor, never a parameter or buffer, never serialized, and
released exactly once with the graph history.

## Phase F — native normalization and stateful buffers, **complete**

**`NativeLayerNorm` (F2), `NativeBatchNorm1d` (F3), and
`NativeBatchNorm2d` (F4) are all implemented, F5 has proved their
state/checkpoint/ownership/graph-safety contracts by exhaustive test, F6
has shipped a deterministic normalized training example with exact
checkpoint resume, F7 has shipped the honest benchmark
characterization, F8 has shipped the cross-cutting integration and
semantic guardrails, and F9 has closed the phase with the Release/Debug
build revalidation, the sanitizer and LeakSanitizer passes, and the
documentation reconciliation.** The authoritative statement of what
exists is the backend registry: all three modules are in
`NATIVE_MODULES`, both `"layernorm"` and `"batchnorm"` have left
`UNSUPPORTED`, and there is **no** normalization entry in
`TENSOR_CORE_OPS`, `AUTOGRAD_OPS`, or `RAW_KERNELS` (all three modules
are compositions of existing operations, not operations).

The Phase-F **architecture contract** is locked in
[native_normalization_design.md](native_normalization_design.md)
(milestone **F0**). It fixes, before any numerical code is written: the
public API (`NativeLayerNorm(normalized_shape, eps=1e-5,
elementwise_affine=True)` with `weight`/`bias`;
`NativeBatchNorm1d(num_features, eps=1e-5, momentum=0.1)` and
`NativeBatchNorm2d(...)` with `gamma`/`beta` and `running_mean` /
`running_var`); the decision to **compose** normalization from existing
native operations (`mean`, `subtract`, `multiply`, `add`, `sqrt`,
`reciprocal`, `reshape`, broadcasting, `contiguous_copy`) so that **no
kernel, C ABI export, ctypes declaration, or `NativeTensorCore` method
is added** and the backward comes from the existing autograd; population
variance with `eps` inside the square root; training statistics that are
differentiated through; the rule that a **live mutable running buffer is
never captured as a rereadable graph operand** (eval mode uses
independent graph-free snapshots, which is why buffers stay unversioned);
atomic two-buffer running-statistics updates with rollback and preserved
buffer identity; and state/checkpoint integration with the format
**unchanged at version 1**.

| Milestone | Deliverable | Status |
|---|---|---|
| F0 | Phase-F architecture contract and repository reconciliation | **Complete** (design and documentation only — no numerical behavior) |
| F1 | Atomic native-buffer state transactions: the private `_native_state.replace_native_state` primitive extracted and generalized from the existing `load_state_dict` staging/commit/rollback — validate-then-stage-then-commit with an explicit commit boundary (every core swap **and** every parameter-version increment), complete rollback of cores *and* versions before it, exactly-once closing of replaced and abandoned cores, identity-preserving swaps, and destination deduplication by object identity (conflicting values for one destination rejected before mutation). `NativeModule.load_state_dict` now delegates to it with its public signature, validation order, error messages, key reporting, version semantics, and atomicity unchanged. Reusable by F3/F4 to commit `running_mean` and `running_var` together **without** a state dictionary. Plus the `STATE_SUPPORT` persistent-buffer correction. Adds **no** normalization mathematics, module, kernel, ABI symbol, tensor operation, or export | **Complete** (state management and capability reporting only) |
| F2 | `NativeLayerNorm` — the first native normalization module: stateless (no buffers, identical in train and eval), differentiable through the mean and the population variance, composed entirely from existing native operations (`mean`, `subtract`, `multiply`, `add`, `sqrt`, `reciprocal`; `sqrt(var + eps)`, no Bessel correction; multi-axis reduction as sequential single-axis means). `weight`/`bias` `NativeParameter`s only when `elementwise_affine=True`. Fresh owning contiguous output. Adds **no** kernel, C ABI symbol, ctypes declaration, `NativeTensorCore` method, custom backward, functional helper, or `NativeTensor.layer_norm` operation — only `NATIVE_MODULES` grew and `"layernorm"` left `UNSUPPORTED` | **Complete** (a module composed from existing operations) |
| F3 | `NativeBatchNorm1d` — the **first stateful native numerical module**: `(N, C)` batch normalization, again composed entirely from existing native operations. Differentiable current-batch statistics (population variance, `sqrt(var + eps)`, gradients through the mean *and* the variance); persistent native `running_mean`/`running_var` buffers updated by `(1 − momentum)·running + momentum·batch` from the *same* batch statistics, computed **graph-free** and committed as one **atomic two-buffer transaction** through the F1 primitive (identities preserved, replaced cores closed exactly once, **no parameter version moved**); evaluation from **independent owning graph-free snapshots** of the running buffers, so no registered buffer is ever a rereadable graph operand and a later training step, buffer-only state load, or buffer-only checkpoint load cannot change an earlier eval graph's gradient (§7 — a full checkpoint load that also replaces `gamma`/`beta` still stales the graph through the unchanged parameter-version rule, which is correct and is proved by its own test); graph-owned snapshot lifetime released exactly once with the graph history; validate → build the output graph → prepare the updates → commit ordering, so a failed forward changes no running state and leaks nothing without GC. `gamma`/`beta` always exist; no `affine=False`, `track_running_stats`, `num_batches_tracked`, or unbiased running variance. Adds **no** kernel, C ABI symbol, ctypes declaration, `NativeTensorCore` method, custom backward, functional helper, or `NativeTensor.batch_norm` operation — only `NATIVE_MODULES` grew, `"batchnorm"` **stayed** in `UNSUPPORTED`, and the checkpoint format stayed at version 1 | **Complete** (a stateful module composed from existing operations) |
| F4 | `NativeBatchNorm2d` — the NCHW `(N, C, H, W)` shape over the **same** shared private implementation: the public class supplies only its rank, its `(0, 2, 3)` reduction axes, its `(1, C, 1, 1)` broadcast layout, and the channels-last permutation its rank-1 affine parameters need, and inherits every method by function identity. Reduces over N, H, and W and never over C; `(C,)` running buffers and `(1, C, 1, 1)` snapshots unchanged. The one shared addition is the channelwise affine step, which transposes the *activation* rather than reshaping `gamma`/`beta` so the existing direct-parameter stale-value guard survives. Adds **no** kernel, C ABI symbol, ctypes declaration, `NativeTensorCore` method, custom backward, or `NativeTensor.batch_norm` operation — `"batchnorm"` finally left `UNSUPPORTED` here, once *both* shapes existed | **Complete** (the normalization module surface, not the phase) |
| F5 | Normalization state, checkpoint, and graph-safety hardening: a focused `tests/test_native_normalization_state.py` plus narrow additions to the generic buffer and checkpoint suites, proving §7–§10 by executable test — canonical dotted buffer keys, independent state snapshots (storage-independent in both directions), strict/non-strict buffer-key loads, exact never-casting shape/dtype/device validation, identity-preserving mixed loads, mixed parameter/buffer transaction rollback, the version-1 checkpoint manifest/archive gaining no normalization field with BatchNorm buffers as ordinary entries, exact eval-output reproduction across a round trip, the buffer-only-versus-full stale-graph distinction, the corrupt/staging/save failure boundaries, eval-graph snapshot safety under `retain_graph` and a failed retryable backward, and explicit parameter/buffer closure to baseline. Adds **no** capability, module, operation, kernel, ABI symbol, or checkpoint schema field | **Complete** (tests and documentation only — no numerical behavior) |
| F6 | Deterministic normalized training and exact resume: `examples/native_normalization_training.py` trains a `Linear → BatchNorm1d → ReLU → LayerNorm → Linear` regressor for 24 deterministic `NativeAdam` steps with `NativeMSELoss` (98.9% loss reduction), proves two uninterrupted runs bit-identical, and resumes an interrupted run into a fresh model/optimizer pair that reproduces the remaining losses, every parameter, the NativeAdam state, both BatchNorm `running_mean`/`running_var`, the final training-step prediction, and the final evaluation-mode output exactly (format version 1 unchanged, training flags runtime-only). One example and its integration test; adds **no** capability, operation, kernel, schema field, benchmark, or export | **Complete** (integration proof only — no numerical behavior) |
| F7 | Native normalization benchmark characterization: `benchmarks/benchmark_native_normalization.py` — nine cases (`layernorm_forward`, `layernorm_backward`, `batchnorm1d_training_forward`, `batchnorm1d_eval_forward`, `batchnorm1d_backward`, `batchnorm2d_training_forward`, `batchnorm2d_eval_forward`, `batchnorm2d_backward`, `normalized_training_step`), each **correctness-gated before any timing** (a failed gate exits nonzero and publishes nothing), each labelled with the reference it actually used: `stable_tensorforge` for the six cases with a real stable counterpart, run on identical inputs, epsilon, momentum, affine values, running state, initial parameters, and optimizer hyperparameters; `native_only` for the three BatchNorm2d cases, because the stable line has **no public `BatchNorm2d`** to time against — those publish no ratio while keeping a rigorous correctness oracle (an explicit NumPy NCHW population-statistics formula, a channelwise-affine probe, eval-mode state neutrality, and for the backward the stable `BatchNorm1d` on the equivalent `(N*H*W, C)` sample matrix transformed back to NCHW, which is a correctness oracle only and deliberately not timed). `time.perf_counter_ns()` with warm-up, one timed call per sample, every sample retained, setup and cleanup outside the timer, medians reported with min/max/spread, `--smoke`/`--json`, **no result file written, no speed assertion, no committed timing number, and no CI timing threshold**. Adds **no** capability, operation, kernel, ABI symbol, ctypes declaration, schema field, example, or export | **Complete** (measurement only — no numerical behavior) |
| F8 | Cross-cutting Phase-F integration and semantic guardrails: `tests/test_native_phase_f.py` — one test-only integrated classifier (`NativeConv2d(1, 4, 3)` → `NativeBatchNorm2d(4)` → `NativeReLU` → `NativeMaxPool2d(2)` → `NativeFlatten` → `NativeLinear(16, 8)` → `NativeBatchNorm1d(8)` → `NativeReLU` → `NativeLayerNorm(8)` → `NativeLinear(8, 3)` → **raw logits** → `NativeCrossEntropyLoss`) over the fixed twelve-image three-class dataset, trained for 12 deterministic `NativeAdam(lr=0.05)` steps, interrupted at step 5, checkpointed, and resumed into a **fresh** model/optimizer pair reproducing the loss suffix, every parameter, the NativeAdam state, **all four** running-statistic buffers, the final training logits, and the final evaluation-mode logits/predictions/accuracy by **exact equality** (format version 1 unchanged, training flag runtime-only, identities preserved). Also proves: BatchNorm eval snapshots, MaxPool2d winners, and cross-entropy probabilities coexisting in one eval graph with no registered buffer object *or storage* reachable, releasing exactly once; buffer-only mutation (including a real buffer-only `load_native_checkpoint()` over all four registered objects) leaving an earlier eval graph's gradients equal to a clean control, while a full checkpoint load or an affine `copy_value_` correctly stales it through the unchanged v3.7 **parameter** rule; the Phase-E versioning archetypes meeting a normalized graph; shared parameters deduplicating to one slot/update/version increment; frozen parameters registered, persisted, and skipped; a non-contiguous NCHW input through the whole stack in both modes; strict stable/native separation; **honest** per-boundary failure atomicity (transactions are **per module** — one whole training step is *not* globally transactional); error-state recovery; a NumPy/conversion tripwire over one complete integrated step; live-storage baselines across success **and** failure cycles; and reality-derived capability/export/artifact guardrails. Adds **no** capability, operation, kernel, ABI symbol, ctypes declaration, schema field, example, benchmark, or export | **Complete** (tests and documentation only — no numerical behavior) |
| F9 | Phase-F closure: validation, documentation reconciliation, and the completion statement. Fresh Windows **Release** and **Debug** CMake builds (Visual Studio 17 2022, MSVC 19.44.35228.0, CMake 4.4.0), each with `TF_BUILD_TESTS=ON`, each built out-of-source outside the repository, each passing the **full existing 10-test CTest suite** (Release 10/10 in 0.78 s, Debug 10/10 in 0.97 s) with **zero project compiler, linker, and CMake warnings**; the Debug library written to an external directory so the active runtime stayed the Release DLL (proved by the linked CRT: `MSVCP140.dll`/`VCRUNTIME140.dll` versus the Debug build's `MSVCP140D.dll`/`ucrtbased.dll`). A fresh Clang **18.1.3** `address,undefined` build in WSL2 Ubuntu 24.04.4 with **instrumentation proved** — `nm -D` shows 22 `__asan*` and 13 `__ubsan*` dynamic symbols beside the 50 exported `tf_*` symbols, and the library fails to load without the sanitizer runtime (`undefined symbol: __ubsan_vptr_type_cache`). **10/10 sanitized native CTests** with `detect_leaks=1`; **1,968 sanitized Python tests** across 32 normalization and dependency suites with **zero ASan and zero UBSan diagnostics** and no backend-unavailable skip; the F6 example reproducing its exact resume and the F7 benchmark passing all nine correctness gates under the sanitized library. A practical LeakSanitizer lifecycle (integrated classifier, training steps, eval and `native_accuracy`, version-1 checkpoint, fresh-pair resume, non-contiguous NCHW input, a retained-then-released eval graph, explicit closure) returned native live storage **exactly to baseline (0 → 0)**; the process-exit LSan report's 925,710 bytes in 830 allocations contain **no** frame naming `_tensorforge_cpp`, `tf_core_`, `tf_storage_`, `tf::`, or any TensorForge source — only CPython, libc, NumPy, `_ctypes`, and the ASan runtime — and **no suppression file was added**. The complete Python regression suite passed. Adds **no** capability, operation, kernel, C ABI symbol, ctypes declaration, `NativeTensorCore` method, CTest, C++ source, schema field, example, benchmark, or export | **Complete** (validation and documentation only — no numerical behavior) |

Explicitly **outside** Phase F: dropout, a native
RNG and RNG checkpoint state (the first two are Phase G, in progress —
see the section below), `tanh`/`sigmoid`/GELU, more losses,
schedulers, data loaders, native integer tensors, indexing/`gather`/
`max`/`argmax`, float32/float16/bfloat16, casting or dtype promotion,
CUDA, AMP, Tensor Core dispatch, pybind11, the Python C API, implicit
stable/native dispatch or conversion, fused normalization kernels,
normalization-specific C ABI exports, custom normalization backward
kernels, synchronized/distributed BatchNorm, CPU optimization,
performance thresholds, checkpoint format version changes, and real
datasets or generalization claims.

## Phase G — native RNG and Dropout, **complete**

The architecture contract is locked in
[native_rng_dropout_design.md](native_rng_dropout_design.md); the
registry above (and `tensorforge.backends.cpp`) stays the authority on
what is live. **All eleven milestones (G0 through G10) have landed**, and
the phase closed at G10 by moving `"dropout"` out of `UNSUPPORTED`.

| Milestone | What it shipped | Status |
|---|---|---|
| G0 | The architecture contract: Python-managed generator state (an explicit 64-bit seed, a call counter, and an algorithm identifier) with **stateless** native random kernels that receive the whole key for one call; inverted Dropout with a graph-owned multiplier mask whose backward never rereads the input; exactly one generator call consumed per **successful** stochastic forward and none on any failure, in evaluation, at `p == 0`, or in backward, behind a lock-protected token-validated reservation protocol; generator state as a fourth `NativeModule` category; and native checkpoint **version 2** recording the generator **alias topology**, with a locked version-1 compatibility rule and whole-checkpoint transaction atomicity | **Complete** (design, documentation, and guardrails only — no numerical behavior) |
| G1 | `NativeGenerator` and module generator-state ownership — see the two rows in the training-stack table above. Random **state**, not randomness | **Complete** (state only — it generates no random values by itself) |
| G2 | The deterministic **stateless Dropout-forward Core**: the exact locked `tensorforge.splitmix64` derivation as hidden `namespace tf` functions in the new `cpp/src/random.cpp` / `cpp/include/tf_random_internal.h` (`mix64` finalizer; per-call stream key `mix64(seed + GOLDEN·(call_index + 1))`; per-element bits `mix64(stream + GOLDEN·(element + 1))`; uniform `(bits >> 11)·2⁻⁵³`; dropped when `u < p`), all `std::uint64_t` wrapping arithmetic with **no** `<random>`, `random_device`, `mt19937`, clock, process id, address, or static/thread-local state; the `tf::dropout_forward_contiguous` inverted-Dropout kernel writing the output **and** the private multiplier mask in one pass with `1/(1 − p)` computed once per call; the self-validating guarded export `tf_core_dropout_forward` (rejecting null handles, negative offset/count, spans exceeding storage, non-finite or out-of-range `p`, and any aliasing between the input and either destination, writing **nothing** to either destination when it rejects); its ctypes declaration carrying the whole key as two `c_uint64` arguments; `"dropout_forward"` in `TENSOR_CORE_OPS` and `"tf_core_dropout_forward"` in the checked-kernel inventory; the public `NativeTensorCore.dropout_forward(p, *, seed, call_index)` and the private `_dropout_forward_with_mask` that keeps the mask (the `maxpool2d_forward` / winner-buffer split); a dependency-free CTest over both layers; and **committed known-answer vectors asserted identically from C++ and Python**. The Core is **stateless** — it reserves, commits, cancels, inspects, and mutates no `NativeGenerator`, and a direct Core call leaves a live generator bit-identical. Randomness is keyed by the **logical** row-major element index, so a transposed, narrowed, or nonzero-offset view gets the same mask as a contiguous tensor of the same logical shape. Both results are fresh **owning contiguous** cores aliasing neither the input nor each other, the input is never mutated, and the two-result boundary is failure-atomic in C++ *and* in the Python wrapper. There is deliberately **no** backward kernel: that gradient is the existing `multiply` over the saved mask. Adds **no** autograd node, module, export, checkpoint change, or capability-registry move | **Complete** (the Core layer only) |
| G3 | Differentiable **`NativeTensor.dropout(p, *, generator)`** over the G2 contract. One method and one registry name (`"dropout"` in `AUTOGRAD_OPS`) — no C++, no C ABI symbol, no ctypes declaration, no `NativeTensorCore` method, no module, no export, and no checkpoint-format change; the backward is the existing native `multiply` over the saved mask, so **no Dropout backward kernel exists**. The `generator` is **required and keyword-only**: no default, process-global, or module-global stream, no implicit per-call generator, and no NumPy or Python `random` fallback, with `p` validated by the *same* shared normalizer the Core uses. It owns the call transaction — validate, reserve one call, run the Core **outside** the generator's lock with the **reservation's own** seed and index (never a reread `calls`), build the graph, then commit as the **last** state-changing action — so a successful stochastic forward consumes exactly one call *with or without gradients*; `p == 0` returns the caller's own tensor object (`result is input`) having reserved, allocated, and drawn nothing; and every ordinary failure before the commit releases the result and the mask, cancels the reservation, and leaves the same unconsumed index, so the next forward reproduces the committed vector the failed one would have. Backward consumes none, ever. The private multiplier mask becomes **graph-owned** state through the unchanged `graph_resources` contract — the third member of the family beside MaxPool2d's winners and cross-entropy's saved probabilities — released **exactly once** with the graph history, retained under `retain_graph=True`, kept alive across a failed retryable backward, freed by an abandoned graph's `close()` (with `__del__` as the fallback), and closed immediately by a no-grad forward. Backward reads **only** the upstream gradient and that mask: it never rereads the input, never redraws, and never reserves, commits, inspects, or mutates a generator, so the node records **no** expected parameter version and a later input mutation, `reseed`, `reset`, `load_state`, or `load_generator_state_dict` cannot change an existing graph's gradient or raise (a *full* checkpoint load still stales such a graph through some **other** node's parameter rule — a parameter contract, never a Dropout effect). Concurrent or reentrant use of one generator raises deterministically without minting an index. Adds **no** module, train/eval behavior, export, kernel, ABI symbol, checkpoint change, or capability-registry move | **Complete** (the operation layer only) |
| G4 | The **`NativeDropout` module** and its public export: `NativeDropout(p=0.5, seed=None, generator=None)`, one file, one export, and one name (`"NativeDropout"`) appended to `NATIVE_MODULES` — no C++, no C ABI symbol, no ctypes declaration, no `NativeTensorCore` method, no autograd operation, and no checkpoint-format change. `p` goes through the **same** shared normalizer the G2 Core and the G3 operation use, never a third rule. `seed` and `generator` are **mutually exclusive**: supplying both raises `TypeError` rather than silently ignoring one. Without an explicit generator the module **creates and owns** `NativeGenerator(seed)` (fresh OS entropy at `seed=None`), so the default gives every layer an independent stream; with one it registers **that exact object**, never a copy, which is how two layers deliberately share one interleaved stream. Everything is validated before a generator is created or registered, so a rejected construction draws no entropy, registers nothing, allocates nothing, and leaves a supplied generator bit-identical. The generator is registered under the canonical name `"generator"` as the **fourth** state category — in `generators()`, `named_generators()`, and `generator_state_dict()`, and deliberately **absent** from `state_dict()`, which stays contractually `{name: NativeTensor}` — and `load_generator_state_dict()` replaces it in place, so identity and any sharing survive. The module owns **no native storage**, and dropping it never closes, resets, or mutates its generator. Forward validates its input **first** (so evaluation is not a way to hand back a closed tensor), then: **training** delegates to `NativeTensor.dropout`, which owns the whole call transaction, so a successful forward consumes exactly one call and a failed one none; **evaluation** returns the **input object itself** (`result is input`), consuming no call and allocating nothing, so an arbitrary number of eval forwards leaves **no gap in the stream** and the next training forward takes the next index; and **`p == 0`** is identity in both modes, deliberately *not* short-circuited in the module because §6.2 assigns that rule to the operation. Mode follows the ordinary `train()`/`eval()` propagation, including through `NativeSequential`, and switching modes never reseeds or resets the generator. **`"dropout"` deliberately stays in `UNSUPPORTED`** | **Complete** (the module layer only — generator state is still **not** checkpointed, so exact stochastic resume does not exist yet) |
| G5 | Native checkpoint **format version 2** and exact generator restoration. The format **name** never moves; `_FORMAT_VERSION` is now **2** and every new save writes 2, whether or not the model has generators. The manifest gained exactly one field, `"generators"` — `null` when the model registers none (absence stated, not inferred) or three subfields: `keys` (the ordered canonical names from the identity-deduplicated `named_generators()` walk), `entries` (one `{algorithm, algorithm_version, seed, calls}` object per canonical name, mapping exactly `keys` in order, with `seed` and `calls` as **canonical decimal strings** — `^(0\|[1-9][0-9]*)$`, ≤ 20 digits, in `[0, 2**64 − 1]` — because a `uint64` above `2**53` is not representable in the IEEE double most JSON readers use), and `aliases` (the complete **registered path → canonical name** map in full traversal order, every canonical name included and mapped to itself). Generator state adds **no array** to the NPZ payload. A shared generator's state is written **once** while its *topology* is written in full, so two paths draw from one stream in the archive exactly when their aliases name the same canonical entry — sharing is **identity**, never state equality, so two generators with the same seed and counter stay two entries. Canonical names and both orders are functions of the model alone, so saving the same model twice is byte-identical, and no caller-supplied mapping order can alter the archive. **Loading** compares the archive against a real `named_generators()` traversal of the live model, strictly in both directions: missing or unexpected canonical keys, missing or unexpected registered paths, an alias targeting an absent entry, a canonical name absent from `aliases` or not self-mapped, a multi-step alias relation, a repeated JSON object key, saved-shared versus live-independent (and the reverse), a canonical name changed by a reordered registration, an algorithm or version mismatch, and a malformed or out-of-range seed/counter string all raise — **in prevalidation, with the model, buffers, optimizer, and generators completely untouched**. Generators are restored **in place** through `load_state`, so identity and every sharing relationship survive and the archive never constructs a `NativeGenerator`; loading generator state moves no parameter version and stales no graph. A save **or** a load is refused, changing nothing and leaving an existing destination byte-intact, while any target generator has a call reservation in flight — published *or* holding a construction claim — and every generator's state is read in **one** locked snapshot, so the states an archive carries were true together. **Version-1 compatibility** is exactly as locked: a v1 archive still loads into a model with **no** registered generators, and one loaded into a model that **has** them fails naming them, fabricating no seed and no counter — not zero, not fresh entropy, not the current value; a v2 archive with a non-null generator section loaded into a generator-free model fails as an unexpected-generator error; any other version fails. A load is **one transaction over the whole archive**: prevalidation touches nothing, staging materializes every staged value *and* an independent rollback snapshot of every live target, and the commit runs model → optimizer → generators through the components' own loaders inside **one** rollback guard — so any exception in it, a deliverable `KeyboardInterrupt` included, restores all four state families, preserves every object identity, moves no parameter version, leaves graph-owned Dropout masks from earlier graphs untouched, and returns native live storage to baseline. Only external process or interpreter death is outside that guarantee. It is **serializable** as well as atomic: every participating in-memory state replacement — the checkpoint load commit, `NativeModule.load_state_dict`, `load_generator_state_dict`, `NativeSGD.load_state_dict`, and `NativeAdam.load_state_dict` — plus the checkpoint **save snapshot** runs under **one** private process-wide `RLock`, in the universal state-replacement lock order (that guard first, then every unique target generator lock in the global `id()` order, never the reverse), so two concurrent loads leave one archive's state followed by the other's rather than model state from one beside optimizer or generator state from the other, and a save describes one coherent serial point. Generator **reservations** deliberately stay outside the guard, taking only their own generator's lock, so a racing reservation precedes or follows a transaction and no state is ever replaced underneath a live token. Ordinary training mutation (`step()`, `copy_value_`, a backward) does **not** take the guard, so thread-safe concurrent training snapshots are not claimed. The whole registry footprint is one reporting-only name, `"checkpoint_generator_state"` in `STATE_SUPPORT` — kept separate from G1's in-memory `"generator_state"` — with **no** C++, C ABI symbol, ctypes declaration, Core method, autograd operation, module, export, or new public entry point. **`"dropout"` deliberately stays in `UNSUPPORTED`** | **Complete** (the persistence layer only — end-to-end exact stochastic *training* resume is G7) |
| G6 | RNG, graph, ownership, and checkpoint hardening — the finished G1–G5 surface attacked adversarially in the new `tests/test_native_phase_g_hardening.py`, which adds **no capability**. The reservation transition matrix, with each rejected transition asserting five invariants at once (no counter movement, no active-reservation change, no construction-claim change, no serial reuse, no native-storage movement) and the four reservation-creation failure positions distinguished by whether a serial was consumed. The exact `uint64` boundary as the design's own table, row by row, with the final index proved retryable until committed and repeated exhaustion failures freezing every field. Forced concurrent interleavings under barriers and events with **bounded joins and no sleeps**: no duplicate call index, unrelated generators independent, no torn state read, a reservation racing a state replacement provably preceding or following it in both orders, a construction claim refusing both a save and a load, and a transaction started *from inside* token construction refused rather than deadlocked. The deterministic Core's **structural** key properties beside its committed vectors — the stream key injective in the call index for one seed (so no generator can ever repeat a stream), the element derivation injective within one call, and the cross-seed collision that a 128→64-bit key makes unavoidable pinned as a *characterized consequence* rather than a defect — plus the probability extremes and logical-layout independence through real transposed and narrowed views. Every pre-commit position of the call transaction × `RuntimeError`, `MemoryError`, `KeyboardInterrupt`, and a non-`Exception` `BaseException`, each proving the retry reproduces the exact mask the failure would have produced, and every post-commit position proving the index spent exactly once with the original exception primary. All **four** graph-owned saved-resource families — a Dropout mask, MaxPool2d winners, BatchNorm eval snapshots, and cross-entropy probabilities — coexisting in one graph and releasing exactly once, across branched, chained, shared-generator, independent-generator, retained, failed-retryable, and abandoned graphs. A **76-case** checkpoint corruption matrix, every case failing before any live change with the model, buffers, optimizer, and generators bit-identical. Whole-transaction rollback injected at every commit position × the same four exception classes, with object identities, parameter versions, unrelated active reservations, and pre-load graph masks all proved untouched. Save-seam destination atomicity at all seven positions. Repeated success-and-failure lifecycle loops returning native live storage exactly to a measured baseline. **No** C++, C ABI symbol, ctypes declaration, Core method, autograd operation, module, export, schema field, benchmark, example, or registry value changed; one runtime defect was found and fixed with the narrowest possible change — a failed cleanup step could make the Dropout transaction's `__context__` chain **cyclic**, hanging any ordinary chain-walking reader — with a dedicated regression guard | **Complete** (hardening only — no capability) |
| G7 | Deterministic stochastic training and **exact stochastic resume**, end to end — `examples/native_dropout_training.py` plus `tests/test_native_dropout_training.py`, and **no new capability**. The model is the smallest one carrying all four TensorForge-owned state families at once: `NativeLinear(4, 8, seed=0)` → `NativeBatchNorm1d(8)` (persistent running buffers) → `NativeReLU` → `NativeDropout(p=0.5, seed=20240707)` (a registered `NativeGenerator`) → `NativeLayerNorm(8)` → `NativeLinear(8, 3, seed=1)`, over **raw logits** with `NativeCrossEntropyLoss` and `NativeAdam(lr=0.05)` (moments plus per-parameter step counters). The task is twelve four-feature samples over three classes computed from an **explicit arithmetic formula** — every value a quarter or an eighth, so exact in float64 — in three fixed batches of four, on a schedule that is a **pure function of the training step** (`step % 3`). Nothing is shuffled, generated randomly, augmented, loaded, or downloaded; neither NumPy's global RNG nor Python's `random` is touched. **Two uninterrupted runs are bit-identical** across the loss sequence, every parameter, both running statistics, the whole optimizer state, the generator state, the final training logits, and the final evaluation output. An **interrupted** run — checkpointed after 7 *completed* steps, deliberately mid-cycle in the batch schedule, with the interrupted model, optimizer, and generator **released before the resume begins**, so the archive is the only continuation boundary — reloads into a completely fresh model/optimizer/generator set built with a *different* Dropout seed and reproduces the uninterrupted run by **exact equality** on every one of those items. Two negative controls make the proof load-bearing rather than decorative: a resume that restores all four state families but restarts the batch schedule at 0 **diverges**, and one that restores everything but re-seeds the generator **diverges**. Evaluation is proved **state-neutral**: repeated eval passes leave `calls` bit-identical, produce identical outputs, restore the caller's training mode, and leave a probed run's loss sequence exactly equal to an unprobed one's — no gap in the random stream. A separate **throwaway** reload (so the resumed run is untouched) matches the restored `NativeDropout`'s next output against `NativeTensorCore.dropout_forward` at the exact restored `(seed, call_index)` and proves it consumes exactly one call; the module's private multiplier mask is never exposed, the Core supplies a reference *output*. **External loop progress is carried explicitly**, as validated JSON metadata (`{"training_step": k, "next_batch_index": k % 3, "lr": ...}`), because checkpoint v2 captures TensorForge-owned state and **not** data-loader position, batch order, shuffle state, epoch counters, scheduler state, Python's `random`, or NumPy's global RNG — and the example's `validated_progress` **raises** on a missing field, a `bool` where an `int` belongs, an out-of-range step, or a `next_batch_index` disagreeing with the schedule, rather than silently restarting from step 0. Reproducibility is exact **for the state actually captured**; full-program determinism is not claimed. Adds **no** C++, C ABI symbol, ctypes declaration, `NativeTensorCore` method, autograd operation, module, export, schema field, checkpoint version, benchmark, or registry value, and defines **no public training API** — none of the example's helpers is exported. **`"dropout"` deliberately stays in `UNSUPPORTED`** | **Complete** (the end-to-end resume proof — no capability) |
| G8 | An honest native Dropout benchmark — `benchmarks/benchmark_native_dropout.py`. **Characterization only: no speed is asserted anywhere, and no timing number is committed.** Thirty-five cases in eight families (`baseline`, `core_reference`, `size_scaling`, `layout`, `probability`, `tensor_operation`, `module`, `training_step`). The stateless Core is timed against an **exact bit-for-bit** vectorized NumPy implementation of the same locked derivation; the `NativeTensor` and `NativeDropout` cases are `native_only` and publish **no ratio**, because no NumPy expression owns a generator call transaction, native ownership, and an autograd graph. Correctness is gated **before** timing everywhere — a prologue pins the harness's reference to the committed G2 known-answer vectors and then pins the native kernel to the same vectors — each stochastic case's generator consumption is verified exactly, evaluation and `p == 0` are proved to consume none, and an untimed lifecycle pass returns native live storage exactly to baseline. `--case`, `--family`, `--warmup`, `--repetitions`, `--smoke` (`--quick`), `--json`, and `--json-out`; **no result file unless `--json-out` names one**. Adds **no** capability, kernel, C ABI symbol, ctypes declaration, Core method, operation, module, export, schema field, or registry value, and changed no runtime file | **Complete** (measurement only — no speed assertion, no committed timing number, and no CI timing threshold anywhere) |
| G9 | Cross-cutting Phase-G integration — `tests/test_native_phase_g.py`, one test-only model carrying every registered state family at once (`NativeConv2d` → `NativeBatchNorm2d` → `NativeReLU` → `NativeMaxPool2d` → `NativeDropout` → `NativeFlatten` → `NativeLinear` → `NativeBatchNorm1d` → `NativeReLU` → `NativeLayerNorm` → `NativeDropout` → `NativeLinear`, raw logits into `NativeCrossEntropyLoss`, with the two Dropout layers sharing **one** registered generator). It proves the interactions: **four** saved-resource families in one graph released exactly once; exact version-2 resume into a completely fresh model/optimizer/generator set, with a negative control that diverges; the generator-topology matrix (shared, independent, equal-valued-but-distinct, renamed, missing, extra) with every mismatch rejected **before** any state family changes; the shared stream consuming indices in execution order against the G2 Core; evaluation consuming none anywhere; `p == 0`; non-contiguous NCHW and strided views; whole-state rollback at **every** commit position; four deterministic concurrency cases proving the participating transactions serialize (ordinary concurrent *training* stays explicitly unclaimed); a Phase A–F regression matrix; and native live storage returning exactly to baseline across success and failure cycles. Adds **no** capability, operation, kernel, C ABI symbol, ctypes declaration, Core method, module, export, schema field, example, or benchmark, and changed no runtime file | **Complete** (integration evidence only — `dropout` stays in `UNSUPPORTED` until the G10 closure) |
| G10 | Phase-G closure, and `"dropout"` finally leaving `UNSUPPORTED`. The §18 validation matrix executed with observed results: fresh Windows **Release** and **Debug** builds (Visual Studio 17 2022, MSVC 19.44.35228.0, CMake 4.4.0), each configured out-of-source outside the repository with `-DTF_BUILD_TESTS=ON` and each passing **11/11 CTests** (0.86 s and 0.94 s) with **zero project compiler, linker, and CMake warnings**, the Debug library written elsewhere so the active runtime stayed the Release DLL (proved by size and linked CRT). A fresh Clang **18.1.3** `-DTF_SANITIZE=address,undefined` build in WSL2 Ubuntu 24.04.4 with **instrumentation proved rather than assumed** — `nm -D` shows 22 `__asan*` and 14 `__ubsan*` dynamic symbols beside the **51** exported `tf_*` symbols, and the library refuses to load without the sanitizer runtime. Under `halt_on_error=1:abort_on_error=1:detect_stack_use_after_return=1:detect_leaks=1` and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`: **11/11 sanitized CTests**, **3,166 sanitized Python tests** across 43 Phase-G and dependency suites, the G7 example reproducing its exact resume, and the G8 benchmark smoke path passing every correctness gate — all with **zero ASan and zero UBSan diagnostics**. A practical LeakSanitizer lifecycle returned native live storage **exactly to baseline (0 → 0)**, and its remaining process-exit allocations (926,478 bytes in 831 allocations) contain **no TensorForge frame** — only CPython, libc, NumPy, `_ctypes`, and the ASan runtime — with **no suppression file added**. Then, and only then, `"dropout"` was removed from `UNSUPPORTED`. **Validation, documentation, and one registry line** — no C++, CTest, C ABI symbol, ctypes declaration, Core method, operation, module, export, schema field, checkpoint version, example, or benchmark changed | **Complete** (the phase closure — the boundary move was its last act) |

So the native line now has the Core kernel, the differentiable
`NativeTensor.dropout` operation, the `NativeDropout` module, a checkpoint
that persists its random stream, a demonstrated exact stochastic resume —
and, since G10, the **capability** to match. `UNSUPPORTED` read
`("float32", "cuda", "amp")` when Phase G closed — that is the historical
record of what G10 left, and it is stated as history because a later phase
legitimately moved one of those names (see below). The gap through G0–G9
was deliberate: the registry reports a *closed, validated* capability, and
a capability whose value is exact reproducibility is not finished until
reproducibility has been demonstrated under fresh Release and Debug builds,
the sanitizers, and a checkpoint that can actually persist the stream. All
of that has now been demonstrated.

The claim stays narrow, and is worth stating precisely: **native Dropout
is supported in TensorForge's experimental native CPU backend.** That is
not a statement about the stable framework (`tensorforge.nn.Dropout` has
always been its own separate NumPy implementation). `cuda` and `amp` remain
unsupported. `float32` did **not** remain unsupported — Phase I milestone
**I9** moved it, on the same discipline Phase G used here, and
`tf_core_dropout_forward` became dtype-general at I7 with its exact ABI
shape unchanged and the random derivation untouched. There is still no
generic `rand`/`randn`/Bernoulli/sampling API, no global or process-wide
random state, and no `Dropout2d`/`Dropout3d`.

One contract detail is recorded rather than glossed: the design's
**empty-tensor** row (`count == 0` draws nothing and consumes one call)
is implemented in the kernel and the C ABI, but the native tensor
representation rejects zero-size dimensions outright, so no empty
`NativeTensorCore` can be constructed to exercise it from Python today.
G2 proves the case at the two layers where it is reachable and pins the
representation's limit with a test.

## Phase H — native CPU performance and runtime efficiency, **complete (H0–H10)**

**Phase H is **complete**: milestones H0 through H10 have all landed, and it is the latest *completed* phase. H10 re-measured the whole phase against a reconstructed and verified H0 baseline (52 cases, **zero checksum mismatches** — every figure compares implementations that produced bit-identical results), resolved the acceleration gate as three documented rejections with measurements (SIMD, threading/OpenMP, BLAS), assessed `tf_core_narrow_backward` and the small-operation boundary floor and implemented neither, ran the full Release/Debug/Linux/sanitizer/lifecycle matrix, and closed the phase. **Every shipped training workload is 1.50×–3.89× faster than at H0**, matmul 4.71×, Conv2d kernels 2.59×–4.64×, reductions 3.78×–5.06×, with no allocation count or memory peak raised anywhere — and across the whole phase **no capability, dtype, device, registry value, public API, checkpoint field, or checkpoint version moved**, with exactly **one** C ABI symbol added (`tf_storage_create_uninitialized`, at H1): 51 → **52**.**
Its architecture contract is
[native_cpu_performance_design.md](native_cpu_performance_design.md).

H0 is an **architecture, profiling, and baseline** milestone. It shipped
the contract, the unified measurement harness
`benchmarks/benchmark_native_cpu_performance.py`, that harness's
behavioral contract tests, and documentation reconciliation — and
**nothing else**.

**Milestone H1 is complete**, and is the first Phase-H change to
production code: the explicit output-allocation contract, which removed
the redundant zero-fill from output storage a kernel *provably*
overwrites in full. H1 is an allocation change and nothing else — it is
**bit-identical**, adds exactly one C ABI symbol
(`tf_storage_create_uninitialized`, taking the library from 51 exported
`tf_*` symbols to **52**) beside the unchanged
zero-initializing default, exposes **no** public empty-tensor API and
**no** poison control or other runtime allocation-content switch, and
moved **no** capability, dtype, device, registry value,
checkpoint field, or checkpoint version. `sum`/`mean` and
`narrow_backward` are explicitly rejected and keep a zeroed
destination. The per-kernel audit, the poison-test methodology, the
bit-identity proof, and the measured results (including the inconclusive
and negative ones) are section 16.1 of the design.

**Milestone H2 is complete**, and is the first Phase-H milestone to
change how a numerical kernel executes: the production matmul's loop
order. `tf_core_matmul` now ships two compute paths behind the same
unchanged export — the pre-H2 `i`-`j`-`k` triple loop, retained verbatim
as the **generic reference path**, and an `i`-`k`-`j` row sweep over four
destination rows at a time — and chooses between them inside the kernel
from the stride metadata it already receives. **Cache blocking was
measured against 22 blocked variants and rejected**; the negative result
is recorded in the design rather than the milestone title being honored
for its own sake. H2's numerical agreement with the reference path is a
four-part contract rather than a blanket bit-identity claim — identical
accumulation order, **bit identity on every non-NaN result**, NaN-class
equivalence, and **NaN payload bits deliberately outside the contract**
— and it preserves H1's
uninitialized-output contract on both paths, adds **no** C ABI symbol —
the count stays **52** — and exposes **no** kernel selector, block-size
setter, dispatch tracer, environment variable, or any other dispatch
control. The exact preconditions, the fallback conditions, the
accumulation-order proof, the block-size and threshold evidence, and the
measured before-and-after (including the shapes where the change is not
measurable) are section 16.2 of the design.

**Milestone H3 is complete**, and is the first Phase-H milestone that is
**Python-only**: no C++, no C ABI symbol, no ctypes declaration, and no
kernel changed, so the export count stays **52**. It removed redundant
metadata *re-validation* from the path to a kernel — before H3 one
`shape_info` call ran `_as_int_tuple` **four** times over a tuple that
was fully validated after the first pass and computed the row-major
strides **twice**, which instrumented counts put at **815**
`_as_int_tuple` calls per MLP training step. H3 introduced one
normalization boundary (`_normalized_layout`, performing exactly the
checks `shape_info` always performed, in the same order and with the same
messages), private `_checked` primitives that derive the strides, element
count, and contiguity without re-validating, a private
`NativeTensorView._from_validated` constructor that skips *only* that
normalization while sharing the public constructor's `_bind` — so the
storage open check and the full reachable-offset bounds check still run,
and the element count and contiguity flag are derived inside rather than
passed in, making an inconsistent pair unrepresentable — and lazy,
**read-only**, per-view `int64` layout arrays for the strided C ABI.
That memoization cannot go stale: a view's layout is assigned exactly
once and every layout-changing operation returns a *new* view, so no
invalidation is required and none exists. **No validation was removed**,
no global cache or interning was introduced, and **no public API** of any
kind was added — no cache control, statistic, reset, profiling counter,
dispatch selector, or environment variable. Measured: view construction
3.2x, `_as_int_tuple` per MLP step 815 -> 149, an MLP training step
1.43x, a CNN step 1.29x, a normalized step 1.51x — and **no measurable
change on large kernel-bound matmul or elementwise work**, published as
such. The immutability proof, the cached-versus-not-cached decisions, the
object footprint, the validation placement table, and the full measured
before-and-after are section 16.3 of the design, and the focused Clang
ASan/UBSan validation of the metadata and view paths — instrumentation
proved, **zero ASan and zero UBSan diagnostics**, a negative control that
makes the detector demonstrably able to fail, live native storage back to
baseline, and **no TensorForge frame in the LeakSanitizer report** — is
section 16.3.15.

**No other kernel has been optimized.** Every kernel in `cpp/src/`
outside `matmul.cpp` is exactly the deliberately plain reference loop it
was after Phase G, and no SIMD, threading, OpenMP, BLAS, memory pool, or
scratch workspace exists anywhere. H0 changed no C++, no C ABI symbol, no
ctypes declaration, no `NativeTensorCore` method, no autograd operation,
no module, no loss, no metric, no optimizer, no export, no capability
registry, no dtype, no device, and no checkpoint format; H1, H2, H3,
H4, H5, H6, H7, and H8 moved no capability either. When Phase H closed,
`UNSUPPORTED` read `("float32", "cuda", "amp")`, `SUPPORTED_DTYPES` read
`("float64",)`, `SUPPORTED_DEVICES` read `("cpu",)`, and the native
checkpoint format was `tensorforge.native_checkpoint` version **2** with
versions **(1, 2)** supported. Those are stated as Phase H's *record*
rather than as today's values, because Phase I has since moved two of them
deliberately — the checkpoint to version **3** at I8 and the dtype registry
to `("float64", "float32")` / `("cuda", "amp")` at I9 — and a closure
record that silently absorbed later changes would stop being a record.
`SUPPORTED_DEVICES` has not moved and is still `("cpu",)`. That was true through H9; **at H10 the phase closed, so Phase H —
Native CPU Performance and Runtime Efficiency — is now the latest
*completed* phase (H0–H10)**, having moved no capability, dtype, device,
registry value, export, checkpoint field, or checkpoint version at any
milestone.

| Milestone | What it shipped | Status |
|---|---|---|
| H0 | The Phase-H architecture contract ([native_cpu_performance_design.md](native_cpu_performance_design.md)): why CPU efficiency precedes CUDA and dtype expansion; the measured bottleneck evidence, separated into *directly measured*, *strongly source-evidenced but not fully measured*, and *unconfirmed hypotheses*, with the minimal instrumentation a later milestone would need where H0's observability could not settle a question; the representative workload families and the shape-selection rules for the smoke, full, and profiler configurations; the timing, warm-up, repetition, and statistics methodology; the correctness-before-timing rule; the exact-versus-tolerance policy and the floating-point **accumulation-order** policy, whose default is that Phase H preserves the existing order bit-for-bit and whose five-condition escape hatch requires every existing exact-resume proof to be re-established; the determinism policy; the optimized-contiguous-dispatch, strided-fallback, and **retained generic reference path** rules; the invariants Phase H may not weaken; the allocation, scratch-workspace, SIMD, threading, and optional-BLAS decision criteria (all four currently **rejected on evidence**, with the criteria recorded rather than an answer invented); the cross-platform and sanitizer requirements; the conditional H0–H11 ladder (H0–H10 as drafted; H5 inserted the copy and mutation-transfer milestone and pushed reduction execution to H6); the explicit non-goals; the closure requirements; and the recorded adopt/adapt/reject decision for every candidate performance design the phase considered. Plus `benchmarks/benchmark_native_cpu_performance.py`, the unified baseline harness — 24 cases at H0 across 12 workload families (H3 later added two more, taking it to 26, and H5 two more again, taking it to 28), measured across up to nine declared implementation layers (`numpy`, `stable_tensorforge`, `raw_kernel`, `raw_kernel_tiled`, `tensor_core`, `native_tensor`, `native_tensor_graph`, `backward`, `optimizer_step`, `training_step`), with a correctness gate that runs **before** the timing helper is ever reached, honest reference labelling (a case with no honest equivalent publishes **no ratio at all** and says why), `--smoke` / `--json` / `--case` / `--workload` / a focused `--profile CASE` mode, deterministic seeded inputs, explicit cleanup with no reliance on garbage collection, fresh state per training-step repetition, an explicit reset generator for the Dropout case, and **no result file of any kind**. Plus `tests/test_native_cpu_performance_benchmark.py`. **No numerical capability, no optimization, no registry move** | **Complete** (architecture, profiling, and baseline only — nothing was made faster) |
| H1 | **The explicit output-allocation contract.** Redundant zero-initialization removed from output storage a kernel *provably* overwrites in full. One new production C ABI symbol, `tf_storage_create_uninitialized`, sharing one file-local body with `tf_storage_create` so the two cannot drift apart: identical size validation, zero/negative rejection, fault injection, allocation-failure handling, error state, handle shape, ownership, destruction, and live-storage accounting, differing **only** in the buffer initial contents. The zero-initializing path stays the default and is byte-for-byte unchanged. On the Python side, one private keyword on `NativeStorage.__init__` (so both allocation kinds pass through the one constructor every live-storage hook wraps), plus the private `NativeStorage._uninitialized` / `NativeTensorCore._uninitialized` helpers — **no** public `empty` API, no registry capability, no stable-framework surface. Every enabled call site opts in explicitly against a per-kernel audit table (design 16.1.2); there is no global switch, environment variable, heuristic, memory pool, or scratch arena, and a failed precondition means the safe path. **Enabled:** the elementwise/unary family across all three traversal paths, `relu`, `sqrt`/`reciprocal`/`exp`/`log`, `contiguous_copy`, `matmul` (at H1 it accumulated into a *local* register and never read the destination; **H2 restated the reason for the optimized path it added** — the row sweep's `k == 0` pass assigns every element of every row before anything accumulates into it, so the destination is still never read before it is written, and both paths now carry poison proofs), `relu_backward`, `softmax`/`log_softmax`, both cross-entropy directions, `conv2d_forward`, **both convolution gradients and the pooling gradient** (their kernels zero their own whole span before accumulating, so the caller fill was pure duplication), `maxpool2d_forward` values and winners, both Dropout destinations, `from_array`, and `full`. **Rejected:** `sum`/`mean` (it accumulates into its output, so the zero is the additive identity) and `narrow_backward` (it writes only the narrowed region, and the untouched zeros *are* the gradient) — both pinned by negative-control tests. Completeness is proved by **deterministic poison** tests, separately from ASan and UBSan, which do not detect uninitialized-*value* reads (MemorySanitizer would, but it needs an instrumented libc and CPython this project does not have, so no MSan result is claimed). The poison is injected **exclusively by test infrastructure around the allocator**: the suite wraps the private `NativeStorage._uninitialized` helper — the single funnel every uninitialized allocation passes through — lets the real constructor allocate, fills the returned storage with a quiet NaN or a large nontrivial finite pattern through the ordinary `fill` primitive, and hands **that same storage** to the real operation, so the pattern is in place after the real allocation and before the real kernel. **The shipped library and the installed Python backend contain no poison control whatsoever** — no exported hook, no thread-local flag, no environment variable, no global mode — and a scope test asserts that against the loaded image's own export table. Four negative controls prove the detector can actually fail, including a real complete kernel given a deliberate hole. Bit-identical: every enabled operation and a complete eight-step training run are compared element-wise against the zero-initializing allocator, with the uninitialized side poisoned so a hole cannot match by luck. No capability, dtype, device, registry value, checkpoint field, or checkpoint version moved, and `tf_storage_create_uninitialized` is the only added export (51 → **52** exported `tf_*` symbols) | **Complete** |
| H2 | **Native matmul memory access.** `tf_core_matmul` ships two compute paths behind the same unchanged export. `tf::matmul_generic_strided` is the pre-H2 `i`-`j`-`k` triple loop kept verbatim — the **retained generic reference path** (design 8.3), shipped, reachable through ordinary production dispatch, and the oracle every optimized result is compared against. `tf::matmul_row_sweep` is the optimized path: `i`-`k`-`j` over `MATMUL_ROW_BLOCK` = 4 destination rows at a time, so the innermost loop walks a row of the right operand and a row of the output sequentially rather than walking a column. **Preconditions, all read from metadata and all required:** the right operand's column stride is exactly 1, the inner dimension is non-empty, and the result has at least `MATMUL_MIN_COLUMNS` = 8 columns. **Everything else takes the generic path** — a transposed right operand, any other non-unit column stride, a narrower result, an empty inner dimension — and that is a design choice rather than a gap, because the `i`-`j`-`k` order is the better one when the right operand's *rows* are the unit stride. A transposed *left* operand beside a row-major right one (which is `db = a.T @ upstream` in the matmul backward) does qualify. Selection is total, pure, deterministic, side-effect free, and independent of pointer values, alignment, wall time, environment variables, and CPU-feature probes; a failed precondition is a fallback, never an error. **Cache blocking was measured and rejected**: 22 blocked variants were timed against unblocked row sweeps across 25 shapes, and the unblocked sweep was faster at every non-trivial size (5.50x versus 3.33x at 384 cubed), so H2 shipped the simpler superior design and published the negative result. Both constants are compile-time values in a shipped header — no autotuning, no runtime probe, no stored machine-specific measurement. **A four-part numerical contract, not a blanket bit-identity claim.** (1) Accumulation order is preserved exactly: for each output element the row sweep starts from the same 0.0 and takes the same products in the same ascending `k` order, with the `0.0 +` on the `k == 0` pass written out deliberately because `0.0 + (-0.0)` is `+0.0`. (2) **Every non-NaN result is bit-identical** — signed zeros, infinities, denormals, the smallest normal and the largest finite magnitudes included — asserted as raw IEEE-754 bit patterns, not tolerances, in the C++ CTest across a 1-65 dimension sweep including primes and both sides of every boundary, in Python across the Core, `NativeTensor`, the autograd node, `NativeLinear`, and both optimizers, and in the harness gate before any timing. (3) **NaN-class equivalence**: NaNs occur in exactly the same positions on both paths and are always quiet; neither path produces a signaling NaN. (4) **NaN payload bits are outside TensorForge's numerical contract** and may differ — measured at 162 of 208 in the MSVC Release build and 0 of 208 in the Debug and Clang builds, with ten source-level formulations tried and only the `i`-`j`-`k` order H2 replaces able to reproduce the reference's payloads. Deterministic training and exact checkpoint resume stay bit-identical for supported finite workloads, which part (2) covers completely. **H1 compatibility** holds on both paths for different reasons — the generic path never reads the destination, and the row sweep assigns every element of every row in its `k == 0` pass before anything accumulates into it — proved by poison tests over both paths with both patterns, a bit-identity check across NaN-poisoned, finite-poisoned and zeroed destinations, and a partial-write negative control. The pre-existing raw `tf_matmul_tiled` was inspected and **not** adopted: it takes plain contiguous buffers rather than storage handles, carries no guard or error contract, and zeroes its destination before accumulating — a full extra write pass, which is exactly what H1 removed. It stays as the standing benchmark experiment on no production path. **No C ABI symbol added** (still **52** exported `tf_*`), no kernel selector, block-size setter, benchmark hook, dispatch tracer, reference-kernel selector, CPU-feature control, or environment variable; no threading, SIMD, OpenMP, BLAS, memory pool, or scratch workspace; no capability, dtype, device, registry value, checkpoint field, or checkpoint version moved | **Complete** |
| H3 | **Native metadata and dispatch efficiency — Python-only.** Repeated Python-side metadata normalization removed from the path to a kernel, with **no C++, no C ABI symbol, no ctypes declaration, and no kernel** touched (export count stays **52**). The measured cause was redundant *re-validation*: one `shape_info` call ran `_as_int_tuple` **four** times over a tuple fully validated after the first pass and computed the row-major strides **twice**, while `NativeTensorCore.zeros` validated the caller's shape a second complete time — **815** `_as_int_tuple` calls per MLP training step, 604 per `NativeAdam` step, instrumented by test-local monkeypatching with **no production counter**. Three pieces. (1) **One normalization boundary**: the private `_normalized_layout` performs exactly the checks `shape_info` always performed, in the same order and with the same messages, and normalizes the shape once; the derived strides, element count, and contiguity come from private `_checked` primitives that validate nothing because nothing is left to validate, and each public helper (`row_major_strides`, `numel`, `reduce_shape`, `broadcast_shapes`) is now its own validation plus the matching primitive, so the two cannot disagree. (2) **Two view constructors, one binding**: the public `NativeTensorView(...)` normalizes; the private `_from_validated` skips **only** that normalization; both funnel through a shared `_bind` that still performs the storage open check and the **full reachable-offset bounds check** — not skipped, because bounds depend on the storage, not the metadata. The element count and contiguity flag are **derived inside** the private constructor rather than passed to it, so an inconsistent pair is unrepresentable — which is why H3 ships a separate private constructor rather than a misusable `validated=True` flag. (3) **Per-view layout arrays**: the `int64` shape/stride arrays the strided C ABI takes, memoized **lazily** and **read-only**. Staleness is impossible by construction, not by invalidation: a view's layout is assigned exactly once, in `_bind`, and `reshape`/`transpose`/`T`/`narrow` all return *new* views, so no invalidation is ever required and **none exists**. Nothing global was introduced — no shape cache, stride interning, weak-reference machinery, or thread-local state — and **no validation was removed**: every rejection still happens with the same exception type, the same message, and the same shape-then-strides-then-offset ordering, asserted by feeding the constructor metadata with two and three simultaneous faults. Cold object footprint is **byte-identical**; a view that actually takes a strided path costs +328 bytes, and in a full MLP step only **5 of 134** views ever populate it. Measured: `shape_info` 2.6–4.5×, view construction 3.2×, `_as_int_tuple` per MLP step **815 → 149**, a one-element allocation 2.1×, `reshape` 3.1×, a small `add` 1.56×, `NativeAdam` on a small MLP 1.42×, an **MLP training step 1.43×**, a **CNN step 1.29×**, a **normalized step 1.51×** — and **no measurable change on large kernel-bound work** (384³/512³/128³ matmul, 256² elementwise, 128² reduction all inside their own spread), published as such. **No public API added** of any kind: no cache control, statistic, reset, profiling counter, dispatch selector, or environment variable. No capability, dtype, device, registry value, checkpoint field, or checkpoint version moved | **Complete** |
| H4 | **Native optimizer step efficiency — Python-only.** The fixed native allocation and call count of `NativeAdam.step()` reduced, and `NativeSGD.step()` changed only where the evidence supported it, with **no C++, no C ABI symbol, no ctypes declaration, and no kernel** touched (export count stays **52**). It is the first Phase-H milestone whose subject is a *training-stack* component rather than the tensor runtime. B4's counts were **re-instrumented on the current post-H3 code** rather than taken from H0, and H0's figure was confirmed exactly: **27 native storage allocations per parameter per Adam step**, of which **ten are one-element** — eight broadcast scalar coefficients (`beta1`, `1 - beta1`, `beta2`, `1 - beta2`, both bias-correction terms, `eps`, and `lr`; design 3.2 said six, and `eps` and `lr` were the two it missed) plus the two `reciprocal` outputs taken on one-element tensors — with `NativeSGD` at five per parameter, and **8 of Adam's 13 binary operations** taking the broadcasting path rather than the contiguous fast path. Three changes. (1) **The step's scalar coefficients are built once per step, not once per parameter**: a private per-step `_StepConstants` holder builds each on first use, keyed by `(dtype, device)` so it never assumes one dtype exists, and hands the same read-only core to every later parameter; the two bias-correction terms are cached per step *counter*, so steady-state training builds one pair while a parameter that skipped earlier steps legitimately gets its own. The holder allocates nothing until the first entry asks for a coefficient — so a step with no active parameter allocates nothing at all — is released before the commit begins, and is **never stored on the optimizer**, so no scalar survives a step, enters `state_dict()`, reaches a checkpoint, or has to be released by `close()`. `NativeSGD` does the same for its single `lr` scalar, the only change its evidence supported. (2) **The bias-correction reciprocal is evaluated in Python**, removing one allocation and one kernel call per coefficient per parameter. This is an **exact substitution, not a reassociation**: the kernel literally is `double op_reciprocal(double x) { return 1.0 / x; }`, a Python `float` and a C++ `double` are the same IEEE-754 binary64 value, and IEEE-754 requires division to be correctly rounded, so there is exactly one possible result — proved over **20,000+ values** spanning the full exponent range, ±0, ±∞, the smallest subnormal, the largest finite magnitude, and every `1 - beta ** t` the optimizer actually forms, compared on **raw `uint64` bit patterns** with zero mismatches. (3) **Temporaries are released at their last use** rather than all together at the end of the staged expression. **Bit-identical, with no carve-out of the kind H2 needed**: no accumulation order, operand position, or kernel changed, so NaN payloads match too. The **pre-H4 composition is retained in the test suite** as a literal transcription executed natively, and every equality is against that rather than a NumPy re-derivation — 60 shape/step/hyperparameter combinations for Adam (including `beta = 0`, betas at `0.99999`/`0.9999999` with `eps = 1e-30`, and `lr = 1e10`), a six-step run over four mixed shapes, and four SGD learning rates from `1e-9` to `1e12` — while a separate test pins the **exact operation sequence** a staged entry issues so a future reorder or fusion fails loudly. **The two-phase contract is untouched**: validation is still four complete passes in the same order with nothing moved behind a mutation; stage mutates no parameter, moment, counter, version, or gradient; the commit is still **one `copy_value_` and exactly one version increment per updated parameter**; gradients are read and never written, by identity, value, and storage identity; and the documented per-entry commit boundary is *tested* by injecting a `copy_value_` failure rather than assumed infallible. Measured by alternating pre/post **subprocess** rounds so drift affects both arms equally, 366 samples per case: **1.58×** at (128, 128), **1.54×** at (256, 256), **1.48×** on a four-parameter MLP with a 256² weight, 1.21–1.22× on a small MLP, 1.15× on a first step; a large MLP training step 1.23×, a small one 1.15×, a normalized step 1.13×, a CNN step 1.09×; and in the shipped harness `adam_step` 1.25×, cutting the gap against `tensorforge.optim.Adam` from **23.8× to 19.7×**. Reported just as honestly: **a (512, 512) parameter is neutral** (1.02×, memory-bandwidth-bound), the **Dropout training step is neutral** (0.99×), and **NativeSGD is neutral-to-slightly-positive** (1.03–1.07×) with one 0.88× row identified as **noise** by a focused re-measurement whose post minima were lower in every pair — and the machine's control-case noise band is stated at **0.84×–1.26×** so no reading inside it is mistaken for a result. **Memory moved with time, not against it**: peak live transient bytes during one Adam step fell **2.6–3.0×** (1,966,160 → 655,424 for a (128, 128) parameter) and per-parameter allocations 27 → **17**, so a four-parameter model allocates **76 instead of 108**. **Six alternatives measured and rejected**, each with its reason recorded: scalar materialization (faster below ~32K elements, slower above, and it would regress the harness's own profile configuration while adding a parameter-sized buffer per scalar operation); same-shape stride-0 views (identical kernel arguments by construction, but four NumPy layout arrays per call where the broadcast path builds three); adopting the staged core instead of `copy_value_`; giving `_native_copy` a `contiguous_copy` implementation (it would stop normalizing `-0.0` to `+0.0`, a real observable change in a helper shared far beyond the optimizer); a persistent per-optimizer scalar cache (the hidden scratch tensor the design forbids); and reassociating the update to fold scalars together (a floating-point order change that would break every exact-resume proof). **No public API added** of any kind: no cache control, statistic, reset, profiling counter, dispatch selector, failure toggle, or environment variable. No capability, dtype, device, registry value, checkpoint field, or checkpoint version moved | **Complete** |
| H5 | **Native copy and mutation-transfer efficiency.** The native line's value-transfer primitive replaced: `_native_copy` was `zeros(shape) + core` (two allocations, a zero-fill pass, and an elementwise-addition pass) and is now the E3.1 native identity gather `contiguous_copy()` (one uninitialized allocation, one pass). A complete inventory found **ten** call sites of that one helper — `copy_value_` staging, both `state_dict()` snapshots, both `load_state_dict()` stagings, both BatchNorm running-statistic commits, and the reshape/transpose/unbroadcast gradient materializations — all of them pure value transfers, and all ten were enabled; `_broadcast_back` was **rejected** because it is a genuine broadcast expansion rather than a copy, and `sum`/`mean` and `narrow_backward` keep their zeroed destinations for H1's unchanged reasons. The semantic question H4 refused to decide in passing was decided by measurement over a fixed 18-pattern IEEE-754 sweep: **exactly three** patterns moved — the addition normalized `-0.0` to `+0.0` and quieted both signs of signaling NaN — while every other pattern, including every NaN payload, was already identical (nothing differed there), so **H2's matmul NaN-payload carve-out does not generalize to copies**. The pre-H5 behavior was accidental and inconsistent, not contracted: `NativeParameter(source)` construction, `detach()`, and the `to_numpy()`/`from_array` boundary always preserved `-0.0`, while `copy_value_` documented the same thing and did not. H5 states the narrowest coherent rule — **a value transfer reproduces its source's bits exactly; an operation follows IEEE arithmetic** — and changed no operation's arithmetic anywhere. One C++ change, and **no ABI change**: a second *traversal* inside the unchanged `tf_core_contiguous_copy` export, chosen by `tf::copy_prefers_contiguous` — hidden-visibility C++ in a new internal header, total, pure, allocation-free, a function of layout metadata alone, never of a pointer value, alignment, clock, environment variable, or CPU-feature probe — sweeping a row-major source with the flat pointer loop and falling back to the retained odometer otherwise. **No numerical carve-out is needed**, unlike H2: the identity map performs no arithmetic, so the two traversals are bit-identical *by construction*, proved at the C++ level by a new dependency-free CTest (13 to 14). Nothing became in-place — every call site still stages an independent materialization before adopting it — so self-copy, a source that views the destination's own storage, a square parameter's own transpose, sibling views, and duplicate parameters across optimizers all stay correct, and no `memcpy` is used anywhere. Parameter identity, storage replacement, gradient retention, the one version increment per commit, the F1 state transaction, checkpoint atomicity, and exact resume are unchanged; H1's full-write contract is proved on both traversals by poison injected purely by test infrastructure around the allocator, with a negative control. Measured by alternating pre/post subprocess rounds (control band 0.96x-1.05x) and by a separate pre-H5-library A/B: the traversal alone **2.5x-5.5x** on contiguous sources and **0.94x-1.02x** on transposed ones (the unchanged odometer, the design's own control); `copy_value_` **2.14x** at (512, 512), optimizer `state_dict()` 2.40x, `load_state_dict()` 1.69x, `NativeSGD.step()` 1.15-1.31x. Reported just as honestly: **`NativeAdam.step()`, every training step, the BatchNorm running update, and copies below ~16 K elements are all neutral**, the last because two `int64` layout arrays cost ~1.1 us each at the ctypes boundary — measured, attributed, and left to a later dispatch milestone rather than paid for by weakening H3's validation. Allocations fell everywhere and **no measured peak rose**: `copy_value_` 2 to 1, module state 4 to 2, optimizer state 16 to 8, Adam 17 to **16** per parameter. Two harness cases added (26 to 28). The ladder was **reordered** here: reduction execution, drafted as H5, moved to H6. No public API, capability, dtype, device, registry value, checkpoint field, or checkpoint version moved; export count stays **52** | **Complete** |
| **H6** | **Native reduction execution efficiency.** Reductions were the last core family always paying the generic strided indexing cost. The pre-H6 kernel was re-read and re-measured rather than trusted from H0's or H5's summaries, and **decomposed**: at `(256, 256)` `axis=0` a `core.sum` costs 99.7 us of which the raw native call is **94.8 us — 95 %** (the C++ traversal alone ~92 %), while the entire Python wrapper — axis normalization, output-shape construction, write-stride construction, the H3-cached layout arrays, and the output allocation — is ~5 us. So unlike B3 this was unambiguously a compiled-loop problem. H6 reused the dispatch shape H2 and H5 each proved: one hidden metadata predicate, inside the existing export, **no new symbol**, the pre-milestone traversal retained. New `cpp/include/tf_reduction_internal.h` declares three hidden-visibility `namespace tf` functions that `cpp/src/reduction.cpp` implements — `tf::sum_generic_strided`, the **pre-H6 odometer retained as the shipped generic reference path** and the only path that can address a transposed, narrowed, non-unit-strided, or broadcast source at all; `tf::reduce_prefers_contiguous_blocks`, total, pure, allocation-free, a function of layout metadata alone (never a pointer value, alignment, clock, environment variable, or CPU probe), with a false answer a fallback and never an error; and `tf::sum_contiguous_blocks`, a flat walk over an `outer x mid x inner` factorization. The predicate accepts a reduction when the source strides are exactly the row-major strides implied by the shape (the same definition `NativeTensorView` uses, so the layers agree by construction), the reduced axes — those with a zero *write* stride — form **one contiguous run**, and the kept axes carry the output's row-major strides. Stride collapsing is **implicit and bounded, not a general layout compiler**; nothing is cached or interned; `keepdims` needs no special case because the kernel cannot observe it. **Per-output accumulation order is preserved exactly** and the source traversal order is not even reordered, with no reassociation, FMA, Kahan, pairwise, tree, parallel, or horizontal-vector reduction anywhere; the `inner == 1` branch's local accumulator is seeded **from the destination**, keeping the export's accumulate-into semantics identical on both paths. Signed zeros are proved as **raw bit patterns** across every position and both `keepdims` values, and the rank-0 branch is recorded precisely (a genuine addition against a zeroed destination, so a rank-0 `-0.0` sums to `+0.0`, exactly as before). **The NaN rule is H6's own, measured rather than inherited from H2**: identical NaN positions, every NaN quiet, signaling NaNs quieted identically, and **bit identity whenever at most one NaN enters an accumulation** — every case that occurs in practice — with payloads outside the contract only when two or more NaNs meet in one cell, asserted in neither direction. Four accumulation spellings (including one accumulating *through memory* exactly as the odometer does) all selected the same NaN and all differed from the odometer, so parity is unavailable at any spelling and the memory form was 1.2x-1.8x slower; the block path keeps the **first** NaN, the odometer the **last**, and the block path's choice is NumPy's. **H1's rejection of this destination stands** — both traversals read it, so it stays zero-initialized; Outcome B was rejected on measurement (2 KB of fill against 512 KB of reads) and on semantics, so H6 adds **no poison test** because it introduces no uninitialized destination. Measured against a pre-H6 library on identical `ctypes` calls with outputs proved bit-identical before timing, 15 alternating rounds, control band **0.90x-1.03x**: full reductions 1.19x to **3.96x**, 2-D axis reductions 3.24x to **6.37x**, and — unpredicted — 3-D/4-D reductions **8.60x-10.94x**, because the odometer's carry loop scales with rank. Layer level: `TensorCore.sum(axis=0)` 4.49x, `mean(axis=0)` 4.11x, NCHW `sum(axis=1)` **8.56x**, `NativeTensor.sum` 3.88x, `sum()` fwd+bwd 1.27x, the **convolution bias gradient 1.46x**, `_unbroadcast` 1.15x, softmax backward 1.14x, LayerNorm forward 1.16x; the NumPy gap on contiguous reductions closed from ~8-13x to **1.67x-3.75x** while the transposed control stayed 10.33x. Reported just as honestly: **every training step is neutral** (0.99x-1.03x), so H6 does not make training faster; **normalization is mostly neutral**, narrowing H7 rather than motivating it; **tiny reductions are neutral** below ~1,000 elements where the fixed ~7 us Python-plus-ctypes cost dominates; and a **real ~10 % regression on 2-D transposed `axis=0` fallbacks** (0.89x-0.93x over four 25-round runs) is published, with the 3-D fallback 1.04x-1.05x *faster* and the cause isolated to whole-translation-unit code layout rather than the extracted call. Memory moved not at all and it is asserted: one allocation per `sum` on both paths at every axis. Harness 28 to **31** cases; native CTests 14 to **15**. No exported symbol, no new translation unit, **no public control of any kind**, no SIMD/threading/OpenMP/BLAS/pool/workspace/fast-math, no multi-axis reduction, and `tf_core_narrow_backward` (the scatter dual) deliberately untouched. No public API, capability, dtype, device, registry value, checkpoint field, or checkpoint version moved; export count stays **52** | **Complete** |
| **H7** | **Native Python/C ABI boundary efficiency - Python-only.** The ctypes argument binding for this project's own layout metadata, with **no C++, no exported symbol, no kernel, no traversal, and no arithmetic** touched (export count stays **52**). **The ladder was revised here, and the revision is on the record**: H0's H7 slot was *composed-module cost*, explicitly conditional on a re-measurement after H1, H3, and H6; H6 made `mean` 3.9x-4.1x faster and moved the normalization modules almost not at all (normalized training step **1.03x**), so the condition was **not met** and that milestone was **dropped on evidence**, its proposal preserved verbatim in the design (16.7.0) rather than deleted. The slot was refilled from the same measurements: H3, H5, and H6 had each ended by deferring one identically named cost to "a later dispatch milestone". **The cost was decomposed, and the assumption that six kernels were involved was checked and found wrong**: all 52 exports are configured in one file (no other module in the repository imports `ctypes`), **57 argument positions are arrays**, and every one was bound as `numpy.ctypeslib.ndpointer` - which re-verifies array-ness, exact dtype, and contiguity at *every call*, then builds `obj.ctypes` and resolves it through `_as_parameter_`, at a measured **~2.1 us per array position**. `tf_core_add` on a 4x4 with three layout arrays cost **7.6 us**, of which **6.1 us** was the binding; the array-free `tf_core_add_contiguous` cost 0.9 us and is the control. Frequency was then counted: an MLP step makes 245 native calls carrying **101** array crossings, a normalized step 692 and **315**, a CNN step 242 and **104** - **20-23 %** of each step - and **~85 % of them are operation-local broadcast strides**, not the H3 per-view cache, which is the finding that decided the architecture. H7 ships **two bindings for two categories and deliberately not one blanket policy**: *data* positions keep the checked `ndpointer` binding - the seven public raw-buffer kernels, the `copy_from`/`copy_to`/`materialize` host boundary, and the cross-entropy **class labels**, which are int64 like layout metadata but stay checked because a label array's required length comes from the *logits*, a different object - while the **32 layout positions across 13 exports** take `ctypes.POINTER(ctypes.c_int64)`, fed by exactly two private producers: `NativeTensorView._native_layout_pointers()`, memoizing `data_as` over the **unchanged** H3 read-only NumPy arrays that remain the owning buffers, and `_layout_vector(values)`, building a fresh `(c_int64 * len(values))` for metadata belonging to one operation. **Nothing was weakened and one thing was strengthened**: ctypes still type-checks every call (a trusted position rejects a NumPy array of any dtype, a differently typed pointer or vector, a `c_void_p`, a list, an int, bytes, and a string), dtype/byte order/contiguity are established *by construction*, and **the length/rank invariant - the one `ndpointer` never checked, because the ABI sees only a pointer and an `ndim`** - became checkable for the first time, asserted per producer, per rank 0-4, and structurally over every strided call in a real workload. The one honest difference is `None` (rejected by `ndpointer`, a NULL for a typed pointer), closed structurally: both producers are total, no public API takes a metadata pointer (only **three** of the thirteen exports reject a null metadata pointer in C++ as well, so the producers being total is the load-bearing part). Ownership is NumPy's own guarantee - `data_as` attaches the array to the pointer - and `POINTER.from_address` was measured **faster** (0.9 us against 1.6 us) and **rejected outright** for producing a pointer with no owner; a cached ctypes vector was measured fastest of all and **rejected** for duplicating a view's layout description and losing H3's `writeable = False`, for ~2 % of a training step. Proved with the cyclic collector **disabled**: no reference cycle, no native storage kept alive, no usable pointer after close, no global pointer cache, and no `from_address`, `byref`, `addressof`, `id`, or weak-reference container anywhere in the module's code (enforced by parsing it). Binding configuration stays a **load-time act** - `argtypes`/`restype`/`errcheck` assigned only inside the two loader functions, asserted by locating each assignment's enclosing function - so nothing reconfigures a shared function object per call and **no thread-safety claim is broadened**. Measured against a **retained pre-H7 `cpp.py` driving the same Release DLL**, 11 alternating pre/post subprocess rounds, every case **bit-identical before either side was timed**, control bands 0.95x-1.10x (core) and 0.99x-1.05x (end to end): a 1-element `sum` **1.94x**, `to_numpy` 16x16 1.83x, `sum(axis=0)` 16x16 1.79x, a 4x4 `contiguous_copy` 1.73x, `narrow_backward` 1.73x, strided `relu_backward` 1.71x, scalar-broadcast `add` 1.67x, 4-D NCHW `sum(axis=1)` 1.54x, `sum(axis=0)` 256x256 1.35x - and end to end the **native Dropout step 1.32x, the normalized step 1.31x, `NativeAdam` at (32, 32) 1.31x, the CNN step 1.30x, the MLP step 1.28x**, `NativeLayerNorm` forward 1.23x, `NativeBatchNorm1d` eval 1.23x, `NativeSGD` 1.13x. **H7 is the first Phase-H milestone to move every training step** (H4 moved them 1.09x-1.23x; H5 and H6 were neutral on all of them). Reported just as honestly: **large kernel-bound work is neutral** - 256-cubed matmul **0.99x** and 8-cubed matmul 1.00x are controls taking no array at all, so **H2's result is structurally untouched**, and 512x512 `copy` 1.02x, 256x256 `to_numpy` 1.04x, full 512x512 `sum` 1.06x, 256x256 broadcast `multiply` 1.08x, and the large MLP step 1.08x are at or inside the band. **Memory did not move** and it is asserted: the same workload allocates 5 native storages, peak 4 live, 584 peak bytes before and after; a view's cold footprint is byte-identical and a strided view costs +296 bytes, with only **9 of 98** views in an MLP step ever populating it. Harness 31 to **34** cases (`ctypes_boundary_strided`, the array-carrying twin of the array-free `ctypes_boundary` - 1.3 us against 0.8 us, where pre-H7 it would have been ~7 us - plus the two broadcast shapes). Validation added an ASan **negative control**: two-entry metadata passed with `ndim = 3` produces a `heap-buffer-overflow` in `reduce_prefers_contiguous_blocks`, which is what makes **zero diagnostics across 2,834 sanitized tests** a real absence rather than a blind detector. No public API, capability, dtype, device, registry value, checkpoint field, or checkpoint version moved | **Complete** |
| **H8** | **Native elementwise traversal and composed allocation efficiency.** Entered with **two** candidate tracks and an explicit instruction not to force both into production; the measurement kept one as a large result, kept a narrow piece of the other, and rejected the rest with reasons. **Track A (elementwise traversal) is the milestone; Track B (composed normalization allocation) is a memory result and is reported as timing-neutral.** The cost was decomposed rather than assumed: the odometer costs **1.60x-6.42x** the flat loop on identical contiguous data, **all broadcasting is on the odometer** (there is no broadcast fast path at all), and a guarded standalone split the odometer's cost four ways at `(256,256)` contiguous `add` - shipped **123.5 us**, templating alone **81.3 us (1.52x)**, collapsing alone **63.6 us (1.94x)**, **both together 11.5 us (10.7x)**. Neither is worth much alone; together they are worth an order of magnitude, because only their combination lets the compiler vectorize. The same run showed the **existing flat contiguous kernel was itself hobbled** by its indirect call (21.0 us against 11.7 us). H8 reuses the dispatch shape H2, H5, and H6 each proved - one hidden metadata builder, inside the existing export, **no new symbol**, the pre-milestone traversal retained. New `cpp/include/tf_elementwise_internal.h` declares `tf::build_unary_plan` / `tf::build_binary_plan` and the templated traversals; `cpp/src/elementwise.cpp` defines and dispatches to them. A **plan is an operation-local normalized descriptor** - stack-built, used by one call, dropped, nothing cached or interned - applying exactly two sequence-preserving transformations: **unit axes dropped**, and **adjacent axes merged** when `stride[outer] == stride[inner] * extent(inner)` for *every* operand at once. Axes are never reordered, split, or transposed; the bound is a fixed **4 axes**; **this is not a layout compiler**. The builders are total, pure, allocation-free, and a function of layout metadata alone, and a rejection is a **fallback, never an error**. `core_unary` / `core_binary` are retained **verbatim** as the shipped generic reference paths. **Only operations IEEE-754 specifies take the templated path**; **`exp` and `log` deliberately keep the paths they had**, since a vector-math library would be free to return different bits and the templated traversal measured **1.05x** on both - inside the noise. **The numerical contract is H8's own**: (1) every result with **at most one NaN operand is bit-identical** to pre-H8 - **zero differing results** across 15 op x layout combinations; (2) NaN positions identical and every NaN the *arithmetic* produces quiet (`relu_backward` and the identity gather **select** rather than compute, so a signaling NaN legitimately survives - exactly as H5 established); (3) **subtraction bit-identical everywhere**, two-NaN pairs included, because it is not commutative; (4) for **addition and multiplication with two NaN operands** the payload is outside the contract, asserted in neither direction. **Part 4 predates H8 and H8 narrows it** - the pre-H8 library's own flat kernel and own odometer already disagreed on **30 of 196** such pairs, post-H8 only a transposed operand differs, on **5 of 196**. A *different* qualification from H2's and H6's, which concerned NaNs meeting inside an accumulation. **H1 holds unchanged**, proved by poison over six layouts with two patterns, guard elements, and a negative control. **Track B**: `_NativeBatchNorm` builds its `(1 - momentum, momentum)` pair **once per forward** instead of once per buffer (the H4 per-step-constants shape, never stored on the module) and releases each blend's temporaries at last use. Against a **retained pre-H8 composition executed natively**, running statistics bit-identical first: BatchNorm1d training forward **25 -> 23** allocations, **peak live 25 -> 17**, fills **5 -> 3**; BatchNorm2d **30 -> 28**, **30 -> 22**, **5 -> 3**. **Timing neutral** (1.007x-1.106x). Four alternatives **rejected with reasons**: early release of graph temporaries (proved impossible), adopting the blend core into the running-state transaction (would move numerical work inside staging, changing a failure ordering F5/F8 prove), caching the eval inverse standard deviation (hidden mutable state), and reshaping `gamma`/`beta` to `(1,C,1,1)` (F4's unversioned-parameter hazard). Measured against a pre-H8 library on identical `ctypes` calls, bit-identical before timing, control band **0.97x-1.08x**: row-broadcast `multiply` **10.58x**, strided same-shape `add` **9.67x**, NCHW-stat `multiply` **7.15x**, col-broadcast **6.70x**, scalar-broadcast **6.31x**, rank-3 broadcast **6.18x**, transposed `add` 2.63x, strided `relu` 2.51x, transposed `copy` 2.31x, `sqrt` 2.03x, `reciprocal` 1.98x, `relu_backward` 1.86x, contiguous `relu` 1.78x, contiguous `add`/`multiply` 1.76x, contiguous `copy` 1.68x. Layer/end-to-end over 11 alternating subprocess rounds, all 31 cases bit-identical first: `TensorCore` row-broadcast **6.81x**, graph broadcast multiply **6.33x**, scalar-broadcast `add` **5.12x**, **`NativeAdam.step()` 2.01x**, **BatchNorm1d eval 1.40x**, **BatchNorm2d eval 1.36x**, **BatchNorm2d training forward 1.33x**, **LayerNorm forward 1.30x**, BatchNorm2d fwd+bwd 1.25x, **large MLP step 1.19x**, **normalized step 1.08x**. **This is the milestone that finally moved the normalization modules**, which H6 measured as almost entirely neutral and which is why H0's composed-module H7 was dropped and this one entered. Reported just as honestly: **small normalization shapes are neutral** (BatchNorm1d `(32,16)` **0.98x**), the **CNN step is neutral (0.99x)**, and one control is **published rather than buried** - **`matmul` 256 cubed reads 0.93x-0.96x**, isolated by a 25-round run to that one size (64 cubed 1.014x, 128 cubed 1.035x, **256 cubed 0.921x**, 384 cubed 0.994x) against an identical-code twin at 0.969x. `matmul.cpp` is byte-identical source; `elementwise.cpp`'s object code grew 127 KB to 188 KB, moving every function's placement - **the same whole-translation-unit code-layout effect H6 documented**, every matmul result bit-identical, the H2 CTest passing, no end-to-end case regressed. **Memory: Track A moved none, and the odometer's heap-allocated counter is now removed on every plannable layout** - a strided elementwise call makes **one** allocation where it made two - which re-anchored one existing fault-injection test with its assertion unchanged. Harness 34 to **38** cases; native CTests 15 to **16**. **No exported C ABI symbol, no new translation unit, no public control of any kind**, no SIMD/threading/OpenMP/BLAS/pool/workspace/fusion/fast-math, no `restrict`. No public API, capability, dtype, device, registry value, checkpoint field, or checkpoint version moved; export count stays **52** | **Complete** |
| **H9** | **Native Conv2d execution efficiency.** **Not in H0's ladder** — that slot was pencilled in for SIMD/threading/BLAS, *conditional and presumed rejected*; none qualified, and a larger, safer result was available, so the acceleration decision moved to H10's decision gate. H6 left every training step neutral, H7's CNN gain came from small calls, and H8 moved the normalization modules while the **CNN step stayed neutral at 0.99×** — because a convolution step's time is in `tf_core_conv2d_*`, still the unmodified Phase-D direct loop from D2–D5. The cost was **decomposed**: the Python wrapper is a fixed ≈ 8–12 µs — **66 %** of a toy `(4,1,6,6)` convolution but **0.2 %** at `(8,3,16,16)` and **≈ 0 %** at `(16,8,32,32)` — so the compiled traversal is essentially **100 %** of any real convolution, H6's finding rather than H3's. The composed **bias gradient was measured and found immaterial**, a recorded negative result. All three kernels were `n,o,i,j` outer / `c,p,q` inner with the padded coordinate recomputed and bounds-tested inside, making the innermost loop typically **3** iterations. H9 reuses the H2/H5/H6/H8 dispatch shape: **one hidden predicate, inside the existing export, no new symbol, the pre-milestone traversal retained**. The three `*_generic` twins are the **Phase-D loops retained verbatim** as the shipped reference paths; beside them `conv2d_forward_row_sweep` (nest becomes `n,o,i | c,p,q | j` into a bias-primed output row), `conv2d_input_backward_gather` (walks `grad_input` rows and gathers instead of scattering), and `conv2d_weight_backward_gather` (one destination per register, written **once** instead of `batch·out_h·out_w` times). **Preconditions**: `min(input_width, output_width) >= 4` for all three, plus **unit stride** for the input gradient. The minimum is **measured** — swept extent 1 gave **0.57×–0.93×**, 2 gave 1.04×–1.38×, 4 gave **1.91×–2.40×** — and keying the input gradient on `input_width` alone **was measured wrong** (0.73×). Total, pure, allocation-free, geometry-only; **a false answer is a fallback, never an error**. The input gradient alone needs unit stride because its downward kernel walk inverts one-for-one only there; forward and weight gradient optimize at **every** stride, and **that asymmetry is deliberate**. **Per-destination accumulation order is preserved exactly in all three directions**, each separately proved; nothing reassociated, no FMA, no fast-math, no tree/pairwise/parallel accumulation. **Numerical contract, measured against a pre-H9 library**: every non-NaN result bit-identical (**256 pairs × 3 directions, zero differences**), NaN positions identical, **at most one NaN per destination agrees exactly including payload** (480 configurations, zero differences), **signed zeros bit-identical** (80 configurations — `−0.0` survives only while every addend is `−0.0`, both halves asserted on both paths), signalling NaNs quieted identically; **not contractual**, two-or-more-NaN payloads (20/256, 20/256, 29/256), the H2/H6 qualification **measured here rather than assumed**. **H1 holds on all three destinations** for a different reason each, proved by poison in C++ and through the real Python allocation seam, both with negative controls. **Layout, autograd, and memory untouched** — byte-identical allocation counts, peak live storages, and peak bytes; **no scratch, workspace, arena, pool, padded copy, or im2col anywhere**. Measured, bit-identical before timing, fallback controls at 1.00×–1.13×: k1×1 **6.23×/8.37×/5.40×**, `(8,16,32,32)→32` 3.64×/5.04×/2.60×, `(16,8,32,32)→16` 3.32×/5.28×/2.47×, padded 2.91×/4.97×/2.54×, `(8,3,16,16)→8` 2.87×/3.75×/2.45×, **stride 2** 2.41×/*1.04× (falls back)*/2.41×, k5×5 1.97×/3.84×/2.09×. End to end: `NativeConv2d` fwd+bwd **3.13×/3.09×**, forward **2.98×**, and **a CNN training step 1.86×**, 1.38×, 1.27× with Dropout, **1.13×** at the shipped shape, 1.11× with BatchNorm2d — **the first Phase-H milestone to move a CNN training step**. Honestly reported: **small convolutions neutral** (1.06×/1.20×), BatchNorm2d and shipped-example steps move least, and **no control regressed** (21 rounds: matmul **0.98×**, MLP **0.97×**, band **0.97×–1.07×**), with the 9-round 0.93× elementwise reading **published** as the low-round artifact it was. **Four candidates rejected with reasons**: im2col+matmul (changes accumulation order; 8× the input allocated), a materialized padded input (moves cost), output-channel blocking (tuning constant for a second-order gain), and a third small-extent path (would leave the Phase-D reference unreachable). Harness 38 to **41** cases; CTests 16 to **17**; **no ABI change, still 52 symbols**; no dilation/groups/channels-last/padding mode added; no capability move | **Complete** |
| *Ladder revisions* | **The ladder was a set of proposals, not commitments, and three of them were revised on evidence.** **H9 is the second worked example of an evidence-driven revision**: H0 pencilled that slot in for SIMD/threading/BLAS, *conditional and presumed rejected*; none qualified, so the acceleration decision moved to H10's decision gate and the slot went to Conv2d execution, the last large compute family still running its unmodified Phase-D implementation. Ordered by measured leverage and each explicitly conditional: a milestone whose premise the preceding measurement does not confirm is narrowed, reordered, or dropped, and the reason is recorded in the design. H5 is the worked example of a *reorder* - the ladder really was reordered, moving reduction execution from the H5 slot to H6 on H4's evidence. **H7 is the worked example of a milestone being *dropped***: the composed-module milestone that held that slot was conditional on a re-measured normalization step still showing a material composed-module cost after H1, H3, and H6, H6 measured that step at **1.03x** with most normalization operations inside the control band, and so it was not entered - the slot was refilled with the boundary work H3, H5, and H6 had each deferred by name. **H8 is the worked example of a proposal being *confirmed and widened***: it was the weakest-evidenced numbered milestone and the one most likely to be dropped, and the measurement instead showed its premise held *and* that the composed-normalization cost the dropped H7 had left as conditional scope was reachable from the elementwise side. **H10 is the third worked example, and the last**: it made the acceleration decision as a *decision*, rejecting SIMD, threading/OpenMP, and BLAS with measurements (design 16.11.7), assessed the two remaining candidates and implemented neither, re-measured the whole phase against the reconstructed H0 baseline, ran the full validation matrix, and closed the phase. H0's H11 slot was **not needed** - H10 carried closure itself, so the ladder runs H0-H10 and ends there | **Complete** |
| **H10** | **Integration, re-measurement, the acceleration decision, and closure.** The H0 baseline was **reconstructed and proved** - all 80 tracked `src/`+`cpp/` files restored to `c9d89e1` and verified file by file, its library built separately, and each side's resolved sources and library hash recorded at run time so the comparison refuses to report if they coincide (H0 exports **51** symbols, final **52**, the difference being exactly `tf_storage_create_uninitialized`). 52 cases, three independent runs, **zero checksum mismatches** - every H0-versus-final figure compares implementations that produced bit-identical results. Measured: matmul **4.71x**, Conv2d kernels **2.59x-4.64x**, reductions **3.78x-5.06x**, `NativeLinear` forward **4.78x**, `NativeAdam.step()` **3.91x**, normalization modules **1.98x-2.20x**, and **every shipped training workload 1.50x-3.89x**. No allocation count or memory peak rose anywhere; 16 workloads fell (an Adam step 108 -> **72** allocations and 1.97 MB -> **0.79 MB** peak). Reported just as honestly: the raw-kernel and NumPy controls held at 0.99x-1.03x, Dropout and storage allocation are neutral, and **`to_numpy` at 0.95x is the one reproducible regression** - attributed to a byte-identical compiled traversal (0.975x-1.008x) that H3/H7's cheaper wrapper no longer hides, and recorded as a remaining limitation. **The acceleration gate was resolved as three documented rejections**: SIMD (disassembly shows elementwise/matmul/reduction **already auto-vectorized** - 24/16/4 packed-double instructions - while scalar convolution offers only a 1.05x-1.4x *compiler hint*, and 4-wide would need a forbidden `/arch` flag); threading (198 native calls per CNN step, **median 1.20 us**, only two above 1 ms, ceiling about 1.4x, against a determinism obligation and a missing TSan matrix); BLAS (gap re-measured at 3.6x-9.3x, but **not bit-identical** - 3.553e-15 at 64 cubed, reproducing the design's prediction - which would break every exact-resume proof). **`tf_core_narrow_backward` was measured at 0 calls in all four real training workloads** and left alone; the small-operation floor was decomposed and found to be **60.5% allocation and 18.7% result wrapping against 11.8% ctypes**, an architectural floor whose remaining levers are all forbidden. Validation: Windows Release **and** Debug with zero warnings and **17/17** CTests each, Linux CI-equivalent and Windows pytest both **6,358 passed / 0 skipped**, Clang ASan/UBSan with instrumentation proved and **zero** diagnostics across 6,358 sanitized tests, a **negative control** producing a real `heap-buffer-overflow`, and a LeakSanitizer lifecycle returning live storage **exactly to baseline** at 20 checkpoints with no TensorForge frame at exit and no suppression file. **No production file changed** | **Complete** |
| ~~H11~~ | **Not needed.** H0 drafted a separate phase-closure milestone; H10 carried integration, re-measurement, the acceleration gate, *and* closure, so splitting them again would have created a milestone whose only content was recording that the previous one finished. Every closure requirement is satisfied by H10 | **Not entered; the ladder ends at H10** |

What the H0 evidence found, in one paragraph, and with the honesty the
design spends a section on: the largest measured factors are **not**
where an unoptimized-kernel intuition would put them. The eager
zero-fill on every native allocation, the matmul's memory **access
pattern** (not its lack of SIMD — and the existing `tf_matmul_tiled`
demonstrates a large improvement that is provably **bit-identical** to
the current kernel), the Python-side per-call metadata path (of which the
ctypes boundary is roughly a tenth), and the optimizer's fixed
per-parameter call and allocation count together account for far more
than the arithmetic does. `NativeTensor` and its autograd graph node are
**not** a bottleneck — a negative result that rules out a whole family of
proposed optimizations. Every number behind those statements is a local
characterization of one machine, is reported with its spread, and is
asserted by no test.

## Phase I — native dtype generalization and float32 CPU support, **complete (I0–I11)**

**Phase I is complete: milestones I0 through I11 have all landed; Phase J closed after it, so the
latest completed phase is Phase J.** I11 revalidated the stack across every
required platform, added the closure guardrails, and reconciled the status
surfaces. Its architecture contract is
[native_dtype_float32_design.md](native_dtype_float32_design.md).
Phase H is unaffected and remains complete — it closed at **52** exports.
Phase J, below, is complete (J0–J9). **Phase K is the latest phase, and only K0 through K7 have landed.** **K8 through K9 are unstarted.** **Phase J is the latest completed phase**, and it remains complete. Phase J moved no
row in this document at any milestone.

**Since milestone I9, `float32` and `float64` are both supported native
CPU dtypes**, and this is the row that supersedes every "float64 only"
sentence recorded for an earlier phase above:

| Registry | Value |
|---|---|
| `SUPPORTED_DTYPES` | `("float64", "float32")` |
| `SUPPORTED_DEVICES` | `("cpu",)` |
| `UNSUPPORTED` | `("cuda", "amp")` |
| `RAW_KERNEL_DTYPES` | `("float64",)` |
| Native checkpoint format | `tensorforge.native_checkpoint`, version **3** |
| Accepted checkpoint versions | `(1, 2, 3)` |
| In-memory optimizer state format | version **1** |
| Exported production `tf_*` symbols | **54** |

Read those four registry rows as four different statements, because they
are: `SUPPORTED_DTYPES` is the **capability**; `backend_info()["dtype"]`
(still `"float64"`) is the **default** an omitted `dtype` selects, at every
constructor, factory, module, and parameter, and that has not changed and
will not; `SUPPORTED_DEVICES` did not move, because a dtype milestone
grants no device; and `RAW_KERNEL_DTYPES` is a permanent limitation of the
**seven handle-free raw utility kernels**, which take `double*` and an
element count and therefore have no dtype to dispatch on. Overall float32
support and raw-kernel float64-only support are separate facts and neither
may be read off the other.

What is *not* implied: no casting, no promotion, no mixed-dtype arithmetic
(a mismatch raises before any allocation or mutation), no dtype inference
from a NumPy array, no global default dtype, no `.float()`/`.double()`/
`.astype()`/`.to()`, no third dtype, no device, no CUDA, and no AMP.

**I0** was a design-and-reconciliation milestone: it shipped the contract,
its guardrail tests, and documentation, and **no runtime behavior at
all**.

**I1** built the dtype foundation. What it changed:

- the C++ dtype model exists — frozen ABI codes (`0 = float64`,
  `1 = float32`), one item-size authority, one canonical-name authority,
  and a total validated code conversion that rejects every unknown code
  without producing a dtype;
- native storage is **dtype-tagged**: one untyped owned buffer, a logical
  element count whose meaning did not move, and one dtype tag that is the
  single authority for how the bytes are read. The storage owns a genuine
  runtime-selected `float[]` or `double[]` array, created with checked
  `numel × itemsize` and type-erased into `void*` only after creation, so
  the kernels' `data[i]` and `data + i` are valid C++17 pointer arithmetic
  over one array object; the immutable dtype tag selects the typed
  accessor and the matching central `delete[]`, so the allocation and
  deallocation forms cannot disagree;
- the library exports **54** production `tf_*` symbols — the two typed
  creators `tf_storage_create_typed` and
  `tf_storage_create_uninitialized_typed`, which are the **only** two
  symbols the whole phase adds. The untyped creators remain, unchanged, as
  thin float64 compatibility wrappers over the same shared body;
- the native CTest inventory moved **17 → 18** (`test_dtype_storage`).

**I2** made that foundation movable, and added **no export**. What it
changed:

- the three exports that carry a storage handle *and* a raw host buffer —
  `tf_storage_copy_from`, `tf_storage_copy_to`, `tf_storage_materialize` —
  are **dtype-general**, through a **source-level retype** of their host
  positions from `double*` to `void*`. Same symbols, same argument counts
  and order, same calling convention, same return types: a previously
  compiled caller would link and run identically, and the export inventory
  did not move;
- `tf_core_contiguous_copy` — the runtime's **value-transfer primitive**
  (H5) and the one compute-shaped export I2 touched — is
  dtype-preserving and dtype-strict. Source and destination dtypes must
  agree; a mixed pair is rejected before a single element is written,
  because the runtime never casts. Its three H5/H8 traversal tiers are
  instantiated for both element types from the *same source*, so float64
  runs the code Phase H measured;
- **transfer is bit-preserving at both widths**: `+0.0`/`-0.0`, both
  infinities, subnormals, quiet NaNs with distinct payloads, and
  signalling NaNs of both signs all reproduce their source's object
  representation exactly. Proved over seventeen IEEE-754 classes per dtype
  as raw `uint32`/`uint64` patterns, in C++ and in Python, never by value
  and never by tolerance;
- `RAW_KERNEL_DTYPES == ("float64",)` records the other half of the ABI's
  division and is reported by `backend_info()` as `raw_kernel_dtypes`,
  beside — and deliberately distinct from — `supported_dtypes`;
- the native CTest inventory moved **18 → 19** (`test_typed_transfer`).

**I3** made that movable foundation computable, and added **no export**.
What it changed:

- the elementwise and unary Core family is **dtype-general**: `add`,
  `subtract`, `multiply`, `relu`, `relu_backward`, `sqrt`, `reciprocal`,
  `exp`, and `log`, across their strided and contiguous forms — seventeen
  exports, every one of them the symbol Python already declared. Each
  validates that its operands agree, dispatches **once** from the storage
  tag, and runs one instantiation of a templated kernel; nothing below that
  point branches on dtype;
- **all three Phase-H traversal tiers are instantiated for both element
  types from the same source** — the contiguous flat row, H8's collapsed
  plan, and the retained generic odometer — so float64 runs the code Phase H
  measured and the two widths cannot drift apart. NumPy-style broadcasting
  works at float32 for every layout it already worked at for float64: zero
  strides, transposed and narrowed operands, negative strides, unit extents,
  multiple broadcast axes, and rank-0 scalars;
- **outputs preserve the operand dtype**, allocated through the private
  typed path I2 added. A float32 operation produces a float32 result and a
  float32 NumPy array on egress; nothing widens on the way out;
- **float32 arithmetic is genuinely binary32.** Every result is asserted
  bit-identical to the binary32 oracle for the IEEE-specified operations,
  and `exp`/`log` — library functions with no correctly-rounded guarantee —
  are held to a measured ULP bound instead, exactly as at float64. No hidden
  float64 accumulator, constant, or intermediate exists, which is asserted
  structurally over the source as well as numerically;
- **mixed dtype is rejected before any allocation or mutation**, in the
  left, right, and destination positions independently, at the Python layer
  and again in C++ at the trust boundary. A rejected call leaves its
  destination byte-for-byte unchanged and allocates nothing;
- the native CTest inventory moved **19 → 20** (`test_dtype_elementwise`).

**I4** extended that to the accumulating families and to the graph built on
them, and added **no export**. What it changed:

- `sum`, `mean`, `matmul`, and `narrow_backward` are **dtype-general** —
  three exports, every one of them the symbol Python already declared, each
  validating that its operands agree and dispatching **once** from the
  storage tag into a templated kernel;
- **both traversals of both families are instantiated for both element types
  from the same source**: H6's contiguous-block factorization and the
  retained generic odometer, H2's `i`-`k`-`j` row sweep and the retained
  `i`-`j`-`k` triple loop. Both metadata predicates are untouched — they read
  `int64` layout only — so the two widths take the *same* path for the same
  layout and every optimized path keeps its oracle **per dtype**;
- `tf_storage_scale` and `tf_storage_fill` are dtype-general with their
  `(handle, double)` signatures **unchanged**: the scalar crosses as the
  ABI's widest binary floating-point type and is narrowed **once, before the
  loop**. That is what makes `mean`'s `1/count` deterministic, identical on
  every platform, and independent of `count`'s magnitude, and it is asserted
  behaviourally — narrow-then-multiply and multiply-then-narrow differ by one
  representable step on a chosen witness, and TensorForge must produce the
  first;
- **float32 accumulates in float32, and I4 is where that became a measured
  claim rather than only a structural one.** On `1.0` followed by eight
  copies of `2**-24`, sequential binary32 stays at exactly `1.0` while
  binary64-then-narrow lands four ULPs higher; TensorForge is asserted equal
  to the first and **unequal to the second**, by raw bit pattern, on both
  reduction traversals and both matmul paths;
- **private/internal float32 `NativeTensor` graphs run forward and backward**
  through every Core operation landed so far. Gradients carry the tensor's
  dtype, backward temporaries carry the graph's, every materialized constant
  — `0.5`, `-1`, `1/count`, the ones seed, broadcast-back's zeros — is built
  at the operand's dtype through a private typed constructor, and a
  mixed-dtype contribution is refused before the accumulation and before any
  allocation;
- the native CTest inventory moved **20 → 21**
  (`test_dtype_reduction_matmul`).

**I5** extended that to the CNN stack, and added **no export**. What it
changed:

- all three Conv2d directions (`tf_core_conv2d_forward`,
  `tf_core_conv2d_input_backward`, `tf_core_conv2d_weight_backward`) and
  both MaxPool2d directions (`tf_core_maxpool2d_forward`,
  `tf_core_maxpool2d_backward`) are **dtype-general** — five exports, every
  one of them the symbol Python already declared, each validating that its
  numeric operands agree and dispatching **once** from the storage tag into
  a templated kernel;
- **both traversals of every Conv2d direction are instantiated for both
  element types from the same source**: the retained Phase-D generic loops
  and H9's row-sweep and gather traversals, with all three geometry
  predicates untouched — they read `int64` geometry only — so the two
  widths take the *same* path for the same geometry and every optimized
  path keeps its oracle **per dtype**. Conv2d accumulates in the element
  type: the float32 accumulation witness (`1.0` plus eight `2**-24`) is
  proved in all three directions, on both traversals of each, equal to the
  sequential binary32 result and unequal to the binary64-then-narrow one;
- MaxPool2d's value path follows the input dtype — the window scan, the
  strict `>` first-winner tie rule, the NaN policy, the signed-zero
  selection, and the padding-as-`-inf` behavior are the identical
  comparison sequence at both widths — while **the winner buffer stays
  private float64 at every value dtype** (design §13.3). The `2**53` exact
  winner-plane bound is float64's and does not shrink to float32's
  `2**24`: a float32 pool over a plane larger than `2**24` still records
  its winner offsets exactly, which is the capability the decision
  preserves. A non-float64 winner buffer is refused before anything is
  read or written, in Python and again at the C ABI;
- the overlapping-window pooling backward accumulates repeated winners in
  the **graph dtype**, witnessed at float32 against the widened
  alternative; unique routing reproduces finite upstream values bitwise;
- **private/internal float32 `NativeTensor` graphs differentiate through
  convolution and pooling**: input, weight, bias, and pooling gradients
  all carry the graph dtype, the float64 winner buffer rides the unchanged
  `graph_resources` contract (released exactly once, retained under
  `retain_graph`, alive across a failed retryable backward, closed
  immediately by a no-grad forward), and mixed dtype is rejected in every
  operand position before any allocation;
- the native CTest inventory moved **21 → 22** (`test_dtype_cnn`).

**I6** extended that to the stable-math and classification stack, and added
**no export**. What it changed:

- softmax, log-softmax, and the fused cross-entropy forward and backward
  (`tf_core_softmax_forward`, `tf_core_log_softmax_forward`,
  `tf_core_cross_entropy_forward`, `tf_core_cross_entropy_backward`) are
  **dtype-general** — four exports, every one of them the symbol Python
  already declared, each validating that **every** participating numeric
  handle agrees (two for each transform, three for each cross-entropy
  direction) and dispatching **once** from the storage tag into a templated
  kernel;
- the four compute kernels became templates over the element type and moved
  into `tf_classification_internal.h`, on I4's and I5's terms: a template
  must be visible where it is instantiated, and both instantiations must
  reach the exported wrappers and the CTests that compile them directly.
  `T = double` is the Phase-E kernel statement for statement; the source
  organization of the classification design's §9.1 is unchanged;
- **everything numerical happens at the element type**: the maximum scan,
  the shift, `std::exp`/`std::log` on the element type (so a float32 slice
  takes the `float` overload rather than widening and narrowing back), the
  normalizing sum, the log-normalizer, the per-row loss, the batch-loss
  accumulator, the mean divisor, the gradient contribution, and the
  upstream scale. There is **no hidden float64 accumulator** anywhere, and
  the batch-loss accumulator is where that became a *measured* claim: on a
  batch whose first row contributes exactly 200 and whose remaining 199
  contribute ~6.1e-6 each, sequential binary32 stays at exactly 200 while
  binary64-then-narrow lands ~1.2e-3 higher, and TensorForge is asserted
  equal to the first and **unequal to the second** by raw bit pattern;
- **log-softmax is still its own fused log-sum-exp kernel** and never
  `softmax().log()` — a decision that only gets stronger at float32, whose
  smallest normal probability is ~1.18e-38 — and the cross-entropy loss
  still comes from `log(sum_exp) - (x[target] - m)`, never from
  `-log(p[target])`;
- the **saved probabilities carry the graph dtype**, ride the unchanged
  `graph_resources` contract, and are still the only thing the backward
  reads: the logits are not a parameter of the kernel, of the export, or of
  the Core wrapper, and no exponential or logarithm is evaluated there;
- the class **targets stay host `int64` metadata at every width** — copied
  into an independently owned array, revalidated index by index in C++,
  caller-independent after the forward, never a tensor, never inferred from
  the logits dtype. **No integer tensor dtype was introduced**, and no
  integer-storage or target export was added;
- the float32 stability statement picked up its **one honest
  qualification**, measured rather than assumed. The maximum shift
  guarantees no *exponent* overflows; it does not make the shifted value
  `x - m` representable, so a slice whose **spread** exceeds the element
  type's largest finite value overflows it to `-inf`. `softmax` is
  unaffected — the affected class gets exactly `+0.0`, which is the
  correctly rounded probability — while `log_softmax` reports `-inf` and
  `cross_entropy` `+inf`, as *values* with the error slot at `TF_OK`. The
  identical thing happens at binary64 past ~1.8e308, so it is a
  dynamic-range fact rather than a float32 defect. No widened intermediate,
  clamp, or special case was added; the counterexample is recorded in the
  contract's §10.5 and asserted in both directions by test;
- **private/internal float32 `NativeTensor` graphs differentiate through
  all three operations**, with no change to the graph structure at all —
  the backwards are composed from Core operations that I3 and I4 had
  already generalized;
- the native CTest inventory moved **22 → 23**
  (`test_dtype_classification`).

**I7** made float32 a *module* dtype and generalized the last float64-only
numerical family, and added **no export**. What it changed:

- six state-owning constructors — `NativeParameter`, `NativeLinear`,
  `NativeConv2d`, `NativeLayerNorm`, `NativeBatchNorm1d`, and
  `NativeBatchNorm2d` — take a **keyword-only** `dtype` accepting exactly
  `"float64"` and `"float32"` and defaulting to `"float64"`, all six
  routing through **one** shared private validator. The set is closed: no
  stateless module, loss, metric, generator, container, or optimizer took
  one, and no `device` argument was added anywhere;
- affine parameters, both BatchNorm running buffers, the graph-safe
  evaluation snapshots, every training and evaluation temporary, and every
  scalar a composed normalization forward materializes (`eps`, `momentum`,
  `1 - momentum`) are at the module's dtype — the scalars through two new
  private tensor constructors, because a literal float64 constant meeting a
  float32 operand would be a mixed-dtype request the runtime refuses;
- the atomic two-buffer running-statistics transaction gained **one** dtype
  validation and nothing else, and the BatchNorm forward re-proves that all
  four numeric state objects still carry the module's dtype before either
  buffer can move;
- **initialization did not move at either width**: the same local
  `numpy.random.default_rng(seed)` stream, in the same order, at the same
  sizes, with the fan-in bound computed once in binary64, so a float32
  layer with seed *S* holds exactly `float32(the float64 draw with seed
  S)` — the seed contract stays dtype-independent;
- `tf_core_dropout_forward` became dtype-general with its **exact ABI shape
  unchanged** (same symbol, same eight arguments, same order, same types),
  one operand-agreement guard over its three handles, and one dispatch into
  a templated kernel. The **random derivation is untouched**: the uniform
  is still the binary64 53-bit conversion at every width, so one
  `(seed, call_index, element count)` key drops exactly the same elements
  at both dtypes — proved at float32 against the *same* committed Phase-G
  keep vectors. Only the two multiplier values differ, and the kept one is
  the binary64 reciprocal narrowed once;
- the generator's algorithm, version, state, locking, and reserve →
  commit/abandon call accounting are unchanged, and the accounting is
  asserted identical at both widths on every path — one call for a
  successful stochastic forward, none for a failure, none in evaluation,
  none at `p == 0`;
- `state_dict()` carries dtype implicitly through its tensors, validates it
  per entry against the live destination, and never casts; one mismatched
  entry among many rolls the whole load back;
- a **version-2 checkpoint refuses a float32 model on the way out**, before
  the temporary file exists. Versions 1 and 2 are float64-only formats
  permanently, so writing a float32 payload under a version-2 manifest
  would have produced an archive this library refuses to read back;
- the native CTest inventory moved **23 → 24** (`test_dtype_dropout`).

**I9** made float32 **public**, and it is the phase's one and only public
capability change. What it changed:

- the registry moved, exactly once and exactly this far:
  `SUPPORTED_DTYPES` `("float64",)` → `("float64", "float32")` and
  `UNSUPPORTED` `("float32", "cuda", "amp")` → `("cuda", "amp")`.
  `SUPPORTED_DEVICES`, `RAW_KERNEL_DTYPES`, the export count, the
  checkpoint version, and the in-memory optimizer state version all stayed
  where they were;
- `normalize_dtype("float32")` succeeds, and every **public** constructor
  builds a float32 tensor: `NativeStorage(...)` and
  `NativeStorage.from_array`, `NativeTensorCore.from_array` / `.zeros` /
  `.full`, and `NativeTensor.from_array` / `.zeros` / `.full`. Views,
  operations, and gradients all preserve the dtype, and `to_numpy()`
  returns a genuinely `np.float32` array — never widened on the way out;
- the **ordering was the discipline, and it was followed**: the integrated
  example and its exact-resume proof were written and passing *first*,
  through the already-approved private typed route and the six I7 module
  constructors, with the registry still reading `("float64",)`; the
  registry moved only afterwards; the example's one ingress helper switched
  to the public `NativeTensor.from_array(values, dtype=...)`; and the whole
  proof was rerun. The example contains no `_typed*` call and a test
  asserts that;
- `examples/native_float32_training.py` is the executable proof —
  `Conv2d(1→4, 3×3, pad 1) → BatchNorm2d(4) → ReLU → MaxPool2d(2) →
  Dropout(p) → Flatten → Linear(36→8) → BatchNorm1d(8) → ReLU →
  LayerNorm(8) → Dropout(p) → Linear(8→3)`, raw logits into
  `NativeCrossEntropyLoss`, trained by `NativeAdam` — run uninterrupted and
  interrupted at **each** dtype and compared **only against itself**;
- **two Dropout layers share one registered `NativeGenerator`**, so the
  model carries a real alias topology rather than a lone counter: one
  canonical path, one alias, two generator calls per training forward, one
  canonical state plus the full alias map written to the archive and
  re-validated against a live traversal on load;
- **every comparison is over raw IEEE-754 bit patterns** — `uint32` at
  float32, `uint64` at float64 — with no tolerance, no `allclose`, and no
  float32-versus-float64 comparison anywhere. Proved equal between the
  uninterrupted and the resumed run: the loss suffix and the whole loss
  sequence, every parameter, every persistent buffer, every Adam `m`, `v`,
  and step counter, the optimizer hyperparameters, the generator's
  algorithm/version/seed/calls, the alias topology, the final training
  logits, the final predictions, the final evaluation output, and the
  validated external-loop metadata;
- **gradients are proved produced, not restored.** They are not
  checkpointed, so the first resumed step captures them after backward and
  before the optimizer commits — the one moment they exist — and compares
  them to the same step of the uninterrupted run;
- the **next Dropout mask** after the resume is bit-identical, proved
  non-degenerate (8 dropped, 24 kept), consuming exactly one call in each
  model, and observed through the shared alias path so the restored
  *sharing relationship* is what is demonstrated rather than two equal
  counters;
- the fourth graph-owned resource family is exercised and **scoped
  honestly**: three ride a *training* graph (both Dropout masks, the
  MaxPool2d winners, cross-entropy's saved probabilities) while the
  BatchNorm evaluation snapshots exist only on an *evaluation* graph, so
  all four are exercised **across** the run rather than coexisting in one
  graph — and the eval graph is proved independent of the live buffers by
  advancing all four underneath it and re-running its backward;
- native live storage returns **exactly to baseline** (`0 / 0`) across both
  dtypes, both runs each, the mask proof, and the snapshot proof;
- `backend_info()`'s flat `"dtype"` key was decided explicitly and
  **kept**, still `"float64"`, now documented everywhere as the *default*
  statement rather than a capability one — accurate rather than merely
  unchanged, and asserted to agree with `normalize_dtype(None)`;
- **no C++ changed**, no export was added (still **54**), no CTest was
  added (still **24**), no checkpoint field or version moved, no new
  module/loss/optimizer/operation appeared, no dependency or build option
  was added, and no benchmark implementation changed — I10 owns
  benchmarking, and it added exactly one new harness rather than changing
  any existing one. The NumPy reference backend's own `supported_dtypes`
  stays `("float64",)`: Phase I is a native-line phase.

The honest statement of what float32 *is* after I9, in four parts because
they are four different things:

1. **Every numerical family in the native runtime is dtype-general.**
   Storage, transfer, views, elementwise/unary execution, reductions,
   matmul, all three Conv2d directions, both MaxPool2d directions, softmax,
   log-softmax, fused cross-entropy, LayerNorm, both BatchNorm shapes,
   Dropout, view backward, and Core autograd. There is no float64-only
   compute path left to name, which is a real milestone and is stated as
   one.
2. **The experimental state-owning modules can be constructed at float32.**
   Six constructors take a dtype; their parameters, buffers, snapshots,
   graphs, and gradients are physically float32; and a float32 model
   forwards, differentiates, normalizes, and drops.
3. **float32 state survives a step and a file (I8).** Both native
   optimizers execute at float32; Adam's `m` and `v` carry their
   parameter's dtype; one optimizer may hold parameters of both widths,
   each with independent dtype-consistent state; and native checkpoint
   **version 3** round-trips float32 model values, persistent buffers, and
   Adam moments **bit for bit**, with accepted versions now `(1, 2, 3)`.
   Versions 1 and 2 stay float64-only formats permanently.
4. **float32 is a supported TensorForge dtype (I9).** The exact float32
   resume proof has been run — integrated, bitwise, and independently at
   each dtype — the registry has moved, and every public tensor
   constructor produces a float32 tensor when asked for one explicitly.
   **Phase I is closed.** I10 (hardening and benchmarking) delivered the
   adversarial matrix and the one loader-validation repair it found, which
   moved no schema, version, or capability; I11 (cross-platform validation
   and closure) then revalidated every platform, reconciled every
   inventory, and ended the ladder without adding a capability of its own.

The capability detail behind (1), unchanged from I6 except for the last
two clauses. float32 can be
copied in from a host buffer, copied out, viewed through any layout the
metadata contract permits, materialized, and copied storage-to-storage, all
bit-exactly; it can be added, subtracted, multiplied, rectified (forwards
and backwards), square-rooted, reciprocated, exponentiated, and logged,
with broadcasting, at genuine binary32; it can be summed, averaged,
matrix-multiplied, and scattered back through a narrow, with the
accumulation genuinely in binary32; it can be convolved in all three
directions and max-pooled in both, through H9's unchanged traversals, again
accumulating in binary32, with the pooling winners kept as private float64
index metadata; it can be softmaxed, log-softmaxed, and reduced to a fused
cross-entropy loss and its gradient, with the maximum shift and the
log-sum-exp entirely in binary32 and the class targets still host `int64`
metadata; it can be normalized by LayerNorm and by both BatchNorm shapes,
with the statistics, the epsilon, the momentum coefficients, the running
updates, and the evaluation snapshots all at binary32; it can be dropped
with the *same* keep/drop pattern a float64 run of the same key produces;
and an internal graph over those operations — convolution, pooling,
classification, normalization, and Dropout included — differentiates end to
end with every gradient at the graph's dtype.

What (3) means concretely. `NativeAdam`'s `m` and `v` match their
parameter in dtype, shape, and device and start at bit-exact `+0.0`; one
optimizer may hold parameters of **both** widths, with independent
dtype-consistent state per parameter and the H4 scalar caches keyed on
`(dtype, device)`, so a mixed collection builds one scalar set per *active
dtype* rather than one per parameter. Neither optimizer gained a `dtype`
or a `device` argument, and neither may: they own no dtype they could
choose, only state that must match a parameter. A version-2 checkpoint
still refuses to *save* a float32 model rather than writing a payload it
could never read back, because versions 1 and 2 are float64-only formats
permanently.

**A float32 tensor still has to be asked for.** Nothing infers it, no
default produces it, and every existing call site that omits `dtype` is
byte-identical to a pre-Phase-I run: `NativeTensor.from_array(x)` on a
float32 NumPy array still produces a **float64** tensor. Passing a float64
array with `dtype="float32"` is the explicit host-to-native conversion
boundary — one conversion, on the way in — and is **not** a tensor cast:
no native tensor changes dtype and none can. There is no `astype`, no
`to`, no `.float()`, no `.double()`, no `result_type`, and no promotion
table.

Two surfaces I6 inspected and deliberately left alone are worth restating
now that the registry has moved, because the reasoning still holds.
`NativeCrossEntropyLoss` is a thin delegate to
`logits.cross_entropy(...)` and `native_accuracy` is a reporting-only
helper over `to_numpy()`; neither has a float64 assumption to remove and
neither gained a dtype argument, so both simply work at either width.
**That is the operation being dtype-general, not a second dtype
authority** — and the same is true of `NativeReLU`, `NativeFlatten`,
`NativeMaxPool2d`, `NativeSequential`, `NativeDropout`, `NativeMSELoss`,
and `NativeGenerator`, none of which takes a dtype or may gain one.

What the phase delivered: float32 CPU tensors beside the
existing float64 ones, dtype-tagged storage, dtype-aware handle-based
operations, float32 autograd, float32 modules, persistent buffers,
optimizers and optimizer state, float32 deterministic Dropout over an
**unchanged** generator algorithm, a dtype-aware checkpoint **version 3**,
exact deterministic float32 resume, public float32 support, unchanged
float64 behavior and performance, at **I10** the cross-cutting adversarial
hardening matrix and separate float32/float64 benchmark characterization,
and at **I11** the cross-platform revalidation, the closure guardrails, and
the final inventory reconciliation. Nothing remains: the ladder is
finished.

The decisions the contract locks, recorded here because they bound what
any later milestone may do:

- **Exactly two new C ABI symbols across the whole phase** —
  `tf_storage_create_typed` and `tf_storage_create_uninitialized_typed`,
  taking the library from 52 to **54** — and no more. **Delivered at
  I1**; no later milestone adds a symbol. Per-operation float32 exports
  are explicitly **rejected**: 42 of Phase H's 52 exports already address
  their operands through opaque handles, so a dtype tag on the storage
  reaches them all, and only *construction* needs new information. The
  existing untyped creators stay, unchanged, as compatibility wrappers.
- **Storage carries the dtype and is its single authority.** Every view
  of one buffer reports the same dtype by construction, and no view
  operation casts or reinterprets. Shapes, strides, offsets, and spans
  stay measured in logical elements; bytes appear only at the allocation
  boundary, with checked `numel × itemsize` arithmetic and one
  allocation form matched by one deallocation form. **Delivered at I1.**
- **A division between dtype-general handle-based paths and float64-only
  raw-buffer utilities.** The seven handle-free reference kernels
  (`elementwise_add`/`subtract`/`multiply`/`divide`, `relu`, `matmul`,
  `matmul_tiled`) have no handle and therefore no dtype to dispatch on;
  they stay float64 permanently, declared by a `RAW_KERNEL_DTYPES`
  registry. **Delivered at I2.** No native float32 training path goes
  through them.
- **No casting, no promotion, and no mixed-dtype arithmetic**, at any
  layer, rejected before any output is allocated or any state is
  mutated. There is no `astype`, no `to()`, and no `map_location`.
  **First reachable and tested at I2**, on the identity copy; **enforced
  across the whole elementwise family at I3**, in the left, right, and
  destination positions independently; **extended at I4** to reductions,
  matmul, narrow-backward, the backward seed, and gradient accumulation.
- **Value transfer is bit-preserving at both widths**, because a transfer
  performs no arithmetic and so has no operand roles to choose between and
  nothing to round. **Delivered at I2**, for host ingress, host egress,
  strided materialization, and the storage-to-storage identity copy.
- **float32 accumulates in float32.** No hidden float64 accumulator is
  introduced anywhere — that would be mixed precision, which is out of
  scope along with AMP, float16, bfloat16, integer tensors, CUDA, data
  loaders, and distributed execution. **First enforced at I3**, where every
  operation is a single correctly-rounded operation per output element and
  the claim could therefore only be carried structurally; **witnessed by
  test at I4**, where accumulation makes binary32 and binary64-then-narrow
  genuinely distinguishable, on both reduction traversals and both matmul
  paths.
- **Checkpoint version 3**, designed at I0 and activated at I8, with
  accepted versions becoming `(1, 2, 3)`. Versions **1 and 2 are
  float64-only formats** and a v1/v2 payload is **never** guessed to be
  float32.
- **Exact deterministic resume proved separately for float32 and
  float64.** A float32 run is never required to reproduce a float64 one,
  and no contract, gate, or closure claim rests on agreement between
  them. **Delivered at I9**, integrated and in raw IEEE-754 bit patterns,
  with each dtype compared only against itself.
- **Every Phase-H float64 optimization preserved**, with each dtype
  characterized on its own and — as in every phase before it — no timing
  assertion, no committed benchmark number, and no result file.
  **Delivered at I10**, in a harness deliberately separate from Phase H's
  so that every number Phase H published keeps its meaning. I10's one
  production change is a validation branch in the checkpoint **loader** —
  no C++, no kernel, no allocation path — so float64 behavior and
  allocation counts are unchanged by construction. The measured results
  are in [backend_experiments.md](backend_experiments.md).

The ladder is I0–I11 — **all twelve landed**: the
contract (I0), the internal dtype model and
tagged storage (I1), typed transfer, views, and materialization (I2),
elementwise and broadcast execution (I3), reductions, matmul, and core
autograd (I4), the convolution and pooling kernels (I5), stable math and
classification (I6), modules, parameters, buffers, and Dropout (I7),
optimizer state and checkpoint version 3 (I8), public integration and the
exact-resume proof (I9), cross-cutting hardening and benchmarking (I10),
and cross-platform validation and closure (I11). **The public support
registry moved at I9** — and at no other milestone — after integrated
float32 training and the exact float32 resume proof both passed, which is
the same discipline that kept `dropout` in `UNSUPPORTED` from G3 through
G9 while the operation and the module both already existed.

## Phase J — deterministic native data pipeline and mini-batching, **complete (J0–J9)**

**Phase J is complete.** **Phase K is the latest phase, and only K0 through K7 have landed.** **Phase J is the latest completed phase**, and it remains complete. It was newly approved
when it opened: the repository closed Phase I at I11 without committing to
a successor, and Phase J was approved afterwards, so nothing here describes
pre-existing roadmap work.
Its architecture contract is
[native_data_pipeline_design.md](native_data_pipeline_design.md).

Milestone **J0** was architecture, contract, and documentation work and
shipped **no runtime behavior at all**. Runtime capability began at
**J1**, and **exactly three public names plus one milestone that added
none** exist in this section today:

- **`NativeTensorDataset`** (J1) — the finite host-backed dataset, in
  `tensorforge.experimental`. One owned copied host snapshot of the
  features and one of the class targets; the native feature dtype
  **explicitly chosen** and never inferred from the input array, still
  defaulting to float64; a locked SHA-256 content fingerprint and a
  JSON-compatible `identity()`; a caller-owned `NativeTensor` feature
  batch and a fresh read-only host `int64` target batch per index
  sequence, order and duplicates preserved exactly; and no native storage
  held between calls. It moved no registry value, added no C ABI symbol,
  and took the experimental Python export inventory from 22 names to 23.
- **`NativeBatchSampler`** (J2) — the deterministic order and batch
  **planner**, in `tensorforge.experimental`. It owns `batch_size`,
  `drop_last`, `shuffle`, the `seed`, the `epoch`, and the `cursor`, and
  emits batch-index groups through `epoch_permutation()`, `plan()`, and
  `next_batch_indices()`, all of them pure. Its permutation is a pure
  function of `(seed, epoch, length)`, derived by the permanently private
  `_native_permutation` module from the locked `tensorforge.splitmix64`
  finalizer under one domain-separated epoch key schedule — **no new RNG
  algorithm, no new global or default generator, no `NativeGenerator`
  coupling, and no raw-random C ABI export** — with unbiased
  rejection-based bounded integers and a downward Fisher–Yates sweep. Its
  compact JSON-compatible `state_dict()` carries the configuration, the
  position, and the dataset's four identity fields (no permutation, no
  payload) and loads transactionally, validating dataset identity against
  live reality and adopting the state's configuration. It allocates
  nothing native, owns nothing releasable — so it has **no `close()`** —
  and works unchanged against a closed dataset. It moved no registry
  value, added no C ABI symbol, and took the experimental Python export
  inventory from 23 names to 24.
- **`NativeDataLoader`** (J3) — the mini-batch iteration surface, in
  `tensorforge.experimental`. `NativeDataLoader(sampler)`, and nothing
  else: no batch-size, shuffle, seed, drop-last, dtype, or device
  argument, because the sampler and the dataset own those.
  `iter(loader)` returns a **private** one-epoch iterator that captures
  the sampler's remaining batch count and supersedes any previous one,
  and each `__next__` runs an explicit five-phase transaction — claim,
  construct, publish, commit-and-deliver, rollback — under one invariant:
  **the committed sampler position advances if and only if a batch was
  successfully delivered to the caller.** Every failure position closes
  the undelivered feature tensor, restores the exact pre-delivery epoch
  and cursor through the same non-failing write seam a state load uses,
  and leaves a retry returning the same indices and the same values. A
  batch is a caller-owned `NativeTensor` at the dataset's dtype beside a
  fresh read-only host `int64` array, and **the caller closes the
  tensor** — no close path on the loader, the iterator, or the sampler
  retains or can reach a delivered batch. It is **not thread-safe** and
  adds no lock, thread, queue, worker, prefetch, collate, or callback
  surface. It moved no registry value, added no C ABI symbol, and took
  the experimental Python export inventory from 24 names to 25.
- **Loader state and exact in-memory mid-epoch restoration** (J4) — two
  methods on the existing `NativeDataLoader`, and **no new public name**:
  the experimental Python export inventory stayed at **25**.
  `state_dict()` returns a compact tagged wrapper with exactly three root
  keys — `format` (`"tensorforge.native_data_loader"`), `format_version`
  (**1**), and `sampler`, the last being exactly the **unchanged**
  version-1 sampler state. Every container is fresh at every call, no
  configuration or position field is duplicated at the root, and the
  structure carries no permutation, no dataset content, no NumPy object,
  no process-local value, and nothing whose size grows with the number of
  samples; it is JSON-compatible and accepted unchanged by the existing
  checkpoint metadata validator. It is allowed between batches, after
  exhaustion or supersession, with a closed dataset, and after the loader
  is closed, and is **refused** while a batch transaction is in flight,
  because the commit-before-delivery window has no honest answer.
  `load_state_dict(state)` runs a closed guard, a transaction guard, and
  an active-iteration guard **before** reading the state, validates the
  wrapper, **delegates** the whole nested sampler validation to the seam
  that already owns it, and commits through the same non-failing write
  seam the delivery uses — so a rejected load leaves the loader, the
  sampler, the dataset, the position, the cache behavior, the iterator
  slot, and native live storage byte-identical, and a successful one
  adopts all six configuration and position values while validating
  dataset identity without adopting it and preserving every object
  identity. It moved no registry value, added no C ABI symbol, and added
  **no** public name.

**J5 added no public name and no production code at all**, and proved the
caller-managed checkpoint-metadata workflow end to end against **real**
version-3 archives. A caller takes `loader.state_dict()`, places it inside
the `metadata` they already control, and on the way back calls
`load_native_checkpoint` **first** and
`fresh_loader.load_state_dict(...)` **second**. The checkpoint runtime
validates that metadata as recursively JSON-compatible, preserves it, and
returns it as an independent plain dict — it does **not** interpret it,
does not know the keys `"training"` or `"data_loader"`, and calls no
loader method. `src/tensorforge/experimental/native_checkpoint.py` is
unchanged, the format is still `tensorforge.native_checkpoint` version
**3** with `(1, 2, 3)` accepted, the manifest keeps its same six root
keys, and the array inventory is identical whether or not loader state
travels — **the archive's own capture set did not grow by one field**.
Restoration into an entirely fresh model, optimizer, generator set,
dataset, sampler, and loader reproduces every parameter, persistent
buffer, Adam moment and step counter, generator state and alias topology,
and all six loader values exactly in raw IEEE-754 bit patterns, then the
exact next batch and remaining sequence; a **failed** delivery resumes the
same candidate batch, a **successful** one the following batch, and an
epoch-boundary save the canonical next epoch. There is deliberately **no
cross-object atomicity** between the two calls, and none is claimed.

**J6 added no public name and no production code either**, and shipped the
deterministic mini-batch training example,
`examples/native_minibatch_training.py`. It trains a
`Linear → BatchNorm1d → ReLU → Dropout → Linear → LayerNorm → Dropout →
Linear` classifier over **shuffled** mini-batches drawn through the
dataset/sampler/loader chain, with `NativeAdam`,
`NativeCrossEntropyLoss`, and two Dropout layers sharing one
`NativeGenerator`, and proves an interrupted-and-resumed run **bit-for-bit
identical** to an uninterrupted one: the complete batch-index sequence,
every feature batch's raw bits, every `int64` target array with its dtype,
shape, contiguity, ownership, and read-only flag, every loss, every
parameter and persistent buffer, Adam's moments and step counters, the
generator state and alias topology, the final loader `state_dict()`, and
the evaluation logits, predictions, and metric. It runs at float32 and
float64 **independently, each compared only against itself**, with **no
tolerance anywhere**; the only cross-dtype equality asserted is the
batch-index sequence, which is dtype-independent by construction. The
interruption is genuinely mid-epoch, the resumed graph is entirely fresh
and deliberately built wrong in every family first, native live storage
returns exactly to baseline, and a negative control that omits
`loader.load_state_dict` alone is proved to diverge. The example uses
**public APIs only** and claims no timing. The example inventory moved 15
→ **16**; `tensorforge.experimental.__all__` stayed at **25**.

**J7 added no public name and no production code either**, and shipped the
cross-cutting adversarial hardening matrix,
`tests/test_native_data_hardening.py`. It asserts every §12.7, §15, §16,
and §17 row by injection rather than by argument: each
dataset-construction row, each iteration row — with the host gather, the
native allocation, the host→native transfer, and the target copy kept as
**four distinct injections** — the **commit step made to fail after the
candidate position was really applied**, a `BaseException` through the
same unconditional rollback, the reentrancy refusal matrix at all three
transaction phases, every abandonment position, both close orderings, and
a **checkpoint taken immediately after a failed delivery proved to resume
the same candidate batch** at both dtypes through a real version-3
archive into an entirely fresh graph. Every rejection carries a complete
before/after fingerprint of the observable world — including an unrelated
`NativeParameter` with its version and gradient, a persistent buffer, a
live optimizer, a registered `NativeGenerator`, the filesystem, both
global RNGs, and every registry — and every injection and every parser
has its own non-vacuity control. **It found no production defect.**
Concurrency stays **documented as unsupported rather than tested as
safe**: no Phase-J module contains a lock, none was added, and no test
starts a thread. The example inventory stayed at **16**, the benchmarks
at **8**, and `tensorforge.experimental.__all__` at **25**.

**J8 added no public name and no production code either**, and shipped
the data-pipeline characterization benchmark,
`benchmarks/benchmark_native_data_pipeline.py`, with its contract module
`tests/test_native_data_benchmark.py`. It measures four layers as four
separate questions — immutable host dataset indexing, deterministic batch
planning, deterministic shuffled-permutation construction, and
host→native batch materialization — beside one clearly separate composed
`next(iterator)` delivery case. float32 and float64 are measured
**separately and never as a ratio of one to the other**; correctness is
gated exactly, with no tolerance anywhere, **before** the timing helper is
reached; a case with no honest equivalent is labelled `native_only` and
publishes **no ratio at all**; cold and warm permutation construction are
separate cases and are never averaged; medians are reported with an
interquartile range after warm-up; and setup, per-repetition state reset,
and every `close()` stay outside the timer. **No speed is asserted, no
threshold or CI timing job exists, no result file of any kind is written,
and no optimization shipped** — a change motivated by a J8 measurement
would be a separate, separately approved decision. The benchmark
inventory moved 8 → **9**; the example inventory stayed at **16** and
`tensorforge.experimental.__all__` at **25**.

**J9 closed the phase**, adding no production code, no public name, and no
export: it shipped `tests/test_native_phase_j_closure.py`, re-ran the
complete cross-platform, sanitizer, and leak-detection matrix, and
reconciled every inventory.

**Everything else in this section is permanently unsupported.** There is
**no automatic loader discovery**, in either direction, and none may be
added. **Phase J is complete and no milestone remains.** That sentence
continued "and no successor phase is defined" for as long as it was true;
**Phase K was approved afterwards**, and is recorded in its own section
below rather than folded into this one.

| Registry | Value at J0 | Value at J9 (final) |
|---|---|---|
| `SUPPORTED_DTYPES` | `("float64", "float32")` | unchanged |
| `SUPPORTED_DEVICES` | `("cpu",)` | unchanged |
| `UNSUPPORTED` | `("cuda", "amp")` | unchanged |
| `RAW_KERNEL_DTYPES` | `("float64",)` | unchanged |
| Native checkpoint format | version **3**, accepted `(1, 2, 3)` | unchanged |
| In-memory optimizer state format | version **1** | unchanged |
| Exported production `tf_*` symbols | **54** | unchanged |
| Registered native CTests | **24** | unchanged |

**Phase J was not a capability phase in any dtype, device, or ABI
direction, and it closed without becoming one.** It added **no** new C ABI
export; §22.3 of the contract states
the one measured condition under which that could even be reopened, and it
would require separate approval rather than being a milestone's to take.

The milestone ladder, with its current status:

| Milestone | Subject | Status |
|---|---|---|
| **J0** | architecture and API contract | **complete** |
| **J1** | host-backed dataset foundation | **complete** |
| **J2** | deterministic sampler | **complete** |
| **J3** | native mini-batch loader | **complete** |
| **J4** | loader state and mid-epoch resume | **complete** |
| **J5** | native checkpoint metadata integration | **complete** |
| **J6** | deterministic mini-batch training example | **complete** |
| **J7** | cross-cutting hardening | **complete** |
| **J8** | performance and transfer characterization | **complete** |
| **J9** | integration and closure | **complete** |

What J0 locked, so that later milestones inherit it: three eventual public
names — `NativeTensorDataset` (J1), `NativeBatchSampler` (J2), and
`NativeDataLoader` (J3), with the permutation helpers and the batch
iterator permanently private; a strict `numpy.ndarray`-only input contract
whose native feature dtype is explicitly chosen and **never inferred** from
the input array, still defaulting to float64; copied host snapshots with no
aliasing, and indexing that returns copies rather than views; a
deterministic **SHA-256** dataset fingerprint over an explicit
little-endian canonical byte stream; a sampler owning `batch_size` and
`drop_last` and emitting batch-index groups, with the epoch boundary
canonicalized immediately; a deterministic shuffle that **reuses the locked
`tensorforge.splitmix64` derivation** under a domain-separated key schedule
— no new RNG algorithm, no new global generator, and deliberately no
coupling to a live `NativeGenerator` — with unbiased rejection-based
bounded integers, a downward Fisher–Yates sweep, and committed reference
vectors; a permutation that is a pure function of `(seed, epoch, length)`;
caller-owned `NativeTensor` feature batches beside fresh read-only host
`int64` targets; strict JSON-compatible state schemas carrying no payload
and no serialized permutation; transactional state loading; and an
explicit **caller-managed** checkpoint-metadata workflow.

**The checkpoint's capture set does not grow.** A native checkpoint still
captures no data-loader position, no shuffle order, and no epoch counter,
exactly as the list above says. What Phase J will eventually add is a
loader whose position the **caller** can serialize and hand back through
the existing validated `metadata` channel — which needs no new checkpoint
field, no new version, and no coupling between the checkpoint runtime and
the pipeline.

## Phase K — native integer tensors and indexing, **K0, K1, and K2 complete**

**Phase K is newly approved**, after Phase J closed at J9 without a
committed successor, and **K0, K1, and K2 have landed**. Its
authoritative contract is
[native_integer_tensors_design.md](native_integer_tensors_design.md).

**K0 is architecture, contract, status reconciliation, and guardrails, and
it moved nothing in this document.** It added no integer dtype, no dtype
code, no C++ enumerator, no kernel, no C ABI symbol, no ctypes
declaration, no `NativeTensorCore` method, no `NativeTensor` operation, no
module, no public export, no capability-registry value, no checkpoint
field or version, no optimizer-state version, no loader- or sampler-state
version, no example, no benchmark, no CTest, and no dependency.

**K1 added the internal `int64` representation and every reachability
barrier, and it moved exactly one row in this document — the CTest
count.**

*Now supported internally, and only internally:* a third C++ dtype
enumerator at ABI code **2** (`TF_DTYPE_INT64` / `tf::Dtype::Int64`);
allocation and destruction of genuine `std::int64_t[]` storage through the
existing `tf_storage_create_typed` / `tf_storage_create_uninitialized_typed`
pair; **exact, bit-for-bit** transfer through `tf_storage_copy_from`,
`tf_storage_copy_to`, `tf_storage_materialize`, and
`tf_core_contiguous_copy`, at the signed extremes and beyond 2⁵³; and a
complete floating/autograd/state reachability barrier — `tf::require_floating`
on **32** float-only exports at the C ABI, and the matching Python
barriers in front of wrapper construction, autograd, parameters, buffers
at **both** `persistent` values, both optimizers, checkpoint entry
validation, and every floating operation entry.

**K2 made the `int64` tensor publicly constructible, and it moved exactly
two rows in this document — `INDEX_DTYPES` and the Python `_DTYPE_CODES`
table. It landed atomically.**

*The public delta, stated exactly:* K2 adds **three** public
`NativeTensor` method names — `from_int64_array`, `item()`, and
`tolist()`. `NativeTensor.from_int64_array` is the **only public
construction or host-ingress door** for an `int64` buffer, and it asks
`cpp._normalize_index_dtype` — the `INDEX_DTYPES` registry gate — before
inspecting the input or allocating anything. `item()` and `tolist()` are
**dtype-general host-inspection** methods that construct nothing and
behave identically at `float64`, `float32`, and `int64`. Storage and Core
integer ingress remain private. "One public *door*" and "three public
*names*" are both true; neither is written as the other.

*Now supported publicly, and asserted so by test:*
`INDEX_DTYPES == ("int64",)`, reported as
`backend_info()["index_dtypes"]` **beside** an unmoved
`SUPPORTED_DTYPES`; **`NativeTensor.from_int64_array(values, *,
requires_grad=False)`**, the one public construction door in the
repository through which an `int64` buffer can come into existence;
native `int64` tensors
with the full inherited metadata surface (`dtype`, `shape`, `strides`,
`ndim`, `numel`, `contiguous`, `offset`, `closed`, `owns_core`, `close()`,
`__enter__`/`__exit__`, `__repr__`); the existing borrowing views at
`int64` (`reshape`, `transpose`, `T`, `narrow`), which preserve the dtype
because a view has none of its own; owning `contiguous_copy`; and exact
host inspection through `to_numpy()` (a fresh independent `numpy.int64`
array), `item()` (a built-in `int`, complete signed 64-bit value, no float
intermediate), and `tolist()` (nested built-in containers with Python
`int` leaves).

*Still unsupported at K2, and asserted so by test then:* `int64` in
`SUPPORTED_DTYPES`, `normalize_dtype("int64")`, an `int64` route through
any generic constructor, public `NativeStorage(size, dtype="int64")`, a
public `NativeStorage.from_int64_array` or
`NativeTensorCore.from_int64_array`, `argmax`, `index_select`, integer
arithmetic, integer reductions, integer autograd, integer parameters or
buffers, integer optimizer state, integer checkpoint entries, promotion or
casting, CUDA, and AMP. Two entries have since **moved rather than been
deleted**: `argmax` left the list at **K3** and `index_select` at **K4**,
each shipped by the milestone that owns it and each recorded above. Every
other entry is still unsupported today.

| Registry | Value at J9 | Value at K0 | Value at K1 | Value at K2 |
|---|---|---|---|---|
| `SUPPORTED_DTYPES` | `("float64", "float32")` | unchanged | unchanged | unchanged |
| `SUPPORTED_DEVICES` | `("cpu",)` | unchanged | unchanged | unchanged |
| `UNSUPPORTED` | `("cuda", "amp")` | unchanged | unchanged | unchanged |
| `RAW_KERNEL_DTYPES` | `("float64",)` | unchanged | unchanged | unchanged |
| `normalize_dtype(None)` · `backend_info()["dtype"]` | `"float64"` | unchanged | unchanged | unchanged |
| `normalize_dtype("int64")` | raises `ValueError` | unchanged | unchanged | unchanged |
| `INDEX_DTYPES` | absent | absent | absent | **`("int64",)`** |
| Python `_DTYPE_CODES` | `float64`, `float32` | unchanged | unchanged | **+ `int64: 2`** |
| C++ `TfDtype` | `FLOAT64`, `FLOAT32` | unchanged | **+ `INT64 = 2`** | unchanged |
| Public integer constructor | absent | absent | absent | **`NativeTensor.from_int64_array`** |
| Native checkpoint format | version **3**, accepted `(1, 2, 3)` | unchanged | unchanged | unchanged |
| In-memory optimizer state format | version **1** | unchanged | unchanged | unchanged |
| Loader state · sampler state | version **1**, accepted `(1,)` | unchanged | unchanged | unchanged |
| Exported production `tf_*` symbols | **54** | unchanged | unchanged | unchanged |
| `tensorforge.experimental.__all__` | **25** | unchanged | unchanged | unchanged |
| Native CTests · examples · benchmarks | **24** · **16** · **9** | unchanged | **25** · **16** · **9** | unchanged |

**What is *not* supported, and is not a bug at K2.** `int64` is **not a
supported native tensor dtype**: it is an **index/result** dtype, in its
own registry, and the two rows answer two different questions — at what
dtypes the kernels *compute*, and what dtypes a native tensor may *carry*
as exact integer data. No omitted `dtype` ever selects it, no generic
constructor accepts it, and no kernel computes at it.

**K3 shipped the phase's first operation and its first C ABI symbol: native `argmax`.** `NativeTensor.argmax(axis=None, keepdims=False)` and `NativeTensorCore.argmax(axis=None, keepdims=False)` search a **floating** tensor at either dtype, at any rank including 0, contiguous or not, and return a fresh owning contiguous **`int64`** tensor — the one operation in the runtime whose result dtype differs from its operand's, which is what an index is. Shapes come from the existing `reduce_shape` authority and axes from the existing `_normalize_axis_checked`, with the axis validated **before** `keepdims`; the value rule is exact rather than adjectival — equal maxima give the **lowest** index, `+0.0` and `-0.0` tie, an all-`-inf` run gives 0, and the **first** NaN wins against every finite value and either infinity. The result is a plain leaf **even when the input requires gradients**, because an index has no derivative, so `"argmax"` joined `TENSOR_CORE_OPS` and deliberately not `AUTOGRAD_OPS`. Exports went **54 → 55** (`tf_core_argmax`, against a phase maximum of 56) and native CTests **25 → 26**; `__all__`, every registry, every version, the examples, and the benchmarks are unmoved, and **no `max` was shipped beside it**.

**K4 shipped the phase's one index-*consuming* operation and its second and final C ABI symbol: native `index_select`, forward only.** `NativeTensor.index_select(axis, indices)` and `NativeTensorCore.index_select(axis, indices)` take a **floating** source at either dtype, any rank ≥ 1, contiguous or not, together with a **rank-1 `int64`** index tensor — never a NumPy array, a list, a tuple, or a Python `int`; a caller with host indices goes through `from_int64_array` first — and return a fresh owning contiguous tensor of the **source's** dtype whose selected axis has `indices.numel` positions. It is `argmax`'s mirror image, and the two compose directly. Duplicates and order are preserved exactly, negative and out-of-range indices are **rejected rather than wrapped**, the complete bounds scan runs in Python *and* independently in C++ before the first destination element is written, and values cross by **object representation**, so signed zeros, infinities, subnormals, and NaN payloads survive bit for bit. It is **forward only**: a source with `requires_grad=True` is rejected with a message naming `detach()` rather than silently detached, so `"index_select"` joined `TENSOR_CORE_OPS` and deliberately not `AUTOGRAD_OPS`. Exports went **55 → 56** (`tf_core_index_select` — the phase maximum, now reached) and native CTests **26 → 27**; `__all__`, every registry, every version, the examples, and the benchmarks are unmoved.

**K5 is the compatibility proof, and it added zero production code.** Its whole deliverable is `tests/test_native_integer_compatibility.py`, which drives every surface K1–K4 could have disturbed and shows none of them moved: no checkpoint archive can declare an `int64` entry at a parameter, persistent-buffer, optimizer-moment, or optimizer-parameter entry, at any accepted version, and such a load rejects **before** publishing anything and without allocating an `int64` storage; the checkpoint format and version (`tensorforge.native_checkpoint`, **3**, accepting `(1, 2, 3)`, with no version-4 constant written, reserved, or accepted), the in-memory optimizer-state version (**1**), and the loader and sampler state versions (**1**, accepting `(1,)`) are exactly what Phase J left; a version-1 archive still loads under its legacy rules and versions 1 and 2 stay float64-only; parameters, buffers at both persistence values, and both optimizers still refuse a real `int64` tensor; the data pipeline still delivers a floating `NativeTensor` feature batch and a read-only host `numpy.ndarray` target batch of dtype `int64` at both feature widths, and gained no option able to request a native label; explicit caller conversion through `NativeTensor.from_int64_array` works on a delivered batch and feeds `index_select` with no pipeline change; `NativeCrossEntropyLoss` accepts and refuses exactly what it did, with every accepted target form giving a bit-identical loss; `native_accuracy` still succeeds with the native `argmax` and `index_select` patched to raise, proving it calls neither; and a real classifier trains, checkpoints, and resumes **bit-identically** at float64 and float32 while `argmax` and a detached `index_select` run beside the training path, with an observational control proving the indexing changes no trainable state. Exports stayed **56**, CTests **27**, examples **16**, benchmarks **9**, and `__all__` **25**. The proof also found one real defect and it is recorded rather than hidden: driving the two module registration routes with a deliberately forged `NativeParameter` showed that `save_native_checkpoint` trusted whatever dtype live state reported, so the **writer** could emit an archive declaring an `int64` entry that its own loader then refused. That was a pre-existing gap in the writer, unreachable through any public API, and it was repaired in a **separate checkpoint-hardening change committed before K5** — a save-side persisted-dtype authority asking the same `cpp.normalize_dtype` question the loader asks, applied in the model preflight and again at the serialization seam, with its own regression in `tests/test_native_checkpoint.py`. No format, field, version, capability, registry, export, CTest, example, or benchmark moved, and K5 itself stays the test-and-documentation milestone.

**K6 is the end-to-end integration example, and it added zero production code.** Its deliverable is `examples/native_integer_indexing.py`, owned by `tests/test_native_integer_indexing_example.py`: a deterministic `NativeLinear(5 → 8) → NativeReLU → NativeLinear(8 → 4)` classifier with `NativeCrossEntropyLoss` and `NativeAdam`, trained over the Phase-J pipeline for ten shuffled batches of six, interrupted **strictly mid-epoch**, and resumed through a real version-3 archive into an entirely fresh object graph proved different before the load. At four fixed steps — two on each side of the interruption — the step's own logits become native `int64` predictions through `argmax` and are then consumed by `index_select` over a **detached** copy of those logits, the two sources differing deliberately because `argmax` returns a plain leaf even from a gradient-tracking input while `index_select` rejects one with a message naming `detach()`. That call is **axis selection, not a per-row gather**: a `(6, 4)` logits batch and a `(6,)` index vector give a `(6, 6)` result whose column *j* is the whole source column `predictions[j]` and whose **diagonal** is each example's own predicted-class logit, both recomputed from raw bit patterns by the owner test, with duplicate predicted classes guaranteed by pigeonhole and proved to give identical columns in their original positions. The uninterrupted and resumed runs agree exactly at float64 and float32 **independently** — every prediction index by exact integer equality, every floating value by raw IEEE-754 bits — and the omitted-loader-state control is proved to diverge. Cross entropy still trains on the loader's read-only host `int64` target arrays. **Examples moved 16 → 17**, the one inventory K6 touches; exports stayed **56**, CTests **27**, benchmarks **9**, and `__all__` **25**, with no registry or version movement and no C ABI symbol.

**K7 is the adversarial hardening milestone, and it added zero production code.** Its whole deliverable is `tests/test_native_integer_hardening.py`, plus status reconciliation. It drives the four injection families — host validation or normalization, native allocation through the backend's **own** thread-local arm, host→native transfer or materialization, and kernel execution — at **every** actual allocating step of `from_int64_array`, `int64` `contiguous_copy`, `argmax`, and `index_select`, resolved from the live call graph into a traceable path-by-position matrix in which a family that genuinely cannot exist on a path is an `N/A` carrying its technical reason rather than a borrowed injection, and in which a **family is not a position**: `index_select` reaches `tf_core_contiguous_copy` at two different call sites — its floating source and its `int64` index — so it carries **two** materialization rows, the second driven by a call journal that delegates the first call to the real export (an injection that fails immediately can never reach it) and proving the two distinguishable by dtype, element count, and rank. Each allocation row arms the real countdown immediately before the production seam it names; each kernel and publication row runs under both an `Exception` and a `BaseException`, at **both** floating widths with each width proved only against itself, while holding the allocated core or storage in an external list, so closure is proved by production cleanup rather than by `__del__` timing; the three-allocation `index_select` failure proves reverse-order cleanup and a per-object release count proves **exactly once** — a cleanup invariant at a position the matrix already names, traced as such rather than added as a row. Every rejection and every injected failure is bracketed by one reusable fingerprint of the observable world — both operands, an unrelated parameter with its version and gradient, both buffer kinds, a live optimizer, a registered generator, every capability registry, dtype table, operation inventory, format and version, `experimental.__all__`, both global RNGs, the environment, a watched directory, the live-storage count, and the native error slot — and every component, every injector, and every parser has a non-vacuity control; a scan of the module's own AST requires every matrix owner to enter that fingerprint, so a narrower check can never be reported as the complete one, and where a rejection needs a deliberate instrument (an emptied `INDEX_DTYPES`, a lowered `_INT64_MAX`) the instrument is applied first and the snapshot taken after it. The two exports keep **separate** malformed-metadata matrices **and separate dtype-role matrices**, because their validation lists are separate: each prefills **every** operand with distinctive values and proves not one byte moved in any of them — two handles for `argmax`, three for `index_select` — with no native allocation and a clean error slot afterwards, and a late invalid index following three valid ones is proved to leave the whole destination byte-identical at both widths. **K7 found no production defect**, performed no native build, and moved no inventory: exports **56**, CTests **27**, examples **17**, benchmarks **9**, `__all__` **25**, and every registry and version exactly what K6 left.

There is **no `max`**, no `max_with_indices`, no `argmin`, no general `gather`, no `scatter`, no `scatter_add`, no embedding lookup, no `__getitem__` or advanced/boolean/multi-axis indexing, no `index_select` backward, no integer arithmetic or reduction, no integer autograd, parameter, buffer, optimizer state, or checkpoint entry, and no casting or promotion, and **K8 through K9 are unstarted**.

**What the contract decides**, so that a later milestone implements rather
than re-argues it: one extended `NativeTensor` rather than a parallel
integer tensor class (a separate class would duplicate storage ownership,
view ownership, `close()` semantics, host transfer, and failure cleanup
without preventing anything the dtype checks do not); storage keeps owning
the dtype and views keep inheriting it; **`int64` is the only integer
dtype in the phase** — `int32`, `int16`, `int8`, every unsigned width,
`bool`, complex, `float16`, and `bfloat16` are all deferred and rejected —
and it is an **exact, non-differentiable index/result dtype**; one strict
`numpy.ndarray`-only construction door with **no dtype inference, no
numeric cast, no truncation, no widening, and no byte-swapping**, where a
non-contiguous exact-`int64` array is copied because layout normalization
is not conversion; integer autograd, parameters, optimizer ownership,
module buffers, and checkpoint entries barred in Python **and**
independently at the C ABI; a complete `argmax` contract (floating input,
`int64` output, `axis`/`keepdims`, first-occurrence ties, signed-zero
ties, **NaN propagates and wins**, no graph ever, and **no `max` exposed
beside it**); a forward-only `index_select` (rank-1 `int64` indices,
negatives rejected rather than wrapped, **every index bounds-checked
before the destination is allocated**, duplicates and order preserved, and
a source that requires grad rejected rather than silently detached); the
Phase-J delivery contract `(NativeTensor, numpy.int64)` left exactly as it
is; **no checkpoint version change and no version 4**; and a C ABI budget
of exactly **+2** symbols — `tf_core_argmax` at K3 and
`tf_core_index_select` at K4 — for a **phase maximum of 56**.

**The one public registry movement of the phase is assigned to K2**, and
it is **not** `SUPPORTED_DTYPES`: that row remains the floating-compute
registry permanently and never gains `int64`, so every generic constructor
keeps rejecting the dtype by name and none of them changes at any
milestone. What appears instead is a separate `INDEX_DTYPES ==
("int64",)` row beside it, in the same commit as the public constructor
and one milestone after every reachability barrier has landed at **K1**. Until then this section's table is the answer: *prove first, then
promise*, the rule Phase G used for `dropout` and Phase I for `float32`.

## How to build and verify

The native backend is built with CMake (`cpp/CMakeLists.txt`), wrapped by
the cross-platform `cpp/build.py` (which falls back to a direct compiler
invocation — `g++`/`clang++`/`ziglang` — when CMake is unavailable, as on
CI). Every fallible native export is exception-guarded so no C++ exception
crosses the ABI; native failures surface as `MemoryError` / `ValueError` /
`RuntimeError` (see docs/native_abi_error_contract.md). All commands are
verified against this repository:

```
uv sync                                                # dependencies
uv sync --group cpp                                    # only if no system C++ compiler
uv run python cpp/build.py                             # build the native backend (Release)
uv run python cpp/build.py --debug                     # unoptimized debug build
uv run python scripts/smoke_cpp_backend.py             # hard-failing smoke check
uv run python examples/native_tensor_demo.py           # runtime and views
uv run python examples/native_autograd_demo.py         # native backward
uv run python examples/native_mlp_training.py          # end-to-end training proof
uv run python examples/native_checkpoint_resume.py     # save, restore, resume bit-for-bit
uv run python examples/native_cnn_training.py          # end-to-end CNN training + resume proof
uv run python examples/native_classification_training.py  # native classification + exact resume
uv run python examples/native_float32_training.py      # integrated float32 + float64 exact resume
uv run python benchmarks/benchmark_native_autograd.py --smoke
uv run python benchmarks/benchmark_native_cnn.py --smoke   # CNN characterization
uv run python benchmarks/benchmark_native_classification.py --smoke        # classification characterization
uv run python benchmarks/benchmark_native_classification.py --smoke --json # machine-readable
uv run python benchmarks/benchmark_native_cpu_performance.py --smoke       # Phase-H CPU baseline
uv run python benchmarks/benchmark_native_cpu_performance.py --workload matmul
uv run python benchmarks/benchmark_native_cpu_performance.py --profile matmul_square_contiguous
uv run pytest                                          # full suite (native tests skip if unbuilt)
```

Sanitizer validation (Linux/WSL with Clang; MSVC supports neither
`-fsanitize=undefined` nor this option form):

```
cmake -S cpp -B build/phase-d-sanitizers -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_COMPILER=clang++ -DTF_SANITIZE=address,undefined \
      -DTF_BUILD_TESTS=ON
cmake --build build/phase-d-sanitizers
ctest --test-dir build/phase-d-sanitizers --output-on-failure
```
