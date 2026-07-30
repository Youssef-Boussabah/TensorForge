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
complete, and is the latest *completed* phase: milestone G0, the
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
- **Targets** are not native tensors (the runtime has no integer dtype):
  they are copied into an independently owned, contiguous, read-only
  `int64` array before anything is allocated. `bool` and floating-point
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
  loading, checkpoint merging, sharding, compression, or encryption
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
  `argmax` (the runtime has no integer dtype for one to return)
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
- float32 / float16 / bfloat16, dtype promotion or casting, AMP
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
and, since G10, the **capability** to match: `UNSUPPORTED` reads
`("float32", "cuda", "amp")`. The gap through G0–G9 was deliberate: the
registry reports a *closed, validated* capability, and a capability whose
value is exact reproducibility is not finished until reproducibility has
been demonstrated under fresh Release and Debug builds, the sanitizers,
and a checkpoint that can actually persist the stream. All of that has now
been demonstrated.

The claim stays narrow, and is worth stating precisely: **native Dropout
is supported in TensorForge's experimental native float64 CPU backend.**
That is not a statement about the stable framework (`tensorforge.nn.Dropout`
has always been its own separate NumPy implementation), and `float32`,
`cuda`, and `amp` remain unsupported. There is still no generic
`rand`/`randn`/Bernoulli/sampling API, no global or process-wide random
state, and no `Dropout2d`/`Dropout3d`.

One contract detail is recorded rather than glossed: the design's
**empty-tensor** row (`count == 0` draws nothing and consumes one call)
is implemented in the kernel and the C ABI, but the native tensor
representation rejects zero-size dimensions outright, so no empty
`NativeTensorCore` can be constructed to exercise it from Python today.
G2 proves the case at the two layers where it is reachable and pins the
representation's limit with a test.

## Phase H — native CPU performance and runtime efficiency, **begun (H0–H6)**

**Phase H is the current phase. It has begun, and milestones H0, H1,
H2, H3, H4, H5, and H6 are complete.**
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
H4, H5, and H6 moved no capability either. `UNSUPPORTED` still reads
`("float32", "cuda", "amp")`, `SUPPORTED_DTYPES` still reads
`("float64",)`, `SUPPORTED_DEVICES` still reads `("cpu",)`, and the
native checkpoint format is still `tensorforge.native_checkpoint`
version **2** with versions **(1, 2)** supported. **Phase G therefore
remains the latest completed phase.**

| Milestone | What it shipped | Status |
|---|---|---|
| H0 | The Phase-H architecture contract ([native_cpu_performance_design.md](native_cpu_performance_design.md)): why CPU efficiency precedes CUDA and dtype expansion; the measured bottleneck evidence, separated into *directly measured*, *strongly source-evidenced but not fully measured*, and *unconfirmed hypotheses*, with the minimal instrumentation a later milestone would need where H0's observability could not settle a question; the representative workload families and the shape-selection rules for the smoke, full, and profiler configurations; the timing, warm-up, repetition, and statistics methodology; the correctness-before-timing rule; the exact-versus-tolerance policy and the floating-point **accumulation-order** policy, whose default is that Phase H preserves the existing order bit-for-bit and whose five-condition escape hatch requires every existing exact-resume proof to be re-established; the determinism policy; the optimized-contiguous-dispatch, strided-fallback, and **retained generic reference path** rules; the invariants Phase H may not weaken; the allocation, scratch-workspace, SIMD, threading, and optional-BLAS decision criteria (all four currently **rejected on evidence**, with the criteria recorded rather than an answer invented); the cross-platform and sanitizer requirements; the conditional H0–H11 ladder (H0–H10 as drafted; H5 inserted the copy and mutation-transfer milestone and pushed reduction execution to H6); the explicit non-goals; the closure requirements; and the recorded adopt/adapt/reject decision for every relevant Daedalus idea. Plus `benchmarks/benchmark_native_cpu_performance.py`, the unified baseline harness — 24 cases at H0 across 12 workload families (H3 later added two more, taking it to 26, and H5 two more again, taking it to 28), measured across up to nine declared implementation layers (`numpy`, `stable_tensorforge`, `raw_kernel`, `raw_kernel_tiled`, `tensor_core`, `native_tensor`, `native_tensor_graph`, `backward`, `optimizer_step`, `training_step`), with a correctness gate that runs **before** the timing helper is ever reached, honest reference labelling (a case with no honest equivalent publishes **no ratio at all** and says why), `--smoke` / `--json` / `--case` / `--workload` / a focused `--profile CASE` mode, deterministic seeded inputs, explicit cleanup with no reliance on garbage collection, fresh state per training-step repetition, an explicit reset generator for the Dropout case, and **no result file of any kind**. Plus `tests/test_native_cpu_performance_benchmark.py`. **No numerical capability, no optimization, no registry move** | **Complete** (architecture, profiling, and baseline only — nothing was made faster) |
| H1 | **The explicit output-allocation contract.** Redundant zero-initialization removed from output storage a kernel *provably* overwrites in full. One new production C ABI symbol, `tf_storage_create_uninitialized`, sharing one file-local body with `tf_storage_create` so the two cannot drift apart: identical size validation, zero/negative rejection, fault injection, allocation-failure handling, error state, handle shape, ownership, destruction, and live-storage accounting, differing **only** in the buffer initial contents. The zero-initializing path stays the default and is byte-for-byte unchanged. On the Python side, one private keyword on `NativeStorage.__init__` (so both allocation kinds pass through the one constructor every live-storage hook wraps), plus the private `NativeStorage._uninitialized` / `NativeTensorCore._uninitialized` helpers — **no** public `empty` API, no registry capability, no stable-framework surface. Every enabled call site opts in explicitly against a per-kernel audit table (design 16.1.2); there is no global switch, environment variable, heuristic, memory pool, or scratch arena, and a failed precondition means the safe path. **Enabled:** the elementwise/unary family across all three traversal paths, `relu`, `sqrt`/`reciprocal`/`exp`/`log`, `contiguous_copy`, `matmul` (at H1 it accumulated into a *local* register and never read the destination; **H2 restated the reason for the optimized path it added** — the row sweep's `k == 0` pass assigns every element of every row before anything accumulates into it, so the destination is still never read before it is written, and both paths now carry poison proofs), `relu_backward`, `softmax`/`log_softmax`, both cross-entropy directions, `conv2d_forward`, **both convolution gradients and the pooling gradient** (their kernels zero their own whole span before accumulating, so the caller fill was pure duplication), `maxpool2d_forward` values and winners, both Dropout destinations, `from_array`, and `full`. **Rejected:** `sum`/`mean` (it accumulates into its output, so the zero is the additive identity) and `narrow_backward` (it writes only the narrowed region, and the untouched zeros *are* the gradient) — both pinned by negative-control tests. Completeness is proved by **deterministic poison** tests, separately from ASan and UBSan, which do not detect uninitialized-*value* reads (MemorySanitizer would, but it needs an instrumented libc and CPython this project does not have, so no MSan result is claimed). The poison is injected **exclusively by test infrastructure around the allocator**: the suite wraps the private `NativeStorage._uninitialized` helper — the single funnel every uninitialized allocation passes through — lets the real constructor allocate, fills the returned storage with a quiet NaN or a large nontrivial finite pattern through the ordinary `fill` primitive, and hands **that same storage** to the real operation, so the pattern is in place after the real allocation and before the real kernel. **The shipped library and the installed Python backend contain no poison control whatsoever** — no exported hook, no thread-local flag, no environment variable, no global mode — and a scope test asserts that against the loaded image's own export table. Four negative controls prove the detector can actually fail, including a real complete kernel given a deliberate hole. Bit-identical: every enabled operation and a complete eight-step training run are compared element-wise against the zero-initializing allocator, with the uninitialized side poisoned so a hole cannot match by luck. No capability, dtype, device, registry value, checkpoint field, or checkpoint version moved, and `tf_storage_create_uninitialized` is the only added export (51 → **52** exported `tf_*` symbols) | **Complete** |
| H2 | **Native matmul memory access.** `tf_core_matmul` ships two compute paths behind the same unchanged export. `tf::matmul_generic_strided` is the pre-H2 `i`-`j`-`k` triple loop kept verbatim — the **retained generic reference path** (design 8.3), shipped, reachable through ordinary production dispatch, and the oracle every optimized result is compared against. `tf::matmul_row_sweep` is the optimized path: `i`-`k`-`j` over `MATMUL_ROW_BLOCK` = 4 destination rows at a time, so the innermost loop walks a row of the right operand and a row of the output sequentially rather than walking a column. **Preconditions, all read from metadata and all required:** the right operand's column stride is exactly 1, the inner dimension is non-empty, and the result has at least `MATMUL_MIN_COLUMNS` = 8 columns. **Everything else takes the generic path** — a transposed right operand, any other non-unit column stride, a narrower result, an empty inner dimension — and that is a design choice rather than a gap, because the `i`-`j`-`k` order is the better one when the right operand's *rows* are the unit stride. A transposed *left* operand beside a row-major right one (which is `db = a.T @ upstream` in the matmul backward) does qualify. Selection is total, pure, deterministic, side-effect free, and independent of pointer values, alignment, wall time, environment variables, and CPU-feature probes; a failed precondition is a fallback, never an error. **Cache blocking was measured and rejected**: 22 blocked variants were timed against unblocked row sweeps across 25 shapes, and the unblocked sweep was faster at every non-trivial size (5.50x versus 3.33x at 384 cubed), so H2 shipped the simpler superior design and published the negative result. Both constants are compile-time values in a shipped header — no autotuning, no runtime probe, no stored machine-specific measurement. **A four-part numerical contract, not a blanket bit-identity claim.** (1) Accumulation order is preserved exactly: for each output element the row sweep starts from the same 0.0 and takes the same products in the same ascending `k` order, with the `0.0 +` on the `k == 0` pass written out deliberately because `0.0 + (-0.0)` is `+0.0`. (2) **Every non-NaN result is bit-identical** — signed zeros, infinities, denormals, the smallest normal and the largest finite magnitudes included — asserted as raw IEEE-754 bit patterns, not tolerances, in the C++ CTest across a 1-65 dimension sweep including primes and both sides of every boundary, in Python across the Core, `NativeTensor`, the autograd node, `NativeLinear`, and both optimizers, and in the harness gate before any timing. (3) **NaN-class equivalence**: NaNs occur in exactly the same positions on both paths and are always quiet; neither path produces a signaling NaN. (4) **NaN payload bits are outside TensorForge's numerical contract** and may differ — measured at 162 of 208 in the MSVC Release build and 0 of 208 in the Debug and Clang builds, with ten source-level formulations tried and only the `i`-`j`-`k` order H2 replaces able to reproduce the reference's payloads. Deterministic training and exact checkpoint resume stay bit-identical for supported finite workloads, which part (2) covers completely. **H1 compatibility** holds on both paths for different reasons — the generic path never reads the destination, and the row sweep assigns every element of every row in its `k == 0` pass before anything accumulates into it — proved by poison tests over both paths with both patterns, a bit-identity check across NaN-poisoned, finite-poisoned and zeroed destinations, and a partial-write negative control. The pre-existing raw `tf_matmul_tiled` was inspected and **not** adopted: it takes plain contiguous buffers rather than storage handles, carries no guard or error contract, and zeroes its destination before accumulating — a full extra write pass, which is exactly what H1 removed. It stays as the standing benchmark experiment on no production path. **No C ABI symbol added** (still **52** exported `tf_*`), no kernel selector, block-size setter, benchmark hook, dispatch tracer, reference-kernel selector, CPU-feature control, or environment variable; no threading, SIMD, OpenMP, BLAS, memory pool, or scratch workspace; no capability, dtype, device, registry value, checkpoint field, or checkpoint version moved | **Complete** |
| H3 | **Native metadata and dispatch efficiency — Python-only.** Repeated Python-side metadata normalization removed from the path to a kernel, with **no C++, no C ABI symbol, no ctypes declaration, and no kernel** touched (export count stays **52**). The measured cause was redundant *re-validation*: one `shape_info` call ran `_as_int_tuple` **four** times over a tuple fully validated after the first pass and computed the row-major strides **twice**, while `NativeTensorCore.zeros` validated the caller's shape a second complete time — **815** `_as_int_tuple` calls per MLP training step, 604 per `NativeAdam` step, instrumented by test-local monkeypatching with **no production counter**. Three pieces. (1) **One normalization boundary**: the private `_normalized_layout` performs exactly the checks `shape_info` always performed, in the same order and with the same messages, and normalizes the shape once; the derived strides, element count, and contiguity come from private `_checked` primitives that validate nothing because nothing is left to validate, and each public helper (`row_major_strides`, `numel`, `reduce_shape`, `broadcast_shapes`) is now its own validation plus the matching primitive, so the two cannot disagree. (2) **Two view constructors, one binding**: the public `NativeTensorView(...)` normalizes; the private `_from_validated` skips **only** that normalization; both funnel through a shared `_bind` that still performs the storage open check and the **full reachable-offset bounds check** — not skipped, because bounds depend on the storage, not the metadata. The element count and contiguity flag are **derived inside** the private constructor rather than passed to it, so an inconsistent pair is unrepresentable — which is why H3 ships a separate private constructor rather than a misusable `validated=True` flag. (3) **Per-view layout arrays**: the `int64` shape/stride arrays the strided C ABI takes, memoized **lazily** and **read-only**. Staleness is impossible by construction, not by invalidation: a view's layout is assigned exactly once, in `_bind`, and `reshape`/`transpose`/`T`/`narrow` all return *new* views, so no invalidation is ever required and **none exists**. Nothing global was introduced — no shape cache, stride interning, weak-reference machinery, or thread-local state — and **no validation was removed**: every rejection still happens with the same exception type, the same message, and the same shape-then-strides-then-offset ordering, asserted by feeding the constructor metadata with two and three simultaneous faults. Cold object footprint is **byte-identical**; a view that actually takes a strided path costs +328 bytes, and in a full MLP step only **5 of 134** views ever populate it. Measured: `shape_info` 2.6–4.5×, view construction 3.2×, `_as_int_tuple` per MLP step **815 → 149**, a one-element allocation 2.1×, `reshape` 3.1×, a small `add` 1.56×, `NativeAdam` on a small MLP 1.42×, an **MLP training step 1.43×**, a **CNN step 1.29×**, a **normalized step 1.51×** — and **no measurable change on large kernel-bound work** (384³/512³/128³ matmul, 256² elementwise, 128² reduction all inside their own spread), published as such. **No public API added** of any kind: no cache control, statistic, reset, profiling counter, dispatch selector, or environment variable. No capability, dtype, device, registry value, checkpoint field, or checkpoint version moved | **Complete** |
| H4 | **Native optimizer step efficiency — Python-only.** The fixed native allocation and call count of `NativeAdam.step()` reduced, and `NativeSGD.step()` changed only where the evidence supported it, with **no C++, no C ABI symbol, no ctypes declaration, and no kernel** touched (export count stays **52**). It is the first Phase-H milestone whose subject is a *training-stack* component rather than the tensor runtime. B4's counts were **re-instrumented on the current post-H3 code** rather than taken from H0, and H0's figure was confirmed exactly: **27 native storage allocations per parameter per Adam step**, of which **ten are one-element** — eight broadcast scalar coefficients (`beta1`, `1 - beta1`, `beta2`, `1 - beta2`, both bias-correction terms, `eps`, and `lr`; design 3.2 said six, and `eps` and `lr` were the two it missed) plus the two `reciprocal` outputs taken on one-element tensors — with `NativeSGD` at five per parameter, and **8 of Adam's 13 binary operations** taking the broadcasting path rather than the contiguous fast path. Three changes. (1) **The step's scalar coefficients are built once per step, not once per parameter**: a private per-step `_StepConstants` holder builds each on first use, keyed by `(dtype, device)` so it never assumes one dtype exists, and hands the same read-only core to every later parameter; the two bias-correction terms are cached per step *counter*, so steady-state training builds one pair while a parameter that skipped earlier steps legitimately gets its own. The holder allocates nothing until the first entry asks for a coefficient — so a step with no active parameter allocates nothing at all — is released before the commit begins, and is **never stored on the optimizer**, so no scalar survives a step, enters `state_dict()`, reaches a checkpoint, or has to be released by `close()`. `NativeSGD` does the same for its single `lr` scalar, the only change its evidence supported. (2) **The bias-correction reciprocal is evaluated in Python**, removing one allocation and one kernel call per coefficient per parameter. This is an **exact substitution, not a reassociation**: the kernel literally is `double op_reciprocal(double x) { return 1.0 / x; }`, a Python `float` and a C++ `double` are the same IEEE-754 binary64 value, and IEEE-754 requires division to be correctly rounded, so there is exactly one possible result — proved over **20,000+ values** spanning the full exponent range, ±0, ±∞, the smallest subnormal, the largest finite magnitude, and every `1 - beta ** t` the optimizer actually forms, compared on **raw `uint64` bit patterns** with zero mismatches. (3) **Temporaries are released at their last use** rather than all together at the end of the staged expression. **Bit-identical, with no carve-out of the kind H2 needed**: no accumulation order, operand position, or kernel changed, so NaN payloads match too. The **pre-H4 composition is retained in the test suite** as a literal transcription executed natively, and every equality is against that rather than a NumPy re-derivation — 60 shape/step/hyperparameter combinations for Adam (including `beta = 0`, betas at `0.99999`/`0.9999999` with `eps = 1e-30`, and `lr = 1e10`), a six-step run over four mixed shapes, and four SGD learning rates from `1e-9` to `1e12` — while a separate test pins the **exact operation sequence** a staged entry issues so a future reorder or fusion fails loudly. **The two-phase contract is untouched**: validation is still four complete passes in the same order with nothing moved behind a mutation; stage mutates no parameter, moment, counter, version, or gradient; the commit is still **one `copy_value_` and exactly one version increment per updated parameter**; gradients are read and never written, by identity, value, and storage identity; and the documented per-entry commit boundary is *tested* by injecting a `copy_value_` failure rather than assumed infallible. Measured by alternating pre/post **subprocess** rounds so drift affects both arms equally, 366 samples per case: **1.58×** at (128, 128), **1.54×** at (256, 256), **1.48×** on a four-parameter MLP with a 256² weight, 1.21–1.22× on a small MLP, 1.15× on a first step; a large MLP training step 1.23×, a small one 1.15×, a normalized step 1.13×, a CNN step 1.09×; and in the shipped harness `adam_step` 1.25×, cutting the gap against `tensorforge.optim.Adam` from **23.8× to 19.7×**. Reported just as honestly: **a (512, 512) parameter is neutral** (1.02×, memory-bandwidth-bound), the **Dropout training step is neutral** (0.99×), and **NativeSGD is neutral-to-slightly-positive** (1.03–1.07×) with one 0.88× row identified as **noise** by a focused re-measurement whose post minima were lower in every pair — and the machine's control-case noise band is stated at **0.84×–1.26×** so no reading inside it is mistaken for a result. **Memory moved with time, not against it**: peak live transient bytes during one Adam step fell **2.6–3.0×** (1,966,160 → 655,424 for a (128, 128) parameter) and per-parameter allocations 27 → **17**, so a four-parameter model allocates **76 instead of 108**. **Six alternatives measured and rejected**, each with its reason recorded: scalar materialization (faster below ~32K elements, slower above, and it would regress the harness's own profile configuration while adding a parameter-sized buffer per scalar operation); same-shape stride-0 views (identical kernel arguments by construction, but four NumPy layout arrays per call where the broadcast path builds three); adopting the staged core instead of `copy_value_`; giving `_native_copy` a `contiguous_copy` implementation (it would stop normalizing `-0.0` to `+0.0`, a real observable change in a helper shared far beyond the optimizer); a persistent per-optimizer scalar cache (the hidden scratch tensor the design forbids); and reassociating the update to fold scalars together (a floating-point order change that would break every exact-resume proof). **No public API added** of any kind: no cache control, statistic, reset, profiling counter, dispatch selector, failure toggle, or environment variable. No capability, dtype, device, registry value, checkpoint field, or checkpoint version moved | **Complete** |
| H5 | **Native copy and mutation-transfer efficiency.** The native line's value-transfer primitive replaced: `_native_copy` was `zeros(shape) + core` (two allocations, a zero-fill pass, and an elementwise-addition pass) and is now the E3.1 native identity gather `contiguous_copy()` (one uninitialized allocation, one pass). A complete inventory found **ten** call sites of that one helper — `copy_value_` staging, both `state_dict()` snapshots, both `load_state_dict()` stagings, both BatchNorm running-statistic commits, and the reshape/transpose/unbroadcast gradient materializations — all of them pure value transfers, and all ten were enabled; `_broadcast_back` was **rejected** because it is a genuine broadcast expansion rather than a copy, and `sum`/`mean` and `narrow_backward` keep their zeroed destinations for H1's unchanged reasons. The semantic question H4 refused to decide in passing was decided by measurement over a fixed 18-pattern IEEE-754 sweep: **exactly three** patterns moved — the addition normalized `-0.0` to `+0.0` and quieted both signs of signaling NaN — while every other pattern, including every NaN payload, was already identical (nothing differed there), so **H2's matmul NaN-payload carve-out does not generalize to copies**. The pre-H5 behavior was accidental and inconsistent, not contracted: `NativeParameter(source)` construction, `detach()`, and the `to_numpy()`/`from_array` boundary always preserved `-0.0`, while `copy_value_` documented the same thing and did not. H5 states the narrowest coherent rule — **a value transfer reproduces its source's bits exactly; an operation follows IEEE arithmetic** — and changed no operation's arithmetic anywhere. One C++ change, and **no ABI change**: a second *traversal* inside the unchanged `tf_core_contiguous_copy` export, chosen by `tf::copy_prefers_contiguous` — hidden-visibility C++ in a new internal header, total, pure, allocation-free, a function of layout metadata alone, never of a pointer value, alignment, clock, environment variable, or CPU-feature probe — sweeping a row-major source with the flat pointer loop and falling back to the retained odometer otherwise. **No numerical carve-out is needed**, unlike H2: the identity map performs no arithmetic, so the two traversals are bit-identical *by construction*, proved at the C++ level by a new dependency-free CTest (13 to 14). Nothing became in-place — every call site still stages an independent materialization before adopting it — so self-copy, a source that views the destination's own storage, a square parameter's own transpose, sibling views, and duplicate parameters across optimizers all stay correct, and no `memcpy` is used anywhere. Parameter identity, storage replacement, gradient retention, the one version increment per commit, the F1 state transaction, checkpoint atomicity, and exact resume are unchanged; H1's full-write contract is proved on both traversals by poison injected purely by test infrastructure around the allocator, with a negative control. Measured by alternating pre/post subprocess rounds (control band 0.96x-1.05x) and by a separate pre-H5-library A/B: the traversal alone **2.5x-5.5x** on contiguous sources and **0.94x-1.02x** on transposed ones (the unchanged odometer, the design's own control); `copy_value_` **2.14x** at (512, 512), optimizer `state_dict()` 2.40x, `load_state_dict()` 1.69x, `NativeSGD.step()` 1.15-1.31x. Reported just as honestly: **`NativeAdam.step()`, every training step, the BatchNorm running update, and copies below ~16 K elements are all neutral**, the last because two `int64` layout arrays cost ~1.1 us each at the ctypes boundary — measured, attributed, and left to a later dispatch milestone rather than paid for by weakening H3's validation. Allocations fell everywhere and **no measured peak rose**: `copy_value_` 2 to 1, module state 4 to 2, optimizer state 16 to 8, Adam 17 to **16** per parameter. Two harness cases added (26 to 28). The ladder was **reordered** here: reduction execution, drafted as H5, moved to H6. No public API, capability, dtype, device, registry value, checkpoint field, or checkpoint version moved; export count stays **52** | **Complete** |
| **H6** | **Native reduction execution efficiency.** Reductions were the last core family always paying the generic strided indexing cost. The pre-H6 kernel was re-read and re-measured rather than trusted from H0's or H5's summaries, and **decomposed**: at `(256, 256)` `axis=0` a `core.sum` costs 99.7 us of which the raw native call is **94.8 us — 95 %** (the C++ traversal alone ~92 %), while the entire Python wrapper — axis normalization, output-shape construction, write-stride construction, the H3-cached layout arrays, and the output allocation — is ~5 us. So unlike B3 this was unambiguously a compiled-loop problem. H6 reused the dispatch shape H2 and H5 each proved: one hidden metadata predicate, inside the existing export, **no new symbol**, the pre-milestone traversal retained. New `cpp/include/tf_reduction_internal.h` declares three hidden-visibility `namespace tf` functions that `cpp/src/reduction.cpp` implements — `tf::sum_generic_strided`, the **pre-H6 odometer retained as the shipped generic reference path** and the only path that can address a transposed, narrowed, non-unit-strided, or broadcast source at all; `tf::reduce_prefers_contiguous_blocks`, total, pure, allocation-free, a function of layout metadata alone (never a pointer value, alignment, clock, environment variable, or CPU probe), with a false answer a fallback and never an error; and `tf::sum_contiguous_blocks`, a flat walk over an `outer x mid x inner` factorization. The predicate accepts a reduction when the source strides are exactly the row-major strides implied by the shape (the same definition `NativeTensorView` uses, so the layers agree by construction), the reduced axes — those with a zero *write* stride — form **one contiguous run**, and the kept axes carry the output's row-major strides. Stride collapsing is **implicit and bounded, not a general layout compiler**; nothing is cached or interned; `keepdims` needs no special case because the kernel cannot observe it. **Per-output accumulation order is preserved exactly** and the source traversal order is not even reordered, with no reassociation, FMA, Kahan, pairwise, tree, parallel, or horizontal-vector reduction anywhere; the `inner == 1` branch's local accumulator is seeded **from the destination**, keeping the export's accumulate-into semantics identical on both paths. Signed zeros are proved as **raw bit patterns** across every position and both `keepdims` values, and the rank-0 branch is recorded precisely (a genuine addition against a zeroed destination, so a rank-0 `-0.0` sums to `+0.0`, exactly as before). **The NaN rule is H6's own, measured rather than inherited from H2**: identical NaN positions, every NaN quiet, signaling NaNs quieted identically, and **bit identity whenever at most one NaN enters an accumulation** — every case that occurs in practice — with payloads outside the contract only when two or more NaNs meet in one cell, asserted in neither direction. Four accumulation spellings (including one accumulating *through memory* exactly as the odometer does) all selected the same NaN and all differed from the odometer, so parity is unavailable at any spelling and the memory form was 1.2x-1.8x slower; the block path keeps the **first** NaN, the odometer the **last**, and the block path's choice is NumPy's. **H1's rejection of this destination stands** — both traversals read it, so it stays zero-initialized; Outcome B was rejected on measurement (2 KB of fill against 512 KB of reads) and on semantics, so H6 adds **no poison test** because it introduces no uninitialized destination. Measured against a pre-H6 library on identical `ctypes` calls with outputs proved bit-identical before timing, 15 alternating rounds, control band **0.90x-1.03x**: full reductions 1.19x to **3.96x**, 2-D axis reductions 3.24x to **6.37x**, and — unpredicted — 3-D/4-D reductions **8.60x-10.94x**, because the odometer's carry loop scales with rank. Layer level: `TensorCore.sum(axis=0)` 4.49x, `mean(axis=0)` 4.11x, NCHW `sum(axis=1)` **8.56x**, `NativeTensor.sum` 3.88x, `sum()` fwd+bwd 1.27x, the **convolution bias gradient 1.46x**, `_unbroadcast` 1.15x, softmax backward 1.14x, LayerNorm forward 1.16x; the NumPy gap on contiguous reductions closed from ~8-13x to **1.67x-3.75x** while the transposed control stayed 10.33x. Reported just as honestly: **every training step is neutral** (0.99x-1.03x), so H6 does not make training faster; **normalization is mostly neutral**, narrowing H7 rather than motivating it; **tiny reductions are neutral** below ~1,000 elements where the fixed ~7 us Python-plus-ctypes cost dominates; and a **real ~10 % regression on 2-D transposed `axis=0` fallbacks** (0.89x-0.93x over four 25-round runs) is published, with the 3-D fallback 1.04x-1.05x *faster* and the cause isolated to whole-translation-unit code layout rather than the extracted call. Memory moved not at all and it is asserted: one allocation per `sum` on both paths at every axis. Harness 28 to **31** cases; native CTests 14 to **15**. No exported symbol, no new translation unit, **no public control of any kind**, no SIMD/threading/OpenMP/BLAS/pool/workspace/fast-math, no multi-axis reduction, and `tf_core_narrow_backward` (the scatter dual) deliberately untouched. No public API, capability, dtype, device, registry value, checkpoint field, or checkpoint version moved; export count stays **52** | **Complete** |
| H7–H11 | **Proposals, not commitments.** Ordered by measured leverage and each explicitly conditional: a milestone whose premise the preceding measurement does not confirm is narrowed, reordered, or dropped, and the reason is recorded in the design. H5 is the worked example of a *reorder* — the ladder really was reordered, moving reduction execution from the H5 slot to H6 on H4's evidence. **H6 is now the worked example of a condition being answered rather than met**: H7 was conditional on a re-measured normalization step still showing a material composed-module cost after H1, H3, and H6, and H6 measured that step at **1.03x** with most normalization operations inside the control band, so H7 as framed should not be entered — what remains in those modules is the count of broadcast elementwise operations, which is H8's subject | **Not started** |
| H9 | Re-measurement, hardening, and the full sanitizer matrix | **Not started** |
| H10 | Phase closure | **Not started** |

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
