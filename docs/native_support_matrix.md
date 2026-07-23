# Native support matrix

The canonical, authoritative statement of what the **experimental
native C++ CPU line** supports today — **Phase E (native classification
and stable math) is complete**, closing the
Phase A (native CPU runtime) → Phase B (native autograd) → Phase C
(native training stack) → Phase D (native CNN stack) → Phase E arc in
code. **Phase F (native normalization and stateful buffers) is in
progress**: `NativeLayerNorm` (milestone F2) is supported, while native
BatchNorm is not. The stable Python
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
**Phase F — Native Normalization and Stateful Buffers — is the current
phase and is *in progress*.** Its architecture contract is locked in
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
`"layernorm"` removed from `UNSUPPORTED`); milestones **F3–F9 are planned
and have not started**. No `NativeBatchNorm1d` or
`NativeBatchNorm2d` exists, no normalization *operation* is
differentiable, and no normalization kernel or C ABI export exists —
`batchnorm` remains in the backend registry's
`UNSUPPORTED` tuple and in the "Unsupported or future" section below.
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
| Buffers | Supported | v3.15: `register_buffer(name, tensor, persistent=True)`, `buffers()` / `named_buffers()`; NativeTensor-backed non-`Parameter` persistent state (infrastructure for future BatchNorm/RNG state — no algorithm yet); identity-deduplicated, cycle-safe traversal; persistent buffers join `state_dict`/`load_state_dict` and checkpoints, non-persistent buffers are never serialized. Reported as `persistent_buffers` in `STATE_SUPPORT` since Phase F milestone **F1** — reconciliation of an under-reported capability, not a new feature |
| `state_dict` / `load_state_dict` | Supported | In-memory, parameters and persistent buffers, atomic validate-then-commit with rollback (buffer identity preserved on restore). Since **F1** the replacement half runs through the private `_native_state.replace_native_state` transaction, shared with the future normalization running-statistics update; `load_state_dict`'s public signature, validation order, error messages, key reporting, version semantics, and atomicity are unchanged |
| Atomic native state transaction | Supported (private) | **F1** (Phase F): `tensorforge.experimental._native_state.replace_native_state` — identity-preserving, exception-safe replacement of one or more registered `NativeParameter`/persistent-buffer cores as one all-or-nothing transaction. Validate → stage → commit, with the commit boundary at "every core swap **and** every parameter-version increment succeeded"; complete rollback of cores and versions before it; exactly-once closing of replaced and abandoned cores after it; destinations deduplicated by object identity (an aliased parameter is swapped, versioned, and released once; conflicting values for one destination are rejected before mutation). **Deliberately private** — absent from `tensorforge.experimental.__all__`, and *not* a public in-place mutation API for `NativeTensor` (`NativeParameter.copy_value_` remains the only public controlled-mutation primitive) |
| `NativeLinear` | Supported | Seeded deterministic init; strictly 2-D input |
| `NativeReLU` | Supported | Parameter-free activation module |
| `NativeFlatten` | Supported | D1 (Phase D): parameter-free, buffer-free batch-preserving flatten `(N, …) → (N, features)`, Python-composed from the existing `reshape`/`contiguous_copy` ops and their autograd — no new kernel, no custom backward; returns an independent owning result so it composes safely in `NativeSequential` |
| `NativeConv2d` | Supported | D7 (Phase D): the trainable convolution **module** over the differentiable `conv2d` operation — OIHW weight / optional `(O,)` bias `NativeParameter`s, deterministic uniform conv fan-in initialization, 4-D NCHW input validation; backward supplied entirely by the operation's autograd (no new kernel, ABI symbol, or custom module backward) |
| `NativeMaxPool2d` | Supported | D10 (Phase D): the pooling **module** over the differentiable `maxpool2d` operation — parameter-free, buffer-free, normalized `(h, w)` `kernel_size`/`stride`/`padding` (`stride=None` ⇒ non-overlapping); no winner state between calls and no state-dictionary or checkpoint keys |
| `NativeSequential` | Supported | Ordered container with contiguous integer-string slots |
| `NativeLayerNorm` | Supported | F2 (Phase F): the first native normalization **module** — normalizes the trailing `len(normalized_shape)` dimensions, stateless (no buffers, identical in train and eval). Population variance with `sqrt(var + eps)` ordering; multi-axis reduction as sequential single-axis `mean(keepdims=True)` calls; composed entirely from existing native operations (`mean`, `subtract`, `multiply`, `add`, `sqrt`, `reciprocal`) so backward comes from the existing autograd. `weight`/`bias` `NativeParameter`s only when `elementwise_affine=True` (registered in that order); fresh owning contiguous output. No new kernel, ABI symbol, ctypes declaration, `NativeTensorCore` method, custom backward, functional helper, or `NativeTensor.layer_norm` operation |
| `NativeMSELoss` | Supported | `"mean"` / `"sum"` reductions; exact shapes, no broadcasting |
| `NativeCrossEntropyLoss` | Supported | E7 (Phase E): the classification **loss module** over the differentiable `cross_entropy` operation — parameter-free, buffer-free, `"mean"`/`"sum"` only (validated in the constructor by the operation's own validator, so an invalid reduction can never reach it), targets validated and copied by the operation itself. Its forward is exactly `logits.cross_entropy(targets, reduction=self.reduction)`: no new kernel, ABI symbol, arithmetic, or custom backward, and no state-dictionary or checkpoint keys |
| `native_accuracy` | Supported (reporting only) | E7 (Phase E): **not** a native operation — a Python helper that validates rank-2 logits and strict `int64` targets, materializes once through the explicit public `to_numpy()` boundary, takes `numpy.argmax(axis=1)` (first-maximal index on ties), and returns a Python `float` in `[0.0, 1.0]`. Builds no graph, touches no gradient/parameter/version, allocates no native storage, retains nothing. Reported in the new `NATIVE_METRICS` inventory, never in the operation inventories |
| `NativeSGD` | Supported | Minimal `value ← value − lr·grad`; identity-deduplicated; two-phase mutation-atomic `step()`; `zero_grad()`; in-memory `state_dict`/`load_state_dict` (v3.13: lr + positional parameter metadata) |
| `NativeAdam` | Supported | Adaptive optimizer (v3.12): validated `lr`/`betas`/`eps`; persistent optimizer-owned native m/v moments and per-parameter step counts; bias correction via `sqrt`/`reciprocal` (no division); graph-free staged updates committed through `copy_value_`; skipped frozen/`grad=None` parameters never age state; explicit state lifetime — `close()` releases the moments; in-memory `state_dict`/`load_state_dict` (v3.13) |
| Optimizer state (in-memory) | Supported | v3.13: one versioned schema (format 1, exact optimizer type tag), ordered positional shape/dtype/device parameter metadata — no object ids, names, values, or gradients — caller-owned independent NativeTensor m/v snapshots and per-parameter step counts (NativeAdam), exact validation with no casting or device movement, staged atomic loading that never touches parameter values, versions, gradients, or retained graphs; deterministic in-memory training continuation with the module state contract |
| Checkpoint files / resume | Supported | v3.14: `save_native_checkpoint` / `load_native_checkpoint` — one pickle-free NPZ archive (format `"tensorforge.native_checkpoint"`, version 1) holding the model state, optionally one native optimizer's v3.13 state, and JSON-compatible metadata; UTF-8/JSON uint8 manifest, indexed float64 array entries, strict full-archive validation before any live mutation, strict optimizer presence/type matching, atomic temporary-file replacement, `allow_pickle=False` loading, deterministic bit-identical file resume (`examples/native_checkpoint_resume.py`); no scheduler or random-state capture, no `map_location` |
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
- scheduler state, random-state capture/restoration, or dataloader
  state in native checkpoints; `map_location`, partial or name-remapped
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
- native **BatchNorm** — **Phase F is in progress**: `NativeLayerNorm`
  (milestone F2) has shipped and is listed in the training-stack table
  above, but BatchNorm has not. The contract is locked in
  [native_normalization_design.md](native_normalization_design.md), and
  `batchnorm` stays in the registry's `UNSUPPORTED` tuple until milestone
  F4 removes it (`layernorm` left at F2)
- dropout or a native RNG — future work **beyond** Phase F
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

## Phase F — native normalization and stateful buffers, **in progress**

**`NativeLayerNorm` (milestone F2) is the only normalization capability
implemented so far.** The rest of this section records a locked
architecture contract, not yet a capability. The authoritative statement
of what exists is the backend registry: `"NativeLayerNorm"` is in
`NATIVE_MODULES` and `"layernorm"` has left `UNSUPPORTED`, while
`batchnorm` remains unsupported and there is **no** normalization entry in
`TENSOR_CORE_OPS`, `AUTOGRAD_OPS`, or `RAW_KERNELS` (LayerNorm is a module
composed from existing operations, not an operation).

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
| F3 | `NativeBatchNorm1d` | Planned — not started |
| F4 | `NativeBatchNorm2d` | Planned — not started |
| F5 | Normalization state, checkpoint, and graph-safety hardening | Planned — not started |
| F6 | Deterministic normalized training and exact resume | Planned — not started |
| F7 | Native normalization benchmark characterization (no speed assertion) | Planned — not started |
| F8 | Cross-cutting Phase-F integration and semantic guardrails | Planned — not started |
| F9 | Phase-F closure | Planned — not started |

Explicitly **outside** Phase F and still unplanned: dropout, a native
RNG and RNG checkpoint state, `tanh`/`sigmoid`/GELU, more losses,
schedulers, data loaders, native integer tensors, indexing/`gather`/
`max`/`argmax`, float32/float16/bfloat16, casting or dtype promotion,
CUDA, AMP, Tensor Core dispatch, pybind11, the Python C API, implicit
stable/native dispatch or conversion, fused normalization kernels,
normalization-specific C ABI exports, custom normalization backward
kernels, synchronized/distributed BatchNorm, CPU optimization,
performance thresholds, checkpoint format version changes, and real
datasets or generalization claims.

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
