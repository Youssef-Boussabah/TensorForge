# Native normalization architecture design (Phase F contract)

This is the **design-and-contract** document for the experimental native
C++ CPU line's normalization stack — **Phase F — Native Normalization and
Stateful Buffers**. It is a milestone-zero (F0) deliverable: it locks the
architecture, public API surface, buffer-safety rules, mutation
atomicity, ownership model, state/checkpoint integration, capability
placements, testing strategy, and milestone sequence **before any
numerical normalization code is written**.

**F0 adds no numerical behavior.** No kernel, C ABI symbol, ctypes
declaration, `NativeTensorCore` method, `NativeTensor` operation,
normalization module, buffer-mutation helper, benchmark, or example ships
with this document. F0 is a design-and-reconciliation milestone: it
writes this contract and corrects documentation drift left after Phase E
closed. The backend capability registry (`tensorforge.backends.cpp` —
`TENSOR_CORE_OPS`, `AUTOGRAD_OPS`, `RAW_KERNELS`, `NATIVE_MODULES`,
`NATIVE_LOSSES`, `NATIVE_METRICS`, `STATE_SUPPORT`, and `UNSUPPORTED`),
mirrored in [native_support_matrix.md](native_support_matrix.md), stays
the **single source of truth** for what is actually live at any moment.
At F0 that registry listed `"batchnorm"` and `"layernorm"` in
`UNSUPPORTED`, and it must keep listing each until the milestone that
implements it removes it — as milestone F2 has now done for
`"layernorm"`.

**Phase-F status: complete.** **F0 is complete** (this contract and
the repository reconciliation that came with it), **F1 is complete** (the
private atomic native-buffer state transaction, the `load_state_dict`
refactor onto it, and the `STATE_SUPPORT` persistent-buffer
reconciliation — **state management and capability reporting only, no
normalization mathematics**), **F2 is complete** (`NativeLayerNorm` —
the first native normalization module: stateless, differentiable through
the mean and the population variance, and **composed entirely from
existing native operations**, adding no kernel, C ABI symbol,
`NativeTensorCore` method, custom backward, or `NativeTensor`
normalization operation; `"NativeLayerNorm"` has joined `NATIVE_MODULES`
and `"layernorm"` has left `UNSUPPORTED`), and **F3 is complete**
(`NativeBatchNorm1d` — the first **stateful** native numerical module:
`(N, C)` batch normalization with differentiable training statistics,
persistent native `running_mean`/`running_var` buffers advanced by a
graph-free atomic two-buffer transaction, and evaluation from graph-safe
immutable snapshots; again composed entirely from existing operations,
adding no kernel, C ABI symbol, `NativeTensorCore` method, custom
backward, or `NativeTensor.batch_norm` operation; `"NativeBatchNorm1d"`
has joined `NATIVE_MODULES` while `"batchnorm"` stayed in
`UNSUPPORTED`), and **F4 is complete** (`NativeBatchNorm2d` — NCHW
`(N, C, H, W)` batch normalization reducing over N, H, and W, built on
the **same** shared private implementation as `NativeBatchNorm1d`;
`"NativeBatchNorm2d"` has joined `NATIVE_MODULES` and, with both shapes
now live, `"batchnorm"` has finally left `UNSUPPORTED`), and **F5 is
complete** (the exhaustive state, checkpoint, ownership, and graph-safety
hardening — a focused `tests/test_native_normalization_state.py` plus
narrow additions to the generic buffer and checkpoint suites, proving
§7–§10 by executable test rather than by prose: canonical dotted buffer
keys, independent state snapshots, strict/non-strict buffer-key handling,
exact never-casting metadata validation, identity-preserving mixed loads,
mixed parameter/buffer transaction rollback, the version-1 checkpoint
schema gaining no normalization field, exact eval-output reproduction
across a round trip, the buffer-only-versus-full stale-graph distinction,
the corrupt/staging/save failure boundaries, eval-graph structural safety
under `retain_graph` and a failed retryable backward, and the
live-storage baseline over the whole matrix. **Tests and documentation
only — no numerical behavior and no new public capability**: the exports
and every capability registry are exactly what F4 left, the checkpoint
format stays version 1, and no normalization operation, kernel, C ABI
symbol, or custom backward exists).
**The numerical normalization *module* surface is therefore complete**:
`NativeLayerNorm`, `NativeBatchNorm1d`, and `NativeBatchNorm2d` all
exist and are exported, and the state/checkpoint/ownership/graph-safety
contracts are proved by test. And **F6 is complete** (the deterministic
normalized training and exact checkpoint-resume proof —
`examples/native_normalization_training.py`: a
`Linear → BatchNorm1d → ReLU → LayerNorm → Linear` regressor trained for
24 deterministic `NativeAdam` steps with `NativeMSELoss` (a 98.9% loss
reduction), whose two uninterrupted runs are bit-identical and whose
interrupted checkpoint resume into a **fresh** model/optimizer pair
reproduces the remaining loss suffix, every parameter, the NativeAdam
state, both BatchNorm `running_mean` and `running_var`, the final
training-step prediction, and the final **evaluation-mode** output
exactly, with format version 1 unchanged and training flags runtime-only;
**one example and its integration test, adding no capability, operation,
kernel, schema field, benchmark, or export**). And **F7 is complete** (the
honest benchmark characterization —
`benchmarks/benchmark_native_normalization.py`: nine cases covering the
LayerNorm forward and backward, the BatchNorm1d training forward,
evaluation forward, and backward, the BatchNorm2d training forward,
evaluation forward, and backward, and one complete F6-style normalized
training step. Every case is **correctness-gated before any timing**, is
labelled with the reference it actually used — `stable_tensorforge` for
the six cases with a real stable counterpart, `native_only` for the three
BatchNorm2d cases because the stable line has no public `BatchNorm2d` to
time against — and reports the median with the minimum, maximum, and
spread after warm-up, with `--smoke`/`--json` modes and **no result file,
no speed assertion, and no timing threshold anywhere**. The BatchNorm2d
cases keep a rigorous correctness oracle even without a timed reference;
**measurement only — no capability, operation, kernel, C ABI symbol,
schema field, or export**). And **F8 is complete** (the cross-cutting
integration and semantic guardrails — `tests/test_native_phase_f.py`: one
integrated `Conv2d → BatchNorm2d → ReLU → MaxPool2d → Flatten → Linear →
BatchNorm1d → ReLU → LayerNorm → Linear` classifier over raw logits and
the fused loss, trained by `NativeAdam` and resumed **exactly** from one
version-1 checkpoint including all four running-statistic buffers and the
evaluation-mode output; three saved-resource families (BatchNorm
snapshots, MaxPool2d winners, cross-entropy probabilities) coexisting and
releasing exactly once; buffer mutation versus parameter mutation
attributed to the right cause; the Phase-E versioning archetypes meeting
a normalized graph; shared and frozen parameters; a non-contiguous NCHW
input through the whole stack; each failure boundary tested honestly
without claiming a whole step is globally transactional; and semantic
capability/export/artifact guardrails derived from real registries and
files. **Tests and documentation only — no capability, and no production
behavior changed**). And **F9 is complete** (the phase closure —
validation, documentation reconciliation, and this completion statement,
with **no numerical capability of any kind**: fresh Windows Release and
Debug builds each passing the full existing 10-test CTest suite with zero
project compiler, linker, or CMake warnings; a fresh Clang 18.1.3
ASan+UBSan build in WSL2 Ubuntu 24.04 whose instrumentation is *proved*
rather than assumed; 10/10 sanitized native CTests with leak detection
enabled; 1,968 sanitized normalization-focused Python tests; the F6
example and the F7 benchmark smoke path under the sanitized library; and
a practical LeakSanitizer lifecycle returning native live storage
**exactly** to baseline with no TensorForge-attributable leak frame. The
full record is in the F9 milestone entry in §15). **Phase F is therefore
complete: F0–F9 all shipped.** And no normalization *operation*,
kernel, or C ABI symbol exists at all — all three modules are
compositions of existing operations, not numerical primitives.

**What Phase F delivered.** A fully native, differentiable, state-safe
normalization stack:

- `NativeLayerNorm` — batch-independent normalization over the trailing
  dimensions, with no persistent state.
- `NativeBatchNorm1d` — `(N, C)` batch normalization with persistent
  native running statistics and distinct train/eval behavior.
- `NativeBatchNorm2d` — NCHW `(N, C, H, W)` batch normalization over the
  same shared implementation.

and, underneath them, the parts that make stateful native modules safe:
native-only normalization mathematics, exact training and evaluation
behavior, persistent native running statistics, graph-safe handling of
mutable buffers, atomic running-state updates, state-dictionary and
checkpoint integration, exact checkpoint resume, explicit ownership and
cleanup, and honest capability reporting.

The stable Python framework (`tensorforge.nn.LayerNorm`,
`tensorforge.nn.BatchNorm1d`) is the **numerical and public-behavior
reference** wherever an equivalent stable capability exists. Where the
native architecture must differ (ownership, lifetime, buffer identity,
the absence of a stable `BatchNorm2d`), the difference is stated and
justified. No implementation code is copied from any other framework.

Read alongside:
[native_tensor_wrapper_design.md](native_tensor_wrapper_design.md) (the
`NativeTensor` wrapper and ownership model),
[native_autograd_design.md](native_autograd_design.md) (the
Python-managed reverse-mode graph and the parameter-version stale-graph
contract),
[native_cnn_design.md](native_cnn_design.md) (Phase D — NCHW layout and
the graph-owned saved-resource model),
[native_classification_design.md](native_classification_design.md)
(Phase E — the fused-kernel and saved-output precedent),
[native_abi_error_contract.md](native_abi_error_contract.md) (the
exception-safe C ABI status contract), and
[backend_experiments.md](backend_experiments.md) (the whole native line).

---

## 0. Invariants Phase F must preserve

Phase F changes nothing about these existing guarantees:

- Stable `Tensor` and `NativeTensor` remain **separate systems**; there is
  no implicit conversion, no dispatch, and no fallback between them.
- Native autograd stays **Python-managed** at the `NativeTensor` layer;
  the `NativeTensorCore` runtime and the C++ kernels stay
  **autograd-unaware**.
- The native runtime targets **CPU float64** only.
- Native storage **ownership and lifetime are explicit** (`owns_core`,
  `close()`); owning cores free storage, borrowing views do not.
- `NativeParameter` preserves **object identity**; value replacement
  increments its monotonic **value version**; graphs record expected
  versions and detect **stale graphs** through version checks.
- `state_dict()` returns **independent snapshots**; state and checkpoint
  loading are **atomic** (validate → stage → commit → rollback on
  failure); native checkpoints stay **pickle-free** and load with
  `allow_pickle=False`.
- Native checkpoints keep **format version 1**
  (`"tensorforge.native_checkpoint"`).
- Failed operations must **not partially mutate** caller-visible state.
- Every fallible native export uses the existing **thread-local status /
  `errcheck` contract**; **no C++ exception crosses the C ABI**.
- Numerical operations stay free of NumPy compute — the NumPy tripwire
  tests keep holding.
- Existing Phase A–E numerical behavior is **unchanged**.

---

## 1. Phase-F scope

### In scope

| # | Deliverable |
|---|---|
| 1 | A private, reusable atomic state transaction for native buffer mutation |
| 2 | `NativeLayerNorm` — a differentiable, stateless normalization module |
| 3 | `NativeBatchNorm1d` — `(N, C)` batch normalization with persistent running statistics |
| 4 | `NativeBatchNorm2d` — NCHW `(N, C, H, W)` batch normalization over the shared implementation |
| 5 | Graph-safe eval-mode snapshots of the mutable running buffers |
| 6 | Atomic two-buffer running-statistics updates with rollback |
| 7 | State-dictionary and checkpoint integration for the running buffers |
| 8 | A deterministic normalized training run with **exact** checkpoint resume |
| 9 | Honest normalization benchmarks (characterization only) |
| 10 | Cross-cutting integration, ownership, sanitizer, and documentation closure |

### Explicitly excluded from Phase F

`dropout`; a native RNG; RNG checkpoint state; `tanh`; `sigmoid`; GELU;
any further loss; schedulers; data loaders; native integer tensors;
indexing, `gather`, `max`, or `argmax`; `float32`, `float16`, or
`bfloat16`; casting or dtype promotion; CUDA; AMP; Tensor Core dispatch;
pybind11; the Python C API; implicit stable/native dispatch; implicit
conversion; fused LayerNorm or BatchNorm kernels; normalization-specific
C ABI exports; custom normalization backward kernels; synchronized or
distributed BatchNorm; CPU optimization; performance thresholds;
checkpoint format version changes; and real datasets or generalization
claims.

These stay in the support matrix's unsupported/future section. As in
Phases D and E, the first implementation of every layer favors
**correctness, readability, sanitizer safety, and explicit composition**
over speed.

---

## 2. Stable/native separation

Locked, and unchanged from every earlier phase:

- The **stable Python/NumPy framework remains the numerical and
  behavioral reference** wherever an equivalent stable capability exists.
  `tensorforge.nn.LayerNorm` and `tensorforge.nn.BatchNorm1d` are the
  references for `NativeLayerNorm` and `NativeBatchNorm1d`. There is no
  stable `BatchNorm2d`, so `NativeBatchNorm2d`'s reference is the stable
  `BatchNorm1d` mathematics applied over the NCHW reduction set, verified
  additionally against hand-computed cases and central finite
  differences.
- Native functionality remains **explicit**, reached only through
  `tensorforge.experimental` and `tensorforge.backends`. `import
  tensorforge` never imports it.
- **No implicit dispatch and no implicit conversion.** A stable `Tensor`
  is never accepted where a `NativeTensor` is expected, and the reverse
  holds too; stable and native modules keep rejecting each other's
  objects.
- **Native numerical paths must not silently use NumPy.** Every value in
  a normalization forward or backward is produced by native kernels
  through `NativeTensorCore`. The NumPy tripwire tests extend to cover a
  complete normalized training step.
- **Explicit conversion boundaries remain allowed** and stay explicit:
  `NativeTensor.from_array()` in, `to_numpy()` out, and the
  reporting-only conversion `native_accuracy` already performs. Phase F
  adds no new conversion boundary and no reporting-only helper of its
  own.

Host Python scalars used as *configuration* — `eps`, `momentum`,
`num_features`, `normalized_shape` — are not tensor data and do not
constitute a NumPy path. They are validated in Python, converted to
native constants through the existing explicit constructors, and the
resulting constants are ordinary owning native tensors.

---

## 3. Autograd and native-layer boundaries

Locked:

- **Python continues to own the `NativeTensor` autograd graph.**
  `NativeTensorCore` and the C++ kernels remain **graph-unaware**.
- **No pybind11.** **No Python headers in C++.** C++17, a plain **C
  ABI**, and **ctypes** remain mandatory.
- **Phase F adds no normalization-specific `NativeTensorCore` method, no
  internal C++ normalization kernel, and no C ABI export.** This is the
  phase's single most important structural decision: normalization is
  arithmetic the existing native operations already express, so composing
  it buys exact gradients for free from the existing autograd and adds
  zero new trust-boundary surface to validate, fuzz, or sanitize.
- Normalization is **initially composed from existing native
  operations** only:

  | Operation | Role in the composition |
  |---|---|
  | `mean` | batch/feature means, and the mean of the squared deviations |
  | `subtract` | centering (`x − mean`) |
  | `multiply` | squaring the deviations, applying `gamma`/`weight`, and applying the inverse standard deviation |
  | `add` | the `eps` shift and the `beta`/`bias` offset |
  | `sqrt` | the standard deviation from the variance |
  | `reciprocal` | the inverse standard deviation (there is deliberately still **no** general `divide`) |
  | `reshape` | shaping affine parameters and statistics for broadcasting |
  | broadcasting | applying per-feature statistics and affine terms across the reduced axes |
  | `contiguous_copy` | producing fresh owning contiguous results and graph-free snapshots where required |

  Every one of these already has a tested backward. The **variance is
  therefore differentiated automatically**: because the batch mean and
  the mean of squared deviations are computed with the differentiable
  `mean`, the training-mode backward flows through the statistics without
  any hand-written normalization backward.

- A **fused normalization kernel is explicitly deferred past Phase F**.
  If a later phase ever adds one, it must reproduce the composed result
  to tolerance and must not change any public contract in this document.

Because normalization introduces **no new operation name**, it introduces
no new entry in the backward-read/versioning matrix either. Each composed
edge keeps exactly the versioning behavior its own operation already has
(`multiply` and `matmul` version-guard a direct `NativeParameter` operand;
`add`, `subtract`, reductions, and views do not). Section 7 covers the
one genuinely new hazard this creates.

---

## 4. Locked public API surface

```
NativeLayerNorm(
    normalized_shape,
    eps=1e-5,
    elementwise_affine=True,
)

NativeBatchNorm1d(
    num_features,
    eps=1e-5,
    momentum=0.1,
)

NativeBatchNorm2d(
    num_features,
    eps=1e-5,
    momentum=0.1,
)
```

Nothing else becomes public. In particular:

- **`NativeLayerNorm` uses `weight` and `bias`** for its affine
  parameters — matching the stable `tensorforge.nn.LayerNorm`.
- **`NativeBatchNorm1d` and `NativeBatchNorm2d` use `gamma` and `beta`** —
  matching TensorForge's stable `BatchNorm1d` naming. The two
  normalizations deliberately differ in parameter naming because their
  stable counterparts do; consistency with the stable reference beats
  internal uniformity.
- **BatchNorm running state is exposed as `running_mean` and
  `running_var`** — again the stable names, and the names the state
  dictionary and checkpoints will use.
- **No public functional `layer_norm` or `batch_norm` helper.** The
  modules are the whole public surface.
- **No `NativeTensor.layer_norm()` and no `NativeTensor.batch_norm()`
  operation.** Normalization is a *module-level* composition, not an
  autograd primitive, so it never appears in `AUTOGRAD_OPS`.
- **No `dtype` or `device` constructor arguments** while the runtime
  supports only `float64`/`cpu`. Adding inert arguments would advertise a
  flexibility the kernels do not have. The v1.21 rule stands: reject
  rather than accept-and-ignore.

`training` remains runtime state set through the existing
`NativeModule.train()` / `eval()`, which already validate a real `bool`
and propagate recursively.

---

## 5. `NativeLayerNorm` contract

`NativeLayerNorm` is the simpler of the two shapes and ships first
(milestone F2) because it is **stateless**: it exercises the composed
normalization mathematics without touching the mutable-buffer machinery
at all.

**Construction.**

- `normalized_shape` accepts a **positive `int`** or a **non-empty
  sequence of positive `int`s** (`bool` rejected at every position, as
  everywhere else in the native line). It is normalized to a tuple and
  stored as `self.normalized_shape`.
- `eps` must be a positive real (`bool` rejected). Stored as a Python
  `float`.
- `elementwise_affine` must be a real `bool`.
- `elementwise_affine=True` creates `weight` initialized to **ones** and
  `bias` initialized to **zeros**, both of shape `normalized_shape`, both
  `NativeParameter`s registered through ordinary attribute assignment.
- `elementwise_affine=False` creates **no parameters at all** — not
  frozen parameters, not `None` placeholders that later code must
  special-case beyond the affine branch itself.
- All validation happens **before** any native allocation, so a rejected
  construction leaks nothing.

**Forward.**

- It **normalizes the trailing `len(normalized_shape)` dimensions**.
- The input's trailing dimensions must **match `normalized_shape`
  exactly**; the input rank must be at least `len(normalized_shape)`.
  Anything else is rejected before allocation, naming the expected and
  received shapes.
- The mean and variance are taken over exactly those trailing
  dimensions, per sample. No other sample participates.
- **Variance is the population variance** (divide by the element count,
  no Bessel correction) — matching the stable `LayerNorm`.
- **Epsilon is added before the square root**: `sqrt(var + eps)`, never
  `sqrt(var) + eps`.
- The normalized value is `(x − mean) * reciprocal(sqrt(var + eps))`,
  then `* weight + bias` when `elementwise_affine=True`.
- **LayerNorm has no persistent buffers** and registers none.
- **LayerNorm behaves identically in train and eval modes.** The same
  input always gives the same output; `training` is never read in its
  forward.
- The output is a **fresh, owning, contiguous** native tensor of the
  input's shape, sharing storage with nothing.
- **Backward is provided entirely through composition** of the existing
  autograd operations (§3). There is no hand-written LayerNorm backward,
  no saved statistics, and no graph-owned resource. Gradients flow to the
  input, and to `weight`/`bias` when affine.

**Deliberately not in the contract:** `reduction`-style options, a
functional form, per-sample `eps`, RMSNorm, GroupNorm, or an
`elementwise_affine` variant with only one of `weight`/`bias`.

---

## 6. `NativeBatchNorm1d` and `NativeBatchNorm2d` contracts

Both modules are built on **one shared private implementation** — the
reduction axis set and the broadcast shape are the only real differences
between them — while remaining **two separate public classes**, because
the input rank they accept is part of each one's contract and silent
rank-polymorphism would hide shape bugs.

### 6.1 `NativeBatchNorm1d`

- Accepts **only** shape `(N, C)` where `C == num_features`. Any other
  rank or a mismatched feature count is rejected before allocation.
- **Reduces over axis 0** (the batch).
- Affine parameters and running statistics broadcast as `(1, C)`.

### 6.2 `NativeBatchNorm2d`

- Accepts **only** NCHW shape `(N, C, H, W)` where `C == num_features` —
  the same activation layout Phase D locked for convolution and pooling.
- **Reduces over axes N, H, and W**, so each channel gets one mean and
  one variance over `N * H * W` values.
- Affine parameters and running statistics broadcast as `(1, C, 1, 1)`.

### 6.3 Shared contract

**Construction.**

- `num_features` must be a positive `int` (`bool` rejected).
- `eps` must be a positive real; `momentum` must be a real in `[0, 1]`.
  Both are stored as Python `float`s. Both are validated before any
  native allocation.
- `gamma` is initialized to **ones**, `beta` to **zeros**; both are
  `NativeParameter`s of shape `(num_features,)`.
- `running_mean` is initialized to **zeros**, `running_var` to **ones**;
  both are of shape `(num_features,)`.
- The running statistics are **owning, contiguous, gradient-free
  persistent `NativeTensor` buffers**, registered through the existing
  `NativeModule.register_buffer(name, tensor, persistent=True)`. They are
  therefore never parameters: `parameters()` never yields them, no
  optimizer ever sees them, and no gradient flows through them.

**Training-mode forward.**

- The batch statistics are computed **through differentiable native
  operations**, so the **training backward includes differentiation
  through the batch mean and variance**. This is not optional and not a
  simplification to be traded away: detaching the statistics would give a
  different, wrong gradient, and the stable reference does not detach
  them.
- The normalized value is `(x − batch_mean) * reciprocal(sqrt(batch_var +
  eps))`, then scaled by `gamma` and shifted by `beta`, with all
  per-feature terms broadcast to the input's layout.
- **Epsilon is added before the square root.**
- **Population variance** is used, consistently with the stable
  TensorForge reference — the *same* population variance is used both in
  the normalization and in the running-variance update. There is **no
  unbiased / Bessel-corrected running variance** in Phase F.
- The **running-statistics update happens outside autograd** (§8): the
  new values are computed as independent graph-free native state, and the
  update never contributes to any gradient.

**Evaluation-mode forward.**

- Evaluation uses the **stored running statistics** rather than the
  batch's own, so a single sample normalizes consistently.
- The running buffers are **never captured directly** in the output
  graph — §7 is the load-bearing rule that governs this, and it applies
  to eval mode specifically.
- `gamma` and `beta` still participate normally and still receive
  gradients when the graph requires them; only the *statistics* change
  source between modes.

**Output.** Fresh, owning, contiguous native tensors in both modes,
sharing storage with neither the input, the parameters, nor the running
buffers.

**Deliberately not in the contract for Phase F:**

- **No `track_running_stats` option.** Running statistics always exist.
- **No `affine=False` option.** `gamma` and `beta` always exist.
- **No `num_batches_tracked`** counter, and therefore no cumulative-average
  momentum mode.
- **No unbiased/Bessel-corrected running variance.**
- No synchronized or distributed BatchNorm, no `BatchNorm3d`, no
  `InstanceNorm`, no `GroupNorm`.

---

## 7. Mutable-buffer graph safety

**This section is load-bearing.** It is the one genuinely new hazard
Phase F introduces, and the reason F1 exists as its own milestone.

### 7.1 The hazard, stated precisely

Two facts about the existing engine combine badly:

1. **Native persistent buffers have no value version.** A buffer is a
   plain `NativeTensor`. The stale-graph guard
   (`_versioned_value_reads`) records an expected version only for
   operands that carry a `_version` slot, and `NativeParameter` is the
   only class that defines one. A buffer operand therefore records
   **nothing**, and `backward()` has no way to notice that it changed.
2. **Some binary operations reread a live operand during backward.**
   `multiply`'s backward computes each parent's gradient from the *other*
   parent's **current** core (`b._require_open()` at backward time), not
   from a value saved at forward time. The same is true of `matmul`. This
   is safe today only because the sole mutation path in the engine —
   `NativeParameter.copy_value_` — is version-guarded, and because plain
   `NativeTensor`s are otherwise immutable for the life of a graph (there
   is no in-place arithmetic).

A BatchNorm running buffer breaks the second assumption: it is a plain
`NativeTensor` that a module **does** mutate, repeatedly, as part of
ordinary training. So if an eval-mode forward captured the live
registered `running_var` object as a `multiply` operand, then a
subsequent training step, `load_state_dict()`, or checkpoint load would
either

- silently change the gradient of the already-built graph (the backward
  reads a value that was never in the forward), with **no** stale-graph
  error to warn anyone — the worst possible failure mode, a wrong number
  reported as a right one; or
- raise a confusing `RuntimeError: this NativeTensorCore has been closed`,
  because `load_state_dict()`'s commit closes each replaced core after
  installing its replacement.

### 7.2 The locked rule

> **A live, mutable, registered `running_mean` or `running_var` buffer
> must never be captured directly as a rereadable graph operand.**

Concretely, for milestones F3 onward:

- **Eval-mode BatchNorm must create independent, owning, graph-free
  snapshots** of `running_mean` and `running_var` **before** using them
  in the output graph. The snapshot is a fresh owning contiguous native
  tensor with `requires_grad=False`, produced by the existing native
  storage-to-storage copy path — no NumPy round trip.
- The graph then holds the **snapshot**, which nothing will ever mutate.
- Any later training forward, `load_state_dict()`, checkpoint load, or
  internal running-statistics update therefore **cannot change the
  gradient of an already-built eval graph**, and cannot invalidate its
  operands.
- Training-mode forwards never read the running buffers at all — they use
  the batch's own statistics — so the rule costs nothing on the hot path.

**Scope, stated precisely.** This rule is about the *running buffers*, and
that is the whole of what it promises. A state or checkpoint load that
replaces **only** `running_mean`/`running_var` leaves an already-built
eval graph **valid and numerically unchanged** — it moves no parameter
version and the graph never read those objects. A load that *also*
replaces `gamma` or `beta` is a different thing: those are
`NativeParameter`s, an eval graph does hold them directly as `multiply`
operands, and the pre-existing v3.7 stale-value guard therefore
**intentionally rejects** the old graph with its deterministic
stale-parameter error, exactly as it does for any other layer. That is
the parameter contract working, not a buffer failure, and normalization
must neither bypass nor weaken it. Because the ordinary
`load_native_checkpoint()` of a whole BatchNorm model replaces all four
state entries at once, **the buffer half of the rule has to be proved on
its own** — over the real checkpoint path, not only over
`load_state_dict()` — which is what F3's buffer-only checkpoint test
does (§15, F3).

### 7.3 Why buffer versions are *not* being added

An obvious alternative is to give buffers a monotonic value version, as
`NativeParameter` has, and let the existing stale-graph detector catch
the hazard. Phase F deliberately does not do that, for four reasons:

1. **It solves a problem the snapshot rule removes entirely.** Versions
   *detect* a stale graph and raise; snapshots make the graph **never
   become stale**. An eval graph built before a training step stays valid
   and correct, which is the behavior a user actually wants — a raised
   error would be honest but useless.
2. **Versioning is a contract on `NativeParameter` identity, and it is
   coupled to `copy_value_`.** Extending it to buffers means extending
   `copy_value_`-style mutation semantics to plain `NativeTensor`s, which
   Phase F explicitly refuses (§8): a general public in-place mutation API
   on ordinary `NativeTensor` is a much larger and more dangerous change
   than anything normalization needs.
3. **It would widen the guard's blast radius for no gain.** Every
   `multiply`/`matmul` node would start recording buffer entries, and
   every optimizer, state load, and checkpoint path would have to reason
   about a second version space.
4. **The snapshot is cheap and bounded.** It is one contiguous copy of
   `(num_features,)` values per eval forward — negligible beside the
   activation-sized work the same forward already does.

**Therefore: running buffers remain unversioned, precisely because graphs
never read the mutable registered buffer objects directly.** The two
halves of that sentence are one decision, not two, and neither may be
changed without the other. A guardrail test in F5 must fail if a
BatchNorm eval graph is ever found holding the registered buffer object
itself.

---

## 8. Running-statistics transaction and atomicity

The two running buffers describe **one** statistical state. Updating one
without the other — or leaving one updated after a failure — produces a
model that is silently, subtly wrong. Phase F therefore treats the update
as a single transaction.

**Locked:**

- `running_mean` and `running_var` update as **one atomic transaction**.
- **Validation and staging happen before mutation.** Nothing
  caller-visible changes until both new values exist and are known good.
- Both new values are computed as **independent, graph-free native
  state** — never views onto, or graph descendants of, the forward's
  tensors.
- The **commit preserves the Python identity of both registered
  buffers**. The buffer objects are the same objects afterwards; only the
  cores they own are swapped. This is what makes `register_buffer`'s
  identity guarantee, alias sharing, and in-place `load_state_dict`
  restore keep working.
- **Both old cores remain valid until the full commit succeeds.** Nothing
  is closed mid-transaction.
- **On any failure or interruption — including `KeyboardInterrupt`
  between the two swaps — both original buffers are restored unchanged.**
- **Replaced cores close exactly once**, after a successful commit.
- **Staged cores close on rollback**, exactly once.
- **Parameter versions do not move** during a running-buffer update.
  `gamma` and `beta` are untouched; no `NativeParameter` is replaced; no
  existing graph becomes stale because the statistics advanced.
- A **failed forward or a failed running-state update leaves no partial
  state change**: the module is exactly as it was, and a later valid
  forward succeeds.
- **No general public in-place mutation API is added to ordinary
  `NativeTensor`.** The transaction is private module machinery, not a
  new tensor capability. `NativeTensor` stays immutable from the public
  surface, and `NativeParameter.copy_value_` stays the only public
  controlled-mutation primitive in the native line.

**Where the transaction comes from.** `NativeModule.load_state_dict`
already implements exactly this shape — preflight validation, a `staged`
list of independent owning copies built with nothing mutated yet, a
commit loop that swaps cores while recording `adopted` pairs, a
`BaseException`-guarded rollback that restores every original core and
closes every staged core, and a final pass that closes each replaced core
exactly once. **F1 extracts and generalizes that behavior into a private
reusable state transaction** and re-expresses `load_state_dict` in terms
of it, proving by the existing test suite that the extraction is
behavior-preserving. F3 then uses the same primitive for the running
update. Writing a second, parallel implementation of these semantics
would be the likeliest place for Phase F to introduce a subtle
state-corruption bug, which is exactly why F1 precedes F3.

**Momentum convention.** The update follows the stable reference:

```
running_mean ← (1 − momentum) * running_mean + momentum * batch_mean
running_var  ← (1 − momentum) * running_var  + momentum * batch_var
```

with `batch_var` the **population** variance (§6.3), computed once and
used both for normalizing and for this update.

---

## 9. Ownership and lifetime

Locked:

- **`NativeModule` registration does not imply exclusive ownership.**
  Registering a parameter or a buffer does not make the module
  responsible for closing it, and does not forbid the caller from holding
  its own reference.
- **`NativeModule.close()` is not introduced in Phase F.** A recursive
  module-level close is a broader lifetime question — it would have to
  decide what happens to shared parameters, aliased buffers, and
  optimizer-held references — and normalization does not need it. Adding
  it here would be a framework-wide contract change smuggled in under a
  normalization phase.
- **Examples and tests using stateful modules must explicitly close
  both** `model.parameters()` **and** `model.buffers()`. This is the
  practical consequence of the previous two points, and it must be shown
  in the F6 example rather than left to a reader to infer. Existing
  examples that use no stateful module are unaffected.
- **Forward-created constants, snapshots, and temporaries must close
  deterministically** when no longer needed — including the eval-mode
  snapshots of §7 once the graph that holds them is released, the `eps`
  constant, and every intermediate the composition allocates that the
  graph does not need.
- **Graph-required snapshots remain valid until graph release.** A
  snapshot the backward will read is graph-owned state with the same
  lifetime the Phase-D pooling winner buffer and the Phase-E saved
  probabilities have: retained under `retain_graph=True`, retained across
  a failed retryable backward, and released exactly once when the graph
  history is.
- **No phase contract may rely solely on garbage collection.**
  `__del__`-based cleanup stays a defensive backstop, never a guarantee.
- **Graph-owned resources remain exactly-once closeable.** Double release
  is a bug, and the existing `graph_resources` contract already enforces
  it.
- **Failed graph construction must release all unadopted native
  objects.** If a node cannot be built, every native object allocated for
  it and not yet adopted by the graph is closed before the exception
  propagates.

F5 and F8 must include live-storage baseline tests: repeated
train/eval/backward cycles over a normalized model must return the native
live-storage counters to their baseline, proving no per-iteration growth
from snapshots, constants, or transaction staging.

---

## 10. State and checkpoints

Locked:

- Persistent `running_mean` and `running_var` participate in
  `state_dict()`, `load_state_dict()`, `save_native_checkpoint()`, and
  `load_native_checkpoint()` **through the existing infrastructure**.
  `NativeModule` already includes persistent buffers in `state_dict()`
  under their canonical dotted names and restores them in the same atomic
  transaction as parameters; the checkpoint layer already serializes
  whatever `state_dict()` produces. Phase F adds **no new state
  mechanism** — it adds a module whose state happens to include buffers.
- The checkpoint format **`"tensorforge.native_checkpoint"`, version 1,
  remains unchanged.** **No new archive schema version is needed merely
  because a model's state has new persistent keys** — the manifest
  already maps arbitrary canonical model keys to indexed float64 arrays,
  which is exactly what a `bn.running_mean` key is. A format bump would
  break every existing checkpoint for no gain.
- **Buffer identity must survive loads.** After `load_state_dict()` or a
  checkpoint load, `model.bn.running_mean` is the *same Python object* it
  was before, holding new values.
- **Buffer loads increment no parameter version**, and make no existing
  graph stale. A load that *also* carries `gamma`/`beta` does increment
  those parameters' versions and therefore does stale a graph that reads
  them — the unchanged v3.7 rule, caused by the parameters and never by
  the buffers (§7.2).
- **Parameter versions behave exactly as before**: one increment per
  matched canonical parameter, after the atomic commit.
- **Exact resume must eventually include** — proved end to end in F6 —
  the remaining loss suffix, the parameters, the optimizer state, the
  **running means**, the **running variances**, the final
  logits/predictions, and the **evaluation-mode output**. Evaluation-mode
  output is the part that would silently pass if the running statistics
  were mishandled, so it is a required element of the proof, not an
  optional extra.
- **Training flags remain runtime state and are not checkpointed.**
  `training` is not serialized; a loaded model is in whatever mode the
  caller puts it in. This matches the stable line.
- **`normalized_shape`, `num_features`, `eps`, `momentum`, and the affine
  configuration remain constructor configuration**, not serialized tensor
  state — the same rule that keeps `NativeCrossEntropyLoss`'s `reduction`
  out of the checkpoint. Reconstructing a model is the caller's job;
  loading state into it is the framework's.

---

## 11. Capability inventory contract

The registry in `tensorforge.backends.cpp` stays the authority. Phase F's
planned changes are locked here so no milestone has to invent a
placement.

**`NATIVE_MODULES` eventually gains** (each entry added by the milestone
that implements it, never before):

- `NativeLayerNorm` (F2)
- `NativeBatchNorm1d` (F3)
- `NativeBatchNorm2d` (F4)

**`UNSUPPORTED` eventually removes:**

- `"layernorm"` (F2)
- `"batchnorm"` (F4 — only when *both* BatchNorm modules exist; the name
  is unqualified, so removing it while only the 1-D form ships would
  over-claim)

**`UNSUPPORTED` retains**, at the end of Phase F and beyond it:

- `"dropout"`, `"float32"`, `"cuda"`, `"amp"`

**Inventories that receive nothing from Phase F:**

- `TENSOR_CORE_OPS` — **no normalization operation.** Normalization is
  composed at the module layer; it adds no Core method.
- `AUTOGRAD_OPS` — **no normalization operation.** There is no
  `NativeTensor.layer_norm` or `.batch_norm` (§4).
- `RAW_KERNELS` — **no normalization kernel.** There is no C++
  normalization kernel at all (§3).
- `NATIVE_LOSSES` and `NATIVE_METRICS` — **unchanged.**

**`STATE_SUPPORT` correction.** The tuple currently reads
`("state_dict", "load_state_dict", "save_native_checkpoint",
"load_native_checkpoint")`. Persistent native buffers have been supported
since the pre-Phase-D hardening milestone (`register_buffer`,
`buffers()`, `named_buffers()`, persistent buffers in `state_dict` and
checkpoints), but the registry does not say so, so `backend_info()`
under-reports an existing capability. **F1 should correct
`STATE_SUPPORT`** so persistent native buffers are represented
explicitly, for example:

```
STATE_SUPPORT = (
    "persistent_buffers",
    "state_dict",
    "load_state_dict",
    "save_native_checkpoint",
    "load_native_checkpoint",
)
```

This is **existing-capability reconciliation, not a Phase-F numerical
claim** — the capability shipped long before Phase F was defined.
**F0 deliberately does not make this change**: the preferred scope is to
design it here and implement it in F1 alongside the transaction
extraction, with the accompanying `tests/test_cpp_backend_info.py` update
that proves the new name maps to a real capability. Until F1 lands,
`STATE_SUPPORT` stays exactly as it is.

---

## 12. Testing contract

Every milestone below carries its own required tests. Across the phase,
these must all hold:

- **Reference parity.** `NativeLayerNorm` and `NativeBatchNorm1d` match
  `tensorforge.nn.LayerNorm` / `tensorforge.nn.BatchNorm1d` to tolerance
  in both modes. `NativeBatchNorm2d` matches hand-computed cases and the
  stable mathematics applied over the NCHW reduction set.
- **Central finite differences** verify the input gradient and every
  affine-parameter gradient, in training mode, for both shapes —
  including through the batch mean and variance.
- **Mode behavior.** LayerNorm's output is mode-independent. BatchNorm's
  output differs between modes on the same input, and eval mode's output
  depends only on the running statistics and the affine parameters.
- **Buffer graph safety** (§7): an eval graph must not hold the
  registered buffer object; a training step, a **buffer-only**
  `load_state_dict()`, and a **buffer-only** `load_native_checkpoint()`
  performed *after* an eval forward must each leave that graph's backward
  unchanged and non-raising. The checkpoint half must exercise the real
  archive path over the *same* registered buffer objects — a
  state-dictionary test does not stand in for it. Separately, a **full**
  model checkpoint load, which also replaces `gamma`/`beta`, must raise
  the existing stale-*parameter* error, and a test must attribute that
  raise to the parameter versions rather than to the buffers.
- **Transaction atomicity** (§8): an injected failure during staging and
  during commit each leaves both buffers, both parameters, every version,
  and every gradient exactly as before, with no leaked staged core and no
  closed live core; a later valid forward succeeds.
- **State and checkpoints** (§10): buffer keys appear in `state_dict()`
  under canonical dotted names; loading preserves buffer identity; no
  parameter version moves for a buffer-only load; format version stays 1.
- **Ownership** (§9): repeated train/eval/backward cycles return the
  live-storage counters to baseline; explicit closes over both
  `parameters()` and `buffers()` release everything.
- **NumPy tripwire**: one complete normalized training step reaches no
  NumPy numerical routine and converts no tensor data.
- **Separation**: stable modules reject native tensors and vice versa;
  no implicit dispatch appears anywhere.
- **Registry agreement**: the inventories, the exports, the support
  matrix, and this document describe the same surface at every milestone.

Dependency-free C++ CTests are **not** part of Phase F: there is no new
C++ code to test. Sanitizer validation (F9) re-runs the existing native
suite under the normalization workload to prove the composition drives
the existing kernels safely, not to validate new kernels.

---

## 13. Benchmark contract

F7 characterizes the normalization stack under exactly the Phase-E rules,
which are not negotiable:

- **Correctness is gated before every measurement.** A failed gate exits
  nonzero and publishes no timing.
- Each case is **labelled with the reference it actually used** —
  `stable_tensorforge` where a stable equivalent exists, `native_only`
  where no honest analogue would exist.
- Timing uses `time.perf_counter_ns` with warm-up, repeated measurements,
  setup and cleanup outside the timed region, and **median** reporting
  alongside min, max, and spread.
- `--smoke` and `--json` modes exist; **no result file is written**.
- **No speed assertion, no committed timing number, and no CI timing
  threshold anywhere.** Observed ratios are local characterizations, never
  promises.

**Shipped.** `benchmarks/benchmark_native_normalization.py` implements
exactly this contract — nine correctness-gated cases (LayerNorm forward
and backward; BatchNorm1d training forward, evaluation forward, and
backward; BatchNorm2d training forward, evaluation forward, and backward;
and one complete F6-style normalized training step), with
`stable_tensorforge` labels where a real stable counterpart exists and
`native_only` for the three BatchNorm2d cases, which publish no timing
ratio because the stable line has no public `BatchNorm2d` and a
layout-transformed stand-in would make the ratio misleading. Those cases
keep a rigorous correctness oracle regardless. See the F7 milestone entry
in §15 for the full record.

---

## 14. Comparison with the Daedalus design

Phase F was designed independently and then compared against the relevant
Daedalus normalization design, as every phase of this line has been. The
comparison is recorded honestly: some ideas are worth taking, several are
actively wrong for this architecture, and none of the implementation is
copied.

### Ideas worth taking

| Idea | Why it fits TensorForge |
|---|---|
| **LayerNorm composed over existing operations** | Exactly the conclusion §3 reaches independently: composition buys an exact backward with no new kernel and no new trust boundary. Convergent, and confirming. |
| **Separate `BatchNorm1d` and `BatchNorm2d` public modules** | Keeps the accepted input rank part of each class's contract instead of hiding it behind rank-polymorphism, so a wrong-rank input is a clear error rather than a silently different reduction. |
| **A shared private BatchNorm implementation behind those two classes** | The reduction axes and broadcast shape are the only real difference; one implementation means one place for the statistics, the transaction, and the mode split to be right. |
| **Clear, explicit train/eval behavior** | Matches the stable line's existing mode contract and makes the eval path's statistics source obvious at the call site. |
| **Stateless RNG-kernel ideas for a future dropout phase** | Recorded as a *future* idea only. Dropout and a native RNG are explicitly outside Phase F (§1), and nothing in this phase depends on them. |

### Ideas rejected

| Idea | Why it is rejected here |
|---|---|
| **pybind11 architecture** | The native line's whole point is an explicit C ABI reached through ctypes, with no Python headers in C++. Adopting pybind11 for one layer would fracture the ABI contract and the error-status contract with it. |
| **C++-managed autograd** | Python owns the graph; the kernels stay graph-unaware. That split is what keeps the C++ side auditable, sanitizable, and free of Python lifetime concerns. |
| **Host NumPy BatchNorm statistics** | Would silently take the numerics off the native path and break the NumPy tripwire. §2 forbids it: native numerical paths must not silently use NumPy. |
| **Detached training statistics** | Gives a *different and wrong* gradient. §6.3 requires the training backward to differentiate through the batch mean and variance. |
| **Host arrays as TensorForge running state** | Running statistics are model state that must ride `state_dict()`, the atomic loader, and the pickle-free checkpoint path. Host arrays would need a parallel serialization route and would break buffer identity. |
| **Untracked checkpoint-critical RNG attributes** | Any state a resume depends on must be visible to `state_dict()` and the checkpoint, or "exact resume" is not exact. Phase F has no RNG state at all, and this is a reason it stays that way. |
| **Copying an implementation directly** | Non-negotiable across the whole project: the design is compared, the code is written here. |

### Why TensorForge's design is cleaner for this architecture

- **Native-only BatchNorm numerics** — every value comes from native
  kernels, so the tripwire holds and the numbers mean what the phase
  claims.
- **Exact training gradients** — differentiating through the batch
  statistics, verified against central finite differences, rather than
  detaching for convenience.
- **Persistent native buffers** — running statistics are real
  `NativeTensor` state in the module's own registry, not host arrays
  bolted alongside it.
- **Graph-safe snapshots** — §7 makes an eval graph immune to later
  mutation *by construction*, instead of detecting the problem after the
  fact or leaving it undetected.
- **Atomic two-buffer updates** — the pair advances together or not at
  all, over the same transaction primitive the state loader already
  proved.
- **A pickle-free checkpoint format, retained** — new persistent keys
  ride the existing version-1 manifest with no schema change and no
  pickle.

---

## 15. Milestone ladder (F0–F9)

| Milestone | Deliverable | Status |
|---|---|---|
| F0 | Phase-F architecture contract and repository reconciliation | **complete** |
| F1 | Atomic native-buffer state transactions | **complete** |
| F2 | `NativeLayerNorm` | **complete** |
| F3 | `NativeBatchNorm1d` | **complete** |
| F4 | `NativeBatchNorm2d` | **complete** |
| F5 | Normalization state, checkpoint, and graph-safety hardening | **complete** |
| F6 | Deterministic normalized training and exact resume | **complete** |
| F7 | Native normalization benchmark characterization | **complete** |
| F8 | Cross-cutting Phase-F integration and semantic guardrails | **complete** |
| F9 | Phase-F closure | **complete** |

Each milestone's full contract follows; the table above is the status
summary, and the registry remains the authority on what is live. **F0,
F1, F2, F3, F4, F5, F6, F7, F8, and F9 are all complete, so Phase F is
complete.**

### F0 — Phase-F architecture contract and repository reconciliation *(this document)* — **complete**

- **Objective:** lock the complete Phase-F contract above, and reconcile
  the documentation drift left after Phase E closed so the phase starts
  from an accurate baseline.
- **Layer / scope:** documentation and guardrail tests only.
- **Expected files:** this document; `docs/native_support_matrix.md`,
  `docs/roadmap.md`, `docs/project_summary.md`, `docs/architecture.md`,
  `docs/release_history.md`, `docs/backend_experiments.md`, `README.md`,
  `CLAUDE.md`, `src/tensorforge/experimental/__init__.py` (docstring
  only), `tests/test_docs.py`.
- **Required tests:** durable semantic guardrails — Phase E is positively
  marked complete on every authoritative status surface; native
  classification is presented as shipped and never as absent, upcoming,
  or in progress; this document exists, is linked, and contains F0–F9;
  Phase F is described as designed but not numerically implemented;
  LayerNorm and BatchNorm remain in `UNSUPPORTED` and absent from the
  experimental exports; no normalization Core op, autograd op, raw
  kernel, C ABI symbol, or module is advertised; the public experimental
  export set is unchanged; the backend capability registries are
  unchanged; dropout, float32, CUDA, and AMP remain unsupported; the
  authoritative docs agree on the phase sequence; every document linked
  from the README exists.
- **Validation:** `uv run pytest tests/test_docs.py
  tests/test_cpp_backend_info.py -q`, then the full `uv run pytest`. The
  existing compiled Release backend stays the active runtime; no Debug
  build, sanitizer run, or benchmark is required.
- **Risks:** advertising an unimplemented capability; writing brittle
  prose-locking tests instead of semantic ones; deleting useful history
  instead of labelling it; silently changing a registry while claiming a
  documentation-only milestone.
- **Dependencies:** completed Phase E.
- **Non-goals:** **any** numerical behavior — no kernel, ABI symbol,
  ctypes declaration, Core method, tensor operation, module, buffer
  helper, benchmark, or example; and no registry change (including the
  `STATE_SUPPORT` correction, which is F1's).
- **Completion criteria:** this document locks the full contract with
  F0–F9 specified; every authoritative status document presents Phase E
  as complete; Phase F is defined but not advertised as implemented;
  BatchNorm and LayerNorm remain unsupported; exports and inventories are
  unchanged; no numerical runtime or C++ file changed behavior; the
  guardrails catch the stale claims found in the audit; focused and full
  test runs pass with no regression.

### F1 — Atomic native-buffer state transactions — **complete**

- **Objective:** extract and generalize the staging/commit/rollback
  behavior already inside `NativeModule.load_state_dict` into a private,
  reusable state transaction, and correct `STATE_SUPPORT` to report the
  persistent-buffer capability that already exists.
- **Layer / scope:** `src/tensorforge/experimental/native_module.py`
  (refactor only) and `src/tensorforge/backends/cpp.py`
  (`STATE_SUPPORT` tuple and its comment).
- **Expected files:** `native_module.py`; `cpp.py`;
  `tests/test_native_buffers.py`; `tests/test_cpp_backend_info.py`;
  possibly a new focused `tests/test_native_state_transaction.py`.
- **Required tests:** the extraction is **behavior-preserving** — every
  existing `load_state_dict`, checkpoint, and buffer test passes
  unchanged; the transaction commits all-or-nothing; staged cores close
  exactly once on rollback; replaced cores close exactly once after
  commit; original cores survive until commit succeeds; identity is
  preserved for both parameters and buffers; parameter versions move only
  after a full commit and never for buffers; an interruption between
  swaps rolls back; `backend_info()["state_support"]` contains the new
  name and every advertised name maps to a real capability.
- **Validation:** the full Python suite; no build or sanitizer work.
- **Risks:** changing `load_state_dict` semantics while "only
  refactoring"; introducing a second mutation path; over-claiming in
  `STATE_SUPPORT`.
- **Dependencies:** F0.
- **Explicit non-goals:** no normalization module; no public in-place
  mutation API on `NativeTensor`; no buffer versioning; no kernel, ABI
  symbol, or checkpoint schema change.
- **Completion criteria:** one private transaction primitive is used by
  both `load_state_dict` and (later) the running-statistics update; every
  pre-existing test passes untouched; `STATE_SUPPORT` reports persistent
  buffers and its guardrail proves the name is real.
- **Shipped (F1).** Exactly the above, plus the refinements the
  implementation settled:

  - **The private helper is
    `src/tensorforge/experimental/_native_state.py`** — a
    `NativeStateEntry(label, destination, make_core, source)` record and
    one `replace_native_state(entries)` transaction. It is deliberately
    absent from `tensorforge.experimental.__all__` and is **not** a
    public in-place mutation API; `NativeParameter.copy_value_` remains
    the only public controlled-mutation primitive in the native line.
  - **The transaction takes ownership of every core a factory returns**
    — it installs it or closes it, exactly once, on every path. Staging
    calls each entry's `make_core()` before any destination is mutated,
    and validates the produced core (a `NativeTensorCore`, open, owning,
    contiguous, metadata-matched, sharing storage with neither its
    destination's current core nor any other staged core).
  - **The commit boundary is the point at which every core swap *and*
    every parameter-version increment has succeeded.** Both live inside
    one `BaseException` rollback guard — a refinement over the inline
    version, where the increments sat *outside* the guard. A failure at
    either step now restores every swapped core **and** every moved
    version and closes every staged core, so a failed transaction moves
    no version at all. On success the outcome is identical to the
    previous inline behavior, so nothing observable changed.
  - **Only the release of the replaced cores is past the boundary.**
    Each replaced core is closed exactly once; every one is attempted
    even if an earlier close raises, so ownership is never ambiguous,
    and the first failure is then re-raised wrapped in a `RuntimeError`
    that states plainly that the state change itself succeeded. Since
    `NativeTensorCore.close()` is idempotent and does not raise, this is
    a defensive path only.
  - **Deduplication is by destination object identity**, not by label: a
    shared parameter or buffer reachable under several registered names
    is one destination, swapped once, version-bumped once, released
    once. Two entries for one destination are the same request only when
    they name the same `source` object; any other duplicate is a
    conflict — two different values for one object — and is rejected
    during planning, before anything is staged or mutated.
  - **A defensive re-check** runs immediately before each swap: the
    destination's core must still be the one planning recorded, so a
    factory side effect, signal handler, or reentrant caller that
    changed it aborts the transaction before any change is made.
  - **`NativeModule.load_state_dict` now delegates to it** with its
    public signature, validation order, error messages, key reporting,
    identity guarantees, version semantics, atomicity, and ownership
    behavior all unchanged. Its staging copies still go through this
    module's own `_native_copy`, so the long-standing staging seam (and
    the tests that use it) is untouched. `state_dict()` output, the
    checkpoint format, and the checkpoint version are unchanged.
  - **`STATE_SUPPORT` now reads
    `("persistent_buffers", "state_dict", "load_state_dict",
    "save_native_checkpoint", "load_native_checkpoint")`.** This is
    reconciliation of an under-report: `register_buffer` / `buffers()` /
    `named_buffers()` and persistent buffers in `state_dict` and
    checkpoints have existed since the pre-Phase-D hardening milestone.
    Unlike the four names beside it, `persistent_buffers` names a
    *capability* rather than one callable, and the guardrails resolve it
    explicitly to that API rather than by relaxing the "every advertised
    name is real" check.
  - **The private seams** `_stage_entry`, `_install_core`,
    `_restore_core`, `_bump_version`, `_restore_version`, and
    `_release_core` are module-level functions so tests can inject a
    failure at exactly one step. They are seams, not production flags:
    nothing in the library ever replaces them and there is no
    user-facing failure control.
  - **F1 added no normalization capability of any kind** — no
    normalization module, formula, forward or backward pass, eval
    snapshot, running-statistic update, kernel, C ABI function, ctypes
    declaration, public tensor operation, or experimental export. F1
    itself is state management and capability reporting only; F3 will be
    the transaction's second caller. (F1 left `"layernorm"` in
    `UNSUPPORTED`; the *later* milestone F2 removed it when it shipped
    `NativeLayerNorm`.)

### F2 — `NativeLayerNorm` — **complete**

**Shipped (F2).** `NativeLayerNorm` is live in
`src/tensorforge/experimental/native_layernorm.py`, exported from
`tensorforge.experimental`, and listed in `NATIVE_MODULES`; `"layernorm"`
has left `UNSUPPORTED` (`"batchnorm"` stays). It is the first native
normalization module and it delivers the §5 contract exactly:

- **Composed, not primitive.** Forward is built only from existing
  differentiable `NativeTensor` operations — `mean`, `subtract`,
  `multiply`, `add`, `sqrt`, `reciprocal` — so the existing native
  autograd **is** the backward. F2 added **no** C++ code, normalization
  kernel, C ABI symbol, ctypes declaration, `NativeTensorCore` method,
  custom backward, functional `layer_norm`, or `NativeTensor.layer_norm`
  operation. No operation inventory grew; only `NATIVE_MODULES` did.
- **Population variance, `sqrt(var + eps)`.** The variance divides by the
  element count (no Bessel correction) and epsilon is added inside the
  square root, both proved against hand-computed values.
- **Multi-axis by sequential single-axis means.** `NativeTensor.mean`
  reduces one axis at a time; the trailing-`k` mean is taken as a
  sequence of `mean(axis=a, keepdims=True)` calls, and because each
  reduced dimension is retained at size 1 the axis numbers stay valid
  across the sequence. No tuple-axis reduction was added to
  `NativeTensor`.
- **Stateless.** No buffers, no running statistics; identical output in
  train and eval mode (forward never reads `training`). `weight`/`bias`
  `NativeParameter`s exist only when `elementwise_affine=True` (registered
  in that order, version 0), and `elementwise_affine=False` registers no
  parameters and contributes no state keys.
- **Owning contiguous output.** Every forward returns a fresh, owning,
  row-major-contiguous tensor, never a `NativeParameter` or a borrowing
  view. Construction validates before any native allocation, and a failed
  bias allocation closes the already-created weight deterministically.
- **State/checkpoint unchanged.** Affine parameters serialize through the
  existing `state_dict`/`load_state_dict` and native checkpoint (format
  version **1**, unchanged), with identity preserved and versions
  advancing under the existing contract.

F2 added no normalization *operation*, kernel, ABI symbol, or
`NativeTensorCore` method; BatchNorm remains unsupported, and **F3
(`NativeBatchNorm1d`) is the next milestone.**

**Original F2 contract (preserved for reference).**

- **Objective:** the first native normalization module — stateless,
  differentiable, composed from existing operations.
- **Layer / scope:** a new experimental module plus its export and
  inventory entry. No C++, no ABI, no Core method.
- **Expected files:**
  `src/tensorforge/experimental/native_layernorm.py`;
  `src/tensorforge/experimental/__init__.py` (import and `__all__`);
  `src/tensorforge/backends/cpp.py` (`NATIVE_MODULES` gains
  `"NativeLayerNorm"`, `UNSUPPORTED` drops `"layernorm"`);
  `tests/test_native_layernorm.py`; the support matrix and this document.
- **Required tests:** constructor validation (int and sequence
  `normalized_shape`, `bool` rejection, positivity, `eps`,
  `elementwise_affine`) with no allocation on rejection; trailing-shape
  matching and rejection; parity with `tensorforge.nn.LayerNorm` to
  tolerance for several ranks; population variance and `sqrt(var + eps)`
  ordering proved against hand-computed values; identical output in train
  and eval mode; `elementwise_affine=False` registers no parameters and
  contributes no state keys; central finite differences for the input,
  `weight`, and `bias`; fresh owning contiguous output; deterministic
  state keys `["weight", "bias"]`; composition inside a
  `NativeSequential`; NumPy tripwire over a forward and backward.
- **Validation:** the full Python suite with the Release backend.
- **Risks:** `sqrt(var) + eps` instead of `sqrt(var + eps)`; a sample
  variance instead of a population variance; returning a borrowing view;
  leaking the `eps` constant or intermediates.
- **Dependencies:** F1 (not strictly required numerically, but F1 lands
  first so the phase's state machinery is settled before any module uses
  it).
- **Explicit non-goals:** BatchNorm; any buffer; any kernel or ABI
  symbol; a functional `layer_norm`; a `NativeTensor.layer_norm`
  operation; RMSNorm or GroupNorm.
- **Completion criteria:** the module ships, matches the stable
  reference, is finite-difference verified, appears in `NATIVE_MODULES`
  and the exports, `"layernorm"` has left `UNSUPPORTED`, and
  `"batchnorm"` has not.

### F3 — `NativeBatchNorm1d` — **complete**

- **Objective:** the first **stateful** native module: `(N, C)` batch
  normalization with persistent running statistics, distinct train/eval
  behavior, graph-safe eval snapshots, and atomic running updates.
- **Layer / scope:** a new experimental module (containing the shared
  private BatchNorm implementation) plus its export and inventory entry.
  No C++, no ABI, no Core method.
- **Expected files:**
  `src/tensorforge/experimental/native_batchnorm.py`;
  `src/tensorforge/experimental/__init__.py`;
  `src/tensorforge/backends/cpp.py` (`NATIVE_MODULES` gains
  `"NativeBatchNorm1d"`; `UNSUPPORTED` **keeps** `"batchnorm"` until F4);
  `tests/test_native_batchnorm1d.py`; the support matrix and this
  document.
- **Required tests:** constructor validation (`num_features`, `eps`,
  `momentum` range) with no allocation on rejection; `(N, C)`-only input
  with clear rejection otherwise; training-mode parity with
  `tensorforge.nn.BatchNorm1d`; eval-mode parity using the running
  statistics; population variance in both the normalization and the
  update; the exact momentum convention; central finite differences for
  the input, `gamma`, and `beta` **through** the batch statistics; the §7
  snapshot rule (an eval graph does not hold the registered buffer
  object; a later training step, a buffer-only state load, and a
  buffer-only `load_native_checkpoint()` over the same registered buffer
  objects each leave that graph's backward unchanged, while a full
  checkpoint load that also replaces `gamma`/`beta` legitimately raises
  the existing stale-*parameter* error); the §8 transaction (failure during
  staging and during commit each leave both buffers, parameters,
  versions, and gradients unchanged, with nothing leaked and a later
  forward succeeding); no parameter version moves on a running update;
  buffers are excluded from `parameters()` and from every optimizer;
  state keys include `running_mean`/`running_var`; live-storage baseline
  over repeated cycles; NumPy tripwire.
- **Validation:** the full Python suite with the Release backend.
- **Risks:** capturing the live buffer in an eval graph (the §7 hazard);
  updating one running buffer without the other; detaching the training
  statistics; letting a running update touch a parameter version;
  per-iteration storage growth from snapshots or staged cores.
- **Dependencies:** F1 (the transaction) and F2 (the composed
  normalization mathematics).
- **Explicit non-goals:** `NativeBatchNorm2d`; `track_running_stats`;
  `affine=False`; `num_batches_tracked`; unbiased running variance; any
  kernel or ABI symbol; a `NativeTensor.batch_norm` operation.
- **Completion criteria:** the module ships with both modes correct and
  finite-difference verified, §7 and §8 are proved by tests,
  `"NativeBatchNorm1d"` is in `NATIVE_MODULES` and the exports, and
  `"batchnorm"` is still listed in `UNSUPPORTED`.
- **Shipped (F3).** Exactly the above, plus the refinements the
  implementation settled:

  - **`src/tensorforge/experimental/native_batchnorm.py`** holds one
    shared private `_NativeBatchNorm(NativeModule)` carrying *every*
    behavior — constructor validation, parameter and buffer creation,
    forward-state validation, both modes' mathematics, the eval
    snapshots, the running update, the transaction, and the cleanup —
    and the public `NativeBatchNorm1d` subclass supplies **only** shape
    configuration: `_INPUT_NDIM = 2`, `_REDUCTION_AXES = (0,)`,
    `_TRAILING_DIMS = 0`, `_LAYOUT = "(N, C)"`. It declares no callable
    of its own, so F4 adds the NCHW shape by supplying `4`, `(0, 2, 3)`,
    `2`, and `"(N, C, H, W)"` — never a second implementation. The
    private base is not exported.
  - **Composed, never primitive.** Forward is built only from existing
    differentiable `NativeTensor` operations — `mean`, `subtract`,
    `multiply`, `add`, `sqrt`, `reciprocal`, plus `reshape`,
    `contiguous_copy`, and `detach` for the graph-free state — so the
    existing autograd **is** the backward. F3 added **no** C++ code,
    normalization kernel, C ABI symbol, ctypes declaration,
    `NativeTensorCore` method, custom backward, functional `batch_norm`
    helper, or `NativeTensor.batch_norm` operation. No operation
    inventory grew; only `NATIVE_MODULES` did.
  - **Training statistics are differentiated through**, verified by
    central finite differences for the input, `gamma`, and `beta`, and
    by a direct test that the gradient is *not* the detached-statistics
    form. Population variance, `sqrt(var + eps)`, no Bessel correction.
  - **The same batch statistics drive both** the normalization graph and
    the running update; nothing is recomputed. They enter the update
    through `detach()` — a native storage-to-storage copy, no NumPy
    round trip — so the blend `(1 - momentum) * running + momentum *
    batch` builds no autograd node and no gradient path reaches it.
    `momentum=0.0` leaves both running values numerically unchanged;
    `momentum=1.0` makes them exactly the batch statistics.
  - **One atomic two-buffer commit** through
    `_native_state.replace_native_state` with two `NativeStateEntry`
    records — no fabricated state dictionary, no internal
    `load_state_dict()`, no attribute reassignment, no second
    transaction system. The transaction owns each factory-produced core,
    so the factories hand it independent copies of the prepared values.
    Injected failures at first-entry staging, second-entry staging,
    first-core install, and second-core install (the interrupted-swap
    case) each leave both buffers, both parameters, every version, and
    every gradient exactly as before, with no staged core leaked and a
    later valid forward succeeding.
  - **Forward ordering** is validate → build the complete output graph →
    prepare both graph-free replacement values → commit atomically →
    return the already-built output. A failure at any earlier point
    changes no running state, and once the commit succeeds no further
    numerical operation can fail before the return.
  - **Deterministic cleanup, without GC.** Two tracking lists with
    different lifetimes: graph temporaries (the output included) are
    closed only on failure, most-recent first; pure scratch (the eps
    constant, the detached statistics, the momentum-blend intermediates,
    the consumed variance snapshot) is closed on every path. Injected
    failures after every intermediate — the batch mean, the centering,
    the squaring, the variance, `sqrt`, `reciprocal`, the affine
    multiply and add, the detached statistics, the blend intermediates,
    and both transaction seams — each return the native live-storage
    counters to the pre-forward baseline **with no `gc.collect()`**,
    which the `sqrt`/`reciprocal` result-capturing cycles would
    otherwise require.
  - **The §7 snapshot rule is implemented and proved structurally.**
    Evaluation copies each running buffer into an independent, owning,
    contiguous, graph-free, gradient-free `(1, C)` tensor through a
    borrowing `reshape` view materialized by `contiguous_copy` — no
    NumPy, no host materialization, and no borrowing view in the graph.
    A recursive walk of the returned graph's parents *and*
    `graph_resources` cannot find either registered buffer by identity,
    and a training step, a buffer-only `load_state_dict()`, **or a
    buffer-only `load_native_checkpoint()` over the same registered
    buffer objects** leaves that graph's backward non-raising and
    numerically identical to the forward-time values. The checkpoint half
    is driven through a test-only buffer-only `NativeModule` that
    registers the BatchNorm's own `running_mean`/`running_var` as
    persistent aliases, so the real archive path replaces exactly those
    two objects while `gamma`/`beta` stay untouched; no production helper
    or public API was added for it. A **full** BatchNorm checkpoint load
    also replaces `gamma`/`beta`, so it moves their versions and the
    existing v3.7 guard rejects the old graph — proved by its own test,
    which attributes the raise to the parameter versions, shows the
    buffers are unchanged in identity and absent from the graph, and
    re-proves the buffer half of the *same* archive leaves an earlier eval
    graph valid.
  - **Snapshot lifetime rides the existing D9 `graph_resources`
    contract**, not a new one: the two values a backward could read (the
    mean snapshot and the derived inverse standard deviation) are
    adopted by the output node's history through one private adoption
    helper, so `retain_graph=True` keeps them, a one-shot `backward()`
    releases them exactly once, an abandoned graph frees them on
    `close()`, and a forward that builds no graph releases them
    immediately. No public operation and no second autograd
    implementation was added to achieve it.
  - **State and checkpoints unchanged.** State order is `gamma`,
    `beta`, `running_mean`, `running_var`; buffer and parameter
    identities survive `load_state_dict()` and checkpoint loading; a
    buffer-only load moves no parameter version; the archive format
    stays `"tensorforge.native_checkpoint"` **version 1**; `training` is
    not serialized.
  - **`"batchnorm"` deliberately stayed in `UNSUPPORTED`.** The name is
    unqualified, and F4 has not shipped `NativeBatchNorm2d`, so removing
    it now would over-claim.

### F4 — `NativeBatchNorm2d` — **complete**

- **Objective:** NCHW `(N, C, H, W)` batch normalization over the **same**
  shared private implementation, completing the normalization surface.
- **Layer / scope:** the second public class in the F3 module file, its
  export, and the inventory entries — including the removal of
  `"batchnorm"` from `UNSUPPORTED`, which is only honest once both shapes
  exist.
- **Expected files:**
  `src/tensorforge/experimental/native_batchnorm.py`;
  `src/tensorforge/experimental/__init__.py`;
  `src/tensorforge/backends/cpp.py`;
  `tests/test_native_batchnorm2d.py`; the support matrix and this
  document.
- **Required tests:** `(N, C, H, W)`-only input with clear rejection
  otherwise; reduction over N, H, W with `(1, C, 1, 1)` broadcasting;
  parity with hand-computed cases and with the stable 1-D mathematics
  applied over the NCHW reduction set; both modes; central finite
  differences through the batch statistics; the §7 and §8 contracts
  re-proved for the 4-D shape; the shared implementation genuinely shared
  (a change in one place affects both, verified structurally rather than
  by duplication); composition in a `NativeSequential` beside
  `NativeConv2d`/`NativeMaxPool2d`; live-storage baseline; NumPy
  tripwire.
- **Validation:** the full Python suite with the Release backend.
- **Risks:** reducing over the wrong axis set; broadcasting as `(1, C)`
  instead of `(1, C, 1, 1)`; duplicating the F3 implementation instead of
  sharing it; removing `"batchnorm"` from `UNSUPPORTED` before both
  shapes exist.
- **Dependencies:** F3.
- **Explicit non-goals:** `BatchNorm3d`; `InstanceNorm`; `GroupNorm`;
  synchronized or distributed BatchNorm; any kernel or ABI symbol.
- **Completion criteria:** both BatchNorm shapes ship over one
  implementation, all three normalization modules are in
  `NATIVE_MODULES` and the exports, and `"batchnorm"` and `"layernorm"`
  have both left `UNSUPPORTED` while `"dropout"`, `"float32"`,
  `"cuda"`, and `"amp"` remain.
- **Shipped (F4).** Exactly the above, plus the one refinement the NCHW
  shape forced:

  - **`NativeBatchNorm2d` declares only shape configuration** —
    `_INPUT_NDIM = 4`, `_REDUCTION_AXES = (0, 2, 3)`,
    `_TRAILING_DIMS = 2`, `_LAYOUT = "(N, C, H, W)"`, and
    `_CHANNELS_LAST = (0, 2, 3, 1)`. It defines a docstring and not one
    callable. Every method — `forward`, `_training_forward`,
    `_eval_forward`, `_mean_over`, `_inverse_std`, `_snapshot`,
    `_blend`, `_affine`, `_commit_running_state`, `_validate_forward`,
    `_registered_running`, `__init__`, `__repr__` — is inherited from
    `_NativeBatchNorm` **by function identity**, proved per method by a
    test, and the source contains exactly one definition of each.
  - **The reduction is three sequential single-axis means** with
    `keepdims=True`: `(N, C, H, W)` → `(1, C, H, W)` → `(1, C, 1, W)` →
    `(1, C, 1, 1)`. Because every reduced dimension is retained at size
    1 the axis numbers stay valid across the sequence, so no tuple-axis
    reduction was added to `NativeTensor`. Each mean is over a full axis
    of equal extent, so the composition is exactly the population mean
    over `N * H * W`. The channel axis is never reduced.
  - **The channelwise-affine refinement.** This is the one genuinely new
    problem the 4-D shape poses, and it is worth stating plainly.
    NumPy-style broadcasting aligns from the *trailing* axis, so
    `(N, C, H, W) * (C,)` would line `gamma` up with **W**, not with the
    channel axis — silently wrong whenever `W == C`, and a shape error
    otherwise. The obvious fix, reshaping `gamma` to `(1, C, 1, 1)`, is
    **rejected**: `multiply` records a stale-value guard entry only for a
    direct operand carrying a value version, and a reshaped `gamma` is an
    ordinary unversioned view. Under that alternative, mutating `gamma`
    after a forward stops raising the deterministic stale-parameter error
    and instead surfaces a bare "this NativeStorage has been closed" —
    exactly the confusing failure §7.1 names, and a silent-wrong-gradient
    hazard whenever the old storage happens to outlive the mutation.
    **So the activation moves instead of the parameter**: a borrowing
    `transpose` carries the channel axis to the trailing position
    (NCHW → NHWC), `gamma` and `beta` apply there as **direct rank-1
    operands** — keeping the existing version guard exactly as F3 left
    it — a second borrowing `transpose` carries the result back, and
    `contiguous_copy` materializes the fresh owning contiguous NCHW
    output. Both transposes are metadata-only and already
    differentiable, so **no gradient logic was added**: `multiply`'s
    existing broadcast-aware backward reduces `gamma`'s gradient over N,
    H, and W, `add`'s does the same for `beta`, and `transpose`'s
    backward applies the inverse permutation. The return permutation is
    *derived* from `_CHANNELS_LAST` rather than configured, so the two
    halves can never drift apart. Channels-last is an internal step of
    one method, never a public layout mode.
  - **`_affine` is the only shared method F4 added**, and it keeps the
    `(N, C)` path byte-identical to F3's (`_CHANNELS_LAST is None` means
    the channel axis is already trailing, so the parameters apply
    directly). Every F3 numerical, snapshot, transaction, and
    cleanup test passes unchanged.
  - **Everything else is inherited and re-proved for the 4-D shape**:
    `(1, C, 1, 1)` batch statistics and eval snapshots, `(C,)` running
    buffers, the population variance driving both the output and the
    running update, the graph-free atomic two-buffer commit, the
    registered buffers' absence from eval graphs by object identity
    *and* by storage, the buffer-only versus full-checkpoint
    distinction, deterministic mid-forward cleanup without GC (including
    at the new transpose/contiguous-copy points), non-contiguous NCHW
    input, and composition beside `NativeConv2d`/`NativeMaxPool2d`/
    `NativeFlatten` in a `NativeSequential`.
  - **`"batchnorm"` has left `UNSUPPORTED`**, which is only honest now
    that both shapes exist. `UNSUPPORTED` is exactly
    `("dropout", "float32", "cuda", "amp")`. The checkpoint format is
    unchanged at **version 1**, and F4 added no C++ code, kernel, C ABI
    symbol, ctypes declaration, `NativeTensorCore` method,
    `NativeTensor.batch_norm` operation, or custom BatchNorm backward.
  - **F4 completed the normalization *module* surface, not Phase F.**
    F5 (state/checkpoint and graph-safety hardening) was next.

### F5 — Normalization state, checkpoint, and graph-safety hardening — **complete**

- **Objective:** prove the state, checkpoint, ownership, and graph-safety
  contracts for the running buffers exhaustively — no new capability.
- **Layer / scope:** tests only, plus any documentation the proofs
  correct.
- **Expected files:** `tests/test_native_buffers.py`,
  `tests/test_native_checkpoint.py`, a focused
  `tests/test_native_normalization_state.py`; the support matrix and this
  document.
- **Required tests:** canonical dotted buffer keys in `state_dict()`;
  independent snapshot values that share no storage with the model in
  either direction; identity-preserving `load_state_dict()` and
  checkpoint loads; strict/non-strict key handling for buffer keys;
  exact shape/dtype/device validation with no casting or reshaping; no
  parameter version movement for buffer-only loads; **format version 1
  unchanged** and no new manifest key; a checkpoint round trip that
  reproduces eval-mode output exactly; the §7 rule under
  `retain_graph=True` and across a failed retryable backward; failure
  atomicity at the staging, commit, save, and corrupt-load boundaries;
  explicit closes over `parameters()` **and** `buffers()`; live-storage
  baselines.
- **Validation:** the full Python suite.
- **Risks:** proving the easy paths only; a checkpoint round trip that
  restores training output but not eval output; hidden reliance on
  garbage collection.
- **Dependencies:** F4.
- **Explicit non-goals:** any capability, kernel, module, or schema
  change.
- **Completion criteria:** §7, §8, §9, and §10 are each proved by tests
  rather than asserted by prose.
- **Shipped (F5).** Exactly the above — **tests and documentation only,
  no numerical behavior and no new public capability.** F5 added no
  module, operation, kernel, C ABI symbol, ctypes declaration,
  `NativeTensorCore` method, custom backward, checkpoint schema field, or
  export; the exports, `NATIVE_MODULES`, `STATE_SUPPORT`, `UNSUPPORTED`,
  and every operation inventory are exactly what F4 left, and the
  checkpoint format stays version 1. What it delivered:

  - **The focused `tests/test_native_normalization_state.py`** carries the
    cross-cutting proofs the single-module milestones could not: it builds
    small test-only fixtures — a nested 1-D model (BatchNorm1d under a
    child `NativeSequential`, dotted keys asserted), a nested 2-D model
    (BatchNorm2d beside `NativeConv2d`/`NativeMaxPool2d`), a mixed model
    (`NativeLinear` + `NativeLayerNorm` + `NativeBatchNorm1d` +
    `NativeBatchNorm2d`, so parameter-only, stateless-normalization, and
    persistent-buffer state coexist), a shared-child-module fixture, and
    an exact buffer-alias fixture — and proves against them: exact ordered
    canonical dotted state keys (parameters in `named_parameters()` order,
    then persistent buffers in `named_buffers()` order, with the BatchNorm
    local order `gamma`, `beta`, `running_mean`, `running_var`);
    identity-deduplicated traversal under shared modules and exact buffer
    aliases with the first-discovered name winning, and cycle-safe
    traversal; graph-free, owning, contiguous, metadata-matched
    `state_dict()` snapshots independent of the model **in both directions
    by storage identity**; a snapshot failure after a partial mapping
    spanning both categories closing every created snapshot with no gc;
    the full strict missing/unexpected/both-lists matrix over the buffer
    keys; non-strict partial loads (`running_mean`-only, `running_var`-only,
    both, `gamma`-plus-a-buffer, nested dotted keys) touching only the
    matching subset, advancing only loaded-parameter versions, moving no
    version on a buffer-only load, and aborting the whole matching subset
    atomically on one invalid entry; exact shape/dtype/device validation
    that never casts, reshapes, or moves — the dtype/device half driven
    through the narrowest property seam, since the runtime is float64/cpu
    only; identity-, version-, gradient-, and traversal-preserving
    successful mixed loads; mixed parameter/buffer transaction rollback at
    staging, first install, a later install after swaps, version
    adjustment, and a `KeyboardInterrupt` between swaps; the version-1
    checkpoint manifest and archive gaining no normalization-specific
    field, with BatchNorm buffers serializing as ordinary model entries;
    exact **eval-output** reproduction across a round trip for both shapes
    (with the training-mode output shown to be insufficient) and through
    the full NCHW convolutional stack; buffer-only checkpoint loads over
    the module's own registered buffers replacing exactly those objects
    while sparing the parameters and leaving an earlier eval graph valid;
    the complementary **full** checkpoint load staling the graph through
    the *parameter*-version guard (attributed to `NativeParameter`, never
    to a buffer); a corrupt-archive matrix targeting the persistent
    running-buffer keys (manifest identity, model section, and archive
    arrays) mutating nothing; checkpoint staging/commit and atomic-save
    failure boundaries that leak nothing and preserve an existing
    destination byte-for-byte; eval graphs holding no registered buffer
    object or storage, only independent `(1, C)` / `(1, C, 1, 1)`
    snapshots, with repeated forwards taking independent snapshot storage;
    the §7 rule under `retain_graph=True` (snapshots retained until the
    final one-shot release, ignoring an intervening buffer mutation) and
    across a **failed retryable backward** (no partial commit, graph not
    freed, retry matching a clean control that ignores the mutated running
    values); and a live-storage baseline over the whole matrix, including
    explicit closes over `parameters()` **and** `buffers()` and
    identity-deduplicated closing of shared state.
  - **Two narrow generic-infrastructure additions** — a `state_dict()`
    partial-snapshot-failure cleanup test in
    `tests/test_native_buffers.py` and a checkpoint load-staging-failure
    rollback test in `tests/test_native_checkpoint.py` — both useful
    beyond normalization.
  - **No production behavior changed.** Every F5 proof passed against the
    F0–F4 implementation as shipped; no locked-contract bug was found, so
    no production file was touched.
  - **F5 completed the hardening, not the phase.** At F5 the remaining
    milestones F6–F9 had not started, so there was as yet no normalized
    end-to-end training example, no normalization benchmark, and no
    Phase-F integration file. All four have since shipped.

### F6 — Deterministic normalized training and exact resume — **complete**

- **Objective:** an end-to-end integration proof — a deterministic
  training run over a normalized native model, interrupted and resumed
  **exactly**, including the running statistics and the evaluation-mode
  output.
- **Layer / scope:** one example plus its integration tests and
  documentation. **No capability, and no inventory entry.**
- **Expected files:**
  `examples/native_normalization_training.py`;
  `tests/test_native_normalization_training.py`; `README.md`,
  `docs/examples.md` if the example is listed there, the support matrix,
  and this document.
- **Required tests:** the run is deterministic and bit-reproducible
  across two uninterrupted runs; the loss falls; interrupting at a fixed
  step, checkpointing model **and** optimizer state (format version 1),
  and resuming into a **fresh** model/optimizer pair reproduces the
  remaining loss suffix, every parameter, the optimizer state, **both
  running means and running variances**, the final logits/predictions,
  **and the evaluation-mode output**, exactly; the example closes both
  `parameters()` and `buffers()` explicitly; repeated steps retain no
  completed graph and grow no native storage; a NumPy tripwire over one
  complete step.
- **Validation:** the full Python suite; the example runs from the
  README command.
- **Risks:** a resume that matches training output but not eval output;
  fixtures that hide a running-statistics bug; presenting an integration
  proof as a benchmark or a generalization claim.
- **Dependencies:** F5.
- **Explicit non-goals:** real datasets; generalization or accuracy
  claims; timing claims; any new capability.
- **Completion criteria:** the example runs deterministically, the exact
  resume covers every element listed in §10, and no inventory grew.
- **Shipped (F6).** Exactly the above — **one example and its integration
  test, no capability, operation, kernel, C ABI symbol, `NativeTensorCore`
  method, custom backward, checkpoint schema field, benchmark, or export;
  no inventory changed and the checkpoint format stays version 1.** What
  it delivered:

  - **`examples/native_normalization_training.py`** trains
    `NativeNormalizedRegressor` — a **named** `NativeModule` subclass
    `hidden: NativeLinear(2, 8, seed=0)` → `batch_norm:
    NativeBatchNorm1d(8, momentum=0.1)` → `relu: NativeReLU()` →
    `layer_norm: NativeLayerNorm(8)` → `output: NativeLinear(8, 1,
    seed=1)`, so **both** normalization families run in every forward and
    the state keys are readable dotted names. `batch_norm` is the only
    stateful module (persistent `running_mean`/`running_var`);
    `layer_norm` contributes `weight`/`bias` affine parameters but **no
    buffers**. There is deliberately no `NativeBatchNorm2d` or
    convolutional layer — the full convolutional integration model is F8's
    scope. The task is one fixed eight-sample two-feature regression over
    frozen literals (nothing generated, shuffled, or sampled), the full
    batch in fixed order every step, driven by `NativeMSELoss` and
    `NativeAdam(lr=0.05)` for 24 steps.
  - **The deterministic evidence, observed and asserted exactly.** The
    training loss falls from ≈2.440245 to ≈0.027000 (a 98.9% reduction);
    two independently constructed uninterrupted runs are **bit-identical**
    in the whole loss history, every final parameter, the NativeAdam
    state, the running statistics, the final training-step prediction, and
    the final evaluation-mode output; and the global NumPy RNG cannot
    perturb the seeded construction. Every parameter is reached by
    backward, the running buffers receive no gradient and are excluded
    from the optimizer, and BatchNorm running state advances once per
    training forward while evaluation reads it without updating it.
  - **The exact checkpoint resume.** `run_resume_proof()` runs the
    schedule uninterrupted, then interrupted at step 10 — saving model
    **and** optimizer state (the BatchNorm running buffers ride as
    ordinary model state, format **version 1**), reloading into a
    **completely fresh** model/optimizer pair, and continuing. The two
    agree **exactly** (equality, never a tolerance): the prefix, the whole
    remaining loss suffix, the first resumed loss at the split, every
    parameter, the complete model state, both `running_mean` and
    `running_var`, the NativeAdam hyperparameters/counters/`m`/`v`, the
    final training-step prediction, and the final **evaluation-mode**
    output. The fresh target's parameter and buffer identities survive the
    load; the target is deliberately put in **eval** mode before loading
    and stays there afterwards, proving the training flag is runtime state
    and not serialized (it is switched back to train explicitly before
    continuing). Parameter *versions* are not compared across the load —
    the checkpoint does not serialize them, by design.
  - **Native and clean.** A complete normalized update — forward through
    BatchNorm and LayerNorm (with the running-statistics update), scalar
    MSE, backward, the NativeAdam step, and zero_grad — passes a strict
    NumPy/conversion tripwire, producing exactly the unarmed reference's
    values. Every public helper representing a completed run returns
    plain Python values only; the reporting helpers close their
    `state_dict()` and optimizer-state snapshots; each run explicitly
    closes its parameters **and** its buffers (there is no
    `NativeModule.close()`); repeated steps and eval passes grow no native
    storage; and the checkpoint lives in a temporary directory removed
    automatically.
  - **No production behavior changed.** The example composed only existing
    modules, loss, optimizer, and checkpoint APIs; no locked-contract bug
    was found, so no production file was touched. **F7 (the honest
    benchmark characterization) was next.**

### F7 — Native normalization benchmark characterization — **complete**

- **Objective:** characterize the normalization stack honestly under the
  §13 rules. Measurement only.
- **Layer / scope:** one benchmark harness plus its smoke test.
- **Expected files:**
  `benchmarks/benchmark_native_normalization.py`;
  `tests/test_native_normalization_benchmark.py`; `README.md`, the
  support matrix, and this document.
- **Required tests:** the smoke mode runs and exits zero; every case is
  correctness-gated before timing; a deliberately broken case fails the
  gate and publishes no timing; the JSON mode is well-formed; no result
  file is written; **no test asserts a duration**.
- **Validation:** `--smoke` in CI-equivalent time; the full Python suite.
- **Risks:** publishing a timing number as a promise; tuning the
  implementation to the benchmark; adding a CI timing threshold.
- **Dependencies:** F6.
- **Explicit non-goals:** optimization; any speed assertion; any
  committed timing number.
- **Completion criteria:** the harness exists, gates correctness first,
  reports medians with spread and honest reference labels, and asserts no
  speed anywhere.
- **Shipped (F7).** Exactly the above — **one benchmark harness and its
  test, measurement only: no capability, operation, kernel, C ABI symbol,
  ctypes declaration, `NativeTensorCore` method, custom backward,
  checkpoint schema field, example, or export; no inventory changed, the
  checkpoint format stays version 1, and no production file was
  modified.** What it delivered:

  - **`benchmarks/benchmark_native_normalization.py`** —
    `BENCHMARK_NAME = "tensorforge.native_normalization"`,
    `BENCHMARK_VERSION = "1.0"` — with exactly **nine** cases, in this
    order: `layernorm_forward`, `layernorm_backward`,
    `batchnorm1d_training_forward`, `batchnorm1d_eval_forward`,
    `batchnorm1d_backward`, `batchnorm2d_training_forward`,
    `batchnorm2d_eval_forward`, `batchnorm2d_backward`, and
    `normalized_training_step`. Nothing else is measured: no checkpoint
    I/O, no `state_dict()`/`load_state_dict()`, no constructor
    validation, no failure path, no `retain_graph`, no fault injection,
    and no isolated running-state transaction — the training-forward
    cases already include the real running-statistics update.
  - **Correctness before timing, structurally.** `_measure_case` calls
    the case's `check()` **before** it reaches the timing helper, so a
    failed gate raises before a single sample is taken and publishes no
    timing; the CLI turns that into `correctness gate failed: …` on
    stderr with a nonzero exit and a completely clean stdout. The tests
    prove it by replacing a native path with a finite, correctly shaped,
    but numerically wrong result (and separately with a non-finite one)
    and asserting that the timing helper was never called.
  - **Honest reference labels.** `stable_tensorforge` for the six cases
    with a real stable counterpart — the LayerNorm forward and backward,
    the BatchNorm1d training forward, evaluation forward, and backward,
    and the normalized training step — each running
    `tensorforge.nn`/`tensorforge.optim` on the *same* input values,
    epsilon, momentum, affine values, running state, initial parameters,
    optimizer hyperparameters, and reduction semantics.
    **`native_only`** for all three `NativeBatchNorm2d` cases, because
    the stable line has **no public `BatchNorm2d`**; they publish no
    native-to-stable ratio at all. Their correctness gates are still
    real: an explicit NumPy NCHW population-statistics formula for the
    output and both running buffers, a channelwise-affine probe (smoke
    mode uses unequal `C`/`H`/`W` so a channel/spatial broadcast mistake
    cannot hide), state neutrality in eval mode, and — for the backward —
    the stable `BatchNorm1d` applied to the equivalent `(N*H*W, C)`
    sample matrix with the input gradient transformed back to NCHW and
    the `gamma`/`beta` gradients compared directly. That transformed
    computation is a **correctness oracle only**: timing it as a
    "BatchNorm2d reference" would compare a different module plus two
    layout transformations, so the ratio would be misleading, and both
    `reference_detail` and the case's notes say so explicitly.
  - **Timing methodology.** `time.perf_counter_ns()`, warm-up before
    measurement, one measured sample per operation call, **every** sample
    retained, no fastest-only reporting, and no timer-overhead
    subtraction. `prepare()` and `cleanup()` run outside the timed
    region on every path; graph construction is inside the timer for the
    forward and training-step cases (it is part of the call) and outside
    it for the backward-only cases, which time exactly one one-shot
    `backward()` on a graph rebuilt from cleared gradients each
    repetition. Because a BatchNorm training forward advances persistent
    state, every training-mode repetition builds a **fresh** module from
    the same deterministic state — a state-advanced module is never
    reused as a sample. Reported per timed path: `sample_count`,
    `samples_s`, `median_s`, `min_s`, `max_s`, `spread_s`,
    `relative_spread`, and `units = "seconds_per_call"`.
  - **A JSON-native payload** (`benchmark`, `version`, `mode`,
    `environment`, `cases`) that survives
    `json.loads(json.dumps(payload)) == payload`, with `--case`,
    `--warmup`, `--repetitions`, `--smoke`, and `--json`; unknown cases
    and non-positive/`bool`/non-`int` counts rejected; **no file of any
    kind written**, and a human report that ends in the local
    characterization disclaimer and carries no speed verdict.
  - **No speed is asserted and there is no CI timing threshold.** The
    only float constants the harness defines are correctness tolerances
    (`FORWARD_ATOL`, `GRADIENT_ATOL`, `STATE_ATOL`, `LOSS_ATOL`,
    `PARAMETER_ATOL`) and module arguments; no test compares a measured
    duration or ratio against a nonzero constant; no timing number is
    committed to any document; and CI runs no benchmark at all.
  - **No production behavior changed.** The harness only composes shipped
    public APIs; no locked-contract bug was found, so no `src/` numerical
    file was touched. **F8 (the cross-cutting Phase-F integration) was
    next.**

### F8 — Cross-cutting Phase-F integration and semantic guardrails — **complete**

- **Objective:** prove the normalization stack composes correctly with
  everything the earlier phases built, and lock the phase's invariants
  with durable semantic guardrails.
- **Layer / scope:** one cross-cutting integration test file plus
  guardrail updates.
- **Expected files:** `tests/test_native_phase_f.py`;
  `tests/test_docs.py`; `tests/test_cpp_backend_info.py`.
- **Required tests:** a full model combining convolution, pooling,
  flatten, linear, both normalization families, the classification loss,
  and `NativeAdam`, trained and resumed; inventory self-consistency
  across every registry, export, and document; stable/native separation
  with no implicit dispatch; shared and frozen parameters through a
  normalized model; the versioning archetypes meeting a normalized graph;
  failure atomicity at every boundary; storage baselines; the capability
  boundary (dropout, float32, CUDA, AMP still unsupported).
- **Validation:** the full Python suite.
- **Risks:** guardrails that freeze prose instead of meaning; tests that
  restate F3–F5 instead of testing the *interactions*.
- **Dependencies:** F7.
- **Explicit non-goals:** any capability, kernel, or schema change.
- **Completion criteria:** the integration file covers the interactions
  no single-module test can, and the guardrails derive their checks from
  real exports, registries, and files.
- **Shipped (F8).** Exactly the above — **one cross-cutting integration
  suite (`tests/test_native_phase_f.py`) plus guardrail updates to
  `tests/test_docs.py` and `tests/test_cpp_backend_info.py`: tests and
  documentation only, no capability, operation, kernel, C ABI symbol,
  ctypes declaration, `NativeTensorCore` method, custom backward,
  checkpoint schema field, example, benchmark, or export; no inventory
  changed, the checkpoint format stays version 1, and no production file
  was modified.** What it delivered:

  - **One integrated model, `NativePhaseFClassifier`** (test-only, not a
    production class): `NativeConv2d(1, 4, 3)` → `NativeBatchNorm2d(4)` →
    `NativeReLU` → `NativeMaxPool2d(2)` → `NativeFlatten` →
    `NativeLinear(16, 8)` → `NativeBatchNorm1d(8)` → `NativeReLU` →
    `NativeLayerNorm(8)` → `NativeLinear(8, 3)` → **raw logits** →
    `NativeCrossEntropyLoss`, over the E8 fixed twelve-image three-class
    dataset. Every Phase-D module family, **both** BatchNorm shapes, and
    LayerNorm participate in one graph; no softmax or log-softmax module
    is inserted; `NativeAdam` sees the twelve parameters and never the
    four buffers.
  - **The full forward/backward/optimizer interaction.** One graph
    reaches every trainable parameter with a finite, correctly shaped
    gradient; the buffers receive none; both BatchNorm pairs advance
    together during the training forward; parameter versions and
    optimizer step counters each advance exactly once; parameter and
    buffer identities never move; and the one-shot backward releases the
    MaxPool2d winners and the cross-entropy probabilities exactly once.
  - **Deterministic integrated training and exact resume.** Twelve
    `NativeAdam(lr=0.05)` steps over the fixed batch, interrupted at step
    5, checkpointed (model **and** optimizer, format **version 1**),
    reloaded into a **completely fresh** model/optimizer pair, and
    continued. The prefix, the remaining loss suffix, the whole loss
    history, every parameter, the complete NativeAdam state, **all four**
    running-statistic buffers, the final training logits, and the final
    evaluation-mode logits, predictions, and accuracy all match by
    **exact equality**. The fresh target is deliberately put in eval mode
    before loading and stays there afterwards, proving the training flag
    is runtime-only; parameter and buffer identities survive the load.
  - **Three saved-resource families in one eval graph.** BatchNorm
    snapshots (`(1, 4, 1, 1)` and `(1, 8)`), MaxPool2d winners, and
    cross-entropy probabilities coexist; neither registered running
    buffer — object **or** storage — is reachable from the graph, while
    `gamma`/`beta` legitimately are; one backward releases all three
    families exactly once, a second release is a no-op, the registered
    buffers stay open and unchanged, and an abandoned eval graph releases
    its snapshots without touching registered state.
  - **Buffer mutation versus parameter mutation, attributed correctly.**
    A buffer-only `load_native_checkpoint()` over a parameter-free holder
    aliasing all four registered buffer objects — and, separately, a full
    training step — leave an earlier eval graph's gradients exactly equal
    to a clean control, with every buffer identity preserved and no
    parameter version moved. A **full** checkpoint load and a direct
    `copy_value_` on a normalization affine parameter each stale the
    graph through the unchanged v3.7 **parameter** rule, commit no
    partial gradient, and leave a fresh forward working.
  - **The versioning archetypes meeting a normalized graph.** One graph
    combining the integrated classification loss with an `exp` branch and
    a `log` branch: mutating all four running buffers and the `exp`
    parameter after the forward leaves every saved-state edge valid and
    reproduces a clean control exactly, while mutating the live-reread
    `log` parameter invalidates the **whole** graph before any branch
    commits a gradient.
  - **Shared and frozen parameters through normalization.** A parameter
    registered under two paths and used twice in one forward is exposed
    once under its first-discovered canonical name, deduplicated by
    identity in `parameters()`, given one NativeAdam state slot,
    accumulates both uses into one gradient (checked against an
    independent two-parameter control), updates once with exactly one
    version increment, survives a checkpoint round trip with both aliases
    pointing at the same object, and closes exactly once. A registered
    `requires_grad=False` parameter that really participates in the
    forward stays discoverable, may sit in the optimizer's list, builds
    no gradient, is skipped with its value, version, step counter, and
    Adam moments untouched, still persists numerically, and reloads still
    frozen — while the normalization buffers update normally.
  - **A non-contiguous NCHW input through the whole stack**, in both
    train and eval mode, matching the contiguous form for the logits, the
    loss, every trainable gradient, and all four running statistics,
    producing fresh owning contiguous logits and leaving the caller's
    base and view untouched.
  - **Strict stable/native separation through normalization**, and a
    representative stable `LayerNorm`/`BatchNorm1d` train-and-eval path
    proved unchanged.
  - **Failure boundaries, tested honestly and never over-claimed.**
    **A** — a BatchNorm running-state transaction failure rolls *that
    pair* back completely, while an earlier module's already-committed
    transaction legitimately stands: transactions are **per module**, and
    one whole training step is *not* globally transactional. **B** — a
    loss or backward failure after a successful forward does **not**
    retroactively roll back the running updates the forward committed,
    and commits no gradient or optimizer change. **C** — an optimizer
    staging failure commits nothing, closes every staged temporary, and
    leaves the gradients usable for a clean retry. **D** — a
    stale-parameter backward keeps the forward's committed buffer update,
    commits nothing, and releases its saved resources on explicit close.
    **E** — a commit failure while loading a real integrated checkpoint
    (twelve parameters, four buffers, and the NativeAdam state) restores
    every value, identity, and version and leaks no staged storage.
  - **Error-state recovery, the NumPy boundary, and ownership.** Handled
    Python and native failures leave `tf_last_error_code() == TF_OK` and
    the next normalized operation succeeds; one complete integrated step
    reaches no NumPy numerical routine and no tensor-data conversion path
    (`native_accuracy` stays deliberately outside); and repeated success
    **and** failure cycles, an exact resume cycle, aliases, frozen
    parameters, and non-contiguous inputs all return the native
    live-storage counters to their baseline.
  - **Semantic guardrails derived from reality**, not from prose: every
    `NATIVE_MODULES` entry resolves to a real exported callable class,
    `_NativeBatchNorm` stays private, no normalization name appears in
    any kernel/operation registry or in `_CHECKED_KERNELS`, no
    `NativeTensor`/`NativeTensorCore` normalization method exists, every
    `STATE_SUPPORT` name maps to a real API, `UNSUPPORTED` still reads
    exactly `("dropout", "float32", "cuda", "amp")`, the shipped Phase-F
    artifacts all exist, and no Phase-F C++ source, header, or CTest was
    added.
  - **No production behavior changed.** The suite composes shipped public
    APIs only; no locked-contract bug was found, so no `src/` numerical
    file was touched. **F9 (the phase closure) was next**, and it has
    since shipped.

### F9 — Phase-F closure — **complete**

- **Objective:** close the phase — validation, documentation
  reconciliation, and the completion statement. **No new numerical
  capability of any kind.**
- **Layer / scope:** builds, sanitizers, documentation.
- **Expected files:** this document (§15 status column, §17, §18);
  `docs/native_support_matrix.md`, `docs/roadmap.md`,
  `docs/project_summary.md`, `docs/architecture.md`,
  `docs/release_history.md`, `docs/backend_experiments.md`, `README.md`,
  `CLAUDE.md`, `src/tensorforge/experimental/__init__.py` (docstring).
- **Required tests:** the complete Python regression suite; Release
  **and** Debug native builds with the full existing CTest suite (no new
  CTest — Phase F adds no C++); Clang ASan/UBSan over a normalized
  training workload with zero diagnostics attributable to TensorForge; a
  practical LeakSanitizer pass with the live-storage counters returning
  to baseline.
- **Validation:** all of the above, reported with actual observed output.
- **Risks:** closing the phase while a status surface still describes it
  as planned; claiming a later phase; letting closure smuggle in a
  capability.
- **Dependencies:** F8.
- **Explicit non-goals:** any numerical capability; any statement about
  a later phase.
- **Completion criteria:** every §17 criterion is met and checked against
  reality, every status surface agrees, and §18 records the closure.

**Shipped (F9).** Exactly the above, as **documentation and
documentation-guardrail tests only** — no C++ source, header, CTest, C
ABI export, ctypes declaration, `NativeTensorCore` method, kernel,
module, operation, loss, metric, optimizer, example, benchmark,
checkpoint schema field, or export changed, and **no numerical
production file changed at all**. Every number below was observed during
this closure; none is carried over from Phase D or Phase E.

- **Windows environment.** Windows 11 Home 10.0.26200 (build 26200.8894),
  PowerShell 5.1.26100.8894 (Desktop), x64 (AMD64, Intel Core Ultra 9
  185H), Python 3.13.14, uv 0.11.26, CMake 4.4.0, generator **Visual
  Studio 17 2022**, MSVC **19.44.35228.0** (toolset 14.44.35207), Windows
  SDK 10.0.26100.0.
- **Release build and CTests.** Configured fresh and out-of-source
  (outside the repository) with `-DTF_BUILD_TESTS=ON` and `TF_OUTPUT_DIR`
  pointing at `src/tensorforge/backends`, then built `--config Release`:
  **zero compiler, zero linker, and zero CMake warnings**, and **10/10
  CTests passed** (0.78 s). The build log's only warnings are 13
  identical MSBuild `MSB8029` notices stating that the build directory
  must not sit under the temporary directory — an artifact of *where this
  validation put its build tree*, not a project diagnostic. The rebuilt
  Release DLL loads and passes `scripts/smoke_cpp_backend.py`.
- **Debug build and CTests.** A second fresh out-of-source configuration
  writing its library to a separate external directory, built `--config
  Debug`: **zero compiler, zero linker, and zero CMake warnings** (the
  same 13 locational `MSB8029` notices), and **10/10 CTests passed**
  (0.97 s). Assertions are genuinely enabled — the Debug configuration
  defines `_DEBUG`, never `NDEBUG`, and compiles `/Od /RTC1`. The Debug
  library never reached the package: the active
  `src/tensorforge/backends/_tensorforge_cpp.dll` stayed the 56,320-byte
  Release build linking `MSVCP140.dll`/`VCRUNTIME140.dll`, while the
  Debug library is a separate 172,032-byte file linking
  `MSVCP140D.dll`/`ucrtbased.dll`.
- **Windows Python regression.** `uv run pytest -q` with the Release
  backend active: **3,628 passed, 5 skipped** before the closure edits
  (40.62 s) and again after the build validation (29.64 s). All five
  skips are the pre-existing "backend is built; the unavailable path
  cannot be forced" cases — no test skipped because the backend was
  missing.
- **WSL/Linux environment.** WSL 2.6.1.0, Ubuntu 24.04.4 LTS, kernel
  6.6.87.2-microsoft-standard-WSL2, x86_64; CMake 3.28.3; Clang and
  clang++ **18.1.3**; `llvm-symbolizer-18`; GNU nm (binutils) 2.42;
  Python 3.12.3 with NumPy 2.5.1 and pytest 9.1.1 in an environment
  **outside** the repository (no repository `.venv` was created or
  replaced, and no dependency manifest or lockfile was touched).
- **Sanitizer build.** A fresh `/tmp` build directory configured
  `-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++
  -DTF_SANITIZE=address,undefined -DTF_BUILD_TESTS=ON`, which reported
  `TensorForge backend: sanitizers enabled -> address,undefined` and
  built with **zero project warnings**.
- **Instrumentation proved, not assumed.** `nm -D` on the produced
  library shows **22 `__asan*` symbols** and **13 `__ubsan*` symbols**
  (`__asan_init`, `__asan_report_load*`, `__ubsan_handle_add_overflow`,
  `__ubsan_handle_divrem_overflow`, …) alongside the **50** exported
  `tf_*` C ABI symbols. `file`/`readelf` confirm an ELF64 x86-64 shared
  object produced by "Ubuntu clang version 18.1.3". Independently,
  loading the library **without** the ASan runtime fails with `undefined
  symbol: __ubsan_vptr_type_cache`, and loading it **with** the runtime
  preloaded succeeds and reports `available=True`, `float64`, `cpu`.
- **Sanitized native CTests.** With
  `ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:detect_stack_use_after_return=1:detect_leaks=1`
  and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`: **10/10 CTests
  passed** (0.09 s) with **leak detection on**, no sanitizer diagnostic,
  no leak, no suppression, and no recovery mode.
- **Sanitized Python.** Because CPython itself is not instrumented, the
  Clang ASan runtime is preloaded (`LD_PRELOAD`, resolved from
  `clang++ -print-file-name=libclang_rt.asan-x86_64.so`) with
  `ASAN_SYMBOLIZER_PATH` set. Thirty-two test files — the whole
  normalization surface (LayerNorm, both BatchNorm shapes, the
  state/checkpoint hardening, the training proof, the benchmark, and the
  Phase-F integration suite) plus the ownership, autograd, buffer, state,
  parameter-versioning, storage, view, module, optimizer, checkpoint,
  backend-introspection, and Phase-E dependencies they exercise — ran
  through the sanitized library: **1,968 passed**, **zero ASan
  diagnostics, zero UBSan diagnostics**, and no backend-unavailable skip.
  `examples/native_normalization_training.py` then ran under the same
  library and reproduced its exact resume (loss suffix, parameters,
  optimizer state, both running statistics, the final training
  prediction, and the final evaluation-mode output all matching), and
  `benchmarks/benchmark_native_normalization.py --smoke` passed **all
  nine correctness gates** and wrote no result file — both with zero
  diagnostics.
- **LeakSanitizer, scope stated honestly.** A temporary (never
  committed) workload drove one complete normalized lifecycle: the
  integrated `Conv2d → BatchNorm2d → ReLU → MaxPool2d → Flatten → Linear
  → BatchNorm1d → ReLU → LayerNorm → Linear` classifier with
  `NativeCrossEntropyLoss` and `NativeAdam`, six training steps, a
  reporting eval pass with `native_accuracy`, a version-1 checkpoint, a
  **fresh** model/optimizer pair loading it, a resumed step matching the
  uninterrupted continuation exactly, a non-contiguous NCHW input through
  the whole stack, an eval graph carrying normalization snapshots
  retained across one backward and released by the next, a second
  reporting eval pass, and explicit closure of the optimizers, every
  unique parameter, every unique buffer, the views, and the base tensors.
  With `detect_leaks=1` and symbolization on, **the native live-storage
  counter returned exactly to its baseline (0 → 0)** before exit. The
  process-exit LSan report does list **925,710 bytes in 830
  allocations** — and **not one frame is attributable to TensorForge**:
  no frame names `_tensorforge_cpp`, `tf_core_`, `tf_storage_`, `tf::`,
  or any TensorForge C++ source path. The only binaries named are CPython
  (6,548 frames), the ASan runtime itself (293), libc (68), NumPy (24),
  and `_ctypes` (8) — interpreter and module-initialization allocations a
  non-instrumented CPython never frees at shutdown. **No suppression file
  was added** and `LSAN_OPTIONS` was left unset, so nothing was hidden.
  The project's leak contract remains the deterministic live-storage
  counters and the explicit-cleanup tests, which assert an *exact* return
  to baseline.
- **Cleanup.** The Linux `.so` was removed from
  `src/tensorforge/backends/` by a shell trap that fires however the
  validation exits; the Windows Release DLL remains in place and active.
  No `.so`, `.json`, `.csv`, `.npz`, sanitizer log, core dump, or build
  directory was left in the repository, and no source-tree build
  directory was created — both native builds and the sanitizer build used
  directories outside the repository.
- **Files changed by F9.** Documentation and documentation guardrails
  only: this document, `docs/native_support_matrix.md`,
  `docs/roadmap.md`, `docs/release_history.md`,
  `docs/backend_experiments.md`, `docs/project_summary.md`,
  `docs/architecture.md`, `README.md`, `CLAUDE.md`,
  `src/tensorforge/experimental/__init__.py` (module docstring status
  wording only), and `tests/test_docs.py`. **No numerical production
  file changed**, no capability registry changed, and no locked-contract
  defect was found.

---

## 16. Why this order

- **The state machinery first** (F1). The running-statistics transaction
  is the phase's riskiest mechanism and it already exists, in
  `load_state_dict`, in tested form. Extracting it before any module
  needs it means F3 inherits proven semantics instead of inventing a
  second implementation of them.
- **The stateless normalization second** (F2). LayerNorm exercises the
  composed mathematics — the mean, the population variance, the
  `sqrt(var + eps)` ordering, the affine application, the composed
  backward — with **no** buffers, so a numerical bug cannot hide behind a
  state bug.
- **The stateful 1-D case third** (F3). It adds exactly one new thing to
  a working composition: mutable state, with its snapshot rule and its
  transaction. It is where §7 and §8 are proved.
- **The 4-D case fourth** (F4). By then only the reduction axes and the
  broadcast shape are new, and the shared implementation makes that
  explicit. `"batchnorm"` leaves `UNSUPPORTED` only here, when the
  unqualified name is finally true.
- **Hardening before proving** (F5 before F6). The state and safety
  contracts are proved in isolation before an end-to-end run depends on
  them, so a failed resume points at the resume rather than at an
  unproven invariant.
- **Proof before measurement** (F6 before F7). Nothing is timed until it
  is correct, and the benchmark's correctness gates inherit a stack that
  is already proved.
- **Behavior-free closure last** (F8, F9). The phase ends with
  integration, hardening, validation, and documentation — never with new
  capability.

---

## 17. Phase-F completion criteria — all met

Phase F was to be complete only when **all** of the following held. Each
was checked against reality at the F9 closure checkpoint — against the
live registry, the live exports, an executable test, or an observed
validation run, never against prose.

1. ✅ `NativeLayerNorm`, `NativeBatchNorm1d`, and `NativeBatchNorm2d` are
   exported from `tensorforge.experimental`, tested by their own suites,
   and registered in `NATIVE_MODULES` (F2–F4).
2. ✅ `"layernorm"` and `"batchnorm"` have left `UNSUPPORTED`, which now
   reads exactly `("dropout", "float32", "cuda", "amp")`.
3. ✅ No normalization operation appears in `TENSOR_CORE_OPS`,
   `AUTOGRAD_OPS`, or `RAW_KERNELS`; no `NativeTensorCore` normalization
   method, ctypes declaration, or C ABI symbol exists; and no C++ source
   mentions `batch_norm` or `layer_norm`. The composition rule of §3 held
   for the whole phase, and `tests/test_docs.py` re-derives this from the
   live registry and the real `cpp/src/*.cpp` files.
4. ✅ Training-mode gradients differentiate through the batch mean and
   variance, verified by central finite differences for the input and
   every affine parameter, for both BatchNorm shapes
   (`tests/test_native_batchnorm1d.py`,
   `tests/test_native_batchnorm2d.py`).
5. ✅ The §7 mutable-buffer graph-safety rule is proved by tests: an eval
   graph never holds a registered buffer object *or its storage*, and a
   training step, a buffer-only `load_state_dict()`, and a real
   buffer-only `load_native_checkpoint()` each leave an earlier eval
   graph's gradients equal to a clean control
   (`tests/test_native_normalization_state.py`,
   `tests/test_native_phase_f.py`). A load that also replaces
   `gamma`/`beta` staling the graph through the unchanged v3.7 parameter
   rule is proved to be the *parameter* contract working, not a buffer
   failure.
6. ✅ The §8 transaction is proved: all-or-nothing updates, preserved
   buffer identity, exactly-once closes, unmoved parameter versions, and
   no partial state after an injected staging or commit failure — with F8
   additionally recording, honestly, that transactions are **per module**
   and that one whole training step is *not* globally transactional.
7. ✅ Persistent running statistics ride `state_dict()`,
   `load_state_dict()`, and the pickle-free checkpoint path with
   **format version 1 unchanged** (`native_checkpoint._FORMAT_VERSION ==
   1`), preserving buffer identity across every load.
8. ✅ A deterministic normalized training run resumes **exactly** —
   including both running means, both running variances, and the
   evaluation-mode output (F6), and again across the integrated
   convolutional classifier with all four running buffers (F8).
9. ✅ Benchmarks characterize the stack with no performance assertion, no
   committed timing number, no result file, and no CI timing threshold
   (F7).
10. ✅ Release **and** Debug builds each passed the full existing 10-test
    CTest suite with zero project warnings, and Clang ASan/UBSan and
    LeakSanitizer found nothing attributable to TensorForge over a
    normalized workload — 10/10 sanitized CTests with leak detection on,
    1,968 sanitized Python tests, the training example, the benchmark
    smoke path, and a practical lifecycle returning native live storage
    exactly to baseline (F9; the full record is in §15's F9 entry).
11. ✅ Every status surface — support matrix, roadmap, README, project
    summary, architecture, release history, the design documents, the
    experimental package docstring, and the backend registry — agrees on
    what shipped, and `tests/test_docs.py` derives that agreement
    semantically rather than by frozen prose.
12. ✅ Phase A–E behavior, the stable framework, the checkpoint schema,
    and the strict stable/native separation are all unchanged: the
    complete Python regression suite passes, and F9 changed no numerical
    production file.

---

## 18. Phase-F completion statement

**Phase F is complete: F0, F1, F2, F3, F4, F5, F6, F7, F8, and F9 have
all shipped.** The experimental native line now has the complete
numerical normalization *module* surface — LayerNorm and both BatchNorm
shapes — their state, checkpoint, ownership, and graph-safety contracts
are proved by executable test, a deterministic normalized training run
resumes exactly from a checkpoint, the stack is characterized by an
honest correctness-gated benchmark, the cross-cutting interactions are
proved by an integration suite, and the phase has been validated and
closed:

- `NativeLayerNorm` exists, is exported from `tensorforge.experimental`,
  and is registered in `NATIVE_MODULES`; `"layernorm"` has left
  `UNSUPPORTED`.
- `NativeBatchNorm1d` exists, is exported, and is registered in
  `NATIVE_MODULES`. It is the first **stateful** native numerical
  module: differentiable training statistics, persistent native
  `running_mean`/`running_var` buffers advanced by a graph-free atomic
  two-buffer transaction, and evaluation from graph-safe immutable
  snapshots.
- `NativeBatchNorm2d` exists, is exported, and is registered in
  `NATIVE_MODULES`. It is NCHW `(N, C, H, W)` batch normalization
  reducing over N, H, and W, built on the **same** shared private
  implementation as `NativeBatchNorm1d` — it declares only its rank, its
  reduction axes, its `(1, C, 1, 1)` broadcast layout, and the
  channels-last permutation its rank-1 affine parameters need.
- With both shapes live, `"batchnorm"` has left `UNSUPPORTED`, which now
  reads exactly `("dropout", "float32", "cuda", "amp")`.
- **F5 is complete.** The exhaustive state/checkpoint, ownership, and
  graph-safety hardening — a focused `tests/test_native_normalization_state.py`
  plus narrow additions to the generic buffer and checkpoint suites —
  proves §7–§10 by executable test rather than by prose: canonical dotted
  buffer keys, independent state snapshots, strict/non-strict loads, exact
  never-casting metadata validation, mixed parameter/buffer transaction
  atomicity, buffer identity across state and checkpoint loads, exact
  eval-output reproduction, the buffer-only-versus-full stale-graph
  distinction, the save/corrupt-load failure boundaries, eval-graph
  snapshot safety under `retain_graph` and a failed retryable backward,
  and explicit parameter/buffer closure returning storage to baseline.
  **Tests and documentation only — no numerical behavior, no new public
  capability, no checkpoint schema change; the checkpoint format stays
  version 1.**
- **F6 is complete.** `examples/native_normalization_training.py` trains
  `NativeNormalizedRegressor` (`Linear → BatchNorm1d → ReLU → LayerNorm →
  Linear`, both normalization families in every forward, BatchNorm the
  only stateful module) for 24 deterministic `NativeAdam` steps with
  `NativeMSELoss` (98.9% loss reduction), proves two uninterrupted runs
  bit-identical, and resumes an interrupted run into a **fresh**
  model/optimizer pair that reproduces the remaining loss suffix, every
  parameter, the NativeAdam state, both `running_mean` and `running_var`,
  the final training-step prediction, and the final **evaluation-mode**
  output exactly — with buffer/parameter identities preserved across the
  load, the training flag proved runtime-only, and the checkpoint format
  version **1**. **One example and its integration test — no capability,
  operation, kernel, schema field, or benchmark.**
- **F7 is complete.** `benchmarks/benchmark_native_normalization.py`
  characterizes the stack with nine correctness-gated cases — the
  LayerNorm forward and backward, the BatchNorm1d training forward,
  evaluation forward, and backward, the BatchNorm2d training forward,
  evaluation forward, and backward, and one complete F6-style normalized
  training step. Correctness runs before any timing and a failed gate
  publishes none; six cases are measured against `stable_tensorforge`
  equivalents on identical state, while the three BatchNorm2d cases are
  `native_only` for timing (there is no public stable `BatchNorm2d`) and
  keep a rigorous NumPy/transformed-oracle correctness gate instead;
  medians are reported with min, max, and spread after warm-up;
  `--smoke` and `--json` exist; and **no result file is written, no speed
  is asserted, no timing number is committed, and no CI job asserts a
  duration**. **Measurement only — one harness and its test, no
  capability, operation, kernel, C ABI symbol, schema field, example, or
  export, and no production behavior changed.**
- **F8 is complete.** `tests/test_native_phase_f.py` proves the
  interactions no single-module suite can: one integrated
  convolution/normalization/pooling/classification model trained by
  `NativeAdam` and resumed **exactly** (all four running buffers, the
  final training logits, and the evaluation-mode logits, predictions, and
  accuracy included); BatchNorm snapshots, MaxPool2d winners, and
  cross-entropy probabilities coexisting in one eval graph and releasing
  exactly once; buffer mutation leaving an earlier graph valid while
  parameter mutation correctly stales it; the Phase-E versioning
  archetypes meeting a normalized graph; shared and frozen parameters;
  a non-contiguous NCHW input through the whole stack; strict
  stable/native separation; each failure boundary tested honestly —
  including the explicit statement that BatchNorm transactions are **per
  module** and that one whole training step is *not* globally
  transactional; error-state recovery; the NumPy boundary; live-storage
  baselines across success and failure cycles; and semantic capability,
  export, and artifact guardrails derived from real registries and files.
  **Tests and documentation only — no capability, no operation, no
  kernel, no schema change, and no production behavior changed.**
- **F9 is complete.** The phase closure: fresh Windows Release and Debug
  builds each passing the full existing 10-test CTest suite with zero
  project compiler, linker, or CMake warnings and the active runtime
  proved to stay Release; a fresh Clang 18.1.3 ASan+UBSan build in WSL2
  Ubuntu 24.04 whose instrumentation is *proved* (22 `__asan*` and 13
  `__ubsan*` dynamic symbols, and a load that fails without the
  sanitizer runtime); 10/10 sanitized native CTests with leak detection
  enabled; 1,968 sanitized normalization-focused Python tests with zero
  ASan and zero UBSan diagnostics; the F6 example and the F7 benchmark
  smoke path clean under the sanitized library; and a practical
  LeakSanitizer lifecycle whose native live-storage counter returned
  **exactly** to baseline, with the remaining process-exit allocations
  identified honestly as CPython/NumPy shutdown retention containing no
  TensorForge frame and no suppression file. **Validation and
  documentation only — no numerical capability, no C++, no CTest, no ABI
  or ctypes surface, no example, no benchmark, and no production
  behavior changed.**
- **No normalization *operation* is differentiable, and none appears in
  any operation inventory** — `NativeLayerNorm`, `NativeBatchNorm1d`,
  and `NativeBatchNorm2d` are all composed from existing operations, so
  no normalization kernel, C ABI export, ctypes declaration, or
  `NativeTensorCore` method exists.

What F0 delivered is this contract and the repository reconciliation that
accompanied it — **no numerical behavior whatsoever**. What **F1**
delivered is the private atomic state transaction of §8
(`_native_state.py`), the `load_state_dict` refactor onto it, and the
`STATE_SUPPORT` persistent-buffer reconciliation — **state management and
capability reporting only**. What **F2** delivered is `NativeLayerNorm`
itself: the stateless, differentiable, composed-from-existing-operations
normalization module of §5, with `sqrt(var + eps)` ordering, population
variance, identical train/eval behavior, and no kernel, ABI symbol,
`NativeTensorCore` method, custom backward, or `NativeTensor`
normalization operation. What **F3** delivered is `NativeBatchNorm1d`:
the §6.1 contract, the §7 snapshot rule, and the §8 atomic two-buffer
transaction, all live in one shared private implementation, with the
checkpoint format unchanged at version 1. What **F4** delivered is
`NativeBatchNorm2d`: the §6.2 NCHW contract over that same
implementation, plus the one shared affine-layout refinement the 4-D
shape needed — and the removal of `"batchnorm"` from `UNSUPPORTED`. What
**F5** delivered is the exhaustive state/checkpoint/ownership/graph-safety
hardening test surface for §7–§10, and the documentation reconciliation
that came with it — **tests and documentation only, no numerical behavior
and no new capability, with the exports, every capability registry, and
the version-1 checkpoint format all exactly what F4 left.** What **F6**
delivered is `examples/native_normalization_training.py` and its
integration test: a deterministic normalized training run whose two
uninterrupted repetitions are bit-identical and whose interrupted
checkpoint resume into a fresh model/optimizer pair is exact — the
running statistics, the NativeAdam state, the final training-step
prediction, and the evaluation-mode output all reproduced — **one example
and one test, no capability and no schema change.** What **F7** delivered
is `benchmarks/benchmark_native_normalization.py` and its test: the
honest §13 characterization of the whole normalization stack — nine
cases, correctness gated before every measurement, honest
`stable_tensorforge`/`native_only` reference labels, medians with min,
max, and spread, `--smoke`/`--json` modes, no result file, and no speed
assertion, committed timing number, or CI timing threshold anywhere —
**measurement only, one harness and one test, no capability and no
production change.** What **F8** delivered is
`tests/test_native_phase_f.py` and the guardrail updates that came with
it: the cross-cutting proof that the normalization stack composes
correctly with everything Phases A–E built — one integrated classifier
trained and resumed exactly, three saved-resource families coexisting in
one eval graph, the buffer/parameter mutation distinction attributed to
the right cause, the versioning archetypes, shared and frozen parameters,
a non-contiguous NCHW input, honest per-boundary failure atomicity, and
capability/export/artifact guardrails derived from reality — **tests and
documentation only, no capability and no production change.** And what
**F9** delivered is the closure itself: the Release and Debug build
revalidation, the sanitizer and LeakSanitizer passes, the exact
live-storage baseline proof, the reconciliation of every authoritative
status surface, the durable semantic closure guardrails in
`tests/test_docs.py`, and this statement — **documentation and
documentation-guardrail tests only, with no numerical production file
changed and no new capability, operation, kernel, C ABI symbol, ctypes
declaration, CTest, example, benchmark, or export.**

**Phase F is finished; the native line is not.** Closing this phase is a
statement about Phase F only. TensorForge as a whole remains an
experimental, explicitly-scoped project: the native runtime is still
**float64 on CPU only**, still reached only through
`tensorforge.experimental` and `tensorforge.backends`, still free of any
implicit stable/native dispatch or conversion, and still not
production-ready and not a PyTorch replacement. `"dropout"`,
`"float32"`, `"cuda"`, and `"amp"` remain in `UNSUPPORTED`, and the
native kernels remain deliberately naive — no optimization work has been
done and no performance is promised anywhere.

Deliberately outside Phase F and still unplanned: dropout, a native RNG
and RNG checkpoint state, further activations (`tanh`, `sigmoid`, GELU),
more losses, schedulers, data loaders, native integer tensors, indexing
and index-producing reductions, further dtypes or devices, casting or
dtype promotion, CUDA, AMP, Tensor Core dispatch, CPU optimization, and
any implicit stable/native dispatch or conversion. Nothing in this
document should be read as a claim about any later phase.
