# Native classification architecture design (Phase E contract)

This is the **design-and-contract** document for the experimental native
C++ CPU line's classification stack — **Phase E — Native Classification
and Stable Math**. It is a milestone-zero (E0) deliverable: it locks the
architecture, public API surface, numerical-stability strategy, ownership
rules, C ABI shape, source organization, testing strategy, and milestone
sequence **before any numerical classification code is written**.

**E0 added no numerical behavior.** No kernel, C ABI symbol, ctypes
declaration, `NativeTensorCore` method, `NativeTensor` operation, loss
module, metric, benchmark, or example shipped with the first version of
this document. The backend capability registry
(`tensorforge.backends.cpp` — `TENSOR_CORE_OPS`, `AUTOGRAD_OPS`,
`NATIVE_MODULES`, `NATIVE_LOSSES`, and `UNSUPPORTED`), mirrored in
[native_support_matrix.md](native_support_matrix.md), stays the
**single source of truth** for what is actually live at any moment.

**Phase-E status: in progress.** **E0 is complete** (this contract),
**E1 is complete** (the differentiable `NativeTensor.exp()`), **E2 is
complete** (the differentiable `NativeTensor.log()`), **E3 is
complete** (the fused, stable `NativeTensor.softmax()`), **E4 is
complete** (the fused, stable `NativeTensor.log_softmax()`), **E5 is
complete** (the fused cross-entropy **Core contract** — forward and
backward), and **E6 is complete** (the differentiable
`NativeTensor.cross_entropy()`) — so `"exp"`, `"log"`, `"softmax"`, and
`"log_softmax"` now live in `TENSOR_CORE_OPS` and `AUTOGRAD_OPS`,
`"cross_entropy_forward"`/`"cross_entropy_backward"` live in
`TENSOR_CORE_OPS`, and `"cross_entropy"` lives in `AUTOGRAD_OPS`; all of
them have left `UNSUPPORTED`.
Shipping exp and log as a pair was deliberate: they are the phase's two
backward archetypes, and §5's matrix is now **proved by tests**, not
merely asserted — `exp` reads its saved output and records no version,
`log` rereads the live input and version-guards a direct parameter, and
`softmax` and `log_softmax` join `exp` on the saved-output side. E3 also
created the phase's C++ source unit, `cpp/src/classification.cpp`, which
E4 extended with its own fused log-sum-exp kernel — **never**
`softmax().log()` — and which E5 extended again with the fused
cross-entropy forward/backward pair.

**E5 shipped the Core layer; E6 shipped the graph node over it, and that
split is load-bearing.** E5 delivered the graph-unaware runtime layer:
the fused forward (scalar loss **and** private saved probabilities in one
pass), the saved-probability backward, both guarded C ABI exports, and
the strict copied-`int64` target contract of §6. **E6 added no numerical
capability at all** — no kernel, no ABI export, no change to any formula.
It added exactly one thing: `NativeTensor.cross_entropy(targets,
reduction="mean")`, a single autograd node that calls the E5 forward
once, adopts the private probabilities as a **graph-owned resource**
(§7), captures the copied targets and the normalized reduction in its
backward closure, and drives the E5 backward — never rereading the
logits, and therefore recording **no expected parameter version**.

**Everything else in Phase E (E7–E10) is still designed-only**:
`NativeCrossEntropyLoss` and `native_accuracy` do not exist in code and
remain in `UNSUPPORTED`, there is no `NATIVE_METRICS` inventory, and no
native classification training, benchmark, or sanitizer work has
started. Per-milestone status is recorded in the ladder (§15); the
completion criteria (§17) are **not** met, so Phase E is **not**
complete.

The stable Python framework (`tensorforge.Tensor.exp/log/softmax`,
`tensorforge.nn.cross_entropy`, `tensorforge.nn.accuracy`) is the
**numerical and public-behavior reference**. Where the native
architecture must differ (ownership, lifetime, the fused-primitive /
autograd split, the absence of an integer dtype), the difference is
stated and justified. No implementation code is copied from any other
framework.

Read alongside:
[native_tensor_wrapper_design.md](native_tensor_wrapper_design.md) (the
`NativeTensor` wrapper and ownership model),
[native_autograd_design.md](native_autograd_design.md) (the
Python-managed reverse-mode graph),
[native_cnn_design.md](native_cnn_design.md) (Phase D — the source of
the graph-owned saved-resource model this phase reuses),
[native_abi_error_contract.md](native_abi_error_contract.md) (the
exception-safe C ABI status contract), and
[backend_experiments.md](backend_experiments.md) (the whole native line).

---

## 0. Invariants Phase E must preserve

Phase E changes nothing about these existing guarantees:

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
  `allow_pickle=False`; buffers stay separate from parameters.
- Native checkpoints keep **format version 1**.
- Failed operations must **not partially mutate** caller-visible state.
- Every fallible native export uses the existing **thread-local status /
  `errcheck` contract**; **no C++ exception crosses the C ABI**.
- Numerical operations stay free of NumPy compute — the NumPy tripwire
  tests keep holding. The one deliberate exception is the reporting-only
  metric of §8, which is outside training and autograd.
- Existing Phase A–D numerical behavior is **unchanged**.

---

## 1. Phase-E scope

### In scope

| # | Deliverable |
|---|---|
| 1 | Differentiable `NativeTensor.exp()` |
| 2 | Differentiable `NativeTensor.log()` |
| 3 | Numerically stable `NativeTensor.softmax(axis=-1)` |
| 4 | Numerically stable `NativeTensor.log_softmax(axis=-1)` |
| 5 | Fused `NativeTensor.cross_entropy(targets, reduction="mean")` from raw logits |
| 6 | `NativeCrossEntropyLoss` module |
| 7 | Reporting-only `native_accuracy(logits, targets)` |
| 8 | Deterministic end-to-end native classification training |
| 9 | Exact model and optimizer checkpoint resume for that run |
| 10 | Honest classification benchmarks (characterization only) |
| 11 | Cross-cutting integration, ownership, sanitizer, and documentation closure |

### Explicitly excluded from Phase E

`tanh`; `sigmoid`; binary cross-entropy; `NLLLoss`; class weighting;
`ignore_index`; label smoothing; probability targets; soft-label targets;
`NativeSoftmax` or `NativeLogSoftmax` **modules**; `max`, `min`, `argmax`,
or product reductions; native integer tensors; normalization; BatchNorm;
LayerNorm; a native RNG; dropout; optimizer changes; scheduler changes;
checkpoint schema changes; CPU optimization; build or packaging redesign;
float32; CUDA; AMP; any stable-framework change; implicit dispatch;
implicit conversion; real datasets; and data loaders.

These stay in the support matrix's unsupported/future section. As in
Phase D, the first implementation of every kernel favors **correctness,
readability, sanitizer safety, and explicit indexing** over speed.

---

## 2. Locked public API surface

```
NativeTensor.exp()
NativeTensor.log()
NativeTensor.softmax(axis=-1)
NativeTensor.log_softmax(axis=-1)
NativeTensor.cross_entropy(targets, reduction="mean")

NativeCrossEntropyLoss(reduction="mean")          # a NativeModule
native_accuracy(logits, targets) -> float          # reporting only
```

Nothing else becomes public. In particular the fused cross-entropy's
**saved probabilities are private** (§7) and never reachable from any
public attribute, method, state dictionary, or checkpoint.

**Reductions.** `cross_entropy` and `NativeCrossEntropyLoss` support
exactly `"mean"` and `"sum"`, validated by exact string match (no case or
whitespace normalization, no coercion) — the `NativeMSELoss` precedent.
**`reduction="none"` is not supported in Phase E**: both supported
reductions produce a scalar, which is what `backward()`'s default seed and
the training loop need; an unreduced loss is a broader API question
deferred past this phase.

**Axis argument.** `softmax` and `log_softmax` take `axis=-1` by default,
accept any valid positive or negative axis, reject `bool` (the existing
`_normalize_axis` rule — a `bool` is never a legal axis even though it is
an `int` in Python), and normalize negatives against `ndim`. Rank ≥ 1 is
required; rank 0 (a scalar) has no axis to normalize and is rejected.

---

## 3. Numerical stability strategy

The whole phase exists because a naive `exp(x) / sum(exp(x))` overflows
for large logits and a naive `log(softmax(x))` loses precision for small
probabilities. Two standard transforms carry the phase, and both are
**mandatory, not optional optimizations**:

- **Maximum shift.** With `m = max_j x_j` along the axis,
  `softmax(x)_i = exp(x_i − m) / Σ_j exp(x_j − m)`. Every exponent is
  ≤ 0, so no term overflows, and at least one term equals 1, so the
  denominator is ≥ 1 and never underflows to zero.
- **Log-sum-exp.** `log_softmax(x)_i = (x_i − m) − log(Σ_j exp(x_j − m))`,
  computed in that order and **never** as `log(softmax(x))`.

`cross_entropy` uses the same log-sum-exp directly on raw logits:
`loss_n = −(x[n, t_n] − m_n) + log(Σ_j exp(x[n, j] − m_n))`.

The max shift is an internal computation, **not** a new public `max`
reduction: Phase E adds no `max`/`min`/`argmax` operation to any
inventory. The maximum is computed inside the fused kernels over the
axis being normalized.

Determinism: fixed row-major traversal order, no parallel reduction, no
SIMD/FMA/Kahan tricks. Floating-point parity with the stable framework is
therefore **to a tolerance** (`atol ≈ 1e-12` for test-sized tensors), not
bit-for-bit — the same honesty the reductions and CNN docs already state.

---

## 4. Operation contracts

Common to all five operations: **float64/CPU only**; inputs must be open
`NativeTensor`s (stable `Tensor`s, NumPy arrays, lists, scalars, and
closed tensors are rejected by type/state before anything is allocated);
every output is a **fresh owning row-major contiguous** tensor that
aliases no input; zero-size dimensions cannot exist (the runtime forbids
them), so "nonempty" is inherited, not re-checked.

### 4.1 `NativeTensor.exp()`

- **Shape support:** any shape already legal in the native runtime,
  including rank-0 scalars and arbitrary strided/offset views.
- **Execution paths:** both the **generic strided odometer** path and the
  **contiguous fast path**, exactly like `sqrt`/`reciprocal` — two guarded
  exports, selected by the Core wrapper on the operand's contiguity.
- **IEEE behavior:** unmodified `std::exp`. Overflow to `+inf`, underflow
  to `0`, `NaN` propagation, and `exp(-inf) == 0` are all inherited, not
  clamped.
- **Backward — reads the saved forward output.** `d/dx exp(x) = exp(x)`,
  so the node keeps its own forward result `y` and computes
  `dx = upstream * y`. **The input is never reread during backward.**
- **Versioning: no version snapshot.** Because backward reads the
  **saved output** and not a parent's current value, `exp` records
  **no parameter version**, exactly like `sqrt`/`reciprocal`. Mutating a
  direct `NativeParameter` after the forward pass therefore leaves this
  edge valid, and backward still produces the mathematically correct
  gradient for the forward that actually ran.
- **Closed saved output:** if the saved output tensor is closed before
  backward runs, backward must fail **clearly and atomically** — the
  existing closed-tensor `RuntimeError`, raised before any gradient is
  accumulated, never a silent wrong answer.

**Status (E1 — implemented as specified).** The op ships end to end:
`op_exp` in `cpp/src/elementwise.cpp` (plain `std::exp`, reusing the
existing `core_unary` odometer and `core_unary_contiguous` fast-path
walkers), the guarded exports `tf_core_exp` / `tf_core_exp_contiguous`,
their ctypes declarations and `_CHECKED_KERNELS` registration,
`NativeTensorCore.exp()` (dispatching on contiguity through the shared
`_unary_compute` helper), and the differentiable `NativeTensor.exp()`
whose backward is one native multiply — `upstream * saved output` —
recording **no** expected version. Two E1 refinements of the D0-era
sketch, both tightening rather than changing the contract:

- **The exports validate their own arguments.** `tf_core_exp` and
  `tf_core_exp_contiguous` check handles, layout metadata, spans, and
  overflow themselves (file-local `checked_mul`/`checked_add` helpers
  and a min/max reachable-index bound that also covers negative strides)
  before touching memory, rejecting a bad call with `TF_ERROR_INVALID`
  → `ValueError` and writing nothing. The older unary exports
  (`relu`/`sqrt`/`reciprocal`) predate this convention and were left
  exactly as they are — E1 tightened only what it added.
- **The saved output is the node's own core, not a separate graph
  resource.** `exp`'s derivative *is* its forward result, which the
  autograd node already owns, so no `graph_resources` entry is needed
  (contrast MaxPool2d's private winner buffer, and the saved
  probabilities cross-entropy adopts as a graph resource at E6). Lifetime
  therefore
  follows the ordinary node rules: retained under `retain_graph=True`,
  released with the graph, and a `close()`d intermediate makes backward
  raise before any gradient is committed.

**Not covered by E1:** a closed *parent* still fails backward, because
the parent is where the gradient is accumulated — that is the existing
ownership contract, not a value read. The distinction is proved in
`tests/test_native_exp.py`.

### 4.2 `NativeTensor.log()`

- **Shape support and execution paths:** identical to `exp` (generic
  strided + contiguous, both guarded exports).
- **IEEE behavior:** unmodified `std::log`, with **no clamping and no
  inserted epsilon**. `log(0) == -inf`, `log(negative) == NaN`,
  `log(+inf) == +inf`, `NaN` propagates. Silently clamping would hide
  real modelling bugs; the stable framework's stability lives in the fused
  losses, not in `log` itself, and native follows that.
- **Backward — rereads the live input.** `d/dx log(x) = 1/x`, and the
  saved output `y = log(x)` cannot recover `x` cheaply or exactly, so the
  backward **rereads the input's current value** and computes
  `dx = upstream * reciprocal(input)` through the existing native
  `reciprocal` (no new division operation; the `reciprocal` + `multiply`
  composition rule is unchanged).
- **Versioning: version-checked.** Because backward reads a parent's
  **live input** value, a direct `NativeParameter` parent **is
  version-guarded** through `_versioned_value_reads("log", (self,))`.
  Mutating that parameter after the forward pass must raise the
  deterministic **stale-graph** `RuntimeError` **before any gradient is
  computed or committed** — the existing pre-callback validation point.
  A non-parameter parent records nothing, as always.

This exp/log asymmetry is deliberate and is the phase's first teaching
point: *what a backward reads decides what it must version-guard.*

**Status (E2 — implemented as specified).** The op ships end to end:
`op_log` in `cpp/src/elementwise.cpp` (plain `std::log`, over the same
`core_unary` / `core_unary_contiguous` walkers), the guarded exports
`tf_core_log` / `tf_core_log_contiguous`, their ctypes declarations and
`_CHECKED_KERNELS` registration, `NativeTensorCore.log()`, and the
differentiable `NativeTensor.log()`. Implementation notes:

- **E1's validators were reused unchanged.** `unary_strided_error` and
  `unary_contiguous_error` were written op-agnostic in E1, so E2 needed
  **no generalization at all** — the log exports call them directly.
  `cpp/tests/test_log.cpp` re-drives the whole rejection matrix as
  regression coverage that sharing them weakened neither operation.
- **IEEE domain results are values, not ABI errors.** `log(0) == -inf`
  and `log(negative) == NaN` set IEEE flags, not C++ exceptions, so they
  flow to the destination and leave the thread-local error slot clear.
  This is asserted at both the C++ and Python layers.
- **The backward's temporary is `finally`-closed.** The closure computes
  `reciprocal(input)` into a transient owning core and closes it in a
  `finally`, so a failing `multiply` cannot leak it and a cleanup problem
  cannot mask the original exception — the pattern conv2d's bias
  reduction established (§D6).
- **Version guarding uses the existing mechanism only.** No new
  stale-check path exists inside the closure: `_versioned_value_reads`
  records the entry and `backward()`'s existing preflight validates it
  before the gradient snapshot, the seed, and every callback. A stale
  graph therefore commits nothing **anywhere** — proved with a mixed
  graph whose healthy branch also stays untouched — is not marked freed,
  and fails identically on retry.
- **The direct-parent rule was not broadened.** `p.exp().log()` after
  mutating `p` stays valid: `log` rereads its direct parent (exp's
  output, a plain tensor with no version slot), and `exp` reads its saved
  output, so no edge reads the mutated value. Version provenance does
  **not** propagate through intermediates in the current architecture,
  and E2 deliberately did not redesign that — the boundary is stated here
  and locked by a test.
- **No saved-output state and no graph resource.** `log`'s node owns
  nothing private (`_graph_resources == ()`); its local derivative is
  rebuilt from the live input each pass.

### 4.3 `NativeTensor.softmax(axis=-1)`

- **Forward:** the §3 maximum-shift form, as one **fused C++ kernel** —
  not a composition of `exp` and `sum` at the Python layer. Fusing keeps
  one pass over the axis, avoids three intermediate allocations, and
  keeps the stability transform inside the kernel where it cannot be
  bypassed.
- **Rank/axis:** any rank ≥ 1; any valid positive or negative axis; a
  `bool` axis is rejected. Output shape equals input shape exactly.
- **Layout:** **contiguous-only C ABI.** The kernel receives contiguous
  storage plus an offset and a three-way factorization of the shape
  around the axis — `(outer, axis_length, inner)` — which is all the
  index arithmetic a general-rank axis-wise pass needs. The Core wrapper
  applies the existing **Policy-B copy-then-compute** rule from Phase D:
  a non-contiguous operand is materialized with `contiguous_copy()` into
  private owning storage that the wrapper closes as soon as the kernel
  returns. The caller's view is never touched.
- **Backward — composed at the graph-unaware Core layer from the saved
  output.** With `y = softmax(x)`:
  `dx = y * (upstream − sum(upstream * y, axis, keepdims=True))`.
  This is expressed with existing Core ops (`multiply`, `sum` with
  `keepdims`, `subtract`) rather than a new backward kernel — the
  identity is exact, the pieces are already tested, and no new C ABI
  symbol is needed. The composition happens **below** the autograd layer
  (in the backward closure, over cores), so the graph never leaks into
  C++ and no intermediate node is created.
- **Backward reads the saved output `y`, never the input.**
- **Versioning: no version snapshot.** Same reasoning as `exp`.

**Status (E3 — implemented as specified).** The op ships end to end:
the internal `tf::softmax_forward_contiguous` kernel and the guarded
export `tf_core_softmax_forward` in the **new**
`cpp/src/classification.cpp` (declared in the new
`cpp/include/tf_classification_internal.h`), the ctypes declaration and
`_CHECKED_KERNELS` registration, `NativeTensorCore.softmax(axis=-1)`
with Policy-B copy-then-compute, and the differentiable
`NativeTensor.softmax(axis=-1)`. Implementation notes:

- **The forward is genuinely fused, and the ABI is contiguous-only.**
  One kernel does max → shifted exponentials → sum → normalize in three
  passes over each slice, writing the exponentials straight into the
  destination so **no second probability buffer is allocated in C++**.
  It is not composed from public max/subtract/exp/sum/divide operations —
  E3 adds no public `max`, `argmax`, or division. The export takes
  `(source handle, source offset, destination handle, outer,
  axis_length, inner)` and self-validates all of it; strides never cross
  the boundary.
- **The axis is validated by the existing `_normalize_axis`.** Bools,
  floats, strings, `None`, tuples, and out-of-range values are rejected;
  negatives normalize; every axis on a rank-0 tensor is out of bounds, so
  rank ≥ 1 falls out of the existing rule rather than a new check.
  Validation runs **before any allocation** (proved by a test that makes
  allocation itself fail).
- **Policy B follows the Phase-D helper exactly.** A non-contiguous input
  is materialized through `_contiguous_temp`, closed in a `finally`
  whether the call succeeds or raises; the output is closed if the kernel
  fails after allocation; the caller's input is never closed. Both the
  success and post-copy failure paths are asserted by instrumenting the
  copy helper.
- **The Policy-B copy is fully native (E3.1).**
  `NativeTensorCore.contiguous_copy` allocates its destination and then
  gathers the strided source **storage to storage** through the
  `tf_core_contiguous_copy` export, so **no tensor data round-trips
  through a NumPy host buffer** at any point of a non-contiguous
  softmax. Only the shape/stride arrays cross the boundary, as ctypes
  metadata. The strict NumPy tripwire therefore applies equally to the
  contiguous and non-contiguous paths, and to the backward. See §9.4.
- **The backward is composed at the Core layer, with no new kernel.**
  `weighted = g * y` → `slice_dot = weighted.sum(axis, keepdims=True)` →
  `centered = g - slice_dot` (broadcasting over the kept axis) →
  `contribution = y * centered`, each temporary closed in a `finally` as
  soon as its consumer has run. **No dedicated softmax backward kernel
  exists**, as E0 requires.
- **Saved-output semantics confirmed.** The node records no expected
  version and owns no private graph resource (`y` is the node's own
  core), so mutating a direct parameter after forward leaves the edge
  valid — proved with a *weighted* loss, since a plain
  `softmax(x).sum()` has a zero gradient and would hide an error.
- **Exceptional values are plain IEEE.** A NaN or `+inf` in a slice makes
  that whole slice NaN (`inf - inf`); `-inf` simply takes zero mass; an
  all-`-inf` slice is NaN. These are values, not ABI errors — the error
  slot stays `TF_OK`. Tests compare against a NumPy reference running the
  *same* maximum-shift order rather than another framework's
  special-casing.

### 4.4 `NativeTensor.log_softmax(axis=-1)`

- **Forward:** a **fused stable log-sum-exp** kernel. **Never implemented
  as `softmax().log()`** — that composition is exactly the precision loss
  this operation exists to avoid, and the design forbids it at every
  layer.
- **Rank/axis, shape, layout:** identical to `softmax`, including the
  contiguous-only C ABI with `(outer, axis_length, inner)` factorization
  and Core-level Policy-B handling.
- **Backward — from the saved output.** With `y = log_softmax(x)`:
  `dx = upstream − exp(y) * sum(upstream, axis, keepdims=True)`.
  The `exp(y)` here is the Phase-E `exp` Core op applied to the **saved
  output**, recovering the probabilities without rereading the input.
- **Backward reads the saved output, never the input.**
- **Versioning: no version snapshot.**

**Status (E4 — implemented as specified).** The op ships end to end:
the internal `tf::log_softmax_forward_contiguous` kernel and the guarded
export `tf_core_log_softmax_forward` in `cpp/src/classification.cpp`
(declared in `cpp/include/tf_classification_internal.h`), the ctypes
declaration and `_CHECKED_KERNELS` registration,
`NativeTensorCore.log_softmax(axis=-1)` with Policy-B copy-then-compute,
and the differentiable `NativeTensor.log_softmax(axis=-1)`.
Implementation notes:

- **The forward is one fused log-sum-exp kernel, never
  `softmax().log()`.** Per slice it computes the maximum, writes the
  shifted logits straight into the destination while accumulating
  `Σ exp(x − m)`, then subtracts `log` of that sum in place — three
  passes, **no probability buffer and no division anywhere**. It is not
  composed from public max/subtract/exp/sum/divide operations, and it
  does not call the softmax kernel. The distinction is pinned both
  structurally (the Core forward calls exactly one classification export,
  and the kernel body contains no division) and numerically: for logits
  `[0, −800]` the composed form gives `−inf` (the probability underflows
  to 0 before its logarithm is taken) while the fused form returns an
  accurate finite `−800`.
- **The ABI is contiguous-only, and reuses E3's call shape exactly.**
  `tf_core_log_softmax_forward(src_handle, src_offset, dst_handle, outer,
  axis_length, inner)` — strides never cross the boundary; the Core layer
  applies the existing native Policy-B copy-then-compute (§9.4) for a
  strided or offset view, closing the private copy in a `finally` whether
  the call succeeds or raises, and closing the output if the kernel fails
  after allocation. No log-softmax **backward** export exists.
- **The two exports now share one file-local validator.** Softmax and
  log-softmax re-prove the identical preconditions (handles, positive
  dimensional factors, non-negative offset, overflow-safe products,
  source span, destination capacity) in the identical order, so E4
  factored those checks into `forward_argument_error` and each export
  supplies its own operation name for the message. Softmax's behavior and
  messages are unchanged, and `cpp/tests/test_log_softmax.cpp` re-drives
  softmax's whole rejection matrix (plus a numeric call) as regression
  coverage for the sharing — the same discipline E2 applied to E1's unary
  validators.
- **The Core wrappers share their Policy-B plumbing too.** Both delegate
  to a private `NativeTensorCore._axis_fused_forward(axis, kernel_name)`,
  so axis normalization, the validate-before-allocate ordering, the
  Policy-B ownership, and the failure cleanup cannot drift apart; each
  public method still names its own exported symbol.
- **The backward is composed at the Core layer, with no new kernel.**
  `probabilities = y.exp()` → `slice_sum = g.sum(axis, keepdims=True)` →
  `scaled = probabilities * slice_sum` (broadcasting over the kept axis)
  → `contribution = g − scaled`, with every temporary closed in a
  `finally` as soon as its consumer has run. `exp(y)` recovers the
  probabilities from the **saved log probabilities** without rereading
  the input. **No dedicated log-softmax backward kernel exists**, as E0
  requires.
- **Saved-output semantics confirmed.** The node records no expected
  version and owns no private graph resource (`y` is the node's own
  core), so mutating a direct parameter after forward leaves the edge
  valid — proved with a non-uniform upstream, which is what makes the
  saved-output distinction observable. `log`'s live-input/version-checked
  contrast is re-proved unchanged in the same module, as are `exp`'s and
  `softmax`'s saved-output behavior and `multiply`'s guard; a mixed graph
  with a stale branch commits nothing on either branch and fails
  identically on retry.
- **Exceptional values are plain IEEE.** A NaN or `+inf` in a slice makes
  that whole slice NaN (`inf − inf`); an all-`-inf` slice is NaN; a
  `-inf` alongside finite values keeps `-inf` at its own position while
  the finite positions stay governed by the stable computation
  (`exp(-inf) == 0` contributes nothing to the sum). A structurally valid
  call that produces NaN or infinity is **not** an ABI failure — the
  error slot stays `TF_OK`. Tests compare against a NumPy reference
  running the *same* maximum-shift order rather than another framework's
  special-casing.
- **No `NativeLogSoftmax` module, no `NLLLoss`, no public `max`/`argmax`,
  and no division** were added, and the native checkpoint format stays
  **version 1**.

### 4.5 `NativeTensor.cross_entropy(targets, reduction="mean")`

- **Fused directly from raw logits.** Never
  `softmax().log()`-then-index, and never `log_softmax()`-then-gather:
  one kernel computes the row maximum, the log-sum-exp, the per-row loss,
  **and** the softmax probabilities the backward needs, in a single
  deterministic pass.
- **Logits contract:** shape exactly `(batch_size, num_classes)`, rank
  **2** (rank 1, rank 3, and higher are rejected with an error naming the
  actual shape). `batch_size` and `num_classes` are both positive by the
  existing nonempty-tensor rule. No broadcasting, no implicit reshape, no
  channel-dimension convention.
- **Targets:** not `NativeTensor`s — see §6.
- **Reduction:** exactly `"mean"` or `"sum"`.
- **Result:** a **scalar** `NativeTensor` (shape `()`), so
  `loss.backward()` works with the existing default seed and an explicit
  scalar upstream scales the gradient exactly as the engine defines.
- **Saved state:** the forward produces **private saved probabilities**
  (the full `(batch_size, num_classes)` softmax) as internal owning
  native storage. It is never a public `NativeTensor`, never has a public
  dtype, and never appears in any inventory (§7).
- **Backward** uses **only**: the saved probabilities, the independently
  copied target metadata, the reduction, and the scalar upstream
  gradient. The rule is
  `grad_logits[n, j] = upstream * (p[n, j] − [j == t_n]) / N` for
  `"mean"` (drop the `/N` for `"sum"`), computed by a dedicated backward
  kernel into fresh owning contiguous `(batch_size, num_classes)`
  storage.
- **Backward never rereads the logits**, so there is **no logits version
  snapshot**: a direct `NativeParameter` logits parent mutated after
  forward leaves this edge valid. (In practice logits are an
  intermediate, not a parameter; the rule is stated because it must be
  provable, not because it is common.)

**Status (E5 + E6 — this section is implemented as specified, at both
layers).** E5 shipped the **graph-unaware** half, end to end — the
internal `tf::cross_entropy_forward_contiguous` and
`tf::cross_entropy_backward_contiguous` kernels and the guarded exports
`tf_core_cross_entropy_forward` / `tf_core_cross_entropy_backward` in
`cpp/src/classification.cpp` (declared in
`cpp/include/tf_classification_internal.h`), their ctypes declarations
and `_CHECKED_KERNELS` registration, and the Core methods
`NativeTensorCore.cross_entropy_forward(targets, reduction="mean")` and
`NativeTensorCore.cross_entropy_backward(targets, upstream,
reduction="mean")`. **E6 then shipped the public differentiable
operation** `NativeTensor.cross_entropy(targets, reduction="mean")` over
it, adding **no new kernel, no new ABI export, and no change to any
formula** — see the E6 status block after these notes. Implementation
notes:

- **The forward is genuinely fused, in C++.** One kernel walks each row
  once for its maximum, once to write `exp(x − m)` straight into the
  saved-probability destination while accumulating `Σ exp(x − m)`, and
  once to normalize in place; the per-example loss is
  `log(Σ exp(x − m)) − (x[target] − m)`, accumulated in deterministic
  batch order. It is emphatically **not** `−log(p[target])` (which
  reports `inf` the moment a probability underflows — pinned numerically
  by a test at a target 800 below the row maximum), not
  `softmax().log()`-then-index, and not `log_softmax()`-then-gather: the
  Core forward calls exactly one classification export and no
  public `max`, `argmax`, `gather`, or division exists at any layer.
  **No second probability buffer is allocated in C++**, and the kernel
  allocates nothing at all.
- **Rank two, class axis fixed.** Logits must be exactly
  `(batch_size, num_classes)`; rank 0, 1, and 3 are rejected by shape
  before anything is allocated. There is no `axis` argument.
- **The targets are copied into owned `int64` metadata.** §6's whole
  rejection matrix is enforced at the Core boundary *before* any
  allocation — `bool` (Python and NumPy), floating-point values
  including integral ones like `1.0`, complex values, strings, bytes,
  nested/ragged sequences, rank-2 arrays, scalars, object arrays,
  values outside the int64 range, and any label outside
  `[0, num_classes)`. Validation never routes through
  `np.asarray(targets, dtype=np.int64)`, which would silently truncate
  `1.9` and reinterpret `True`; values are compared as Python ints and
  only then copied into a fresh, contiguous, **read-only** `np.int64`
  array. The copy is taken **even when the caller already passed a
  contiguous `int64` array**, so no view into caller memory is ever
  retained and post-forward caller mutation cannot reach the kernel —
  proved for a list and for an array.
- **The reduction is normalized once, in Python.** Exactly `"mean"` and
  `"sum"` by exact string match (the `NativeMSELoss` precedent:
  TypeError for a non-string, ValueError for an unknown string), mapped
  to the E0-locked integer code — `0` = mean, `1` = sum — which is
  revalidated in C++. A test proves only `0` and `1` ever reach the ABI.
- **The C ABI is contiguous-only for tensor data.** The forward takes
  `(logits handle + offset, const int64_t* targets, target_count, loss
  handle, probabilities handle, batch_size, num_classes, reduction
  code)`; the backward takes `(probabilities handle + offset,
  const int64_t* targets, target_count, upstream handle + offset,
  grad_logits handle, batch_size, num_classes, reduction code)`. Strides
  never cross the boundary — the Core layer applies the existing native
  Policy-B copy-then-compute (§9.4) for a strided or offset view and
  closes the private copy in a `finally` whether the call succeeds or
  raises. **The logits are not an argument of the backward at all**,
  which is the structural half of "backward never rereads the logits".
- **Both exports self-validate, and re-prove every target index.** They
  share one file-local validator (`cross_entropy_common_error`) plus
  `reject_target_range`, and check handles, the target pointer,
  dimensions, offsets, the target count, the reduction code, overflow,
  every span, every destination capacity, and destination/operand
  aliasing **before a single destination element is written** — so a
  rejected forward leaves *both* the loss and the probability
  destinations byte-for-byte unchanged, and a rejected backward leaves
  the gradient unchanged. C++ revalidates all `batch_size` target
  indices rather than trusting Python, the same discipline the pooling
  backward applies to its saved winners.
- **Multiple outputs fail atomically.** The forward allocates the scalar
  loss and then the probability block; if the second allocation, the
  kernel, or anything else raises, every object the method allocated is
  closed exactly once, the Policy-B temporary is closed exactly once, no
  partial result object escapes, and the caller's logits stay open and
  unchanged. Injected allocation failures across the whole sweep leak
  nothing and retry cleanly.
- **The upstream stays native.** The backward takes an open
  `NativeTensorCore` holding **exactly one element** — rank-0 `()`,
  `(1,)`, `(1, 1)`, or a one-element view at a nonzero offset — and
  passes its *storage handle and offset* to the kernel. The value is
  **never** extracted through `to_numpy()`, `float()`, or any NumPy
  buffer.
- **The saved probabilities are private Core state.** They are returned
  through the internal `_CrossEntropyForwardResult` record (loss,
  probabilities, copied targets, normalized reduction); they are not a
  public `NativeTensor`, not a module parameter or buffer, not in any
  `state_dict()` or checkpoint, and not in any capability inventory. The
  **Core** caller owns them and releases them with `result.close()`;
  §7's graph-owned lifetime is what E6's autograd node applies on top.
- **Exceptional values are plain IEEE.** A NaN or `+inf` in a row makes
  that row NaN (`inf − inf`); an all-`-inf` row is NaN; a `-inf`
  alongside finite values simply takes zero probability, and a `-inf`
  *target* logit gives an infinite loss. A structurally valid call that
  produces NaN or infinity is **not** an ABI failure — the error slot
  stays `TF_OK`. Tests compare against a NumPy reference running the
  *same* maximum-shift order rather than another framework's
  special-casing.
- **No tensor-data NumPy round-trip.** NumPy is used on this path only
  to build the owned `int64` target copy and to marshal shape/stride
  metadata for ctypes; a tripwire test proves every value handed to a
  NumPy constructor is a small tuple or list of Python ints, and a
  stricter second tripwire blocks *every* NumPy constructor across a
  backward whose target copy is already prepared.

**Status (E6 — the differentiable operation, implemented as specified).**
`NativeTensor.cross_entropy(targets, reduction="mean")` lives in
`src/tensorforge/experimental/native_tensor.py` and is registered as
`"cross_entropy"` in `AUTOGRAD_OPS`. It is **autograd integration only**:
E6 changed no C++ file, added no C ABI export, altered no ctypes
signature, and touched no numerical formula. Implementation notes:

- **One Core call, one graph node.** The method calls
  `NativeTensorCore.cross_entropy_forward` exactly once — which is where
  every validation, the target copy, Policy B, and the fused kernel
  already live — unpacks the E5 result record immediately, and hands the
  scalar loss core to `_from_op(..., graph_resources=(probabilities,))`
  with the logits as its single parent. There is no second cross-entropy
  path and no revalidation at this layer.
- **Ownership transfer is explicit, and needs no new machinery.** The E5
  result record is a plain `__slots__` carrier with **no** `__del__`, so
  once the four fields are unpacked into the calling frame the record
  owns nothing that could be double-closed. That places this operation in
  exactly MaxPool2d's position when its Core returns
  `(out_core, winners)`, and the same cleanup applies: if graph
  construction raises, the probabilities and then the loss are closed
  before the exception propagates. No release/take helper and no resource
  manager was added.
- **Saved probabilities are graph-owned.** Exactly one resource belongs
  to the node. It survives forward, survives `retain_graph=True`, survives
  a failed retryable backward, and is released exactly once at the
  deterministic points §7 names — a one-shot `backward()`'s history
  release, or `close()`. A **no-grad forward closes it immediately**
  inside `_from_op`, proved by instrumenting the core's own lifetime
  rather than by observing an empty `graph_resources` tuple.
- **Targets are closure metadata, not a graph resource.** The owned,
  read-only `int64` copy is captured by the backward closure and released
  with it when the history is. `graph_resources` is only for closeable
  *native* objects, so nothing was added there; no native integer tensor
  exists.
- **Backward never rereads the logits, and records no version.**
  `_expected_versions == ()`. The Core backward's signature does not
  accept logits at all, so this is structural rather than a promise:
  mutating a direct `NativeParameter` logits parent with `copy_value_`
  after the forward neither raises a stale-graph error nor changes the
  gradient, including across a retained graph, and a fresh forward
  afterwards uses the new values. This is the `maxpool2d` archetype and
  the deliberate contrast with `log`.
- **A closed parent is a lifetime failure, not a value reread.** The
  gradient still computes correctly from saved state; it is
  `_accumulate_grad` that requires the parent tensor open, because that
  is where the gradient is *stored*. The failure commits nothing, keeps
  the saved probabilities, and leaves the graph retryable. No version
  snapshot would help and none is recorded.
- **Nothing partial survives a failure.** An E5 validation, allocation,
  or kernel failure returns no tensor and builds no node; a graph
  construction failure closes both E5 outputs; a backward failure
  (allocation, native, or accumulation) commits no gradient, leaks no
  gradient core — proved while holding a strong reference that disables
  the `__del__` fallback — keeps the probabilities for a retry, and does
  not mark the graph freed. In a mixed graph whose other branch is stale,
  the engine's preflight raises before any callback runs, so
  cross-entropy commits nothing and pre-existing gradients are restored
  by reference.
- **Still no tensor-data NumPy round-trip.** The public forward and
  backward run under the same strict tripwire as the Core layer, on the
  contiguous and the Policy-B paths; a stricter variant blocks *every*
  NumPy constructor across a backward, and an instrumented probe proves
  the only values a NumPy constructor ever receives are small integer
  target labels and shape/stride metadata.

---

## 5. Backward-read and versioning matrix

The one table to consult when implementing any Phase-E backward. "Saved"
means state the forward recorded; "live" means the parent's current value
read at backward time.

| Operation | Backward reads | Rereads the input? | Parameter version snapshot |
|---|---|---|---|
| `exp` | the **saved output** `y` | no | **none** |
| `log` | the **live input** `x` | **yes** | **version-checked** (stale-graph error on a mutated direct parameter) |
| `softmax` | the **saved output** `y` | no | **none** |
| `log_softmax` | the **saved output** `y` | no | **none** |
| `cross_entropy` | the **saved probabilities** + copied targets | no | **none** (no logits version snapshot) |

Only `log` joins the version-checked set (`multiply`, `matmul`, `relu`,
and conditionally `conv2d`). Everything else in Phase E is a
saved-state backward and therefore survives post-forward mutation, the
same property `sqrt`/`reciprocal`/`maxpool2d` already have.

---

## 6. Target contract

The native runtime has **no integer dtype**, and Phase E deliberately
does not add one (a public integer `NativeTensor` is a much larger
change: storage, ABI, dtype normalization, promotion rules). So
classification targets are **not** `NativeTensor`s.

**Targets are accepted as Python or NumPy integer data and immediately
converted to an independently owned contiguous `int64` copy.** That copy
— never the caller's object — is what the forward validates, what the
kernel reads, and what the backward closure keeps.

Validation, all performed **before any allocation or kernel call**:

- must be **one-dimensional** (a nested sequence or a 2-D array is
  rejected; a scalar is rejected);
- length must equal `batch_size` exactly;
- every element must be an **actual integer scalar value**;
- **`bool` is rejected** (a `bool` is an `int` in Python, but a boolean
  target is a modelling error, and the native line rejects `bool`
  wherever an integer is meant);
- **floating-point values are rejected, including integral ones such as
  `1.0`** — no silent truncation, no "it happened to be whole" coercion;
- **nested arrays are rejected**;
- **negative values are rejected**;
- values **≥ `num_classes` are rejected**;
- errors name the offending index and value.

**Caller mutation after forward can never affect backward.** Because the
forward took an independent `int64` copy, mutating the caller's list or
NumPy array afterwards changes nothing — targets carry no version
counter and need none. This must be proved by a dedicated test, not
assumed.

**Crossing the ABI.** Targets travel as a raw `const int64_t*` span plus
its length — the honest representation, since they are not native
tensors. The owning buffer stays alive for the whole call (and, for the
backward, for the life of the graph node). C++ **revalidates the range**
of every target it reads (`0 ≤ t < num_classes`) rather than trusting the
Python layer, per §10.

---

## 7. Saved-probability lifetime contract

Phase E reuses, unchanged, the **graph-owned resource model** Phase D
established for MaxPool2d's private winner buffer
(`NativeTensor._from_op(..., graph_resources=(...))`).

When `cross_entropy` **builds an autograd graph**:

- the output node **owns** the private probability storage;
- it is **retained under `retain_graph=True`**, so a second backward pass
  still has it;
- it is **released after a successful one-shot backward**, at the same
  deterministic point the graph history is released;
- it is **released by `output.close()`** if the graph is abandoned;
- it **stays alive after a failed retryable backward**, so a caller who
  fixes the error and retries is not left with a half-freed graph;
- it is **released exactly once** (idempotent, never double-freed);
- it **never appears in `state_dict()`**;
- it **never appears in a checkpoint**.

When **no graph is built** (no operand requires grad), the private
probabilities are **closed immediately inside the forward** — the
existing `_from_op` behavior — so a no-grad forward can never leak saved
state.

The copied `int64` targets are ordinary Python-side data held by the same
backward closure and released with it. They are graph data: never
serialized, never public.

**Status: live as of E6.** Every rule above now binds and is directly
tested. The Core forward still hands the private probabilities back
through its internal result record — at that layer the **caller** owns
them and releases them with `result.close()` — and
`NativeTensor.cross_entropy` is the caller that adopts them into
`_from_op(..., graph_resources=(probabilities,))`. The E5 result record
has no finalizer, so unpacking it is a complete ownership transfer; if
graph construction raises, the probabilities and the loss are both closed
explicitly before the exception propagates. Exactly one probability
resource belongs to the node; a no-grad forward closes it inside
`_from_op` before returning; the copied targets ride in the backward
closure rather than in `graph_resources`, which is reserved for closeable
native objects.

---

## 8. Metric contract

`native_accuracy(logits, targets) -> float` is a **reporting helper**,
not a native kernel and not an autograd operation.

It must:

- accept an **open `NativeTensor` of rank 2** (`(batch_size,
  num_classes)`), rejecting everything else by type/state/shape;
- validate `targets` with **the same contract as §6**;
- **explicitly materialize** through `logits.to_numpy()` — the documented
  native→host copy, the one sanctioned exit from the native world;
- compute `argmax` with **NumPy**;
- return a **Python `float`** (the fraction correct), matching the stable
  `accuracy` metric's return type;
- **build no graph**, modify no gradients, modify no parameters, and
  leave the input tensor open and unchanged.

**Why NumPy is allowed here, and only here.** This is a metric computed
outside training and outside autograd, exactly as the stable framework's
metrics are "plain NumPy returning Python floats, outside autograd". It
is not a native operation and must never be described as one. Every
**numerical** native operation keeps its NumPy-tripwire protection: the
Phase-E training proof asserts no NumPy compute inside the training step,
and the metric is called outside that guarded region (or the guard
explicitly permits the reporting call, and says so).

`native_accuracy` therefore belongs in a **new `NATIVE_METRICS`
inventory** (§11) — not in `TENSOR_CORE_OPS`, not in `AUTOGRAD_OPS`, not
in `NATIVE_MODULES`.

---

## 9. C++ source organization and the C ABI

### 9.1 Source units

| Unit | Contents |
|---|---|
| `cpp/src/elementwise.cpp` (existing) | `exp` and `log` — they are unary elementwise ops and belong with `relu`/`sqrt`/`reciprocal`, reusing the existing `core_unary` odometer and `core_unary_contiguous` fast-path helpers |
| `cpp/src/classification.cpp` (**new**) | `softmax` forward, `log_softmax` forward, cross-entropy forward, cross-entropy backward |
| `cpp/include/tf_classification_internal.h` (**new**) | declarations of the internal `tf::…_contiguous` compute functions, so the dependency-free CTests can link them without going through the C ABI |

`cpp/CMakeLists.txt` globs `cpp/src/*.cpp`, so a new translation unit
needs **no build-system change**; `cpp/build.py`'s direct-compiler
fallback enumerates sources the same way. Verify, do not assume, at the
milestone that adds the file.

### 9.2 Guarded C ABI families

| Symbol | Milestone | Shape of the call |
|---|---|---|
| `tf_core_exp` | E1 | generic strided: source handle, destination handle, `shape`, `strides`, `offset`, `ndim` |
| `tf_core_exp_contiguous` | E1 | contiguous: source handle, destination handle, `numel`, `offset` |
| `tf_core_log` | E2 | as `tf_core_exp` |
| `tf_core_log_contiguous` | E2 | as `tf_core_exp_contiguous` |
| `tf_core_softmax_forward` | E3 | contiguous-only: source handle + offset, destination handle, `outer`, `axis_length`, `inner` |
| `tf_core_log_softmax_forward` | E4 | as `tf_core_softmax_forward` |
| `tf_core_cross_entropy_forward` | E5 | logits handle + offset, `const int64_t* targets`, loss handle, probabilities handle, `batch_size`, `num_classes`, reduction code |
| `tf_core_cross_entropy_backward` | E5 | probabilities handle, `const int64_t* targets`, upstream handle + offset, grad-logits handle, `batch_size`, `num_classes`, reduction code |

The **final exact argument lists** follow the repository's existing
handle / shape / stride / offset / dimension / status / overflow /
span-validation conventions (compare `tf_core_relu`, `tf_core_sum`,
`tf_core_conv2d_forward`, `tf_core_maxpool2d_forward`); the table fixes
the *shape* of each call, not its literal signature.

The reduction crosses as a small integer **code** (`0` = mean, `1` =
sum), validated on both sides — never a string, which would put
allocation and encoding concerns on the ABI.

### 9.3 ABI rules (unchanged, restated because they bind Phase E)

- **No C++ exception crosses the ABI.** Every export is wrapped in
  `TF_GUARD_BEGIN` / `TF_GUARD_END_VOID`; failures become the
  thread-local status the ctypes `errcheck` hook turns into
  `MemoryError` / `ValueError` / `RuntimeError`.
- **Raw ABI validation is self-contained.** Each export re-proves its own
  preconditions — non-null handles, positive dimensions, non-negative
  offsets, overflow-safe products, and every span it touches inside its
  storage — because the exports are reachable by any ctypes caller, not
  only by the Python wrapper.
- **Python validates before allocation** wherever it can (dtype/device,
  rank, shape, axis, reduction, targets, output-shape arithmetic in
  arbitrary-precision Python ints), so a rejected call allocates nothing.
- **C++ revalidates trust-boundary data** — most importantly every target
  index it dereferences, the same way the pooling backward revalidates
  every saved winner instead of trusting it.
- **Ordinary failures mutate nothing.** A rejected call leaves inputs,
  gradients, parameters, versions, and graphs untouched.
- **Partial allocations are cleaned up.** Where a forward allocates both
  an output and a private buffer, the wrapper closes what it already owns
  before propagating a failure (the `_maxpool2d_forward_with_winners`
  pattern).
- **No implicit fallback exists.** An unavailable native operation raises
  with build instructions; it never quietly computes in NumPy.

### 9.4 The native Policy-B copy (E3.1 — shared runtime hardening)

Phase E's contiguous-only ABIs depend on Policy-B copy-then-compute, so
that copy must itself stay inside native memory. It now does.

`NativeTensorCore.contiguous_copy` (and its `NativeTensorView`
counterpart) previously materialized through
`from_array(self.to_numpy())` — a host round-trip that exported tensor
values into a NumPy buffer and imported them back. E3.1 replaced it with
a **native storage-to-storage gather**:

| Export | Shape of the call |
|---|---|
| `tf_core_contiguous_copy` | source handle, destination handle, `shape`, `strides`, `offset`, `ndim` — the same generic-strided signature the unary exports use |

The wrapper allocates the destination first, calls the export, and closes
the destination if the call raises, so no partially initialized core
escapes. The kernel reuses `elementwise.cpp`'s existing odometer walker
(the operation is the identity map; only the traversal matters) and its
existing `unary_strided_error` trust-boundary validation — handles,
layout metadata, spans in both stride directions, overflow, and
destination capacity — so the copy inherits validation already exercised
by the E1/E2 CTests rather than re-deriving it.

**No tensor-data NumPy round-trip remains** in any copy path; only
shape/stride arrays cross as ctypes metadata. Softmax stays
**contiguous-only at the C ABI boundary**, and the Core continues to
handle non-contiguous inputs through this native Policy-B copy.

This is **shared native runtime hardening, not new Conv2d or MaxPool2d
behavior**: those Phase-D operations, `NativeFlatten`, `NativeParameter`
construction, and the differentiable `contiguous_copy` operation all use
the same helper and gained the same property with no change to their own
contracts, numerics, or public surface.

---

## 10. Failure atomicity across the layers

Every Phase-E milestone must preserve the four-layer failure story:

1. **Kernel:** validates, then computes; on failure records a status and
   writes nothing meaningful (allocation failures inside a kernel surface
   as `MemoryError`).
2. **Core wrapper:** validates → allocates → calls → cleans up on any
   raise, so no partially built output or orphaned private buffer
   escapes.
3. **`NativeTensor` op:** builds a graph node only after a successful
   forward; a failed forward creates no node and adopts no resource.
4. **`backward()`:** the existing snapshot/rollback engine restores every
   node's gradient on failure, so a failed pass commits no partial
   gradients — and, per §7, keeps the saved probabilities alive for a
   retry.

---

## 11. Capability inventory contract

Later milestones must update the **correct layer-specific inventory** —
never a convenient one:

| Name | Inventory | Milestone |
|---|---|---|
| `exp`, `log` | `TENSOR_CORE_OPS` **and** `AUTOGRAD_OPS` | E1, E2 |
| `softmax`, `log_softmax` | `TENSOR_CORE_OPS` **and** `AUTOGRAD_OPS` | E3, E4 |
| `cross_entropy_forward`, `cross_entropy_backward` | `TENSOR_CORE_OPS` (layer-qualified Core wrappers) | E5 |
| `cross_entropy` | `AUTOGRAD_OPS` (the differentiable operation) | E6 |
| `NativeCrossEntropyLoss` | `NATIVE_LOSSES` | E7 |
| `native_accuracy` | a **new `NATIVE_METRICS`** inventory, surfaced by `backend_info()` | E7 |

Each name leaves `UNSUPPORTED` in the same milestone that implements it,
and not before.

**How E5 reported a half-implemented capability.** `UNSUPPORTED` is a
flat list of names; it cannot express "the Core layer is implemented but
the autograd operation is not". The registry has always resolved that by
being **layer-specific** — Conv2d and MaxPool2d each carry
`*_forward`/`*_backward` Core names, a separate differentiable operation
name, and a separate module name. E5 followed exactly that rule:
`"cross_entropy_forward"` and `"cross_entropy_backward"` entered
`TENSOR_CORE_OPS`, and the bare `"cross_entropy"` token left
`UNSUPPORTED` because it no longer describes anything absent as a whole.
The differentiable operation's absence was reported the way this registry
reports every unimplemented operation — **by `"cross_entropy"` not being
in `AUTOGRAD_OPS`** — and **E6 added it there**, leaving the two Core
wrappers exactly where E5 put them. The bare name is therefore in
`AUTOGRAD_OPS` only; it is deliberately *not* an alias in
`TENSOR_CORE_OPS`, and there is no `NativeTensorCore.cross_entropy`.
`NativeCrossEntropyLoss` and `native_accuracy` stay in `UNSUPPORTED`
until E7. Guardrail tests assert each half of this directly, so the
boundary cannot blur.

Explicitly forbidden placements:

- `NativeCrossEntropyLoss` in `AUTOGRAD_OPS` — it is a module, not an
  operation (the `NativeMSELoss` precedent).
- `native_accuracy` in `TENSOR_CORE_OPS`, `AUTOGRAD_OPS`, or
  `NATIVE_MODULES` — it is a reporting function, not a runtime op, not
  differentiable, and not a module.
- `softmax` or `cross_entropy` in `RAW_KERNELS` — that tuple is the
  frozen set of raw NumPy-buffer benchmark kernels. Phase E designs **no**
  separate raw NumPy-buffer classification kernels, so nothing is added
  there unless a later milestone explicitly designs them.

`TENSOR_CORE_KERNELS` stays frozen at its historical five, as always.

---

## 12. Checkpoint contract

Phase E **retains native checkpoint format version 1**. Classification
adds **no persistent model or optimizer state**:

- `NativeCrossEntropyLoss` is parameter-free and buffer-free; its
  `state_dict()` is empty and it contributes no checkpoint keys (the
  `NativeMSELoss` precedent).
- Saved probabilities and copied targets are **graph data** and must
  never be serialized.
- `native_accuracy` holds no state at all.
- No manifest field, section, key ordering, array-naming rule, or
  validation step changes. A Phase-E model checkpoints and resumes
  through exactly the existing `save_native_checkpoint` /
  `load_native_checkpoint` paths, and a checkpoint written before Phase E
  remains loadable.

---

## 13. Testing contract

Each milestone lands with focused tests; the phase closes with
cross-cutting ones. Recurring obligations:

- **Exact hand-computed cases** for every forward (small tensors whose
  values can be written down).
- **Stable-framework parity** to a stated tolerance — native `softmax` /
  `log_softmax` / `cross_entropy` against `tensorforge.Tensor.softmax`
  and `tensorforge.nn.cross_entropy`, comparing *values only* (no shared
  objects, no shared graph).
- **Central finite differences** for every gradient, over every
  `requires_grad` combination.
- **Stability tests with extreme logits** (e.g. `±1000`) proving no
  overflow, no `NaN`, and probabilities summing to 1 — the tests that
  would fail against a naive implementation.
- **Versioning tests** proving the §5 matrix: mutate a direct parameter
  after forward and assert `log` raises stale-graph while `exp`,
  `softmax`, `log_softmax`, and `cross_entropy` still differentiate
  correctly.
- **Ownership/lifetime tests**: outputs own their storage; strided and
  offset inputs work through Policy B without mutating the caller's view;
  saved probabilities follow §7 across one-shot backward, `retain_graph`,
  `close()`, abandonment, and a failed retryable backward.
- **Target-contract tests**: every rejection in §6, plus post-forward
  caller mutation having no effect.
- **Failure-atomicity tests** at every boundary, including the existing
  deterministic allocation-failure injection.
- **NumPy-tripwire tests** proving no NumPy compute in the native
  training path (with the §8 metric exception stated explicitly).
- **Dependency-free C++ CTests** for each new internal compute function,
  and **ASan/UBSan** validation of the whole classification stack.

---

## 14. Benchmark contract

E9 adds a characterization harness in the established style
(`benchmarks/benchmark_native_cnn.py` is the model): correctness gate
first, medians over repeated runs after warmup, a `--smoke` mode, a JSON
mode, honest hardware-specific reporting, and **no performance
assertions anywhere**. It measures what the phase actually built —
`exp`/`log` elementwise, `softmax`/`log_softmax` along an axis, fused
cross-entropy forward and forward+backward, and a full classification
training step — against a stable-framework reference row.

---

## 15. Milestone ladder (E0–E10)

| Milestone | Deliverable | Status |
|---|---|---|
| E0 | Classification architecture contract and Phase-D baseline reconciliation | **complete** |
| E1 | Native exponential | **complete** |
| E2 | Native logarithm | **complete** |
| E3 | Stable differentiable softmax | **complete** |
| E4 | Stable differentiable log-softmax | **complete** |
| E5 | Fused cross-entropy forward and backward Core contract | **complete** |
| E6 | Differentiable `NativeTensor` cross-entropy | **complete** |
| E7 | `NativeCrossEntropyLoss` and reporting-only `native_accuracy` | not started |
| E8 | Deterministic native classification training and exact checkpoint resume | not started |
| E9 | Native classification benchmark characterization | not started |
| E10 | Phase-E integration, sanitizer validation, documentation reconciliation, and closure | not started |

Each milestone's full contract follows; the table above is the status
summary, and the registry remains the authority on what is live.

### E0 — Classification architecture contract and Phase-D baseline reconciliation *(this document)* — **complete**

- **Objective:** lock the complete Phase-E contract above, and reconcile
  the documentation and comment drift left after Phase D so the phase
  starts from an accurate baseline.
- **Layer:** documentation and guardrail tests only.
- **Expected files:** this document; `docs/native_support_matrix.md`,
  `docs/roadmap.md`, `docs/backend_experiments.md`,
  `docs/architecture.md`, `docs/project_summary.md`, `README.md`,
  `CLAUDE.md`, `src/tensorforge/backends/cpp.py` (comments and the
  `UNSUPPORTED` boundary only), `src/tensorforge/experimental/__init__.py`
  (docstring only), `tests/test_docs.py`.
- **Required tests:** durable semantic guardrails — the design document
  exists, is linked, and states its load-bearing decisions; the ladder is
  in order; the four backward/versioning distinctions are pinned; every
  Phase-E capability is still `UNSUPPORTED` in the live registries; the
  Phase-D shipped surface is unchanged; no authoritative status surface
  claims a shipped Phase-D module is unimplemented.
- **Risks:** accidentally advertising an unimplemented capability;
  writing brittle prose-locking tests; deleting useful history instead of
  labelling it.
- **Dependencies:** completed Phase D.
- **Non-goals:** **any** numerical behavior — no kernel, ABI symbol,
  ctypes declaration, Core method, tensor operation, module, metric,
  benchmark, or example.

### E1 — Native exponential — **complete**

- **Objective:** a differentiable `NativeTensor.exp()` through the whole
  stack.
- **Layer:** C++ kernels → C ABI → ctypes → `NativeTensorCore` →
  `NativeTensor` autograd.
- **Expected files:** `cpp/src/elementwise.cpp`;
  `src/tensorforge/backends/cpp.py` (two guarded exports, their ctypes
  declarations, `NativeTensorCore.exp`, inventories);
  `src/tensorforge/experimental/native_tensor.py`;
  `tests/test_native_exp.py`.
- **Required tests:** exact values; IEEE edge cases (`0`, `-inf`,
  overflow to `+inf`, `NaN`); strided/offset views through both execution
  paths, bit-identical to each other; fresh owning contiguous output;
  finite-difference gradients; saved-output backward valid after a
  post-forward parameter mutation (**no** version snapshot); closed
  saved-output failure is clear and atomic.
- **Risks:** forgetting the contiguous fast path and silently regressing
  performance characteristics; letting the backward reread the input,
  which would wrongly demand versioning.
- **Dependencies:** E0.
- **Non-goals:** `log`, any probability transform, any module.
- **Shipped (E1).** Exactly the above, plus the two refinements recorded
  in §4.1: the exports self-validate at the trust boundary (handles,
  layout metadata, spans, overflow, negative strides), and the saved
  output is the autograd node's own core rather than a separate
  `graph_resources` entry. Files touched: `cpp/src/elementwise.cpp`
  (`op_exp`, the file-local validators, and the two guarded exports),
  `cpp/tests/test_exp.cpp` + `cpp/CMakeLists.txt` (a new dependency-free
  CTest driving the exported pair, including its rejection cases),
  `src/tensorforge/backends/cpp.py` (ctypes declarations,
  `_CHECKED_KERNELS`, `NativeTensorCore.exp`, `TENSOR_CORE_OPS`,
  `AUTOGRAD_OPS`, `UNSUPPORTED`),
  `src/tensorforge/experimental/native_tensor.py` (`NativeTensor.exp`),
  and `tests/test_native_exp.py`. No new source file, no module, no
  benchmark, no example, no schema change, and no change to any existing
  kernel or operation.

### E2 — Native logarithm — **complete**

- **Objective:** a differentiable `NativeTensor.log()`.
- **Layer:** same four layers as E1.
- **Expected files:** as E1, plus `tests/test_native_log.py`.
- **Required tests:** exact values; `log(0) == -inf`,
  `log(negative) == NaN`, `NaN` propagation — proving **no clamping and
  no epsilon**; both execution paths; finite differences;
  `dx = upstream * reciprocal(x)` verified against an analytic reference;
  **stale-graph raised** when a direct parameter parent is mutated after
  forward, raised **before** any gradient changes; a non-parameter parent
  records no version.
- **Risks:** copying E1's no-version backward by habit; introducing a
  division operation instead of reusing `reciprocal`.
- **Dependencies:** E1 (shares the unary kernel scaffolding).
- **Non-goals:** stability clamping, `log1p`, `log2`, `log10`.
- **Shipped (E2).** Exactly the above; see §4.2's status block for the
  implementation notes. Files touched: `cpp/src/elementwise.cpp`
  (`op_log` and the two guarded exports — E1's validators reused
  verbatim), `cpp/tests/test_log.cpp` + `cpp/CMakeLists.txt` (a seventh
  CTest target), `src/tensorforge/backends/cpp.py` (ctypes declarations,
  `_CHECKED_KERNELS`, `NativeTensorCore.log`, `TENSOR_CORE_OPS`,
  `AUTOGRAD_OPS`, `UNSUPPORTED`),
  `src/tensorforge/experimental/native_tensor.py` (`NativeTensor.log`),
  and `tests/test_native_log.py`. Neither risk materialized: the backward
  records a version through the existing helper, and the derivative goes
  through the existing native `reciprocal` — **no division operation was
  added at any layer**. No new source file, module, benchmark, example,
  schema change, or change to any existing kernel or operation.

### E3 — Stable differentiable softmax — **complete**

- **Objective:** `NativeTensor.softmax(axis=-1)` with the fused
  maximum-shift forward and the saved-output backward.
- **Layer:** new `cpp/src/classification.cpp` + internal header → C ABI →
  Core → autograd.
- **Expected files:** `cpp/src/classification.cpp`;
  `cpp/include/tf_classification_internal.h`; `cpp/tests/test_softmax.cpp`;
  `src/tensorforge/backends/cpp.py`;
  `src/tensorforge/experimental/native_tensor.py`;
  `tests/test_native_softmax.py`.
- **Required tests:** rows sum to 1; exact small cases; extreme logits
  (`±1000`) with no overflow/`NaN`; every rank ≥ 1 and every valid
  positive/negative axis; `bool` axis rejected; non-contiguous input via
  Policy B leaves the caller's view untouched; output owns fresh
  contiguous storage; stable parity; finite differences; saved-output
  backward with **no** version snapshot; the C++ CTest for the internal
  compute function.
- **Risks:** the `(outer, axis_length, inner)` factorization getting
  negative-axis normalization wrong; a non-contiguous copy leaking on a
  failure path; composing the backward with an extra graph node instead
  of at the Core layer.
- **Dependencies:** E1 (the `exp` math and the unary precedent).
- **Non-goals:** a `NativeSoftmax` module; a public `max` reduction;
  `log_softmax`.
- **Shipped (E3).** Exactly the above; see §4.3's status block for the
  implementation notes. Files created: `cpp/src/classification.cpp`,
  `cpp/include/tf_classification_internal.h`, `cpp/tests/test_softmax.cpp`,
  `tests/test_native_softmax.py`. Files touched: `cpp/CMakeLists.txt` (an
  eighth CTest target only — the `src/*.cpp` glob discovers the new source
  unit automatically, so no source-list entry was added),
  `src/tensorforge/backends/cpp.py` (the ctypes declaration,
  `_CHECKED_KERNELS`, `NativeTensorCore.softmax`, `TENSOR_CORE_OPS`,
  `AUTOGRAD_OPS`, `UNSUPPORTED`), and
  `src/tensorforge/experimental/native_tensor.py`
  (`NativeTensor.softmax`). None of the named risks materialized: the
  axis factorization handles negative axes through the shared validator,
  the Policy-B copy is closed on both paths, and the backward adds no
  graph node of its own. No module, no public `max`/`argmax`/division, no
  benchmark, no example, no schema change.

### E4 — Stable differentiable log-softmax — **complete**

- **Objective:** `NativeTensor.log_softmax(axis=-1)` as a **fused**
  log-sum-exp, never `softmax().log()`.
- **Layer:** as E3.
- **Expected files:** `cpp/src/classification.cpp`;
  `cpp/tests/test_log_softmax.cpp`; `src/tensorforge/backends/cpp.py`;
  `src/tensorforge/experimental/native_tensor.py`;
  `tests/test_native_log_softmax.py`.
- **Required tests:** `exp(log_softmax(x))` matches `softmax(x)` to
  tolerance; precision in the small-probability regime where
  `log(softmax(x))` degrades; extreme logits; rank/axis coverage as E3;
  finite differences; saved-output backward with **no** version snapshot;
  a test asserting the implementation does not route through `softmax`
  followed by `log`; the C++ CTest.
- **Risks:** silently reintroducing the composed form; drift between the
  softmax and log-softmax axis handling.
- **Dependencies:** E2 (log semantics), E3 (axis machinery).
- **Non-goals:** a `NativeLogSoftmax` module; `NLLLoss`.
- **Shipped (E4).** Exactly the above; see §4.4's status block for the
  implementation notes. Files created: `cpp/tests/test_log_softmax.cpp`,
  `tests/test_native_log_softmax.py`. Files touched:
  `cpp/src/classification.cpp` (the internal
  `tf::log_softmax_forward_contiguous` kernel, the guarded
  `tf_core_log_softmax_forward` export, and the shared
  `forward_argument_error` validator the two exports now use),
  `cpp/include/tf_classification_internal.h` (the new internal
  declaration), `cpp/CMakeLists.txt` (a ninth CTest target),
  `src/tensorforge/backends/cpp.py` (the ctypes declaration,
  `_CHECKED_KERNELS`, `NativeTensorCore.log_softmax` over the shared
  private `_axis_fused_forward` helper, `TENSOR_CORE_OPS`,
  `AUTOGRAD_OPS`, `UNSUPPORTED`), and
  `src/tensorforge/experimental/native_tensor.py`
  (`NativeTensor.log_softmax`). Neither named risk materialized: the
  forward is a single fused kernel that neither divides nor calls the
  softmax kernel (pinned structurally *and* numerically in the
  small-probability regime), and the axis handling cannot drift because
  both Core wrappers now go through one shared helper. No module, no
  `NLLLoss`, no public `max`/`argmax`/division, no backward kernel, no
  benchmark, no example, no schema change.

### E5 — Fused cross-entropy forward and backward Core contract — **complete**

- **Objective:** the autograd-unaware Core layer of cross-entropy: the
  fused forward (loss **and** private probabilities) and the backward
  kernel, reachable from Python as forward-only Core methods.
- **Layer:** C++ kernels → C ABI → ctypes → `NativeTensorCore`.
- **Expected files:** `cpp/src/classification.cpp`;
  `cpp/include/tf_classification_internal.h`;
  `cpp/tests/test_cross_entropy.cpp`;
  `src/tensorforge/backends/cpp.py` (both guarded exports, ctypes
  declarations, `NativeTensorCore.cross_entropy_forward` — including the
  private with-probabilities helper — and
  `NativeTensorCore.cross_entropy_backward`);
  `tests/test_native_cross_entropy_core.py`.
- **Required tests:** exact loss for hand-computed cases under both
  reductions; extreme-logit stability; the full §6 target-rejection
  matrix at the Core boundary; C++-side target-range revalidation;
  probabilities equal to `softmax(logits)` to tolerance; backward values
  equal to `(p − onehot)/N` scaled by upstream; failure atomicity
  including injected allocation failure with no orphaned private buffer;
  the C++ CTest.
- **Risks:** ordering the forward's two allocations so a failure orphans
  one; trusting Python-validated targets at the ABI; computing
  probabilities in a second pass that could disagree with the loss.
- **Dependencies:** E3, E4 (the shared stable-math machinery).
- **Non-goals:** any graph construction; any public tensor operation;
  the loss module.
- **Shipped (E5).** Exactly the above; see §4.5's status block for the
  implementation notes. Files created: `cpp/tests/test_cross_entropy.cpp`,
  `tests/test_native_cross_entropy_core.py`. Files touched:
  `cpp/src/classification.cpp` (the two internal kernels, the two guarded
  exports, and their shared `cross_entropy_common_error` /
  `reject_target_range` validators),
  `cpp/include/tf_classification_internal.h` (the two internal
  declarations and the `kCrossEntropyReduction{Mean,Sum}` codes),
  `cpp/CMakeLists.txt` (a tenth CTest target),
  `src/tensorforge/backends/cpp.py` (the two ctypes declarations,
  `_CHECKED_KERNELS`, `NativeTensorCore.cross_entropy_forward` /
  `cross_entropy_backward`, the private `_normalize_reduction`,
  `_prepare_class_targets`, `_require_target_copy`, and
  `_CrossEntropyForwardResult` helpers, `TENSOR_CORE_OPS`, and
  `UNSUPPORTED`). None of the named risks materialized: the forward's two
  allocations are closed in reverse order on any failure (proved with an
  injected failure of the *second* allocation), C++ revalidates every
  target index it dereferences, and the probabilities are produced by the
  *same* pass that produces the loss rather than a second one.
  `src/tensorforge/experimental/native_tensor.py` was **not** touched:
  E5 adds no tensor operation and no autograd behavior. No module, no
  metric, no `NATIVE_METRICS`, no public `max`/`argmax`/`gather`/
  division, no integer tensor, no benchmark, no example, no schema
  change.

### E6 — Differentiable `NativeTensor` cross-entropy — **complete**

- **Objective:** `NativeTensor.cross_entropy(targets, reduction="mean")`
  — the graph node over the E5 Core contract.
- **Layer:** `NativeTensor` autograd.
- **Expected files:**
  `src/tensorforge/experimental/native_tensor.py`;
  `src/tensorforge/backends/cpp.py` (inventories);
  `tests/test_native_cross_entropy.py`.
- **Required tests:** scalar output; single `(logits,)` parent;
  finite-difference gradients under both reductions; explicit scalar
  upstream scaling; **no** logits version snapshot (post-forward mutation
  of a direct parameter still differentiates); saved-probability lifetime
  across one-shot backward, `retain_graph=True`, `close()`, abandonment,
  and a failed retryable backward; targets mutated after forward change
  nothing; no-grad forward releases the probabilities immediately;
  stable-framework parity.
- **Risks:** the private buffer outliving or predeceasing the graph;
  double release; retaining the logits unnecessarily and implying a
  version guard that does not exist.
- **Dependencies:** E5.
- **Non-goals:** the loss module; the metric; training.
- **Shipped (E6).** Exactly the above; see §4.5's E6 status block for the
  implementation notes and §7 for the now-live lifetime contract. Files
  touched: `src/tensorforge/experimental/native_tensor.py` (the one new
  method), `src/tensorforge/backends/cpp.py` (`"cross_entropy"` into
  `AUTOGRAD_OPS`, plus the inventory comments), the new
  `tests/test_native_cross_entropy.py`, the Phase-D/E milestone-boundary
  tests, and the status documents. **No C++ source, header, C ABI export,
  ctypes signature, CMake target, or C++ test changed** — E6 adds no
  numerical capability. No module, no metric, no `NATIVE_METRICS`, no
  `reduction="none"`, no public saved probabilities or target objects, no
  integer tensor, no benchmark, no example, no schema change (the native
  checkpoint format stays version 1).

### E7 — `NativeCrossEntropyLoss` and reporting-only `native_accuracy`

- **Objective:** the public classification surface.
- **Layer:** native modules + a reporting helper.
- **Expected files:**
  `src/tensorforge/experimental/native_cross_entropy_loss.py`;
  `src/tensorforge/experimental/native_metrics.py`;
  `src/tensorforge/experimental/__init__.py` (exports);
  `src/tensorforge/backends/cpp.py` (`NATIVE_LOSSES`, the new
  `NATIVE_METRICS`, `backend_info()`);
  `tests/test_native_cross_entropy_loss.py`;
  `tests/test_native_metrics.py`.
- **Required tests:** exact reduction validation; parameter-free,
  buffer-free, empty `state_dict()`; train/eval independence; input
  rejection matrix; composition in a `NativeSequential` model; the metric
  returning a Python `float`, building no graph, touching no gradient,
  leaving its input open; the metric's target validation; the inventory
  placements of §11 asserted directly.
- **Risks:** registering the metric in the wrong inventory; the metric
  quietly acquiring gradient side effects; describing the metric as a
  native kernel.
- **Dependencies:** E6.
- **Non-goals:** BCE, `NLLLoss`, class weights, `ignore_index`, label
  smoothing, soft labels.

### E8 — Deterministic native classification training and exact checkpoint resume

- **Objective:** prove the stack trains and resumes, end to end, with no
  new capability.
- **Layer:** example + integration tests.
- **Expected files:** `examples/native_classification_training.py`;
  `tests/test_native_classification_training.py`;
  `docs/examples.md` / `README.md` / matrix references.
- **Required tests:** deterministic seeded loss trajectory reproduced
  bit-identically across runs; accuracy improving; a run interrupted
  mid-training, checkpointed (model **and** optimizer state) and resumed
  into a fresh model/optimizer pair reproducing the uninterrupted run
  **exactly**; checkpoint **format version 1** unchanged and no new keys;
  NumPy tripwire over the training step.
- **Risks:** hiding a real defect behind a too-easy synthetic task;
  letting the reporting metric run inside the tripwire region without
  saying so.
- **Dependencies:** E7.
- **Non-goals:** real datasets; data loaders; schedulers; any schema
  change.

### E9 — Native classification benchmark characterization

- **Objective:** honest measurement of what Phase E built.
- **Layer:** benchmarks only (measurement-only, no behavior).
- **Expected files:** `benchmarks/benchmark_native_classification.py`;
  a benchmark section in the docs.
- **Required tests:** the harness's correctness gate; `--smoke` and JSON
  modes; **no test asserts a speed**.
- **Risks:** drifting into optimization work; publishing numbers as
  claims rather than characterizations.
- **Dependencies:** E8.
- **Non-goals:** CPU optimization; kernel tuning; threading; SIMD.

### E10 — Phase-E integration, sanitizer validation, documentation reconciliation, and closure

- **Objective:** close the phase with **no new numerical behavior**.
- **Layer:** cross-cutting tests and documentation.
- **Expected files:** `tests/test_native_phase_e.py`; the support matrix,
  roadmap, this document's status sections, `README.md`,
  `backend_experiments.md`, and the durable doc guardrails.
- **Required tests:** cross-cutting integration (the full classification
  model stack, shared graphs, the versioning contracts meeting in one
  backward, saved-resource lifetime under stress, state/checkpoint
  integration, cross-layer failure atomicity, the capability boundary);
  ASan/UBSan validation of the classification kernels under Clang on
  Linux; a practical LeakSanitizer pass over the native CTests.
- **Risks:** slipping a behavior change into a closure milestone;
  leaving a status surface claiming Phase E is unimplemented.
- **Dependencies:** E9.
- **Non-goals:** anything in §1's exclusion list; Phase F planning.

---

## 16. Why this order

The ladder is **architectural, not chronological convenience**:

- **Stable scalar math first** (E1, E2). `exp` and `log` are the smallest
  operations that introduce the phase's two backward archetypes —
  saved-output and live-input — and every later operation depends on one
  of them.
- **Probability transforms next** (E3, E4). They need `exp`'s math and
  `log`'s semantics, and they establish the axis factorization and the
  contiguous-only classification ABI that cross-entropy reuses.
- **Fused loss Core contract next** (E5). The numerically hardest piece
  lands where it can be tested in isolation — autograd-unaware,
  forward-only, with no graph to confuse a failure.
- **Graph integration after the Core contract** (E6). The saved-resource
  lifetime is subtle enough to deserve a milestone where the numerics are
  already known correct.
- **Public loss and metric surface after operation correctness** (E7).
  A module is a thin layer; it should never be where a numerical bug is
  found.
- **End-to-end proof after the public stack** (E8). A training proof is
  only meaningful once the surface it exercises is final.
- **Benchmark after behavior is complete** (E9). Measuring a moving
  target teaches nothing.
- **Behavior-free closure last** (E10). The phase ends with hardening,
  validation, and documentation — never with new capability.

---

## 17. Phase-E completion criteria

Phase E is complete when **all** of the following hold:

1. `exp`, `log`, `softmax`, `log_softmax` are differentiable
   `NativeTensor` operations, and `cross_entropy` is a fused
   differentiable operation from raw logits.
2. `NativeCrossEntropyLoss` and `native_accuracy` are exported, tested,
   and registered in the correct inventories; no Phase-E name remains in
   `UNSUPPORTED`.
3. The §5 backward/versioning matrix is proved by tests, not asserted by
   prose.
4. The §6 target contract and the §7 saved-probability lifetime contract
   are proved, including the failed-retryable-backward and abandoned-graph
   paths.
5. A deterministic native classification training run learns and resumes
   **exactly** from a checkpoint, with **format version 1** unchanged.
6. Benchmarks characterize the stack with no performance assertion.
7. ASan/UBSan (and a practical LeakSanitizer pass) find nothing in the
   classification stack.
8. Every status surface — support matrix, roadmap, README, summary,
   architecture, and the backend registry — agrees on what shipped.
9. Phase A–D behavior, the stable framework, the checkpoint schema, and
   the strict stable/native separation are all unchanged.
