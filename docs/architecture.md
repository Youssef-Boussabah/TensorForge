# Architecture

TensorForge is a from-scratch deep learning and ML systems framework
with **two strictly separate lines**. The **stable Python framework**
reimplements how a framework like PyTorch works under the hood in
Python + NumPy, with every piece kept deliberately small and readable.
The **experimental native line** is a
ctypes-loaded C++ CPU runtime with its own explicit tensor, a
Python-managed native autograd graph, and a native training stack —
merged into `main` and living in its own explicit namespaces
(`tensorforge.backends`, `tensorforge.experimental`), which the stable
framework never imports
(see [backend_experiments.md](backend_experiments.md) and the
[native support matrix](native_support_matrix.md)).

## Package layout

```
src/tensorforge/
  tensor.py          Tensor + the stable autograd engine
  data.py            batches() and train_test_split()
  serialization.py   save/load parameters, save/load checkpoints
  nn/                everything model-related
    parameter.py     Parameter (a trainable Tensor)
    module.py        Module base class: parameters, buffers,
                     train/eval mode, state_dict, summary
    linear.py        Linear layer
    activations.py   ReLU, Sigmoid, Tanh
    dropout.py       Dropout
    batchnorm.py     BatchNorm1d
    layernorm.py     LayerNorm
    conv.py          Conv2d (NCHW image-shaped inputs)
    pool.py          MaxPool2d
    flatten.py       Flatten
    sequential.py    Sequential container
    losses.py        mse_loss, cross_entropy, binary_cross_entropy
    metrics.py       accuracy, binary_accuracy, evaluators
  optim/             optimizers and training control
    sgd.py           SGD
    adam.py          Adam
    clip.py          clip_grad_norm, clip_grad_value
    lr_scheduler.py  StepLR
  backends/          the explicit backend boundary (never imported
                     by the stable framework)
    cpp.py           ctypes loader + NativeStorage / NativeTensorView /
                     NativeTensorCore + raw kernel entry points
    registry.py      get_backend()/available_backends() — explicit only
    numpy_backend.py, native_backend.py
  experimental/      the native training line (explicit import only)
    native_tensor.py     NativeTensor + Python-managed native autograd
    native_parameter.py  NativeParameter + versioning + registry
    native_module.py     NativeModule + state_dict/load_state_dict
    native_linear.py     NativeLinear
    native_relu.py       NativeReLU
    native_flatten.py    NativeFlatten (Phase D)
    native_conv2d.py     NativeConv2d (Phase D)
    native_maxpool2d.py  NativeMaxPool2d (Phase D)
    native_sequential.py NativeSequential
    native_mse_loss.py   NativeMSELoss
    native_cross_entropy_loss.py  NativeCrossEntropyLoss (Phase E)
    native_metrics.py    native_accuracy (Phase E, reporting only)
    native_sgd.py        NativeSGD
    native_adam.py       NativeAdam (persistent moment state)
    native_optimizer_state.py  optimizer state_dict schema helpers
    native_checkpoint.py save/load_native_checkpoint (pickle-free NPZ)
cpp/                 C++ kernel sources + build.py (nothing compiled
                     is checked in; CI builds from source)
examples/            runnable training scripts (stable + native)
scripts/             smoke_cpp_backend.py — hard-failing backend check
benchmarks/          measurement-only characterization harnesses
tests/               pytest suite (one test file per feature area)
```

## How the pieces fit together

**Tensor** (`tensor.py`) is the foundation. It wraps a NumPy array and
records enough information to run backpropagation. Everything else in
the framework is built on top of Tensor operations — see
[autograd.md](autograd.md) for how that works.

**Parameter** is a Tensor subclass that always has
`requires_grad=True`. Being a distinct class is what lets the rest of
the framework find trainable state by type.

**Module** is the base class for layers and models. It discovers
Parameters by walking its own attributes (including child modules and
lists of modules), which gives every model `parameters()`,
`state_dict()`, `summary()`, and train/eval mode for free — no
registration calls needed. Modules can also declare non-trainable
*buffers* (like BatchNorm's running statistics) that travel with
`state_dict()` but are never optimized.

The recursive walks are **identity-aware and cycle-safe**: a shared or
tied Parameter/buffer is yielded once (first-encountered name wins), and
a module graph with shared children or a reference cycle terminates
instead of recursing forever. `train(mode)` requires a real `bool` (so an
accidental `model.train("eval")` raises rather than silently staying in
training mode), and `eval()` is `train(False)`. `load_state_dict()` is
**atomic — validate then commit**: every key, value type, and shape is
checked and every replacement prepared before any live Parameter or
buffer is mutated, so a failure (e.g. a later shape mismatch) leaves the
whole model unchanged with Parameter identities intact, and the commit is
rollback-guarded.

**Layers** (Linear, activations, Dropout, BatchNorm1d, LayerNorm,
Conv2d, Flatten) are small Module subclasses. The two normalizations
differ in what they average over: BatchNorm1d normalizes each feature
*across the batch* (so it keeps running statistics and changes behavior
in eval mode), while LayerNorm normalizes each sample over *its own*
trailing dimensions (no buffers, identical in train and eval mode). Each one implements `forward()`
using Tensor operations, so gradients flow through them automatically
(Conv2d and MaxPool2d are the exceptions — fused ops with explicit
backwards, because composing them from elementwise ops would be
unreadable). Conv2d and MaxPool2d work on image-shaped `(batch,
channels, height, width)` input; MaxPool2d has no Parameters at all and
its backward routes each window's gradient only to the position that
won the max. Flatten bridges image-shaped activations back to the
`(batch, features)` shape Linear expects. Sequential chains layers so
the output of one feeds the next.

**Losses** are either plain Tensor expressions (`mse_loss`) or fused
operations with a hand-written backward pass where numerical stability
demands it (`cross_entropy`, `binary_cross_entropy`).

**Optimizers** (SGD, Adam) are plain classes, not Modules. They hold a
list of Parameters and update `param.data` from `param.grad` in
`step()`. They skip parameters with no gradient and frozen parameters
(`requires_grad=False`). Gradient clipping and the StepLR scheduler
sit alongside them in `optim/`.

**Examples** tie it all together. Each one is a standalone script with
a `train()` function (so tests can import and run it) and a `main()`
that prints progress. See [examples.md](examples.md).

## The experimental native line

The native line rebuilds the same ideas against real memory, one
explicit layer at a time:

- **`NativeStorage`** owns a raw native allocation with an explicit
  `close()` lifetime.
- **`NativeTensorView`** adds shape/strides/offset layout over a
  storage — views are metadata, not copies.
- **`NativeTensorCore`** is the forward runtime: elementwise ops with
  NumPy-style broadcasting, `relu`, a 2-D `matmul`, `sum`/`mean`
  reductions, reshape/transpose/narrow views, contiguous
  materialization, and float64/cpu dtype/device metadata — all
  executing in C++ kernels behind a plain C ABI, loaded with ctypes.
  The core and the kernels are completely autograd-unaware.
- **`NativeTensor`** wraps one core and adds the **Python-managed
  native autograd graph**: every operation listed in the backend's
  `AUTOGRAD_OPS` registry is differentiable (elementwise math, `matmul`,
  reductions, the view ops, the Phase-D `conv2d`/`maxpool2d`
  primitives, and the Phase-E stable math — `exp`, `log`, the fused
  `softmax` and `log_softmax`, and the fused `cross_entropy` from raw
  logits — the registry, mirrored in the
  [native support matrix](native_support_matrix.md), is the exact list),
  with gradient un-broadcasting, view backwards (including a native
  scatter for `narrow`), one-shot graph release with `retain_graph`
  opt-in, and failure rollback. Backward math runs at the core level, so
  the graph never leaks into C++.
- **The native training stack (Phase C, complete)** builds on that:
  `NativeParameter` (graph-free trainable leaves with value versioning,
  a controlled mutation path, and stale-graph detection), `NativeModule`
  (registration by assignment, recursive traversal, atomic in-memory
  `state_dict`/`load_state_dict`), `NativeLinear` / `NativeReLU` /
  `NativeSequential`, `NativeMSELoss`, `NativeSGD` and the adaptive
  `NativeAdam` (persistent native moment state, per-parameter bias
  correction, explicit `close()` lifetime), in-memory optimizer
  `state_dict`/`load_state_dict`, and pickle-free native checkpoint
  files (`save_native_checkpoint`/`load_native_checkpoint`) with
  deterministic in-memory and file resume — and, with **Phase D (the
  native CNN stack) complete**, `NativeFlatten`, the differentiable
  `conv2d` operation with its trainable `NativeConv2d` module, and the
  `maxpool2d` operation (private saved winners, scatter backward) with its
  parameter-free `NativeMaxPool2d` module, proven together by a
  deterministic CNN training + exact checkpoint-resume run
  (`examples/native_cnn_training.py`) — proven end to end by
  `examples/native_mlp_training.py` and
  `examples/native_checkpoint_resume.py`.
- **The native classification stack (Phase E, complete)** adds the
  stable math the training stack needed to classify: differentiable
  `exp` and `log` (the phase's two backward archetypes — `exp` reads its
  saved output and records no parameter version, `log` rereads the live
  input and version-guards a direct parameter), the fused, numerically
  stable `softmax` (maximum shift) and `log_softmax` (its own
  log-sum-exp kernel, never `softmax().log()`), and the fused
  `cross_entropy` from raw logits — a Core contract plus one autograd
  node whose graph-owned private saved probabilities drive the backward,
  so the logits are never reread. On top of them sit the stateless
  `NativeCrossEntropyLoss` module and the deliberately **reporting-only**
  `native_accuracy`, which converts through the explicit public
  `to_numpy()` boundary and builds no graph. Proven by
  `examples/native_classification_training.py`, whose
  checkpoint-interrupted run resumes exactly.
- **Native normalization (Phase F) is complete (F0–F9).** The
  contract for `NativeLayerNorm`, `NativeBatchNorm1d`, and
  `NativeBatchNorm2d` is locked in
  [native_normalization_design.md](native_normalization_design.md)
  (milestone F0), including the decision to **compose** normalization
  from existing native operations rather than add any kernel, C ABI
  export, or `NativeTensorCore` method, and the rule that a live mutable
  running-statistics buffer is never captured as a rereadable graph
  operand. Milestone F1 shipped the private atomic native-buffer
  state transaction that contract calls for (`_native_state.py`, now the
  single implementation behind `load_state_dict`) — state management
  only. **Milestone F2 shipped `NativeLayerNorm`** — the first native
  normalization module, stateless and differentiable, composed entirely
  from existing native operations (`mean`/`subtract`/`multiply`/`add`/
  `sqrt`/`reciprocal`, `sqrt(var + eps)`, population variance) with no
  kernel, ABI symbol, Core method, custom backward, or `NativeTensor`
  normalization operation; `"NativeLayerNorm"` is in `NATIVE_MODULES` and
  `"layernorm"` has left `UNSUPPORTED`. **Milestone F3 shipped
  `NativeBatchNorm1d`** — the first *stateful* native numerical module:
  `(N, C)` batch normalization with differentiable current-batch
  statistics, persistent native `running_mean`/`running_var` buffers
  advanced graph-free by one **atomic two-buffer transaction** over the
  F1 primitive (both identities preserved, no parameter version moved),
  and evaluation from **independent graph-free snapshots** of those
  buffers, so the §7 rule above holds by construction: no running-buffer
  mutation — a training step, a buffer-only `load_state_dict()`, or a
  buffer-only `load_native_checkpoint()` — can change an earlier eval
  graph's gradient. (A full checkpoint load also replaces `gamma`/`beta`
  and therefore still stales such a graph through the unchanged
  parameter-version rule, which BatchNorm neither bypasses nor weakens.) It is composed
  from the same existing operations, so again no kernel, ABI symbol,
  Core method, custom backward, or `NativeTensor.batch_norm` operation
  exists, and the native checkpoint format stays at version 1;
  `"NativeBatchNorm1d"` is in `NATIVE_MODULES`, while `"batchnorm"`
  stayed listed as unsupported until the NCHW shape shipped.
  **Milestone F4 shipped `NativeBatchNorm2d`** — NCHW `(N, C, H, W)`
  batch normalization reducing over N, H, and W, so each channel gets
  one population mean and one population variance over `N * H * W`
  values. It is built on the **same** shared private implementation and
  declares nothing but its rank, its reduction axes, its `(1, C, 1, 1)`
  broadcast layout, and the channels-last permutation its rank-1
  `gamma`/`beta` need: rank-1 parameters broadcast from the *trailing*
  axis, so the **activation** is transposed for the affine step and back
  again rather than the parameters being reshaped — which keeps `gamma` a
  direct versioned `multiply` operand and preserves the stale-value guard
  exactly. Running buffers stay `(C,)`. All three normalization modules
  are now in `NATIVE_MODULES` and the exports, and `"batchnorm"` has
  **left** `UNSUPPORTED`. That completes the normalization **module**
  surface. **Milestone F5 shipped the state/checkpoint/graph-safety
  hardening** — a focused `tests/test_native_normalization_state.py` plus
  narrow additions to the generic buffer and checkpoint suites — proving
  §7–§10 by executable test: canonical dotted buffer keys, independent
  state snapshots, strict/non-strict loads, exact never-casting metadata
  validation, mixed parameter/buffer transaction atomicity, buffer
  identity across state and checkpoint loads, exact eval-output
  reproduction, the buffer-only-versus-full stale-graph distinction, the
  save/corrupt-load failure boundaries, eval-graph snapshot safety under
  `retain_graph` and a failed retryable backward, and explicit
  parameter/buffer closure returning storage to baseline. F5 is **tests
  and documentation only** — no numerical behavior, no new public
  capability, and the checkpoint format stays version 1. **Milestone F6
  shipped the deterministic normalized training and exact-resume proof**
  (`examples/native_normalization_training.py`): a
  `Linear → BatchNorm1d → ReLU → LayerNorm → Linear` regressor trained for
  24 deterministic `NativeAdam` steps with `NativeMSELoss`, whose two
  uninterrupted runs are bit-identical and whose interrupted checkpoint
  resume into a fresh model/optimizer pair reproduces the remaining loss
  suffix, every parameter, the NativeAdam state, both BatchNorm
  `running_mean`/`running_var`, the final training-step prediction, and
  the final evaluation-mode output exactly — one example and its
  integration test, no capability or schema change, format version 1
  unchanged, training flags runtime-only. **Milestone F7 shipped the
  honest benchmark characterization**
  (`benchmarks/benchmark_native_normalization.py`): nine cases — both
  LayerNorm directions, all three BatchNorm1d paths, all three
  BatchNorm2d paths, and one complete F6-style normalized training step —
  each **correctness-gated before any timing**, six measured against
  `stable_tensorforge` equivalents on identical state and three (the
  BatchNorm2d shapes) labelled `native_only` because the stable line has
  no public `BatchNorm2d` to time against, though those keep a rigorous
  NumPy NCHW and transformed-oracle correctness gate. Medians are
  reported with min, max, and spread after warm-up, `--smoke`/`--json`
  modes exist, and **no result file is written, no speed is asserted, no
  timing number is committed, and no CI job asserts a duration** —
  measurement only, no capability. **Milestone F8 shipped the
  cross-cutting integration and semantic guardrails**
  (`tests/test_native_phase_f.py`): one integrated `Conv2d → BatchNorm2d
  → ReLU → MaxPool2d → Flatten → Linear → BatchNorm1d → ReLU → LayerNorm
  → Linear` classifier over raw logits and the fused loss, trained by
  `NativeAdam` and resumed **exactly** from one version-1 checkpoint —
  all four running-statistic buffers, the final training logits, and the
  evaluation-mode logits, predictions, and accuracy included. It also
  proves BatchNorm snapshots, MaxPool2d winners, and cross-entropy
  probabilities coexisting in one eval graph and releasing exactly once;
  buffer mutation leaving an earlier graph valid while parameter mutation
  correctly stales it; the versioning archetypes; shared and frozen
  parameters; a non-contiguous NCHW input; strict stable/native
  separation; and each failure boundary tested honestly — BatchNorm
  transactions are **per module**, and one whole training step is *not*
  presented as globally transactional. Tests and documentation only, no
  capability. **Milestone F9 closed the phase**: fresh Windows Release
  and Debug builds each passing the full existing 10-test CTest suite
  with zero project warnings and the active runtime proved to stay
  Release; a fresh Clang 18.1.3 ASan+UBSan build with instrumentation
  proved by `nm -D` (22 `__asan*`, 13 `__ubsan*`) and by the library's
  refusal to load without the sanitizer runtime; 10/10 sanitized native
  CTests with leak detection enabled; 1,968 sanitized
  normalization-focused Python tests with zero ASan and zero UBSan
  diagnostics; the F6 example and the F7 benchmark smoke path clean
  under the sanitized library; and a practical LeakSanitizer lifecycle
  returning native live storage **exactly** to baseline with no
  TensorForge-attributable leak frame and no suppression file —
  validation and documentation only, adding no numerical capability. All
  of Phase F (F0–F9) has therefore shipped.
- **Native RNG and Dropout (Phase G) is complete (G0–G10).** Random state
  is **Python-managed** and native random kernels stay **stateless**,
  receiving the complete key (an unsigned 64-bit seed plus a call index)
  for one operation. `NativeGenerator` holds exactly that state, owns no
  native storage, and never consults a global or process-wide random
  source. `cpp/src/random.cpp` carries the locked `tensorforge.splitmix64`
  derivation and the inverted-Dropout forward kernel behind one guarded C
  ABI export, `tf_core_dropout_forward`, with known-answer vectors
  asserted identically from C++ and Python. Above it sit the
  differentiable `NativeTensor.dropout(p, *, generator)` — a
  **required, keyword-only** generator, no default or global stream —
  whose backward is the existing `multiply` over a **graph-owned**
  multiplier mask (the fourth member of the saved-resource family beside
  MaxPool2d winners, BatchNorm eval snapshots, and cross-entropy
  probabilities), and the `NativeDropout` module, which registers its
  generator as a **fourth** state category alongside parameters, buffers,
  and child modules. Exactly one generator call is consumed per
  *successful* stochastic forward, and none on any failure, in evaluation,
  at `p == 0`, or in backward. Native checkpoint **format version 2**
  persists every registered generator's state **and** its alias topology,
  restoring in place so identity and sharing survive, with version 1 still
  loadable under its locked rules and the whole load one synchronous-atomic
  transaction. **Milestone G10 closed the phase**: fresh Windows Release
  and Debug builds each passing the full 11-test CTest suite with zero
  project warnings and the active runtime proved to stay Release; a fresh
  Clang 18.1.3 ASan+UBSan build with instrumentation proved by `nm -D`
  (22 `__asan*`, 14 `__ubsan*`, beside 51 exported `tf_*` symbols) and by
  the library's refusal to load without the sanitizer runtime; 11/11
  sanitized native CTests with leak detection enabled; 3,166 sanitized
  Python tests with zero ASan and zero UBSan diagnostics; the G7 example
  and the G8 benchmark clean under the sanitized library; and a practical
  LeakSanitizer lifecycle returning native live storage **exactly** to
  baseline with no TensorForge-attributable leak frame and no suppression
  file. Only then did `dropout` leave `UNSUPPORTED`, which now reads
  `("float32", "cuda", "amp")` — a claim scoped to the **experimental
  native float64 CPU** line, never to the stable framework, which keeps
  its own separate `Dropout`.
- **Native CPU performance (Phase H) has begun; H0, H1, H2, H3, H4, and H5
  are complete.**
  H0 is architecture, profiling, and baseline work and **shipped no
  optimization**: the contract in
  [native_cpu_performance_design.md](native_cpu_performance_design.md),
  the unified measurement harness
  `benchmarks/benchmark_native_cpu_performance.py`, its contract tests,
  and documentation reconciliation. Every kernel in `cpp/src/` is still
  the deliberately plain reference loop Phase G left behind, and no
  numerical capability, kernel, C ABI symbol, ctypes declaration, Core
  method, operation, module, export, capability-registry value, dtype,
  device, or checkpoint version changed — the format stays version 2 with
  versions 1 and 2 supported, so **Phase G remains the latest completed
  phase**. What H0 added is the ability to *attribute* cost: the harness
  measures each of the layers in the execution path below separately,
  gates correctness before timing, and publishes no ratio where no honest
  equivalent exists. Its ranked evidence, and the explicitly conditional
  H2–H8 ladder derived from it, live in the design document; a memory
  pool, scratch allocation, SIMD, threading, and BLAS are all currently
  rejected on that evidence, with the criteria that would reopen each
  recorded rather than an answer invented.
- **Milestone H1 — the output-allocation contract — is complete**, and is
  the first Phase-H change to production code. Native storage was
  value-initialized on construction, a full write pass over a buffer most
  kernels then overwrite completely; H1 removed that fill wherever a
  kernel *provably* writes every destination element before reading any
  of it. It added one production C ABI symbol,
  `tf_storage_create_uninitialized`, sharing one body with the unchanged
  zero-initializing `tf_storage_create` so the two cannot drift apart on
  size validation, allocation failure, error state, ownership,
  destruction, or live-storage accounting. The zero-initializing path
  remains the default; each call site opts in explicitly against a
  per-kernel audit table, with `sum`/`mean` and `narrow_backward`
  deliberately **rejected** because the first accumulates into its output
  and the second leaves untouched zeros that *are* the gradient.
  Completeness is proved by deterministic **poison** tests rather than by
  ASan or UBSan, which do not detect uninitialized-value reads and stay
  separate from this proof. The poison belongs to the test suite alone:
  it wraps the private uninitialized allocation helper, fills the real
  storage the real constructor returned, and hands that same storage to
  the real operation, so **no poison control exists in the shipped
  library or the installed Python backend** — no exported hook, no
  thread-local flag, no environment variable, no global mode.
  H1 is bit-identical and moved no capability, dtype, device,
  registry value, or checkpoint version; it added **no** public
  empty-tensor API, and `tf_storage_create_uninitialized` is the only C
  ABI symbol it added, taking the library to **52** exported `tf_*`
  symbols.
- **Milestone H2 — native matmul memory access — is complete**, and is
  the first Phase-H milestone to change how a numerical kernel executes.
  `tf_core_matmul` now ships **two** compute paths behind the same
  unchanged export: `tf::matmul_generic_strided`, the pre-H2 `i`-`j`-`k`
  triple loop kept verbatim as the **retained generic reference path**
  (§8.3 of the design), and `tf::matmul_row_sweep`, an `i`-`k`-`j` sweep
  over four destination rows at a time whose innermost loop walks a row
  of the right operand and a row of the output sequentially instead of a
  column. **Cache blocking was measured against 22 blocked variants and
  rejected** — an unblocked full-width sweep was faster at every
  non-trivial size — so H2 shipped the simpler design and recorded the
  negative result. The choice is made inside the kernel from the stride
  metadata it already receives: unit column stride on the right operand,
  a non-empty inner dimension, and at least 8 result columns select the
  row sweep; everything else falls to the generic path, which is the loop
  order those cases already suit. Selection is total, pure, deterministic,
  and independent of pointer values, alignment, timing, environment
  variables, and CPU-feature probes; a failed precondition is a fallback,
  never an error. **No exported C ABI symbol was added** — the count
  stays **52** — and no kernel selector, block-size setter, dispatch
  tracer, or public dispatch control exists; the two kernels and the
  predicate are hidden-visibility C++, which is exactly why the native
  test target compiles `matmul.cpp` in rather than linking the library.
  Accumulation order is unchanged per output element. The resulting
  agreement is stated in four parts rather than as a blanket
  bit-identity claim: the accumulation sequence is preserved exactly,
  **every non-NaN result is bit-identical** (asserted as raw bit
  patterns), NaNs occur in exactly the same positions on both paths and
  are always quiet, and the **payload bits of a NaN result are outside
  TensorForge's numerical contract** and may differ — they follow from
  the compiler's instruction operand ordering, which follows from the
  loop order, and ten source-level formulations failed to close that gap
  without reverting the optimization. Every committed loss trajectory and
  every exact-resume proof runs on finite data, so part two covers all of
  them. H1's uninitialized-output contract holds on both — the generic path never
  reads the destination, and the row sweep assigns every element of every
  row in its `k == 0` pass before anything accumulates into it. No
  capability, dtype, device, registry value, or checkpoint version moved.

- **Milestone H3 — native metadata and dispatch efficiency — is
  complete**, and is the first Phase-H milestone that is **Python-only**:
  no C++, no C ABI symbol, no ctypes declaration, and no kernel changed,
  so the library still exports exactly **52** `tf_*` symbols. It removed
  redundant metadata *re-validation* from the path to a kernel. Before
  H3 one `shape_info` call ran `_as_int_tuple` **four** times over a
  tuple that was fully validated after the first pass and computed the
  row-major strides **twice**, and `NativeTensorCore.zeros` validated the
  caller's shape a second complete time; instrumented counts put that at
  **815** `_as_int_tuple` calls per MLP training step. The architecture
  is three pieces. **One normalization boundary**: the private
  `_normalized_layout` performs exactly the checks `shape_info` always
  performed, in the same order and with the same messages, and normalizes
  the shape once, with everything derived from it computed by private
  `_checked` primitives that validate nothing because nothing is left to
  validate — each public helper is now its own validation plus the
  matching primitive, so the two cannot disagree. **Two view
  constructors, one binding**: `NativeTensorView` keeps its normalizing
  public constructor and gains a private `_from_validated` that skips
  *only* that normalization, with both funnelling through a shared
  `_bind` that still performs the storage open check and the full
  reachable-offset bounds check; the element count and contiguity flag
  are derived *inside* the private constructor rather than passed to it,
  so an inconsistent pair cannot be supplied — which is why this is a
  separate constructor and not a misusable `validated=True` flag.
  **Per-view layout arrays**: the `int64` shape/stride arrays the strided
  C ABI takes are memoized lazily and read-only. That memoization cannot
  go stale, because a view's layout is assigned exactly once in `_bind`
  and every layout-changing operation returns a *new* view — so no
  invalidation is ever required and none exists. Nothing global was
  introduced and **no validation was removed**: every rejection still
  happens with the same exception type, message, and ordering. Measured:
  view construction 3.2x, `_as_int_tuple` per MLP step 815 -> 149, an MLP
  training step 1.43x, a CNN step 1.29x, a normalized step 1.51x — with
  **no measurable change on large kernel-bound matmul or elementwise
  work**, which is reported as such. No capability, dtype, device,
  registry value, or checkpoint version moved.

- **Milestone H4 — native optimizer step efficiency — is complete**, and
  is **Python-only** like H3: no C++, no C ABI symbol, no ctypes
  declaration, and no kernel changed, so the library still exports
  exactly **52** `tf_*` symbols. It is the first Phase-H milestone whose
  subject is a *training-stack* component rather than the tensor runtime.
  Re-instrumented on the current code, `NativeAdam.step()` cost **27
  native allocations per parameter**, **ten of them one-element**: eight
  broadcast scalar coefficients (`beta1`, `1 - beta1`, `beta2`,
  `1 - beta2`, both bias corrections, `eps`, `lr`) plus two `reciprocal`
  outputs taken on one-element tensors. H4 changed three things. **The
  scalar coefficients are built once per step, not once per parameter** —
  a private per-step `_StepConstants` holder, keyed by `(dtype, device)`,
  with the bias corrections cached per step *counter* so a parameter that
  skipped earlier steps still gets its own; the holder allocates nothing
  until the first entry asks for a coefficient, is released before the
  commit begins, and is never stored on the optimizer, so no scalar
  survives a step or reaches `state_dict()`, a checkpoint, or `close()`.
  `NativeSGD` does the same for its single `lr` scalar — the only change
  its evidence supported. **The bias-correction reciprocal is evaluated
  in Python**, which is an exact substitution rather than a
  reassociation: the kernel *is* `1.0 / x` on the same IEEE-754 binary64
  value and IEEE division is correctly rounded, proved over 20,000+
  values on raw bit patterns. **Temporaries are released at their last
  use** instead of all together at the end of the staged expression.
  Everything is **bit-identical** to the pre-H4 composition, which is
  retained in the test suite and executed natively as the reference; the
  exact operation sequence per staged entry is pinned by test. The
  two-phase contract is untouched — validation still four complete passes
  in the same order, stage mutating nothing, one `copy_value_` and one
  version increment per updated parameter, gradients never written.
  Measured by alternating pre/post subprocess rounds: `NativeAdam.step()`
  **1.58x** at (128, 128), **1.48x** on a four-parameter MLP with a 256²
  weight, 1.21x on a small MLP; a large MLP training step 1.23x, a
  normalized step 1.13x, a CNN step 1.09x; and against
  `tensorforge.optim.Adam` **23.8x -> 19.7x**. Reported as honestly:
  **a (512, 512) parameter, the Dropout training step, and NativeSGD are
  all neutral**, and the machine's control-case noise band is 0.84x-1.26x.
  **Peak live transient bytes during an Adam step fell 2.6-3.0x** and
  per-parameter allocations 27 -> 17, so the time was not bought with
  memory. Six alternatives were measured and rejected, including scalar
  materialization (faster small, slower large) and a persistent scalar
  cache (the forbidden hidden scratch tensor). No capability, dtype,
  device, registry value, or checkpoint version moved, and no public API
  of any kind was added.

- **Milestone H5 — native copy and mutation-transfer efficiency — is
  complete**, and is the first Phase-H milestone since H2 to change C++
  though **not the ABI** (still exactly **52** exported `tf_*` symbols).
  H5 replaced the native line's value-transfer primitive: `_native_copy`
  was `zeros(shape) + core` — two allocations, a zero-fill pass, and an
  elementwise-addition pass — and is now the E3.1 native identity gather,
  `contiguous_copy()`, at one uninitialized allocation and one pass. All
  **ten** call sites of that helper (`copy_value_` staging, both
  `state_dict()` snapshots, both `load_state_dict()` stagings, both
  BatchNorm running-statistic commits, and the
  reshape/transpose/unbroadcast gradient materializations) are pure value
  transfers and were enabled; `_broadcast_back` was **rejected** because it
  is a genuine broadcast, not a copy. Over a fixed 18-pattern IEEE-754
  sweep, **exactly three** patterns moved: the addition normalized `-0.0`
  and quieted both signs of signaling NaN, and the gather preserves all
  three — and **no** NaN payload differed at all, so **H2's matmul
  payload carve-out does not generalize to copies**. The rule H5 states is
  the narrowest coherent one: **a value transfer reproduces its source's
  bits; an operation follows IEEE arithmetic.** One C++ change, inside the
  unchanged export: a metadata-driven second *traversal*
  (`tf::copy_prefers_contiguous`, hidden visibility, total, pure, no
  environment variable or CPU probe) that sweeps a row-major source with
  the flat loop and falls back to the retained odometer otherwise —
  bit-identical **by construction**, since the identity map performs no
  arithmetic, and proved by a new dependency-free CTest (13 to 14). Nothing
  became in-place; every call site still stages, so self-copy, own-storage
  views, own transposes, and sibling views all stay correct, and identity,
  storage, version, gradient, state-transaction, checkpoint, and
  exact-resume behavior are all unchanged. Measured by alternating
  pre/post subprocess rounds (control band 0.96x-1.05x) and by a separate
  pre-H5-library A/B: the traversal alone **2.5x-5.5x** on contiguous
  sources and **0.94x-1.02x** on transposed ones (the unchanged odometer —
  the design's own control); `copy_value_` **2.14x** at (512, 512),
  optimizer `state_dict()` 2.40x, `load_state_dict()` 1.69x, `NativeSGD`
  1.15-1.31x. Reported as honestly: **`NativeAdam.step()`, every training
  step, the normalization running-statistic update, and copies
    below ~16 K elements are
  all neutral**, the latter because two `int64` layout arrays cost ~1.1 us
  each at the ctypes boundary — measured, attributed, and left to a later
  dispatch milestone. Allocations fell everywhere and **no peak rose**:
  `copy_value_` 2 to 1, module state 4 to 2, optimizer state 16 to 8, Adam
  17 to **16** per parameter. The harness gained two cases (26 to 28), the
  ladder was **reordered** (reduction execution moved to H6), and no
  public API, capability, dtype, device, registry value, or checkpoint
  version moved.

The execution path for a native training step is:

```
Python native modules (NativeSequential → NativeLinear → NativeMSELoss)
  → NativeTensor operations + the Python-managed autograd graph
    → NativeTensorCore forward calls
      → ctypes boundary
        → C++ CPU kernels over strided native memory
```

**The separation is absolute and deliberate.** There is no implicit
conversion between `tensorforge.Tensor` and `NativeTensor`, no shared
autograd graph, and no automatic backend dispatch. The native world is
entered only explicitly (`NativeTensor.from_array`, or the
`tensorforge.backends` APIs) and exited only explicitly
(`to_numpy()`), both as copies. Stable and native parameters,
modules, and optimizers reject each other's objects. CUDA does not
exist anywhere in the current architecture — it remains a future
experiment, as does any dispatch integration (see
[dispatch_design.md](dispatch_design.md)).

## Design habits

A few conventions hold across the codebase:

- Forward computation is eager NumPy; only the gradient bookkeeping is
  deferred.
- Comments explain math and autograd reasoning, not obvious Python.
- Randomness is always seedable: helpers take a `seed` argument and
  use their own `np.random.default_rng`. The one deliberate exception
  is unseeded Dropout, which draws from NumPy's global RNG so that
  checkpoint RNG save/restore can make dropout training resume
  bit-for-bit.
- Every feature ships with tests, and training tests assert learning
  (loss decreased, accuracy above a threshold) rather than exact
  floating-point values.
