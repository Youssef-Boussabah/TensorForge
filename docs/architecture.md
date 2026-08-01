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
- **Native CPU performance (Phase H) is complete (H0–H10).**
  Phase H is **complete**: milestones H0 through H10 have all landed, and it is the latest *completed* phase. H10 re-measured the whole phase against a reconstructed and verified H0 baseline (52 cases, **zero checksum mismatches** — every figure compares implementations that produced bit-identical results), resolved the acceleration gate as three documented rejections with measurements (SIMD, threading/OpenMP, BLAS), assessed `tf_core_narrow_backward` and the small-operation boundary floor and implemented neither, ran the full Release/Debug/Linux/sanitizer/lifecycle matrix, and closed the phase. **Every shipped training workload is 1.50×–3.89× faster than at H0**, matmul 4.71×, Conv2d kernels 2.59×–4.64×, reductions 3.78×–5.06×, with no allocation count or memory peak raised anywhere — and across the whole phase **no capability, dtype, device, registry value, public API, checkpoint field, or checkpoint version moved**, with exactly **one** C ABI symbol added (`tf_storage_create_uninitialized`, at H1): 51 → **52**.
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
  MLP step 1.08x are all at or inside the band. **H7 did not make matmul
  faster**, and no reading should say otherwise.

  A second, independent 11-round run reproduced every row: all cases again
  bit-identical, every control again holding (256-cubed matmul 0.98x, 8-cubed
  matmul 1.04x, contiguous 16x16 `add` 1.08x, 512x512 `copy` 1.01x), and every
  training step again improving, with individual ratios moving by roughly the
  control band's width in both directions (a 1-element `sum` 2.12x against
  1.94x, `NativeSGD` 1.26x against 1.13x, `NativeAdam` at (128, 128) 1.08x
  against 1.14x). The figures quoted are the first run's; the second is
  recorded in the design so no single number is read as more precise than the
  method supports.

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

- **Milestone H8 — native elementwise traversal and composed allocation
  efficiency — is complete**, the fourth Phase-H milestone to change C++
  and, like H2, H5, and H6, **not the ABI**: still exactly **52** exported
  `tf_*` symbols. It entered with **two** candidate tracks and an explicit
  instruction not to force both into production; the measurement kept one
  as a large result, kept a narrow piece of the other, and rejected the
  rest. **Track A (elementwise traversal) is the milestone; Track B
  (composed normalization allocation) is a memory result, reported as
  timing-neutral.**

  The cost was **decomposed rather than assumed**. The odometer costs
  **1.60x-6.42x** the flat loop on identical contiguous data; **all
  broadcasting is on the odometer**, because no broadcast fast path
  exists; and a guarded standalone split the odometer's cost four ways at
  `(256, 256)` contiguous `add` — shipped **123.5 us**, templating alone
  **81.3 us (1.52x)**, collapsing alone **63.6 us (1.94x)**, **both
  together 11.5 us (10.7x)**. Neither change is worth much alone; only
  their combination lets the compiler emit a vector loop. The same run
  showed the **existing flat contiguous kernel was itself hobbled** by its
  indirect call, at 21.0 us against 11.7 us.

  **The architecture** is the shape H2, H5, and H6 each proved: one hidden
  metadata builder, inside the existing export, no new symbol, the
  pre-milestone traversal retained. `tf::build_unary_plan` /
  `tf::build_binary_plan` (in `cpp/include/tf_elementwise_internal.h`)
  produce an **operation-local normalized descriptor** — stack-built, used
  by one call, dropped, with nothing cached, interned, or shared — that
  applies exactly two sequence-preserving transformations: **unit axes are
  dropped**, and **adjacent axes are merged** when
  `stride[outer] == stride[inner] * extent(inner)` holds for *every*
  operand at once. Axes are never reordered, split, or transposed; the
  bound is a fixed **4 axes**; **this is not a layout compiler**. The
  builders are total, pure, allocation-free, and a function of layout
  metadata alone, and a rejection is a **fallback, never an error**;
  `core_unary` and `core_binary` are retained **verbatim** as the shipped
  generic reference paths. **`exp` and `log` are deliberately excluded** —
  IEEE-754 does not specify them, so a vectorizing toolchain could
  legitimately return different bits, and the templated traversal measured
  **1.05x** on both, inside the noise.

  **The numerical contract is H8's own, in four parts.** Every result in
  which at most one operand is a NaN is **bit-identical** to the pre-H8
  kernel's (**zero differing results** across 15 op x layout
  combinations); NaN positions are identical and every NaN the
  *arithmetic* produces is quiet, while `relu_backward` and the identity
  gather **select** an operand and so legitimately preserve a signaling
  NaN, identically on both traversals; **subtraction is bit-identical
  everywhere**, two-NaN pairs included, because it is not commutative; and
  for **addition and multiplication with two NaN operands** the surviving
  payload is outside the contract. **That last part predates H8 and H8
  narrows it** — the pre-H8 library's own flat kernel and its own odometer
  already disagreed on **30 of 196** such pairs, while post-H8 only a
  transposed operand differs, on **5 of 196**. It is a *different*
  qualification from H2's and H6's, which concerned NaNs meeting inside an
  accumulation. **H1's contract holds unchanged**, proved by poison over
  six layouts with two patterns, guard elements, and a negative control.

  **Track B** shipped the one composed-allocation change the evidence
  supported: `_NativeBatchNorm` builds its `(1 - momentum, momentum)` pair
  **once per forward** instead of once per buffer — the H4
  per-step-constants shape, never stored on the module — and each blend
  releases its temporaries at last use. Against a **retained pre-H8
  composition executed natively**, running statistics bit-identical first,
  a `NativeBatchNorm1d` training forward goes **25 -> 23** allocations with
  **peak live storages 25 -> 17** and constant fills **5 -> 3**, and
  `NativeBatchNorm2d` **30 -> 28**, **30 -> 22**, and **5 -> 3**. Its
  **timing effect is neutral**, and it is reported as a memory result
  rather than a speed one. Four alternatives were rejected with reasons,
  including adopting the blend result into the running-state transaction —
  which would have moved numerical work inside the staging phase and
  changed a failure ordering F5 and F8 prove by test.

  Measured against a pre-H8 library on identical `ctypes` calls with every
  case bit-identical before timing, control band **0.97x-1.08x**:
  row-broadcast `multiply` **10.58x**, strided same-shape `add` **9.67x**,
  NCHW-statistic `multiply` **7.15x**, column broadcast **6.70x**, scalar
  broadcast **6.31x**, transposed `add` 2.63x, strided `relu` 2.51x,
  `sqrt` 2.03x, `relu_backward` 1.86x, contiguous `add`/`multiply` 1.76x.
  End to end, over 11 alternating subprocess rounds with all 31 cases
  bit-identical first: **`NativeAdam.step()` 2.01x**, **`NativeBatchNorm1d`
  eval forward 1.40x**, **`NativeBatchNorm2d` eval forward 1.36x**,
  **`NativeBatchNorm2d` training forward 1.33x**, **`NativeLayerNorm`
  forward 1.30x**, **the large MLP training step 1.19x**, and **the
  normalized training step 1.08x**. **This is the milestone that finally
  moved the normalization modules** — which H6 measured as almost entirely
  neutral, and which is precisely why H0's composed-module H7 was dropped
  and this one entered.

  Reported just as honestly: **small normalization shapes are neutral**
  (`NativeBatchNorm1d` training at `(32, 16)` **0.98x**), **the CNN step is
  neutral (0.99x)** because its time is in the convolution kernels H8 did
  not touch, the `exp`/`log` controls read 0.97x-1.07x exactly as the
  deliberate exclusion predicts, and one control is **published rather
  than buried** — 256-cubed `matmul` reads **0.93x-0.96x**, isolated by a
  25-round run to that one size while 64, 128, and 384 cubed are neutral
  and an identical-code twin reads 0.969x on the same case. `matmul.cpp`
  is byte-identical source; `elementwise.cpp`'s object code grew 127 KB to
  188 KB, moving every function's placement — the same
  whole-translation-unit code-layout effect H6 documented, and the
  machine-specific tuning the design rejects chasing.

  **Memory: Track A moved none, and the odometer's heap-allocated counter
  is now removed on every plannable layout** — a strided elementwise call
  makes **one** allocation where it previously made two, which re-anchored
  one existing fault-injection test with its assertion unchanged. The
  harness gained four cases, 34 to **38**, and the native CTests 15 to
  **16**. Validation added its own **sanitizer negative control**: under
  Clang ASan, test-only code handing `tf_core_add` two-entry metadata with
  `ndim = 3` produces a `heap-buffer-overflow`, `READ of size 8`, in
  `tf::build_binary_plan` — H8's own new code path — which is what makes
  the **zero diagnostics across 2,726 sanitized tests** a real absence
  rather than a blind detector. **No exported C ABI symbol, no new
  translation unit, and no public control of any kind**, and no SIMD,
  threading, OpenMP, BLAS, memory pool, scratch workspace, general fusion,
  fast-math, or `restrict`. No public API, capability, dtype, device,
  registry value, checkpoint field, or checkpoint version moved.

  **Milestone H9 — native Conv2d execution efficiency — has since shipped**,
  the fifth Phase-H milestone to change C++ and, like H2/H5/H6/H8, **not the
  ABI**: still exactly **52** exported `tf_*` symbols. **It was not in H0's
  ladder** — that slot held SIMD/threading/BLAS, *conditional and presumed
  rejected*; none qualified, so the acceleration decision moved to H10's
  decision gate and the slot went to the last large compute family still
  running its unmodified Phase-D implementation. Measurement decided it:
  the Python wrapper is a fixed ≈ 8–12 µs — **66 %** of a toy `(4,1,6,6)`
  convolution but **≈ 0 %** at `(16,8,32,32)` — so the compiled traversal is
  essentially **100 %** of any real convolution. Each of the three exports
  now ships two paths behind one unchanged symbol: the **Phase-D direct loop
  retained verbatim** as the generic reference, and one optimized traversal
  — a forward row sweep, and gathers for both gradients — chosen by a hidden
  predicate from the integer geometry alone
  (`min(input_width, output_width) >= 4`, plus **unit stride** for the input
  gradient, whose downward kernel walk inverts one-for-one only there). **A
  false answer is a fallback, never an error**, and **per-destination
  accumulation order is preserved exactly in all three directions**, each
  separately proved. Measured against a pre-H9 library on identical `ctypes`
  calls, bit-identical before timing: kernels **1.97×–8.37×**, `NativeConv2d`
  forward+backward **3.09×–3.13×**, and **a CNN training step 1.86×** —
  **the first Phase-H milestone to move one**. Honestly reported: small
  convolutions are neutral (1.06×), the strided input gradient falls back by
  design (1.04×), and no control regressed (band **0.97×–1.07×**). Memory is
  **byte-identical**; there is no scratch, workspace, im2col, or padded copy
  anywhere. The harness gained three cases, 38 to **41**, and the native
  CTests 16 to **17**. No public API, capability, dtype, device, registry
  value, checkpoint field, or checkpoint version moved, and no convolution
  option was added.
- **Native dtype generalization and float32 CPU support (Phase I) has
  begun, at milestone I0.** Phase I is the latest phase; Phase H is
  unaffected and remains complete. **I0 is the architecture contract
  ([native_dtype_float32_design.md](native_dtype_float32_design.md)), its
  guardrail tests, and documentation reconciliation — and no runtime
  behavior**: no dtype, no storage change, no kernel, no C ABI symbol, no
  ctypes declaration, no Core method, no operation, no module, no
  optimizer, no export, no registry value, and no checkpoint-format
  change. The runtime is still float64 CPU only, still **52** exported
  `tf_*` symbols, still checkpoint version 2 with versions 1 and 2
  accepted, and `float32` is still listed in `UNSUPPORTED`. The
  architectural fact the contract rests on is visible in the layering
  above: **42 of the 52 exports already address their operands through
  opaque handles**, so putting a dtype tag on the storage behind those
  handles reaches every one of them, and only *construction* needs new
  information — hence **exactly two** planned new symbols
  (`tf_storage_create_typed`, `tf_storage_create_uninitialized_typed`,
  52 → **54**) and an explicit rejection of per-operation float32
  exports. Storage becomes the single dtype authority, so every view over
  a buffer agrees by construction and no view operation casts; shapes,
  strides, and offsets stay in logical elements, with bytes appearing only
  at the allocation boundary; each exported call dispatches **once** into
  a templated `float`/`double` kernel and nothing below that branches on
  dtype; casting, promotion, and mixed-dtype arithmetic do not exist and
  are rejected before any allocation or mutation; float32 accumulates in
  float32; checkpoint **version 3** is designed but not activated, with
  versions 1 and 2 defined as float64-only formats; exact deterministic
  resume is proved **separately** per dtype rather than as agreement
  between them; and the public support registry moves at milestone
  **I9**, not before.

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
