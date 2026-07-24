# Architecture

TensorForge is a from-scratch deep learning and ML systems framework
with **two strictly separate lines**. The **stable Python framework**
reimplements how a framework like PyTorch works under the hood in
Python + NumPy, with every piece kept deliberately small and readable.
The **experimental native line** (the advanced branch) is a
ctypes-loaded C++ CPU runtime with its own explicit tensor, a
Python-managed native autograd graph, and a native training stack —
living in its own namespaces (`tensorforge.backends`,
`tensorforge.experimental`) that the stable framework never imports
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
- **Native normalization (Phase F) is in progress.** The
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
  capability. Milestone F9 — the phase closure — has not started, so
  Phase F itself is still in progress.

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
