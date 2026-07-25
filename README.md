# TensorForge

A from-scratch deep learning and ML systems framework in two
lines: a **stable Python framework** built on NumPy — PyTorch-style
autograd, modules, optimizers, checkpointing, and CNN support, complete
as of v3.0 — and an **experimental native C++ CPU backend**, living in
its own `tensorforge.backends` / `tensorforge.experimental` namespaces,
with its own explicit tensor runtime, Python-managed
native autograd, and a native training stack that already trains an MLP
end to end. The two lines stay strictly separate: `NativeTensor` never
masquerades as `tensorforge.Tensor`, nothing dispatches implicitly, and
the native stack is reached only by explicit import.

Everything is implemented by hand and kept readable: the autograd
engines, the layers, the optimizers, the losses, the C++ kernels. If
you want to see what `loss.backward()` actually does — in NumPy or
through a ctypes boundary into native code — this repo shows you.

**Framework internals covered:** reverse-mode autograd, how neural
network modules and parameters fit together, what optimizers actually
do, regularization (Dropout) and normalization (BatchNorm, LayerNorm),
checkpoint/resume mechanics down to the RNG, the internals of
convolution and pooling, and native-backend mechanics: an owning
storage/view/tensor runtime, strided kernels behind a C ABI, a
Python-managed autograd graph over them, parameter versioning with
stale-graph detection, and honest benchmarks.

## Features — stable Python framework

**Core engine**
- Tensor with reverse-mode autograd: `+ - * / ** @`, `sum`, `mean`,
  `reshape`, `exp`, `log`, `tanh`, `sigmoid`, `relu`, `softmax`, with
  broadcasting-aware gradients — all verified against finite differences

**Models (`tensorforge.nn`)**
- `Parameter`, `Module`, `Linear`, `ReLU`/`Sigmoid`/`Tanh`, `Dropout`,
  `BatchNorm1d`, `LayerNorm`, `Conv2d`, `MaxPool2d`, `Flatten`,
  `Sequential`
- Train/eval mode (`model.train()` / `model.eval()`), frozen
  parameters, model summaries, parameter counting
- Losses: `mse_loss`, numerically stable `cross_entropy` and
  `binary_cross_entropy` (both on raw logits)
- Metrics: `accuracy`, `binary_accuracy`, and eval-safe
  `evaluate_classifier` / `evaluate_binary_classifier`

**Training (`tensorforge.optim`, `tensorforge.data`)**
- `SGD`, `Adam`, `StepLR` scheduler, `clip_grad_norm` /
  `clip_grad_value`
- `batches` mini-batch iterator, `train_test_split` validation holdout

**Persistence**
- `save_parameters` / `load_parameters` for weights, and
  `save_checkpoint` / `load_checkpoint` with optimizer state, optional
  scheduler state, optional RNG state (for dropout-safe, bit-exact
  resume), and metadata — all plain `.npz`, no pickle

## Features — experimental native C++ backend

The explicit experimental native line — merged into `main` and reached
only through `tensorforge.experimental` and `tensorforge.backends`, never
by the stable framework — carries a complete native CPU training line:

- **Native runtime**: `NativeStorage` → `NativeTensorView` →
  `NativeTensorCore` → `NativeTensor` — explicit ownership and
  lifetime, shapes/strides/offsets, borrowing views, contiguity
  tracking, NumPy-style broadcasting, sum/mean reductions, and
  float64/cpu dtype/device metadata over ctypes-loaded C++ kernels.
- **Native autograd (Phase B, complete)**: a Python-managed
  reverse-mode graph over autograd-unaware kernels — the differentiable
  operations Phase B shipped (`add`, `subtract`, `multiply`, `relu`,
  `sqrt`, `reciprocal`, `matmul`, `sum`, `mean`, `reshape`,
  `transpose`/`T`, `narrow`, `contiguous_copy`), joined by the Phase-D
  `conv2d` and `maxpool2d` primitives below and the Phase-E
  `exp`/`log`/`softmax`, plus broadcasting and view
  gradients, a native scatter backward for `narrow`, one-shot graph
  release with `retain_graph` opt-in, and failure rollback. The backend's
  `AUTOGRAD_OPS` registry is the exact, current list.
- **Native training stack (Phase C, complete)**: `NativeParameter`
  (value versioning and a controlled mutation path with stale-graph
  detection), `NativeModule` with atomic `state_dict`/
  `load_state_dict`, `NativeLinear`, `NativeReLU`, `NativeSequential`,
  `NativeMSELoss`, the minimal `NativeSGD` optimizer, and the
  adaptive `NativeAdam` optimizer (persistent native moment state,
  per-parameter bias correction, explicit state lifetime) — both with
  in-memory `state_dict`/`load_state_dict`, plus pickle-free native
  checkpoint files (`save_native_checkpoint` /
  `load_native_checkpoint`: model + optional optimizer state + JSON
  metadata, atomic writes, strict validation, deterministic
  bit-identical file resume).
- **A native MLP training proof**: `examples/native_mlp_training.py`
  trains a 2→8→ReLU→1 MLP for 25 deterministic native SGD steps with a
  monotonic 99.5% loss reduction — model, loss, gradients, and updates
  all native.
- **A native CNN training proof**: `examples/native_cnn_training.py`
  trains Conv2d→ReLU→MaxPool2d→Flatten→Linear on eight fixed 6×6 images
  for 40 deterministic native Adam steps (98.6% loss reduction), then
  checkpoints mid-run and resumes into a fresh model/optimizer pair that
  reproduces the uninterrupted run exactly.
- **A native classification proof**:
  `examples/native_classification_training.py` trains the same layer
  stack as a three-class classifier over **raw logits** into
  `NativeCrossEntropyLoss` on twelve fixed 6×6 images, for 40
  deterministic native Adam steps (loss 1.159638 → 0.000101, reporting
  accuracy 0.3333 → 1.0000 via `native_accuracy`), then checkpoints at
  step 15 and resumes into a fresh model/optimizer pair that reproduces
  the remaining losses, parameters, optimizer state, logits,
  predictions, and accuracy exactly.
- **A native normalized-training proof**:
  `examples/native_normalization_training.py` trains a
  `Linear → BatchNorm1d → ReLU → LayerNorm → Linear` regressor — running
  **both** normalization families in every forward, with `NativeBatchNorm1d`
  the only stateful module — for 24 deterministic native Adam steps with
  `NativeMSELoss` (98.9% loss reduction), then checkpoints at step 10 and
  resumes into a fresh model/optimizer pair that reproduces the remaining
  losses, every parameter, the NativeAdam state, the **BatchNorm running
  statistics**, the final training-step prediction, and the final
  **evaluation-mode** output exactly (format version 1, training flags
  runtime-only).

The exact operation-by-operation status lives in the
[native support matrix](docs/native_support_matrix.md).

## Quickstart

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/):

```
uv sync
uv run pytest
uv run python examples/train_mlp_with_dropout.py
```

The stable API looks the way you'd expect:

```python
from tensorforge import Tensor, Adam
from tensorforge.nn import Linear, ReLU, Sequential, cross_entropy

x = Tensor(4.0, requires_grad=True)
(x * x).backward()
print(x.grad)  # 8.0

model = Sequential(Linear(2, 16), ReLU(), Linear(16, 3))
optimizer = Adam(model.parameters(), lr=0.01)

loss = cross_entropy(model(Tensor(inputs)), targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Native quickstart

Build the experimental backend once, then run the native line:

```
uv run python cpp/build.py                        # uv sync --group cpp first if you have no C++ compiler
uv run python scripts/smoke_cpp_backend.py        # hard-failing backend check
uv run python examples/native_tensor_demo.py      # the native runtime and views
uv run python examples/native_autograd_demo.py    # native backward
uv run python examples/native_mlp_training.py     # end-to-end native training
uv run python examples/native_checkpoint_resume.py # save, restore, resume bit-for-bit
uv run python examples/native_cnn_training.py     # end-to-end native CNN training + resume
uv run python examples/native_classification_training.py  # native classification + exact resume
uv run python examples/native_normalization_training.py   # native BatchNorm+LayerNorm training + exact resume
uv run python benchmarks/benchmark_native_autograd.py --smoke
uv run python benchmarks/benchmark_native_cnn.py --smoke  # CNN characterization
uv run python benchmarks/benchmark_native_classification.py --smoke        # classification characterization
uv run python benchmarks/benchmark_native_classification.py --smoke --json # machine-readable JSON
uv run python benchmarks/benchmark_native_normalization.py --smoke         # normalization characterization
uv run python benchmarks/benchmark_native_normalization.py --smoke --json  # machine-readable JSON
```

The native API mirrors the stable one, explicitly:

```python
from tensorforge.experimental import (
    NativeLinear, NativeMSELoss, NativeReLU, NativeSequential,
    NativeSGD, NativeTensor,
)

model = NativeSequential(NativeLinear(2, 8, seed=0), NativeReLU(),
                         NativeLinear(8, 1, seed=1))
optimizer = NativeSGD(model.parameters(), lr=0.1)

loss = NativeMSELoss()(model(NativeTensor.from_array(inputs)),
                       NativeTensor.from_array(targets))
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

## Examples

Six runnable, seeded stable-framework examples forming a progression —
each one introduces a single idea:

```
uv run python examples/train_linear_regression.py    # the bare training loop
uv run python examples/train_xor.py                  # why hidden layers exist
uv run python examples/train_multiclass.py           # real classification + mini-batches
uv run python examples/train_binary_classification.py  # logits and stable losses
uv run python examples/train_mlp_with_dropout.py     # train mode vs eval mode
uv run python examples/train_tiny_cnn.py             # convolution and pooling
```

What each one teaches, and what to expect: [docs/examples.md](docs/examples.md).
The native examples and demos are listed in the native quickstart above.

## Documentation

- [docs/project_summary.md](docs/project_summary.md) — the whole project in two minutes
- [docs/architecture.md](docs/architecture.md) — how both framework lines fit together
- [docs/autograd.md](docs/autograd.md) — the stable autograd engine, explained
- [docs/training.md](docs/training.md) — training loops, train/eval mode, saving
- [docs/examples.md](docs/examples.md) — the stable examples and what they teach
- [docs/roadmap.md](docs/roadmap.md) — what's done and what's next
- [docs/release_history.md](docs/release_history.md) — how the project grew, by version
- [docs/native_support_matrix.md](docs/native_support_matrix.md) — exactly what the native stack supports
- [docs/backend_experiments.md](docs/backend_experiments.md) — the experimental C++ backend line
- [docs/dispatch_design.md](docs/dispatch_design.md) — how backends might eventually meet the Tensor
- [docs/native_tensor_wrapper_design.md](docs/native_tensor_wrapper_design.md) — Stage-2 design for the native tensor wrapper
- [docs/native_contiguous_fast_path_design.md](docs/native_contiguous_fast_path_design.md) — design for a contiguous elementwise fast path in the native runtime
- [docs/native_broadcasting_design.md](docs/native_broadcasting_design.md) — design for NumPy-style broadcasting in the native elementwise runtime
- [docs/native_reductions_design.md](docs/native_reductions_design.md) — design for native sum/mean reductions in the native runtime
- [docs/native_dtype_device_metadata_design.md](docs/native_dtype_device_metadata_design.md) — design for explicit dtype/device metadata in the native runtime
- [docs/native_autograd_design.md](docs/native_autograd_design.md) — design for native reverse-mode autograd over NativeTensor (Phase B)
- [docs/native_autograd_benchmarks.md](docs/native_autograd_benchmarks.md) — characterization benchmark for the native autograd stack (Phase B)
- [docs/native_cnn_design.md](docs/native_cnn_design.md) — architecture contract for the native CNN stack (Phase D)
- [docs/native_classification_design.md](docs/native_classification_design.md) — architecture contract for the native classification stack (Phase E — complete: E0–E10 shipped)
- [docs/native_normalization_design.md](docs/native_normalization_design.md) — architecture contract for the native normalization stack (Phase F — **complete**: F0, F1, F2 (`NativeLayerNorm`), F3 (`NativeBatchNorm1d`), F4 (`NativeBatchNorm2d`), F5 (state/checkpoint/graph-safety hardening), F6 (a deterministic normalized training example with exact resume), F7 (the honest benchmark characterization), F8 (the cross-cutting integration and semantic guardrails), and F9 (the phase closure — validation and documentation only) have all shipped)
- [docs/native_rng_dropout_design.md](docs/native_rng_dropout_design.md) — architecture contract for native RNG and Dropout (Phase G — **in progress**: milestone G0, the design lock, is complete; G1–G10 have not started, and no generator, kernel, operation, module, or export exists yet)

## Limitations

Honest expectations:

- Not production-ready — clarity and correctness take priority over
  performance everywhere, in both lines.
- The stable framework is NumPy on CPU. The native line is an
  experimental C++ **CPU** backend: float64/cpu only, no CUDA backend
  yet, no dtype promotion or casting, and no implicit dispatch into
  `tensorforge.Tensor`.
- The native CNN stack (Phase D), the native classification stack
  (Phase E), and the native normalization stack (Phase F) are all
  complete — but "complete" means *these* capabilities
  work and are validated, not that the native line is finished. All three
  normalization modules
  (`NativeLayerNorm`, `NativeBatchNorm1d`, `NativeBatchNorm2d`) shipped
  as modules composed from existing operations with no kernel, F5
  proved their state/checkpoint/ownership/graph-safety contracts by
  exhaustive test, F6 shipped a deterministic normalized training
  example with exact checkpoint resume
  (`examples/native_normalization_training.py`), F7 shipped the
  honest benchmark characterization
  (`benchmarks/benchmark_native_normalization.py`), F8 shipped the
  cross-cutting integration and semantic guardrails
  (`tests/test_native_phase_f.py`), and F9 closed the phase under
  Release/Debug builds and Clang ASan/UBSan/LeakSanitizer. What the
  native line still does **not** have: dropout
  or a native RNG, data loaders, native integer
  tensors, further dtypes or devices, CUDA, AMP, and any implicit
  dispatch into `tensorforge.Tensor`. Native checkpoints capture no
  scheduler or random state, and the classification loss supports
  `"mean"`/`"sum"` only — no `reduction="none"`, class weights,
  `ignore_index`, label smoothing, or soft targets. See the
  [native support matrix](docs/native_support_matrix.md).
- Both lines' convolution and pooling use deliberately naive loops (the
  stable `Conv2d`/`MaxPool2d` and the native kernels alike: no im2col,
  BLAS, threading, or SIMD).
- Benchmarks are hardware-specific characterizations with no universal
  speed claims; the naive native kernels can lose to NumPy's BLAS.
- No real datasets and no external ML libraries; every example runs on
  small synthetic data.

## Status

**v3.0 — the stable Python framework line is complete**, covered by the
test suite and documented. **The experimental native line — merged into
`main`, and reached only through the explicit `tensorforge.backends` and
`tensorforge.experimental` namespaces — has completed Phases A–F**:
Phase A (native CPU runtime),
Phase B (native autograd), Phase C (the native training stack),
Phase D (the native CNN stack), Phase E (native classification and
stable math), and Phase F (native normalization and stateful buffers)
are all complete. Phase C shipped
parameters, modules, state dictionaries,
Linear/ReLU/Sequential, MSE loss, parameter versioning with stale-graph
safety, `sqrt`/`reciprocal` optimizer primitives, SGD and adaptive Adam,
in-memory optimizer state snapshots, pickle-free native checkpoint files,
end-to-end deterministic MLP training, and deterministic in-memory and
file resume — with cross-cutting failure, lifetime, and ownership
guardrails. **Phase D** added every native CNN layer — `NativeFlatten`,
the differentiable convolution layer (`NativeConv2d`) over the native
`conv2d` operation, and the pooling layer (`NativeMaxPool2d`) over the
native `maxpool2d` operation with its private saved winners — plus the
end-to-end training + checkpoint-resume proof
(`examples/native_cnn_training.py`: 40 deterministic steps, 98.6% loss
reduction, and a checkpoint-interrupted run that reproduces the
uninterrupted one exactly), cross-cutting integration tests, honest CNN
benchmarks, and ASan/UBSan validation of the whole native stack.
**Phase E — Native Classification and Stable Math — is complete**
(milestones E0–E10): its architecture contract is locked
([docs/native_classification_design.md](docs/native_classification_design.md),
milestone E0) and milestones **E1–E4 shipped the differentiable native
`exp`, `log`, and the fused stable `softmax` and `log_softmax`** — C++
kernels, self-validating guarded C ABI, `NativeTensorCore` and
`NativeTensor` layers. `exp` and `log` are the phase's two backward
archetypes: `exp` reads its saved output and records no parameter
version, while `log` rereads the live input (`upstream × reciprocal(x)`,
no division operation added) and version-guards a direct parameter so a
post-forward mutation fails before any gradient moves. `softmax` and
`log_softmax` are the phase's two fused probability transforms — a
maximum-shift kernel and a log-sum-exp kernel over any axis behind a
contiguous-only ABI, each with a saved-output backward composed from
existing Core operations rather than a dedicated kernel. `log_softmax`
is deliberately **never** `softmax().log()`: it forms no probability and
performs no division, so it stays accurate where the composed form
collapses to `-inf`. **E5 shipped the fused `cross_entropy` Core
contract** — a single kernel producing the stable scalar loss *and* the
private saved probabilities per row (never `-log(p[target])`), a
backward that reads only those saved probabilities, the copied `int64`
targets, and a native one-element upstream (**it never rereads the
logits**), and strict targets that reject `bool` and floating-point
labels and are copied so caller mutation cannot reach the kernel. **E6
shipped the differentiable `NativeTensor.cross_entropy(targets,
reduction="mean")`** over that contract, adding no kernel and no
numerical change: one scalar-output autograd node whose private saved
probabilities are **graph-owned** — retained under `retain_graph=True`,
released exactly once with the graph history, closed immediately when no
gradient is required — and whose backward never rereads the logits, so
no parameter version is recorded and mutating the logits after the
forward leaves the gradient correct for the forward that ran.
**E7 completed the public surface**: `NativeCrossEntropyLoss`, a
stateless module whose whole forward delegates to that operation, and
`native_accuracy`, a deliberately **reporting-only** helper — no kernel,
no Core method, no autograd node — that materializes once through the
explicit public `to_numpy()` boundary, takes a NumPy argmax, and returns
a plain `float` without building a graph or touching any gradient.
**E8 proved the assembled stack end to end without adding anything to
it**: `examples/native_classification_training.py` trains a
`NativeConv2d(1, 4, 3)` → `NativeReLU` → `NativeMaxPool2d(2)` →
`NativeFlatten` → `NativeLinear(16, 3)` classifier — no softmax layer,
raw logits straight into `NativeCrossEntropyLoss` — on twelve fixed 6×6
images in three classes for 40 deterministic `NativeAdam(lr=0.05)` steps
(loss 1.159638 → 0.000101, reporting accuracy 0.3333 → 1.0000), then
interrupts at step 15, checkpoints model **and** optimizer state through
the existing pickle-free path (format **version 1**, unchanged), and
resumes into a **fresh** model/optimizer pair that reproduces the
uninterrupted run **exactly** — the whole remaining loss suffix, every
parameter, both Adam moment buffers and step counters, the final logits,
the predictions, and the accuracy. It is an integration proof on one
fixed task: no speed and no generalization is claimed.
**E9 added the characterization benchmark**,
`benchmarks/benchmark_native_classification.py` — seven cases (`exp`,
`log`, `softmax`, `log_softmax`, cross-entropy forward, cross-entropy
backward, and one complete classification training step), each with a
correctness gate that runs **before** any timing, each labelled with the
reference it actually used (`stable_tensorforge`, `numpy` where the
stable line has no direct operation, or `native_only`), and each reported
as a median with min/max/spread over repeated `time.perf_counter_ns`
measurements taken after warm-up with setup and cleanup outside the
timer. It has `--smoke` and `--json` modes and writes no result file.
Observed ratios are **local characterizations, not guarantees**: no test
asserts a speed, no timing number is committed as a promise, and there is
no CI performance gate. **E10 closed the phase with no new numerical
capability**: cross-cutting integration tests
(`tests/test_native_phase_e.py`), Release **and** Debug native builds
(10/10 CTests each, zero warnings), Clang AddressSanitizer and
UndefinedBehaviorSanitizer validation of the whole classification stack
with zero diagnostics attributable to TensorForge, a practical
LeakSanitizer pass finding no native leak, the full Python regression
suite, and documentation reconciliation across every status surface.
Phase E expanded nothing beyond float64/CPU and added no implicit
stable/native dispatch.

**Phase F — Native Normalization and Stateful Buffers — is complete
(F0–F9).**
Its architecture
contract is locked in
[docs/native_normalization_design.md](docs/native_normalization_design.md)
(milestone **F0**, complete: design and repository reconciliation only,
adding no numerical behavior), as is **F1** (a private atomic
native-buffer state transaction, the `load_state_dict` refactor onto it,
and the `STATE_SUPPORT` persistent-buffer reconciliation — state
management and capability reporting only, no normalization mathematics),
and **F2** (`NativeLayerNorm` — the first native normalization module:
stateless, differentiable through the mean and the population variance,
and composed entirely from existing native operations with `sqrt(var +
eps)` ordering and no kernel, ABI symbol, `NativeTensorCore` method,
custom backward, or `NativeTensor` normalization operation;
`"NativeLayerNorm"` is now in `NATIVE_MODULES` and the exports, and
`"layernorm"` has left `UNSUPPORTED`), and **F3**
(`NativeBatchNorm1d` — the **first stateful native numerical module**:
`(N, C)` batch normalization with differentiable training statistics
(gradients flow through the batch mean *and* the population variance),
persistent native `running_mean`/`running_var` buffers advanced
graph-free by one **atomic two-buffer transaction** over the F1
primitive (both identities preserved, replaced cores closed exactly
once, no parameter version moved), and evaluation from **graph-safe
immutable snapshots** of those buffers, so a later training step, or a
buffer-only state or checkpoint load, can never change an earlier eval
graph's gradient — while a *full* checkpoint load that also replaces
`gamma`/`beta` still stales that graph through the unchanged
parameter-version rule, which is correct and deliberately unweakened; again composed from existing operations, so again no kernel,
ABI symbol, `NativeTensorCore` method, custom backward, or
`NativeTensor.batch_norm` operation, and the native checkpoint format
stays at version 1; `"NativeBatchNorm1d"` is now in `NATIVE_MODULES` and
the exports, while `"batchnorm"` stayed in `UNSUPPORTED`), and **F4**
(`NativeBatchNorm2d` — NCHW `(N, C, H, W)` batch normalization reducing
over **N, H, and W**, so each channel gets one population mean and one
population variance over `N * H * W` values. It is built on the **same**
shared private implementation as the 1-D shape and declares only its
rank, its reduction axes, its `(1, C, 1, 1)` broadcast layout, and the
channels-last permutation its rank-1 `gamma`/`beta` need — the
activation is transposed for the affine step, never the parameters, so
the existing direct-parameter stale-value guard is preserved exactly.
Running buffers stay `(C,)`; again no kernel, ABI symbol,
`NativeTensorCore` method, custom backward, or `NativeTensor.batch_norm`
operation, and the checkpoint format stays version 1;
`"NativeBatchNorm2d"` is now in `NATIVE_MODULES` and the exports, and
with both shapes live **`batchnorm` has left `UNSUPPORTED`**, which now
reads exactly `("dropout", "float32", "cuda", "amp")`).
**That completes the numerical normalization *module* surface.** **F5 is
complete**: the exhaustive state/checkpoint, ownership, and graph-safety
hardening — a focused `tests/test_native_normalization_state.py` plus
narrow additions to the generic buffer and checkpoint suites — proves the
design's §7–§10 contracts by executable test (canonical dotted buffer
keys, independent state snapshots, strict/non-strict loads, exact
never-casting metadata validation, mixed parameter/buffer transaction
atomicity, buffer identity across state and checkpoint loads, exact
eval-output reproduction, the buffer-only-versus-full stale-graph
distinction, the save/corrupt-load failure boundaries, eval-graph snapshot
safety under `retain_graph` and a failed retryable backward, and the
live-storage baselines); it is **tests and documentation only — no
numerical behavior, no new capability, and the checkpoint format stays
version 1**. **F6 is complete**: `examples/native_normalization_training.py`
trains a `Linear → BatchNorm1d → ReLU → LayerNorm → Linear` regressor for
24 deterministic `NativeAdam` steps with `NativeMSELoss` (a 98.9% loss
reduction), then checkpoints at step 10 and resumes into a completely
fresh model/optimizer pair that reproduces the remaining losses, every
parameter, the NativeAdam state, the **BatchNorm running statistics**, the
final training-step prediction, and the final **evaluation-mode** output
exactly — proving two uninterrupted runs are bit-identical and the
interrupted resume is exact (format version 1 unchanged, training flags
runtime-only). It **adds no capability**: one example and its integration
test, no operation, kernel, or schema change. **F7 is complete**:
`benchmarks/benchmark_native_normalization.py` characterizes the stack
with nine cases — the LayerNorm forward and backward, the BatchNorm1d
training forward, evaluation forward, and backward, the BatchNorm2d
training forward, evaluation forward, and backward, and one complete
F6-style normalized training step. Every case is **correctness-gated
before any timing** (a failed gate exits nonzero and publishes nothing);
six are measured against `stable_tensorforge` equivalents on the same
inputs, epsilon, momentum, affine values, running state, initial
parameters, and optimizer hyperparameters, while the three BatchNorm2d
cases are labelled `native_only` for timing because the stable line has
**no public `BatchNorm2d`** to compare against — they still carry a
rigorous correctness oracle (an explicit NumPy NCHW population-statistics
formula, and for the backward the stable `BatchNorm1d` on the equivalent
`(N*H*W, C)` sample matrix transformed back to NCHW). Medians are
reported with min, max, and spread after warm-up, `--smoke` and `--json`
modes exist, **no result file is written, no speed is asserted, no timing
number is committed, and no CI job asserts a duration** — measurement
only, adding no capability. **F8 is complete**:
`tests/test_native_phase_f.py` proves the cross-cutting interactions no
single-module suite can — one integrated `Conv2d → BatchNorm2d → ReLU →
MaxPool2d → Flatten → Linear → BatchNorm1d → ReLU → LayerNorm → Linear`
classifier over **raw logits** and the fused loss, trained by
`NativeAdam` and resumed **exactly** from one version-1 checkpoint (the
loss suffix, every parameter, the NativeAdam state, **all four**
running-statistic buffers, the final training logits, and the final
evaluation-mode logits, predictions, and accuracy); BatchNorm eval
snapshots, MaxPool2d winners, and cross-entropy probabilities coexisting
in one eval graph and releasing exactly once; buffer-only mutation
leaving an earlier eval graph valid while a full checkpoint load or an
affine `copy_value_` correctly stales it through the unchanged parameter
rule; the versioning archetypes; shared and frozen parameters; a
non-contiguous NCHW input through the whole stack; strict stable/native
separation; and each failure boundary tested **honestly** — BatchNorm
transactions are per module, so one whole training step is *not*
presented as globally transactional. Tests and documentation only,
adding no capability.
Milestone **F9 is complete** — the phase closure: fresh Windows Release
**and** Debug builds each passing the full existing 10-test CTest suite
with zero project compiler, linker, or CMake warnings, and the active
runtime proved to stay the Release DLL; a fresh Clang 18.1.3
ASan+UBSan build in WSL2 Ubuntu 24.04 whose instrumentation is *proved*
rather than assumed (22 `__asan*` and 13 `__ubsan*` dynamic symbols, and
a library that refuses to load without the sanitizer runtime); 10/10
sanitized native CTests with leak detection enabled; **1,968** sanitized
normalization-focused Python tests with zero ASan and zero UBSan
diagnostics; the F6 example reproducing its exact resume and the F7
benchmark passing all nine correctness gates under the sanitized
library; and a practical LeakSanitizer lifecycle whose native
live-storage counter returned **exactly** to baseline, its remaining
process-exit allocations identified honestly as CPython/NumPy shutdown
retention with no TensorForge frame and no suppression file —
**validation and documentation only, adding no numerical capability**.
So **Phase F is complete**, and no
normalization *operation*, kernel, or C ABI symbol exists at all.
Closing Phase F closes that phase, not the project: the native line is
still experimental, still float64/CPU only, and still not
production-ready.

**Phase G — Native RNG and Dropout — is the current phase, and it is in
progress.** Only its first milestone has landed: **G0**, the architecture
contract locked in
[docs/native_rng_dropout_design.md](docs/native_rng_dropout_design.md) —
Python-managed generator state (an explicit 64-bit seed and call counter
with an algorithm identifier), stateless native random kernels, inverted
Dropout with a graph-owned multiplier mask, one generator call consumed
per *successful* stochastic forward and none on any failure — behind a
lock-protected, token-validated reservation protocol so no two callers
can ever receive the same call index — generator state registered on
`NativeModule`, native checkpoint **version 2** that records the
generator *alias topology* (which layers share a stream, not just the
saved states) with a locked version-1 compatibility rule, and
whole-checkpoint transaction atomicity in which any ordinary synchronous
commit failure rolls back parameters, buffers, optimizer state, and
generator state together. **G0 is design,
documentation, and guardrails only**: milestones G1–G10 have not started,
no `NativeGenerator`, kernel, C ABI symbol, operation, module, or export
exists yet, the checkpoint format is still version 1, and `dropout`
(with `float32`, `cuda`, and `amp`) is still listed as unsupported —
G4 will implement and export `NativeDropout` without moving that
boundary, and `dropout` leaves the unsupported list only at **G10**,
after the full closure matrix passes. More
activations/math, data loaders, and CPU optimization sit beyond Phase G,
and CUDA/GPU experiments remain future work. See
[docs/roadmap.md](docs/roadmap.md) and
[docs/release_history.md](docs/release_history.md).

TensorForge is a from-scratch look at how a deep learning framework
works under the hood — not a PyTorch replacement. Start reading at
`src/tensorforge/tensor.py`, then cross the ctypes boundary at
`src/tensorforge/backends/cpp.py`.
