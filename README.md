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
- **A native stochastic-training proof**:
  `examples/native_dropout_training.py` trains a
  `Linear → BatchNorm1d → ReLU → Dropout → LayerNorm → Linear` classifier
  over raw logits with `NativeCrossEntropyLoss` and native Adam — the
  smallest model carrying **all four** TensorForge-owned state families at
  once (parameters, persistent BatchNorm buffers, a registered
  `NativeGenerator`, and Adam moments) — then checkpoints after 7 completed
  steps, releases that model, and resumes into a completely fresh
  model/optimizer/generator set built from a *different* Dropout seed,
  reproducing the whole loss sequence, every parameter, both running
  statistics, the full optimizer state, the generator's seed and call
  counter, the final training logits, and the final evaluation output
  **exactly** (checkpoint format version 2). External loop progress travels
  as explicit, validated metadata — a checkpoint captures TensorForge-owned
  state, not a data loader, a shuffle order, a scheduler, Python's `random`,
  or NumPy's global RNG.

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
uv run python examples/native_dropout_training.py         # native Dropout training + exact STOCHASTIC resume
uv run python benchmarks/benchmark_native_autograd.py --smoke
uv run python benchmarks/benchmark_native_cnn.py --smoke  # CNN characterization
uv run python benchmarks/benchmark_native_classification.py --smoke        # classification characterization
uv run python benchmarks/benchmark_native_classification.py --smoke --json # machine-readable JSON
uv run python benchmarks/benchmark_native_normalization.py --smoke         # normalization characterization
uv run python benchmarks/benchmark_native_normalization.py --smoke --json  # machine-readable JSON
uv run python benchmarks/benchmark_native_dropout.py --smoke               # Dropout characterization
uv run python benchmarks/benchmark_native_dropout.py --smoke --json        # machine-readable JSON
uv run python benchmarks/benchmark_native_cpu_performance.py --smoke       # Phase-H CPU baseline
uv run python benchmarks/benchmark_native_cpu_performance.py --workload matmul
uv run python benchmarks/benchmark_native_cpu_performance.py --profile matmul_square_contiguous
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
- [docs/native_rng_dropout_design.md](docs/native_rng_dropout_design.md) — architecture contract for native RNG and Dropout (Phase G — **complete**: milestone G0, the design lock, milestone G1, `NativeGenerator` and module generator-state ownership, milestone G2, the stateless `dropout_forward` **Core** kernel and its C ABI, milestone G3, the differentiable `NativeTensor.dropout(p, *, generator)` with its graph-owned saved mask and generator call transaction, milestone G4, the `NativeDropout` module and its public export, milestone G5, native checkpoint **format version 2** — persisted generator state with its shared-generator alias topology, strict topology validation, version-1 compatibility rules, and the whole-checkpoint load transaction — and milestone G6, the RNG/graph/ownership/checkpoint hardening that added no capability, and milestone G7, the deterministic stochastic training example and its exact checkpoint resume (no capability), and milestone G8, the honest benchmark characterization `benchmarks/benchmark_native_dropout.py` (also no capability — correctness gated before timing, no speed asserted), and milestone G9, the cross-cutting integration suite `tests/test_native_phase_g.py` (integration evidence only — no capability, and no runtime file changed), and milestone G10, the phase closure — the Release/Debug/sanitizer validation matrix, the documentation reconciliation, and the single registry line that finally removed `dropout` from `UNSUPPORTED` — are all complete, so end-to-end **exact stochastic training resume is demonstrated** and native Dropout is now supported on the experimental native float64 CPU line)
- [docs/native_cpu_performance_design.md](docs/native_cpu_performance_design.md) — architecture contract for native CPU performance and runtime efficiency (Phase H — the **current** phase, **begun at milestone H0 only**: the design lock, the unified baseline harness `benchmarks/benchmark_native_cpu_performance.py`, its contract tests, and documentation reconciliation. H0 is architecture, profiling, and baseline work — **no performance optimization has shipped**, no numerical capability, dtype, device, export, registry value, or checkpoint version changed, and the proposed H1–H8 ladder is explicitly evidence-driven and conditional, so a milestone whose premise the measurement does not support is narrowed, reordered, or dropped. **Milestone H1 — the output-allocation contract — has since shipped**: redundant zero-initialization removed from output storage a kernel provably overwrites in full, behind one new C ABI symbol, bit-identical, with the zero-initializing path still the default, `sum` and `narrow_backward` explicitly rejected, completeness proved by deterministic poison tests, and no capability, dtype, device, export, registry value, or checkpoint version changed. **Milestone H2 — native matmul memory access — has since shipped too**: the production matmul's loop order swapped from `i`-`j`-`k` to `i`-`k`-`j` over four destination rows at a time, with **cache blocking measured and rejected**, the pre-H2 triple loop retained verbatim as the shipped generic reference path, metadata-driven dispatch between them inside the kernel, a four-part numerical contract rather than a blanket bit-identity claim (identical accumulation order, bit identity on every non-NaN result, NaN-class equivalence, and NaN payload bits deliberately outside the contract), H1's uninitialized-output contract preserved on both paths, **no exported C ABI symbol added** (still 52), and no capability, dtype, device, registry value, or checkpoint version changed)

## Limitations

Honest expectations:

- Not production-ready — clarity and correctness take priority over
  performance everywhere, in both lines.
- The stable framework is NumPy on CPU. The native line is an
  experimental C++ **CPU** backend: float64/cpu only, no CUDA backend
  yet, no dtype promotion or casting, and no implicit dispatch into
  `tensorforge.Tensor`.
- The native CNN stack (Phase D), the native classification stack
  (Phase E), the native normalization stack (Phase F), and native RNG and
  Dropout (Phase G) are all
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
  Release/Debug builds and Clang ASan/UBSan/LeakSanitizer. Phase G then
  added native RNG and Dropout — `NativeGenerator`, the stateless Dropout
  Core, the differentiable `NativeTensor.dropout`, and `NativeDropout` —
  closing at G10 under the same validation matrix. What the
  native line still does **not** have: data loaders, native integer
  tensors, further dtypes or devices, CUDA, AMP, a generic random-number
  API, `Dropout2d`/`Dropout3d`, and any implicit
  dispatch into `tensorforge.Tensor`. Native checkpoints capture
  parameters, persistent buffers, optimizer state, and generator state,
  but **no** data-loader position, shuffle or epoch state, scheduler
  state, Python `random`, or NumPy global RNG, and the classification
  loss supports
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
`tensorforge.experimental` namespaces — has completed Phases A–G**:
Phase A (native CPU runtime),
Phase B (native autograd), Phase C (the native training stack),
Phase D (the native CNN stack), Phase E (native classification and
stable math), and Phase F (native normalization and stateful buffers)
are all complete, and so is Phase G (native RNG and Dropout), the latest
completed native phase. **Phase H — native CPU performance and runtime
efficiency — is the current phase and has begun at milestone H0**, which
is architecture, profiling, and baseline work only: it shipped the design
contract, the unified measurement harness, and its tests, and **no
performance optimization, numerical capability, dtype, device, export,
registry value, or checkpoint version came with it**. Phase C shipped
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

**Phase G — Native RNG and Dropout — is complete, and is the latest
*completed* phase.** All eleven milestones, G0 through G10, have landed. **G0**
locked the architecture
contract in
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
generator state together. **G0 was design, documentation, and guardrails
only.**

**G1** then shipped the state itself: `NativeGenerator` — a pure-Python
value holder carrying the algorithm identity, a 64-bit seed, and a
counter of *committed* stochastic calls, with atomic
`state()`/`load_state()`/`reseed()`/`reset()`, identity (never value)
semantics, no copying and no `close()`, and a lock-protected
token-validated call transaction — plus generators as a **fourth**
`NativeModule` registration category beside parameters, buffers, and
child modules (`register_generator`, `generators()`,
`named_generators()`, `generator_state_dict()`,
`load_generator_state_dict()`). **G1 generates no random values by
itself.**

**G2** then shipped the stateless Dropout-forward **Core**: the exact
`tensorforge.splitmix64` derivation in unsigned 64-bit arithmetic
(`mix64(seed + GOLDEN*(call+1))`, then `mix64(stream + GOLDEN*(i+1))`,
then `(bits >> 11) * 2**-53` compared against `p`), an inverted-Dropout
float64 CPU kernel in `cpp/src/random.cpp`, the guarded
`tf_core_dropout_forward` export, and
`NativeTensorCore.dropout_forward(p, *, seed, call_index)` with the
private `_dropout_forward_with_mask` that also returns the multiplier
mask. It is **stateless**: the whole random key arrives as two explicit
integers, and the Core touches no `NativeGenerator` — no reservation, no
commit, no counter movement — while no C++ translation unit holds random
state of any kind. The keep/drop pattern is keyed by the **logical**
row-major element index, so a transposed, narrowed, or nonzero-offset
view gets the same mask as a contiguous tensor of the same shape.
Committed known-answer vectors are asserted identically from C++ and
Python.

**G3** then shipped the differentiable operation over that Core:

```python
from tensorforge.experimental import NativeGenerator, NativeTensor

g = NativeGenerator(1234)
x = NativeTensor.from_array(values, requires_grad=True)
y = x.dropout(0.25, generator=g)        # g.calls: 0 -> 1
y.backward(gradient=upstream)           # grad = upstream * the saved mask
```

The generator is **required and keyword-only** — there is no default,
process-global, or module-global stream, no implicit per-call generator,
and no NumPy or Python `random` fallback. One successful stochastic
forward consumes exactly one call; `p == 0` returns the input object
itself and consumes none; every failure before the commit cancels the
reservation, so the next forward reuses the same index; and backward
consumes none, ever. Backward reads only the upstream gradient and the
**graph-owned multiplier mask** the forward saved — the third member of
the saved-state family beside MaxPool2d's winners and cross-entropy's
probabilities — so it never rereads the input, never redraws, and never
touches the generator, which is why mutating the input or reseeding the
generator afterwards leaves an existing graph's gradient exactly as it
was. There is no Dropout backward kernel: the gradient is the existing
native `multiply`. The whole registry footprint is one name, `dropout`,
in `AUTOGRAD_OPS`.

**G4** then shipped the public module over that operation:

```python
from tensorforge.experimental import NativeDropout, NativeGenerator

layer = NativeDropout(0.5, seed=1234)   # owns its generator
shared = NativeGenerator(7)             # ...or two layers share one stream
a, b = NativeDropout(0.5, generator=shared), NativeDropout(0.5, generator=shared)

layer.train();  y = layer(x)            # stochastic; calls 0 -> 1
layer.eval();   assert layer(x) is x    # identity; consumes nothing
```

`NativeDropout(p=0.5, seed=None, generator=None)` — `seed` and
`generator` are **mutually exclusive**; without an explicit generator the
module creates and owns one, and with one it registers **that exact
object**, never a copy. The generator is first-class registered state:
it appears in `named_generators()` and `generator_state_dict()` and is
deliberately absent from `state_dict()`, which stays tensor-only. Training
delegates to the G3 operation, so a successful forward consumes exactly
one call and a failed one none; evaluation returns the input object
itself, so any number of eval forwards leaves **no gap in the stream**;
and `p == 0` is identity in both modes. The module owns no native storage.

**G5** closed the persistence gap G4 left open: the native checkpoint
format is now **version 2** (the format *name* is unchanged), and it
carries a `generators` manifest section holding every registered
generator's `algorithm`, `algorithm_version`, `seed`, and `calls` — seeds
and counters as canonical decimal strings, because a `uint64` above
`2**53` cannot survive a JSON double — plus the complete
**alias topology**: every registered path mapped to its canonical
generator, so *shared versus independent* streams are restored, not just
the numbers. Generator state adds **no array** to the archive. A load
restores each generator **in place**, preserving object identity and
every sharing relationship, and validates the archive's topology strictly
in both directions against a real `named_generators()` traversal — a
missing path, an extra path, or a shared-versus-independent difference
fails before anything changes. A version-1 archive still loads into a
model with **no** generators and is **rejected** for one that has them,
naming them: no seed and no counter is ever fabricated. A load is one
transaction over the whole archive — model, buffers, optimizer, and
generators commit under a single rollback guard, so any synchronous
failure (a deliverable `KeyboardInterrupt` included) restores all four
together rather than leaving a mixed checkpoint. It is also
**serializable**, not merely deadlock-free: every participating state
replacement — the checkpoint load, `load_state_dict`,
`load_generator_state_dict`, both optimizers' state loads — and the
checkpoint save snapshot run under one private shared `RLock`, with
generator locks taken under it in the global `id()` order. Two concurrent
loads therefore produce one archive's state followed by the other's,
never model state from one beside optimizer or generator state from the
other. Ordinary training mutation deliberately does not take that guard,
so thread-safe concurrent training snapshots are not claimed.

Milestone **G6 is complete** — hardening, and no new capability.
`tests/test_native_phase_g_hardening.py` attacks the finished G1–G5
surface: the reservation transition matrix, the exact `uint64` boundary,
forced concurrent interleavings (bounded joins, no sleeps), the
deterministic Core's structural key properties beside its committed
vectors, every pre-commit and post-commit failure position of the call
transaction across four exception classes, all four graph-owned
saved-resource families in one graph, a 76-case checkpoint corruption
matrix, whole-transaction rollback at every commit position, save-seam
destination atomicity, and repeated success-and-failure lifecycle loops
measured against a real native live-storage baseline. It changed no C++,
ABI, ctypes, Core method, operation, module, export, schema field, or
registry value, and added no benchmark and no example; it found and fixed
exactly one runtime defect — a cleanup-failure `__context__` chain that
could become cyclic — with a dedicated regression guard.

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
measures thirty-five cases in eight families: the stateless Core against
an **exact, bit-for-bit** vectorized NumPy implementation of the same
locked derivation, size scaling from a rank-0 scalar to a
six-figure-element tensor, four physical layouts over one logical shape
(contiguous, transposed, narrowed non-contiguous, and offset-contiguous),
a five-value probability sweep at three layers, the no-grad /
differentiable / backward-only / forward-plus-backward operation layers,
the module's training and identity paths, and one complete Dropout
training step. Correctness is gated **before** timing everywhere — the
committed known-answer vectors first pin the harness's reference and then
the native kernel — and the `NativeTensor` and `NativeDropout` cases are
labelled `native_only`, publishing no ratio, because no NumPy expression
has their generator transaction, ownership, or graph. `--smoke`
(`--quick`), `--json`, and `--json-out` are supported; **no result file
is written unless a destination is named**, and there is no speed
assertion, no committed timing number, and no CI timing threshold
anywhere. Results are a machine-specific snapshot, not a performance
contract, and **nothing was optimized to improve a number** — G8 changed
no runtime file.

Milestone **G9 is complete** — the cross-cutting Phase-G integration
suite, and **no new capability**. `tests/test_native_phase_g.py` builds one
test-only model that carries every registered state family at once —
`NativeConv2d` -> `NativeBatchNorm2d` -> `NativeReLU` -> `NativeMaxPool2d`
-> `NativeDropout` -> `NativeFlatten` -> `NativeLinear` ->
`NativeBatchNorm1d` -> `NativeReLU` -> `NativeLayerNorm` ->
`NativeDropout` -> `NativeLinear` over raw logits with
`NativeCrossEntropyLoss` — with the two Dropout layers sharing **one**
registered generator, and proves the interactions no single-module suite
can: all four saved-resource families (Dropout masks, MaxPool2d winners,
BatchNorm eval snapshots, and cross-entropy probabilities) alive in one
graph and released exactly once; deterministic training and **exact**
version-2 resume into a completely fresh model, optimizer, and generator
set, with a negative control that diverges; the generator-topology matrix
(shared, independent, equal-valued-but-distinct, renamed, missing, extra)
with every mismatch rejected **before** any state family changes;
evaluation consuming no generator call anywhere and training resuming at
the exact next index; `p == 0` through the whole model; non-contiguous
NCHW and strided views; the whole-checkpoint transaction rolled back at
every commit position; four deterministic concurrency cases proving the
participating state transactions serialize; a Phase A–F regression
matrix; and native live storage returning exactly to baseline across
success and failure cycles. It changed **no** runtime file and found no
defect.

Milestone **G10 is complete, and it closed the phase.** The validation
matrix ran with observed results: fresh Windows **Release** and **Debug**
builds (Visual Studio 17 2022, MSVC 19.44.35228.0), each passing
**11/11 native CTests** with zero project compiler, linker, and CMake
warnings, and the active runtime proved to remain the Release DLL. A fresh
Clang 18.1.3 `-DTF_SANITIZE=address,undefined` build in WSL2 Ubuntu
24.04.4 with **instrumentation proved rather than assumed** — 22 `__asan*`
and 14 `__ubsan*` dynamic symbols beside the 51 exported `tf_*` symbols,
and a library that refuses to load without the sanitizer runtime — then
**11/11 sanitized CTests** with leak detection on, **3,166 sanitized
Python tests** across 43 suites, the G7 example reproducing its exact
resume, and the G8 benchmark passing every correctness gate, all with
**zero ASan and zero UBSan diagnostics**. A practical LeakSanitizer
lifecycle returned native live storage **exactly to baseline (0 → 0)**,
and not one of its remaining process-exit allocations names a TensorForge
frame — **no suppression file was added**.

Only after all of that did the single registry line move: at **G10**, and
not one milestone earlier, `dropout` **left** the unsupported list, which
now reads exactly `("float32", "cuda", "amp")`. The claim is deliberately narrow — **native Dropout is
supported in TensorForge's experimental native float64 CPU backend**. That
is not a stable-framework claim (`tensorforge.nn.Dropout` is its own
separate NumPy implementation), not float32, CUDA, or AMP support, not a
generic random-number API, and not a production-readiness or speed claim.
Reproducibility is exact for the state actually captured; Python's
`random`, NumPy's global RNG, data-loader position, and scheduler state
are **not** captured and full-program determinism is not claimed. Ordinary
concurrent *training* is not claimed thread-safe either: the
serializability guarantee covers the participating state transactions, not
an optimizer `step()` racing a forward.

**Phase H — Native CPU Performance and Runtime Efficiency — is the
current phase, and it has begun at milestone H0 only.** H0 is an
architecture, profiling, and baseline milestone: it locked the contract
in
[docs/native_cpu_performance_design.md](docs/native_cpu_performance_design.md),
shipped `benchmarks/benchmark_native_cpu_performance.py` — a unified
measurement harness that separates the layers a caller actually pays for
(NumPy, the stable line, the raw-buffer kernels, `NativeTensorCore`,
`NativeTensor` with and without a graph, backward, an optimizer step, and
a whole training step) across twelve workload families, gates correctness
**before** timing everywhere, publishes no ratio where no honest
equivalent exists, and writes no result file — and its contract tests.

**Nothing was made faster.** Every kernel is exactly the deliberately
plain reference loop Phase G left behind; no numerical capability, dtype,
device, export, capability-registry value, or checkpoint version changed,
and the checkpoint format remains version 2 with versions 1 and 2
supported. What H0 produced is *evidence*, and it is deliberately
surprising in places: the largest measured factors are an allocator
behavior and a memory access pattern rather than raw arithmetic, the
Python-side per-call metadata path costs several times the ctypes
boundary it wraps, and the `NativeTensor` wrapper and its autograd graph
node are measurably **not** a bottleneck — a negative result that rules
out a family of plausible optimizations.

**Milestone H1 — the output-allocation contract — has since shipped**, and
it is the first Phase-H change to production code. Every native operation
allocates a fresh owning output, and native storage was value-initialized
on construction — a full write pass over a buffer most kernels then
overwrite completely. H1 added one C ABI symbol,
`tf_storage_create_uninitialized`, identical to the zero-initializing
constructor in size validation, allocation-failure handling, error state,
ownership, destruction, and live-storage accounting, and differing only in
the buffer's initial contents. The zero-initializing path is still the
default: there is no global allocator policy, no environment variable, no
heuristic, no memory pool, no scratch arena, and **no public
empty-tensor API**. Each call site opts in explicitly against a
per-kernel audit table, and two operations are deliberately **rejected** —
`sum`/`mean`, which accumulates into its output, and `narrow_backward`,
whose untouched zeros *are* the gradient.

Because ASan and UBSan do not detect uninitialized-value reads (that is
MemorySanitizer's job, and MSan needs an instrumented libc and CPython,
which this project does not have), completeness is proved separately, by
**deterministic poison tests**: an uninitialized allocation is filled with
a quiet NaN or a nontrivial finite pattern, and no poison may survive into
a result. The poison is applied **entirely by test infrastructure wrapped
around the allocator** — the real constructor allocates, the test fills
the returned storage through the ordinary fill primitive, and that same
storage goes to the real operation — so **the shipped library and the
installed Python backend contain no poison control at all**: no exported
hook, no thread-local flag, no environment variable, no global mode.
Negative controls prove the detector can actually fail. H1 is
bit-identical — every enabled operation and a complete training run are
compared element-wise against the zero-initializing allocator — and no
capability, dtype, device, registry value, or checkpoint version changed;
`tf_storage_create_uninitialized` is its only added export, taking the
library from 51 exported `tf_*` symbols to 52. The measured result is reported honestly rather than as a headline: isolated, the zero-fill is enormous and scales with the buffer (about 52x at 2 MB, 119x at 8 MB, 552x at 32 MB, and *negative* below roughly 16,000 elements, where it sits inside the noise). End to end it is much smaller and often inconclusive — clearly real for large memory-bound elementwise work (about 1.5-1.8x on an 8 MB output), small and variable for normalization and Adam, and with no measurable effect on Conv2d, the MLP step, or matmul, whose arithmetic dwarfs its allocation. Those inconclusive and negative rows are published as such.

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

The proposed H3–H8 ladder is
explicitly conditional on that evidence: a milestone whose premise the
measurements do not support is narrowed, reordered, or dropped, and
allocation pooling, SIMD, threading, and BLAS are all currently
**rejected on evidence**, with the criteria that would reopen each one
recorded rather than an answer invented. Every number in that document is
a local characterization of one machine, reported with its spread, and
asserted by no test — there is no CI timing threshold anywhere in this
repository.

More activations/math, data loaders, further dtypes, and CUDA/GPU
experiments remain future work beyond Phase H. See
[docs/roadmap.md](docs/roadmap.md) and
[docs/release_history.md](docs/release_history.md).

TensorForge is a from-scratch look at how a deep learning framework
works under the hood — not a PyTorch replacement. Start reading at
`src/tensorforge/tensor.py`, then cross the ctypes boundary at
`src/tensorforge/backends/cpp.py`.
