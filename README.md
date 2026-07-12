# TensorForge

A serious from-scratch deep learning and ML systems framework in two
lines: a **stable Python framework** built on NumPy — PyTorch-style
autograd, modules, optimizers, checkpointing, and CNN support, complete
as of v3.0 — and an **experimental native C++ CPU backend** (this
advanced branch) with its own explicit tensor runtime, Python-managed
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

The advanced branch carries a complete native CPU training line,
reached explicitly through `tensorforge.experimental` and
`tensorforge.backends` (the stable framework never imports it):

- **Native runtime**: `NativeStorage` → `NativeTensorView` →
  `NativeTensorCore` → `NativeTensor` — explicit ownership and
  lifetime, shapes/strides/offsets, borrowing views, contiguity
  tracking, NumPy-style broadcasting, sum/mean reductions, and
  float64/cpu dtype/device metadata over ctypes-loaded C++ kernels.
- **Native autograd (Phase B, complete)**: a Python-managed
  reverse-mode graph over autograd-unaware kernels — fourteen
  differentiable operations (`add`, `subtract`, `multiply`, `relu`,
  `sqrt`, `reciprocal`, `matmul`, `sum`, `mean`, `reshape`,
  `transpose`/`T`, `narrow`, `contiguous_copy`), broadcasting and view
  gradients, a native scatter backward for `narrow`, one-shot graph
  release with `retain_graph` opt-in, and failure rollback.
- **Native training stack (Phase C, in progress)**: `NativeParameter`
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
uv run python benchmarks/benchmark_native_autograd.py --smoke
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
The four native examples are listed in the native quickstart above.

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

## Limitations

Honest expectations:

- Not production-ready — clarity and correctness take priority over
  performance everywhere, in both lines.
- The stable framework is NumPy on CPU. The native line is an
  experimental C++ **CPU** backend: float64/cpu only, no CUDA backend
  yet, no dtype promotion or casting, and no implicit dispatch into
  `tensorforge.Tensor`.
- The native stack has no CNN layers, and native checkpoints capture
  no scheduler or random state — see the
  [native support matrix](docs/native_support_matrix.md).
- `Conv2d` and `MaxPool2d` (stable line) use deliberately naive loops.
- Benchmarks are hardware-specific characterizations with no universal
  speed claims; the naive native kernels can lose to NumPy's BLAS.
- No real datasets and no external ML libraries; every example runs on
  small synthetic data.

## Status

**v3.0 — the stable Python framework line is complete**, covered by the
test suite and documented. **The advanced branch has reached its first
major native checkpoint (Advanced C++ v3.10)**: Phase A (native CPU
runtime) and Phase B (native autograd) are complete, and Phase C (the
native training stack) has shipped parameters, modules, state
dictionaries, Linear/ReLU/Sequential, MSE loss, SGD, parameter
versioning with stale-graph safety, and an end-to-end deterministic MLP
training proof — 1264 tests pass. CUDA/GPU experiments have not
started and remain future work. See [docs/roadmap.md](docs/roadmap.md)
and [docs/release_history.md](docs/release_history.md).

TensorForge is a from-scratch look at how a deep learning framework
works under the hood — not a PyTorch replacement. Start reading at
`src/tensorforge/tensor.py`, then cross the ctypes boundary at
`src/tensorforge/backends/cpp.py`.
