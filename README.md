# TensorForge

A from-scratch deep learning framework built in Python and NumPy —
a serious, Daedalus-inspired ML systems project covering PyTorch-style
framework internals: autograd, modules, optimizers, checkpointing,
CNN support, and an experimental native C++ backend.

Everything is implemented by hand and kept readable: the autograd
engine, the layers, the optimizers, the losses. NumPy is the only
numeric dependency. If you want to see what `loss.backward()`
actually does, this repo is a few hundred lines away from showing you.

**Framework internals covered:** reverse-mode autograd, how neural
network modules and parameters fit together, what optimizers actually
do, regularization (Dropout) and normalization (BatchNorm, LayerNorm),
checkpoint/resume mechanics down to the RNG, the internals of
convolution and pooling, and native-backend mechanics (ctypes-loaded
C++ kernels with honest benchmarks).

## Features

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

## Quickstart

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/):

```
uv sync
uv run pytest
uv run python examples/train_mlp_with_dropout.py
```

The API looks the way you'd expect:

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

## Examples

Six runnable, seeded examples forming a progression — each one
introduces a single idea:

```
uv run python examples/train_linear_regression.py    # the bare training loop
uv run python examples/train_xor.py                  # why hidden layers exist
uv run python examples/train_multiclass.py           # real classification + mini-batches
uv run python examples/train_binary_classification.py  # logits and stable losses
uv run python examples/train_mlp_with_dropout.py     # train mode vs eval mode
uv run python examples/train_tiny_cnn.py             # convolution and pooling
```

What each one teaches, and what to expect: [docs/examples.md](docs/examples.md).

## Documentation

- [docs/project_summary.md](docs/project_summary.md) — the whole project in two minutes
- [docs/architecture.md](docs/architecture.md) — how the package fits together
- [docs/autograd.md](docs/autograd.md) — the autograd engine, explained
- [docs/training.md](docs/training.md) — training loops, train/eval mode, saving
- [docs/examples.md](docs/examples.md) — the examples and what they teach
- [docs/roadmap.md](docs/roadmap.md) — what's done and what's next
- [docs/release_history.md](docs/release_history.md) — how the project grew, by version
- [docs/backend_experiments.md](docs/backend_experiments.md) — the experimental C++ backend line
- [docs/dispatch_design.md](docs/dispatch_design.md) — how backends might eventually meet the Tensor
- [docs/native_tensor_wrapper_design.md](docs/native_tensor_wrapper_design.md) — Stage-2 design for a forward-only native tensor wrapper
- [docs/native_contiguous_fast_path_design.md](docs/native_contiguous_fast_path_design.md) — design for a contiguous elementwise fast path in the native runtime
- [docs/native_broadcasting_design.md](docs/native_broadcasting_design.md) — design for NumPy-style broadcasting in the native elementwise runtime

## Limitations

Honest expectations:

- Not production-ready — clarity and correctness take priority over
  performance everywhere.
- NumPy on CPU only. No C++ backend yet, no CUDA backend yet — both
  are future experiments, not current features.
- `Conv2d` and `MaxPool2d` use deliberately naive loops.
- No real datasets and no external ML libraries; every example runs on
  small synthetic data.

## Status

**v3.0 — the Python framework line is complete.** Everything
above works, is covered by 370+ tests, and is documented. Advanced
work now happens in experimental branches: a C++ backend experiment
has started (ctypes-loaded elementwise kernels and a naive matmul,
proving the mechanism — see
[docs/backend_experiments.md](docs/backend_experiments.md)); CUDA/GPU
experiments are still future work. There is still no production C++
backend yet and no CUDA backend yet. See
[docs/roadmap.md](docs/roadmap.md) and
[docs/release_history.md](docs/release_history.md).

TensorForge is a from-scratch look at how a deep learning framework
works under the hood — not a PyTorch replacement. Start reading at
`src/tensorforge/tensor.py`.
