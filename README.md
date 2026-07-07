# TensorForge

An educational deep learning framework built from scratch in Python
and NumPy — a tiny, readable take on PyTorch, with autograd, modules,
optimizers, checkpointing, and basic CNN support.

Everything is implemented by hand and kept readable: the autograd
engine, the layers, the optimizers, the losses. NumPy is the only
numeric dependency. If you want to understand what `loss.backward()`
actually does, this repo is a few hundred lines away from telling you.

**What it teaches:** reverse-mode autograd, how neural network modules
and parameters fit together, what optimizers actually do,
regularization (Dropout) and normalization (BatchNorm, LayerNorm),
checkpoint/resume mechanics down to the RNG, and the internals of
convolution and pooling.

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

## Limitations

Honest expectations:

- Educational, not production-ready — clarity beats performance every
  time.
- NumPy on CPU only. No C++ backend yet, no CUDA backend yet — both
  are future experiments, not current features.
- `Conv2d` and `MaxPool2d` use deliberately naive loops.
- No real datasets and no external ML libraries; every example runs on
  small synthetic data.

## Status

**v3.0 — the Python educational framework is complete.** Everything
above works, is covered by 370+ tests, and is documented. Advanced
work now happens in experimental branches: a C++ backend experiment
has started (a handful of ctypes-loaded elementwise kernels proving
the mechanism — see
[docs/backend_experiments.md](docs/backend_experiments.md)); CUDA/GPU
experiments are still future work. There is still no production C++
backend yet and no CUDA backend yet. See
[docs/roadmap.md](docs/roadmap.md) and
[docs/release_history.md](docs/release_history.md).

TensorForge exists to show how a deep learning framework works under
the hood, not to compete with real ones. Start reading at
`src/tensorforge/tensor.py`.
