# TensorForge

A mini deep learning framework built from scratch in Python and NumPy —
a tiny, educational take on PyTorch.

Everything is implemented by hand and kept readable: the autograd
engine, the layers, the optimizers, the losses. NumPy is the only
numeric dependency. If you want to understand what `loss.backward()`
actually does, this repo is a few hundred lines away from telling you.

## Features

**Core engine**
- Tensor with reverse-mode autograd: `+ - * / ** @`, `sum`, `mean`,
  `exp`, `log`, `tanh`, `sigmoid`, `relu`, `softmax`, with
  broadcasting-aware gradients — all verified against finite differences

**Models (`tensorforge.nn`)**
- `Parameter`, `Module`, `Linear`, `ReLU`/`Sigmoid`/`Tanh`, `Dropout`,
  `BatchNorm1d`, `Sequential`
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
  `save_checkpoint` / `load_checkpoint` with optimizer state and
  metadata so training resumes exactly — all plain `.npz`, no pickle

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

Five runnable, seeded examples, from a bare linear regression up to a
dropout-regularized MLP that demonstrates train/eval mode:

```
uv run python examples/train_linear_regression.py
uv run python examples/train_xor.py
uv run python examples/train_multiclass.py
uv run python examples/train_binary_classification.py
uv run python examples/train_mlp_with_dropout.py
```

What each one teaches, and what to expect: [docs/examples.md](docs/examples.md).

## Documentation

- [docs/architecture.md](docs/architecture.md) — how the package fits together
- [docs/autograd.md](docs/autograd.md) — the autograd engine, explained
- [docs/training.md](docs/training.md) — training loops, train/eval mode, saving
- [docs/examples.md](docs/examples.md) — the examples and what they teach
- [docs/roadmap.md](docs/roadmap.md) — what's done and what's next

## Status

The core educational framework is complete — model definition,
training, regularization, honest evaluation, and checkpointing all
work and are covered by 300+ tests. The v2.x milestones are expanding
it (normalization, training control, docs). See
[docs/roadmap.md](docs/roadmap.md) for direction.

TensorForge is an educational project: it exists to show how a deep
learning framework works under the hood, not to compete with real
ones. Start reading at `src/tensorforge/tensor.py`.
