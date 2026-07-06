# TensorForge

A mini deep learning framework built from scratch in Python and NumPy —
a tiny, educational take on PyTorch.

Everything is implemented by hand and kept readable: the autograd
engine, the layers, the optimizers, the losses. NumPy is the only
numeric dependency.

## Features

- [x] **Tensor with reverse-mode autograd** — `+`, `*`, `-`, `/`, `**`,
  `@` (matmul), `sum`, `mean`, `exp`, `log`, `tanh`, `sigmoid`, `relu`,
  `softmax`, with broadcasting-aware gradients
- [x] **`tensorforge.nn`** — `Parameter`, `Module`, `Linear`, `ReLU`,
  `Sigmoid`, `Tanh`, `Sequential`
- [x] **Losses** — `mse_loss`, numerically stable `cross_entropy`
- [x] **Metrics** — `accuracy`
- [x] **`tensorforge.optim`** — `SGD`, `Adam`
- [x] **`tensorforge.data`** — `batches` mini-batch iterator
- [x] **Save/load parameters** — `save_parameters` / `load_parameters`
  (plain NumPy `.npz`, plus `state_dict()` on every module)
- [x] **Model inspection** — `model.summary()` and `count_parameters`;
  parameters with `requires_grad=False` are frozen: skipped by
  optimizers and excluded from trainable counts
- [x] **Runnable examples** — linear regression, XOR, 3-class spiral

## Setup

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/):

```
uv sync
```

## Quickstart

```python
from tensorforge import Tensor

x = Tensor(4.0, requires_grad=True)
y = x * x
y.backward()

print(x.grad)  # 8.0
```

Training a model looks just like you'd expect:

```python
from tensorforge import Tensor, Adam
from tensorforge.nn import Linear, ReLU, Sequential, cross_entropy

model = Sequential(Linear(2, 16), ReLU(), Linear(16, 3))
optimizer = Adam(model.parameters(), lr=0.01)

logits = model(Tensor(inputs))          # inputs: (batch, 2) array
loss = cross_entropy(logits, targets)   # targets: integer class IDs

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Trained weights can be saved and loaded into a same-architecture model:

```python
from tensorforge import save_parameters, load_parameters

save_parameters(model, "model.npz")
load_parameters(new_model, "model.npz")
```

Models can describe themselves:

```python
from tensorforge import count_parameters

print(model.summary())         # per-parameter names, shapes, counts
print(count_parameters(model)) # total trainable scalars
```

## Examples

Fit a `Linear(1, 1)` model to the line `y = 2x + 1`:

```
uv run python examples/train_linear_regression.py
```

Solve XOR with a small MLP — the classic problem a single linear layer
cannot learn:

```
uv run python examples/train_xor.py
```

Train a two-hidden-layer MLP on a 3-class spiral with cross-entropy
(reaches 100% training accuracy; also supports mini-batches via
`train(batch_size=...)`):

```
uv run python examples/train_multiclass.py
```

## Tests

Every feature is covered by tests:

```
uv run pytest
```

## Project structure

```
src/tensorforge/
  tensor.py        Tensor + autograd engine
  data.py          batches() mini-batch iterator
  nn/              Parameter, Module, Linear, activations,
                   Sequential, losses, metrics
  optim/           SGD, Adam
examples/          runnable training scripts
tests/             pytest suite
```

## Roadmap

Possible next steps, in rough order:

- Numerical gradient checking
- More layers (Dropout, normalization)
- A larger dataset example
- C++ backend experiments
- GPU/CUDA experiments

## Note

TensorForge is an educational project: it exists to show how a deep
learning framework works under the hood, not to compete with real ones.
If you want to understand autograd, read `src/tensorforge/tensor.py` —
the whole engine is a few hundred readable lines.
