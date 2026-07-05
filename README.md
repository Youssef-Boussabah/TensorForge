# TensorForge

A mini deep learning framework built from scratch in Python and NumPy —
a tiny, educational take on PyTorch.

What exists so far:

- **Tensor** with reverse-mode autograd (`+`, `*`, `-`, `/`, `**`, `@`,
  `sum`, `mean`, `relu`, `exp`, `log`, `tanh`, `sigmoid`)
- **tensorforge.nn** — `Parameter`, `Module`, `Linear`, `ReLU`, `Sigmoid`,
  `Tanh`, `Sequential`, `mse_loss`
- **tensorforge.optim** — `SGD`

## Quick taste

```python
from tensorforge import Tensor

x = Tensor(4.0, requires_grad=True)
y = x * x
y.backward()

print(x.grad)  # 8.0
```

## Run the training examples

Train a `Linear(1, 1)` model to recover the line `y = 2x + 1`:

```
uv run python examples/train_regression.py
```

It prints the loss as it falls and the learned weight (≈ 2.0) and
bias (≈ 1.0).

Train a small MLP (`Linear → Tanh → Linear → Sigmoid`) to solve XOR —
the classic problem a single linear layer cannot learn:

```
uv run python examples/train_xor.py
```

It prints the loss as it falls and the final predictions for all four
XOR inputs (near 0, 1, 1, 0).

## Run the tests

```
uv run pytest tests/ -v
```
