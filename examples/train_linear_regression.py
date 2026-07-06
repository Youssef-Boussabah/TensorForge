"""TensorForge's first end-to-end training example.

Trains a Linear(1, 1) model to recover the line y = 2x + 1 from data.
Run it with:

    uv run python examples/train_linear_regression.py
"""

import numpy as np

from tensorforge import Tensor
from tensorforge.nn import Linear, mse_loss
from tensorforge.optim import SGD


def train(epochs=300, lr=0.05, seed=0, log_every=None):
    """Train the model and return (model, final_loss)."""
    np.random.seed(seed)  # stable weight init and data, so runs are reproducible

    # Dataset: 20 points on the line y = 2x + 1, with a little noise so
    # the model has to average rather than memorize.
    x_data = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)   # shape (N, 1)
    y_data = 2.0 * x_data + 1.0 + np.random.randn(20, 1) * 0.01

    x = Tensor(x_data)
    y = Tensor(y_data)

    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=lr)

    loss = None
    for epoch in range(1, epochs + 1):
        # 1. Forward pass: run the data through the model.
        pred = model(x)

        # 2. Compute how wrong the predictions are.
        loss = mse_loss(pred, y)

        # 3. Clear old gradients — backward() accumulates, so without
        #    this every epoch would add onto the previous gradients.
        optimizer.zero_grad()

        # 4. Backward pass: autograd fills in param.grad for every
        #    parameter that contributed to the loss.
        loss.backward()

        # 5. Descend: nudge each parameter against its gradient.
        optimizer.step()

        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:4d}  loss {float(loss.data):.6f}")

    return model, loss


def main():
    model, loss = train(epochs=300, lr=0.05, seed=0, log_every=50)

    weight = float(model.weight.data[0, 0])
    bias = float(model.bias.data[0])
    print()
    print(f"final loss:     {float(loss.data):.6f}")
    print(f"learned weight: {weight:.4f}  (true value: 2.0)")
    print(f"learned bias:   {bias:.4f}  (true value: 1.0)")


if __name__ == "__main__":
    main()
