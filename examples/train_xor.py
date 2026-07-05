"""Train a small MLP to learn XOR.

XOR is the classic problem a single Linear layer cannot solve: no
straight line separates {[0,1], [1,0]} from {[0,0], [1,1]}. A hidden
layer with a nonlinearity lets the model bend the decision boundary,
which is the whole point of this example.

Run it with:

    uv run python examples/train_xor.py
"""

import numpy as np

from tensorforge import Tensor
from tensorforge.nn import Linear, Sequential, Sigmoid, Tanh, mse_loss
from tensorforge.optim import SGD


def train(epochs=3000, lr=0.5, seed=0, log_every=None):
    """Train the XOR model and return (model, final_loss)."""
    np.random.seed(seed)  # stable weight init, so runs are reproducible

    # The full XOR truth table — all four cases, no noise.
    x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = Tensor([[0.0], [1.0], [1.0], [0.0]])

    # Linear(2, 4) + Tanh is the hidden layer that makes XOR solvable:
    # without it the model collapses to a single linear map, which can
    # never output high for [0,1]/[1,0] but low for [0,0]/[1,1].
    # The final Sigmoid squashes the output into (0, 1) so it reads
    # as a probability.
    model = Sequential(
        Linear(2, 4),
        Tanh(),
        Linear(4, 1),
        Sigmoid(),
    )
    optimizer = SGD(model.parameters(), lr=lr)

    loss = None
    for epoch in range(1, epochs + 1):
        pred = model(x)
        loss = mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:4d}  loss {float(loss.data):.6f}")

    return model, loss


def main():
    model, loss = train(epochs=3000, lr=0.5, seed=0, log_every=500)

    print()
    print(f"final loss: {float(loss.data):.6f}")
    print()
    print("input     target  prediction")
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [0, 1, 1, 0]
    for point, target in zip(inputs, targets):
        pred = float(model(Tensor([point])).data[0, 0])
        print(f"{point}  {target}       {pred:.4f}")


if __name__ == "__main__":
    main()
