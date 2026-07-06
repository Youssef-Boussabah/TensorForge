"""Train an MLP on a 3-class spiral — TensorForge's first real classifier.

Each class is one arm of a spiral, so no straight line (and no single
Linear layer) can separate them: the model needs hidden layers to bend
the decision boundary around the arms. Targets are integer class IDs
and the loss is cross-entropy, the standard setup for multi-class
classification.

Run it with:

    uv run python examples/train_multiclass.py
"""

import numpy as np

from tensorforge import Tensor
from tensorforge.nn import Linear, Sequential, Tanh, cross_entropy
from tensorforge.optim import SGD

NUM_CLASSES = 3


def make_spiral(points_per_class, num_classes=NUM_CLASSES, noise=0.15):
    """Build a spiral dataset: inputs (N, 2), integer targets (N,).

    Each class j is a set of points along an arm that starts at its own
    angle and winds outward, with a little angular noise so the arms
    are fuzzy instead of perfect curves.
    """
    n = points_per_class
    x = np.zeros((n * num_classes, 2))
    y = np.zeros(n * num_classes, dtype=int)
    for j in range(num_classes):
        rows = slice(n * j, n * (j + 1))
        radius = np.linspace(0.05, 1.0, n)
        start_angle = j * 2.0 * np.pi / num_classes
        angle = np.linspace(start_angle, start_angle + 2.5, n)
        angle = angle + np.random.randn(n) * noise
        x[rows] = np.column_stack([radius * np.sin(angle), radius * np.cos(angle)])
        y[rows] = j
    return x, y


def train(
    epochs=1000,
    lr=0.1,
    hidden_size=16,
    num_samples_per_class=40,
    seed=0,
    verbose=True,
):
    """Train the classifier and return a dictionary of training stats."""
    np.random.seed(seed)  # fixes both the dataset and the weight init

    x_np, y_np = make_spiral(num_samples_per_class)
    x = Tensor(x_np)

    # Two hidden layers give the model enough capacity to wrap its
    # decision boundary around the spiral arms. The last layer outputs
    # raw scores (logits) — cross_entropy applies softmax internally.
    model = Sequential(
        Linear(2, hidden_size),
        Tanh(),
        Linear(hidden_size, hidden_size),
        Tanh(),
        Linear(hidden_size, NUM_CLASSES),
    )
    optimizer = SGD(model.parameters(), lr=lr)

    def evaluate():
        """Current loss and accuracy on the full dataset."""
        logits = model(x)
        loss = cross_entropy(logits, y_np)
        predictions = np.argmax(logits.data, axis=1)
        accuracy = float((predictions == y_np).mean())
        return logits, loss, accuracy

    losses = []
    accuracies = []
    for epoch in range(epochs):
        logits, loss, accuracy = evaluate()
        losses.append(float(loss.data))
        accuracies.append(accuracy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and epoch % 100 == 0:
            print(f"epoch {epoch:04d} | loss {float(loss.data):.4f} | accuracy {accuracy:.1%}")

    # One last forward pass to measure the model after the final update.
    _, final_loss_tensor, final_accuracy = evaluate()
    final_loss = float(final_loss_tensor.data)
    losses.append(final_loss)
    accuracies.append(final_accuracy)

    if verbose:
        print()
        print(f"final loss = {final_loss:.4f}")
        print(f"final accuracy = {final_accuracy:.1%}")

    return {
        "model": model,
        "initial_loss": losses[0],
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "losses": losses,
        "accuracies": accuracies,
    }


def main():
    train()


if __name__ == "__main__":
    main()
