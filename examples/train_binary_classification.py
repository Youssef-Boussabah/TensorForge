"""Train a binary classifier on two 2-D point clouds.

This is logistic regression built from TensorForge parts: a single
Linear(2, 1) layer produces one raw logit per row, and
binary_cross_entropy turns that logit into a loss. No Sigmoid layer is
needed — the loss works on raw logits directly (which is also what
keeps it numerically stable), and a logit >= 0 already means
"class 1 is more likely".

Run it with:

    uv run python examples/train_binary_classification.py
"""

import numpy as np

from tensorforge import (
    Tensor,
    binary_accuracy,
    binary_cross_entropy,
    train_test_split,
)
from tensorforge.nn import Linear
from tensorforge.optim import SGD


def make_blobs(points_per_class=60, noise=0.7):
    """Two Gaussian clouds: class 0 around (-1, -1), class 1 around (1, 1)."""
    n = points_per_class
    class0 = np.random.randn(n, 2) * noise + np.array([-1.0, -1.0])
    class1 = np.random.randn(n, 2) * noise + np.array([1.0, 1.0])
    x = np.vstack([class0, class1])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return x, y


def train(epochs=200, lr=0.1, seed=0, validation_split=0.25, verbose=True):
    """Train the classifier and return a dictionary of training stats."""
    np.random.seed(seed)  # fixes both the dataset and the weight init

    x_np, y_np = make_blobs()
    if validation_split > 0:
        x_np, x_val, y_np, y_val = train_test_split(
            x_np, y_np, test_size=validation_split, shuffle=True, seed=seed
        )
    x = Tensor(x_np)

    model = Linear(2, 1)  # one raw logit per row
    optimizer = SGD(model.parameters(), lr=lr)

    losses = []
    accuracies = []
    validation_losses = []
    validation_accuracies = []
    for epoch in range(epochs):
        logits = model(x)
        loss = binary_cross_entropy(logits, y_np)
        losses.append(float(loss.data))
        accuracies.append(binary_accuracy(logits, y_np))
        if validation_split > 0:
            val_logits = model(Tensor(x_val))
            validation_losses.append(float(binary_cross_entropy(val_logits, y_val).data))
            validation_accuracies.append(binary_accuracy(val_logits, y_val))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and epoch % 50 == 0:
            line = f"epoch {epoch:03d} | loss {losses[-1]:.4f} | accuracy {accuracies[-1]:.1%}"
            if validation_split > 0:
                line += (
                    f" | val loss {validation_losses[-1]:.4f}"
                    f" | val accuracy {validation_accuracies[-1]:.1%}"
                )
            print(line)

    # One last measurement after the final update.
    logits = model(x)
    final_loss = float(binary_cross_entropy(logits, y_np).data)
    final_accuracy = binary_accuracy(logits, y_np)
    losses.append(final_loss)
    accuracies.append(final_accuracy)

    stats = {
        "model": model,
        "initial_loss": losses[0],
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "losses": losses,
        "accuracies": accuracies,
    }
    if validation_split > 0:
        val_logits = model(Tensor(x_val))
        final_val_loss = float(binary_cross_entropy(val_logits, y_val).data)
        final_val_accuracy = binary_accuracy(val_logits, y_val)
        validation_losses.append(final_val_loss)
        validation_accuracies.append(final_val_accuracy)
        stats["final_validation_loss"] = final_val_loss
        stats["final_validation_accuracy"] = final_val_accuracy
        stats["validation_losses"] = validation_losses
        stats["validation_accuracies"] = validation_accuracies

    if verbose:
        print()
        print(f"final loss = {final_loss:.4f}")
        print(f"final accuracy = {final_accuracy:.1%}")
        if validation_split > 0:
            print(f"final validation loss = {final_val_loss:.4f}")
            print(f"final validation accuracy = {final_val_accuracy:.1%}")

    return stats


def main():
    train()


if __name__ == "__main__":
    main()
