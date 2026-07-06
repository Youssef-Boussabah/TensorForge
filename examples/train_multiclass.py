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

from tensorforge import Tensor, batches, train_test_split
from tensorforge.nn import (
    Linear,
    Sequential,
    Tanh,
    accuracy,
    cross_entropy,
    evaluate_classifier,
)
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
    batch_size=None,
    validation_split=0.0,
):
    """Train the classifier and return a dictionary of training stats."""
    np.random.seed(seed)  # fixes both the dataset and the weight init

    x_np, y_np = make_spiral(num_samples_per_class)
    if validation_split > 0:
        # Hold out part of the data: the model never trains on it, so
        # its metrics show how well the model generalizes.
        x_np, x_val, y_np, y_val = train_test_split(
            x_np, y_np, test_size=validation_split, shuffle=True, seed=seed
        )
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
        return logits, loss, accuracy(logits, y_np)

    losses = []
    accuracies = []
    validation_losses = []
    validation_accuracies = []
    for epoch in range(epochs):
        logits, loss, epoch_accuracy = evaluate()
        losses.append(float(loss.data))
        accuracies.append(epoch_accuracy)
        if validation_split > 0:
            val = evaluate_classifier(model, x_val, y_val)
            validation_losses.append(val["loss"])
            validation_accuracies.append(val["accuracy"])

        if batch_size is None:
            # Full-batch: one update per epoch on the whole dataset.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # Mini-batch: several smaller updates per epoch. Seeding the
            # shuffle with seed + epoch keeps the run reproducible while
            # still reshuffling every epoch.
            for xb, yb in batches(x_np, y_np, batch_size, shuffle=True, seed=seed + epoch):
                batch_loss = cross_entropy(model(Tensor(xb)), yb)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

        if verbose and epoch % 100 == 0:
            line = f"epoch {epoch:04d} | loss {float(loss.data):.4f} | accuracy {epoch_accuracy:.1%}"
            if validation_split > 0:
                line += f" | val loss {val['loss']:.4f} | val accuracy {val['accuracy']:.1%}"
            print(line)

    # One last forward pass to measure the model after the final update.
    _, final_loss_tensor, final_accuracy = evaluate()
    final_loss = float(final_loss_tensor.data)
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
        final_val = evaluate_classifier(model, x_val, y_val)
        validation_losses.append(final_val["loss"])
        validation_accuracies.append(final_val["accuracy"])
        stats["final_validation_loss"] = final_val["loss"]
        stats["final_validation_accuracy"] = final_val["accuracy"]
        stats["validation_losses"] = validation_losses
        stats["validation_accuracies"] = validation_accuracies

    if verbose:
        print()
        print(f"final loss = {final_loss:.4f}")
        print(f"final accuracy = {final_accuracy:.1%}")
        if validation_split > 0:
            print(f"final validation loss = {final_val['loss']:.4f}")
            print(f"final validation accuracy = {final_val['accuracy']:.1%}")

    return stats


def main():
    train()


if __name__ == "__main__":
    main()
