"""Train a tiny CNN to tell vertical bars from horizontal bars.

The dataset is synthetic 6x6 grayscale images: each one contains a
single bright bar, either vertical (class 0) or horizontal (class 1),
plus noise. A convolution is the natural tool here — one small kernel
slid across the image can detect a bar *wherever* it appears, which is
exactly what a Linear layer on raw pixels cannot do without learning
every position separately.

Model: Conv2d -> ReLU -> Flatten -> Linear, trained with cross-entropy.

Run it with:

    uv run python examples/train_tiny_cnn.py
"""

import numpy as np

from tensorforge import Tensor, accuracy
from tensorforge.nn import Conv2d, Flatten, Linear, ReLU, Sequential, cross_entropy
from tensorforge.optim import Adam

IMAGE_SIZE = 6


def make_bars(samples_per_class=40, noise=0.15):
    """6x6 images shaped (N, 1, 6, 6): vertical bar = 0, horizontal = 1."""
    n = samples_per_class
    images = np.random.randn(2 * n, 1, IMAGE_SIZE, IMAGE_SIZE) * noise
    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    columns = np.random.randint(0, IMAGE_SIZE, size=n)
    rows = np.random.randint(0, IMAGE_SIZE, size=n)
    for k in range(n):
        images[k, 0, :, columns[k]] += 1.0        # vertical bar
        images[n + k, 0, rows[k], :] += 1.0       # horizontal bar
    return images, labels


def train(epochs=150, lr=0.01, seed=0, verbose=True):
    """Train the CNN and return a dictionary of training stats."""
    np.random.seed(seed)  # fixes the dataset and the weight init

    x_np, y_np = make_bars()
    x = Tensor(x_np)

    # One conv layer learns bar-shaped kernels; Flatten bridges the
    # image-shaped activations into the final Linear classifier.
    model = Sequential(
        Conv2d(1, 4, kernel_size=3),   # (N, 1, 6, 6) -> (N, 4, 4, 4)
        ReLU(),
        Flatten(),                     # -> (N, 64)
        Linear(4 * 4 * 4, 2),          # -> (N, 2) class logits
    )
    optimizer = Adam(model.parameters(), lr=lr)

    losses = []
    accuracies = []
    for epoch in range(epochs):
        logits = model(x)
        loss = cross_entropy(logits, y_np)
        losses.append(float(loss.data))
        accuracies.append(accuracy(logits, y_np))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and epoch % 25 == 0:
            print(f"epoch {epoch:03d} | loss {losses[-1]:.4f} | accuracy {accuracies[-1]:.1%}")

    # Final measurement after the last update.
    logits = model(x)
    final_loss = float(cross_entropy(logits, y_np).data)
    final_accuracy = accuracy(logits, y_np)
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
