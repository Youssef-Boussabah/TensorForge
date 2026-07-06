"""Train a deeper MLP with Dropout on concentric circles.

The dataset is two rings — no straight line separates them, so this
takes a real MLP. The model uses Dropout, which makes the train/eval
distinction matter for the first time:

- model.train():  Dropout randomly zeroes activations. This is the
  mode for optimization steps — the noise is the regularization.
- model.eval():   Dropout is the identity. This is the mode for
  measuring anything, otherwise metrics jump around randomly.

All histories below are measured through evaluate_binary_classifier,
which switches to eval mode for the measurement and restores the
previous mode afterwards.

Run it with:

    uv run python examples/train_mlp_with_dropout.py
"""

import numpy as np

from tensorforge import Tensor, evaluate_binary_classifier, train_test_split
from tensorforge.nn import Dropout, Linear, ReLU, Sequential, binary_cross_entropy
from tensorforge.optim import Adam, StepLR, clip_grad_norm


def make_circles(points_per_class=100, noise=0.1):
    """Two rings: class 0 is an inner circle, class 1 an outer one."""
    n = points_per_class
    angles = np.random.uniform(0.0, 2.0 * np.pi, 2 * n)
    radii = np.concatenate([
        0.5 + np.random.randn(n) * noise,   # inner ring
        1.5 + np.random.randn(n) * noise,   # outer ring
    ])
    x = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return x, y


def train(
    epochs=400,
    lr=0.03,
    hidden_size=16,
    dropout_p=0.25,
    seed=0,
    validation_split=0.25,
    verbose=True,
    max_grad_norm=None,
    scheduler_step_size=None,
    scheduler_gamma=0.5,
):
    """Train the classifier and return a dictionary of training stats."""
    np.random.seed(seed)  # fixes the dataset and the weight init

    x_np, y_np = make_circles()
    x_np, x_val, y_np, y_val = train_test_split(
        x_np, y_np, test_size=validation_split, shuffle=True, seed=seed
    )
    x = Tensor(x_np)

    model = Sequential(
        Linear(2, hidden_size),
        ReLU(),
        Dropout(dropout_p, seed=seed),
        Linear(hidden_size, hidden_size),
        ReLU(),
        Dropout(dropout_p, seed=seed + 1),
        Linear(hidden_size, 1),  # one raw logit per row
    )
    optimizer = Adam(model.parameters(), lr=lr)
    # Optional: decay the learning rate every scheduler_step_size epochs.
    scheduler = None
    if scheduler_step_size is not None:
        scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    gradient_norms = []
    losses = []
    accuracies = []
    validation_losses = []
    validation_accuracies = []
    for epoch in range(epochs):
        # Measure first, with Dropout OFF (the evaluator handles the
        # mode switch), so the histories are stable numbers.
        train_stats = evaluate_binary_classifier(model, x_np, y_np)
        val_stats = evaluate_binary_classifier(model, x_val, y_val)
        losses.append(train_stats["loss"])
        accuracies.append(train_stats["accuracy"])
        validation_losses.append(val_stats["loss"])
        validation_accuracies.append(val_stats["accuracy"])

        # Then take the optimization step with Dropout ON.
        model.train()
        loss = binary_cross_entropy(model(x), y_np)
        optimizer.zero_grad()
        loss.backward()
        # Optional: cap the gradient norm before stepping, so one bad
        # batch can't blow up the weights.
        if max_grad_norm is not None:
            gradient_norms.append(clip_grad_norm(model.parameters(), max_grad_norm))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if verbose and epoch % 100 == 0:
            print(
                f"epoch {epoch:03d} | loss {train_stats['loss']:.4f}"
                f" | accuracy {train_stats['accuracy']:.1%}"
                f" | val loss {val_stats['loss']:.4f}"
                f" | val accuracy {val_stats['accuracy']:.1%}"
            )

    # Final measurement after the last update.
    train_stats = evaluate_binary_classifier(model, x_np, y_np)
    val_stats = evaluate_binary_classifier(model, x_val, y_val)
    losses.append(train_stats["loss"])
    accuracies.append(train_stats["accuracy"])
    validation_losses.append(val_stats["loss"])
    validation_accuracies.append(val_stats["accuracy"])

    # Leave the model in eval mode: anyone using it now gets
    # deterministic predictions with Dropout inactive.
    model.eval()

    if verbose:
        print()
        print(f"final loss = {losses[-1]:.4f}")
        print(f"final accuracy = {accuracies[-1]:.1%}")
        print(f"final validation loss = {validation_losses[-1]:.4f}")
        print(f"final validation accuracy = {validation_accuracies[-1]:.1%}")

    stats = {
        "model": model,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "final_accuracy": accuracies[-1],
        "losses": losses,
        "accuracies": accuracies,
        "final_validation_loss": validation_losses[-1],
        "final_validation_accuracy": validation_accuracies[-1],
        "validation_losses": validation_losses,
        "validation_accuracies": validation_accuracies,
        "final_lr": float(optimizer.lr),
    }
    if max_grad_norm is not None:
        stats["gradient_norms"] = gradient_norms
    return stats


def main():
    train()


if __name__ == "__main__":
    main()
