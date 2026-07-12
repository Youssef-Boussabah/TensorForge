"""Native checkpointing and deterministic file resume (Advanced C++
v3.14).

Trains the small native MLP with NativeAdam for a few steps, saves one
pickle-free native checkpoint archive (model + optimizer state +
metadata), restores it into a completely fresh model/optimizer pair,
and continues **both** runs on identical data — proving the resumed
run reproduces the uninterrupted run bit for bit. The checkpoint lives
in a temporary directory that is removed automatically; nothing is
left behind.

Run it:

    uv run python examples/native_checkpoint_resume.py
"""

import os
import tempfile

from tensorforge.experimental import (
    NativeAdam,
    NativeLinear,
    NativeMSELoss,
    NativeReLU,
    NativeSequential,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)

# The same fixed synthetic regression data the MLP training proof uses:
# 8 samples, 2 features -> 1 target, handed once to from_array (the
# explicit native entry boundary).
X_VALUES = [
    [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],
    [0.5, -1.0], [-1.0, 0.5], [-0.5, -0.5], [1.5, 0.5],
]
Y_VALUES = [[0.0], [1.0], [1.0], [2.0], [-0.5], [0.0], [-1.0], [2.5]]

STEPS_BEFORE_SAVE = 6
STEPS_AFTER_RESUME = 6
LR = 0.05


def _build():
    """A deterministic fresh model/optimizer pair (fixed seeds)."""
    model = NativeSequential(
        NativeLinear(2, 8, seed=0),
        NativeReLU(),
        NativeLinear(8, 1, seed=1),
    )
    return model, NativeAdam(model.parameters(), lr=LR)


def _train(model, optimizer, loss_fn, x, y, steps):
    """``steps`` full iterations; returns the loss history as floats.
    Per-iteration tensors are closed and gradients cleared each step."""
    losses = []
    for _ in range(steps):
        prediction = model(x)
        loss = loss_fn(prediction, y)
        losses.append(float(loss.to_numpy()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss.close()
        prediction.close()
    return losses


def train(steps_before=STEPS_BEFORE_SAVE, steps_after=STEPS_AFTER_RESUME):
    """Train, checkpoint, resume, and compare. Returns plain Python
    stats only — every native tensor and optimizer is closed before
    returning, and the temporary checkpoint directory is removed."""
    loss_fn = NativeMSELoss()
    x = NativeTensor.from_array(X_VALUES)
    y = NativeTensor.from_array(Y_VALUES)
    model_a, optimizer_a = _build()
    model_b = optimizer_b = None
    try:
        before = _train(model_a, optimizer_a, loss_fn, x, y, steps_before)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "native_mlp.checkpoint.npz")
            save_native_checkpoint(
                path, model_a, optimizer=optimizer_a,
                metadata={"steps_completed": steps_before, "lr": LR},
            )
            model_b, optimizer_b = _build()
            metadata = load_native_checkpoint(
                path, model_b, optimizer=optimizer_b
            )

        # Continue the uninterrupted run and the resumed run on the
        # same data: they must match bit for bit.
        continued = _train(model_a, optimizer_a, loss_fn, x, y, steps_after)
        resumed = _train(model_b, optimizer_b, loss_fn, x, y, steps_after)
        losses_match = continued == resumed
        parameters_match = all(
            (a.to_numpy() == b.to_numpy()).all()
            for a, b in zip(model_a.parameters(), model_b.parameters())
        )
        return {
            "steps_before": steps_before,
            "steps_after": steps_after,
            "metadata": metadata,
            "losses_before_save": before,
            "continued_losses": continued,
            "resumed_losses": resumed,
            "losses_match": losses_match,
            "parameters_match": parameters_match,
            "final_loss": resumed[-1],
        }
    finally:
        optimizer_a.close()
        if optimizer_b is not None:
            optimizer_b.close()
        for model in (model_a, model_b):
            if model is not None:
                for parameter in model.parameters():
                    parameter.close()
        x.close()
        y.close()


def main():
    stats = train()
    print(
        f"trained {stats['steps_before']} steps, saved a checkpoint, "
        f"resumed into a fresh model/optimizer"
    )
    print(f"checkpoint metadata: {stats['metadata']}")
    print(
        f"continued {stats['steps_after']} more steps on both runs: "
        f"losses match = {stats['losses_match']}, "
        f"parameters match = {stats['parameters_match']}"
    )
    print(f"final loss: {stats['final_loss']:.6f}")
    if not (stats["losses_match"] and stats["parameters_match"]):
        raise SystemExit("resumed run diverged from the uninterrupted run")
    print("native checkpoint resume ok")


if __name__ == "__main__":
    main()
