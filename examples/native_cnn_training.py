"""The native CNN training and checkpoint-resume proof (Advanced C++
Phase D, milestone D11).

A small convolutional network — ``NativeSequential(NativeConv2d(1, 2, 2,
seed=0), NativeReLU(), NativeMaxPool2d(2), NativeFlatten(),
NativeLinear(8, 1, seed=1))`` — learns a genuinely **spatial** regression
target on eight fixed 6x6 single-channel images, trained for 40
deterministic ``NativeAdam(lr=0.05)`` steps entirely through the
experimental native stack: native forward, native ``NativeMSELoss``,
native ``backward()``, and graph-free native optimizer updates through the
v3.7 ``copy_value_`` contract. The whole Phase-D layer set participates —
convolution, activation, max-pooling (with its private saved winners),
flatten, and the linear head.

**The task.** Each image carries bright blocks on a dim background. The
target is the strength of the strongest *bright-to-dark vertical edge*::

    target = 0.25 * max over every 2x2 window of
             (top-left + bottom-left - top-right - bottom-right),
             floored at 0

That rule is exactly a 2x2 convolution, a ReLU, and a maximum over
positions, so it needs the convolutional path: no linear map of the raw
pixels can reproduce the maximum, and the flat image (sample 3) and the
purely dark-to-bright image (sample 6, whose response the floor cancels
before the max picks a different window) keep the target from collapsing
into "sum of pixels". Images and targets are **fixed Python literals**
handed once to ``NativeTensor.from_array`` — nothing is generated,
augmented, shuffled, or loaded, and no random number is drawn during
training.

**The proof.** ``train()`` runs the uninterrupted schedule and reports the
deterministic loss curve (about 0.7713 -> 0.0111, a ~98.6% reduction),
every parameter's gradient/version evidence, and the final predictions.
``checkpoint_resume_proof()`` then runs the same schedule twice: once
uninterrupted, and once interrupted at step 15, saved to one pickle-free
native checkpoint (model **and** optimizer state), reloaded into a
completely fresh model/optimizer pair, and continued to step 40. The two
runs must agree **exactly** — loss history, final predictions, every
parameter value, and every optimizer state entry — which they do, because
the native CPU float64 kernels are deterministic (fixed loop orders, no
parallel reduction, no fast-math) and nothing random happens between the
checkpoint and the resume.

Lifetime is explicit: each step builds a completely fresh graph, closes
its prediction and loss tensors after the one-shot ``backward()`` has
released that graph (and with it max-pooling's private winner buffer), and
clears gradients before the next step. The checkpoint lives in a temporary
directory that is removed automatically; nothing is left behind.

This is an integration proof for the native CNN stack — not a benchmark,
and no performance is claimed. It needs the experimental C++ backend to be
built — run:

    uv run python examples/native_cnn_training.py

``train()`` and ``checkpoint_resume_proof()`` return plain Python
values only (never live native tensors) so the tests can import and verify
them; ``main()`` prints them.
"""

import os
import tempfile

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeConv2d,
    NativeFlatten,
    NativeLinear,
    NativeMaxPool2d,
    NativeMSELoss,
    NativeReLU,
    NativeSequential,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)

# Eight fixed 6x6 single-channel images (NCHW: 8 x 1 x 6 x 6). Every value
# is a half or a quarter, so the literals are exactly representable in
# float64 and the arithmetic below is reproducible everywhere.
IMAGE_VALUES = [
    # 0 — one wide bright block, upper left: a strong edge.
    [[[2.0, 2.0, 0.0, 0.0, 0.5, 0.5],
      [2.0, 2.0, 0.0, 0.0, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
    # 1 — a medium block in the middle of a dim frame.
    [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 1.5, 1.5, 0.0, 0.0, 0.5],
      [0.5, 1.5, 1.5, 0.0, 0.0, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
    # 2 — a weak narrow edge low and to the right.
    [[[0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 1.00, 0.25, 0.25],
      [0.25, 0.25, 0.25, 1.00, 0.25, 0.25]]],
    # 3 — perfectly flat: no edge anywhere, so the target is exactly 0.
    [[[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]],
    # 4 — one very bright column: the strongest edge in the set.
    [[[0.0, 0.0, 0.0, 2.5, 0.0, 0.0],
      [0.0, 0.0, 0.0, 2.5, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
    # 5 — two blocks; the stronger one decides the target (a maximum,
    # not a sum — this is what makes the task non-linear).
    [[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 2.0, 2.0, 0.0],
      [0.0, 0.0, 0.0, 2.0, 2.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
    # 6 — a block whose left side is a dark-to-bright edge; only its
    # right side counts under the rule.
    [[[0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
      [0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
    # 7 — a smooth ramp: mild edges everywhere, small target.
    [[[0.0, 0.5, 1.0, 1.5, 1.0, 0.5],
      [0.0, 0.5, 1.0, 1.5, 1.0, 0.5],
      [0.0, 0.5, 1.0, 1.5, 1.0, 0.5],
      [0.0, 0.5, 1.0, 1.5, 1.0, 0.5],
      [0.0, 0.5, 1.0, 1.5, 1.0, 0.5],
      [0.0, 0.5, 1.0, 1.5, 1.0, 0.5]]],
]

# The rule applied to each image above, frozen as literals (see the
# module docstring; ``strongest_edge`` recomputes them for the tests).
TARGET_VALUES = [
    [1.0],    # 0: (2.0 + 2.0 - 0.0 - 0.0) * 0.25
    [0.75],   # 1: (1.5 + 1.5 - 0.0 - 0.0) * 0.25
    [0.375],  # 2: (1.00 + 1.00 - 0.25 - 0.25) * 0.25
    [0.0],    # 3: flat image, no edge
    [1.25],   # 4: (2.5 + 2.5 - 0.0 - 0.0) * 0.25
    [1.0],    # 5: (2.0 + 2.0 - 0.0 - 0.0) * 0.25
    [1.0],    # 6: (2.0 + 2.0 - 0.0 - 0.0) * 0.25, from the block's right edge
    [0.25],   # 7: (1.5 + 1.5 - 1.0 - 1.0) * 0.25
]

IN_CHANNELS = 1
CONV_CHANNELS = 2
KERNEL_SIZE = 2
POOL_SIZE = 2
IMAGE_SIZE = 6
# 6x6 -image- conv(2) -> 5x5 -pool(2, floor)-> 2x2 per channel.
FLAT_FEATURES = CONV_CHANNELS * 2 * 2
OUT_FEATURES = 1

CONV_SEED = 0
LINEAR_SEED = 1
TOTAL_STEPS = 40
SPLIT_STEP = 15
DEFAULT_LR = 0.05


def strongest_edge(image):
    """The task's target rule for one ``(1, 6, 6)`` image, in plain
    Python: ``0.25 * max(0, strongest 2x2 bright-to-dark vertical edge)``.
    Used to document and re-derive ``TARGET_VALUES``; training never
    calls it."""
    plane = image[0]
    best = 0.0
    for row in range(len(plane) - 1):
        for column in range(len(plane[0]) - 1):
            response = (
                plane[row][column] + plane[row + 1][column]
                - plane[row][column + 1] - plane[row + 1][column + 1]
            )
            best = max(best, response)
    return 0.25 * best


def build_model():
    """The deterministic native CNN: Conv -> ReLU -> MaxPool -> Flatten
    -> Linear, with fixed seeds on both trainable layers."""
    return NativeSequential(
        NativeConv2d(IN_CHANNELS, CONV_CHANNELS, KERNEL_SIZE, seed=CONV_SEED),
        NativeReLU(),
        NativeMaxPool2d(POOL_SIZE),
        NativeFlatten(),
        NativeLinear(FLAT_FEATURES, OUT_FEATURES, seed=LINEAR_SEED),
    )


def build_optimizer(model, lr=DEFAULT_LR):
    """The canonical optimizer — NativeAdam, whose persistent moment and
    step-count state makes the resume proof meaningful."""
    return NativeAdam(model.parameters(), lr=lr)


def _training_step(model, loss_fn, optimizer, x, y):
    """One full iteration: fresh graph -> forward -> scalar loss ->
    record -> backward -> step -> zero_grad, closing this step's tensors.

    The one-shot ``backward()`` releases the operation graph — and with it
    max-pooling's private saved-winner buffer — so nothing accumulates
    between steps and no stale graph can be reused after ``step()``."""
    prediction = model(x)
    loss = loss_fn(prediction, y)
    try:
        value = float(loss.to_numpy())  # inspection exit, before release
        loss.backward()
        optimizer.step()
    finally:
        loss.close()
        prediction.close()
    optimizer.zero_grad()
    return value


def _evaluate(model, loss_fn, x, y):
    """A no-update pass: returns ``(loss, predictions)`` as Python values
    and closes both native tensors. The graph it builds is released by
    the caller's ``zero_grad`` discipline (no backward runs here)."""
    prediction = model(x)
    loss = loss_fn(prediction, y)
    try:
        return float(loss.to_numpy()), prediction.to_numpy().ravel().tolist()
    finally:
        loss.close()
        prediction.close()


def _close_run(model, optimizer):
    """Release everything a run owns, in the established order."""
    if optimizer is not None:
        optimizer.close()
    if model is not None:
        for parameter in model.parameters():
            parameter.close()


def train(steps=TOTAL_STEPS, lr=DEFAULT_LR):
    """Train the deterministic native CNN for ``steps`` NativeAdam
    updates and return the run's evidence as plain Python values.

    The returned dict contains the configuration (``steps``, ``lr``), the
    per-step ``loss_history`` (each entry the loss *before* that step's
    update), ``initial_loss`` / ``final_loss`` (the latter a fresh
    evaluation pass after the last update), ``initial_predictions`` /
    ``final_predictions`` / ``targets``, the canonical
    ``parameter_names``, per-parameter ``initial_versions`` /
    ``final_versions``, ``initial_parameters`` / ``final_parameters``
    (name -> nested lists), ``gradient_shapes`` and
    ``gradient_nonzero`` recorded after the first backward,
    ``identity_stable``, and ``state_keys``.

    Everything the run creates — the fixed data tensors, the model's
    parameters, and the optimizer's state — is closed before returning,
    success or failure."""
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError(f"steps must be an int, got {type(steps).__name__}")
    if steps <= 0:
        raise ValueError(f"steps must be strictly positive, got {steps}")

    model = build_model()
    loss_fn = NativeMSELoss()
    optimizer = build_optimizer(model, lr=lr)  # validates lr

    named = list(model.named_parameters())
    names = [name for name, _ in named]
    parameters = [parameter for _, parameter in named]
    identities = [id(parameter) for parameter in parameters]

    x = NativeTensor.from_array(IMAGE_VALUES)
    y = NativeTensor.from_array(TARGET_VALUES)
    try:
        initial_loss, initial_predictions = _evaluate(model, loss_fn, x, y)
        results = {
            "steps": steps,
            "lr": optimizer.lr,
            "parameter_names": names,
            "state_keys": sorted(model.state_dict()),
            "initial_versions": [p.version for p in parameters],
            "initial_parameters": {
                name: parameter.to_numpy().tolist()
                for name, parameter in named
            },
            "initial_predictions": initial_predictions,
            "targets": [row[0] for row in TARGET_VALUES],
        }

        loss_history = []
        identity_stable = True
        gradient_shapes = None
        gradient_nonzero = None
        for step in range(steps):
            prediction = model(x)
            loss = loss_fn(prediction, y)
            try:
                loss_history.append(float(loss.to_numpy()))
                loss.backward()
                if step == 0:
                    # Evidence that every trainable layer is reached, taken
                    # from the very first backward (not inferred from the
                    # final parameter difference).
                    gradient_shapes = {
                        name: parameter.grad.shape
                        for name, parameter in named
                    }
                    gradient_nonzero = {
                        name: bool((parameter.grad.to_numpy() != 0.0).any())
                        for name, parameter in named
                    }
                optimizer.step()
            finally:
                loss.close()
                prediction.close()
            optimizer.zero_grad()
            identity_stable &= (
                [id(p) for _, p in model.named_parameters()] == identities
            )

        final_loss, final_predictions = _evaluate(model, loss_fn, x, y)
        results.update(
            loss_history=loss_history,
            initial_loss=initial_loss,
            final_loss=final_loss,
            final_predictions=final_predictions,
            final_versions=[p.version for p in parameters],
            final_parameters={
                name: parameter.to_numpy().tolist()
                for name, parameter in named
            },
            gradient_shapes=gradient_shapes,
            gradient_nonzero=gradient_nonzero,
            identity_stable=identity_stable,
            gradients_cleared=all(p.grad is None for p in parameters),
        )
        return results
    finally:
        x.close()
        y.close()
        _close_run(model, optimizer)


def _optimizer_state_values(optimizer):
    """The optimizer's state as plain Python values, so two runs can be
    compared without holding native tensors."""
    state = optimizer.state_dict()
    return {
        "format_version": state["format_version"],
        "optimizer": state["optimizer"],
        "lr": state["lr"],
        "betas": list(state["betas"]),
        "eps": state["eps"],
        "step_counts": list(state["step_counts"]),
        "m": [tensor.to_numpy().tolist() for tensor in state["m"]],
        "v": [tensor.to_numpy().tolist() for tensor in state["v"]],
    }


def checkpoint_resume_proof(total_steps=TOTAL_STEPS, split_step=SPLIT_STEP,
                            lr=DEFAULT_LR):
    """Run the same schedule twice — once uninterrupted, once interrupted
    at ``split_step``, checkpointed, reloaded into a **fresh** model and
    optimizer, and continued — then compare them exactly.

    Returns plain Python values: both loss histories, the prefix/suffix
    comparisons, the final predictions, whether every parameter value and
    every optimizer state entry matches, the checkpoint metadata, the
    model state keys stored in the archive, and the fresh model's
    parameter-identity stability across the load."""
    loss_fn = NativeMSELoss()
    x = NativeTensor.from_array(IMAGE_VALUES)
    y = NativeTensor.from_array(TARGET_VALUES)

    model_a = optimizer_a = None
    model_b = optimizer_b = model_c = optimizer_c = None
    try:
        # -- Path A: uninterrupted -------------------------------------
        model_a = build_model()
        optimizer_a = build_optimizer(model_a, lr=lr)
        losses_a = [
            _training_step(model_a, loss_fn, optimizer_a, x, y)
            for _ in range(total_steps)
        ]
        final_loss_a, predictions_a = _evaluate(model_a, loss_fn, x, y)
        parameters_a = [p.to_numpy().tolist() for p in model_a.parameters()]
        state_a = _optimizer_state_values(optimizer_a)
        names_a = [name for name, _ in model_a.named_parameters()]

        # -- Path B: train, checkpoint, resume into a fresh pair --------
        model_b = build_model()
        optimizer_b = build_optimizer(model_b, lr=lr)
        losses_b_prefix = [
            _training_step(model_b, loss_fn, optimizer_b, x, y)
            for _ in range(split_step)
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "native_cnn.checkpoint.npz")
            save_native_checkpoint(
                path, model_b, optimizer=optimizer_b,
                metadata={"steps_completed": split_step, "lr": lr},
            )
            model_c = build_model()
            optimizer_c = build_optimizer(model_c, lr=lr)
            identities_before = [id(p) for p in model_c.parameters()]
            metadata = load_native_checkpoint(
                path, model_c, optimizer=optimizer_c
            )
            identities_after = [id(p) for p in model_c.parameters()]
            archive_keys = sorted(model_c.state_dict())

        losses_b_suffix = [
            _training_step(model_c, loss_fn, optimizer_c, x, y)
            for _ in range(total_steps - split_step)
        ]
        final_loss_c, predictions_c = _evaluate(model_c, loss_fn, x, y)
        parameters_c = [p.to_numpy().tolist() for p in model_c.parameters()]
        state_c = _optimizer_state_values(optimizer_c)
        names_c = [name for name, _ in model_c.named_parameters()]

        losses_b = losses_b_prefix + losses_b_suffix
        return {
            "total_steps": total_steps,
            "split_step": split_step,
            "lr": lr,
            "metadata": metadata,
            "uninterrupted_losses": losses_a,
            "resumed_losses": losses_b,
            "prefix_matches": losses_a[:split_step] == losses_b_prefix,
            "suffix_matches": losses_a[split_step:] == losses_b_suffix,
            "losses_match": losses_a == losses_b,
            "final_loss_uninterrupted": final_loss_a,
            "final_loss_resumed": final_loss_c,
            "final_losses_match": final_loss_a == final_loss_c,
            "predictions_match": predictions_a == predictions_c,
            "parameters_match": parameters_a == parameters_c,
            "optimizer_state_matches": state_a == state_c,
            "parameter_order_matches": names_a == names_c,
            "identities_preserved": identities_before == identities_after,
            "state_keys": archive_keys,
            "final_predictions": predictions_c,
        }
    finally:
        x.close()
        y.close()
        for model, optimizer in (
            (model_a, optimizer_a), (model_b, optimizer_b),
            (model_c, optimizer_c),
        ):
            if model is not None or optimizer is not None:
                _close_run(model, optimizer)


def main():
    if not cpp.is_available():
        print("The experimental C++ backend is not built.")
        print(cpp.build_instructions())
        return

    run = train()
    reduction = 100.0 * (1.0 - run["final_loss"] / run["initial_loss"])
    print(
        f"native CNN: Conv2d(1 -> {CONV_CHANNELS}, {KERNEL_SIZE}x"
        f"{KERNEL_SIZE}) -> ReLU -> MaxPool2d({POOL_SIZE}) -> Flatten -> "
        f"Linear({FLAT_FEATURES} -> {OUT_FEATURES})"
    )
    print(
        f"trained {run['steps']} NativeAdam steps (lr={run['lr']}) on "
        f"{len(IMAGE_VALUES)} fixed {IMAGE_SIZE}x{IMAGE_SIZE} images"
    )
    print(
        f"loss: {run['initial_loss']:.6f} -> {run['final_loss']:.6f} "
        f"({reduction:.1f}% reduction)"
    )
    print("prediction vs target:")
    for index, (predicted, target) in enumerate(
        zip(run["final_predictions"], run["targets"])
    ):
        print(f"  image {index}: {predicted:+.4f}   target {target:+.4f}")
    print(
        f"trainable parameters: {run['parameter_names']} "
        f"(pooling and flatten contribute none)"
    )
    print(f"first-step gradients nonzero: {run['gradient_nonzero']}")
    print(f"parameter identities stable: {run['identity_stable']}")

    proof = checkpoint_resume_proof()
    print()
    print(
        f"checkpoint resume: trained {proof['split_step']} steps, saved model "
        f"+ optimizer state, reloaded into a fresh pair, continued to "
        f"{proof['total_steps']}"
    )
    print(f"  checkpoint metadata: {proof['metadata']}")
    print(f"  archived model state keys: {proof['state_keys']}")
    print(f"  prefix losses match:      {proof['prefix_matches']}")
    print(f"  resumed losses match:     {proof['suffix_matches']}")
    print(f"  final loss match:         {proof['final_losses_match']}")
    print(f"  final predictions match:  {proof['predictions_match']}")
    print(f"  final parameters match:   {proof['parameters_match']}")
    print(f"  optimizer state matches:  {proof['optimizer_state_matches']}")
    print(f"  fresh identities stable:  {proof['identities_preserved']}")

    if not (
        proof["losses_match"]
        and proof["final_losses_match"]
        and proof["predictions_match"]
        and proof["parameters_match"]
        and proof["optimizer_state_matches"]
        and proof["identities_preserved"]
    ):
        raise SystemExit("resumed run diverged from the uninterrupted run")
    print("native CNN training + checkpoint resume ok")


if __name__ == "__main__":
    main()
