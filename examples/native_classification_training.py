"""The native classification training and checkpoint-resume proof
(Advanced C++ Phase E, milestone E8).

A compact native CNN classifier —
``NativeConv2d(1, 4, 3, seed=0) -> NativeReLU -> NativeMaxPool2d(2) ->
NativeFlatten -> NativeLinear(16, 3, seed=1)`` — learns a **three-class**
shape task on twelve fixed 6x6 single-channel images, trained for 40
deterministic ``NativeAdam(lr=0.05)`` steps entirely through the
experimental native stack: native forward, the native
``NativeCrossEntropyLoss`` over **raw logits**, native ``backward()``,
and graph-free native optimizer updates through the v3.7 ``copy_value_``
contract. Every Phase-D layer and the whole Phase-E classification path
participate; **E8 adds no numerical operation and no runtime
capability** — no kernel, no C ABI export, no new module, loss, metric,
or optimizer.

**The task.** Each image carries one bright shape on a dark background:
a **vertical bar** (class 0), a **horizontal bar** (class 1), or a
**diagonal line** (class 2), each at four different positions. Position
varies within every class, so no fixed pixel template separates the
classes — a small convolution kernel slid across the image is what finds
the shape wherever it sits. Images and labels are **fixed literals**
(``IMAGE_VALUES`` / ``TARGET_VALUES``); nothing is generated, loaded,
downloaded, augmented, shuffled, or split, and no random number is drawn
during training. Labels stay **host integers** — the native runtime has
no integer dtype, and the classification stack's strict ``int64`` target
contract is exactly why.

**The path.** The model's last layer is the linear head, and its output
goes to the loss unchanged: there is deliberately **no softmax or
log-softmax module** in the model, because ``NativeCrossEntropyLoss``
consumes raw logits through the fused, numerically stable E5/E6 kernel.
``native_accuracy`` is **reporting only** — it copies logits to the host
through the explicit public ``to_numpy()`` boundary — so it is called
before training, after training, and never inside the training
mathematics.

**The proof.** ``run_training()`` runs the uninterrupted schedule and
reports the deterministic loss curve (1.159638 -> 0.000101, a 99.99%
reduction) and accuracy (0.3333 -> 1.0000 on the fixed task).
``run_resume_proof()`` then runs the same schedule twice: once
uninterrupted, and once interrupted at step 15, saved to one pickle-free
native checkpoint (model **and** optimizer state, format **version 1**,
unchanged), reloaded into a completely fresh model/optimizer pair, and
continued to step 40. The two runs must agree **exactly** — the whole
remaining loss suffix, every parameter value, every optimizer state
entry, the final logits, the predictions, and the accuracy — which they
do, because the native CPU float64 kernels are deterministic (fixed loop
orders, no parallel reduction, no fast-math) and nothing random happens
between the checkpoint and the resume.

Lifetime is explicit: each step builds a completely fresh graph and
closes its logits and loss after the one-shot ``backward()`` has released
that graph — and with it max-pooling's private winner buffer and
cross-entropy's private **saved probabilities** — so nothing accumulates
between steps. The checkpoint lives in a temporary directory that is
removed automatically; nothing is left behind.

This is an integration proof for the native classification stack on one
fixed task — not a benchmark, not a generalization claim, and no
performance is claimed. Honest benchmark characterization is E9's job.
It needs the experimental C++ backend to be built — run:

    uv run python examples/native_classification_training.py

Every public helper returns plain Python values only (never live native
tensors) so the tests can import and verify them; ``main()`` prints them.
"""

import os
import tempfile

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeConv2d,
    NativeCrossEntropyLoss,
    NativeFlatten,
    NativeLinear,
    NativeMaxPool2d,
    NativeModule,
    NativeReLU,
    NativeTensor,
    load_native_checkpoint,
    native_accuracy,
    save_native_checkpoint,
)

# Twelve fixed 6x6 single-channel images (NCHW: 12 x 1 x 6 x 6), four per
# class. Every value is exactly 0.0 or 1.0, so the literals are exactly
# representable in float64 and the arithmetic below is reproducible
# everywhere. The shape's *position* varies inside each class — that is
# what makes the task convolutional rather than a pixel lookup.
IMAGE_VALUES = [
    # --- class 0: a vertical bar, at columns 1, 2, 3, 4 ------------------
    [[[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]],
    [[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]],
    [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]],
    [[[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]],
    # --- class 1: a horizontal bar, at rows 1, 2, 3, 4 -------------------
    [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
    [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
    [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
    [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
    # --- class 2: a diagonal line, at offsets -1, 0, +1, +2 --------------
    [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]],
    [[[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
    [[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
]

# The class label of each image above: **host integers**, never a native
# tensor (the native runtime has no integer dtype).
TARGET_VALUES = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

CLASS_NAMES = ("vertical", "horizontal", "diagonal")

IN_CHANNELS = 1
CONV_CHANNELS = 4
KERNEL_SIZE = 3
POOL_SIZE = 2
IMAGE_SIZE = 6
NUM_CLASSES = 3
# 6x6 -image- conv(3) -> 4x4 -pool(2, floor)-> 2x2 per channel.
FLAT_FEATURES = CONV_CHANNELS * 2 * 2

CONV_SEED = 0
LINEAR_SEED = 1
TOTAL_STEPS = 40
SPLIT_STEP = 15
DEFAULT_LR = 0.05


def build_dataset():
    """The fixed task as plain host data: ``(images, targets)``, where
    ``images`` is the ``12 x 1 x 6 x 6`` nested list of literals and
    ``targets`` is the list of integer class labels.

    Deterministic by construction — these are copies of the module-level
    literals, so repeated calls are equal and a caller mutating one
    result cannot perturb the next."""
    images = [[[list(row) for row in plane] for plane in image]
              for image in IMAGE_VALUES]
    return images, list(TARGET_VALUES)


class NativeImageClassifier(NativeModule):
    """The deterministic native CNN classifier:
    ``Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear``, producing **raw
    logits** of shape ``(batch_size, num_classes)``.

    Every child is registered through the normal ``NativeModule``
    attribute-assignment path, so parameters, ``state_dict()`` keys, and
    checkpoint keys all come from the module system rather than from a
    hand-maintained list. There is deliberately **no softmax or
    log-softmax layer**: the cross-entropy loss is fused and stable and
    takes the logits directly."""

    def __init__(self, conv_seed=CONV_SEED, linear_seed=LINEAR_SEED):
        super().__init__()
        self.conv = NativeConv2d(IN_CHANNELS, CONV_CHANNELS, KERNEL_SIZE,
                                 seed=conv_seed)
        self.relu = NativeReLU()
        self.pool = NativeMaxPool2d(POOL_SIZE)
        self.flatten = NativeFlatten()
        self.linear = NativeLinear(FLAT_FEATURES, NUM_CLASSES,
                                   seed=linear_seed)

    def forward(self, images):
        """``(N, 1, 6, 6)`` images to ``(N, 3)`` logits. The
        intermediates are dropped as locals — the autograd graph holds
        what backward needs, and releases it all at once."""
        hidden = self.conv(images)
        hidden = self.relu(hidden)
        hidden = self.pool(hidden)
        hidden = self.flatten(hidden)
        return self.linear(hidden)


def build_model():
    """A freshly initialized classifier. Deterministic: both trainable
    layers draw their fan-in uniform initialization from a *local*
    seeded generator, so two independently built models start
    numerically identical and the global NumPy RNG is never touched."""
    return NativeImageClassifier()


def build_loss():
    """The native classification loss, over raw logits."""
    return NativeCrossEntropyLoss()


def build_optimizer(model, lr=DEFAULT_LR):
    """The canonical optimizer — NativeAdam, whose persistent moment
    buffers and per-parameter step counters make the resume proof
    meaningful: restoring them is what makes the resumed trajectory
    match, and comparing them is what proves it."""
    return NativeAdam(model.parameters(), lr=lr)


def train_step(model, loss_fn, optimizer, images, targets):
    """One full iteration: fresh graph -> logits -> scalar loss ->
    record -> backward -> step -> zero_grad, closing this step's tensors.

    The one-shot ``backward()`` releases the operation graph — and with
    it max-pooling's private saved winners and cross-entropy's private
    saved probabilities — so nothing accumulates between steps and no
    stale graph can be reused after ``step()``. Returns the loss
    *before* this step's update, as a plain float.

    Deliberately native throughout: no metric, no argmax, and no host
    conversion of tensor data happens here (the single ``to_numpy()`` on
    the scalar loss is the established inspection exit, and the tests
    prove the step needs nothing else)."""
    logits = model(images)
    loss = loss_fn(logits, targets)
    try:
        value = float(loss.to_numpy())   # scalar inspection, before release
        loss.backward()
        optimizer.step()
    finally:
        loss.close()
        logits.close()
    optimizer.zero_grad()
    return value


def evaluate(model, loss_fn, images, targets):
    """A no-update reporting pass: ``(loss, accuracy, predictions,
    logits)`` as plain Python values, closing both native tensors.

    ``native_accuracy`` and the predicted classes are **reporting only**
    — both leave native memory through the explicit public
    ``to_numpy()`` boundary, which is exactly why neither belongs in
    ``train_step``. The predicted class is the first maximal logit,
    matching ``native_accuracy``'s ``numpy.argmax`` tie rule."""
    logits = model(images)
    loss = loss_fn(logits, targets)
    try:
        rows = logits.to_numpy().tolist()
        predictions = [max(range(len(row)), key=row.__getitem__)
                       for row in rows]
        return (float(loss.to_numpy()), native_accuracy(logits, targets),
                predictions, rows)
    finally:
        loss.close()
        logits.close()


def _close_run(model, optimizer):
    """Release everything a run owns, in the established order."""
    if optimizer is not None:
        optimizer.close()
    if model is not None:
        for parameter in model.parameters():
            parameter.close()


def _optimizer_state_values(optimizer):
    """The optimizer's state as plain Python values, so two runs can be
    compared exactly without holding native tensors."""
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


def run_training(steps=TOTAL_STEPS, lr=DEFAULT_LR):
    """Train the deterministic native classifier for ``steps`` NativeAdam
    updates and return the run's evidence as plain Python values.

    The returned dict contains the configuration (``steps``, ``lr``), the
    per-step ``loss_history`` (each entry the loss *before* that step's
    update), ``initial_loss`` / ``final_loss`` and ``initial_accuracy`` /
    ``final_accuracy`` (fresh reporting passes before the first and after
    the last update), ``initial_predictions`` / ``final_predictions`` /
    ``final_logits`` / ``targets``, the canonical ``parameter_names``,
    per-parameter ``initial_versions`` / ``final_versions``,
    ``initial_parameters`` / ``final_parameters`` (name -> nested lists),
    ``gradient_shapes`` and ``gradient_nonzero`` recorded after the first
    backward, ``optimizer_state``, ``identity_stable``,
    ``gradients_cleared``, and ``state_keys``.

    Everything the run creates — the fixed input tensor, the model's
    parameters, and the optimizer's state — is closed before returning,
    success or failure."""
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError(f"steps must be an int, got {type(steps).__name__}")
    if steps <= 0:
        raise ValueError(f"steps must be strictly positive, got {steps}")

    images, targets = build_dataset()
    model = build_model()
    loss_fn = build_loss()
    optimizer = build_optimizer(model, lr=lr)   # validates lr

    named = list(model.named_parameters())
    names = [name for name, _ in named]
    parameters = [parameter for _, parameter in named]
    identities = [id(parameter) for parameter in parameters]

    x = NativeTensor.from_array(images)
    try:
        initial_loss, initial_accuracy, initial_predictions, _ = evaluate(
            model, loss_fn, x, targets
        )
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
            "initial_loss": initial_loss,
            "initial_accuracy": initial_accuracy,
            "initial_predictions": initial_predictions,
            "targets": list(targets),
        }

        loss_history = []
        identity_stable = True
        gradient_shapes = None
        gradient_nonzero = None
        for step in range(steps):
            logits = model(x)
            loss = loss_fn(logits, targets)
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
                logits.close()
            optimizer.zero_grad()
            identity_stable &= (
                [id(p) for _, p in model.named_parameters()] == identities
            )

        final_loss, final_accuracy, final_predictions, final_logits = evaluate(
            model, loss_fn, x, targets
        )
        results.update(
            loss_history=loss_history,
            final_loss=final_loss,
            final_accuracy=final_accuracy,
            final_predictions=final_predictions,
            final_logits=final_logits,
            final_versions=[p.version for p in parameters],
            final_parameters={
                name: parameter.to_numpy().tolist()
                for name, parameter in named
            },
            gradient_shapes=gradient_shapes,
            gradient_nonzero=gradient_nonzero,
            optimizer_state=_optimizer_state_values(optimizer),
            identity_stable=identity_stable,
            gradients_cleared=all(p.grad is None for p in parameters),
        )
        return results
    finally:
        x.close()
        _close_run(model, optimizer)


def run_resume_proof(total_steps=TOTAL_STEPS, split_step=SPLIT_STEP,
                     lr=DEFAULT_LR, directory=None):
    """Run the same schedule twice — once uninterrupted, once interrupted
    at ``split_step``, checkpointed, reloaded into a **fresh** model and
    optimizer, and continued — then compare them exactly.

    ``directory`` is an optional existing directory for the checkpoint;
    by default a temporary one is created and removed, so the default run
    leaves no file behind.

    Returns plain Python values: both loss histories, the prefix/suffix
    comparisons, the final losses/accuracies/predictions/logits, whether
    every parameter value and every optimizer state entry matches, the
    checkpoint metadata, the model state keys stored in the archive, and
    the fresh model's parameter-identity stability across the load."""
    images, targets = build_dataset()
    loss_fn = build_loss()
    x = NativeTensor.from_array(images)

    model_a = optimizer_a = None
    model_b = optimizer_b = model_c = optimizer_c = None
    try:
        # -- Path A: uninterrupted -------------------------------------
        model_a = build_model()
        optimizer_a = build_optimizer(model_a, lr=lr)
        initial_parameters = [p.to_numpy().tolist() for p in model_a.parameters()]
        losses_a = [
            train_step(model_a, loss_fn, optimizer_a, x, targets)
            for _ in range(total_steps)
        ]
        loss_a, accuracy_a, predictions_a, logits_a = evaluate(
            model_a, loss_fn, x, targets
        )
        parameters_a = [p.to_numpy().tolist() for p in model_a.parameters()]
        state_a = _optimizer_state_values(optimizer_a)
        names_a = [name for name, _ in model_a.named_parameters()]

        # -- Path B: train, checkpoint, resume into a fresh pair --------
        model_b = build_model()
        optimizer_b = build_optimizer(model_b, lr=lr)
        start_parameters = [p.to_numpy().tolist() for p in model_b.parameters()]
        losses_b_prefix = [
            train_step(model_b, loss_fn, optimizer_b, x, targets)
            for _ in range(split_step)
        ]

        if directory is None:
            context = tempfile.TemporaryDirectory()
        else:
            context = _ExistingDirectory(directory)
        with context as checkpoint_directory:
            path = os.path.join(checkpoint_directory,
                                "native_classification.checkpoint.npz")
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
            train_step(model_c, loss_fn, optimizer_c, x, targets)
            for _ in range(total_steps - split_step)
        ]
        loss_c, accuracy_c, predictions_c, logits_c = evaluate(
            model_c, loss_fn, x, targets
        )
        parameters_c = [p.to_numpy().tolist() for p in model_c.parameters()]
        state_c = _optimizer_state_values(optimizer_c)
        names_c = [name for name, _ in model_c.named_parameters()]

        losses_b = losses_b_prefix + losses_b_suffix
        return {
            "total_steps": total_steps,
            "split_step": split_step,
            "lr": lr,
            "metadata": metadata,
            "identical_start": initial_parameters == start_parameters,
            "uninterrupted_losses": losses_a,
            "resumed_losses": losses_b,
            "resumed_suffix": losses_b_suffix,
            "prefix_matches": losses_a[:split_step] == losses_b_prefix,
            "suffix_matches": losses_a[split_step:] == losses_b_suffix,
            "first_resumed_loss_matches": (
                losses_b_suffix[0] == losses_a[split_step]
            ),
            "losses_match": losses_a == losses_b,
            "final_loss_uninterrupted": loss_a,
            "final_loss_resumed": loss_c,
            "final_losses_match": loss_a == loss_c,
            "final_accuracy_uninterrupted": accuracy_a,
            "final_accuracy_resumed": accuracy_c,
            "accuracies_match": accuracy_a == accuracy_c,
            "logits_match": logits_a == logits_c,
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
        for model, optimizer in (
            (model_a, optimizer_a), (model_b, optimizer_b),
            (model_c, optimizer_c),
        ):
            if model is not None or optimizer is not None:
                _close_run(model, optimizer)


class _ExistingDirectory:
    """A context manager mirroring ``TemporaryDirectory``'s interface for
    a caller-supplied directory — entered and left without creating or
    removing anything, so the optional explicit output path is handled by
    the same ``with`` block as the default temporary one."""

    def __init__(self, path):
        self._path = str(path)

    def __enter__(self):
        return self._path

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def main():
    if not cpp.is_available():
        print("The experimental C++ backend is not built.")
        print(cpp.build_instructions())
        return

    run = run_training()
    reduction = 100.0 * (1.0 - run["final_loss"] / run["initial_loss"])
    print(
        f"native classifier: Conv2d(1 -> {CONV_CHANNELS}, {KERNEL_SIZE}x"
        f"{KERNEL_SIZE}) -> ReLU -> MaxPool2d({POOL_SIZE}) -> Flatten -> "
        f"Linear({FLAT_FEATURES} -> {NUM_CLASSES})  [raw logits]"
    )
    print(
        f"trained {run['steps']} NativeAdam steps (lr={run['lr']}) on "
        f"{len(IMAGE_VALUES)} fixed {IMAGE_SIZE}x{IMAGE_SIZE} images in "
        f"{NUM_CLASSES} classes {CLASS_NAMES} with NativeCrossEntropyLoss"
    )
    print(f"initial loss: {run['initial_loss']:.6f}")
    print(f"initial accuracy: {run['initial_accuracy']:.4f}")
    print(f"final loss: {run['final_loss']:.6f}")
    print(f"final accuracy: {run['final_accuracy']:.4f}")
    print(f"loss reduction: {reduction:.2f}%")
    print(f"predicted classes: {run['final_predictions']}")
    print(f"true classes:      {run['targets']}")
    print(f"trainable parameters: {run['parameter_names']}")
    print(f"first-step gradients nonzero: {run['gradient_nonzero']}")

    proof = run_resume_proof()
    exact = (
        proof["identical_start"]
        and proof["prefix_matches"]
        and proof["suffix_matches"]
        and proof["losses_match"]
        and proof["final_losses_match"]
        and proof["logits_match"]
        and proof["predictions_match"]
        and proof["accuracies_match"]
        and proof["parameters_match"]
        and proof["optimizer_state_matches"]
        and proof["identities_preserved"]
    )
    print()
    print(
        f"checkpoint resume: trained {proof['split_step']} steps, saved model "
        f"+ optimizer state (format version 1), reloaded into a fresh pair, "
        f"continued to {proof['total_steps']}"
    )
    print(f"  checkpoint metadata: {proof['metadata']}")
    print(f"  archived model state keys: {proof['state_keys']}")
    print(f"  resumed loss suffix match: {proof['suffix_matches']}")
    print(f"  final parameters match:    {proof['parameters_match']}")
    print(f"  optimizer state matches:   {proof['optimizer_state_matches']}")
    print(f"  final logits match:        {proof['logits_match']}")
    print(f"  predictions match:         {proof['predictions_match']}")
    print(f"  final accuracy match:      {proof['accuracies_match']}")
    print(f"exact resume: {'yes' if exact else 'no'}")

    if not exact:
        raise SystemExit("resumed run diverged from the uninterrupted run")
    print("native classification training + checkpoint resume ok")


if __name__ == "__main__":
    main()
