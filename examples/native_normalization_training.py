"""The native normalized training and exact checkpoint-resume proof
(Advanced C++ Phase F, milestone F6).

A compact native regressor that runs **both** normalization families in
every forward —

    NativeLinear(2, 8, seed=0)
    -> NativeBatchNorm1d(8, momentum=0.1)   # the only stateful module
    -> NativeReLU()
    -> NativeLayerNorm(8)                    # affine parameters, no buffers
    -> NativeLinear(8, 1, seed=1)

— learns a fixed eight-sample two-feature regression task for a fixed
number of deterministic ``NativeAdam`` steps, with ``NativeMSELoss``,
entirely through the experimental native stack: native forward through
both normalization modules, native ``backward()`` (BatchNorm and
LayerNorm are compositions of existing operations, so the existing
autograd *is* their backward), and graph-free native optimizer updates
through the v3.7 ``copy_value_`` contract. **F6 adds no numerical
operation and no runtime capability** — no kernel, no C ABI export, no
new module, loss, metric, or optimizer, and no checkpoint schema change.
It assembles the pieces F0–F5 already shipped into one deterministic
training-and-resume proof.

**The stateful part.** ``NativeBatchNorm1d`` is the only stateful
normalization module here: it carries persistent ``running_mean`` /
``running_var`` buffers that advance once per training forward and that
evaluation mode reads instead of the batch's own statistics.
``NativeLayerNorm`` contributes learnable ``weight`` / ``bias`` affine
parameters but **no buffers** — it normalizes each sample over its own
features and behaves identically in train and eval. The resume proof is
meaningful precisely because it must restore the BatchNorm running
statistics *and* the NativeAdam moment buffers, and then reproduce both
the training-path prediction and the **evaluation-mode** output exactly.

**The data.** ``X_VALUES`` / ``Y_VALUES`` are fixed Python literals
(eight samples, two input features, one target), handed once to
``NativeTensor.from_array``. Nothing is generated, shuffled, downloaded,
augmented, or randomly sampled; the same full batch is used in the same
order for every step; no random number is drawn during training.

**The proof.** ``run_training()`` runs the uninterrupted schedule and
reports the deterministic loss curve and the run's evidence.
``run_resume_proof()`` runs the same schedule twice: once uninterrupted,
and once interrupted at a fixed step, saved to one pickle-free native
checkpoint (model **and** optimizer state, format **version 1**,
unchanged — including the BatchNorm running buffers as ordinary model
state), reloaded into a **completely fresh** model/optimizer pair, and
continued to the end. The two runs must agree **exactly** — the whole
remaining loss suffix, every parameter, every NativeAdam moment, the
BatchNorm running statistics, the final training-step prediction, and the
final evaluation-mode output — because the native CPU float64 kernels are
deterministic and nothing random happens between the checkpoint and the
resume. Exact equality is asserted, never a tolerance.

Lifetime is explicit: each step builds a completely fresh graph and
closes its prediction and loss after the one-shot ``backward()`` has
released that graph — and with it the BatchNorm eval-mode snapshots when
they exist — so nothing accumulates between steps; and every run closes
its parameters, its **buffers**, its optimizer, and its data tensors on
the way out, since a stateful native module has no
``NativeModule.close()``. The checkpoint lives in a temporary directory
that is removed automatically; nothing is left behind.

This is an integration proof for the native normalization stack on one
fixed task — not a benchmark, not a generalization claim, and no
performance is claimed. Honest benchmark characterization is F7's job. It
needs the experimental C++ backend to be built — run:

    uv run python examples/native_normalization_training.py

Every public helper that represents a completed run returns plain Python
values only (never a live native tensor, parameter, optimizer-state
tensor, model, or optimizer), so the tests can import and verify them;
``main()`` prints them.
"""

import os
import tempfile

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchNorm1d,
    NativeLayerNorm,
    NativeLinear,
    NativeMSELoss,
    NativeModule,
    NativeReLU,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)

# Fixed synthetic regression data: 8 samples, 2 input features, 1 target
# feature — the same literals the first native MLP proof used, chosen once
# from y = 0.5*x0 - x1 + 0.25 and frozen here so the example is
# self-contained. Nothing is generated, loaded, or preprocessed; every
# value is exactly representable in float64.
X_VALUES = [
    [0.0, 0.5],
    [1.0, -1.0],
    [-0.5, 1.5],
    [2.0, 0.0],
    [-1.5, -0.5],
    [0.5, 2.0],
    [1.5, 1.0],
    [-2.0, 1.0],
]
Y_VALUES = [
    [-0.25],
    [1.75],
    [-1.5],
    [1.25],
    [0.0],
    [-1.5],
    [0.0],
    [-1.75],
]

IN_FEATURES = 2
HIDDEN_FEATURES = 8
OUT_FEATURES = 1
HIDDEN_SEED = 0
OUTPUT_SEED = 1
MOMENTUM = 0.1
TOTAL_STEPS = 24
SPLIT_STEP = 10
DEFAULT_LR = 0.05

# The canonical child-module names, in registration (execution) order.
BATCH_NORM_NAME = "batch_norm"


def build_dataset():
    """The fixed task as plain host data: ``(inputs, targets)`` nested
    lists of literals.

    Deterministic by construction — these are fresh copies of the
    module-level literals, so repeated calls are equal and a caller
    mutating one result cannot perturb the next or the literals."""
    inputs = [list(row) for row in X_VALUES]
    targets = [list(row) for row in Y_VALUES]
    return inputs, targets


class NativeNormalizedRegressor(NativeModule):
    """The deterministic native regressor that runs **both** normalization
    families in every forward::

        hidden      = NativeLinear(2, 8, seed=0)
        batch_norm  = NativeBatchNorm1d(8, momentum=0.1)   # stateful
        relu        = NativeReLU()
        layer_norm  = NativeLayerNorm(8)                   # affine, stateless
        output      = NativeLinear(8, 1, seed=1)

    producing ``(N, 1)`` predictions. Every child is registered through
    the normal ``NativeModule`` attribute-assignment path under a **named**
    attribute, so parameter names, ``state_dict()`` keys, and checkpoint
    keys are readable and exact rather than anonymous ``NativeSequential``
    slot numbers.

    ``batch_norm`` is the only stateful module (persistent
    ``running_mean`` / ``running_var`` buffers); ``layer_norm`` contributes
    ``weight`` / ``bias`` affine parameters but no buffers. There is no
    ``NativeBatchNorm2d`` and no convolutional layer — the full
    convolutional integration model is F8's scope, not this proof's."""

    def __init__(self, hidden_seed=HIDDEN_SEED, output_seed=OUTPUT_SEED,
                 momentum=MOMENTUM):
        super().__init__()
        self.hidden = NativeLinear(IN_FEATURES, HIDDEN_FEATURES,
                                   seed=hidden_seed)
        self.batch_norm = NativeBatchNorm1d(HIDDEN_FEATURES, momentum=momentum)
        self.relu = NativeReLU()
        self.layer_norm = NativeLayerNorm(HIDDEN_FEATURES)
        self.output = NativeLinear(HIDDEN_FEATURES, OUT_FEATURES,
                                   seed=output_seed)

    def forward(self, inputs):
        """``(N, 2)`` inputs to ``(N, 1)`` predictions, through both
        normalization families. The intermediates are dropped as locals —
        the autograd graph holds what backward needs and releases it all
        at once."""
        hidden = self.hidden(inputs)
        hidden = self.batch_norm(hidden)
        hidden = self.relu(hidden)
        hidden = self.layer_norm(hidden)
        return self.output(hidden)


def build_model():
    """A freshly initialized regressor. Deterministic: both linear layers
    draw their fan-in uniform initialization from a *local* seeded
    generator, and the normalization parameters/buffers start from fixed
    constants (ones/zeros), so two independently built models start
    numerically identical and the global NumPy RNG is never touched."""
    return NativeNormalizedRegressor()


def build_loss():
    """The native mean-squared-error loss (scalar output)."""
    return NativeMSELoss()


def build_optimizer(model, lr=DEFAULT_LR):
    """The canonical adaptive optimizer — NativeAdam over the model's
    trainable parameters only. Its persistent moment buffers and
    per-parameter step counters are what make the resume proof meaningful:
    restoring them is what makes the resumed trajectory match, and
    comparing them is what proves it. Buffers are never handed to it."""
    return NativeAdam(model.parameters(), lr=lr)


def train_step(model, loss_fn, optimizer, inputs, targets,
               capture_prediction=False):
    """One full training iteration: model in training mode -> fresh graph
    -> scalar loss -> record the pre-update loss -> backward -> NativeAdam
    step -> zero_grad, closing this step's prediction and loss.

    The one-shot ``backward()`` releases the operation graph so nothing
    accumulates between steps and no stale graph can be reused after
    ``step()``. The single BatchNorm running-statistics update happens
    inside this one training forward. Returns the loss *before* this step's
    update as a plain float; with ``capture_prediction=True`` returns
    ``(loss, prediction_values)`` where ``prediction_values`` is the
    training-mode prediction recorded (as plain nested lists) before the
    graph is closed — so the final training-path prediction can be
    compared without an extra state-mutating training forward.

    Deliberately native throughout: the only host conversions are the
    scalar-loss inspection exit and, when requested, the reporting capture
    of the prediction — both after backward has run, neither part of the
    training mathematics (the tripwire test proves the numerical step
    reaches no NumPy)."""
    model.train()
    prediction = model(inputs)
    loss = loss_fn(prediction, targets)
    captured = None
    try:
        value = float(loss.to_numpy())   # scalar inspection, before release
        if capture_prediction:
            captured = prediction.to_numpy().tolist()
        loss.backward()
        optimizer.step()
    finally:
        loss.close()
        prediction.close()
    optimizer.zero_grad()
    if capture_prediction:
        return value, captured
    return value


def evaluate(model, loss_fn, inputs, targets):
    """A no-update reporting pass in **evaluation mode**: returns
    ``(loss, output)`` as plain Python values, where ``output`` is the
    ``(N, 1)`` prediction computed from the model's **stored BatchNorm
    running statistics** (not the batch's own).

    Evaluation performs no optimizer update and mutates no running state;
    it closes every native tensor it creates, and it restores the caller's
    previous training mode before returning — so a reporting pass never
    silently leaves the model in eval mode."""
    was_training = model.training
    model.eval()
    prediction = model(inputs)
    loss = loss_fn(prediction, targets)
    try:
        return float(loss.to_numpy()), prediction.to_numpy().tolist()
    finally:
        loss.close()
        prediction.close()
        model.train(was_training)


def _running_stats(model):
    """The BatchNorm running statistics as plain Python values, read by
    name so the resume evidence is obvious rather than hidden inside one
    generic state comparison. Reading a live buffer with ``to_numpy()``
    materializes a fresh host array and never mutates the buffer."""
    batch_norm = getattr(model, BATCH_NORM_NAME)
    return {
        "running_mean": batch_norm.running_mean.to_numpy().tolist(),
        "running_var": batch_norm.running_var.to_numpy().tolist(),
    }


def _model_state_values(model):
    """Snapshot ``model.state_dict()`` into an ordered mapping of plain
    values — every parameter first, then the BatchNorm ``running_mean`` /
    ``running_var`` buffers, in canonical order — and **close every
    snapshot** in a finally block, returning no native tensor.

    The returned dict preserves the canonical ``state_dict()`` order, so
    ``list(result)`` is the canonical state-key order."""
    state = model.state_dict()
    try:
        return {name: tensor.to_numpy().tolist()
                for name, tensor in state.items()}
    finally:
        for snapshot in state.values():
            snapshot.close()


def _optimizer_state_values(optimizer):
    """Materialize the NativeAdam state as plain Python values so two runs
    can be compared exactly without holding native tensors.

    ``state_dict()`` returns **caller-owned** ``m`` / ``v`` snapshots, so
    every one is closed after materialization (in a finally block) — the
    reporting helper must never leak optimizer-state storage."""
    state = optimizer.state_dict()
    try:
        return {
            "format_version": state["format_version"],
            "optimizer": state["optimizer"],
            "lr": state["lr"],
            "betas": list(state["betas"]),
            "eps": state["eps"],
            "parameters": [
                {"shape": list(entry["shape"]),
                 "dtype": entry["dtype"],
                 "device": entry["device"]}
                for entry in state["parameters"]
            ],
            "step_counts": list(state["step_counts"]),
            "m": [tensor.to_numpy().tolist() for tensor in state["m"]],
            "v": [tensor.to_numpy().tolist() for tensor in state["v"]],
        }
    finally:
        for tensor in state["m"]:
            tensor.close()
        for tensor in state["v"]:
            tensor.close()


def _close_run(model, optimizer, *tensors):
    """Release everything a run owns, in the established order: the
    optimizer, then every unique model parameter, then every unique model
    buffer (both traversals are identity-deduplicated, so shared state
    closes exactly once), then any fixed data tensors. ``close()`` is
    idempotent, and nothing closed here is ever returned to the caller."""
    if optimizer is not None:
        optimizer.close()
    if model is not None:
        for parameter in model.parameters():
            parameter.close()
        for buffer in model.buffers():
            buffer.close()
    for tensor in tensors:
        if tensor is not None:
            tensor.close()


def run_training(steps=TOTAL_STEPS, lr=DEFAULT_LR):
    """Train the deterministic native normalized regressor for ``steps``
    NativeAdam updates and return the run's evidence as plain Python
    values.

    The returned dict contains the configuration (``steps``, ``lr``,
    ``momentum``), the canonical ``parameter_names`` / ``buffer_names`` /
    ``state_keys`` (in canonical order), the per-step ``loss_history``
    (each entry the training loss *before* that step's update),
    ``initial_loss`` / ``final_loss`` (first and last training losses),
    ``initial_parameters`` / ``final_parameters`` and
    ``initial_running_stats`` / ``final_running_stats``,
    ``final_train_prediction`` (the last training-step prediction),
    ``final_eval`` (``(loss, output)`` from a post-training evaluation
    pass), ``final_optimizer_state``, the first-step gradient evidence
    (``gradient_shapes``, ``gradient_nonzero``, ``all_parameters_reached``,
    ``buffers_grad_free``), ``identity_stable``, ``gradients_cleared``,
    ``running_stats_advanced``, and the train/eval mode evidence
    (``eval_differs_from_train``, ``mode_restored``).

    Everything the run creates — the fixed data tensors, the model's
    parameters and buffers, and the optimizer — is closed before
    returning, success or failure; the caller receives Python values
    only."""
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError(f"steps must be an int, got {type(steps).__name__}")
    if steps <= 0:
        raise ValueError(f"steps must be strictly positive, got {steps}")

    inputs, targets = build_dataset()
    model = build_model()
    loss_fn = build_loss()
    optimizer = build_optimizer(model, lr=lr)   # validates lr

    named = list(model.named_parameters())
    names = [name for name, _ in named]
    parameters = [parameter for _, parameter in named]
    identities = [id(parameter) for parameter in parameters]
    buffer_named = list(model.named_buffers())
    buffer_names = [name for name, _ in buffer_named]
    buffer_identities = [id(buffer) for _, buffer in buffer_named]

    x = NativeTensor.from_array(inputs)
    y = NativeTensor.from_array(targets)
    try:
        results = {
            "steps": steps,
            "lr": optimizer.lr,
            "momentum": MOMENTUM,
            "parameter_names": names,
            "buffer_names": buffer_names,
            "state_keys": list(model.state_dict()),
            "initial_parameters": _model_state_values(model),
            "initial_running_stats": _running_stats(model),
        }

        loss_history = []
        identity_stable = True
        gradient_shapes = None
        gradient_nonzero = None
        all_parameters_reached = None
        buffers_grad_free = None
        final_train_prediction = None
        for step in range(steps):
            model.train()
            prediction = model(x)
            loss = loss_fn(prediction, y)
            try:
                loss_history.append(float(loss.to_numpy()))
                if step == steps - 1:
                    final_train_prediction = prediction.to_numpy().tolist()
                loss.backward()
                if step == 0:
                    # Evidence from the very first backward that every
                    # trainable parameter is reached and every buffer stays
                    # gradient-free — taken here, not inferred from the
                    # final parameter difference.
                    gradient_shapes = {
                        name: parameter.grad.shape for name, parameter in named
                    }
                    gradient_nonzero = {
                        name: bool((parameter.grad.to_numpy() != 0.0).any())
                        for name, parameter in named
                    }
                    all_parameters_reached = all(
                        parameter.grad is not None for _, parameter in named
                    )
                    buffers_grad_free = all(
                        buffer.grad is None for _, buffer in buffer_named
                    )
                optimizer.step()
            finally:
                loss.close()
                prediction.close()
            optimizer.zero_grad()
            identity_stable &= (
                [id(p) for _, p in model.named_parameters()] == identities
                and [id(b) for _, b in model.named_buffers()]
                == buffer_identities
            )

        final_running_stats = _running_stats(model)
        # The model is left in training mode by the loop; evaluate() reads
        # the stored running statistics and restores that mode on the way
        # out, so ``mode_restored`` proves evaluation does not leak eval
        # mode. The eval-mode output differs from the final training-step
        # prediction because eval uses the running statistics rather than
        # the batch's own — the reason BatchNorm carries running state.
        final_eval_loss, final_eval_output = evaluate(model, loss_fn, x, y)
        results.update(
            loss_history=loss_history,
            initial_loss=loss_history[0],
            final_loss=loss_history[-1],
            final_parameters=_model_state_values(model),
            final_running_stats=final_running_stats,
            final_train_prediction=final_train_prediction,
            final_eval=(final_eval_loss, final_eval_output),
            final_optimizer_state=_optimizer_state_values(optimizer),
            gradient_shapes=gradient_shapes,
            gradient_nonzero=gradient_nonzero,
            all_parameters_reached=all_parameters_reached,
            buffers_grad_free=buffers_grad_free,
            identity_stable=identity_stable,
            gradients_cleared=all(p.grad is None for p in parameters),
            running_stats_advanced=(
                final_running_stats != results["initial_running_stats"]
            ),
            eval_differs_from_train=(
                final_train_prediction != final_eval_output
            ),
            mode_restored=model.training is True,
        )
        return results
    finally:
        _close_run(model, optimizer, x, y)


def run_resume_proof(total_steps=TOTAL_STEPS, split_step=SPLIT_STEP,
                     lr=DEFAULT_LR, directory=None):
    """Run the same schedule twice — once uninterrupted, once interrupted
    at ``split_step``, checkpointed, reloaded into a **fresh** model and
    optimizer, and continued — then compare them exactly.

    ``directory`` is an optional existing directory for the checkpoint; by
    default a temporary one is created and removed, so the default run
    leaves no file behind.

    Returns plain Python values: both loss histories, the prefix/suffix
    comparisons, the final training-step predictions and evaluation
    outputs, whether every parameter value, the complete model state, the
    BatchNorm running statistics, and every NativeAdam state entry match,
    the checkpoint metadata and the archived model state keys, and — for
    the fresh load target — whether parameter and buffer identities were
    preserved and the training flag stayed unserialized."""
    if split_step <= 0 or split_step >= total_steps:
        raise ValueError(
            f"split_step must satisfy 0 < split_step < total_steps, got "
            f"split_step={split_step}, total_steps={total_steps}"
        )

    inputs, targets = build_dataset()
    loss_fn = build_loss()
    x = NativeTensor.from_array(inputs)
    y = NativeTensor.from_array(targets)

    model_a = optimizer_a = None
    model_b = optimizer_b = model_c = optimizer_c = None
    try:
        # -- Path A: uninterrupted -------------------------------------
        model_a = build_model()
        optimizer_a = build_optimizer(model_a, lr=lr)
        initial_state_a = _model_state_values(model_a)
        losses_a = []
        final_train_prediction_a = None
        for step in range(total_steps):
            if step == total_steps - 1:
                value, final_train_prediction_a = train_step(
                    model_a, loss_fn, optimizer_a, x, y, capture_prediction=True
                )
            else:
                value = train_step(model_a, loss_fn, optimizer_a, x, y)
            losses_a.append(value)
        eval_loss_a, eval_output_a = evaluate(model_a, loss_fn, x, y)
        parameters_a = _model_state_values(model_a)
        running_a = _running_stats(model_a)
        state_a = _optimizer_state_values(optimizer_a)
        names_a = [name for name, _ in model_a.named_parameters()]
        buffer_names_a = [name for name, _ in model_a.named_buffers()]

        # -- Path B: train, checkpoint, resume into a fresh pair --------
        model_b = build_model()
        optimizer_b = build_optimizer(model_b, lr=lr)
        start_state_b = _model_state_values(model_b)
        losses_b_prefix = [
            train_step(model_b, loss_fn, optimizer_b, x, y)
            for _ in range(split_step)
        ]

        if directory is None:
            context = tempfile.TemporaryDirectory()
        else:
            context = _ExistingDirectory(directory)
        with context as checkpoint_directory:
            path = os.path.join(checkpoint_directory,
                                "native_normalization.checkpoint.npz")
            save_native_checkpoint(
                path, model_b, optimizer=optimizer_b,
                metadata={"steps_completed": split_step, "lr": lr},
            )
            model_c = build_model()
            optimizer_c = build_optimizer(model_c, lr=lr)
            parameter_ids_before = [id(p) for p in model_c.parameters()]
            buffer_ids_before = [id(b) for b in model_c.buffers()]
            # Deliberately put the fresh target in eval mode before loading,
            # so the load is proved not to serialize or overwrite the flag.
            model_c.eval()
            metadata = load_native_checkpoint(
                path, model_c, optimizer=optimizer_c
            )
            mode_after_load = model_c.training
            parameter_ids_after = [id(p) for p in model_c.parameters()]
            buffer_ids_after = [id(b) for b in model_c.buffers()]
            archive_keys = list(model_c.state_dict())
            # The training flag is runtime state; switch it back explicitly.
            model_c.train()

        losses_b_suffix = []
        final_train_prediction_c = None
        remaining = total_steps - split_step
        for step in range(remaining):
            if step == remaining - 1:
                value, final_train_prediction_c = train_step(
                    model_c, loss_fn, optimizer_c, x, y, capture_prediction=True
                )
            else:
                value = train_step(model_c, loss_fn, optimizer_c, x, y)
            losses_b_suffix.append(value)
        eval_loss_c, eval_output_c = evaluate(model_c, loss_fn, x, y)
        parameters_c = _model_state_values(model_c)
        running_c = _running_stats(model_c)
        state_c = _optimizer_state_values(optimizer_c)
        names_c = [name for name, _ in model_c.named_parameters()]
        buffer_names_c = [name for name, _ in model_c.named_buffers()]

        losses_b = losses_b_prefix + losses_b_suffix
        return {
            "total_steps": total_steps,
            "split_step": split_step,
            "lr": lr,
            "metadata": metadata,
            "identical_start": initial_state_a == start_state_b,
            "uninterrupted_losses": losses_a,
            "resumed_losses": losses_b,
            "resumed_suffix": losses_b_suffix,
            "prefix_matches": losses_a[:split_step] == losses_b_prefix,
            "suffix_matches": losses_a[split_step:] == losses_b_suffix,
            "first_resumed_loss_matches": (
                losses_b_suffix[0] == losses_a[split_step]
            ),
            "losses_match": losses_a == losses_b,
            "final_train_prediction_uninterrupted": final_train_prediction_a,
            "final_train_prediction_resumed": final_train_prediction_c,
            "final_train_predictions_match": (
                final_train_prediction_a == final_train_prediction_c
            ),
            "final_eval_uninterrupted": (eval_loss_a, eval_output_a),
            "final_eval_resumed": (eval_loss_c, eval_output_c),
            "final_eval_matches": (
                eval_loss_a == eval_loss_c and eval_output_a == eval_output_c
            ),
            "parameters_match": parameters_a == parameters_c,
            "running_mean_matches": (
                running_a["running_mean"] == running_c["running_mean"]
            ),
            "running_var_matches": (
                running_a["running_var"] == running_c["running_var"]
            ),
            "optimizer_state_matches": state_a == state_c,
            "parameter_order_matches": names_a == names_c,
            "buffer_order_matches": buffer_names_a == buffer_names_c,
            "state_keys": archive_keys,
            "identities_preserved": (
                parameter_ids_before == parameter_ids_after
                and buffer_ids_before == buffer_ids_after
            ),
            "mode_not_serialized": mode_after_load is False,
        }
    finally:
        for model, optimizer in (
            (model_a, optimizer_a), (model_b, optimizer_b),
            (model_c, optimizer_c),
        ):
            if model is not None or optimizer is not None:
                _close_run(model, optimizer)
        x.close()
        y.close()


class _ExistingDirectory:
    """A context manager mirroring ``TemporaryDirectory``'s interface for a
    caller-supplied directory — entered and left without creating or
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
        f"native normalized regressor: "
        f"Linear({IN_FEATURES} -> {HIDDEN_FEATURES}) -> "
        f"BatchNorm1d({HIDDEN_FEATURES}, momentum={MOMENTUM}) -> ReLU -> "
        f"LayerNorm({HIDDEN_FEATURES}) -> Linear({HIDDEN_FEATURES} -> "
        f"{OUT_FEATURES})"
    )
    print(
        f"trained {run['steps']} NativeAdam steps (lr={run['lr']}) on "
        f"{len(X_VALUES)} fixed samples ({IN_FEATURES} features -> "
        f"{OUT_FEATURES} target) with NativeMSELoss"
    )
    print(f"parameters: {run['parameter_names']}")
    print(f"buffers:    {run['buffer_names']}  (BatchNorm running statistics)")
    print(f"initial training loss: {run['initial_loss']:.6f}")
    print(f"final training loss:   {run['final_loss']:.6f}")
    print(f"loss reduction:        {reduction:.2f}%")
    print(f"final eval loss:       {run['final_eval'][0]:.6f}")
    print(f"BatchNorm running stats advanced: {run['running_stats_advanced']}")
    print(f"LayerNorm has buffers: "
          f"{'no' if 'layer_norm' not in ' '.join(run['buffer_names']) else 'yes'}")
    print(f"every parameter reached by backward: {run['all_parameters_reached']}")
    print(f"buffers stayed gradient-free: {run['buffers_grad_free']}")

    proof = run_resume_proof()
    exact = (
        proof["identical_start"]
        and proof["prefix_matches"]
        and proof["first_resumed_loss_matches"]
        and proof["suffix_matches"]
        and proof["losses_match"]
        and proof["final_train_predictions_match"]
        and proof["final_eval_matches"]
        and proof["parameters_match"]
        and proof["running_mean_matches"]
        and proof["running_var_matches"]
        and proof["optimizer_state_matches"]
        and proof["identities_preserved"]
        and proof["mode_not_serialized"]
    )
    print()
    print(
        f"checkpoint resume: trained {proof['split_step']} steps, saved model "
        f"+ optimizer state (format version 1), reloaded into a fresh pair, "
        f"continued to {proof['total_steps']}"
    )
    print(f"  checkpoint metadata:        {proof['metadata']}")
    print(f"  archived model state keys:  {proof['state_keys']}")
    print(f"  resumed loss suffix match:  {proof['suffix_matches']}")
    print(f"  final parameters match:     {proof['parameters_match']}")
    print(f"  running_mean match:         {proof['running_mean_matches']}")
    print(f"  running_var match:          {proof['running_var_matches']}")
    print(f"  optimizer state matches:    {proof['optimizer_state_matches']}")
    print(f"  final train prediction match: {proof['final_train_predictions_match']}")
    print(f"  final eval output match:    {proof['final_eval_matches']}")
    print(f"  identities preserved:       {proof['identities_preserved']}")
    print(f"  training mode unserialized: {proof['mode_not_serialized']}")
    print(f"exact resume: {'yes' if exact else 'no'}")

    if not exact:
        raise SystemExit("resumed run diverged from the uninterrupted run")
    print("native normalized training + checkpoint resume ok")


if __name__ == "__main__":
    main()
