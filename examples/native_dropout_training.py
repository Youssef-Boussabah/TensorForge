"""Deterministic native Dropout training and exact stochastic resume
(Advanced C++ Phase G, milestone G7).

A compact native classifier that carries **all four** TensorForge-owned
state families in one model —

    NativeLinear(4, 8, seed=0)
    -> NativeBatchNorm1d(8)      # persistent running_mean / running_var
    -> NativeReLU()
    -> NativeDropout(p=0.5, seed=20240707)   # a registered NativeGenerator
    -> NativeLayerNorm(8)        # affine parameters, no buffers
    -> NativeLinear(8, 3, seed=1)

— learns a fixed twelve-sample three-class task for a fixed number of
deterministic ``NativeAdam`` steps over **raw logits** with
``NativeCrossEntropyLoss``, entirely through the experimental native
stack. **G7 adds no numerical operation and no runtime capability** — no
kernel, no C ABI export, no new module, loss, metric, or optimizer, and
no checkpoint schema or version change. It assembles what G1-G6 already
shipped into one deterministic training-and-resume proof.

**Why this model.** The resume is only interesting if something would
diverge without it. Every layer here contributes a distinct state family
that a checkpoint must restore: the two ``NativeLinear`` layers and the
affine parameters contribute **trainable parameters**;
``NativeBatchNorm1d`` contributes **persistent running buffers** that
advance once per training forward and that evaluation reads instead of
the batch's own statistics; ``NativeDropout`` contributes a registered
**``NativeGenerator``** whose call counter advances once per training
forward; and ``NativeAdam`` contributes **moment buffers and per-parameter
step counters**. Miss any one of the four and the resumed trajectory
diverges immediately.

**The data.** ``build_dataset()`` computes twelve four-feature samples
from an explicit arithmetic formula over the sample index — every value is
a quarter or an eighth, so all of them are exactly representable in
float64 and identical on every platform. Nothing is generated randomly,
loaded, downloaded, augmented, shuffled, or split. Labels stay **host
integers**, because the native runtime has no integer dtype and the
classification stack's strict ``int64`` target contract is exactly why.

**The batch schedule is a pure function of the step.** The twelve samples
form three fixed batches of four, and step *s* always trains on batch
``s % 3`` — see ``batch_index_for_step``. That is the whole reason the
external loop position can be carried by a single integer, and it is
tested: resuming at the wrong batch changes the loss sequence.

**What the checkpoint captures, and what it does not.** Format version 2
captures TensorForge-owned state: model parameters, persistent buffers,
optimizer state, and every registered generator's state **and sharing
topology**. It does **not** capture — and this example does not pretend it
does — data-loader position, batch order, shuffle state, epoch counters,
scheduler state, Python's ``random`` module, or NumPy's global RNG. There
is no data loader in the native line to capture. The external loop
position is therefore carried **explicitly**, as ordinary JSON metadata
(``{"training_step": ..., "next_batch_index": ...}``), validated on the
way back in by ``validated_progress`` — which rejects a missing field, a
wrong type, an out-of-range value, or a ``next_batch_index`` that
disagrees with the schedule, rather than silently restarting from step
zero. Reproducibility is exact **for the state actually captured**;
full-program determinism is not claimed.

**The proof.** ``run_training()`` runs the uninterrupted schedule and
reports the deterministic loss curve and the run's evidence.
``run_resume_proof()`` runs the same schedule twice: once uninterrupted,
and once interrupted after a fixed number of **completed** steps, saved to
one pickle-free native checkpoint (model, optimizer, generator, and
progress metadata, format **version 2**), reloaded into a **completely
fresh** model/optimizer/generator set — built with a deliberately
*different* Dropout seed, so a load that failed to restore the stream
would be obvious — and continued to the end. The two runs must agree
**exactly**: the whole loss sequence, every parameter, every persistent
BatchNorm statistic, every NativeAdam moment and step counter, the
generator's algorithm/version/seed/calls, the final training-step logits,
and the final evaluation-mode output. Exact equality is asserted
throughout, never a tolerance — the native CPU float64 kernels are
deterministic (fixed loop orders, no parallel reduction, no fast-math) and
the restored generator makes the *stochastic* part deterministic too.

``run_next_mask_proof()`` closes the loop back to the G2 Core: it reloads
the same checkpoint into a **throwaway** model (so the resumed run is
untouched), pushes a fixed probe tensor through that model's restored
``NativeDropout``, and checks the result against the stateless Core called
with the exact restored ``(seed, call_index)`` — proving the next mask
after a resume is the one the saved stream owed, and that it consumes
exactly one call.

Lifetime is explicit: each step builds a completely fresh graph and closes
its logits and loss after the one-shot ``backward()`` has released that
graph — and with it the Dropout multiplier mask, the BatchNorm eval
snapshots when they exist, and cross-entropy's saved probabilities — so
nothing accumulates between steps; and every run closes its parameters,
its buffers, its optimizer, and its data tensors on the way out, since a
stateful native module has no ``NativeModule.close()``. A generator owns
no native storage and has no ``close()``. The checkpoint lives in a
temporary directory that is removed automatically; nothing is left behind,
and ``main()`` reports the native live-storage count before and after the
whole workflow.

This is an integration proof for the native RNG and Dropout stack on one
fixed task — not a benchmark, not a generalization claim, and no
performance is claimed. Honest benchmark characterization is G8's job. It
needs the experimental C++ backend to be built — run:

    uv run python examples/native_dropout_training.py

Every public helper that represents a completed run returns plain Python
values only (never a live native tensor, parameter, generator, model, or
optimizer), so the tests can import and verify them; ``main()`` prints
them.
"""

import gc
import os
import tempfile

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchNorm1d,
    NativeCrossEntropyLoss,
    NativeDropout,
    NativeLayerNorm,
    NativeLinear,
    NativeModule,
    NativeReLU,
    NativeTensor,
    load_native_checkpoint,
    native_accuracy,
    save_native_checkpoint,
)

FEATURES = 4
HIDDEN_FEATURES = 8
NUM_CLASSES = 3
SAMPLES = 12
BATCH_SIZE = 4
NUM_BATCHES = SAMPLES // BATCH_SIZE          # 3

HIDDEN_SEED = 0
OUTPUT_SEED = 1
DROPOUT_SEED = 20240707
DROPOUT_P = 0.5
# A deliberately different seed for the fresh restore target, so a load
# that failed to restore the stream could not possibly go unnoticed.
FRESH_DROPOUT_SEED = 999999

TOTAL_STEPS = 16
# Not a multiple of NUM_BATCHES on purpose: the resume lands mid-cycle in
# the batch schedule, so a loop that restarted the schedule at batch 0
# would diverge. That is exactly what the progress metadata prevents.
SPLIT_STEP = 7
DEFAULT_LR = 0.05

# The canonical child-module names, in registration (execution) order.
DROPOUT_NAME = "dropout"
BATCH_NORM_NAME = "batch_norm"
# The canonical name the registered generator gets in a checkpoint:
# "<module attribute>.<generator attribute>".
GENERATOR_KEY = f"{DROPOUT_NAME}.generator"

# The exact keys the progress metadata carries, and nothing else.
PROGRESS_FIELDS = ("training_step", "next_batch_index")

# A fixed probe tensor for the next-mask proof. Twelve exactly
# representable values, unrelated to the dataset — the mask depends only
# on (seed, call_index, element index, p), never on the values.
PROBE_VALUES = [0.5, -1.0, 1.5, -0.25, 2.0, 0.75,
                -1.75, 1.25, -0.5, 0.25, -2.0, 1.0]


def build_dataset():
    """The fixed task as plain host data: ``(inputs, targets)``, where
    ``inputs`` is a ``12 x 4`` nested list and ``targets`` is the list of
    integer class labels.

    Computed from an explicit formula over the sample index rather than
    stored as literals, so the structure is visible: sample *i* belongs to
    class ``i % 3`` and sits at position ``i // 3``, its own class feature
    carries the strong positive signal ``1.0 + offset``, the next feature
    carries ``-0.75 + offset``, and the rest carry ``offset - 0.375``,
    where ``offset`` is one of ``0.0, 0.25, 0.5, 0.75``. Position varies
    within every class, so no single feature threshold separates them.

    Every value is a quarter or an eighth and therefore exact in float64.
    Nothing consults the clock, a random source, the filesystem, or the
    network, and repeated calls return equal, independent lists."""
    inputs = []
    targets = []
    for index in range(SAMPLES):
        label = index % NUM_CLASSES
        offset = (index // NUM_CLASSES) / 4.0        # 0.0, 0.25, 0.5, 0.75
        row = [offset - 0.375] * FEATURES
        row[label] = 1.0 + offset
        row[(label + 1) % FEATURES] = -0.75 + offset
        inputs.append(row)
        targets.append(label)
    return inputs, targets


def batch_index_for_step(step):
    """Which batch training step ``step`` uses — ``step % NUM_BATCHES``.

    **A pure function of the step, deliberately.** The whole external
    loop position therefore collapses to one integer, which is what makes
    ``{"training_step": k}`` a complete and honest description of where a
    run stopped. There is no shuffling, no epoch state, and no data-loader
    object, so there is nothing else to carry — and nothing is silently
    inferred, because the saved ``next_batch_index`` is checked against
    this function on the way back in."""
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError(f"step must be an int, got {type(step).__name__}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    return step % NUM_BATCHES


def build_batches(inputs, targets):
    """The fixed batch schedule as ``[(NativeTensor, [label, ...]), ...]``
    — three contiguous batches of four samples, in a fixed order.

    The caller owns every returned tensor and must close them. Labels stay
    host integers: the native runtime has no integer dtype."""
    batches = []
    for index in range(NUM_BATCHES):
        start = index * BATCH_SIZE
        stop = start + BATCH_SIZE
        batches.append((NativeTensor.from_array(inputs[start:stop]),
                        list(targets[start:stop])))
    return batches


class NativeDropoutClassifier(NativeModule):
    """The deterministic native classifier carrying all four state
    families::

        hidden      = NativeLinear(4, 8, seed=0)
        batch_norm  = NativeBatchNorm1d(8)        # persistent buffers
        relu        = NativeReLU()
        dropout     = NativeDropout(p, seed=...)  # a registered generator
        layer_norm  = NativeLayerNorm(8)          # affine, stateless
        output      = NativeLinear(8, 3, seed=1)

    producing **raw logits** of shape ``(batch_size, 3)``. There is
    deliberately no softmax or log-softmax module: the fused, numerically
    stable ``NativeCrossEntropyLoss`` consumes logits directly.

    Every child is registered through the normal ``NativeModule``
    attribute-assignment path under a **named** attribute, so parameter
    names, ``state_dict()`` keys, the generator's canonical name, and the
    checkpoint keys are all readable and exact rather than anonymous
    ``NativeSequential`` slot numbers.

    Dropout sits **after** the activation and **before** the layer
    normalization, which is the ordinary placement: it perturbs the
    activations, and the following normalization sees the perturbed
    batch. One Dropout module and one generator — shared-generator
    topology across several layers is proved by the G5 and G6 suites, and
    broader multi-layer integration is G9's scope, not this proof's."""

    def __init__(self, hidden_seed=HIDDEN_SEED, output_seed=OUTPUT_SEED,
                 dropout_seed=DROPOUT_SEED, p=DROPOUT_P):
        super().__init__()
        self.hidden = NativeLinear(FEATURES, HIDDEN_FEATURES,
                                   seed=hidden_seed)
        self.batch_norm = NativeBatchNorm1d(HIDDEN_FEATURES)
        self.relu = NativeReLU()
        self.dropout = NativeDropout(p, seed=dropout_seed)
        self.layer_norm = NativeLayerNorm(HIDDEN_FEATURES)
        self.output = NativeLinear(HIDDEN_FEATURES, NUM_CLASSES,
                                   seed=output_seed)

    def forward(self, inputs):
        """``(N, 4)`` inputs to ``(N, 3)`` raw logits. The intermediates
        are dropped as locals — the autograd graph holds what backward
        needs (including the Dropout multiplier mask) and releases it all
        at once."""
        hidden = self.hidden(inputs)
        hidden = self.batch_norm(hidden)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.layer_norm(hidden)
        return self.output(hidden)


def build_model(dropout_seed=DROPOUT_SEED):
    """A freshly initialized classifier. Deterministic: both linear layers
    draw their fan-in uniform initialization from a *local* seeded
    generator, the normalization parameters and buffers start from fixed
    constants, and the Dropout generator starts from the explicit
    ``dropout_seed`` — so two independently built models start numerically
    identical and neither the global NumPy RNG nor Python's ``random`` is
    ever touched."""
    return NativeDropoutClassifier(dropout_seed=dropout_seed)


def build_loss():
    """The native classification loss, over raw logits."""
    return NativeCrossEntropyLoss()


def build_optimizer(model, lr=DEFAULT_LR):
    """The canonical adaptive optimizer — NativeAdam over the model's
    trainable parameters only. Its persistent moment buffers and
    per-parameter step counters are part of what makes the resume proof
    meaningful: restoring them is what makes the resumed trajectory match,
    and comparing them is what proves it. Buffers and generators are never
    handed to it."""
    return NativeAdam(model.parameters(), lr=lr)


def train_step(model, loss_fn, optimizer, batches, step,
               capture_logits=False):
    """One full training iteration at schedule position ``step``: model in
    training mode -> the step's fixed batch -> fresh graph -> raw logits ->
    scalar loss -> record the pre-update loss -> backward -> NativeAdam
    step -> zero_grad, closing this step's logits and loss.

    The one-shot ``backward()`` releases the operation graph — and with it
    the Dropout multiplier mask and cross-entropy's saved probabilities —
    so nothing accumulates between steps and no stale graph can be reused
    after ``step()``. Exactly **one** generator call is consumed, by the
    single training-mode Dropout forward; the BatchNorm running statistics
    advance once, in the same forward.

    Returns the loss *before* this step's update as a plain float; with
    ``capture_logits=True`` returns ``(loss, logit_values)`` where
    ``logit_values`` is the training-mode logits recorded (as plain nested
    lists) before the graph is closed — so the final training-path output
    can be compared without an extra state-mutating training forward.

    Deliberately native throughout: the only host conversions are the
    scalar-loss inspection exit and, when requested, the reporting capture
    of the logits — both after backward has run, neither part of the
    training mathematics."""
    inputs, targets = batches[batch_index_for_step(step)]
    model.train()
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    captured = None
    try:
        value = float(loss.to_numpy())   # scalar inspection, before release
        if capture_logits:
            captured = logits.to_numpy().tolist()
        loss.backward()
        optimizer.step()
    finally:
        loss.close()
        logits.close()
    optimizer.zero_grad()
    if capture_logits:
        return value, captured
    return value


def evaluate(model, loss_fn, inputs, targets):
    """A no-update reporting pass in **evaluation mode** over the full
    dataset: ``(loss, accuracy, predictions, logits)`` as plain Python
    values.

    Evaluation is **state-neutral on every axis that matters here**: the
    Dropout module returns its input object and consumes no generator call,
    BatchNorm reads its stored running statistics instead of the batch's
    own and advances nothing, and no optimizer update happens. It closes
    every native tensor it creates and restores the caller's previous
    training mode before returning, so a reporting pass never silently
    leaves the model in eval mode — and never puts a gap in the random
    stream.

    ``native_accuracy`` and the predicted classes are **reporting only**:
    both leave native memory through the explicit public ``to_numpy()``
    boundary, which is exactly why neither belongs in ``train_step``."""
    was_training = model.training
    model.eval()
    logits = model(inputs)
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
        model.train(was_training)


# --------------------------------------------------------------------------
# External loop progress — carried explicitly, never inferred
# --------------------------------------------------------------------------


def progress_metadata(completed_steps, lr=DEFAULT_LR):
    """The external loop position as JSON-compatible checkpoint metadata.

    ``completed_steps`` is the number of **fully completed** training steps
    — a step counts only once its backward, optimizer update, and
    ``zero_grad`` have all run, so an interrupted step is never presented
    as a completed one. Because the schedule is a pure function of the
    step, ``training_step`` alone would be sufficient;
    ``next_batch_index`` is stored **as well** so the archive states the
    position it implies and a load can check the two against each other
    instead of trusting either alone.

    ``lr`` is recorded for the reader's benefit only. It is *not* the
    authority on the optimizer's learning rate — the optimizer section of
    the checkpoint is — and nothing in the resume path reads it back."""
    if isinstance(completed_steps, bool) or not isinstance(completed_steps, int):
        raise TypeError(
            f"completed_steps must be an int, got "
            f"{type(completed_steps).__name__}"
        )
    if completed_steps < 0:
        raise ValueError(
            f"completed_steps must be non-negative, got {completed_steps}"
        )
    return {
        "training_step": completed_steps,
        "next_batch_index": batch_index_for_step(completed_steps),
        "lr": lr,
    }


def validated_progress(metadata, total_steps=TOTAL_STEPS):
    """Validate loaded checkpoint metadata and return
    ``(training_step, next_batch_index)``.

    This is the honest half of "checkpoint version 2 does not capture a
    data loader": the external position is ordinary metadata, so it is the
    *example's* job to check it, and it is checked strictly rather than
    defaulted. A missing field, a non-``int`` (``bool`` included — ``True``
    is not a step), a negative or out-of-range step, or a
    ``next_batch_index`` that disagrees with ``batch_index_for_step`` all
    raise, naming the problem.

    Defaulting a missing ``training_step`` to ``0`` is exactly the failure
    mode this prevents: a resume that silently restarted from the
    beginning would still "work", would still converge, and would be a
    different run — so it must be an error, not a fallback."""
    if not isinstance(metadata, dict):
        raise TypeError(
            f"checkpoint metadata must be a dict, got "
            f"{type(metadata).__name__}"
        )
    for field in PROGRESS_FIELDS:
        if field not in metadata:
            raise ValueError(
                f"checkpoint metadata is missing the required progress "
                f"field {field!r}: this example carries its external loop "
                f"position explicitly, and never restarts from step 0 by "
                f"default (got keys {sorted(metadata)})"
            )
        value = metadata[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"checkpoint metadata {field!r} must be an int, got "
                f"{type(value).__name__}"
            )
    training_step = metadata["training_step"]
    next_batch_index = metadata["next_batch_index"]
    if not 0 <= training_step <= total_steps:
        raise ValueError(
            f"checkpoint metadata 'training_step' must be in "
            f"[0, {total_steps}], got {training_step}"
        )
    expected = batch_index_for_step(training_step)
    if next_batch_index != expected:
        raise ValueError(
            f"checkpoint metadata is inconsistent: 'training_step' "
            f"{training_step} implies batch {expected}, but "
            f"'next_batch_index' is {next_batch_index}"
        )
    return training_step, next_batch_index


# --------------------------------------------------------------------------
# Reporting helpers — plain Python values only
# --------------------------------------------------------------------------


def generator_state(model):
    """The registered Dropout generator's complete state as a plain dict
    (``algorithm``, ``algorithm_version``, ``seed``, ``calls``).

    Read by name so the resume evidence is obvious rather than buried in
    one generic comparison. Reading state creates no reservation, advances
    no counter, and allocates nothing native."""
    return getattr(model, DROPOUT_NAME).generator.state()


def running_stats(model):
    """The BatchNorm running statistics as plain Python values, read by
    name for the same reason. ``to_numpy()`` materializes a fresh host
    array and never mutates the buffer."""
    batch_norm = getattr(model, BATCH_NORM_NAME)
    return {
        "running_mean": batch_norm.running_mean.to_numpy().tolist(),
        "running_var": batch_norm.running_var.to_numpy().tolist(),
    }


def _model_state_values(model):
    """Snapshot ``model.state_dict()`` into an ordered mapping of plain
    values — every parameter first, then the BatchNorm buffers, in
    canonical order — closing **every** snapshot in a finally block and
    returning no native tensor.

    Generators are deliberately absent: ``state_dict()`` is contractually
    ``{name: NativeTensor}``, and generator state is reported separately by
    ``generator_state``."""
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
    every one is closed after materialization (in a finally block) — a
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
    closes exactly once), then any fixed data tensors.

    Generators are **not** closed and deliberately have no ``close()``:
    a ``NativeGenerator`` is a pure-Python value holder that owns no native
    storage. ``close()`` is idempotent, and nothing closed here is ever
    returned to the caller."""
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


def _close_batches(batches):
    """Close every batch input tensor. Labels are host lists."""
    for inputs, _ in batches:
        inputs.close()


class _LiveStorageMeter:
    """Counts open native storages for the duration of a ``with`` block.

    A **measurement instrument**, not behavior: it wraps
    ``NativeStorage.__init__`` / ``close`` to record which ones are open,
    changes nothing about what either does, and restores both in a
    ``finally``. It exists because the native runtime exposes no
    live-allocation counter and this example's final claim — that the
    whole workflow returns to its starting baseline — has to be measured
    rather than asserted.

    ``count()`` is the number currently open. Explicit ``close()`` is what
    the example relies on; the counter merely observes it."""

    def __init__(self):
        self._open = set()
        self._original_init = None
        self._original_close = None

    def count(self):
        return len(self._open)

    def settled_count(self):
        """``count()`` after one collection.

        Every run above closes its parameters, buffers, optimizer, and
        data tensors explicitly — that is the release mechanism, and it is
        what the baseline claim rests on. Two things are nevertheless left
        to the collector by contract, and both are reference *cycles* that
        refcounting alone cannot break: ``zero_grad()`` drops gradient
        objects without closing them (the documented optimizer contract),
        and the Python-managed autograd graph holds its parents through
        backward closures. Collecting here settles those into a
        deterministic number; it releases nothing the explicit cleanup was
        responsible for."""
        gc.collect()
        return len(self._open)

    def __enter__(self):
        storage = cpp.NativeStorage
        self._original_init = storage.__init__
        self._original_close = storage.close
        open_ids = self._open
        original_init = self._original_init
        original_close = self._original_close

        def tracked_init(instance, *args, **kwargs):
            original_init(instance, *args, **kwargs)   # raises => not recorded
            open_ids.add(id(instance))

        def tracked_close(instance):
            original_close(instance)
            open_ids.discard(id(instance))

        storage.__init__ = tracked_init
        storage.close = tracked_close
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        cpp.NativeStorage.__init__ = self._original_init
        cpp.NativeStorage.close = self._original_close
        return False


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


# --------------------------------------------------------------------------
# The runs
# --------------------------------------------------------------------------


def run_training(steps=TOTAL_STEPS, lr=DEFAULT_LR,
                 dropout_seed=DROPOUT_SEED, eval_probe_step=None):
    """Train the deterministic native Dropout classifier for ``steps``
    NativeAdam updates and return the run's evidence as plain Python
    values.

    With ``eval_probe_step`` set to a step index, the run pauses there and
    runs three evaluation passes back to back, recording the generator
    state before and after and whether the outputs were identical — the
    in-line proof that evaluation puts **no gap** in the random stream.

    Everything the run creates — the batch tensors, the full-dataset
    tensor, the model's parameters and buffers, and the optimizer — is
    closed before returning, success or failure; the caller receives Python
    values only."""
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError(f"steps must be an int, got {type(steps).__name__}")
    if steps <= 0:
        raise ValueError(f"steps must be strictly positive, got {steps}")

    inputs, targets = build_dataset()
    model = build_model(dropout_seed=dropout_seed)
    loss_fn = build_loss()
    optimizer = build_optimizer(model, lr=lr)   # validates lr

    named = list(model.named_parameters())
    identities = [id(parameter) for _, parameter in named]
    buffer_named = list(model.named_buffers())
    buffer_identities = [id(buffer) for _, buffer in buffer_named]
    generator = getattr(model, DROPOUT_NAME).generator

    batches = build_batches(inputs, targets)
    full = NativeTensor.from_array(inputs)
    try:
        results = {
            "steps": steps,
            "lr": optimizer.lr,
            "dropout_p": getattr(model, DROPOUT_NAME).p,
            "parameter_names": [name for name, _ in named],
            "buffer_names": [name for name, _ in buffer_named],
            "state_keys": list(model.state_dict()),
            "generator_keys": [name for name, _ in model.named_generators()],
            "initial_generator": generator_state(model),
            "initial_running_stats": running_stats(model),
            "batch_schedule": [batch_index_for_step(step)
                               for step in range(steps)],
        }
        initial_eval = evaluate(model, loss_fn, full, targets)

        loss_history = []
        identity_stable = True
        eval_probe = None
        final_train_logits = None
        for step in range(steps):
            if step == eval_probe_step:
                before = generator_state(model)
                outputs = [evaluate(model, loss_fn, full, targets)
                           for _ in range(3)]
                eval_probe = {
                    "step": step,
                    "generator_before": before,
                    "generator_after": generator_state(model),
                    "calls_unchanged": (
                        generator_state(model)["calls"] == before["calls"]
                    ),
                    "outputs_identical": all(
                        output == outputs[0] for output in outputs
                    ),
                    "mode_restored": model.training is True,
                }
            if step == steps - 1:
                value, final_train_logits = train_step(
                    model, loss_fn, optimizer, batches, step,
                    capture_logits=True,
                )
            else:
                value = train_step(model, loss_fn, optimizer, batches, step)
            loss_history.append(value)
            identity_stable &= (
                [id(p) for _, p in model.named_parameters()] == identities
                and [id(b) for _, b in model.named_buffers()]
                == buffer_identities
            )

        final_running = running_stats(model)
        final_eval = evaluate(model, loss_fn, full, targets)
        results.update(
            loss_history=loss_history,
            initial_loss=loss_history[0],
            final_loss=loss_history[-1],
            best_loss=min(loss_history),
            initial_eval=initial_eval,
            final_eval=final_eval,
            initial_accuracy=initial_eval[1],
            final_accuracy=final_eval[1],
            final_parameters=_model_state_values(model),
            final_running_stats=final_running,
            final_train_logits=final_train_logits,
            final_generator=generator_state(model),
            final_optimizer_state=_optimizer_state_values(optimizer),
            calls_equal_steps=(generator_state(model)["calls"] == steps),
            running_stats_advanced=(
                final_running != results["initial_running_stats"]
            ),
            identity_stable=identity_stable,
            gradients_cleared=all(p.grad is None for _, p in named),
            generator_identity_stable=(
                getattr(model, DROPOUT_NAME).generator is generator
            ),
            eval_probe=eval_probe,
            mode_restored=model.training is True,
        )
        return results
    finally:
        _close_batches(batches)
        _close_run(model, optimizer, full)


def run_resume_proof(total_steps=TOTAL_STEPS, split_step=SPLIT_STEP,
                     lr=DEFAULT_LR, directory=None):
    """Run the same schedule twice — once uninterrupted, once interrupted
    after ``split_step`` **completed** steps, checkpointed, reloaded into a
    completely fresh model/optimizer/generator set, and continued — then
    compare them exactly.

    The fresh restore target is built with ``FRESH_DROPOUT_SEED``, a
    different stream from the one that was saved, so "the load restored the
    generator" cannot be true by accident. Its parameters, buffers,
    optimizer, and generator are all different objects from the
    interrupted run's, which are released before the resume begins: the
    **checkpoint file is the only continuation boundary**.

    ``directory`` is an optional existing directory for the checkpoint; by
    default a temporary one is created and removed, so the default run
    leaves no file behind.

    Returns plain Python values only."""
    if isinstance(split_step, bool) or not isinstance(split_step, int):
        raise TypeError(
            f"split_step must be an int, got {type(split_step).__name__}"
        )
    if split_step <= 0 or split_step >= total_steps:
        raise ValueError(
            f"split_step must satisfy 0 < split_step < total_steps, got "
            f"split_step={split_step}, total_steps={total_steps}"
        )

    inputs, targets = build_dataset()
    loss_fn = build_loss()
    batches = build_batches(inputs, targets)
    full = NativeTensor.from_array(inputs)

    model_a = optimizer_a = None
    model_b = optimizer_b = model_c = optimizer_c = None
    try:
        # -- Path A: uninterrupted -------------------------------------
        model_a = build_model()
        optimizer_a = build_optimizer(model_a, lr=lr)
        initial_state_a = _model_state_values(model_a)
        initial_generator_a = generator_state(model_a)
        losses_a = []
        final_train_logits_a = None
        for step in range(total_steps):
            if step == total_steps - 1:
                value, final_train_logits_a = train_step(
                    model_a, loss_fn, optimizer_a, batches, step,
                    capture_logits=True,
                )
            else:
                value = train_step(model_a, loss_fn, optimizer_a, batches,
                                   step)
            losses_a.append(value)
        eval_a = evaluate(model_a, loss_fn, full, targets)
        parameters_a = _model_state_values(model_a)
        running_a = running_stats(model_a)
        optimizer_values_a = _optimizer_state_values(optimizer_a)
        generator_a = generator_state(model_a)
        names_a = [name for name, _ in model_a.named_parameters()]
        buffer_names_a = [name for name, _ in model_a.named_buffers()]

        # -- Path B: train to the split, then checkpoint ----------------
        model_b = build_model()
        optimizer_b = build_optimizer(model_b, lr=lr)
        start_state_b = _model_state_values(model_b)
        losses_b_prefix = [
            train_step(model_b, loss_fn, optimizer_b, batches, step)
            for step in range(split_step)
        ]
        # Saved only here — after the loop has fully completed
        # ``split_step`` steps, so the metadata can never describe a step
        # whose optimizer update did not run.
        saved_generator = generator_state(model_b)
        saved_running = running_stats(model_b)
        saved_parameters = _model_state_values(model_b)
        saved_optimizer = _optimizer_state_values(optimizer_b)

        if directory is None:
            context = tempfile.TemporaryDirectory()
        else:
            context = _ExistingDirectory(directory)
        with context as checkpoint_directory:
            path = os.path.join(checkpoint_directory,
                                "native_dropout.checkpoint.npz")
            save_native_checkpoint(
                path, model_b, optimizer=optimizer_b,
                metadata=progress_metadata(split_step, lr=lr),
            )
            # The interrupted run is released *before* the resume begins:
            # nothing below may depend on a live object from it.
            _close_run(model_b, optimizer_b)
            model_b = optimizer_b = None

            # -- Path C: a completely fresh set, from a different stream.
            model_c = build_model(dropout_seed=FRESH_DROPOUT_SEED)
            optimizer_c = build_optimizer(model_c, lr=lr)
            fresh_generator = generator_state(model_c)
            fresh_parameters = _model_state_values(model_c)
            parameter_ids_before = [id(p) for p in model_c.parameters()]
            buffer_ids_before = [id(b) for b in model_c.buffers()]
            generator_id_before = id(getattr(model_c, DROPOUT_NAME).generator)
            # Deliberately put the fresh target in eval mode before loading,
            # so the load is proved not to serialize or overwrite the flag.
            model_c.eval()
            metadata = load_native_checkpoint(
                path, model_c, optimizer=optimizer_c
            )
            mode_after_load = model_c.training
            parameter_ids_after = [id(p) for p in model_c.parameters()]
            buffer_ids_after = [id(b) for b in model_c.buffers()]
            generator_id_after = id(getattr(model_c, DROPOUT_NAME).generator)
            restored_generator = generator_state(model_c)
            restored_running = running_stats(model_c)
            restored_parameters = _model_state_values(model_c)
            restored_optimizer = _optimizer_state_values(optimizer_c)
            archive_keys = list(model_c.state_dict())
            archive_generator_keys = [
                name for name, _ in model_c.named_generators()
            ]
            # The external loop position is metadata, and it is validated
            # rather than defaulted.
            resumed_step, resumed_batch = validated_progress(
                metadata, total_steps=total_steps
            )
            # The training flag is runtime state; switch it back explicitly.
            model_c.train()

        losses_b_suffix = []
        final_train_logits_c = None
        for step in range(resumed_step, total_steps):
            if step == total_steps - 1:
                value, final_train_logits_c = train_step(
                    model_c, loss_fn, optimizer_c, batches, step,
                    capture_logits=True,
                )
            else:
                value = train_step(model_c, loss_fn, optimizer_c, batches,
                                   step)
            losses_b_suffix.append(value)
        eval_c = evaluate(model_c, loss_fn, full, targets)
        parameters_c = _model_state_values(model_c)
        running_c = running_stats(model_c)
        optimizer_values_c = _optimizer_state_values(optimizer_c)
        generator_c = generator_state(model_c)
        names_c = [name for name, _ in model_c.named_parameters()]
        buffer_names_c = [name for name, _ in model_c.named_buffers()]

        losses_b = losses_b_prefix + losses_b_suffix
        return {
            "total_steps": total_steps,
            "split_step": split_step,
            "lr": lr,
            "metadata": metadata,
            "resumed_step": resumed_step,
            "resumed_batch_index": resumed_batch,
            "resumed_step_is_split": resumed_step == split_step,
            "resumed_batch_is_scheduled": (
                resumed_batch == batch_index_for_step(split_step)
            ),
            "identical_start": initial_state_a == start_state_b,
            "fresh_target_started_different": (
                fresh_generator != saved_generator
                and fresh_parameters != saved_parameters
            ),
            "fresh_generator": fresh_generator,
            "saved_generator": saved_generator,
            "restored_generator": restored_generator,
            "generator_restored_exactly": restored_generator == saved_generator,
            "running_restored_exactly": restored_running == saved_running,
            "parameters_restored_exactly": (
                restored_parameters == saved_parameters
            ),
            "optimizer_restored_exactly": (
                restored_optimizer == saved_optimizer
            ),
            "uninterrupted_losses": losses_a,
            "resumed_losses": losses_b,
            "resumed_suffix": losses_b_suffix,
            "prefix_matches": losses_a[:split_step] == losses_b_prefix,
            "suffix_matches": losses_a[split_step:] == losses_b_suffix,
            "first_resumed_loss_matches": (
                losses_b_suffix[0] == losses_a[split_step]
            ),
            "losses_match": losses_a == losses_b,
            "final_train_logits_uninterrupted": final_train_logits_a,
            "final_train_logits_resumed": final_train_logits_c,
            "final_train_logits_match": (
                final_train_logits_a == final_train_logits_c
            ),
            "final_eval_uninterrupted": eval_a,
            "final_eval_resumed": eval_c,
            "final_eval_matches": eval_a == eval_c,
            "parameters_match": parameters_a == parameters_c,
            "running_mean_matches": (
                running_a["running_mean"] == running_c["running_mean"]
            ),
            "running_var_matches": (
                running_a["running_var"] == running_c["running_var"]
            ),
            "optimizer_state_matches": (
                optimizer_values_a == optimizer_values_c
            ),
            "generator_matches": generator_a == generator_c,
            "uninterrupted_generator": generator_a,
            "resumed_generator": generator_c,
            "initial_generator": initial_generator_a,
            "parameter_order_matches": names_a == names_c,
            "buffer_order_matches": buffer_names_a == buffer_names_c,
            "state_keys": archive_keys,
            "generator_keys": archive_generator_keys,
            "identities_preserved": (
                parameter_ids_before == parameter_ids_after
                and buffer_ids_before == buffer_ids_after
                and generator_id_before == generator_id_after
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
        _close_batches(batches)
        full.close()


def run_next_mask_proof(split_step=SPLIT_STEP, lr=DEFAULT_LR,
                        directory=None):
    """Tie the resume back to the stateless G2 Core.

    Trains a model for ``split_step`` steps, checkpoints it, reloads it
    into a **throwaway** model — separate from any resumed run, so this
    proof consumes no call the training resume depends on — and pushes the
    fixed ``PROBE_VALUES`` tensor through that model's restored
    ``NativeDropout``. The result must equal
    ``NativeTensorCore.dropout_forward`` called with the **exact restored**
    ``(seed, call_index)``, and the generator must advance from
    ``restored_calls`` to ``restored_calls + 1``.

    The module's private multiplier mask is never exposed: the comparison
    is between two *outputs*, and the Core is used purely as a reference.

    Returns plain Python values only."""
    inputs, targets = build_dataset()
    loss_fn = build_loss()
    batches = build_batches(inputs, targets)

    model = optimizer = probe_model = probe_optimizer = None
    probe = None
    try:
        model = build_model()
        optimizer = build_optimizer(model, lr=lr)
        for step in range(split_step):
            train_step(model, loss_fn, optimizer, batches, step)

        if directory is None:
            context = tempfile.TemporaryDirectory()
        else:
            context = _ExistingDirectory(directory)
        with context as checkpoint_directory:
            path = os.path.join(checkpoint_directory,
                                "native_dropout.probe.npz")
            save_native_checkpoint(
                path, model, optimizer=optimizer,
                metadata=progress_metadata(split_step, lr=lr),
            )
            probe_model = build_model(dropout_seed=FRESH_DROPOUT_SEED)
            probe_optimizer = build_optimizer(probe_model, lr=lr)
            load_native_checkpoint(path, probe_model,
                                   optimizer=probe_optimizer)

        dropout = getattr(probe_model, DROPOUT_NAME)
        restored = dropout.generator.state()
        # The Core reference: the same probability, the exact restored seed
        # and call index, and the same probe values. Stateless by
        # construction — it takes no generator and advances nothing.
        source = cpp.NativeTensorCore.from_array([list(PROBE_VALUES)])
        try:
            reference_core = source.dropout_forward(
                dropout.p, seed=restored["seed"],
                call_index=restored["calls"],
            )
            try:
                reference = reference_core.to_numpy().tolist()
            finally:
                reference_core.close()
        finally:
            source.close()
        calls_before = dropout.generator.calls

        probe = NativeTensor.from_array([list(PROBE_VALUES)])
        dropout.train()
        result_tensor = dropout(probe)
        try:
            result = result_tensor.to_numpy().tolist()
        finally:
            result_tensor.close()
        calls_after = dropout.generator.calls

        return {
            "split_step": split_step,
            "restored_seed": restored["seed"],
            "restored_calls": restored["calls"],
            "restored_calls_equal_split": restored["calls"] == split_step,
            "core_reference": reference,
            "module_result": result,
            "next_mask_matches": reference == result,
            "calls_before": calls_before,
            "calls_after": calls_after,
            "consumed_exactly_one_call": calls_after == calls_before + 1,
            "used_the_restored_index": calls_before == restored["calls"],
        }
    finally:
        if probe is not None:
            probe.close()
        _close_run(model, optimizer)
        _close_run(probe_model, probe_optimizer)
        _close_batches(batches)


def _format_losses(values, per_line=8):
    """A compact multi-line rendering of a loss sequence — never a giant
    single line, and never a truncated one."""
    rendered = [f"{value:.6f}" for value in values]
    lines = []
    for start in range(0, len(rendered), per_line):
        lines.append("  " + " ".join(rendered[start:start + per_line]))
    return "\n".join(lines)


def main():
    if not cpp.is_available():
        print("The experimental C++ backend is not built.")
        print(cpp.build_instructions())
        return

    with _LiveStorageMeter() as meter:
        baseline = meter.settled_count()

        run = run_training(eval_probe_step=SPLIT_STEP)
        proof = run_resume_proof()
        mask = run_next_mask_proof()

        final_live = meter.settled_count()

    print(
        f"native Dropout classifier: "
        f"Linear({FEATURES} -> {HIDDEN_FEATURES}) -> "
        f"BatchNorm1d({HIDDEN_FEATURES}) -> ReLU -> "
        f"Dropout(p={DROPOUT_P}) -> LayerNorm({HIDDEN_FEATURES}) -> "
        f"Linear({HIDDEN_FEATURES} -> {NUM_CLASSES})"
    )
    print(
        f"trained {run['steps']} NativeAdam steps (lr={run['lr']}) on "
        f"{SAMPLES} fixed samples in {NUM_BATCHES} fixed batches of "
        f"{BATCH_SIZE}, with NativeCrossEntropyLoss over raw logits"
    )
    print(f"parameters: {run['parameter_names']}")
    print(f"buffers:    {run['buffer_names']}  (BatchNorm running statistics)")
    print(f"generators: {run['generator_keys']}")
    print(f"batch schedule (step % {NUM_BATCHES}): {run['batch_schedule']}")
    print()
    print("uninterrupted loss sequence:")
    print(_format_losses(run["loss_history"]))
    print(f"initial training loss: {run['initial_loss']:.6f}")
    print(f"final training loss:   {run['final_loss']:.6f}   "
          f"(best {run['best_loss']:.6f}; Dropout makes the training-mode "
          f"curve genuinely noisy)")
    print(f"eval loss / accuracy:  {run['initial_eval'][0]:.6f} / "
          f"{run['initial_accuracy']:.4f}  ->  "
          f"{run['final_eval'][0]:.6f} / {run['final_accuracy']:.4f}")
    print(f"generator calls == training steps: {run['calls_equal_steps']} "
          f"({run['final_generator']['calls']})")
    print(f"BatchNorm running stats advanced:  {run['running_stats_advanced']}")

    probe = run["eval_probe"]
    print()
    print(f"evaluation probe at step {probe['step']} (3 eval passes):")
    print(f"  generator calls before/after: "
          f"{probe['generator_before']['calls']} / "
          f"{probe['generator_after']['calls']}")
    print(f"  evaluation consumed no calls: {probe['calls_unchanged']}")
    print(f"  repeated eval outputs identical: {probe['outputs_identical']}")
    print(f"  training mode restored: {probe['mode_restored']}")

    print()
    print(
        f"checkpoint resume: trained {proof['split_step']} complete steps, "
        f"saved model + buffers + optimizer + generator (format version 2) "
        f"with explicit progress metadata, released the interrupted run, "
        f"reloaded into a FRESH set, continued to {proof['total_steps']}"
    )
    print(f"  checkpoint metadata:         {proof['metadata']}")
    print(f"  archived state keys:         {proof['state_keys']}")
    print(f"  archived generator keys:     {proof['generator_keys']}")
    print(f"  fresh target began elsewhere: "
          f"{proof['fresh_target_started_different']} "
          f"(seed {proof['fresh_generator']['seed']} -> "
          f"{proof['restored_generator']['seed']})")
    print(f"  restored generator seed/calls: "
          f"{proof['restored_generator']['seed']} / "
          f"{proof['restored_generator']['calls']}")
    print(f"  resumed at step / batch:     {proof['resumed_step']} / "
          f"{proof['resumed_batch_index']} "
          f"(from metadata, not inferred)")
    print("resumed loss sequence:")
    print(_format_losses(proof["resumed_losses"]))
    print(f"  exact loss match:            {proof['losses_match']}")
    print(f"  final parameters match:      {proof['parameters_match']}")
    print(f"  running_mean match:          {proof['running_mean_matches']}")
    print(f"  running_var match:           {proof['running_var_matches']}")
    print(f"  optimizer state matches:     {proof['optimizer_state_matches']}")
    print(f"  generator state matches:     {proof['generator_matches']} "
          f"{proof['resumed_generator']}")
    print(f"  final train logits match:    {proof['final_train_logits_match']}")
    print(f"  final eval output match:     {proof['final_eval_matches']}")
    print(f"  identities preserved:        {proof['identities_preserved']}")
    print(f"  training mode unserialized:  {proof['mode_not_serialized']}")

    print()
    print("next-mask proof against the stateless G2 Core "
          "(separate throwaway load):")
    print(f"  restored seed / calls:       {mask['restored_seed']} / "
          f"{mask['restored_calls']}")
    print(f"  used the restored index:     {mask['used_the_restored_index']}")
    print(f"  next Dropout == Core output: {mask['next_mask_matches']}")
    print(f"  consumed exactly one call:   "
          f"{mask['consumed_exactly_one_call']} "
          f"({mask['calls_before']} -> {mask['calls_after']})")

    print()
    print(f"checkpoint format: tensorforge.native_checkpoint, version 2")
    print(f"captured by the checkpoint: model parameters, persistent "
          f"buffers, optimizer state, generator state + topology")
    print(f"NOT captured (carried as metadata or not at all): data-loader "
          f"position, batch order, shuffle state, scheduler state, "
          f"Python random, NumPy global RNG")
    print(f"live native storage baseline / final: {baseline} / {final_live}")

    exact = (
        run["calls_equal_steps"]
        and run["identity_stable"]
        and run["gradients_cleared"]
        and run["generator_identity_stable"]
        and probe["calls_unchanged"]
        and probe["outputs_identical"]
        and probe["mode_restored"]
        and proof["identical_start"]
        and proof["fresh_target_started_different"]
        and proof["generator_restored_exactly"]
        and proof["running_restored_exactly"]
        and proof["parameters_restored_exactly"]
        and proof["optimizer_restored_exactly"]
        and proof["resumed_step_is_split"]
        and proof["resumed_batch_is_scheduled"]
        and proof["prefix_matches"]
        and proof["first_resumed_loss_matches"]
        and proof["suffix_matches"]
        and proof["losses_match"]
        and proof["final_train_logits_match"]
        and proof["final_eval_matches"]
        and proof["parameters_match"]
        and proof["running_mean_matches"]
        and proof["running_var_matches"]
        and proof["optimizer_state_matches"]
        and proof["generator_matches"]
        and proof["identities_preserved"]
        and proof["mode_not_serialized"]
        and mask["next_mask_matches"]
        and mask["consumed_exactly_one_call"]
        and mask["used_the_restored_index"]
        and mask["restored_calls_equal_split"]
        and final_live == baseline
    )
    print(f"exact stochastic resume: {'yes' if exact else 'no'}")

    if not exact:
        raise SystemExit("resumed run diverged from the uninterrupted run")
    print("native Dropout training + exact stochastic resume ok")


if __name__ == "__main__":
    main()
