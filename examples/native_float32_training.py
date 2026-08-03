"""Integrated native float32 training and exact deterministic resume
(Advanced C++ Phase I, milestone I9).

The milestone's executable correctness proof. One deep native model —
convolution, 2-D batch normalization, pooling, flattening, two linear
layers, 1-D batch normalization, layer normalization, two Dropout layers
sharing **one** registered generator, and a fused cross-entropy loss over
raw logits, trained with ``NativeAdam`` — is run twice at **each** of the
two supported dtypes, and the interrupted-and-resumed run is proved
**bitwise identical** to the uninterrupted one::

    Conv2d(1 -> 4, 3x3, pad 1)
      -> BatchNorm2d(4) -> ReLU -> MaxPool2d(2) -> Dropout(p)
      -> Flatten
      -> Linear(36 -> 8) -> BatchNorm1d(8) -> ReLU -> LayerNorm(8)
      -> Dropout(p)
      -> Linear(8 -> 3)
      -> NativeCrossEntropyLoss  (+ NativeAdam)

**I9 adds no numerical operation and no runtime capability** — no kernel,
no C ABI export, no module, loss, metric or optimizer, no checkpoint
schema and no checkpoint version change. It assembles what I1-I8 already
shipped into one integrated proof, and it is the milestone at which
``"float32"`` leaves ``UNSUPPORTED`` and joins ``SUPPORTED_DTYPES``.

**Why this model.** A resume proof is only interesting if something would
diverge without it, and this network is built so that *every* kind of
TensorForge-owned state is load-bearing at once:

- **parameters** — both convolution tensors, both linear weight/bias
  pairs, and the three affine pairs (BatchNorm2d, BatchNorm1d,
  LayerNorm);
- **persistent buffers** — the four BatchNorm running statistics, which
  advance once per training forward and which *evaluation* reads instead
  of the batch's own statistics;
- **a registered generator** — one ``NativeGenerator``, shared by **two**
  ``NativeDropout`` layers, so the two layers consume one interleaved
  stream and the checkpoint has a real alias topology to restore, not
  just a scalar counter;
- **optimizer state** — ``NativeAdam``'s per-parameter first and second
  moments and its per-parameter step counters.

Miss any one of the four and the resumed trajectory diverges immediately.

**All four graph-owned saved-resource families are exercised, and the
claim is scoped exactly.** A *training* graph carries three of them at
once — the two Dropout multiplier masks, the MaxPool2d winner buffer, and
cross-entropy's saved probabilities — while the BatchNorm **evaluation
snapshots** exist only on an *evaluation* graph, because training-mode
BatchNorm normalizes with the batch's own statistics and takes no
snapshot at all. So the honest statement, and the one the proof makes, is
that all four families are exercised **across** the integrated run rather
than coexisting in one graph: three in every training step, the fourth in
``run_eval_snapshot_proof()``, which deliberately builds a
gradient-enabled evaluation graph, advances the live running buffers
underneath it with an ordinary training forward, and shows the earlier
graph's backward is completely unaffected.

**Two dtypes, two independent proofs, and no comparison between them.**
``run_dtype_proof("float32")`` and ``run_dtype_proof("float64")`` each
build their own host data, their own native state and their own
checkpoint, and each is compared **only against itself**. A float32 run
is *not* required to reproduce a float64 run's numbers and nothing here
asserts that it does — the contract is bitwise equality between the
interrupted-and-resumed run and the uninterrupted run *at the same
dtype* (design §18.3). The float64 path is the regression half: it runs
the same integrated architecture and must behave exactly as it always
has.

**Exactness is measured in bits.** Every comparison in this file is made
over raw IEEE-754 bit patterns — ``uint32`` views at float32, ``uint64``
views at float64 — never with a tolerance and never with ``allclose``.
The helper that produces them refuses an array whose dtype is not exactly
the run's, so "the values matched" can never quietly mean "the values
were converted and then matched".

**The data.** ``build_dataset()`` computes twelve ``1 x 6 x 6`` images
from an explicit arithmetic formula over the sample index. Every value is
a quarter or an eighth, so all of them are exactly representable in
*both* binary32 and binary64 and identical on every platform. Nothing is
generated randomly, loaded, downloaded, augmented, shuffled, or split.
Labels stay **host integers**: the native runtime has no integer dtype,
and the classification stack's strict ``int64`` target contract is
exactly why.

**The batch schedule is a pure function of the step.** The twelve samples
form three fixed batches of four, and step *s* always trains on batch
``s % 3`` — see ``batch_index_for_step``. That is the whole reason the
external loop position can be carried by a single integer, and it is
carried **explicitly**, as ordinary JSON metadata validated by
``validated_progress`` on the way back in. Checkpoint version 3 captures
TensorForge-owned state; it does not capture a data loader, a batch
order, a shuffle state, a scheduler, Python's ``random``, or NumPy's
global RNG, and this example does not pretend otherwise. Reproducibility
is exact **for the state TensorForge captures**.

**The proof, in five stages, per dtype.**

1. ``run_uninterrupted()`` runs the whole fixed schedule and records
   everything: every loss, the gradients produced at the split step, every
   parameter, every buffer, every Adam moment and counter, the generator
   state and alias topology, the final training logits, the final
   predictions, and the final evaluation output.
2. ``run_resume_proof()`` runs the same schedule interrupted after
   ``SPLIT_STEP`` **completed** steps, saves one version-3 checkpoint,
   releases the interrupted run entirely, builds a **completely fresh**
   model/optimizer/generator set from **deliberately different seeds**
   (so a load that restored nothing could not possibly pass), proves the
   fresh state differs from the saved state before loading, loads, and
   continues to the end.
3. ``run_next_mask_proof()`` proves the *next stochastic event* after the
   resume is equal too: the same registered Dropout path in both final
   models, an all-ones probe so the output *is* the multiplier mask,
   compared bit for bit, shown non-degenerate, consuming exactly one call
   in each, and observed through the **shared alias path** so the second
   Dropout is proved to see the same advanced generator object.
4. ``run_eval_snapshot_proof()`` exercises the fourth graph-owned
   resource family and proves the evaluation graph is independent of the
   live buffers it was built from.
5. Every recorded value is compared between the two runs by exact bit
   equality.

Lifetime is explicit throughout. Each step builds a completely fresh
graph and closes its logits and loss after the one-shot ``backward()``
has released that graph — and with it both Dropout masks, the MaxPool2d
winners, and cross-entropy's saved probabilities — so nothing accumulates
between steps; every run closes its parameters, buffers, optimizer, and
data tensors on the way out; and ``main()`` reports the native
live-storage count before and after the whole workflow, which must return
exactly to its baseline. Checkpoints live in a temporary directory that
is removed automatically; nothing is left behind.

This is an integration proof on one fixed task — not a benchmark, not a
generalization claim, and **no performance is claimed at either dtype**.
It needs the experimental C++ backend to be built — run:

    uv run python examples/native_float32_training.py

Every public helper that represents a completed run returns plain Python
values only (never a live native tensor, parameter, generator, model, or
optimizer), so the tests can import and verify them; ``main()`` prints
them.
"""

import gc
import os
import tempfile

import numpy as np

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchNorm1d,
    NativeBatchNorm2d,
    NativeConv2d,
    NativeCrossEntropyLoss,
    NativeDropout,
    NativeFlatten,
    NativeGenerator,
    NativeLayerNorm,
    NativeLinear,
    NativeMaxPool2d,
    NativeModule,
    NativeReLU,
    NativeTensor,
    load_native_checkpoint,
    native_accuracy,
    save_native_checkpoint,
)

# --------------------------------------------------------------------------
# The fixed architecture
# --------------------------------------------------------------------------

CHANNELS = 1
HEIGHT = 6
WIDTH = 6
CONV_CHANNELS = 4
KERNEL_SIZE = 3
PADDING = 1
POOL_SIZE = 2
POOLED_HEIGHT = HEIGHT // POOL_SIZE
POOLED_WIDTH = WIDTH // POOL_SIZE
# Conv2d keeps the spatial size (3x3 kernel, padding 1), pooling halves it.
POOLED_FEATURES = CONV_CHANNELS * POOLED_HEIGHT * POOLED_WIDTH   # 36
HIDDEN_FEATURES = 8
NUM_CLASSES = 3

SAMPLES = 12
BATCH_SIZE = 4
NUM_BATCHES = SAMPLES // BATCH_SIZE          # 3

# The two dtypes the native runtime supports, proved independently and
# never against each other. float64 is first because it is the default and
# the regression half of the proof.
RUN_DTYPES = ("float64", "float32")

# Fixed initialization seeds. Each layer draws from its own *local* seeded
# NumPy generator, so nothing here touches a global RNG.
CONV_SEED = 0
HIDDEN_SEED = 1
OUTPUT_SEED = 2
GENERATOR_SEED = 20260802
DROPOUT_P = 0.25

# Deliberately different seeds for the fresh restore target, so a load that
# failed to restore parameters *or* the random stream could not pass by
# accident. Every one of them differs from its counterpart above.
FRESH_CONV_SEED = 7001
FRESH_HIDDEN_SEED = 7002
FRESH_OUTPUT_SEED = 7003
FRESH_GENERATOR_SEED = 999999

TOTAL_STEPS = 12
# Neither the first step nor the last, and not a multiple of NUM_BATCHES:
# the resume lands mid-cycle in the batch schedule, so a loop that
# restarted the schedule at batch 0 would diverge. That is exactly what the
# explicit progress metadata prevents.
SPLIT_STEP = 5
DEFAULT_LR = 0.05

# The canonical child-module names, in registration (execution) order.
CONV_NAME = "conv"
BATCH_NORM_2D_NAME = "batch_norm2d"
POOL_NAME = "pool"
CONV_DROPOUT_NAME = "conv_dropout"
HIDDEN_NAME = "hidden"
BATCH_NORM_1D_NAME = "batch_norm1d"
LAYER_NORM_NAME = "layer_norm"
DENSE_DROPOUT_NAME = "dense_dropout"
OUTPUT_NAME = "output"

# The two registered generator paths. Both resolve to the **same** object;
# the first is the canonical one (the traversal reaches it first) and the
# second is its alias. This is the topology the checkpoint records and a
# load re-validates.
CANONICAL_GENERATOR_KEY = f"{CONV_DROPOUT_NAME}.generator"
ALIAS_GENERATOR_KEY = f"{DENSE_DROPOUT_NAME}.generator"
EXPECTED_GENERATOR_ALIASES = {
    CANONICAL_GENERATOR_KEY: CANONICAL_GENERATOR_KEY,
    ALIAS_GENERATOR_KEY: CANONICAL_GENERATOR_KEY,
}
# Two Dropout layers, both in training-mode traversal, so one training
# forward consumes exactly two generator calls.
DROPOUT_CALLS_PER_STEP = 2

# The exact keys the progress metadata carries, and nothing else.
PROGRESS_FIELDS = ("training_step", "next_batch_index", "run_dtype")

# The host NumPy type each run dtype physically is, and the unsigned
# integer type its raw IEEE-754 bits are read through. Two small explicit
# tables, used only by this example's *reporting* helpers — the runtime has
# its own single dtype authority and this is not a second one.
_HOST_DTYPES = {"float64": np.float64, "float32": np.float32}
_BIT_DTYPES = {"float64": np.uint64, "float32": np.uint32}


# --------------------------------------------------------------------------
# Bit-exact comparison helpers
# --------------------------------------------------------------------------


def bits(array, dtype):
    """``array``'s raw IEEE-754 bit patterns as a flat list of Python ints.

    The **only** comparison mechanism this example uses for numeric values:
    a ``uint32`` view at float32 and a ``uint64`` view at float64, never a
    tolerance and never ``np.allclose``. Bits distinguish ``+0.0`` from
    ``-0.0`` and never call two different values equal, which is what
    "exact" has to mean in a resume proof.

    ``array``'s dtype must be **exactly** the run's — a mismatch raises
    rather than converting. That strictness is the point: a helper that
    quietly accepted a float64 array for a float32 run could report a
    match that only existed after a conversion this runtime does not
    perform."""
    expected = _HOST_DTYPES[dtype]
    array = np.asarray(array)
    if array.dtype != expected:
        raise TypeError(
            f"expected a {expected.__name__} array for a {dtype} run, got "
            f"{array.dtype}; nothing in this proof converts between widths"
        )
    return np.ascontiguousarray(array).reshape(-1).view(
        _BIT_DTYPES[dtype]).tolist()


def tensor_bits(tensor, dtype):
    """A live native tensor's values as raw bits, through the explicit
    public ``to_numpy()`` boundary. Materializes a fresh host array and
    mutates nothing."""
    if tensor.dtype != dtype:
        raise TypeError(
            f"expected a {dtype} tensor, got {tensor.dtype}"
        )
    return bits(tensor.to_numpy(), dtype)


# --------------------------------------------------------------------------
# The fixed task
# --------------------------------------------------------------------------


def build_dataset():
    """The fixed task as plain host data: ``(images, targets)``, where
    ``images`` is a ``12 x 1 x 6 x 6`` nested list and ``targets`` is the
    list of integer class labels.

    Computed from an explicit formula over the sample index rather than
    stored as literals, so the structure is visible: sample *i* belongs to
    class ``i % 3`` and sits at position ``i // 3``; every third row
    starting at the class index carries the strong positive signal
    ``1.0 + offset``, the column at ``(class + 1) % 6`` carries
    ``-0.75 + offset``, and the background is ``offset - 0.375``, where
    ``offset`` is one of ``0.0, 0.25, 0.5, 0.75``. Position varies within
    every class, so no single pixel threshold separates them.

    **Every value is a quarter or an eighth**, and therefore exactly
    representable in binary32 *and* binary64 — which is why the same
    nested list can seed both runs without either one being a rounded
    version of the other. Nothing consults the clock, a random source, the
    filesystem, or the network, and repeated calls return equal,
    independent lists."""
    images = []
    targets = []
    for index in range(SAMPLES):
        label = index % NUM_CLASSES
        offset = (index // NUM_CLASSES) / 4.0     # 0.0, 0.25, 0.5, 0.75
        plane = [[offset - 0.375] * WIDTH for _ in range(HEIGHT)]
        for row in range(label, HEIGHT, NUM_CLASSES):
            plane[row] = [1.0 + offset] * WIDTH
        column = (label + 1) % WIDTH
        for row in range(HEIGHT):
            plane[row][column] = -0.75 + offset
        images.append([plane])                    # one channel
        targets.append(label)
    return images, targets


def host_images(images, dtype):
    """``images`` as a host NumPy array physically of the run's dtype.

    This is where the run's width is chosen, **once**, on the host — a
    float32 run's data is genuinely ``np.float32`` before it ever reaches
    the native boundary. Each dtype gets its own independent array built
    from the same exactly representable literals, so neither is a narrowed
    copy of the other."""
    return np.asarray(images, dtype=_HOST_DTYPES[dtype])


def native_input(values, dtype):
    """The **public** native ingress boundary: host data in, an owning
    native tensor of the requested dtype out.

    ``NativeTensor.from_array(values, dtype=...)`` is the explicit
    host-to-native conversion boundary and it has always converted; at I9
    ``"float32"`` simply became one of the widths it can convert *to*.
    That is not a tensor cast — no native tensor changes dtype and none
    can — and the dtype is never inferred from ``values``: it is passed
    explicitly here, every time, exactly as the contract requires.

    The caller owns the result and must close it."""
    return NativeTensor.from_array(values, dtype=dtype)


def batch_index_for_step(step):
    """Which batch training step ``step`` uses — ``step % NUM_BATCHES``.

    **A pure function of the step, deliberately.** The whole external loop
    position therefore collapses to one integer, which is what makes
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


def build_batches(images, targets, dtype):
    """The fixed batch schedule as ``[(NativeTensor, [label, ...]), ...]``
    — three contiguous batches of four samples, in a fixed order.

    The caller owns every returned tensor and must close them. Labels stay
    host integers: the native runtime has no integer dtype."""
    array = host_images(images, dtype)
    batches = []
    for index in range(NUM_BATCHES):
        start = index * BATCH_SIZE
        stop = start + BATCH_SIZE
        batches.append((native_input(array[start:stop], dtype),
                        list(targets[start:stop])))
    return batches


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


class NativeFloat32Classifier(NativeModule):
    """The integrated native classifier, at one explicit dtype::

        conv          = NativeConv2d(1, 4, 3, padding=1)   # parameters
        batch_norm2d  = NativeBatchNorm2d(4)               # + buffers
        relu1         = NativeReLU()
        pool          = NativeMaxPool2d(2)                 # winner buffer
        conv_dropout  = NativeDropout(p, generator=shared) # + generator
        flatten       = NativeFlatten()
        hidden        = NativeLinear(36, 8)                # parameters
        batch_norm1d  = NativeBatchNorm1d(8)               # + buffers
        relu2         = NativeReLU()
        layer_norm    = NativeLayerNorm(8)                 # affine only
        dense_dropout = NativeDropout(p, generator=shared) # SAME generator
        output        = NativeLinear(8, 3)                 # parameters

    producing **raw logits** of shape ``(batch_size, 3)``. There is
    deliberately no softmax or log-softmax module: the fused, numerically
    stable ``NativeCrossEntropyLoss`` consumes logits directly.

    **Every state-owning child receives the run dtype explicitly** — the
    convolution, both BatchNorm shapes, the LayerNorm, and both linear
    layers. The stateless children (ReLU, MaxPool2d, Flatten, both
    Dropouts) take **no** dtype argument and must not gain one: they own no
    dtype-bearing numeric state, so an argument there would be a second
    authority that could disagree with the data. They inherit the dtype of
    whatever flows through them.

    **The two Dropout layers share one generator object.** It is
    constructed here and handed to both, which registers *the exact
    object* twice under two different paths — never a copy, never a
    re-seed. So the two layers draw from one interleaved stream, one
    training forward consumes exactly two consecutive calls, and the
    checkpoint has a genuine alias topology to record and re-validate
    rather than a single scalar counter. ``named_generators()``
    deduplicates by identity and reports one canonical entry; the archive's
    ``aliases`` map records both paths.

    Dropout sits after the pooling in the convolutional stage and after the
    layer normalization in the dense stage, which is the ordinary placement
    in both cases."""

    def __init__(self, dtype, conv_seed=CONV_SEED, hidden_seed=HIDDEN_SEED,
                 output_seed=OUTPUT_SEED, generator_seed=GENERATOR_SEED,
                 p=DROPOUT_P):
        super().__init__()
        # One generator, two registrations. Built before ``super().__init__``
        # would matter only for ordering; it is built here so that both
        # Dropout constructors adopt the identical object.
        generator = NativeGenerator(generator_seed)
        self.conv = NativeConv2d(CHANNELS, CONV_CHANNELS, KERNEL_SIZE,
                                 padding=PADDING, seed=conv_seed, dtype=dtype)
        self.batch_norm2d = NativeBatchNorm2d(CONV_CHANNELS, dtype=dtype)
        self.relu1 = NativeReLU()
        self.pool = NativeMaxPool2d(POOL_SIZE)
        self.conv_dropout = NativeDropout(p, generator=generator)
        self.flatten = NativeFlatten()
        self.hidden = NativeLinear(POOLED_FEATURES, HIDDEN_FEATURES,
                                   seed=hidden_seed, dtype=dtype)
        self.batch_norm1d = NativeBatchNorm1d(HIDDEN_FEATURES, dtype=dtype)
        self.relu2 = NativeReLU()
        self.layer_norm = NativeLayerNorm(HIDDEN_FEATURES, dtype=dtype)
        self.dense_dropout = NativeDropout(p, generator=generator)
        self.output = NativeLinear(HIDDEN_FEATURES, NUM_CLASSES,
                                   seed=output_seed, dtype=dtype)
        # Recorded for reporting only. The authority on any tensor's dtype
        # is that tensor's own storage tag, never this attribute.
        self._dtype = dtype

    @property
    def dtype(self):
        """The dtype this model's parameters and buffers were built at."""
        return self._dtype

    def forward(self, images):
        """``(N, 1, 6, 6)`` images to ``(N, 3)`` raw logits.

        The intermediates are dropped as locals — the autograd graph holds
        what backward needs (both Dropout multiplier masks, the MaxPool2d
        winner buffer, and, once the loss is taken, cross-entropy's saved
        probabilities) and releases all of it at once."""
        h = self.conv(images)
        h = self.batch_norm2d(h)
        h = self.relu1(h)
        h = self.pool(h)
        h = self.conv_dropout(h)
        h = self.flatten(h)
        h = self.hidden(h)
        h = self.batch_norm1d(h)
        h = self.relu2(h)
        h = self.layer_norm(h)
        h = self.dense_dropout(h)
        return self.output(h)


def build_model(dtype, fresh=False):
    """A freshly initialized classifier at ``dtype``.

    Deterministic: every layer draws its initialization from a *local*
    seeded generator, the normalization parameters and buffers start from
    fixed constants, and the shared Dropout generator starts from an
    explicit seed — so two independently built models start numerically
    identical and neither the global NumPy RNG nor Python's ``random`` is
    ever touched.

    ``fresh=True`` selects the **deliberately different** seed set used for
    the restore target. Every seed differs, so the fresh model's parameters
    *and* its random stream both start somewhere else, and a load that
    restored nothing could not possibly produce a matching run."""
    if fresh:
        return NativeFloat32Classifier(
            dtype, conv_seed=FRESH_CONV_SEED, hidden_seed=FRESH_HIDDEN_SEED,
            output_seed=FRESH_OUTPUT_SEED,
            generator_seed=FRESH_GENERATOR_SEED,
        )
    return NativeFloat32Classifier(dtype)


def build_loss():
    """The native classification loss, over raw logits. It takes **no**
    dtype argument and must not gain one: it is a thin delegate that
    inherits the dtype of the logits it is handed."""
    return NativeCrossEntropyLoss()


def build_optimizer(model, lr=DEFAULT_LR):
    """``NativeAdam`` over the model's trainable parameters only.

    It takes **no** dtype argument and must not gain one: it owns no dtype
    it could choose, only state that must match a parameter. Each moment
    pair is allocated at its own parameter's width. Buffers and generators
    are never handed to it."""
    return NativeAdam(model.parameters(), lr=lr)


# --------------------------------------------------------------------------
# One training step
# --------------------------------------------------------------------------


def train_step(model, loss_fn, optimizer, batches, step, dtype,
               capture_logits=False, capture_gradients=False):
    """One full training iteration at schedule position ``step``: model in
    training mode -> the step's fixed batch -> fresh graph -> raw logits ->
    fused scalar loss -> record the pre-update loss -> backward -> optional
    gradient capture -> ``NativeAdam.step()`` -> ``zero_grad()``, closing
    this step's logits and loss.

    The one-shot ``backward()`` releases the operation graph — and with it
    both Dropout multiplier masks, the MaxPool2d winner buffer, and
    cross-entropy's saved probabilities — so nothing accumulates between
    steps and no stale graph can be reused after ``step()``. Exactly
    ``DROPOUT_CALLS_PER_STEP`` generator calls are consumed, by the two
    training-mode Dropout forwards; both BatchNorm running-statistics pairs
    advance once each, in the same forward.

    ``capture_gradients=True`` records every parameter's gradient **after
    backward and before the optimizer commits**, as raw bits. That
    ordering is the whole point: gradients are not checkpointed, so the
    contractual claim is that the first resumed step *produces* the same
    gradients, and it can only be checked at the one moment they exist and
    nothing has yet consumed them.

    Returns the pre-update loss as ``(value, bit_pattern)``; with
    ``capture_logits`` and/or ``capture_gradients`` the extra records
    follow in a dict. Deliberately native throughout: the only host
    conversions are the scalar-loss inspection and the reporting captures,
    none of which is part of the training mathematics."""
    inputs, targets = batches[batch_index_for_step(step)]
    model.train()
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    captured = {}
    try:
        loss_array = loss.to_numpy()
        value = float(loss_array)
        loss_pattern = bits(loss_array, dtype)
        if capture_logits:
            captured["logits"] = tensor_bits(logits, dtype)
        loss.backward()
        if capture_gradients:
            # After backward, before step(): the one moment the gradients
            # exist and nothing has consumed them.
            captured["gradients"] = {
                name: tensor_bits(parameter.grad, dtype)
                for name, parameter in model.named_parameters()
            }
        optimizer.step()
    finally:
        loss.close()
        logits.close()
    optimizer.zero_grad()
    if captured:
        return value, loss_pattern, captured
    return value, loss_pattern


def evaluate(model, loss_fn, inputs, targets, dtype):
    """A no-update reporting pass in **evaluation mode** over the full
    dataset, as plain Python values.

    Evaluation is **state-neutral on every axis that matters here**: both
    Dropout modules return their input object and consume no generator
    call, both BatchNorm layers read their stored running statistics
    instead of the batch's own and advance nothing, and no optimizer update
    happens. It closes every native tensor it creates and restores the
    caller's previous training mode before returning, so a reporting pass
    never silently leaves the model in eval mode — and never puts a gap in
    the random stream.

    ``native_accuracy`` and the predicted classes are **reporting only**:
    both leave native memory through the explicit public ``to_numpy()``
    boundary."""
    was_training = model.training
    model.eval()
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    try:
        rows = logits.to_numpy()
        predictions = [int(index) for index in np.argmax(rows, axis=1)]
        return {
            "loss": float(loss.to_numpy()),
            "loss_bits": bits(loss.to_numpy(), dtype),
            "accuracy": native_accuracy(logits, targets),
            "predictions": predictions,
            "logit_bits": bits(rows, dtype),
        }
    finally:
        loss.close()
        logits.close()
        model.train(was_training)


# --------------------------------------------------------------------------
# External loop progress — carried explicitly, never inferred
# --------------------------------------------------------------------------


def progress_metadata(completed_steps, dtype, lr=DEFAULT_LR):
    """The external loop position as JSON-compatible checkpoint metadata.

    ``completed_steps`` is the number of **fully completed** training steps
    — a step counts only once its backward, optimizer update, and
    ``zero_grad`` have all run, so an interrupted step is never presented
    as a completed one. Because the schedule is a pure function of the
    step, ``training_step`` alone would be sufficient; ``next_batch_index``
    is stored **as well** so the archive states the position it implies and
    a load can check the two against each other instead of trusting either
    alone.

    ``run_dtype`` is this example's own record of which run wrote the file.
    It is **ordinary metadata**, not checkpoint schema: version 3 already
    declares every numeric entry's dtype authoritatively, and a load
    validates the payload against *that*, never against this string.
    Carrying it here lets the example refuse to resume a float32 run from a
    float64 archive with a clear message instead of a shape or dtype error
    from deeper down.

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
    if dtype not in _HOST_DTYPES:
        raise ValueError(
            f"run_dtype must be one of {tuple(_HOST_DTYPES)}, got {dtype!r}"
        )
    return {
        "training_step": completed_steps,
        "next_batch_index": batch_index_for_step(completed_steps),
        "run_dtype": dtype,
        "lr": lr,
    }


def validated_progress(metadata, dtype, total_steps=TOTAL_STEPS):
    """Validate loaded checkpoint metadata and return
    ``(training_step, next_batch_index)``.

    This is the honest half of "the checkpoint does not capture a data
    loader": the external position is ordinary metadata, so it is the
    *example's* job to check it, and it is checked strictly rather than
    defaulted. A missing field, a non-``int`` (``bool`` included — ``True``
    is not a step), a negative or out-of-range step, a
    ``next_batch_index`` that disagrees with ``batch_index_for_step``, or a
    ``run_dtype`` that is not this run's all raise, naming the problem.

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
    for field in ("training_step", "next_batch_index"):
        value = metadata[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"checkpoint metadata {field!r} must be an int, got "
                f"{type(value).__name__}"
            )
    if metadata["run_dtype"] != dtype:
        raise ValueError(
            f"checkpoint metadata 'run_dtype' is "
            f"{metadata['run_dtype']!r}, but this run is {dtype!r}; there "
            f"is no dtype conversion at load and none may be added"
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
    """The shared generator's complete state as a plain dict
    (``algorithm``, ``algorithm_version``, ``seed``, ``calls``), read
    through the canonical registered path.

    Reading state creates no reservation, advances no counter, and
    allocates nothing native."""
    return getattr(model, CONV_DROPOUT_NAME).generator.state()


def alias_topology(model):
    """The registered generator topology as plain values:
    ``(canonical_keys, aliases, shared)``.

    ``canonical_keys`` comes from the identity-deduplicated
    ``named_generators()`` walk, so a shared generator appears **once**.
    ``aliases`` is the complete registered-path map this example expects,
    rebuilt from a real traversal by object identity rather than by name —
    and ``shared`` is the direct identity check, which is the fact the
    whole topology rests on."""
    canonical = [(name, generator)
                 for name, generator in model.named_generators()]
    canonical_by_id = {id(generator): name for name, generator in canonical}
    aliases = {}
    for name, generator in model.named_generators(recurse=True):
        aliases[name] = canonical_by_id[id(generator)]
    # Both registered paths, resolved by identity against the canonical set.
    for path in (CANONICAL_GENERATOR_KEY, ALIAS_GENERATOR_KEY):
        module_name, _, attribute = path.partition(".")
        generator = getattr(getattr(model, module_name), attribute)
        aliases[path] = canonical_by_id[id(generator)]
    return {
        "canonical_keys": [name for name, _ in canonical],
        "aliases": aliases,
        "shared": (getattr(model, CONV_DROPOUT_NAME).generator
                   is getattr(model, DENSE_DROPOUT_NAME).generator),
    }


def model_bits(model, dtype):
    """``model.state_dict()`` as ``{name: bit_pattern}`` — every parameter
    first, then all four BatchNorm buffers, in canonical order — closing
    **every** caller-owned snapshot in a ``finally`` and returning no
    native tensor.

    Generators are deliberately absent: ``state_dict()`` is contractually
    ``{name: NativeTensor}``, and generator state is reported separately."""
    state = model.state_dict()
    try:
        return {name: tensor_bits(tensor, dtype)
                for name, tensor in state.items()}
    finally:
        for snapshot in state.values():
            snapshot.close()


def model_dtypes(model):
    """Every parameter's and buffer's dtype tag, read off the live state.
    The proof that "the model is at dtype X" is made here, from the
    storage tags themselves, never from the constructor argument."""
    state = model.state_dict()
    try:
        return {name: tensor.dtype for name, tensor in state.items()}
    finally:
        for snapshot in state.values():
            snapshot.close()


def optimizer_bits(optimizer, dtype):
    """The ``NativeAdam`` state as plain values, with both moment families
    as raw bits.

    ``state_dict()`` returns **caller-owned** ``m``/``v`` snapshots, so
    every one is closed after materialization (in a ``finally``) — a
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
            "m": [bits(tensor.to_numpy(), dtype) for tensor in state["m"]],
            "v": [bits(tensor.to_numpy(), dtype) for tensor in state["v"]],
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

    Generators are **not** closed and deliberately have no ``close()``: a
    ``NativeGenerator`` is a pure-Python value holder that owns no native
    storage — which is also why one shared by two modules needs no
    ownership bookkeeping here. ``close()`` is idempotent, and nothing
    closed here is ever returned to the caller."""
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
    live-allocation counter and this example's final claim — that the whole
    workflow returns to its starting baseline — has to be measured rather
    than asserted.

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

        Every run above closes its parameters, buffers, optimizer, and data
        tensors explicitly — that is the release mechanism, and it is what
        the baseline claim rests on. Two things are nevertheless left to the
        collector by contract, and both are reference *cycles* that
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


def _run_record(model, optimizer, dtype, full, targets, loss_fn,
                losses, loss_patterns, split_gradients, final_logit_bits):
    """Everything a completed run is compared on, as plain Python values.

    Collected in one place so the uninterrupted and resumed runs cannot
    accidentally record different things — the comparison is
    ``record_a == record_b`` over this exact structure."""
    calls_before_eval = generator_state(model)["calls"]
    final_eval = evaluate(model, loss_fn, full, targets, dtype)
    calls_after_eval = generator_state(model)["calls"]
    return {
        "losses": losses,
        "loss_bits": loss_patterns,
        "split_step_gradients": split_gradients,
        "final_train_logit_bits": final_logit_bits,
        "parameters": model_bits(model, dtype),
        "dtypes": model_dtypes(model),
        "optimizer": optimizer_bits(optimizer, dtype),
        "generator": generator_state(model),
        "topology": alias_topology(model),
        "final_eval": final_eval,
        "eval_consumed_no_call": calls_after_eval == calls_before_eval,
    }


# --------------------------------------------------------------------------
# The runs
# --------------------------------------------------------------------------


def run_uninterrupted(dtype, total_steps=TOTAL_STEPS, split_step=SPLIT_STEP,
                      lr=DEFAULT_LR):
    """Build a fresh model/optimizer/generator set at ``dtype``, run the
    complete fixed schedule, and return the run's record.

    Everything the run creates — the batch tensors, the full-dataset
    tensor, the model's parameters and buffers, and the optimizer — is
    closed before returning, success or failure; the caller receives Python
    values only."""
    images, targets = build_dataset()
    array = host_images(images, dtype)
    loss_fn = build_loss()
    batches = build_batches(images, targets, dtype)
    full = native_input(array, dtype)
    model = build_model(dtype)
    optimizer = build_optimizer(model, lr=lr)
    try:
        initial_generator = generator_state(model)
        initial_parameters = model_bits(model, dtype)
        initial_eval = evaluate(model, loss_fn, full, targets, dtype)
        losses = []
        loss_patterns = []
        split_gradients = None
        final_logit_bits = None
        for step in range(total_steps):
            capture_logits = step == total_steps - 1
            capture_gradients = step == split_step
            result = train_step(
                model, loss_fn, optimizer, batches, step, dtype,
                capture_logits=capture_logits,
                capture_gradients=capture_gradients,
            )
            if capture_logits or capture_gradients:
                value, pattern, captured = result
                if capture_gradients:
                    split_gradients = captured["gradients"]
                if capture_logits:
                    final_logit_bits = captured["logits"]
            else:
                value, pattern = result
            losses.append(value)
            loss_patterns.append(pattern)
        record = _run_record(model, optimizer, dtype, full, targets, loss_fn,
                             losses, loss_patterns, split_gradients,
                             final_logit_bits)
        record.update(
            dtype=dtype,
            steps=total_steps,
            lr=optimizer.lr,
            initial_generator=initial_generator,
            initial_eval=initial_eval,
            parameters_changed=record["parameters"] != initial_parameters,
            calls_equal_expected=(
                record["generator"]["calls"]
                == total_steps * DROPOUT_CALLS_PER_STEP
            ),
            gradients_cleared=all(parameter.grad is None
                                  for parameter in model.parameters()),
        )
        return record
    finally:
        _close_batches(batches)
        _close_run(model, optimizer, full)


def run_dtype_proof(dtype, total_steps=TOTAL_STEPS, split_step=SPLIT_STEP,
                    lr=DEFAULT_LR, directory=None):
    """The complete integrated proof at one dtype.

    Runs the fixed schedule twice — once uninterrupted, once interrupted
    after ``split_step`` **completed** steps, saved to one version-3
    checkpoint, reloaded into a **completely fresh** model/optimizer/
    generator set built from different seeds, and continued to the end —
    then compares the two by exact bit equality, proves the next Dropout
    mask is equal, and proves the evaluation graph's independence.

    The interrupted run is released **before** the resume begins, so the
    **checkpoint file is the only continuation boundary**. The fresh
    destination shares no parameter, buffer, moment, generator, graph
    resource, native storage, or Python wrapper with either earlier run;
    only the architecture and the required alias topology match.

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

    images, targets = build_dataset()
    array = host_images(images, dtype)
    loss_fn = build_loss()
    batches = build_batches(images, targets, dtype)
    full = native_input(array, dtype)

    uninterrupted = run_uninterrupted(dtype, total_steps=total_steps,
                                      split_step=split_step, lr=lr)

    # The uninterrupted run above closed itself, so it is rebuilt here to
    # stay live beside the resumed one — the next-mask and evaluation
    # proofs need both final models at once. It is the *same* deterministic
    # schedule, and its record is asserted identical to the one just taken,
    # which is itself a determinism check that costs nothing.
    model_a = optimizer_a = None
    model_b = optimizer_b = model_c = optimizer_c = None
    try:
        model_a = build_model(dtype)
        optimizer_a = build_optimizer(model_a, lr=lr)
        losses_a = []
        patterns_a = []
        gradients_a = None
        logits_a = None
        for step in range(total_steps):
            capture_logits = step == total_steps - 1
            capture_gradients = step == split_step
            result = train_step(
                model_a, loss_fn, optimizer_a, batches, step, dtype,
                capture_logits=capture_logits,
                capture_gradients=capture_gradients,
            )
            if capture_logits or capture_gradients:
                value, pattern, captured = result
                if capture_gradients:
                    gradients_a = captured["gradients"]
                if capture_logits:
                    logits_a = captured["logits"]
            else:
                value, pattern = result
            losses_a.append(value)
            patterns_a.append(pattern)
        record_a = _run_record(model_a, optimizer_a, dtype, full, targets,
                               loss_fn, losses_a, patterns_a, gradients_a,
                               logits_a)

        # -- Path B: train to the split, then checkpoint ----------------
        model_b = build_model(dtype)
        optimizer_b = build_optimizer(model_b, lr=lr)
        losses_b_prefix = []
        patterns_b_prefix = []
        for step in range(split_step):
            value, pattern = train_step(model_b, loss_fn, optimizer_b,
                                        batches, step, dtype)
            losses_b_prefix.append(value)
            patterns_b_prefix.append(pattern)
        # Read only here — after the loop has fully completed ``split_step``
        # steps, so the metadata can never describe a step whose optimizer
        # update did not run.
        saved_parameters = model_bits(model_b, dtype)
        saved_optimizer = optimizer_bits(optimizer_b, dtype)
        saved_generator = generator_state(model_b)
        saved_topology = alias_topology(model_b)

        if directory is None:
            context = tempfile.TemporaryDirectory()
        else:
            context = _ExistingDirectory(directory)
        with context as checkpoint_directory:
            path = os.path.join(checkpoint_directory,
                                f"native_{dtype}.checkpoint.npz")
            save_native_checkpoint(
                path, model_b, optimizer=optimizer_b,
                metadata=progress_metadata(split_step, dtype, lr=lr),
            )
            # The interrupted run is released *before* the resume begins:
            # nothing below may depend on a live object from it.
            _close_run(model_b, optimizer_b)
            model_b = optimizer_b = None

            # -- Path C: a completely fresh set, from different seeds ----
            model_c = build_model(dtype, fresh=True)
            optimizer_c = build_optimizer(model_c, lr=lr)
            fresh_parameters = model_bits(model_c, dtype)
            fresh_generator = generator_state(model_c)
            parameter_ids_before = [id(p) for p in model_c.parameters()]
            buffer_ids_before = [id(b) for b in model_c.buffers()]
            generator_id_before = id(
                getattr(model_c, CONV_DROPOUT_NAME).generator)
            # Deliberately put the fresh target in eval mode before loading,
            # so the load is proved not to serialize or overwrite the flag.
            model_c.eval()
            metadata = load_native_checkpoint(path, model_c,
                                              optimizer=optimizer_c)
            mode_after_load = model_c.training
            parameter_ids_after = [id(p) for p in model_c.parameters()]
            buffer_ids_after = [id(b) for b in model_c.buffers()]
            generator_id_after = id(
                getattr(model_c, CONV_DROPOUT_NAME).generator)
            restored_parameters = model_bits(model_c, dtype)
            restored_optimizer = optimizer_bits(optimizer_c, dtype)
            restored_generator = generator_state(model_c)
            restored_topology = alias_topology(model_c)
            # The external loop position is metadata, and it is validated
            # rather than defaulted.
            resumed_step, resumed_batch = validated_progress(
                metadata, dtype, total_steps=total_steps
            )
            # The training flag is runtime state; switch it back explicitly.
            model_c.train()

        losses_c = []
        patterns_c = []
        gradients_c = None
        logits_c = None
        for step in range(resumed_step, total_steps):
            capture_logits = step == total_steps - 1
            capture_gradients = step == split_step   # the FIRST resumed step
            result = train_step(
                model_c, loss_fn, optimizer_c, batches, step, dtype,
                capture_logits=capture_logits,
                capture_gradients=capture_gradients,
            )
            if capture_logits or capture_gradients:
                value, pattern, captured = result
                if capture_gradients:
                    gradients_c = captured["gradients"]
                if capture_logits:
                    logits_c = captured["logits"]
            else:
                value, pattern = result
            losses_c.append(value)
            patterns_c.append(pattern)
        record_c = _run_record(model_c, optimizer_c, dtype, full, targets,
                               loss_fn, losses_b_prefix + losses_c,
                               patterns_b_prefix + patterns_c, gradients_c,
                               logits_c)

        # -- The next stochastic event, and the evaluation graph ---------
        mask = run_next_mask_proof(model_a, model_c, dtype)
        snapshots = run_eval_snapshot_proof(model_a, model_c, loss_fn, full,
                                            targets, dtype)

        return {
            "dtype": dtype,
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
            "metadata_validated": metadata == progress_metadata(
                split_step, dtype, lr=lr),
            # The determinism cross-check: two independently built
            # uninterrupted runs agree exactly.
            "uninterrupted_reproducible": record_a == uninterrupted_record(
                uninterrupted),
            # The fresh target genuinely started somewhere else.
            "fresh_started_different": (
                fresh_parameters != saved_parameters
                and fresh_generator != saved_generator
            ),
            "fresh_generator": fresh_generator,
            "saved_generator": saved_generator,
            "restored_generator": restored_generator,
            "load_restored_parameters": (
                restored_parameters == saved_parameters
            ),
            "load_restored_optimizer": restored_optimizer == saved_optimizer,
            "load_restored_generator": restored_generator == saved_generator,
            "load_restored_topology": restored_topology == saved_topology,
            "topology_is_expected": (
                restored_topology["aliases"] == EXPECTED_GENERATOR_ALIASES
                and restored_topology["shared"] is True
                and restored_topology["canonical_keys"]
                == [CANONICAL_GENERATOR_KEY]
            ),
            "identities_preserved": (
                parameter_ids_before == parameter_ids_after
                and buffer_ids_before == buffer_ids_after
                and generator_id_before == generator_id_after
            ),
            "mode_not_serialized": mode_after_load is False,
            # -- the equality matrix, item by item ---------------------
            "losses_match": record_a["losses"] == record_c["losses"],
            "loss_bits_match": record_a["loss_bits"] == record_c["loss_bits"],
            "prefix_matches": (
                record_a["loss_bits"][:split_step] == patterns_b_prefix
            ),
            "suffix_matches": (
                record_a["loss_bits"][split_step:] == patterns_c
            ),
            "first_resumed_loss_matches": (
                patterns_c[0] == record_a["loss_bits"][split_step]
            ),
            "split_gradients_match": (
                record_a["split_step_gradients"]
                == record_c["split_step_gradients"]
            ),
            "gradients_nonempty": bool(record_c["split_step_gradients"]),
            "parameters_match": (
                record_a["parameters"] == record_c["parameters"]
            ),
            "buffers_match": all(
                record_a["parameters"][name] == record_c["parameters"][name]
                for name in record_a["parameters"]
                if "running_" in name
            ),
            "moments_match": (
                record_a["optimizer"]["m"] == record_c["optimizer"]["m"]
                and record_a["optimizer"]["v"] == record_c["optimizer"]["v"]
            ),
            "counters_match": (
                record_a["optimizer"]["step_counts"]
                == record_c["optimizer"]["step_counts"]
            ),
            "optimizer_matches": (
                record_a["optimizer"] == record_c["optimizer"]
            ),
            "generator_matches": record_a["generator"] == record_c["generator"],
            "topology_matches": record_a["topology"] == record_c["topology"],
            "final_train_logits_match": (
                record_a["final_train_logit_bits"]
                == record_c["final_train_logit_bits"]
            ),
            "final_eval_matches": (
                record_a["final_eval"] == record_c["final_eval"]
            ),
            "predictions_match": (
                record_a["final_eval"]["predictions"]
                == record_c["final_eval"]["predictions"]
            ),
            "eval_consumed_no_call": (
                record_a["eval_consumed_no_call"] is True
                and record_c["eval_consumed_no_call"] is True
            ),
            "dtypes_match": record_a["dtypes"] == record_c["dtypes"],
            "all_state_at_run_dtype": all(
                tag == dtype for tag in record_c["dtypes"].values()
            ),
            "next_mask": mask,
            "eval_snapshots": snapshots,
            # Reporting values (not comparisons).
            "uninterrupted_losses": record_a["losses"],
            "resumed_losses": record_c["losses"],
            "final_eval": record_c["final_eval"],
            "generator": record_c["generator"],
            "optimizer_step_counts": record_c["optimizer"]["step_counts"],
            "parameter_names": [name for name, _ in
                                model_c.named_parameters()],
            "buffer_names": [name for name, _ in model_c.named_buffers()],
            "state_keys": list(record_c["parameters"]),
            "calls_equal_expected": (
                record_c["generator"]["calls"]
                == total_steps * DROPOUT_CALLS_PER_STEP
            ),
        }
    finally:
        for model, optimizer in ((model_a, optimizer_a),
                                 (model_b, optimizer_b),
                                 (model_c, optimizer_c)):
            if model is not None or optimizer is not None:
                _close_run(model, optimizer)
        _close_batches(batches)
        full.close()


def uninterrupted_record(record):
    """The comparable subset of an ``run_uninterrupted`` result — the exact
    keys ``_run_record`` produces, with the run's own reporting extras
    dropped. Used only for the determinism cross-check."""
    return {key: record[key] for key in (
        "losses", "loss_bits", "split_step_gradients",
        "final_train_logit_bits", "parameters", "dtypes", "optimizer",
        "generator", "topology", "final_eval", "eval_consumed_no_call",
    )}


def run_next_mask_proof(model_a, model_c, dtype):
    """Prove the **next stochastic event** after the resume is equal too.

    Both final models are put in training mode and the *same registered
    Dropout path* is called once in each with an identical all-ones tensor
    of the run's dtype. An all-ones input is deliberate: inverted Dropout
    multiplies by ``0`` or by ``1 / (1 - p)``, so the output **is** the
    multiplier mask, exposed without any new API and without reaching into
    the module's private saved state.

    Four things are checked: the outputs are bit-identical; the pattern is
    **non-degenerate** (it really contains both dropped and kept elements,
    so an all-kept mask cannot pass vacuously); each call consumes exactly
    one generator call; and the **shared alias path** observes the same
    advanced generator object, which is what proves the topology was
    restored rather than merely the counter.

    The models are left with equal state — both consumed exactly one call
    — so later comparisons stay valid. Every tensor created here is
    closed."""
    results = {}
    kept = 1.0 / (1.0 - DROPOUT_P)
    for label, model in (("uninterrupted", model_a), ("resumed", model_c)):
        dropout = getattr(model, CONV_DROPOUT_NAME)
        alias = getattr(model, DENSE_DROPOUT_NAME)
        before = dropout.generator.state()
        probe = native_input(
            np.ones((BATCH_SIZE, HIDDEN_FEATURES), dtype=_HOST_DTYPES[dtype]),
            dtype,
        )
        was_training = model.training
        model.train()
        output = dropout(probe)
        try:
            values = output.to_numpy()
            pattern = bits(values, dtype)
            dropped = int(np.count_nonzero(values == 0.0))
            results[label] = {
                "mask_bits": pattern,
                "dtype": output.dtype,
                "dropped": dropped,
                "kept": int(values.size - dropped),
                "non_degenerate": 0 < dropped < values.size,
                "kept_value_is_inverted_scale": bool(
                    np.all(values[values != 0.0]
                           == _HOST_DTYPES[dtype](kept))
                ),
                "calls_before": before["calls"],
                "calls_after": dropout.generator.calls,
                "consumed_exactly_one_call": (
                    dropout.generator.calls == before["calls"] + 1
                ),
                # The alias path is the same object and therefore sees the
                # same advanced counter — identity, not a copied value.
                "alias_is_same_object": alias.generator is dropout.generator,
                "alias_sees_advanced_calls": (
                    alias.generator.calls == before["calls"] + 1
                ),
            }
        finally:
            output.close()
            probe.close()
            model.train(was_training)
    return {
        "uninterrupted": results["uninterrupted"],
        "resumed": results["resumed"],
        "mask_bits_match": (results["uninterrupted"]["mask_bits"]
                            == results["resumed"]["mask_bits"]),
        "calls_match": (results["uninterrupted"]["calls_after"]
                        == results["resumed"]["calls_after"]),
        "both_non_degenerate": (results["uninterrupted"]["non_degenerate"]
                                and results["resumed"]["non_degenerate"]),
        "both_consumed_one_call": (
            results["uninterrupted"]["consumed_exactly_one_call"]
            and results["resumed"]["consumed_exactly_one_call"]
        ),
        "both_aliases_shared": (
            results["uninterrupted"]["alias_is_same_object"]
            and results["resumed"]["alias_is_same_object"]
            and results["uninterrupted"]["alias_sees_advanced_calls"]
            and results["resumed"]["alias_sees_advanced_calls"]
        ),
    }


def _drain_gradients(model, dtype):
    """Record every parameter's gradient as raw bits, then clear and close
    it — returning ``{name: bit_pattern}`` and leaving every parameter with
    ``grad is None``.

    The order matters and is the documented one: ``zero_grad()`` *drops*
    the gradient object without closing it (so a caller holding a
    reference is never invalidated out from under it), which is precisely
    why this helper holds that reference itself and closes it explicitly
    afterwards. Nothing here is left to the collector."""
    recorded = {}
    grads = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        recorded[name] = tensor_bits(grad, dtype)
        grads.append(grad)
        parameter.zero_grad()
    for grad in grads:
        grad.close()
    return recorded


def run_eval_snapshot_proof(model_a, model_c, loss_fn, full, targets, dtype):
    """Exercise the **fourth** graph-owned saved-resource family and prove
    the evaluation graph is independent of the buffers it was built from.

    Training-mode BatchNorm normalizes with the batch's own statistics and
    takes no snapshot; only *evaluation* mode does, capturing independent
    owning copies of the running mean and variance as ``graph_resources``.
    So this is where those snapshots come from, and the established safe
    way to prove they are genuinely independent is to advance the live
    running buffers underneath an already-built graph and show the graph's
    backward is unchanged:

    1. build a gradient-enabled **evaluation** graph and record every
       parameter's gradient (this releases that graph);
    2. build a second, identical evaluation graph and hold it;
    3. run one ordinary **training** forward, which advances all four
       running buffers (and consumes two generator calls);
    4. run the held graph's backward and prove every gradient is bit-
       identical to step 1's — the graph answered for the forward it
       recorded, not for the buffers as they are now.

    Both models do exactly the same sequence, so their generator counters
    stay equal and the two runs remain comparable afterwards. Every tensor
    and every gradient created here is closed."""
    results = {}
    for label, model in (("uninterrupted", model_a), ("resumed", model_c)):
        was_training = model.training
        buffers_before = {name: tensor_bits(buffer, dtype)
                          for name, buffer in model.named_buffers()}

        # 1. A complete eval forward/backward, to establish the control.
        model.eval()
        logits = model(full)
        loss = loss_fn(logits, targets)
        loss.backward()
        control = _drain_gradients(model, dtype)
        loss.close()
        logits.close()

        # 2. A second identical eval graph, held open across the mutation.
        held_logits = model(full)
        held_loss = loss_fn(held_logits, targets)

        # 3. One training forward advances all four running buffers.
        model.train()
        training_logits = model(full)
        training_logits.close()
        buffers_after = {name: tensor_bits(buffer, dtype)
                         for name, buffer in model.named_buffers()}

        # 4. The held graph still answers for the forward it recorded.
        model.eval()
        held_loss.backward()
        after = _drain_gradients(model, dtype)
        held_loss.close()
        held_logits.close()
        model.train(was_training)

        results[label] = {
            "control_gradients": control,
            "gradients_after_mutation": after,
            "graph_independent_of_live_buffers": control == after,
            "buffers_actually_advanced": all(
                buffers_before[name] != buffers_after[name]
                for name in buffers_before
            ),
            "gradients_cleared": all(parameter.grad is None
                                     for parameter in model.parameters()),
        }
    return {
        "uninterrupted": results["uninterrupted"],
        "resumed": results["resumed"],
        "both_independent": (
            results["uninterrupted"]["graph_independent_of_live_buffers"]
            and results["resumed"]["graph_independent_of_live_buffers"]
        ),
        "both_advanced_buffers": (
            results["uninterrupted"]["buffers_actually_advanced"]
            and results["resumed"]["buffers_actually_advanced"]
        ),
        "gradients_match": (
            results["uninterrupted"]["control_gradients"]
            == results["resumed"]["control_gradients"]
        ),
    }


# --------------------------------------------------------------------------
# The exit gate
# --------------------------------------------------------------------------

# Every boolean the proof must satisfy, at every dtype. Listed once, by
# name, so ``main()`` reports exactly what it checks and a new claim cannot
# be added to the output without being added to the gate.
REQUIRED = (
    "resumed_step_is_split",
    "resumed_batch_is_scheduled",
    "metadata_validated",
    "uninterrupted_reproducible",
    "fresh_started_different",
    "load_restored_parameters",
    "load_restored_optimizer",
    "load_restored_generator",
    "load_restored_topology",
    "topology_is_expected",
    "identities_preserved",
    "mode_not_serialized",
    "losses_match",
    "loss_bits_match",
    "prefix_matches",
    "suffix_matches",
    "first_resumed_loss_matches",
    "split_gradients_match",
    "gradients_nonempty",
    "parameters_match",
    "buffers_match",
    "moments_match",
    "counters_match",
    "optimizer_matches",
    "generator_matches",
    "topology_matches",
    "final_train_logits_match",
    "final_eval_matches",
    "predictions_match",
    "eval_consumed_no_call",
    "dtypes_match",
    "all_state_at_run_dtype",
    "calls_equal_expected",
)


def failed_checks(proof):
    """Every required check that did not hold, by name — empty when the
    proof passed."""
    failures = [name for name in REQUIRED if proof[name] is not True]
    mask = proof["next_mask"]
    for name in ("mask_bits_match", "calls_match", "both_non_degenerate",
                 "both_consumed_one_call", "both_aliases_shared"):
        if mask[name] is not True:
            failures.append(f"next_mask.{name}")
    snapshots = proof["eval_snapshots"]
    for name in ("both_independent", "both_advanced_buffers",
                 "gradients_match"):
        if snapshots[name] is not True:
            failures.append(f"eval_snapshots.{name}")
    return failures


def _format_losses(values, per_line=6):
    """A compact multi-line rendering of a loss sequence — never a giant
    single line, and never a truncated one.

    Printed to eight decimals rather than six on purpose: at six the two
    dtypes' curves happen to render identically on this task, which would
    read as though one were being compared to the other. They are not, and
    the extra digits make the difference visible."""
    rendered = [f"{value:.8f}" for value in values]
    lines = []
    for start in range(0, len(rendered), per_line):
        lines.append("  " + " ".join(rendered[start:start + per_line]))
    return "\n".join(lines)


def _report(proof):
    dtype = proof["dtype"]
    print()
    print("=" * 72)
    print(f"run dtype: {dtype}   (proved only against itself — a float32 "
          f"run is never compared to a float64 one)")
    print("=" * 72)
    print(f"parameters: {proof['parameter_names']}")
    print(f"buffers:    {proof['buffer_names']}")
    print(f"generators: canonical "
          f"{proof['next_mask']['resumed']['alias_is_same_object'] and '1' or '?'}"
          f" object under 2 registered paths "
          f"{sorted(EXPECTED_GENERATOR_ALIASES)}")
    print("uninterrupted loss sequence:")
    print(_format_losses(proof["uninterrupted_losses"]))
    print("resumed loss sequence:")
    print(_format_losses(proof["resumed_losses"]))
    print(f"  resumed at step / batch:     {proof['resumed_step']} / "
          f"{proof['resumed_batch_index']} (from validated metadata)")
    print(f"  checkpoint metadata:         {proof['metadata']}")
    print(f"  fresh target began elsewhere: {proof['fresh_started_different']}"
          f" (seed {proof['fresh_generator']['seed']} -> "
          f"{proof['restored_generator']['seed']})")
    print(f"  every loss bit-identical:    {proof['loss_bits_match']}")
    print(f"  first resumed step gradients: "
          f"{proof['split_gradients_match']} "
          f"(produced, not restored — gradients are not checkpointed)")
    print(f"  parameters / buffers:        {proof['parameters_match']} / "
          f"{proof['buffers_match']}")
    print(f"  Adam m, v / step counters:   {proof['moments_match']} / "
          f"{proof['counters_match']} {proof['optimizer_step_counts']}")
    print(f"  generator state:             {proof['generator_matches']} "
          f"{proof['generator']}")
    print(f"  alias topology:              {proof['topology_matches']} "
          f"(expected map: {proof['topology_is_expected']})")
    print(f"  final train logits:          "
          f"{proof['final_train_logits_match']}")
    print(f"  final predictions:           {proof['predictions_match']} "
          f"{proof['final_eval']['predictions']}")
    print(f"  final eval loss / accuracy:  "
          f"{proof['final_eval']['loss']:.6f} / "
          f"{proof['final_eval']['accuracy']:.4f} "
          f"(exact match: {proof['final_eval_matches']})")
    print(f"  every value at {dtype}:  {proof['all_state_at_run_dtype']}")
    mask = proof["next_mask"]
    print(f"  next Dropout mask identical: {mask['mask_bits_match']} "
          f"(dropped {mask['resumed']['dropped']} / kept "
          f"{mask['resumed']['kept']}, non-degenerate "
          f"{mask['both_non_degenerate']}, one call each "
          f"{mask['both_consumed_one_call']}, shared alias advanced "
          f"{mask['both_aliases_shared']})")
    snapshots = proof["eval_snapshots"]
    print(f"  BatchNorm eval snapshots:    "
          f"independent {snapshots['both_independent']}, live buffers "
          f"advanced underneath {snapshots['both_advanced_buffers']}, "
          f"gradients equal {snapshots['gradients_match']}")


def main():
    if not cpp.is_available():
        print("The experimental C++ backend is not built.")
        print(cpp.build_instructions())
        return

    print("integrated native classifier "
          f"(Conv2d({CHANNELS}->{CONV_CHANNELS}, {KERNEL_SIZE}x{KERNEL_SIZE},"
          f" pad {PADDING}) -> BatchNorm2d({CONV_CHANNELS}) -> ReLU -> "
          f"MaxPool2d({POOL_SIZE}) -> Dropout(p={DROPOUT_P}) -> Flatten -> "
          f"Linear({POOLED_FEATURES}->{HIDDEN_FEATURES}) -> "
          f"BatchNorm1d({HIDDEN_FEATURES}) -> ReLU -> "
          f"LayerNorm({HIDDEN_FEATURES}) -> Dropout(p={DROPOUT_P}) -> "
          f"Linear({HIDDEN_FEATURES}->{NUM_CLASSES}))")
    print(f"NativeCrossEntropyLoss over raw logits, NativeAdam "
          f"(lr={DEFAULT_LR}), {TOTAL_STEPS} steps, interrupted after "
          f"{SPLIT_STEP}, {SAMPLES} fixed samples in {NUM_BATCHES} batches "
          f"of {BATCH_SIZE}")
    print(f"supported native dtypes: {cpp.SUPPORTED_DTYPES}   "
          f"devices: {cpp.SUPPORTED_DEVICES}   "
          f"raw-kernel dtypes: {cpp.RAW_KERNEL_DTYPES}")

    with _LiveStorageMeter() as meter:
        baseline = meter.settled_count()
        proofs = {dtype: run_dtype_proof(dtype) for dtype in RUN_DTYPES}
        final_live = meter.settled_count()

    failures = {}
    for dtype in RUN_DTYPES:
        _report(proofs[dtype])
        broken = failed_checks(proofs[dtype])
        if broken:
            failures[dtype] = broken

    print()
    print("checkpoint format: tensorforge.native_checkpoint, version 3 "
          "(every numeric entry declares its own dtype)")
    print("captured by the checkpoint: model parameters, persistent "
          "buffers, optimizer state and counters, generator state + "
          "sharing topology")
    print("NOT captured (carried as metadata or not at all): data-loader "
          "position, batch order, shuffle state, scheduler state, Python "
          "random, NumPy global RNG")
    print("comparison mechanism: raw IEEE-754 bit patterns "
          "(uint32 at float32, uint64 at float64) — no tolerance, no "
          "allclose, and no float32-versus-float64 comparison anywhere")
    print(f"live native storage baseline / final: {baseline} / {final_live}")

    if final_live != baseline:
        failures["lifecycle"] = [
            f"live native storage {baseline} -> {final_live}"
        ]

    for dtype in RUN_DTYPES:
        status = "no" if dtype in failures else "yes"
        print(f"exact deterministic resume at {dtype}: {status}")

    if failures:
        for scope, broken in failures.items():
            print(f"FAILED [{scope}]: {', '.join(broken)}")
        raise SystemExit("the resumed run diverged from the uninterrupted run")
    print("integrated native float32 and float64 training + exact "
          "deterministic resume ok")


if __name__ == "__main__":
    main()
