"""Deterministic native mini-batch training and exact resume over the
Phase-J data pipeline (Advanced C++ Phase J, milestone J6).

The first end-user program that trains a native model through
``NativeTensorDataset`` -> ``NativeBatchSampler`` -> ``NativeDataLoader``
instead of a hand-indexed array, and the milestone's executable proof that
an interrupted run resumes **exactly** — including the shuffled batch
order, which no previous native example could carry at all::

    Linear(6 -> 8)
      -> BatchNorm1d(8) -> ReLU -> Dropout(p, shared generator)
      -> Linear(8 -> 8) -> LayerNorm(8) -> Dropout(p, the SAME generator)
      -> Linear(8 -> 3)
      -> NativeCrossEntropyLoss  (+ NativeAdam)

**J6 adds no runtime capability.** No kernel, no C ABI export, no module,
loss, metric, or optimizer, no checkpoint field or version, no public
package export, and no line of ``src/``. It composes what J1-J5 already
shipped into one ordinary training program, written entirely against the
**public** experimental surface.

**What is genuinely new here.** Every native training proof before this one
— the MLP, the CNN, the classifier, the normalized regressor, the
stochastic model, and the dual-dtype float32 model — trained on a fixed,
whole-batch, hand-indexed array whose schedule was a pure function of the
step. Their honest caveat was always the same: *reproducibility is exact
for the state TensorForge captures, which is not a data loader, a shuffle
order, or an epoch counter.* This program shuffles, and it still resumes
exactly, because the loader's position travels as **ordinary caller
metadata** through the existing version-3 checkpoint. The archive did not
grow a field and its version did not move.

**The supported workflow, and its order.** Saving is
``loader.state_dict()`` first, then ``save_native_checkpoint`` with that
state nested in the caller's own ``metadata`` — with **no delivery in
between**, or the archive would describe a position the run has already
left. Restoring is ``load_native_checkpoint`` **first**, then
``loader.load_state_dict``. The two calls are on two unrelated objects and
there is **no cross-object atomicity**: if the first succeeds and the
second fails, nothing rolls back, and the documented recovery is to discard
everything and repeat both calls from the same unchanged archive.
``"training"``, ``"data_loader"``, and ``"next_step"`` are this
repository's **caller conventions**; no runtime code knows them.

**One delivered batch is one completed step.** The loader's committed
position advances **if and only if** the caller received the batch, so
``next_step`` after completed step ``k`` is ``k + 1`` and the loader's
state always describes the exact next batch. The two cannot drift by one,
which is the error every resume proof turns on.

**The interruption is genuinely mid-epoch.** Twenty-four samples in
batches of six give four batches per epoch; the run is ten steps and the
interruption lands after five, so the saved position is epoch 1, cursor 1
— three batches still owed by the *active* epoch, with two epoch
boundaries crossed across the whole run. A loop that restarted the epoch,
or that reset the sampler, would diverge on the very next batch.

**The restore target is deliberately built wrong.** Different parameter
seeds, a different generator seed, a different learning rate, a separately
constructed dataset, and a loader with a different seed, a different batch
size, a different shuffle setting, and a different position — advanced
there by really delivering batches. Every one of those differences is
proved *before* the load, so the proof cannot pass vacuously, and
``run_omitted_loader_control()`` shows the run genuinely **diverges** when
the loader restoration alone is left out.

**Two dtypes, two independent proofs, and no comparison between them.**
``run_dtype_proof("float64")`` and ``run_dtype_proof("float32")`` each
build their own host data, their own native state, and their own
checkpoint, and each is compared **only against itself**. A float32 run is
not required to reproduce a float64 run's numbers and nothing here asserts
that it does. The one thing that *is* asserted across the two is the
**batch-index sequence**, because a permutation is a pure function of
``(seed, epoch, samples)`` and carries no dtype at all.

**Exactness is measured in bits.** Every numeric comparison is over raw
IEEE-754 bit patterns — ``uint32`` views at float32, ``uint64`` at float64
— never a tolerance, never ``allclose``. ``bits()`` refuses an array whose
dtype is not exactly the run's, so "the values matched" can never quietly
mean "the values were converted and then matched".

**The data.** ``build_features()`` computes twenty-four six-feature rows
from an explicit formula over the sample index. Every value is a multiple
of one eighth, so all of them are exactly representable in *both* binary32
and binary64 and identical on every platform. Nothing is generated
randomly, loaded, downloaded, or read from a file, and no global random
stream — Python's or NumPy's — is touched anywhere in this module. Labels
stay **host integers**: the native runtime has no integer dtype.

**Ownership is explicit throughout.** A delivered feature batch is the
**caller's**, and this program closes each one at the end of its step,
along with that step's logits, loss, and gradients; every run closes its
loader, dataset, optimizer, parameters, and buffers on the way out; and
``main()`` reports the native live-storage count before and after the whole
workflow, which must return exactly to its baseline. The target array is
ordinary read-only host memory and is never closed. Checkpoints live in a
temporary directory that is removed automatically; nothing is left behind.

This is an integration proof on one fixed task — not a benchmark, not a
generalization claim, and **no timing or performance is claimed or
measured anywhere**. It needs the experimental C++ backend to be built —
run:

    uv run python examples/native_minibatch_training.py

Every helper that represents a completed proof returns plain Python values
only — never a live ``NativeTensor``, model, optimizer, loader, dataset,
sampler, or generator — so the tests can import and verify them, and
``main()`` prints them.
"""

import gc
import os
import tempfile

import numpy as np

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchNorm1d,
    NativeBatchSampler,
    NativeCrossEntropyLoss,
    NativeDataLoader,
    NativeDropout,
    NativeGenerator,
    NativeLayerNorm,
    NativeLinear,
    NativeModule,
    NativeReLU,
    NativeTensorDataset,
    load_native_checkpoint,
    native_accuracy,
    save_native_checkpoint,
)

# --------------------------------------------------------------------------
# The fixed task
# --------------------------------------------------------------------------

SAMPLES = 24
FEATURES = 6
NUM_CLASSES = 3
HIDDEN = 8

# Four batches per epoch, so an interruption can land strictly inside one.
BATCH_SIZE = 6
DROP_LAST = False
SHUFFLE = True
SAMPLER_SEED = 20260803
BATCHES_PER_EPOCH = 4          # ceil(24 / 6); asserted against the sampler

# Ten steps span two whole epochs and half of a third, so the run crosses
# two epoch boundaries. The interruption is deliberately *not* one of them:
# 5 is neither 0 nor the last step, and 5 % 4 == 1, so the saved position is
# epoch 1, cursor 1 — three batches still owed by the active epoch.
TOTAL_STEPS = 10
SPLIT_STEP = 5

DROPOUT_P = 0.25
DEFAULT_LR = 0.05

# The two dtypes the native runtime supports, proved independently and never
# against each other. float64 is first because it is the default and the
# regression half of the proof.
RUN_DTYPES = ("float64", "float32")

# Fixed initialization seeds. Each layer draws from its own *local* seeded
# generator, so nothing here touches a global RNG.
HIDDEN_SEED = 11
MIXING_SEED = 12
OUTPUT_SEED = 13
GENERATOR_SEED = 20260804

# The deliberately different set for the fresh restore target. Every seed
# differs and so does the learning rate, so a load that restored nothing
# could not possibly produce a matching run.
FRESH_HIDDEN_SEED = 9101
FRESH_MIXING_SEED = 9102
FRESH_OUTPUT_SEED = 9103
FRESH_GENERATOR_SEED = 777777
FRESH_LR = 0.011

# ...and the deliberately different loader the target is built with. All
# four of §14.2's axes move: the seed, the batch size, the shuffle setting,
# and — once ``advance_loader`` has really delivered batches — the position.
# Configuration is *adopted* from a loaded state (design §12.4), so every one
# of these is replaced by the restore rather than having to be guessed right.
FRESH_SAMPLER_SEED = 99991
FRESH_BATCH_SIZE = 4
FRESH_SHUFFLE = False
FRESH_DROP_LAST = False
FRESH_ADVANCE_BATCHES = 2

# The canonical child-module names, in registration (execution) order.
HIDDEN_NAME = "hidden"
BATCH_NORM_NAME = "batch_norm"
HIDDEN_DROPOUT_NAME = "hidden_dropout"
MIXING_NAME = "mixing"
LAYER_NORM_NAME = "layer_norm"
MIXING_DROPOUT_NAME = "mixing_dropout"
OUTPUT_NAME = "output"

# The two registered generator paths. Both resolve to the **same** object;
# the first is canonical (the traversal reaches it first) and the second is
# its alias. This is the topology the checkpoint records and a load
# re-validates.
CANONICAL_GENERATOR_KEY = f"{HIDDEN_DROPOUT_NAME}.generator"
ALIAS_GENERATOR_KEY = f"{MIXING_DROPOUT_NAME}.generator"
EXPECTED_GENERATOR_ALIASES = {
    CANONICAL_GENERATOR_KEY: CANONICAL_GENERATOR_KEY,
    ALIAS_GENERATOR_KEY: CANONICAL_GENERATOR_KEY,
}
# Two Dropout layers, both in training-mode traversal, so one training
# forward consumes exactly two generator calls.
DROPOUT_CALLS_PER_STEP = 2

# The caller-side metadata keys this repository speaks. They are
# **conventions**, not schema: no runtime code knows any of them, and a
# caller may nest the loader state anywhere under any name.
TRAINING_KEY = "training"
LOADER_KEY = "data_loader"
NEXT_STEP_KEY = "next_step"

# The host NumPy type each run dtype physically is, and the unsigned integer
# type its raw IEEE-754 bits are read through. Two small explicit tables used
# only by this example's *reporting* helpers — the runtime has its own single
# dtype authority and this is not a second one.
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
    quietly accepted a float64 array for a float32 run could report a match
    that only existed after a conversion this runtime does not perform."""
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
    """A live native tensor's values as raw bits, through the explicit public
    ``to_numpy()`` boundary. Materializes a fresh host array and mutates
    nothing."""
    if tensor.dtype != dtype:
        raise TypeError(f"expected a {dtype} tensor, got {tensor.dtype}")
    return bits(tensor.to_numpy(), dtype)


def target_facts(targets):
    """Everything a delivered target batch is contractually required to be,
    as plain Python: a fresh, independently owned, C-contiguous, read-only
    host ``int64`` array. Values included, because they are what the step
    actually trained on."""
    return {
        "dtype": str(targets.dtype),
        "shape": tuple(targets.shape),
        "values": [int(value) for value in targets],
        "c_contiguous": bool(targets.flags["C_CONTIGUOUS"]),
        "owndata": bool(targets.flags["OWNDATA"]),
        "writeable": bool(targets.flags["WRITEABLE"]),
        "is_ndarray": type(targets) is np.ndarray,
    }


def feature_facts(features, dtype):
    """Everything a delivered feature batch is contractually required to be,
    plus its raw bits. Reads the tensor and changes nothing."""
    return {
        "dtype": features.dtype,
        "shape": tuple(features.shape),
        "device": features.device,
        "contiguous": bool(features.contiguous),
        "owns_core": bool(features.owns_core),
        "requires_grad": bool(features.requires_grad),
        "bits": tensor_bits(features, dtype),
    }


# --------------------------------------------------------------------------
# The deterministic host dataset
# --------------------------------------------------------------------------


def build_features():
    """The fixed task's features as a ``24 x 6`` nested list of floats.

    Computed from an explicit formula over the sample index rather than
    stored as literals, so the structure is visible: sample *i* belongs to
    class ``i % 3`` and sits at position ``i // 3``, which contributes an
    ``offset`` of ``k / 8`` for ``k`` in ``0..7``. Column ``class`` carries
    the strong positive marker ``1.0 + offset``, column ``class + 3`` the
    weaker positive marker ``0.5 + offset``, column ``(class + 1) % 6`` the
    negative marker ``-0.75 + offset``, and every other column the
    background ``offset - 0.375``. The three marked columns are distinct for
    every class, and position varies within each class, so no single feature
    threshold separates them.

    **Every value is a multiple of one eighth**, and therefore exactly
    representable in binary32 *and* binary64 — which is why the same nested
    list can seed both runs without either one being a rounded version of
    the other. Nothing consults the clock, a random source, an environment
    variable, the filesystem, or the network, and repeated calls return
    equal values in independent containers."""
    rows = []
    for index in range(SAMPLES):
        label = index % NUM_CLASSES
        offset = (index // NUM_CLASSES) / 8.0      # 0.0, 0.125 ... 0.875
        row = [offset - 0.375] * FEATURES
        row[label] = 1.0 + offset
        row[label + NUM_CLASSES] = 0.5 + offset
        row[(label + 1) % FEATURES] = -0.75 + offset
        rows.append(row)
    return rows


def build_targets():
    """The fixed class labels as a plain list of Python ints — sample ``i``
    belongs to class ``i % 3``, so all three classes occur eight times
    each."""
    return [index % NUM_CLASSES for index in range(SAMPLES)]


def host_arrays(dtype):
    """``(features, targets)`` as host NumPy arrays, the features physically
    at the run's dtype and the targets exact ``int64``.

    This is where the run's width is chosen, **once**, on the host: a
    float32 run's data is genuinely ``np.float32`` before it ever reaches
    the dataset constructor. Each dtype gets its own independent array built
    from the same exactly representable literals, so neither is a narrowed
    copy of the other, and the two arrays hold the same logical values."""
    features = np.asarray(build_features(), dtype=_HOST_DTYPES[dtype])
    targets = np.asarray(build_targets(), dtype=np.int64)
    return features, targets


def build_dataset(dtype):
    """A fresh ``NativeTensorDataset`` at ``dtype``, from freshly built host
    arrays.

    The dtype is passed **explicitly**, every time. It is never inferred
    from the NumPy array: an omitted ``dtype`` would mean float64 whatever
    the input array is, which is the rule the whole native line keeps. The
    dataset takes unconditional copies of both arrays, so nothing this
    function's caller does afterwards can reach it — and the caller closes
    the dataset."""
    features, targets = host_arrays(dtype)
    return NativeTensorDataset(features, targets, dtype=dtype)


def build_loader(dataset, *, seed=SAMPLER_SEED, batch_size=BATCH_SIZE,
                 shuffle=SHUFFLE, drop_last=DROP_LAST):
    """``(loader, sampler)`` over ``dataset``.

    The loader takes a **sampler**, not a dataset plus six keyword
    arguments, so the composition is explicit and each configuration value
    is spelled in exactly one constructor. The caller closes the loader."""
    sampler = NativeBatchSampler(dataset, batch_size=batch_size,
                                 shuffle=shuffle, seed=seed,
                                 drop_last=drop_last)
    return NativeDataLoader(sampler), sampler


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


class NativeMiniBatchClassifier(NativeModule):
    """The compact native classifier this example trains, at one explicit
    dtype::

        hidden         = NativeLinear(6, 8)                 # parameters
        batch_norm     = NativeBatchNorm1d(8)               # + buffers
        relu           = NativeReLU()
        hidden_dropout = NativeDropout(p, generator=shared) # + generator
        mixing         = NativeLinear(8, 8)                 # parameters
        layer_norm     = NativeLayerNorm(8)                 # affine only
        mixing_dropout = NativeDropout(p, generator=shared) # SAME generator
        output         = NativeLinear(8, 3)                 # parameters

    producing **raw logits** of shape ``(batch_size, 3)``. There is
    deliberately no softmax or log-softmax module: the fused, numerically
    stable ``NativeCrossEntropyLoss`` consumes logits directly.

    It is built so that every kind of TensorForge-owned state is
    load-bearing at once — trainable parameters, persistent BatchNorm
    running statistics, a registered generator, and (through the optimizer)
    Adam's moments and step counters. Miss any one on a resume and the
    trajectory diverges immediately.

    **Every state-owning child receives the run dtype explicitly.** The
    stateless children — ReLU, both Dropouts — take **no** dtype argument
    and must not gain one: they own no dtype-bearing numeric state, so an
    argument there would be a second authority that could disagree with the
    data.

    **The two Dropout layers share one generator object.** It is constructed
    here and handed to both, which registers *the exact object* twice under
    two different paths — never a copy, never a re-seed. So the two layers
    draw from one interleaved stream, one training forward consumes exactly
    two consecutive calls, and the checkpoint has a genuine alias topology
    to record and re-validate rather than a single scalar counter.

    This class is an **example implementation detail**. It is not exported,
    is not a public module, and no milestone adds it to
    ``tensorforge.experimental``."""

    def __init__(self, dtype, hidden_seed=HIDDEN_SEED,
                 mixing_seed=MIXING_SEED, output_seed=OUTPUT_SEED,
                 generator_seed=GENERATOR_SEED, p=DROPOUT_P):
        super().__init__()
        generator = NativeGenerator(generator_seed)
        self.hidden = NativeLinear(FEATURES, HIDDEN, seed=hidden_seed,
                                   dtype=dtype)
        self.batch_norm = NativeBatchNorm1d(HIDDEN, dtype=dtype)
        self.relu = NativeReLU()
        self.hidden_dropout = NativeDropout(p, generator=generator)
        self.mixing = NativeLinear(HIDDEN, HIDDEN, seed=mixing_seed,
                                   dtype=dtype)
        self.layer_norm = NativeLayerNorm(HIDDEN, dtype=dtype)
        self.mixing_dropout = NativeDropout(p, generator=generator)
        self.output = NativeLinear(HIDDEN, NUM_CLASSES, seed=output_seed,
                                   dtype=dtype)
        # Recorded for reporting only. The authority on any tensor's dtype is
        # that tensor's own storage tag, never this attribute.
        self._dtype = dtype

    @property
    def dtype(self):
        """The dtype this model's parameters and buffers were built at."""
        return self._dtype

    def forward(self, features):
        """``(N, 6)`` features to ``(N, 3)`` raw logits.

        The intermediates are dropped as locals — the autograd graph holds
        what backward needs (both Dropout multiplier masks and, once the
        loss is taken, cross-entropy's saved probabilities) and releases all
        of it at once."""
        h = self.hidden(features)
        h = self.batch_norm(h)
        h = self.relu(h)
        h = self.hidden_dropout(h)
        h = self.mixing(h)
        h = self.layer_norm(h)
        h = self.mixing_dropout(h)
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
    the restore target."""
    if fresh:
        return NativeMiniBatchClassifier(
            dtype, hidden_seed=FRESH_HIDDEN_SEED,
            mixing_seed=FRESH_MIXING_SEED, output_seed=FRESH_OUTPUT_SEED,
            generator_seed=FRESH_GENERATOR_SEED,
        )
    return NativeMiniBatchClassifier(dtype)


def build_loss():
    """The native classification loss, over raw logits. It takes **no** dtype
    argument and must not gain one: it is a thin delegate that inherits the
    dtype of the logits it is handed."""
    return NativeCrossEntropyLoss()


def build_optimizer(model, lr=DEFAULT_LR):
    """``NativeAdam`` over the model's trainable parameters only.

    It takes **no** dtype argument and must not gain one: it owns no dtype
    it could choose, only state that must match a parameter. Buffers and
    generators are never handed to it."""
    return NativeAdam(model.parameters(), lr=lr)


# --------------------------------------------------------------------------
# Reporting helpers — plain Python values only
# --------------------------------------------------------------------------


def generator_state(model):
    """The shared generator's complete state as a plain dict (``algorithm``,
    ``algorithm_version``, ``seed``, ``calls``), read through the canonical
    registered path. Reading state creates no reservation, advances no
    counter, and allocates nothing native."""
    return getattr(model, HIDDEN_DROPOUT_NAME).generator.state()


def alias_topology(model):
    """The registered generator topology as plain values.

    ``canonical_keys`` comes from the identity-deduplicated
    ``named_generators()`` walk, so a shared generator appears **once**.
    ``aliases`` is the complete registered-path map, rebuilt from a real
    traversal by object identity rather than by name, and ``shared`` is the
    direct identity check the whole topology rests on."""
    canonical = list(model.named_generators())
    canonical_by_id = {id(generator): name for name, generator in canonical}
    aliases = {}
    for name, generator in model.named_generators(recurse=True):
        aliases[name] = canonical_by_id[id(generator)]
    for path in (CANONICAL_GENERATOR_KEY, ALIAS_GENERATOR_KEY):
        module_name, _, attribute = path.partition(".")
        generator = getattr(getattr(model, module_name), attribute)
        aliases[path] = canonical_by_id[id(generator)]
    return {
        "canonical_keys": [name for name, _ in canonical],
        "aliases": aliases,
        "shared": (getattr(model, HIDDEN_DROPOUT_NAME).generator
                   is getattr(model, MIXING_DROPOUT_NAME).generator),
    }


def model_facts(model, dtype):
    """``model.state_dict()`` as ``{name: {shape, dtype, device, bits}}`` —
    every parameter first, then the two BatchNorm buffers, in canonical
    order — closing **every** caller-owned snapshot in a ``finally`` and
    returning no native tensor."""
    state = model.state_dict()
    try:
        return {
            name: {
                "shape": tuple(tensor.shape),
                "dtype": tensor.dtype,
                "device": tensor.device,
                "bits": tensor_bits(tensor, dtype),
            }
            for name, tensor in state.items()
        }
    finally:
        for snapshot in state.values():
            snapshot.close()


def optimizer_facts(optimizer, dtype):
    """The ``NativeAdam`` state as plain values, with both moment families as
    raw bits.

    ``state_dict()`` returns **caller-owned** ``m``/``v`` snapshots, so every
    one is closed after materialization (in a ``finally``) — a reporting
    helper must never leak optimizer-state storage."""
    state = optimizer.state_dict()
    try:
        return {
            "format_version": state["format_version"],
            "optimizer": state["optimizer"],
            "lr": state["lr"],
            "betas": list(state["betas"]),
            "eps": state["eps"],
            "parameters": [
                {"shape": list(entry["shape"]), "dtype": entry["dtype"],
                 "device": entry["device"]}
                for entry in state["parameters"]
            ],
            "step_counts": list(state["step_counts"]),
            "m": [{"shape": tuple(tensor.shape), "dtype": tensor.dtype,
                   "device": tensor.device,
                   "bits": tensor_bits(tensor, dtype)}
                  for tensor in state["m"]],
            "v": [{"shape": tuple(tensor.shape), "dtype": tensor.dtype,
                   "device": tensor.device,
                   "bits": tensor_bits(tensor, dtype)}
                  for tensor in state["v"]],
        }
    finally:
        for tensor in state["m"]:
            tensor.close()
        for tensor in state["v"]:
            tensor.close()


def loader_facts(loader):
    """Everything a loader's position and configuration are, as plain Python
    — the six owned values, the two derived counts, and the complete
    ``state_dict()``. Pure: it delivers nothing and advances nothing."""
    sampler = loader.sampler
    return {
        "seed": sampler.seed,
        "shuffle": sampler.shuffle,
        "batch_size": sampler.batch_size,
        "drop_last": sampler.drop_last,
        "epoch": sampler.epoch,
        "cursor": sampler.cursor,
        "batches_per_epoch": sampler.batches_per_epoch,
        "remaining": sampler.remaining,
        "state_dict": loader.state_dict(),
        "next_batch_indices": sampler.next_batch_indices(),
    }


def evaluate(model, criterion, dataset, dtype):
    """A no-update reporting pass in **evaluation mode** over the full
    dataset, as plain Python values.

    The whole dataset is materialized through the dataset's own public
    ``feature_batch``/``target_batch`` pair rather than the loader, because
    evaluation has no position, no shuffle, and no epoch — it is a single
    fixed gather of every row in index order, and routing it through the
    loader would advance a cursor that describes *training*.

    Evaluation is **state-neutral on every axis that matters here**: both
    Dropout modules return their input and consume no generator call, the
    BatchNorm layer reads its stored running statistics instead of the
    batch's own and advances nothing, and no optimizer update happens. It
    closes every native tensor it creates and restores the caller's previous
    training mode before returning, so a reporting pass never silently
    leaves the model in eval mode — and never puts a gap in the random
    stream.

    ``native_accuracy`` and the predicted classes are **reporting only**:
    both leave native memory through the explicit public ``to_numpy()``
    boundary."""
    was_training = model.training
    indices = tuple(range(dataset.samples))
    features = dataset.feature_batch(indices)
    targets = dataset.target_batch(indices)
    model.eval()
    logits = model(features)
    loss = criterion(logits, targets)
    try:
        rows = logits.to_numpy()
        loss_array = loss.to_numpy()
        return {
            "loss": float(loss_array),
            "loss_bits": bits(loss_array, dtype),
            "logit_shape": tuple(logits.shape),
            "logit_dtype": logits.dtype,
            "logit_bits": bits(rows, dtype),
            "predictions": [int(index) for index in np.argmax(rows, axis=1)],
            "accuracy": native_accuracy(logits, targets),
            "targets": target_facts(targets),
        }
    finally:
        loss.close()
        logits.close()
        features.close()
        model.train(was_training)


def evaluation_record(model, criterion, dataset, dtype):
    """``evaluate()`` plus the two neutrality facts it claims: the persistent
    buffers did not move and the generator consumed no call."""
    buffers_before = {name: tensor_bits(buffer, dtype)
                      for name, buffer in model.named_buffers()}
    calls_before = generator_state(model)["calls"]
    result = evaluate(model, criterion, dataset, dtype)
    buffers_after = {name: tensor_bits(buffer, dtype)
                     for name, buffer in model.named_buffers()}
    result["buffers_unchanged"] = buffers_before == buffers_after
    result["consumed_no_generator_call"] = (
        generator_state(model)["calls"] == calls_before
    )
    return result


# --------------------------------------------------------------------------
# The training loop
# --------------------------------------------------------------------------


def _release_gradients(model, optimizer):
    """Clear and close every parameter gradient, in the documented order.

    ``zero_grad()`` *drops* the gradient object without closing it — so a
    caller holding a reference is never invalidated out from under them —
    which is precisely why the references are taken first here and closed
    explicitly afterwards. Nothing is left to the collector."""
    grads = [parameter.grad for parameter in model.parameters()
             if parameter.grad is not None]
    optimizer.zero_grad()
    for grad in grads:
        grad.close()


def train(model, optimizer, criterion, loader, dtype, *, start_step,
          stop_step, journal=None):
    """Run training steps ``[start_step, stop_step)`` over ``loader`` and
    return the list of plain-Python step records.

    **One delivered batch is one completed step**, and a step is complete
    only once its forward, loss, backward, optimizer update, gradient clear,
    and temporary cleanup have all run. So a caller that has run this to
    step ``k`` inclusive saves ``next_step = k + 1``, which is exactly the
    ``stop_step`` handed back here — and it is exactly what the loader's own
    position already says, because the committed cursor advances **if and
    only if** a batch was delivered.

    The iterator is created here and **closed here**. One iterator is one
    epoch: when its captured countdown is spent it raises ``StopIteration``,
    and this loop answers by creating a **new** iterator from the *same*
    loader, which continues at the sampler's canonical next-epoch position.
    Nothing resets or rebuilds the sampler and nothing increments an epoch
    or a cursor by hand; the position moves only through delivery.

    The batch's indices are recorded from the **public** pure planning API
    (``loader.sampler.next_batch_indices()``) *before* the delivery, so the
    record states what the loader was about to hand over and the delivered
    rows can be checked against it. They are deliberately not added to the
    yielded structure, which stays the plain 2-tuple the design contracts.

    Every native object this function creates is closed explicitly: the
    delivered feature batch, the step's logits and loss, and every gradient.
    The delivered target array is ordinary host memory and is never closed.
    No live native object, model, optimizer, loader, or generator appears in
    the returned records — they are plain Python throughout.

    ``journal`` is an optional list this function appends ``("deliver",
    step)`` to as each batch actually arrives, so a caller can prove the
    ordering of its own checkpoint calls against real delivery events rather
    than reconstructing it afterwards."""
    if isinstance(start_step, bool) or not isinstance(start_step, int):
        raise TypeError(
            f"start_step must be an int, got {type(start_step).__name__}")
    if isinstance(stop_step, bool) or not isinstance(stop_step, int):
        raise TypeError(
            f"stop_step must be an int, got {type(stop_step).__name__}")
    if start_step < 0 or stop_step < start_step:
        raise ValueError(
            f"require 0 <= start_step <= stop_step, got start_step="
            f"{start_step}, stop_step={stop_step}")

    sampler = loader.sampler
    records = []
    iterator = iter(loader)
    try:
        for step in range(start_step, stop_step):
            epoch_before = sampler.epoch
            cursor_before = sampler.cursor
            indices = sampler.next_batch_indices()
            try:
                features, targets = next(iterator)
            except StopIteration:
                # The epoch's captured countdown is spent. A *new* iterator
                # over the same loader continues from the canonical
                # next-epoch position the last delivery already committed.
                iterator = iter(loader)
                features, targets = next(iterator)
            if journal is not None:
                journal.append(("deliver", step))

            model.train()
            logits = model(features)
            loss = criterion(logits, targets)
            try:
                loss_array = loss.to_numpy()
                record = {
                    "step": step,
                    "epoch_before": epoch_before,
                    "cursor_before": cursor_before,
                    "indices": tuple(indices),
                    "features": feature_facts(features, dtype),
                    "targets": target_facts(targets),
                    "loss": float(loss_array),
                    "loss_bits": bits(loss_array, dtype),
                    "loss_dtype": loss.dtype,
                    "loss_shape": tuple(loss.shape),
                    "logit_bits": tensor_bits(logits, dtype),
                    "predictions": [
                        int(index)
                        for index in np.argmax(logits.to_numpy(), axis=1)
                    ],
                }
                loss.backward()
                optimizer.step()
            finally:
                loss.close()
                logits.close()
                features.close()
            _release_gradients(model, optimizer)
            record["epoch_after"] = sampler.epoch
            record["cursor_after"] = sampler.cursor
            record["generator_calls"] = generator_state(model)["calls"]
            records.append(record)
        return records
    finally:
        # Explicit, so the loader's active-iteration count is released here
        # rather than by a finalizer — and so ``load_state_dict`` is legal
        # again the moment this returns.
        iterator.close()


def advance_loader(loader, batches):
    """Deliver and immediately close ``batches`` batches, moving the loader's
    committed position through the **public** iteration path.

    Used only to give the restore target a genuinely different starting
    position than the checkpoint holds — reached by really consuming
    batches, never by assigning a private epoch or cursor field. Returns the
    index tuples that were delivered, as plain Python."""
    delivered = []
    iterator = iter(loader)
    try:
        for _ in range(batches):
            indices = loader.sampler.next_batch_indices()
            try:
                # The target array is ordinary host memory and needs no
                # close; only the feature batch owns native storage.
                features, _ = next(iterator)
            except StopIteration:
                # Same rollover rule as ``train``: one iterator is one
                # epoch, and a new one continues at the canonical
                # next-epoch position.
                iterator = iter(loader)
                features, _ = next(iterator)
            features.close()
            delivered.append(tuple(indices))
    finally:
        iterator.close()
    return delivered


# --------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------


def _close_run(model=None, optimizer=None, loader=None, dataset=None):
    """Release everything a run owns, in the established order: the loader
    (which closes its iterator and rolls back any in-flight transaction),
    then the dataset's host snapshots, then the optimizer's moment state,
    then every unique model parameter and buffer.

    Both model traversals are identity-deduplicated, so shared state closes
    exactly once. Generators are **not** closed and deliberately have no
    ``close()``: a ``NativeGenerator`` is a pure-Python value holder that
    owns no native storage. ``close()`` is idempotent everywhere, and
    nothing closed here is ever returned to a caller — in particular, a
    *delivered* batch is the caller's and no close path here can reach
    one."""
    if loader is not None:
        loader.close()
    if dataset is not None:
        dataset.close()
    if optimizer is not None:
        optimizer.close()
    if model is not None:
        for parameter in model.parameters():
            parameter.close()
        for buffer in model.buffers():
            buffer.close()


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

        Every run above closes its loader, dataset, optimizer, parameters,
        buffers, batches, and gradients explicitly — that is the release
        mechanism, and it is what the baseline claim rests on. What is left
        to the collector is a reference *cycle* refcounting alone cannot
        break: the Python-managed autograd graph holds its parents through
        backward closures. Collecting here settles that into a deterministic
        number; it releases nothing the explicit cleanup was responsible
        for."""
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


def _checkpoint_directory(directory):
    return (tempfile.TemporaryDirectory() if directory is None
            else _ExistingDirectory(directory))


# --------------------------------------------------------------------------
# The runs
# --------------------------------------------------------------------------


def _final_record(model, optimizer, criterion, loader, dataset, dtype, steps):
    """Everything a completed run is compared on, as plain Python values.

    Collected in one place so two runs cannot accidentally record different
    things — the comparison is ``record_a == record_b`` over this exact
    structure."""
    return {
        "steps": steps,
        "parameters": model_facts(model, dtype),
        "optimizer": optimizer_facts(optimizer, dtype),
        "generator": generator_state(model),
        "topology": alias_topology(model),
        "loader": loader_facts(loader),
        "dataset_identity": dataset.identity(),
        "evaluation": evaluation_record(model, criterion, dataset, dtype),
    }


def run_uninterrupted(dtype, total_steps=TOTAL_STEPS, lr=DEFAULT_LR):
    """Build a fresh dataset/sampler/loader and model/optimizer set at
    ``dtype``, train the whole schedule in one go, and return the run's
    record.

    Everything the run creates is closed before returning, success or
    failure; the caller receives plain Python values only."""
    dataset = build_dataset(dtype)
    loader = sampler = model = optimizer = None
    try:
        loader, sampler = build_loader(dataset)
        model = build_model(dtype)
        optimizer = build_optimizer(model, lr=lr)
        criterion = build_loss()
        initial = {
            "parameters": model_facts(model, dtype),
            "generator": generator_state(model),
            "optimizer": optimizer_facts(optimizer, dtype),
            "loader": loader_facts(loader),
            "buffers": {name: tensor_bits(buffer, dtype)
                        for name, buffer in model.named_buffers()},
            # An evaluation of the untrained model, over exactly the inputs
            # the final one uses. It is the control the "training actually
            # moved something" claim is measured against, and it is
            # state-neutral: eval mode consumes no generator call and
            # advances no running buffer.
            "evaluation": evaluation_record(model, criterion, dataset, dtype),
        }
        steps = train(model, optimizer, criterion, loader, dtype,
                      start_step=0, stop_step=total_steps)
        record = _final_record(model, optimizer, criterion, loader, dataset,
                               dtype, steps)
        record.update(
            dtype=dtype,
            total_steps=total_steps,
            lr=optimizer.lr,
            initial=initial,
            batches_per_epoch=sampler.batches_per_epoch,
            index_sequence=[step["indices"] for step in steps],
            position_sequence=[(step["epoch_before"], step["cursor_before"])
                               for step in steps],
            epoch_permutations=[
                sampler.epoch_permutation(epoch)
                for epoch in range(record_epochs(total_steps,
                                                 sampler.batches_per_epoch))
            ],
            gradients_cleared=all(parameter.grad is None
                                  for parameter in model.parameters()),
        )
        return record
    finally:
        _close_run(model, optimizer, loader, dataset)


def record_epochs(total_steps, batches_per_epoch):
    """How many epochs a ``total_steps`` run touches — the number whose
    permutations are actually exercised, so a proof never records an epoch
    the run never reached."""
    return (total_steps + batches_per_epoch - 1) // batches_per_epoch


def _identity_snapshot(model, optimizer, loader, dataset):
    """Object identities for every family, as ids. Used only to prove that a
    restored graph shares **nothing** with the graph that saved it, and that
    a load restores in place rather than constructing replacements."""
    return {
        "model": id(model),
        "parameters": [id(parameter) for parameter in model.parameters()],
        "buffers": [id(buffer) for buffer in model.buffers()],
        "generators": [id(generator)
                       for _, generator in model.named_generators()],
        "optimizer": id(optimizer),
        "loader": id(loader),
        "sampler": id(loader.sampler),
        "dataset": id(dataset),
    }


def _moment_ids(optimizer):
    """The ids of the optimizer's live moment tensors, read through the
    caller-owned ``state_dict()`` snapshots and closed immediately.

    Snapshot identity is not moment identity, so this records the *snapshot*
    ids only as a shape check; the identity claim that matters is asserted
    over parameters, buffers, and generators, which ``state_dict()`` does not
    copy."""
    state = optimizer.state_dict()
    try:
        return {"m": len(state["m"]), "v": len(state["v"])}
    finally:
        for tensor in state["m"]:
            tensor.close()
        for tensor in state["v"]:
            tensor.close()


def run_resume_proof(dtype, total_steps=TOTAL_STEPS, split_step=SPLIT_STEP,
                     lr=DEFAULT_LR, directory=None):
    """Train to ``split_step``, checkpoint, discard **everything**, rebuild
    an entirely fresh graph from deliberately different seeds and a
    deliberately different loader, restore, and finish the run.

    Returns plain Python values only: the prefix and suffix step records, the
    final record, and every fact the proof needs about what was saved, what
    the fresh objects looked like before the load, and what the load
    actually did."""
    # -- Path B: train to the split, then checkpoint --------------------
    dataset_b = build_dataset(dtype)
    loader_b = model_b = optimizer_b = None
    # Every native resource of the interrupted run is released by an
    # explicit ``close()`` the moment the archive is written. The emptied
    # Python objects are then held in ``retired`` for exactly as long as the
    # identity comparison below runs, and for one reason: CPython recycles
    # ``id()`` values, so "the fresh graph shares no object with the saved
    # one" would be measured against reusable addresses if the originals
    # were already collected. Holding them keeps every id unique and the
    # claim honest. **None of them is ever passed into the restored graph**,
    # none owns native storage any more, and the list is cleared on the way
    # out.
    retired = []
    journal = []
    try:
        loader_b, sampler_b = build_loader(dataset_b)
        model_b = build_model(dtype)
        optimizer_b = build_optimizer(model_b, lr=lr)
        criterion = build_loss()
        prefix = train(model_b, optimizer_b, criterion, loader_b, dtype,
                       start_step=0, stop_step=split_step, journal=journal)

        # The supported save order: read the loader state **first**, do not
        # iterate, then write the archive. The journal records both calls
        # beside the real delivery events, so "no delivery happened in
        # between" is observed rather than asserted.
        journal.append(("loader_state_dict", split_step))
        loader_state = loader_b.state_dict()
        saved = {
            "loader": loader_facts(loader_b),
            "parameters": model_facts(model_b, dtype),
            "optimizer": optimizer_facts(optimizer_b, dtype),
            "generator": generator_state(model_b),
            "topology": alias_topology(model_b),
            "identities": _identity_snapshot(model_b, optimizer_b, loader_b,
                                             dataset_b),
            "moment_counts": _moment_ids(optimizer_b),
        }
        metadata_written = {
            TRAINING_KEY: {
                NEXT_STEP_KEY: split_step,
                LOADER_KEY: loader_state,
            }
        }
        with _checkpoint_directory(directory) as checkpoint_directory:
            path = os.path.join(checkpoint_directory,
                                f"native_minibatch_{dtype}.checkpoint.npz")
            journal.append(("save_checkpoint", split_step))
            save_native_checkpoint(path, model_b, optimizer=optimizer_b,
                                   metadata=metadata_written)
            # The interrupted run is released **before** the resume begins:
            # nothing below may depend on a live object from it, and the
            # archive is the only continuation boundary.
            _close_run(model_b, optimizer_b, loader_b, dataset_b)
            retired = [model_b, optimizer_b, loader_b, dataset_b]
            model_b = optimizer_b = loader_b = dataset_b = None
            resumed = _restore_and_finish(dtype, path, total_steps,
                                          split_step, saved)
    finally:
        _close_run(model_b, optimizer_b, loader_b, dataset_b)
        retired.clear()

    resumed.update(prefix=prefix, saved=saved, journal=journal,
                   metadata_written=metadata_written, split_step=split_step,
                   total_steps=total_steps)
    return resumed


def _restore_and_finish(dtype, path, total_steps, split_step, saved,
                        restore_loader=True):
    """Build the **entirely fresh** restore target, prove it started
    somewhere else, load, and finish the run.

    ``restore_loader=False`` is the §14.2 negative control: everything else
    is identical, the loader restoration alone is skipped, and the resulting
    run must **differ**."""
    dataset_c = build_dataset(dtype)
    loader_c = model_c = optimizer_c = None
    try:
        loader_c, sampler_c = build_loader(
            dataset_c, seed=FRESH_SAMPLER_SEED, batch_size=FRESH_BATCH_SIZE,
            shuffle=FRESH_SHUFFLE, drop_last=FRESH_DROP_LAST)
        model_c = build_model(dtype, fresh=True)
        optimizer_c = build_optimizer(model_c, lr=FRESH_LR)
        criterion = build_loss()
        # A genuinely different position, reached by really delivering
        # batches through the public iteration path.
        advanced = advance_loader(loader_c, FRESH_ADVANCE_BATCHES)

        fresh = {
            "loader": loader_facts(loader_c),
            "parameters": model_facts(model_c, dtype),
            "optimizer": optimizer_facts(optimizer_c, dtype),
            "generator": generator_state(model_c),
            "identities": _identity_snapshot(model_c, optimizer_c, loader_c,
                                             dataset_c),
            "advanced_batches": advanced,
        }
        identities_before = _identity_snapshot(model_c, optimizer_c, loader_c,
                                               dataset_c)

        # -- the two calls, in the one supported order ------------------
        metadata = load_native_checkpoint(path, model_c, optimizer=optimizer_c)
        training = metadata[TRAINING_KEY]
        loaded_loader_state = training[LOADER_KEY]
        if restore_loader:
            loader_c.load_state_dict(loaded_loader_state)
        next_step = training[NEXT_STEP_KEY]

        identities_after = _identity_snapshot(model_c, optimizer_c, loader_c,
                                              dataset_c)
        restored = {
            "loader": loader_facts(loader_c),
            "parameters": model_facts(model_c, dtype),
            "optimizer": optimizer_facts(optimizer_c, dtype),
            "generator": generator_state(model_c),
            "topology": alias_topology(model_c),
        }
        suffix = train(model_c, optimizer_c, criterion, loader_c, dtype,
                       start_step=next_step, stop_step=total_steps)
        record = _final_record(model_c, optimizer_c, criterion, loader_c,
                               dataset_c, dtype, suffix)
        record.update(
            dtype=dtype,
            lr=optimizer_c.lr,
            metadata=metadata,
            next_step=next_step,
            fresh=fresh,
            restored=restored,
            restore_loader=restore_loader,
            identities_before=identities_before,
            identities_after=identities_after,
            identities_preserved=identities_before == identities_after,
            fresh_loader_differs=(
                fresh["loader"]["state_dict"]
                != saved["loader"]["state_dict"]
            ),
            fresh_next_batch_differs=(
                fresh["loader"]["next_batch_indices"]
                != saved["loader"]["next_batch_indices"]
            ),
            fresh_parameters_differ=(
                fresh["parameters"] != saved["parameters"]
            ),
            fresh_generator_differs=fresh["generator"] != saved["generator"],
            fresh_optimizer_differs=(
                fresh["optimizer"] != saved["optimizer"]
            ),
            fresh_shares_no_identity=_no_shared_identity(
                fresh["identities"], saved["identities"]),
            loader_adopted_saved_state=(
                restored["loader"]["state_dict"]
                == saved["loader"]["state_dict"]
            ),
            loader_next_batch_matches_saved=(
                restored["loader"]["next_batch_indices"]
                == saved["loader"]["next_batch_indices"]
            ),
            load_restored_parameters=(
                restored["parameters"] == saved["parameters"]
            ),
            load_restored_optimizer=(
                restored["optimizer"] == saved["optimizer"]
            ),
            load_restored_generator=(
                restored["generator"] == saved["generator"]
            ),
            load_restored_topology=restored["topology"] == saved["topology"],
            index_sequence=[step["indices"] for step in suffix],
            position_sequence=[(step["epoch_before"], step["cursor_before"])
                               for step in suffix],
        )
        return record
    finally:
        _close_run(model_c, optimizer_c, loader_c, dataset_c)


def _no_shared_identity(fresh, saved):
    """``True`` when two identity snapshots share **no** object at all, at
    any level — the executable form of "the resumed graph is genuinely
    fresh"."""
    for key, value in fresh.items():
        other = saved[key]
        if isinstance(value, list):
            if set(value) & set(other):
                return False
        elif value == other:
            return False
    return True


def run_omitted_loader_control(dtype, total_steps=TOTAL_STEPS,
                               split_step=SPLIT_STEP, lr=DEFAULT_LR,
                               directory=None):
    """The §14.2 negative control: restore the model, optimizer, and
    generators from the archive and **omit** ``loader.load_state_dict``.

    The remaining batch order must then be wrong, and the finished run must
    differ from the uninterrupted one — otherwise the positive proof could
    be passing without the loader restoration doing anything at all. The
    whole leg builds and closes its own graph and returns plain Python."""
    dataset_b = build_dataset(dtype)
    loader_b = model_b = optimizer_b = None
    retired = []
    try:
        loader_b, _sampler_b = build_loader(dataset_b)
        model_b = build_model(dtype)
        optimizer_b = build_optimizer(model_b, lr=lr)
        criterion = build_loss()
        train(model_b, optimizer_b, criterion, loader_b, dtype, start_step=0,
              stop_step=split_step)
        loader_state = loader_b.state_dict()
        saved = {
            "loader": loader_facts(loader_b),
            "parameters": model_facts(model_b, dtype),
            "optimizer": optimizer_facts(optimizer_b, dtype),
            "generator": generator_state(model_b),
            "topology": alias_topology(model_b),
            "identities": _identity_snapshot(model_b, optimizer_b, loader_b,
                                             dataset_b),
            "moment_counts": _moment_ids(optimizer_b),
        }
        with _checkpoint_directory(directory) as checkpoint_directory:
            path = os.path.join(
                checkpoint_directory,
                f"native_minibatch_{dtype}.omitted.checkpoint.npz")
            save_native_checkpoint(
                path, model_b, optimizer=optimizer_b,
                metadata={TRAINING_KEY: {NEXT_STEP_KEY: split_step,
                                         LOADER_KEY: loader_state}})
            _close_run(model_b, optimizer_b, loader_b, dataset_b)
            retired = [model_b, optimizer_b, loader_b, dataset_b]
            model_b = optimizer_b = loader_b = dataset_b = None
            return _restore_and_finish(dtype, path, total_steps, split_step,
                                       saved, restore_loader=False)
    finally:
        _close_run(model_b, optimizer_b, loader_b, dataset_b)
        retired.clear()


# --------------------------------------------------------------------------
# The proof
# --------------------------------------------------------------------------


def _training_moved_everything(uninterrupted):
    """The non-vacuity half: this task really trains, so a resume proof over
    it is not comparing two runs of nothing.

    Every claim is about *state that moved*, never about an accuracy or a
    loss threshold — a training example must not turn a numeric target into
    a gate."""
    initial = uninterrupted["initial"]
    final_optimizer = uninterrupted["optimizer"]
    losses = [step["loss_bits"] for step in uninterrupted["steps"]]
    buffers_after = {
        name: facts["bits"]
        for name, facts in uninterrupted["parameters"].items()
        if "running_" in name
    }
    return {
        "parameters_moved": (
            uninterrupted["parameters"] != initial["parameters"]
        ),
        "buffers_moved": all(
            buffers_after[name] != initial["buffers"][name]
            for name in buffers_after
        ),
        "buffer_count": len(buffers_after),
        "moments_became_nonzero": any(
            any(pattern != 0 for pattern in moment["bits"])
            for moment in final_optimizer["m"] + final_optimizer["v"]
        ),
        "step_counters_advanced": all(
            count == uninterrupted["total_steps"]
            for count in final_optimizer["step_counts"]
        ),
        "generator_calls_advanced": (
            uninterrupted["generator"]["calls"]
            == uninterrupted["total_steps"] * DROPOUT_CALLS_PER_STEP
            > initial["generator"]["calls"]
        ),
        "loss_sequence_varies": len(set(map(tuple, losses))) > 1,
        # The anti-hard-coding claim, made on the one comparison that is
        # meaningful: the **same** evaluation inputs in the **same** mode,
        # before and after training. Predictions themselves are deliberately
        # not gated — a class label may legitimately survive training — and
        # they are proved to be a real ``argmax`` of these logits instead.
        "evaluation_output_changed": (
            uninterrupted["evaluation"]["logit_bits"]
            != initial["evaluation"]["logit_bits"]
        ),
        "optimizer_state_nonempty": bool(final_optimizer["m"]),
    }


def _schedule_is_non_vacuous(uninterrupted, split_step):
    """The split and the shuffle really are what the proof needs them to be,
    proved from the run's own observed plan rather than from probability."""
    batches_per_epoch = uninterrupted["batches_per_epoch"]
    positions = uninterrupted["position_sequence"]
    permutations = uninterrupted["epoch_permutations"]
    identity = tuple(range(SAMPLES))
    return {
        "batches_per_epoch": batches_per_epoch,
        "split_is_not_zero": split_step > 0,
        "split_is_not_final": split_step < uninterrupted["total_steps"] - 1,
        "split_is_mid_epoch": split_step % batches_per_epoch != 0,
        "split_position": positions[split_step],
        "batches_left_in_active_epoch": (
            batches_per_epoch - positions[split_step][1]
        ),
        "suffix_is_multi_step": (
            uninterrupted["total_steps"] - split_step > 1
        ),
        "epoch_boundaries_crossed": len({epoch for epoch, _ in positions}) - 1,
        "shuffle_is_on": uninterrupted["loader"]["shuffle"] is True,
        "order_is_not_identity": all(order != identity
                                     for order in permutations),
        "epochs_have_distinct_orders": (
            len(set(permutations)) == len(permutations)
        ),
        "exercised_epochs": len(permutations),
    }


def run_dtype_proof(dtype, total_steps=TOTAL_STEPS, split_step=SPLIT_STEP,
                    lr=DEFAULT_LR, directory=None):
    """The complete proof at one dtype: an uninterrupted run, an
    interrupted-and-resumed run through a real version-3 archive into
    entirely fresh objects, and the omitted-loader negative control —
    compared by exact equality and returned as plain Python values."""
    if isinstance(split_step, bool) or not isinstance(split_step, int):
        raise TypeError(
            f"split_step must be an int, got {type(split_step).__name__}")
    if split_step <= 0 or split_step >= total_steps:
        raise ValueError(
            f"split_step must satisfy 0 < split_step < total_steps, got "
            f"split_step={split_step}, total_steps={total_steps}")

    uninterrupted = run_uninterrupted(dtype, total_steps=total_steps, lr=lr)
    resumed = run_resume_proof(dtype, total_steps=total_steps,
                               split_step=split_step, lr=lr,
                               directory=directory)
    omitted = run_omitted_loader_control(dtype, total_steps=total_steps,
                                         split_step=split_step, lr=lr,
                                         directory=directory)

    prefix = resumed["prefix"]
    suffix = resumed["steps"]
    combined = prefix + suffix
    reference = uninterrupted["steps"]
    journal = resumed["journal"]
    # The two checkpoint calls, and everything before them, as they really
    # happened. The last delivery must precede the snapshot, and nothing may
    # sit between the snapshot and the save.
    tail = journal[-3:]

    return {
        "dtype": dtype,
        "total_steps": total_steps,
        "split_step": split_step,
        "lr": lr,
        "next_step": resumed["next_step"],
        "next_step_is_split": resumed["next_step"] == split_step,
        "one_batch_per_step": (
            len(combined) == total_steps and len(suffix) == total_steps
            - split_step
        ),
        # -- ordering ---------------------------------------------------
        "journal_tail": tail,
        "snapshot_immediately_precedes_save": (
            tail == [("deliver", split_step - 1),
                     ("loader_state_dict", split_step),
                     ("save_checkpoint", split_step)]
        ),
        # -- the fresh target really started elsewhere ------------------
        "fresh_started_different": all((
            resumed["fresh_loader_differs"],
            resumed["fresh_next_batch_differs"],
            resumed["fresh_parameters_differ"],
            resumed["fresh_generator_differs"],
            resumed["fresh_optimizer_differs"],
        )),
        "fresh_shares_no_identity": resumed["fresh_shares_no_identity"],
        "identities_preserved": resumed["identities_preserved"],
        # -- what the two calls restored --------------------------------
        "load_restored_parameters": resumed["load_restored_parameters"],
        "load_restored_optimizer": resumed["load_restored_optimizer"],
        "load_restored_generator": resumed["load_restored_generator"],
        "load_restored_topology": resumed["load_restored_topology"],
        "loader_adopted_saved_state": resumed["loader_adopted_saved_state"],
        "loader_next_batch_matches_saved": (
            resumed["loader_next_batch_matches_saved"]
        ),
        "next_batch_after_restore_matches": (
            resumed["restored"]["loader"]["next_batch_indices"]
            == reference[split_step]["indices"]
        ),
        # -- the batch sequence -----------------------------------------
        "prefix_indices_match": (
            [step["indices"] for step in prefix]
            == [step["indices"] for step in reference[:split_step]]
        ),
        "suffix_indices_match": (
            [step["indices"] for step in suffix]
            == [step["indices"] for step in reference[split_step:]]
        ),
        "whole_index_sequence_matches": (
            [step["indices"] for step in combined]
            == uninterrupted["index_sequence"]
        ),
        "position_sequence_matches": (
            [(step["epoch_before"], step["cursor_before"])
             for step in combined]
            == uninterrupted["position_sequence"]
        ),
        "epoch_boundaries_match": (
            [(step["epoch_after"], step["cursor_after"])
             for step in combined]
            == [(step["epoch_after"], step["cursor_after"])
                for step in reference]
        ),
        # -- the values --------------------------------------------------
        "feature_batches_match": (
            [step["features"] for step in combined]
            == [step["features"] for step in reference]
        ),
        "target_batches_match": (
            [step["targets"] for step in combined]
            == [step["targets"] for step in reference]
        ),
        "loss_sequence_matches": (
            [step["loss_bits"] for step in combined]
            == [step["loss_bits"] for step in reference]
        ),
        "suffix_losses_match": (
            [step["loss_bits"] for step in suffix]
            == [step["loss_bits"] for step in reference[split_step:]]
        ),
        "logits_match": (
            [step["logit_bits"] for step in combined]
            == [step["logit_bits"] for step in reference]
        ),
        "step_predictions_match": (
            [step["predictions"] for step in combined]
            == [step["predictions"] for step in reference]
        ),
        "parameters_match": (
            resumed["parameters"] == uninterrupted["parameters"]
        ),
        "buffers_match": all(
            resumed["parameters"][name] == uninterrupted["parameters"][name]
            for name in uninterrupted["parameters"] if "running_" in name
        ),
        "optimizer_matches": resumed["optimizer"] == uninterrupted["optimizer"],
        "moments_match": (
            resumed["optimizer"]["m"] == uninterrupted["optimizer"]["m"]
            and resumed["optimizer"]["v"] == uninterrupted["optimizer"]["v"]
        ),
        "counters_match": (
            resumed["optimizer"]["step_counts"]
            == uninterrupted["optimizer"]["step_counts"]
        ),
        "hyperparameters_match": all(
            resumed["optimizer"][key] == uninterrupted["optimizer"][key]
            for key in ("lr", "betas", "eps", "format_version", "optimizer",
                        "parameters")
        ),
        "generator_matches": (
            resumed["generator"] == uninterrupted["generator"]
        ),
        "topology_matches": resumed["topology"] == uninterrupted["topology"],
        "topology_is_expected": (
            resumed["topology"]["aliases"] == EXPECTED_GENERATOR_ALIASES
            and resumed["topology"]["shared"] is True
            and resumed["topology"]["canonical_keys"]
            == [CANONICAL_GENERATOR_KEY]
        ),
        "final_loader_state_matches": (
            resumed["loader"]["state_dict"]
            == uninterrupted["loader"]["state_dict"]
        ),
        "final_loader_position": (uninterrupted["loader"]["epoch"],
                                  uninterrupted["loader"]["cursor"]),
        "dataset_identity_matches": (
            resumed["dataset_identity"] == uninterrupted["dataset_identity"]
        ),
        "evaluation_matches": (
            resumed["evaluation"] == uninterrupted["evaluation"]
        ),
        "evaluation_is_neutral": (
            uninterrupted["evaluation"]["buffers_unchanged"] is True
            and uninterrupted["evaluation"]["consumed_no_generator_call"]
            is True
            and resumed["evaluation"]["buffers_unchanged"] is True
            and resumed["evaluation"]["consumed_no_generator_call"] is True
        ),
        "all_state_at_run_dtype": all(
            facts["dtype"] == dtype
            for facts in resumed["parameters"].values()
        ),
        "all_state_on_cpu": all(
            facts["device"] == "cpu"
            for facts in resumed["parameters"].values()
        ),
        # -- the negative control ---------------------------------------
        "omitted_next_batch_differs": (
            omitted["restored"]["loader"]["next_batch_indices"]
            != reference[split_step]["indices"]
        ),
        "omitted_indices_differ": (
            omitted["index_sequence"]
            != [step["indices"] for step in reference[split_step:]]
        ),
        "omitted_losses_differ": (
            [step["loss_bits"] for step in omitted["steps"]]
            != [step["loss_bits"] for step in reference[split_step:]]
        ),
        "omitted_parameters_differ": (
            omitted["parameters"] != uninterrupted["parameters"]
        ),
        "omitted_evaluation_differs": (
            omitted["evaluation"] != uninterrupted["evaluation"]
        ),
        # -- non-vacuity -------------------------------------------------
        "training": _training_moved_everything(uninterrupted),
        "schedule": _schedule_is_non_vacuous(uninterrupted, split_step),
        # -- reporting / cross-dtype facts --------------------------------
        "index_sequence": uninterrupted["index_sequence"],
        "position_sequence": uninterrupted["position_sequence"],
        "epoch_permutations": uninterrupted["epoch_permutations"],
        "next_batch_at_interruption": (
            resumed["saved"]["loader"]["next_batch_indices"]
        ),
        "uninterrupted_losses": [step["loss"] for step in reference],
        "resumed_losses": [step["loss"] for step in combined],
        "final_evaluation": uninterrupted["evaluation"],
        "checkpoint_metadata": resumed["metadata"],
        "step_counts": uninterrupted["optimizer"]["step_counts"],
        "parameter_names": list(uninterrupted["parameters"]),
    }


def cross_dtype_facts(proofs):
    """The **only** things two dtypes' proofs may be compared on — all of
    them dtype-independent, because a permutation is a pure function of
    ``(seed, epoch, samples)`` and carries no dtype at all.

    Losses, logits, parameters, buffers, optimizer moments, and evaluation
    outputs are deliberately **absent**: cross-dtype numeric equality is not
    a TensorForge contract and nothing here asserts it."""
    first, second = (proofs[dtype] for dtype in RUN_DTYPES)
    return {
        "dtypes": list(RUN_DTYPES),
        "index_sequences_match": (
            first["index_sequence"] == second["index_sequence"]
        ),
        "permutations_match": (
            first["epoch_permutations"] == second["epoch_permutations"]
        ),
        "positions_match": (
            first["position_sequence"] == second["position_sequence"]
        ),
        "next_batch_at_interruption_matches": (
            first["next_batch_at_interruption"]
            == second["next_batch_at_interruption"]
        ),
        "final_loader_position_matches": (
            first["final_loader_position"] == second["final_loader_position"]
        ),
    }


# --------------------------------------------------------------------------
# The exit gate
# --------------------------------------------------------------------------

# Every boolean the proof must satisfy, at every dtype. Listed once, by name,
# so ``main()`` reports exactly what it checks and a new claim cannot be added
# to the output without being added to the gate.
REQUIRED = (
    "next_step_is_split",
    "one_batch_per_step",
    "snapshot_immediately_precedes_save",
    "fresh_started_different",
    "fresh_shares_no_identity",
    "identities_preserved",
    "load_restored_parameters",
    "load_restored_optimizer",
    "load_restored_generator",
    "load_restored_topology",
    "loader_adopted_saved_state",
    "loader_next_batch_matches_saved",
    "next_batch_after_restore_matches",
    "prefix_indices_match",
    "suffix_indices_match",
    "whole_index_sequence_matches",
    "position_sequence_matches",
    "epoch_boundaries_match",
    "feature_batches_match",
    "target_batches_match",
    "loss_sequence_matches",
    "suffix_losses_match",
    "logits_match",
    "step_predictions_match",
    "parameters_match",
    "buffers_match",
    "optimizer_matches",
    "moments_match",
    "counters_match",
    "hyperparameters_match",
    "generator_matches",
    "topology_matches",
    "topology_is_expected",
    "final_loader_state_matches",
    "dataset_identity_matches",
    "evaluation_matches",
    "evaluation_is_neutral",
    "all_state_at_run_dtype",
    "all_state_on_cpu",
    "omitted_next_batch_differs",
    "omitted_indices_differ",
    "omitted_losses_differ",
    "omitted_parameters_differ",
    "omitted_evaluation_differs",
)

REQUIRED_TRAINING = (
    "parameters_moved",
    "buffers_moved",
    "moments_became_nonzero",
    "step_counters_advanced",
    "generator_calls_advanced",
    "loss_sequence_varies",
    "evaluation_output_changed",
    "optimizer_state_nonempty",
)

REQUIRED_SCHEDULE = (
    "split_is_not_zero",
    "split_is_not_final",
    "split_is_mid_epoch",
    "suffix_is_multi_step",
    "shuffle_is_on",
    "order_is_not_identity",
    "epochs_have_distinct_orders",
)

REQUIRED_CROSS_DTYPE = (
    "index_sequences_match",
    "permutations_match",
    "positions_match",
    "next_batch_at_interruption_matches",
    "final_loader_position_matches",
)


def failed_checks(proof):
    """Every required check that did not hold, by name — empty when the proof
    passed."""
    failures = [name for name in REQUIRED if proof[name] is not True]
    failures += [f"training.{name}" for name in REQUIRED_TRAINING
                 if proof["training"][name] is not True]
    failures += [f"schedule.{name}" for name in REQUIRED_SCHEDULE
                 if proof["schedule"][name] is not True]
    if proof["schedule"]["epoch_boundaries_crossed"] < 1:
        failures.append("schedule.epoch_boundaries_crossed")
    if proof["schedule"]["batches_left_in_active_epoch"] < 1:
        failures.append("schedule.batches_left_in_active_epoch")
    return failures


def failed_cross_dtype_checks(facts):
    return [name for name in REQUIRED_CROSS_DTYPE if facts[name] is not True]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _format_losses(values, per_line=5):
    """A compact multi-line rendering of a loss sequence — never a giant
    single line, and never a truncated one."""
    rendered = [f"{value:.8f}" for value in values]
    return "\n".join(
        "  " + " ".join(rendered[start:start + per_line])
        for start in range(0, len(rendered), per_line)
    )


def _report(proof):
    dtype = proof["dtype"]
    schedule = proof["schedule"]
    training = proof["training"]
    print()
    print("=" * 72)
    print(f"run dtype: {dtype}   (proved only against itself — a float32 run "
          f"is never compared to a float64 one)")
    print("=" * 72)
    print(f"parameters and buffers: {proof['parameter_names']}")
    print(f"batches per epoch:      {schedule['batches_per_epoch']} "
          f"({SAMPLES} samples / batch size {BATCH_SIZE}, drop_last="
          f"{DROP_LAST}, shuffle={SHUFFLE}, seed={SAMPLER_SEED})")
    print(f"batch index sequence ({len(proof['index_sequence'])} steps):")
    for step, (indices, position) in enumerate(
            zip(proof["index_sequence"], proof["position_sequence"])):
        marker = "  <- checkpoint taken here (next_step)" if (
            step == proof["split_step"]) else ""
        print(f"  step {step:>2}  epoch {position[0]} cursor {position[1]}  "
              f"{list(indices)}{marker}")
    print("uninterrupted loss sequence:")
    print(_format_losses(proof["uninterrupted_losses"]))
    print("interrupted + resumed loss sequence:")
    print(_format_losses(proof["resumed_losses"]))
    print(f"  split is mid-epoch:           {schedule['split_is_mid_epoch']} "
          f"(position {schedule['split_position']}, "
          f"{schedule['batches_left_in_active_epoch']} batches still owed by "
          f"that epoch)")
    print(f"  epoch boundaries crossed:     "
          f"{schedule['epoch_boundaries_crossed']} over "
          f"{schedule['exercised_epochs']} exercised epochs, each with a "
          f"distinct non-identity order "
          f"({schedule['epochs_have_distinct_orders']})")
    print(f"  next_step from metadata:      {proof['next_step']} "
          f"(one delivered batch = one completed step: "
          f"{proof['one_batch_per_step']})")
    print(f"  snapshot then save, no delivery between: "
          f"{proof['snapshot_immediately_precedes_save']} "
          f"{proof['journal_tail']}")
    print(f"  fresh target began elsewhere: "
          f"{proof['fresh_started_different']} and shares no object "
          f"({proof['fresh_shares_no_identity']}); load preserved every "
          f"identity ({proof['identities_preserved']})")
    print(f"  loader adopted saved state:   "
          f"{proof['loader_adopted_saved_state']}; next batch after restore "
          f"is the uninterrupted continuation "
          f"({proof['next_batch_after_restore_matches']})")
    print(f"  whole batch-index sequence:   "
          f"{proof['whole_index_sequence_matches']}")
    print(f"  feature bits / target arrays: "
          f"{proof['feature_batches_match']} / "
          f"{proof['target_batches_match']}")
    print(f"  every loss bit-identical:     "
          f"{proof['loss_sequence_matches']}")
    print(f"  parameters / buffers:         {proof['parameters_match']} / "
          f"{proof['buffers_match']}")
    print(f"  Adam m, v / step counters:    {proof['moments_match']} / "
          f"{proof['counters_match']} {proof['step_counts']}")
    print(f"  generator / alias topology:   "
          f"{proof['generator_matches']} / {proof['topology_matches']} "
          f"(expected map: {proof['topology_is_expected']})")
    print(f"  final loader state_dict:      "
          f"{proof['final_loader_state_matches']} "
          f"(epoch {proof['final_loader_position'][0]}, cursor "
          f"{proof['final_loader_position'][1]})")
    print(f"  final eval loss / accuracy:   "
          f"{proof['final_evaluation']['loss']:.6f} / "
          f"{proof['final_evaluation']['accuracy']:.4f} "
          f"(exact match: {proof['evaluation_matches']}, buffers and "
          f"generator untouched: {proof['evaluation_is_neutral']})")
    print(f"  final predictions:            "
          f"{proof['final_evaluation']['predictions']}")
    print(f"  training actually moved:      parameters "
          f"{training['parameters_moved']}, "
          f"{training['buffer_count']} running buffers "
          f"{training['buffers_moved']}, Adam moments "
          f"{training['moments_became_nonzero']}, generator calls "
          f"{training['generator_calls_advanced']}, evaluation output "
          f"{training['evaluation_output_changed']}, loss varies "
          f"{training['loss_sequence_varies']}")
    print(f"  omitting loader restore diverges: indices "
          f"{proof['omitted_indices_differ']}, losses "
          f"{proof['omitted_losses_differ']}, parameters "
          f"{proof['omitted_parameters_differ']}, evaluation "
          f"{proof['omitted_evaluation_differs']}")
    print(f"  every value at {dtype} on cpu: "
          f"{proof['all_state_at_run_dtype']} / {proof['all_state_on_cpu']}")


def main():
    if not cpp.is_available():
        print("The experimental C++ backend is not built.")
        print(cpp.build_instructions())
        return

    print(f"deterministic native mini-batch training "
          f"(Linear({FEATURES}->{HIDDEN}) -> BatchNorm1d({HIDDEN}) -> ReLU -> "
          f"Dropout(p={DROPOUT_P}) -> Linear({HIDDEN}->{HIDDEN}) -> "
          f"LayerNorm({HIDDEN}) -> Dropout(p={DROPOUT_P}) -> "
          f"Linear({HIDDEN}->{NUM_CLASSES}))")
    print(f"NativeCrossEntropyLoss over raw logits, NativeAdam "
          f"(lr={DEFAULT_LR}), {TOTAL_STEPS} steps interrupted after "
          f"{SPLIT_STEP}, over NativeTensorDataset -> NativeBatchSampler -> "
          f"NativeDataLoader ({SAMPLES} samples, {NUM_CLASSES} classes, "
          f"shuffled batches of {BATCH_SIZE})")
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

    cross = cross_dtype_facts(proofs)
    broken_cross = failed_cross_dtype_checks(cross)
    if broken_cross:
        failures["cross-dtype"] = broken_cross

    print()
    print("checkpoint format: tensorforge.native_checkpoint, version 3 — "
          "unchanged, with no loader field and no version 4")
    print("the loader position travels as ORDINARY CALLER METADATA under "
          f"{TRAINING_KEY!r} / {LOADER_KEY!r}; no runtime code knows those "
          "names, and there is no automatic loader discovery")
    print("save order: loader.state_dict() -> no iteration -> "
          "save_native_checkpoint;  restore order: load_native_checkpoint "
          "-> loader.load_state_dict  (no cross-object atomicity)")
    print("comparison mechanism: raw IEEE-754 bit patterns (uint32 at "
          "float32, uint64 at float64) — no tolerance, no allclose, and no "
          "float32-versus-float64 numeric comparison anywhere")
    print(f"identical across dtypes (dtype-independent only): batch indices "
          f"{cross['index_sequences_match']}, permutations "
          f"{cross['permutations_match']}, positions "
          f"{cross['positions_match']}, next batch at interruption "
          f"{cross['next_batch_at_interruption_matches']}, final position "
          f"{cross['final_loader_position_matches']}")
    print(f"live native storage baseline / final: {baseline} / {final_live}")

    if final_live != baseline:
        failures["lifecycle"] = [
            f"live native storage {baseline} -> {final_live}"
        ]

    for dtype in RUN_DTYPES:
        status = "no" if dtype in failures else "yes"
        print(f"exact deterministic mini-batch resume at {dtype}: {status}")

    if failures:
        for scope, broken in failures.items():
            print(f"FAILED [{scope}]: {', '.join(broken)}")
        raise SystemExit("the resumed run diverged from the uninterrupted run")
    print("deterministic native mini-batch training + exact interrupted "
          "resume ok at float64 and float32")


if __name__ == "__main__":
    main()
