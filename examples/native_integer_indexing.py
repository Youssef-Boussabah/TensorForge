"""End-to-end native integer indexing: `argmax` predictions, `index_select`
over the predicted classes, and an exact interrupted resume (Advanced C++
Phase K, milestone K6).

The first end-user program in which the native runtime's **integer** side
carries real work. A small classifier trains over the Phase-J pipeline, and
at fixed evaluation points its raw logits are turned into **native
``int64`` prediction indices** by ``NativeTensor.argmax`` and those indices
are then **consumed** by ``NativeTensor.index_select`` over a detached copy
of the same logits::

    Linear(5 -> 8) -> ReLU -> Linear(8 -> 4)
      -> NativeCrossEntropyLoss  (+ NativeAdam)

    predictions      = logits.argmax(axis=1)          # int64, K3
    detached_logits  = logits.detach()                 # graph-free source
    selected         = detached_logits.index_select(1, predictions)   # K4

**K6 adds no runtime capability.** No kernel, no C ABI export, no module,
loss, metric, or optimizer, no checkpoint field or version, no public
package export, and no executable line of ``src/``. It composes what K1-K5
already shipped into one ordinary program, written entirely against the
**public** experimental surface.

**``index_select`` here is axis selection, not a per-row gather.** This is
the single most important thing to read correctly. Given logits of shape
``(batch, classes)`` and predictions of shape ``(batch,)``, the call

    detached_logits.index_select(1, predictions)

returns shape ``(batch, batch)``: it selects **the same ordered prediction
vector along the class axis for every row**, so column *j* of the result is
the whole source column ``predictions[j]``. Each example's own
predicted-class logit therefore sits on the **diagonal** —
``selected[row, row] == logits[row, predictions[row]]`` — and this program
verifies both halves on the host in raw IEEE-754 bits: the diagonal
relation, *and* every column against its source column, so duplicates and
order are proved preserved rather than assumed. A per-row gather is a
different operation with a different index shape, and TensorForge does not
have one (design §18.1).

**Why the ``argmax`` reads the live logits and the ``index_select`` reads a
detached copy.** ``argmax`` returns a plain leaf even from a
gradient-tracking input, because the derivative of an index with respect to
a value does not exist (§17.9). ``index_select`` ships **forward only**, and
a source with ``requires_grad=True`` is **rejected with a message naming
``detach()``** rather than being silently detached, because a graph-free
result from a gradient-tracking source would be a silent gradient hole
(§18.9). So the two calls take deliberately different sources, and neither
touches the training graph.

**Native ``int64`` never becomes a training target.** Cross entropy takes
the loader's **read-only host ``numpy.ndarray`` targets of dtype ``int64``**
at every width, exactly as Phase E and Phase J left it; the native integer
tensors here are *evaluation* values only. ``int64`` is an index/result
dtype in its own registry, not a supported compute dtype, and
``NativeTensor.from_int64_array`` remains the one public door through which
an integer buffer can come into existence.

**The interruption is genuinely mid-epoch.** Twenty-four samples in batches
of six give four batches per epoch; the run is ten steps and the
interruption lands after five, so the saved position is epoch 1, cursor 1 —
three batches still owed by the *active* epoch, with two epoch boundaries
crossed across the whole run. Evaluations happen on **both sides** of the
checkpoint (steps 1 and 4 before it, 6 and 9 after it).

**The supported save/restore order, unchanged.** Saving is
``loader.state_dict()`` first, then ``save_native_checkpoint`` with that
state nested in the caller's own ``metadata`` — with **no delivery in
between**. Restoring is ``load_native_checkpoint`` **first**, then
``loader.load_state_dict``. There is **no cross-object atomicity**: if the
first succeeds and the second fails nothing rolls back, and the documented
recovery is to discard everything and repeat both calls. ``"training"``,
``"data_loader"``, and ``"next_step"`` are this repository's **caller
conventions**; no runtime code knows them.

**The restore target is deliberately built wrong**: different parameter
seeds, a different learning rate, a separately constructed dataset, and a
loader with a different seed, batch size, shuffle setting, and position —
advanced there by really delivering batches. Every difference is proved
*before* the load, and ``run_omitted_loader_control()`` shows the run
genuinely **diverges** when the loader restoration alone is left out.

**Two dtypes, two independent proofs, and no numeric comparison between
them.** ``run_dtype_proof("float64")`` and ``run_dtype_proof("float32")``
each build their own host data, native state, and checkpoint, and each is
compared **only against itself**. The only cross-dtype claims gated here
are dtype-independent: the batch-index schedule, the permutations, the
positions, and structural metadata. Whether the two widths happen to
predict the same classes is **observed and reported, never required** — it
is legitimate for them to differ, and each run must still reproduce itself
exactly.

**Exactness is measured in bits for floats and in integers for indices.**
Floating values are compared through raw IEEE-754 bit patterns —
``uint32`` views at float32, ``uint64`` at float64 — never a tolerance and
never ``allclose``; ``bits()`` refuses an array whose dtype is not exactly
the run's. Prediction indices are compared as **exact Python integers** and
are never converted to floating values.

**The data.** ``build_features()`` computes twenty-four five-feature rows
from an explicit formula over the sample index. Every value is a multiple
of one eighth, so all of them are exactly representable in *both* binary32
and binary64 and identical on every platform. Nothing is generated
randomly, loaded, downloaded, or read from a file, and no global random
stream — Python's or NumPy's — is touched anywhere in this module. Six
predictions over four classes means duplicate predicted classes occur in
**every** evaluation batch by pigeonhole, which is what makes the
duplicate-preservation check meaningful without distorting the model.

**Ownership is explicit throughout.** Every ``argmax`` result, every
detached source, and every ``index_select`` result is closed in a
``finally``, as is every delivered feature batch, forward output, loss, and
gradient; each run closes its loader, dataset, optimizer, and parameters on
the way out; and ``main()`` reports the native live-storage count before
and after the whole workflow, which must return exactly to its baseline.
The target array is ordinary read-only host memory and is never closed.
Checkpoints live in a temporary directory that is removed automatically;
nothing is left behind.

This is an integration proof on one fixed task — not a benchmark, not a
generalization claim, and **no timing or performance is claimed or measured
anywhere**. It needs the experimental C++ backend to be built — run:

    uv run python examples/native_integer_indexing.py

Every helper that represents a completed proof returns plain Python values
only — never a live ``NativeTensor``, model, optimizer, loader, dataset, or
sampler — so the tests can import and verify them, and ``main()`` prints
them.
"""

import gc
import os
import tempfile

import numpy as np

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchSampler,
    NativeCrossEntropyLoss,
    NativeDataLoader,
    NativeLinear,
    NativeModule,
    NativeReLU,
    NativeTensorDataset,
    load_native_checkpoint,
    save_native_checkpoint,
)

# --------------------------------------------------------------------------
# The fixed task
# --------------------------------------------------------------------------

SAMPLES = 24
FEATURES = 5
NUM_CLASSES = 4
HIDDEN = 8

# The axis logits are classified along, and the axis both integer operations
# work on: `argmax` reduces it, `index_select` selects along it.
CLASS_AXIS = 1

# Four batches per epoch, so an interruption can land strictly inside one.
# BATCH_SIZE > NUM_CLASSES, so six predictions over four classes must repeat
# a class — duplicates are guaranteed by pigeonhole rather than hoped for.
BATCH_SIZE = 6
DROP_LAST = False
SHUFFLE = True
SAMPLER_SEED = 20260806
BATCHES_PER_EPOCH = 4          # ceil(24 / 6); asserted against the sampler

# Ten steps span two whole epochs and half of a third, so the run crosses two
# epoch boundaries. The interruption is deliberately *not* one of them: 5 is
# neither 0 nor the last step, and 5 % 4 == 1, so the saved position is
# epoch 1, cursor 1 — three batches still owed by the active epoch.
TOTAL_STEPS = 10
SPLIT_STEP = 5
EXERCISED_EPOCHS = (TOTAL_STEPS + BATCHES_PER_EPOCH - 1) // BATCHES_PER_EPOCH

# The steps whose logits are put through the integer evaluation path. Two sit
# before the interruption and two after it, so the indexing is exercised on
# **both sides** of the checkpoint and the resumed run has to reproduce both.
EVAL_STEPS = (1, 4, 6, 9)

DEFAULT_LR = 0.05

# The two dtypes the native runtime computes at, proved independently and
# never against each other. float64 is first because it is the default.
RUN_DTYPES = ("float64", "float32")

# The dtype every prediction tensor must physically be. It is an index dtype
# in its own registry (`cpp.INDEX_DTYPES`), never a member of
# `cpp.SUPPORTED_DTYPES`, and nothing in this program computes at it.
INDEX_DTYPE = "int64"

# Fixed initialization seeds. Each layer draws from its own *local* seeded
# generator, so nothing here touches a global RNG.
HIDDEN_SEED = 41
OUTPUT_SEED = 42

# The deliberately different set for the fresh restore target. Both seeds
# differ and so does the learning rate, so a load that restored nothing could
# not possibly produce a matching run.
FRESH_HIDDEN_SEED = 8101
FRESH_OUTPUT_SEED = 8102
FRESH_LR = 0.017

# ...and the deliberately different loader the target is built with: the
# seed, the batch size, the shuffle setting, and — once ``advance_loader``
# has really delivered batches — the position. Configuration is *adopted*
# from a loaded state, so every one of these is replaced by the restore
# rather than having to be guessed right.
FRESH_SAMPLER_SEED = 55501
FRESH_BATCH_SIZE = 4
FRESH_SHUFFLE = False
FRESH_DROP_LAST = False
FRESH_ADVANCE_BATCHES = 3

# The canonical child-module names, in registration (execution) order.
HIDDEN_NAME = "hidden"
RELU_NAME = "relu"
OUTPUT_NAME = "output"

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
# Exact comparison helpers — bits for floats, integers for indices
# --------------------------------------------------------------------------


def bits(array, dtype):
    """``array``'s raw IEEE-754 bit patterns as a flat list of Python ints.

    The **only** comparison mechanism this example uses for floating values:
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
    """A live floating tensor's values as raw bits, through the explicit
    public ``to_numpy()`` boundary. Materializes a fresh host array and
    mutates nothing."""
    if tensor.dtype != dtype:
        raise TypeError(f"expected a {dtype} tensor, got {tensor.dtype}")
    return bits(tensor.to_numpy(), dtype)


def index_values(tensor):
    """A live ``int64`` tensor's values as **exact Python integers**.

    Deliberately not routed through ``bits()``: an index is an integer, and
    the only honest comparison for one is integer equality. Nothing here
    converts an index to a floating value, at any point, in either
    direction."""
    if tensor.dtype != INDEX_DTYPE:
        raise TypeError(
            f"expected an {INDEX_DTYPE} tensor, got {tensor.dtype}; "
            f"prediction indices are never read at a floating width")
    return [int(value) for value in tensor.tolist()]


def target_facts(targets):
    """Everything a delivered target batch is contractually required to be,
    as plain Python: a fresh, independently owned, C-contiguous, read-only
    host ``int64`` array — **host** metadata, never a native tensor, and
    never what ``argmax`` produced."""
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
    """The fixed task's features as a ``24 x 5`` nested list of floats.

    Computed from an explicit formula over the sample index rather than
    stored as literals, so the structure is visible: sample *i* belongs to
    class ``i % 4`` and sits at position ``i // 4``, which contributes an
    ``offset`` of ``k / 8`` for ``k`` in ``0..5``. Column ``class`` carries
    the strong positive marker ``1.0 + offset``, column ``(class + 2) % 5``
    a negative marker ``-0.75 + offset``, the last column a per-class code
    ``0.25 * class - 0.375``, and every other column the background
    ``offset - 0.5``. The assignments are applied in that fixed order, so a
    column named twice takes the later value deterministically.

    **Every value is a multiple of one eighth**, and therefore exactly
    representable in binary32 *and* binary64 — which is why the same nested
    list can seed both runs without either one being a rounded version of
    the other. Nothing consults the clock, a random source, an environment
    variable, the filesystem, or the network, and repeated calls return
    equal values in independent containers."""
    rows = []
    for index in range(SAMPLES):
        label = index % NUM_CLASSES
        offset = (index // NUM_CLASSES) / 8.0      # 0.0, 0.125 ... 0.625
        row = [offset - 0.5] * FEATURES
        row[(label + 2) % FEATURES] = -0.75 + offset
        row[label] = 1.0 + offset
        row[FEATURES - 1] = 0.25 * label - 0.375
        rows.append(row)
    return rows


def build_targets():
    """The fixed class labels as a plain list of Python ints — sample ``i``
    belongs to class ``i % 4``, so all four classes occur six times each."""
    return [index % NUM_CLASSES for index in range(SAMPLES)]


def host_arrays(dtype):
    """``(features, targets)`` as host NumPy arrays, the features physically
    at the run's dtype and the targets exact ``int64``.

    This is where the run's width is chosen, **once**, on the host: a
    float32 run's data is genuinely ``np.float32`` before it ever reaches
    the dataset constructor. Each dtype gets its own independent array built
    from the same exactly representable literals, so neither is a narrowed
    copy of the other, and the two arrays hold the same logical values. The
    targets are ``int64`` at both widths and are **host label metadata**,
    which no Phase-K milestone changed."""
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


class NativeIndexingClassifier(NativeModule):
    """The compact native classifier this example trains, at one explicit
    dtype::

        hidden = NativeLinear(5, 8)     # parameters
        relu   = NativeReLU()
        output = NativeLinear(8, 4)     # parameters

    producing **raw logits** of shape ``(batch_size, 4)``. There is
    deliberately no softmax or log-softmax module: the fused, numerically
    stable ``NativeCrossEntropyLoss`` consumes logits directly.

    It is deliberately smaller than the Phase-J mini-batch classifier. K6's
    subject is the *integer indexing* of a prediction, not the breadth of
    the module library, and every extra stochastic or normalizing layer
    would add state to the resume proof without adding anything to the
    indexing one. Trainable parameters plus Adam's moments and step counters
    are enough for the resume half to be non-vacuous; there are no buffers
    and no registered generator, and this program never claims otherwise.

    **Every state-owning child receives the run dtype explicitly.** The
    stateless child — ReLU — takes **no** dtype argument and must not gain
    one: it owns no dtype-bearing numeric state, so an argument there would
    be a second authority that could disagree with the data.

    This class is an **example implementation detail**. It is not exported,
    is not a public module, and no milestone adds it to
    ``tensorforge.experimental``."""

    def __init__(self, dtype, hidden_seed=HIDDEN_SEED,
                 output_seed=OUTPUT_SEED):
        super().__init__()
        self.hidden = NativeLinear(FEATURES, HIDDEN, seed=hidden_seed,
                                   dtype=dtype)
        self.relu = NativeReLU()
        self.output = NativeLinear(HIDDEN, NUM_CLASSES, seed=output_seed,
                                   dtype=dtype)
        # Recorded for reporting only. The authority on any tensor's dtype is
        # that tensor's own storage tag, never this attribute.
        self._dtype = dtype

    @property
    def dtype(self):
        """The dtype this model's parameters were built at."""
        return self._dtype

    def forward(self, features):
        """``(N, 5)`` features to ``(N, 4)`` raw logits."""
        return self.output(self.relu(self.hidden(features)))


def build_model(dtype, fresh=False):
    """A freshly initialized classifier at ``dtype``.

    Deterministic: every layer draws its initialization from a *local*
    seeded generator, so two independently built models start numerically
    identical and neither the global NumPy RNG nor Python's ``random`` is
    ever touched.

    ``fresh=True`` selects the **deliberately different** seed set used for
    the restore target."""
    if fresh:
        return NativeIndexingClassifier(dtype, hidden_seed=FRESH_HIDDEN_SEED,
                                        output_seed=FRESH_OUTPUT_SEED)
    return NativeIndexingClassifier(dtype)


def build_loss():
    """The native classification loss, over raw logits and **host** ``int64``
    targets. It takes no dtype argument and must not gain one: it is a thin
    delegate that inherits the dtype of the logits it is handed, and its
    target contract is host label metadata at every width."""
    return NativeCrossEntropyLoss()


def build_optimizer(model, lr=DEFAULT_LR):
    """``NativeAdam`` over the model's trainable parameters only.

    It takes **no** dtype argument and must not gain one: it owns no dtype
    it could choose, only state that must match a parameter."""
    return NativeAdam(model.parameters(), lr=lr)


# --------------------------------------------------------------------------
# The integer evaluation path — argmax, then index_select
# --------------------------------------------------------------------------


def evaluate_indexing(logits, dtype):
    """The K3 + K4 evaluation path over one step's logits, as plain Python.

    Exactly two native integer operations run here, in this order::

        predictions     = logits.argmax(axis=1)                    # K3
        detached_logits = logits.detach()
        selected        = detached_logits.index_select(1, predictions)   # K4

    The ``argmax`` reads the **live, gradient-tracking** logits because
    §17.9 promises a plain leaf even then; the ``index_select`` reads a
    **detached** source because §18.9 rejects a ``requires_grad=True`` one
    rather than detaching it silently. Neither call joins the training
    graph, consumes a random draw, or mutates the model.

    ``index_select`` selects **the same ordered index vector along the class
    axis for every row** — it is not a per-row gather — so for a batch of
    ``B`` the result is ``(B, B)``: column *j* is the whole source column
    ``predictions[j]``, and each example's own predicted-class logit sits on
    the **diagonal**. Both relations are verified here on the host, in raw
    bits, and the columns check is what proves duplicates and order are
    preserved rather than merely plausible.

    Every native temporary is closed explicitly, in reverse order of
    creation, under ``try``/``finally`` — including under ``BaseException``.
    The returned record holds plain Python only."""
    predictions = logits.argmax(axis=CLASS_AXIS)
    try:
        indices = index_values(predictions)
        rows = int(logits.shape[0])
        record = {
            "prediction_dtype": predictions.dtype,
            "prediction_shape": tuple(predictions.shape),
            "prediction_device": predictions.device,
            "prediction_contiguous": bool(predictions.contiguous),
            "prediction_owns_core": bool(predictions.owns_core),
            "prediction_requires_grad": bool(predictions.requires_grad),
            "prediction_is_leaf": bool(predictions.is_leaf),
            "prediction_grad_is_none": predictions.grad is None,
            "prediction_numel": int(predictions.numel),
            "predictions": list(indices),
            "predictions_in_range": all(0 <= value < NUM_CLASSES
                                        for value in indices),
            "predictions_are_exact_ints": all(type(value) is int
                                              for value in indices),
            "duplicates_present": len(set(indices)) < len(indices),
            "distinct_predicted_classes": len(set(indices)),
            "logit_shape": tuple(logits.shape),
            "logit_dtype": logits.dtype,
            "logit_bits": tensor_bits(logits, dtype),
        }
        detached = logits.detach()
        try:
            selected = detached.index_select(CLASS_AXIS, predictions)
            try:
                host_selected = selected.to_numpy()
                host_logits = logits.to_numpy()
                positions = np.arange(rows)
                diagonal = np.ascontiguousarray(np.diagonal(host_selected))
                predicted_logits = np.ascontiguousarray(
                    host_logits[positions, np.asarray(indices)])
                columns_match = all(
                    bits(np.ascontiguousarray(host_selected[:, position]),
                         dtype)
                    == bits(np.ascontiguousarray(host_logits[:, index]), dtype)
                    for position, index in enumerate(indices)
                )
                duplicate_columns_identical = all(
                    bits(np.ascontiguousarray(host_selected[:, left]), dtype)
                    == bits(np.ascontiguousarray(host_selected[:, right]),
                            dtype)
                    for left in range(len(indices))
                    for right in range(left + 1, len(indices))
                    if indices[left] == indices[right]
                )
                record.update(
                    detached_requires_grad=bool(detached.requires_grad),
                    detached_dtype=detached.dtype,
                    selected_dtype=selected.dtype,
                    selected_shape=tuple(selected.shape),
                    selected_device=selected.device,
                    selected_contiguous=bool(selected.contiguous),
                    selected_owns_core=bool(selected.owns_core),
                    selected_requires_grad=bool(selected.requires_grad),
                    selected_is_leaf=bool(selected.is_leaf),
                    selected_grad_is_none=selected.grad is None,
                    selected_bits=bits(host_selected, dtype),
                    diagonal_bits=bits(diagonal, dtype),
                    predicted_logit_bits=bits(predicted_logits, dtype),
                    diagonal_is_predicted_logits=(
                        bits(diagonal, dtype) == bits(predicted_logits, dtype)
                    ),
                    columns_match_source_columns=columns_match,
                    duplicate_columns_identical=duplicate_columns_identical,
                    class_axis_length_is_prediction_count=(
                        int(selected.shape[CLASS_AXIS]) == len(indices)
                    ),
                    result_is_square_batch=(
                        tuple(selected.shape) == (rows, len(indices))
                    ),
                )
                return record
            finally:
                selected.close()
        finally:
            detached.close()
    finally:
        predictions.close()


# --------------------------------------------------------------------------
# Reporting helpers — plain Python values only
# --------------------------------------------------------------------------


def model_facts(model, dtype):
    """``model.state_dict()`` as ``{name: {shape, dtype, device, bits}}``, in
    canonical order — closing **every** caller-owned snapshot in a
    ``finally`` and returning no native tensor."""
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


def train_steps(model, optimizer, criterion, loader, dtype, *, start_step,
                stop_step, journal=None):
    """Run training steps ``[start_step, stop_step)`` over ``loader`` and
    return the list of plain-Python step records.

    **One delivered batch is one completed step**, and a step is complete
    only once its forward, integer evaluation (when the step is one of
    ``EVAL_STEPS``), loss, backward, optimizer update, gradient clear, and
    temporary cleanup have all run. So a caller that has run this to step
    ``k`` inclusive saves ``next_step = k + 1``, which is exactly the
    ``stop_step`` handed back here — and it is exactly what the loader's own
    position already says, because the committed cursor advances **if and
    only if** a batch was delivered.

    **The evaluation reuses the step's own logits**, deliberately. A second
    forward pass would be a second traversal of the model, and the claim
    "the integer evaluation changed nothing" would then be measuring the
    extra forward rather than ``argmax`` and ``index_select``.

    The iterator is created here and **closed here**. One iterator is one
    epoch: when its captured countdown is spent it raises ``StopIteration``,
    and this loop answers by creating a **new** iterator from the *same*
    loader, which continues at the sampler's canonical next-epoch position.
    Nothing resets or rebuilds the sampler and nothing increments an epoch
    or a cursor by hand; the position moves only through delivery.

    Every native object this function creates is closed explicitly: the
    delivered feature batch, the step's logits and loss, every gradient, and
    (inside ``evaluate_indexing``) every integer temporary. The delivered
    target array is ordinary host memory and is never closed.

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
            try:
                evaluation = (evaluate_indexing(logits, dtype)
                              if step in EVAL_STEPS else None)
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
                        "evaluation": evaluation,
                    }
                    loss.backward()
                    optimizer.step()
                finally:
                    loss.close()
            finally:
                logits.close()
                features.close()
            _release_gradients(model, optimizer)
            record["epoch_after"] = sampler.epoch
            record["cursor_after"] = sampler.cursor
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
    then every unique model parameter.

    The model traversal is identity-deduplicated, so shared state closes
    exactly once. This model registers no buffers and no generators, so
    neither traversal appears here. ``close()`` is idempotent everywhere,
    and nothing closed here is ever returned to a caller — in particular, a
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
        batches, integer temporaries, and gradients explicitly — that is the
        release mechanism, and it is what the baseline claim rests on. What
        is left to the collector is a reference *cycle* refcounting alone
        cannot break: the Python-managed autograd graph holds its parents
        through backward closures. Collecting here settles that into a
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


# --------------------------------------------------------------------------
# The runs
# --------------------------------------------------------------------------


def _final_record(model, optimizer, loader, dataset, dtype, steps):
    """Everything a completed run is compared on, as plain Python values.

    Collected in one place so two runs cannot accidentally record different
    things — the comparison is ``record_a == record_b`` over this exact
    structure."""
    return {
        "steps": steps,
        "parameters": model_facts(model, dtype),
        "optimizer": optimizer_facts(optimizer, dtype),
        "loader": loader_facts(loader),
        "dataset_identity": dataset.identity(),
    }


def _evaluations(steps):
    """``{step: evaluation record}`` for exactly the steps that evaluated —
    the comparable form of "the integer evaluation reproduced exactly"."""
    return {record["step"]: record["evaluation"] for record in steps
            if record["evaluation"] is not None}


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
            "optimizer": optimizer_facts(optimizer, dtype),
            "loader": loader_facts(loader),
        }
        steps = train_steps(model, optimizer, criterion, loader, dtype,
                            start_step=0, stop_step=total_steps)
        record = _final_record(model, optimizer, loader, dataset, dtype, steps)
        record.update(
            dtype=dtype,
            total_steps=total_steps,
            lr=optimizer.lr,
            initial=initial,
            batches_per_epoch=sampler.batches_per_epoch,
            index_sequence=[step["indices"] for step in steps],
            position_sequence=[(step["epoch_before"], step["cursor_before"])
                               for step in steps],
            epoch_permutations=[sampler.epoch_permutation(epoch)
                                for epoch in range(EXERCISED_EPOCHS)],
            evaluations=_evaluations(steps),
            gradients_cleared=all(parameter.grad is None
                                  for parameter in model.parameters()),
        )
        return record
    finally:
        _close_run(model, optimizer, loader, dataset)


def _identity_snapshot(model, optimizer, loader, dataset):
    """Object identities for every family, as ids. Used only to prove that a
    restored graph shares **nothing** with the graph that saved it, and that
    a load restores in place rather than constructing replacements."""
    return {
        "model": id(model),
        "parameters": [id(parameter) for parameter in model.parameters()],
        "optimizer": id(optimizer),
        "loader": id(loader),
        "sampler": id(loader.sampler),
        "dataset": id(dataset),
    }


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


def _restore_and_finish(dtype, path, total_steps, saved,
                        restore_loader=True):
    """Build the **entirely fresh** restore target, prove it started
    somewhere else, load, and finish the run.

    ``restore_loader=False`` is the negative control: everything else is
    identical, the loader restoration alone is skipped, and the resulting
    run must **differ**."""
    dataset_c = build_dataset(dtype)
    loader_c = model_c = optimizer_c = None
    try:
        loader_c, _sampler_c = build_loader(
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
        }
        suffix = train_steps(model_c, optimizer_c, criterion, loader_c, dtype,
                             start_step=next_step, stop_step=total_steps)
        record = _final_record(model_c, optimizer_c, loader_c, dataset_c,
                               dtype, suffix)
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
                fresh["loader"]["state_dict"] != saved["loader"]["state_dict"]
            ),
            fresh_next_batch_differs=(
                fresh["loader"]["next_batch_indices"]
                != saved["loader"]["next_batch_indices"]
            ),
            fresh_parameters_differ=fresh["parameters"] != saved["parameters"],
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
            index_sequence=[step["indices"] for step in suffix],
            position_sequence=[(step["epoch_before"], step["cursor_before"])
                               for step in suffix],
            evaluations=_evaluations(suffix),
        )
        return record
    finally:
        _close_run(model_c, optimizer_c, loader_c, dataset_c)


def _train_to_split_and_save(dtype, path, split_step, lr, journal=None):
    """Train ``[0, split_step)``, snapshot the loader, write the archive, and
    release **everything** the interrupted run owns.

    Returns ``(prefix records, saved facts, metadata written, retired
    objects)`` as plain Python plus one list of *emptied* objects. The
    archive is the only continuation boundary — nothing after this call may
    depend on a live object from before it.

    ``retired`` exists for one reason: CPython recycles ``id()`` values, so
    "the fresh graph shares no object with the saved one" would be measured
    against reusable addresses if the originals were already collected.
    Holding the emptied objects keeps every id unique and the claim honest.
    **None of them is ever passed into the restored graph**, none owns native
    storage any more, and the caller clears the list as soon as the identity
    comparison is done."""
    dataset_b = build_dataset(dtype)
    loader_b = model_b = optimizer_b = None
    retired = []
    try:
        loader_b, _sampler_b = build_loader(dataset_b)
        model_b = build_model(dtype)
        optimizer_b = build_optimizer(model_b, lr=lr)
        criterion = build_loss()
        prefix = train_steps(model_b, optimizer_b, criterion, loader_b, dtype,
                             start_step=0, stop_step=split_step,
                             journal=journal)

        # The supported save order: read the loader state **first**, do not
        # iterate, then write the archive.
        if journal is not None:
            journal.append(("loader_state_dict", split_step))
        loader_state = loader_b.state_dict()
        saved = {
            "loader": loader_facts(loader_b),
            "parameters": model_facts(model_b, dtype),
            "optimizer": optimizer_facts(optimizer_b, dtype),
            "identities": _identity_snapshot(model_b, optimizer_b, loader_b,
                                             dataset_b),
        }
        metadata_written = {
            TRAINING_KEY: {
                NEXT_STEP_KEY: split_step,
                LOADER_KEY: loader_state,
            }
        }
        if journal is not None:
            journal.append(("save_checkpoint", split_step))
        save_native_checkpoint(path, model_b, optimizer=optimizer_b,
                               metadata=metadata_written)
        # The interrupted run is released **before** the resume begins.
        _close_run(model_b, optimizer_b, loader_b, dataset_b)
        retired = [model_b, optimizer_b, loader_b, dataset_b]
        model_b = optimizer_b = loader_b = dataset_b = None
        return prefix, saved, metadata_written, retired
    finally:
        _close_run(model_b, optimizer_b, loader_b, dataset_b)


def run_interrupted_and_resumed(dtype, total_steps=TOTAL_STEPS,
                                split_step=SPLIT_STEP, lr=DEFAULT_LR):
    """Train to ``split_step``, checkpoint, discard **everything**, rebuild
    an entirely fresh graph from deliberately different seeds and a
    deliberately different loader, restore, and finish the run.

    Returns plain Python values only: the prefix and suffix step records, the
    final record, and every fact the proof needs about what was saved, what
    the fresh objects looked like before the load, and what the load actually
    did."""
    journal = []
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory,
                            f"native_integer_indexing_{dtype}.checkpoint.npz")
        prefix, saved, metadata_written, retired = _train_to_split_and_save(
            dtype, path, split_step, lr, journal=journal)
        try:
            resumed = _restore_and_finish(dtype, path, total_steps, saved)
        finally:
            retired.clear()
    resumed.update(prefix=prefix, saved=saved, journal=journal,
                   metadata_written=metadata_written, split_step=split_step,
                   total_steps=total_steps)
    return resumed


def run_omitted_loader_control(dtype, total_steps=TOTAL_STEPS,
                               split_step=SPLIT_STEP, lr=DEFAULT_LR):
    """The negative control: restore the model and optimizer from the archive
    and **omit** ``loader.load_state_dict``.

    The remaining batch order must then be wrong, and the finished run must
    differ from the uninterrupted one — otherwise the positive proof could be
    passing without the loader restoration doing anything at all. The whole
    leg builds and closes its own graph and returns plain Python."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(
            directory,
            f"native_integer_indexing_{dtype}.omitted.checkpoint.npz")
        _prefix, saved, _metadata, retired = _train_to_split_and_save(
            dtype, path, split_step, lr)
        try:
            return _restore_and_finish(dtype, path, total_steps, saved,
                                       restore_loader=False)
        finally:
            retired.clear()


# --------------------------------------------------------------------------
# The proof
# --------------------------------------------------------------------------


def _indexing_is_non_vacuous(uninterrupted, split_step):
    """The integer half's own non-vacuity: the evaluation really ran, on both
    sides of the interruption, and really produced duplicates."""
    evaluations = uninterrupted["evaluations"]
    steps = sorted(evaluations)
    return {
        "evaluated_steps": steps,
        "evaluated_before_split": [step for step in steps if step < split_step],
        "evaluated_after_split": [step for step in steps if step >= split_step],
        "evaluations_on_both_sides": (
            any(step < split_step for step in steps)
            and any(step >= split_step for step in steps)
        ),
        "duplicates_occurred": any(record["duplicates_present"]
                                   for record in evaluations.values()),
        "duplicates_guaranteed_by_pigeonhole": BATCH_SIZE > NUM_CLASSES,
        "every_prediction_is_int64": all(
            record["prediction_dtype"] == INDEX_DTYPE
            for record in evaluations.values()),
        "every_prediction_is_a_plain_leaf": all(
            record["prediction_requires_grad"] is False
            and record["prediction_is_leaf"] is True
            and record["prediction_grad_is_none"] is True
            for record in evaluations.values()),
        "every_prediction_owns_contiguous_storage": all(
            record["prediction_owns_core"] is True
            and record["prediction_contiguous"] is True
            for record in evaluations.values()),
        "every_prediction_in_range": all(
            record["predictions_in_range"] is True
            and record["predictions_are_exact_ints"] is True
            for record in evaluations.values()),
        "every_prediction_covers_the_batch": all(
            record["prediction_shape"] == (BATCH_SIZE,)
            and record["prediction_numel"] == BATCH_SIZE
            for record in evaluations.values()),
        "every_selection_keeps_the_source_dtype": all(
            record["selected_dtype"] == uninterrupted["dtype"]
            and record["detached_dtype"] == uninterrupted["dtype"]
            for record in evaluations.values()),
        "every_selection_is_a_fresh_graph_free_copy": all(
            record["selected_owns_core"] is True
            and record["selected_contiguous"] is True
            and record["selected_requires_grad"] is False
            and record["selected_is_leaf"] is True
            and record["selected_grad_is_none"] is True
            and record["detached_requires_grad"] is False
            for record in evaluations.values()),
        "every_selection_is_batch_by_batch": all(
            record["result_is_square_batch"] is True
            and record["class_axis_length_is_prediction_count"] is True
            for record in evaluations.values()),
        "every_diagonal_is_the_predicted_logit": all(
            record["diagonal_is_predicted_logits"] is True
            for record in evaluations.values()),
        "every_column_matches_its_source_column": all(
            record["columns_match_source_columns"] is True
            for record in evaluations.values()),
        "duplicate_columns_are_identical": all(
            record["duplicate_columns_identical"] is True
            for record in evaluations.values()),
    }


def _training_moved_everything(uninterrupted):
    """The training half: this task really trains, so a resume proof over it
    is not comparing two runs of nothing.

    Every claim is about *state that moved*, never about an accuracy or a
    loss threshold — a training example must not turn a numeric target into
    a gate."""
    initial = uninterrupted["initial"]
    final_optimizer = uninterrupted["optimizer"]
    losses = [step["loss_bits"] for step in uninterrupted["steps"]]
    return {
        "parameters_moved": (
            uninterrupted["parameters"] != initial["parameters"]
        ),
        "moments_became_nonzero": any(
            any(pattern != 0 for pattern in moment["bits"])
            for moment in final_optimizer["m"] + final_optimizer["v"]
        ),
        "step_counters_advanced": all(
            count == uninterrupted["total_steps"]
            for count in final_optimizer["step_counts"]
        ),
        "optimizer_state_was_empty_at_the_start": all(
            all(pattern == 0 for pattern in moment["bits"])
            for moment in initial["optimizer"]["m"] + initial["optimizer"]["v"]
        ),
        "optimizer_state_nonempty": bool(final_optimizer["m"]),
        "loss_sequence_varies": len(set(map(tuple, losses))) > 1,
        "logits_changed_over_training": (
            uninterrupted["steps"][0]["logit_bits"]
            != uninterrupted["steps"][-1]["logit_bits"]
        ),
        "gradients_cleared": uninterrupted["gradients_cleared"],
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
        "suffix_is_multi_step": uninterrupted["total_steps"] - split_step > 1,
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
                    lr=DEFAULT_LR):
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
    resumed = run_interrupted_and_resumed(dtype, total_steps=total_steps,
                                          split_step=split_step, lr=lr)
    omitted = run_omitted_loader_control(dtype, total_steps=total_steps,
                                         split_step=split_step, lr=lr)

    prefix = resumed["prefix"]
    suffix = resumed["steps"]
    combined = prefix + suffix
    reference = uninterrupted["steps"]
    combined_evaluations = _evaluations(combined)
    reference_evaluations = uninterrupted["evaluations"]
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
            len(combined) == total_steps
            and len(suffix) == total_steps - split_step
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
            resumed["fresh_optimizer_differs"],
        )),
        "fresh_shares_no_identity": resumed["fresh_shares_no_identity"],
        "identities_preserved": resumed["identities_preserved"],
        # -- what the two calls restored --------------------------------
        "load_restored_parameters": resumed["load_restored_parameters"],
        "load_restored_optimizer": resumed["load_restored_optimizer"],
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
            [(step["epoch_after"], step["cursor_after"]) for step in combined]
            == [(step["epoch_after"], step["cursor_after"])
                for step in reference]
        ),
        # -- the floating values ----------------------------------------
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
        "parameters_match": (
            resumed["parameters"] == uninterrupted["parameters"]
        ),
        "optimizer_matches": (
            resumed["optimizer"] == uninterrupted["optimizer"]
        ),
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
        "final_loader_state_matches": (
            resumed["loader"]["state_dict"]
            == uninterrupted["loader"]["state_dict"]
        ),
        "final_loader_position": (uninterrupted["loader"]["epoch"],
                                  uninterrupted["loader"]["cursor"]),
        "dataset_identity_matches": (
            resumed["dataset_identity"] == uninterrupted["dataset_identity"]
        ),
        # -- the integer values -----------------------------------------
        "evaluation_steps_match": (
            sorted(combined_evaluations) == sorted(reference_evaluations)
            == sorted(EVAL_STEPS)
        ),
        "prediction_indices_match": all(
            combined_evaluations[step]["predictions"]
            == reference_evaluations[step]["predictions"]
            for step in reference_evaluations
        ),
        "selected_bits_match": all(
            combined_evaluations[step]["selected_bits"]
            == reference_evaluations[step]["selected_bits"]
            for step in reference_evaluations
        ),
        "diagonal_bits_match": all(
            combined_evaluations[step]["diagonal_bits"]
            == reference_evaluations[step]["diagonal_bits"]
            for step in reference_evaluations
        ),
        "whole_evaluation_record_matches": (
            combined_evaluations == reference_evaluations
        ),
        "suffix_evaluations_match": all(
            resumed["evaluations"][step] == reference_evaluations[step]
            for step in resumed["evaluations"]
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
        "omitted_evaluations_differ": (
            omitted["evaluations"]
            != {step: reference_evaluations[step]
                for step in reference_evaluations if step >= split_step}
        ),
        # -- non-vacuity -------------------------------------------------
        "indexing": _indexing_is_non_vacuous(uninterrupted, split_step),
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
        "evaluations": reference_evaluations,
        "predictions_by_step": {step: record["predictions"]
                                for step, record in
                                reference_evaluations.items()},
        "selection_shapes": {step: record["selected_shape"]
                             for step, record in
                             reference_evaluations.items()},
        "checkpoint_metadata": resumed["metadata"],
        "step_counts": uninterrupted["optimizer"]["step_counts"],
        "parameter_names": list(uninterrupted["parameters"]),
    }


def cross_dtype_facts(proofs):
    """The **only** things two dtypes' proofs are *required* to agree on —
    all of them dtype-independent, because a permutation is a pure function
    of ``(seed, epoch, samples)`` and carries no dtype at all, and a shape is
    a property of the schedule rather than of a width.

    Losses, logits, parameters, optimizer moments, and selected values are
    deliberately **absent**: cross-dtype numeric equality is not a
    TensorForge contract and nothing here asserts it.

    ``prediction_indices_agree`` is an **observation, not a requirement**. A
    float32 run may legitimately predict a different class than a float64 one
    on the same inputs; each run still has to reproduce *itself* exactly,
    which is what the per-dtype proofs check. It is reported so a reader can
    see what actually happened, and it is deliberately not in
    ``REQUIRED_CROSS_DTYPE``."""
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
        "evaluation_steps_match": (
            sorted(first["evaluations"]) == sorted(second["evaluations"])
        ),
        "selection_shapes_match": (
            first["selection_shapes"] == second["selection_shapes"]
        ),
        # An observation, never a gate — see the docstring.
        "prediction_indices_agree": (
            first["predictions_by_step"] == second["predictions_by_step"]
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
    "parameters_match",
    "optimizer_matches",
    "moments_match",
    "counters_match",
    "hyperparameters_match",
    "final_loader_state_matches",
    "dataset_identity_matches",
    "evaluation_steps_match",
    "prediction_indices_match",
    "selected_bits_match",
    "diagonal_bits_match",
    "whole_evaluation_record_matches",
    "suffix_evaluations_match",
    "all_state_at_run_dtype",
    "all_state_on_cpu",
    "omitted_next_batch_differs",
    "omitted_indices_differ",
    "omitted_losses_differ",
    "omitted_parameters_differ",
    "omitted_evaluations_differ",
)

REQUIRED_INDEXING = (
    "evaluations_on_both_sides",
    "duplicates_occurred",
    "duplicates_guaranteed_by_pigeonhole",
    "every_prediction_is_int64",
    "every_prediction_is_a_plain_leaf",
    "every_prediction_owns_contiguous_storage",
    "every_prediction_in_range",
    "every_prediction_covers_the_batch",
    "every_selection_keeps_the_source_dtype",
    "every_selection_is_a_fresh_graph_free_copy",
    "every_selection_is_batch_by_batch",
    "every_diagonal_is_the_predicted_logit",
    "every_column_matches_its_source_column",
    "duplicate_columns_are_identical",
)

REQUIRED_TRAINING = (
    "parameters_moved",
    "moments_became_nonzero",
    "step_counters_advanced",
    "optimizer_state_was_empty_at_the_start",
    "optimizer_state_nonempty",
    "loss_sequence_varies",
    "logits_changed_over_training",
    "gradients_cleared",
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
    "evaluation_steps_match",
    "selection_shapes_match",
)


def failed_checks(proof):
    """Every required check that did not hold, by name — empty when the proof
    passed."""
    failures = [name for name in REQUIRED if proof[name] is not True]
    failures += [f"indexing.{name}" for name in REQUIRED_INDEXING
                 if proof["indexing"][name] is not True]
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
    indexing = proof["indexing"]
    print()
    print("=" * 72)
    print(f"run dtype: {dtype}   (proved only against itself — a float32 run "
          f"is never compared to a float64 one)")
    print("=" * 72)
    print(f"parameters:             {proof['parameter_names']}")
    print(f"batches per epoch:      {schedule['batches_per_epoch']} "
          f"({SAMPLES} samples / batch size {BATCH_SIZE}, drop_last="
          f"{DROP_LAST}, shuffle={SHUFFLE}, seed={SAMPLER_SEED})")
    print(f"batch index sequence ({len(proof['index_sequence'])} steps):")
    for step, (indices, position) in enumerate(
            zip(proof["index_sequence"], proof["position_sequence"])):
        marks = []
        if step == proof["split_step"]:
            marks.append("checkpoint taken here (next_step)")
        if step in EVAL_STEPS:
            marks.append("integer evaluation")
        marker = f"  <- {'; '.join(marks)}" if marks else ""
        print(f"  step {step:>2}  epoch {position[0]} cursor {position[1]}  "
              f"{list(indices)}{marker}")
    print("uninterrupted loss sequence:")
    print(_format_losses(proof["uninterrupted_losses"]))
    print("interrupted + resumed loss sequence:")
    print(_format_losses(proof["resumed_losses"]))
    print()
    print("integer evaluation path — logits.argmax(axis=1) then "
          "logits.detach().index_select(1, predictions):")
    for step in sorted(proof["evaluations"]):
        record = proof["evaluations"][step]
        print(f"  step {step:>2}  argmax -> {record['prediction_dtype']} "
              f"{record['prediction_shape']} {record['predictions']}  "
              f"(duplicates: {record['duplicates_present']}, distinct "
              f"classes: {record['distinct_predicted_classes']})")
        print(f"           index_select -> {record['selected_dtype']} "
              f"{record['selected_shape']} from logits "
              f"{record['logit_shape']}  (axis selection, not a per-row "
              f"gather: column j is the whole source column predictions[j])")
    print(f"  every diagonal is the predicted-class logit: "
          f"{indexing['every_diagonal_is_the_predicted_logit']}")
    print(f"  every column matches its source column: "
          f"{indexing['every_column_matches_its_source_column']}  "
          f"(duplicate columns identical: "
          f"{indexing['duplicate_columns_are_identical']})")
    print(f"  predictions are int64 plain leaves, owning and contiguous: "
          f"{indexing['every_prediction_is_int64']} / "
          f"{indexing['every_prediction_is_a_plain_leaf']} / "
          f"{indexing['every_prediction_owns_contiguous_storage']}")
    print(f"  selections keep the source dtype, own fresh graph-free "
          f"storage: "
          f"{indexing['every_selection_keeps_the_source_dtype']} / "
          f"{indexing['every_selection_is_a_fresh_graph_free_copy']}")
    print(f"  evaluated at steps {indexing['evaluated_steps']} — before the "
          f"checkpoint {indexing['evaluated_before_split']}, after it "
          f"{indexing['evaluated_after_split']} "
          f"({indexing['evaluations_on_both_sides']})")
    print(f"  duplicate predicted classes occurred: "
          f"{indexing['duplicates_occurred']} (guaranteed by pigeonhole, "
          f"{BATCH_SIZE} predictions over {NUM_CLASSES} classes: "
          f"{indexing['duplicates_guaranteed_by_pigeonhole']})")
    print()
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
    print(f"  parameters / Adam m, v:       {proof['parameters_match']} / "
          f"{proof['moments_match']} (step counters "
          f"{proof['counters_match']} {proof['step_counts']})")
    print(f"  prediction indices exact:     "
          f"{proof['prediction_indices_match']} (exact integer equality — no "
          f"tolerance, and no index is ever read at a floating width)")
    print(f"  selected values bit-exact:    "
          f"{proof['selected_bits_match']} (diagonal "
          f"{proof['diagonal_bits_match']}, whole evaluation record "
          f"{proof['whole_evaluation_record_matches']})")
    print(f"  final loader state_dict:      "
          f"{proof['final_loader_state_matches']} "
          f"(epoch {proof['final_loader_position'][0]}, cursor "
          f"{proof['final_loader_position'][1]})")
    print(f"  training actually moved:      parameters "
          f"{training['parameters_moved']}, Adam moments "
          f"{training['moments_became_nonzero']} (zero at the start: "
          f"{training['optimizer_state_was_empty_at_the_start']}), logits "
          f"{training['logits_changed_over_training']}, loss varies "
          f"{training['loss_sequence_varies']}")
    print(f"  omitting loader restore diverges: indices "
          f"{proof['omitted_indices_differ']}, losses "
          f"{proof['omitted_losses_differ']}, parameters "
          f"{proof['omitted_parameters_differ']}, integer evaluations "
          f"{proof['omitted_evaluations_differ']}")
    print(f"  every value at {dtype} on cpu: "
          f"{proof['all_state_at_run_dtype']} / {proof['all_state_on_cpu']}")


def main():
    if not cpp.is_available():
        print("The experimental C++ backend is not built.")
        print(cpp.build_instructions())
        return

    print(f"deterministic native integer indexing "
          f"(Linear({FEATURES}->{HIDDEN}) -> ReLU -> "
          f"Linear({HIDDEN}->{NUM_CLASSES}))")
    print(f"NativeCrossEntropyLoss over raw logits, NativeAdam "
          f"(lr={DEFAULT_LR}), {TOTAL_STEPS} steps interrupted after "
          f"{SPLIT_STEP}, over NativeTensorDataset -> NativeBatchSampler -> "
          f"NativeDataLoader ({SAMPLES} samples, {NUM_CLASSES} classes, "
          f"shuffled batches of {BATCH_SIZE})")
    print(f"compute dtypes: {cpp.SUPPORTED_DTYPES}   "
          f"index dtypes: {cpp.INDEX_DTYPES}   "
          f"devices: {cpp.SUPPORTED_DEVICES}")
    print("int64 is an index/result dtype in its own registry, never a "
          "supported compute dtype: nothing here computes at int64, and "
          "NativeTensor.from_int64_array stays the one public integer door")

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
    print("native argmax produced int64 predictions and native index_select "
          "consumed them, over a detached graph-free source; neither joined "
          "the training graph and neither is differentiable")
    print("index_select selects one index vector along the class axis for "
          "every row, so a (6, 4) logits batch gives a (6, 6) result whose "
          "diagonal is each example's own predicted-class logit — it is NOT "
          "a per-row gather, and TensorForge has no gather, scatter, or "
          "embedding")
    print("cross entropy trained on the loader's read-only host int64 target "
          "arrays at both widths; no native integer tensor is ever used as a "
          "target, a parameter, a buffer, optimizer state, or a checkpoint "
          "entry")
    print("checkpoint format: tensorforge.native_checkpoint, version 3 — "
          "unchanged, with no loader field and no version 4")
    print("the loader position travels as ORDINARY CALLER METADATA under "
          f"{TRAINING_KEY!r} / {LOADER_KEY!r}; no runtime code knows those "
          "names, and there is no automatic loader discovery")
    print("save order: loader.state_dict() -> no iteration -> "
          "save_native_checkpoint;  restore order: load_native_checkpoint "
          "-> loader.load_state_dict  (no cross-object atomicity)")
    print("comparison mechanism: exact Python integer equality for every "
          "prediction index, and raw IEEE-754 bit patterns (uint32 at "
          "float32, uint64 at float64) for every floating value — no "
          "tolerance, no allclose, and no float32-versus-float64 numeric "
          "comparison anywhere")
    print(f"identical across dtypes (dtype-independent only): batch indices "
          f"{cross['index_sequences_match']}, permutations "
          f"{cross['permutations_match']}, positions "
          f"{cross['positions_match']}, evaluation steps "
          f"{cross['evaluation_steps_match']}, selection shapes "
          f"{cross['selection_shapes_match']}")
    print(f"observed, not required: the two widths predicted the same "
          f"classes at every evaluation: {cross['prediction_indices_agree']} "
          f"— each run is still proved only against itself")
    print(f"live native storage baseline / final: {baseline} / {final_live}")
    print("this is an integration proof on one fixed task: no timing or "
          "performance is claimed or measured anywhere")

    if final_live != baseline:
        failures["lifecycle"] = [
            f"live native storage {baseline} -> {final_live}"
        ]

    for dtype in RUN_DTYPES:
        status = "no" if dtype in failures else "yes"
        print(f"exact native integer indexing resume at {dtype}: {status}")

    if failures:
        for scope, broken in failures.items():
            print(f"FAILED [{scope}]: {', '.join(broken)}")
        raise SystemExit("the resumed run diverged from the uninterrupted run")
    print("native argmax + index_select evaluation with exact interrupted "
          "resume ok at float64 and float32")


if __name__ == "__main__":
    main()
