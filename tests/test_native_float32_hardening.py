"""Cross-cutting float32/float64 hardening (Phase I, milestone I10).

I1 through I9 built the dtype-general native stack one layer at a time,
and each milestone proved its own layer. **I10 proves the seams between
them**, adversarially, at both widths, and adds no capability at all: no
operation, module, loss, optimizer, dtype, device, checkpoint version,
optimizer-state version, C ABI symbol, or dependency.

What this module covers, and why each part is here rather than in the
milestone module that shipped the feature:

1. **The mixed-dtype rejection authority map** (design §9.2) — every
   layer, every operand position *independently*. A guard that fires for
   a mismatched left operand and silently accepts a mismatched right one
   is a hole no single-position test can see.
2. **The C ABI as an independent authority** (design §9.2's
   defence-in-depth sentence). Python rejects mixed dtype before the
   call; C++ rejects it again at the trust boundary. Those are two
   authorities, so they are proved *separately* — the C++ half is reached
   by forcing a mismatch production Python would never emit, through
   production geometry.
3. **Validation ordering** (design §20). Which of two simultaneous
   defects is reported is part of the contract; this module records the
   established order rather than normalizing every error into one
   message.
4. **Allocation and wrapper-failure cleanup at both widths** (design
   §20's table), through the pre-existing deterministic hooks. No second
   fault-injection framework is introduced.
5. **All four graph-owned saved-resource families in one float32 graph.**
   I9 exercised them honestly *across* a run — three on a training graph,
   the BatchNorm evaluation snapshots only on an evaluation graph — and
   said so. I10 builds the configuration in which all four genuinely
   coexist, and walks the whole graph to prove it rather than asserting
   it.
6. **Concurrency**, strictly as the contract already claims it (design
   §21): participating state-replacement operations serialize; ordinary
   training mutation does not participate and is still not safe to race.
   **I10 adds no thread-safety promise.**
7. **Stable/native isolation**, re-proved now that float32 is public.
8. **ABI, registry, ctypes, and export inventories** reconciled.

The malformed-checkpoint matrix is large enough to own a file and lives
in ``tests/test_native_float32_checkpoint_corruption.py``.

**Nothing here weakens or replaces an I1-I9 test.** Those prove their own
layers; this adds adversarial coverage on top of them.

Contract: docs/native_dtype_float32_design.md §9 (no casting, no
promotion), §11 (autograd dtype invariants), §13 (buffers and the
winner-buffer exception), §20 (error handling and failure atomicity),
§21 (ownership, lifetime, concurrency), §26 (testing strategy).

Selector: python -m pytest -q -k native_float32_hardening
"""

import ctypes
import gc
import subprocess
import sys
import threading

import numpy as np
import pytest

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
    NativeMSELoss,
    NativeParameter,
    NativeReLU,
    NativeSequential,
    NativeSGD,
    NativeTensor,
)
from tensorforge.experimental import _native_state_lock

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="the experimental C++ backend is not built"
)
needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="deterministic allocation-failure injection is not compiled in",
)

BOTH_DTYPES = ("float64", "float32")
OTHER_DTYPE = {"float64": "float32", "float32": "float64"}
NUMPY_DTYPES = {"float64": np.float64, "float32": np.float32}
BIT_DTYPES = {"float64": np.uint64, "float32": np.uint32}

# Bounded joins everywhere a thread is started. A timeout here is a
# **deadlock detector**, not a performance requirement: the work under it
# is microseconds, so any expiry means an ordering defect, never a slow
# machine.
JOIN_TIMEOUT = 30.0


# ==========================================================================
# Helpers
# ==========================================================================


def bits(array, dtype):
    """Raw IEEE-754 bit patterns, with the dtype asserted rather than
    coerced. A helper that quietly converted could report a match that
    existed only after a conversion this runtime does not perform — which
    is precisely the bug every test in this file is hunting."""
    array = np.asarray(array)
    assert array.dtype == NUMPY_DTYPES[dtype], (array.dtype, dtype)
    return np.ascontiguousarray(array).reshape(-1).view(
        BIT_DTYPES[dtype]).tolist()


def core(values, dtype):
    """A ``NativeTensorCore`` at exactly ``dtype``, from host values.

    The host array is built at the *target* dtype so nothing here depends
    on ingress conversion; the dtype is then asserted, so a helper can
    never be the thing that made a test pass."""
    array = np.asarray(values, dtype=NUMPY_DTYPES[dtype])
    built = cpp.NativeTensorCore.from_array(array, dtype=dtype)
    assert built.dtype == dtype
    return built


def tensor(values, dtype, requires_grad=False):
    """A public ``NativeTensor`` at exactly ``dtype``."""
    array = np.asarray(values, dtype=NUMPY_DTYPES[dtype])
    built = NativeTensor.from_array(array, dtype=dtype,
                                    requires_grad=requires_grad)
    assert built.dtype == dtype
    return built


def close_all(*objects):
    """Close everything that can be closed, whatever else failed."""
    for item in objects:
        if item is not None and hasattr(item, "close"):
            try:
                item.close()
            except Exception:                       # pragma: no cover
                pass


def close_module(module):
    for parameter in module.parameters():
        close_all(parameter)
    for _, buffer in getattr(module, "named_buffers", lambda: ())():
        close_all(buffer)


@pytest.fixture()
def live_storages(monkeypatch):
    """The ids of every open ``NativeStorage``.

    This is the project's deterministic native-allocation instrumentation
    (the same fixture I9 uses): a rejection test can then prove *nothing
    was allocated* rather than trusting garbage collection to hide the
    evidence."""
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    monkeypatch.setattr(cpp.NativeStorage, "__init__", tracked_init)
    monkeypatch.setattr(cpp.NativeStorage, "close", tracked_close)
    return open_ids


def graph_resource_inventory(root):
    """Every graph-owned saved resource reachable from ``root``.

    Walks the autograd history and collects each node's private
    ``_graph_resources``, tagged with the **operation that created the
    node holding it**. Provenance rather than shape-guessing is what makes
    "all four families are present" an observation instead of a
    coincidence of dimensions.

    **Test-only introspection.** The four families are deliberately
    private, no production API exposes them, and none is added here.
    Returns ``(op, resource, dtype, shape)`` tuples."""
    seen, stack, found = set(), [root], []
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        op = getattr(node, "_op", "") or ""
        for resource in getattr(node, "_graph_resources", ()) or ():
            found.append((op, resource, resource.dtype, resource.shape))
        for parent in getattr(node, "_parents", ()) or ():
            stack.append(parent)
    return found


# The four graph-owned families, keyed by the op of the node that adopted
# them. BatchNorm's evaluation snapshots are the one family with no
# operation of its own: the normalization forward is a *composition*, so
# its snapshots ride on whichever arithmetic node ends it. Everything that
# is not one of the three named operations is therefore a snapshot — and
# the negative control below proves that bucket really does empty when
# BatchNorm is in training mode.
RESOURCE_OPS = {"dropout": "dropout_mask",
                "maxpool2d": "winner",
                "cross_entropy": "probabilities"}


def close_graph(root):
    """Close every node reachable from ``root``, and return how many.

    ``close()`` releases the resources of **the node it is called on** —
    it deliberately does not walk the ancestry, because a parent may be a
    tensor the caller still owns. An *abandoned* graph therefore has to be
    released node by node, and a module forward creates intermediates the
    caller never sees (BatchNorm's centered value, ``NativeLinear``'s
    pre-bias product), so the walk is how a test reaches them.

    This is the ownership model as written, not a workaround for it: the
    deterministic release points are a one-shot ``backward()``'s history
    release and ``close()``; ``__del__`` is the fallback, and a test that
    leaned on it would be measuring the collector.

    Leaves are deliberately skipped: a leaf is the caller's input or one
    of the model's registered parameters, and neither belongs to the
    graph. Closing them would be closing the model."""
    seen, stack, closed = set(), [root], 0
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        for parent in getattr(node, "_parents", ()) or ():
            stack.append(parent)
        if getattr(node, "_is_leaf", True):
            continue                    # an input or a registered parameter
        node.close()
        closed += 1
    return closed


def is_closed(obj):
    """Whether a saved resource has been released.

    The four families are a mix of ``NativeTensorCore`` (private
    ``_closed``) and ``NativeTensor`` (public ``closed``), so the question
    is asked of whichever the object actually is."""
    if hasattr(obj, "closed"):
        return obj.closed
    return obj._closed


def _storage_id(obj):
    """The identity of whatever native storage ``obj`` stands on.

    Saved resources are a mix of ``NativeTensorCore`` and ``NativeTensor``
    (each family adopts whichever it produced), so aliasing has to be
    compared at the storage rather than at the wrapper."""
    core_obj = getattr(obj, "_core", obj)
    return id(core_obj._storage)


def classify_resources(inventory):
    """Split an inventory into the four families by provenance."""
    families = {"dropout_mask": [], "winner": [], "probabilities": [],
                "bn_snapshot": []}
    for op, resource, dtype, shape in inventory:
        families[RESOURCE_OPS.get(op, "bn_snapshot")].append(
            (resource, dtype, shape))
    return families


class ThreadRunner:
    """Threads with captured exceptions and a bounded join.

    A worker that raises inside ``threading.Thread`` prints a traceback
    and is otherwise invisible to pytest, so every failure would read as
    a pass. Exceptions are captured and re-raised on the main thread."""

    def __init__(self):
        self.errors = []
        self._threads = []
        self._guard = threading.Lock()

    def start(self, target, name=None):
        def wrapper():
            try:
                target()
            except BaseException as error:          # noqa: BLE001
                with self._guard:
                    self.errors.append(error)

        thread = threading.Thread(target=wrapper, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()
        return thread

    def join(self):
        for thread in self._threads:
            thread.join(JOIN_TIMEOUT)
            assert not thread.is_alive(), (
                f"thread {thread.name!r} did not finish within "
                f"{JOIN_TIMEOUT}s — this is a deadlock detector, not a "
                f"performance budget"
            )
        if self.errors:
            raise self.errors[0]


# ==========================================================================
# 1. The mixed-dtype rejection authority map — Python layer
#
#    Design §9.2 lists the sites. This section reaches each one and, where
#    an authority has more than one operand position, reaches **each
#    position independently and in both directions**.
# ==========================================================================


@needs_native
@pytest.mark.parametrize("operation", ["add", "subtract", "multiply",
                                       "matmul"])
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_core_binary_rejects_a_mismatch_in_either_operand_position(
    operation, dtype, live_storages
):
    """``NativeTensorCore._require_matching_metadata`` is one guard shared
    by the whole binary family, so the risk is not that it is missing but
    that a *position* escapes it. Both orders are exercised at both
    dtypes: a guard that only compared "self against other" in one
    direction would pass a one-sided test."""
    same = core(np.ones((4, 4)), dtype)
    other = core(np.ones((4, 4)), OTHER_DTYPE[dtype])
    try:
        baseline = len(live_storages)
        for left, right in ((same, other), (other, same)):
            with pytest.raises(ValueError, match="matching dtype") as caught:
                getattr(left, operation)(right)
            message = str(caught.value)
            # Both dtypes are named, so the message identifies the
            # disagreement rather than merely reporting one.
            assert "float32" in message and "float64" in message
            assert len(live_storages) == baseline, (operation, dtype)
        assert same.dtype == dtype and other.dtype == OTHER_DTYPE[dtype]
        assert same._closed is False and other._closed is False
    finally:
        close_all(same, other)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_relu_backward_rejects_a_mismatched_upstream(dtype, live_storages):
    """``relu_backward`` is the one elementwise export whose second
    operand is a *gradient* rather than a peer, and I3 generalized it as a
    forward-shaped numerical primitive. It gets the same guard."""
    x = core([[1.0, -2.0], [3.0, -4.0]], dtype)
    upstream = core([[1.0, 1.0], [1.0, 1.0]], OTHER_DTYPE[dtype])
    try:
        baseline = len(live_storages)
        with pytest.raises(ValueError, match="matching dtype"):
            x.relu_backward(upstream)
        assert len(live_storages) == baseline
    finally:
        close_all(x, upstream)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_pooling_backward_has_exactly_one_value_operand(dtype):
    """Recorded because an absence is easy to mistake for a gap.

    ``maxpool2d_backward`` takes an upstream gradient and the winner
    buffer, and **only the upstream is a value tensor** — the output
    follows it, and the winner buffer is float64 metadata whatever the
    values are. So there is no second value position for a mixed-dtype
    rule to govern here, and the operation's one dtype authority is the
    winner-buffer check covered by the two tests that follow.

    Proved by construction: the same winner buffer drives a correct
    backward at **either** value width."""
    values = np.arange(16, dtype=NUMPY_DTYPES[dtype]).reshape(1, 1, 4, 4)
    x = core(values, dtype)
    pooled = winners = None
    try:
        pooled, winners = x._maxpool2d_forward_with_winners(
            kernel_size=2, stride=2, padding=0)
        assert winners.dtype == "float64"
        for value_dtype in BOTH_DTYPES:
            upstream = core(np.ones((1, 1, 2, 2)), value_dtype)
            try:
                grad = upstream.maxpool2d_backward(
                    winners, input_shape=(1, 1, 4, 4))
                try:
                    assert grad.dtype == value_dtype, (
                        "the gradient follows the upstream, and the "
                        "float64 winner buffer does not drag it along"
                    )
                finally:
                    grad.close()
            finally:
                upstream.close()
    finally:
        close_all(x, pooled, winners)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_maxpool_winner_buffer_is_float64_at_every_value_dtype(dtype):
    """Design §13.3, stated as the exception it is.

    The winner buffer holds **offsets**, not values, and stays private
    float64 at every value dtype so that a float32 pool over a plane
    beyond binary32's ``2**24`` exact-integer range still records an exact
    offset. This is not a mixed-dtype defect and must not be "fixed": the
    value tensors agree with each other, and the winner buffer is
    deliberately outside that agreement."""
    values = np.arange(16, dtype=NUMPY_DTYPES[dtype]).reshape(1, 1, 4, 4)
    x = core(values, dtype)
    pooled = winners = None
    try:
        pooled, winners = x._maxpool2d_forward_with_winners(
            kernel_size=2, stride=2, padding=0)
        assert pooled.dtype == dtype, "the value output follows the input"
        assert winners.dtype == "float64", (
            "the winner buffer is private float64 metadata at every value "
            "dtype (design §13.3)"
        )
        # ...and the backward accepts that pairing rather than rejecting it,
        # so the exception is proved in both directions.
        upstream = core(np.ones((1, 1, 2, 2)), dtype)
        try:
            grad = upstream.maxpool2d_backward(
                winners, input_shape=(1, 1, 4, 4))
            try:
                assert grad.dtype == dtype
            finally:
                grad.close()
        finally:
            upstream.close()
    finally:
        close_all(x, pooled, winners)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_wrong_dtype_winner_buffer_is_rejected(dtype, live_storages):
    """The winner buffer is float64 *beside* the value dtype, never
    *against* it — so a float32 winner buffer is rejected even when the
    gradient is float32 too. Without this, "the winner buffer is
    metadata" would degrade into "the winner buffer is whatever."""
    upstream = core(np.ones((1, 1, 2, 2)), dtype)
    wrong = core(np.zeros((1, 1, 2, 2)), "float32")
    try:
        baseline = len(live_storages)
        with pytest.raises(ValueError):
            upstream.maxpool2d_backward(wrong, input_shape=(1, 1, 4, 4))
        assert len(live_storages) == baseline
    finally:
        close_all(upstream, wrong)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_conv2d_core_rejects_a_mismatch_at_weight_and_at_bias(
    dtype, live_storages
):
    """Convolution has three numeric inputs, and the bias is the one most
    easily forgotten because it is optional. Each is mismatched on its
    own, with the others correct."""
    other = OTHER_DTYPE[dtype]
    x = core(np.ones((1, 1, 4, 4)), dtype)
    weight_ok = core(np.ones((2, 1, 3, 3)), dtype)
    weight_bad = core(np.ones((2, 1, 3, 3)), other)
    bias_ok = core(np.zeros(2), dtype)
    bias_bad = core(np.zeros(2), other)
    try:
        baseline = len(live_storages)
        with pytest.raises(ValueError):
            x.conv2d_forward(weight_bad, bias_ok, padding=1)
        assert len(live_storages) == baseline, "mismatched weight"
        with pytest.raises(ValueError):
            x.conv2d_forward(weight_ok, bias_bad, padding=1)
        assert len(live_storages) == baseline, "mismatched bias"
        # The control: all three agreeing really does succeed, so the two
        # rejections above are about dtype and not about the geometry.
        out = x.conv2d_forward(weight_ok, bias_ok, padding=1)
        try:
            assert out.dtype == dtype
        finally:
            out.close()
    finally:
        close_all(x, weight_ok, weight_bad, bias_ok, bias_bad)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_conv2d_backward_directions_reject_their_mismatched_partner(
    dtype, live_storages
):
    """Both backward directions take two numeric operands, and each has
    its own export and its own guard."""
    other = OTHER_DTYPE[dtype]
    upstream = core(np.ones((1, 2, 4, 4)), dtype)
    weight = core(np.ones((2, 1, 3, 3)), other)
    x = core(np.ones((1, 1, 4, 4)), other)
    try:
        baseline = len(live_storages)
        with pytest.raises(ValueError):
            upstream.conv2d_input_backward(
                weight, input_shape=(1, 1, 4, 4), padding=1)
        assert len(live_storages) == baseline, "input-backward"
        with pytest.raises(ValueError):
            upstream.conv2d_weight_backward(
                x, weight_shape=(2, 1, 3, 3), padding=1)
        assert len(live_storages) == baseline, "weight-backward"
    finally:
        close_all(upstream, weight, x)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_cross_entropy_backward_rejects_a_mismatched_probability_set(
    dtype, live_storages
):
    """The fused backward reads the **saved probabilities** and an
    upstream gradient, and cannot even name the logits. Both of the two
    numeric handles it does read are covered."""
    other = OTHER_DTYPE[dtype]
    logits = core([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype)
    targets = np.asarray([0, 2], dtype=np.int64)
    forward = None
    try:
        forward = logits.cross_entropy_forward(targets)
        assert forward.probabilities.dtype == dtype, (
            "the saved probabilities carry the graph dtype"
        )
        assert forward.loss.dtype == dtype
        upstream = core([1.0], other)
        try:
            baseline = len(live_storages)
            with pytest.raises(ValueError):
                forward.probabilities.cross_entropy_backward(
                    targets, upstream)
            assert len(live_storages) == baseline
        finally:
            upstream.close()
    finally:
        close_all(logits, forward)


@needs_native
def test_cross_entropy_targets_stay_host_int64_at_both_widths():
    """Design §16 of the classification contract, re-proved at the seam:
    the class targets are **host int64 metadata** at every value width.
    They carry no tensor dtype, gain none, and are never inferred from the
    logits — so there is no integer tensor dtype for a mixed-dtype rule to
    apply to."""
    for dtype in BOTH_DTYPES:
        logits = core([[1.0, 2.0, 3.0]], dtype)
        try:
            # An integer host array is accepted and normalized to int64
            # **host metadata** — the same explicit host-ingress boundary
            # §9.4 describes for values, and not a tensor dtype.
            for accepted in (np.asarray([1], dtype=np.int64),
                             np.asarray([1], dtype=np.int32),
                             [1]):
                close_all(logits.cross_entropy_forward(accepted))
            # A float or bool target is not a "float32 versus float64"
            # question at all: targets are class indices, so they are
            # rejected rather than converted.
            for bad in (np.asarray([1], dtype=np.float64),
                        np.asarray([1], dtype=np.float32),
                        np.asarray([1], dtype=np.bool_)):
                with pytest.raises((TypeError, ValueError)):
                    logits.cross_entropy_forward(bad)
            # ...and no integer tensor dtype exists for them to become.
            assert "int64" not in cpp.SUPPORTED_DTYPES
            assert "int32" not in cpp.SUPPORTED_DTYPES
            with pytest.raises((TypeError, ValueError)):
                cpp.NativeStorage(4, dtype="int64")
        finally:
            logits.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_module_forward_authorities_reject_a_mismatched_input(
    dtype, live_storages
):
    """Every state-owning module validates its input against its own
    parameters before building a graph node. One table, so a module added
    later without a guard shows up as a missing row rather than as silent
    acceptance."""
    other = OTHER_DTYPE[dtype]
    cases = [
        ("NativeLinear", NativeLinear(3, 2, seed=0, dtype=dtype),
         [[1.0, 2.0, 3.0]]),
        ("NativeConv2d", NativeConv2d(1, 2, 3, padding=1, seed=0,
                                      dtype=dtype),
         np.ones((1, 1, 4, 4))),
        ("NativeLayerNorm", NativeLayerNorm(3, dtype=dtype),
         [[1.0, 2.0, 3.0]]),
        ("NativeBatchNorm1d", NativeBatchNorm1d(3, dtype=dtype),
         [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
        ("NativeBatchNorm2d", NativeBatchNorm2d(2, dtype=dtype),
         np.ones((2, 2, 2, 2))),
    ]
    for label, module, values in cases:
        x = tensor(values, other)
        try:
            baseline = len(live_storages)
            versions = [p.version for p in module.parameters()]
            buffers = [(name, b.to_numpy().copy())
                       for name, b in module.named_buffers()]
            with pytest.raises(ValueError) as caught:
                module(x)
            assert dtype in str(caught.value), label
            assert other in str(caught.value), label
            assert len(live_storages) == baseline, label
            assert [p.version for p in module.parameters()] == versions, label
            for (name, before), (_, after) in zip(
                buffers, [(n, b.to_numpy()) for n, b in module.named_buffers()]
            ):
                assert np.array_equal(before, after), (label, name)
        finally:
            close_all(x)
            close_module(module)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_mse_loss_rejects_a_mismatch_in_either_position(dtype,
                                                        live_storages):
    other = OTHER_DTYPE[dtype]
    for prediction_dtype, target_dtype in ((dtype, other), (other, dtype)):
        prediction = tensor([[1.0, 2.0]], prediction_dtype)
        target = tensor([[1.0, 2.0]], target_dtype)
        try:
            baseline = len(live_storages)
            with pytest.raises(ValueError, match="matching dtype"):
                NativeMSELoss()(prediction, target)
            assert len(live_storages) == baseline
        finally:
            close_all(prediction, target)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_sequential_rejects_at_the_mismatched_child_and_bridges_nothing(
    dtype, live_storages
):
    """A container takes no dtype and enforces none, so a model may hold
    both widths — and the failure surfaces **at the child that disagrees**,
    with no implicit bridge inserted between children.

    The intermediate the first child produced is released when the raising
    call's frames go away, without a ``gc.collect()``. Note the deliberate
    plain ``try``/``except``: ``pytest.raises`` keeps the traceback alive,
    which keeps ``forward``'s locals alive, so measuring live storage
    inside that block would measure pytest rather than TensorForge."""
    other = OTHER_DTYPE[dtype]
    model = NativeSequential(
        NativeLinear(3, 4, seed=0, dtype=dtype),
        NativeReLU(),
        NativeLinear(4, 2, seed=1, dtype=other),
    )
    x = tensor([[1.0, 2.0, 3.0]], dtype)
    try:
        baseline = len(live_storages)
        message = None
        try:
            model(x)
        except ValueError as error:
            message = str(error)
        assert message is not None, "the mismatched child did not raise"
        # The message names the child that refused, not the container.
        assert "NativeLinear" in message
        assert other in message and dtype in message
        assert len(live_storages) == baseline, (
            "the first child's output must not outlive the failed forward"
        )
        assert not hasattr(model, "dtype"), (
            "a container owns no dtype and must not acquire one"
        )
    finally:
        close_all(x)
        for parameter in model.parameters():
            close_all(parameter)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_parameter_mutation_authorities_reject_a_mismatched_source(
    dtype, live_storages
):
    """``copy_value_`` and ``_adopt_value_core`` are the two ways a
    parameter's value can be replaced, and both validate dtype before
    anything moves. Identity, version, and value all survive the
    rejection."""
    other = OTHER_DTYPE[dtype]
    parameter = NativeParameter([[1.0, 2.0]], dtype=dtype)
    source = tensor([[3.0, 4.0]], other)
    replacement = core([[3.0, 4.0]], other)
    try:
        baseline = len(live_storages)
        identity = id(parameter)
        version = parameter.version
        before = parameter.to_numpy().copy()

        with pytest.raises(ValueError, match="no casting"):
            parameter.copy_value_(source)
        with pytest.raises(ValueError, match="metadata mismatch"):
            parameter._adopt_value_core(replacement)

        assert len(live_storages) == baseline
        assert id(parameter) == identity
        assert parameter.version == version
        assert bits(parameter.to_numpy(), dtype) == bits(before, dtype)
        assert parameter.dtype == dtype
    finally:
        close_all(parameter, source, replacement)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_parameter_refuses_to_adopt_a_live_tensor_of_the_other_dtype(
    dtype
):
    """There is no tensor cast, so a live ``NativeTensor`` of the other
    width is a rejection rather than a conversion — while **host** data at
    any NumPy dtype converts once at the ingress boundary (design §9.4).
    Both halves are asserted, because the difference between them is the
    whole of §9.4."""
    other = OTHER_DTYPE[dtype]
    live = tensor([[1.0, 2.0]], other)
    try:
        with pytest.raises(ValueError, match="no casting"):
            NativeParameter(live, dtype=dtype)
        # ...and the host path at the same "wrong" NumPy dtype is fine,
        # because it is a documented conversion boundary and not a cast.
        host = np.asarray([[1.0, 2.0]], dtype=NUMPY_DTYPES[other])
        converted = NativeParameter(host, dtype=dtype)
        try:
            assert converted.dtype == dtype
        finally:
            converted.close()
    finally:
        close_all(live)


@needs_native
@pytest.mark.parametrize("optimizer_class", [NativeSGD, NativeAdam])
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_an_optimizer_rejects_a_mismatched_gradient_and_commits_nothing(
    optimizer_class, dtype, live_storages
):
    """One invalid entry prevents **every** commit, including the valid
    ones beside it: the optimizer validates in complete passes before it
    mutates anything (design §15.2). A per-parameter loop that updated the
    good entry and then raised would pass a single-parameter test."""
    other = OTHER_DTYPE[dtype]
    good = NativeParameter([[1.0, 2.0]], dtype=dtype)
    bad = NativeParameter([[3.0, 4.0]], dtype=dtype)
    good_grad = tensor([[0.5, 0.5]], dtype)
    bad_grad = tensor([[0.5, 0.5]], other)
    optimizer = optimizer_class([good, bad], lr=0.1)
    try:
        good._grad = good_grad
        bad._grad = bad_grad
        baseline = len(live_storages)
        versions = (good.version, bad.version)
        values = (good.to_numpy().copy(), bad.to_numpy().copy())

        with pytest.raises(ValueError) as caught:
            optimizer.step()
        assert "dtype" in str(caught.value)

        assert len(live_storages) == baseline
        assert (good.version, bad.version) == versions, (
            "the valid parameter beside the invalid one must not commit"
        )
        assert bits(good.to_numpy(), dtype) == bits(values[0], dtype)
        assert bits(bad.to_numpy(), dtype) == bits(values[1], dtype)
        # The gradients are untouched too — a rejected step consumes
        # nothing.
        assert good._grad is good_grad and bad._grad is bad_grad
    finally:
        close_all(optimizer if hasattr(optimizer, "close") else None)
        close_all(good_grad, bad_grad, good, bad)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_adam_rejects_a_moment_that_drifted_from_its_parameter(
    dtype, live_storages
):
    """Adam's ``m``/``v`` must match their parameter in dtype, and the
    check is against the *parameter*, so a coherent-looking pair that
    agrees with itself but not with the parameter is still refused."""
    other = OTHER_DTYPE[dtype]
    parameter = NativeParameter([[1.0, 2.0]], dtype=dtype)
    parameter._grad = tensor([[0.5, 0.5]], dtype)
    optimizer = NativeAdam([parameter], lr=0.1)
    try:
        optimizer.step()                     # builds the moments
        assert optimizer._m[0].dtype == dtype and optimizer._v[0].dtype == dtype
        original_m = optimizer._m[0]
        drifted = NativeTensor.zeros((1, 2), dtype=other)
        try:
            optimizer._m[0] = drifted        # the one seam a test can set
            baseline = len(live_storages)
            version = parameter.version
            values = parameter.to_numpy().copy()
            with pytest.raises(ValueError) as caught:
                optimizer.step()
            message = str(caught.value)
            assert "m state" in message and "parameter is" in message
            assert dtype in message and other in message
            assert len(live_storages) == baseline
            assert parameter.version == version
            assert bits(parameter.to_numpy(), dtype) == bits(values, dtype)
        finally:
            optimizer._m[0] = original_m
            drifted.close()
    finally:
        close_all(parameter._grad, optimizer, parameter)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_module_state_loading_rejects_one_mismatched_entry_among_many(
    dtype, live_storages
):
    """``load_state_dict`` is a transaction: one bad entry among several
    good ones rolls the whole load back, leaving identities, versions, and
    values exactly as they were."""
    other = OTHER_DTYPE[dtype]
    destination = NativeLinear(3, 2, seed=0, dtype=dtype)
    donor = NativeLinear(3, 2, seed=5, dtype=dtype)
    stray = NativeParameter(np.zeros((2,)), dtype=other)
    try:
        state = donor.state_dict()
        state["bias"] = stray                    # exactly one bad entry
        baseline = len(live_storages)
        identities = {n: id(p) for n, p in destination.named_parameters()}
        versions = {n: p.version for n, p in destination.named_parameters()}
        before = {n: p.to_numpy().copy()
                  for n, p in destination.named_parameters()}

        with pytest.raises(ValueError, match="dtype"):
            destination.load_state_dict(state)

        assert len(live_storages) == baseline
        for name, parameter in destination.named_parameters():
            assert id(parameter) == identities[name], name
            assert parameter.version == versions[name], name
            assert bits(parameter.to_numpy(), dtype) == bits(
                before[name], dtype), name
        close_all(*state.values())
    finally:
        close_all(stray)
        close_module(destination)
        close_module(donor)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_batchnorm_two_buffer_transaction_refuses_a_drifted_buffer(
    dtype, live_storages
):
    """Design §13: all four numeric state objects describe one statistical
    state, so a running buffer that has drifted from the module's dtype is
    rejected **before** either buffer can move — never half-way through
    the two-buffer transaction."""
    other = OTHER_DTYPE[dtype]
    module = NativeBatchNorm1d(3, dtype=dtype)
    module.train(True)
    x = tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype)
    original_entry = module._buffers["running_var"]
    drifted = NativeTensor.zeros((3,), dtype=other)
    try:
        module._buffers["running_var"] = original_entry._replace(
            tensor=drifted)
        mean_before = module.running_mean.to_numpy().copy()
        baseline = len(live_storages)
        with pytest.raises(ValueError) as caught:
            module(x)
        assert "dtype" in str(caught.value)
        assert len(live_storages) == baseline
        assert np.array_equal(module.running_mean.to_numpy(), mean_before), (
            "running_mean must not move when running_var is rejected"
        )
    finally:
        module._buffers["running_var"] = original_entry
        drifted.close()
        close_all(x)
        close_module(module)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_autograd_rejects_a_mismatched_seed_gradient_before_any_leaf_moves(
    dtype, live_storages
):
    """Invariant 5 of design §11 at the *root*: an explicit
    ``backward(gradient=...)`` of the wrong dtype is refused before the
    traversal starts, so no leaf holds a partial gradient afterwards."""
    other = OTHER_DTYPE[dtype]
    x = tensor([[1.0, 2.0]], dtype, requires_grad=True)
    seed = tensor([[1.0, 1.0]], other)
    y = None
    try:
        y = x.multiply(x)
        baseline = len(live_storages)
        with pytest.raises(ValueError) as caught:
            y.backward(gradient=seed)
        assert "dtype" in str(caught.value)
        assert x.grad is None, "no leaf may hold a partially committed grad"
        assert len(live_storages) == baseline
        # ...and a corrected retry on the same graph succeeds, so the
        # rejection left the graph usable rather than poisoned.
        good = tensor([[1.0, 1.0]], dtype)
        try:
            y.backward(gradient=good)
            assert x.grad is not None and x.grad.dtype == dtype
        finally:
            good.close()
    finally:
        close_all(y, seed, x)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_gradient_accumulation_rejects_a_mismatched_contribution(
    dtype, live_storages
):
    """The accumulation guard, reached directly. A gradient already held
    is left exactly as it was — bit for bit — and the leaf stays usable."""
    other = OTHER_DTYPE[dtype]
    x = tensor([[1.0, 2.0]], dtype, requires_grad=True)
    first = tensor([[1.0, 1.0]], dtype)
    wrong = tensor([[1.0, 1.0]], other)
    try:
        x._accumulate_grad(first)
        before = x.grad.to_numpy().copy()
        baseline = len(live_storages)
        with pytest.raises(ValueError, match="matching dtype"):
            x._accumulate_grad(wrong)
        assert len(live_storages) == baseline
        assert bits(x.grad.to_numpy(), dtype) == bits(before, dtype)
        assert x.grad.dtype == dtype
    finally:
        close_all(wrong, x)


# ==========================================================================
# 2. The C ABI is an independent authority
#
#    Design §9.2: the C++ dispatch points confirm operand agreement "as a
#    defence-in-depth revalidation at the trust boundary". Python already
#    rejects these calls, so proving the C++ half means constructing a
#    mismatch production Python would never emit — through production
#    geometry, so the arguments are the real ones.
# ==========================================================================


def _force_output_dtype(monkeypatch, forced):
    """Make every derived output allocation land at ``forced``.

    The Core allocates its output through ``_uninitialized`` / ``zeros``
    with ``dtype=self.dtype``; overriding the dtype there yields a call
    whose operands agree and whose **destination** does not. That is
    unreachable from Python's own guard, which is the point: it isolates
    the C ABI's check from the Python one."""
    original_uninitialized = cpp.NativeTensorCore._uninitialized.__func__
    original_zeros = cpp.NativeTensorCore.zeros.__func__

    def uninitialized(cls, shape, dtype="float64", device="cpu"):
        return original_uninitialized(cls, shape, dtype=forced, device=device)

    def zeros(cls, shape, dtype="float64", device="cpu", **kwargs):
        kwargs["_trusted_dtype"] = True
        return original_zeros(cls, shape, dtype=forced, device=device,
                              **kwargs)

    monkeypatch.setattr(cpp.NativeTensorCore, "_uninitialized",
                        classmethod(uninitialized))
    monkeypatch.setattr(cpp.NativeTensorCore, "zeros", classmethod(zeros))


ABI_OUTPUT_CASES = (
    # (label, callable taking a same-dtype operand factory)
    ("relu", lambda make: make(np.ones((4, 4))).relu()),
    ("sqrt", lambda make: make(np.ones((4, 4))).sqrt()),
    ("reciprocal", lambda make: make(np.ones((4, 4))).reciprocal()),
    ("exp", lambda make: make(np.ones((4, 4))).exp()),
    ("log", lambda make: make(np.ones((4, 4))).log()),
    ("add", lambda make: make(np.ones((4, 4))).add(make(np.ones((4, 4))))),
    ("subtract",
     lambda make: make(np.ones((4, 4))).subtract(make(np.ones((4, 4))))),
    ("multiply",
     lambda make: make(np.ones((4, 4))).multiply(make(np.ones((4, 4))))),
    ("matmul",
     lambda make: make(np.ones((4, 4))).matmul(make(np.ones((4, 4))))),
    ("sum", lambda make: make(np.ones((4, 4))).sum()),
    ("mean", lambda make: make(np.ones((4, 4))).mean()),
    ("contiguous_copy", lambda make: make(np.ones((4, 4))).contiguous_copy()),
    ("softmax", lambda make: make(np.ones((2, 4))).softmax()),
    ("log_softmax", lambda make: make(np.ones((2, 4))).log_softmax()),
    ("conv2d_forward",
     lambda make: make(np.ones((1, 1, 4, 4))).conv2d_forward(
         make(np.ones((2, 1, 3, 3))), padding=1)),
)


@needs_native
@pytest.mark.parametrize("label, call", ABI_OUTPUT_CASES,
                         ids=[c[0] for c in ABI_OUTPUT_CASES])
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_c_abi_rejects_a_mismatched_destination_on_its_own(
    label, call, dtype, monkeypatch
):
    """Every dtype-general compute export revalidates at the boundary.

    The operands agree; only the destination disagrees — a state Python's
    own guard cannot produce, so a pass here is the **C++** guard and
    nothing else. The message is the C ABI's own wording, and both dtypes
    are named in it."""
    _force_output_dtype(monkeypatch, OTHER_DTYPE[dtype])
    made = []

    def make(values):
        built = core(values, dtype)
        made.append(built)
        return built

    try:
        with pytest.raises(ValueError) as caught:
            call(make)
        message = str(caught.value)
        assert "same dtype" in message, (label, message)
        assert "float32" in message and "float64" in message, (label, message)
        assert "no casting or promotion" in message, (label, message)
    finally:
        close_all(*made)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_c_abi_rejects_mismatched_operands_when_python_is_bypassed(
    dtype, monkeypatch
):
    """The two authorities are independent, proved by removing one.

    With ``_require_matching_metadata`` neutered, the Python layer accepts
    the call and the C ABI still refuses it. If the C++ guard were absent,
    this would silently compute — which is exactly the failure mode
    defence-in-depth exists to prevent."""
    monkeypatch.setattr(cpp.NativeTensorCore, "_require_matching_metadata",
                        lambda self, other, op_name: None)
    same = core(np.ones((4, 4)), dtype)
    other = core(np.ones((4, 4)), OTHER_DTYPE[dtype])
    try:
        for left, right in ((same, other), (other, same)):
            with pytest.raises(ValueError) as caught:
                left.add(right)
            assert "same dtype" in str(caught.value)
    finally:
        close_all(same, other)


@needs_native
def test_the_bypass_control_is_not_vacuous():
    """The negative control for the test above: with the Python guard
    neutered and **matching** dtypes, the very same call succeeds. So the
    rejection above came from the dtype disagreement, not from the
    monkeypatch having broken the call."""
    original = cpp.NativeTensorCore._require_matching_metadata
    cpp.NativeTensorCore._require_matching_metadata = (
        lambda self, other, op_name: None)
    a = core(np.ones((4, 4)), "float32")
    b = core(np.ones((4, 4)), "float32")
    try:
        out = a.add(b)
        try:
            assert out.dtype == "float32"
            assert np.array_equal(out.to_numpy(),
                                  np.full((4, 4), 2.0, dtype=np.float32))
        finally:
            out.close()
    finally:
        cpp.NativeTensorCore._require_matching_metadata = original
        close_all(a, b)


@needs_native
@pytest.mark.parametrize("call", ("sum", "matmul", "narrow_backward",
                                  "contiguous_copy", "relu", "relu_backward"))
def test_a_rejected_abi_call_leaves_the_destination_byte_identical(call):
    """A rejecting export **writes nothing**. Proved as raw bit patterns
    over a destination deliberately pre-filled with a recognizable value,
    so a partial write of even one element is visible."""
    library = cpp._require_library()
    narrow = cpp.NativeStorage._typed(16, "float32")
    wide = cpp.NativeStorage(16, dtype="float64")
    try:
        wide.fill(-7.5)
        before = wide.to_numpy().copy()
        shape = (ctypes.c_int64 * 2)(4, 4)
        strides = (ctypes.c_int64 * 2)(4, 1)
        writes = (ctypes.c_int64 * 2)(0, 1)
        a = narrow._require_open()
        dst = wide._require_open()
        library.tf_clear_error()
        with pytest.raises(ValueError, match="same dtype"):
            if call == "sum":
                library.tf_core_sum(a, dst, shape, strides, writes, 0, 2)
            elif call == "matmul":
                library.tf_core_matmul(a, dst, dst, 4, 4, 4, 4, 1, 4, 1, 0, 0)
            elif call == "narrow_backward":
                library.tf_core_narrow_backward(a, dst, shape, strides,
                                                strides, 0, 0, 2)
            elif call == "contiguous_copy":
                library.tf_core_contiguous_copy(a, dst, shape, strides, 0, 2)
            elif call == "relu":
                library.tf_core_relu(a, dst, shape, strides, 0, 2)
            else:
                library.tf_core_relu_backward(a, a, dst, shape, strides,
                                              strides, 0, 0, 2)
        assert bits(wide.to_numpy(), "float64") == bits(before, "float64")
        library.tf_clear_error()
    finally:
        close_all(wide, narrow)


# ==========================================================================
# 3. Validation ordering
#
#    Which defect is reported when a call is invalid in two ways at once
#    is part of the contract. This section **records** the established
#    order; it does not normalize errors into one message and does not
#    move a guard to make a test tidier.
# ==========================================================================


@needs_native
def test_a_closed_operand_is_reported_before_its_dtype():
    """Liveness first: a closed handle cannot be asked what dtype it is,
    so ``_require_open`` runs before the dtype comparison and the error
    names the closure."""
    live = core(np.ones((2, 2)), "float32")
    dead = core(np.ones((2, 2)), "float64")
    dead.close()
    try:
        with pytest.raises((RuntimeError, ValueError)) as caught:
            live.add(dead)
        assert "closed" in str(caught.value).lower()
    finally:
        live.close()


@needs_native
def test_a_non_tensor_operand_is_reported_before_its_dtype():
    """Type before value: a plain list has no dtype to disagree with."""
    live = core(np.ones((2, 2)), "float32")
    try:
        with pytest.raises(TypeError, match="NativeTensorCore"):
            live.add([[1.0, 2.0], [3.0, 4.0]])
    finally:
        live.close()


@needs_native
def test_dtype_is_reported_before_a_broadcast_shape_conflict():
    """Design §9.3 orders the dtype guard **before** span/shape
    validation, so a call that is both mixed-dtype and unbroadcastable
    reports the dtype. That ordering is what lets the rule "no allocation
    happens on a mixed-dtype call" be unconditional."""
    a = core(np.ones((4, 4)), "float32")
    b = core(np.ones((3, 7)), "float64")
    try:
        with pytest.raises(ValueError) as caught:
            a.add(b)
        assert "matching dtype" in str(caught.value)
    finally:
        close_all(a, b)


@needs_native
def test_copy_value_reports_shape_before_dtype():
    """The parameter mutation primitive established the opposite order,
    and it is recorded rather than changed: shape is compared first, so a
    source that is wrong in both ways names the shape. Both guards run
    before anything is staged, so the ordering costs no atomicity."""
    parameter = NativeParameter([[1.0, 2.0]], dtype="float32")
    source = tensor([[1.0, 2.0, 3.0]], "float64")
    try:
        with pytest.raises(ValueError, match="shape mismatch"):
            parameter.copy_value_(source)
    finally:
        close_all(parameter, source)


@needs_native
def test_a_module_reports_its_own_shape_rule_before_a_dtype_mismatch():
    """Each module states its own order. BatchNorm2d validates rank
    first — a 2-D input is not a BatchNorm2d input whatever its dtype —
    and the message says so."""
    module = NativeBatchNorm2d(2, dtype="float32")
    x = tensor([[1.0, 2.0]], "float64")
    try:
        with pytest.raises(ValueError) as caught:
            module(x)
        assert "shape" in str(caught.value)
    finally:
        close_all(x)
        close_module(module)


@needs_native
def test_the_seed_gradient_dtype_is_reported_before_graph_staleness():
    """The established order at ``backward()``, recorded rather than
    chosen: the seed gradient's dtype is validated against the output
    **before** the traversal begins, so a call that is both mixed-dtype
    and standing on a stale graph reports the dtype. The staleness rule
    fires on the next attempt, once the seed is right.

    Both orderings are defensible; this test exists so the one that ships
    is the one that is documented, and so a later milestone cannot quietly
    swap them."""
    parameter = NativeParameter([[2.0, 3.0]], dtype="float32")
    x = tensor([[1.0, 1.0]], "float32", requires_grad=True)
    replacement = tensor([[5.0, 5.0]], "float32")
    seed_wrong = tensor([[1.0, 1.0]], "float64")
    seed_right = tensor([[1.0, 1.0]], "float32")
    y = None
    try:
        y = x.multiply(parameter)
        parameter.copy_value_(replacement)          # the graph is now stale
        with pytest.raises(ValueError) as caught:
            y.backward(gradient=seed_wrong, retain_graph=True)
        assert "dtype" in str(caught.value)
        assert "stale" not in str(caught.value).lower()
        # ...and with a correctly typed seed the staleness rule is what
        # fires, so the second guard really is behind the first.
        with pytest.raises(RuntimeError) as stale:
            y.backward(gradient=seed_right, retain_graph=True)
        assert "stale" in str(stale.value).lower()
    finally:
        close_all(y, seed_wrong, seed_right, replacement, x, parameter)


@needs_native
def test_dropout_rejects_a_bad_probability_without_consuming_a_call():
    """A generator call is a scarce, ordered resource: a rejected Dropout
    forward must consume none. Proved on the *validation* failure, which
    is the case where an implementation is most tempted to reserve
    first and validate second."""
    generator = NativeGenerator(7)
    layer = NativeDropout(0.5, generator=generator)
    x = tensor([[1.0, 2.0, 3.0]], "float32")
    try:
        before = generator.calls
        for bad in (-0.1, 1.5, "0.5", None):
            with pytest.raises((ValueError, TypeError)):
                NativeDropout(bad, generator=generator)
            assert generator.calls == before
        # ...and a mismatched forward likewise burns nothing.
        wrong = tensor([[1.0, 2.0, 3.0]], "float64")
        try:
            layer.train(True)
            out = layer(x)          # the control: one call is consumed
            assert generator.calls == before + 1
            out.close()
        finally:
            wrong.close()
    finally:
        close_all(x)


# ==========================================================================
# 4. Allocation and wrapper-failure cleanup, at both widths
#
#    The deterministic hooks already in the tree, used as they are. No
#    second production fault-injection framework is introduced (§4.1 of
#    CLAUDE.md forbids one).
# ==========================================================================


@needs_native
@needs_fault_injection
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_an_injected_allocation_failure_leaves_live_storage_at_baseline(
    dtype, live_storages
):
    """Every public factory, at both widths, with the *first* native
    allocation failed. Live storage returns exactly to baseline and no
    partially published object escapes."""
    factories = (
        ("storage", lambda: cpp.NativeStorage(16, dtype=dtype)),
        ("core.zeros", lambda: cpp.NativeTensorCore.zeros((4, 4),
                                                          dtype=dtype)),
        ("core.full", lambda: cpp.NativeTensorCore.full((4, 4), 2.0,
                                                        dtype=dtype)),
        ("core.from_array",
         lambda: cpp.NativeTensorCore.from_array(
             np.ones((4, 4), dtype=NUMPY_DTYPES[dtype]), dtype=dtype)),
        ("tensor.zeros", lambda: NativeTensor.zeros((4, 4), dtype=dtype)),
        ("tensor.full", lambda: NativeTensor.full((4, 4), 2.0, dtype=dtype)),
        ("tensor.from_array",
         lambda: NativeTensor.from_array(
             np.ones((4, 4), dtype=NUMPY_DTYPES[dtype]), dtype=dtype)),
    )
    for label, factory in factories:
        baseline = len(live_storages)
        cpp._arm_alloc_failure(1)
        try:
            with pytest.raises(MemoryError):
                factory()
        finally:
            cpp._arm_alloc_failure(0)
        assert len(live_storages) == baseline, label
        # ...and the very same factory works immediately afterwards, so
        # the failure left no armed or poisoned state behind.
        built = factory()
        try:
            assert built.dtype == dtype, label
        finally:
            built.close()
        assert len(live_storages) == baseline, label


@needs_native
@needs_fault_injection
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_an_operation_output_allocation_failure_closes_everything(
    dtype, live_storages
):
    """Every operation family's **output** allocation, failed. The
    operands survive open and unchanged, no output is published, and a
    corrected retry produces the right answer — so the failure path is
    recoverable rather than merely quiet."""
    operations = (
        ("relu", lambda a, b: a.relu()),
        ("sqrt", lambda a, b: a.sqrt()),
        ("exp", lambda a, b: a.exp()),
        ("add", lambda a, b: a.add(b)),
        ("multiply", lambda a, b: a.multiply(b)),
        ("matmul", lambda a, b: a.matmul(b)),
        ("sum", lambda a, b: a.sum()),
        ("contiguous_copy", lambda a, b: a.contiguous_copy()),
        ("softmax", lambda a, b: a.softmax()),
    )
    a = core(np.ones((4, 4)), dtype)
    b = core(np.full((4, 4), 2.0), dtype)
    try:
        for label, operation in operations:
            baseline = len(live_storages)
            before_a = a.to_numpy().copy()
            # Different operations allocate a different number of native
            # buffers (softmax copies before it computes), so every
            # allocation position each one reaches is failed in turn
            # rather than assuming the first is the only one.
            failed_at_least_once = False
            for nth in (1, 2, 3):
                cpp._arm_alloc_failure(nth)
                try:
                    operation(a, b)
                except MemoryError:
                    failed_at_least_once = True
                finally:
                    cpp._arm_alloc_failure(0)
                assert len(live_storages) == baseline, (label, nth)
                assert a._closed is False and b._closed is False, (label, nth)
                assert bits(a.to_numpy(), dtype) == bits(before_a, dtype), (
                    label, nth)
            assert failed_at_least_once, (
                f"{label} never allocated, so this proved nothing"
            )
            result = operation(a, b)
            try:
                assert result.dtype == dtype, label
            finally:
                result.close()
            assert len(live_storages) == baseline, label
    finally:
        close_all(a, b)


@needs_native
@needs_fault_injection
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_module_constructor_failure_closes_its_earlier_allocations(
    dtype, live_storages
):
    """The multi-allocation constructors: a failure at the *second*
    allocation must close the first. ``NativeLinear`` gained this at I7
    (its younger siblings already had it), and it is held here at both
    widths for every state-owning constructor."""
    constructors = (
        ("NativeLinear", lambda: NativeLinear(4, 3, seed=0, dtype=dtype)),
        ("NativeConv2d",
         lambda: NativeConv2d(1, 2, 3, padding=1, seed=0, dtype=dtype)),
        ("NativeLayerNorm", lambda: NativeLayerNorm(4, dtype=dtype)),
        ("NativeBatchNorm1d", lambda: NativeBatchNorm1d(4, dtype=dtype)),
        ("NativeBatchNorm2d", lambda: NativeBatchNorm2d(4, dtype=dtype)),
    )
    for label, build in constructors:
        baseline = len(live_storages)
        failures = 0
        for nth in (1, 2, 3, 4):
            cpp._arm_alloc_failure(nth)
            try:
                close_module(build())
            except MemoryError:
                failures += 1
            finally:
                cpp._arm_alloc_failure(0)
            assert len(live_storages) == baseline, (label, nth)
        assert failures >= 2, (
            f"{label} allocates more than once, so failing the first two "
            f"positions must be observable"
        )
        close_module(build())
        assert len(live_storages) == baseline, label


@needs_native
@needs_fault_injection
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_an_optimizer_state_allocation_failure_leaves_the_optimizer_intact(
    dtype, live_storages
):
    """Adam builds ``m`` and ``v`` per parameter and a scalar set per
    active dtype. A failure at any of those positions must leave the
    optimizer exactly as it was — no half-built moment pair, no orphaned
    scalar — and a retry must then succeed."""
    parameter = NativeParameter([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    parameter._grad = tensor([[0.5, 0.5], [0.5, 0.5]], dtype)
    optimizer = NativeAdam([parameter], lr=0.1)
    try:
        baseline = len(live_storages)
        version = parameter.version
        values = parameter.to_numpy().copy()
        failures = 0
        for nth in (1, 2, 3, 4, 5):
            cpp._arm_alloc_failure(nth)
            try:
                optimizer.step()
            except MemoryError:
                failures += 1
            finally:
                cpp._arm_alloc_failure(0)
            assert len(live_storages) == baseline, nth
            assert parameter.version == version, nth
            assert bits(parameter.to_numpy(), dtype) == bits(values, dtype), nth
            # No half-built moment pair survives: either there is no state
            # at all, or both moments exist at the parameter's own width.
            assert len(optimizer._m) == len(optimizer._v), nth
            for moment in list(optimizer._m) + list(optimizer._v):
                if moment is not None:
                    assert moment.dtype == dtype, nth
        assert failures >= 2, (
            "Adam's first step allocates moments and scalars, so several "
            "positions must be failable"
        )
        # A clean step now works and really does move the parameter.
        optimizer.step()
        assert parameter.version == version + 1
        assert optimizer._m[0].dtype == dtype
    finally:
        close_all(parameter._grad, optimizer, parameter)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_python_wrapper_failure_after_native_allocation_closes_it(
    dtype, live_storages, monkeypatch
):
    """Design §20's last row: an exception raised during **Python wrapper
    construction**, after the native allocation has already succeeded,
    still closes every native resource the attempt created. Injected at
    the wrapper rather than at the allocator, because that is a different
    seam from the one above."""
    baseline = len(live_storages)
    original = cpp.NativeTensorCore.__init__

    def exploding(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("injected wrapper failure")

    monkeypatch.setattr(cpp.NativeTensorCore, "__init__", exploding)
    with pytest.raises(RuntimeError, match="injected wrapper failure"):
        cpp.NativeTensorCore.zeros((4, 4), dtype=dtype)
    monkeypatch.undo()
    assert len(live_storages) == baseline


# ==========================================================================
# 5. All four saved-resource families in ONE float32 graph
#
#    I9 exercised all four across a run and scoped the claim honestly:
#    three on a training graph, the BatchNorm evaluation snapshots only on
#    an evaluation graph, because training-mode BatchNorm takes no
#    snapshot at all. I10 builds the configuration where all four
#    genuinely coexist and walks the graph to prove it.
# ==========================================================================


class FourResourceModel(NativeModule):
    """The smallest model whose one loss graph owns all four families.

    ``conv → batch_norm2d → relu → pool → dropout_a → flatten → hidden
    → dropout_b`` into a fused cross-entropy loss. The two Dropout layers
    share **one** generator object, so the model carries a real alias
    topology rather than two independent counters.

    The four families arrive as:

    - **Dropout multiplier masks** — one per Dropout forward, in training;
    - **MaxPool2d winners** — from the pooling forward;
    - **BatchNorm evaluation snapshots** — only when BatchNorm is in eval;
    - **cross-entropy saved probabilities** — from the fused loss.

    Which is why the configuration is deliberately mixed-mode."""

    def __init__(self, dtype, seed=3, generator_seed=4242, p=0.5):
        super().__init__()
        generator = NativeGenerator(generator_seed)
        self.conv = NativeConv2d(1, 2, 3, padding=1, seed=seed, dtype=dtype)
        self.batch_norm2d = NativeBatchNorm2d(2, dtype=dtype)
        self.relu = NativeReLU()
        self.pool = NativeMaxPool2d(2)
        self.dropout_a = NativeDropout(p, generator=generator)
        self.flatten = NativeFlatten()
        self.hidden = NativeLinear(2 * 2 * 2, 3, seed=seed + 1, dtype=dtype)
        self.dropout_b = NativeDropout(p, generator=generator)
        self._dtype = dtype

    def forward(self, images):
        return self.forward_keeping_intermediates(images)[0]

    def forward_keeping_intermediates(self, images):
        """The same forward, returning ``(output, intermediates)``.

        ``close()`` releases the resources **the node it is called on**
        owns — it does not walk the ancestry, and the module's own locals
        are unreachable once ``forward`` returns. So the abandoned-graph
        proof needs every owner in hand, which is what this gives it. The
        ordinary ``forward`` is unchanged for every other test."""
        kept = []
        h = self.conv(images)
        kept.append(h)
        for layer in (self.batch_norm2d, self.relu, self.pool,
                      self.dropout_a, self.flatten, self.hidden,
                      self.dropout_b):
            h = layer(h)
            kept.append(h)
        return kept[-1], kept[:-1]


def _four_resource_setup(dtype, p=0.5, frozen=False):
    """Model, loss, input, and targets in the mixed-mode configuration.

    The whole model goes to **eval** so BatchNorm normalizes with
    immutable snapshots of its running buffers; the two Dropout layers are
    then put back into **training** through the ordinary public per-module
    ``train()`` API. No undocumented flag is touched.

    ``frozen=True`` additionally clears ``requires_grad`` on every
    parameter, which is how the no-grad path is reached honestly: a model
    whose parameters track gradients builds a graph whatever the input
    does."""
    model = FourResourceModel(dtype, p=p)
    loss_fn = NativeCrossEntropyLoss()
    model.eval()
    model.dropout_a.train(True)
    model.dropout_b.train(True)
    if frozen:
        for parameter in model.parameters():
            # ``requires_grad`` is deliberately read-only on a native
            # parameter (it is fixed at construction), so a test that
            # needs a gradient-free model uses the private initializer —
            # the same kind of narrow test seam I9 used for ``_grad``.
            parameter._init_requires_grad(False)
    values = np.linspace(-1.0, 1.0, 2 * 1 * 4 * 4).reshape(2, 1, 4, 4)
    x = tensor(values, dtype)
    targets = np.asarray([0, 2], dtype=np.int64)
    return model, loss_fn, x, targets


@needs_native
def test_one_float32_graph_owns_all_four_saved_resource_families():
    """The coexistence proof, and the reason this milestone exists.

    Everything is asserted from the live graph: the module modes, that the
    running buffers did **not** move, that the generator advanced by
    exactly the number of Dropout forwards, that all four families are
    present, that every *value* resource is float32, that the winner
    buffer is float64 metadata, and that no resource aliases a parameter,
    a buffer, the input, or another resource."""
    dtype = "float32"
    model, loss_fn, x, targets = _four_resource_setup(dtype)
    loss = logits = None
    try:
        # The configuration itself, proved rather than assumed.
        assert model.batch_norm2d.training is False
        assert model.dropout_a.training is True
        assert model.dropout_b.training is True

        generator = list(model.generators())[0]
        calls_before = generator.calls
        running_before = {
            name: buffer.to_numpy().copy()
            for name, buffer in model.named_buffers()
        }

        logits = model(x)
        loss = loss_fn(logits, targets)

        # BatchNorm is in eval, so no running buffer moved.
        for name, buffer in model.named_buffers():
            assert np.array_equal(buffer.to_numpy(), running_before[name]), name
        # Exactly two Dropout forwards happened, on the shared stream.
        assert generator.calls == calls_before + 2

        inventory = graph_resource_inventory(loss)
        families = classify_resources(inventory)

        assert families["dropout_mask"], "no Dropout mask in the graph"
        assert families["winner"], "no MaxPool2d winner buffer in the graph"
        assert families["bn_snapshot"], "no BatchNorm snapshot in the graph"
        assert families["probabilities"], "no saved probabilities in the graph"
        assert len(families["dropout_mask"]) == 2, families["dropout_mask"]
        assert len(families["winner"]) == 1, families["winner"]
        assert len(families["probabilities"]) == 1, families["probabilities"]
        # One BatchNorm layer in eval, two graph-safe snapshots (the
        # running-mean snapshot and the inverse standard deviation).
        assert len(families["bn_snapshot"]) == 2, families["bn_snapshot"]

        # Every *value* resource carries the graph dtype...
        for label in ("dropout_mask", "bn_snapshot", "probabilities"):
            for _, resource_dtype, shape in families[label]:
                assert resource_dtype == dtype, (label, shape, resource_dtype)
        # ...and the winner buffer is the one deliberate exception, still
        # float64 while every value beside it is float32.
        for _, resource_dtype, shape in families["winner"]:
            assert resource_dtype == "float64", shape

        # Nothing aliases anything: every resource has its own storage,
        # and none of them is a parameter, a buffer, the input, or the
        # output.
        storages = [_storage_id(resource) for _, resource, _, _ in inventory]
        assert len(set(storages)) == len(storages), "two resources alias"
        protected = {_storage_id(p) for p in model.parameters()}
        protected |= {_storage_id(b) for _, b in model.named_buffers()}
        protected.add(_storage_id(x))
        protected.add(_storage_id(logits))
        assert not (set(storages) & protected), (
            "a saved resource aliases a parameter, buffer, input, or output"
        )
    finally:
        close_all(loss, logits, x)
        close_module(model)


@needs_native
def test_the_four_resource_proof_is_not_vacuous():
    """The negative control. With the model left wholly in **training**
    mode, BatchNorm uses the batch's own statistics and takes no snapshot
    — so the very same walk finds three families, not four, and the
    coexistence claim above is a real observation rather than a helper
    that always reports success."""
    dtype = "float32"
    model, loss_fn, x, targets = _four_resource_setup(dtype)
    loss = logits = None
    try:
        model.train(True)                      # every child, including BN
        assert model.batch_norm2d.training is True
        logits = model(x)
        loss = loss_fn(logits, targets)
        families = classify_resources(graph_resource_inventory(loss))
        assert not families["bn_snapshot"], (
            "training-mode BatchNorm must take no evaluation snapshot"
        )
        assert families["dropout_mask"] and families["winner"]
        assert families["probabilities"]
    finally:
        close_all(loss, logits, x)
        close_module(model)


@needs_native
def test_the_four_resource_graph_survives_a_retained_backward(live_storages):
    """``retain_graph=True`` keeps all four families alive for another
    pass, and the second backward accumulates rather than recomputing from
    a released graph."""
    dtype = "float32"
    model, loss_fn, x, targets = _four_resource_setup(dtype)
    loss = logits = None
    try:
        x_leaf = tensor(x.to_numpy(), dtype, requires_grad=True)
        logits = model(x_leaf)
        loss = loss_fn(logits, targets)
        resources = [r for _, r, _, _ in graph_resource_inventory(loss)]
        assert len(resources) >= 4

        loss.backward(retain_graph=True)
        assert all(not is_closed(r) for r in resources), (
            "a retained graph must keep every saved resource open"
        )
        first = x_leaf.grad.to_numpy().copy()

        loss.backward(retain_graph=True)
        second = x_leaf.grad.to_numpy().copy()
        assert np.allclose(second, 2.0 * first, rtol=1e-5, atol=1e-6), (
            "the second retained backward must accumulate"
        )
        assert all(not is_closed(r) for r in resources)

        # The final one-shot pass releases all four, exactly once.
        loss.backward()
        assert all(is_closed(r) for r in resources), (
            "a one-shot backward must release every saved resource"
        )
        with pytest.raises(RuntimeError):
            loss.backward()
        close_all(x_leaf)
    finally:
        close_all(loss, logits, x)
        close_module(model)


@needs_native
def test_a_failed_backward_retains_every_saved_resource_and_retries(
    monkeypatch
):
    """Design §20's last row and §21's retryability rule, on the graph
    that owns all four families: a failure raised **after** backward
    temporaries exist commits no leaf gradient, keeps every saved resource
    available, and a corrected retry then succeeds."""
    dtype = "float32"
    model, loss_fn, x, targets = _four_resource_setup(dtype)
    loss = logits = x_leaf = None
    try:
        x_leaf = tensor(x.to_numpy(), dtype, requires_grad=True)
        logits = model(x_leaf)
        loss = loss_fn(logits, targets)
        resources = [r for _, r, _, _ in graph_resource_inventory(loss)]
        generator = list(model.generators())[0]
        calls_before = generator.calls

        # Fail partway through the traversal, after temporaries exist.
        state = {"n": 0}
        original = cpp.NativeTensorCore.multiply

        def flaky(self, other):
            state["n"] += 1
            if state["n"] == 3:
                raise RuntimeError("injected backward failure")
            return original(self, other)

        monkeypatch.setattr(cpp.NativeTensorCore, "multiply", flaky)
        with pytest.raises(RuntimeError, match="injected backward failure"):
            loss.backward(retain_graph=True)
        monkeypatch.undo()

        assert x_leaf.grad is None, "no leaf gradient may be committed"
        assert all(not is_closed(r) for r in resources), (
            "a failed retryable backward keeps its saved resources"
        )
        assert generator.calls == calls_before, (
            "backward consumes no generator call"
        )

        loss.backward()
        assert x_leaf.grad is not None and x_leaf.grad.dtype == dtype
        assert all(is_closed(r) for r in resources)
    finally:
        close_all(loss, logits, x_leaf, x)
        close_module(model)


@needs_native
def test_an_abandoned_four_resource_graph_releases_everything_once(
    live_storages
):
    """An abandoned graph — built and never differentiated — releases all
    four families exactly once, deterministically, and live storage
    returns to baseline without a ``gc.collect()``.

    ``close()`` is per node, so every node that owns a resource is closed
    explicitly. That is the contract as written: the deterministic release
    points are a one-shot ``backward()``'s history release and ``close()``
    on the owning node; ``__del__`` is a fallback, not a contract."""
    dtype = "float32"
    model, loss_fn, x, targets = _four_resource_setup(dtype)
    logits = loss = None
    intermediates = []
    try:
        gc.collect()
        baseline = len(live_storages)
        logits, intermediates = model.forward_keeping_intermediates(x)
        loss = loss_fn(logits, targets)
        inventory = graph_resource_inventory(loss)
        families = classify_resources(inventory)
        assert all(families[name] for name in families), families
        resources = [r for _, r, _, _ in inventory]
        assert len(resources) == 6, [(op, s) for op, _, _, s in inventory]

        closed = close_graph(loss)
        assert closed >= len(intermediates) + 2, closed
        close_graph(loss)                    # idempotent: never twice
        loss = logits = None
        intermediates = []

        assert all(is_closed(r) for r in resources), (
            "an abandoned graph must release every saved resource"
        )
        # Explicit ``close()`` is the release mechanism and the assertion
        # above is what proves it. The collection settles only the two
        # documented Python-level reference cycles I9 recorded — a dropped
        # gradient object, and a graph node reachable through a backward
        # closure — so the count below is a statement about TensorForge.
        gc.collect()
        assert len(live_storages) == baseline
    finally:
        close_all(loss, logits, *intermediates, x)
        close_module(model)


@needs_native
def test_a_no_grad_forward_closes_its_saved_resources_immediately(
    live_storages
):
    """When no backward can consume them, the resources are closed at
    once rather than waiting for a graph that will never exist.

    Reaching this honestly needs **every** leaf gradient-free: a model
    whose parameters track gradients builds a graph however the input was
    constructed, so the parameters are frozen too."""
    dtype = "float32"
    model, loss_fn, x, targets = _four_resource_setup(dtype, frozen=True)
    try:
        assert not any(p.requires_grad for p in model.parameters())
        baseline = len(live_storages)
        for _ in range(3):
            logits = model(x)               # nothing requires grad
            loss = loss_fn(logits, targets)
            assert logits.requires_grad is False
            assert graph_resource_inventory(loss) == [], (
                "a no-grad forward must build no graph-owned resource"
            )
            close_all(loss, logits)
            assert len(live_storages) == baseline
    finally:
        close_all(x)
        close_module(model)


@needs_native
def test_repeated_four_resource_lifecycles_return_to_baseline(live_storages):
    """The whole cycle, repeated: build, backward, release. Live storage
    must return **exactly** to baseline every time, at both widths.

    A one-shot ``backward()`` is itself a deterministic release point — it
    clears each traversed node's graph and releases every saved resource —
    and gradients are closed explicitly, because ``zero_grad()``
    deliberately drops rather than closes them. The collection settles the
    two documented reference cycles before counting, exactly as the I9
    lifecycle proof does."""
    for dtype in BOTH_DTYPES:
        model, loss_fn, x, targets = _four_resource_setup(dtype)
        try:
            gc.collect()
            baseline = len(live_storages)
            for _ in range(4):
                leaf = tensor(x.to_numpy(), dtype, requires_grad=True)
                logits = model(leaf)
                loss = loss_fn(logits, targets)
                loss.backward()
                # A gradient is real native storage, and ``zero_grad()``
                # deliberately **drops rather than closes** it so a caller
                # holding ``t.grad`` is never invalidated. Releasing it is
                # therefore the caller's job, and doing it here is what
                # makes "returns exactly to baseline" a statement about
                # TensorForge rather than about the collector.
                for owner in [leaf] + list(model.parameters()):
                    if owner.grad is not None:
                        owner.grad.close()
                    owner.zero_grad()
                close_graph(loss)
                close_all(logits, leaf)
                gc.collect()
                assert len(live_storages) == baseline, dtype
        finally:
            close_all(x)
            close_module(model)


# ==========================================================================
# 6. Concurrency — exactly the contract already claimed, no more
#
#    Design §21: participating **state-replacement** operations serialize
#    with respect to each other in the universal lock order. Ordinary
#    training mutation does not participate and concurrent training
#    against a model being checkpointed is still not supported. I10 tests
#    what is claimed and adds no promise.
# ==========================================================================


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_concurrent_readers_observe_one_immutable_dtype(dtype):
    """The dtype tag is immutable after construction, so concurrent
    readers cannot disagree. Barriered so every thread genuinely reaches
    the read window rather than finishing before the others start."""
    workers = 8
    barrier = threading.Barrier(workers, timeout=JOIN_TIMEOUT)
    x = tensor(np.ones((32, 32)), dtype)
    seen, guard = [], threading.Lock()
    runner = ThreadRunner()

    def read():
        barrier.wait()
        observations = {(x.dtype, x.device, x.shape, x._core.dtype)
                        for _ in range(200)}
        with guard:
            seen.append(observations)

    try:
        for index in range(workers):
            runner.start(read, name=f"reader-{index}")
        runner.join()
        assert len(seen) == workers, "not every reader reached the window"
        merged = set().union(*seen)
        assert merged == {(dtype, "cpu", (32, 32), dtype)}, merged
    finally:
        close_all(x)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_concurrent_state_snapshots_are_each_internally_coherent(dtype):
    """Two threads snapshotting one model's state concurrently each get a
    coherent snapshot, and no snapshot is a mixture. Identity is preserved
    throughout, and no partial transaction is ever visible."""
    workers = 4
    barrier = threading.Barrier(workers, timeout=JOIN_TIMEOUT)
    model = NativeLinear(4, 3, seed=1, dtype=dtype)
    identities = {n: id(p) for n, p in model.named_parameters()}
    snapshots, guard = [], threading.Lock()
    runner = ThreadRunner()

    def snapshot():
        barrier.wait()
        for _ in range(10):
            state = model.state_dict()
            captured = {name: value.to_numpy().copy()
                        for name, value in state.items()}
            close_all(*state.values())
            with guard:
                snapshots.append(captured)

    try:
        for index in range(workers):
            runner.start(snapshot, name=f"snap-{index}")
        runner.join()
        assert len(snapshots) == workers * 10
        reference = snapshots[0]
        for captured in snapshots:
            assert set(captured) == set(reference)
            for name, values in captured.items():
                assert values.dtype == NUMPY_DTYPES[dtype], name
                assert bits(values, dtype) == bits(reference[name], dtype)
        assert {n: id(p) for n, p in model.named_parameters()} == identities
    finally:
        close_module(model)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_concurrent_state_loads_serialize_and_never_mix(dtype):
    """Two whole-model loads racing each other: the shared guard gives the
    execution a valid serial order, so the final state equals **one** of
    the two donors exactly, never a mixture of both."""
    barrier = threading.Barrier(2, timeout=JOIN_TIMEOUT)
    destination = NativeLinear(4, 3, seed=1, dtype=dtype)
    donors = [NativeLinear(4, 3, seed=s, dtype=dtype) for s in (11, 22)]
    states = [d.state_dict() for d in donors]
    identities = {n: id(p) for n, p in destination.named_parameters()}
    runner = ThreadRunner()

    def load(state):
        def run():
            barrier.wait()
            for _ in range(20):
                destination.load_state_dict(state)
        return run

    try:
        for index, state in enumerate(states):
            runner.start(load(state), name=f"loader-{index}")
        runner.join()
        final = {n: p.to_numpy().copy()
                 for n, p in destination.named_parameters()}
        matches = [
            all(bits(final[n], dtype)
                == bits(state[n].to_numpy(), dtype) for n in final)
            for state in states
        ]
        assert any(matches), (
            "the final state is a mixture of both donors, which is exactly "
            "what the shared state-transaction guard exists to prevent"
        )
        assert {n: id(p) for n, p in destination.named_parameters()} \
            == identities
    finally:
        for state in states:
            close_all(*state.values())
        close_module(destination)
        for donor in donors:
            close_module(donor)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_checkpoint_snapshot_never_tears_a_batchnorm_update(dtype,
                                                              tmp_path):
    """Phase F's two-buffer transaction under contention, at both widths,
    against the **participating** reader.

    A BatchNorm training forward commits its running statistics through
    ``replace_native_state``, which takes the shared guard; a checkpoint
    save snapshots under the same guard. So a saved archive holds a mean
    and a variance from **one** commit, never a mean from one step beside
    a variance from another.

    The reader here is deliberately ``save_native_checkpoint`` and not
    ``state_dict()``: a bare snapshot does *not* take the guard, and the
    project does not offer thread-safe concurrent training snapshots
    (design §21). Testing the unsupported path would be inventing a
    promise."""
    from tensorforge.experimental import (
        load_native_checkpoint, save_native_checkpoint,
    )

    module = NativeBatchNorm1d(3, dtype=dtype)
    module.train(True)
    barrier = threading.Barrier(2, timeout=JOIN_TIMEOUT)
    stop = threading.Event()
    saved, guard = [], threading.Lock()
    runner = ThreadRunner()

    def writer():
        barrier.wait()
        try:
            for step in range(30):
                x = tensor(np.full((4, 3), float(step) + 1.0), dtype)
                try:
                    module(x).close()
                finally:
                    x.close()
        finally:
            stop.set()

    def saver():
        barrier.wait()
        index = 0
        while not stop.is_set() and index < 30:
            path = tmp_path / f"snapshot_{index}.npz"
            save_native_checkpoint(path, module)
            with guard:
                saved.append(path)
            index += 1

    try:
        runner.start(writer, name="bn-writer")
        runner.start(saver, name="bn-saver")
        runner.join()
        assert saved, "the saver never completed an archive"
        # Every archive loads back into a fresh module of the same width,
        # which is only possible if each was internally coherent.
        probe = NativeBatchNorm1d(3, dtype=dtype)
        try:
            for path in saved:
                load_native_checkpoint(path, probe)
                for name, buffer in probe.named_buffers():
                    values = buffer.to_numpy()
                    assert values.dtype == NUMPY_DTYPES[dtype], name
                    assert np.all(np.isfinite(values)), (name, path)
        finally:
            close_module(probe)
    finally:
        close_module(module)


@needs_native
def test_one_generator_serves_one_stochastic_call_at_a_time(dtype=None):
    """Phase G's reservation rule, unchanged and dtype-independent.

    A generator serves **one** stochastic call at a time. Concurrent
    stochastic use of a single generator is explicitly *not supported*,
    and the runtime says so deterministically rather than interleaving two
    draws — which is the behavior that keeps the stream reproducible. This
    test asserts that documented refusal; it does not claim concurrent
    Dropout is safe, because it is not."""
    generator = NativeGenerator(99)
    layer = NativeDropout(0.5, generator=generator)
    layer.train(True)
    workers, draws = 4, 20
    barrier = threading.Barrier(workers, timeout=JOIN_TIMEOUT)
    runner = ThreadRunner()
    outcomes, guard = {"ok": 0, "refused": 0}, threading.Lock()

    def draw(dtype):
        def run():
            barrier.wait()
            ok = refused = 0
            for _ in range(draws):
                x = tensor(np.ones((4, 4)), dtype)
                try:
                    out = layer(x)
                    out.close()
                    ok += 1
                except RuntimeError as error:
                    assert "reservation" in str(error), str(error)
                    refused += 1
                finally:
                    x.close()
            with guard:
                outcomes["ok"] += ok
                outcomes["refused"] += refused
        return run

    for index in range(workers):
        runner.start(draw(BOTH_DTYPES[index % 2]), name=f"draw-{index}")
    runner.join()

    assert outcomes["ok"] + outcomes["refused"] == workers * draws
    # The invariant that matters: the counter equals the number of
    # **successful** draws exactly. A refusal burns nothing, and no two
    # successful draws ever shared a call index.
    assert generator.calls == outcomes["ok"], (
        f"counter {generator.calls} vs {outcomes['ok']} successful draws — "
        f"a refused reservation must burn no call"
    )


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_serialized_dropout_draws_account_for_every_call(dtype):
    """The supported shape of the same thing: used serially, one shared
    generator across two registered Dropout paths advances by exactly one
    per forward, at both widths, and the two paths observe one stream."""
    generator = NativeGenerator(1234)
    first = NativeDropout(0.5, generator=generator)
    second = NativeDropout(0.5, generator=generator)
    first.train(True)
    second.train(True)
    x = tensor(np.ones((4, 4)), dtype)
    try:
        assert generator.calls == 0
        for expected in range(1, 11):
            layer = first if expected % 2 else second
            out = layer(x)
            try:
                assert out.dtype == dtype
            finally:
                out.close()
            assert generator.calls == expected
    finally:
        close_all(x)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_rejected_dropout_forward_burns_no_generator_call(dtype):
    """A mismatched Dropout forward is a Python-level rejection, so no
    reservation is taken and the counter does not move."""
    generator = NativeGenerator(5)
    module = NativeSequential(
        NativeLinear(3, 3, seed=0, dtype=dtype),
        NativeDropout(0.5, generator=generator),
    )
    module.train(True)
    wrong = tensor([[1.0, 2.0, 3.0]], OTHER_DTYPE[dtype])
    try:
        before = generator.calls
        with pytest.raises(ValueError):
            module(wrong)
        assert generator.calls == before
    finally:
        close_all(wrong)
        for parameter in module.parameters():
            close_all(parameter)


@needs_native
def test_the_state_transaction_guard_is_one_reentrant_process_wide_lock():
    """The universal lock order rests on there being exactly one guard, so
    that is asserted from the live object rather than argued from source.
    Reentrancy is required, not incidental: the whole-checkpoint
    transaction holds it and then calls loaders that take it again."""
    guard = _native_state_lock._STATE_TRANSACTION_LOCK
    assert type(guard).__name__ in ("RLock", "_thread.RLock")
    assert _native_state_lock.held_by_current_thread() is False
    with _native_state_lock.state_transaction():
        assert _native_state_lock.held_by_current_thread() is True
        with _native_state_lock.state_transaction():   # reentrant
            assert _native_state_lock.held_by_current_thread() is True
    assert _native_state_lock.held_by_current_thread() is False


@needs_native
def test_the_barrier_controls_really_detect_a_thread_that_never_arrives():
    """The negative control for every barriered test above: a barrier
    sized for more parties than will arrive times out, so "both threads
    reached the seam" is a real observation. Without this, a test whose
    worker died early would still pass."""
    barrier = threading.Barrier(2, timeout=0.25)
    with pytest.raises(threading.BrokenBarrierError):
        barrier.wait()


# ==========================================================================
# 7. Stable / native isolation, re-proved after the public float32 move
# ==========================================================================


@needs_native
def test_importing_the_stable_framework_loads_no_native_library():
    """Design §19. Public float32 is a **native-line** capability and says
    nothing about the stable line; importing ``tensorforge`` must still not
    pull in ctypes or the C++ library. Run in a subprocess because the
    parent has already imported everything."""
    program = (
        "import sys, tensorforge\n"
        "assert 'tensorforge.backends.cpp' not in sys.modules, \\\n"
        "    'the stable import pulled in the native backend'\n"
        "t = tensorforge.Tensor([[1.0, 2.0]])\n"
        "assert t.data.dtype.name == 'float64'\n"
        "from tensorforge.backends import cpp\n"
        "assert cpp._lib is None, 'importing tensorforge loaded the library'\n"
        "assert cpp.SUPPORTED_DTYPES == ('float64', 'float32')\n"
        "assert cpp.backend_info()['stable_framework_integration'] is False\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", program],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


@needs_native
def test_the_stable_line_gained_no_dtype_surface_from_phase_i():
    """The stable ``Tensor`` is feature-frozen at v3.0 and Phase I did not
    touch it, its dtype behavior, or its serialization."""
    import tensorforge as tf

    t = tf.Tensor([[1.0, 2.0]])
    assert t.data.dtype == np.float64
    for banned in ("dtype", "astype", "float", "double", "to"):
        assert not hasattr(t, banned), banned
    for banned in ("dtype", "device"):
        assert banned not in tf.Tensor.__init__.__code__.co_varnames, banned


@needs_native
def test_the_numpy_reference_backend_dtype_row_did_not_move():
    """Phase I is a native-line phase. The NumPy reference backend keeps
    its own ``supported_dtypes == ("float64",)``, and that is a different
    statement from the native registry."""
    from tensorforge.backends.numpy_backend import NumpyBackend

    info = NumpyBackend().backend_info()
    assert info["supported_dtypes"] == ("float64",)
    assert info["dtype"] == "float64"
    assert info["experimental"] is False


@needs_native
def test_stable_and_native_objects_reject_one_another_in_both_directions():
    """No implicit conversion exists in either direction, at any dtype."""
    import tensorforge as tf

    stable = tf.Tensor([[1.0, 2.0]])
    native = tensor([[1.0, 2.0]], "float32")
    try:
        with pytest.raises((TypeError, ValueError)):
            native.add(stable)
        with pytest.raises((TypeError, ValueError)):
            NativeParameter(stable, dtype="float32")
        with pytest.raises((TypeError, ValueError)):
            stable + native
    finally:
        close_all(native)


@needs_native
def test_no_environment_variable_selects_a_dtype_or_a_backend():
    """Which dtype a tensor has is a property of how it was constructed,
    full stop. Asserted structurally over the runtime source, so a future
    ``os.environ`` read in the dtype path fails here."""
    import inspect

    source = inspect.getsource(cpp)
    for banned in ("os.environ", "getenv", "TF_DTYPE", "TENSORFORGE_DTYPE"):
        assert banned not in source, banned
    assert cpp.backend_info()["stable_framework_integration"] is False


# ==========================================================================
# 8. ABI, registry, ctypes, and export inventories
# ==========================================================================


@needs_native
def test_the_registries_are_exactly_the_post_i9_truth():
    """The four rows are four different questions and none may be read off
    another. Asserted as exact tuples, order included."""
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert not set(cpp.SUPPORTED_DTYPES) & set(cpp.UNSUPPORTED)
    # The flat key is the **default**, not the capability.
    assert cpp.backend_info()["dtype"] == "float64"
    assert cpp.normalize_dtype(None) == "float64"
    assert cpp.backend_info()["dtype"] == cpp.normalize_dtype(None)
    assert cpp.backend_info()["supported_dtypes"] == cpp.SUPPORTED_DTYPES
    assert cpp.backend_info()["raw_kernel_dtypes"] == cpp.RAW_KERNEL_DTYPES


@needs_native
def test_the_dtype_abi_codes_and_item_sizes_are_frozen():
    """``0 = float64``, ``1 = float32``, 8 and 4 bytes — and exactly one
    item-size authority, so no second table can drift."""
    assert cpp._DTYPE_CODES == {"float64": 0, "float32": 1}
    assert set(cpp._DTYPE_NUMPY) == {"float64", "float32"}
    assert cpp._DTYPE_NUMPY["float64"] == np.float64
    assert cpp._DTYPE_NUMPY["float32"] == np.float32
    assert np.dtype(np.float64).itemsize == 8
    assert np.dtype(np.float32).itemsize == 4
    assert set(cpp._CHECKED_HOST_ARRAYS) == {"float64", "float32"}


@needs_native
def test_the_export_inventory_is_still_exactly_fifty_four():
    """Source declarations and the built library's export table agree, and
    the count has not moved since I1. Phase I adds exactly two symbols
    across the whole phase and both were spent at I1."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    declared = set()
    for path in list((root / "cpp" / "include").glob("*.h")) + \
            list((root / "cpp" / "src").glob("*.cpp")):
        text = path.read_text(encoding="utf-8")
        declared |= set(re.findall(r"TF_EXPORT[^(;]*?\b(tf_[a-z0-9_]+)\s*\(",
                                   text))
    assert len(declared) == 54, sorted(declared)
    assert "tf_storage_create_typed" in declared
    assert "tf_storage_create_uninitialized_typed" in declared
    # No per-dtype compute symbol, no cast symbol, no dtype query.
    for symbol in sorted(declared):
        assert not symbol.endswith(("_f32", "_f64")), symbol
        for banned in ("cast", "astype", "convert", "promote", "dtype_of",
                       "query_dtype"):
            assert banned not in symbol, symbol


@needs_native
def test_every_checked_kernel_is_real_and_every_fallible_one_is_checked():
    """``_CHECKED_KERNELS`` is the errcheck registry: every name in it must
    be a real exported symbol, and every fallible compute export must be
    in it — otherwise a native failure would return a benign value with
    nobody translating the error slot."""
    library = cpp._require_library()
    for name in cpp._CHECKED_KERNELS:
        assert hasattr(library, name), name
    compute = [n for n in cpp._CHECKED_KERNELS if n.startswith("tf_core_")]
    assert len(compute) >= 20, sorted(compute)
    # The infallible and test-only symbols are deliberately absent.
    for infallible in ("tf_clear_error", "tf_last_error_code",
                       "tf_last_error_message", "tf_storage_size",
                       "tf_storage_destroy", "tf_fault_injection_available",
                       "tf_test_arm_alloc_failure"):
        assert infallible not in cpp._CHECKED_KERNELS, infallible


@needs_native
def test_the_raw_kernels_are_float64_only_and_say_so():
    """``RAW_KERNEL_DTYPES`` is a permanent limitation of the seven
    handle-free utility kernels, which take only ``double*`` and an element
    count and so have no dtype to dispatch on. It is not the overall dtype
    support row and must never be read as one."""
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.RAW_KERNEL_DTYPES != cpp.SUPPORTED_DTYPES
    result = cpp.matmul(np.ones((4, 4), dtype=np.float32),
                        np.ones((4, 4), dtype=np.float32))
    assert result.dtype == np.float64, (
        "a raw kernel converts to float64 and returns float64; it does not "
        "gain a float32 path"
    )


@needs_native
def test_the_checkpoint_and_optimizer_state_constants_did_not_move_at_i10():
    from tensorforge.experimental import (
        native_checkpoint, native_optimizer_state,
    )

    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert native_optimizer_state.FORMAT_VERSION == 1


@needs_native
def test_no_casting_promotion_or_dtype_inference_entered_the_public_surface():
    """The whole phase in one assertion set: two dtypes is not a way to
    move between them, and no constructor infers a dtype from its input."""
    from tensorforge import experimental

    for banned in ("astype", "cast", "promote", "result_type",
                   "can_cast", "set_default_dtype", "get_default_dtype"):
        assert not hasattr(experimental, banned), banned
        assert not hasattr(cpp, banned), banned
    # A float32 host array with no dtype still gives float64.
    built = NativeTensor.from_array(np.ones((2, 2), dtype=np.float32))
    try:
        assert built.dtype == "float64"
    finally:
        built.close()


@needs_native
def test_no_device_argument_was_added_anywhere_by_the_hardening_milestone():
    import inspect

    for factory in (NativeLinear, NativeConv2d, NativeLayerNorm,
                    NativeBatchNorm1d, NativeBatchNorm2d, NativeParameter,
                    NativeSGD, NativeAdam, NativeDropout, NativeMaxPool2d,
                    NativeReLU, NativeFlatten, NativeCrossEntropyLoss,
                    NativeMSELoss, NativeGenerator):
        parameters = inspect.signature(factory.__init__).parameters
        assert "device" not in parameters, factory.__name__
    # ...and the stateless ones still take no dtype either.
    for stateless in (NativeReLU, NativeFlatten, NativeMaxPool2d,
                      NativeDropout, NativeCrossEntropyLoss, NativeMSELoss,
                      NativeGenerator, NativeSequential):
        parameters = inspect.signature(stateless.__init__).parameters
        assert "dtype" not in parameters, stateless.__name__
