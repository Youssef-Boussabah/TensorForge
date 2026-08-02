"""native_accuracy — the reporting-only native metric (Phase E,
milestone E7).

`native_accuracy(logits, targets) -> float` is deliberately **not**
native compute: there is no accuracy kernel, no C ABI export, no
`NativeTensorCore` method, and no autograd node. It validates rank-2
logits and targets under the same strict Phase-E contract the
cross-entropy forward uses, materializes the logits **once** through the
explicit public `to_numpy()` boundary, takes `numpy.argmax(axis=1)`, and
returns a plain Python `float` in `[0.0, 1.0]`.

That conversion is the whole point of these tests being different from
the cross-entropy ones. Native *training* operations are guarded by
tripwires that forbid any tensor-data NumPy round trip; this helper is
required to make one, so instead of a tripwire these tests
**instrument** the boundary and prove the conversion happens
deliberately, exactly once, through the public method — and that
`numpy.argmax` is what decides the winners, including its
first-maximal-index tie rule.

The other half of the contract is non-interference: even when the logits
require gradients, calling the metric must build no graph, touch no
`.grad`, no graph history, no `requires_grad` flag, and no
`NativeParameter` version, and must leave a pre-existing graph fully
usable.

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Cleanup is explicit via close().

Selector: python -m pytest -q -k native_metrics
"""

import numpy as np
import pytest

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import (NativeCrossEntropyLoss, NativeParameter,
                                      NativeTensor, native_accuracy)

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

LOGITS = np.array([[1.0, 2.0, 0.5], [-1.0, 0.25, 3.0]])
TARGETS = [1, 2]                     # both rows predicted correctly


@pytest.fixture(autouse=True)
def _disarm_after_each():
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


# ======================================================================
# Numerical behavior
# ======================================================================


@needs_native
def test_native_accuracy_perfect_predictions():
    x = NativeTensor.from_array(LOGITS)
    assert native_accuracy(x, TARGETS) == 1.0
    x.close()


@needs_native
def test_native_accuracy_no_correct_predictions():
    x = NativeTensor.from_array(LOGITS)
    assert native_accuracy(x, [0, 0]) == 0.0
    x.close()


@needs_native
def test_native_accuracy_partial():
    x = NativeTensor.from_array(LOGITS)
    assert native_accuracy(x, [1, 0]) == 0.5
    assert native_accuracy(x, [0, 2]) == 0.5
    x.close()


@needs_native
def test_native_accuracy_batch_size_one():
    x = NativeTensor.from_array(np.array([[0.5, 2.0, -1.0]]))
    assert native_accuracy(x, [1]) == 1.0
    assert native_accuracy(x, [0]) == 0.0
    x.close()


@needs_native
def test_native_accuracy_two_classes():
    logits = np.array([[1.0, -1.0], [-2.0, 3.0], [0.5, 0.25]])
    x = NativeTensor.from_array(logits)
    assert native_accuracy(x, [0, 1, 0]) == 1.0
    assert native_accuracy(x, [0, 1, 1]) == pytest.approx(2 / 3)
    x.close()


@needs_native
def test_native_accuracy_many_classes():
    rng = np.random.default_rng(9)
    logits = rng.standard_normal((7, 6))
    winners = np.argmax(logits, axis=1).tolist()
    x = NativeTensor.from_array(logits)
    assert native_accuracy(x, winners) == 1.0
    shifted = [(w + 1) % 6 for w in winners]
    assert native_accuracy(x, shifted) == 0.0
    x.close()


@needs_native
def test_native_accuracy_matches_the_plain_numpy_expression():
    """The definition, stated as a test: mean(argmax(logits, 1) ==
    targets) as a fraction — no rounding and no percentage."""
    rng = np.random.default_rng(12)
    logits = rng.standard_normal((9, 4))
    targets = rng.integers(0, 4, size=9).tolist()
    x = NativeTensor.from_array(logits)
    expected = float(np.mean(np.argmax(logits, axis=1) == np.array(targets)))
    result = native_accuracy(x, targets)
    assert result == expected
    assert 0.0 <= result <= 1.0
    x.close()


@needs_native
def test_native_accuracy_returns_a_builtin_python_float():
    x = NativeTensor.from_array(LOGITS)
    result = native_accuracy(x, [1, 0])
    assert type(result) is float                  # not np.float64
    assert not isinstance(result, np.floating)
    x.close()


@needs_native
def test_native_accuracy_is_deterministic_on_repeat():
    x = NativeTensor.from_array(LOGITS)
    results = [native_accuracy(x, [1, 0]) for _ in range(5)]
    assert results == [0.5] * 5
    assert np.array_equal(x.to_numpy(), LOGITS)   # and non-mutating
    x.close()


@needs_native
def test_native_accuracy_ties_follow_numpy_argmax_first_index():
    """No special tie semantics are invented: `numpy.argmax` gives the
    whole row to the **first** maximal index, and this helper adopts
    that rule unchanged."""
    logits = np.array([[1.0, 1.0, 1.0], [0.0, 5.0, 5.0]])
    assert np.argmax(logits, axis=1).tolist() == [0, 1]   # the premise
    x = NativeTensor.from_array(logits)
    assert native_accuracy(x, [0, 1]) == 1.0
    assert native_accuracy(x, [1, 2]) == 0.0      # later tied index loses
    assert native_accuracy(x, [2, 2]) == 0.0
    x.close()


@needs_native
@pytest.mark.parametrize("offset", [0.0, 700.0, -700.0, 1e10, -1e10])
def test_native_accuracy_large_finite_logits(offset):
    """A common offset never changes which class is largest, and no
    structural exception is raised for large finite values."""
    x = NativeTensor.from_array(LOGITS + offset)
    assert native_accuracy(x, TARGETS) == 1.0
    x.close()


@needs_native
def test_native_accuracy_all_negative_logits():
    logits = np.array([[-5.0, -1.0, -3.0], [-0.5, -9.0, -2.0]])
    x = NativeTensor.from_array(logits)
    assert native_accuracy(x, [1, 0]) == 1.0
    x.close()


# ======================================================================
# Views: non-contiguous and offset logits
# ======================================================================


@needs_native
def test_native_accuracy_transposed_logits():
    base = NativeTensor.from_array(LOGITS.T)
    strided = base.T
    assert strided.contiguous is False
    assert native_accuracy(strided, TARGETS) == 1.0
    assert native_accuracy(strided, [0, 0]) == 0.0
    assert np.array_equal(base.to_numpy(), LOGITS.T)   # view unmutated
    strided.close()
    base.close()


@needs_native
def test_native_accuracy_narrowed_nonzero_offset_logits():
    values = np.array([[9.0, 0.0, 0.0],
                       [1.0, 2.0, 0.5],
                       [-1.0, 0.25, 3.0],
                       [0.0, 0.0, 9.0]])
    base = NativeTensor.from_array(values)
    window = base.narrow(0, 1, 2)                  # rows 1..2
    assert native_accuracy(window, TARGETS) == 1.0
    assert native_accuracy(window, [0, 0]) == 0.0
    assert np.array_equal(base.to_numpy(), values)
    window.close()
    base.close()


@needs_native
def test_native_accuracy_transpose_of_a_narrow():
    values = np.arange(12, dtype=float).reshape(3, 4)
    base = NativeTensor.from_array(values)
    strided = base.narrow(1, 1, 2).T               # (2, 3), strided + offset
    logits = values[:, 1:3].T
    expected = np.argmax(logits, axis=1).tolist()
    assert native_accuracy(strided, expected) == 1.0
    strided.close()
    base.close()


# ======================================================================
# The deliberate conversion boundary
# ======================================================================


@needs_native
def test_native_accuracy_converts_through_public_to_numpy_exactly_once(
    monkeypatch
):
    """Instrumentation, not a tripwire: the metric is *required* to leave
    native memory, and this pins that it does so deliberately — through
    the public `NativeTensor.to_numpy()`, once per call, on the caller's
    own logits object."""
    calls = []
    original = NativeTensor.to_numpy

    def recording(self):
        calls.append(id(self))
        return original(self)

    monkeypatch.setattr(NativeTensor, "to_numpy", recording)
    x = NativeTensor.from_array(LOGITS)
    result = native_accuracy(x, TARGETS)
    monkeypatch.undo()

    assert result == 1.0
    assert calls == [id(x)], f"expected one to_numpy() on the logits, got {calls}"
    x.close()


@needs_native
def test_native_accuracy_uses_numpy_argmax_on_the_class_axis(monkeypatch):
    """`numpy.argmax(axis=1)` decides the predictions — dimension 1 is
    the class axis, as the rank-2 contract states."""
    seen = []
    original = np.argmax

    def recording(values, *args, **kwargs):
        seen.append((np.array(values, copy=True), args, kwargs))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np, "argmax", recording)
    x = NativeTensor.from_array(LOGITS)
    assert native_accuracy(x, TARGETS) == 1.0
    monkeypatch.undo()

    assert len(seen) == 1, "argmax was not the single prediction step"
    values, args, kwargs = seen[0]
    assert np.array_equal(values, LOGITS)          # the materialized logits
    assert kwargs.get("axis", args[0] if args else None) == 1
    x.close()


@needs_native
def test_native_accuracy_rejects_targets_before_materializing(monkeypatch):
    """Validation runs first: a rejected call converts nothing at all."""
    calls = []
    original = NativeTensor.to_numpy
    monkeypatch.setattr(
        NativeTensor, "to_numpy",
        lambda self: (calls.append(1), original(self))[1],
    )
    x = NativeTensor.from_array(LOGITS)
    with pytest.raises(TypeError):
        native_accuracy(x, [1.0, 2.0])
    with pytest.raises(ValueError):
        native_accuracy(x, [1, 9])
    monkeypatch.undo()
    assert calls == [], "a rejected metric call still materialized the logits"
    x.close()


@needs_native
def test_native_accuracy_leaves_the_logits_native_and_unchanged():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    native_accuracy(x, TARGETS)
    assert isinstance(x, NativeTensor)
    assert x.closed is False
    assert x.owns_core is True
    assert x.shape == LOGITS.shape
    assert x.dtype == "float64" and x.device == "cpu"
    assert np.array_equal(x.to_numpy(), LOGITS)
    x.close()


# ======================================================================
# Graph, gradient, and state non-interference
# ======================================================================


@needs_native
def test_native_accuracy_builds_no_graph_on_grad_tracking_logits():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    result = native_accuracy(x, TARGETS)
    assert type(result) is float
    assert x.requires_grad is True                 # untouched
    assert x.grad is None
    assert x.is_leaf is True
    assert x._parents == ()
    assert x._backward is None
    assert x._graph_resources == ()
    assert x._expected_versions == ()
    assert x._graph_freed is False
    x.close()


@needs_native
def test_native_accuracy_does_not_disturb_a_preexisting_graph():
    """The required sequence: build a graph, record its state, evaluate
    the metric, then prove the graph is still usable and its gradient is
    exactly what it would have been."""
    x = NativeParameter(LOGITS)
    loss = NativeCrossEntropyLoss()(x, TARGETS)
    saved = loss._graph_resources[0]
    before = (loss._op, loss._parents, loss._backward, loss._graph_freed,
              loss._expected_versions, loss._graph_resources)
    version_before = x._version

    accuracy = native_accuracy(x, TARGETS)

    # Nothing about the graph changed.
    assert accuracy == 1.0
    assert (loss._op, loss._parents, loss._backward, loss._graph_freed,
            loss._expected_versions, loss._graph_resources) == before
    assert not saved._closed                       # the metric freed nothing
    assert x.grad is None                          # and committed nothing
    assert x._version == version_before

    # The graph still works, and gives the untouched gradient.
    loss.backward()
    shifted = LOGITS - LOGITS.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    expected = probabilities.copy()
    expected[np.arange(2), TARGETS] -= 1.0
    expected /= 2
    assert np.allclose(x.grad.to_numpy(), expected, atol=1e-15)
    assert saved._closed is True                   # released by the backward
    loss.close()
    x.close()


@needs_native
def test_native_accuracy_does_not_alter_an_existing_gradient():
    x = NativeParameter(LOGITS)
    seed = NativeTensor.full((), 1.0)
    loss = NativeCrossEntropyLoss()(x, TARGETS)
    loss.backward(gradient=seed, retain_graph=True)
    gradient_object = x.grad
    before = x.grad.to_numpy().copy()

    native_accuracy(x, TARGETS)

    assert x.grad is gradient_object               # same object, not replaced
    assert np.array_equal(x.grad.to_numpy(), before)
    for t in (loss, seed, x):
        t.close()


@needs_native
def test_native_accuracy_does_not_bump_a_parameter_version():
    x = NativeParameter(LOGITS)
    assert x._version == 0
    for _ in range(3):
        native_accuracy(x, TARGETS)
    assert x._version == 0
    # ...and a graph built before the metric is still not stale.
    logged = x.log()
    native_accuracy(x, TARGETS)
    assert logged._expected_versions[0][2] == x._version
    logged.close()
    x.close()


@needs_native
def test_native_accuracy_retains_no_state():
    """No native output is created and nothing is retained: the metric is
    a pure function of its arguments."""
    x = NativeTensor.from_array(LOGITS)
    assert not hasattr(native_accuracy, "state")
    assert not hasattr(native_accuracy, "_cache")
    # Repeated calls with different targets never influence each other.
    assert native_accuracy(x, TARGETS) == 1.0
    assert native_accuracy(x, [0, 0]) == 0.0
    assert native_accuracy(x, TARGETS) == 1.0
    x.close()


@needs_native
def test_native_accuracy_allocates_no_native_storage(monkeypatch):
    """It creates no native output — no NativeStorage is constructed by
    the call at all."""
    created = []
    original = cpp.NativeStorage.__init__

    def tracking(self, *args, **kwargs):
        original(self, *args, **kwargs)
        created.append(id(self))

    x = NativeTensor.from_array(LOGITS)
    monkeypatch.setattr(cpp.NativeStorage, "__init__", tracking)
    assert native_accuracy(x, TARGETS) == 1.0
    monkeypatch.undo()
    assert created == [], "the metric allocated native storage"
    x.close()


# ======================================================================
# Strict target contract (shared with cross-entropy)
# ======================================================================


@needs_native
@pytest.mark.parametrize("targets", [
    [1, 2], (1, 2), np.array([1, 2], dtype=np.int8),
    np.array([1, 2], dtype=np.int32), np.array([1, 2], dtype=np.int64),
    np.array([1, 2], dtype=np.uint8), np.array([1, 2], dtype=np.uint32),
    np.array([1, 2], dtype=np.uint64), [np.int64(1), np.int32(2)],
    np.array([9, 1, 2, 9], dtype=np.int64)[1:3],          # contiguous slice
    np.array([1, 9, 2, 9], dtype=np.int64)[::2],          # non-contiguous view
])
def test_native_accuracy_accepts_the_same_target_forms_as_cross_entropy(targets):
    x = NativeTensor.from_array(LOGITS)
    assert native_accuracy(x, targets) == 1.0
    x.close()


@needs_native
@pytest.mark.parametrize("targets, error", [
    (True, TypeError), (np.bool_(True), TypeError),
    ([True, False], TypeError), (np.array([True, False]), TypeError),
    (1.0, TypeError), ([1.0, 2.0], TypeError), (np.array([1.0, 2.0]), TypeError),
    (np.array([1, 2], dtype=np.float32), TypeError),
    ([1, 2.0], TypeError), (1 + 2j, TypeError),
    (np.array([1 + 2j, 2 + 0j]), TypeError),
    ("12", TypeError), (b"12", TypeError), (bytearray(b"12"), TypeError),
    (np.array([1, 2.5], dtype=object), TypeError),
    ([[1], [2]], TypeError), (np.array([[1], [2]]), ValueError),
    (np.array(1), ValueError), (1, TypeError), (np.int64(1), TypeError),
    ([[1, 2], [3]], TypeError), (None, TypeError),
    ([1], ValueError), ([1, 2, 0], ValueError), ([], ValueError),
    ([1, -1], ValueError), ([1, 3], ValueError), ([1, 99], ValueError),
    ([1, 2 ** 64], ValueError),
])
def test_native_accuracy_rejects_the_same_target_forms_as_cross_entropy(
    targets, error
):
    x = NativeTensor.from_array(LOGITS)
    with pytest.raises(error):
        native_accuracy(x, targets)
    x.close()


@needs_native
@pytest.mark.parametrize("targets", [
    [1.0, 2.0], [True, False], "12", 1, np.array([[1], [2]]), [1],
    [1, 3], [1, -1], [], None,
])
def test_native_accuracy_target_errors_match_cross_entropy_exactly(targets):
    """One validator, one rule: the metric and the loss must raise the
    same exception type with the same message for the same bad input
    (only the operation name in the text differs)."""
    x = NativeTensor.from_array(LOGITS)
    with pytest.raises(Exception) as metric_error:
        native_accuracy(x, targets)
    with pytest.raises(Exception) as loss_error:
        x.cross_entropy(targets)
    assert type(metric_error.value) is type(loss_error.value)
    assert (str(metric_error.value).replace("native_accuracy", "OP")
            == str(loss_error.value).replace("cross_entropy_forward", "OP"))
    x.close()


@needs_native
def test_native_accuracy_uses_the_shared_target_helper(monkeypatch):
    """Structural proof that no second validator exists: blocking the E5
    preparation helper breaks the metric."""
    def _blocked(*args, **kwargs):
        raise AssertionError("the metric bypassed the shared target helper")

    monkeypatch.setattr(cpp, "_prepare_class_targets", _blocked)
    x = NativeTensor.from_array(LOGITS)
    with pytest.raises(AssertionError, match="bypassed the shared target"):
        native_accuracy(x, TARGETS)
    monkeypatch.undo()
    x.close()


@needs_native
def test_native_accuracy_target_mutation_after_the_call_changes_nothing():
    """The helper copies, so the caller's object is never aliased — and
    the returned float was computed before any later mutation anyway."""
    caller = np.array(TARGETS, dtype=np.int64)
    x = NativeTensor.from_array(LOGITS)
    first = native_accuracy(x, caller)
    caller[0] = 0
    assert first == 1.0
    assert native_accuracy(x, [1, 2]) == 1.0       # unaffected by the mutation
    x.close()


# ======================================================================
# Invalid logits
# ======================================================================


@needs_native
@pytest.mark.parametrize("logits", [
    LOGITS, [[1.0, 2.0]], 1.0, None, "logits",
])
def test_native_accuracy_non_native_logits_rejected(logits):
    with pytest.raises(TypeError, match="NativeTensor"):
        native_accuracy(logits, [0])


@needs_native
def test_native_accuracy_stable_tensor_logits_rejected():
    """No implicit dispatch: the stable framework's Tensor is not a
    native tensor and is not converted."""
    with pytest.raises(TypeError, match="NativeTensor"):
        native_accuracy(tensorforge.Tensor(LOGITS), TARGETS)


@needs_native
def test_native_accuracy_closed_logits_rejected():
    x = NativeTensor.from_array(LOGITS)
    x.close()
    with pytest.raises(RuntimeError, match="closed"):
        native_accuracy(x, TARGETS)


@needs_native
@pytest.mark.parametrize("shape", [(), (3,), (2, 2, 2), (1, 2, 3, 4)])
def test_native_accuracy_rank_other_than_two_rejected(shape):
    x = NativeTensor.zeros(shape) if shape else NativeTensor.full((), 1.0)
    with pytest.raises(ValueError, match="2-D|batch_size"):
        native_accuracy(x, [0])
    x.close()


@needs_native
def test_native_accuracy_failure_leaves_no_partial_state():
    x = NativeParameter(LOGITS)
    seed = NativeTensor.full((), 1.0)
    loss = NativeCrossEntropyLoss()(x, TARGETS)
    loss.backward(gradient=seed, retain_graph=True)
    before = x.grad.to_numpy().copy()
    version_before = x._version

    for bad in ([1.0, 2.0], [1, 3], [1], "12"):
        with pytest.raises((TypeError, ValueError)):
            native_accuracy(x, bad)

    assert np.array_equal(x.grad.to_numpy(), before)
    assert x._version == version_before
    assert not loss._graph_resources[0]._closed
    assert np.array_equal(x.to_numpy(), LOGITS)
    # The graph still runs after every rejected metric call.
    loss.backward(gradient=seed)
    assert np.allclose(x.grad.to_numpy(), 2 * before, atol=1e-15)
    for t in (loss, seed, x):
        t.close()


# ======================================================================
# Capability boundaries
# ======================================================================


def test_native_metrics_inventory_placement():
    """A reporting helper is its own capability: NATIVE_METRICS only.
    Runs without the compiled backend — pure inventory facts."""
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert "native_accuracy" in cpp.NATIVE_METRICS
    for wrong in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS,
                  cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS, cpp.NATIVE_MODULES,
                  cpp.NATIVE_LOSSES, cpp.NATIVE_OPTIMIZERS, cpp.STATE_SUPPORT,
                  cpp.UNSUPPORTED):
        assert "native_accuracy" not in wrong
    # No other inventory absorbed the metric, and no metric name leaked
    # into the loss inventory.
    for name in cpp.NATIVE_METRICS:
        assert name not in cpp.NATIVE_LOSSES, name


def test_native_metrics_backend_info_reports_the_inventory():
    info = cpp.backend_info()
    assert info["native_metrics"] == cpp.NATIVE_METRICS
    assert info["native_metrics"] is cpp.NATIVE_METRICS   # not a copy/literal
    assert "native_accuracy" not in info["unsupported"]
    assert "native_accuracy" not in info["autograd_ops"]
    assert "native_accuracy" not in info["tensor_core_ops"]
    assert "native_accuracy" not in info["native_modules"]
    # Implemented and unsupported inventories are disjoint. Phase G held
    # one deliberate exception for G3-G9 — "dropout" named both a shipped
    # operation and an unclosed capability (design §19) — and the G10
    # closure ended it. Nothing in this metric's inventory was ever
    # involved either way.
    implemented = (set(info["raw_kernels"]) | set(info["tensor_core_ops"])
                   | set(info["autograd_ops"]) | set(info["native_modules"])
                   | set(info["native_losses"]) | set(info["native_metrics"])
                   | set(info["native_optimizers"]))
    assert implemented & set(info["unsupported"]) == set()
    # Existing keys are unchanged.
    for key in ("raw_kernels", "kernels", "tensor_core_ops", "autograd_ops",
                "native_modules", "native_losses", "native_optimizers",
                "state_support", "unsupported"):
        assert key in info, key


def test_native_metrics_names_are_publicly_exported():
    import tensorforge.experimental as experimental

    for name in cpp.NATIVE_METRICS:
        assert hasattr(experimental, name), name
        assert name in experimental.__all__, name
        assert callable(getattr(experimental, name)), name


@needs_native
def test_native_accuracy_is_not_a_kernel_core_or_autograd_operation():
    """The honest claim: this is Python + NumPy over one explicit
    conversion, not native compute."""
    x = NativeTensor.from_array(LOGITS)
    core = cpp.NativeTensorCore.from_array(LOGITS)
    for absent in ("native_accuracy", "accuracy", "argmax"):
        assert not hasattr(x, absent), absent
        assert not hasattr(core, absent), absent
    # No ABI symbol exists for it, checked or otherwise.
    library = cpp._require_library()
    for symbol in ("tf_core_accuracy", "tf_core_argmax", "tf_native_accuracy"):
        assert symbol not in cpp._CHECKED_KERNELS, symbol
        assert not hasattr(library, symbol) or getattr(
            library, symbol, None) is None, symbol
    core.close()
    x.close()


@needs_native
def test_native_accuracy_scope_boundaries_hold():
    """E7's metric is accuracy only: no top-k, no per-class, no confusion
    matrix, no streaming or stateful metric, and no native argmax."""
    import tensorforge.experimental as experimental

    for absent in ("native_top_k_accuracy", "native_per_class_accuracy",
                   "native_confusion_matrix", "NativeAccuracy",
                   "NativeMetric", "native_precision", "native_recall",
                   "native_f1"):
        assert not hasattr(experimental, absent), absent
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    # No integer tensors appeared to support an index-producing reduction.
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    with pytest.raises(ValueError):
        cpp.NativeTensorCore.zeros((2, 2), dtype="int64")
    # The stable framework keeps its own accuracy, entirely separately.
    assert callable(tensorforge.accuracy)
    assert not hasattr(tensorforge, "native_accuracy")
    stable = tensorforge.accuracy(tensorforge.Tensor(LOGITS), [1, 2])
    assert isinstance(stable, float)


def test_native_accuracy_adds_no_persistent_state():
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT_VERSION == 3
    assert not hasattr(native_accuracy, "state_dict")
    assert not hasattr(native_accuracy, "parameters")
