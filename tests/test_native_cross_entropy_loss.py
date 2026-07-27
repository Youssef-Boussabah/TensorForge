"""NativeCrossEntropyLoss — the native classification loss module
(Phase E, milestone E7).

The module is a **thin delegating wrapper**: its whole forward is
`logits.cross_entropy(targets, reduction=self.reduction)`. These tests
therefore do two things and only two things — prove the delegation is
real (the wrapper adds no arithmetic, no second target validator, and no
second reduction rule), and prove that delegating does not *weaken* any
E5/E6 contract. The full cross-entropy contract itself is pinned in
tests/test_native_cross_entropy_core.py (Core) and
tests/test_native_cross_entropy.py (autograd); nothing here duplicates
it.

Also covered: the parameter-free/buffer-free module state contract, the
empty `state_dict()` and strict loading, train/eval numerical
invariance, the repr, and the milestone's capability boundaries (E7 adds
no kernel, no ABI symbol, no operation, and no checkpoint change).

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Cleanup is explicit via close().

Selector: python -m pytest -q -k native_cross_entropy_loss
"""

import numpy as np
import pytest

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import (NativeCrossEntropyLoss, NativeLinear,
                                      NativeMSELoss, NativeParameter,
                                      NativeSequential, NativeTensor)

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)
needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)

LOGITS = np.array([[1.0, 2.0, 0.5], [-1.0, 0.25, 3.0]])
TARGETS = [1, 2]
REDUCTIONS = ("mean", "sum")


@pytest.fixture(autouse=True)
def _disarm_after_each():
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


def loss_reference(logits, targets, reduction):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    batch_size = logits.shape[0]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    per_example = (np.log(np.sum(np.exp(shifted), axis=1))
                   - shifted[np.arange(batch_size), targets])
    total = float(per_example.sum())
    return total / batch_size if reduction == "mean" else total


def grad_reference(logits, targets, reduction, upstream=1.0):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    base = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    base[np.arange(base.shape[0]), np.asarray(targets, dtype=np.int64)] -= 1.0
    if reduction == "mean":
        base /= base.shape[0]
    return upstream * base


def one_saved(output):
    resources = output._graph_resources
    assert len(resources) == 1, (
        f"expected exactly one saved-probability resource, got {resources!r}"
    )
    return resources[0]


# ======================================================================
# Delegation: the wrapper adds no mathematics of its own
# ======================================================================


@needs_native
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_native_cross_entropy_loss_calls_the_operation_verbatim(reduction,
                                                                monkeypatch):
    """The load-bearing test of this milestone: `forward` must reach
    `NativeTensor.cross_entropy` exactly once, with the module's
    reduction and the caller's targets passed straight through — and
    must return precisely that operation's result object."""
    calls = []
    original = NativeTensor.cross_entropy

    def recording(self, targets, reduction="mean"):
        calls.append((id(self), targets, reduction))
        return original(self, targets, reduction)

    monkeypatch.setattr(NativeTensor, "cross_entropy", recording)
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    caller_targets = list(TARGETS)
    loss = NativeCrossEntropyLoss(reduction)(x, caller_targets)
    monkeypatch.undo()

    assert len(calls) == 1, "the module did not delegate exactly once"
    logits_id, seen_targets, seen_reduction = calls[0]
    assert logits_id == id(x)
    assert seen_targets is caller_targets      # passed through, not copied
    assert seen_reduction == reduction
    assert isinstance(loss, NativeTensor)
    loss.close()
    x.close()


@needs_native
def test_native_cross_entropy_loss_uses_no_other_numerical_path(monkeypatch):
    """No softmax/log_softmax composition, no exp/log, no NumPy
    arithmetic: the module's forward runs to completion with every
    alternative route blocked, because the only route it takes is the
    fused E6 operation."""
    def _tripwire(*args, **kwargs):
        raise AssertionError("the loss module took a non-fused numerical path")

    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    for name in ("softmax", "log_softmax", "exp", "log"):
        monkeypatch.setattr(NativeTensor, name, _tripwire)
        monkeypatch.setattr(cpp.NativeTensorCore, name, _tripwire)
    for name in ("exp", "log", "logaddexp", "max", "amax", "argmax", "sum",
                 "divide", "true_divide", "subtract", "multiply", "mean",
                 "take", "take_along_axis"):
        monkeypatch.setattr(np, name, _tripwire)

    loss = NativeCrossEntropyLoss()(x, TARGETS)
    loss.backward()
    monkeypatch.undo()

    assert np.isclose(float(loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, "mean"), atol=1e-14)
    assert np.allclose(x.grad.to_numpy(),
                       grad_reference(LOGITS, TARGETS, "mean"), atol=1e-15)

    # ...and the sanity check that the tripwire can fire at all: the
    # module really does reach the fused Core forward.
    reached = []
    original = cpp.NativeTensorCore.cross_entropy_forward
    monkeypatch.setattr(
        cpp.NativeTensorCore, "cross_entropy_forward",
        lambda self, targets, reduction="mean": (
            reached.append(reduction), original(self, targets, reduction)
        )[1],
    )
    second = NativeCrossEntropyLoss("sum")(x, TARGETS)
    monkeypatch.undo()
    assert reached == ["sum"]
    for t in (loss, second, x):
        t.close()


@needs_native
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_native_cross_entropy_loss_forward_matches_the_operation(reduction):
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    y = NativeTensor.from_array(LOGITS, requires_grad=True)
    module_loss = NativeCrossEntropyLoss(reduction)(x, TARGETS)
    direct_loss = y.cross_entropy(TARGETS, reduction)
    assert float(module_loss.to_numpy()) == float(direct_loss.to_numpy())
    assert np.isclose(float(module_loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, reduction), atol=1e-14)
    for t in (module_loss, direct_loss, x, y):
        t.close()


@needs_native
def test_native_cross_entropy_loss_default_reduction_is_mean():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    default = NativeCrossEntropyLoss()
    explicit = NativeCrossEntropyLoss("mean")
    assert default.reduction == "mean"
    first = default(x, TARGETS)
    second = explicit(x, TARGETS)
    assert float(first.to_numpy()) == float(second.to_numpy())
    for t in (first, second, x):
        t.close()


@needs_native
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_native_cross_entropy_loss_backward_matches_the_operation(reduction):
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    y = NativeTensor.from_array(LOGITS, requires_grad=True)
    NativeCrossEntropyLoss(reduction)(x, TARGETS).backward()
    y.cross_entropy(TARGETS, reduction).backward()
    assert np.array_equal(x.grad.to_numpy(), y.grad.to_numpy())
    assert np.allclose(x.grad.to_numpy(),
                       grad_reference(LOGITS, TARGETS, reduction), atol=1e-15)
    x.close()
    y.close()


@needs_native
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_native_cross_entropy_loss_finite_differences(reduction):
    """The gradient is checked *through the module*, not only through the
    operation — moderate finite logits, central differences."""
    rng = np.random.default_rng(7)
    logits = rng.standard_normal((3, 4)) * 0.8
    targets = [2, 0, 3]
    loss_fn = NativeCrossEntropyLoss(reduction)
    x = NativeTensor.from_array(logits, requires_grad=True)
    loss_fn(x, targets).backward()
    analytic = x.grad.to_numpy()

    step = 1e-6
    numeric = np.zeros_like(logits)
    for row in range(logits.shape[0]):
        for column in range(logits.shape[1]):
            plus, minus = logits.copy(), logits.copy()
            plus[row, column] += step
            minus[row, column] -= step
            numeric[row, column] = (
                loss_reference(plus, targets, reduction)
                - loss_reference(minus, targets, reduction)
            ) / (2 * step)
    assert np.allclose(analytic, numeric, atol=1e-7)
    x.close()


@needs_native
def test_native_cross_entropy_loss_mean_is_sum_over_the_batch():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    mean = NativeCrossEntropyLoss("mean")(x, TARGETS)
    total = NativeCrossEntropyLoss("sum")(x, TARGETS)
    assert np.isclose(float(mean.to_numpy()) * LOGITS.shape[0],
                      float(total.to_numpy()), atol=1e-14)
    for t in (mean, total, x):
        t.close()


@needs_native
def test_native_cross_entropy_loss_scalar_output_and_graph_shape():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = NativeCrossEntropyLoss()(x, TARGETS)
    assert loss.shape == () and loss.numel == 1
    assert loss.requires_grad is True and loss.is_leaf is False
    assert loss._op == "cross_entropy"        # the operation's node, not a new one
    assert loss._parents == (x,)
    assert loss._expected_versions == ()
    loss.close()
    x.close()


# ======================================================================
# Inherited E5/E6 contracts — proved not to be weakened by the wrapper
# ======================================================================


@needs_native
def test_native_cross_entropy_loss_owns_one_graph_resource_with_e6_lifetime():
    loss_fn = NativeCrossEntropyLoss()
    x = NativeParameter(LOGITS)
    loss = loss_fn(x, TARGETS)
    saved = one_saved(loss)
    assert not saved._closed
    seed = NativeTensor.full((), 1.0)
    loss.backward(gradient=seed, retain_graph=True)
    assert not saved._closed                    # retained
    once = x.grad.to_numpy().copy()
    loss.backward(gradient=seed)                # one-shot: releases it
    assert saved._closed is True
    assert loss._graph_resources == ()
    assert np.allclose(x.grad.to_numpy(), 2 * once, atol=1e-15)
    for t in (loss, seed, x):
        t.close()


@needs_native
def test_native_cross_entropy_loss_no_grad_forward_keeps_no_graph_state():
    x = NativeTensor.from_array(LOGITS)          # requires_grad False
    loss = NativeCrossEntropyLoss()(x, TARGETS)
    assert loss.requires_grad is False and loss.is_leaf is True
    assert loss._graph_resources == () and loss._backward is None
    assert np.isclose(float(loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, "mean"), atol=1e-14)
    loss.close()
    x.close()


@needs_native
def test_native_cross_entropy_loss_abandoned_graph_releases_probabilities():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = NativeCrossEntropyLoss()(x, TARGETS)
    saved = one_saved(loss)
    loss.close()
    assert saved._closed is True
    loss.close()                                 # idempotent
    x.close()


@needs_native
@pytest.mark.parametrize("as_array", [False, True])
def test_native_cross_entropy_loss_caller_target_mutation_immunity(as_array):
    caller = np.array(TARGETS, dtype=np.int64) if as_array else list(TARGETS)
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = NativeCrossEntropyLoss()(x, caller)
    caller[0] = 0
    caller[1] = 0
    loss.backward()
    original = grad_reference(LOGITS, TARGETS, "mean")
    mutated = grad_reference(LOGITS, [0, 0], "mean")
    assert not np.allclose(original, mutated)
    assert np.allclose(x.grad.to_numpy(), original, atol=1e-15)
    loss.close()
    x.close()


@needs_native
def test_native_cross_entropy_loss_parameter_mutation_after_forward():
    """The E6 versioning contract survives the wrapper: no expected
    version is recorded, so a mutated logits parameter still
    differentiates against the probabilities the forward saved."""
    original = LOGITS.copy()
    x = NativeParameter(original)
    loss = NativeCrossEntropyLoss()(x, TARGETS)
    assert loss._expected_versions == ()
    replacement = NativeTensor.from_array(
        np.array([[-3.0, 0.5, 4.0], [2.0, -2.0, 0.25]])
    )
    x.copy_value_(replacement)
    loss.backward()                              # no stale-graph error
    expected = grad_reference(original, TARGETS, "mean")
    assert np.allclose(x.grad.to_numpy(), expected, atol=1e-15)
    assert not np.allclose(
        expected, grad_reference(replacement.to_numpy(), TARGETS, "mean"),
        atol=1e-3,
    )
    for t in (loss, replacement, x):
        t.close()


@needs_native
def test_native_cross_entropy_loss_non_contiguous_logits():
    base = NativeTensor.from_array(LOGITS.T, requires_grad=True)
    strided = base.T
    assert strided.contiguous is False
    loss = NativeCrossEntropyLoss("sum")(strided, TARGETS)
    assert np.isclose(float(loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, "sum"), atol=1e-14)
    loss.backward()
    assert np.allclose(base.grad.to_numpy(),
                       grad_reference(LOGITS, TARGETS, "sum").T, atol=1e-14)
    assert np.array_equal(base.to_numpy(), LOGITS.T)   # view unmutated
    for t in (loss, strided, base):
        t.close()


@needs_native
def test_native_cross_entropy_loss_narrowed_offset_logits():
    values = np.arange(12, dtype=float).reshape(4, 3) / 10.0
    base = NativeTensor.from_array(values, requires_grad=True)
    window = base.narrow(0, 1, 2)
    loss = NativeCrossEntropyLoss()(window, TARGETS)
    assert np.isclose(float(loss.to_numpy()),
                      loss_reference(values[1:3], TARGETS, "mean"), atol=1e-14)
    loss.backward()
    expected = np.zeros_like(values)
    expected[1:3] = grad_reference(values[1:3], TARGETS, "mean")
    assert np.allclose(base.grad.to_numpy(), expected, atol=1e-15)
    for t in (loss, window, base):
        t.close()


@needs_native
def test_native_cross_entropy_loss_in_a_model_graph():
    """The realistic path: a NativeLinear model, the loss module, and one
    backward reaching the model's parameters."""
    rng = np.random.default_rng(3)
    model = NativeSequential(NativeLinear(4, 3, seed=0))
    loss_fn = NativeCrossEntropyLoss()
    inputs = NativeTensor.from_array(rng.standard_normal((5, 4)))
    logits = model(inputs)
    loss = loss_fn(logits, [0, 2, 1, 1, 0])
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert np.all(np.isfinite(parameter.grad.to_numpy()))
    # The loss module contributed no state to the model it was used with.
    assert set(model.state_dict()) == {"0.weight", "0.bias"}
    for t in (loss, logits, inputs, *model.parameters()):
        t.close()


# ======================================================================
# Validation
# ======================================================================


@pytest.mark.parametrize("reduction", ["none", "Mean", "SUM", "", " mean"])
def test_native_cross_entropy_loss_unknown_reduction_rejected(reduction):
    """Rejected at construction, so an invalid reduction can never reach
    the operation. Runs without the compiled backend."""
    with pytest.raises(ValueError, match="reduction"):
        NativeCrossEntropyLoss(reduction)


@pytest.mark.parametrize("reduction", [None, 0, 1.0, True, ["mean"],
                                       ("mean",), b"mean"])
def test_native_cross_entropy_loss_non_string_reduction_rejected(reduction):
    with pytest.raises(TypeError, match="reduction"):
        NativeCrossEntropyLoss(reduction)


def test_native_cross_entropy_loss_reduction_errors_match_the_operation():
    """Same validator, therefore same error types and same messages as
    `NativeTensor.cross_entropy` — no second reduction rule exists."""
    for bad, error in (("none", ValueError), (None, TypeError)):
        with pytest.raises(error) as module_error:
            NativeCrossEntropyLoss(bad)
        with pytest.raises(error) as core_error:
            cpp._normalize_reduction(bad, "NativeCrossEntropyLoss")
        assert str(module_error.value) == str(core_error.value)


@needs_native
@pytest.mark.parametrize("logits", [
    np.array([[1.0, 2.0]]), [[1.0, 2.0]], 1.0, None,
])
def test_native_cross_entropy_loss_non_native_logits_rejected(logits):
    with pytest.raises(TypeError, match="NativeTensor"):
        NativeCrossEntropyLoss()(logits, [0])


@needs_native
def test_native_cross_entropy_loss_stable_tensor_logits_rejected():
    with pytest.raises(TypeError, match="NativeTensor"):
        NativeCrossEntropyLoss()(tensorforge.Tensor(LOGITS), TARGETS)


@needs_native
def test_native_cross_entropy_loss_closed_logits_rejected():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    x.close()
    with pytest.raises(RuntimeError, match="closed"):
        NativeCrossEntropyLoss()(x, TARGETS)


@needs_native
@pytest.mark.parametrize("targets, error", [
    ([1.0, 2.0], TypeError), ([True, False], TypeError),
    (np.array([1.0, 2.0]), TypeError), ("12", TypeError), (1, TypeError),
    (np.array([[1], [2]]), ValueError), ([1], ValueError),
    ([1, 3], ValueError), ([1, -1], ValueError), ([], ValueError),
])
def test_native_cross_entropy_loss_invalid_targets_propagate(targets, error):
    """Target validation is delegated in full: the module implements no
    second matrix, so the operation's errors surface unchanged."""
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    with pytest.raises(error):
        NativeCrossEntropyLoss()(x, targets)
    assert np.array_equal(x.to_numpy(), LOGITS)
    assert x.grad is None
    x.close()


@needs_native
@pytest.mark.parametrize("shape", [(3,), (2, 2, 2)])
def test_native_cross_entropy_loss_wrong_rank_rejected(shape):
    x = NativeTensor.zeros(shape, requires_grad=True)
    with pytest.raises(ValueError, match="2-D|batch_size"):
        NativeCrossEntropyLoss()(x, [0] * shape[0])
    x.close()


# ======================================================================
# Failure atomicity, inherited
# ======================================================================


@needs_native
@needs_fault_injection
def test_native_cross_entropy_loss_forward_allocation_failure_is_atomic():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss_fn = NativeCrossEntropyLoss()
    reached = 0
    for nth in range(1, 4):
        cpp._arm_alloc_failure(nth)
        try:
            loss_fn(x, TARGETS)
        except MemoryError:
            reached += 1
        finally:
            cpp._arm_alloc_failure(0)
            cpp._require_library().tf_clear_error()
        assert x.closed is False
        assert np.array_equal(x.to_numpy(), LOGITS)
        assert x.grad is None
    assert reached >= 2
    loss = loss_fn(x, TARGETS)                    # disarmed: works
    loss.backward()
    assert np.allclose(x.grad.to_numpy(),
                       grad_reference(LOGITS, TARGETS, "mean"), atol=1e-15)
    loss.close()
    x.close()


@needs_native
@needs_fault_injection
def test_native_cross_entropy_loss_backward_failure_rolls_back_and_retries():
    x = NativeParameter(LOGITS)
    seed = NativeTensor.full((), 1.0)
    loss = NativeCrossEntropyLoss()(x, TARGETS)
    loss.backward(gradient=seed, retain_graph=True)
    before = x.grad.to_numpy().copy()
    saved = one_saved(loss)

    cpp._arm_alloc_failure(1)
    with pytest.raises(MemoryError):
        loss.backward(gradient=seed, retain_graph=True)
    cpp._arm_alloc_failure(0)
    cpp._require_library().tf_clear_error()
    assert np.array_equal(x.grad.to_numpy(), before)   # no partial commit
    assert loss._graph_freed is False
    assert not saved._closed                           # retryable
    loss.backward(gradient=seed)
    assert np.allclose(x.grad.to_numpy(), 2 * before, atol=1e-15)
    assert saved._closed is True
    for t in (loss, seed, x):
        t.close()


# ======================================================================
# Module state contract
# ======================================================================


def test_native_cross_entropy_loss_is_a_native_module():
    from tensorforge.experimental import NativeModule

    loss_fn = NativeCrossEntropyLoss()
    assert isinstance(loss_fn, NativeModule)
    assert callable(loss_fn)
    assert loss_fn.training is True


def test_native_cross_entropy_loss_has_no_parameters_or_buffers():
    loss_fn = NativeCrossEntropyLoss("sum")
    assert loss_fn.parameters() == []
    assert list(loss_fn.named_parameters()) == []
    assert loss_fn.buffers() == []
    assert list(loss_fn.named_buffers()) == []
    assert list(loss_fn._modules) == []


def test_native_cross_entropy_loss_state_dict_is_empty():
    loss_fn = NativeCrossEntropyLoss("sum")
    state = loss_fn.state_dict()
    assert state == {}
    # The reduction is constructor configuration, never model state.
    assert "reduction" not in state
    loss_fn.load_state_dict({})                    # loading nothing succeeds
    assert loss_fn.reduction == "sum"
    assert loss_fn.state_dict() == {}


def test_native_cross_entropy_loss_strict_loading_rejects_unexpected_keys():
    loss_fn = NativeCrossEntropyLoss()
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        loss_fn.load_state_dict({"reduction": "sum"})
    assert loss_fn.reduction == "mean"             # unchanged by the failure


@needs_native
def test_native_cross_entropy_loss_train_eval_does_not_change_numerics():
    loss_fn = NativeCrossEntropyLoss()
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    training_loss = float(loss_fn(x, TARGETS).to_numpy())
    loss_fn.eval()
    assert loss_fn.training is False
    eval_loss = float(loss_fn(x, TARGETS).to_numpy())
    loss_fn.train()
    assert loss_fn.training is True
    again = float(loss_fn(x, TARGETS).to_numpy())
    assert training_loss == eval_loss == again
    x.close()


@needs_native
def test_native_cross_entropy_loss_inside_a_model_contributes_no_state():
    class Classifier(tensorforge.experimental.NativeModule):
        def __init__(self):
            super().__init__()
            self.linear = NativeLinear(4, 3, seed=1)
            self.criterion = NativeCrossEntropyLoss()

        def forward(self, inputs, targets):
            return self.criterion(self.linear(inputs), targets)

    model = Classifier()
    assert set(model.state_dict()) == {"linear.weight", "linear.bias"}
    for key in model.state_dict():
        assert "criterion" not in key
        assert "probabilit" not in key and "target" not in key
    assert model.buffers() == []
    for parameter in model.parameters():
        parameter.close()


def test_native_cross_entropy_loss_repr():
    assert repr(NativeCrossEntropyLoss()) == (
        "NativeCrossEntropyLoss(reduction='mean')"
    )
    assert repr(NativeCrossEntropyLoss("sum")) == (
        "NativeCrossEntropyLoss(reduction='sum')"
    )
    # The same shape the existing native loss uses.
    assert repr(NativeMSELoss("sum")) == "NativeMSELoss(reduction='sum')"


# ======================================================================
# Capability boundaries
# ======================================================================


def test_native_cross_entropy_loss_registry_placement():
    """A loss module is its own capability: NATIVE_LOSSES only, never an
    autograd op, a Core op, a kernel, or a model-building module."""
    assert "NativeCrossEntropyLoss" in cpp.NATIVE_LOSSES
    assert "NativeMSELoss" in cpp.NATIVE_LOSSES     # unchanged
    for wrong in (cpp.AUTOGRAD_OPS, cpp.TENSOR_CORE_OPS, cpp.RAW_KERNELS,
                  cpp.TENSOR_CORE_KERNELS, cpp.NATIVE_MODULES,
                  cpp.NATIVE_METRICS, cpp.UNSUPPORTED):
        assert "NativeCrossEntropyLoss" not in wrong
    # The operation it wraps stays exactly where E6 put it.
    assert "cross_entropy" in cpp.AUTOGRAD_OPS
    assert "cross_entropy_forward" in cpp.TENSOR_CORE_OPS
    assert "cross_entropy_backward" in cpp.TENSOR_CORE_OPS
    info = cpp.backend_info()
    assert "NativeCrossEntropyLoss" in info["native_losses"]
    assert "NativeCrossEntropyLoss" not in info["unsupported"]


def test_native_cross_entropy_loss_is_exported():
    import tensorforge.experimental as experimental

    assert "NativeCrossEntropyLoss" in experimental.__all__
    assert experimental.NativeCrossEntropyLoss is NativeCrossEntropyLoss
    # Not on the stable framework, which keeps its own separate function.
    assert not hasattr(tensorforge, "NativeCrossEntropyLoss")
    assert callable(tensorforge.nn.cross_entropy)


@needs_native
def test_native_cross_entropy_loss_scope_boundaries_hold():
    """E7's loss adds no operation, no kernel, and no later Phase-E
    surface."""
    import tensorforge.experimental as experimental

    loss_fn = NativeCrossEntropyLoss()
    # No numerical surface of its own.
    for absent in ("softmax", "log_softmax", "argmax", "to_numpy",
                   "cross_entropy_forward", "cross_entropy_backward"):
        assert not hasattr(loss_fn, absent), absent
    # No E8+ surface appeared.
    for absent in ("NativeNLLLoss", "NativeBCELoss", "NativeSoftmax",
                   "NativeLogSoftmax", "native_top_k_accuracy",
                   "native_confusion_matrix"):
        assert not hasattr(experimental, absent), absent
    # No class weights, ignore_index, or label smoothing crept in.
    for kwargs in ({"weight": None}, {"ignore_index": 0},
                   {"label_smoothing": 0.1}):
        with pytest.raises(TypeError):
            NativeCrossEntropyLoss(**kwargs)
    # No new checked ABI symbol for a loss.
    for absent in ("tf_core_cross_entropy_loss", "tf_core_nll_loss"):
        assert absent not in cpp._CHECKED_KERNELS, absent


def test_native_cross_entropy_loss_checkpoint_schema_is_untouched():
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT_VERSION == 2
    assert NativeCrossEntropyLoss("sum").state_dict() == {}
