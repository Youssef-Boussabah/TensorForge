"""Tests for NativeSGD — the first native optimizer (Advanced C++
v3.8, the eighth Phase C milestone).

NativeSGD(parameters, lr) stores unique open NativeParameter objects by
identity in first-occurrence order and applies ``value ← value -
lr * grad`` per step: frozen (requires_grad=False) parameters are
skipped before their gradients are examined, grad=None parameters are
skipped, every active gradient is validated (open, exact
shape/dtype/device), updates are staged natively at the graph-free
core level, and commits go through the v3.7 ``copy_value_()`` path —
identity, registration, requires_grad, and gradients (by identity and
value) preserved, one version increment per updated parameter, staged
temporaries always released. The public failure surface is
mutation-atomic: any preflight/validation/staging failure changes no
value, version, or gradient. Gradients persist until ``zero_grad()``.
No momentum, weight decay, parameter groups, optimizer state, or
training loop. See src/tensorforge/experimental/native_sgd.py.

NumPy appears below only for input preparation and exact references
(``lr * grad`` and the subtraction are single float64 operations, so
native and NumPy references are bit-identical); the staging and commit
paths are native, and a tripwire test proves it.

Selector: python -m pytest -q -k "native_sgd"
"""

import numpy as np
import pytest

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeLinear,
    NativeModule,
    NativeMSELoss,
    NativeParameter,
    NativeReLU,
    NativeSequential,
    NativeSGD,
    NativeTensor,
)

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


P_VALUES = np.array([[1.0, -2.0], [0.5, 3.0]])
G_VALUES = np.array([[0.5, -1.0], [2.0, 0.25]])
X_VALUES = np.array([[1.0, 2.0], [3.0, -1.0]])
LR = 0.5


def _param_with_grad(values=P_VALUES, grad_values=G_VALUES):
    """A NativeParameter whose grad is exactly ``grad_values``:
    d(sum(p * c))/dp = c, so one backward through multiply sets it."""
    parameter = NativeParameter(values)
    parameter.multiply(NativeTensor.from_array(grad_values)).sum().backward()
    return parameter


# ======================================================================
# Constructor: parameter storage
# ======================================================================


@needs_native
def test_native_sgd_constructs_from_lists_model_parameters_and_generators():
    layer = NativeLinear(2, 3, seed=0)
    from_model = NativeSGD(layer.parameters(), lr=LR)
    assert from_model.parameters() == [layer.weight, layer.bias]
    from_generator = NativeSGD(
        (parameter for parameter in layer.parameters()), lr=LR
    )
    assert from_generator.parameters() == [layer.weight, layer.bias]
    # The stored objects are the caller's exact parameters — never
    # copies — and construction touches nothing.
    assert from_model.parameters()[0] is layer.weight
    assert layer.weight.version == 0 and layer.weight.grad is None
    assert np.array_equal(
        from_model.parameters()[0].to_numpy(), layer.weight.to_numpy()
    )
    # The returned list is a snapshot: mutating it changes nothing.
    view = from_model.parameters()
    view.clear()
    assert from_model.parameters() == [layer.weight, layer.bias]


@needs_native
def test_native_sgd_deduplicates_by_identity_in_first_occurrence_order():
    a, b = NativeParameter(P_VALUES), NativeParameter(G_VALUES)
    optimizer = NativeSGD([a, b, a, a, b], lr=LR)
    assert optimizer.parameters() == [a, b]
    # Shared-module aliases collapse the same way.
    module = NativeModule()
    module.first = a
    module.second = a
    assert NativeSGD(module.parameters(), lr=LR).parameters() == [a]
    # Equal values are never deduplicated — identity only.
    twin = NativeParameter(P_VALUES)
    assert NativeSGD([a, twin], lr=LR).parameters() == [a, twin]


@needs_native
def test_native_sgd_rejects_bad_parameter_arguments():
    good = NativeParameter(P_VALUES)
    with pytest.raises(ValueError, match="at least one"):
        NativeSGD([], lr=LR)
    for not_iterable in (42, None, good):  # a bare parameter included
        with pytest.raises(TypeError, match="iterable"):
            NativeSGD(not_iterable, lr=LR)
    with pytest.raises(TypeError, match=r"parameters\[1\].*NativeTensor"):
        NativeSGD([good, NativeTensor.from_array(P_VALUES)], lr=LR)
    with pytest.raises(TypeError, match=r"parameters\[0\]"):
        NativeSGD("weights", lr=LR)  # str iterates to non-parameters
    with pytest.raises(TypeError, match=r"parameters\[1\].*Parameter"):
        NativeSGD([good, tensorforge.Parameter(P_VALUES)], lr=LR)
    closed = NativeParameter(P_VALUES)
    closed.close()
    with pytest.raises(RuntimeError, match=r"parameters\[1\].*closed"):
        NativeSGD([good, closed], lr=LR)
    # Constructor failure never closes or touches the caller's params.
    assert not good.closed and good.version == 0


# ======================================================================
# Constructor: learning-rate validation
# ======================================================================


@needs_native
def test_native_sgd_lr_accepts_real_values_and_normalizes_to_float():
    parameter = NativeParameter(P_VALUES)
    assert NativeSGD([parameter], lr=0.01).lr == 0.01
    from_int = NativeSGD([parameter], lr=1)
    assert from_int.lr == 1.0 and isinstance(from_int.lr, float)
    assert NativeSGD([parameter], lr=np.float64(0.25)).lr == 0.25
    with pytest.raises(AttributeError):
        from_int.lr = 0.5  # read-only


@needs_native
def test_native_sgd_lr_rejects_invalid_values():
    parameter = NativeParameter(P_VALUES)

    class Coercible:
        def __float__(self):
            return 0.01

    for bad_type in (True, False, "0.01", None, 1 + 0j, Coercible()):
        with pytest.raises(TypeError, match="real number"):
            NativeSGD([parameter], lr=bad_type)
    for bad_value in (0, 0.0, -0.01, -1):
        with pytest.raises(ValueError, match="strictly positive"):
            NativeSGD([parameter], lr=bad_value)
    for not_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            NativeSGD([parameter], lr=not_finite)
    assert not parameter.closed and parameter.version == 0


# ======================================================================
# Numerical updates
# ======================================================================


@needs_native
def test_native_sgd_step_single_parameter_exact_update():
    parameter = _param_with_grad()
    assert np.array_equal(parameter.grad.to_numpy(), G_VALUES)
    optimizer = NativeSGD([parameter], lr=LR)
    assert optimizer.step() is None
    assert np.array_equal(parameter.to_numpy(), P_VALUES - LR * G_VALUES)
    assert parameter.version == 1


@needs_native
def test_native_sgd_step_multiple_parameters_and_skips():
    active = _param_with_grad()
    no_grad = NativeParameter(P_VALUES)  # grad is None → skipped
    frozen = NativeParameter(G_VALUES, requires_grad=False)
    # A frozen parameter is skipped before its grad is examined — even
    # a (simulated) stale gradient never updates it.
    frozen._grad = NativeTensor.from_array(P_VALUES)
    zero_grad_update = _param_with_grad(G_VALUES, np.zeros((2, 2)))
    optimizer = NativeSGD([active, no_grad, frozen, zero_grad_update], lr=LR)
    optimizer.step()
    assert np.array_equal(active.to_numpy(), P_VALUES - LR * G_VALUES)
    assert active.version == 1
    assert np.array_equal(no_grad.to_numpy(), P_VALUES)
    assert no_grad.version == 0
    assert np.array_equal(frozen.to_numpy(), G_VALUES)
    assert frozen.version == 0
    # A numerically unchanged update (zero gradient) still replaced the
    # owned value — it increments exactly once.
    assert np.array_equal(zero_grad_update.to_numpy(), G_VALUES)
    assert zero_grad_update.version == 1


@needs_native
def test_native_sgd_repeated_steps_use_the_current_value():
    parameter = _param_with_grad()
    optimizer = NativeSGD([parameter], lr=LR)
    optimizer.step()
    optimizer.step()  # the gradient was deliberately retained
    assert np.array_equal(parameter.to_numpy(), P_VALUES - 2 * LR * G_VALUES)
    assert parameter.version == 2
    optimizer.step()
    assert np.array_equal(parameter.to_numpy(), P_VALUES - 3 * LR * G_VALUES)
    assert parameter.version == 3


# ======================================================================
# Identity, version, gradient, and zero_grad behavior
# ======================================================================


@needs_native
def test_native_sgd_step_preserves_identity_registration_and_gradients():
    module = NativeModule()
    parameter = _param_with_grad()
    module.weight = parameter
    module.alias = parameter
    grad_before = parameter.grad
    grad_values = grad_before.to_numpy()
    NativeSGD(module.parameters(), lr=LR).step()
    assert module.weight is parameter and module.alias is parameter
    assert parameter.requires_grad and parameter.is_leaf
    assert parameter.owns_core and not parameter.closed
    assert parameter._parents == () and parameter._backward is None
    assert parameter._op == "" and not parameter._graph_freed
    # The gradient survives by identity and value, until zero_grad().
    assert parameter.grad is grad_before
    assert np.array_equal(parameter.grad.to_numpy(), grad_values)
    assert not grad_before.closed


@needs_native
def test_native_sgd_zero_grad_clears_without_touching_values_or_versions():
    active = _param_with_grad()
    frozen = NativeParameter(G_VALUES, requires_grad=False)
    frozen._grad = NativeTensor.from_array(P_VALUES)  # simulated stale grad
    optimizer = NativeSGD([active, frozen], lr=LR)
    optimizer.step()
    value_after_step = active.to_numpy()
    assert optimizer.zero_grad() is None
    assert active.grad is None
    assert frozen.grad is None  # frozen gradients clear too
    assert np.array_equal(active.to_numpy(), value_after_step)
    assert active.version == 1 and frozen.version == 0
    # With every gradient cleared, the next step is a no-op.
    optimizer.step()
    assert active.version == 1


@needs_native
def test_native_sgd_shared_and_duplicate_parameters_update_once():
    shared = _param_with_grad()
    module = NativeModule()
    module.a = shared
    module.b = shared
    optimizer = NativeSGD(
        list(module.parameters()) + [shared, shared], lr=LR
    )
    assert optimizer.parameters() == [shared]
    optimizer.step()
    # One update — not one per reference — and one version increment.
    assert np.array_equal(shared.to_numpy(), P_VALUES - LR * G_VALUES)
    assert shared.version == 1
    assert module.a is shared and module.b is shared


# ======================================================================
# Graph safety
# ======================================================================


@needs_native
def test_native_sgd_step_builds_no_graph_and_uses_no_numpy(monkeypatch):
    parameter = _param_with_grad()
    optimizer = NativeSGD([parameter], lr=LR)

    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in ("add", "subtract", "multiply", "matmul", "sum", "mean",
                 "divide", "negative", "copyto"):
        monkeypatch.setattr(np, name, _tripwire)
    optimizer.step()
    optimizer.zero_grad()
    monkeypatch.undo()
    # The permanent parameter never became a graph node.
    assert parameter.is_leaf and parameter._parents == ()
    assert parameter._backward is None and parameter._op == ""
    assert np.array_equal(parameter.to_numpy(), P_VALUES - LR * G_VALUES)


# The matmul-sum gradient of a weight: d(sum(x @ w))/dw = x.T @ 1.
G_VALUES_FOR_MATMUL = X_VALUES.T @ np.ones((2, 2))


@needs_native
def test_native_sgd_step_makes_old_sensitive_graphs_stale():
    weight = NativeParameter(P_VALUES)
    x = NativeTensor.from_array(X_VALUES, requires_grad=True)
    out = x.matmul(weight).sum()
    out.backward(retain_graph=True)  # grads committed, graph retained
    x_grad_before = x.grad
    w_grad_before = weight.grad
    optimizer = NativeSGD([weight], lr=LR)
    optimizer.step()
    assert weight.version == 1
    # The retained value-sensitive graph is now stale — the existing
    # deterministic v3.7 error, with every gradient untouched.
    with pytest.raises(RuntimeError, match="stale"):
        out.backward(retain_graph=True)
    assert x.grad is x_grad_before and weight.grad is w_grad_before
    # A fresh forward/backward uses — and matches — the updated value.
    updated = P_VALUES - LR * G_VALUES_FOR_MATMUL
    weight.zero_grad()
    x.zero_grad()
    fresh = x.matmul(weight).sum()
    fresh.backward()
    ones = np.ones((2, 2))
    assert np.array_equal(x.grad.to_numpy(), ones @ updated.T)
    assert np.array_equal(weight.grad.to_numpy(), X_VALUES.T @ ones)


# ======================================================================
# Validation and failure atomicity
# ======================================================================


@needs_native
def test_native_sgd_step_bad_later_gradient_prevents_earlier_mutation():
    first = _param_with_grad()
    second = NativeParameter(G_VALUES)
    second._grad = NativeTensor.from_array([1.0, 2.0])  # shape mismatch
    optimizer = NativeSGD([first, second], lr=LR)
    with pytest.raises(ValueError, match=r"parameters\[1\].grad shape"):
        optimizer.step()
    # Mutation-atomic: the earlier valid parameter did not move.
    assert np.array_equal(first.to_numpy(), P_VALUES)
    assert first.version == 0 and second.version == 0
    assert first.grad is not None
    # dtype/device mismatched gradients are unconstructible on the
    # float64/cpu-only runtime; the check exists in step()'s preflight.


@needs_native
def test_native_sgd_step_closed_parameter_or_gradient_fails_before_mutation():
    first = _param_with_grad()
    second = _param_with_grad()
    optimizer = NativeSGD([first, second], lr=LR)
    second.grad.close()
    with pytest.raises(RuntimeError, match=r"parameters\[1\].grad.*closed"):
        optimizer.step()
    assert np.array_equal(first.to_numpy(), P_VALUES)
    assert first.version == 0 and second.version == 0
    # A parameter closed after construction fails the step preflight.
    third = _param_with_grad()
    late_closed = _param_with_grad()
    late_optimizer = NativeSGD([third, late_closed], lr=LR)
    late_closed.close()
    with pytest.raises(RuntimeError, match=r"parameters\[1\] has been closed"):
        late_optimizer.step()
    assert np.array_equal(third.to_numpy(), P_VALUES)
    assert third.version == 0
    assert not third.closed and third.grad is not None  # caller-owned


@needs_native
def test_native_sgd_staging_failure_changes_nothing_and_recovers(monkeypatch):
    first = _param_with_grad()
    second = _param_with_grad()
    optimizer = NativeSGD([first, second], lr=LR)
    real_full = cpp.NativeTensorCore.full
    calls = {"n": 0}

    def flaky_full(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # the second staged update's lr scalar
            raise MemoryError("forced staging failure")
        return real_full(*args, **kwargs)

    monkeypatch.setattr(cpp.NativeTensorCore, "full", flaky_full)
    with pytest.raises(MemoryError, match="forced staging failure"):
        optimizer.step()
    monkeypatch.undo()
    # No value, version, or gradient changed, and nothing leaked or
    # was double-closed: the same optimizer recovers completely.
    for parameter in (first, second):
        assert np.array_equal(parameter.to_numpy(), P_VALUES)
        assert parameter.version == 0
        assert parameter.grad is not None and not parameter.grad.closed
    optimizer.step()
    for parameter in (first, second):
        assert np.array_equal(parameter.to_numpy(), P_VALUES - LR * G_VALUES)
        assert parameter.version == 1


@needs_native
def test_native_sgd_zero_grad_preflight_prevents_partial_clearing():
    first = _param_with_grad()
    closed = _param_with_grad()
    optimizer = NativeSGD([first, closed], lr=LR)
    closed.close()
    with pytest.raises(RuntimeError, match=r"parameters\[1\]"):
        optimizer.zero_grad()
    assert first.grad is not None  # nothing was cleared


# ======================================================================
# Integration: model → loss → backward → step → zero_grad → fresh pass
# ======================================================================


@needs_native
def test_native_sgd_one_step_model_integration():
    model = NativeSequential(
        NativeLinear(2, 3, seed=0),
        NativeReLU(),
        NativeLinear(3, 2, seed=1),
    )
    loss_fn = NativeMSELoss()
    x = NativeTensor.from_array(X_VALUES)
    target = NativeTensor.from_array([[1.0, -0.5], [0.5, 2.0]])
    parameters = model.parameters()
    identities = [id(parameter) for parameter in parameters]

    loss_fn(model(x), target).backward()
    grads_before = [parameter.grad for parameter in parameters]
    assert all(grad is not None for grad in grads_before)
    values_before = [parameter.to_numpy() for parameter in parameters]
    grad_values = [grad.to_numpy() for grad in grads_before]

    optimizer = NativeSGD(model.parameters(), lr=0.1)
    optimizer.step()

    # Exact SGD arithmetic, identity stability, one increment each, and
    # gradients retained by identity and value.
    for parameter, value, grad, grad_value in zip(
        parameters, values_before, grads_before, grad_values
    ):
        assert np.array_equal(parameter.to_numpy(), value - 0.1 * grad_value)
        assert parameter.version == 1
        assert parameter.grad is grad
        assert np.array_equal(parameter.grad.to_numpy(), grad_value)
    assert [id(parameter) for parameter in model.parameters()] == identities

    optimizer.zero_grad()
    assert all(parameter.grad is None for parameter in parameters)
    assert [parameter.version for parameter in parameters] == [1, 1, 1, 1]

    # A fresh graph over the updated values trains on: forward, loss,
    # backward, and gradient shapes all behave normally.
    fresh_loss = loss_fn(model(x), target)
    fresh_loss.backward()
    assert all(
        parameter.grad is not None and parameter.grad.shape == parameter.shape
        for parameter in parameters
    )


@needs_native
def test_native_sgd_shared_parameter_model_compatibility():
    shared = NativeParameter(P_VALUES)
    module = NativeModule()
    module.a = shared
    module.b = shared
    x = NativeTensor.from_array(X_VALUES)
    # The shared parameter contributes through two use-sites; its
    # gradient accumulates both matmul contributions.
    x.matmul(module.a).add(x.matmul(module.b)).sum().backward()
    expected_grad = 2 * (X_VALUES.T @ np.ones((2, 2)))
    assert np.array_equal(shared.grad.to_numpy(), expected_grad)
    NativeSGD(module.parameters(), lr=LR).step()
    assert np.array_equal(
        shared.to_numpy(), P_VALUES - LR * expected_grad
    )
    assert shared.version == 1  # once, despite two aliases and two uses
