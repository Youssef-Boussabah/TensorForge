"""Tests for NativeAdam — the native adaptive optimizer (Advanced C++
v3.12, the twelfth Phase C milestone).

NativeAdam(parameters, lr, betas, eps) stores unique open
NativeParameter objects by identity in first-occurrence order, owns one
persistent native first/second-moment pair and one integer step counter
per entry (allocated eagerly after validation), and applies per active
parameter, entirely at the graph-free NativeTensorCore level::

    t      = previous_step + 1
    m_new  = beta1 * m + (1 - beta1) * g
    v_new  = beta2 * v + (1 - beta2) * (g * g)
    m_hat  = m_new * reciprocal(1 - beta1 ** t)
    v_hat  = v_new * reciprocal(1 - beta2 ** t)
    update = lr * m_hat * reciprocal(sqrt(v_hat) + eps)

committed through the v3.7 ``copy_value_()`` path — identity,
registration, requires_grad, and gradients preserved, one version and
one step-count increment per updated parameter, old moments closed only
after their staged replacements are installed. The public failure
surface is mutation-atomic; frozen and grad=None parameters are skipped
without aging their state; ``close()`` releases the optimizer-owned
moments exactly once. No weight decay, AMSGrad, parameter groups,
optimizer state_dict, or checkpointing. See
src/tensorforge/experimental/native_adam.py.

NumPy appears below only for input preparation and as the test oracle
(mirroring the native composition operation by operation, so the
comparison tolerance can stay at 1e-15); staging and commits are
native, and a tripwire test proves it.

Selector: python -m pytest -q -k "native_adam"
"""

import inspect
import math

import numpy as np
import pytest

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeLinear,
    NativeModule,
    NativeMSELoss,
    NativeParameter,
    NativeReLU,
    NativeSequential,
    NativeTensor,
)

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


P_VALUES = np.array([[1.0, -2.0], [0.5, 3.0]])
G_VALUES = np.array([[0.5, -1.0], [2.0, 0.25]])
X_VALUES = np.array([[1.0, 2.0], [3.0, -1.0]])
LR = 0.1
BETAS = (0.9, 0.999)
EPS = 1e-8


def _param_with_grad(values=P_VALUES, grad_values=G_VALUES):
    """A NativeParameter whose grad is exactly ``grad_values``:
    d(sum(p * c))/dp = c, so one backward through multiply sets it."""
    parameter = NativeParameter(values)
    parameter.multiply(NativeTensor.from_array(grad_values)).sum().backward()
    return parameter


def _set_grad(parameter, grad_values):
    """Replace ``parameter``'s gradient with exactly ``grad_values``
    through a fresh multiply/sum/backward pass."""
    parameter.zero_grad()
    parameter.multiply(NativeTensor.from_array(grad_values)).sum().backward()


def _adam_reference(value, grad, m, v, t, lr=LR, betas=BETAS, eps=EPS):
    """The test oracle: one Adam update mirroring the native staging
    composition operation by operation (reciprocals instead of
    division, the native multiplication order), so native results
    match it to float64 round-off. Returns (value_new, m_new, v_new)."""
    beta1, beta2 = betas
    m_new = beta1 * m + grad * (1.0 - beta1)
    v_new = beta2 * v + (grad * grad) * (1.0 - beta2)
    m_hat = m_new * (1.0 / (1.0 - beta1 ** t))
    v_hat = v_new * (1.0 / (1.0 - beta2 ** t))
    update = (m_hat * lr) * (1.0 / (np.sqrt(v_hat) + eps))
    return value - update, m_new, v_new


def _close(actual, expected):
    return np.allclose(actual, expected, rtol=0.0, atol=1e-15)


# ======================================================================
# 1. Constructor and parameter storage
# ======================================================================


@needs_native
def test_native_adam_constructs_from_lists_model_parameters_and_generators():
    layer = NativeLinear(2, 3, seed=0)
    from_model = NativeAdam(layer.parameters(), lr=LR)
    assert from_model.parameters() == [layer.weight, layer.bias]
    from_generator = NativeAdam(
        (parameter for parameter in layer.parameters()), lr=LR
    )
    assert from_generator.parameters() == [layer.weight, layer.bias]
    # The stored objects are the caller's exact parameters — never
    # copies — and construction touches nothing.
    assert from_model.parameters()[0] is layer.weight
    assert layer.weight.version == 0 and layer.weight.grad is None
    # The returned list is a snapshot: mutating it changes nothing.
    view = from_model.parameters()
    view.clear()
    assert from_model.parameters() == [layer.weight, layer.bias]


@needs_native
def test_native_adam_deduplicates_by_identity_in_first_occurrence_order():
    a, b = NativeParameter(P_VALUES), NativeParameter(G_VALUES)
    optimizer = NativeAdam([a, b, a, a, b], lr=LR)
    assert optimizer.parameters() == [a, b]
    assert optimizer.step_counts == (0, 0)  # one state entry per unique
    # Shared-module aliases collapse the same way.
    module = NativeModule()
    module.first = a
    module.second = a
    assert NativeAdam(module.parameters(), lr=LR).parameters() == [a]
    # Equal values are never deduplicated — identity only.
    twin = NativeParameter(P_VALUES)
    assert NativeAdam([a, twin], lr=LR).parameters() == [a, twin]


@needs_native
def test_native_adam_rejects_bad_parameter_arguments():
    good = NativeParameter(P_VALUES)
    with pytest.raises(ValueError, match="at least one"):
        NativeAdam([], lr=LR)
    for not_iterable in (42, None, good):  # a bare parameter included
        with pytest.raises(TypeError, match="iterable"):
            NativeAdam(not_iterable, lr=LR)
    with pytest.raises(TypeError, match=r"parameters\[1\].*NativeTensor"):
        NativeAdam([good, NativeTensor.from_array(P_VALUES)], lr=LR)
    with pytest.raises(TypeError, match=r"parameters\[0\]"):
        NativeAdam("weights", lr=LR)  # str iterates to non-parameters
    with pytest.raises(TypeError, match=r"parameters\[1\].*Parameter"):
        NativeAdam([good, tensorforge.Parameter(P_VALUES)], lr=LR)
    closed = NativeParameter(P_VALUES)
    closed.close()
    with pytest.raises(RuntimeError, match=r"parameters\[1\].*closed"):
        NativeAdam([good, closed], lr=LR)
    # Constructor failure never closes or touches the caller's params.
    assert not good.closed and good.version == 0


@needs_native
def test_native_adam_constructor_failure_releases_partial_state(monkeypatch):
    first = _param_with_grad()
    second = _param_with_grad()
    created = []
    real_zeros = NativeTensor.zeros

    def tracking_zeros(*args, **kwargs):
        if len(created) == 3:  # fail allocating the fourth buffer
            raise MemoryError("forced allocation failure")
        tensor = real_zeros(*args, **kwargs)
        created.append(tensor)
        return tensor

    monkeypatch.setattr(NativeTensor, "zeros", tracking_zeros)
    with pytest.raises(MemoryError, match="forced allocation failure"):
        NativeAdam([first, second], lr=LR)
    monkeypatch.undo()
    # Every optimizer-created buffer was released; the caller's
    # parameters and gradients were never closed or mutated.
    assert len(created) == 3
    assert all(buffer.closed for buffer in created)
    for parameter in (first, second):
        assert not parameter.closed and parameter.version == 0
        assert parameter.grad is not None and not parameter.grad.closed
        assert np.array_equal(parameter.to_numpy(), P_VALUES)


# ======================================================================
# 2. Hyperparameter validation
# ======================================================================


@needs_native
def test_native_adam_lr_accepts_real_values_and_normalizes_to_float():
    parameter = NativeParameter(P_VALUES)
    assert NativeAdam([parameter], lr=0.01).lr == 0.01
    from_int = NativeAdam([parameter], lr=1)
    assert from_int.lr == 1.0 and isinstance(from_int.lr, float)
    assert NativeAdam([parameter], lr=np.float64(0.25)).lr == 0.25
    assert NativeAdam([parameter]).lr == 0.001  # the documented default
    with pytest.raises(AttributeError):
        from_int.lr = 0.5  # read-only


@needs_native
def test_native_adam_lr_rejects_invalid_values():
    parameter = NativeParameter(P_VALUES)

    class Coercible:
        def __float__(self):
            return 0.01

    for bad_type in (True, False, "0.01", None, 1 + 0j, Coercible()):
        with pytest.raises(TypeError, match="lr must be a real number"):
            NativeAdam([parameter], lr=bad_type)
    for bad_value in (0, 0.0, -0.01, -1):
        with pytest.raises(ValueError, match="lr must be strictly positive"):
            NativeAdam([parameter], lr=bad_value)
    for not_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="lr must be finite"):
            NativeAdam([parameter], lr=not_finite)
    assert not parameter.closed and parameter.version == 0


@needs_native
def test_native_adam_betas_accept_valid_values_and_normalize():
    parameter = NativeParameter(P_VALUES)
    assert NativeAdam([parameter]).betas == (0.9, 0.999)  # the default
    assert NativeAdam([parameter], betas=[0.5, 0.75]).betas == (0.5, 0.75)
    # The 0.0 boundary is valid; integers normalize to floats.
    boundary = NativeAdam([parameter], betas=(0, 0.999))
    assert boundary.betas == (0.0, 0.999)
    assert all(isinstance(beta, float) for beta in boundary.betas)
    assert isinstance(boundary.betas, tuple)
    assert NativeAdam([parameter], betas=(np.float64(0.9), 0.999)).betas == BETAS
    with pytest.raises(AttributeError):
        boundary.betas = (0.8, 0.9)  # read-only


@needs_native
def test_native_adam_betas_reject_invalid_values():
    parameter = NativeParameter(P_VALUES)
    for bad_collection in (0.9, None, "09", {"beta1": 0.9, "beta2": 0.999}):
        with pytest.raises(TypeError, match="betas must be a tuple or list"):
            NativeAdam([parameter], betas=bad_collection)
    for bad_length in ((), (0.9,), (0.9, 0.99, 0.999)):
        with pytest.raises(ValueError, match="exactly two values"):
            NativeAdam([parameter], betas=bad_length)
    for bad_element in (True, "0.9", None):
        with pytest.raises(TypeError, match=r"betas\[0\] must be a real"):
            NativeAdam([parameter], betas=(bad_element, 0.999))
    for out_of_range in (1.0, 1.5, -0.1, -1.0):
        with pytest.raises(ValueError, match=r"betas\[1\].*0.0 <= beta < 1.0"):
            NativeAdam([parameter], betas=(0.9, out_of_range))
    for not_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match=r"betas\[0\] must be finite"):
            NativeAdam([parameter], betas=(not_finite, 0.999))
    assert not parameter.closed and parameter.version == 0


@needs_native
def test_native_adam_eps_validation():
    parameter = NativeParameter(P_VALUES)
    assert NativeAdam([parameter]).eps == 1e-8  # the default
    from_int = NativeAdam([parameter], eps=1)
    assert from_int.eps == 1.0 and isinstance(from_int.eps, float)

    class Coercible:
        def __float__(self):
            return 1e-8

    for bad_type in (True, False, "1e-8", None, Coercible()):
        with pytest.raises(TypeError, match="eps must be a real number"):
            NativeAdam([parameter], eps=bad_type)
    for bad_value in (0, 0.0, -1e-8):
        with pytest.raises(ValueError, match="eps must be strictly positive"):
            NativeAdam([parameter], eps=bad_value)
    for not_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="eps must be finite"):
            NativeAdam([parameter], eps=not_finite)
    with pytest.raises(AttributeError):
        from_int.eps = 1e-6  # read-only
    assert not parameter.closed and parameter.version == 0


# ======================================================================
# 3. State initialization and ownership
# ======================================================================


@needs_native
def test_native_adam_state_starts_at_zero_with_matching_metadata():
    a = _param_with_grad()
    b = NativeParameter(np.array([1.0, 2.0, 3.0]))
    optimizer = NativeAdam([a, b], lr=LR)
    assert optimizer.step_counts == (0, 0)
    assert isinstance(optimizer.step_counts, tuple)
    for index, parameter in enumerate((a, b)):
        for state in (optimizer._m[index], optimizer._v[index]):
            # Plain graph-free NativeTensor state — never a parameter,
            # never gradient-tracking, always fresh owning storage.
            assert isinstance(state, NativeTensor)
            assert not isinstance(state, NativeParameter)
            assert not state.requires_grad and state.is_leaf
            assert state.owns_core and not state.closed
            assert state.shape == parameter.shape
            assert state.dtype == parameter.dtype
            assert state.device == parameter.device
            assert np.array_equal(state.to_numpy(), np.zeros(parameter.shape))
    # Eager allocation is deterministic: a second optimizer over the
    # same parameters starts from the identical zero state.
    again = NativeAdam([a, b], lr=LR)
    assert again.step_counts == (0, 0)
    assert np.array_equal(again._m[0].to_numpy(), np.zeros(a.shape))


@needs_native
def test_native_adam_state_buffers_are_independent_and_unaliased():
    a = _param_with_grad()
    b = _param_with_grad()
    optimizer = NativeAdam([a, b], lr=LR)
    buffers = [*optimizer._m, *optimizer._v]
    # m and v never alias each other, another entry's state, a
    # parameter, or a gradient — distinct objects over distinct storage.
    assert len({id(buffer) for buffer in buffers}) == 4
    assert len({id(buffer._core.storage) for buffer in buffers}) == 4
    for buffer in buffers:
        assert buffer._core.storage is not a._core.storage
        assert buffer._core.storage is not b._core.storage
        assert buffer._core.storage is not a.grad._core.storage
    # State never directly changes a parameter's value or version.
    assert a.version == 0 and np.array_equal(a.to_numpy(), P_VALUES)


@needs_native
def test_native_adam_state_never_appears_in_model_state_dict():
    module = NativeModule()
    module.layer = NativeLinear(2, 3, seed=0)
    keys_before = list(module.state_dict())
    NativeAdam(module.parameters(), lr=LR)
    assert list(module.state_dict()) == keys_before
    assert keys_before == ["layer.weight", "layer.bias"]


# ======================================================================
# 4. One-step numerical correctness
# ======================================================================


@needs_native
def test_native_adam_single_parameter_one_step_exact():
    parameter = _param_with_grad()
    grad_before = parameter.grad
    optimizer = NativeAdam([parameter], lr=LR, betas=BETAS, eps=EPS)
    assert optimizer.step() is None
    expected_value, expected_m, expected_v = _adam_reference(
        P_VALUES, G_VALUES, np.zeros((2, 2)), np.zeros((2, 2)), t=1
    )
    assert _close(parameter.to_numpy(), expected_value)
    assert _close(optimizer._m[0].to_numpy(), expected_m)
    assert _close(optimizer._v[0].to_numpy(), expected_v)
    assert parameter.version == 1
    assert optimizer.step_counts == (1,)
    # The gradient survives by identity and value, until zero_grad().
    assert parameter.grad is grad_before
    assert np.array_equal(parameter.grad.to_numpy(), G_VALUES)
    assert not grad_before.closed


@needs_native
def test_native_adam_multiple_parameters_one_step_exact():
    first = _param_with_grad(P_VALUES, G_VALUES)
    second_grad = np.array([[-0.25, 4.0], [0.0, 1.5]])
    second = _param_with_grad(G_VALUES, second_grad)
    optimizer = NativeAdam([first, second], lr=LR, betas=BETAS, eps=EPS)
    optimizer.step()
    zeros = np.zeros((2, 2))
    for parameter, value, grad in (
        (first, P_VALUES, G_VALUES), (second, G_VALUES, second_grad)
    ):
        expected_value, _, _ = _adam_reference(value, grad, zeros, zeros, t=1)
        assert _close(parameter.to_numpy(), expected_value)
        assert parameter.version == 1
    assert optimizer.step_counts == (1, 1)


# ======================================================================
# 5. Repeated-step correctness
# ======================================================================


@needs_native
def test_native_adam_repeated_steps_with_changing_gradients_match_oracle():
    gradients = [
        G_VALUES,
        np.array([[-1.0, 0.5], [0.25, -2.0]]),
        np.array([[3.0, -0.125], [1.0, 0.75]]),
    ]
    parameter = NativeParameter(P_VALUES)
    optimizer = NativeAdam([parameter], lr=LR, betas=BETAS, eps=EPS)
    value = P_VALUES.copy()
    m = np.zeros((2, 2))
    v = np.zeros((2, 2))
    for step, grad in enumerate(gradients, start=1):
        _set_grad(parameter, grad)
        optimizer.step()
        # Moments persist across steps and updates read the parameter's
        # *current* value — the oracle carries both forward.
        value, m, v = _adam_reference(value, grad, m, v, t=step)
        assert _close(parameter.to_numpy(), value)
        assert _close(optimizer._m[0].to_numpy(), m)
        assert _close(optimizer._v[0].to_numpy(), v)
        assert parameter.version == step
        assert optimizer.step_counts == (step,)


@needs_native
def test_native_adam_trajectory_is_deterministic_across_runs():
    def run():
        parameter = NativeParameter(P_VALUES)
        optimizer = NativeAdam([parameter], lr=LR, betas=BETAS, eps=EPS)
        history = []
        for grad in (G_VALUES, -G_VALUES, 2.0 * G_VALUES):
            _set_grad(parameter, grad)
            optimizer.step()
            history.append(parameter.to_numpy())
        return history

    first_run = run()
    second_run = run()
    for one, other in zip(first_run, second_run):
        assert np.array_equal(one, other)  # bit-identical repeats


# ======================================================================
# 6. Skipping behavior
# ======================================================================


@needs_native
def test_native_adam_skips_frozen_and_gradientless_parameters():
    active = _param_with_grad()
    no_grad = NativeParameter(P_VALUES)  # grad is None → skipped
    frozen = NativeParameter(G_VALUES, requires_grad=False)
    # A frozen parameter is skipped before its grad is examined — even
    # a closed, invalid stale gradient is never inspected.
    stale = NativeTensor.from_array(P_VALUES)
    stale.close()
    frozen._grad = stale
    optimizer = NativeAdam([active, no_grad, frozen], lr=LR)
    m_no_grad = optimizer._m[1]
    v_frozen = optimizer._v[2]
    optimizer.step()
    # The active parameter advanced …
    assert active.version == 1 and optimizer.step_counts[0] == 1
    # … and every skipped parameter kept value, version, moments (by
    # identity and value), counter, and gradient state untouched.
    for index, parameter, value in ((1, no_grad, P_VALUES), (2, frozen, G_VALUES)):
        assert np.array_equal(parameter.to_numpy(), value)
        assert parameter.version == 0
        assert optimizer.step_counts[index] == 0
        assert np.array_equal(
            optimizer._m[index].to_numpy(), np.zeros((2, 2))
        )
    assert optimizer._m[1] is m_no_grad and optimizer._v[2] is v_frozen
    assert no_grad.grad is None
    assert frozen._grad is stale  # untouched, still the closed stale object


@needs_native
def test_native_adam_parameter_becoming_active_later_starts_at_step_one():
    early = _param_with_grad()
    late = NativeParameter(G_VALUES)
    optimizer = NativeAdam([early, late], lr=LR, betas=BETAS, eps=EPS)
    optimizer.step()
    assert optimizer.step_counts == (1, 0)
    # The late parameter becomes active now: its first update is the
    # t=1 bias-corrected step from zero moments — it never aged.
    late_grad = np.array([[0.5, 0.5], [-1.0, 2.0]])
    _set_grad(late, late_grad)
    optimizer.step()
    assert optimizer.step_counts == (2, 1)
    zeros = np.zeros((2, 2))
    expected_value, expected_m, expected_v = _adam_reference(
        G_VALUES, late_grad, zeros, zeros, t=1
    )
    assert _close(late.to_numpy(), expected_value)
    assert _close(optimizer._m[1].to_numpy(), expected_m)
    assert _close(optimizer._v[1].to_numpy(), expected_v)
    assert late.version == 1


@needs_native
def test_native_adam_shared_parameter_advances_once():
    shared = NativeParameter(P_VALUES)
    module = NativeModule()
    module.a = shared
    module.b = shared
    x = NativeTensor.from_array(X_VALUES)
    x.matmul(module.a).add(x.matmul(module.b)).sum().backward()
    expected_grad = 2 * (X_VALUES.T @ np.ones((2, 2)))
    assert np.array_equal(shared.grad.to_numpy(), expected_grad)
    optimizer = NativeAdam(
        list(module.parameters()) + [shared, shared], lr=LR
    )
    assert optimizer.parameters() == [shared]
    optimizer.step()
    # One entry, one update, one counter, one version — despite two
    # aliases, two uses, and duplicate references.
    zeros = np.zeros((2, 2))
    expected_value, _, _ = _adam_reference(
        P_VALUES, expected_grad, zeros, zeros, t=1
    )
    assert _close(shared.to_numpy(), expected_value)
    assert shared.version == 1
    assert optimizer.step_counts == (1,)


@needs_native
def test_native_adam_zero_valued_present_gradient_is_active():
    parameter = _param_with_grad(P_VALUES, np.zeros((2, 2)))
    optimizer = NativeAdam([parameter], lr=LR)
    m_before = optimizer._m[0]
    optimizer.step()
    # A zero gradient still replaces the owned value and advances the
    # state: version +1, count +1, fresh (still-zero) moment buffers.
    assert np.array_equal(parameter.to_numpy(), P_VALUES)  # update is 0
    assert parameter.version == 1
    assert optimizer.step_counts == (1,)
    assert optimizer._m[0] is not m_before  # replaced …
    assert m_before.closed                  # … and the old buffer released
    assert np.array_equal(optimizer._m[0].to_numpy(), np.zeros((2, 2)))


# ======================================================================
# 7. Gradient and zero_grad behavior
# ======================================================================


@needs_native
def test_native_adam_zero_grad_clears_gradients_and_preserves_state():
    active = _param_with_grad()
    frozen = NativeParameter(G_VALUES, requires_grad=False)
    frozen._grad = NativeTensor.from_array(P_VALUES)  # simulated stale grad
    optimizer = NativeAdam([active, frozen], lr=LR)
    optimizer.step()
    value_after_step = active.to_numpy()
    m_after_step = optimizer._m[0]
    m_values = m_after_step.to_numpy()
    assert optimizer.zero_grad() is None
    assert active.grad is None
    assert frozen.grad is None  # frozen gradients clear too
    # Values, versions, moments (identity and value), and counters all
    # survive zero_grad untouched.
    assert np.array_equal(active.to_numpy(), value_after_step)
    assert active.version == 1 and frozen.version == 0
    assert optimizer._m[0] is m_after_step
    assert np.array_equal(optimizer._m[0].to_numpy(), m_values)
    assert optimizer.step_counts == (1, 0)
    # With every gradient cleared, the next step is a no-op.
    optimizer.step()
    assert active.version == 1 and optimizer.step_counts == (1, 0)


@needs_native
def test_native_adam_zero_grad_preflight_prevents_partial_clearing():
    first = _param_with_grad()
    closed = _param_with_grad()
    optimizer = NativeAdam([first, closed], lr=LR)
    closed.close()
    with pytest.raises(RuntimeError, match=r"parameters\[1\]"):
        optimizer.zero_grad()
    assert first.grad is not None  # nothing was cleared


# ======================================================================
# 8. Failure atomicity
# ======================================================================


def _assert_untouched(optimizer, parameters, moments):
    """Every parameter value/version/gradient and every moment buffer
    (by identity and value) is exactly as captured in ``moments``."""
    for index, parameter in enumerate(parameters):
        assert np.array_equal(parameter.to_numpy(), P_VALUES)
        assert parameter.version == 0
        assert parameter.grad is not None and not parameter.grad.closed
        m_before, v_before = moments[index]
        assert optimizer._m[index] is m_before
        assert optimizer._v[index] is v_before
        assert np.array_equal(m_before.to_numpy(), np.zeros((2, 2)))
        assert np.array_equal(v_before.to_numpy(), np.zeros((2, 2)))
    assert optimizer.step_counts == (0,) * len(parameters)


@needs_native
def test_native_adam_bad_later_gradient_prevents_earlier_update():
    first = _param_with_grad()
    second = _param_with_grad()
    optimizer = NativeAdam([first, second], lr=LR)
    moments = [(optimizer._m[i], optimizer._v[i]) for i in range(2)]
    second._grad = NativeTensor.from_array([1.0, 2.0])  # shape mismatch
    with pytest.raises(ValueError, match=r"parameters\[1\].grad shape"):
        optimizer.step()
    second._grad = _param_with_grad().grad  # restore a valid-shape grad
    _assert_untouched(optimizer, (first, second), moments)
    # dtype/device mismatched gradients are unconstructible on the
    # float64/cpu-only runtime; the checks exist in step()'s preflight.
    # The same optimizer completes a later valid step.
    optimizer.step()
    assert first.version == 1 and second.version == 1
    assert optimizer.step_counts == (1, 1)


@needs_native
def test_native_adam_closed_gradient_or_parameter_fails_before_mutation():
    first = _param_with_grad()
    second = _param_with_grad()
    optimizer = NativeAdam([first, second], lr=LR)
    moments = [(optimizer._m[i], optimizer._v[i]) for i in range(2)]
    second.grad.close()
    with pytest.raises(RuntimeError, match=r"parameters\[1\].grad.*closed"):
        optimizer.step()
    assert np.array_equal(first.to_numpy(), P_VALUES)
    assert first.version == 0 and second.version == 0
    assert optimizer._m[0] is moments[0][0]
    # A parameter closed after construction fails the step preflight.
    third = _param_with_grad()
    late_closed = _param_with_grad()
    late_optimizer = NativeAdam([third, late_closed], lr=LR)
    late_closed.close()
    with pytest.raises(RuntimeError, match=r"parameters\[1\] has been closed"):
        late_optimizer.step()
    assert np.array_equal(third.to_numpy(), P_VALUES)
    assert third.version == 0 and late_optimizer.step_counts == (0, 0)
    assert not third.closed and third.grad is not None  # caller-owned


@needs_native
def test_native_adam_closed_later_state_prevents_earlier_update():
    first = _param_with_grad()
    second = _param_with_grad()
    optimizer = NativeAdam([first, second], lr=LR)
    optimizer._v[1].close()  # white-box corruption of a later entry
    with pytest.raises(RuntimeError, match=r"v state for parameters\[1\]"):
        optimizer.step()
    # The earlier valid entry did not move.
    assert np.array_equal(first.to_numpy(), P_VALUES)
    assert first.version == 0 and second.version == 0
    assert optimizer.step_counts == (0, 0)
    assert first.grad is not None and second.grad is not None


@needs_native
def test_native_adam_staging_failure_changes_nothing_and_recovers(monkeypatch):
    first = _param_with_grad()
    second = _param_with_grad()
    optimizer = NativeAdam([first, second], lr=LR, betas=BETAS, eps=EPS)
    moments = [(optimizer._m[i], optimizer._v[i]) for i in range(2)]
    real_full = cpp.NativeTensorCore.full
    calls = {"n": 0}

    def flaky_full(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 9:  # the second entry's first scalar: entry 1
            raise MemoryError("forced staging failure")  # is fully staged
        return real_full(*args, **kwargs)

    monkeypatch.setattr(cpp.NativeTensorCore, "full", flaky_full)
    with pytest.raises(MemoryError, match="forced staging failure"):
        optimizer.step()
    monkeypatch.undo()
    # A failure *after* at least one fully staged entry still commits
    # nothing: no value, version, moment, counter, or gradient changed,
    # nothing leaked or was double-closed, and the same optimizer
    # recovers completely on the next valid step.
    _assert_untouched(optimizer, (first, second), moments)
    optimizer.step()
    zeros = np.zeros((2, 2))
    expected_value, _, _ = _adam_reference(P_VALUES, G_VALUES, zeros, zeros, t=1)
    for parameter in (first, second):
        assert _close(parameter.to_numpy(), expected_value)
        assert parameter.version == 1
    assert optimizer.step_counts == (1, 1)


# ======================================================================
# 9. Versioning and graph safety
# ======================================================================


# The matmul-sum gradient of a weight: d(sum(x @ w))/dw = x.T @ 1.
G_VALUES_FOR_MATMUL = X_VALUES.T @ np.ones((2, 2))


@needs_native
def test_native_adam_step_makes_old_sensitive_graphs_stale():
    weight = NativeParameter(P_VALUES)
    x = NativeTensor.from_array(X_VALUES, requires_grad=True)
    out = x.matmul(weight).sum()
    out.backward(retain_graph=True)  # grads committed, graph retained
    x_grad_before = x.grad
    w_grad_before = weight.grad
    optimizer = NativeAdam([weight], lr=LR)
    optimizer.step()
    assert weight.version == 1
    # The retained value-sensitive graph is now stale — the existing
    # deterministic v3.7 error, with every gradient untouched.
    with pytest.raises(RuntimeError, match="stale"):
        out.backward(retain_graph=True)
    assert x.grad is x_grad_before and weight.grad is w_grad_before
    # A fresh forward/backward uses — and matches — the updated value,
    # and the parameter's identity and registration stayed stable.
    zeros = np.zeros((2, 2))
    updated, _, _ = _adam_reference(
        P_VALUES, G_VALUES_FOR_MATMUL, zeros, zeros, t=1
    )
    weight.zero_grad()
    x.zero_grad()
    fresh = x.matmul(weight).sum()
    fresh.backward()
    ones = np.ones((2, 2))
    assert _close(x.grad.to_numpy(), ones @ updated.T)
    assert np.array_equal(weight.grad.to_numpy(), X_VALUES.T @ ones)


@needs_native
def test_native_adam_step_preserves_identity_registration_and_leaf_state():
    module = NativeModule()
    parameter = _param_with_grad()
    module.weight = parameter
    module.alias = parameter
    NativeAdam(module.parameters(), lr=LR).step()
    assert module.weight is parameter and module.alias is parameter
    assert parameter.requires_grad and parameter.is_leaf
    assert parameter.owns_core and not parameter.closed
    assert parameter._parents == () and parameter._backward is None
    assert parameter._op == "" and not parameter._graph_freed
    assert parameter.version == 1  # moved by copy_value_ alone


# ======================================================================
# 10. Optimizer lifetime
# ======================================================================


@needs_native
def test_native_adam_close_releases_state_and_is_idempotent():
    parameter = _param_with_grad()
    optimizer = NativeAdam([parameter], lr=LR)
    optimizer.step()
    value_after_step = parameter.to_numpy()
    grad_after_step = parameter.grad
    m_buffer, v_buffer = optimizer._m[0], optimizer._v[0]
    assert not optimizer.closed
    assert optimizer.close() is None
    assert optimizer.closed
    assert m_buffer.closed and v_buffer.closed
    optimizer.close()  # idempotent — no double-close of the buffers
    with pytest.raises(RuntimeError, match="closed"):
        optimizer.step()
    with pytest.raises(RuntimeError, match="closed"):
        optimizer.zero_grad()
    # Caller-owned objects are untouched: parameter open, value and
    # version as after the step, gradient alive by identity and value.
    assert not parameter.closed
    assert np.array_equal(parameter.to_numpy(), value_after_step)
    assert parameter.version == 1
    assert parameter.grad is grad_after_step and not grad_after_step.closed
    # The documented post-close introspection surface stays readable.
    assert optimizer.parameters() == [parameter]
    assert optimizer.lr == LR and optimizer.betas == BETAS
    assert optimizer.eps == EPS
    assert optimizer.step_counts == (1,)
    assert repr(optimizer) == "NativeAdam(closed)"


@needs_native
def test_native_adam_context_manager_closes_state():
    parameter = _param_with_grad()
    with NativeAdam([parameter], lr=LR) as optimizer:
        optimizer.step()
        buffers = [optimizer._m[0], optimizer._v[0]]
    assert optimizer.closed
    assert all(buffer.closed for buffer in buffers)
    assert not parameter.closed and parameter.grad is not None


# ======================================================================
# 11. Native-only guardrails
# ======================================================================


@needs_native
def test_native_adam_step_builds_no_graph_and_uses_no_numpy(monkeypatch):
    parameter = _param_with_grad()
    optimizer = NativeAdam([parameter], lr=LR, betas=BETAS, eps=EPS)

    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in ("sqrt", "reciprocal", "divide", "add", "subtract",
                 "multiply", "matmul", "sum", "mean", "negative",
                 "power", "copyto"):
        monkeypatch.setattr(np, name, _tripwire)
    optimizer.step()
    optimizer.zero_grad()
    monkeypatch.undo()
    # Neither the parameter nor the optimizer state became graph nodes.
    assert parameter.is_leaf and parameter._parents == ()
    assert parameter._backward is None and parameter._op == ""
    for state in (optimizer._m[0], optimizer._v[0]):
        assert state.is_leaf and not state.requires_grad
        assert state._parents == () and state._backward is None
    zeros = np.zeros((2, 2))
    expected_value, _, _ = _adam_reference(P_VALUES, G_VALUES, zeros, zeros, t=1)
    assert _close(parameter.to_numpy(), expected_value)


@needs_native
def test_native_adam_scope_boundaries_hold():
    parameter = _param_with_grad()
    optimizer = NativeAdam([parameter], lr=LR)
    # No PyTorch-style extras and no division. In-memory
    # state_dict/load_state_dict shipped in v3.13 (see
    # tests/test_native_optimizer_state.py); file checkpointing has not.
    for absent in ("add_param_group", "param_groups", "weight_decay",
                   "amsgrad", "save", "load", "save_checkpoint",
                   "load_checkpoint"):
        assert not hasattr(optimizer, absent)
    assert hasattr(optimizer, "state_dict")
    assert hasattr(optimizer, "load_state_dict")
    assert not hasattr(NativeTensor.from_array(P_VALUES), "divide")
    assert not hasattr(parameter, "add_")  # no in-place arithmetic
    # zero_grad has no set_to_none (or any other) option.
    assert len(inspect.signature(optimizer.zero_grad).parameters) == 0
    # The stable Adam is untouched — it keeps its own state_dict
    # surface, fully separate from the native optimizer.
    assert hasattr(tensorforge.Adam, "state_dict")


# ======================================================================
# 12. Integration: model → loss → backward → step → zero_grad → repeat
# ======================================================================


@needs_native
def test_native_adam_trains_a_native_mlp_deterministically():
    model = NativeSequential(
        NativeLinear(2, 8, seed=0),
        NativeReLU(),
        NativeLinear(8, 1, seed=1),
    )
    loss_fn = NativeMSELoss()
    x = NativeTensor.from_array(X_VALUES)
    target = NativeTensor.from_array([[1.0], [-0.5]])
    parameters = model.parameters()
    identities = [id(parameter) for parameter in parameters]
    optimizer = NativeAdam(model.parameters(), lr=0.05)

    steps = 20
    losses = []
    for _ in range(steps):
        prediction = model(x)
        loss = loss_fn(prediction, target)
        losses.append(float(loss.to_numpy()))
        loss.backward()
        assert all(parameter.grad is not None for parameter in parameters)
        optimizer.step()
        # Gradients are retained through the step …
        assert all(parameter.grad is not None for parameter in parameters)
        optimizer.zero_grad()
        loss.close()
        prediction.close()

    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < 0.5 * losses[0]  # meaningful reduction
    # Every trainable parameter received state and updates: version
    # delta equals the update count, counters advanced once per step,
    # and the moment buffers moved off zero.
    assert [parameter.version for parameter in parameters] == [steps] * 4
    assert optimizer.step_counts == (steps,) * 4
    for index in range(4):
        assert not np.array_equal(
            optimizer._m[index].to_numpy(),
            np.zeros(parameters[index].shape),
        )
    # Identity and registration stayed stable; final gradients cleared;
    # a fresh forward/backward still works after the run.
    assert [id(parameter) for parameter in model.parameters()] == identities
    assert all(parameter.grad is None for parameter in parameters)
    fresh_loss = loss_fn(model(x), target)
    fresh_loss.backward()
    assert all(
        parameter.grad is not None and parameter.grad.shape == parameter.shape
        for parameter in parameters
    )
    optimizer.close()
    assert all(not parameter.closed for parameter in parameters)
