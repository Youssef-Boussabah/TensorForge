"""Tests for the native logarithm — Phase E, milestone E2.

The second Phase-E stable-math primitive, and the phase's deliberate
counterpart to E1's exponential
(docs/native_classification_design.md §4.2, §5). One shape-preserving
unary operation through the complete native stack: the C++ kernel
(odometer + contiguous fast path) → the guarded, self-validating C ABI
pair (``tf_core_log`` / ``tf_core_log_contiguous``, reusing E1's
validators unchanged) → ctypes → ``NativeTensorCore.log()`` →
differentiable ``NativeTensor.log()``.

The autograd design is a **live-input backward**, the opposite of
``exp``: d(log(x))/dx = 1/x cannot be recovered from the saved output
``log(x)``, so the backward computes ``upstream * reciprocal(input)``
by rereading the parent's *current* value. That makes a direct
``NativeParameter`` parent **version-guarded** (v3.7): mutating it after
forward raises the deterministic stale-graph error, and — because the
engine validates every recorded version before the snapshot, the seed,
and any callback — the failure commits no gradient anywhere in the
graph. ``exp`` records no version and stays valid across the same
mutation; both are asserted here so the contrast cannot silently erode.

Forward is unclamped IEEE ``std::log``: log(1) == 0, log(±0) == -inf,
log(negative) is NaN, log(+inf) == +inf, NaN propagates. Those are
**values**, not ABI errors.

NumPy appears below only as an external oracle and for inspection (with
expected warnings suppressed locally via ``np.errstate``); a tripwire
test proves the forward/backward paths never compute with it.

Selector: python -m pytest -q -k "native_log"
"""

import math

import numpy as np
import pytest

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import NativeParameter, NativeTensor

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)
needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)


# Safely positive and well conditioned: exact and finite-difference
# comparisons are both meaningful, and 1/x stays bounded.
VALUES = np.array([[1.0, 2.0, 0.5], [4.0, 0.25, 10.0]])


@pytest.fixture(autouse=True)
def _disarm_after_each():
    """No test may leave the allocation injector armed or the native
    error slot dirty (the test_native_abi_errors.py convention)."""
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


def _numeric_grad(f, x, eps=1e-6):
    """Central finite differences of the scalar-valued ``f`` at ``x``."""
    x = np.array(x, dtype=np.float64)
    grad = np.zeros_like(x)
    flat, gflat = x.ravel(), grad.ravel()
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        f_plus = f(x)
        flat[i] = original - eps
        f_minus = f(x)
        flat[i] = original
        gflat[i] = (f_plus - f_minus) / (2 * eps)
    return grad


# ======================================================================
# Kernel symbols and the capability boundary
# ======================================================================


@needs_native
def test_native_log_kernel_symbols_are_bound():
    library = cpp._require_library()
    for name in ("tf_core_log", "tf_core_log_contiguous"):
        assert getattr(library, name) is not None
        # Both participate in the error contract (they self-validate).
        assert name in cpp._CHECKED_KERNELS
        # An ABI symbol is not a capability name.
        assert name not in cpp.RAW_KERNELS
        assert name not in cpp.TENSOR_CORE_OPS
    # E2 added no raw NumPy-buffer kernel and did not touch the frozen
    # historical registry.
    assert cpp.TENSOR_CORE_KERNELS == ("relu", "add", "subtract",
                                       "multiply", "matmul")
    assert "log" not in cpp.RAW_KERNELS
    assert not hasattr(cpp, "log")


def test_native_log_registry_placement():
    """log is a Core op and an autograd op — nothing else. Runs without
    the compiled backend: these are pure inventory facts."""
    assert "log" in cpp.TENSOR_CORE_OPS
    assert "log" in cpp.AUTOGRAD_OPS
    assert "log" not in cpp.UNSUPPORTED
    assert "log" not in cpp.NATIVE_MODULES
    assert "log" not in cpp.NATIVE_LOSSES
    assert not hasattr(cpp, "NATIVE_METRICS")
    # E1 stays implemented alongside it.
    assert "exp" in cpp.TENSOR_CORE_OPS and "exp" in cpp.AUTOGRAD_OPS
    assert "exp" not in cpp.UNSUPPORTED
    info = cpp.backend_info()
    assert "log" in info["tensor_core_ops"] and "log" in info["autograd_ops"]
    assert "log" not in info["unsupported"]
    # E3/E4 landed `softmax` and `log_softmax` (their contracts live in
    # tests/test_native_softmax.py and tests/test_native_log_softmax.py).
    # "log_softmax" is a different capability from "log" — it shipped as
    # its own fused kernel and must not be conflated with this one.
    assert "log_softmax" in cpp.TENSOR_CORE_OPS
    assert "log_softmax" in cpp.AUTOGRAD_OPS
    # E5 landed the cross-entropy **Core** layer and E6 the differentiable
    # operation over it (tests/test_native_cross_entropy_core.py and
    # tests/test_native_cross_entropy.py); the loss module and the metric
    # (E7) are still absent.
    for core_op in ("cross_entropy_forward", "cross_entropy_backward"):
        assert core_op in cpp.TENSOR_CORE_OPS, core_op
        assert core_op not in cpp.AUTOGRAD_OPS, core_op
    # ...and E6 added the differentiable operation itself under the bare
    # name, exactly where an autograd operation belongs.
    assert "cross_entropy" in cpp.AUTOGRAD_OPS
    assert "cross_entropy" not in cpp.TENSOR_CORE_OPS
    for absent in ("NativeCrossEntropyLoss", "native_accuracy"):
        assert absent in cpp.UNSUPPORTED, absent
        assert absent not in cpp.TENSOR_CORE_OPS
        assert absent not in cpp.AUTOGRAD_OPS


# ======================================================================
# NativeTensorCore forward
# ======================================================================


@needs_native
def test_native_log_core_forward_contiguous_and_scalar():
    core = cpp.NativeTensorCore.from_array(VALUES)
    out = core.log()
    assert np.array_equal(out.to_numpy(), np.log(VALUES))
    assert out.shape == core.shape and out.contiguous
    assert out.dtype == "float64" and out.device == "cpu"
    assert out.offset == 0
    # The input is untouched and shares no storage with the output.
    assert np.array_equal(core.to_numpy(), VALUES)
    assert out.storage is not core.storage
    # Scalars follow the runtime's existing conventions unchanged:
    # from_array(1.0) is a one-element (1,) core, a full reduction gives
    # a true rank-0 () core, and log preserves either shape.
    scalar = cpp.NativeTensorCore.from_array(1.0)
    assert scalar.shape == (1,) and scalar.log().shape == (1,)
    assert scalar.log().to_numpy().item() == 0.0
    rank_zero = cpp.NativeTensorCore.from_array(VALUES).sum()
    assert rank_zero.shape == ()
    assert rank_zero.log().shape == ()
    assert math.isclose(rank_zero.log().to_numpy().item(),
                        math.log(float(VALUES.sum())), rel_tol=1e-15)


@needs_native
def test_native_log_core_forward_one_dimensional_and_multidimensional():
    flat = np.array([1.0, 2.0, 0.5, 8.0, 0.125])
    assert np.array_equal(
        cpp.NativeTensorCore.from_array(flat).log().to_numpy(), np.log(flat)
    )
    cube = np.arange(1.0, 13.0).reshape(2, 3, 2)
    out = cpp.NativeTensorCore.from_array(cube).log()
    assert out.shape == (2, 3, 2)
    assert np.array_equal(out.to_numpy(), np.log(cube))


@needs_native
def test_native_log_core_forward_random_values():
    rng = np.random.default_rng(20260721)
    values = rng.uniform(0.01, 100.0, size=(4, 7))
    out = cpp.NativeTensorCore.from_array(values).log()
    assert np.allclose(out.to_numpy(), np.log(values), rtol=1e-15, atol=0.0)


@needs_native
def test_native_log_core_forward_strided_views():
    core = cpp.NativeTensorCore.from_array(VALUES)
    transposed = core.T
    assert not transposed.contiguous
    assert np.array_equal(transposed.log().to_numpy(), np.log(VALUES.T))
    narrowed = core.narrow(1, 1, 2)  # nonzero offset
    assert narrowed.offset != 0
    assert np.array_equal(narrowed.log().to_numpy(), np.log(VALUES[:, 1:3]))
    combined = core.narrow(1, 1, 2).T  # transpose of a narrow
    result = combined.log()
    assert np.array_equal(result.to_numpy(), np.log(VALUES[:, 1:3].T))
    # Every output is fresh, owning, row-major contiguous, offset 0.
    assert result.contiguous and result.offset == 0
    assert result.storage is not core.storage
    # A reshaped (contiguous) view takes the fast path and agrees.
    reshaped = core.reshape((3, 2))
    assert np.array_equal(reshaped.log().to_numpy(),
                          np.log(VALUES.reshape(3, 2)))
    # The strided and contiguous paths agree bit-for-bit.
    assert np.array_equal(
        transposed.log().to_numpy(),
        cpp.NativeTensorCore.from_array(
            np.ascontiguousarray(VALUES.T)).log().to_numpy())
    assert np.array_equal(core.to_numpy(), VALUES)  # still unmutated


@needs_native
def test_native_log_core_rejects_closed_input():
    core = cpp.NativeTensorCore.from_array(VALUES)
    core.close()
    with pytest.raises(RuntimeError, match="closed"):
        core.log()


# ======================================================================
# IEEE domain behavior (values, never errors)
# ======================================================================


@needs_native
def test_native_log_domain_values_are_not_errors():
    values = [1.0, 0.0, -0.0, -1.0, -0.5, float("inf"), float("nan")]
    core = cpp.NativeTensorCore.from_array(values)
    out = core.log().to_numpy()
    assert out[0] == 0.0                              # log(1) == 0 exactly
    assert out[1] == -math.inf                        # +0 -> -inf
    assert out[2] == -math.inf                        # -0 -> -inf
    assert math.isnan(out[3])                         # negative -> NaN
    assert math.isnan(out[4])
    assert out[5] == math.inf                         # +inf -> +inf
    assert math.isnan(out[6])                         # NaN propagates
    # No clamping, no epsilon: exactly the values NumPy produces.
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.log(np.asarray(values, dtype=np.float64))
    assert np.array_equal(out, expected, equal_nan=True)
    # -inf is a negative value, so it is NaN (not -inf).
    neg_inf = cpp.NativeTensorCore.from_array([float("-inf")]).log().to_numpy()
    assert math.isnan(neg_inf[0])
    # A domain result leaves the native error slot clear — it is not a
    # failure, so no exception and no stale status.
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK


@needs_native
def test_native_log_values_between_zero_and_one_and_extremes():
    small = np.array([0.5, 0.25, 0.1, 1e-8, 1e-300])
    out = cpp.NativeTensorCore.from_array(small).log().to_numpy()
    assert np.all(out < 0.0) and np.all(np.isfinite(out))
    assert np.allclose(out, np.log(small), rtol=1e-14, atol=0.0)
    large = np.array([1e8, 1e150, 1e300])
    big = cpp.NativeTensorCore.from_array(large).log().to_numpy()
    assert np.allclose(big, np.log(large), rtol=1e-14, atol=0.0)
    assert np.all(np.isfinite(big))


# ======================================================================
# NativeTensor forward and graph construction
# ======================================================================


@needs_native
def test_native_log_wrapper_forward_and_graph_construction():
    plain = NativeTensor.from_array(VALUES)
    result = plain.log()
    assert np.array_equal(result.to_numpy(), np.log(VALUES))
    assert result._op == "" and result._parents == ()  # graph-free
    assert result.is_leaf and not result.requires_grad
    assert result._graph_resources == ()
    result.close()
    assert not plain.closed  # output lifetime is independent

    tracked = NativeTensor.from_array(VALUES, requires_grad=True)
    node = tracked.log()
    assert node._op == "log" and node._parents == (tracked,)
    assert node.requires_grad and not node.is_leaf
    # A plain NativeTensor parent carries no version, so nothing is
    # recorded even though the backward *does* reread it.
    assert node._expected_versions == ()
    # log owns no private graph resource (contrast maxpool2d's winners).
    assert node._graph_resources == ()

    # Parameter type never propagates through operations.
    parameter = NativeParameter(VALUES)
    assert type(parameter.log()) is NativeTensor

    closed = NativeTensor.from_array(VALUES)
    closed.close()
    with pytest.raises(RuntimeError, match="closed"):
        closed.log()


@needs_native
def test_native_log_wrapper_forward_through_views():
    x = NativeTensor.from_array(VALUES)
    assert np.array_equal(x.T.log().to_numpy(), np.log(VALUES.T))
    assert np.array_equal(x.narrow(1, 0, 2).log().to_numpy(),
                          np.log(VALUES[:, 0:2]))
    assert np.array_equal(x.to_numpy(), VALUES)


# ======================================================================
# Analytical backward
# ======================================================================


@needs_native
def test_native_log_gradient_exact():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    x.log().sum().backward()
    assert np.array_equal(x.grad.to_numpy(), 1.0 / VALUES)
    scalar = NativeTensor.from_array(4.0, requires_grad=True)
    scalar.log().backward()  # implicit scalar seeding
    assert scalar.grad.to_numpy().item() == 0.25


@needs_native
def test_native_log_explicit_upstream_gradient():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.log()
    upstream_values = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    out.backward(NativeTensor.from_array(upstream_values))
    # dx = upstream * reciprocal(input), exactly.
    assert np.array_equal(x.grad.to_numpy(), upstream_values / VALUES)


@needs_native
def test_native_log_gradient_through_views():
    y = NativeTensor.from_array(VALUES, requires_grad=True)
    y.T.log().sum().backward()
    assert np.array_equal(y.grad.to_numpy(), 1.0 / VALUES)
    # A narrowed (offset) parent differentiates at the logical shape.
    z = NativeTensor.from_array(VALUES, requires_grad=True)
    z.narrow(1, 1, 2).log().sum().backward()
    expected = np.zeros_like(VALUES)
    expected[:, 1:3] = 1.0 / VALUES[:, 1:3]
    assert np.array_equal(z.grad.to_numpy(), expected)


@needs_native
def test_native_log_finite_differences():
    # Safely inside the domain: positive, bounded away from zero.
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    x.log().sum().backward()
    numeric = _numeric_grad(lambda a: float(np.sum(np.log(a))), VALUES)
    assert np.allclose(x.grad.to_numpy(), numeric, atol=1e-6)
    # And through a strided input.
    y = NativeTensor.from_array(VALUES, requires_grad=True)
    y.T.log().sum().backward()
    assert np.allclose(y.grad.to_numpy(), numeric, atol=1e-6)


@needs_native
def test_native_log_chained_operations():
    # log(exp(x)) == x, so d/dx == 1 exactly through the composition.
    base = np.array([[0.5, -1.0], [2.0, 0.25]])
    x = NativeTensor.from_array(base, requires_grad=True)
    out = x.exp().log()
    assert np.allclose(out.to_numpy(), base, atol=1e-15)
    out.sum().backward()
    assert np.allclose(x.grad.to_numpy(), np.ones_like(base), atol=1e-14)
    # mean(x * log(x)): d/dx = (log(x) + 1) / N.
    w = NativeTensor.from_array(VALUES, requires_grad=True)
    loss = w.multiply(w.log()).mean()
    assert np.allclose(loss.to_numpy(), np.mean(VALUES * np.log(VALUES)),
                       atol=1e-15)
    loss.backward()
    assert np.allclose(w.grad.to_numpy(),
                       (np.log(VALUES) + 1.0) / VALUES.size, atol=1e-15)


@needs_native
def test_native_log_shared_subgraph_and_accumulation():
    # y = log(x); loss = sum(y + y) -> dx = 2 / x through one node.
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    y = x.log()
    y.add(y).sum().backward()
    assert np.allclose(x.grad.to_numpy(), 2.0 / VALUES, atol=1e-15)
    # Two independent log nodes over the same leaf also accumulate.
    w = NativeTensor.from_array(VALUES, requires_grad=True)
    w.log().sum().backward()
    w.log().sum().backward()
    assert np.allclose(w.grad.to_numpy(), 2.0 / VALUES, atol=1e-15)


@needs_native
def test_native_log_repeated_accumulation_until_zero_grad():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.log().sum()
    out.backward(retain_graph=True)
    out.backward(retain_graph=True)
    assert np.allclose(x.grad.to_numpy(), 2.0 / VALUES, atol=1e-15)
    x.zero_grad()
    assert x.grad is None
    out.backward()  # the final default pass frees the history
    assert np.array_equal(x.grad.to_numpy(), 1.0 / VALUES)


@needs_native
def test_native_log_backward_at_exceptional_inputs_follows_the_formula():
    """Not a well-conditioned finite-difference case — just the
    documented ``upstream * reciprocal(x)`` rule at IEEE edges."""
    x = NativeTensor.from_array([1.0, 0.0, -2.0], requires_grad=True)
    x.log().sum().backward()
    grad = x.grad.to_numpy()
    assert grad[0] == 1.0
    assert grad[1] == math.inf     # 1/0 -> +inf (reciprocal's IEEE rule)
    assert grad[2] == -0.5         # 1/(-2), even though log(-2) is NaN


# ======================================================================
# Versioning: live-input backward => direct parameters are guarded
# ======================================================================


@needs_native
def test_native_log_records_a_version_for_a_direct_parameter():
    p = NativeParameter(VALUES)
    node = p.log()
    assert len(node._expected_versions) == 1
    op_name, tracked, expected = node._expected_versions[0]
    assert op_name == "log" and tracked is p and expected == p.version
    # A plain tensor parent records nothing.
    plain = NativeTensor.from_array(VALUES, requires_grad=True)
    assert plain.log()._expected_versions == ()


@needs_native
def test_native_log_direct_parameter_mutation_raises_stale():
    """The E2 invariant: backward rereads the input, so mutating a direct
    parameter after forward must fail — before any gradient changes."""
    p = NativeParameter(VALUES)
    loss = p.log().sum()
    p.copy_value_(NativeTensor.from_array(np.full(VALUES.shape, 3.0)))
    with pytest.raises(RuntimeError, match="stale"):
        loss.backward()
    assert p.grad is None                      # nothing committed
    assert loss._parents != () and not loss._graph_freed  # not freed
    # Deterministic on retry while the version stays stale.
    with pytest.raises(RuntimeError, match="stale"):
        loss.backward()
    assert p.grad is None
    # A fresh forward after mutation works normally, on the new value.
    p.log().sum().backward()
    assert np.allclose(p.grad.to_numpy(), 1.0 / 3.0, atol=1e-15)


@needs_native
def test_native_log_mixed_graph_rollback_is_global():
    """A stale log edge must not let *any* branch commit a gradient."""
    p = NativeParameter(VALUES)                       # the log branch
    q = NativeParameter(np.full(VALUES.shape, 2.0))   # an unrelated branch
    loss = p.log().sum().add(q.multiply(q).sum())
    p.copy_value_(NativeTensor.from_array(np.full(VALUES.shape, 5.0)))
    with pytest.raises(RuntimeError, match="stale"):
        loss.backward()
    assert p.grad is None
    assert q.grad is None      # the healthy branch committed nothing either
    # Still deterministic, still nothing committed.
    with pytest.raises(RuntimeError, match="stale"):
        loss.backward()
    assert p.grad is None and q.grad is None


@needs_native
def test_native_log_stale_failure_preserves_preexisting_gradients():
    p = NativeParameter(VALUES)
    q = NativeParameter(np.full(VALUES.shape, 2.0))
    # One valid pass establishes real gradients on both parameters.
    p.log().sum().add(q.multiply(q).sum()).backward()
    p_before = p.grad.to_numpy().copy()
    q_before = q.grad.to_numpy().copy()
    assert np.allclose(p_before, 1.0 / VALUES, atol=1e-15)
    # A new graph over the same parameters then goes stale.
    stale = p.log().sum().add(q.multiply(q).sum())
    p.copy_value_(NativeTensor.from_array(np.full(VALUES.shape, 7.0)))
    with pytest.raises(RuntimeError, match="stale"):
        stale.backward()
    # The earlier gradients are numerically unchanged — no partial
    # accumulation leaked into them.
    assert np.array_equal(p.grad.to_numpy(), p_before)
    assert np.array_equal(q.grad.to_numpy(), q_before)


@needs_native
def test_native_log_retained_graph_becomes_stale_after_mutation():
    p = NativeParameter(VALUES)
    loss = p.log().sum()
    loss.backward(retain_graph=True)
    recorded = p.grad.to_numpy().copy()
    assert np.allclose(recorded, 1.0 / VALUES, atol=1e-15)
    p.copy_value_(NativeTensor.from_array(np.full(VALUES.shape, 9.0)))
    with pytest.raises(RuntimeError, match="stale"):
        loss.backward()
    # retain_graph does not rescue a stale graph, and the gradient the
    # successful pass produced is untouched.
    assert np.array_equal(p.grad.to_numpy(), recorded)


@needs_native
def test_native_log_indirect_saved_intermediate_stays_valid():
    """log rereads its *direct* parent. When that parent is a saved
    intermediate with independent storage (here exp's output), mutating
    the ancestor parameter does not invalidate the graph — exp's own
    backward reads its saved output, so no edge reads the mutated value.
    This is the locked direct-parent rule, not a broadened alias model.
    """
    original = np.array([[0.5, -1.0], [2.0, 0.25]])
    p = NativeParameter(original)
    y = p.exp()
    assert y._expected_versions == ()          # exp: saved-output backward
    loss = y.log().sum()
    # The log node's parent is y (a plain tensor), so nothing is recorded.
    assert loss._parents[0]._expected_versions == ()
    p.copy_value_(NativeTensor.from_array(np.full((2, 2), 3.0)))
    loss.backward()                            # valid: no stale edge
    # d/dp log(exp(p)) == 1, from the forward that actually ran.
    assert np.allclose(p.grad.to_numpy(), np.ones_like(original), atol=1e-14)


@needs_native
def test_native_log_does_not_change_other_operations_versioning():
    """E2 must not perturb the established classifications."""
    p = NativeParameter(VALUES)
    # exp stays saved-output / no version...
    assert p.exp()._expected_versions == ()
    exp_loss = p.exp().sum()
    p.copy_value_(NativeTensor.from_array(np.full(VALUES.shape, 2.0)))
    exp_loss.backward()                        # still valid after mutation
    assert p.grad is not None
    # ...while relu and multiply stay guarded.
    q = NativeParameter(VALUES)
    for build in (lambda t: t.relu().sum(), lambda t: t.multiply(t).sum()):
        q.zero_grad()
        loss = build(q)
        q.copy_value_(NativeTensor.from_array(
            np.full(VALUES.shape, float(q.version + 2))))
        with pytest.raises(RuntimeError, match="stale"):
            loss.backward()
        assert q.grad is None


# ======================================================================
# Graph lifetime
# ======================================================================


@needs_native
def test_native_log_one_shot_release_and_freed_history():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.log().sum()
    out.backward()
    with pytest.raises(RuntimeError, match="freed"):
        out.backward()
    assert np.array_equal(x.grad.to_numpy(), 1.0 / VALUES)


@needs_native
def test_native_log_no_grad_forward_creates_no_graph():
    x = NativeTensor.from_array(VALUES)  # requires_grad=False
    out = x.log()
    assert out.is_leaf and out._backward is None and out._parents == ()
    assert not out.requires_grad and out._expected_versions == ()
    with pytest.raises(RuntimeError, match="does not require grad|requires_grad"):
        out.sum().backward()


@needs_native
def test_native_log_detached_input_creates_no_graph():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    detached = x.detach()
    out = detached.log()
    assert not out.requires_grad and out.is_leaf
    out.close()
    x.log().sum().backward()
    assert np.array_equal(x.grad.to_numpy(), 1.0 / VALUES)


@needs_native
def test_native_log_closed_parent_fails_atomically():
    """Because the backward rereads the parent, a parent closed after
    forward makes backward fail clearly — and commit nothing."""
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    intermediate = x.add(x)
    out = intermediate.log().sum()
    intermediate.close()
    with pytest.raises(RuntimeError, match="closed"):
        out.backward()
    assert x.grad is None            # nothing committed anywhere
    with pytest.raises(RuntimeError, match="closed"):
        out.backward()               # deterministic on retry
    assert x.grad is None
    # With the parent left open the same shape of graph works:
    # d/dy log(y + y) = 2 * 1/(2y) = 1/y.
    y = NativeTensor.from_array(VALUES, requires_grad=True)
    y.add(y).log().sum().backward()
    assert np.allclose(y.grad.to_numpy(), 1.0 / VALUES, atol=1e-15)


@needs_native
def test_native_log_saves_no_derivative_state():
    """log stores no saved-output derivative state: the node owns no
    private graph resource, and the local derivative is rebuilt from the
    **live input** at backward time — the output's value is never read.

    Closing a *consumed* output still fails, but for the ordinary reason
    every operation shares (a downstream node must accumulate a gradient
    into it), not because log needed its value."""
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    y = x.log()
    assert y._graph_resources == ()
    # The log output as the backward root: nothing reads its value.
    y.backward(NativeTensor.from_array(np.ones(VALUES.shape)))
    assert np.array_equal(x.grad.to_numpy(), 1.0 / VALUES)
    # A closed, consumed output fails only through the accumulation rule.
    z = NativeTensor.from_array(VALUES, requires_grad=True)
    w = z.log()
    loss = w.sum()
    w.close()
    with pytest.raises(RuntimeError, match="closed"):
        loss.backward()
    assert z.grad is None


@needs_native
def test_native_log_abandoned_graph_closes_cleanly():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.log()
    out.close()
    out.close()  # idempotent: no double free
    assert out.closed and not x.closed
    assert x.grad is None
    x.log().sum().backward()
    assert np.array_equal(x.grad.to_numpy(), 1.0 / VALUES)


# ======================================================================
# Failure behavior
# ======================================================================


@needs_native
def test_native_log_rejects_invalid_backward_arguments():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.log()  # non-scalar output
    with pytest.raises(ValueError, match="non-scalar"):
        out.backward()
    with pytest.raises(ValueError, match="shape"):
        out.backward(NativeTensor.from_array(np.ones((2, 2))))
    with pytest.raises(TypeError):
        out.backward(np.ones(VALUES.shape))
    closed_upstream = NativeTensor.from_array(np.ones(VALUES.shape))
    closed_upstream.close()
    with pytest.raises(RuntimeError, match="closed"):
        out.backward(closed_upstream)
    # Every rejection happened before any gradient was committed.
    assert x.grad is None
    out.backward(NativeTensor.from_array(np.ones(VALUES.shape)))
    assert np.array_equal(x.grad.to_numpy(), 1.0 / VALUES)


@needs_native
def test_native_log_abi_rejects_invalid_calls():
    """The E2 exports reuse E1's validators, so a malformed direct ctypes
    call raises ValueError instead of reading out of bounds — and leaves
    the destination byte-for-byte unchanged."""
    library = cpp._require_library()
    source = cpp.NativeTensorCore.from_array(np.arange(1.0, 5.0))
    destination = cpp.NativeTensorCore.from_array(np.full(4, 7777.5))
    src_handle = source.storage._require_open()
    dst_handle = destination.storage._require_open()
    shape = np.asarray([4], dtype=np.int64)
    good_strides = np.asarray([1], dtype=np.int64)

    with pytest.raises(ValueError, match="span exceeds its storage"):
        library.tf_core_log_contiguous(src_handle, dst_handle, 4, 1)
    with pytest.raises(ValueError, match="span exceeds its storage"):
        library.tf_core_log_contiguous(src_handle, dst_handle, 8, 0)
    with pytest.raises(ValueError, match="span exceeds its storage"):
        library.tf_core_log(src_handle, dst_handle, shape,
                            np.asarray([2], dtype=np.int64), 0, 1)
    with pytest.raises(ValueError, match="non-positive dimension"):
        library.tf_core_log(src_handle, dst_handle,
                            np.asarray([0], dtype=np.int64), good_strides, 0, 1)
    with pytest.raises(ValueError, match="negative offset"):
        library.tf_core_log(src_handle, dst_handle, shape, good_strides, -1, 1)
    # The destination was never written by any rejected call.
    assert np.array_equal(destination.to_numpy(), np.full(4, 7777.5))
    # A stale error does not survive the next valid call.
    assert np.array_equal(source.log().to_numpy(), np.log(np.arange(1.0, 5.0)))
    assert library.tf_last_error_code() == cpp.TF_OK


@needs_fault_injection
def test_native_log_allocation_failure_is_atomic():
    """Sweep the injector across the allocations a log forward+backward
    performs. Every stage that can be hit must raise MemoryError, mutate
    no input, commit no gradient, and leave the stack usable afterwards.
    (The injector counts allocations rather than naming stages, so the
    sweep is honest about which stages are reachable instead of assuming
    a fixed mapping.)"""
    values = np.array([[1.0, 2.0], [4.0, 0.5]])

    # -- forward output allocation --
    x = NativeTensor.from_array(values, requires_grad=True)
    with pytest.raises(MemoryError):
        cpp._arm_alloc_failure(1)
        x.log()
    cpp._arm_alloc_failure(0)
    assert np.array_equal(x.to_numpy(), values)  # input untouched
    assert x.grad is None
    assert np.array_equal(x.log().to_numpy(), np.log(values))  # recovers

    # -- backward allocations (seed, reciprocal temporary, multiply) --
    hit = 0
    for nth in range(1, 7):
        y = NativeTensor.from_array(values, requires_grad=True)
        loss = y.log().sum()
        try:
            cpp._arm_alloc_failure(nth)
            loss.backward()
        except MemoryError:
            hit += 1
            cpp._arm_alloc_failure(0)
            # Nothing committed, input intact, graph not freed.
            assert y.grad is None, nth
            assert np.array_equal(y.to_numpy(), values), nth
            assert not loss._graph_freed, nth
            # And the same graph completes once the injector is disarmed
            # (no leaked or double-closed native state).
            loss.backward()
            assert np.allclose(y.grad.to_numpy(), 1.0 / values, atol=1e-15)
        else:
            cpp._arm_alloc_failure(0)
            assert np.allclose(y.grad.to_numpy(), 1.0 / values, atol=1e-15)
    assert hit >= 2, "expected several reachable backward allocation stages"
    # The backend is healthy afterwards: no stale error, ops still work.
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    assert np.allclose(
        cpp.NativeTensorCore.from_array(values).log().to_numpy(),
        np.log(values), atol=1e-15)


def test_native_log_requires_the_built_backend():
    """Without the compiled library the operation raises ImportError with
    build instructions — never a silent NumPy fallback."""
    if cpp.is_available():
        pytest.skip("backend is built; the unavailable path cannot be forced")
    with pytest.raises(ImportError, match="cpp/build.py"):
        cpp.NativeTensorCore.from_array(VALUES).log()


# ======================================================================
# Guardrails
# ======================================================================


@needs_native
def test_native_log_uses_no_numpy_compute(monkeypatch):
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    core = cpp.NativeTensorCore.from_array(VALUES)

    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in ("log", "log2", "log10", "log1p", "exp", "sqrt",
                 "reciprocal", "divide", "true_divide", "add", "subtract",
                 "multiply", "matmul", "sum", "mean", "negative", "power",
                 "copyto"):
        monkeypatch.setattr(np, name, _tripwire)
    core.log()                       # NativeTensorCore forward
    x.log().sum().backward()         # NativeTensor forward + backward
    monkeypatch.undo()
    assert np.array_equal(x.grad.to_numpy(), 1.0 / VALUES)


@needs_native
def test_native_log_scope_boundaries_hold():
    """E2 is logarithm only: no other Phase-E surface appeared, no public
    division, and the stable framework is untouched."""
    x = NativeTensor.from_array(VALUES)
    core = cpp.NativeTensorCore.from_array(VALUES)
    for absent in ("tanh", "sigmoid"):
        assert not hasattr(x, absent), absent
        assert not hasattr(core, absent), absent
    # `cross_entropy` shipped at E6 as a NativeTensor operation, and is
    # still absent from the layer-qualified Core surface.
    assert not hasattr(core, "cross_entropy")
    assert not hasattr(x, "divide") and not hasattr(x, "__truediv__")
    assert not hasattr(core, "divide")
    import tensorforge.experimental as experimental
    assert not hasattr(experimental, "NativeCrossEntropyLoss")
    assert not hasattr(experimental, "native_accuracy")
    # The stable Tensor keeps its own log, entirely separately.
    stable = tensorforge.Tensor(2.0, requires_grad=True)
    stable.log().backward()
    assert np.allclose(stable.grad, 0.5)
    assert type(stable.log()) is tensorforge.Tensor
    # No implicit dispatch in either direction.
    with pytest.raises((TypeError, AttributeError)):
        NativeTensor.from_array(VALUES).multiply(tensorforge.Tensor(VALUES))


@needs_native
def test_native_log_checkpoint_schema_is_untouched():
    """E2 adds no persistent state: the native checkpoint format version
    is still 1 (docs/native_classification_design.md §12)."""
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT_VERSION == 1
