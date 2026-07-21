"""Tests for the native exponential — Phase E, milestone E1.

The first Phase-E operation, and the first of the two stable-math
primitives the classification stack is built from
(docs/native_classification_design.md §4.1). One shape-preserving unary
operation through the complete native stack: the C++ kernel (odometer +
contiguous fast path, sharing the relu signature) → the guarded,
**self-validating** C ABI pair (``tf_core_exp`` /
``tf_core_exp_contiguous``) → ctypes → ``NativeTensorCore.exp()`` →
differentiable ``NativeTensor.exp()``.

The autograd design is a **saved forward output**: d(exp(x))/dx =
exp(x), so backward is exactly ``upstream * saved output``. It never
rereads the parent, so the graph records **no** expected parameter
version (v3.7): mutating a direct parameter after forward leaves the
edge valid and the gradient correct for the forward that was recorded —
the deliberate contrast with the (still guarded) value-reading edges
like multiply/matmul/relu, and with the live-input ``log`` backward E2
will add.

Exceptional values follow IEEE float64 with no clamping and no inserted
bound: exp(0) == 1, large positive arguments overflow to +inf, large
negative ones underflow toward +0, -inf gives +0, and NaN propagates.

NumPy appears below only as an external oracle and for inspection; a
tripwire test proves the forward/backward paths never compute with it.

Selector: python -m pytest -q -k "native_exp"
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


# Moderate magnitudes: exp stays finite and well conditioned, so exact
# and finite-difference comparisons are both meaningful.
VALUES = np.array([[0.0, 1.0, -1.0], [2.5, -0.75, 0.25]])


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
def test_native_exp_kernel_symbols_are_bound():
    library = cpp._require_library()
    for name in ("tf_core_exp", "tf_core_exp_contiguous"):
        assert getattr(library, name) is not None
        # Both participate in the error contract (they self-validate).
        assert name in cpp._CHECKED_KERNELS
        # An ABI symbol is not a capability name.
        assert name not in cpp.RAW_KERNELS
        assert name not in cpp.TENSOR_CORE_OPS
    # The frozen historical registry and the raw-buffer set are unchanged:
    # E1 added no raw NumPy-buffer kernel (the sum/mean/sqrt precedent).
    assert cpp.TENSOR_CORE_KERNELS == ("relu", "add", "subtract",
                                       "multiply", "matmul")
    assert "exp" not in cpp.RAW_KERNELS
    assert not hasattr(cpp, "exp")


def test_native_exp_registry_placement():
    """exp is a Core op and an autograd op — nothing else. Runs without
    the compiled backend: these are pure inventory facts."""
    assert "exp" in cpp.TENSOR_CORE_OPS
    assert "exp" in cpp.AUTOGRAD_OPS
    assert "exp" not in cpp.UNSUPPORTED
    assert "exp" not in cpp.NATIVE_MODULES
    assert "exp" not in cpp.NATIVE_LOSSES
    assert not hasattr(cpp, "NATIVE_METRICS")
    info = cpp.backend_info()
    assert "exp" in info["tensor_core_ops"] and "exp" in info["autograd_ops"]
    assert "exp" not in info["unsupported"]
    # E2/E3/E4 landed `log`, `softmax`, and `log_softmax` beside it
    # (their contracts live in tests/test_native_log.py,
    # tests/test_native_softmax.py, and tests/test_native_log_softmax.py);
    # everything after E4 is still absent.
    for absent in ("cross_entropy",
                   "NativeCrossEntropyLoss", "native_accuracy"):
        assert absent in cpp.UNSUPPORTED, absent
        assert absent not in cpp.TENSOR_CORE_OPS
        assert absent not in cpp.AUTOGRAD_OPS


# ======================================================================
# NativeTensorCore forward
# ======================================================================


@needs_native
def test_native_exp_core_forward_contiguous_and_scalar():
    core = cpp.NativeTensorCore.from_array(VALUES)
    out = core.exp()
    assert np.array_equal(out.to_numpy(), np.exp(VALUES))
    assert out.shape == core.shape and out.contiguous
    assert out.dtype == "float64" and out.device == "cpu"
    assert out.offset == 0
    # The input is untouched and shares no storage with the output.
    assert np.array_equal(core.to_numpy(), VALUES)
    assert out.storage is not core.storage
    # Scalars follow the runtime's existing convention unchanged:
    # from_array(0.0) is a one-element (1,) core, while a rank-0 () core
    # is what a full reduction produces. exp preserves either shape.
    scalar = cpp.NativeTensorCore.from_array(0.0)
    assert scalar.shape == (1,) and scalar.exp().shape == (1,)
    assert scalar.exp().to_numpy().item() == 1.0
    rank_zero = cpp.NativeTensorCore.from_array(VALUES).sum()
    assert rank_zero.shape == ()
    assert rank_zero.exp().shape == ()
    assert math.isclose(rank_zero.exp().to_numpy().item(),
                        math.exp(float(VALUES.sum())), rel_tol=1e-15)


@needs_native
def test_native_exp_core_forward_one_dimensional_and_multidimensional():
    flat = np.array([0.0, 1.0, -1.0, 3.0, -2.0])
    assert np.array_equal(
        cpp.NativeTensorCore.from_array(flat).exp().to_numpy(), np.exp(flat)
    )
    cube = np.arange(-6.0, 6.0).reshape(2, 3, 2) / 4.0
    out = cpp.NativeTensorCore.from_array(cube).exp()
    assert out.shape == (2, 3, 2)
    assert np.array_equal(out.to_numpy(), np.exp(cube))


@needs_native
def test_native_exp_core_forward_random_values():
    rng = np.random.default_rng(20260720)
    values = rng.uniform(-5.0, 5.0, size=(4, 7))
    out = cpp.NativeTensorCore.from_array(values).exp()
    assert np.allclose(out.to_numpy(), np.exp(values), rtol=1e-15, atol=0.0)


@needs_native
def test_native_exp_core_forward_strided_views():
    core = cpp.NativeTensorCore.from_array(VALUES)
    transposed = core.T
    assert not transposed.contiguous
    assert np.array_equal(transposed.exp().to_numpy(), np.exp(VALUES.T))
    narrowed = core.narrow(1, 1, 2)  # nonzero offset
    assert narrowed.offset != 0
    assert np.array_equal(narrowed.exp().to_numpy(), np.exp(VALUES[:, 1:3]))
    combined = core.narrow(1, 1, 2).T  # transpose of a narrow
    result = combined.exp()
    assert np.array_equal(result.to_numpy(), np.exp(VALUES[:, 1:3].T))
    # Every output is fresh, owning, row-major contiguous, offset 0.
    assert result.contiguous and result.offset == 0
    assert result.storage is not core.storage
    # A reshaped (contiguous) view takes the fast path and agrees.
    reshaped = core.reshape((3, 2))
    assert np.array_equal(reshaped.exp().to_numpy(),
                          np.exp(VALUES.reshape(3, 2)))
    # The strided and contiguous paths agree bit-for-bit.
    assert np.array_equal(transposed.exp().to_numpy(),
                          cpp.NativeTensorCore.from_array(
                              np.ascontiguousarray(VALUES.T)).exp().to_numpy())
    assert np.array_equal(core.to_numpy(), VALUES)  # still unmutated


@needs_native
def test_native_exp_core_rejects_closed_input():
    core = cpp.NativeTensorCore.from_array(VALUES)
    core.close()
    with pytest.raises(RuntimeError, match="closed"):
        core.exp()


# ======================================================================
# Exceptional values (IEEE float64, locked)
# ======================================================================


@needs_native
def test_native_exp_exceptional_values():
    core = cpp.NativeTensorCore.from_array(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1000.0, -1000.0]
    )
    out = core.exp().to_numpy()
    assert out[0] == 1.0                              # exp(0) == 1 exactly
    assert out[1] == 1.0                              # exp(-0.0) == 1
    assert out[2] == math.inf                         # +inf -> +inf
    assert out[3] == 0.0 and not np.signbit(out[3])   # -inf -> +0
    assert math.isnan(out[4])                         # NaN propagates
    assert out[5] == math.inf                         # overflow -> +inf
    assert out[6] == 0.0                              # underflow -> 0
    # No clamping: the same values NumPy produces (it warns, we do not).
    with np.errstate(over="ignore"):
        expected = np.exp(core.to_numpy())
    assert np.array_equal(out, expected, equal_nan=True)


@needs_native
def test_native_exp_large_finite_values_match_numpy():
    values = np.array([-700.0, -50.0, -5.0, 5.0, 50.0, 700.0])
    out = cpp.NativeTensorCore.from_array(values).exp().to_numpy()
    assert np.allclose(out, np.exp(values), rtol=1e-14, atol=0.0)
    assert np.all(np.isfinite(out))


# ======================================================================
# NativeTensor forward and graph construction
# ======================================================================


@needs_native
def test_native_exp_wrapper_forward_and_graph_construction():
    plain = NativeTensor.from_array(VALUES)
    result = plain.exp()
    assert np.array_equal(result.to_numpy(), np.exp(VALUES))
    assert result._op == "" and result._parents == ()  # graph-free
    assert result.is_leaf and not result.requires_grad
    assert result._graph_resources == ()
    result.close()
    assert not plain.closed  # output lifetime is independent

    tracked = NativeTensor.from_array(VALUES, requires_grad=True)
    node = tracked.exp()
    assert node._op == "exp" and node._parents == (tracked,)
    assert node.requires_grad and not node.is_leaf
    assert node._expected_versions == ()  # saved output: no version reads
    assert node._graph_resources == ()    # the node's own core is the state

    # Parameter type never propagates through operations.
    parameter = NativeParameter(VALUES)
    assert type(parameter.exp()) is NativeTensor

    closed = NativeTensor.from_array(VALUES)
    closed.close()
    with pytest.raises(RuntimeError, match="closed"):
        closed.exp()


@needs_native
def test_native_exp_wrapper_forward_through_views():
    x = NativeTensor.from_array(VALUES)
    assert np.array_equal(x.T.exp().to_numpy(), np.exp(VALUES.T))
    assert np.array_equal(x.narrow(1, 0, 2).exp().to_numpy(),
                          np.exp(VALUES[:, 0:2]))
    assert np.array_equal(x.to_numpy(), VALUES)


# ======================================================================
# Analytical backward
# ======================================================================


@needs_native
def test_native_exp_gradient_exact():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    x.exp().sum().backward()
    assert np.array_equal(x.grad.to_numpy(), np.exp(VALUES))
    scalar = NativeTensor.from_array(2.0, requires_grad=True)
    scalar.exp().backward()  # implicit scalar seeding
    assert scalar.grad.to_numpy().item() == math.exp(2.0)


@needs_native
def test_native_exp_explicit_upstream_gradient():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.exp()
    upstream_values = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    out.backward(NativeTensor.from_array(upstream_values))
    # dx = upstream * exp(input), exactly.
    assert np.array_equal(x.grad.to_numpy(), upstream_values * np.exp(VALUES))


@needs_native
def test_native_exp_gradient_through_views():
    y = NativeTensor.from_array(VALUES, requires_grad=True)
    y.T.exp().sum().backward()
    assert np.array_equal(y.grad.to_numpy(), np.exp(VALUES))


@needs_native
def test_native_exp_finite_differences():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    x.exp().sum().backward()
    numeric = _numeric_grad(lambda a: float(np.sum(np.exp(a))), VALUES)
    assert np.allclose(x.grad.to_numpy(), numeric, atol=1e-6)
    # And through a strided input.
    y = NativeTensor.from_array(VALUES, requires_grad=True)
    y.T.exp().sum().backward()
    assert np.allclose(y.grad.to_numpy(), numeric, atol=1e-6)


@needs_native
def test_native_exp_chained_operations():
    # mean(exp(x) * x): d/dx = (exp(x) * x + exp(x)) / N.
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    loss = x.exp().multiply(x).mean()
    assert np.allclose(loss.to_numpy(), np.mean(np.exp(VALUES) * VALUES),
                       atol=1e-15)
    loss.backward()
    expected = (np.exp(VALUES) * VALUES + np.exp(VALUES)) / VALUES.size
    assert np.allclose(x.grad.to_numpy(), expected, atol=1e-15)
    # exp of an exp: d/dx exp(exp(x)) = exp(exp(x)) * exp(x).
    z = NativeTensor.from_array(np.array([0.0, 0.5, -0.5]), requires_grad=True)
    z.exp().exp().sum().backward()
    base = np.array([0.0, 0.5, -0.5])
    assert np.allclose(z.grad.to_numpy(), np.exp(np.exp(base)) * np.exp(base),
                       atol=1e-14)


@needs_native
def test_native_exp_shared_subgraph_accumulates():
    # y = exp(x); loss = sum(y + y) -> dx = 2 * exp(x) through one node.
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    y = x.exp()
    y.add(y).sum().backward()
    assert np.allclose(x.grad.to_numpy(), 2.0 * np.exp(VALUES), atol=1e-15)
    # Two independent exp nodes over the same leaf also accumulate.
    w = NativeTensor.from_array(VALUES, requires_grad=True)
    w.exp().sum().backward()
    w.exp().sum().backward()
    assert np.allclose(w.grad.to_numpy(), 2.0 * np.exp(VALUES), atol=1e-15)


@needs_native
def test_native_exp_repeated_accumulation_until_zero_grad():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.exp().sum()
    out.backward(retain_graph=True)
    out.backward(retain_graph=True)
    assert np.allclose(x.grad.to_numpy(), 2.0 * np.exp(VALUES), atol=1e-15)
    x.zero_grad()
    assert x.grad is None
    out.backward()  # the final default pass frees the history
    assert np.array_equal(x.grad.to_numpy(), np.exp(VALUES))


# ======================================================================
# Graph lifetime
# ======================================================================


@needs_native
def test_native_exp_one_shot_release_and_retain_graph():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.exp().sum()
    out.backward()
    with pytest.raises(RuntimeError, match="freed"):
        out.backward()
    assert np.array_equal(x.grad.to_numpy(), np.exp(VALUES))


@needs_native
def test_native_exp_no_grad_forward_creates_no_graph():
    x = NativeTensor.from_array(VALUES)  # requires_grad=False
    out = x.exp()
    assert out.is_leaf and out._backward is None and out._parents == ()
    assert not out.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad|requires_grad"):
        out.sum().backward()


@needs_native
def test_native_exp_detached_input_creates_no_graph():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    detached = x.detach()
    out = detached.exp()
    assert not out.requires_grad and out.is_leaf
    out.close()
    # The original still differentiates normally.
    x.exp().sum().backward()
    assert np.array_equal(x.grad.to_numpy(), np.exp(VALUES))


@needs_native
def test_native_exp_closed_saved_output_fails_atomically():
    """The saved forward output *is* the local derivative, so closing it
    before a downstream backward must fail clearly — and commit nothing."""
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    y = x.exp()
    z = y.sum()
    y.close()  # the saved output backward must read
    with pytest.raises(RuntimeError, match="closed"):
        z.backward()
    # The failed pass rolled back: no partial gradients, graph intact.
    assert x.grad is None
    assert z._parents != () and not z._graph_freed
    with pytest.raises(RuntimeError, match="closed"):
        z.backward()  # deterministic on retry, still nothing committed
    assert x.grad is None


@needs_native
def test_native_exp_closed_parent_fails_on_accumulation_not_on_a_value_read():
    """Backward never rereads the parent's *value* — but it still has to
    accumulate a gradient **into** the parent, and the established
    contract is that a closed tensor accepts nothing. So a closed parent
    raises clearly and atomically, and the reason is accumulation, never
    a value read (the mutation tests above prove the value is not read).
    """
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    intermediate = x.add(x)          # a value-independent parent
    out = intermediate.exp().sum()
    intermediate.close()
    with pytest.raises(RuntimeError, match="closed"):
        out.backward()
    assert x.grad is None            # nothing committed
    assert intermediate.closed       # (a closed tensor reports no grad)
    # With the parent left open the same graph differentiates fine —
    # confirming the failure was the closed handle, not the exp edge.
    y = NativeTensor.from_array(VALUES, requires_grad=True)
    y.add(y).exp().sum().backward()
    assert np.allclose(y.grad.to_numpy(), 2.0 * np.exp(2.0 * VALUES),
                       atol=1e-15)


@needs_native
def test_native_exp_abandoned_graph_closes_cleanly():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.exp()
    out.close()
    out.close()  # idempotent: no double free
    assert out.closed and not x.closed
    assert x.grad is None
    # A fresh graph over the same leaf still works.
    x.exp().sum().backward()
    assert np.array_equal(x.grad.to_numpy(), np.exp(VALUES))


# ======================================================================
# Versioning: saved output => value-independent of later mutation
# ======================================================================


@needs_native
def test_native_exp_parameter_mutation_keeps_graph_valid():
    """The E1 invariant: mutate a direct parameter after forward and
    backward still succeeds, using exp(original values)."""
    original = np.array([[0.0, 1.0], [-1.0, 2.0]])
    p = NativeParameter(original)
    out = p.exp().sum()
    p.copy_value_(NativeTensor.from_array(np.full((2, 2), 5.0)))
    out.backward()  # reads the saved forward output — still valid
    assert np.array_equal(p.grad.to_numpy(), np.exp(original))
    # Explicitly NOT the gradient of the mutated value.
    assert not np.allclose(p.grad.to_numpy(), np.exp(np.full((2, 2), 5.0)))
    # A fresh forward uses the new value, deterministically.
    p.zero_grad()
    p.exp().sum().backward()
    assert np.allclose(p.grad.to_numpy(), np.exp(np.full((2, 2), 5.0)),
                       atol=1e-15)


@needs_native
def test_native_exp_records_no_version_snapshot():
    p = NativeParameter([[0.0, 1.0], [-1.0, 2.0]])
    node = p.exp()
    assert node._expected_versions == ()
    # Two mutations, then backward: still no stale-graph error.
    loss = node.sum()
    p.copy_value_(NativeTensor.from_array(np.ones((2, 2))))
    p.copy_value_(NativeTensor.from_array(np.zeros((2, 2))))
    loss.backward()
    assert p.grad is not None


@needs_native
def test_native_exp_does_not_weaken_value_reading_edges():
    """Existing guarded edges stay guarded: an exp edge on a mutated
    parameter is fine, but a multiply edge on the same parameter still
    goes stale — and the failed pass commits no gradient."""
    p = NativeParameter([[0.0, 1.0], [-1.0, 2.0]])
    mixed = p.exp().multiply(p).sum()
    p.copy_value_(NativeTensor.from_array(np.ones((2, 2))))
    with pytest.raises(RuntimeError, match="stale"):
        mixed.backward()
    assert p.grad is None
    # relu (a value-reading edge) is likewise still guarded.
    q = NativeParameter([[0.5, -0.5], [1.5, -1.5]])
    relu_loss = q.relu().exp().sum()
    q.copy_value_(NativeTensor.from_array(np.zeros((2, 2))))
    with pytest.raises(RuntimeError, match="stale"):
        relu_loss.backward()
    assert q.grad is None


# ======================================================================
# Failure behavior
# ======================================================================


@needs_native
def test_native_exp_rejects_invalid_backward_arguments():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.exp()  # non-scalar output
    with pytest.raises(ValueError, match="non-scalar"):
        out.backward()  # no implicit seed for a non-scalar
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
    assert np.array_equal(x.grad.to_numpy(), np.exp(VALUES))


@needs_native
def test_native_exp_abi_rejects_invalid_spans():
    """The E1 exports validate their own arguments, so a malformed direct
    ctypes call raises ValueError through the existing error contract
    instead of reading out of bounds."""
    library = cpp._require_library()
    source = cpp.NativeTensorCore.from_array(np.arange(4.0))
    destination = cpp.NativeTensorCore.zeros((4,))
    src_handle = source.storage._require_open()
    dst_handle = destination.storage._require_open()
    with pytest.raises(ValueError, match="span exceeds its storage"):
        library.tf_core_exp_contiguous(src_handle, dst_handle, 4, 1)
    with pytest.raises(ValueError, match="span exceeds its storage"):
        library.tf_core_exp_contiguous(src_handle, dst_handle, 8, 0)
    shape = np.asarray([4], dtype=np.int64)
    strides = np.asarray([2], dtype=np.int64)
    with pytest.raises(ValueError, match="span exceeds its storage"):
        library.tf_core_exp(src_handle, dst_handle, shape, strides, 0, 1)
    bad_shape = np.asarray([0], dtype=np.int64)
    good_strides = np.asarray([1], dtype=np.int64)
    with pytest.raises(ValueError, match="non-positive dimension"):
        library.tf_core_exp(src_handle, dst_handle, bad_shape, good_strides,
                            0, 1)
    # The failures changed nothing: a valid call still computes.
    assert np.array_equal(source.exp().to_numpy(), np.exp(np.arange(4.0)))


def test_native_exp_requires_the_built_backend():
    """Without the compiled library the operation raises ImportError with
    build instructions — never a silent NumPy fallback."""
    if cpp.is_available():
        pytest.skip("backend is built; the unavailable path cannot be forced")
    with pytest.raises(ImportError, match="cpp/build.py"):
        cpp.NativeTensorCore.from_array(VALUES).exp()


# ======================================================================
# Guardrails
# ======================================================================


@needs_native
def test_native_exp_uses_no_numpy_compute(monkeypatch):
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    core = cpp.NativeTensorCore.from_array(VALUES)

    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in ("exp", "log", "sqrt", "reciprocal", "divide", "add",
                 "subtract", "multiply", "matmul", "sum", "mean",
                 "negative", "power", "copyto"):
        monkeypatch.setattr(np, name, _tripwire)
    core.exp()                       # NativeTensorCore forward
    x.exp().sum().backward()         # NativeTensor forward + backward
    monkeypatch.undo()
    assert np.array_equal(x.grad.to_numpy(), np.exp(VALUES))


@needs_native
def test_native_exp_scope_boundaries_hold():
    """E1 added the exponential and nothing else. (`log`, `softmax`, and
    `log_softmax` arrived separately in E2/E3/E4, so they are no longer
    listed as absent here — the still-unshipped Phase-E surface is.) The
    stable framework is untouched."""
    x = NativeTensor.from_array(VALUES)
    core = cpp.NativeTensorCore.from_array(VALUES)
    for absent in ("cross_entropy",):
        assert not hasattr(x, absent), absent
        assert not hasattr(core, absent), absent
    assert not hasattr(x, "divide") and not hasattr(x, "__truediv__")
    import tensorforge.experimental as experimental
    assert not hasattr(experimental, "NativeCrossEntropyLoss")
    assert not hasattr(experimental, "native_accuracy")
    # The stable Tensor keeps its own exp, entirely separately.
    stable = tensorforge.Tensor(1.0, requires_grad=True)
    stable.exp().backward()
    assert np.allclose(stable.grad, math.e)
    assert type(stable.exp()) is tensorforge.Tensor
    # No implicit dispatch in either direction.
    with pytest.raises((TypeError, AttributeError)):
        NativeTensor.from_array(VALUES).multiply(tensorforge.Tensor(VALUES))


@needs_native
def test_native_exp_checkpoint_schema_is_untouched():
    """E1 adds no persistent state: the native checkpoint format version
    is still 1 (docs/native_classification_design.md §12)."""
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT_VERSION == 1
