"""Tests for the native optimizer math primitives — sqrt and
reciprocal (Advanced C++ v3.11).

Two shape-preserving unary operations through the complete native
stack: C++ kernels (odometer + contiguous fast path, sharing the relu
signature) → ctypes bindings → NativeTensorCore.sqrt()/reciprocal() →
NativeTensor.sqrt()/reciprocal() → native autograd. They exist because
NativeAdam (v3.12) needs a square-root denominator and reciprocal
scaling; NativeAdam itself is not implemented.

The autograd design is **saved forward results**: d(sqrt(x))/dx =
1/(2*sqrt(x)) is computed as 0.5 * reciprocal(saved output) and
d(1/x)/dx = -1/x² as -(saved output)² — each backward reads the
recorded forward output, never the parent's current value, so neither
operation records an expected parameter version (v3.7): mutating a
direct parameter input after forward leaves the graph valid and the
gradient correct for the forward that was recorded. Exceptional values
follow IEEE float64 (documented in the kernels): sqrt of negatives is
NaN, signed zeros are preserved; reciprocal of ±0 is ±inf and of ±inf
is ±0; NaN propagates. General division remains unsupported.

NumPy appears below only for references and inspection; a tripwire
test proves the forward/backward paths never touch it.

Selector: python -m pytest -q -k "native_optimizer_math"
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


VALUES = np.array([[4.0, 9.0, 16.0], [25.0, 0.25, 2.25]])  # exact roots


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
# Kernel symbols and NativeTensorCore forward
# ======================================================================


@needs_native
def test_native_optimizer_math_kernel_symbols_are_bound():
    library = cpp._require_library()
    for name in ("tf_core_sqrt", "tf_core_sqrt_contiguous",
                 "tf_core_reciprocal", "tf_core_reciprocal_contiguous"):
        assert getattr(library, name) is not None
    # The registry boundary is intact: the raw-buffer kernel list and
    # the locked tensor-core tuple are unchanged (the sum/mean
    # precedent — later core ops are not appended).
    assert cpp.TENSOR_CORE_KERNELS == ("relu", "add", "subtract",
                                       "multiply", "matmul")
    assert not hasattr(cpp, "sqrt") and not hasattr(cpp, "reciprocal")


@needs_native
def test_native_optimizer_math_core_forward_contiguous_and_scalar():
    core = cpp.NativeTensorCore.from_array(VALUES)
    out = core.sqrt()
    assert np.array_equal(out.to_numpy(), np.sqrt(VALUES))
    assert out.shape == core.shape and out.contiguous
    assert out.dtype == "float64" and out.device == "cpu"
    rec = core.reciprocal()
    assert np.array_equal(rec.to_numpy(), 1.0 / VALUES)
    # The input was not mutated and shares no storage with the outputs.
    assert np.array_equal(core.to_numpy(), VALUES)
    assert out.storage is not core.storage and rec.storage is not core.storage
    scalar = cpp.NativeTensorCore.from_array(9.0)
    assert scalar.sqrt().to_numpy().item() == 3.0
    assert scalar.reciprocal().to_numpy().item() == 1.0 / 9.0


@needs_native
def test_native_optimizer_math_core_forward_strided_views():
    core = cpp.NativeTensorCore.from_array(VALUES)
    transposed = core.T
    assert not transposed.contiguous
    assert np.array_equal(transposed.sqrt().to_numpy(), np.sqrt(VALUES.T))
    assert np.array_equal(transposed.reciprocal().to_numpy(), 1.0 / VALUES.T)
    narrowed = core.narrow(1, 1, 2)  # nonzero offset
    assert narrowed.offset != 0
    assert np.array_equal(narrowed.sqrt().to_numpy(),
                          np.sqrt(VALUES[:, 1:3]))
    combined = core.narrow(1, 1, 2).T  # transpose of a narrow
    assert np.array_equal(combined.reciprocal().to_numpy(),
                          1.0 / VALUES[:, 1:3].T)
    # Every output is fresh, owning, row-major contiguous.
    assert combined.reciprocal().contiguous


@needs_native
def test_native_optimizer_math_core_rejects_closed_input():
    core = cpp.NativeTensorCore.from_array(VALUES)
    core.close()
    with pytest.raises(RuntimeError, match="closed"):
        core.sqrt()
    with pytest.raises(RuntimeError, match="closed"):
        core.reciprocal()


# ======================================================================
# Exceptional values (IEEE float64, locked)
# ======================================================================


@needs_native
def test_native_optimizer_math_sqrt_exceptional_values():
    core = cpp.NativeTensorCore.from_array(
        [4.0, 0.0, -0.0, -1.0, float("inf"), float("nan")]
    )
    out = core.sqrt().to_numpy()
    assert out[0] == 2.0
    assert out[1] == 0.0 and not np.signbit(out[1])  # +0 -> +0
    assert out[2] == 0.0 and np.signbit(out[2])      # -0 -> -0 (IEEE)
    assert math.isnan(out[3])                        # negative -> NaN
    assert out[4] == math.inf                        # +inf -> +inf
    assert math.isnan(out[5])                        # NaN propagates


@needs_native
def test_native_optimizer_math_reciprocal_exceptional_values():
    core = cpp.NativeTensorCore.from_array(
        [4.0, -2.0, 0.0, -0.0, float("inf"), float("-inf"), float("nan")]
    )
    out = core.reciprocal().to_numpy()
    assert out[0] == 0.25 and out[1] == -0.5
    assert out[2] == math.inf                        # +0 -> +inf
    assert out[3] == -math.inf                       # -0 -> -inf
    assert out[4] == 0.0 and not np.signbit(out[4])  # +inf -> +0
    assert out[5] == 0.0 and np.signbit(out[5])      # -inf -> -0
    assert math.isnan(out[6])                        # NaN propagates


# ======================================================================
# NativeTensor forward
# ======================================================================


@needs_native
def test_native_optimizer_math_wrapper_forward_and_graph_construction():
    plain = NativeTensor.from_array(VALUES)
    for op_name in ("sqrt", "reciprocal"):
        result = getattr(plain, op_name)()
        assert result._op == "" and result._parents == ()  # graph-free
        assert result.is_leaf and not result.requires_grad
        result.close()
    assert not plain.closed  # output lifetime is independent
    tracked = NativeTensor.from_array(VALUES, requires_grad=True)
    node = tracked.sqrt()
    assert node._op == "sqrt" and node._parents == (tracked,)
    assert node.requires_grad and not node.is_leaf
    assert node._expected_versions == ()  # saved-output: no version reads
    rec_node = tracked.reciprocal()
    assert rec_node._op == "reciprocal"
    assert rec_node._expected_versions == ()
    # Parameter type never propagates through operations.
    parameter = NativeParameter(VALUES)
    assert type(parameter.sqrt()) is NativeTensor
    assert type(parameter.reciprocal()) is NativeTensor
    closed = NativeTensor.from_array(VALUES)
    closed.close()
    with pytest.raises(RuntimeError, match="closed"):
        closed.sqrt()
    with pytest.raises(RuntimeError, match="closed"):
        closed.reciprocal()


# ======================================================================
# Analytical backward
# ======================================================================


@needs_native
def test_native_optimizer_math_sqrt_gradient_exact():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    x.sqrt().sum().backward()
    assert np.array_equal(x.grad.to_numpy(), 0.5 / np.sqrt(VALUES))
    scalar = NativeTensor.from_array(16.0, requires_grad=True)
    scalar.sqrt().backward()
    assert scalar.grad.to_numpy().item() == 0.5 / 4.0


@needs_native
def test_native_optimizer_math_reciprocal_gradient_exact():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    x.reciprocal().sum().backward()
    assert np.array_equal(x.grad.to_numpy(), -1.0 / VALUES**2)
    scalar = NativeTensor.from_array(2.0, requires_grad=True)
    scalar.reciprocal().backward()
    assert scalar.grad.to_numpy().item() == -0.25


@needs_native
def test_native_optimizer_math_explicit_upstream_and_views():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.sqrt()
    upstream = NativeTensor.from_array(np.full(VALUES.shape, 2.0))
    out.backward(upstream)
    assert np.array_equal(x.grad.to_numpy(), 2.0 * 0.5 / np.sqrt(VALUES))
    # Through a view chain: transpose -> reciprocal -> sum.
    y = NativeTensor.from_array(VALUES, requires_grad=True)
    y.T.reciprocal().sum().backward()
    assert np.array_equal(y.grad.to_numpy(), -1.0 / VALUES**2)


@needs_native
def test_native_optimizer_math_chain_and_shared_subgraph():
    # Chain composition: mean(x * reciprocal(sqrt(x))) = mean(sqrt(x)).
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    inv_root = x.sqrt().reciprocal()
    loss = x.multiply(inv_root).mean()
    assert np.allclose(loss.to_numpy(), np.mean(np.sqrt(VALUES)),
                       atol=1e-15)
    loss.backward()
    # d mean(sqrt(x))/dx = 0.5/sqrt(x)/N, accumulated across the shared
    # subgraph (x feeds both multiply and the sqrt branch).
    assert np.allclose(x.grad.to_numpy(),
                       0.5 / np.sqrt(VALUES) / VALUES.size, atol=1e-15)


@needs_native
def test_native_optimizer_math_finite_differences():
    for op_name, reference in (
        ("sqrt", lambda a: float(np.sum(np.sqrt(a)))),
        ("reciprocal", lambda a: float(np.sum(1.0 / a))),
    ):
        # Safely inside the domain: positive, bounded away from zero.
        x = NativeTensor.from_array(VALUES, requires_grad=True)
        getattr(x, op_name)().sum().backward()
        numeric = _numeric_grad(reference, VALUES)
        assert np.allclose(x.grad.to_numpy(), numeric, atol=1e-6)
        # And at least one strided/view input.
        y = NativeTensor.from_array(VALUES, requires_grad=True)
        getattr(y.T, op_name)().sum().backward()
        assert np.allclose(y.grad.to_numpy(), numeric, atol=1e-6)


# ======================================================================
# Graph lifetime
# ======================================================================


@needs_native
def test_native_optimizer_math_one_shot_release_and_retain_graph():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.sqrt().sum()
    out.backward()
    with pytest.raises(RuntimeError, match="freed"):
        out.backward()
    grad_once = x.grad.to_numpy()
    x.zero_grad()
    retained = x.reciprocal().sum()
    retained.backward(retain_graph=True)
    retained.backward(retain_graph=True)  # saved output still available
    assert np.array_equal(x.grad.to_numpy(), -2.0 / VALUES**2)
    retained.backward()  # final default pass frees
    with pytest.raises(RuntimeError, match="freed"):
        retained.backward()
    assert np.array_equal(grad_once, 0.5 / np.sqrt(VALUES))


@needs_native
def test_native_optimizer_math_closed_saved_output_fails_cleanly():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    root = x.sqrt()
    loss = root.sum()
    root.close()  # the saved forward result backward must read
    with pytest.raises(RuntimeError, match="closed"):
        loss.backward()
    # The failed pass rolled back: no partial gradients, graph intact.
    assert x.grad is None
    assert loss._parents != () and not loss._graph_freed
    with pytest.raises(RuntimeError, match="closed"):
        loss.backward()  # deterministic on retry


# ======================================================================
# Versioning: saved-output => value-independent of later mutation
# ======================================================================


@needs_native
def test_native_optimizer_math_parameter_mutation_keeps_graph_valid():
    original = np.array([[4.0, 9.0], [16.0, 25.0]])
    p = NativeParameter(original)
    out = p.sqrt().sum()
    p.copy_value_(NativeTensor.from_array(np.ones((2, 2))))
    out.backward()  # reads the saved forward output — still valid
    assert np.array_equal(p.grad.to_numpy(), 0.5 / np.sqrt(original))
    q = NativeParameter(original)
    rec = q.reciprocal().sum()
    q.copy_value_(NativeTensor.from_array(np.full((2, 2), 3.0)))
    rec.backward()
    assert np.array_equal(q.grad.to_numpy(), -1.0 / original**2)
    # Deterministic and repeatable: a fresh forward uses the new value.
    q.zero_grad()
    q.reciprocal().sum().backward()
    assert np.array_equal(q.grad.to_numpy(), np.full((2, 2), -1.0 / 9.0))


@needs_native
def test_native_optimizer_math_sensitive_edges_still_guarded():
    # The new ops change no existing classification: a multiply edge on
    # the same parameter still goes stale after mutation.
    p = NativeParameter([[4.0, 9.0], [16.0, 25.0]])
    mixed = p.sqrt().multiply(p).sum()
    p.copy_value_(NativeTensor.from_array(np.ones((2, 2))))
    with pytest.raises(RuntimeError, match="stale"):
        mixed.backward()
    assert p.grad is None


# ======================================================================
# Guardrails
# ======================================================================


@needs_native
def test_native_optimizer_math_uses_no_numpy_compute(monkeypatch):
    x = NativeTensor.from_array(VALUES, requires_grad=True)

    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in ("sqrt", "reciprocal", "divide", "add", "subtract",
                 "multiply", "matmul", "sum", "mean", "negative",
                 "power", "copyto"):
        monkeypatch.setattr(np, name, _tripwire)
    x.sqrt().sum().backward()
    x.reciprocal().sum().backward()
    monkeypatch.undo()
    expected = 0.5 / np.sqrt(VALUES) + (-1.0 / VALUES**2)
    assert np.allclose(x.grad.to_numpy(), expected, atol=1e-15)


@needs_native
def test_native_optimizer_math_scope_boundaries_hold():
    # No general division API, no NativeAdam, no operator overloads.
    x = NativeTensor.from_array(VALUES)
    assert not hasattr(x, "divide") and not hasattr(x, "__truediv__")
    assert not hasattr(cpp.NativeTensorCore.from_array(VALUES), "divide")
    import tensorforge.experimental as experimental
    assert not hasattr(experimental, "NativeAdam")
    # The stable Tensor is untouched and still has its own sqrt-free
    # native isolation: the native ops reject nothing new from it
    # because they are methods, not dispatchers.
    stable = tensorforge.Tensor(4.0, requires_grad=True)
    (stable * stable).backward()
    assert np.allclose(stable.grad, 8.0)
