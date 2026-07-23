"""Tests for the fused native softmax — Phase E, milestone E3.

The classification stack's first probability transform
(docs/native_classification_design.md §4.3). One axis-wise fused
operation through the complete native stack: the new C++ classification
unit (`tf::softmax_forward_contiguous`) → the guarded, self-validating,
**contiguous-only** C ABI (`tf_core_softmax_forward`) → ctypes →
`NativeTensorCore.softmax(axis=-1)` with Policy-B copy-then-compute →
differentiable `NativeTensor.softmax(axis=-1)`.

Forward is a **fused maximum shift**: per slice `exp(x - max(x)) /
sum(exp(x - max(x)))`, computed inside the kernel in float64, never
composed from public max/subtract/exp/sum/divide operations (none of
which softmax adds). Backward is the closed-form Jacobian-vector
product

    dx = y * (upstream - sum(upstream * y, axis, keepdims=True))

**composed at the graph-unaware Core layer** — there is no dedicated
softmax backward kernel — and it reads only the **saved output** `y`, so
the node records no expected parameter version (the `exp` archetype,
deliberately unlike `log`).

Exceptional values follow plain IEEE arithmetic with no special-casing:
a NaN or `+inf` in a slice propagates through the shift and sum, making
that slice NaN, while `-inf` simply takes zero mass. The reference below
is the *same* maximum-shift algorithm in NumPy rather than another
framework's infinity handling.

NumPy appears only as an external oracle and for inspection; a tripwire
test proves the forward/backward paths never compute with it.

Selector: python -m pytest -q -k "native_softmax"
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


VALUES = np.array([[1.0, 2.0, 0.5], [-1.0, 0.25, 3.0]])
CUBE = np.arange(24.0).reshape(2, 3, 4) / 5.0 - 2.0


@pytest.fixture(autouse=True)
def _disarm_after_each():
    """No test may leave the allocation injector armed or the native
    error slot dirty (the test_native_abi_errors.py convention)."""
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


def softmax_reference(x, axis):
    """The same maximum-shift algorithm the kernel implements, in NumPy.

    Deliberately not another framework's softmax: the contract is that
    the fused kernel agrees with *this* algorithm, including at IEEE
    edges where implementations differ."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


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
# Kernel symbols, source unit, and the capability boundary
# ======================================================================


@needs_native
def test_native_softmax_kernel_symbol_is_bound():
    library = cpp._require_library()
    assert library.tf_core_softmax_forward is not None
    assert "tf_core_softmax_forward" in cpp._CHECKED_KERNELS
    # An ABI symbol is not a capability name.
    assert "tf_core_softmax_forward" not in cpp.RAW_KERNELS
    assert "tf_core_softmax_forward" not in cpp.TENSOR_CORE_OPS
    # E3 added no raw NumPy-buffer kernel and left the frozen registry.
    assert cpp.TENSOR_CORE_KERNELS == ("relu", "add", "subtract",
                                       "multiply", "matmul")
    assert "softmax" not in cpp.RAW_KERNELS
    assert not hasattr(cpp, "softmax")
    # E3 deliberately ships **no** backward kernel: the gradient is
    # composed from existing Core ops.
    assert not any("softmax" in name and "backward" in name
                   for name in cpp._CHECKED_KERNELS)


def test_native_softmax_source_unit_exists():
    """The Phase-E classification source unit locked by E0 §9.1."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    assert (root / "cpp" / "src" / "classification.cpp").is_file()
    assert (root / "cpp" / "tests" / "test_softmax.cpp").is_file()
    # The softmax kernel and its export live in the classification unit,
    # not the elementwise one. Checked by *symbol definition* rather than
    # by banning the word, so cross-referencing prose stays possible.
    classification = (root / "cpp" / "src" / "classification.cpp").read_text(
        encoding="utf-8"
    )
    elementwise = (root / "cpp" / "src" / "elementwise.cpp").read_text(
        encoding="utf-8"
    )
    assert "tf_core_softmax_forward" in classification
    assert "softmax_forward_contiguous" in classification
    for symbol in ("tf_core_softmax_forward(", "softmax_forward_contiguous("):
        assert symbol not in elementwise, symbol


def test_native_softmax_registry_placement():
    """softmax is a Core op and an autograd op — nothing else. Runs
    without the compiled backend: these are pure inventory facts."""
    assert "softmax" in cpp.TENSOR_CORE_OPS
    assert "softmax" in cpp.AUTOGRAD_OPS
    assert "softmax" not in cpp.UNSUPPORTED
    assert "softmax" not in cpp.NATIVE_MODULES
    assert "softmax" not in cpp.NATIVE_LOSSES
    assert not hasattr(cpp, "NATIVE_METRICS")
    # E1/E2 stay implemented alongside it.
    for shipped in ("exp", "log"):
        assert shipped in cpp.TENSOR_CORE_OPS and shipped in cpp.AUTOGRAD_OPS
        assert shipped not in cpp.UNSUPPORTED
    info = cpp.backend_info()
    assert "softmax" in info["tensor_core_ops"]
    assert "softmax" in info["autograd_ops"]
    assert "softmax" not in info["unsupported"]
    # E4 landed "log_softmax" as a distinct fused capability, deliberately
    # not composed from log and softmax (its contract lives in
    # tests/test_native_log_softmax.py), and E5 landed the cross-entropy
    # **Core** layer (tests/test_native_cross_entropy_core.py).
    assert "log_softmax" in cpp.TENSOR_CORE_OPS
    assert "log_softmax" in cpp.AUTOGRAD_OPS
    assert "log_softmax" not in cpp.UNSUPPORTED
    # E5 is Core-only: no differentiable operation (E6), no module or
    # metric (E7).
    for core_op in ("cross_entropy_forward", "cross_entropy_backward"):
        assert core_op in cpp.TENSOR_CORE_OPS, core_op
        assert core_op not in cpp.AUTOGRAD_OPS, core_op
    assert "cross_entropy" not in cpp.AUTOGRAD_OPS
    assert "cross_entropy" not in cpp.TENSOR_CORE_OPS
    for absent in ("NativeCrossEntropyLoss", "native_accuracy"):
        assert absent in cpp.UNSUPPORTED, absent
        assert absent not in cpp.TENSOR_CORE_OPS
        assert absent not in cpp.AUTOGRAD_OPS


# ======================================================================
# Forward correctness
# ======================================================================


@needs_native
def test_native_softmax_forward_rank1():
    values = np.array([1.0, 2.0, 3.0])
    for axis in (0, -1):
        out = cpp.NativeTensorCore.from_array(values).softmax(axis).to_numpy()
        assert np.allclose(out, softmax_reference(values, axis), atol=1e-15)
        assert math.isclose(float(out.sum()), 1.0, rel_tol=1e-12)


@needs_native
def test_native_softmax_forward_rank2_every_axis():
    for axis in (0, 1, -1, -2):
        out = cpp.NativeTensorCore.from_array(VALUES).softmax(axis).to_numpy()
        assert np.allclose(out, softmax_reference(VALUES, axis), atol=1e-15)
        assert np.allclose(out.sum(axis=axis), 1.0, atol=1e-12)
        assert np.all(out >= 0.0)


@needs_native
def test_native_softmax_forward_rank3_first_middle_last():
    for axis in (0, 1, 2):
        out = cpp.NativeTensorCore.from_array(CUBE).softmax(axis).to_numpy()
        assert np.allclose(out, softmax_reference(CUBE, axis), atol=1e-15)
        assert np.allclose(out.sum(axis=axis), 1.0, atol=1e-12)
    # A rank-4 tensor exercises a middle axis with both outer and inner > 1.
    quad = np.linspace(-2.0, 2.0, 2 * 3 * 4 * 2).reshape(2, 3, 4, 2)
    for axis in (1, 2):
        out = cpp.NativeTensorCore.from_array(quad).softmax(axis).to_numpy()
        assert np.allclose(out, softmax_reference(quad, axis), atol=1e-15)


@needs_native
def test_native_softmax_negative_axes_match_positive():
    for negative, positive in ((-1, 2), (-2, 1), (-3, 0)):
        left = cpp.NativeTensorCore.from_array(CUBE).softmax(negative).to_numpy()
        right = cpp.NativeTensorCore.from_array(CUBE).softmax(positive).to_numpy()
        assert np.array_equal(left, right)


@needs_native
def test_native_softmax_axis_length_one_is_all_mass():
    values = np.array([[5.0], [-3.0], [100.0]])
    out = cpp.NativeTensorCore.from_array(values).softmax(1).to_numpy()
    assert np.array_equal(out, np.ones_like(values))


@needs_native
def test_native_softmax_equal_values_are_uniform():
    values = np.full((3, 4), 2.75)
    out = cpp.NativeTensorCore.from_array(values).softmax(-1).to_numpy()
    assert np.array_equal(out, np.full((3, 4), 0.25))


@needs_native
def test_native_softmax_random_values():
    rng = np.random.default_rng(20260721)
    values = rng.uniform(-8.0, 8.0, size=(5, 7, 3))
    for axis in (0, 1, 2, -1):
        out = cpp.NativeTensorCore.from_array(values).softmax(axis).to_numpy()
        assert np.allclose(out, softmax_reference(values, axis),
                           rtol=1e-14, atol=0.0)
        assert np.allclose(out.sum(axis=axis), 1.0, atol=1e-12)


@needs_native
def test_native_softmax_is_stable_under_large_offsets():
    """A naive exp-then-normalize gives inf/inf at +800 and 0/0 at -800;
    the fused maximum shift stays exact."""
    base = np.array([0.0, 1.0, 2.0, 3.0])
    expected = softmax_reference(base, -1)
    for offset in (800.0, -800.0, 1e5, -1e5, 1e10, -1e10):
        shifted = base + offset
        out = cpp.NativeTensorCore.from_array(shifted).softmax(-1).to_numpy()
        assert np.all(np.isfinite(out)), offset
        assert np.allclose(out, expected, atol=1e-6), offset
        assert math.isclose(float(out.sum()), 1.0, rel_tol=1e-12)
    # Sanity: the naive formulation really would have failed here.
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(base + 800.0)
        assert not np.all(np.isfinite(naive / naive.sum()))


@needs_native
def test_native_softmax_additive_shift_invariance_per_slice():
    """Adding a constant to one slice leaves that slice's softmax
    unchanged (and does not disturb any other slice)."""
    values = VALUES.copy()
    reference = cpp.NativeTensorCore.from_array(values).softmax(-1).to_numpy()
    shifted = values.copy()
    shifted[0] += 37.5           # only the first slice moves
    out = cpp.NativeTensorCore.from_array(shifted).softmax(-1).to_numpy()
    assert np.allclose(out, reference, atol=1e-13)


@needs_native
def test_native_softmax_forward_is_deterministic():
    core = cpp.NativeTensorCore.from_array(CUBE)
    first = core.softmax(1).to_numpy()
    second = core.softmax(1).to_numpy()
    assert np.array_equal(first, second)


# ======================================================================
# Exceptional values (plain IEEE, matching the implemented algorithm)
# ======================================================================


@needs_native
def test_native_softmax_exceptional_values_follow_the_max_shift_algorithm():
    values = np.array([
        [1.0, float("nan"), 2.0],          # NaN poisons its slice
        [float("inf"), 1.0, 2.0],          # inf - inf -> NaN
        [float("-inf"), 1.0, 2.0],         # -inf simply gets zero mass
    ])
    out = cpp.NativeTensorCore.from_array(values).softmax(-1).to_numpy()
    expected = softmax_reference(values, -1)
    assert np.array_equal(out, expected, equal_nan=True)
    assert np.all(np.isnan(out[0])), "a NaN makes its whole slice NaN"
    assert np.all(np.isnan(out[1])), "+inf makes its whole slice NaN"
    assert out[2][0] == 0.0, "-inf takes zero mass"
    assert math.isclose(float(out[2][1:].sum()), 1.0, rel_tol=1e-12)
    # A NaN is never silently turned into a finite probability.
    assert not np.any(np.isfinite(out[0]))
    # An all -inf slice is NaN (-inf - -inf).
    all_neg = cpp.NativeTensorCore.from_array(
        np.array([float("-inf"), float("-inf")])
    ).softmax(-1).to_numpy()
    assert np.all(np.isnan(all_neg))


@needs_native
def test_native_softmax_exceptional_values_are_not_abi_errors():
    """A numerically exceptional but structurally valid call must not
    raise and must leave the native error slot clear."""
    values = np.array([float("nan"), float("inf"), float("-inf")])
    out = cpp.NativeTensorCore.from_array(values).softmax(0).to_numpy()
    assert out.shape == (3,)
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    # And through the autograd layer too.
    tracked = NativeTensor.from_array(values, requires_grad=True)
    tracked.softmax(0).sum().backward()
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK


# ======================================================================
# Axis validation
# ======================================================================


@needs_native
@pytest.mark.parametrize("bad_axis", [True, False, 1.0, 0.5, "0", None,
                                      (0,), [0]])
def test_native_softmax_rejects_non_integer_axis(bad_axis):
    core = cpp.NativeTensorCore.from_array(VALUES)
    with pytest.raises(TypeError, match="axis"):
        core.softmax(bad_axis)
    tracked = NativeTensor.from_array(VALUES, requires_grad=True)
    with pytest.raises(TypeError, match="axis"):
        tracked.softmax(bad_axis)
    # Nothing was consumed: a valid call still works.
    assert core.softmax(-1).shape == VALUES.shape


@needs_native
@pytest.mark.parametrize("bad_axis", [2, 3, 99, -3, -4, -99])
def test_native_softmax_rejects_out_of_range_axis(bad_axis):
    core = cpp.NativeTensorCore.from_array(VALUES)   # rank 2
    with pytest.raises(ValueError, match="out of bounds"):
        core.softmax(bad_axis)
    tracked = NativeTensor.from_array(VALUES, requires_grad=True)
    with pytest.raises(ValueError, match="out of bounds"):
        tracked.softmax(bad_axis)


@needs_native
def test_native_softmax_rejects_rank_zero_input():
    """softmax needs an axis to normalize over, so rank 0 is rejected —
    every integer axis is out of bounds on a scalar."""
    rank_zero = cpp.NativeTensorCore.from_array(VALUES).sum()
    assert rank_zero.shape == ()
    for axis in (0, -1):
        with pytest.raises(ValueError, match="out of bounds"):
            rank_zero.softmax(axis)


@needs_native
def test_native_softmax_axis_validation_precedes_allocation(monkeypatch):
    """A rejected axis must allocate nothing — the validation runs before
    the output (and before any Policy-B copy)."""
    core = cpp.NativeTensorCore.from_array(VALUES)

    def _fail(*args, **kwargs):
        raise AssertionError("allocation happened before axis validation")

    monkeypatch.setattr(cpp.NativeTensorCore, "zeros", staticmethod(_fail))
    monkeypatch.setattr(cpp.NativeTensorCore, "contiguous_copy", _fail)
    with pytest.raises(ValueError, match="out of bounds"):
        core.softmax(5)
    with pytest.raises(TypeError, match="axis"):
        core.softmax(True)


@needs_native
def test_native_softmax_rejects_closed_input():
    core = cpp.NativeTensorCore.from_array(VALUES)
    core.close()
    with pytest.raises(RuntimeError, match="closed"):
        core.softmax(-1)
    tracked = NativeTensor.from_array(VALUES)
    tracked.close()
    with pytest.raises(RuntimeError, match="closed"):
        tracked.softmax(-1)


# ======================================================================
# Policy-B, layout, and ownership
# ======================================================================


@needs_native
def test_native_softmax_output_is_fresh_owning_contiguous():
    core = cpp.NativeTensorCore.from_array(VALUES)
    out = core.softmax(-1)
    assert out.shape == core.shape
    assert out.contiguous and out.offset == 0
    assert out.dtype == "float64" and out.device == "cpu"
    assert out.storage is not core.storage
    assert np.array_equal(core.to_numpy(), VALUES)  # input unmutated


@needs_native
def test_native_softmax_handles_non_contiguous_inputs():
    core = cpp.NativeTensorCore.from_array(CUBE)
    transposed = core.transpose((2, 1, 0))
    assert not transposed.contiguous
    out = transposed.softmax(-1)
    assert np.allclose(out.to_numpy(),
                       softmax_reference(CUBE.transpose(2, 1, 0), -1),
                       atol=1e-15)
    assert out.contiguous and out.offset == 0
    # A narrowed view with a nonzero offset. (Views borrow their base's
    # storage, so the base must stay referenced for the view to be
    # usable — the existing ownership rule.)
    base = cpp.NativeTensorCore.from_array(VALUES)
    narrowed = base.narrow(1, 1, 2)
    assert narrowed.offset != 0
    assert np.allclose(narrowed.softmax(-1).to_numpy(),
                       softmax_reference(VALUES[:, 1:3], -1), atol=1e-15)
    # ...and the transpose of a narrow (non-contiguous *and* offset).
    combined = base.narrow(1, 1, 2).T
    assert not combined.contiguous and combined.offset != 0
    assert np.allclose(combined.softmax(0).to_numpy(),
                       softmax_reference(VALUES[:, 1:3].T, 0), atol=1e-15)
    # A reshaped (still contiguous) view takes the direct path.
    cube_base = cpp.NativeTensorCore.from_array(CUBE)
    reshaped = cube_base.reshape((6, 4))
    assert reshaped.contiguous
    assert np.allclose(reshaped.softmax(-1).to_numpy(),
                       softmax_reference(CUBE.reshape(6, 4), -1), atol=1e-15)
    assert np.array_equal(core.to_numpy(), CUBE)   # never mutated
    assert np.array_equal(base.to_numpy(), VALUES)


@needs_native
def test_native_softmax_policy_b_temporary_is_closed_on_success(monkeypatch):
    """The private contiguous copy must be closed exactly once when the
    call succeeds — instrumented through the real copy helper."""
    made = []
    original = cpp.NativeTensorCore.contiguous_copy

    def _recording_copy(self):
        temp = original(self)
        made.append(temp)
        return temp

    monkeypatch.setattr(cpp.NativeTensorCore, "contiguous_copy",
                        _recording_copy)
    base = cpp.NativeTensorCore.from_array(VALUES)  # keep the base alive
    strided = base.T
    out = strided.softmax(0)
    assert len(made) == 1, "a non-contiguous input must be copied once"
    assert made[0]._closed, "the Policy-B temporary was not closed"
    assert not out._closed and out.contiguous
    # A contiguous input makes no copy at all.
    made.clear()
    contiguous = cpp.NativeTensorCore.from_array(VALUES)
    contiguous.softmax(-1)
    assert made == []


@needs_fault_injection
def test_native_softmax_policy_b_temporary_is_closed_on_failure(monkeypatch):
    """...and also when the output allocation fails after the copy."""
    made = []
    original = cpp.NativeTensorCore.contiguous_copy

    def _recording_copy(self):
        temp = original(self)
        made.append(temp)
        return temp

    monkeypatch.setattr(cpp.NativeTensorCore, "contiguous_copy",
                        _recording_copy)
    base = cpp.NativeTensorCore.from_array(VALUES)  # keep the base alive
    strided = base.T
    # The injector counts allocations and the copy itself consumes more
    # than one, so sweep rather than assume a stage number. For every
    # failure: either it happened *inside* the copy (nothing recorded yet)
    # or the copy had already been made, in which case it must have been
    # closed by the Policy-B cleanup.
    post_copy_failures = 0
    for nth in range(1, 5):
        made.clear()
        try:
            cpp._arm_alloc_failure(nth)
            strided.softmax(0)
        except MemoryError:
            cpp._arm_alloc_failure(0)
            if made:
                assert made[0]._closed, (
                    f"the Policy-B temporary leaked when allocation {nth} failed"
                )
                post_copy_failures += 1
        else:
            cpp._arm_alloc_failure(0)
            assert made and made[0]._closed
    assert post_copy_failures >= 1, (
        "the sweep never reached a failure after the contiguous copy"
    )
    # The caller's input is untouched and the operation still works.
    assert np.array_equal(strided.to_numpy(), VALUES.T)
    assert np.allclose(strided.softmax(0).to_numpy(),
                       softmax_reference(VALUES.T, 0), atol=1e-15)


# ======================================================================
# NativeTensor forward and graph construction
# ======================================================================


@needs_native
def test_native_softmax_wrapper_forward_and_graph_construction():
    plain = NativeTensor.from_array(VALUES)
    result = plain.softmax(-1)
    assert np.allclose(result.to_numpy(), softmax_reference(VALUES, -1),
                       atol=1e-15)
    assert result._op == "" and result._parents == ()
    assert result.is_leaf and not result.requires_grad
    assert result._graph_resources == ()
    result.close()
    assert not plain.closed

    tracked = NativeTensor.from_array(VALUES, requires_grad=True)
    node = tracked.softmax(-1)
    assert node._op == "softmax" and node._parents == (tracked,)
    assert node.requires_grad and not node.is_leaf
    # Saved-output backward: no version snapshot, no private resource
    # (y is the node's own core).
    assert node._expected_versions == ()
    assert node._graph_resources == ()

    parameter = NativeParameter(VALUES)
    assert type(parameter.softmax(-1)) is NativeTensor


# ======================================================================
# Backward correctness
# ======================================================================


def _analytic_grad(x, upstream, axis):
    y = softmax_reference(x, axis)
    return y * (upstream - np.sum(upstream * y, axis=axis, keepdims=True))


@needs_native
@pytest.mark.parametrize("shape,axis", [
    ((5,), 0), ((5,), -1),
    ((2, 3), 0), ((2, 3), 1), ((2, 3), -1), ((2, 3), -2),
    ((2, 3, 4), 0), ((2, 3, 4), 1), ((2, 3, 4), 2),
    ((2, 3, 4), -1), ((2, 3, 4), -2), ((2, 3, 4), -3),
])
def test_native_softmax_backward_matches_the_analytic_jacobian(shape, axis):
    rng = np.random.default_rng(7)
    values = rng.uniform(-3.0, 3.0, size=shape)
    upstream = rng.uniform(-2.0, 2.0, size=shape)
    x = NativeTensor.from_array(values, requires_grad=True)
    x.softmax(axis).backward(NativeTensor.from_array(upstream))
    assert np.allclose(x.grad.to_numpy(),
                       _analytic_grad(values, upstream, axis), atol=1e-14)


@needs_native
def test_native_softmax_backward_negative_axis_equivalence():
    rng = np.random.default_rng(11)
    values = rng.uniform(-2.0, 2.0, size=(2, 3, 4))
    upstream = rng.uniform(-1.0, 1.0, size=(2, 3, 4))
    grads = []
    for axis in (1, -2):
        x = NativeTensor.from_array(values, requires_grad=True)
        x.softmax(axis).backward(NativeTensor.from_array(upstream))
        grads.append(x.grad.to_numpy())
    assert np.array_equal(grads[0], grads[1])


@needs_native
def test_native_softmax_gradient_sums_to_zero_along_the_axis():
    """A structural invariant of the softmax Jacobian: each slice's
    gradient sums to zero, because probabilities are constrained to 1."""
    rng = np.random.default_rng(3)
    values = rng.uniform(-2.0, 2.0, size=(3, 5))
    upstream = rng.uniform(-4.0, 4.0, size=(3, 5))
    x = NativeTensor.from_array(values, requires_grad=True)
    x.softmax(1).backward(NativeTensor.from_array(upstream))
    assert np.allclose(x.grad.to_numpy().sum(axis=1), 0.0, atol=1e-13)


@needs_native
def test_native_softmax_uniform_upstream_gives_zero_gradient():
    """A constant upstream over a slice cancels exactly: the softmax of
    a slice cannot change its own total."""
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    x.softmax(-1).backward(NativeTensor.from_array(np.full(VALUES.shape, 2.5)))
    assert np.allclose(x.grad.to_numpy(), 0.0, atol=1e-15)
    # Which is also why plain .sum() has a zero gradient — the reason the
    # versioning test below uses a *weighted* loss instead.
    y = NativeTensor.from_array(VALUES, requires_grad=True)
    y.softmax(-1).sum().backward()
    assert np.allclose(y.grad.to_numpy(), 0.0, atol=1e-15)


@needs_native
def test_native_softmax_finite_differences():
    rng = np.random.default_rng(5)
    values = rng.uniform(-2.0, 2.0, size=(3, 4))
    weights = rng.uniform(0.5, 2.0, size=(3, 4))

    def loss_of(a):
        return float(np.sum(softmax_reference(a, -1) * weights))

    x = NativeTensor.from_array(values, requires_grad=True)
    x.softmax(-1).multiply(NativeTensor.from_array(weights)).sum().backward()
    assert np.allclose(x.grad.to_numpy(), _numeric_grad(loss_of, values),
                       atol=1e-6)
    # And along axis 0.
    def loss_axis0(a):
        return float(np.sum(softmax_reference(a, 0) * weights))

    z = NativeTensor.from_array(values, requires_grad=True)
    z.softmax(0).multiply(NativeTensor.from_array(weights)).sum().backward()
    assert np.allclose(z.grad.to_numpy(), _numeric_grad(loss_axis0, values),
                       atol=1e-6)


@needs_native
def test_native_softmax_backward_through_a_non_contiguous_parent():
    rng = np.random.default_rng(13)
    values = rng.uniform(-2.0, 2.0, size=(3, 4))
    upstream = rng.uniform(-1.0, 1.0, size=(4, 3))
    x = NativeTensor.from_array(values, requires_grad=True)
    x.T.softmax(-1).backward(NativeTensor.from_array(upstream))
    expected = _analytic_grad(values.T, upstream, -1).T
    assert np.allclose(x.grad.to_numpy(), expected, atol=1e-14)


@needs_native
def test_native_softmax_chained_and_shared_subgraph():
    rng = np.random.default_rng(17)
    values = rng.uniform(-2.0, 2.0, size=(2, 4))
    weights = rng.uniform(0.5, 1.5, size=(2, 4))
    # Chained: log(softmax(x)) . weights — a genuine composition of the
    # three Phase-E ops shipped so far.
    x = NativeTensor.from_array(values, requires_grad=True)
    x.softmax(-1).log().multiply(
        NativeTensor.from_array(weights)
    ).sum().backward()
    probabilities = softmax_reference(values, -1)
    upstream = weights / probabilities          # d/dy of (w * log y)
    assert np.allclose(x.grad.to_numpy(),
                       _analytic_grad(values, upstream, -1), atol=1e-12)
    # Shared subgraph: y used twice accumulates through one node.
    z = NativeTensor.from_array(values, requires_grad=True)
    y = z.softmax(-1)
    y.add(y).multiply(NativeTensor.from_array(weights)).sum().backward()
    assert np.allclose(z.grad.to_numpy(),
                       _analytic_grad(values, 2.0 * weights, -1), atol=1e-13)


@needs_native
def test_native_softmax_repeated_accumulation_and_retain_graph():
    weights = np.linspace(0.5, 2.0, VALUES.size).reshape(VALUES.shape)
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    loss = x.softmax(-1).multiply(NativeTensor.from_array(weights)).sum()
    loss.backward(retain_graph=True)
    once = x.grad.to_numpy().copy()
    loss.backward(retain_graph=True)
    assert np.allclose(x.grad.to_numpy(), 2.0 * once, atol=1e-15)
    x.zero_grad()
    assert x.grad is None
    loss.backward()  # the final default pass frees the history
    assert np.allclose(x.grad.to_numpy(), once, atol=1e-15)
    with pytest.raises(RuntimeError, match="freed"):
        loss.backward()


# ======================================================================
# Versioning: saved output => value-independent of later mutation
# ======================================================================


@needs_native
def test_native_softmax_records_no_version_snapshot():
    p = NativeParameter(VALUES)
    node = p.softmax(-1)
    assert node._expected_versions == ()
    assert node._graph_resources == ()


@needs_native
def test_native_softmax_parameter_mutation_keeps_graph_valid():
    """The E3 invariant: backward reads only the saved probabilities, so
    mutating a direct parameter after forward leaves the edge valid — and
    the gradient is the one for the forward that actually ran.

    A *weighted* loss is used deliberately: the gradient of a plain
    ``softmax(x).sum()`` is zero and would hide an incorrect result."""
    original = np.array([[0.5, 1.5, -1.0], [2.0, 0.0, 1.0]])
    weights = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    p = NativeParameter(original)
    loss = p.softmax(-1).multiply(NativeTensor.from_array(weights)).sum()
    mutated = np.full(original.shape, 4.0)
    p.copy_value_(NativeTensor.from_array(mutated))
    loss.backward()  # no stale error: nothing rereads p

    expected = _analytic_grad(original, weights, -1)
    assert np.allclose(p.grad.to_numpy(), expected, atol=1e-14)
    assert not np.allclose(p.grad.to_numpy(), 0.0), "gradient must be nonzero"
    # Explicitly NOT the gradient of the mutated values (which, being
    # uniform, would give an entirely different result).
    mutated_grad = _analytic_grad(mutated, weights, -1)
    assert not np.allclose(p.grad.to_numpy(), mutated_grad, atol=1e-8)
    # A fresh forward after mutation uses the new values.
    p.zero_grad()
    p.softmax(-1).multiply(NativeTensor.from_array(weights)).sum().backward()
    assert np.allclose(p.grad.to_numpy(), mutated_grad, atol=1e-14)


@needs_native
def test_native_softmax_does_not_weaken_other_versioning():
    """E3 must not perturb the established classifications."""
    p = NativeParameter(np.array([[1.0, 2.0], [3.0, 4.0]]))
    # log stays live-input / version-checked.
    log_loss = p.log().sum()
    p.copy_value_(NativeTensor.from_array(np.full((2, 2), 5.0)))
    with pytest.raises(RuntimeError, match="stale"):
        log_loss.backward()
    assert p.grad is None
    # exp stays saved-output / no version.
    q = NativeParameter(np.array([[1.0, 2.0], [3.0, 4.0]]))
    exp_loss = q.exp().sum()
    q.copy_value_(NativeTensor.from_array(np.full((2, 2), 2.0)))
    exp_loss.backward()
    assert q.grad is not None
    # multiply stays guarded.
    r = NativeParameter(np.array([[1.0, 2.0], [3.0, 4.0]]))
    mul_loss = r.multiply(r).sum()
    r.copy_value_(NativeTensor.from_array(np.full((2, 2), 6.0)))
    with pytest.raises(RuntimeError, match="stale"):
        mul_loss.backward()
    assert r.grad is None


@needs_native
def test_native_softmax_mixed_graph_with_a_stale_log_rolls_back_globally():
    """A softmax branch is valid across mutation, but if another branch
    goes stale the whole pass must commit nothing."""
    p = NativeParameter(np.array([[1.0, 2.0], [3.0, 4.0]]))
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    loss = (p.softmax(-1).multiply(NativeTensor.from_array(weights)).sum()
            .add(p.log().sum()))
    p.copy_value_(NativeTensor.from_array(np.full((2, 2), 7.0)))
    with pytest.raises(RuntimeError, match="stale"):
        loss.backward()
    assert p.grad is None       # the softmax branch committed nothing either


# ======================================================================
# Graph lifetime
# ======================================================================


@needs_native
def test_native_softmax_no_grad_and_detached_paths():
    x = NativeTensor.from_array(VALUES)
    out = x.softmax(-1)
    assert out.is_leaf and out._backward is None and out._parents == ()
    assert not out.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad|requires_grad"):
        out.sum().backward()
    tracked = NativeTensor.from_array(VALUES, requires_grad=True)
    detached = tracked.detach().softmax(-1)
    assert not detached.requires_grad and detached.is_leaf
    detached.close()


@needs_native
def test_native_softmax_closed_saved_output_fails_atomically():
    """The saved probabilities *are* the derivative, so closing them
    before a downstream backward must fail — and commit nothing."""
    weights = np.linspace(0.5, 2.0, VALUES.size).reshape(VALUES.shape)
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    y = x.softmax(-1)
    loss = y.multiply(NativeTensor.from_array(weights)).sum()
    y.close()
    with pytest.raises(RuntimeError, match="closed"):
        loss.backward()
    assert x.grad is None
    assert loss._parents != () and not loss._graph_freed
    with pytest.raises(RuntimeError, match="closed"):
        loss.backward()          # deterministic on retry
    assert x.grad is None


@needs_native
def test_native_softmax_closed_parent_follows_the_accumulation_rule():
    """softmax backward never rereads the parent's value, but the parent
    must still be open to *receive* its gradient — the engine's ordinary
    ownership rule, not a version requirement."""
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    intermediate = x.add(x)
    loss = intermediate.softmax(-1).multiply(
        NativeTensor.from_array(np.linspace(1.0, 2.0, VALUES.size)
                                .reshape(VALUES.shape))
    ).sum()
    intermediate.close()
    with pytest.raises(RuntimeError, match="closed"):
        loss.backward()
    assert x.grad is None


@needs_native
def test_native_softmax_abandoned_graph_closes_cleanly():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.softmax(-1)
    out.close()
    out.close()  # idempotent
    assert out.closed and not x.closed
    assert x.grad is None
    x.softmax(-1).multiply(
        NativeTensor.from_array(np.full(VALUES.shape, 2.0))
    ).sum().backward()
    assert x.grad is not None


@needs_native
def test_native_softmax_rejects_invalid_backward_arguments():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.softmax(-1)
    with pytest.raises(ValueError, match="non-scalar"):
        out.backward()
    with pytest.raises(ValueError, match="shape"):
        out.backward(NativeTensor.from_array(np.ones((5, 5))))
    with pytest.raises(TypeError):
        out.backward(np.ones(VALUES.shape))
    closed_upstream = NativeTensor.from_array(np.ones(VALUES.shape))
    closed_upstream.close()
    with pytest.raises(RuntimeError, match="closed"):
        out.backward(closed_upstream)
    assert x.grad is None
    out.backward(NativeTensor.from_array(np.ones(VALUES.shape)))
    assert x.grad is not None


# ======================================================================
# Allocation failure
# ======================================================================


@needs_fault_injection
def test_native_softmax_allocation_failure_is_atomic():
    """Sweep the injector across a forward+backward. Every stage that can
    be hit must raise MemoryError, mutate nothing, commit no gradient, and
    leave the graph retryable. (The injector counts allocations rather
    than naming stages, so the sweep reports what is reachable instead of
    assuming a fixed mapping.)"""
    values = np.array([[1.0, 2.0], [0.5, -1.0]])
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])

    # -- forward output allocation --
    x = NativeTensor.from_array(values, requires_grad=True)
    with pytest.raises(MemoryError):
        cpp._arm_alloc_failure(1)
        x.softmax(-1)
    cpp._arm_alloc_failure(0)
    assert np.array_equal(x.to_numpy(), values)
    assert x.grad is None
    assert np.allclose(x.softmax(-1).to_numpy(),
                       softmax_reference(values, -1), atol=1e-15)

    # -- backward allocations: weighted product, reduction, subtraction,
    #    final product, accumulation --
    hit = 0
    for nth in range(1, 9):
        y = NativeTensor.from_array(values, requires_grad=True)
        loss = y.softmax(-1).multiply(
            NativeTensor.from_array(weights)
        ).sum()
        try:
            cpp._arm_alloc_failure(nth)
            loss.backward()
        except MemoryError:
            hit += 1
            cpp._arm_alloc_failure(0)
            assert y.grad is None, nth
            assert np.array_equal(y.to_numpy(), values), nth
            assert not loss._graph_freed, nth
            # The same graph completes once the injector is disarmed —
            # nothing leaked, nothing double-closed.
            loss.backward()
            assert np.allclose(
                y.grad.to_numpy(),
                _analytic_grad(values, weights, -1), atol=1e-14)
        else:
            cpp._arm_alloc_failure(0)
            assert y.grad is not None
    assert hit >= 3, "expected several reachable backward allocation stages"
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK


# ======================================================================
# Raw ABI misuse
# ======================================================================


@needs_native
def test_native_softmax_abi_rejects_invalid_calls():
    """The export validates its own arguments, so a malformed direct
    ctypes call raises ValueError and leaves the destination unchanged."""
    library = cpp._require_library()
    source = cpp.NativeTensorCore.from_array(np.arange(1.0, 5.0))
    destination = cpp.NativeTensorCore.from_array(np.full(4, 7777.5))
    src_handle = source.storage._require_open()
    dst_handle = destination.storage._require_open()

    for outer, axis_length, inner, message in (
        (0, 2, 1, "must each be >= 1"),
        (2, 0, 1, "must each be >= 1"),
        (2, 2, 0, "must each be >= 1"),
        (-1, 2, 1, "must each be >= 1"),
        (2, 3, 1, "source span exceeds its storage"),
        (2, 2, 2, "source span exceeds its storage"),
    ):
        with pytest.raises(ValueError, match=message):
            library.tf_core_softmax_forward(src_handle, 0, dst_handle,
                                            outer, axis_length, inner)
    with pytest.raises(ValueError, match="negative source offset"):
        library.tf_core_softmax_forward(src_handle, -1, dst_handle, 2, 2, 1)
    with pytest.raises(ValueError, match="source span exceeds its storage"):
        library.tf_core_softmax_forward(src_handle, 2, dst_handle, 1, 4, 1)
    # A destination smaller than numel.
    small = cpp.NativeTensorCore.from_array(np.full(2, 7777.5))
    with pytest.raises(ValueError, match="destination storage smaller"):
        library.tf_core_softmax_forward(src_handle, 0,
                                        small.storage._require_open(), 2, 2, 1)
    # Nothing was written by any rejected call.
    assert np.array_equal(destination.to_numpy(), np.full(4, 7777.5))
    assert np.array_equal(small.to_numpy(), np.full(2, 7777.5))
    # The stale error does not survive the next valid call.
    assert np.allclose(source.softmax(0).to_numpy(),
                       softmax_reference(np.arange(1.0, 5.0), 0), atol=1e-15)
    assert library.tf_last_error_code() == cpp.TF_OK


def test_native_softmax_requires_the_built_backend():
    """Without the compiled library the operation raises ImportError with
    build instructions — never a silent NumPy fallback."""
    if cpp.is_available():
        pytest.skip("backend is built; the unavailable path cannot be forced")
    with pytest.raises(ImportError, match="cpp/build.py"):
        cpp.NativeTensorCore.from_array(VALUES).softmax(-1)


# ======================================================================
# Guardrails
# ======================================================================


def _numpy_tripwire(monkeypatch, names):
    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in names:
        monkeypatch.setattr(np, name, _tripwire)


# Everything NumPy could plausibly be used to *compute* a softmax with.
_NUMERICAL_NUMPY = (
    "max", "amax", "argmax", "exp", "log", "sum", "divide", "true_divide",
    "add", "subtract", "multiply", "matmul", "mean", "negative", "power",
    "copyto",
)
# Every route by which tensor *data* could enter or leave a NumPy host
# buffer: the two Core conversion methods plus the array constructors
# they would have to use. (np.asarray is deliberately absent: every
# native op marshals its shape/stride arrays through it because that is
# the ctypes calling convention — small layout metadata, never tensor
# data. See test_native_softmax_metadata_marshalling_is_not_data below,
# which pins that distinction.)
_DATA_NUMPY = ("empty", "array", "copy", "frombuffer", "zeros")


def _data_conversion_tripwire(monkeypatch):
    """Arm every tensor-data conversion route, including the Core-level
    ``to_numpy``/``from_array`` methods themselves."""
    def _tripwire(*args, **kwargs):
        raise AssertionError("tensor data was converted through NumPy")

    _numpy_tripwire(monkeypatch, _NUMERICAL_NUMPY + _DATA_NUMPY)
    monkeypatch.setattr(cpp.NativeTensorCore, "to_numpy", _tripwire)
    monkeypatch.setattr(cpp.NativeTensorCore, "from_array",
                        staticmethod(_tripwire))
    monkeypatch.setattr(cpp.NativeTensorView, "to_numpy", _tripwire)
    monkeypatch.setattr(cpp.NativeStorage, "from_array", staticmethod(_tripwire))
    monkeypatch.setattr(cpp.NativeStorage, "to_numpy", _tripwire)


@needs_native
def test_native_softmax_contiguous_path_and_backward_use_no_numpy(monkeypatch):
    """Strict tripwire for the contiguous path: no NumPy arithmetic and
    no tensor-data conversion in the Core forward, the wrapper forward,
    or the backward."""
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    core = cpp.NativeTensorCore.from_array(VALUES)
    upstream = NativeTensor.from_array(np.full(VALUES.shape, 1.5))

    _data_conversion_tripwire(monkeypatch)
    core.softmax(-1)                          # Core forward
    x.softmax(-1).backward(upstream)          # wrapper forward + backward
    monkeypatch.undo()
    assert np.allclose(x.grad.to_numpy(),
                       _analytic_grad(VALUES, np.full(VALUES.shape, 1.5), -1),
                       atol=1e-15)


@needs_native
def test_native_softmax_policy_b_path_uses_no_numpy(monkeypatch):
    """The **same strict** tripwire for the non-contiguous path.

    As of E3.1 the Policy-B copy is a native storage-to-storage gather
    (``tf_core_contiguous_copy``), so a strided softmax keeps tensor data
    in native memory end to end — there is no host round-trip left to
    exempt."""
    base = cpp.NativeTensorCore.from_array(VALUES)
    strided = base.T
    assert not strided.contiguous
    _data_conversion_tripwire(monkeypatch)
    out = strided.softmax(0)                  # Policy-B copy + kernel
    monkeypatch.undo()
    assert np.allclose(out.to_numpy(), softmax_reference(VALUES.T, 0),
                       atol=1e-15)
    # The same through the autograd layer, backward included.
    tracked = NativeTensor.from_array(VALUES, requires_grad=True)
    upstream_values = np.linspace(0.5, 2.0, VALUES.size).reshape(VALUES.T.shape)
    upstream = NativeTensor.from_array(upstream_values)
    _data_conversion_tripwire(monkeypatch)
    tracked.T.softmax(0).backward(upstream)
    monkeypatch.undo()
    assert np.allclose(tracked.grad.to_numpy(),
                       _analytic_grad(VALUES.T, upstream_values, 0).T,
                       atol=1e-14)


@needs_native
def test_native_contiguous_copy_uses_no_numpy(monkeypatch):
    """The shared Policy-B helper itself: a strided, offset copy must
    move tensor data natively, storage to storage."""
    base = cpp.NativeTensorCore.from_array(CUBE)
    for source in (base.T, base.narrow(1, 1, 2), base.narrow(1, 1, 2).T,
                   base):
        _data_conversion_tripwire(monkeypatch)
        copy = source.contiguous_copy()
        monkeypatch.undo()
        assert copy.contiguous and copy.offset == 0
        copy.close()


@needs_native
def test_native_softmax_metadata_marshalling_is_not_data(monkeypatch):
    """The deliberate boundary: ``np.asarray`` *is* used, but only to
    marshal shape/stride arrays for ctypes. This test pins that
    distinction so the tripwire above cannot be quietly widened into a
    false claim — every asarray call the softmax path makes receives a
    small tuple of ints, never tensor values."""
    seen = []
    original = np.asarray

    def _recording_asarray(values, *args, **kwargs):
        seen.append(values)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", _recording_asarray)
    base = cpp.NativeTensorCore.from_array(CUBE)
    base.T.softmax(0)                          # Policy-B copy + kernel
    monkeypatch.undo()
    assert seen, "the softmax path marshals layout metadata through asarray"
    for value in seen:
        assert isinstance(value, tuple), value
        assert all(isinstance(extent, int) for extent in value), value
        # Shape/stride tuples are rank-sized, never element-sized.
        assert len(value) <= CUBE.ndim, value


@needs_native
def test_native_softmax_scope_boundaries_hold():
    """E3 is softmax only: no later Phase-E surface, no public max/argmax
    or division, no module, and the stable framework is untouched.
    (`log_softmax` arrived separately in E4 as its own fused kernel, so
    it is no longer listed as absent here.)"""
    x = NativeTensor.from_array(VALUES)
    core = cpp.NativeTensorCore.from_array(VALUES)
    for absent in ("cross_entropy", "max", "argmax", "amax",
                   "divide", "sigmoid", "tanh"):
        assert not hasattr(x, absent), absent
        assert not hasattr(core, absent), absent
    assert not hasattr(x, "__truediv__")
    import tensorforge.experimental as experimental
    for absent in ("NativeSoftmax", "NativeCrossEntropyLoss",
                   "native_accuracy"):
        assert not hasattr(experimental, absent), absent
    assert "NativeSoftmax" not in cpp.NATIVE_MODULES
    # The stable Tensor keeps its own softmax, entirely separately.
    stable = tensorforge.Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
    stable.softmax().sum().backward()
    assert type(stable.softmax()) is tensorforge.Tensor
    # No implicit dispatch in either direction.
    with pytest.raises((TypeError, AttributeError)):
        NativeTensor.from_array(VALUES).multiply(tensorforge.Tensor(VALUES))


# ======================================================================
# E3.1 — the shared native Policy-B contiguous copy
# ======================================================================


@needs_native
@pytest.mark.parametrize("build,expected", [
    (lambda base: base.T, lambda v: v.T),
    (lambda base: base.narrow(1, 1, 2), lambda v: v[:, 1:3]),
    (lambda base: base.narrow(1, 1, 2).T, lambda v: v[:, 1:3].T),
    (lambda base: base, lambda v: v),
])
def test_native_contiguous_copy_matches_the_logical_view(build, expected):
    values = np.arange(12.0).reshape(3, 4)
    base = cpp.NativeTensorCore.from_array(values)
    source = build(base)
    copy = source.contiguous_copy()
    assert np.array_equal(copy.to_numpy(), expected(values))
    # Fresh, owning, contiguous, offset zero, non-aliasing.
    assert copy.contiguous and copy.offset == 0
    assert copy.shape == source.shape
    assert copy.storage is not base.storage
    assert copy.dtype == "float64" and copy.device == "cpu"
    # The source is untouched, and the copy is independent of it.
    assert np.array_equal(base.to_numpy(), values)
    base.close()
    assert np.array_equal(copy.to_numpy(), expected(values))
    copy.close()


@needs_native
def test_native_contiguous_copy_of_a_rank_zero_core():
    rank_zero = cpp.NativeTensorCore.from_array(np.arange(6.0)).sum()
    assert rank_zero.shape == ()
    copy = rank_zero.contiguous_copy()
    assert copy.shape == () and copy.contiguous and copy.offset == 0
    assert copy.to_numpy().item() == 15.0


@needs_native
def test_native_contiguous_copy_rejects_a_closed_input():
    core = cpp.NativeTensorCore.from_array(VALUES)
    core.close()
    with pytest.raises(RuntimeError, match="closed"):
        core.contiguous_copy()


@needs_fault_injection
def test_native_contiguous_copy_allocation_failure_is_atomic():
    """A failed destination allocation leaves the source untouched and
    nothing half-built; a failed gather closes the destination."""
    values = np.arange(12.0).reshape(3, 4)
    base = cpp.NativeTensorCore.from_array(values)
    strided = base.T
    hit = 0
    for nth in range(1, 4):
        try:
            cpp._arm_alloc_failure(nth)
            copy = strided.contiguous_copy()
        except MemoryError:
            hit += 1
            cpp._arm_alloc_failure(0)
            # The source is intact and immediately reusable.
            assert np.array_equal(base.to_numpy(), values), nth
        else:
            cpp._arm_alloc_failure(0)
            assert np.array_equal(copy.to_numpy(), values.T)
            copy.close()
    assert hit >= 1, "the sweep never triggered an allocation failure"
    # The backend recovered: no stale error, and the copy still works.
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    recovered = strided.contiguous_copy()
    assert np.array_equal(recovered.to_numpy(), values.T)
    recovered.close()
    base.close()


@needs_native
def test_native_contiguous_copy_abi_rejects_invalid_calls():
    """The new export self-validates: malformed direct ctypes calls raise
    ValueError and leave the destination byte-for-byte unchanged."""
    library = cpp._require_library()
    source = cpp.NativeTensorCore.from_array(np.arange(4.0))
    destination = cpp.NativeTensorCore.from_array(np.full(4, 7777.5))
    src_handle = source.storage._require_open()
    dst_handle = destination.storage._require_open()
    shape = np.asarray([4], dtype=np.int64)
    strides = np.asarray([1], dtype=np.int64)

    with pytest.raises(ValueError, match="span exceeds its storage"):
        library.tf_core_contiguous_copy(src_handle, dst_handle, shape,
                                        np.asarray([2], dtype=np.int64), 0, 1)
    with pytest.raises(ValueError, match="negative offset"):
        library.tf_core_contiguous_copy(src_handle, dst_handle, shape,
                                        strides, -1, 1)
    with pytest.raises(ValueError, match="non-positive dimension"):
        library.tf_core_contiguous_copy(src_handle, dst_handle,
                                        np.asarray([0], dtype=np.int64),
                                        strides, 0, 1)
    small = cpp.NativeTensorCore.from_array(np.full(2, 7777.5))
    with pytest.raises(ValueError, match="output storage smaller"):
        library.tf_core_contiguous_copy(src_handle,
                                        small.storage._require_open(),
                                        shape, strides, 0, 1)
    # Nothing was written by any rejected call.
    assert np.array_equal(destination.to_numpy(), np.full(4, 7777.5))
    assert np.array_equal(small.to_numpy(), np.full(2, 7777.5))
    # A valid call afterwards succeeds and clears the stale error state.
    assert np.array_equal(source.contiguous_copy().to_numpy(),
                          np.arange(4.0))
    assert library.tf_last_error_code() == cpp.TF_OK


@needs_native
def test_shared_policy_b_users_still_work_on_non_contiguous_inputs():
    """contiguous_copy is shared infrastructure: the Phase-D Policy-B
    consumers must be unaffected by the E3.1 native rewrite."""
    rng = np.random.default_rng(4)
    # conv2d: a non-contiguous NCHW input and OIHW weight.
    x = rng.standard_normal((1, 2, 4, 4))
    w = rng.standard_normal((2, 2, 2, 2))
    x_base = cpp.NativeTensorCore.from_array(
        np.ascontiguousarray(x.transpose(0, 1, 3, 2)))
    w_base = cpp.NativeTensorCore.from_array(
        np.ascontiguousarray(w.transpose(0, 1, 3, 2)))
    x_view = x_base.transpose((0, 1, 3, 2))
    w_view = w_base.transpose((0, 1, 3, 2))
    assert not x_view.contiguous and not w_view.contiguous
    strided_conv = x_view.conv2d_forward(w_view)
    direct_conv = cpp.NativeTensorCore.from_array(x).conv2d_forward(
        cpp.NativeTensorCore.from_array(w))
    assert np.allclose(strided_conv.to_numpy(), direct_conv.to_numpy(),
                       atol=1e-12)
    # maxpool2d: the same input through the pooling Policy-B path.
    strided_pool = x_view.maxpool2d_forward(kernel_size=2)
    direct_pool = cpp.NativeTensorCore.from_array(x).maxpool2d_forward(
        kernel_size=2)
    assert np.allclose(strided_pool.to_numpy(), direct_pool.to_numpy(),
                       atol=1e-12)
    # softmax: the same guarantee for the E3 path.
    strided_softmax = x_view.softmax(1)
    assert np.allclose(strided_softmax.to_numpy(),
                       softmax_reference(x, 1), atol=1e-14)
    # Every input is unchanged and still usable.
    assert np.allclose(x_view.to_numpy(), x, atol=1e-15)
    assert np.allclose(w_view.to_numpy(), w, atol=1e-15)


@needs_native
def test_native_softmax_checkpoint_schema_is_untouched():
    """E3 adds no persistent state: the native checkpoint format version
    is still 1 (docs/native_classification_design.md §12)."""
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT_VERSION == 1
