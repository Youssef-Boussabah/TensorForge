"""Tests for the fused native log-softmax — Phase E, milestone E4.

The classification stack's second probability transform
(docs/native_classification_design.md §4.4). One axis-wise fused
operation through the complete native stack: the internal
`tf::log_softmax_forward_contiguous` kernel in the Phase-E
classification unit → the guarded, self-validating, **contiguous-only**
C ABI (`tf_core_log_softmax_forward`) → ctypes →
`NativeTensorCore.log_softmax(axis=-1)` with Policy-B copy-then-compute
→ differentiable `NativeTensor.log_softmax(axis=-1)`.

Forward is a **fused maximum shift + log-sum-exp**: per slice
`(x - max(x)) - log(sum(exp(x - max(x))))`, computed inside the kernel
in float64. It is **never** `softmax(x).log()` — that composition is
exactly the precision loss the operation exists to avoid, and a test
below pins the distinction numerically as well as structurally.
Backward is the closed-form Jacobian-vector product

    dx = upstream - exp(y) * sum(upstream, axis, keepdims=True)

**composed at the graph-unaware Core layer** — there is no dedicated
log-softmax backward kernel — and it reads only the **saved output** `y`,
so the node records no expected parameter version (the `exp`/`softmax`
archetype, deliberately unlike `log`).

Exceptional values follow plain IEEE arithmetic with no special-casing:
a NaN or `+inf` in a slice propagates through the shift and sum, making
that slice NaN, while `-inf` gets `-inf` and leaves its finite
neighbours governed by the stable computation. The reference below is
the *same* maximum-shift algorithm in NumPy rather than another
framework's infinity handling.

NumPy appears only as an external oracle and for inspection; a tripwire
test proves the forward/backward paths never compute with it.

Selector: python -m pytest -q -k "native_log_softmax"
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


def log_softmax_reference(x, axis):
    """The same maximum-shift / log-sum-exp algorithm the kernel
    implements, in NumPy.

    Deliberately not another framework's log_softmax, and deliberately
    not ``log(softmax(x))``: the contract is that the fused kernel agrees
    with *this* algorithm, including at IEEE edges where implementations
    differ."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        shifted = x - np.max(x, axis=axis, keepdims=True)
        return shifted - np.log(
            np.sum(np.exp(shifted), axis=axis, keepdims=True)
        )


def softmax_reference(x, axis):
    """The E3 maximum-shift softmax, for the cross-check that
    ``exp(log_softmax(x))`` agrees with ``softmax(x)``."""
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


def _analytic_grad(x, upstream, axis):
    """dx = g - exp(y) * sum(g, axis, keepdims=True)."""
    y = log_softmax_reference(x, axis)
    return upstream - np.exp(y) * np.sum(upstream, axis=axis, keepdims=True)


# ======================================================================
# Kernel symbols, source unit, and the capability boundary
# ======================================================================


@needs_native
def test_native_log_softmax_kernel_symbol_is_bound():
    library = cpp._require_library()
    assert library.tf_core_log_softmax_forward is not None
    assert "tf_core_log_softmax_forward" in cpp._CHECKED_KERNELS
    # An ABI symbol is not a capability name.
    assert "tf_core_log_softmax_forward" not in cpp.RAW_KERNELS
    assert "tf_core_log_softmax_forward" not in cpp.TENSOR_CORE_OPS
    # E4 added no raw NumPy-buffer kernel and left the frozen registry.
    assert cpp.TENSOR_CORE_KERNELS == ("relu", "add", "subtract",
                                       "multiply", "matmul")
    assert "log_softmax" not in cpp.RAW_KERNELS
    assert not hasattr(cpp, "log_softmax")
    # E4 deliberately ships **no** backward kernel: the gradient is
    # composed from existing Core ops.
    assert not any("log_softmax" in name and "backward" in name
                   for name in cpp._CHECKED_KERNELS)
    assert not hasattr(library, "tf_core_log_softmax_backward")


def test_native_log_softmax_lives_in_the_classification_source_unit():
    """The E4 kernel and export belong to the Phase-E classification
    unit locked by E0 §9.1 — not to the elementwise one."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    classification_path = root / "cpp" / "src" / "classification.cpp"
    assert classification_path.is_file()
    assert (root / "cpp" / "tests" / "test_log_softmax.cpp").is_file()
    classification = classification_path.read_text(encoding="utf-8")
    elementwise = (root / "cpp" / "src" / "elementwise.cpp").read_text(
        encoding="utf-8"
    )
    header = (root / "cpp" / "include" / "tf_classification_internal.h"
              ).read_text(encoding="utf-8")
    assert "tf_core_log_softmax_forward" in classification
    assert "log_softmax_forward_contiguous" in classification
    assert "log_softmax_forward_contiguous" in header
    for symbol in ("tf_core_log_softmax_forward(",
                   "log_softmax_forward_contiguous("):
        assert symbol not in elementwise, symbol
    # No backward export exists at the C ABI, in any spelling.
    assert "tf_core_log_softmax_backward" not in classification
    # The CTest is registered as its own target.
    cmake = (root / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "test_log_softmax" in cmake
    assert "add_test(NAME log_softmax" in cmake


def test_native_log_softmax_forward_is_fused_not_softmax_then_log():
    """The structural half of the "never softmax().log()" contract: the
    Core forward calls exactly one native classification export, and it
    is the log-softmax one. (The numerical half is pinned separately, in
    the small-probability regime test below.)"""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    # Phase I milestone I6 made the four classification kernels templates
    # over the element type, so their definitions moved from
    # cpp/src/classification.cpp into the internal header beside it — a
    # template has to be visible where it is instantiated. The kernel is the
    # same kernel and this contract is unchanged; only where the source
    # lives moved.
    classification = (
        root / "cpp" / "include" / "tf_classification_internal.h"
    ).read_text(encoding="utf-8")
    # The internal kernel accumulates exponentials and takes ONE
    # logarithm of the sum; it never forms or divides by a probability.
    body = classification.split(
        "inline void log_softmax_forward_contiguous(", 1)[1]
    body = body.split("\n}\n", 1)[0]
    assert "std::exp(" in body and "std::log(" in body
    assert "/=" not in body, "the fused kernel must not divide"
    assert "softmax_forward_contiguous(" not in body.replace(
        "log_softmax_forward_contiguous(", ""
    ), "the fused kernel must not call the softmax kernel"


@needs_native
def test_native_log_softmax_core_forward_calls_only_its_own_kernel(monkeypatch):
    """Dynamic proof: a Core log_softmax must not route through the
    softmax export (nor through the Core-level exp/log/divide ops)."""
    library = cpp._require_library()
    calls = []

    def _forbidden(name):
        def _tripwire(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"log_softmax routed through {name}")
        return _tripwire

    monkeypatch.setattr(library, "tf_core_softmax_forward",
                        _forbidden("tf_core_softmax_forward"))
    for op in ("exp", "log", "softmax"):
        monkeypatch.setattr(cpp.NativeTensorCore, op, _forbidden(op))
    out = cpp.NativeTensorCore.from_array(VALUES).log_softmax(-1)
    monkeypatch.undo()
    assert calls == []
    assert np.allclose(out.to_numpy(), log_softmax_reference(VALUES, -1),
                       atol=1e-15)


def test_native_log_softmax_registry_placement():
    """log_softmax is a Core op and an autograd op — nothing else. Runs
    without the compiled backend: these are pure inventory facts."""
    assert "log_softmax" in cpp.TENSOR_CORE_OPS
    assert "log_softmax" in cpp.AUTOGRAD_OPS
    assert "log_softmax" not in cpp.UNSUPPORTED
    assert "log_softmax" not in cpp.NATIVE_MODULES
    assert "log_softmax" not in cpp.NATIVE_LOSSES
    assert "log_softmax" not in cpp.TENSOR_CORE_KERNELS
    assert "log_softmax" not in cpp.NATIVE_METRICS
    # E1/E2/E3 stay implemented alongside it.
    for shipped in ("exp", "log", "softmax"):
        assert shipped in cpp.TENSOR_CORE_OPS and shipped in cpp.AUTOGRAD_OPS
        assert shipped not in cpp.UNSUPPORTED
    info = cpp.backend_info()
    assert "log_softmax" in info["tensor_core_ops"]
    assert "log_softmax" in info["autograd_ops"]
    assert "log_softmax" not in info["unsupported"]
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
    # E7 shipped the public classification surface, each name into the
    # one inventory that describes its layer — never into an operation
    # inventory.
    assert "NativeCrossEntropyLoss" in cpp.NATIVE_LOSSES
    assert "native_accuracy" in cpp.NATIVE_METRICS
    for shipped in ("NativeCrossEntropyLoss", "native_accuracy"):
        assert shipped not in cpp.UNSUPPORTED, shipped
        assert shipped not in cpp.TENSOR_CORE_OPS
        assert shipped not in cpp.AUTOGRAD_OPS


# ======================================================================
# Forward correctness
# ======================================================================


@needs_native
def test_native_log_softmax_forward_rank1():
    values = np.array([1.0, 2.0, 3.0])
    for axis in (0, -1):
        out = (cpp.NativeTensorCore.from_array(values)
               .log_softmax(axis).to_numpy())
        assert np.allclose(out, log_softmax_reference(values, axis),
                           atol=1e-15)
        assert math.isclose(float(np.exp(out).sum()), 1.0, rel_tol=1e-12)
        assert np.all(out <= 0.0)


@needs_native
def test_native_log_softmax_forward_rank2_every_axis():
    for axis in (0, 1, -1, -2):
        out = (cpp.NativeTensorCore.from_array(VALUES)
               .log_softmax(axis).to_numpy())
        assert np.allclose(out, log_softmax_reference(VALUES, axis),
                           atol=1e-15)
        assert np.allclose(np.exp(out).sum(axis=axis), 1.0, atol=1e-12)
        # Log-probabilities are non-positive up to tiny rounding.
        assert np.all(out <= 1e-15)


@needs_native
def test_native_log_softmax_forward_rank3_first_middle_last():
    for axis in (0, 1, 2):
        out = (cpp.NativeTensorCore.from_array(CUBE)
               .log_softmax(axis).to_numpy())
        assert np.allclose(out, log_softmax_reference(CUBE, axis), atol=1e-15)
        assert np.allclose(np.exp(out).sum(axis=axis), 1.0, atol=1e-12)
    # A rank-4 tensor exercises a middle axis with both outer and inner > 1.
    quad = np.linspace(-2.0, 2.0, 2 * 3 * 4 * 2).reshape(2, 3, 4, 2)
    for axis in (1, 2):
        out = (cpp.NativeTensorCore.from_array(quad)
               .log_softmax(axis).to_numpy())
        assert np.allclose(out, log_softmax_reference(quad, axis), atol=1e-15)


@needs_native
def test_native_log_softmax_negative_axes_match_positive():
    for negative, positive in ((-1, 2), (-2, 1), (-3, 0)):
        left = (cpp.NativeTensorCore.from_array(CUBE)
                .log_softmax(negative).to_numpy())
        right = (cpp.NativeTensorCore.from_array(CUBE)
                 .log_softmax(positive).to_numpy())
        assert np.array_equal(left, right)


@needs_native
def test_native_log_softmax_axis_length_one_is_exactly_zero():
    """A slice with one member has probability 1, so log(1) == 0 —
    exactly, since the shift makes it 0 - log(exp(0))."""
    values = np.array([[5.0], [-3.0], [100.0]])
    out = cpp.NativeTensorCore.from_array(values).log_softmax(1).to_numpy()
    assert np.array_equal(out, np.zeros_like(values))


@needs_native
def test_native_log_softmax_equal_values_give_minus_log_axis_length():
    values = np.full((3, 4), 2.75)
    out = cpp.NativeTensorCore.from_array(values).log_softmax(-1).to_numpy()
    assert np.array_equal(out, np.full((3, 4), -math.log(4.0)))
    # And along the other axis, where the length differs.
    other = cpp.NativeTensorCore.from_array(values).log_softmax(0).to_numpy()
    assert np.array_equal(other, np.full((3, 4), -math.log(3.0)))


@needs_native
def test_native_log_softmax_random_values():
    rng = np.random.default_rng(20260721)
    values = rng.uniform(-8.0, 8.0, size=(5, 7, 3))
    for axis in (0, 1, 2, -1):
        out = (cpp.NativeTensorCore.from_array(values)
               .log_softmax(axis).to_numpy())
        assert np.allclose(out, log_softmax_reference(values, axis),
                           rtol=1e-13, atol=1e-14)
        assert np.allclose(np.exp(out).sum(axis=axis), 1.0, atol=1e-12)


@needs_native
def test_native_log_softmax_is_stable_under_large_offsets():
    """A naive log(sum(exp(x))) gives +inf at +800 and -inf at -800; the
    fused maximum shift stays exact."""
    base = np.array([0.0, 1.0, 2.0, 3.0])
    expected = log_softmax_reference(base, -1)
    for offset in (800.0, -800.0, 1e5, -1e5, 1e10, -1e10):
        shifted = base + offset
        out = (cpp.NativeTensorCore.from_array(shifted)
               .log_softmax(-1).to_numpy())
        assert np.all(np.isfinite(out)), offset
        assert np.allclose(out, expected, atol=1e-6), offset
        assert math.isclose(float(np.exp(out).sum()), 1.0, rel_tol=1e-12)
    # Sanity: the naive formulation really would have failed here.
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.log(np.sum(np.exp(base + 800.0)))
        assert not np.isfinite(naive)


@needs_native
def test_native_log_softmax_beats_the_composed_form_on_small_probabilities():
    """The numerical half of "never softmax().log()". A logit 800 below
    the maximum has probability exp(-800), which underflows float64 to
    exactly 0 — so ``softmax(x).log()`` reports -inf. The fused
    log-sum-exp form reports the accurate finite value."""
    values = np.array([0.0, -800.0])
    out = cpp.NativeTensorCore.from_array(values).log_softmax(-1).to_numpy()
    assert np.all(np.isfinite(out))
    assert np.allclose(out, [0.0, -800.0], atol=1e-9)
    # What the forbidden composition would have produced, through the
    # native softmax and log ops themselves.
    composed = (cpp.NativeTensorCore.from_array(values)
                .softmax(-1).log().to_numpy())
    assert composed[1] == -np.inf
    assert np.isfinite(out[1]), "the fused form must not degrade to -inf"


@needs_native
def test_native_log_softmax_additive_shift_invariance_per_slice():
    """Adding a constant to one slice leaves that slice's log-softmax
    unchanged (and does not disturb any other slice)."""
    values = VALUES.copy()
    reference = (cpp.NativeTensorCore.from_array(values)
                 .log_softmax(-1).to_numpy())
    shifted = values.copy()
    shifted[0] += 37.5           # only the first slice moves
    out = cpp.NativeTensorCore.from_array(shifted).log_softmax(-1).to_numpy()
    assert np.allclose(out, reference, atol=1e-13)


@needs_native
def test_native_log_softmax_agrees_with_softmax():
    """exp(log_softmax(x)) == softmax(x) to tolerance. Bit equality is
    NOT claimed: the two kernels do different arithmetic."""
    rng = np.random.default_rng(99)
    values = rng.uniform(-6.0, 6.0, size=(4, 5))
    for axis in (0, 1, -1):
        logs = (cpp.NativeTensorCore.from_array(values)
                .log_softmax(axis).to_numpy())
        probabilities = (cpp.NativeTensorCore.from_array(values)
                         .softmax(axis).to_numpy())
        assert np.allclose(np.exp(logs), probabilities, atol=1e-15)
        assert np.allclose(np.exp(logs), softmax_reference(values, axis),
                           atol=1e-15)


@needs_native
def test_native_log_softmax_forward_is_deterministic():
    core = cpp.NativeTensorCore.from_array(CUBE)
    first = core.log_softmax(1).to_numpy()
    second = core.log_softmax(1).to_numpy()
    assert np.array_equal(first, second)


# ======================================================================
# Exceptional values (plain IEEE, matching the implemented algorithm)
# ======================================================================


@needs_native
def test_native_log_softmax_exceptional_values_follow_the_max_shift_algorithm():
    values = np.array([
        [1.0, float("nan"), 2.0],          # NaN poisons its slice
        [float("inf"), 1.0, 2.0],          # inf - inf -> NaN
        [float("-inf"), 1.0, 2.0],         # -inf keeps -inf, rest is stable
    ])
    out = cpp.NativeTensorCore.from_array(values).log_softmax(-1).to_numpy()
    expected = log_softmax_reference(values, -1)
    assert np.array_equal(out, expected, equal_nan=True)
    assert np.all(np.isnan(out[0])), "a NaN makes its whole slice NaN"
    assert np.all(np.isnan(out[1])), "+inf makes its whole slice NaN"
    assert out[2][0] == -np.inf, "-inf keeps a -inf log-probability"
    assert np.all(np.isfinite(out[2][1:]))
    # The finite members are exactly the stable result over the finite
    # sub-slice: -inf contributes exp(-inf) == 0 to the sum.
    assert np.allclose(out[2][1:],
                       log_softmax_reference(np.array([1.0, 2.0]), -1),
                       atol=1e-15)
    assert math.isclose(float(np.exp(out[2][1:]).sum()), 1.0, rel_tol=1e-12)
    # A NaN is never silently turned into a finite log-probability.
    assert not np.any(np.isfinite(out[0]))
    # An all -inf slice is NaN (-inf - -inf).
    all_neg = cpp.NativeTensorCore.from_array(
        np.array([float("-inf"), float("-inf")])
    ).log_softmax(-1).to_numpy()
    assert np.all(np.isnan(all_neg))


@needs_native
def test_native_log_softmax_neighbouring_slices_are_independent():
    values = np.array([[1.0, float("nan")], [1.0, 2.0]])
    out = cpp.NativeTensorCore.from_array(values).log_softmax(-1).to_numpy()
    assert np.all(np.isnan(out[0]))
    assert np.allclose(out[1], log_softmax_reference(values[1], -1),
                       atol=1e-15)


@needs_native
def test_native_log_softmax_exceptional_values_are_not_abi_errors():
    """A numerically exceptional but structurally valid call must not
    raise and must leave the native error slot clear."""
    values = np.array([float("nan"), float("inf"), float("-inf")])
    out = cpp.NativeTensorCore.from_array(values).log_softmax(0).to_numpy()
    assert out.shape == (3,)
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    # And through the autograd layer too.
    tracked = NativeTensor.from_array(values, requires_grad=True)
    tracked.log_softmax(0).sum().backward()
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK


# ======================================================================
# Axis validation
# ======================================================================


@needs_native
@pytest.mark.parametrize("bad_axis", [True, False, 1.0, 0.5, "0", None,
                                      (0,), [0]])
def test_native_log_softmax_rejects_non_integer_axis(bad_axis):
    core = cpp.NativeTensorCore.from_array(VALUES)
    with pytest.raises(TypeError, match="axis"):
        core.log_softmax(bad_axis)
    tracked = NativeTensor.from_array(VALUES, requires_grad=True)
    with pytest.raises(TypeError, match="axis"):
        tracked.log_softmax(bad_axis)
    # Nothing was consumed: a valid call still works.
    assert core.log_softmax(-1).shape == VALUES.shape


@needs_native
@pytest.mark.parametrize("bad_axis", [2, 3, 99, -3, -4, -99])
def test_native_log_softmax_rejects_out_of_range_axis(bad_axis):
    core = cpp.NativeTensorCore.from_array(VALUES)   # rank 2
    with pytest.raises(ValueError, match="out of bounds"):
        core.log_softmax(bad_axis)
    tracked = NativeTensor.from_array(VALUES, requires_grad=True)
    with pytest.raises(ValueError, match="out of bounds"):
        tracked.log_softmax(bad_axis)


@needs_native
def test_native_log_softmax_rejects_rank_zero_input():
    """log_softmax needs an axis to normalize over, so rank 0 is
    rejected — every integer axis is out of bounds on a scalar."""
    rank_zero = cpp.NativeTensorCore.from_array(VALUES).sum()
    assert rank_zero.shape == ()
    for axis in (0, -1):
        with pytest.raises(ValueError, match="out of bounds"):
            rank_zero.log_softmax(axis)
    tracked = NativeTensor.from_array(VALUES, requires_grad=True).sum()
    with pytest.raises(ValueError, match="out of bounds"):
        tracked.log_softmax(-1)


@needs_native
def test_native_log_softmax_axis_validation_precedes_allocation(monkeypatch):
    """A rejected axis must allocate nothing — the validation runs before
    the output (and before any Policy-B copy)."""
    core = cpp.NativeTensorCore.from_array(VALUES)

    def _fail(*args, **kwargs):
        raise AssertionError("allocation happened before axis validation")

    monkeypatch.setattr(cpp.NativeTensorCore, "zeros",
                        staticmethod(_fail))
    # H1: the enabled output-allocation sites construct through
    # _uninitialized, so the same probe must watch both
    # constructors for this test to still observe the real path.
    monkeypatch.setattr(cpp.NativeTensorCore, "_uninitialized",
                        staticmethod(_fail))
    monkeypatch.setattr(cpp.NativeTensorCore, "contiguous_copy", _fail)
    with pytest.raises(ValueError, match="out of bounds"):
        core.log_softmax(5)
    with pytest.raises(TypeError, match="axis"):
        core.log_softmax(True)


@needs_native
def test_native_log_softmax_rejects_closed_input():
    core = cpp.NativeTensorCore.from_array(VALUES)
    core.close()
    with pytest.raises(RuntimeError, match="closed"):
        core.log_softmax(-1)
    tracked = NativeTensor.from_array(VALUES)
    tracked.close()
    with pytest.raises(RuntimeError, match="closed"):
        tracked.log_softmax(-1)


# ======================================================================
# Policy-B, layout, and ownership
# ======================================================================


@needs_native
def test_native_log_softmax_output_is_fresh_owning_contiguous():
    core = cpp.NativeTensorCore.from_array(VALUES)
    out = core.log_softmax(-1)
    assert out.shape == core.shape
    assert out.contiguous and out.offset == 0
    assert out.dtype == "float64" and out.device == "cpu"
    assert out.storage is not core.storage
    assert np.array_equal(core.to_numpy(), VALUES)  # input unmutated


@needs_native
def test_native_log_softmax_handles_non_contiguous_inputs():
    core = cpp.NativeTensorCore.from_array(CUBE)
    transposed = core.transpose((2, 1, 0))
    assert not transposed.contiguous
    out = transposed.log_softmax(-1)
    assert np.allclose(out.to_numpy(),
                       log_softmax_reference(CUBE.transpose(2, 1, 0), -1),
                       atol=1e-15)
    assert out.contiguous and out.offset == 0
    # A narrowed view with a nonzero offset. (Views borrow their base's
    # storage, so the base must stay referenced for the view to be
    # usable — the existing ownership rule.)
    base = cpp.NativeTensorCore.from_array(VALUES)
    narrowed = base.narrow(1, 1, 2)
    assert narrowed.offset != 0
    assert np.allclose(narrowed.log_softmax(-1).to_numpy(),
                       log_softmax_reference(VALUES[:, 1:3], -1), atol=1e-15)
    # ...and the transpose of a narrow (non-contiguous *and* offset).
    combined = base.narrow(1, 1, 2).T
    assert not combined.contiguous and combined.offset != 0
    assert np.allclose(combined.log_softmax(0).to_numpy(),
                       log_softmax_reference(VALUES[:, 1:3].T, 0), atol=1e-15)
    # A reshaped (still contiguous) view takes the direct path.
    cube_base = cpp.NativeTensorCore.from_array(CUBE)
    reshaped = cube_base.reshape((6, 4))
    assert reshaped.contiguous
    assert np.allclose(reshaped.log_softmax(-1).to_numpy(),
                       log_softmax_reference(CUBE.reshape(6, 4), -1),
                       atol=1e-15)
    assert np.array_equal(core.to_numpy(), CUBE)   # never mutated
    assert np.array_equal(base.to_numpy(), VALUES)


@needs_native
def test_native_log_softmax_policy_b_temporary_is_closed_on_success(monkeypatch):
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
    out = strided.log_softmax(0)
    assert len(made) == 1, "a non-contiguous input must be copied once"
    assert made[0]._closed, "the Policy-B temporary was not closed"
    assert not out._closed and out.contiguous
    # A contiguous input makes no copy at all.
    made.clear()
    contiguous = cpp.NativeTensorCore.from_array(VALUES)
    contiguous.log_softmax(-1)
    assert made == []


@needs_fault_injection
def test_native_log_softmax_policy_b_temporary_is_closed_on_failure(monkeypatch):
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
            strided.log_softmax(0)
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
    assert np.allclose(strided.log_softmax(0).to_numpy(),
                       log_softmax_reference(VALUES.T, 0), atol=1e-15)


@needs_native
def test_native_log_softmax_output_is_closed_if_the_kernel_fails(monkeypatch):
    """A native call that raises after the output was allocated must
    discard that output rather than returning a half-built tensor."""
    library = cpp._require_library()
    allocated = []
    original_zeros = cpp.NativeTensorCore.zeros

    def _recording_zeros(shape, **kwargs):
        core = original_zeros(shape, **kwargs)
        allocated.append(core)
        return core

    def _failing_kernel(*args, **kwargs):
        raise RuntimeError("simulated native log_softmax failure")

    monkeypatch.setattr(cpp.NativeTensorCore, "zeros",
                        staticmethod(_recording_zeros))
    # H1: the enabled output-allocation sites construct through
    # _uninitialized, so the same probe must watch both
    # constructors for this test to still observe the real path.
    monkeypatch.setattr(cpp.NativeTensorCore, "_uninitialized",
                        staticmethod(_recording_zeros))
    monkeypatch.setattr(library, "tf_core_log_softmax_forward",
                        _failing_kernel)
    core = cpp.NativeTensorCore.from_array(VALUES)
    with pytest.raises(RuntimeError, match="simulated"):
        core.log_softmax(-1)
    monkeypatch.undo()
    assert allocated, "the forward allocated an output before calling"
    assert all(out._closed for out in allocated), (
        "the freshly allocated output leaked when the kernel failed"
    )
    # The input is untouched and the operation still works.
    assert np.array_equal(core.to_numpy(), VALUES)
    assert np.allclose(core.log_softmax(-1).to_numpy(),
                       log_softmax_reference(VALUES, -1), atol=1e-15)


# ======================================================================
# NativeTensor forward and graph construction
# ======================================================================


@needs_native
def test_native_log_softmax_wrapper_forward_and_graph_construction():
    plain = NativeTensor.from_array(VALUES)
    result = plain.log_softmax(-1)
    assert np.allclose(result.to_numpy(), log_softmax_reference(VALUES, -1),
                       atol=1e-15)
    assert result._op == "" and result._parents == ()
    assert result.is_leaf and not result.requires_grad
    assert result._graph_resources == ()
    result.close()
    assert not plain.closed

    tracked = NativeTensor.from_array(VALUES, requires_grad=True)
    node = tracked.log_softmax(-1)
    assert node._op == "log_softmax" and node._parents == (tracked,)
    assert node.requires_grad and not node.is_leaf
    # Saved-output backward: no version snapshot, no private resource
    # (y is the node's own core).
    assert node._expected_versions == ()
    assert node._graph_resources == ()

    parameter = NativeParameter(VALUES)
    assert type(parameter.log_softmax(-1)) is NativeTensor


# ======================================================================
# Backward correctness
# ======================================================================


@needs_native
@pytest.mark.parametrize("shape,axis", [
    ((5,), 0), ((5,), -1),
    ((2, 3), 0), ((2, 3), 1), ((2, 3), -1), ((2, 3), -2),
    ((2, 3, 4), 0), ((2, 3, 4), 1), ((2, 3, 4), 2),
    ((2, 3, 4), -1), ((2, 3, 4), -2), ((2, 3, 4), -3),
])
def test_native_log_softmax_backward_matches_the_analytic_jacobian(shape, axis):
    rng = np.random.default_rng(7)
    values = rng.uniform(-3.0, 3.0, size=shape)
    upstream = rng.uniform(-2.0, 2.0, size=shape)
    x = NativeTensor.from_array(values, requires_grad=True)
    x.log_softmax(axis).backward(NativeTensor.from_array(upstream))
    assert np.allclose(x.grad.to_numpy(),
                       _analytic_grad(values, upstream, axis), atol=1e-14)


@needs_native
def test_native_log_softmax_backward_negative_axis_equivalence():
    rng = np.random.default_rng(11)
    values = rng.uniform(-2.0, 2.0, size=(2, 3, 4))
    upstream = rng.uniform(-1.0, 1.0, size=(2, 3, 4))
    grads = []
    for axis in (1, -2):
        x = NativeTensor.from_array(values, requires_grad=True)
        x.log_softmax(axis).backward(NativeTensor.from_array(upstream))
        grads.append(x.grad.to_numpy())
    assert np.array_equal(grads[0], grads[1])


@needs_native
def test_native_log_softmax_gradient_sums_to_zero_along_the_axis():
    """A structural invariant: each slice's gradient sums to zero,
    because the probabilities it subtracts sum to one."""
    rng = np.random.default_rng(3)
    values = rng.uniform(-2.0, 2.0, size=(3, 5))
    upstream = rng.uniform(-4.0, 4.0, size=(3, 5))
    x = NativeTensor.from_array(values, requires_grad=True)
    x.log_softmax(1).backward(NativeTensor.from_array(upstream))
    assert np.allclose(x.grad.to_numpy().sum(axis=1), 0.0, atol=1e-13)


@needs_native
def test_native_log_softmax_zero_slice_sum_upstream_passes_through():
    """When an upstream slice sums to zero the correction term vanishes
    exactly, so dx == upstream."""
    upstream = np.array([[1.0, -3.0, 2.0], [0.5, 0.25, -0.75]])
    assert np.allclose(upstream.sum(axis=1), 0.0)
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    x.log_softmax(-1).backward(NativeTensor.from_array(upstream))
    assert np.allclose(x.grad.to_numpy(), upstream, atol=1e-15)
    # A *uniform* upstream, by contrast, does not cancel here — the
    # contrast with softmax's Jacobian, and the reason the versioning
    # tests below can use a plain sum() loss.
    y = NativeTensor.from_array(VALUES, requires_grad=True)
    y.log_softmax(-1).sum().backward()
    assert not np.allclose(y.grad.to_numpy(), 0.0)
    assert np.allclose(y.grad.to_numpy(),
                       _analytic_grad(VALUES, np.ones_like(VALUES), -1),
                       atol=1e-14)


@needs_native
def test_native_log_softmax_finite_differences():
    rng = np.random.default_rng(5)
    values = rng.uniform(-2.0, 2.0, size=(3, 4))
    weights = rng.uniform(0.5, 2.0, size=(3, 4))

    def loss_of(a):
        return float(np.sum(log_softmax_reference(a, -1) * weights))

    x = NativeTensor.from_array(values, requires_grad=True)
    x.log_softmax(-1).multiply(
        NativeTensor.from_array(weights)
    ).sum().backward()
    assert np.allclose(x.grad.to_numpy(), _numeric_grad(loss_of, values),
                       atol=1e-6)

    # And along axis 0.
    def loss_axis0(a):
        return float(np.sum(log_softmax_reference(a, 0) * weights))

    z = NativeTensor.from_array(values, requires_grad=True)
    z.log_softmax(0).multiply(
        NativeTensor.from_array(weights)
    ).sum().backward()
    assert np.allclose(z.grad.to_numpy(), _numeric_grad(loss_axis0, values),
                       atol=1e-6)


@needs_native
def test_native_log_softmax_backward_through_a_non_contiguous_parent():
    rng = np.random.default_rng(13)
    values = rng.uniform(-2.0, 2.0, size=(3, 4))
    upstream = rng.uniform(-1.0, 1.0, size=(4, 3))
    x = NativeTensor.from_array(values, requires_grad=True)
    x.T.log_softmax(-1).backward(NativeTensor.from_array(upstream))
    expected = _analytic_grad(values.T, upstream, -1).T
    assert np.allclose(x.grad.to_numpy(), expected, atol=1e-14)


@needs_native
def test_native_log_softmax_chained_and_shared_subgraph():
    rng = np.random.default_rng(17)
    values = rng.uniform(-2.0, 2.0, size=(2, 4))
    weights = rng.uniform(0.5, 1.5, size=(2, 4))
    # Chained: exp(log_softmax(x)) . weights — the probabilities again,
    # so the gradient must equal softmax's own Jacobian-vector product.
    x = NativeTensor.from_array(values, requires_grad=True)
    x.log_softmax(-1).exp().multiply(
        NativeTensor.from_array(weights)
    ).sum().backward()
    probabilities = softmax_reference(values, -1)
    softmax_grad = probabilities * (
        weights - np.sum(weights * probabilities, axis=-1, keepdims=True)
    )
    assert np.allclose(x.grad.to_numpy(), softmax_grad, atol=1e-13)
    # Shared subgraph: y used twice accumulates through one node.
    z = NativeTensor.from_array(values, requires_grad=True)
    y = z.log_softmax(-1)
    y.add(y).multiply(NativeTensor.from_array(weights)).sum().backward()
    assert np.allclose(z.grad.to_numpy(),
                       _analytic_grad(values, 2.0 * weights, -1), atol=1e-13)


@needs_native
def test_native_log_softmax_repeated_accumulation_and_retain_graph():
    weights = np.linspace(0.5, 2.0, VALUES.size).reshape(VALUES.shape)
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    loss = x.log_softmax(-1).multiply(
        NativeTensor.from_array(weights)
    ).sum()
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
def test_native_log_softmax_records_no_version_snapshot():
    p = NativeParameter(VALUES)
    node = p.log_softmax(-1)
    assert node._expected_versions == ()
    assert node._graph_resources == ()


@needs_native
def test_native_log_softmax_parameter_mutation_keeps_graph_valid():
    """The E4 invariant: backward reads only the saved log
    probabilities, so mutating a direct parameter after forward leaves
    the edge valid — and the gradient is the one for the forward that
    actually ran.

    A *weighted* (non-uniform) upstream is used deliberately: it makes
    the saved-output distinction observable."""
    original = np.array([[0.5, 1.5, -1.0], [2.0, 0.0, 1.0]])
    weights = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    p = NativeParameter(original)
    loss = p.log_softmax(-1).multiply(
        NativeTensor.from_array(weights)
    ).sum()
    mutated = np.array([[3.0, -2.0, 0.25], [-1.5, 2.5, 0.0]])
    p.copy_value_(NativeTensor.from_array(mutated))
    loss.backward()  # no stale error: nothing rereads p

    expected = _analytic_grad(original, weights, -1)
    assert np.allclose(p.grad.to_numpy(), expected, atol=1e-14)
    assert not np.allclose(p.grad.to_numpy(), 0.0), "gradient must be nonzero"
    # Explicitly NOT the gradient of the mutated values.
    mutated_grad = _analytic_grad(mutated, weights, -1)
    assert not np.allclose(p.grad.to_numpy(), mutated_grad, atol=1e-8)
    # A fresh forward after mutation uses the new values.
    p.zero_grad()
    p.log_softmax(-1).multiply(
        NativeTensor.from_array(weights)
    ).sum().backward()
    assert np.allclose(p.grad.to_numpy(), mutated_grad, atol=1e-14)


@needs_native
def test_native_log_softmax_does_not_weaken_other_versioning():
    """E4 must not perturb the established classifications."""
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
    # softmax stays saved-output / no version (weighted loss, since a
    # plain softmax().sum() has a zero gradient).
    s = NativeParameter(np.array([[1.0, 2.0], [3.0, 4.0]]))
    softmax_loss = s.softmax(-1).multiply(
        NativeTensor.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
    ).sum()
    s.copy_value_(NativeTensor.from_array(np.full((2, 2), 2.0)))
    softmax_loss.backward()
    assert s.grad is not None
    # multiply stays guarded.
    r = NativeParameter(np.array([[1.0, 2.0], [3.0, 4.0]]))
    mul_loss = r.multiply(r).sum()
    r.copy_value_(NativeTensor.from_array(np.full((2, 2), 6.0)))
    with pytest.raises(RuntimeError, match="stale"):
        mul_loss.backward()
    assert r.grad is None


@needs_native
@pytest.mark.parametrize("stale_branch", ["log", "multiply"])
def test_native_log_softmax_mixed_stale_graph_commits_nothing(stale_branch):
    """A log-softmax branch is valid across mutation, but if another
    branch goes stale the whole pass must commit nothing — twice, so the
    failure is deterministic on retry."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    p = NativeParameter(values)
    healthy = p.log_softmax(-1).multiply(
        NativeTensor.from_array(weights)
    ).sum()
    stale = p.log().sum() if stale_branch == "log" else p.multiply(p).sum()
    loss = healthy.add(stale)
    p.copy_value_(NativeTensor.from_array(np.full((2, 2), 7.0)))
    with pytest.raises(RuntimeError, match="stale"):
        loss.backward()
    assert p.grad is None       # the log-softmax branch committed nothing
    assert not loss._graph_freed
    with pytest.raises(RuntimeError, match="stale"):
        loss.backward()         # deterministic on retry
    assert p.grad is None


@needs_native
def test_native_log_softmax_mixed_stale_graph_leaves_existing_grads_alone():
    """A preexisting gradient on an untouched parameter must survive a
    failed pass unchanged."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    other = NativeParameter(np.array([[0.5, 1.5], [2.5, 3.5]]))
    other.log_softmax(-1).multiply(
        NativeTensor.from_array(values)
    ).sum().backward()
    before = other.grad.to_numpy().copy()

    p = NativeParameter(values)
    loss = (p.log_softmax(-1).multiply(NativeTensor.from_array(values)).sum()
            .add(p.log().sum())
            .add(other.log_softmax(-1).sum()))
    p.copy_value_(NativeTensor.from_array(np.full((2, 2), 7.0)))
    with pytest.raises(RuntimeError, match="stale"):
        loss.backward()
    assert p.grad is None
    assert np.array_equal(other.grad.to_numpy(), before)


# ======================================================================
# Graph lifetime
# ======================================================================


@needs_native
def test_native_log_softmax_no_grad_and_detached_paths():
    x = NativeTensor.from_array(VALUES)
    out = x.log_softmax(-1)
    assert out.is_leaf and out._backward is None and out._parents == ()
    assert not out.requires_grad
    with pytest.raises(RuntimeError,
                       match="does not require grad|requires_grad"):
        out.sum().backward()
    tracked = NativeTensor.from_array(VALUES, requires_grad=True)
    detached = tracked.detach().log_softmax(-1)
    assert not detached.requires_grad and detached.is_leaf
    detached.close()


@needs_native
def test_native_log_softmax_closed_saved_output_fails_atomically():
    """The saved log probabilities *are* the derivative's source, so
    closing them before a downstream backward must fail — and commit
    nothing."""
    weights = np.linspace(0.5, 2.0, VALUES.size).reshape(VALUES.shape)
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    y = x.log_softmax(-1)
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
def test_native_log_softmax_closed_parent_follows_the_accumulation_rule():
    """log_softmax backward never rereads the parent's value, but the
    parent must still be open to *receive* its gradient — the engine's
    ordinary ownership rule, not a version requirement."""
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    intermediate = x.add(x)
    loss = intermediate.log_softmax(-1).multiply(
        NativeTensor.from_array(np.linspace(1.0, 2.0, VALUES.size)
                                .reshape(VALUES.shape))
    ).sum()
    intermediate.close()
    with pytest.raises(RuntimeError, match="closed"):
        loss.backward()
    assert x.grad is None


@needs_native
def test_native_log_softmax_abandoned_graph_closes_cleanly():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.log_softmax(-1)
    out.close()
    out.close()  # idempotent
    assert out.closed and not x.closed
    assert x.grad is None
    x.log_softmax(-1).multiply(
        NativeTensor.from_array(np.full(VALUES.shape, 2.0))
    ).sum().backward()
    assert x.grad is not None


@needs_native
def test_native_log_softmax_rejects_invalid_backward_arguments():
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = x.log_softmax(-1)
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
def test_native_log_softmax_allocation_failure_is_atomic():
    """Sweep the injector across a forward+backward. Every stage that can
    be hit must raise MemoryError, mutate nothing, commit no gradient, and
    leave the graph retryable. (The injector counts allocations rather
    than naming stages, so the sweep reports what is reachable instead of
    assuming a fixed mapping — the reachable backward stages here are the
    exp of the saved output, the axis reduction, the broadcast multiply,
    the subtraction, and the gradient accumulation.)"""
    values = np.array([[1.0, 2.0], [0.5, -1.0]])
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])

    # -- forward output allocation --
    x = NativeTensor.from_array(values, requires_grad=True)
    with pytest.raises(MemoryError):
        cpp._arm_alloc_failure(1)
        x.log_softmax(-1)
    cpp._arm_alloc_failure(0)
    assert np.array_equal(x.to_numpy(), values)
    assert x.grad is None
    assert np.allclose(x.log_softmax(-1).to_numpy(),
                       log_softmax_reference(values, -1), atol=1e-15)

    # -- backward allocations --
    hit = 0
    for nth in range(1, 10):
        y = NativeTensor.from_array(values, requires_grad=True)
        loss = y.log_softmax(-1).multiply(
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
    assert hit >= 4, "expected several reachable backward allocation stages"
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK


@needs_fault_injection
def test_native_log_softmax_policy_b_allocation_failure_is_atomic():
    """The same sweep over a non-contiguous forward, where the Policy-B
    copy adds its own reachable allocations."""
    base = cpp.NativeTensorCore.from_array(VALUES)
    strided = base.T
    hit = 0
    for nth in range(1, 6):
        try:
            cpp._arm_alloc_failure(nth)
            out = strided.log_softmax(0)
        except MemoryError:
            hit += 1
            cpp._arm_alloc_failure(0)
            assert np.array_equal(base.to_numpy(), VALUES), nth
        else:
            cpp._arm_alloc_failure(0)
            assert np.allclose(out.to_numpy(),
                               log_softmax_reference(VALUES.T, 0), atol=1e-15)
            out.close()
    assert hit >= 1, "the sweep never triggered an allocation failure"
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    recovered = strided.log_softmax(0)
    assert np.allclose(recovered.to_numpy(),
                       log_softmax_reference(VALUES.T, 0), atol=1e-15)


# ======================================================================
# Raw ABI misuse
# ======================================================================


@needs_native
def test_native_log_softmax_abi_rejects_invalid_calls():
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
        (2, -2, 1, "must each be >= 1"),
        (2, 2, -1, "must each be >= 1"),
        (2, 3, 1, "source span exceeds its storage"),
        (2, 2, 2, "source span exceeds its storage"),
    ):
        with pytest.raises(ValueError, match=message):
            library.tf_core_log_softmax_forward(src_handle, 0, dst_handle,
                                                outer, axis_length, inner)
    with pytest.raises(ValueError, match="negative source offset"):
        library.tf_core_log_softmax_forward(src_handle, -1, dst_handle,
                                            2, 2, 1)
    with pytest.raises(ValueError, match="source span exceeds its storage"):
        library.tf_core_log_softmax_forward(src_handle, 2, dst_handle, 1, 4, 1)
    # Null handles.
    with pytest.raises(ValueError, match="null storage handle"):
        library.tf_core_log_softmax_forward(None, 0, dst_handle, 2, 2, 1)
    with pytest.raises(ValueError, match="null storage handle"):
        library.tf_core_log_softmax_forward(src_handle, 0, None, 2, 2, 1)
    # Overflow in the dimension product and in offset + numel.
    huge = 2 ** 62 + 4
    with pytest.raises(ValueError, match="overflows int64"):
        library.tf_core_log_softmax_forward(src_handle, 0, dst_handle,
                                            huge, huge, 1)
    with pytest.raises(ValueError, match="overflows int64"):
        library.tf_core_log_softmax_forward(src_handle, 0, dst_handle,
                                            huge, 1, huge)
    with pytest.raises(ValueError, match="source span exceeds its storage"):
        library.tf_core_log_softmax_forward(src_handle, 2 ** 63 - 1,
                                            dst_handle, 1, 2, 1)
    # A destination smaller than numel.
    small = cpp.NativeTensorCore.from_array(np.full(2, 7777.5))
    with pytest.raises(ValueError, match="destination storage smaller"):
        library.tf_core_log_softmax_forward(src_handle, 0,
                                            small.storage._require_open(),
                                            2, 2, 1)
    # Nothing was written by any rejected call.
    assert np.array_equal(destination.to_numpy(), np.full(4, 7777.5))
    assert np.array_equal(small.to_numpy(), np.full(2, 7777.5))
    # Each message is attributed to log_softmax, not to softmax.
    with pytest.raises(ValueError, match="log_softmax_forward"):
        library.tf_core_log_softmax_forward(src_handle, -1, dst_handle,
                                            2, 2, 1)
    # The stale error does not survive the next valid call.
    assert np.allclose(source.log_softmax(0).to_numpy(),
                       log_softmax_reference(np.arange(1.0, 5.0), 0),
                       atol=1e-15)
    assert library.tf_last_error_code() == cpp.TF_OK


@needs_native
def test_shared_abi_validation_left_softmax_messages_unchanged():
    """E4 factored the two exports' precondition checks into one
    file-local validator. Softmax's own rejections must be identical —
    same errors, same messages, same untouched destination."""
    library = cpp._require_library()
    source = cpp.NativeTensorCore.from_array(np.arange(1.0, 5.0))
    destination = cpp.NativeTensorCore.from_array(np.full(4, 7777.5))
    src_handle = source.storage._require_open()
    dst_handle = destination.storage._require_open()

    with pytest.raises(ValueError, match="softmax_forward: outer"):
        library.tf_core_softmax_forward(src_handle, 0, dst_handle, 0, 2, 1)
    with pytest.raises(ValueError,
                       match="softmax_forward: negative source offset"):
        library.tf_core_softmax_forward(src_handle, -1, dst_handle, 2, 2, 1)
    with pytest.raises(ValueError,
                       match="softmax_forward: source span exceeds"):
        library.tf_core_softmax_forward(src_handle, 0, dst_handle, 2, 3, 1)
    with pytest.raises(ValueError, match="softmax_forward: null storage"):
        library.tf_core_softmax_forward(None, 0, dst_handle, 2, 2, 1)
    assert np.array_equal(destination.to_numpy(), np.full(4, 7777.5))
    # ...and softmax still computes what E3 shipped.
    assert np.allclose(source.softmax(0).to_numpy(),
                       softmax_reference(np.arange(1.0, 5.0), 0), atol=1e-15)
    assert library.tf_last_error_code() == cpp.TF_OK


# The unavailable-backend contract for `log_softmax` — ImportError with
# build instructions, never a silent NumPy fallback — is proved in
# tests/test_native_backend_unavailable.py (the "log_softmax" case). It
# used to live here and skip whenever the backend was built, which is
# every machine that can run this file at all; it now runs
# unconditionally, in a fresh child process that repoints only its *own*
# library path.


# ======================================================================
# Guardrails
# ======================================================================


def _numpy_tripwire(monkeypatch, names):
    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in names:
        monkeypatch.setattr(np, name, _tripwire)


# Everything NumPy could plausibly be used to *compute* a log-softmax
# with.
_NUMERICAL_NUMPY = (
    "max", "amax", "argmax", "exp", "log", "logaddexp", "sum", "divide",
    "true_divide", "add", "subtract", "multiply", "matmul", "mean",
    "negative", "power", "copyto",
)
# Every route by which tensor *data* could enter or leave a NumPy host
# buffer: the two Core conversion methods plus the array constructors
# they would have to use. (np.asarray is deliberately absent: every
# native op marshals its shape/stride arrays through it because that is
# the ctypes calling convention — small layout metadata, never tensor
# data. See test_native_log_softmax_metadata_marshalling_is_not_data
# below, which pins that distinction.)
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
def test_native_log_softmax_contiguous_path_and_backward_use_no_numpy(
        monkeypatch):
    """Strict tripwire for the contiguous path: no NumPy arithmetic and
    no tensor-data conversion in the Core forward, the wrapper forward,
    or the backward."""
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    core = cpp.NativeTensorCore.from_array(VALUES)
    upstream_values = np.linspace(0.5, 2.0, VALUES.size).reshape(VALUES.shape)
    upstream = NativeTensor.from_array(upstream_values)

    _data_conversion_tripwire(monkeypatch)
    core.log_softmax(-1)                          # Core forward
    x.log_softmax(-1).backward(upstream)          # wrapper forward + backward
    monkeypatch.undo()
    assert np.allclose(x.grad.to_numpy(),
                       _analytic_grad(VALUES, upstream_values, -1), atol=1e-15)


@needs_native
def test_native_log_softmax_policy_b_path_uses_no_numpy(monkeypatch):
    """The **same strict** tripwire for the non-contiguous path.

    The Policy-B copy is a native storage-to-storage gather
    (``tf_core_contiguous_copy``, E3.1), so a strided log-softmax keeps
    tensor data in native memory end to end — there is no host round-trip
    left to exempt."""
    base = cpp.NativeTensorCore.from_array(VALUES)
    strided = base.T
    assert not strided.contiguous
    _data_conversion_tripwire(monkeypatch)
    out = strided.log_softmax(0)                  # Policy-B copy + kernel
    monkeypatch.undo()
    assert np.allclose(out.to_numpy(), log_softmax_reference(VALUES.T, 0),
                       atol=1e-15)
    # The same through the autograd layer, backward included.
    tracked = NativeTensor.from_array(VALUES, requires_grad=True)
    upstream_values = np.linspace(0.5, 2.0, VALUES.size).reshape(VALUES.T.shape)
    upstream = NativeTensor.from_array(upstream_values)
    _data_conversion_tripwire(monkeypatch)
    tracked.T.log_softmax(0).backward(upstream)
    monkeypatch.undo()
    assert np.allclose(tracked.grad.to_numpy(),
                       _analytic_grad(VALUES.T, upstream_values, 0).T,
                       atol=1e-14)


@needs_native
def test_native_log_softmax_metadata_marshalling_is_not_data(monkeypatch):
    """The deliberate boundary: ``np.asarray`` *is* used, but only to
    marshal shape/stride arrays for ctypes. This test pins that
    distinction so the tripwire above cannot be quietly widened into a
    false claim — every asarray call the log-softmax path makes receives
    a small tuple of ints, never tensor values."""
    seen = []
    original = np.asarray

    def _recording_asarray(values, *args, **kwargs):
        seen.append(values)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", _recording_asarray)
    base = cpp.NativeTensorCore.from_array(CUBE)
    base.T.log_softmax(0)                      # Policy-B copy + kernel
    monkeypatch.undo()
    assert seen, "the log_softmax path marshals layout metadata through asarray"
    for value in seen:
        assert isinstance(value, tuple), value
        assert all(isinstance(extent, int) for extent in value), value
        # Shape/stride tuples are rank-sized, never element-sized.
        assert len(value) <= CUBE.ndim, value


@needs_native
def test_native_log_softmax_scope_boundaries_hold():
    """E4 is log-softmax only: no later Phase-E surface, no public
    max/argmax or division, no module, and the stable framework is
    untouched."""
    x = NativeTensor.from_array(VALUES)
    core = cpp.NativeTensorCore.from_array(VALUES)
    for absent in ("max", "argmax", "amax", "divide",
                   "sigmoid", "tanh", "nll_loss"):
        assert not hasattr(x, absent), absent
        assert not hasattr(core, absent), absent
    assert not hasattr(x, "__truediv__")
    import tensorforge.experimental as experimental
    # (`NativeCrossEntropyLoss` and `native_accuracy` left this list at
    # E7, which shipped both; neither is a log-softmax capability.)
    for absent in ("NativeLogSoftmax", "NativeSoftmax", "NativeNLLLoss"):
        assert not hasattr(experimental, absent), absent
    assert "NativeLogSoftmax" not in cpp.NATIVE_MODULES
    assert "NativeSoftmax" not in cpp.NATIVE_MODULES
    # The stable Tensor keeps its own softmax, entirely separately, and
    # gained no log_softmax from this milestone.
    stable = tensorforge.Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
    stable.softmax().sum().backward()
    assert type(stable.softmax()) is tensorforge.Tensor
    assert not hasattr(stable, "log_softmax")
    # No implicit dispatch in either direction.
    with pytest.raises((TypeError, AttributeError)):
        NativeTensor.from_array(VALUES).multiply(tensorforge.Tensor(VALUES))


@needs_native
def test_native_log_softmax_checkpoint_schema_is_untouched():
    """E4 adds no persistent state: the native checkpoint format version
    is still 1 (docs/native_classification_design.md §12)."""
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT_VERSION == 3
