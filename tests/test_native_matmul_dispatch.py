"""The H2 matmul memory-access contract (Phase H, milestone H2).

H2 changed *how* ``tf_core_matmul`` walks memory and nothing else. The
native side now ships two compute paths behind one unchanged export:

* ``tf::matmul_generic_strided`` — the pre-H2 ``i``-``j``-``k`` triple
  loop that addresses both operands through their own strides and
  offsets. It is the **retained generic reference path**: shipped,
  reachable through ordinary production dispatch, and the oracle every
  optimized result is compared against.
* ``tf::matmul_row_sweep`` — an ``i``-``k``-``j`` sweep over
  ``MATMUL_ROW_BLOCK`` destination rows at a time, used when the right
  operand's column stride is 1, ``n >= 1``, and the result has at least
  ``MATMUL_MIN_COLUMNS`` columns.

The choice is made inside the kernel from the stride metadata the export
already receives. It is deterministic, total, side-effect free, and
independent of pointer values, alignment, wall time, environment
variables, and CPU-feature probes. **No selector, block-size setter,
dispatch tracer, or "which kernel ran" hook exists in the ABI**, and §5
below asserts that against the built image's own export table.

What this file proves, and how it observes a decision it cannot query:

1. **H2's numerical contract**, which has four parts and is deliberately
   *not* an unqualified claim of bit identity:

   a. **Accumulation order is preserved exactly.** Both paths start each
      output element at ``0.0`` and take the same products in the same
      ascending ``k`` order. Nothing is reassociated, no partial sums are
      combined, no accumulator width changes, no fused multiply-add is
      requested, and no parallel or vector reduction exists.
   b. **Every non-NaN result is bit-identical** — ``+0.0`` versus
      ``-0.0``, ``±inf``, denormals, the smallest normal, and the largest
      finite magnitudes included.
   c. **NaN-class equivalence.** Whenever either path produces a NaN,
      both do, in exactly the same positions, and both are **quiet**.
      Neither path can produce a signaling NaN.
   d. **NaN payload bits are outside TensorForge's numerical contract**
      and may differ between the paths.

   (a)–(c) are asserted exactly here. (d) is asserted in **neither**
   direction: a build whose payloads all agree and a build whose payloads
   all differ are equally conforming. On this toolchain MSVC Release
   differs on 162 of 208 results in a NaN-saturated matrix and MSVC Debug
   and Clang differ on none — the selection follows from the compiler's
   instruction operand ordering, which follows from the loop order rather
   than from anything C++ can express.

   The comparison itself is real and uses only the public API: a
   contiguous right operand takes the row sweep, the same values
   delivered as a transposed view take the generic path, so no production
   observation hook is needed. The independent raw-buffer ``cpp.matmul``
   kernel is a third reading.
2. **Deterministic training and exact checkpoint resume stay
   bit-identical for supported finite workloads**, which part (b) covers
   completely — every committed loss trajectory and every exact-resume
   proof in this project is over finite data.
3. **The H1 output-allocation contract still holds.** The row sweep
   accumulates in the destination, so its ``k == 0`` pass assigns every
   element of every row before anything reads one. Poison proofs for the
   optimized path live in ``tests/test_native_storage_allocation.py``
   beside H1's own.
4. **Autograd, Linear, optimizer, and training parity.**
5. **Scope.** No new export, no capability move, no public control.

The predicate itself, both kernels in isolation, the special-value
matrix, and the partial-write negative control are additionally driven
directly in ``cpp/tests/test_matmul.cpp``, which compiles
``cpp/src/matmul.cpp`` in so it can reach the hidden internals.

Nothing here asserts a duration. H2's measurement lives in
``benchmarks/benchmark_native_cpu_performance.py``.

Selector: python -m pytest -q -k native_matmul_dispatch
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeLinear,
    NativeMSELoss,
    NativeSGD,
    NativeTensor,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

# ==========================================================================
# The dispatch policy, mirrored here as a test-only reference.
#
# This mirrors the C++ predicate rather than calling it — the predicate is
# hidden-visibility internal C++ and the library exports no way to ask it
# anything, which is exactly the property H2 must keep. The same table is
# asserted against the real predicate in cpp/tests/test_matmul.cpp, and
# the two constants are pinned against the shipped header below, so the
# mirror cannot drift silently.
# ==========================================================================

MATMUL_ROW_BLOCK = 4
MATMUL_MIN_COLUMNS = 8


def prefers_row_sweep(m, n, p, b_stride1):
    """The documented precondition, mirrored. See
    docs/native_cpu_performance_design.md §16.2."""
    return b_stride1 == 1 and n >= 1 and p >= MATMUL_MIN_COLUMNS


def core_takes_row_sweep(left, right):
    """Whether these two live tensor cores would take the optimized path,
    read from the very metadata the Core passes to the kernel."""
    return prefers_row_sweep(left.shape[0], left.shape[1],
                             right.shape[1], right.strides[1])


def rng(seed=0):
    return np.random.default_rng(seed)


def bits(values):
    """The raw IEEE-754 bit pattern of every element, so a comparison sees
    +0.0 as different from -0.0 and does not treat NaN as equal to
    itself."""
    return np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)


def same_bits(left, right):
    return np.array_equal(bits(left), bits(right))


def agrees_under_the_numerical_contract(left, right):
    """Parts (b) and (c) of the contract together: every element is
    bit-identical, or **both** sides are NaN. One NaN against a number, or
    two different non-NaN numbers, is a real difference and fails.

    Says nothing about NaN payload bits, which part (d) puts outside the
    contract."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return False
    equal = bits(left) == bits(right)
    both_nan = np.isnan(left) & np.isnan(right)
    return bool(np.all(equal | both_nan))


def as_generic_operand(values):
    """The same logical (n, p) matrix, delivered through a layout whose
    column stride is **not** 1 — which routes the product through the
    generic reference path.

    The first choice is a real transposed view, because that is the
    layout production actually produces (``da = upstream @ b.T`` in the
    matmul backward). A transposed ``(n, p)`` view has column stride
    ``n``, which is 1 only when ``n == 1``; in that one case the same
    values are laid out into every second column of a doubled buffer and
    read back with a column stride of 2, which is an equally genuine
    strided operand.

    Returns ``(base, view)``; the caller closes both. The column stride is
    asserted, so a test can never accidentally compare the optimized path
    against itself."""
    values = np.ascontiguousarray(values, dtype=np.float64)
    n, p = values.shape
    if n != 1:
        base = cpp.NativeTensorCore.from_array(np.ascontiguousarray(values.T))
        view = base.transpose(1, 0)
    else:
        interleaved = np.zeros((n, p * 2), dtype=np.float64)
        interleaved[:, ::2] = values
        base = cpp.NativeTensorCore.from_array(interleaved)
        view = cpp.NativeTensorCore(
            base.storage, cpp.NativeTensorView(base.storage, (n, p),
                                               (p * 2, 2), 0),
            owns_storage=False)
    assert view.strides[1] != 1, (
        f"a {values.shape} operand still has unit column stride, so it does "
        f"not force the generic path"
    )
    # Bit equality, not value equality: these operands deliberately carry
    # NaNs, which never compare equal to themselves.
    assert same_bits(view.to_numpy(), values)
    return base, view


def strided_view(core, shape, strides, offset=0):
    """A borrowing view with arbitrary positive strides over an existing
    core's storage."""
    return cpp.NativeTensorCore(
        core.storage,
        cpp.NativeTensorView(core.storage, shape, strides, offset),
        owns_storage=False)


def close_module(module):
    """There is no ``NativeModule.close()``; an owner releases the
    parameters and buffers explicitly."""
    for parameter in module.parameters():
        parameter.close()
    for buffer in module.buffers():
        buffer.close()


def product_through_both_paths(left_values, right_values):
    """Compute ``left @ right`` twice through the real production Core —
    once with a right operand that qualifies for the row sweep, once with
    the same logical values through a layout that cannot — and return both
    results plus which path the first one took."""
    left = cpp.NativeTensorCore.from_array(left_values)
    fast_right = cpp.NativeTensorCore.from_array(right_values)
    base, generic_right = as_generic_operand(right_values)
    try:
        assert fast_right.strides[1] == 1
        took_row_sweep = core_takes_row_sweep(left, fast_right)
        assert not core_takes_row_sweep(left, generic_right)
        fast_out = left.matmul(fast_right)
        try:
            fast = fast_out.to_numpy().copy()
        finally:
            fast_out.close()
        generic_out = left.matmul(generic_right)
        try:
            generic = generic_out.to_numpy().copy()
        finally:
            generic_out.close()
    finally:
        for core in (left, fast_right, base, generic_right):
            if not core._closed:
                core.close()
    return fast, generic, took_row_sweep


# ==========================================================================
# 1. The dispatch policy is exactly what is documented
# ==========================================================================

@needs_native
def test_the_policy_constants_match_the_shipped_header():
    """The mirror above is only useful if it cannot drift from the
    constants the kernel actually compiles with."""
    header = (REPO_ROOT / "cpp" / "include" / "tf_matmul_internal.h").read_text(
        encoding="utf-8")
    assert f"MATMUL_ROW_BLOCK = {MATMUL_ROW_BLOCK};" in header
    assert f"MATMUL_MIN_COLUMNS = {MATMUL_MIN_COLUMNS};" in header


@pytest.mark.parametrize("m,n,p,b_stride1,expected", [
    (4, 4, 8, 1, True),          # the canonical qualifying layout
    (1, 1, 64, 1, True),         # a single row, a single k
    (0, 3, 16, 1, True),         # no rows at all: still well defined
    (1000, 1, 1000, 1, True),    # n == 1 is the smallest assigning pass
    (64, 64, 64, 64, False),     # transposed right operand
    (64, 64, 64, 2, False),      # non-unit column stride
    (64, 64, 64, 0, False),      # zero column stride
    (64, 64, 64, -1, False),     # negative column stride
    (4, 0, 64, 1, False),        # no k, so no assigning pass
    (8, 8, 7, 1, False),         # one column below the threshold
    (8, 8, 8, 1, True),          # exactly at the threshold
    (8, 8, 9, 1, True),          # one above
    (8, 8, 1, 1, False),
    (8, 8, 0, 1, False),
])
def test_the_documented_precondition_table(m, n, p, b_stride1, expected):
    assert prefers_row_sweep(m, n, p, b_stride1) is expected


def test_the_precondition_is_pure_and_repeatable():
    """Selection allocates nothing, mutates nothing, and gives the same
    answer every time it is asked."""
    for _ in range(5):
        assert prefers_row_sweep(17, 23, 29, 1) is True
        assert prefers_row_sweep(17, 23, 29, 23) is False


@needs_native
def test_a_contiguous_right_operand_qualifies_and_a_transposed_one_does_not():
    """The two layouts the parity tests rely on really do land on
    different paths, read from live tensor metadata."""
    left = cpp.NativeTensorCore.from_array(rng(1).standard_normal((6, 5)))
    right = cpp.NativeTensorCore.from_array(rng(2).standard_normal((5, 16)))
    base, view = as_generic_operand(rng(2).standard_normal((5, 16)))
    try:
        assert right.strides == (16, 1)
        assert core_takes_row_sweep(left, right)
        assert view.strides == (1, 5)
        assert not core_takes_row_sweep(left, view)
    finally:
        for core in (left, right, base, view):
            core.close()


# ==========================================================================
# 2. Bit identity between the two shipped paths
# ==========================================================================

SHAPES = [
    (1, 1, 1), (1, 1, 8), (1, 8, 1), (8, 1, 1),
    (2, 3, 4), (4, 4, 4), (5, 7, 8), (8, 8, 8),
    # Row-block boundaries: MATMUL_ROW_BLOCK is 4, so m of 3/4/5 and 7/8/9
    # cover a partial group, an exact group, and multiple groups with a
    # partial tail.
    (3, 6, 16), (4, 6, 16), (5, 6, 16), (7, 6, 16), (8, 6, 16), (9, 6, 16),
    # Column threshold, both sides.
    (6, 6, 7), (6, 6, 8), (6, 6, 9),
    # Primes, and dimensions that share no factor with the row block.
    (11, 13, 17), (17, 23, 29), (31, 31, 31),
    # Rectangular: tall-skinny, short-wide, and the Linear shapes.
    (32, 8, 4), (4, 8, 32), (16, 24, 12), (12, 4, 64),
    (2, 2, 2), (64, 16, 8),
]


@needs_native
@pytest.mark.parametrize("m,n,p", SHAPES)
def test_both_paths_agree_bit_for_bit(m, n, p):
    left = rng(m * 100 + n).standard_normal((m, n))
    right = rng(n * 100 + p).standard_normal((n, p))
    fast, generic, took_row_sweep = product_through_both_paths(left, right)
    assert took_row_sweep == prefers_row_sweep(m, n, p, 1)
    assert same_bits(fast, generic), (m, n, p)
    assert fast.shape == (m, p)
    assert np.allclose(fast, left @ right, atol=1e-10)


@needs_native
@pytest.mark.parametrize("m,n,p", SHAPES)
def test_both_paths_agree_with_the_raw_naive_kernel(m, n, p):
    """A third, independent reading: ``cpp.matmul`` is a separate
    transcription of the same triple loop over plain row-major buffers, on
    no production path at all."""
    left = rng(m + n * 3).standard_normal((m, n))
    right = rng(n + p * 5).standard_normal((n, p))
    fast, generic, _ = product_through_both_paths(left, right)
    reference = cpp.matmul(left, right)
    assert same_bits(fast, reference), (m, n, p)
    assert same_bits(generic, reference), (m, n, p)


@needs_native
def test_the_column_threshold_boundary_is_correct_on_both_sides():
    """The one metadata threshold H2 introduces, exercised at p - 1, p,
    and p + 1 with the *same* left operand and the same logical values, so
    only the dispatch decision differs."""
    left = rng(11).standard_normal((6, 5))
    for p in range(MATMUL_MIN_COLUMNS - 2, MATMUL_MIN_COLUMNS + 3):
        right = rng(12).standard_normal((5, p))
        fast, generic, took_row_sweep = product_through_both_paths(left, right)
        assert took_row_sweep is (p >= MATMUL_MIN_COLUMNS)
        assert same_bits(fast, generic), p
        assert same_bits(fast, cpp.matmul(left, right)), p


@needs_native
def test_larger_shapes_agree_bit_for_bit():
    """Sizes where the row sweep spans many row groups and the working set
    leaves L1 — the regime the optimization is actually for."""
    for m, n, p in ((128, 128, 128), (127, 65, 33), (64, 256, 96)):
        left = rng(m).standard_normal((m, n))
        right = rng(p).standard_normal((n, p))
        fast, generic, took_row_sweep = product_through_both_paths(left, right)
        assert took_row_sweep
        assert same_bits(fast, generic), (m, n, p)


# ==========================================================================
# 3. Layouts: the generic path stays reachable and stays correct
# ==========================================================================

@needs_native
def test_a_transposed_left_operand_still_qualifies():
    """``db = a.T @ upstream`` in the matmul backward feeds the kernel a
    transposed *left* operand beside a contiguous right one. The row sweep
    reads the left operand through its own strides, so that layout
    qualifies and must still be exact."""
    values = rng(21).standard_normal((5, 12))     # logical left is (12, 5)
    right = rng(22).standard_normal((5, 16))
    base = cpp.NativeTensorCore.from_array(values)
    left = base.transpose(1, 0)
    fast_right = cpp.NativeTensorCore.from_array(right)
    generic_base, generic_right = as_generic_operand(right)
    try:
        assert left.strides == (1, 12) and not left.contiguous
        assert core_takes_row_sweep(left, fast_right)
        fast_out = left.matmul(fast_right)
        generic_out = left.matmul(generic_right)
        try:
            assert same_bits(fast_out.to_numpy(), generic_out.to_numpy())
            assert np.allclose(fast_out.to_numpy(), values.T @ right,
                               atol=1e-10)
        finally:
            fast_out.close()
            generic_out.close()
    finally:
        for core in (base, left, fast_right, generic_base, generic_right):
            core.close()


@needs_native
def test_a_transposed_right_operand_takes_the_generic_path_and_is_exact():
    """The layout the optimized path deliberately declines. It is not an
    error, it is the fallback — and it is the case the generic loop order
    already suits, because its inner ``k`` loop is then the contiguous
    one."""
    left_values = rng(23).standard_normal((9, 7))
    right_values = rng(24).standard_normal((7, 11))
    left = cpp.NativeTensorCore.from_array(left_values)
    base, view = as_generic_operand(right_values)
    try:
        assert not core_takes_row_sweep(left, view)
        out = left.matmul(view)
        try:
            assert np.allclose(out.to_numpy(), left_values @ right_values,
                               atol=1e-10)
            assert same_bits(out.to_numpy(),
                             cpp.matmul(left_values, right_values))
        finally:
            out.close()
    finally:
        for core in (left, base, view):
            core.close()


@needs_native
def test_offset_and_narrowed_views_on_both_operands():
    """Nonzero offsets and row strides wider than the logical row, on the
    left operand, the right operand, and both at once."""
    left_base = cpp.NativeTensorCore.from_array(rng(25).standard_normal((10, 9)))
    right_base = cpp.NativeTensorCore.from_array(rng(26).standard_normal((12, 16)))
    try:
        left = left_base.narrow(0, 2, 6)          # (6, 9), offset 18
        right = right_base.narrow(0, 3, 9)        # (9, 16), offset 48
        try:
            assert left.offset == 18 and right.offset == 48
            assert right.strides[1] == 1 and core_takes_row_sweep(left, right)
            expected = left.to_numpy() @ right.to_numpy()
            out = left.matmul(right)
            try:
                assert np.allclose(out.to_numpy(), expected, atol=1e-10)
                assert same_bits(out.to_numpy(),
                                 cpp.matmul(left.to_numpy(),
                                            right.to_numpy()))
            finally:
                out.close()
        finally:
            left.close()
            right.close()
    finally:
        left_base.close()
        right_base.close()


@needs_native
def test_a_narrowed_right_operand_keeps_its_unit_column_stride():
    """Narrowing rows off a contiguous matrix leaves the column stride at
    1, so the view still qualifies — and its row stride is then wider than
    its logical row, which is the case a kernel that assumed
    ``b_stride0 == p`` would get wrong."""
    base = cpp.NativeTensorCore.from_array(rng(27).standard_normal((20, 16)))
    left_values = rng(28).standard_normal((5, 8))
    left = cpp.NativeTensorCore.from_array(left_values)
    try:
        right = base.narrow(0, 6, 8)
        try:
            assert right.strides == (16, 1) and right.offset == 96
            assert core_takes_row_sweep(left, right)
            out = left.matmul(right)
            try:
                assert same_bits(
                    out.to_numpy(),
                    cpp.matmul(left_values, right.to_numpy()))
            finally:
                out.close()
        finally:
            right.close()
    finally:
        base.close()
        left.close()


@needs_native
def test_non_unit_positive_strides_on_the_left_operand():
    """Every second row and every third column of a larger buffer, which
    the row sweep reads through the left operand's own strides."""
    values = rng(29).standard_normal((24, 27))
    base = cpp.NativeTensorCore.from_array(values)
    right_values = rng(30).standard_normal((9, 16))
    right = cpp.NativeTensorCore.from_array(right_values)
    try:
        # A (12, 9) view with strides (54, 3): both positive, neither 1.
        sparse = strided_view(base, (12, 9), (54, 3))
        try:
            assert sparse.strides == (54, 3)
            assert np.array_equal(sparse.to_numpy(), values[::2, ::3])
            assert core_takes_row_sweep(sparse, right)
            out = sparse.matmul(right)
            try:
                assert same_bits(out.to_numpy(),
                                 cpp.matmul(values[::2, ::3], right_values))
            finally:
                out.close()
        finally:
            sparse.close()
    finally:
        base.close()
        right.close()


@needs_native
def test_both_operands_non_contiguous_together():
    left_values = rng(31).standard_normal((7, 6))
    right_values = rng(32).standard_normal((6, 9))
    left_base = cpp.NativeTensorCore.from_array(
        np.ascontiguousarray(left_values.T))
    right_base, right_view = as_generic_operand(right_values)
    try:
        left_view = left_base.transpose(1, 0)
        try:
            assert not left_view.contiguous and not right_view.contiguous
            assert not core_takes_row_sweep(left_view, right_view)
            out = left_view.matmul(right_view)
            try:
                assert np.allclose(out.to_numpy(),
                                   left_values @ right_values, atol=1e-10)
            finally:
                out.close()
        finally:
            left_view.close()
    finally:
        for core in (left_base, right_base, right_view):
            core.close()


# ==========================================================================
# 4. Special floating-point values
# ==========================================================================

# Magnitudes chosen so no product and no partial sum can overflow to
# infinity, which is what keeps this set provably NaN-free: the largest
# product is 1e150 * 1e150 = 1e300 and a sum of a dozen of those stays
# well inside the float64 range.
FINITE_SPECIALS = (0.0, -0.0, 1.0, -1.0, 5e-324, -5e-324,
                   2.2250738585072014e-308, 1e-300, 1e150, -1e150,
                   0.5, -0.5)
# The overflowing and non-finite values, which *can* manufacture a NaN
# (0 * inf, inf - inf) and are therefore covered by the payload rule.
ALL_SPECIALS = FINITE_SPECIALS + (1.7976931348623157e308,
                                  -1.7976931348623157e308,
                                  np.inf, -np.inf, np.nan)


def _special_operands(values, p):
    count = len(values)
    left = np.repeat(np.asarray(values, dtype=np.float64)[:, None], count,
                     axis=1)
    right = np.empty((count, p), dtype=np.float64)
    for k in range(count):
        for j in range(p):
            right[k, j] = values[(k + j) % count]
    return left, right


@needs_native
def test_non_nan_special_values_are_bit_identical():
    """Signed zeros, denormals, the smallest normal, the largest finite
    magnitudes. No NaN can arise here, so **exact** bit equality is
    claimed and asserted."""
    left, right = _special_operands(FINITE_SPECIALS, 16)
    fast, generic, took_row_sweep = product_through_both_paths(left, right)
    assert took_row_sweep
    assert not np.any(np.isnan(fast)), "this case is meant to stay NaN-free"
    assert same_bits(fast, generic)
    assert same_bits(fast, cpp.matmul(left, right))


@needs_native
def test_a_negative_zero_product_keeps_the_reference_sign():
    """The single reason the row sweep's ``k == 0`` pass writes
    ``0.0 + product`` rather than the product: ``0.0 + (-0.0)`` is
    ``+0.0``, so dropping the addition would hand back ``-0.0`` where the
    reference gives ``+0.0``."""
    left = np.full((2, 1), -1.0)
    right = np.zeros((1, 8))
    fast, generic, took_row_sweep = product_through_both_paths(left, right)
    assert took_row_sweep
    assert same_bits(fast, generic)
    assert not np.signbit(fast).any(), "the row sweep produced -0.0"
    assert not np.signbit(generic).any()


@needs_native
def test_the_numerical_contract_holds_over_nan_and_infinity():
    """Parts (b) and (c), on the matrix where NaN is reachable.

    What is asserted: every non-NaN result is bit-identical, NaN results
    occur in exactly the same positions on both paths, and every NaN is
    quiet. What is deliberately **not** asserted, in either direction, is
    whether the NaN payload bits match — part (d) puts them outside the
    contract, and no TensorForge surface has ever specified them."""
    left, right = _special_operands(ALL_SPECIALS, 16)
    fast, generic, took_row_sweep = product_through_both_paths(left, right)
    assert took_row_sweep
    assert agrees_under_the_numerical_contract(fast, generic)
    # NaN-*ness* itself agrees exactly, which is the part that matters.
    assert np.array_equal(np.isnan(fast), np.isnan(generic))
    assert np.array_equal(np.isinf(fast), np.isinf(generic))
    assert np.any(np.isnan(fast)), "the NaN case never arose"
    # ...and every element that is not NaN is bit-identical.
    finite_mask = ~np.isnan(fast)
    assert np.array_equal(bits(fast)[finite_mask], bits(generic)[finite_mask])


def quiet_nan_with(payload, negative=False):
    """A quiet NaN carrying a chosen payload, so several distinguishable
    NaNs can be placed in one problem."""
    raw = np.uint64(0x7FF8000000000000) | np.uint64(payload)
    if negative:
        raw |= np.uint64(0x8000000000000000)
    return np.array([raw], dtype=np.uint64).view(np.float64)[0]


def require_contract(label, left, right, expect_a_nan):
    """Run one problem through both production paths and enforce parts
    (b) and (c) — and, when the result is NaN-free, the strong claim."""
    fast, generic, took_row_sweep = product_through_both_paths(left, right)
    assert took_row_sweep, f"{label}: the optimized path did not run"
    assert agrees_under_the_numerical_contract(fast, generic), label
    assert np.array_equal(np.isnan(fast), np.isnan(generic)), label
    assert np.array_equal(np.isinf(fast), np.isinf(generic)), label
    for produced in (fast, generic):
        nans = produced[np.isnan(produced)]
        raw = np.ascontiguousarray(nans).view(np.uint64)
        assert np.all((raw & np.uint64(0x0008000000000000)) != 0), label
    assert bool(np.any(np.isnan(fast))) is expect_a_nan, label
    if not expect_a_nan:
        assert same_bits(fast, generic), label
        assert same_bits(fast, cpp.matmul(left, right)), label
    return fast, generic


@needs_native
def test_the_contract_holds_with_a_nan_in_the_left_operand():
    """A NaN in the left operand contaminates a whole output row."""
    left = rng(90).standard_normal((4, 6))
    right = rng(91).standard_normal((6, 16))
    left[1, 2] = quiet_nan_with(0x0111111111111)
    fast, _ = require_contract("nan in the left operand", left, right, True)
    assert np.all(np.isnan(fast[1])), "the NaN did not reach its whole row"
    assert not np.any(np.isnan(fast[[0, 2, 3]])), "the NaN escaped its row"


@needs_native
def test_the_contract_holds_with_a_nan_in_the_right_operand():
    """A NaN in the right operand contaminates a whole output column."""
    left = rng(92).standard_normal((4, 6))
    right = rng(93).standard_normal((6, 16))
    right[3, 5] = quiet_nan_with(0x0222222222222)
    fast, _ = require_contract("nan in the right operand", left, right, True)
    assert np.all(np.isnan(fast[:, 5])), "the NaN did not reach its column"
    assert not np.any(np.isnan(np.delete(fast, 5, axis=1)))


@needs_native
def test_the_contract_holds_with_multiple_nan_payloads_and_signs():
    """Several distinct payloads, including a negative NaN, meeting inside
    one accumulation from both operands at once — the only situation in
    which payload selection is even reachable."""
    left = rng(94).standard_normal((4, 6))
    right = rng(95).standard_normal((6, 16))
    left[0, 0] = quiet_nan_with(0x0111111111111)
    left[0, 4] = quiet_nan_with(0x0333333333333, negative=True)
    right[2, 0] = quiet_nan_with(0x0222222222222)
    right[5, 0] = quiet_nan_with(0x0444444444444, negative=True)
    fast, generic = require_contract("multiple payloads", left, right, True)
    # Both results really are NaN at the contested element, whatever the
    # payload; that is the whole of what the contract promises there.
    assert np.isnan(fast[0, 0]) and np.isnan(generic[0, 0])


@needs_native
def test_the_contract_holds_for_a_nan_manufactured_by_the_arithmetic():
    """``0 * inf`` and ``inf - inf`` produce a NaN neither operand
    carried, which is the case a NaN-in-the-input scan would miss."""
    left = rng(96).standard_normal((4, 6))
    right = rng(97).standard_normal((6, 16))
    left[2, 1] = 0.0
    right[1, 3] = np.inf
    left[3, 0], left[3, 1] = np.inf, -np.inf
    require_contract("manufactured nan", left, right, True)


@needs_native
def test_the_contract_holds_for_denormals_and_largest_finite_values():
    """No NaN can arise, so the strong claim applies: exact bit identity
    across signed zeros, denormals, the smallest normal, and magnitudes
    large enough to matter without overflowing."""
    values = (0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308,
              1e150, -1e150, 0.5)
    left = np.array([[values[(i + k) % 8] for k in range(6)]
                     for i in range(4)], dtype=np.float64)
    right = np.array([[values[(k + j + 3) % 8] for j in range(16)]
                      for k in range(6)], dtype=np.float64)
    require_contract("denormals and huge finite values", left, right, False)


@needs_native
def test_every_nan_produced_is_quiet():
    """Neither path may hand back a signaling NaN."""
    left, right = _special_operands(ALL_SPECIALS, 16)
    fast, generic, _ = product_through_both_paths(left, right)
    for produced in (fast, generic):
        nans = produced[np.isnan(produced)]
        raw = np.ascontiguousarray(nans).view(np.uint64)
        assert np.all((raw & np.uint64(0x0008000000000000)) != 0)


@needs_native
def test_infinities_alone_still_satisfy_the_contract():
    """Infinities without any NaN operand: the products are ±inf and
    finite values, and any NaN in the result comes from ``inf - inf``,
    which both paths reach at the same k."""
    left, right = _special_operands(
        (1.0, -1.0, np.inf, -np.inf, 2.0, 0.5), 12)
    fast, generic, _ = product_through_both_paths(left, right)
    assert agrees_under_the_numerical_contract(fast, generic)
    assert np.array_equal(np.isinf(fast), np.isinf(generic))


# ==========================================================================
# 5. Shape and layout contract — unchanged by H2
# ==========================================================================

@needs_native
def test_zero_sized_dimensions_are_still_rejected_by_the_representation():
    """The native tensor representation rejects zero-size dimensions
    outright, so no empty operand can reach the kernel from Python. H2
    did not change that, and its ``n >= 1`` precondition is therefore a
    defensive statement rather than a reachable branch from here."""
    storage = cpp.NativeStorage(8)
    try:
        with pytest.raises(ValueError):
            cpp.NativeTensorCore(storage, (0, 4))
        with pytest.raises(ValueError):
            cpp.NativeTensorCore(storage, (4, 0))
    finally:
        storage.close()


@needs_native
def test_dimension_and_operand_rejections_are_unchanged():
    a = cpp.NativeTensorCore.from_array(np.ones((3, 2)))
    b = cpp.NativeTensorCore.from_array(np.ones((2, 16)))
    vector = cpp.NativeTensorCore.from_array(np.ones(2))
    mismatched = cpp.NativeTensorCore.from_array(np.ones((4, 16)))
    try:
        with pytest.raises(TypeError):
            a.matmul(np.ones((2, 16)))
        with pytest.raises(ValueError):
            a.matmul(vector)
        with pytest.raises(ValueError):
            vector.matmul(b)
        with pytest.raises(ValueError):
            a.matmul(mismatched)
    finally:
        for core in (a, b, vector, mismatched):
            core.close()


@needs_native
def test_one_element_dimensions():
    """A single row, a single column, and a single k — each of which makes
    one of the row sweep's loops degenerate."""
    for m, n, p in ((1, 5, 16), (16, 1, 16), (16, 5, 1), (1, 1, 16)):
        left = rng(m * 7 + n).standard_normal((m, n))
        right = rng(p * 7 + n).standard_normal((n, p))
        core_left = cpp.NativeTensorCore.from_array(left)
        core_right = cpp.NativeTensorCore.from_array(right)
        try:
            out = core_left.matmul(core_right)
            try:
                assert out.shape == (m, p)
                assert same_bits(out.to_numpy(), cpp.matmul(left, right))
            finally:
                out.close()
        finally:
            core_left.close()
            core_right.close()


@needs_native
def test_output_is_fresh_owning_and_contiguous_and_operands_are_unmutated():
    left = rng(33).standard_normal((6, 5))
    right = rng(34).standard_normal((5, 16))
    core_left = cpp.NativeTensorCore.from_array(left)
    core_right = cpp.NativeTensorCore.from_array(right)
    try:
        out = core_left.matmul(core_right)
        try:
            assert out.contiguous and out.offset == 0
            assert out.storage is not core_left.storage
            assert out.storage is not core_right.storage
            assert out.strides == (16, 1)
            assert np.array_equal(core_left.to_numpy(), left)
            assert np.array_equal(core_right.to_numpy(), right)
            # Writing the result cannot touch either operand.
            out.storage.fill(0.0)
            assert np.array_equal(core_left.to_numpy(), left)
            assert np.array_equal(core_right.to_numpy(), right)
        finally:
            out.close()
    finally:
        core_left.close()
        core_right.close()


@needs_native
def test_repeated_calls_are_identical():
    left = cpp.NativeTensorCore.from_array(rng(35).standard_normal((9, 7)))
    right = cpp.NativeTensorCore.from_array(rng(36).standard_normal((7, 16)))
    try:
        first = left.matmul(right)
        try:
            baseline = first.to_numpy().copy()
        finally:
            first.close()
        for _ in range(5):
            again = left.matmul(right)
            try:
                assert same_bits(again.to_numpy(), baseline)
            finally:
                again.close()
    finally:
        left.close()
        right.close()


@needs_native
def test_a_squared_operand_multiplies_itself():
    """``a.matmul(a)`` — the same storage on both sides, which the kernel
    reads twice and must not disturb."""
    values = rng(37).standard_normal((16, 16))
    core = cpp.NativeTensorCore.from_array(values)
    try:
        out = core.matmul(core)
        try:
            assert same_bits(out.to_numpy(), cpp.matmul(values, values))
            assert np.array_equal(core.to_numpy(), values)
        finally:
            out.close()
    finally:
        core.close()


# ==========================================================================
# 6. Ownership, failure, and live storage
# ==========================================================================

@pytest.fixture
def live_storages(monkeypatch):
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


@needs_native
def test_repeated_success_cycles_return_live_storage_to_baseline(live_storages):
    left = cpp.NativeTensorCore.from_array(rng(38).standard_normal((8, 6)))
    right = cpp.NativeTensorCore.from_array(rng(39).standard_normal((6, 16)))
    try:
        baseline = len(live_storages)
        for _ in range(25):
            out = left.matmul(right)
            out.close()
            assert len(live_storages) == baseline
    finally:
        left.close()
        right.close()


@needs_native
def test_a_failing_kernel_call_closes_the_output(live_storages, monkeypatch):
    """H1 requires that no uninitialized destination reaches a caller
    after a failure. H2 did not change that, on either path."""
    left = cpp.NativeTensorCore.from_array(rng(40).standard_normal((8, 6)))
    right = cpp.NativeTensorCore.from_array(rng(41).standard_normal((6, 16)))
    try:
        baseline = len(live_storages)
        original = left.storage._lib.tf_core_matmul

        def exploding(*args, **kwargs):
            raise RuntimeError("injected native failure")

        monkeypatch.setattr(left.storage._lib, "tf_core_matmul", exploding)
        with pytest.raises(RuntimeError, match="injected native failure"):
            left.matmul(right)
        assert len(live_storages) == baseline
        monkeypatch.setattr(left.storage._lib, "tf_core_matmul", original)
        # ...and the very next call still works, on the fast path.
        out = left.matmul(right)
        try:
            assert np.allclose(out.to_numpy(),
                               left.to_numpy() @ right.to_numpy(), atol=1e-10)
        finally:
            out.close()
        assert len(live_storages) == baseline
    finally:
        left.close()
        right.close()


@needs_native
def test_closed_operands_are_still_rejected_before_any_allocation(live_storages):
    left = cpp.NativeTensorCore.from_array(rng(42).standard_normal((8, 6)))
    right = cpp.NativeTensorCore.from_array(rng(43).standard_normal((6, 16)))
    right.close()
    try:
        baseline = len(live_storages)
        with pytest.raises(RuntimeError):
            left.matmul(right)
        assert len(live_storages) == baseline
    finally:
        left.close()


# ==========================================================================
# 7. Autograd, Linear, optimizer, and training parity
# ==========================================================================

def _linear_pair(in_features, out_features, seed):
    """Two independently constructed but identically seeded layers."""
    return (NativeLinear(in_features, out_features, seed=seed),
            NativeLinear(in_features, out_features, seed=seed))


@needs_native
@pytest.mark.parametrize("m,n,p", [(6, 5, 16), (6, 5, 7), (9, 12, 8)])
def test_native_tensor_forward_and_backward_agree_across_paths(m, n, p):
    """The full autograd node, run on operands that qualify and on the
    same logical values through a layout that does not. Both the forward
    and **both** gradients must be bit-identical."""
    left_values = rng(m * 3 + p).standard_normal((m, n))
    right_values = rng(p * 3 + n).standard_normal((n, p))
    upstream = rng(99).standard_normal((m, p))

    def run(strided_right):
        left = NativeTensor.from_array(left_values, requires_grad=True)
        if strided_right:
            base = NativeTensor.from_array(
                np.ascontiguousarray(right_values.T), requires_grad=True)
            right = base.transpose(1, 0)
            owned = [left, base, right]
        else:
            right = NativeTensor.from_array(right_values, requires_grad=True)
            base = right
            owned = [left, right]
        out = right  # placeholder so `finally` always has something valid
        try:
            out = left.matmul(right)
            seed = NativeTensor.from_array(upstream)
            try:
                out.backward(seed)
            finally:
                seed.close()
            forward = out.to_numpy().copy()
            left_grad = left.grad.to_numpy().copy()
            right_grad = base.grad.to_numpy().copy()
            if strided_right:
                right_grad = np.ascontiguousarray(right_grad.T)
            return forward, left_grad, right_grad
        finally:
            if out is not right and not out._closed:
                out.close()
            for tensor in owned:
                if not tensor._closed:
                    tensor.close()

    fast = run(False)
    generic = run(True)
    for produced, expected, label in zip(fast, generic,
                                         ("forward", "grad_a", "grad_b")):
        assert same_bits(produced, expected), label
    assert np.allclose(fast[0], left_values @ right_values, atol=1e-10)
    assert np.allclose(fast[1], upstream @ right_values.T, atol=1e-9)
    assert np.allclose(fast[2], left_values.T @ upstream, atol=1e-9)


@needs_native
def test_linear_forward_and_backward_match_numpy_and_are_reproducible():
    """A real layer, whose forward is ``input @ weight`` with a contiguous
    weight — the shape the row sweep was chosen for — and whose backward
    feeds the kernel a transposed operand on each side."""
    inputs = rng(50).standard_normal((12, 9))
    layer, twin = _linear_pair(9, 16, seed=5)
    try:
        results = []
        for module in (layer, twin):
            x = NativeTensor.from_array(inputs, requires_grad=True)
            try:
                out = module(x)
                seed = NativeTensor.from_array(np.ones((12, 16)))
                try:
                    out.backward(seed)
                finally:
                    seed.close()
                results.append((
                    out.to_numpy().copy(),
                    x.grad.to_numpy().copy(),
                    module.weight.grad.to_numpy().copy(),
                    module.bias.grad.to_numpy().copy(),
                ))
                out.close()
            finally:
                x.close()
        for produced, expected in zip(results[0], results[1]):
            assert same_bits(produced, expected)
        weight = layer.weight.to_numpy()
        bias = layer.bias.to_numpy()
        assert np.allclose(results[0][0], inputs @ weight + bias, atol=1e-10)
        assert np.allclose(results[0][2], inputs.T @ np.ones((12, 16)),
                           atol=1e-9)
    finally:
        for module in (layer, twin):
            close_module(module)


@needs_native
@pytest.mark.parametrize("optimizer_class", [NativeSGD, NativeAdam])
def test_optimizer_updates_after_matmul_gradients_are_reproducible(
        optimizer_class):
    """Two independent runs of the same deterministic loop must produce
    the same parameters bit for bit, including through the optimizer."""
    inputs = rng(60).standard_normal((16, 8))
    targets = rng(61).standard_normal((16, 12))

    def run():
        layer = NativeLinear(8, 12, seed=9)
        optimizer = optimizer_class(layer.parameters(), lr=0.01)
        loss_fn = NativeMSELoss()
        losses = []
        try:
            for _ in range(6):
                optimizer.zero_grad()
                x = NativeTensor.from_array(inputs)
                y = NativeTensor.from_array(targets)
                try:
                    prediction = layer(x)
                    loss = loss_fn(prediction, y)
                    losses.append(float(loss.to_numpy()))
                    loss.backward()
                    loss.close()
                    prediction.close()
                finally:
                    x.close()
                    y.close()
                optimizer.step()
            return losses, layer.weight.to_numpy().copy(), \
                layer.bias.to_numpy().copy()
        finally:
            close_module(layer)

    first = run()
    second = run()
    assert first[0] == second[0]
    assert same_bits(first[1], second[1])
    assert same_bits(first[2], second[2])
    assert first[0][-1] < first[0][0], "the loop did not learn"


# ==========================================================================
# 8. Scope — no new export, no capability move, no dispatch control
# ==========================================================================

# H2 adds no exported symbol: Phase H's surface is exactly what H1 left.
# The live library exports two more — Phase I milestone I1's typed storage
# creators — so the Phase-H claim is checked against the Phase-H subset.
PHASE_H_TF_EXPORTS = 52
EXPECTED_TF_EXPORTS = 54

# Names that would constitute a runtime dispatch control. None may exist
# in the shipped library or the installed Python backend.
FORBIDDEN_DISPATCH_NAMES = (
    "tf_matmul_set_block_size",
    "tf_matmul_set_kernel",
    "tf_core_matmul_blocked",
    "tf_core_matmul_generic",
    "tf_core_matmul_row_sweep",
    "tf_matmul_select",
    "tf_matmul_last_path",
    "tf_matmul_which_kernel",
    "tf_set_matmul_path",
)


@needs_native
def test_the_loaded_library_exports_no_matmul_dispatch_control():
    """The decisive check: ask the **loaded** library for each forbidden
    symbol through the platform loader."""
    library = cpp._require_library()
    for name in FORBIDDEN_DISPATCH_NAMES:
        with pytest.raises(AttributeError):
            getattr(library, name)
    # The internal kernels and the predicate are hidden-visibility C++ and
    # are not exported either — which is why the C++ CTest compiles the
    # source in rather than linking the library.
    for name in ("matmul_row_sweep", "matmul_generic_strided",
                 "matmul_prefers_row_sweep"):
        with pytest.raises(AttributeError):
            getattr(library, name)
    # ...and the one symbol that legitimately exists still resolves, so
    # the probe is proved able to find a symbol that is there.
    assert getattr(library, "tf_core_matmul") is not None


@needs_native
def test_h2_added_no_exported_symbol():
    import test_native_storage_allocation as h1

    image, names = h1.exported_names(cpp._LIBRARY_PATH)
    if names is None:
        pytest.skip("this image format is not parsed here")
    exported = sorted(name for name in names if name.startswith("tf_"))
    assert len(exported) == EXPECTED_TF_EXPORTS, exported
    assert len(h1.phase_h_export_names(exported)) == PHASE_H_TF_EXPORTS
    assert "tf_core_matmul" in exported
    assert not [name for name in exported if "row_sweep" in name]
    assert not [name for name in exported
                if "block" in name.lower() and name != "tf_matmul_tiled"]


def test_no_environment_variable_or_dispatch_hook_exists_in_the_sources():
    """A structural pin. The dispatch must stay a pure function of the
    metadata the kernel already receives — never of ambient process
    state."""
    sources = list((REPO_ROOT / "cpp" / "src").glob("*.cpp"))
    sources += list((REPO_ROOT / "cpp" / "include").glob("*.h"))
    sources += [REPO_ROOT / "src" / "tensorforge" / "backends" / "cpp.py"]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for forbidden in ("getenv", "GetEnvironmentVariable", "std::thread",
                          "omp_", "#pragma omp", "__cpuid", "cpuid("):
            assert forbidden not in text, f"{path.name}: {forbidden}"


def test_the_matmul_source_ships_both_paths_and_one_predicate():
    """The retained generic reference path (§8.3) is shipped code, not a
    comment or a test fixture, and the export routes through the
    predicate.

    Phase I, milestone I4 made both paths templates over the element type,
    so their **definitions** moved into tf_matmul_internal.h — the ordinary
    reason a template must, and the same place H8's elementwise traversals
    already live. What the definitions say did not change, and this test
    checks that too: the two load-bearing single lines are asserted at
    ``T``, where ``T = double`` reproduces the pre-I4 literals exactly.
    """
    source = (REPO_ROOT / "cpp" / "src" / "matmul.cpp").read_text(
        encoding="utf-8")
    header = (REPO_ROOT / "cpp" / "include" / "tf_matmul_internal.h").read_text(
        encoding="utf-8")
    assert "void matmul_generic_strided(" in header
    assert "void matmul_row_sweep(" in header
    assert "bool matmul_prefers_row_sweep(" in source
    # One dispatch helper chooses between them, and calls each exactly once.
    assert source.count("tf::matmul_prefers_row_sweep(m, n, p, b_stride1)") == 1
    assert source.count("tf::matmul_row_sweep(") == 1
    assert source.count("tf::matmul_generic_strided(") == 1
    # The k == 0 assigning pass is what makes the H1 uninitialized
    # destination safe, and the explicit `T(0) +` is what preserves the
    # sign of a zero result. Both are load-bearing single lines.
    assert "out[j] = T(0) + a_ik * b_row[j];" in header
    assert "out[j] += a_ik * b_row[j];" in header
    # The accumulator follows the element type at both widths: no `double`
    # local survives in either kernel, which is what makes "float32
    # accumulates in float32" (design §10.1) a property of the source
    # rather than of a comment. Asserted over the *code*, with the comment
    # lines stripped — the prose above the kernels quotes the pre-I4
    # spellings on purpose, to record what they became.
    code = "\n".join(line for line in header.splitlines()
                     if not line.lstrip().startswith("//"))
    assert "T sum = T(0);" in code
    assert "double" not in code, "a binary64 local survived in a typed kernel"
    assert "float" not in code, "a binary32 local was hard-coded"


@needs_native
def test_no_capability_registry_moved():
    """H2 is a memory-access change. Nothing about what the native line
    supports moved."""
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert "matmul" in cpp.TENSOR_CORE_OPS
    assert "matmul" in cpp.AUTOGRAD_OPS
    assert cpp.TENSOR_CORE_KERNELS == ("relu", "add", "subtract", "multiply",
                                       "matmul")
    # The two raw-buffer benchmark kernels are still exposed, unchanged
    # and still on no production path.
    assert "matmul" in cpp.RAW_KERNELS and "matmul_tiled" in cpp.RAW_KERNELS


@needs_native
def test_the_legacy_raw_tiled_kernel_is_unchanged_and_not_a_production_path():
    """H2 measured ``tf_matmul_tiled``'s shape and did **not** adopt it:
    it takes plain contiguous buffers rather than storage handles, carries
    no guard or error contract, and accumulates into its destination after
    zeroing it — a full extra write pass, which is exactly what H1
    removed. It stays as the standing cache-blocking experiment its
    benchmark and tests measure."""
    a = rng(70).standard_normal((16, 12))
    b = rng(71).standard_normal((12, 20))
    assert np.allclose(cpp.matmul_tiled(a, b), a @ b, atol=1e-10)
    assert same_bits(cpp.matmul_tiled(a, b, block_size=8), cpp.matmul(a, b))
    source = (REPO_ROOT / "src" / "tensorforge" / "backends"
              / "cpp.py").read_text(encoding="utf-8")
    # The production Core matmul calls tf_core_matmul and nothing else.
    core_matmul = source.split("    def matmul(self, other):")[1]
    core_matmul = core_matmul.split("    # -- reductions")[0]
    assert "tf_core_matmul" in core_matmul
    assert "tf_matmul_tiled" not in core_matmul
    assert "tf_matmul(" not in core_matmul
