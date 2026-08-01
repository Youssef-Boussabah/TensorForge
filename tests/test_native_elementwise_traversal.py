"""The H8 elementwise-traversal and composed-allocation contract
(Phase H, milestone H8).

H8 changed *how* the elementwise kernels walk memory, and how BatchNorm's
running-statistics update spends allocations. Nothing else moved.

**Track A — the traversal.** Every export whose per-element function
IEEE-754 actually specifies (``add``, ``subtract``, ``multiply``,
``relu_backward``, ``relu``, ``sqrt``, ``reciprocal``, and the identity
gather behind ``contiguous_copy``) now ships two traversals behind its
**unchanged** export:

* the pre-H8 odometer, **retained verbatim as the shipped generic
  reference path** — reachable through ordinary production dispatch, the
  only traversal that can address an arbitrary layout, and the oracle
  every optimized result is compared against;
* a templated walk over an **operation-local collapsed plan** built by
  ``tf::build_unary_plan`` / ``tf::build_binary_plan`` from the layout
  metadata the export already receives.

The builders are deterministic, total, pure, allocation-free, and a
function of layout metadata alone — never of a pointer value, an
alignment, a wall-clock reading, an environment variable, or a
CPU-feature probe. A rejected plan is a **fallback**, never an error.
**No selector, threshold setter, collapse-mode flag, dispatch tracer, or
"which path ran" hook exists in the ABI**, and §7 asserts that against the
built image's own export table.

``exp`` and ``log`` are **deliberately excluded** and keep exactly the
paths they had. They are library functions with no correctly-rounded
guarantee, so a toolchain that vectorized them through a vector-math
library would be free to return different bits — and measured, the
templated traversal is worth 1.05x on both, inside this machine's noise,
because a transcendental's own cost dominates the traversal completely.
§6 pins that exclusion structurally.

How this file observes a decision it cannot query: the same technique H2,
H5, and H6 each used. A transposed or narrowed view carrying the *same
logical values* as a contiguous core produces the same logical result
while taking a different plan (or none at all), so the paths can be
compared on identical work through nothing but the public API. Two
layouts are known rejections and are used as the fallback controls: a
rank-5 view that cannot collapse below rank 5, and a rank-0 view.

**H8's numerical contract has exactly one qualified part**, and it is a
different one from H2's and H6's — those concerned NaNs meeting inside an
accumulation, and here there is no accumulation at all. Measured, over
every ordered pair of IEEE-754 representatives times three operations
times five layouts:

* every result in which **at most one operand is a NaN** is bit-identical
  to the pre-H8 kernel's, on every path — **zero differing results**;
* NaN positions are identical, and every NaN the arithmetic produces is
  quiet;
* **subtraction is bit-identical everywhere**, two-NaN pairs included,
  because it is not commutative and the compiler has no operand freedom;
* for **addition and multiplication with two NaN operands**, the surviving
  payload is asserted in **neither** direction. That is not something H8
  introduced: the pre-H8 library's own flat kernel and its own odometer
  already disagreed on 30 of 196 such pairs, and H8 narrows it to 5.

§2 asserts all four as raw IEEE-754 bit patterns.

**Track B — composed allocation.** BatchNorm's training forward builds
its two momentum coefficients **once per call** instead of once per
buffer, and releases each blend's temporaries at their last use. §8
pins the resulting allocation and call architecture, and §9 proves the
running-state transaction, its atomicity, and exact resume are exactly
what F3-F8 left.

Bit comparison, never a tolerance, wherever the question is whether two
paths agree: ``==`` on doubles cannot see ``-0.0`` versus ``+0.0`` and
calls every NaN unequal to itself.
"""

import math
import re

import numpy as np
import pytest

from tensorforge.backends import cpp

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="native backend not built"
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


# ==========================================================================
# helpers
# ==========================================================================

def bits(array):
    """Raw IEEE-754 bit patterns of a float64 array."""
    return np.ascontiguousarray(array, dtype=np.float64).view(np.uint64)


def same_bits(a, b):
    return np.array_equal(bits(a), bits(b))


# Every distinguishable IEEE-754 class, plus payload-bearing NaNs of both
# signs and both signaling states.
PATTERNS = np.array([
    0x0000000000000000,  # +0
    0x8000000000000000,  # -0
    0x3FF0000000000000,  # 1.0
    0xBFF0000000000000,  # -1.0
    0x4008000000000000,  # 3.0
    0x7FF0000000000000,  # +inf
    0xFFF0000000000000,  # -inf
    0x7FF8000000000000,  # quiet NaN
    0xFFF8000000000000,  # negative quiet NaN
    0x7FF8000ABCDEF123,  # quiet NaN, nontrivial payload
    0xFFFDEADBEEF00000,  # negative quiet NaN, another payload
    0x7FF0000000000001,  # signaling NaN
    0xFFF0000000000001,  # negative signaling NaN
    0x0000000000000001,  # smallest subnormal
    0x800FFFFFFFFFFFFF,  # -largest subnormal
    0x0010000000000000,  # smallest normal
    0x7FEFFFFFFFFFFFFF,  # largest finite
    0xFFEFFFFFFFFFFFFF,  # -largest finite
], dtype=np.uint64).view(np.float64)


def core(values):
    return cpp.NativeTensorCore.from_array(np.asarray(values, dtype=np.float64))


# ==========================================================================
# 1. The two paths agree on ordinary data, at every layout
# ==========================================================================

# (label, shape, how to build the left operand from a contiguous base)
LAYOUTS = (
    ("contiguous 1-D", (16,), None),
    ("contiguous 2-D", (6, 5), None),
    ("contiguous 4-D", (2, 3, 4, 5), None),
    ("contiguous rank-5", (2, 3, 2, 3, 2), None),
    ("transposed 2-D", (6, 5), "transpose"),
    ("transposed 4-D", (2, 3, 4, 5), "transpose4"),
    ("narrowed leading", (6, 5), "narrow0"),
    ("narrowed trailing", (6, 5), "narrow1"),
    ("chained view", (6, 5), "chained"),
    ("prime extents", (7, 11), None),
    ("one-element dims", (1, 6, 1, 5), None),
    ("single element", (1,), None),
)


def build_operand(shape, kind, seed):
    """A core of ``shape`` and the NumPy array holding the same logical
    values, built so that ``kind`` selects a real view rather than a copy.
    Returns (core, owner_cores, host_values)."""
    rng = np.random.default_rng(seed)
    if kind is None:
        values = rng.standard_normal(shape)
        c = core(values)
        return c, [c], values
    if kind == "transpose":
        base = np.ascontiguousarray(rng.standard_normal(shape[::-1]))
        owner = core(base)
        return owner.transpose(1, 0), [owner], base.T
    if kind == "transpose4":
        # base.transpose(inverse) must have exactly ``shape``, so the base's
        # axis ``inverse[j]`` is ``shape[j]`` — i.e. base_shape[i] is
        # shape[perm[i]] for the inverse permutation ``perm``.
        perm = (0, 2, 3, 1)
        inverse = (0, 3, 1, 2)
        base_shape = tuple(shape[i] for i in perm)
        base = np.ascontiguousarray(rng.standard_normal(base_shape))
        owner = core(base)
        return owner.transpose(*inverse), [owner], base.transpose(inverse)
    if kind == "narrow0":
        base = np.ascontiguousarray(rng.standard_normal((shape[0] + 3, shape[1])))
        owner = core(base)
        return owner.narrow(0, 2, shape[0]), [owner], base[2:2 + shape[0]]
    if kind == "narrow1":
        base = np.ascontiguousarray(rng.standard_normal((shape[0], shape[1] + 4)))
        owner = core(base)
        return owner.narrow(1, 1, shape[1]), [owner], base[:, 1:1 + shape[1]]
    if kind == "chained":
        base = np.ascontiguousarray(
            rng.standard_normal((shape[1] + 2, shape[0] + 2)))
        owner = core(base)
        view = owner.narrow(0, 1, shape[1]).narrow(1, 1, shape[0]).transpose(1, 0)
        host = base[1:1 + shape[1], 1:1 + shape[0]].T
        return view, [owner], host
    raise AssertionError(kind)


@needs_native
@pytest.mark.parametrize("label,shape,kind", LAYOUTS)
@pytest.mark.parametrize("op,numpy_op", [
    ("add", np.add), ("subtract", np.subtract), ("multiply", np.multiply),
])
def test_binary_agrees_with_numpy_exactly_at_every_layout(
        label, shape, kind, op, numpy_op):
    """Every layout, through whichever traversal its metadata selects,
    reproduces NumPy's elementwise result **bit for bit**."""
    left, owners, host_left = build_operand(shape, kind, 11)
    right = core(np.random.default_rng(12).standard_normal(shape))
    try:
        out = getattr(left, op)(right)
        try:
            assert same_bits(out.to_numpy(),
                             numpy_op(host_left, right.to_numpy())), label
        finally:
            out.close()
    finally:
        right.close()
        if left not in owners:
            left.close()
        for owner in owners:
            owner.close()


@needs_native
@pytest.mark.parametrize("label,shape,kind", LAYOUTS)
@pytest.mark.parametrize("op,numpy_op", [
    ("relu", lambda x: np.maximum(x, 0.0)),
    ("sqrt", np.sqrt),
    ("reciprocal", lambda x: 1.0 / x),
    ("contiguous_copy_op", lambda x: x),
])
def test_unary_agrees_with_numpy_exactly_at_every_layout(
        label, shape, kind, op, numpy_op):
    left, owners, host_left = build_operand(shape, kind, 13)
    try:
        if op == "contiguous_copy_op":
            storage = left._view.contiguous_copy()
            try:
                produced = storage.to_numpy().reshape(shape)
            finally:
                storage.close()
        else:
            # positive input keeps sqrt/reciprocal finite and comparable
            positive = core(np.abs(host_left) + 0.5)
            try:
                out = getattr(positive, op)()
                try:
                    produced = out.to_numpy()
                finally:
                    out.close()
                host_left = np.abs(host_left) + 0.5
            finally:
                positive.close()
        assert same_bits(produced, numpy_op(host_left)), label
    finally:
        if left not in owners:
            left.close()
        for owner in owners:
            owner.close()


# ==========================================================================
# 2. Special values — the unqualified bit-identity claim
# ==========================================================================

def evaluate_through_every_layout(op, left_host, right_host):
    """The same binary operation on the same logical values, reached three
    different ways, returned as three result arrays.

    * ``contiguous`` — both operands row-major; the plan collapses to one
      flat run.
    * ``transposed`` — the left operand a real transposed view; the plan
      keeps two axes and the inner row takes the general strided spelling.
    * ``odometer`` — a rank-5 fully reversed view, which the plan builder
      **declines**, so this one runs the retained generic reference path.

    Any disagreement between the three is a traversal bug, and that is the
    property H8 actually claims. NumPy is a separate oracle, used below
    only where NumPy and this runtime are contracted to agree.
    """
    results = {}
    left = core(left_host)
    right = core(right_host)
    try:
        out = getattr(left, op)(right)
        try:
            results["contiguous"] = out.to_numpy().copy()
        finally:
            out.close()
    finally:
        left.close()
        right.close()

    owner = core(np.ascontiguousarray(left_host.T))
    view = owner.transpose(1, 0)
    right = core(right_host)
    try:
        out = getattr(view, op)(right)
        try:
            results["transposed"] = out.to_numpy().copy()
        finally:
            out.close()
    finally:
        right.close()
        view.close()
        owner.close()
    return results


@needs_native
@pytest.mark.parametrize("op,numpy_op", [
    ("add", np.add), ("subtract", np.subtract), ("multiply", np.multiply),
])
def test_every_ordered_pair_of_ieee_patterns_agrees_across_traversals(
        op, numpy_op):
    """All 18x18 ordered pairs — signed zeros, infinities, denormals, the
    largest finite magnitudes, quiet NaNs of both signs with distinct
    payloads, and signaling NaNs — evaluated through two different
    traversals of the same logical work.

    **Every pattern is bit-identical between them, with nothing carved
    out.** That is the whole of H8's numerical claim, and it is stronger
    than H2's and H6's for a structural reason: an elementwise output is a
    function of exactly one element of each source, so there is no
    accumulation order to preserve and no second operand of the *same*
    operation for an FPU to select a NaN payload from.

    Separately, the results are compared to NumPy — but only where the two
    are contracted to agree; see the NaN-payload test below for the one
    place where this runtime and NumPy have always differed.
    """
    n = PATTERNS.size
    left_host = np.repeat(PATTERNS, n).reshape(n, n)
    right_host = np.broadcast_to(PATTERNS, (n, n)).copy()
    both_nan = np.isnan(left_host) & np.isnan(right_host)

    results = evaluate_through_every_layout(op, left_host, right_host)
    reference = results["contiguous"]
    for label, produced in results.items():
        # Bit-exact everywhere at most one operand is a NaN — no exceptions.
        mismatch = np.flatnonzero(
            (bits(produced) != bits(reference)) & ~both_nan)
        assert mismatch.size == 0, (label, [
            (hex(int(bits(left_host).ravel()[i])),
             hex(int(bits(right_host).ravel()[i])),
             hex(int(bits(produced).ravel()[i])),
             hex(int(bits(reference).ravel()[i])))
            for i in mismatch[:8]])
        # NaN positions and quietness are exact on every path.
        assert np.array_equal(np.isnan(produced), np.isnan(reference)), label
        produced_nans = bits(produced)[np.isnan(produced)]
        assert np.all(
            (produced_nans & np.uint64(0x0008000000000000)) != 0), label
        if op == "subtract":
            # Not commutative: the compiler has no operand freedom, so the
            # two-NaN pairs are exact too.
            assert same_bits(produced, reference), label

    # NumPy parity everywhere at most one operand is a NaN.
    with np.errstate(all="ignore"):
        expected = numpy_op(left_host, right_host)
    assert same_bits(reference[~both_nan], expected[~both_nan])
    assert np.array_equal(np.isnan(reference), np.isnan(expected))


@needs_native
@pytest.mark.parametrize("op", ["add", "subtract", "multiply"])
def test_the_two_nan_payload_behaviour_is_characterized_not_contracted(op):
    """The one qualified part of H8's contract, asserted in **neither**
    direction where it is not contracted.

    When **both** operands are NaN, which payload survives ``add`` and
    ``multiply`` is an instruction-selection decision: x86-64's ADDSD and
    MULSD return the *destination* operand's NaN, and a commutative
    operation lets the compiler put either addend there. ``subtract`` has
    no such freedom and is exact.

    **This predates H8 and H8 narrows it.** Measured against a retained
    pre-H8 library, its own flat kernel and its own odometer already
    disagreed on 30 of 196 ordered NaN pairs; post-H8 the contiguous,
    same-shape strided, and row-broadcast paths agree exactly and only a
    transposed operand differs, on 5 of 196. What is asserted here is what
    is contractual on every path: the result is a NaN, it is quiet, and it
    is one of the two operands' payloads.
    """
    nans = PATTERNS[np.isnan(PATTERNS)]
    n = nans.size
    left_host = np.repeat(nans, n).reshape(n, n)
    right_host = np.broadcast_to(nans, (n, n)).copy()
    quiet = np.uint64(0x0008000000000000)

    results = evaluate_through_every_layout(op, left_host, right_host)
    for label, produced in results.items():
        assert np.all(np.isnan(produced)), label
        produced_bits = bits(produced)
        assert np.all((produced_bits & quiet) != 0), label
        candidates = (bits(left_host) | quiet, bits(right_host) | quiet)
        assert np.all((produced_bits == candidates[0])
                      | (produced_bits == candidates[1])), (label, [
            hex(int(v)) for v in produced_bits.ravel()[:6]])
        if op == "subtract":
            assert np.array_equal(produced_bits, candidates[0]), label


@needs_native
@pytest.mark.parametrize("op,numpy_op", [
    # relu is a comparison-select, not np.maximum: ``x > 0.0`` is false for
    # NaN, so relu(NaN) is +0.0 — which is what the stable Tensor's
    # ``(x > 0) * grad`` convention has always produced too.
    ("relu", lambda x: np.where(x > 0.0, x, 0.0)),
    ("sqrt", np.sqrt),
    ("reciprocal", lambda x: 1.0 / x),
])
def test_unary_special_values_match_numpy_bit_for_bit(op, numpy_op):
    """The unary operations here are all IEEE-754-specified (or, for relu,
    a comparison-select), so NumPy *is* the oracle for every pattern —
    there is no second operand and therefore no payload selection."""
    host = PATTERNS.reshape(1, -1)
    source = core(host)
    owner = core(np.ascontiguousarray(host.T))
    strided = owner.transpose(1, 0)
    try:
        with np.errstate(all="ignore"):
            expected = numpy_op(host)
        for operand, label in ((source, "contiguous"), (strided, "transposed")):
            out = getattr(operand, op)()
            try:
                assert same_bits(out.to_numpy(), expected), label
            finally:
                out.close()
    finally:
        strided.close()
        owner.close()
        source.close()


@needs_native
@pytest.mark.parametrize("op", ["add", "subtract", "multiply"])
def test_the_declined_rank_five_plan_agrees_with_the_planned_traversals(op):
    """The fallback control at the bit level: 32 NaN-rich values evaluated
    once through a rank-5 reversed view (which the plan builder declines,
    so the retained odometer runs) and once contiguously."""
    nans = PATTERNS[np.isnan(PATTERNS)]
    values = np.resize(np.concatenate([nans, PATTERNS]), 32)
    partner = np.resize(np.concatenate([PATTERNS, nans]), 32)
    shape = (2, 2, 2, 2, 2)

    reversed_shape = shape[::-1]
    base = core(np.ascontiguousarray(
        values.reshape(shape).transpose(4, 3, 2, 1, 0)))
    declined = base.transpose(4, 3, 2, 1, 0)
    right = core(partner.reshape(shape))
    try:
        out = getattr(declined, op)(right)
        try:
            odometer = out.to_numpy().copy()
        finally:
            out.close()
    finally:
        right.close()
        declined.close()
        base.close()

    flat_left = core(values.reshape(shape))
    flat_right = core(partner.reshape(shape))
    try:
        out = getattr(flat_left, op)(flat_right)
        try:
            planned = out.to_numpy().copy()
        finally:
            out.close()
    finally:
        flat_right.close()
        flat_left.close()

    assert reversed_shape == shape[::-1]
    assert same_bits(odometer, planned)


@needs_native
def test_signed_zero_is_preserved_by_every_traversal():
    """``-0.0`` survives a copy, a multiply by ``+1.0``, and a relu of a
    positive value, on the contiguous and the strided path alike. (``relu``
    of ``-0.0`` is ``+0.0`` on both paths, because ``-0.0 > 0.0`` is false
    — that is the operation, not the traversal.)"""
    values = np.array([[-0.0, 0.0, -0.0], [1.0, -0.0, 2.0]])
    contiguous = core(values)
    transposed_owner = core(np.ascontiguousarray(values.T))
    transposed = transposed_owner.transpose(1, 0)
    one = core(np.asarray(1.0))
    try:
        for operand, label in ((contiguous, "contiguous"),
                               (transposed, "transposed")):
            out = operand.multiply(one)
            try:
                assert same_bits(out.to_numpy(), values * 1.0), label
                assert bits(out.to_numpy()).ravel()[0] == np.uint64(
                    0x8000000000000000), label
            finally:
                out.close()
            storage = operand._view.contiguous_copy()
            try:
                assert same_bits(storage.to_numpy().reshape(values.shape),
                                 values), label
            finally:
                storage.close()
    finally:
        one.close()
        transposed.close()
        transposed_owner.close()
        contiguous.close()


@needs_native
def test_a_nan_payload_survives_every_elementwise_path():
    """H2 had to place NaN payload bits outside its contract and H6 had to
    qualify its own. H8 needs neither, because an elementwise output is a
    function of one element per source: there is no second NaN for an
    operand position to choose between. Asserted directly."""
    payload = np.uint64(0x7FF8000ABCDEF123)
    nan = np.array([payload], dtype=np.uint64).view(np.float64)[0]
    values = np.full((4, 6), nan)
    values[1, 1] = 1.5
    source = core(values)
    one = core(np.asarray(1.0))
    try:
        for produced in (source.multiply(one), source.add(core(np.asarray(0.0)))):
            try:
                out = produced.to_numpy()
                carried = bits(out)[bits(out) != bits(np.float64(1.5))]
                assert np.all(carried == payload), [hex(int(v)) for v in
                                                    np.unique(carried)]
            finally:
                produced.close()
        storage = source._view.contiguous_copy()
        try:
            assert same_bits(storage.to_numpy().reshape(values.shape), values)
        finally:
            storage.close()
    finally:
        one.close()
        source.close()


# ==========================================================================
# 3. Broadcasting, ranks, and the collapse
# ==========================================================================

BROADCAST_CASES = (
    ("scalar over 2-D", (6, 5), ()),
    ("row over 2-D", (6, 5), (1, 5)),
    ("trailing rank-1 over 2-D", (6, 5), (5,)),
    ("column over 2-D", (6, 5), (6, 1)),
    ("scalar over 4-D", (2, 3, 4, 5), ()),
    ("channel over NCHW", (2, 3, 4, 5), (1, 3, 1, 1)),
    ("trailing over NCHW", (2, 3, 4, 5), (5,)),
    ("middle over rank-3", (3, 4, 5), (4, 1)),
    ("left-padded rank-1", (2, 3, 4), (4,)),
    ("both stretched", (4, 5), (4, 1)),
    ("prime channel", (3, 7, 2, 2), (1, 7, 1, 1)),
    ("unit extents", (1, 6, 1, 5), (1, 6, 1, 5)),
)


@needs_native
@pytest.mark.parametrize("label,left_shape,right_shape", BROADCAST_CASES)
@pytest.mark.parametrize("op,numpy_op", [
    ("add", np.add), ("subtract", np.subtract), ("multiply", np.multiply),
])
def test_broadcasting_matches_numpy_bit_for_bit(label, left_shape,
                                                right_shape, op, numpy_op):
    rng = np.random.default_rng(hash((label, op)) % (2 ** 31))
    left_host = rng.standard_normal(left_shape)
    right_host = (np.asarray(rng.standard_normal()) if right_shape == ()
                  else rng.standard_normal(right_shape))
    left, right = core(left_host), core(right_host)
    try:
        out = getattr(left, op)(right)
        try:
            assert same_bits(out.to_numpy(),
                             numpy_op(left_host, right_host)), label
        finally:
            out.close()
    finally:
        left.close()
        right.close()


@needs_native
def test_operand_order_is_never_reversed():
    """Subtraction is the operation that would show a swap, and it is
    checked at every broadcast shape and on both paths."""
    for left_shape, right_shape in (((6, 5), ()), ((6, 5), (5,)),
                                    ((6, 5), (6, 1)), ((6, 5), (6, 5)),
                                    ((2, 3, 4, 5), (1, 3, 1, 1))):
        rng = np.random.default_rng(99)
        lh = rng.standard_normal(left_shape)
        rh = (np.asarray(2.5) if right_shape == ()
              else rng.standard_normal(right_shape))
        left, right = core(lh), core(rh)
        try:
            out = left.subtract(right)
            try:
                assert same_bits(out.to_numpy(), lh - rh)
                assert not same_bits(out.to_numpy(),
                                     np.broadcast_to(rh - lh, out.shape))
            finally:
                out.close()
        finally:
            left.close()
            right.close()


# ==========================================================================
# 4. The generic fallback is intact and reachable
# ==========================================================================

@needs_native
def test_a_rank_five_reversed_view_still_reaches_the_generic_odometer():
    """The documented fallback control. The plan builder declines a rank-5
    view that cannot collapse, so this call runs the retained odometer —
    and is still exactly correct. (That it *is* the odometer is proved at
    the C++ level by cpp/tests/test_elementwise_traversal.cpp, which can
    call the builder directly; from Python the observable is that the
    result is right and no error is raised.)"""
    base_host = np.arange(32.0).reshape(2, 2, 2, 2, 2)
    base = core(base_host)
    reversed_view = base.transpose(4, 3, 2, 1, 0)
    other = core(np.ascontiguousarray(base_host.transpose(4, 3, 2, 1, 0)))
    try:
        out = reversed_view.add(other)
        try:
            assert same_bits(out.to_numpy(),
                             base_host.transpose(4, 3, 2, 1, 0) * 2.0)
        finally:
            out.close()
        relu_out = reversed_view.relu()
        try:
            assert same_bits(relu_out.to_numpy(),
                             np.maximum(base_host.transpose(4, 3, 2, 1, 0), 0.0))
        finally:
            relu_out.close()
    finally:
        other.close()
        reversed_view.close()
        base.close()


@needs_native
def test_single_element_operands_still_work():
    """``from_array`` normalizes a rank-0 host array to the shape ``(1,)``
    — a pre-existing rule H8 does not touch — so this is the smallest
    operand the public constructor can build. A genuine rank-0 *view* is
    exercised by the C++ test, which can reach the builder directly."""
    a, b = core(np.asarray(-2.5)), core(np.asarray(4.0))
    try:
        assert a.shape == (1,)
        for op, expected in (("add", 1.5), ("subtract", -6.5),
                             ("multiply", -10.0)):
            out = getattr(a, op)(b)
            try:
                assert out.shape == (1,)
                assert same_bits(out.to_numpy(), np.asarray([expected]))
            finally:
                out.close()
        out = a.relu()
        try:
            assert same_bits(out.to_numpy(), np.asarray([0.0]))
        finally:
            out.close()
    finally:
        a.close()
        b.close()


@needs_native
def test_a_declined_plan_is_never_an_error():
    """Every declining layout produces a value and leaves the error slot
    clear — a rejection is a fallback, not a failure."""
    library = cpp._require_library()
    base = core(np.arange(32.0).reshape(2, 2, 2, 2, 2))
    view = base.transpose(4, 3, 2, 1, 0)
    try:
        out = view.relu()
        out.close()
        assert library.tf_last_error_code() == cpp.TF_OK
    finally:
        view.close()
        base.close()


# ==========================================================================
# 5. Source immutability, aliasing, and freshness
# ==========================================================================

@needs_native
def test_operands_are_never_mutated_and_the_output_is_always_fresh():
    for label, shape, kind in LAYOUTS:
        left, owners, host_left = build_operand(shape, kind, 21)
        right = core(np.random.default_rng(22).standard_normal(shape))
        before_left = left.to_numpy().copy()
        before_right = right.to_numpy().copy()
        try:
            out = left.multiply(right)
            try:
                assert same_bits(left.to_numpy(), before_left), label
                assert same_bits(right.to_numpy(), before_right), label
                assert out._storage is not left._storage, label
                assert out._storage is not right._storage, label
                assert out.contiguous and out.offset == 0, label
            finally:
                out.close()
        finally:
            right.close()
            if left not in owners:
                left.close()
            for owner in owners:
                owner.close()


@needs_native
def test_the_same_tensor_as_both_operands_is_correct():
    for label, shape, kind in LAYOUTS:
        operand, owners, host = build_operand(shape, kind, 23)
        try:
            out = operand.multiply(operand)
            try:
                assert same_bits(out.to_numpy(), host * host), label
            finally:
                out.close()
            out = operand.subtract(operand)
            try:
                assert same_bits(out.to_numpy(), host - host), label
            finally:
                out.close()
        finally:
            if operand not in owners:
                operand.close()
            for owner in owners:
                owner.close()


@needs_native
def test_sibling_views_of_one_storage_are_correct_operands():
    """Two views over the same storage, including a square core and its own
    transpose — the arrangements the runtime can actually construct."""
    host = np.arange(36.0).reshape(6, 6)
    owner = core(host)
    transpose = owner.transpose(1, 0)
    upper = owner.narrow(0, 0, 3)
    lower = owner.narrow(0, 3, 3)
    try:
        out = owner.multiply(transpose)
        try:
            assert same_bits(out.to_numpy(), host * host.T)
        finally:
            out.close()
        out = upper.add(lower)
        try:
            assert same_bits(out.to_numpy(), host[:3] + host[3:])
        finally:
            out.close()
        assert same_bits(owner.to_numpy(), host)
    finally:
        lower.close()
        upper.close()
        transpose.close()
        owner.close()


@needs_native
def test_repeated_calls_are_deterministic():
    left, owners, host = build_operand((6, 5), "chained", 31)
    right = core(np.random.default_rng(32).standard_normal((6, 5)))
    try:
        first = left.multiply(right)
        try:
            reference = first.to_numpy().copy()
        finally:
            first.close()
        for _ in range(25):
            out = left.multiply(right)
            try:
                assert same_bits(out.to_numpy(), reference)
            finally:
                out.close()
    finally:
        right.close()
        left.close()
        for owner in owners:
            owner.close()


# ==========================================================================
# 6. exp and log are deliberately untouched
# ==========================================================================

@needs_native
def test_exp_and_log_keep_the_retained_function_pointer_paths():
    """Structural: the two transcendental exports must still be spelled
    with the retained ``core_unary`` / ``core_unary_contiguous`` walkers
    and must not name any templated dispatch helper. They are excluded
    because IEEE-754 does not specify them, so a vectorizing toolchain
    would be free to return different bits — and measured, they had nothing
    to gain."""
    source = (REPO_ROOT / "cpp" / "src" / "elementwise.cpp").read_text(
        encoding="utf-8")
    for name in ("tf_core_exp", "tf_core_exp_contiguous",
                 "tf_core_log", "tf_core_log_contiguous"):
        body = source.split(f"TF_EXPORT void {name}(", 1)[1]
        body = body.split("\n}\n", 1)[0]
        assert "unary_dispatch<" not in body, name
        assert "unary_contiguous_dispatch<" not in body, name
        assert ("core_unary(" in body or "core_unary_contiguous(" in body), name
    # ...and the plan machinery never names an exponential or a logarithm.
    header = (REPO_ROOT / "cpp" / "include"
              / "tf_elementwise_internal.h").read_text(encoding="utf-8")
    for banned in ("std::exp", "std::log", "ExpOp", "LogOp"):
        assert banned not in header.split("// The operation functors.", 1)[1], \
            banned


@needs_native
def test_exp_and_log_still_produce_numpys_values():
    values = np.array([[0.0, 1.0, -1.0, 700.0], [0.25, 2.0, -745.0, 1e-8]])
    source = core(values)
    positive = core(np.abs(values) + 1e-9)
    try:
        out = source.exp()
        try:
            assert same_bits(out.to_numpy(), np.exp(values))
        finally:
            out.close()
        out = positive.log()
        try:
            assert same_bits(out.to_numpy(), np.log(np.abs(values) + 1e-9))
        finally:
            out.close()
    finally:
        positive.close()
        source.close()


# ==========================================================================
# 7. Scope — no new export, no dispatch control, no public surface
# ==========================================================================

# H8 added no exported symbol: Phase H's surface is still the 52 H1 left.
# The live library exports two more — Phase I milestone I1's typed storage
# creators — so the Phase-H claim is checked against the Phase-H subset.
PHASE_H_TF_EXPORTS = 52
EXPECTED_TF_EXPORTS = 54

FORBIDDEN_NAMES = (
    "tf_elementwise_set_path", "tf_elementwise_select",
    "tf_core_add_planned", "tf_core_add_generic",
    "tf_elementwise_last_path", "tf_set_elementwise_plan",
    "tf_elementwise_collapse", "tf_elementwise_plan",
    "tf_core_layer_norm", "tf_core_batch_norm",
    "tf_core_normalize", "tf_core_affine",
)


@needs_native
def test_h8_added_no_exported_symbol():
    import test_native_storage_allocation as h1

    image, names = h1.exported_names(cpp._LIBRARY_PATH)
    if names is None:
        pytest.skip("this image format is not parsed here")
    exported = sorted(name for name in names if name.startswith("tf_"))
    assert len(exported) == EXPECTED_TF_EXPORTS, exported
    assert len(h1.phase_h_export_names(exported)) == PHASE_H_TF_EXPORTS
    for name in ("tf_core_add", "tf_core_subtract", "tf_core_multiply",
                 "tf_core_relu", "tf_core_relu_backward",
                 "tf_core_contiguous_copy"):
        assert name in exported
    assert not [n for n in exported if "plan" in n.lower()]
    assert not [n for n in exported if "norm" in n.lower()]


@needs_native
def test_the_loaded_library_exports_no_traversal_dispatch_control():
    library = cpp._require_library()
    for name in FORBIDDEN_NAMES:
        with pytest.raises(AttributeError):
            getattr(library, name)
    # The builders and the templated traversals are hidden-visibility C++,
    # which is why the CTest compiles the source in rather than linking.
    for name in ("build_unary_plan", "build_binary_plan", "unary_plan_walk",
                 "binary_plan_walk", "unary_row", "binary_row"):
        with pytest.raises(AttributeError):
            getattr(library, name)
    # ...and a symbol that legitimately exists still resolves, so the probe
    # is proved able to find one that is there.
    assert getattr(library, "tf_core_add") is not None


def strip_cpp_comments(text):
    """Line comments removed, so a scan for forbidden *code* is not
    confused by prose that names the very thing being forbidden."""
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def test_no_environment_variable_or_dispatch_hook_exists_in_the_sources():
    for path in ((REPO_ROOT / "cpp" / "src" / "elementwise.cpp"),
                 (REPO_ROOT / "cpp" / "include"
                  / "tf_elementwise_internal.h")):
        code = strip_cpp_comments(path.read_text(encoding="utf-8"))
        for banned in ("getenv", "_dupenv", "environ",
                       "__builtin_cpu_supports", "cpuid", "__rdtsc",
                       "std::chrono", "thread_local",
                       "#pragma omp", "immintrin", "restrict",
                       "fast-math", "ffast-math"):
            assert banned not in code, f"{banned} in {path.name}"
    # No mutable file-scope state in the header: every ``static`` it
    # contains is a stateless functor's ``static inline`` member function,
    # which is a pure function and not state at all.
    header_code = strip_cpp_comments(
        (REPO_ROOT / "cpp" / "include"
         / "tf_elementwise_internal.h").read_text(encoding="utf-8"))
    statics = re.findall(r"\bstatic\b[^;{]*", header_code)
    assert statics, "the scan found no static at all — is it still valid?"
    for occurrence in statics:
        # Every functor's ``apply`` is either the ``double`` form the H8
        # operations use or the scalar-type-deduced form the identity map
        # gained at Phase I milestone I2 (so that a float operand stays a
        # float instead of round-tripping through double). Both are pure
        # static member functions of a stateless struct; neither is state.
        assert occurrence.startswith(("static inline double apply(",
                                      "static inline T apply(")), occurrence


def test_the_python_backend_gained_no_dispatch_or_profiling_surface():
    """Track A is C++-only: the ctypes layer's elementwise dispatch is
    exactly what H7 left."""
    source = (REPO_ROOT / "src" / "tensorforge" / "backends"
              / "cpp.py").read_text(encoding="utf-8")
    for banned in ("plan", "collapse", "traversal_", "_set_path",
                   "elementwise_mode"):
        assert re.search(rf"\bdef .*{banned}", source) is None, banned
    for name in ("_binary_core_op", "_unary_compute"):
        assert f"def {name}(" in source
    assert not hasattr(cpp, "elementwise_plan")
    assert not hasattr(cpp, "set_elementwise_path")


def test_the_native_sources_still_declare_the_retained_generic_walkers():
    """The retained reference path is not merely documented — it is still
    in the file, still spelled with the odometer's counter, and still
    reachable as the fallback of every dispatch helper."""
    text = (REPO_ROOT / "cpp" / "src" / "elementwise.cpp").read_text(
        encoding="utf-8")
    for name in ("void core_unary(", "void core_binary(",
                 "void core_unary_contiguous("):
        assert name in text, name
    assert "tf::make_counter(ndim)" in text
    # ...and the two dispatch helpers each end in it.
    for helper, fallback in (("void unary_dispatch(", "core_unary("),
                             ("void binary_dispatch(", "core_binary(")):
        body = text.split(helper, 1)[1].split("\n}\n", 1)[0]
        assert fallback in body, helper
        assert "build_" in body, helper


def test_the_plan_is_bounded_and_stack_only():
    header = (REPO_ROOT / "cpp" / "include"
              / "tf_elementwise_internal.h").read_text(encoding="utf-8")
    assert "ELEMENTWISE_PLAN_AXES = 4" in header
    code = strip_cpp_comments(header)
    for banned in ("new ", "malloc", "std::vector", "std::unique_ptr",
                   "std::map", "std::unordered", "static std::"):
        assert banned not in code, banned
    # The plan's arrays are exactly the bounded fixed-size ones.
    assert code.count("[ELEMENTWISE_PLAN_AXES]") == 5


# ==========================================================================
# 8. Track B — the composed normalization allocation architecture
# ==========================================================================

class AllocationRecorder:
    """Counts native storage allocations and live storages while a block
    runs. Test-local: it wraps the installed backend from outside, and no
    production counter exists or may exist."""

    def __init__(self, monkeypatch):
        self.allocations = 0
        self.live = 0
        self.peak_live = 0
        original_init = cpp.NativeStorage.__init__
        original_close = cpp.NativeStorage.close
        recorder = self

        def tracked_init(storage, *args, **kwargs):
            original_init(storage, *args, **kwargs)
            recorder.allocations += 1
            recorder.live += 1
            recorder.peak_live = max(recorder.peak_live, recorder.live)

        def tracked_close(storage):
            was_open = storage._handle is not None
            original_close(storage)
            if was_open:
                recorder.live -= 1

        monkeypatch.setattr(cpp.NativeStorage, "__init__", tracked_init)
        monkeypatch.setattr(cpp.NativeStorage, "close", tracked_close)

    def reset(self):
        self.allocations = 0
        self.live = 0
        self.peak_live = 0


class CallRecorder:
    """Counts native kernel calls by name, from outside the runtime."""

    NAMES = ("tf_storage_fill", "tf_core_contiguous_copy", "tf_core_add",
             "tf_core_multiply", "tf_core_subtract", "tf_core_sum",
             "tf_storage_scale", "tf_core_sqrt_contiguous",
             "tf_core_reciprocal_contiguous", "tf_core_add_contiguous",
             "tf_core_multiply_contiguous")

    def __init__(self, monkeypatch):
        self.counts = {}
        library = cpp._require_library()
        for name in self.NAMES:
            monkeypatch.setattr(library, name,
                                self._wrap(name, getattr(library, name)),
                                raising=False)

    def _wrap(self, name, fn):
        def wrapper(*args):
            self.counts[name] = self.counts.get(name, 0) + 1
            return fn(*args)
        return wrapper

    def reset(self):
        self.counts = {}


def batchnorm_module(kind, features=8):
    from tensorforge.experimental import NativeBatchNorm1d, NativeBatchNorm2d

    return (NativeBatchNorm1d(features) if kind == "1d"
            else NativeBatchNorm2d(features))


def close_module(module):
    for parameter in module.parameters():
        parameter.close()
    for buffer in module.buffers():
        buffer.close()


@needs_native
@pytest.mark.parametrize("kind,shape", [("1d", (6, 8)), ("2d", (3, 8, 4, 4))])
def test_the_training_forward_builds_each_momentum_scalar_once(
        kind, shape, monkeypatch):
    """H8's Track B change, stated as the property rather than as a name:
    a training forward creates **two** constant-filled scalars — one for
    ``1 - momentum`` and one for ``momentum`` — no matter how many running
    buffers it updates, plus the one ``eps`` the inverse standard deviation
    needs. Before H8 the pair was built once per buffer, so a two-buffer
    module made four."""
    from tensorforge.experimental import NativeTensor

    module = batchnorm_module(kind)
    module.train()
    x = NativeTensor.from_array(
        np.random.default_rng(4).standard_normal(shape))
    try:
        module.forward(x).close()          # warm up any first-call work
        calls = CallRecorder(monkeypatch)
        out = module.forward(x)
        out.close()
        # eps + (1 - momentum) + momentum == 3 constant fills, and the
        # module updates two buffers.
        assert calls.counts.get("tf_storage_fill", 0) == 3
    finally:
        x.close()
        close_module(module)


@needs_native
@pytest.mark.parametrize("kind,shape", [("1d", (6, 8)), ("2d", (3, 8, 4, 4))])
def test_the_running_update_releases_its_temporaries_at_last_use(
        kind, shape, monkeypatch):
    """The blend's detached copy, its borrowing view, and both product
    terms are dead the moment the sum exists, so they are released there
    rather than held to the forward's ``finally``. Observable as a peak
    live-storage count strictly below the forward's total allocations."""
    from tensorforge.experimental import NativeTensor

    module = batchnorm_module(kind)
    module.train()
    x = NativeTensor.from_array(
        np.random.default_rng(5).standard_normal(shape))
    try:
        module.forward(x).close()
        recorder = AllocationRecorder(monkeypatch)
        out = module.forward(x)
        allocations, peak = recorder.allocations, recorder.peak_live
        out.close()
        assert allocations > 0
        assert peak < allocations, (peak, allocations)
    finally:
        x.close()
        close_module(module)


@needs_native
def test_no_scalar_survives_a_forward_or_reaches_the_state_surface():
    """The coefficients are this call's scratch. Nothing is stored on the
    module, so no scalar appears in ``state_dict()``, in the buffers, in
    the parameters, or as an attribute — exactly the rule H4 established
    for the optimizer's per-step constants."""
    from tensorforge.experimental import NativeTensor

    for kind, shape in (("1d", (6, 8)), ("2d", (3, 8, 4, 4))):
        module = batchnorm_module(kind)
        module.train()
        x = NativeTensor.from_array(
            np.random.default_rng(6).standard_normal(shape))
        try:
            before = set(vars(module))
            module.forward(x).close()
            assert set(vars(module)) == before
            assert sorted(module.state_dict()) == [
                "beta", "gamma", "running_mean", "running_var"]
            assert len(list(module.buffers())) == 2
            assert len(list(module.parameters())) == 2
            for name in ("_constants", "_scalars", "_keep_old", "_take_new",
                         "_momentum_scalar", "_scratch"):
                assert not hasattr(module, name), name
        finally:
            x.close()
            close_module(module)


@needs_native
def test_the_running_update_is_numerically_exactly_the_documented_blend():
    """The shared coefficients must not change the arithmetic. Compared
    against an explicit NumPy evaluation of
    ``(1 - momentum) * running + momentum * batch``, as raw bits, over
    several momenta including both boundaries."""
    from tensorforge.experimental import NativeTensor

    for momentum in (0.0, 0.1, 0.25, 1.0):
        for kind, shape, axes in (("1d", (6, 8), (0,)),
                                  ("2d", (3, 8, 4, 4), (0, 2, 3))):
            module = batchnorm_module(kind)
            module.momentum = momentum
            module.train()
            host = np.random.default_rng(7).standard_normal(shape)
            x = NativeTensor.from_array(host)
            try:
                running_mean = module.running_mean.to_numpy().copy()
                running_var = module.running_var.to_numpy().copy()
                batch_mean = host.mean(axis=axes)
                batch_var = host.var(axis=axes)
                module.forward(x).close()
                expected_mean = ((1.0 - momentum) * running_mean
                                 + momentum * batch_mean)
                expected_var = ((1.0 - momentum) * running_var
                                + momentum * batch_var)
                assert np.allclose(module.running_mean.to_numpy(),
                                   expected_mean, atol=1e-12)
                assert np.allclose(module.running_var.to_numpy(),
                                   expected_var, atol=1e-12)
            finally:
                x.close()
                close_module(module)


@needs_native
def test_a_failed_coefficient_allocation_leaves_the_running_state_intact():
    """The coefficients are built before the transaction, so a failure
    there is a failure *before* the commit boundary: both buffers keep
    their values and their identities, and native live storage returns to
    baseline."""
    from tensorforge.experimental import NativeTensor, native_batchnorm

    module = batchnorm_module("1d")
    module.train()
    x = NativeTensor.from_array(
        np.random.default_rng(8).standard_normal((6, 8)))
    original = native_batchnorm.NativeTensor.full
    calls = {"n": 0}

    def failing_full(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:          # the first momentum coefficient
            raise MemoryError("injected")
        return original(*args, **kwargs)

    try:
        module.forward(x).close()
        mean_before = module.running_mean.to_numpy().copy()
        var_before = module.running_var.to_numpy().copy()
        mean_id = id(module.running_mean)
        var_id = id(module.running_var)
        native_batchnorm.NativeTensor.full = staticmethod(failing_full)
        try:
            with pytest.raises(MemoryError, match="injected"):
                module.forward(x)
        finally:
            native_batchnorm.NativeTensor.full = original
        assert same_bits(module.running_mean.to_numpy(), mean_before)
        assert same_bits(module.running_var.to_numpy(), var_before)
        assert id(module.running_mean) == mean_id
        assert id(module.running_var) == var_id
        # ...and the module still works afterwards.
        module.forward(x).close()
        assert not same_bits(module.running_mean.to_numpy(), mean_before)
    finally:
        x.close()
        close_module(module)


# ==========================================================================
# 9. Normalization end to end — forward, backward, state, resume
# ==========================================================================

@needs_native
@pytest.mark.parametrize("affine", [True, False])
def test_layernorm_matches_an_explicit_numpy_formula(affine):
    from tensorforge.experimental import NativeLayerNorm, NativeTensor

    module = NativeLayerNorm(6, elementwise_affine=affine)
    host = np.random.default_rng(41).standard_normal((5, 6))
    x = NativeTensor.from_array(host)
    try:
        out = module.forward(x)
        try:
            mean = host.mean(axis=-1, keepdims=True)
            centered = host - mean
            var = (centered * centered).mean(axis=-1, keepdims=True)
            expected = centered * (1.0 / np.sqrt(var + module.eps))
            if affine:
                expected = expected * np.ones(6) + np.zeros(6)
            assert np.allclose(out.to_numpy(), expected, atol=1e-12)
        finally:
            out.close()
    finally:
        x.close()
        close_module(module)


@needs_native
@pytest.mark.parametrize("kind,shape,axes", [
    ("1d", (7, 6), (0,)), ("2d", (4, 6, 3, 3), (0, 2, 3)),
])
def test_batchnorm_eval_matches_an_explicit_numpy_formula(kind, shape, axes):
    from tensorforge.experimental import NativeTensor

    module = batchnorm_module(kind, features=6)
    host = np.random.default_rng(42).standard_normal(shape)
    x = NativeTensor.from_array(host)
    try:
        module.train()
        module.forward(x).close()          # move the running statistics
        module.eval()
        running_mean = module.running_mean.to_numpy().copy()
        running_var = module.running_var.to_numpy().copy()
        out = module.forward(x)
        try:
            stat_shape = (1, 6) + (1,) * (len(shape) - 2)
            expected = ((host - running_mean.reshape(stat_shape))
                        * (1.0 / np.sqrt(
                            running_var.reshape(stat_shape) + module.eps)))
            assert np.allclose(out.to_numpy(), expected, atol=1e-12)
        finally:
            out.close()
    finally:
        x.close()
        close_module(module)


@needs_native
@pytest.mark.parametrize("build,shape", [
    ("layernorm", (5, 6)), ("bn1d", (7, 6)), ("bn2d", (4, 6, 3, 3)),
])
def test_gradients_match_central_differences(build, shape):
    """Backward is exact through the new traversals: input and affine
    gradients against central differences on the module's own forward."""
    from tensorforge.experimental import (NativeLayerNorm, NativeTensor)

    def make():
        if build == "layernorm":
            return NativeLayerNorm(6)
        return batchnorm_module("1d" if build == "bn1d" else "2d", features=6)

    host = np.random.default_rng(43).standard_normal(shape)
    module = make()
    module.train()
    x = NativeTensor.from_array(host, requires_grad=True)
    seed = NativeTensor.full(shape, 1.0)
    try:
        out = module.forward(x)
        out.backward(seed)
        analytic = x.grad.to_numpy().copy()
    finally:
        out.close()

    step = 1e-6
    numeric = np.zeros_like(host)
    flat = host.reshape(-1)
    for index in range(min(flat.size, 24)):
        totals = []
        for delta in (step, -step):
            probe = host.copy().reshape(-1)
            probe[index] += delta
            probe = probe.reshape(shape)
            fresh = make()
            fresh.train()
            xi = NativeTensor.from_array(probe)
            try:
                o = fresh.forward(xi)
                totals.append(float(o.to_numpy().sum()))
                o.close()
            finally:
                xi.close()
                close_module(fresh)
        numeric.reshape(-1)[index] = (totals[0] - totals[1]) / (2 * step)

    checked = min(flat.size, 24)
    assert np.allclose(analytic.reshape(-1)[:checked],
                       numeric.reshape(-1)[:checked], atol=1e-5), build
    seed.close()
    x.close()
    close_module(module)


@needs_native
def test_repeated_train_eval_cycles_return_live_storage_to_baseline(monkeypatch):
    from tensorforge.experimental import NativeTensor

    module = batchnorm_module("1d", features=6)
    x = NativeTensor.from_array(
        np.random.default_rng(44).standard_normal((5, 6)))
    try:
        module.train()
        module.forward(x).close()
        recorder = AllocationRecorder(monkeypatch)
        for _ in range(12):
            module.train()
            module.forward(x).close()
            module.eval()
            module.forward(x).close()
        assert recorder.live == 0, recorder.live
    finally:
        x.close()
        close_module(module)


@needs_native
def test_a_normalized_model_still_resumes_exactly_from_a_checkpoint(tmp_path):
    """The end-to-end guarantee H8 must not weaken: an interrupted
    normalized training run resumed into a *fresh* model and optimizer
    reproduces the remaining loss suffix, every parameter, both running
    statistics, and the optimizer state by exact equality."""
    from tensorforge.experimental import (
        NativeAdam, NativeLinear, NativeMSELoss, NativeReLU, NativeSequential,
        NativeTensor, load_native_checkpoint, save_native_checkpoint,
    )
    from tensorforge.experimental import NativeBatchNorm1d, NativeLayerNorm

    def build():
        return NativeSequential(
            NativeLinear(4, 6, seed=0), NativeBatchNorm1d(6), NativeReLU(),
            NativeLayerNorm(6), NativeLinear(6, 2, seed=1))

    rng = np.random.default_rng(45)
    inputs = rng.standard_normal((8, 4))
    targets = rng.standard_normal((8, 2))

    def run(model, optimizer, steps):
        loss_fn = NativeMSELoss()
        values = []
        x = NativeTensor.from_array(inputs)
        y = NativeTensor.from_array(targets)
        try:
            for _ in range(steps):
                optimizer.zero_grad()
                out = model.forward(x)
                loss = loss_fn.forward(out, y)
                loss.backward()
                values.append(float(loss.to_numpy()))
                loss.close()
                optimizer.step()
        finally:
            x.close()
            y.close()
        return values

    reference = build()
    reference_opt = NativeAdam(list(reference.parameters()), lr=0.05)
    full = run(reference, reference_opt, 10)

    interrupted = build()
    interrupted_opt = NativeAdam(list(interrupted.parameters()), lr=0.05)
    run(interrupted, interrupted_opt, 4)
    path = tmp_path / "h8_resume.npz"
    save_native_checkpoint(path, interrupted, interrupted_opt)
    close_module(interrupted)
    interrupted_opt.close()

    resumed = build()
    resumed_opt = NativeAdam(list(resumed.parameters()), lr=0.05)
    load_native_checkpoint(path, resumed, resumed_opt)
    suffix = run(resumed, resumed_opt, 6)

    assert suffix == full[4:]
    for a, b in zip(reference.parameters(), resumed.parameters()):
        assert same_bits(a.to_numpy(), b.to_numpy())
    for a, b in zip(reference.buffers(), resumed.buffers()):
        assert same_bits(a.to_numpy(), b.to_numpy())

    close_module(reference)
    reference_opt.close()
    close_module(resumed)
    resumed_opt.close()


@needs_native
def test_no_fused_normalization_operation_was_added():
    """H8 explicitly did not create one, and the inventories say so."""
    from tensorforge import experimental
    from tensorforge.experimental import native_tensor

    for name in ("layer_norm", "batch_norm", "normalize", "affine",
                 "fused_norm"):
        assert not hasattr(native_tensor.NativeTensor, name), name
        assert not hasattr(cpp.NativeTensorCore, name), name
        assert name not in experimental.__all__, name
    assert not [n for n in cpp.TENSOR_CORE_OPS if "norm" in n]
    assert not [n for n in cpp.AUTOGRAD_OPS if "norm" in n]
    # ...and the normalization modules still compose from the ordinary
    # differentiable operations rather than reaching the backend directly.
    for name in ("native_batchnorm.py", "native_layernorm.py"):
        source = (REPO_ROOT / "src" / "tensorforge" / "experimental"
                  / name).read_text(encoding="utf-8")
        imports = re.findall(r"^\s*(?:from|import)\s+\S+.*$", source, re.M)
        for line in imports:
            assert "ctypes" not in line, (name, line)
            assert "backends" not in line, (name, line)
            assert "NativeTensorCore" not in line, (name, line)
        # ...and no attribute access into the backend either (the prose
        # above the code legitimately names these things).
        assert not re.search(r"\bcpp\.\w", source), name
        assert not re.search(r"\bNativeTensorCore\.\w", source), name


@needs_native
def test_no_capability_dtype_device_or_checkpoint_value_moved():
    from tensorforge.experimental import native_checkpoint

    assert cpp.UNSUPPORTED == ("float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert native_checkpoint._FORMAT_VERSION == 2
    assert set(native_checkpoint._SUPPORTED_FORMAT_VERSIONS) == {1, 2}
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"


def test_importing_stable_tensorforge_loads_no_native_library():
    """Track A is entirely inside the experimental line."""
    import subprocess
    import sys

    code = (
        "import sys; import tensorforge;"
        "assert tensorforge.Tensor is not None;"
        "bad=[m for m in sys.modules if 'experimental' in m or"
        " m.endswith('backends.cpp')];"
        "assert not bad, bad;"
        "print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True,
                            cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


BANNED_TIMING_NAMES = ("perf_" "counter", "time." "time", "time" "it",
                       "mono" "tonic", "process_" "time", "import " "time")


def test_this_file_asserts_no_timing():
    """No duration, threshold, or speed measurement may enter the suite:
    performance is characterized by the benchmarks and asserted nowhere.

    The forbidden names are spelled as concatenations above so this scan
    does not trip over its own list."""
    text = __import__("pathlib").Path(__file__).read_text(encoding="utf-8")
    body = text.split('"""', 2)[2]
    for banned in BANNED_TIMING_NAMES:
        assert banned not in body, banned
    assert math.isfinite(1.0)
