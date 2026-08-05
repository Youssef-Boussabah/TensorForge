"""The H6 reduction-execution contract (Phase H, milestone H6).

H6 changed *how* ``tf_core_sum`` walks memory and nothing else. The native
side now ships two traversals behind one unchanged export:

* ``tf::sum_generic_strided`` — the pre-H6 odometer that addresses the
  source through its own shape/strides/offset while an output position
  advances through the write strides. It is the **retained generic
  reference path**: shipped, reachable through ordinary production
  dispatch, the only path that can address a transposed, narrowed,
  non-unit-strided, or broadcast source at all, and the oracle every
  optimized result is compared against.
* ``tf::sum_contiguous_blocks`` — a flat traversal over the
  ``outer x mid x inner`` factorization, used when the source is row-major
  contiguous and the reduced axes form one contiguous run.

The choice is made inside the kernel by
``tf::reduce_prefers_contiguous_blocks`` from the layout metadata the
export already receives. It is deterministic, total, side-effect free, and
independent of pointer values, alignment, wall time, environment
variables, and CPU-feature probes. **No selector, threshold setter,
dispatch tracer, or "which path ran" hook exists in the ABI**, and §9
below asserts that against the built image's own export table.

How this file observes a decision it cannot query — the H2/H5 technique,
which happens to be unusually clean for a reduction. The odometer walks
the source in **logical** row-major order, so a transposed view carrying
the *same logical values* is reduced over the same elements, in the same
order, into the same destination cells as a contiguous core — while
failing the block predicate on its strides. So the two paths can be
compared on identical logical work using nothing but the public API.

What this file proves:

1. **H6's numerical contract**, which unlike H2's has only one qualified
   part:

   a. **Per-output accumulation order is preserved exactly.** Both paths
      add the same source values into each destination cell in the same
      ascending order, starting from the same initial value.
   b. **Every result is bit-identical** — ``+0.0`` versus ``-0.0``,
      ``+/-inf``, denormals, the smallest normal, the largest finite
      magnitudes, and NaNs in the same positions, all quiet.
   c. **NaN payload bits are outside TensorForge's reduction contract**,
      and the two paths select differently when two or more NaNs are
      accumulated into one cell. This is asserted in **neither**
      direction. It is not H2's rule copied over: it was measured here,
      and four candidate spellings of the optimized accumulation —
      including the one that accumulates through memory exactly as the
      odometer does — all selected the *same* NaN and all differed from
      the odometer, so payload parity is unavailable at any spelling
      short of abandoning the optimization. The block path keeps the
      **first** NaN in accumulation order, which is also what NumPy
      keeps; the odometer keeps the **last**.
2. **Axis, shape, and keepdims behavior is exactly unchanged**, error
   types and messages included.
3. **Backward is exact** for sum and mean, at every layout.
4. **Normalization, softmax, log-softmax, and cross-entropy are exact**,
   and deterministic training and exact checkpoint resume still hold.
5. **The H1 allocation decision stands**: this destination is still
   zero-initialized, on both paths, because both accumulate into it.
6. **Failure atomicity and ownership are unchanged**, with live storage
   returning to baseline after success and failure cycles alike.
7. **Scope**: no new export, no capability move, no public control.

The predicate itself, both traversals in isolation, the special-value
matrix, and the accumulate-into contract are additionally driven directly
in ``cpp/tests/test_sum_reduction.cpp``, which compiles
``cpp/src/reduction.cpp`` in so it can reach the hidden internals.

No test here asserts a duration. Timing lives in
``benchmarks/benchmark_native_cpu_performance.py`` and is never a
pass/fail criterion.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchNorm1d,
    NativeBatchNorm2d,
    NativeCrossEntropyLoss,
    NativeLayerNorm,
    NativeLinear,
    NativeMSELoss,
    NativeParameter,
    NativeReLU,
    NativeSequential,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)

Core = cpp.NativeTensorCore
REPO_ROOT = Path(__file__).resolve().parents[1]

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bits(values):
    """Raw IEEE-754 bit patterns. A value comparison cannot answer this
    file's questions: ``-0.0 == +0.0`` is True and ``nan == nan`` is
    False."""
    return np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)


def _same_bits(left, right):
    left = np.ascontiguousarray(left, dtype=np.float64)
    right = np.ascontiguousarray(right, dtype=np.float64)
    return (left.shape == right.shape
            and np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def _from_bit_patterns(patterns):
    return np.array(list(patterns), dtype=np.uint64).view(np.float64)


def _is_nan_bits(raw):
    raw = np.asarray(raw, dtype=np.uint64)
    return ((raw & np.uint64(0x7FF0000000000000))
            == np.uint64(0x7FF0000000000000)) & (
        (raw & np.uint64(0x000FFFFFFFFFFFFF)) != np.uint64(0))


def _is_quiet_bits(raw):
    raw = np.asarray(raw, dtype=np.uint64)
    return (raw & np.uint64(0x0008000000000000)) != np.uint64(0)


# The IEEE-754 sweep, mirrored in cpp/tests/test_sum_reduction.cpp so the
# two layers cannot drift.
PATTERNS = {
    "+0.0": 0x0000000000000000,
    "-0.0": 0x8000000000000000,
    "1.0": 0x3FF0000000000000,
    "-1.0": 0xBFF0000000000000,
    "smallest subnormal": 0x0000000000000001,
    "-smallest subnormal": 0x8000000000000001,
    "largest subnormal": 0x000FFFFFFFFFFFFF,
    "smallest normal": 0x0010000000000000,
    "largest finite": 0x7FEFFFFFFFFFFFFF,
    "-largest finite": 0xFFEFFFFFFFFFFFFF,
    "+inf": 0x7FF0000000000000,
    "-inf": 0xFFF0000000000000,
    "qNaN payload A": 0x7FF800DEADBEEF01,
    "qNaN payload B": 0x7FF8000000000042,
    "-qNaN payload C": 0xFFF80000000000AA,
    "sNaN payload 1": 0x7FF0000000000001,
    "-sNaN payload 1": 0xFFF0000000000001,
    "2^-53": 0x3CB0000000000000,
}
PATTERN_NAMES = list(PATTERNS)


def _transposed_twin(values):
    """A real transposed view carrying exactly ``values``' logical
    contents, so the reduction it feeds is *the same logical reduction*
    over *the same logical order* while the block predicate rejects its
    strides. Returns ``(view, owner)``; the caller closes ``owner``."""
    order = tuple(reversed(range(np.ndim(values))))
    owner = Core.from_array(np.ascontiguousarray(np.transpose(values, order)))
    return owner.transpose(*order), owner


def _both_paths(values, axis=None, keepdims=False, op="sum"):
    """``(block_result, generic_result)`` for one logical reduction."""
    contiguous = Core.from_array(values)
    view, owner = _transposed_twin(values)
    try:
        assert contiguous.contiguous
        # The twin is genuinely strided whenever the shape makes that
        # possible. An all-ones shape has only one layout, so there the
        # twin *is* contiguous and both sides take the block path — still a
        # valid comparison, just not a cross-path one.
        if np.ndim(values) > 1 and any(d > 1 for d in np.shape(values)):
            assert not view.contiguous, "the twin should not be contiguous"
        block = getattr(contiguous, op)(axis=axis, keepdims=keepdims)
        try:
            generic = getattr(view, op)(axis=axis, keepdims=keepdims)
            try:
                assert block.shape == generic.shape
                return block.to_numpy().copy(), generic.to_numpy().copy()
            finally:
                generic.close()
        finally:
            block.close()
    finally:
        contiguous.close()
        owner.close()


def _close_module(module):
    for _name, parameter in module.named_parameters():
        parameter.close()
    for _name, buffer in module.named_buffers():
        buffer.close()


class _Boom(BaseException):
    """A non-``Exception`` BaseException, so cleanup paths are proved to
    run for something ``except Exception`` would never catch."""


FAILURE_CLASSES = (RuntimeError, MemoryError, KeyboardInterrupt, _Boom)


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — exact, because it
    hooks ``close()`` rather than relying on garbage collection."""
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


# ===========================================================================
# 1. The two paths agree, across every shape and axis the layer accepts
# ===========================================================================

# Shapes chosen to hit each block form (outer==1, inner==1, both above 1),
# unit dimensions in every position, prime extents, and both sides of a
# cache line.
SHAPES = [
    (1,), (2,), (7,), (16,), (17,), (64,), (257,),
    (1, 1), (1, 5), (5, 1), (2, 3), (3, 2), (8, 8), (7, 11), (13, 3),
    (33, 3), (3, 33), (64, 65),
    (1, 2, 3), (2, 1, 3), (2, 3, 1), (3, 4, 5), (5, 13, 2),
    (2, 3, 2, 3), (1, 2, 1, 2), (2, 2, 3, 5), (4, 3, 8, 8),
]


@needs_native
@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
def test_both_paths_agree_bit_for_bit_over_every_axis(shape):
    """(1a)+(1b): the block traversal reproduces the retained odometer's
    bits, for every axis, both ``keepdims`` values, and ``axis=None``."""
    rng = np.random.default_rng(20260729 + len(shape))
    values = rng.standard_normal(shape) * 7.5
    axes = [None] + list(range(len(shape))) + [-1, -len(shape)]
    for axis in axes:
        for keepdims in (False, True):
            block, generic = _both_paths(values, axis, keepdims)
            assert _same_bits(block, generic), (shape, axis, keepdims)


@needs_native
@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
def test_both_paths_agree_with_numpy_to_a_tolerance(shape):
    """The reduction is still the same *mathematics*. Compared to NumPy by
    tolerance, deliberately not bit-for-bit: BLAS-era NumPy uses its own
    pairwise summation order, which is exactly the §7.3 order change this
    project does not make."""
    rng = np.random.default_rng(90210 + len(shape))
    values = rng.standard_normal(shape)
    for axis in [None] + list(range(len(shape))):
        for keepdims in (False, True):
            block, generic = _both_paths(values, axis, keepdims)
            expected = values.sum(axis=axis, keepdims=keepdims)
            assert np.allclose(block, expected, rtol=1e-12, atol=1e-10)
            assert np.allclose(generic, expected, rtol=1e-12, atol=1e-10)


@needs_native
def test_a_large_reduction_agrees_on_both_paths_and_with_numpy():
    """Past every plausible block or unrolling boundary."""
    rng = np.random.default_rng(4242)
    values = rng.standard_normal((129, 257))
    for axis in (None, 0, 1):
        block, generic = _both_paths(values, axis)
        assert _same_bits(block, generic), axis
        assert np.allclose(block, values.sum(axis=axis), rtol=1e-11,
                           atol=1e-9)


@needs_native
def test_mean_agrees_on_both_paths():
    """``mean`` is ``sum`` plus the existing in-place ``1/count`` scale.
    H6 changed neither the division placement nor the scaling, so the two
    paths must agree bit for bit here too."""
    rng = np.random.default_rng(777)
    for shape in [(6,), (5, 7), (3, 4, 5), (2, 3, 4, 5)]:
        values = rng.standard_normal(shape) * 3.0
        for axis in [None] + list(range(len(shape))):
            for keepdims in (False, True):
                block, generic = _both_paths(values, axis, keepdims,
                                             op="mean")
                assert _same_bits(block, generic), (shape, axis, keepdims)
                assert np.allclose(block,
                                   values.mean(axis=axis, keepdims=keepdims),
                                   rtol=1e-12, atol=1e-12)


@needs_native
def test_repeated_reductions_are_identical():
    """Nothing is cached, warmed, or path-dependent: the tenth call equals
    the first, on both paths."""
    rng = np.random.default_rng(31337)
    values = rng.standard_normal((17, 19))
    first_block, first_generic = _both_paths(values, 0)
    for _ in range(9):
        block, generic = _both_paths(values, 0)
        assert _same_bits(block, first_block)
        assert _same_bits(generic, first_generic)


@needs_native
def test_the_predicate_is_a_function_of_metadata_not_of_data():
    """Two tensors with the same layout and different contents take the
    same path, which is what "dispatch is metadata-driven" means
    observably: the ratio of block-to-generic agreement cannot depend on
    the values, so wildly different data still agrees."""
    for factory in (
        lambda: np.zeros((8, 9)),
        lambda: np.full((8, 9), 1e300),
        lambda: np.full((8, 9), 5e-324),
        lambda: np.arange(72, dtype=np.float64).reshape(8, 9),
    ):
        values = factory()
        for axis in (None, 0, 1):
            block, generic = _both_paths(values, axis)
            assert _same_bits(block, generic), axis


# ===========================================================================
# 2. Signed zeros
# ===========================================================================

ZERO_CASES = {
    "all +0.0": np.zeros((4, 6)),
    "all -0.0": np.full((4, 6), -0.0),
    "alternating": np.tile(np.array([0.0, -0.0, 0.0, -0.0, 0.0, -0.0]), (4, 1)),
    "-0.0 first": np.tile(np.array([-0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (4, 1)),
    "-0.0 last": np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0]), (4, 1)),
    "-0.0 with finite": np.tile(
        np.array([-0.0, 1.0, -1.0, -0.0, 2.0, -2.0]), (4, 1)),
    "cancelling finite": np.tile(
        np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0]), (4, 1)),
    "column of -0.0": np.column_stack(
        [np.full(4, -0.0), np.arange(4, dtype=np.float64)]),
}


@needs_native
@pytest.mark.parametrize("name", list(ZERO_CASES))
def test_signed_zeros_are_bit_identical_on_both_paths(name):
    """The sign of a zero sum depends on the initial accumulator and on
    the addition order, which is exactly what the block traversal's local
    accumulator could have changed. Compared as raw bits."""
    values = ZERO_CASES[name]
    for axis in [None] + list(range(values.ndim)):
        for keepdims in (False, True):
            block, generic = _both_paths(values, axis, keepdims)
            assert _same_bits(block, generic), (name, axis, keepdims)


@needs_native
def test_a_run_of_negative_zeros_sums_to_positive_zero_on_both_paths():
    """The exact value, not just agreement: the destination starts at
    ``+0.0`` (the additive identity a zeroed buffer holds) and
    ``+0.0 + -0.0`` is ``+0.0``, so the sum of any number of ``-0.0``
    values is ``+0.0``. Both paths, and NumPy, agree."""
    for shape, axis in [((4,), None), ((3, 4), None), ((3, 4), 0), ((3, 4), 1)]:
        values = np.full(shape, -0.0)
        block, generic = _both_paths(values, axis)
        expected = values.sum(axis=axis)
        assert _same_bits(block, expected), (shape, axis)
        assert _same_bits(generic, expected), (shape, axis)
        assert np.all(_bits(block) == np.uint64(0)), (shape, axis)


@needs_native
def test_a_scalar_reduction_of_negative_zero_keeps_numpys_answer():
    """Rank 1, one element: the smallest case where the accumulator's seed
    is the whole answer."""
    block, generic = _both_paths(np.array([-0.0]), None)
    assert _same_bits(block, generic)
    assert _same_bits(block, np.array(-0.0).sum())
    assert int(_bits(block).ravel()[0]) == 0x0000000000000000


@needs_native
def test_the_rank_zero_export_path_still_places_its_single_element():
    """A rank-0 source is handled by the export *before* either traversal
    is consulted, and H6 left that branch untouched.

    Recorded precisely rather than idealized: the branch is
    ``dst[0] += src[offset]``, so it really is an addition against the
    zeroed destination, and ``+0.0 + -0.0`` is ``+0.0``. A rank-0 ``-0.0``
    therefore sums to ``+0.0`` — which is exactly what it did before H6,
    and is asserted here so a future change to that branch is caught.
    """
    for pattern in (0x8000000000000000, 0x0000000000000000,
                    0x3FF0000000000000, 0xBFF0000000000000,
                    0x7FF800DEADBEEF01):
        value = _from_bit_patterns([pattern])[0]
        expected = _bits(np.array([0.0 + value])).ravel()[0]
        holder = Core.from_array(np.array([value]))
        try:
            scalar = holder.reshape(())
            assert scalar.ndim == 0
            for keepdims in (False, True):
                out = scalar.sum(keepdims=keepdims)
                try:
                    assert out.shape == ()
                    assert int(_bits(out.to_numpy()).ravel()[0]) == int(
                        expected), hex(pattern)
                finally:
                    out.close()
        finally:
            holder.close()


# ===========================================================================
# 3. Infinities, denormals, extremes, and NaN
# ===========================================================================

@needs_native
def test_the_non_nan_special_values_are_bit_identical():
    """Every finite and infinite pattern in the sweep, laid out so each
    row and each column mixes magnitudes."""
    finite_names = [n for n in PATTERN_NAMES if "NaN" not in n]
    row = _from_bit_patterns(PATTERNS[n] for n in finite_names)
    values = np.tile(row, (3, 1))
    values[1] = values[1][::-1]
    for axis in (None, 0, 1):
        for keepdims in (False, True):
            block, generic = _both_paths(values, axis, keepdims)
            assert _same_bits(block, generic), (axis, keepdims)


@needs_native
@pytest.mark.parametrize("name", PATTERN_NAMES)
@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_one_special_value_among_finite_addends_is_bit_identical(name,
                                                                position):
    """One exceptional value per accumulation — the case where **no**
    payload choice arises because at most one operand of any addition is a
    NaN. Both paths must agree bit for bit, payloads included."""
    row = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    row[{"first": 0, "middle": 2, "last": 4}[position]] = _from_bit_patterns(
        [PATTERNS[name]])[0]
    values = np.tile(row, (3, 1))
    for axis in (None, 0, 1):
        block, generic = _both_paths(values, axis)
        assert _same_bits(block, generic), (name, position, axis)


@needs_native
def test_denormals_and_the_extremes_survive_both_paths():
    values = np.array([
        [5e-324, 1e-320, 2.2250738585072014e-308, 1.0],
        [1.7976931348623157e308, -1.7976931348623157e308, 0.0, -0.0],
        [1e-300, 1e300, -1e-300, -1e300],
    ])
    for axis in (None, 0, 1):
        block, generic = _both_paths(values, axis)
        assert _same_bits(block, generic), axis
        assert np.allclose(block, values.sum(axis=axis), rtol=1e-12,
                           atol=0.0, equal_nan=True)


@needs_native
def test_nan_positions_are_identical_on_both_paths():
    """(1c) part one, which **is** contractual: whenever either path
    produces a NaN, both do, in exactly the same positions."""
    rng = np.random.default_rng(5150)
    for shape in [(6,), (4, 5), (3, 4, 5)]:
        values = rng.standard_normal(shape)
        flat = values.reshape(-1)
        flat[::3] = np.nan
        for axis in [None] + list(range(len(shape))):
            block, generic = _both_paths(values, axis)
            assert np.array_equal(np.isnan(block), np.isnan(generic)), (
                shape, axis)


@needs_native
def test_every_nan_either_path_produces_is_quiet():
    """(1c) part two, also contractual: neither path can emit a signaling
    NaN, and a signaling NaN *input* is quieted by both."""
    rows = [
        _from_bit_patterns([PATTERNS["sNaN payload 1"], 0x3FF0000000000000,
                            0x4000000000000000]),
        _from_bit_patterns([PATTERNS["-sNaN payload 1"], 0x3FF0000000000000,
                            0x4000000000000000]),
        _from_bit_patterns([PATTERNS["qNaN payload A"], 0x3FF0000000000000,
                            0x4000000000000000]),
        np.array([np.inf, -np.inf, 1.0]),
    ]
    seen_a_nan = False
    for row in rows:
        values = np.tile(row, (2, 1))
        for axis in (None, 0, 1):
            block, generic = _both_paths(values, axis)
            # NaN *positions* agree on both paths; wherever a NaN appears,
            # it is quiet. Not every (row, axis) combination produces one —
            # summing a column of +inf gives +inf, not a NaN — so quietness
            # is asserted where NaNs exist and their existence is asserted
            # once over the whole matrix.
            assert np.array_equal(_is_nan_bits(_bits(block).ravel()),
                                  _is_nan_bits(_bits(generic).ravel())), (
                row, axis)
            for produced in (block, generic):
                raw = _bits(produced).ravel()
                nan_mask = _is_nan_bits(raw)
                if nan_mask.any():
                    seen_a_nan = True
                    assert bool(np.all(_is_quiet_bits(raw[nan_mask]))), (
                        row, axis)
    assert seen_a_nan, "the matrix produced no NaN at all — it proves nothing"


@needs_native
def test_a_signaling_nan_with_finite_addends_agrees_bit_for_bit():
    """Only *one* NaN is involved, so there is no payload choice to make
    and the two paths must produce identical bits — including the quieted
    payload."""
    for name in ("sNaN payload 1", "-sNaN payload 1"):
        row = _from_bit_patterns([PATTERNS[name], 0x3FF0000000000000,
                                  0x4000000000000000])
        values = np.tile(row, (3, 1))
        for axis in (None, 0, 1):
            block, generic = _both_paths(values, axis)
            assert _same_bits(block, generic), (name, axis)


@needs_native
def test_a_manufactured_nan_appears_in_the_same_places_and_is_quiet():
    """``+inf + -inf`` makes a NaN out of non-NaN inputs. Its position and
    quietness are contractual; its payload is not."""
    values = np.array([[np.inf, -np.inf, 1.0],
                       [1.0, np.inf, -np.inf],
                       [1.0, 2.0, 3.0]])
    for axis in (None, 0, 1):
        block, generic = _both_paths(values, axis)
        assert np.array_equal(np.isnan(block), np.isnan(generic)), axis
        for produced in (block, generic):
            raw = _bits(produced).ravel()
            mask = _is_nan_bits(raw)
            if mask.any():
                assert bool(np.all(_is_quiet_bits(raw[mask]))), axis


@needs_native
def test_the_nan_payload_rule_is_asserted_in_neither_direction():
    """(1c) part three: when **two or more** NaNs are accumulated into one
    cell, the paths may choose different payloads, and H6 does not promise
    either answer.

    This test deliberately asserts only what the contract says — both are
    NaN, both are quiet — and records the measured behavior in its own
    body rather than pinning it. A build whose payloads agree and a build
    whose payloads differ are equally conforming: on this toolchain the
    block path keeps the first NaN in accumulation order and the odometer
    keeps the last, and four candidate spellings of the optimized
    accumulation all agreed with each other and differed from the
    odometer, so parity is not available at any spelling.
    """
    a = _from_bit_patterns([PATTERNS["qNaN payload A"]])[0]
    b = _from_bit_patterns([PATTERNS["qNaN payload B"]])[0]
    c = _from_bit_patterns([PATTERNS["-qNaN payload C"]])[0]
    for values, axis in [
        (np.array([[a, b, 1.0]]), None),
        (np.array([[a, b, 1.0]]), 1),
        (np.array([[a, b, c]]), None),
        (np.array([[1.0, a, b, 2.0]]), None),
        (np.array([[a, b], [b, a]]), 1),
    ]:
        block, generic = _both_paths(values, axis)
        block_raw = _bits(block).ravel()
        generic_raw = _bits(generic).ravel()
        # Contractual: same NaN positions, all quiet.
        assert np.array_equal(_is_nan_bits(block_raw),
                              _is_nan_bits(generic_raw))
        for raw in (block_raw, generic_raw):
            mask = _is_nan_bits(raw)
            assert bool(np.all(_is_quiet_bits(raw[mask])))
        # NOT asserted: block_raw == generic_raw. Payloads are outside the
        # contract in both directions.


@needs_native
def test_the_block_paths_two_nan_choice_matches_numpy():
    """A recorded observation rather than a promise, kept because it is
    the reason H6 did not treat the payload difference as a regression:
    where the paths differ, the *optimized* one agrees with NumPy.

    Asserted only for the shapes and toolchain this project builds on; if
    a future toolchain changes the selection, the contract above still
    holds and only this observation would need re-recording.
    """
    a = _from_bit_patterns([PATTERNS["qNaN payload A"]])[0]
    b = _from_bit_patterns([PATTERNS["qNaN payload B"]])[0]
    values = np.array([[a, b, 1.0]])
    contiguous = Core.from_array(values)
    try:
        out = contiguous.sum(axis=1)
        try:
            produced = out.to_numpy().copy()
        finally:
            out.close()
    finally:
        contiguous.close()
    reference = values.sum(axis=1)
    if _bits(produced).ravel()[0] != _bits(reference).ravel()[0]:
        pytest.skip("this toolchain selects a different NaN payload; the "
                    "contract in the previous test is the binding one")
    assert _same_bits(produced, reference)


# ===========================================================================
# 4. The generic fallback: which layouts keep the odometer, and are correct
# ===========================================================================

@needs_native
def test_a_transposed_source_falls_back_and_is_correct():
    rng = np.random.default_rng(1001)
    values = rng.standard_normal((8, 6, 4))
    base = Core.from_array(values)
    try:
        for order in [(2, 1, 0), (1, 0, 2), (0, 2, 1)]:
            view = base.transpose(*order)
            assert not view.contiguous, order
            host = np.transpose(values, order)
            for axis in [None] + list(range(3)):
                out = view.sum(axis=axis)
                try:
                    assert np.allclose(out.to_numpy(), host.sum(axis=axis),
                                       rtol=1e-12, atol=1e-10), (order, axis)
                finally:
                    out.close()
    finally:
        base.close()


@needs_native
def test_a_last_axis_narrow_falls_back_and_is_correct():
    """Narrowing the *last* axis breaks row-major contiguity, so the
    predicate rejects it and the odometer runs."""
    rng = np.random.default_rng(1002)
    values = rng.standard_normal((10, 12))
    base = Core.from_array(values)
    try:
        view = base.narrow(1, 2, 7)
        assert not view.contiguous
        host = values[:, 2:9]
        for axis in [None, 0, 1]:
            out = view.sum(axis=axis)
            try:
                assert np.allclose(out.to_numpy(), host.sum(axis=axis),
                                   rtol=1e-12, atol=1e-10), axis
            finally:
                out.close()
    finally:
        base.close()


@needs_native
def test_a_leading_axis_narrow_stays_contiguous_and_is_correct():
    """Narrowing a *leading* axis keeps row-major strides and only moves
    the offset, so the block path takes it — with a nonzero offset, which
    the traversal must honor."""
    rng = np.random.default_rng(1003)
    values = rng.standard_normal((10, 12))
    base = Core.from_array(values)
    try:
        view = base.narrow(0, 3, 5)
        assert view.contiguous and view.offset != 0
        host = values[3:8]
        for axis in [None, 0, 1]:
            for keepdims in (False, True):
                out = view.sum(axis=axis, keepdims=keepdims)
                try:
                    assert np.allclose(
                        out.to_numpy(),
                        host.sum(axis=axis, keepdims=keepdims),
                        rtol=1e-12, atol=1e-10), (axis, keepdims)
                finally:
                    out.close()
    finally:
        base.close()


@needs_native
def test_chained_views_are_correct_on_whichever_path_they_take():
    rng = np.random.default_rng(1004)
    values = rng.standard_normal((12, 10, 6))
    base = Core.from_array(values)
    try:
        chains = {
            "narrow-narrow contiguous": (base.narrow(0, 2, 6).narrow(1, 0, 10),
                                         values[2:8, :, :]),
            "narrow then transpose": (base.narrow(0, 1, 4).transpose(2, 1, 0),
                                      np.transpose(values[1:5], (2, 1, 0))),
            "transpose then narrow": (base.transpose(1, 0, 2).narrow(0, 3, 4),
                                      np.transpose(values,
                                                   (1, 0, 2))[3:7]),
            "double narrow non-contiguous": (
                base.narrow(2, 1, 3).narrow(1, 2, 5),
                values[:, 2:7, 1:4]),
        }
        for label, (view, host) in chains.items():
            for axis in [None] + list(range(3)):
                out = view.sum(axis=axis)
                try:
                    assert np.allclose(out.to_numpy(), host.sum(axis=axis),
                                       rtol=1e-12, atol=1e-10), (label, axis)
                finally:
                    out.close()
    finally:
        base.close()


@needs_native
def test_a_positive_non_unit_stride_view_falls_back_and_is_correct():
    """Reached through the public view API: transposing a narrowed axis
    produces genuine non-unit positive strides on both axes."""
    rng = np.random.default_rng(1005)
    values = rng.standard_normal((9, 11))
    base = Core.from_array(values)
    try:
        view = base.narrow(1, 1, 5).transpose(1, 0)
        assert not view.contiguous
        assert all(s > 0 for s in view.strides)
        assert 1 not in view.strides or view.strides != (1, 1)
        host = values[:, 1:6].T
        for axis in [None, 0, 1]:
            out = view.sum(axis=axis)
            try:
                assert np.allclose(out.to_numpy(), host.sum(axis=axis),
                                   rtol=1e-12, atol=1e-10), axis
            finally:
                out.close()
    finally:
        base.close()


@needs_native
def test_a_one_element_view_of_a_larger_storage_is_correct_on_both_paths():
    rng = np.random.default_rng(1006)
    values = rng.standard_normal((6, 7))
    base = Core.from_array(values)
    try:
        cell = base.narrow(0, 2, 1).narrow(1, 4, 1)
        for axis in [None, 0, 1]:
            out = cell.sum(axis=axis)
            try:
                assert np.allclose(out.to_numpy(),
                                   values[2:3, 4:5].sum(axis=axis),
                                   rtol=1e-12, atol=1e-12), axis
            finally:
                out.close()
    finally:
        base.close()


# ===========================================================================
# 5. Axis, shape, keepdims, and the unchanged rejections
# ===========================================================================

@needs_native
def test_negative_axes_match_their_positive_equivalents_exactly():
    rng = np.random.default_rng(2001)
    for shape in [(5,), (4, 6), (3, 4, 5), (2, 3, 4, 5)]:
        values = rng.standard_normal(shape)
        core = Core.from_array(values)
        try:
            for negative in range(-len(shape), 0):
                positive = negative + len(shape)
                for keepdims in (False, True):
                    a = core.sum(axis=negative, keepdims=keepdims)
                    b = core.sum(axis=positive, keepdims=keepdims)
                    try:
                        assert a.shape == b.shape
                        assert _same_bits(a.to_numpy(), b.to_numpy())
                    finally:
                        a.close()
                        b.close()
        finally:
            core.close()


@needs_native
def test_keepdims_shapes_are_exactly_reduce_shapes_answer():
    rng = np.random.default_rng(2002)
    for shape in [(5,), (4, 6), (3, 4, 5), (1, 4, 1)]:
        values = rng.standard_normal(shape)
        core = Core.from_array(values)
        try:
            for axis in [None] + list(range(len(shape))):
                for keepdims in (False, True):
                    out = core.sum(axis=axis, keepdims=keepdims)
                    try:
                        assert out.shape == cpp.reduce_shape(
                            shape, axis=axis, keepdims=keepdims)
                        assert out.shape == np.shape(
                            values.sum(axis=axis, keepdims=keepdims))
                    finally:
                        out.close()
        finally:
            core.close()


@needs_native
@pytest.mark.parametrize("bad_axis", [2, -3, 100, -100])
def test_an_out_of_bounds_axis_still_raises_valueerror_naming_both(bad_axis):
    """The message shape is contracted by the existing suite; H6 must not
    have moved it, and validation must still precede any allocation."""
    core = Core.from_array(np.arange(6.0).reshape(2, 3))
    try:
        with pytest.raises(ValueError) as info:
            core.sum(axis=bad_axis)
        message = str(info.value)
        assert str(bad_axis) in message
        assert "(2, 3)" in message
        assert "out of bounds" in message
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("bad_axis", [1.0, "0", True, False, [0], (0,),
                                      np.float64(0.0)])
def test_a_non_int_axis_still_raises_typeerror(bad_axis):
    core = Core.from_array(np.arange(6.0).reshape(2, 3))
    try:
        with pytest.raises(TypeError) as info:
            core.sum(axis=bad_axis)
        assert "axis must be None or an int" in str(info.value)
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("bad", [1, 0, "yes", None, np.bool_(True)])
def test_a_non_bool_keepdims_still_raises_typeerror(bad):
    core = Core.from_array(np.arange(6.0).reshape(2, 3))
    try:
        with pytest.raises(TypeError) as info:
            core.sum(keepdims=bad)
        assert "keepdims must be a bool" in str(info.value)
    finally:
        core.close()


@needs_native
def test_a_tuple_axis_is_still_unsupported():
    """H6 did **not** add multi-axis reduction. The kernel's predicate can
    factorize a contiguous run of reduced axes, but the Python layer still
    accepts one int or ``None``, and that boundary did not move."""
    core = Core.from_array(np.arange(24.0).reshape(2, 3, 4))
    try:
        with pytest.raises(TypeError):
            core.sum(axis=(0, 1))
        with pytest.raises(TypeError):
            core.sum(axis=[0, 2])
    finally:
        core.close()


@needs_native
def test_an_integer_axis_on_a_scalar_is_still_rejected():
    holder = Core.from_array(np.array([1.5]))
    try:
        scalar = holder.reshape(())
        assert scalar.ndim == 0
        with pytest.raises(ValueError):
            scalar.sum(axis=0)
        with pytest.raises(ValueError):
            scalar.sum(axis=-1)
        out = scalar.sum()
        try:
            assert out.shape == ()
            assert out.to_numpy().reshape(()) == 1.5
        finally:
            out.close()
    finally:
        holder.close()


@needs_native
def test_a_closed_core_is_rejected_before_anything_is_allocated(live_storages):
    core = Core.from_array(np.arange(6.0).reshape(2, 3))
    core.close()
    baseline = len(live_storages)
    with pytest.raises(RuntimeError):
        core.sum(axis=0)
    with pytest.raises(RuntimeError):
        core.mean()
    assert len(live_storages) == baseline


@needs_native
def test_zero_sized_dimensions_are_still_rejected_by_the_representation():
    """The kernel and the C ABI handle a zero element count, but the
    native tensor representation refuses zero-size dimensions outright, so
    no empty core can be constructed from Python. H6 did not change
    that."""
    with pytest.raises(ValueError):
        Core.zeros((0,))
    with pytest.raises(ValueError):
        Core.zeros((3, 0))


@needs_native
def test_a_shape_whose_product_overflows_is_still_rejected():
    huge = (2 ** 40, 2 ** 40)
    with pytest.raises((ValueError, MemoryError, OverflowError)):
        Core.zeros(huge)


# ===========================================================================
# 6. Ownership, immutability, and failure atomicity
# ===========================================================================

@needs_native
def test_the_source_is_never_mutated_on_either_path():
    rng = np.random.default_rng(3001)
    values = rng.standard_normal((7, 5, 3))
    for view_factory, host in (
        (lambda v: (Core.from_array(v), None), values),
        (lambda v: _transposed_twin(v), np.transpose(values, (2, 1, 0))),
    ):
        made = view_factory(values)
        view, owner = made if isinstance(made, tuple) else (made, None)
        try:
            before = view.to_numpy().copy()
            metadata = (view.shape, view.strides, view.offset, view.ndim,
                        view.contiguous)
            for axis in [None, 0, 1, 2]:
                view.sum(axis=axis).close()
                view.mean(axis=axis).close()
            assert _same_bits(view.to_numpy(), before)
            assert (view.shape, view.strides, view.offset, view.ndim,
                    view.contiguous) == metadata
        finally:
            if owner is not None and owner is not view:
                owner.close()
            else:
                view.close()


@needs_native
def test_the_result_is_fresh_owning_row_major_storage_at_offset_zero():
    rng = np.random.default_rng(3002)
    values = rng.standard_normal((6, 4))
    core = Core.from_array(values)
    view, owner = _transposed_twin(values)
    try:
        for source in (core, view):
            for axis in [None, 0, 1]:
                out = source.sum(axis=axis)
                try:
                    assert out.contiguous
                    assert out.offset == 0
                    assert out.storage is not source.storage
                    assert out.storage is not core.storage
                    assert out.storage is not owner.storage
                finally:
                    out.close()
    finally:
        core.close()
        owner.close()


@needs_native
def test_a_failing_native_call_closes_the_output(live_storages, monkeypatch):
    """The output is allocated before the kernel runs, so a kernel failure
    must release it. Injected at the ABI seam, on both paths."""
    rng = np.random.default_rng(3003)
    values = rng.standard_normal((6, 4))
    for source_factory in (lambda: (Core.from_array(values), None),
                           lambda: _transposed_twin(values)):
        made = source_factory()
        source, owner = made
        try:
            baseline = len(live_storages)
            library = cpp._require_library()
            original = library.tf_core_sum

            def exploding(*args, **kwargs):
                raise RuntimeError("injected native failure")

            monkeypatch.setattr(library, "tf_core_sum", exploding)
            try:
                with pytest.raises(RuntimeError, match="injected"):
                    source.sum(axis=0)
            finally:
                monkeypatch.setattr(library, "tf_core_sum", original)
            assert len(live_storages) == baseline
            # And the operation still works afterwards.
            source.sum(axis=0).close()
            assert len(live_storages) == baseline
        finally:
            if owner is not None and owner is not source:
                owner.close()
            else:
                source.close()


@needs_native
@pytest.mark.parametrize("error", FAILURE_CLASSES)
def test_a_failing_output_allocation_leaves_nothing_behind(error,
                                                           live_storages,
                                                           monkeypatch):
    """Allocation is the first thing ``sum`` does after validation; a
    failure there must leave the source untouched and storage at
    baseline."""
    rng = np.random.default_rng(3004)
    values = rng.standard_normal((6, 4))
    core = Core.from_array(values)
    try:
        before = core.to_numpy().copy()
        baseline = len(live_storages)
        original = Core.zeros

        def exploding(*args, **kwargs):
            raise error("injected allocation failure")

        monkeypatch.setattr(Core, "zeros", staticmethod(exploding))
        try:
            with pytest.raises(error):
                core.sum(axis=0)
        finally:
            monkeypatch.setattr(Core, "zeros", staticmethod(original))
        assert len(live_storages) == baseline
        assert _same_bits(core.to_numpy(), before)
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("error", FAILURE_CLASSES)
def test_a_failing_mean_scale_releases_the_summed_output(error, live_storages,
                                                         monkeypatch):
    """``mean`` is sum-then-scale, and the sum's output is already
    allocated when the scale runs, so a failure there must release it
    **explicitly** — not leave it to a refcount or the collector."""
    rng = np.random.default_rng(3005)
    core = Core.from_array(rng.standard_normal((6, 4)))
    try:
        baseline = len(live_storages)
        library = cpp._require_library()
        original = library.tf_storage_scale

        def exploding(*args, **kwargs):
            raise error("injected scale failure")

        monkeypatch.setattr(library, "tf_storage_scale", exploding)
        try:
            with pytest.raises(error):
                core.mean(axis=0)
        finally:
            monkeypatch.setattr(library, "tf_storage_scale", original)
        # No gc.collect(): the release is explicit.
        assert len(live_storages) == baseline
        for _ in range(3):
            core.mean(axis=0).close()
        assert len(live_storages) == baseline
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("seam", ["write_strides", "layout_pointers"])
@pytest.mark.parametrize("error", FAILURE_CLASSES)
def test_a_post_allocation_failure_releases_the_output(seam, error,
                                                       live_storages,
                                                       monkeypatch):
    """The two seams that run *after* the output is allocated and *before*
    the kernel: the write-stride construction and the layout metadata.

    The second seam is ``_layout_pointers`` as of Phase H milestone H7,
    which is where a reduction now obtains the ``shape``/``strides`` the
    strided C ABI takes; before H7 it was ``_layout_arrays``. The seam
    moved, the contract did not — every assertion below is unchanged.

    ``sum`` runs both inside the same cleanup boundary every other
    allocating Core op uses, so a failure at either releases the output
    explicitly. Asserted with **no** ``gc.collect()``, because relying on a
    refcount would be relying on an implementation detail rather than on
    the contract.
    """
    rng = np.random.default_rng(3008)
    core = Core.from_array(rng.standard_normal((4, 5, 6)))
    try:
        before = core.to_numpy().copy()
        baseline = len(live_storages)

        def exploding(*args, **kwargs):
            raise error(f"injected {seam} failure")

        if seam == "write_strides":
            monkeypatch.setattr(cpp, "_reduce_out_strides", exploding)
        else:
            monkeypatch.setattr(Core, "_layout_pointers", exploding)
        with pytest.raises(error):
            core.sum(axis=1)
        monkeypatch.undo()

        assert len(live_storages) == baseline
        assert _same_bits(core.to_numpy(), before)
        # And the operation still works afterwards.
        core.sum(axis=1).close()
        assert len(live_storages) == baseline
    finally:
        core.close()


@needs_native
def test_a_pre_allocation_failure_allocates_nothing_at_all(live_storages,
                                                           monkeypatch):
    """Axis and output-shape validation run *before* the allocation, so a
    failure there must not allocate in the first place."""
    core = Core.from_array(np.arange(24.0).reshape(2, 3, 4))
    try:
        baseline = len(live_storages)
        # A real out-of-range axis.
        with pytest.raises(ValueError):
            core.sum(axis=9)
        assert len(live_storages) == baseline
        # And an injected failure in the output-shape construction itself.
        def exploding(*args, **kwargs):
            raise MemoryError("injected shape failure")

        monkeypatch.setattr(cpp, "_reduce_shape_checked", exploding)
        with pytest.raises(MemoryError):
            core.sum(axis=0)
        monkeypatch.undo()
        assert len(live_storages) == baseline
    finally:
        core.close()


@needs_native
def test_repeated_success_cycles_return_live_storage_to_baseline(
        live_storages):
    rng = np.random.default_rng(3006)
    values = rng.standard_normal((9, 7))
    core = Core.from_array(values)
    view, owner = _transposed_twin(values)
    try:
        baseline = len(live_storages)
        for _ in range(40):
            for axis in (None, 0, 1):
                core.sum(axis=axis).close()
                core.mean(axis=axis).close()
                view.sum(axis=axis).close()
        assert len(live_storages) == baseline
    finally:
        core.close()
        owner.close()


@needs_native
def test_repeated_failure_cycles_also_return_live_storage_to_baseline(
        live_storages, monkeypatch):
    rng = np.random.default_rng(3007)
    core = Core.from_array(rng.standard_normal((9, 7)))
    try:
        baseline = len(live_storages)
        library = cpp._require_library()
        original = library.tf_core_sum

        def exploding(*args, **kwargs):
            raise RuntimeError("injected")

        for _ in range(25):
            monkeypatch.setattr(library, "tf_core_sum", exploding)
            try:
                with pytest.raises(RuntimeError):
                    core.sum(axis=0)
            finally:
                monkeypatch.setattr(library, "tf_core_sum", original)
            core.sum(axis=0).close()
        assert len(live_storages) == baseline
    finally:
        core.close()


# ===========================================================================
# 7. The H1 allocation decision
# ===========================================================================

@needs_native
def test_the_reduction_output_is_still_zero_initialized(monkeypatch):
    """H1 rejected this destination and H6 confirms that rejection: both
    traversals *read* the destination, so the zero is the additive
    identity. Observed structurally — ``sum`` must reach the
    zero-initializing constructor and never the uninitialized one."""
    zeroed = []
    uninitialized = []
    original_zeros = Core.zeros
    original_uninit = Core._uninitialized

    def spy_zeros(*args, **kwargs):
        zeroed.append(args)
        return original_zeros(*args, **kwargs)

    def spy_uninit(*args, **kwargs):
        uninitialized.append(args)
        return original_uninit(*args, **kwargs)

    monkeypatch.setattr(Core, "zeros", staticmethod(spy_zeros))
    monkeypatch.setattr(Core, "_uninitialized", staticmethod(spy_uninit))
    core = Core.from_array(np.arange(24.0).reshape(4, 6))
    try:
        for axis in (None, 0, 1):
            core.sum(axis=axis).close()
        assert len(zeroed) == 3
        assert uninitialized == []
    finally:
        core.close()


@needs_native
def test_a_nonzero_destination_would_change_the_answer_on_both_paths():
    """The negative control that makes the previous test load-bearing: the
    kernel really does accumulate into what it is given, so a non-zero
    destination is *not* equivalent. Driven at the ABI, where a
    destination can be pre-filled."""
    values = np.arange(12.0).reshape(3, 4)
    library = cpp._require_library()
    for source, label in ((Core.from_array(values), "block"),
                          (_transposed_twin(values), "generic")):
        view, owner = source if isinstance(source, tuple) else (source, None)
        try:
            out = Core.zeros((4,))
            try:
                shape_ptr, strides_ptr = view._layout_pointers()
                out_strides = cpp._layout_vector(
                    cpp._reduce_out_strides(view.shape, {0}, False, (4,)))
                library.tf_storage_fill(out._storage._require_open(), 100.0)
                library.tf_core_sum(
                    view._storage._require_open(),
                    out._storage._require_open(),
                    shape_ptr, strides_ptr, out_strides,
                    view.offset, view.ndim)
                accumulated = out.to_numpy().copy()
            finally:
                out.close()
            expected = values.sum(axis=0) + 100.0
            assert np.allclose(accumulated, expected), label
        finally:
            if owner is not None and owner is not view:
                owner.close()
            else:
                view.close()


# ===========================================================================
# 8. Autograd: sum and mean backward
# ===========================================================================

@needs_native
def test_sum_backward_is_exact_at_every_axis_and_layout():
    rng = np.random.default_rng(4001)
    for shape in [(5,), (4, 6), (3, 4, 5)]:
        values = rng.standard_normal(shape)
        for axis in [None] + list(range(len(shape))):
            for keepdims in (False, True):
                tensor = NativeTensor.from_array(values, requires_grad=True)
                try:
                    out = tensor.sum(axis=axis, keepdims=keepdims)
                    total = out.sum() if out.shape != () else out
                    total.backward()
                    grad = tensor.grad.to_numpy().copy()
                    # d(sum)/dx is 1 everywhere.
                    assert _same_bits(grad, np.ones(shape)), (
                        shape, axis, keepdims)
                    if total is not out:
                        total.close()
                    out.close()
                finally:
                    tensor.close()


@needs_native
def test_mean_backward_is_exact_at_every_axis():
    rng = np.random.default_rng(4002)
    for shape in [(5,), (4, 6), (3, 4, 5)]:
        values = rng.standard_normal(shape)
        for axis in [None] + list(range(len(shape))):
            tensor = NativeTensor.from_array(values, requires_grad=True)
            try:
                out = tensor.mean(axis=axis)
                total = out.sum() if out.shape != () else out
                total.backward()
                count = (np.prod(shape) if axis is None else shape[axis])
                expected = np.full(shape, 1.0 / count)
                assert np.allclose(tensor.grad.to_numpy(), expected,
                                   rtol=0.0, atol=0.0), (shape, axis)
                if total is not out:
                    total.close()
                out.close()
            finally:
                tensor.close()


@needs_native
def test_backward_agrees_between_the_two_forward_paths():
    """The forward chose a traversal; the gradient must not know. Compared
    bit for bit between a contiguous source and its transposed twin (whose
    gradient is transposed back for the comparison)."""
    rng = np.random.default_rng(4003)
    values = rng.standard_normal((5, 7))
    for axis in [None, 0, 1]:
        contiguous = NativeTensor.from_array(values, requires_grad=True)
        base = NativeTensor.from_array(np.ascontiguousarray(values.T),
                                       requires_grad=True)
        try:
            twin = base.transpose(1, 0)
            out_a = contiguous.sum(axis=axis)
            out_b = twin.sum(axis=axis)
            total_a = out_a.sum() if out_a.shape != () else out_a
            total_b = out_b.sum() if out_b.shape != () else out_b
            total_a.backward()
            total_b.backward()
            grad_a = contiguous.grad.to_numpy().copy()
            grad_b = base.grad.to_numpy().copy().T
            assert _same_bits(grad_a, grad_b), axis
            if total_a is not out_a:
                total_a.close()
            if total_b is not out_b:
                total_b.close()
            out_a.close()
            out_b.close()
            twin.close()
        finally:
            contiguous.close()
            base.close()


@needs_native
def test_gradient_accumulation_still_sums_across_two_reductions():
    rng = np.random.default_rng(4004)
    values = rng.standard_normal((4, 5))
    tensor = NativeTensor.from_array(values, requires_grad=True)
    try:
        first = tensor.sum(axis=0)
        second = tensor.sum(axis=1)
        a = first.sum()
        b = second.sum()
        a.backward()
        after_first = tensor.grad.to_numpy().copy()
        b.backward()
        after_both = tensor.grad.to_numpy().copy()
        assert np.allclose(after_first, np.ones((4, 5)))
        assert np.allclose(after_both, np.full((4, 5), 2.0))
        a.close()
        b.close()
        first.close()
        second.close()
    finally:
        tensor.close()


@needs_native
def test_repeated_backward_with_retain_graph_is_stable():
    rng = np.random.default_rng(4005)
    tensor = NativeTensor.from_array(rng.standard_normal((4, 5)),
                                     requires_grad=True)
    try:
        out = tensor.mean(axis=1, keepdims=True)
        total = out.sum()
        readings = []
        for _ in range(4):
            total.backward(retain_graph=True)
            readings.append(tensor.grad.to_numpy().copy())
            tensor.zero_grad()
        for reading in readings[1:]:
            assert _same_bits(reading, readings[0])
        total.close()
        out.close()
    finally:
        tensor.close()


@needs_native
def test_a_reduction_node_records_no_expected_parameter_version():
    """sum/mean backward reads only the upstream gradient and the reduced
    shape — never the parent's value — so mutating a parameter operand
    after the forward must leave the edge valid. Unchanged by H6."""
    rng = np.random.default_rng(4006)
    parameter = NativeParameter(rng.standard_normal((4, 5)))
    try:
        out = parameter.sum(axis=0)
        total = out.sum()
        parameter.copy_value_(NativeTensor.from_array(
            rng.standard_normal((4, 5))))
        total.backward()  # must not raise a stale-graph error
        assert np.allclose(parameter.grad.to_numpy(), np.ones((4, 5)))
        total.close()
        out.close()
    finally:
        parameter.close()


@needs_native
def test_a_graph_reduction_builds_a_node_and_the_plain_one_does_not():
    rng = np.random.default_rng(4007)
    values = rng.standard_normal((4, 5))
    plain = NativeTensor.from_array(values)
    graphed = NativeTensor.from_array(values, requires_grad=True)
    try:
        a = plain.sum(axis=0)
        b = graphed.sum(axis=0)
        try:
            assert a.requires_grad is False
            assert b.requires_grad is True
            assert _same_bits(a.to_numpy(), b.to_numpy())
        finally:
            a.close()
            b.close()
    finally:
        plain.close()
        graphed.close()


@needs_native
def test_a_backward_over_a_narrowed_and_chained_source_is_exact():
    rng = np.random.default_rng(4008)
    values = rng.standard_normal((8, 6))
    tensor = NativeTensor.from_array(values, requires_grad=True)
    try:
        narrowed = tensor.narrow(0, 2, 4)
        out = narrowed.sum(axis=1)
        total = out.sum()
        total.backward()
        expected = np.zeros((8, 6))
        expected[2:6] = 1.0
        assert np.allclose(tensor.grad.to_numpy(), expected)
        total.close()
        out.close()
        narrowed.close()
    finally:
        tensor.close()


@needs_native
def test_unbroadcast_through_a_reduction_is_exact():
    """``_unbroadcast`` is the reduction's largest internal consumer: a
    broadcast add's backward sums the stretched axes back down."""
    rng = np.random.default_rng(4009)
    a_values = rng.standard_normal((6, 5))
    b_values = rng.standard_normal((1, 5))
    a = NativeTensor.from_array(a_values, requires_grad=True)
    b = NativeTensor.from_array(b_values, requires_grad=True)
    try:
        out = a.add(b)
        total = out.sum()
        total.backward()
        assert np.allclose(a.grad.to_numpy(), np.ones((6, 5)))
        assert np.allclose(b.grad.to_numpy(), np.full((1, 5), 6.0))
        total.close()
        out.close()
    finally:
        a.close()
        b.close()


@needs_native
def test_a_backward_failure_leaves_storage_at_baseline(live_storages,
                                                       monkeypatch):
    rng = np.random.default_rng(4010)
    tensor = NativeTensor.from_array(rng.standard_normal((4, 5)),
                                     requires_grad=True)
    try:
        out = tensor.sum(axis=0)
        total = out.sum()
        baseline = len(live_storages)
        library = cpp._require_library()
        original = library.tf_core_add

        def exploding(*args, **kwargs):
            raise RuntimeError("injected backward failure")

        monkeypatch.setattr(library, "tf_core_add", exploding)
        try:
            with pytest.raises(RuntimeError):
                total.backward(retain_graph=True)
        finally:
            monkeypatch.setattr(library, "tf_core_add", original)
        # The graph is retained, so the retry must succeed and the
        # allocation count must not have drifted upward permanently.
        total.backward(retain_graph=True)
        assert tensor.grad is not None
        tensor.zero_grad()
        total.close()
        out.close()
        assert len(live_storages) <= baseline + 1
    finally:
        tensor.close()


# ===========================================================================
# 9. Composed users: normalization, softmax, cross-entropy
# ===========================================================================

@needs_native
def test_layernorm_matches_an_explicit_numpy_formula():
    rng = np.random.default_rng(5001)
    values = rng.standard_normal((16, 8))
    module = NativeLayerNorm(8, eps=1e-5)
    x = NativeTensor.from_array(values)
    try:
        out = module(x)
        try:
            mean = values.mean(axis=1, keepdims=True)
            variance = ((values - mean) ** 2).mean(axis=1, keepdims=True)
            expected = (values - mean) / np.sqrt(variance + 1e-5)
            assert np.allclose(out.to_numpy(), expected, rtol=1e-11,
                               atol=1e-11)
        finally:
            out.close()
    finally:
        x.close()
        _close_module(module)


@needs_native
def test_layernorm_agrees_between_the_two_reduction_paths():
    """A non-contiguous input drives the *same* module through the
    fallback reduction; the outputs must agree to float64 tolerance
    (LayerNorm's own compositions differ in operand layout, so this is a
    tolerance claim, not a bit claim)."""
    rng = np.random.default_rng(5002)
    values = rng.standard_normal((12, 8))
    module = NativeLayerNorm(8)
    contiguous = NativeTensor.from_array(values)
    base = NativeTensor.from_array(np.ascontiguousarray(values.T))
    try:
        twin = base.transpose(1, 0)
        a = module(contiguous)
        b = module(twin)
        try:
            assert np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12,
                               atol=1e-12)
        finally:
            a.close()
            b.close()
        twin.close()
    finally:
        contiguous.close()
        base.close()
        _close_module(module)


@needs_native
def test_batchnorm1d_running_statistics_match_an_explicit_formula():
    rng = np.random.default_rng(5003)
    values = rng.standard_normal((10, 4))
    module = NativeBatchNorm1d(4, momentum=0.1, eps=1e-5)
    x = NativeTensor.from_array(values)
    try:
        module.train()
        out = module(x)
        try:
            batch_mean = values.mean(axis=0)
            batch_var = values.var(axis=0)  # population, no Bessel
            expected = (values - batch_mean) / np.sqrt(batch_var + 1e-5)
            assert np.allclose(out.to_numpy(), expected, rtol=1e-11,
                               atol=1e-11)
            expected_mean = 0.9 * 0.0 + 0.1 * batch_mean
            expected_var = 0.9 * 1.0 + 0.1 * batch_var
            assert np.allclose(module.running_mean.to_numpy(), expected_mean,
                               rtol=1e-12, atol=1e-12)
            assert np.allclose(module.running_var.to_numpy(), expected_var,
                               rtol=1e-12, atol=1e-12)
        finally:
            out.close()
    finally:
        x.close()
        _close_module(module)


@needs_native
def test_batchnorm2d_matches_an_explicit_nchw_formula():
    rng = np.random.default_rng(5004)
    values = rng.standard_normal((4, 3, 5, 5))
    module = NativeBatchNorm2d(3, eps=1e-5)
    x = NativeTensor.from_array(values)
    try:
        module.train()
        out = module(x)
        try:
            axes = (0, 2, 3)
            mean = values.mean(axis=axes, keepdims=True)
            variance = values.var(axis=axes, keepdims=True)
            expected = (values - mean) / np.sqrt(variance + 1e-5)
            assert np.allclose(out.to_numpy(), expected, rtol=1e-10,
                               atol=1e-10)
        finally:
            out.close()
    finally:
        x.close()
        _close_module(module)


@needs_native
def test_normalization_backward_is_finite_and_reproducible():
    rng = np.random.default_rng(5005)
    values = rng.standard_normal((8, 6))
    readings = []
    for _ in range(3):
        module = NativeLayerNorm(6)
        x = NativeTensor.from_array(values, requires_grad=True)
        try:
            out = module(x)
            total = out.sum()
            total.backward()
            readings.append(x.grad.to_numpy().copy())
            total.close()
            out.close()
        finally:
            x.close()
            _close_module(module)
    assert np.all(np.isfinite(readings[0]))
    for reading in readings[1:]:
        assert _same_bits(reading, readings[0])


@needs_native
def test_softmax_and_log_softmax_backward_match_closed_form():
    """Both backwards contain a ``sum(axis, keepdims=True)`` — the suffix
    form H6 gave a local accumulator."""
    rng = np.random.default_rng(5006)
    values = rng.standard_normal((7, 5))
    upstream = rng.standard_normal((7, 5))

    tensor = NativeTensor.from_array(values, requires_grad=True)
    try:
        y = tensor.softmax(-1)
        y.backward(NativeTensor.from_array(upstream))
        probabilities = np.exp(values - values.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        expected = probabilities * (
            upstream - (upstream * probabilities).sum(axis=-1, keepdims=True))
        assert np.allclose(tensor.grad.to_numpy(), expected, rtol=1e-11,
                           atol=1e-11)
        y.close()
    finally:
        tensor.close()

    tensor = NativeTensor.from_array(values, requires_grad=True)
    try:
        y = tensor.log_softmax(-1)
        y.backward(NativeTensor.from_array(upstream))
        shifted = values - values.max(axis=-1, keepdims=True)
        log_probabilities = shifted - np.log(np.exp(shifted).sum(
            axis=-1, keepdims=True))
        expected = upstream - np.exp(log_probabilities) * upstream.sum(
            axis=-1, keepdims=True)
        assert np.allclose(tensor.grad.to_numpy(), expected, rtol=1e-11,
                           atol=1e-11)
        y.close()
    finally:
        tensor.close()


@needs_native
def test_cross_entropy_forward_and_backward_match_an_explicit_formula():
    rng = np.random.default_rng(5007)
    logits = rng.standard_normal((9, 4))
    targets = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0])
    tensor = NativeTensor.from_array(logits, requires_grad=True)
    try:
        loss = NativeCrossEntropyLoss()(tensor, targets)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        expected_loss = -np.log(
            probabilities[np.arange(9), targets]).mean()
        assert np.allclose(loss.to_numpy(), expected_loss, rtol=1e-12,
                           atol=1e-12)
        loss.backward()
        one_hot = np.zeros_like(logits)
        one_hot[np.arange(9), targets] = 1.0
        assert np.allclose(tensor.grad.to_numpy(),
                           (probabilities - one_hot) / 9.0,
                           rtol=1e-11, atol=1e-11)
        loss.close()
    finally:
        tensor.close()


@needs_native
def test_mse_loss_which_ends_in_a_full_reduction_is_exact():
    rng = np.random.default_rng(5008)
    prediction_values = rng.standard_normal((11, 3))
    target_values = rng.standard_normal((11, 3))
    prediction = NativeTensor.from_array(prediction_values,
                                         requires_grad=True)
    target = NativeTensor.from_array(target_values)
    try:
        loss = NativeMSELoss()(prediction, target)
        expected = ((prediction_values - target_values) ** 2).mean()
        assert np.allclose(loss.to_numpy(), expected, rtol=1e-12, atol=1e-12)
        loss.backward()
        assert np.allclose(
            prediction.grad.to_numpy(),
            2.0 * (prediction_values - target_values) / 33.0,
            rtol=1e-11, atol=1e-11)
        loss.close()
    finally:
        prediction.close()
        target.close()


@needs_native
def test_the_conv_bias_gradient_composition_is_exact():
    """``g.sum(0).sum(1).sum(1)`` — three chained reductions, each on a
    contiguous source, so all three take the block path."""
    rng = np.random.default_rng(5009)
    upstream = rng.standard_normal((4, 3, 5, 6))
    core = Core.from_array(upstream)
    try:
        first = core.sum(axis=0)
        try:
            second = first.sum(axis=1)
            try:
                third = second.sum(axis=1)
                try:
                    assert np.allclose(third.to_numpy(),
                                       upstream.sum(axis=(0, 2, 3)),
                                       rtol=1e-11, atol=1e-11)
                finally:
                    third.close()
            finally:
                second.close()
        finally:
            first.close()
    finally:
        core.close()


# ===========================================================================
# 10. Deterministic training and exact checkpoint resume
# ===========================================================================

def _training_model():
    return NativeSequential(
        NativeLinear(4, 6, seed=0),
        NativeBatchNorm1d(6),
        NativeReLU(),
        NativeLayerNorm(6),
        NativeLinear(6, 3, seed=1),
    )


def _training_data():
    """An explicit arithmetic formula: every value is a quarter or an
    eighth, exact in float64. No RNG, nothing loaded."""
    rows = []
    for index in range(12):
        rows.append([(index % 4) * 0.25, (index % 3) * 0.5 - 0.5,
                     (index % 5) * 0.125, ((index + 1) % 4) * 0.25 - 0.25])
    return np.array(rows), np.arange(12) % 3


def _train(model, steps, *, optimizer=None):
    inputs, targets = _training_data()
    optimizer = optimizer or NativeAdam(model.parameters(), lr=0.05)
    loss_fn = NativeCrossEntropyLoss()
    losses = []
    x = NativeTensor.from_array(inputs)
    try:
        for _ in range(steps):
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, targets)
            losses.append(float(loss.to_numpy()))
            loss.backward()
            optimizer.step()
            loss.close()
            logits.close()
    finally:
        x.close()
    return losses, optimizer


@needs_native
def test_two_identical_training_runs_are_bit_identical():
    """Every reduction in this model — LayerNorm's mean, BatchNorm's mean,
    cross-entropy's internals, and the backward broadcasts — runs on the
    H6 traversal. The trajectory must be reproducible to the bit."""
    trajectories = []
    finals = []
    for _ in range(2):
        model = _training_model()
        losses, optimizer = _train(model, 10)
        trajectories.append(losses)
        finals.append({name: parameter.to_numpy().copy()
                       for name, parameter in model.named_parameters()})
        optimizer.close()
        _close_module(model)
    assert trajectories[0] == trajectories[1]
    assert set(finals[0]) == set(finals[1])
    for name in finals[0]:
        assert _same_bits(finals[0][name], finals[1][name]), name
    # And it actually learned.
    assert trajectories[0][-1] < trajectories[0][0]


@needs_native
def test_an_interrupted_run_resumes_bit_exactly(tmp_path):
    """The full exact-resume proof over a model whose every forward uses
    reductions, with H6's traversal active throughout."""
    reference_model = _training_model()
    reference_losses, reference_optimizer = _train(reference_model, 10)
    reference_final = {name: parameter.to_numpy().copy()
                       for name, parameter in
                       reference_model.named_parameters()}
    reference_buffers = {name: buffer.to_numpy().copy()
                         for name, buffer in reference_model.named_buffers()}
    reference_optimizer.close()
    _close_module(reference_model)

    interrupted = _training_model()
    first_losses, first_optimizer = _train(interrupted, 4)
    archive = tmp_path / "resume.npz"
    save_native_checkpoint(archive, interrupted, first_optimizer,
                           metadata={"training_step": 4})
    first_optimizer.close()
    _close_module(interrupted)

    resumed = _training_model()
    resumed_optimizer = NativeAdam(resumed.parameters(), lr=0.05)
    load_native_checkpoint(archive, resumed, resumed_optimizer)
    suffix, resumed_optimizer = _train(resumed, 6,
                                       optimizer=resumed_optimizer)

    assert first_losses == reference_losses[:4]
    assert suffix == reference_losses[4:]
    for name, parameter in resumed.named_parameters():
        assert _same_bits(parameter.to_numpy(), reference_final[name]), name
    for name, buffer in resumed.named_buffers():
        assert _same_bits(buffer.to_numpy(), reference_buffers[name]), name
    resumed_optimizer.close()
    _close_module(resumed)


@needs_native
def test_an_mlp_regression_run_learns_and_is_reproducible():
    """A second architecture, ending in a mean-reduced loss."""
    inputs = np.linspace(-1.0, 1.0, 24).reshape(12, 2)
    targets = (inputs[:, :1] * 0.5 - inputs[:, 1:] * 0.25)
    trajectories = []
    for _ in range(2):
        model = NativeSequential(NativeLinear(2, 8, seed=3), NativeReLU(),
                                 NativeLinear(8, 1, seed=4))
        optimizer = NativeAdam(model.parameters(), lr=0.05)
        loss_fn = NativeMSELoss()
        x = NativeTensor.from_array(inputs)
        y = NativeTensor.from_array(targets)
        losses = []
        try:
            for _ in range(15):
                optimizer.zero_grad()
                prediction = model(x)
                loss = loss_fn(prediction, y)
                losses.append(float(loss.to_numpy()))
                loss.backward()
                optimizer.step()
                loss.close()
                prediction.close()
        finally:
            x.close()
            y.close()
            optimizer.close()
            _close_module(model)
        trajectories.append(losses)
    assert trajectories[0] == trajectories[1]
    assert trajectories[0][-1] < trajectories[0][0] * 0.5


@needs_native
def test_a_reduction_allocates_exactly_one_storage_on_either_path(monkeypatch):
    """H6's memory claim: the optimized traversal introduced **no**
    temporary, scratch buffer, workspace, or pool.

    A ``sum`` allocates exactly one native storage — its own output — on
    both paths and at every axis, and ``mean`` allocates the same one
    (its scale is in place). Counted as an exact opens-minus-closes delta,
    because CPython reuses a freed object's address and an ``id()`` set
    would merge two distinct storages.
    """
    counts = {"opened": 0}
    original_init = cpp.NativeStorage.__init__

    def counting_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        counts["opened"] += 1

    monkeypatch.setattr(cpp.NativeStorage, "__init__", counting_init)

    rng = np.random.default_rng(6001)
    values = rng.standard_normal((6, 5, 4))
    contiguous = Core.from_array(values)
    view, owner = _transposed_twin(values)
    try:
        for source, label in ((contiguous, "block"), (view, "generic")):
            for axis in [None, 0, 1, 2]:
                for keepdims in (False, True):
                    counts["opened"] = 0
                    source.sum(axis=axis, keepdims=keepdims).close()
                    assert counts["opened"] == 1, (label, axis, keepdims,
                                                   counts["opened"])
                    counts["opened"] = 0
                    source.mean(axis=axis, keepdims=keepdims).close()
                    assert counts["opened"] == 1, (label, axis, keepdims,
                                                   counts["opened"])
    finally:
        contiguous.close()
        owner.close()


@needs_native
def test_a_training_steps_reduction_count_is_a_function_of_the_model_only():
    """The complementary claim, stated as something durable: H6 changed no
    *number* of reductions anywhere. A step over this model issues exactly
    the same reductions every time it runs, so a future change that fuses,
    adds, or removes one fails here.

    Deliberately **not** a live-storage-baseline assertion for the whole
    step: a training step's live count oscillates as CPython's collector
    runs (the project's contract is that correctness never depends on the
    collector, not that a step is collector-free), and that oscillation was
    measured to be bit-identical before and after H6.
    """
    calls = []
    original_sum = Core.sum
    original_mean = Core.mean

    def spy_sum(self, axis=None, keepdims=False):
        calls.append(("sum", self.shape, axis, keepdims, self.contiguous))
        return original_sum(self, axis=axis, keepdims=keepdims)

    def spy_mean(self, axis=None, keepdims=False):
        calls.append(("mean", self.shape, axis, keepdims, self.contiguous))
        return original_mean(self, axis=axis, keepdims=keepdims)

    model = _training_model()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    inputs, targets = _training_data()
    loss_fn = NativeCrossEntropyLoss()
    x = NativeTensor.from_array(inputs)
    try:
        # One warm-up step outside the spy, so first-step optimizer state
        # exists and the recorded step is a steady-state one.
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        loss.close()
        logits.close()

        recorded = []
        for _ in range(3):
            calls.clear()
            Core.sum = spy_sum
            Core.mean = spy_mean
            try:
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()
                loss.close()
                logits.close()
            finally:
                Core.sum = original_sum
                Core.mean = original_mean
            recorded.append(list(calls))

        # Identical reduction sequence on every step — same operations,
        # same shapes, same axes, same keepdims, same source layouts.
        for later in recorded[1:]:
            assert later == recorded[0]
        assert recorded[0], "the step performed no reduction at all"
        # And every one of them is on a contiguous source, i.e. the model
        # really does exercise the H6 block path rather than the fallback.
        assert all(entry[4] for entry in recorded[0]), recorded[0]
    finally:
        Core.sum = original_sum
        Core.mean = original_mean
        x.close()
        optimizer.close()
        _close_module(model)


# ===========================================================================
# 11. Scope: no new ABI, no public control, nothing else moved
# ===========================================================================

def _exported_names():
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    try:
        storage_tests = importlib.import_module(
            "test_native_storage_allocation")
    finally:
        sys.path.pop(0)
    return storage_tests.exported_names(cpp._LIBRARY_PATH)


@needs_native
def test_h6_added_no_exported_symbol():
    """H6 added no exported symbol: its traversal choice lives *inside* an
    existing export, so Phase H's surface is still exactly the 52 symbols
    H1 left. (The live library exports 54 — Phase I milestone I1 added the
    two typed storage creators — which is why the Phase-H claim is checked
    against the Phase-H subset.)"""
    import test_native_storage_allocation as h1

    _image, names = _exported_names()
    if names is None:
        pytest.skip("this image format is not parsed here")
    exported = sorted(name for name in names if name.startswith("tf_"))
    assert len(exported) == h1.EXPECTED_TF_EXPORTS, exported
    assert len(h1.phase_h_export_names(exported)) == h1.PHASE_H_TF_EXPORTS
    assert "tf_core_sum" in exported
    assert "tf_core_narrow_backward" in exported
    # Nothing reduction-dispatch-flavored was added.
    for banned in ("reduce_prefers", "reduction_mode", "set_reduction",
                   "block_size", "traversal", "prefers", "select",
                   "sum_generic", "sum_contiguous"):
        assert not [n for n in exported if banned in n.lower()], banned


@needs_native
def test_no_reduction_selector_or_profiling_control_exists_anywhere():
    library = cpp._require_library()
    for name in ("tf_reduce_set_mode", "tf_core_sum_contiguous",
                 "tf_set_reduction_path", "tf_reduce_prefers_contiguous_blocks",
                 "tf_reduction_block_size", "tf_core_sum_generic"):
        with pytest.raises(AttributeError):
            getattr(library, name)
    for module_name in ("tensorforge.backends.cpp",
                        "tensorforge.experimental.native_tensor",
                        "tensorforge.experimental.native_layernorm",
                        "tensorforge.experimental.native_batchnorm"):
        module = importlib.import_module(module_name)
        for attribute in dir(module):
            lowered = attribute.lower()
            assert "reduction_mode" not in lowered, (module_name, attribute)
            assert "reduction_path" not in lowered, (module_name, attribute)
            assert "block_size" not in lowered, (module_name, attribute)
            assert not lowered.startswith("set_reduc"), (module_name,
                                                         attribute)


def test_no_environment_variable_or_dispatch_hook_in_the_reduction_sources():
    """Read from the sources themselves: the dispatch is a function of
    metadata, so nothing may consult the environment, the clock, a
    CPU-feature probe, a thread, or a vector reduction intrinsic."""
    sources = [
        REPO_ROOT / "cpp" / "src" / "reduction.cpp",
        REPO_ROOT / "cpp" / "include" / "tf_reduction_internal.h",
    ]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for banned in ("getenv", "std::chrono", "__cpuid", "cpuid",
                       "std::thread", "#pragma omp", "immintrin",
                       "_mm_", "__m128", "__m256", "fast_math",
                       "cblas", "openblas"):
            assert banned not in text, (path.name, banned)


def test_the_reduction_source_ships_both_paths_and_one_predicate():
    """Structural, not name-trivia: the retained reference traversal, the
    optimized traversal, and exactly one predicate all exist, the export
    is unchanged in name and arity, and the scatter dual was left alone."""
    text = (REPO_ROOT / "cpp" / "src" / "reduction.cpp").read_text(
        encoding="utf-8")
    assert "sum_generic_strided" in text
    assert "sum_contiguous_blocks" in text
    assert text.count("reduce_prefers_contiguous_blocks") >= 2
    assert "TF_EXPORT void tf_core_sum(" in text
    assert "TF_EXPORT void tf_core_narrow_backward(" in text
    # Exactly two exports in this translation unit, as before H6.
    assert text.count("TF_EXPORT") == 2
    # The scatter dual keeps the odometer: it must not call either H6 helper.
    dual = text[text.index("TF_EXPORT void tf_core_narrow_backward("):]
    assert "sum_contiguous_blocks" not in dual
    assert "reduce_prefers_contiguous_blocks" not in dual


def test_the_internal_header_is_not_part_of_the_ctypes_surface():
    """The predicate and both traversals are hidden C++; Python declares
    neither, and the checked-kernel inventory did not grow."""
    text = (REPO_ROOT / "src" / "tensorforge" / "backends" / "cpp.py"
            ).read_text(encoding="utf-8")
    for name in ("reduce_prefers_contiguous_blocks", "sum_contiguous_blocks",
                 "sum_generic_strided"):
        assert f"library.{name}" not in text, name
    # H6 added no hooked kernel, so the tuple still ends with the last entry
    # that existed when it landed once the later phases' additions — Phase K
    # milestone K3's argmax forward — are removed.
    post_h6 = ("tf_core_argmax",)
    assert [name for name in cpp._CHECKED_KERNELS
            if name not in post_h6][-1] == "tf_core_dropout_forward"
    # ``tf_core_sum`` was already an errcheck-hooked kernel before H6 and
    # still is exactly one entry; the inventory did not grow.
    assert cpp._CHECKED_KERNELS.count("tf_core_sum") == 1
    assert len(cpp._CHECKED_KERNELS) == len(set(cpp._CHECKED_KERNELS))


@needs_native
def test_the_public_reduction_surface_did_not_change():
    """Signatures, defaults, and the operation inventories."""
    import inspect

    for owner in (Core, NativeTensor):
        for name in ("sum", "mean"):
            signature = inspect.signature(getattr(owner, name))
            assert list(signature.parameters) == ["self", "axis", "keepdims"], (
                owner, name)
            assert signature.parameters["axis"].default is None
            assert signature.parameters["keepdims"].default is False
    assert "sum" in cpp.TENSOR_CORE_OPS
    assert "mean" in cpp.TENSOR_CORE_OPS
    assert "sum" in cpp.AUTOGRAD_OPS
    assert "mean" in cpp.AUTOGRAD_OPS


@needs_native
def test_no_capability_dtype_device_or_checkpoint_value_moved():
    from tensorforge.experimental import native_checkpoint

    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 3
    assert set(native_checkpoint._SUPPORTED_FORMAT_VERSIONS) == {1, 2, 3}


def test_the_stable_framework_still_imports_nothing_native():
    """Importing stable TensorForge must not pull in the native backend,
    the experimental modules, or the DLL — unchanged by H6."""
    import subprocess

    program = (
        "import sys\n"
        "import tensorforge\n"
        "native = [n for n in sys.modules\n"
        "          if n.startswith('tensorforge.backends')\n"
        "          or n.startswith('tensorforge.experimental')]\n"
        "print(native)\n"
    )
    result = subprocess.run([sys.executable, "-c", program],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout


# ===========================================================================
# 12. Documentation guardrails (semantic, not exact prose)
# ===========================================================================

H6_SURFACES = (
    Path("docs/native_cpu_performance_design.md"),
    Path("docs/backend_experiments.md"),
    Path("docs/roadmap.md"),
    Path("docs/project_summary.md"),
    Path("docs/release_history.md"),
    Path("CLAUDE.md"),
)


@pytest.mark.parametrize("relative", H6_SURFACES,
                         ids=[p.name for p in H6_SURFACES])
def test_every_h6_surface_records_the_milestone_semantically(relative):
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert "H6" in text, relative
    lowered = text.lower()
    assert "reduction" in lowered, relative


def test_the_design_states_the_h6_contract_and_its_carve_out():
    text = (REPO_ROOT / "docs" / "native_cpu_performance_design.md").read_text(
        encoding="utf-8")
    section = text[text.index("H6 — reduction execution, as shipped"):]
    lowered = section.lower()
    # The retained reference path, the predicate, and the fallback.
    assert "sum_generic_strided" in section
    assert "sum_contiguous_blocks" in section
    assert "reduce_prefers_contiguous_blocks" in section
    # The three contract halves.
    assert "accumulation order" in lowered
    assert "signed zero" in lowered or "-0.0" in section
    assert "nan" in lowered
    # The H1 decision is restated, not silently reversed.
    assert "zero-initial" in lowered or "additive identity" in lowered
    # No exported symbol was added.
    assert "52" in section


def test_no_h6_surface_claims_an_added_export_or_a_public_control():
    """The one invariant worth an exact check: no surface may say H6 added
    an ABI symbol or a dispatch control."""
    for relative in H6_SURFACES:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for claim in (r"H6 add(s|ed) (one|a|two) (new )?(exported |C ABI )?symbol",
                      r"H6 add(s|ed) a (reduction )?(path )?selector",
                      r"reduction (path )?selector is (now )?public"):
            assert not re.search(claim, text, re.IGNORECASE), (relative,
                                                               claim)


def test_the_support_matrix_still_reports_the_unchanged_boundary():
    text = (REPO_ROOT / "docs" / "native_support_matrix.md").read_text(
        encoding="utf-8")
    assert "float64" in text
    assert "cpu" in text.lower()
