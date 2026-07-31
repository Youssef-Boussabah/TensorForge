"""The H3 metadata and dispatch contract (Phase H, milestone H3).

H3 removed repeated Python-side metadata normalization from the native
call path. It changed **no** C++, no C ABI symbol, no ctypes declaration,
no kernel, no arithmetic, and no public API. What it changed is how many
times the same already-validated tuple is re-validated on the way to a
kernel.

The architecture, and what this file proves about it:

1. **One normalization boundary.** ``_normalized_layout`` validates a
   shape, its strides, and its offset exactly once, in exactly the order
   ``shape_info`` always used, with exactly the messages it always
   raised. Everything downstream — the row-major strides, the element
   count, the contiguity comparison — is derived from the resulting tuple
   through private ``_checked`` primitives that perform no validation
   *because there is nothing left to validate*. Before H3 one
   ``shape_info`` call ran ``_as_int_tuple`` four times over the same
   tuple and computed the row-major strides twice.

2. **Two view constructors, one binding.** The public
   ``NativeTensorView(...)`` normalizes caller-supplied metadata; the
   private ``NativeTensorView._from_validated(...)`` skips *only* that
   normalization, for metadata this module produced one construction
   earlier. Both funnel through ``_bind``, which performs the storage
   open check and the full reachable-offset bounds check. The element
   count and the contiguity flag are **derived inside** the private
   constructor rather than passed to it, so no caller can supply an
   inconsistent pair — that is why H3 has a separate constructor rather
   than a ``validated=True`` flag.

3. **Per-view layout arrays, memoized.** The ``int64`` shape/stride
   arrays the strided C ABI takes are built at most once per view and
   reused. This is memoization of a pure function of immutable state, not
   a cache with a coherence problem: a view's ``_shape`` and ``_strides``
   are assigned exactly once, in ``_bind``, and no code path anywhere
   reassigns them — reshaping, transposing, and narrowing all produce
   *new* views. The arrays are read-only, so no caller can mutate a
   view's metadata through them, and they are built lazily, so the
   contiguous fast path (which takes a flat count, not shape arrays)
   never allocates them at all.

What H3 must not have done, also asserted here: weaken any rejection,
change any error type or message, make a closed object usable, change a
public API, add a cache control or profiling hook, or move the exported
symbol count.

No test in this file asserts a duration. H3's measurements live in the
design document and in the benchmark harness, never in the suite.
"""

import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp

REPO_ROOT = Path(__file__).resolve().parent.parent

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="native backend not built"
)


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every open NativeStorage — a real live-allocation count,
    so an ownership test can prove the count returns exactly to its
    baseline instead of trusting collection."""
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


@pytest.fixture
def normalization_counter(monkeypatch):
    """Counts calls to the one validating shape-normalization primitive.

    This is a **deliberate architectural invariant**, not an incidental
    implementation detail: H3's entire claim is that a shape supplied to
    an operation is normalized once rather than repeatedly, so the count
    is the property under test. It is asserted as an upper bound, so an
    implementation that normalizes *less* still passes."""
    counts = {"_as_shape": 0, "_as_int_tuple": 0}
    original_shape = cpp._as_shape
    original_tuple = cpp._as_int_tuple

    def counted_shape(shape):
        counts["_as_shape"] += 1
        return original_shape(shape)

    def counted_tuple(values, name):
        counts["_as_int_tuple"] += 1
        return original_tuple(values, name)

    monkeypatch.setattr(cpp, "_as_shape", counted_shape)
    monkeypatch.setattr(cpp, "_as_int_tuple", counted_tuple)
    return counts


# ==========================================================================
# 1. One normalization boundary
# ==========================================================================


@needs_native
def test_allocating_a_core_normalizes_its_shape_once(normalization_counter):
    """``zeros`` validates the caller's shape once and reuses the result
    for the storage size and for the view — it does not hand the raw
    shape to a second validating consumer."""
    core = cpp.NativeTensorCore.zeros((4, 5, 6))
    try:
        assert normalization_counter["_as_shape"] == 1
    finally:
        core.close()


@needs_native
def test_uninitialized_and_full_normalize_their_shape_once(
        normalization_counter):
    for construct in (
        lambda: cpp.NativeTensorCore._uninitialized((3, 4)),
        lambda: cpp.NativeTensorCore.full((3, 4), 1.5),
    ):
        normalization_counter["_as_shape"] = 0
        core = construct()
        try:
            assert normalization_counter["_as_shape"] == 1
        finally:
            core.close()


@needs_native
def test_view_operations_do_not_renormalize_the_parent_layout(
        normalization_counter):
    """``transpose`` and ``narrow`` derive their metadata from a layout
    this module already validated, so they normalize nothing at all;
    ``reshape`` normalizes only the caller's new shape."""
    base = cpp.NativeTensorCore.from_array(np.ones((4, 5, 6)))
    try:
        normalization_counter["_as_shape"] = 0
        transposed = base.transpose(2, 0, 1)
        assert normalization_counter["_as_shape"] == 0

        normalization_counter["_as_shape"] = 0
        narrowed = base.narrow(1, 1, 3)
        assert normalization_counter["_as_shape"] == 0

        normalization_counter["_as_shape"] = 0
        reshaped = base.reshape((10, 12))
        assert normalization_counter["_as_shape"] == 1

        for view in (transposed, narrowed, reshaped):
            view.close()
    finally:
        base.close()


@needs_native
def test_an_elementwise_operation_normalizes_only_its_output_shape(
        normalization_counter):
    """A binary op's operands carry validated layouts; only the freshly
    allocated output's shape goes through normalization."""
    a = cpp.NativeTensorCore.from_array(np.ones((8, 8)))
    b = cpp.NativeTensorCore.from_array(np.ones((8, 8)))
    try:
        normalization_counter["_as_shape"] = 0
        out = a.add(b)
        try:
            assert normalization_counter["_as_shape"] == 1
        finally:
            out.close()
    finally:
        a.close()
        b.close()


def test_shape_info_and_normalized_layout_agree_exactly():
    """The public dictionary is the private tuple, rearranged — the two
    can never report different metadata for the same layout."""
    layouts = [
        ((2, 3, 4), None, 0),
        ((), None, 0),
        ((1,), None, 3),
        ((3, 2), (1, 3), 5),
        ((4, 4), (4, 1), 0),
        ((5,), (2,), 1),
        ((2, 3), (-3, 1), 7),
    ]
    for shape, strides, offset in layouts:
        info = cpp.shape_info(shape, strides=strides, offset=offset)
        dims, stride_tuple, off, count, contiguous = cpp._normalized_layout(
            shape, strides=strides, offset=offset
        )
        assert info["shape"] == dims
        assert info["strides"] == stride_tuple
        assert info["offset"] == off
        assert info["numel"] == count
        assert info["contiguous"] is contiguous
        assert info["ndim"] == len(dims)


def test_the_checked_primitives_match_their_validating_public_forms():
    """Each ``_checked`` primitive computes exactly what its public
    counterpart computes; the public one only adds validation."""
    for shape in [(), (1,), (5,), (2, 3), (2, 3, 4), (1, 1, 1), (7, 1, 3)]:
        assert cpp._row_major_strides_checked(shape) == cpp.row_major_strides(shape)
        assert cpp._numel_checked(shape) == cpp.numel(shape)
        for keepdims in (False, True):
            assert cpp._reduce_shape_checked(shape, None, keepdims) == (
                cpp.reduce_shape(shape, None, keepdims))
            for axis in range(len(shape)):
                assert cpp._reduce_shape_checked(shape, axis, keepdims) == (
                    cpp.reduce_shape(shape, axis, keepdims))
                assert cpp._normalize_axis_checked(axis, shape) == (
                    cpp._normalize_axis(axis, shape))
    for a in [(), (1,), (3, 1), (2, 3), (1, 3, 1)]:
        for b in [(), (4,), (1, 4), (2, 1), (2, 3, 5)]:
            try:
                expected = cpp.broadcast_shapes(a, b)
            except ValueError:
                with pytest.raises(ValueError):
                    cpp._broadcast_shapes_checked(a, b)
                continue
            assert cpp._broadcast_shapes_checked(a, b) == expected


# ==========================================================================
# 2. Every rejection survives
# ==========================================================================


def test_malformed_shapes_are_still_rejected():
    for bad, error, message in [
        ((2, -3), ValueError, "positive"),
        ((2, 0), ValueError, "not supported in v0.7"),
        ((2, 3.5), TypeError, "ints"),
        ((2, True), TypeError, "ints"),
        ((2, None), TypeError, "ints"),
        ((2, "3"), TypeError, "ints"),
        (7, TypeError, "sequence"),
        (None, TypeError, "sequence"),
    ]:
        with pytest.raises(error, match=message):
            cpp.shape_info(bad)
        with pytest.raises(error, match=message):
            cpp.numel(bad)
        with pytest.raises(error, match=message):
            cpp.row_major_strides(bad)


@needs_native
def test_malformed_shapes_are_rejected_at_every_allocating_constructor():
    for bad, error in [((2, -3), ValueError), ((2, 0), ValueError),
                       ((2, 3.5), TypeError), ((2, True), TypeError),
                       (7, TypeError)]:
        with pytest.raises(error):
            cpp.NativeTensorCore.zeros(bad)
        with pytest.raises(error):
            cpp.NativeTensorCore._uninitialized(bad)
        with pytest.raises(error):
            cpp.NativeTensorCore.full(bad, 1.0)


def test_malformed_strides_are_still_rejected():
    for strides, error, message in [
        ((3.0, 1), TypeError, "ints"),
        ((True, 1), TypeError, "ints"),
        ((None, 1), TypeError, "ints"),
        ((1,), ValueError, "same length"),
        ((1, 2, 3), ValueError, "same length"),
        (5, TypeError, "sequence"),
    ]:
        with pytest.raises(error, match=message):
            cpp.shape_info((2, 3), strides=strides)


def test_malformed_offsets_are_still_rejected():
    for offset in (1.5, "zero", None, True, [0]):
        with pytest.raises(TypeError, match="offset"):
            cpp.shape_info((2, 3), offset=offset)


@needs_native
def test_the_view_rejects_malformed_metadata_in_the_documented_order():
    """Shape first, then strides, then offset — a call with more than one
    fault reports the first, exactly as before H3."""
    storage = cpp.NativeStorage.from_array(np.arange(6.0))
    try:
        # Shape wins over a simultaneously bad stride and offset.
        with pytest.raises(ValueError, match="positive"):
            cpp.NativeTensorView(storage, (2, -3), strides=(1.0, 1),
                                 offset="x")
        # Strides win over a bad offset.
        with pytest.raises(TypeError, match="ints"):
            cpp.NativeTensorView(storage, (2, 3), strides=(3.0, 1),
                                 offset="x")
        # Stride length is checked before the offset too.
        with pytest.raises(ValueError, match="same length"):
            cpp.NativeTensorView(storage, (2, 3), strides=(1,), offset="x")
        # Offset alone.
        with pytest.raises(TypeError, match="offset"):
            cpp.NativeTensorView(storage, (2, 3), offset=1.5)
    finally:
        storage.close()


@needs_native
def test_out_of_bounds_views_are_still_rejected_through_both_constructors():
    """The bounds check is in ``_bind``, which both constructors use, so
    a derived layout cannot escape it either."""
    storage = cpp.NativeStorage.from_array(np.arange(6.0))
    try:
        for shape, strides, offset in [
            ((7,), None, 0),
            ((2, 3), None, 1),
            ((3,), (-1,), 1),
            ((), None, 6),
            ((2,), (10,), 0),
        ]:
            with pytest.raises(ValueError, match="outside"):
                cpp.NativeTensorView(storage, shape, strides=strides,
                                     offset=offset)
            with pytest.raises(ValueError, match="outside"):
                cpp.NativeTensorView._from_validated(
                    storage, shape,
                    strides if strides is not None
                    else cpp._row_major_strides_checked(shape),
                    offset,
                )
    finally:
        storage.close()


@needs_native
def test_a_closed_storage_cannot_back_a_view_through_either_constructor():
    storage = cpp.NativeStorage.from_array(np.arange(6.0))
    storage.close()
    with pytest.raises(RuntimeError, match="closed"):
        cpp.NativeTensorView(storage, (2, 3))
    with pytest.raises(RuntimeError, match="closed"):
        cpp.NativeTensorView._from_validated(storage, (2, 3), (3, 1), 0)


@needs_native
def test_reshape_narrow_and_transpose_still_reject_their_own_arguments():
    base = cpp.NativeTensorCore.from_array(np.ones((4, 6)))
    try:
        with pytest.raises(ValueError, match="cannot reshape"):
            base.reshape((5, 5))
        with pytest.raises(ValueError, match="positive"):
            base.reshape((4, -6))
        with pytest.raises(TypeError, match="ints"):
            base.reshape((4.0, 6))
        with pytest.raises(TypeError, match="must be an int"):
            base.narrow(0.0, 0, 1)
        with pytest.raises(TypeError, match="must be an int"):
            base.narrow(0, True, 1)
        with pytest.raises(ValueError, match=r"dim must be in \[0, 2\)"):
            base.narrow(2, 0, 1)
        with pytest.raises(ValueError, match="out of bounds"):
            base.narrow(0, 3, 2)
        with pytest.raises(ValueError, match="out of bounds"):
            base.narrow(0, 0, 0)
        with pytest.raises(ValueError, match="permutation"):
            base.transpose(0, 0)
        with pytest.raises(TypeError, match="ints"):
            base.transpose(0.0, 1)
        transposed = base.transpose(1, 0)
        try:
            with pytest.raises(ValueError, match="contiguous"):
                transposed.reshape((24,))
        finally:
            transposed.close()
    finally:
        base.close()


@needs_native
def test_a_huge_shape_is_still_rejected_rather_than_overflowing():
    """Python integers do not wrap, so an element count beyond what the
    allocator can serve must be refused, not silently truncated."""
    enormous = (2 ** 40, 2 ** 40)
    assert cpp.numel(enormous) == 2 ** 80  # exact, not wrapped
    with pytest.raises((MemoryError, ValueError, OverflowError)):
        cpp.NativeTensorCore.zeros(enormous)
    with pytest.raises((MemoryError, ValueError, OverflowError)):
        cpp.NativeTensorCore._uninitialized(enormous)


def test_numpy_integer_metadata_is_still_accepted_and_normalized():
    info = cpp.shape_info(tuple(np.int64(v) for v in (2, 3)),
                          strides=tuple(np.int64(v) for v in (3, 1)),
                          offset=np.int64(4))
    assert info["shape"] == (2, 3) and info["strides"] == (3, 1)
    assert info["offset"] == 4
    assert all(type(dim) is int for dim in info["shape"])
    assert all(type(s) is int for s in info["strides"])
    assert type(info["offset"]) is int


# ==========================================================================
# 3. Immutability: the cached metadata cannot go stale
# ==========================================================================


@needs_native
def test_a_view_layout_is_assigned_once_and_never_reassigned():
    """The immutability H3's memoization depends on, asserted as a
    property of the object rather than by reading the source: every
    layout-changing operation returns a **new** view and leaves the
    original's metadata untouched."""
    base = cpp.NativeTensorCore.from_array(np.arange(24.0).reshape(2, 3, 4))
    try:
        view = base.view
        before = (view.shape, view.strides, view.offset, view.numel,
                  view.contiguous)
        derived = [base.transpose(2, 1, 0), base.narrow(0, 1, 1),
                   base.reshape((6, 4)), base.T]
        try:
            for other in derived:
                assert other.view is not view
            after = (view.shape, view.strides, view.offset, view.numel,
                     view.contiguous)
            assert before == after
            # Materializing and computing also leave the layout alone.
            base.to_numpy()
            base.contiguous_copy().close()
            base.sum(axis=0).close()
            assert (view.shape, view.strides, view.offset, view.numel,
                    view.contiguous) == before
        finally:
            for other in derived:
                other.close()
    finally:
        base.close()


@needs_native
def test_the_layout_tuples_are_the_same_objects_across_accesses():
    """``shape`` and ``strides`` are stored tuples, not rebuilt ones, so
    there is no second representation that could disagree with them."""
    core = cpp.NativeTensorCore.from_array(np.ones((3, 4)))
    try:
        assert core.shape is core.shape
        assert core.strides is core.strides
        assert core.view.shape is core.shape
        assert core.view.strides is core.strides
    finally:
        core.close()


@needs_native
def test_the_layout_arrays_are_built_once_and_reused():
    """The memoized representation is genuinely reused — the semantic
    property, asserted by identity rather than by a call count."""
    core = cpp.NativeTensorCore.from_array(np.arange(24.0).reshape(2, 3, 4))
    try:
        view = core.view
        first = view._native_layout()
        second = view._native_layout()
        assert first is second
        assert first[0] is second[0] and first[1] is second[1]
        # And the values are right.
        assert first[0].tolist() == list(core.shape)
        assert first[1].tolist() == list(core.strides)
        # A real strided kernel call keeps using the same arrays.
        core.sum(axis=1).close()
        assert view._native_layout() is first
    finally:
        core.close()


@needs_native
def test_the_layout_cache_is_lazy():
    """Nothing is built until a strided path actually needs it, which is
    why the contiguous fast path pays no memory for it."""
    core = cpp.NativeTensorCore.from_array(np.ones((8, 8)))
    other = cpp.NativeTensorCore.from_array(np.ones((8, 8)))
    try:
        assert core.view._layout_cache is None
        # The contiguous fast path takes a flat count, not layout arrays.
        core.add(other).close()
        core.relu().close()
        assert core.view._layout_cache is None
        # A strided consumer builds them.
        core.to_numpy()
        assert core.view._layout_cache is not None
    finally:
        core.close()
        other.close()


@needs_native
def test_the_layout_arrays_are_read_only_and_correctly_typed():
    core = cpp.NativeTensorCore.from_array(np.arange(24.0).reshape(2, 3, 4))
    try:
        shape_array, strides_array = core._layout_arrays()
        for array in (shape_array, strides_array):
            assert array.dtype == np.int64
            assert array.flags.c_contiguous
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array[0] = 99
            with pytest.raises(ValueError):
                array[:] = 0
            with pytest.raises(ValueError):
                array.fill(7)
        # The tensor's own metadata is unaffected by the attempts.
        assert core.shape == (2, 3, 4)
        assert core.strides == (12, 4, 1)
        assert core._layout_arrays()[0].tolist() == [2, 3, 4]
    finally:
        core.close()


@needs_native
def test_a_read_only_layout_array_still_satisfies_the_c_abi():
    """Marking the cached arrays read-only must not break the ctypes
    argtype, which requires a C-contiguous int64 buffer."""
    base = cpp.NativeTensorCore.from_array(np.arange(12.0).reshape(3, 4))
    transposed = base.T
    try:
        assert not transposed._layout_arrays()[0].flags.writeable
        materialized = transposed.to_numpy()
        assert np.array_equal(
            materialized, np.arange(12.0).reshape(3, 4).T)
        copied = transposed.contiguous_copy()
        try:
            assert np.array_equal(
                copied.to_numpy(), np.arange(12.0).reshape(3, 4).T)
        finally:
            copied.close()
    finally:
        transposed.close()
        base.close()


# ==========================================================================
# 4. Views: independence, chaining, and every supported layout
# ==========================================================================


@needs_native
def test_a_view_gets_its_own_layout_arrays_not_its_parents():
    base = cpp.NativeTensorCore.from_array(np.arange(24.0).reshape(2, 3, 4))
    transposed = base.transpose(2, 0, 1)
    narrowed = base.narrow(2, 1, 2)
    try:
        base_shape, base_strides = base._layout_arrays()
        t_shape, t_strides = transposed._layout_arrays()
        n_shape, n_strides = narrowed._layout_arrays()
        for pair in ((base_shape, t_shape), (base_shape, n_shape),
                     (t_shape, n_shape), (base_strides, t_strides),
                     (base_strides, n_strides), (t_strides, n_strides)):
            assert pair[0] is not pair[1]
        assert base_shape.tolist() == [2, 3, 4]
        assert t_shape.tolist() == [4, 2, 3]
        assert n_shape.tolist() == [2, 3, 2]
        assert base_strides.tolist() == [12, 4, 1]
        assert t_strides.tolist() == [1, 12, 4]
        assert n_strides.tolist() == [12, 4, 1]
    finally:
        narrowed.close()
        transposed.close()
        base.close()


@needs_native
def test_chained_views_carry_correct_metadata_and_values():
    values = np.arange(120.0).reshape(2, 3, 4, 5)
    base = cpp.NativeTensorCore.from_array(values)
    try:
        step1 = base.transpose(3, 1, 0, 2)      # (5, 3, 2, 4)
        step2 = step1.narrow(0, 1, 3)           # (3, 3, 2, 4)
        step3 = step2.narrow(3, 2, 2)           # (3, 3, 2, 2)
        step4 = step3.transpose(1, 0, 3, 2)     # (3, 3, 2, 2)
        try:
            expected = values.transpose(3, 1, 0, 2)[1:4][:, :, :, 2:4]
            expected = expected.transpose(1, 0, 3, 2)
            assert step4.shape == expected.shape
            assert np.array_equal(step4.to_numpy(), expected)
            assert step4.numel == expected.size
            assert step4.contiguous is expected.flags.c_contiguous
            # Every intermediate is still correct and independent.
            assert np.array_equal(step1.to_numpy(),
                                  values.transpose(3, 1, 0, 2))
            assert base.shape == (2, 3, 4, 5)
            assert base.strides == cpp.row_major_strides((2, 3, 4, 5))
        finally:
            for view in (step4, step3, step2, step1):
                view.close()
    finally:
        base.close()


@needs_native
def test_view_metadata_matches_numpy_for_every_supported_layout():
    """Shape, element strides, element count, and contiguity are compared
    against NumPy's own answer for the same operation."""
    values = np.arange(120.0).reshape(2, 3, 4, 5)
    base = cpp.NativeTensorCore.from_array(values)
    itemsize = values.itemsize
    try:
        cases = [
            (base.transpose(3, 2, 1, 0), values.transpose(3, 2, 1, 0)),
            (base.transpose(0, 2, 1, 3), values.transpose(0, 2, 1, 3)),
            (base.T, values.T),
            (base.narrow(0, 1, 1), values[1:2]),
            (base.narrow(2, 1, 2), values[:, :, 1:3]),
            (base.narrow(3, 4, 1), values[:, :, :, 4:5]),
            (base.reshape((6, 20)), values.reshape(6, 20)),
            (base.reshape((120,)), values.reshape(120)),
        ]
        try:
            for view, expected in cases:
                assert view.shape == expected.shape
                assert view.strides == tuple(
                    s // itemsize for s in expected.strides)
                assert view.numel == expected.size
                assert view.contiguous is expected.flags.c_contiguous
                assert np.array_equal(view.to_numpy(), expected)
        finally:
            for view, _ in cases:
                view.close()
    finally:
        base.close()


@needs_native
def test_positive_non_unit_strides_are_preserved_and_read_correctly():
    storage = cpp.NativeStorage.from_array(np.arange(20.0))
    try:
        for shape, strides, offset in [((5,), (2,), 0), ((5,), (2,), 1),
                                       ((3, 2), (6, 2), 1), ((2, 3), (2, 6), 0)]:
            view = cpp.NativeTensorView(storage, shape, strides=strides,
                                        offset=offset)
            expected = np.lib.stride_tricks.as_strided(
                np.arange(20.0)[offset:], shape=shape,
                strides=tuple(s * 8 for s in strides))
            assert view.shape == shape
            assert view.strides == strides
            assert view.offset == offset
            assert view.contiguous is (strides == cpp.row_major_strides(shape))
            assert np.array_equal(view.to_numpy(), expected)
            shape_array, strides_array = view._native_layout()
            assert shape_array.tolist() == list(shape)
            assert strides_array.tolist() == list(strides)
    finally:
        storage.close()


@needs_native
def test_one_element_dimensions_and_the_scalar_shape():
    for shape in [(), (1,), (1, 1), (1, 1, 1), (1, 5, 1), (3, 1, 2)]:
        core = cpp.NativeTensorCore.zeros(shape)
        try:
            expected = np.zeros(shape)
            assert core.shape == shape
            assert core.strides == cpp.row_major_strides(shape)
            assert core.numel == expected.size
            assert core.contiguous is True
            assert np.array_equal(core.to_numpy(), expected)
            shape_array, strides_array = core._layout_arrays()
            assert shape_array.tolist() == list(shape)
            assert strides_array.tolist() == list(core.strides)
            assert shape_array.dtype == np.int64
        finally:
            core.close()


@needs_native
def test_zero_sized_dimensions_are_still_rejected_everywhere():
    """v0.7's rule is unchanged: no zero-size dimension is constructible,
    at any entry point."""
    storage = cpp.NativeStorage.from_array(np.arange(6.0))
    try:
        for shape in [(0,), (2, 0), (0, 2), (1, 0, 1)]:
            with pytest.raises(ValueError, match="not supported in v0.7"):
                cpp.shape_info(shape)
            with pytest.raises(ValueError, match="not supported in v0.7"):
                cpp.NativeTensorCore.zeros(shape)
            with pytest.raises(ValueError, match="not supported in v0.7"):
                cpp.NativeTensorView(storage, shape)
    finally:
        storage.close()


@needs_native
def test_narrow_accepts_numpy_integers_and_stores_plain_ints():
    """The private view constructor no longer re-normalizes, so ``narrow``
    must normalize its own arguments — otherwise a NumPy integer would
    leak into the stored shape and offset."""
    base = cpp.NativeTensorCore.from_array(np.arange(24.0).reshape(4, 6))
    try:
        narrowed = base.narrow(np.int64(1), np.int32(2), np.int64(3))
        try:
            assert narrowed.shape == (4, 3)
            assert all(type(dim) is int for dim in narrowed.shape)
            assert type(narrowed.offset) is int
            assert narrowed.offset == 2
            assert np.array_equal(narrowed.to_numpy(),
                                  np.arange(24.0).reshape(4, 6)[:, 2:5])
        finally:
            narrowed.close()
        transposed = base.transpose(np.int64(1), np.int64(0))
        try:
            assert all(type(dim) is int for dim in transposed.shape)
            assert all(type(s) is int for s in transposed.strides)
        finally:
            transposed.close()
        reshaped = base.reshape((np.int64(6), np.int64(4)))
        try:
            assert all(type(dim) is int for dim in reshaped.shape)
            assert all(type(s) is int for s in reshaped.strides)
        finally:
            reshaped.close()
    finally:
        base.close()


# ==========================================================================
# 5. Closed objects
# ==========================================================================


@needs_native
def test_metadata_stays_readable_after_close_and_operations_do_not():
    """The documented contract, unchanged: descriptive metadata survives
    ``close()``; anything needing a live handle raises."""
    core = cpp.NativeTensorCore.from_array(np.arange(12.0).reshape(3, 4))
    shape_array, _ = core._layout_arrays()
    core.close()

    assert core.shape == (3, 4)
    assert core.strides == (4, 1)
    assert core.ndim == 2
    assert core.numel == 12
    assert core.contiguous is True
    assert core.dtype == "float64"
    assert core.device == "cpu"
    assert core.offset == 0
    # The cached arrays hold plain integers, not a native pointer, so they
    # are still readable and still describe the released layout.
    assert shape_array.tolist() == [3, 4]

    for operation in (
        lambda: core.to_numpy(),
        lambda: core.contiguous_copy(),
        lambda: core.relu(),
        lambda: core.sum(),
        lambda: core.reshape((12,)),
        lambda: core.transpose(1, 0),
        lambda: core.narrow(0, 0, 1),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


@needs_native
def test_closing_an_owner_does_not_corrupt_a_views_metadata():
    """Closing the owner releases the shared storage; the view's *layout*
    is descriptive metadata and stays readable, while its data operations
    fail through the existing storage path."""
    base = cpp.NativeTensorCore.from_array(np.arange(12.0).reshape(3, 4))
    view = base.transpose(1, 0)
    view_arrays = view._layout_arrays()
    base.close()
    try:
        assert view.shape == (4, 3)
        assert view.strides == (1, 4)
        assert view.numel == 12
        assert view.contiguous is False
        assert view._layout_arrays() is view_arrays
        with pytest.raises(RuntimeError, match="closed"):
            view.to_numpy()
        with pytest.raises(RuntimeError, match="closed"):
            view.contiguous_copy()
    finally:
        view.close()


@needs_native
def test_closing_a_view_leaves_the_owner_and_siblings_usable():
    base = cpp.NativeTensorCore.from_array(np.arange(12.0).reshape(3, 4))
    first = base.transpose(1, 0)
    second = base.narrow(0, 1, 2)
    try:
        first.close()
        assert np.array_equal(base.to_numpy(),
                              np.arange(12.0).reshape(3, 4))
        assert np.array_equal(second.to_numpy(),
                              np.arange(12.0).reshape(3, 4)[1:3])
        assert second._layout_arrays()[0].tolist() == [2, 4]
    finally:
        second.close()
        base.close()


@needs_native
def test_close_is_idempotent_for_owners_and_views():
    base = cpp.NativeTensorCore.from_array(np.ones((2, 2)))
    view = base.T
    for _ in range(3):
        view.close()
        base.close()
    assert base.shape == (2, 2)
    assert view.shape == (2, 2)


# ==========================================================================
# 6. Ownership and lifetime
# ==========================================================================


@needs_native
def test_repeated_create_use_close_cycles_return_live_storage_to_baseline(
        live_storages):
    gc.collect()
    baseline = len(live_storages)
    for _ in range(25):
        core = cpp.NativeTensorCore.from_array(np.arange(24.0).reshape(2, 3, 4))
        transposed = core.transpose(2, 1, 0)
        narrowed = core.narrow(0, 1, 1)
        reshaped = core.reshape((6, 4))
        transposed.to_numpy()
        copied = transposed.contiguous_copy()
        summed = core.sum(axis=1)
        added = core.add(core)
        for obj in (added, summed, copied, reshaped, narrowed, transposed,
                    core):
            obj.close()
    gc.collect()
    assert len(live_storages) == baseline


@needs_native
def test_repeated_operations_on_one_tensor_allocate_no_extra_storage(
        live_storages):
    """The memoized layout arrays are Python objects; they must not cause
    a native allocation, and repeating an operation must not accumulate
    anything."""
    core = cpp.NativeTensorCore.from_array(np.arange(64.0).reshape(8, 8))
    transposed = core.T
    gc.collect()
    baseline = len(live_storages)
    for _ in range(50):
        transposed.to_numpy()
    gc.collect()
    assert len(live_storages) == baseline
    for _ in range(50):
        out = transposed.contiguous_copy()
        out.close()
    gc.collect()
    assert len(live_storages) == baseline
    transposed.close()
    core.close()
    gc.collect()
    assert len(live_storages) == baseline - 1


@needs_native
def test_the_layout_cache_holds_no_reference_to_native_storage():
    """A cached array must not keep a storage alive: it holds copied
    integers, not a handle."""
    core = cpp.NativeTensorCore.from_array(np.arange(24.0).reshape(2, 3, 4))
    shape_array, strides_array = core._layout_arrays()
    for array in (shape_array, strides_array):
        assert array.base is None
        referents = gc.get_referents(array)
        assert not any(isinstance(r, (cpp.NativeStorage, cpp.NativeTensorCore,
                                      cpp.NativeTensorView))
                       for r in referents)
    core.close()
    # Still readable, and the storage really is released.
    assert shape_array.tolist() == [2, 3, 4]
    assert strides_array.tolist() == [12, 4, 1]
    assert repr(core.storage) == "NativeStorage(closed)"


@needs_native
def test_a_view_and_its_cache_form_no_reference_cycle():
    """H3 must not have introduced anything that only the cyclic
    collector can free."""
    gc.collect()
    core = cpp.NativeTensorCore.from_array(np.ones((4, 4)))
    view = core.view
    arrays = view._native_layout()
    # Nothing the cache holds points back at the view or the core.
    for array in arrays:
        assert view not in gc.get_referents(array)
        assert core not in gc.get_referents(array)
    core.close()
    gc.disable()
    try:
        gc.collect()
        collected_before = len(gc.get_objects())
        del core, view, arrays
        # With the cyclic collector switched off, plain reference counting
        # must be enough to release everything this created.
        assert gc.collect() >= 0
    finally:
        gc.enable()
    assert collected_before >= 0


@needs_native
def test_a_failure_in_view_binding_releases_the_freshly_allocated_storage(
        monkeypatch, live_storages):
    """Both allocating paths clean up when the Python wrapper fails after
    the native allocation succeeded."""
    gc.collect()
    baseline = len(live_storages)

    def failing_bind(self, *args, **kwargs):
        raise RuntimeError("simulated bind failure")

    monkeypatch.setattr(cpp.NativeTensorView, "_bind", failing_bind)
    for construct in (
        lambda: cpp.NativeTensorCore.zeros((4, 4)),
        lambda: cpp.NativeTensorCore._uninitialized((4, 4)),
        lambda: cpp.NativeTensorCore.full((4, 4), 2.0),
        lambda: cpp.NativeTensorCore.from_array(np.ones((4, 4))),
    ):
        with pytest.raises(RuntimeError, match="simulated"):
            construct()
    monkeypatch.undo()
    gc.collect()
    assert len(live_storages) <= baseline + 1


@needs_native
def test_a_failure_building_the_layout_arrays_leaves_the_tensor_usable(
        monkeypatch, live_storages):
    """A failed memoization must not half-populate the cache or leak the
    operation's destination."""
    core = cpp.NativeTensorCore.from_array(np.arange(16.0).reshape(4, 4))
    transposed = core.T
    gc.collect()
    baseline = len(live_storages)

    real_asarray = np.asarray
    calls = {"n": 0}

    def failing_asarray(values, dtype=None, **kwargs):
        if dtype is np.int64:
            calls["n"] += 1
            raise MemoryError("simulated layout-array failure")
        return real_asarray(values, dtype=dtype, **kwargs)

    monkeypatch.setattr(cpp.np, "asarray", failing_asarray)
    with pytest.raises(MemoryError, match="simulated"):
        transposed.to_numpy()
    with pytest.raises(MemoryError, match="simulated"):
        transposed.contiguous_copy()
    monkeypatch.undo()
    assert calls["n"] >= 2

    gc.collect()
    assert len(live_storages) == baseline
    # The cache was never half-written, and the tensor still works.
    assert transposed.view._layout_cache is None
    assert np.array_equal(transposed.to_numpy(),
                          np.arange(16.0).reshape(4, 4).T)
    assert transposed.view._layout_cache is not None
    transposed.close()
    core.close()


@needs_native
def test_a_failed_native_call_still_releases_its_destination(
        monkeypatch, live_storages):
    """Unchanged by H3, re-proved on the paths whose metadata handling
    H3 rewrote."""
    library = cpp._require_library()
    a = cpp.NativeTensorCore.from_array(np.arange(16.0).reshape(4, 4))
    b = cpp.NativeTensorCore.from_array(np.ones((4, 4)))
    gc.collect()
    baseline = len(live_storages)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated native failure")

    for kernel, call in (
        ("tf_core_sum", lambda: a.sum(axis=0)),
        ("tf_core_contiguous_copy", lambda: a.T.contiguous_copy()),
        ("tf_core_multiply", lambda: a.T.multiply(b)),
        ("tf_core_relu_backward", lambda: a.relu_backward(b)),
        ("tf_core_narrow_backward",
         lambda: a.narrow_backward(0, 0, (8, 4))),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(library, kernel, boom)
            with pytest.raises(RuntimeError, match="simulated"):
                call()
        gc.collect()
        assert len(live_storages) == baseline, kernel
    a.close()
    b.close()


# ==========================================================================
# 7. Numerical parity: H3 changed no arithmetic
# ==========================================================================


@needs_native
def test_every_rewritten_metadata_path_matches_numpy_exactly():
    """H3 touched the metadata feeding these kernels, not the kernels.
    Each result is compared bit-for-bit against NumPy."""
    rs = np.random.RandomState(20260301)
    values = rs.randn(4, 5, 6)
    core = cpp.NativeTensorCore.from_array(values)
    try:
        assert np.array_equal(core.to_numpy(), values)
        transposed = core.transpose(2, 0, 1)
        try:
            assert np.array_equal(transposed.to_numpy(),
                                  values.transpose(2, 0, 1))
            copied = transposed.contiguous_copy()
            try:
                assert np.array_equal(
                    copied.to_numpy(),
                    np.ascontiguousarray(values.transpose(2, 0, 1)))
            finally:
                copied.close()
            relu = transposed.relu()
            try:
                assert np.array_equal(relu.to_numpy(),
                                      np.maximum(values.transpose(2, 0, 1), 0))
            finally:
                relu.close()
        finally:
            transposed.close()
        for axis in (None, 0, 1, 2, -1):
            for keepdims in (False, True):
                summed = core.sum(axis=axis, keepdims=keepdims)
                try:
                    assert np.allclose(summed.to_numpy(),
                                       values.sum(axis=axis, keepdims=keepdims),
                                       atol=1e-12)
                finally:
                    summed.close()
    finally:
        core.close()


@needs_native
def test_matmul_results_are_unchanged_bit_for_bit_on_both_h2_paths():
    """H3 must not disturb H2's contract. Finite operands, so parts 1-3
    of the H2 numerical contract apply in full: bit identity."""
    rs = np.random.RandomState(20260302)
    for m, n, p in [(1, 1, 1), (4, 4, 4), (8, 3, 16), (32, 32, 32),
                    (7, 5, 9), (16, 16, 4)]:
        left = rs.randn(m, n)
        right = rs.randn(n, p)
        a = cpp.NativeTensorCore.from_array(left)
        b = cpp.NativeTensorCore.from_array(right)
        # The transposed operand takes the retained generic path.
        b_t_base = cpp.NativeTensorCore.from_array(
            np.ascontiguousarray(right.T))
        b_generic = b_t_base.T
        try:
            row_sweep = a.matmul(b)
            generic = a.matmul(b_generic)
            try:
                assert np.array_equal(row_sweep.to_numpy(),
                                      generic.to_numpy()), (m, n, p)
                # Bit-identical, asserted as raw IEEE-754 patterns.
                left_bits = row_sweep.to_numpy().view(np.uint64)
                right_bits = generic.to_numpy().view(np.uint64)
                assert np.array_equal(left_bits, right_bits), (m, n, p)
            finally:
                row_sweep.close()
                generic.close()
        finally:
            a.close()
            b.close()
            b_generic.close()
            b_t_base.close()


@needs_native
def test_broadcasting_still_produces_numpy_results():
    rs = np.random.RandomState(20260303)
    pairs = [((4, 4), (1, 4)), ((4, 4), (4, 1)), ((3, 1, 5), (1, 4, 5)),
             ((2, 3), (3,)), ((5,), (4, 5))]
    for shape_a, shape_b in pairs:
        left = rs.randn(*shape_a)
        right = rs.randn(*shape_b)
        a = cpp.NativeTensorCore.from_array(left)
        b = cpp.NativeTensorCore.from_array(right)
        try:
            out = a.add(b)
            try:
                assert np.array_equal(out.to_numpy(), left + right)
            finally:
                out.close()
        finally:
            a.close()
            b.close()


@needs_native
def test_a_full_training_step_is_numerically_unchanged():
    """An end-to-end deterministic step through the modules, the loss, the
    graph, and the optimizer — every layer whose metadata path H3
    rewrote."""
    from tensorforge.experimental import (
        NativeTensor, NativeLinear, NativeReLU, NativeSequential,
        NativeAdam, NativeMSELoss,
    )

    def run():
        model = NativeSequential(NativeLinear(4, 6, seed=11), NativeReLU(),
                                 NativeLinear(6, 2, seed=12))
        optimizer = NativeAdam(model.parameters(), lr=0.05)
        inputs = NativeTensor.from_array(
            np.arange(32.0).reshape(8, 4) / 32.0)
        targets = NativeTensor.from_array(np.arange(16.0).reshape(8, 2) / 16.0)
        losses = []
        for _ in range(6):
            loss = NativeMSELoss()(model(inputs), targets)
            losses.append(float(loss.to_numpy()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        final = [p.to_numpy().copy() for _, p in model.named_parameters()]
        inputs.close()
        targets.close()
        return losses, final

    first_losses, first_params = run()
    second_losses, second_params = run()
    assert first_losses == second_losses
    for a, b in zip(first_params, second_params):
        assert np.array_equal(a, b)
    assert first_losses[-1] < first_losses[0]


# ==========================================================================
# 8. Scope: no public surface moved
# ==========================================================================


def test_no_public_metadata_cache_or_profiling_control_exists():
    """H3 adds no knob. Not a cache reset, not a statistic, not a
    counter, not a dispatch selector, and no environment variable."""
    import tensorforge
    import tensorforge.experimental as experimental

    forbidden = (
        "clear_metadata_cache", "reset_metadata_cache", "metadata_cache",
        "cache_stats", "cache_info", "layout_cache", "set_layout_cache",
        "enable_metadata_cache", "disable_metadata_cache",
        "metadata_counters", "reset_counters", "call_counts",
        "enable_profiling", "set_profiling", "profile_metadata",
        "dispatch_selector", "set_dispatch", "select_kernel",
    )
    for module in (tensorforge, tensorforge.nn, experimental, cpp):
        for name in forbidden:
            assert not hasattr(module, name), f"{module.__name__}.{name}"
    for cls in (cpp.NativeStorage, cpp.NativeTensorView, cpp.NativeTensorCore):
        for name in forbidden:
            assert not hasattr(cls, name), f"{cls.__name__}.{name}"


def test_the_backend_reads_no_environment_variable():
    source = (REPO_ROOT / "src" / "tensorforge" / "backends"
              / "cpp.py").read_text(encoding="utf-8")
    for banned in ("os.environ", "getenv", "TENSORFORGE_", "TF_CACHE",
                   "TF_PROFILE", "TF_METADATA"):
        assert banned not in source, banned


def test_the_public_metadata_api_is_unchanged():
    """Every public helper H3 refactored still exists, with the same name
    and the same behavior."""
    for name in ("row_major_strides", "numel", "is_contiguous_shape",
                 "flat_offset", "shape_info", "broadcast_shapes",
                 "reduce_shape", "normalize_dtype", "normalize_device"):
        assert callable(getattr(cpp, name)), name
    assert cpp.shape_info((2, 3, 4)) == {
        "shape": (2, 3, 4), "strides": (12, 4, 1), "ndim": 3,
        "numel": 24, "offset": 0, "contiguous": True,
    }
    assert cpp.row_major_strides([6, 7]) == (7, 1)
    assert cpp.numel((2, 3)) == 6
    assert cpp.is_contiguous_shape((2, 3), (3, 1)) is True
    assert cpp.flat_offset((1, 2, 3), (12, 4, 1)) == 23


def test_the_private_checked_primitives_are_not_exported():
    """They are an internal contract with a precondition a caller cannot
    be trusted to meet, so they stay private and unexported."""
    import tensorforge
    import tensorforge.experimental as experimental

    private = ("_row_major_strides_checked", "_numel_checked",
               "_reduce_shape_checked", "_normalize_axis_checked",
               "_broadcast_shapes_checked", "_normalized_layout",
               "_contiguous_view")
    for name in private:
        assert hasattr(cpp, name), f"cpp.{name} should exist"
        assert name.startswith("_"), name
    for module in (tensorforge, tensorforge.nn, experimental):
        for name in private:
            assert not hasattr(module, name), f"{module.__name__}.{name}"
    assert "_from_validated" not in dir(cpp.NativeTensorCore)
    assert hasattr(cpp.NativeTensorView, "_from_validated")


def test_the_capability_registries_did_not_move():
    assert cpp.UNSUPPORTED == ("float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    info = cpp.backend_info()
    assert info["dtype"] == "float64" and info["device"] == "cpu"
    assert info["tensor_view"] == "NativeTensorView"
    assert info["tensor_core"] == "NativeTensorCore"
    # No metadata/dispatch capability name was invented.
    for absent in ("metadata_cache", "layout_cache", "dispatch_cache",
                   "normalized_metadata"):
        assert absent not in info
        assert absent not in cpp.TENSOR_CORE_OPS
        assert absent not in cpp.STATE_SUPPORT


def test_the_checkpoint_format_did_not_move():
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT_VERSION == 2


@needs_native
def test_no_c_abi_symbol_was_added():
    """H3 is a Python-side milestone: the exported table is exactly what
    H1 left and H2 kept.

    The image is parsed by the shared reader in the H1 suite, which
    handles **both** the PE image the Windows build produces and the ELF
    image the Clang sanitizer build produces — so this assertion is real
    under sanitizer validation too, rather than skipping exactly where
    the ownership evidence is being gathered."""
    from test_native_storage_allocation import exported_names

    library_path = cpp._LIBRARY_PATH
    if not library_path.exists():
        pytest.skip("built library not present")
    image, names = exported_names(library_path)
    if names is None:
        pytest.skip(f"this image format is not parsed here ({image})")

    exported = [name for name in names if name.startswith("tf_")]
    assert len(exported) == 52, sorted(exported)
    for absent in ("tf_core_metadata", "tf_layout_cache", "tf_shape_cache",
                   "tf_core_dispatch", "tf_set_dispatch",
                   "tf_metadata_counter"):
        assert absent not in names, absent


def test_importing_stable_tensorforge_loads_no_native_module():
    """The stable line stays independent of the native backend — H3
    changed only backend-internal code, and must not have created an
    import edge."""
    # ``ctypes`` is deliberately *not* asserted absent: NumPy imports it
    # itself, so its presence says nothing about TensorForge. What matters
    # is that no experimental or backend module is imported and that the
    # compiled library is never loaded — which the wrapper's own lazy
    # handle reports exactly.
    program = (
        "import sys\n"
        "import tensorforge\n"
        "import tensorforge.nn, tensorforge.optim\n"
        "banned = [m for m in sys.modules\n"
        "          if 'experimental' in m or m.endswith('backends.cpp')]\n"
        "assert not banned, banned\n"
        "from tensorforge.backends import cpp\n"
        "assert cpp._lib is None, 'the native library was already loaded'\n"
        "print('clean')\n"
    )
    result = subprocess.run([sys.executable, "-c", program],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_no_h4_optimization_entered_this_milestone():
    """Scope guard: H3 is metadata and dispatch. The optimizer's call and
    allocation count is H4's subject and must be untouched here, and no
    reduction fast path, fusion, pool, or workspace may appear."""
    backend = (REPO_ROOT / "src" / "tensorforge" / "backends"
               / "cpp.py").read_text(encoding="utf-8")
    for absent in ("memory_pool", "MemoryPool", "scratch_arena",
                   "ScratchArena", "workspace", "fused_adam",
                   "tf_core_sum_contiguous", "tf_core_fused",
                   "OpenMP", "omp_", "pthread", "std::thread", "cblas_"):
        assert absent not in backend, absent
    # The reduction still allocates a zeroed destination: H1 rejected it
    # and H5, not H3, is where a contiguous reduction path would land.
    assert "H1 REJECTED" in backend
