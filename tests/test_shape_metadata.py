"""Tests for the backend shape/stride metadata layer.

These helpers are pure Python (the metadata contract for a future
native storage object), so they run whether or not the compiled
backend is built. Strides count elements, not bytes.
"""

import numpy as np
import pytest

from tensorforge.backends.cpp import (
    broadcast_shapes,
    flat_offset,
    is_contiguous_shape,
    numel,
    row_major_strides,
    shape_info,
)


def test_row_major_strides():
    assert row_major_strides(()) == ()
    assert row_major_strides((5,)) == (1,)
    assert row_major_strides((2, 3)) == (3, 1)
    assert row_major_strides((2, 3, 4)) == (12, 4, 1)
    assert row_major_strides([6, 7]) == (7, 1)  # lists work too


def test_row_major_strides_matches_numpy_element_strides():
    # NumPy strides are in bytes; divide by itemsize to compare.
    for shape in ((3,), (2, 5), (2, 3, 4), (1, 1, 7)):
        arr = np.zeros(shape)
        numpy_element_strides = tuple(s // arr.itemsize for s in arr.strides)
        assert row_major_strides(shape) == numpy_element_strides


def test_numel():
    assert numel(()) == 1  # a scalar holds one element
    assert numel((5,)) == 5
    assert numel((2, 3)) == 6
    assert numel((2, 3, 4)) == 24


def test_is_contiguous_shape_true_for_row_major():
    assert is_contiguous_shape((), ()) is True
    assert is_contiguous_shape((5,), (1,)) is True
    assert is_contiguous_shape((2, 3, 4), (12, 4, 1)) is True


def test_is_contiguous_shape_false_for_other_layouts():
    # Transposed-style strides for a (3, 2) view of a (2, 3) buffer.
    assert is_contiguous_shape((3, 2), (1, 3)) is False
    # Every-other-element slice of a length-10 buffer.
    assert is_contiguous_shape((5,), (2,)) is False
    # Column-major layout is not row-major contiguous.
    assert is_contiguous_shape((2, 3), (1, 2)) is False


def test_flat_offset():
    assert flat_offset((1, 2, 3), (12, 4, 1)) == 23
    assert flat_offset((0, 0, 0), (12, 4, 1)) == 0
    assert flat_offset((), ()) == 0  # scalar: the single element
    # Cross-check against NumPy on a strided view.
    arr = np.arange(24.0).reshape(2, 3, 4)
    element_strides = tuple(s // arr.itemsize for s in arr.strides)
    for index in ((0, 1, 2), (1, 0, 3), (1, 2, 0)):
        assert arr.flat[flat_offset(index, element_strides)] == arr[index]


def test_flat_offset_with_base_offset():
    assert flat_offset((1, 1), (3, 1), offset=10) == 14
    assert flat_offset((), (), offset=7) == 7


def test_flat_offset_allows_negative_strides():
    # Pure stride math: negative strides (reversed views) are legal.
    assert flat_offset((2,), (-1,), offset=4) == 2


def test_shape_info_defaults_to_row_major():
    info = shape_info((2, 3, 4))
    assert info == {
        "shape": (2, 3, 4),
        "strides": (12, 4, 1),
        "ndim": 3,
        "numel": 24,
        "offset": 0,
        "contiguous": True,
    }


def test_shape_info_scalar():
    info = shape_info(())
    assert info["shape"] == ()
    assert info["strides"] == ()
    assert info["ndim"] == 0
    assert info["numel"] == 1
    assert info["contiguous"] is True


def test_shape_info_with_explicit_noncontiguous_strides():
    info = shape_info((3, 2), strides=(1, 3), offset=5)
    assert info["shape"] == (3, 2)
    assert info["strides"] == (1, 3)
    assert info["offset"] == 5
    assert info["contiguous"] is False


def test_shape_info_accepts_lists_and_returns_tuples():
    info = shape_info([2, 3], strides=[3, 1])
    assert info["shape"] == (2, 3)
    assert info["strides"] == (3, 1)
    assert isinstance(info["shape"], tuple)
    assert isinstance(info["strides"], tuple)


def test_invalid_inputs_raise_clearly():
    with pytest.raises(ValueError, match="positive"):
        row_major_strides((2, -3))
    with pytest.raises(ValueError, match="not supported in v0.7"):
        row_major_strides((2, 0))
    with pytest.raises(TypeError, match="ints"):
        row_major_strides((2, 3.5))
    with pytest.raises(TypeError, match="ints"):
        row_major_strides((2, True))
    with pytest.raises(TypeError, match="sequence"):
        numel(7)
    with pytest.raises(ValueError, match="same length"):
        is_contiguous_shape((2, 3), (1,))
    with pytest.raises(ValueError, match="same length"):
        flat_offset((1, 2), (4,))
    with pytest.raises(TypeError, match="ints"):
        flat_offset((1.0, 2), (4, 1))
    with pytest.raises(TypeError, match="offset"):
        flat_offset((1, 2), (4, 1), offset=0.5)
    with pytest.raises(ValueError, match="same length"):
        shape_info((2, 3), strides=(1, 2, 3))
    with pytest.raises(TypeError, match="offset"):
        shape_info((2, 3), offset="zero")


def test_numpy_integer_dimensions_are_accepted():
    shape = tuple(np.int64(v) for v in (2, 3))
    assert row_major_strides(shape) == (3, 1)
    assert numel(shape) == 6


# ---------------------------------------------------------------------------
# broadcast_shapes (v1.17): pure NumPy-style broadcast shape inference.
# ---------------------------------------------------------------------------


def test_broadcast_shapes_scalar_and_tensor():
    assert broadcast_shapes((), (3, 4)) == (3, 4)
    assert broadcast_shapes((3, 4), ()) == (3, 4)
    assert broadcast_shapes((), ()) == ()


def test_broadcast_shapes_vector_and_matrix():
    assert broadcast_shapes((4,), (3, 4)) == (3, 4)  # left-pad (4,) -> (1, 4)
    assert broadcast_shapes((3, 4), (4,)) == (3, 4)
    assert broadcast_shapes((5,), (5,)) == (5,)


def test_broadcast_shapes_same_rank_stretching():
    assert broadcast_shapes((3, 1), (1, 4)) == (3, 4)
    assert broadcast_shapes((3, 1), (3, 4)) == (3, 4)
    assert broadcast_shapes((1, 4), (3, 4)) == (3, 4)


def test_broadcast_shapes_leading_dimension():
    assert broadcast_shapes((1, 3, 4), (2, 3, 4)) == (2, 3, 4)
    assert broadcast_shapes((3, 4), (2, 3, 4)) == (2, 3, 4)  # left-pad


def test_broadcast_shapes_both_operands_broadcast():
    assert broadcast_shapes((1, 3, 1), (2, 1, 5)) == (2, 3, 5)
    assert broadcast_shapes((2, 1, 5), (1, 3, 1)) == (2, 3, 5)  # commutative


def test_broadcast_shapes_matches_numpy_where_defined():
    for a, b in (((), (3, 4)), ((3, 1), (1, 4)), ((4,), (3, 4)),
                 ((1, 3, 1), (2, 1, 5)), ((2, 1, 5), (1, 3, 1))):
        assert broadcast_shapes(a, b) == np.broadcast_shapes(a, b)


def test_broadcast_shapes_incompatible_raises_naming_both_shapes():
    with pytest.raises(ValueError) as excinfo:
        broadcast_shapes((2, 3), (4, 3))
    message = str(excinfo.value)
    assert "(2, 3)" in message and "(4, 3)" in message
    # A trailing-axis conflict is caught too.
    with pytest.raises(ValueError, match=r"\(3,\).*\(4,\)|\(4,\).*\(3,\)"):
        broadcast_shapes((3,), (4,))


def test_broadcast_shapes_rejects_invalid_dims():
    # Reuses the v0.7 shape validation: zero/negative dims and non-ints.
    with pytest.raises(ValueError, match="positive"):
        broadcast_shapes((2, -1), (2, 3))
    with pytest.raises(ValueError, match="not supported"):
        broadcast_shapes((2, 0), (2, 3))
    with pytest.raises(TypeError, match="ints"):
        broadcast_shapes((2, 1.5), (2, 3))
