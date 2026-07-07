"""Tests for NativeTensorView: NativeStorage + shape/stride metadata.

Backend-dependent, so the module skips cleanly when the compiled
backend is not built. NumPy strided views are used as the reference
for every materialization case.
"""

import numpy as np
import pytest

from tensorforge.backends import cpp

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)


def _storage(values):
    return cpp.NativeStorage.from_array(values)


def test_native_tensor_view_is_importable():
    from tensorforge.backends.cpp import NativeTensorView  # noqa: F401

    assert callable(cpp.NativeTensorView)


def test_from_array_preserves_shape_and_values():
    array = np.arange(24.0).reshape(2, 3, 4)
    view = cpp.NativeTensorView.from_array(array)
    assert view.shape == (2, 3, 4)
    assert view.strides == (12, 4, 1)
    assert view.contiguous is True
    assert np.array_equal(view.to_numpy(), array)


def test_contiguous_view_over_storage():
    with _storage([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) as storage:
        view = cpp.NativeTensorView(storage, shape=(2, 3))
        assert view.to_numpy().tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_strided_1d_view():
    with _storage(np.arange(10.0)) as storage:
        view = cpp.NativeTensorView(storage, shape=(5,), strides=(2,))
        assert view.to_numpy().tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]
        assert view.contiguous is False


def test_transposed_2d_view():
    base = np.arange(6.0).reshape(2, 3)
    with _storage(base) as storage:
        view = cpp.NativeTensorView(storage, shape=(3, 2), strides=(1, 3))
        assert np.array_equal(view.to_numpy(), base.T)


def test_view_with_nonzero_offset():
    with _storage(np.arange(10.0)) as storage:
        view = cpp.NativeTensorView(storage, shape=(2, 2), strides=(2, 1), offset=3)
        # offsets 3,4 / 5,6
        assert view.to_numpy().tolist() == [[3.0, 4.0], [5.0, 6.0]]
        assert view.offset == 3


def test_negative_strides_reverse_a_view():
    with _storage(np.arange(5.0)) as storage:
        view = cpp.NativeTensorView(storage, shape=(5,), strides=(-1,), offset=4)
        assert view.to_numpy().tolist() == [4.0, 3.0, 2.0, 1.0, 0.0]


def test_negative_stride_2d_matches_numpy():
    base = np.arange(12.0).reshape(3, 4)
    with _storage(base) as storage:
        # Rows reversed: like base[::-1] — first row starts at offset 8.
        view = cpp.NativeTensorView(storage, shape=(3, 4), strides=(-4, 1), offset=8)
        assert np.array_equal(view.to_numpy(), base[::-1])


def test_scalar_view():
    with _storage([7.0, 8.0]) as storage:
        view = cpp.NativeTensorView(storage, shape=(), offset=1)
        assert view.ndim == 0
        assert view.numel == 1
        result = view.to_numpy()
        assert result.shape == ()
        assert float(result) == 8.0


def test_to_numpy_is_fresh_and_storage_unmutated():
    with _storage([1.0, 2.0, 3.0, 4.0]) as storage:
        view = cpp.NativeTensorView(storage, shape=(2, 2))
        first = view.to_numpy()
        second = view.to_numpy()
        assert not np.shares_memory(first, second)
        first[0, 0] = 99.0
        assert storage.to_numpy().tolist() == [1.0, 2.0, 3.0, 4.0]
        assert view.to_numpy()[0, 0] == 1.0


def test_contiguous_copy_creates_row_major_storage():
    base = np.arange(6.0).reshape(2, 3)
    with _storage(base) as storage:
        transposed = cpp.NativeTensorView(storage, shape=(3, 2), strides=(1, 3))
        with transposed.contiguous_copy() as copy:
            assert isinstance(copy, cpp.NativeStorage)
            assert copy.size == 6
            assert copy.to_numpy().tolist() == base.T.ravel().tolist()
        # The original storage still holds the original layout.
        assert storage.to_numpy().tolist() == base.ravel().tolist()


def test_view_properties():
    with _storage(np.arange(24.0)) as storage:
        view = cpp.NativeTensorView(storage, shape=(2, 3), strides=(3, 1), offset=4)
        assert view.shape == (2, 3)
        assert view.strides == (3, 1)
        assert view.offset == 4
        assert view.ndim == 2
        assert view.numel == 6
        assert view.contiguous is True  # strides are row-major for the shape
        assert "shape=(2, 3)" in repr(view)


def test_rejects_non_storage():
    with pytest.raises(TypeError, match="NativeStorage"):
        cpp.NativeTensorView(np.zeros(4), shape=(4,))


def test_rejects_closed_storage_at_construction_and_use():
    storage = _storage([1.0, 2.0])
    storage.close()
    with pytest.raises(RuntimeError, match="closed"):
        cpp.NativeTensorView(storage, shape=(2,))

    live = _storage([1.0, 2.0])
    view = cpp.NativeTensorView(live, shape=(2,))
    live.close()  # closing after view creation makes operations fail
    with pytest.raises(RuntimeError, match="closed"):
        view.to_numpy()


def test_rejects_invalid_shape_strides_offset():
    with _storage(np.arange(6.0)) as storage:
        with pytest.raises(ValueError, match="positive"):
            cpp.NativeTensorView(storage, shape=(2, -3))
        with pytest.raises(TypeError, match="ints"):
            cpp.NativeTensorView(storage, shape=(2, 3), strides=(3.0, 1))
        with pytest.raises(ValueError, match="same length"):
            cpp.NativeTensorView(storage, shape=(2, 3), strides=(1,))
        with pytest.raises(TypeError, match="offset"):
            cpp.NativeTensorView(storage, shape=(2, 3), offset=1.5)


def test_rejects_out_of_bounds_views():
    with _storage(np.arange(6.0)) as storage:  # valid offsets 0..5
        with pytest.raises(ValueError, match="outside"):
            cpp.NativeTensorView(storage, shape=(7,))  # too long
        with pytest.raises(ValueError, match="outside"):
            cpp.NativeTensorView(storage, shape=(2, 3), offset=1)  # reaches 6
        with pytest.raises(ValueError, match="outside"):
            cpp.NativeTensorView(storage, shape=(3,), strides=(-1,), offset=1)  # reaches -1
        with pytest.raises(ValueError, match="outside"):
            cpp.NativeTensorView(storage, shape=(), offset=6)  # scalar past the end


def test_backend_info_advertises_tensor_view_and_kernels_stay_kernels():
    info = cpp.backend_info()
    assert info["tensor_view"] == "NativeTensorView"
    assert "NativeTensorView" not in cpp.list_kernels()
    assert "NativeStorage" not in cpp.list_kernels()
