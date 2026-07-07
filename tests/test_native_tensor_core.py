"""Tests for NativeTensorCore, the first native tensor runtime object.

Backend-dependent, so the module skips cleanly when the compiled
backend is not built. Cleanup is explicit via close() or context
managers — nothing depends on garbage-collection timing.
"""

import numpy as np
import pytest

from tensorforge.backends import cpp

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)


def test_native_tensor_core_is_importable():
    from tensorforge.backends.cpp import NativeTensorCore  # noqa: F401

    assert callable(cpp.NativeTensorCore)


def test_from_array_preserves_shape_values_and_layout():
    base = np.arange(24.0).reshape(2, 3, 4)
    with cpp.NativeTensorCore.from_array(base) as tensor:
        assert tensor.shape == (2, 3, 4)
        assert tensor.strides == (12, 4, 1)
        assert tensor.contiguous is True
        result = tensor.to_numpy()
        assert result.dtype == np.float64
        assert np.array_equal(result, base)


def test_from_array_accepts_nested_lists():
    with cpp.NativeTensorCore.from_array([[1, 2], [3, 4]]) as tensor:
        assert tensor.shape == (2, 2)
        assert tensor.to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_zeros():
    with cpp.NativeTensorCore.zeros((3, 2)) as tensor:
        assert tensor.shape == (3, 2)
        assert tensor.to_numpy().tolist() == [[0.0, 0.0]] * 3
    with cpp.NativeTensorCore.zeros(()) as scalar:
        assert scalar.ndim == 0
        assert float(scalar.to_numpy()) == 0.0


def test_full():
    with cpp.NativeTensorCore.full((2, 2), 7.0) as tensor:
        assert tensor.to_numpy().tolist() == [[7.0, 7.0], [7.0, 7.0]]
    with cpp.NativeTensorCore.full((3,), -2) as ints:  # ints convert
        assert ints.to_numpy().tolist() == [-2.0, -2.0, -2.0]


def test_properties():
    with cpp.NativeTensorCore.from_array(np.zeros((2, 3))) as tensor:
        assert tensor.shape == (2, 3)
        assert tensor.strides == (3, 1)
        assert tensor.offset == 0
        assert tensor.ndim == 2
        assert tensor.numel == 6
        assert tensor.contiguous is True
        assert isinstance(tensor.storage, cpp.NativeStorage)
        assert isinstance(tensor.view, cpp.NativeTensorView)
        assert "shape=(2, 3)" in repr(tensor)


def test_to_numpy_is_fresh_and_independent():
    with cpp.NativeTensorCore.from_array([1.0, 2.0]) as tensor:
        first = tensor.to_numpy()
        second = tensor.to_numpy()
        assert not np.shares_memory(first, second)
        first[0] = 99.0
        assert tensor.to_numpy().tolist() == [1.0, 2.0]


def test_contiguous_copy_is_independent():
    with cpp.NativeTensorCore.from_array([[1.0, 2.0], [3.0, 4.0]]) as tensor:
        with tensor.contiguous_copy() as copy:
            assert isinstance(copy, cpp.NativeTensorCore)
            assert copy is not tensor
            assert copy.storage is not tensor.storage
            assert copy.contiguous is True
            assert copy.to_numpy().tolist() == tensor.to_numpy().tolist()
            # Mutating the copy's storage leaves the original alone.
            copy.storage.fill(0.0)
            assert tensor.to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_tensor_retains_its_storage():
    # No external reference to the storage is kept; the tensor must.
    tensor = cpp.NativeTensorCore.from_array([5.0, 6.0])
    assert tensor.to_numpy().tolist() == [5.0, 6.0]
    tensor.close()


def test_invalid_shapes_rejected():
    with pytest.raises(ValueError, match="positive"):
        cpp.NativeTensorCore.zeros((2, -1))
    with pytest.raises(ValueError, match="not supported"):
        cpp.NativeTensorCore.zeros((2, 0))
    with pytest.raises(TypeError, match="ints"):
        cpp.NativeTensorCore.full((2, 1.5), 0.0)


def test_empty_arrays_rejected():
    with pytest.raises(ValueError, match="positive int"):
        cpp.NativeTensorCore.from_array([])
    with pytest.raises(ValueError, match="positive int"):
        cpp.NativeTensorCore.from_array(np.zeros((0, 3)))


def test_invalid_fill_values_rejected():
    with pytest.raises((TypeError, ValueError)):
        cpp.NativeTensorCore.full((2,), "seven")
    with pytest.raises((TypeError, ValueError)):
        cpp.NativeTensorCore.full((2,), None)


def test_close_behavior():
    tensor = cpp.NativeTensorCore.from_array([1.0])
    tensor.close()
    tensor.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        tensor.to_numpy()
    with pytest.raises(RuntimeError, match="closed"):
        tensor.contiguous_copy()
    assert tensor.shape == (1,)  # metadata stays readable
    assert "closed" in repr(tensor)


def test_context_manager_closes():
    with cpp.NativeTensorCore.zeros((2,)) as tensor:
        assert tensor.to_numpy().tolist() == [0.0, 0.0]
    with pytest.raises(RuntimeError, match="closed"):
        tensor.to_numpy()


def test_advanced_constructor_validation():
    with pytest.raises(TypeError, match="NativeStorage"):
        cpp.NativeTensorCore(np.zeros(4), None)
    storage_a = cpp.NativeStorage.from_array([1.0, 2.0])
    storage_b = cpp.NativeStorage.from_array([3.0, 4.0])
    view_b = cpp.NativeTensorView(storage_b, (2,))
    with pytest.raises(ValueError, match="over the given storage"):
        cpp.NativeTensorCore(storage_a, view_b)
    storage_a.close()
    storage_b.close()


def test_backend_info_advertises_tensor_core_and_kernels_stay_kernels():
    info = cpp.backend_info()
    assert info["tensor_core"] == "NativeTensorCore"
    for name in ("NativeStorage", "NativeTensorView", "NativeTensorCore"):
        assert name not in cpp.list_kernels()
