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


# ---------------------------------------------------------------------------
# v1.1: metadata-only view operations
# ---------------------------------------------------------------------------


def test_reshape_is_a_shared_storage_view():
    base = np.arange(6.0).reshape(2, 3)
    with cpp.NativeTensorCore.from_array(base) as tensor:
        view = tensor.reshape((3, 2))
        assert isinstance(view, cpp.NativeTensorCore)
        assert view.shape == (3, 2)
        assert view.strides == (2, 1)
        assert view.storage is tensor.storage  # shared, not copied
        assert np.array_equal(view.to_numpy(), base.reshape(3, 2))
        # Writing through the shared storage is visible in both.
        tensor.storage.fill(9.0)
        assert view.to_numpy().tolist() == [[9.0, 9.0]] * 3


def test_reshape_rejects_wrong_numel_and_bad_shapes():
    with cpp.NativeTensorCore.zeros((2, 3)) as tensor:
        with pytest.raises(ValueError, match="cannot reshape"):
            tensor.reshape((4, 2))
        with pytest.raises(ValueError, match="positive"):
            tensor.reshape((2, -3))


def test_reshape_rejects_non_contiguous():
    with cpp.NativeTensorCore.zeros((2, 3)) as tensor:
        transposed = tensor.transpose()
        assert transposed.contiguous is False
        with pytest.raises(ValueError, match="contiguous"):
            transposed.reshape((6,))


def test_transpose_default_reverses_axes():
    base = np.arange(24.0).reshape(2, 3, 4)
    with cpp.NativeTensorCore.from_array(base) as tensor:
        rev = tensor.transpose()
        assert rev.shape == (4, 3, 2)
        assert rev.strides == (1, 4, 12)
        assert np.array_equal(rev.to_numpy(), base.transpose())


def test_transpose_explicit_axes():
    base = np.arange(24.0).reshape(2, 3, 4)
    with cpp.NativeTensorCore.from_array(base) as tensor:
        moved = tensor.transpose(1, 0, 2)
        assert np.array_equal(moved.to_numpy(), base.transpose(1, 0, 2))
        two_d = tensor.reshape((6, 4)).transpose(1, 0)
        assert np.array_equal(two_d.to_numpy(), base.reshape(6, 4).T)


def test_transpose_scalar_and_1d_are_noops():
    with cpp.NativeTensorCore.from_array([1.0, 2.0, 3.0]) as vector:
        assert vector.transpose().to_numpy().tolist() == [1.0, 2.0, 3.0]
        assert vector.T.shape == (3,)
    with cpp.NativeTensorCore.zeros(()) as scalar:
        assert scalar.T.ndim == 0


def test_transpose_rejects_bad_axes():
    with cpp.NativeTensorCore.zeros((2, 3, 4)) as tensor:
        with pytest.raises(ValueError, match="permutation"):
            tensor.transpose(0, 1)  # missing an axis
        with pytest.raises(ValueError, match="permutation"):
            tensor.transpose(0, 1, 1)  # duplicate
        with pytest.raises(ValueError, match="permutation"):
            tensor.transpose(0, 1, 3)  # out of range


def test_T_matches_numpy():
    base = np.arange(6.0).reshape(2, 3)
    with cpp.NativeTensorCore.from_array(base) as tensor:
        assert np.array_equal(tensor.T.to_numpy(), base.T)
    cube = np.arange(8.0).reshape(2, 2, 2)
    with cpp.NativeTensorCore.from_array(cube) as tensor:
        assert np.array_equal(tensor.T.to_numpy(), cube.T)  # reversed axes


def test_narrow():
    base = np.arange(12.0).reshape(3, 4)
    with cpp.NativeTensorCore.from_array(base) as tensor:
        rows = tensor.narrow(0, 1, 2)
        assert rows.shape == (2, 4)
        assert rows.strides == (4, 1)
        assert rows.offset == 4  # start * strides[0]
        assert rows.storage is tensor.storage
        assert np.array_equal(rows.to_numpy(), base[1:3])

        cols = tensor.narrow(1, 1, 2)
        assert cols.shape == (3, 2)
        assert cols.offset == 1
        assert cols.contiguous is False
        assert np.array_equal(cols.to_numpy(), base[:, 1:3])


def test_narrow_rejects_invalid_arguments():
    with cpp.NativeTensorCore.zeros((3, 4)) as tensor:
        with pytest.raises(ValueError, match="dim"):
            tensor.narrow(2, 0, 1)
        with pytest.raises(ValueError, match="out of bounds"):
            tensor.narrow(0, 2, 2)  # reaches row 4 of 3
        with pytest.raises(ValueError, match="out of bounds"):
            tensor.narrow(0, -1, 2)
        with pytest.raises(ValueError, match="out of bounds"):
            tensor.narrow(0, 0, 0)  # zero-size shapes unsupported
        with pytest.raises(TypeError, match="int"):
            tensor.narrow(0, 0.5, 1)


def test_chained_views():
    base = np.arange(24.0).reshape(4, 6)
    with cpp.NativeTensorCore.from_array(base) as tensor:
        chained = tensor.transpose().narrow(0, 1, 3)  # cols 1..3, transposed
        assert np.array_equal(chained.to_numpy(), base.T[1:4])
        assert chained.storage is tensor.storage


def test_contiguous_copy_of_noncontiguous_view():
    base = np.arange(6.0).reshape(2, 3)
    with cpp.NativeTensorCore.from_array(base) as tensor:
        transposed = tensor.T
        with transposed.contiguous_copy() as copy:
            assert copy.contiguous is True
            assert copy.strides == (2, 1)
            assert copy.storage is not tensor.storage  # independent memory
            assert np.array_equal(copy.to_numpy(), base.T)
            tensor.storage.fill(0.0)  # mutating the original...
            assert np.array_equal(copy.to_numpy(), base.T)  # ...leaves the copy


def test_closing_a_view_leaves_the_owner_and_siblings_usable():
    tensor = cpp.NativeTensorCore.from_array([[1.0, 2.0], [3.0, 4.0]])
    view_a = tensor.T
    view_b = tensor.reshape((4,))
    view_a.close()
    with pytest.raises(RuntimeError, match="closed"):
        view_a.to_numpy()
    # The owner and the sibling view still work: shared storage lives.
    assert tensor.to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert view_b.to_numpy().tolist() == [1.0, 2.0, 3.0, 4.0]
    tensor.close()


def test_closing_the_owner_invalidates_views():
    tensor = cpp.NativeTensorCore.from_array([1.0, 2.0])
    view = tensor.reshape((2,))
    tensor.close()
    # The owner released the shared storage; the view's data ops fail.
    with pytest.raises(RuntimeError, match="closed"):
        view.to_numpy()
    view.close()  # still safe: views never touch shared storage


def test_view_operations_on_closed_core_raise():
    tensor = cpp.NativeTensorCore.zeros((2, 2))
    tensor.close()
    for operation in (
        lambda: tensor.reshape((4,)),
        tensor.transpose,
        lambda: tensor.narrow(0, 0, 1),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


def test_backend_info_advertises_tensor_core_and_kernels_stay_kernels():
    info = cpp.backend_info()
    assert info["tensor_core"] == "NativeTensorCore"
    for name in ("NativeStorage", "NativeTensorView", "NativeTensorCore"):
        assert name not in cpp.list_kernels()
