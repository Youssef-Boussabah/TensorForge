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


# ---------------------------------------------------------------------------
# v1.2: native kernels over tensor cores
# ---------------------------------------------------------------------------


def test_relu_on_contiguous_tensor():
    base = np.array([[-1.0, 2.0], [-3.0, 4.0]])
    with cpp.NativeTensorCore.from_array(base) as tensor:
        result = tensor.relu()
        assert isinstance(result, cpp.NativeTensorCore)
        assert result.contiguous is True
        assert np.array_equal(result.to_numpy(), np.maximum(base, 0.0))
        result.close()


def test_relu_on_noncontiguous_views():
    base = np.arange(-6.0, 6.0).reshape(3, 4)
    with cpp.NativeTensorCore.from_array(base) as tensor:
        assert np.array_equal(tensor.T.relu().to_numpy(), np.maximum(base.T, 0.0))
        narrowed = tensor.narrow(1, 1, 2)
        assert np.array_equal(narrowed.relu().to_numpy(), np.maximum(base[:, 1:3], 0.0))


@pytest.mark.parametrize("method,numpy_op", (
    ("add", np.add),
    ("subtract", np.subtract),
    ("multiply", np.multiply),
))
def test_binary_ops_match_numpy(method, numpy_op):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(3, 4))
    y = rng.normal(size=(3, 4))
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
        result = getattr(a, method)(b)
        assert isinstance(result, cpp.NativeTensorCore)
        assert result.contiguous is True
        assert np.array_equal(result.to_numpy(), numpy_op(x, y))


@pytest.mark.parametrize("method,numpy_op", (
    ("add", np.add),
    ("subtract", np.subtract),
    ("multiply", np.multiply),
))
def test_binary_ops_on_noncontiguous_views(method, numpy_op):
    rng = np.random.default_rng(13)
    x = rng.normal(size=(3, 4))
    y = rng.normal(size=(4, 3))
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
        # A transposed view against a contiguous tensor of matching shape.
        result = getattr(a.T, method)(b)
        assert np.array_equal(result.to_numpy(), numpy_op(x.T, y))
        # Two strided views: narrowed slices of each.
        left = a.narrow(1, 1, 3)                 # (3, 3) view of x
        right = b.narrow(0, 0, 3)                # (3, 3) view of y
        result = getattr(left, method)(right)
        assert np.array_equal(result.to_numpy(), numpy_op(x[:, 1:4], y[:3]))


def test_outputs_are_independent_and_inputs_unmutated():
    x = np.array([[1.0, -2.0]])
    y = np.array([[3.0, 4.0]])
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
        result = a.add(b)
        assert result.storage is not a.storage and result.storage is not b.storage
        result.storage.fill(0.0)  # mutating the output...
        assert a.to_numpy().tolist() == [[1.0, -2.0]]  # ...touches no input
        assert b.to_numpy().tolist() == [[3.0, 4.0]]
        assert a.relu().to_numpy().tolist() == [[1.0, 0.0]]
        assert a.to_numpy().tolist() == [[1.0, -2.0]]  # relu didn't mutate


def test_binary_ops_reject_shape_mismatch():
    # (2, 3) and (3, 2) do not broadcast (2 vs 3 on axis 0), so the
    # broadcast path raises a ValueError naming both shapes.
    with cpp.NativeTensorCore.zeros((2, 3)) as a, cpp.NativeTensorCore.zeros((3, 2)) as b:
        for method in ("add", "subtract", "multiply"):
            with pytest.raises(ValueError, match="broadcast") as excinfo:
                getattr(a, method)(b)
            assert "(2, 3)" in str(excinfo.value) and "(3, 2)" in str(excinfo.value)


def test_binary_ops_reject_non_tensor_core_operands():
    with cpp.NativeTensorCore.zeros((2,)) as a:
        for bad in (np.zeros(2), [0.0, 0.0], 3.0):
            with pytest.raises(TypeError, match="NativeTensorCore"):
                a.add(bad)


def test_kernels_reject_closed_tensors():
    a = cpp.NativeTensorCore.zeros((2,))
    b = cpp.NativeTensorCore.zeros((2,))
    b.close()
    with pytest.raises(RuntimeError, match="closed"):
        a.add(b)  # closed right operand
    a.close()
    with pytest.raises(RuntimeError, match="closed"):
        a.relu()  # closed self


def test_scalar_tensor_core_kernels():
    with cpp.NativeTensorCore.full((), -3.0) as a, cpp.NativeTensorCore.full((), 5.0) as b:
        assert float(a.relu().to_numpy()) == 0.0
        assert float(a.add(b).to_numpy()) == 2.0
        assert float(a.multiply(b).to_numpy()) == -15.0


# ---------------------------------------------------------------------------
# v1.3: TensorCore matmul
# ---------------------------------------------------------------------------


def test_matmul_matches_numpy():
    rng = np.random.default_rng(14)
    for (m, n, p) in ((2, 3, 4), (5, 1, 3), (4, 4, 4)):
        x = rng.normal(size=(m, n))
        y = rng.normal(size=(n, p))
        with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
            result = a.matmul(b)
            assert isinstance(result, cpp.NativeTensorCore)
            assert result.shape == (m, p)
            assert result.contiguous is True
            assert np.allclose(result.to_numpy(), x @ y, atol=1e-12)


def test_matmul_known_values():
    a = cpp.NativeTensorCore.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = cpp.NativeTensorCore.from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    assert a.matmul(b).to_numpy().tolist() == [[58.0, 64.0], [139.0, 154.0]]
    a.close()
    b.close()


def test_matmul_vector_like_matrices():
    with cpp.NativeTensorCore.from_array([[1.0, 2.0, 3.0]]) as row:      # (1, 3)
        with cpp.NativeTensorCore.from_array([[4.0], [5.0], [6.0]]) as col:  # (3, 1)
            inner = row.matmul(col)
            assert inner.shape == (1, 1)
            assert inner.to_numpy()[0, 0] == 32.0
            outer = col.matmul(row)
            assert outer.shape == (3, 3)
            assert np.array_equal(
                outer.to_numpy(),
                np.array([[4.0], [5.0], [6.0]]) @ np.array([[1.0, 2.0, 3.0]]),
            )


def test_matmul_with_transposed_left():
    rng = np.random.default_rng(15)
    x = rng.normal(size=(3, 2))
    y = rng.normal(size=(3, 4))
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
        result = a.T.matmul(b)  # (2, 3) view @ (3, 4)
        assert np.allclose(result.to_numpy(), x.T @ y, atol=1e-12)


def test_matmul_with_transposed_right():
    rng = np.random.default_rng(16)
    x = rng.normal(size=(2, 3))
    y = rng.normal(size=(4, 3))
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
        result = a.matmul(b.T)  # (2, 3) @ (3, 4) view
        assert np.allclose(result.to_numpy(), x @ y.T, atol=1e-12)


def test_matmul_with_narrowed_views():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(4, 6))
    y = rng.normal(size=(5, 4))
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
        left = a.narrow(1, 1, 3)    # (4, 3) view of x
        right = b.narrow(0, 2, 3)   # (3, 4) view of y
        result = left.matmul(right)
        assert np.allclose(result.to_numpy(), x[:, 1:4] @ y[2:5], atol=1e-12)


def test_matmul_output_independent_and_inputs_unmutated():
    x = np.array([[1.0, 2.0]])
    y = np.array([[3.0], [4.0]])
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
        result = a.matmul(b)
        assert result.storage is not a.storage and result.storage is not b.storage
        result.storage.fill(0.0)
        assert a.to_numpy().tolist() == [[1.0, 2.0]]
        assert b.to_numpy().tolist() == [[3.0], [4.0]]


def test_matmul_rejects_bad_operands():
    with cpp.NativeTensorCore.zeros((2, 3)) as a:
        with pytest.raises(TypeError, match="NativeTensorCore"):
            a.matmul(np.ones((3, 2)))
        with cpp.NativeTensorCore.zeros((3,)) as vector:
            with pytest.raises(ValueError, match="2-D right"):
                a.matmul(vector)
            with pytest.raises(ValueError, match="2-D left"):
                vector.matmul(a)
        with cpp.NativeTensorCore.zeros((4, 2)) as mismatched:
            with pytest.raises(ValueError, match="inner dimensions"):
                a.matmul(mismatched)


def test_matmul_rejects_closed_tensors():
    a = cpp.NativeTensorCore.zeros((2, 2))
    b = cpp.NativeTensorCore.zeros((2, 2))
    b.close()
    with pytest.raises(RuntimeError, match="closed"):
        a.matmul(b)
    a.close()
    with pytest.raises(RuntimeError, match="closed"):
        a.matmul(a)


def test_backend_info_advertises_tensor_core_kernels():
    info = cpp.backend_info()
    assert info["tensor_core_kernels"] == ("relu", "add", "subtract", "multiply", "matmul")
    # TensorCore methods are not raw-buffer kernels; list_kernels is
    # unchanged ("relu" appears there as the raw kernel, "add" only in
    # its "elementwise_add" form).
    assert "add" not in cpp.list_kernels()
    assert "elementwise_add" in cpp.list_kernels()


def test_backend_info_advertises_tensor_core_and_kernels_stay_kernels():
    info = cpp.backend_info()
    assert info["tensor_core"] == "NativeTensorCore"
    for name in ("NativeStorage", "NativeTensorView", "NativeTensorCore"):
        assert name not in cpp.list_kernels()


# ---------------------------------------------------------------------------
# v1.14: contiguous elementwise fast path
#
# Contiguous relu/add/subtract/multiply take a flat, index-free kernel;
# strided views keep the generic odometer kernel. The two paths must be
# bit-for-bit equal, so these tests use exact equality (np.array_equal),
# never a tolerance. There is no assertion about *which* kernel ran (the
# choice is an internal traversal detail) — correctness of both the
# contiguous and the non-contiguous case is what pins the behavior.
# ---------------------------------------------------------------------------


BINARY_OPS = (
    ("add", np.add),
    ("subtract", np.subtract),
    ("multiply", np.multiply),
)


def _noncontiguous_same_values(x):
    """An owning core plus a non-contiguous view over it that
    materializes to exactly ``x``. Storing x.T contiguously and
    transposing the view back yields row-major-mismatched strides (so the
    generic odometer path runs) over the original values. The owner is
    returned too and must be kept alive while the view is used."""
    owner = cpp.NativeTensorCore.from_array(np.ascontiguousarray(x.T))
    view = owner.T
    assert view.contiguous is False
    assert np.array_equal(view.to_numpy(), x)
    return owner, view


def test_relu_contiguous_fast_path_matches_numpy_exactly():
    rng = np.random.default_rng(140)
    x = rng.normal(size=(4, 5))
    with cpp.NativeTensorCore.from_array(x) as a:
        assert a.contiguous is True
        result = a.relu()
        assert result.contiguous is True
        assert np.array_equal(result.to_numpy(), np.maximum(x, 0.0))
        result.close()


@pytest.mark.parametrize("method,numpy_op", BINARY_OPS)
def test_binary_contiguous_fast_path_matches_numpy_exactly(method, numpy_op):
    rng = np.random.default_rng(141)
    x = rng.normal(size=(4, 5))
    y = rng.normal(size=(4, 5))
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
        assert a.contiguous and b.contiguous
        result = getattr(a, method)(b)
        assert result.contiguous is True
        assert np.array_equal(result.to_numpy(), numpy_op(x, y))
        result.close()


def test_relu_fast_path_equals_generic_path_on_identical_values():
    # The fast path (contiguous input) and the generic odometer path
    # (a strided view with the same values) must agree exactly.
    rng = np.random.default_rng(142)
    x = rng.normal(size=(3, 4))
    with cpp.NativeTensorCore.from_array(x) as contiguous:
        owner, strided = _noncontiguous_same_values(x)
        fast = contiguous.relu().to_numpy()
        generic = strided.relu().to_numpy()
        assert np.array_equal(fast, generic)
        assert np.array_equal(fast, np.maximum(x, 0.0))
        owner.close()


@pytest.mark.parametrize("method,numpy_op", BINARY_OPS)
def test_binary_fast_path_equals_generic_path_on_identical_values(method, numpy_op):
    rng = np.random.default_rng(143)
    x = rng.normal(size=(3, 4))
    y = rng.normal(size=(3, 4))
    with cpp.NativeTensorCore.from_array(x) as ca, cpp.NativeTensorCore.from_array(y) as cb:
        owner_a, sa = _noncontiguous_same_values(x)
        owner_b, sb = _noncontiguous_same_values(y)
        fast = getattr(ca, method)(cb).to_numpy()
        generic = getattr(sa, method)(sb).to_numpy()
        assert np.array_equal(fast, generic)
        assert np.array_equal(fast, numpy_op(x, y))
        owner_a.close()
        owner_b.close()


def test_fast_path_handles_nonzero_offset_contiguous_row_slice():
    # narrow(0, ...) keeps row-major strides but shifts the offset, so
    # the result is contiguous with a nonzero offset — the fast path must
    # start from data + offset, not offset 0.
    rng = np.random.default_rng(144)
    x = rng.normal(size=(5, 3))
    with cpp.NativeTensorCore.from_array(x) as a:
        rows = a.narrow(0, 1, 3)  # x[1:4], contiguous, offset 3
        assert rows.contiguous is True
        assert rows.offset == 3
        assert np.array_equal(rows.relu().to_numpy(), np.maximum(x[1:4], 0.0))
        other = a.narrow(0, 2, 3)  # x[2:5], contiguous, offset 6
        assert other.contiguous is True and other.offset == 6
        for method, numpy_op in BINARY_OPS:
            got = getattr(rows, method)(other).to_numpy()
            assert np.array_equal(got, numpy_op(x[1:4], x[2:5]))


def test_fast_path_on_scalar_and_size_one_dimensions():
    # Scalars (numel == 1) and size-1 dimensions are contiguous by the
    # exact row-major stride test, so they exercise the flat kernel.
    with cpp.NativeTensorCore.full((), -3.0) as s, cpp.NativeTensorCore.full((), 5.0) as t:
        assert s.contiguous and t.contiguous
        assert float(s.relu().to_numpy()) == 0.0
        assert float(s.add(t).to_numpy()) == 2.0
        assert float(s.subtract(t).to_numpy()) == -8.0
        assert float(s.multiply(t).to_numpy()) == -15.0
    x = np.array([[1.0, -2.0, 3.0, -4.0]])   # (1, 4)
    y = np.array([[5.0, 6.0, -7.0, 8.0]])
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
        assert a.contiguous and b.contiguous
        assert np.array_equal(a.relu().to_numpy(), np.maximum(x, 0.0))
        assert np.array_equal(a.add(b).to_numpy(), x + y)
    col_x = x.reshape(4, 1)                   # (4, 1)
    col_y = y.reshape(4, 1)
    with cpp.NativeTensorCore.from_array(col_x) as a, cpp.NativeTensorCore.from_array(col_y) as b:
        assert a.contiguous and b.contiguous
        assert np.array_equal(a.multiply(b).to_numpy(), col_x * col_y)


def test_generic_fallback_still_serves_noncontiguous_views():
    # The odometer path is retained: transposed and inner-narrowed views
    # are non-contiguous and must keep matching NumPy exactly.
    rng = np.random.default_rng(145)
    x = rng.normal(size=(3, 4))
    y = rng.normal(size=(4, 3))
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
        assert a.T.contiguous is False
        assert np.array_equal(a.T.relu().to_numpy(), np.maximum(x.T, 0.0))
        for method, numpy_op in BINARY_OPS:
            # transposed view (non-contiguous) against a contiguous tensor
            assert np.array_equal(getattr(a.T, method)(b).to_numpy(), numpy_op(x.T, y))
        inner = a.narrow(1, 1, 2)  # narrowing an inner axis -> non-contiguous
        assert inner.contiguous is False
        assert np.array_equal(inner.relu().to_numpy(), np.maximum(x[:, 1:3], 0.0))


def test_fast_path_output_is_independent_and_inputs_unmutated():
    # Same ownership guarantee as the generic path: the flat kernel
    # writes a fresh contiguous output that aliases neither input.
    x = np.array([[1.0, -2.0], [3.0, 4.0]])
    y = np.array([[5.0, 6.0], [-7.0, 8.0]])
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
        result = a.add(b)
        assert result.storage is not a.storage and result.storage is not b.storage
        result.storage.fill(0.0)  # mutating the output touches no input
        assert np.array_equal(a.to_numpy(), x)
        assert np.array_equal(b.to_numpy(), y)
        assert np.array_equal(a.relu().to_numpy(), np.maximum(x, 0.0))
        assert np.array_equal(a.to_numpy(), x)  # relu didn't mutate self


# ---------------------------------------------------------------------------
# v1.17: native broadcasting for add/subtract/multiply
#
# The three-way dispatch: same-shape contiguous keeps the v1.14 fast
# path, same-shape strided keeps the generic odometer, and differing but
# compatible shapes take the broadcast traversal (zero-stride reads over
# a freshly allocated contiguous output). NumPy is the reference — the
# native result must equal it exactly (np.array_equal).
# ---------------------------------------------------------------------------


BROADCAST_SHAPE_PAIRS = (
    ((), (3, 4)),          # scalar + tensor
    ((3, 4), ()),          # tensor + scalar
    ((3, 1), (1, 4)),      # same-rank, both stretch
    ((4,), (3, 4)),        # leading-dimension (left-pad) broadcast
    ((3, 4), (4,)),        # symmetric
    ((1, 3, 1), (2, 1, 5)),  # higher-rank, both stretch
    ((2, 1), (2, 3)),      # one axis stretches
)


@pytest.mark.parametrize("shape_a,shape_b", BROADCAST_SHAPE_PAIRS)
@pytest.mark.parametrize("method,numpy_op", BINARY_OPS)
def test_broadcasting_matches_numpy(shape_a, shape_b, method, numpy_op):
    rng = np.random.default_rng(170)
    x = rng.normal(size=shape_a)
    y = rng.normal(size=shape_b)
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
        result = getattr(a, method)(b)
        expected = numpy_op(x, y)
        assert result.shape == expected.shape
        assert result.contiguous is True  # output is row-major contiguous
        assert np.array_equal(result.to_numpy(), expected)
        result.close()


def test_broadcasting_with_transposed_operand():
    # A non-contiguous (transposed) operand broadcasting against a vector:
    # real axes keep their strides, the broadcast axis reads stride 0.
    rng = np.random.default_rng(171)
    x = rng.normal(size=(3, 4))
    y = rng.normal(size=(4,))
    # owner laid out as x.T so that owner.T is a strided (3, 4) view == x.
    with cpp.NativeTensorCore.from_array(np.ascontiguousarray(x.T)) as owner, \
         cpp.NativeTensorCore.from_array(y) as b:
        strided = owner.T
        assert strided.contiguous is False
        for method, numpy_op in BINARY_OPS:
            assert np.array_equal(getattr(strided, method)(b).to_numpy(), numpy_op(x, y))


def test_broadcasting_with_narrowed_nonzero_offset_operand():
    # A narrowed operand with a nonzero offset and a size-1 axis is a
    # legitimate broadcastable operand.
    rng = np.random.default_rng(172)
    base = rng.normal(size=(3, 4))
    row = rng.normal(size=(1, 4))
    with cpp.NativeTensorCore.from_array(base) as a, cpp.NativeTensorCore.from_array(row) as b:
        col = a.narrow(1, 2, 1)  # (3, 1) view, offset 2
        assert col.shape == (3, 1) and col.offset == 2
        result = col.multiply(b)  # (3, 1) * (1, 4) -> (3, 4)
        assert result.shape == (3, 4)
        assert np.array_equal(result.to_numpy(), base[:, 2:3] * row)


def test_broadcasting_output_is_contiguous_and_operands_unchanged():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    y = np.array([10.0, 20.0, 30.0])                   # (3,)
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
        result = a.add(b)
        assert result.shape == (2, 3)
        assert result.contiguous is True
        assert result.strides == (3, 1)
        assert result.storage is not a.storage and result.storage is not b.storage
        result.storage.fill(0.0)  # mutating the output...
        assert np.array_equal(a.to_numpy(), x)  # ...leaves both operands
        assert np.array_equal(b.to_numpy(), y)


def test_broadcasting_rejects_incompatible_shapes_naming_both():
    with cpp.NativeTensorCore.from_array(np.ones((2, 3))) as a, \
         cpp.NativeTensorCore.from_array(np.ones((4, 3))) as b:
        for method in ("add", "subtract", "multiply"):
            with pytest.raises(ValueError, match="broadcast") as excinfo:
                getattr(a, method)(b)
            assert "(2, 3)" in str(excinfo.value) and "(4, 3)" in str(excinfo.value)


def test_broadcasting_does_not_disturb_same_shape_paths():
    # Regression guard: same-shape contiguous (fast path) and same-shape
    # strided (generic odometer) still compute correctly after the
    # broadcast dispatch was added.
    rng = np.random.default_rng(173)
    x = rng.normal(size=(3, 4))
    y = rng.normal(size=(3, 4))
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
        # A: same shape, both contiguous -> v1.14 fast path.
        assert a.contiguous and b.contiguous
        assert np.array_equal(a.add(b).to_numpy(), x + y)
    # B: same shape, both strided (transposed) -> generic odometer.
    with cpp.NativeTensorCore.from_array(np.ascontiguousarray(x.T)) as oa, \
         cpp.NativeTensorCore.from_array(np.ascontiguousarray(y.T)) as ob:
        sa, sb = oa.T, ob.T
        assert sa.shape == sb.shape and not sa.contiguous
        assert np.array_equal(sa.multiply(sb).to_numpy(), x * y)


# ---------------------------------------------------------------------------
# v1.19: native sum/mean reductions
#
# NumPy is the reference. Float sums are order-sensitive, so reduction
# VALUES are compared with np.allclose (not np.array_equal); output
# SHAPES are compared exactly.
# ---------------------------------------------------------------------------


REDUCE_CASES = [
    (shape, axis, keep)
    for shape in ((2, 3), (2, 3, 4), (5,))
    for axis in (None, 0, -1)
    for keep in (False, True)
]


@pytest.mark.parametrize("shape,axis,keepdims", REDUCE_CASES)
@pytest.mark.parametrize("method", ("sum", "mean"))
def test_reductions_match_numpy(shape, axis, keepdims, method):
    rng = np.random.default_rng(190)
    x = rng.normal(size=shape)
    numpy_op = np.sum if method == "sum" else np.mean
    with cpp.NativeTensorCore.from_array(x) as a:
        result = getattr(a, method)(axis=axis, keepdims=keepdims)
        expected = numpy_op(x, axis=axis, keepdims=keepdims)
        assert isinstance(result, cpp.NativeTensorCore)
        assert result.shape == np.shape(expected)
        assert result.contiguous is True  # output row-major contiguous
        assert np.allclose(result.to_numpy(), expected)
        result.close()


def test_reduction_axis0_and_axis1_values():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with cpp.NativeTensorCore.from_array(x) as a:
        assert np.allclose(a.sum(axis=0).to_numpy(), [5.0, 7.0, 9.0])
        assert np.allclose(a.sum(axis=1).to_numpy(), [6.0, 15.0])
        assert np.allclose(a.mean(axis=0).to_numpy(), [2.5, 3.5, 4.5])
        assert np.allclose(a.mean(axis=1).to_numpy(), [2.0, 5.0])
        assert np.allclose(a.sum().to_numpy(), 21.0)
        assert np.allclose(a.mean().to_numpy(), 3.5)


def test_reduction_negative_axis_equals_positive():
    rng = np.random.default_rng(191)
    x = rng.normal(size=(2, 3, 4))
    with cpp.NativeTensorCore.from_array(x) as a:
        assert np.allclose(a.sum(axis=-1).to_numpy(), a.sum(axis=2).to_numpy())
        assert np.allclose(a.mean(axis=-2).to_numpy(), a.mean(axis=1).to_numpy())


def test_reduction_keepdims_shapes():
    with cpp.NativeTensorCore.zeros((2, 3, 4)) as a:
        assert a.sum(axis=1).shape == (2, 4)
        assert a.sum(axis=1, keepdims=True).shape == (2, 1, 4)
        assert a.sum(keepdims=True).shape == (1, 1, 1)
        assert a.sum().shape == ()


def test_scalar_reductions():
    with cpp.NativeTensorCore.full((), -3.0) as s:
        assert s.sum().shape == ()
        assert float(s.sum().to_numpy()) == -3.0
        assert float(s.mean().to_numpy()) == -3.0
        # A scalar has no axes: an integer axis is out of bounds.
        with pytest.raises(ValueError, match=r"axis 0.*\(\)"):
            s.sum(axis=0)


def test_reduction_over_transposed_view():
    # Layout independence: a strided (transposed) input reduces correctly.
    rng = np.random.default_rng(192)
    x = rng.normal(size=(3, 4))
    with cpp.NativeTensorCore.from_array(np.ascontiguousarray(x.T)) as owner:
        view = owner.T  # (3, 4), non-contiguous, materializes to x
        assert view.contiguous is False
        assert np.allclose(view.sum(axis=0).to_numpy(), x.sum(axis=0))
        assert np.allclose(view.mean(axis=1).to_numpy(), x.mean(axis=1))
        assert np.allclose(view.sum().to_numpy(), x.sum())


def test_reduction_over_narrowed_nonzero_offset_view():
    rng = np.random.default_rng(193)
    x = rng.normal(size=(5, 4))
    with cpp.NativeTensorCore.from_array(x) as a:
        rows = a.narrow(0, 1, 3)  # x[1:4], contiguous, offset 4
        assert rows.offset == 4
        assert np.allclose(rows.sum(axis=1).to_numpy(), x[1:4].sum(axis=1))
        cols = a.narrow(1, 1, 2)  # (5, 2) inner narrow -> non-contiguous
        assert cols.contiguous is False
        assert np.allclose(cols.mean(axis=0).to_numpy(), x[:, 1:3].mean(axis=0))


def test_reduction_after_broadcasted_elementwise():
    # Chains A2 (broadcasting) into A3 (reductions): sum a broadcast result.
    rng = np.random.default_rng(194)
    x = rng.normal(size=(2, 3))
    bias = rng.normal(size=(3,))
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(bias) as b:
        summed = a.add(b).sum(axis=0)
        assert np.allclose(summed.to_numpy(), (x + bias).sum(axis=0))


def test_reduction_output_independent_and_input_unchanged():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with cpp.NativeTensorCore.from_array(x) as a:
        result = a.sum(axis=0)
        assert result.storage is not a.storage
        result.storage.fill(0.0)  # mutating the output...
        assert np.array_equal(a.to_numpy(), x)  # ...leaves the input
        # mean does not mutate the input either.
        _ = a.mean(axis=1)
        assert np.array_equal(a.to_numpy(), x)


def test_reduction_invalid_axis_and_types_raise():
    with cpp.NativeTensorCore.zeros((2, 3)) as a:
        for bad in (2, -3):
            with pytest.raises(ValueError) as excinfo:
                a.sum(axis=bad)
            assert str(bad) in str(excinfo.value) and "(2, 3)" in str(excinfo.value)
        with pytest.raises(TypeError, match="axis"):
            a.sum(axis=1.0)
        with pytest.raises(TypeError, match="keepdims"):
            a.mean(keepdims="yes")


def test_reduction_on_closed_core_raises():
    a = cpp.NativeTensorCore.zeros((2, 2))
    a.close()
    with pytest.raises(RuntimeError, match="closed"):
        a.sum()
    with pytest.raises(RuntimeError, match="closed"):
        a.mean(axis=0)


# ---------------------------------------------------------------------------
# v1.21: native dtype/device metadata (float64/cpu only)
#
# Metadata is explicit and inspectable but does not change any compute:
# the only supported dtype/device is float64/cpu, owned by NativeStorage
# and surfaced read-only through NativeTensorCore. Unsupported values are
# rejected at construction (never silently coerced), and every op/view
# carries the metadata through unchanged.
# ---------------------------------------------------------------------------


def test_storage_default_dtype_device_and_readable_after_close():
    storage = cpp.NativeStorage.from_array([1.0, 2.0, 3.0])
    assert storage.dtype == "float64"
    assert storage.device == "cpu"
    storage.close()
    # Metadata stays readable after close, like size.
    assert storage.dtype == "float64" and storage.device == "cpu"


def test_storage_rejects_unsupported_dtype_and_device():
    # "float16" stands where "float32" used to: float32 became a supported
    # dtype at Phase I milestone I9, so it belongs in the acceptance test
    # below rather than here.
    with pytest.raises(ValueError, match="float16"):
        cpp.NativeStorage(4, dtype="float16")
    with pytest.raises(ValueError, match="cuda"):
        cpp.NativeStorage(4, device="cuda")
    with pytest.raises(TypeError, match="dtype"):
        cpp.NativeStorage(4, dtype=object())
    with pytest.raises(TypeError, match="device"):
        cpp.NativeStorage(4, device=7)


def test_core_exposes_dtype_device_and_defaults():
    for tensor in (
        cpp.NativeTensorCore.from_array([[1.0, 2.0], [3.0, 4.0]]),
        cpp.NativeTensorCore.zeros((2, 3)),
        cpp.NativeTensorCore.full((2,), 5.0),
    ):
        assert tensor.dtype == "float64"
        assert tensor.device == "cpu"
        tensor.close()


def test_core_explicit_dtype_device_args_accept_defaults():
    with cpp.NativeTensorCore.zeros((2, 2), dtype="float64", device="cpu") as z:
        assert z.dtype == "float64" and z.device == "cpu"
    with cpp.NativeTensorCore.full((3,), 1.0, dtype="float64", device="cpu") as f:
        assert f.dtype == "float64" and f.device == "cpu"
    with cpp.NativeTensorCore.from_array([1.0], dtype=None, device="cpu") as a:
        assert a.dtype == "float64" and a.device == "cpu"


def test_core_rejects_unsupported_dtype_device_before_allocation():
    for ctor in (
        lambda: cpp.NativeTensorCore.zeros((2, 2), dtype="float16"),
        lambda: cpp.NativeTensorCore.full((2,), 0.0, device="cuda"),
        lambda: cpp.NativeTensorCore.from_array([1.0], dtype="int64"),
    ):
        with pytest.raises(ValueError):
            ctor()


def test_core_accepts_both_supported_dtypes():
    """The other half of the boundary, since Phase I milestone I9: the two
    supported widths really are constructible through every public
    factory, and the storage tag is what reports them."""
    for dtype in ("float64", "float32"):
        for ctor in (
            lambda: cpp.NativeTensorCore.zeros((2, 2), dtype=dtype),
            lambda: cpp.NativeTensorCore.full((2,), 1.0, dtype=dtype),
            lambda: cpp.NativeTensorCore.from_array([1.0, 2.0], dtype=dtype),
        ):
            tensor = ctor()
            try:
                assert tensor.dtype == dtype
                assert tensor.storage.dtype == dtype
            finally:
                tensor.close()


def test_core_dtype_device_readable_after_close():
    tensor = cpp.NativeTensorCore.from_array([1.0, 2.0])
    tensor.close()
    # Like shape, metadata stays readable on a closed core.
    assert tensor.dtype == "float64" and tensor.device == "cpu"


def test_views_share_storage_metadata():
    base = np.arange(12.0).reshape(3, 4)
    with cpp.NativeTensorCore.from_array(base) as tensor:
        for view in (
            tensor.reshape((4, 3)),
            tensor.transpose(),
            tensor.T,
            tensor.narrow(1, 1, 2),
        ):
            assert view.dtype == tensor.dtype  # shared storage, one truth
            assert view.device == tensor.device


def test_operations_preserve_dtype_device():
    x = np.array([[1.0, -2.0], [3.0, 4.0]])
    y = np.array([[5.0, 6.0], [7.0, 8.0]])
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(y) as b:
        for result in (
            a.relu(),
            a.add(b),
            a.subtract(b),
            a.multiply(b),
            a.matmul(b),
            a.sum(),
            a.sum(axis=0),
            a.mean(axis=1, keepdims=True),
        ):
            assert result.dtype == "float64"
            assert result.device == "cpu"
            result.close()


def test_broadcasting_result_preserves_dtype_device():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    row = np.array([10.0, 20.0, 30.0])                 # (3,)
    with cpp.NativeTensorCore.from_array(x) as a, cpp.NativeTensorCore.from_array(row) as b:
        result = a.add(b)  # broadcasts to (2, 3)
        assert result.shape == (2, 3)
        assert result.dtype == "float64" and result.device == "cpu"
        result.close()


def test_contiguous_copy_preserves_dtype_device():
    with cpp.NativeTensorCore.from_array(np.arange(6.0).reshape(2, 3)) as tensor:
        transposed = tensor.T  # non-contiguous
        with transposed.contiguous_copy() as copy:
            assert copy.dtype == "float64" and copy.device == "cpu"


def test_to_numpy_still_returns_float64():
    x = np.array([[1.5, -2.5], [3.5, 4.5]])
    with cpp.NativeTensorCore.from_array(x) as tensor:
        out = tensor.to_numpy()
        assert out.dtype == np.float64
        assert np.array_equal(out, x)


def test_backend_info_reports_supported_dtype_device_sets():
    info = cpp.backend_info()
    assert info["dtype"] == "float64"
    assert info["device"] == "cpu"
    assert info["supported_dtypes"] == ("float64", "float32")
    assert info["supported_devices"] == ("cpu",)
