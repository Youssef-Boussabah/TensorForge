"""Tests for NativeStorage, the C++-owned float64 buffer.

Everything here needs the compiled backend, so the module skips
cleanly when it is not built — same convention as the kernel tests.
No test depends on garbage-collection timing; cleanup is explicit
via close() or context managers.
"""

import numpy as np
import pytest

from tensorforge.backends import cpp

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)


def test_native_storage_is_importable():
    from tensorforge.backends.cpp import NativeStorage  # noqa: F401

    assert callable(cpp.NativeStorage)


def test_new_storage_is_zero_initialized():
    with cpp.NativeStorage(5) as storage:
        assert storage.size == 5
        assert storage.to_numpy().tolist() == [0.0] * 5


def test_fill():
    with cpp.NativeStorage(4) as storage:
        storage.fill(2.5)
        assert storage.to_numpy().tolist() == [2.5] * 4
        storage.fill(-1)  # ints convert to float64
        assert storage.to_numpy().tolist() == [-1.0] * 4


def test_copy_from_list_and_numpy_and_2d():
    with cpp.NativeStorage(4) as storage:
        storage.copy_from([1, 2, 3, 4])
        assert storage.to_numpy().tolist() == [1.0, 2.0, 3.0, 4.0]
        storage.copy_from(np.array([5.0, 6.0, 7.0, 8.0]))
        assert storage.to_numpy().tolist() == [5.0, 6.0, 7.0, 8.0]
        # Multi-dimensional input is flattened (documented behavior).
        storage.copy_from(np.array([[9.0, 10.0], [11.0, 12.0]]))
        assert storage.to_numpy().tolist() == [9.0, 10.0, 11.0, 12.0]


def test_from_array():
    with cpp.NativeStorage.from_array(np.arange(6.0).reshape(2, 3)) as storage:
        assert storage.size == 6
        assert storage.to_numpy().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_to_numpy_is_a_new_independent_array():
    with cpp.NativeStorage.from_array([1.0, 2.0]) as storage:
        first = storage.to_numpy()
        second = storage.to_numpy()
        assert first.dtype == np.float64
        assert first is not second
        assert not np.shares_memory(first, second)
        first[0] = 99.0  # mutating the copy must not touch the storage
        assert storage.to_numpy().tolist() == [1.0, 2.0]


def test_source_array_is_decoupled_after_copy_from():
    source = np.array([1.0, 2.0, 3.0])
    with cpp.NativeStorage.from_array(source) as storage:
        source[0] = 99.0  # the storage holds its own copy
        assert storage.to_numpy().tolist() == [1.0, 2.0, 3.0]


def test_copy_from_rejects_wrong_sizes():
    with cpp.NativeStorage(3) as storage:
        with pytest.raises(ValueError, match="exactly 3"):
            storage.copy_from([1.0, 2.0])
        with pytest.raises(ValueError, match="exactly 3"):
            storage.copy_from([1.0, 2.0, 3.0, 4.0])


def test_invalid_sizes_are_rejected():
    for bad in (0, -1, 2.5, True, "3", None):
        with pytest.raises(ValueError, match="positive int"):
            cpp.NativeStorage(bad)
    with pytest.raises(ValueError, match="positive int"):
        cpp.NativeStorage.from_array([])  # empty input -> size 0


def test_operations_after_close_raise():
    storage = cpp.NativeStorage(2)
    storage.close()
    for operation in (
        storage.to_numpy,
        lambda: storage.fill(1.0),
        lambda: storage.copy_from([1.0, 2.0]),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


def test_double_close_is_safe():
    storage = cpp.NativeStorage(2)
    storage.close()
    storage.close()  # must not crash or double-free
    assert "closed" in repr(storage)


def test_context_manager_closes():
    with cpp.NativeStorage.from_array([1.0]) as storage:
        assert storage.to_numpy().tolist() == [1.0]
    with pytest.raises(RuntimeError, match="closed"):
        storage.to_numpy()


def test_larger_roundtrip():
    rng = np.random.default_rng(11)
    values = rng.normal(size=10_000)
    with cpp.NativeStorage.from_array(values) as storage:
        assert np.array_equal(storage.to_numpy(), values)


def test_backend_info_advertises_storage_and_kernels_stay_kernels():
    info = cpp.backend_info()
    assert info["storage_object"] == "NativeStorage"
    assert "NativeStorage" not in cpp.list_kernels()
