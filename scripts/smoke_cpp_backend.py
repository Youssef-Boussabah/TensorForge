"""Hard-failing smoke check for the experimental C++ backend.

CI runs this right after building the backend, before pytest. Unlike
the pytest backend tests (which skip when the library is missing),
every failure here — import, kernel math, runtime objects — fails the
run. Locally: uv run python scripts/smoke_cpp_backend.py
"""

import numpy as np

from tensorforge.backends.cpp import (
    NativeStorage,
    NativeTensorCore,
    NativeTensorView,
    backend_info,
    elementwise_add,
    elementwise_divide,
    elementwise_multiply,
    elementwise_subtract,
    is_available,
    list_kernels,
    matmul,
    matmul_tiled,
    relu,
    shape_info,
)


def main():
    # Kernels compute correct values.
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    assert elementwise_add(a, b).tolist() == [4.0, 6.0]
    assert elementwise_subtract(b, a).tolist() == [2.0, 2.0]
    assert elementwise_multiply(a, b).tolist() == [3.0, 8.0]
    assert elementwise_divide(b, a).tolist() == [3.0, 2.0]
    assert relu(np.array([-1.0, 0.0, 2.0])).tolist() == [0.0, 0.0, 2.0]

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    expected = [[58.0, 64.0], [139.0, 154.0]]
    assert matmul(x, y).tolist() == expected
    assert matmul_tiled(x, y).tolist() == expected

    # Introspection.
    assert is_available() is True
    assert "matmul_tiled" in list_kernels()
    assert backend_info()["tensor_core"] == "NativeTensorCore"
    assert shape_info((2, 3))["strides"] == (3, 1)

    # Runtime objects: storage, view, tensor core.
    storage = NativeStorage.from_array([1.0, 2.0])
    assert storage.to_numpy().tolist() == [1.0, 2.0]
    storage.close()

    view = NativeTensorView.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert view.to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]

    tensor = NativeTensorCore.from_array(x)
    assert tensor.T.to_numpy().tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert tensor.reshape((3, 2)).to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert tensor.narrow(1, 1, 2).to_numpy().tolist() == [[2.0, 3.0], [5.0, 6.0]]

    # Native kernels over tensor cores, strided inputs included.
    other = NativeTensorCore.from_array(np.ones((2, 3)))
    assert tensor.add(other).to_numpy().tolist() == [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
    signs = NativeTensorCore.from_array([[-1.0, 2.0], [3.0, -4.0]])
    assert signs.relu().to_numpy().tolist() == [[0.0, 2.0], [3.0, 0.0]]
    assert signs.T.multiply(signs.T).to_numpy().tolist() == [[1.0, 9.0], [4.0, 16.0]]
    other.close()
    signs.close()
    tensor.close()

    print("cpp backend smoke ok")


if __name__ == "__main__":
    main()
