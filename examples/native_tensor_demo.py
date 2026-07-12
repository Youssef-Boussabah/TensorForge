"""A tour of the experimental NativeTensor wrapper.

NativeTensor lives in ``tensorforge.experimental`` and wraps the native
C++ runtime (NativeTensorCore): C++-owned storage, shape/stride
metadata, and forward kernels. It is deliberately **not**
``tensorforge.Tensor`` — no autograd, no optimizer/Module integration,
no CUDA, no operator overloads. Data crosses the native boundary only by
explicit call (``from_array`` in, ``to_numpy`` out), and native memory is
released explicitly (``close()`` or a ``with`` block).

This script is a demonstration, not a benchmark: it makes no performance
claims. It needs the experimental C++ backend to be built — run:

    uv run python examples/native_tensor_demo.py

``demo()`` returns its results as a dict of NumPy arrays so the tests can
import and verify it; ``main()`` prints them.
"""

import numpy as np

from tensorforge.backends import cpp
from tensorforge.experimental import NativeTensor


def demo():
    """Run the whole NativeTensor tour and return the results as a dict
    of NumPy arrays. Assumes the native backend is built."""
    results = {}

    # -- construction (explicit entry boundary) -----------------------
    a = NativeTensor.from_array([[1.0, -2.0, 3.0], [-4.0, 5.0, 6.0]])
    ones = NativeTensor.full((2, 3), 1.0)
    results["a"] = a.to_numpy()
    results["ones"] = ones.to_numpy()

    # -- forward compute (each returns a new owning NativeTensor) ------
    relud = a.relu()
    summed = a.add(ones)
    results["relu"] = relud.to_numpy()
    results["add"] = summed.to_numpy()

    # matmul: (2, 3) @ (3, 2)
    b = NativeTensor.from_array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    product = a.matmul(b)
    results["matmul"] = product.to_numpy()

    # -- metadata-only views (borrow a's storage, no copy) ------------
    reshaped = a.reshape((3, 2))
    transposed = a.T
    narrowed = a.narrow(1, 0, 2)  # keep the first two columns
    results["reshape"] = reshaped.to_numpy()
    results["transpose"] = transposed.to_numpy()
    results["narrow"] = narrowed.to_numpy()

    # -- materialize a strided view into fresh owning storage ----------
    made_contiguous = transposed.contiguous_copy()
    results["contiguous_copy"] = made_contiguous.to_numpy()
    results["contiguous_flag"] = made_contiguous.contiguous

    # -- explicit release (views first, then owners) ------------------
    for tensor in (
        relud, summed, product, reshaped, transposed, narrowed,
        made_contiguous, b, ones, a,
    ):
        tensor.close()

    # -- context-manager form frees deterministically on block exit ----
    with NativeTensor.zeros((2, 2)) as z:
        results["zeros"] = z.to_numpy()

    return results


def main():
    if not cpp.is_available():
        print("The experimental C++ backend is not built.")
        print(cpp.build_instructions())
        return

    np.set_printoptions(precision=4, suppress=True)
    r = demo()
    print("NativeTensor - experimental native C++ backend")
    print("=" * 46)
    print("from_array a =\n", r["a"])
    print("full ones =\n", r["ones"])
    print("relu(a) =\n", r["relu"])
    print("a.add(ones) =\n", r["add"])
    print("a.matmul(b) =\n", r["matmul"])
    print("a.reshape((3, 2)) =\n", r["reshape"])
    print("a.T =\n", r["transpose"])
    print("a.narrow(1, 0, 2) =\n", r["narrow"])
    print(
        "a.T.contiguous_copy() =\n", r["contiguous_copy"],
        "\n  (contiguous =", r["contiguous_flag"], ")",
    )
    print("zeros((2, 2)) =\n", r["zeros"])


if __name__ == "__main__":
    main()
