"""A tour of native autograd on the experimental NativeTensor.

As of Advanced C++ v2.2 the native runtime differentiates its core
operations: this demo builds ``x.matmul(w).add(b).relu().mean()`` out of
NativeTensors — the data ``x`` is a plain forward tensor, the weights
``w`` and bias ``b`` require grad — calls ``backward()`` on the scalar
loss, and reads the native gradients back. The bias add broadcasts a
``(2,)`` over a ``(4, 2)``, so its backward exercises the native
unbroadcast reduction; every gradient is computed by native kernels
(NumPy appears only to print and to hand results to the tests).

This is a demonstration, not a training loop: no optimizer, no dataset,
no ``tensorforge.Tensor``, no performance claims. It needs the
experimental C++ backend to be built — run:

    uv run python examples/native_autograd_demo.py

``demo()`` returns its results as a dict of NumPy arrays/floats so the
tests can import and verify it; ``main()`` prints them.
"""

import numpy as np

from tensorforge.backends import cpp
from tensorforge.experimental import NativeTensor

X_VALUES = [
    [1.0, -2.0, 0.5],
    [3.0, 0.0, -1.0],
    [-0.5, 2.0, 1.0],
    [2.0, 1.0, -3.0],
]
W_VALUES = [
    [0.5, -1.0],
    [1.0, 0.5],
    [-0.5, 1.0],
]
B_VALUES = [0.5, -0.5]


def demo():
    """Run one deterministic native forward + backward and return the
    inputs, the loss, and the gradients as NumPy copies. Assumes the
    native backend is built."""
    x = NativeTensor.from_array(X_VALUES)                      # data: no grad
    w = NativeTensor.from_array(W_VALUES, requires_grad=True)  # (3, 2) leaf
    b = NativeTensor.from_array(B_VALUES, requires_grad=True)  # (2,) leaf

    # (4, 3) @ (3, 2) -> broadcast bias add -> relu -> scalar mean.
    hidden = x.matmul(w)
    shifted = hidden.add(b)
    activated = shifted.relu()
    loss = activated.mean()

    loss.backward()  # scalar output: the seed defaults to a native 1.0

    results = {
        "x": x.to_numpy(),
        "w": w.to_numpy(),
        "b": b.to_numpy(),
        "activated": activated.to_numpy(),
        "loss": float(loss.to_numpy()),
        "w_grad": w.grad.to_numpy(),
        "b_grad": b.grad.to_numpy(),
    }

    # Explicit release: intermediates first, then the leaves. Gradients
    # are dropped with their tensors; nothing here relies on GC timing.
    for tensor in (loss, activated, shifted, hidden, b, w, x):
        tensor.close()
    return results


def main():
    if not cpp.is_available():
        print("The experimental C++ backend is not built.")
        print(cpp.build_instructions())
        return

    np.set_printoptions(precision=4, suppress=True)
    r = demo()
    print("Native autograd - x.matmul(w).add(b).relu().mean()")
    print("=" * 50)
    print("x (data, requires_grad=False) =\n", r["x"])
    print("w (leaf, requires_grad=True) =\n", r["w"])
    print("b (leaf, requires_grad=True, broadcast over rows) =", r["b"])
    print("relu(x @ w + b) =\n", r["activated"])
    print("loss = mean(...) =", r["loss"])
    print("after loss.backward():")
    print("w.grad =\n", r["w_grad"])
    print("b.grad =", r["b_grad"])


if __name__ == "__main__":
    main()
