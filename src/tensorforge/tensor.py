"""TensorForge v0.1 — a tiny NumPy-backed Tensor with reverse-mode autodiff.

The design follows the classic "define-by-run" recipe:

1. Every operation computes its result eagerly with NumPy.
2. The result Tensor remembers which Tensors produced it (``_prev``) and
   how to send gradients back to them (``_backward``).
3. ``backward()`` walks that recorded graph in reverse topological order,
   applying the chain rule one node at a time.
"""

import numpy as np


def _ensure_tensor(value):
    """Wrap plain numbers / arrays so ops like ``x + 2.0`` work."""
    return value if isinstance(value, Tensor) else Tensor(value)


def _unbroadcast(grad, shape):
    """Undo NumPy broadcasting so ``grad`` matches ``shape``.

    If the forward pass broadcast an input (e.g. a row vector added to a
    matrix), the incoming gradient has the *broadcast* shape. Summing over
    the broadcast axes gives the gradient for the original input, because
    each broadcast copy contributed to the output.
    """
    # Axes that broadcasting prepended: sum them away entirely.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Axes that were stretched from size 1: sum back down to size 1.
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """A NumPy array plus the bookkeeping needed for autograd."""

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.requires_grad = requires_grad
        # Autograd graph bookkeeping (set by the op that created this Tensor).
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def _accumulate_grad(self, grad):
        """Add ``grad`` into ``self.grad`` (gradients from multiple uses sum up)."""
        if not self.requires_grad:
            return
        grad = _unbroadcast(grad, self.data.shape)
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad = self.grad + grad

    # ------------------------------------------------------------------
    # Primitive operations (each defines its own local derivative)
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="+",
        )

        def _backward():
            # d(a + b)/da = 1 and d(a + b)/db = 1, so the gradient
            # flows through unchanged to both inputs.
            self._accumulate_grad(out.grad)
            other._accumulate_grad(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="*",
        )

        def _backward():
            # d(a * b)/da = b and d(a * b)/db = a.
            self._accumulate_grad(other.data * out.grad)
            other._accumulate_grad(self.data * out.grad)

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("Tensor ** only supports int/float exponents for now")
        out = Tensor(
            self.data ** exponent,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{exponent}",
        )

        def _backward():
            # d(a**n)/da = n * a**(n-1).
            self._accumulate_grad(exponent * self.data ** (exponent - 1) * out.grad)

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(
            self.data.sum(),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward():
            # Every element contributed with weight 1, so the scalar
            # gradient is spread back to every position.
            self._accumulate_grad(np.ones_like(self.data) * out.grad)

        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(
            self.data.mean(),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="mean",
        )

        def _backward():
            # Like sum, but each element contributed with weight 1/N.
            self._accumulate_grad(np.ones_like(self.data) * out.grad / self.data.size)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(
            np.maximum(self.data, 0.0),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        def _backward():
            # Gradient passes through where the input was positive,
            # and is blocked where relu clamped to zero.
            self._accumulate_grad((self.data > 0) * out.grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Derived operations (built from the primitives above, so they get
    # their gradients for free)
    # ------------------------------------------------------------------

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-_ensure_tensor(other))

    def __truediv__(self, other):
        return self * _ensure_tensor(other) ** -1.0

    # Reflected variants so plain numbers can appear on the left: 2.0 * x.
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return _ensure_tensor(other) - self

    def __rtruediv__(self, other):
        return _ensure_tensor(other) / self

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def backward(self):
        """Run reverse-mode autodiff from this Tensor back to its inputs."""
        if not self.requires_grad:
            raise RuntimeError(
                "backward() called on a Tensor that does not require grad"
            )

        # Topologically sort the graph so every node runs its _backward
        # only after all of its consumers have contributed gradient to it.
        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    build_topo(parent)
                topo.append(tensor)

        build_topo(self)

        # Seed: d(self)/d(self) = 1.
        self.grad = np.ones_like(self.data)
        for tensor in reversed(topo):
            # Nodes that don't require grad never receive a gradient,
            # so they have nothing to propagate to their parents.
            if tensor.requires_grad:
                tensor._backward()
