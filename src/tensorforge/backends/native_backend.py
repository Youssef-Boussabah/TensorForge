"""The native (C++) backend, wrapping the NativeTensorCore runtime.

This object is always constructible — importing it never requires the
compiled library. ``available()`` reports whether the library is
built; operations require it and raise the ``cpp`` module's helpful
ImportError (with build instructions) at call time if it is missing.

Operations produce and consume NativeTensorCore objects. Conversion
is explicit at both boundaries: ``tensor_from_array`` copies
NumPy/Python data *into* native storage, ``to_numpy`` materializes a
native value back *out*. The native backend never silently accepts a
tensorforge.Tensor. See docs/dispatch_design.md.
"""

from tensorforge.backends import cpp


class NativeBackend:
    name = "native"

    def available(self):
        return cpp.is_available()

    def backend_info(self):
        return cpp.backend_info()

    def tensor_from_array(self, values, dtype=None, device="cpu"):
        """The explicit conversion boundary in: NumPy/Python data in, a
        new NativeTensorCore out (a copy — no hidden aliasing).
        ``dtype``/``device`` default to ``"float64"``/``"cpu"`` and are
        rejected if unsupported."""
        return cpp.NativeTensorCore.from_array(values, dtype=dtype, device=device)

    def to_numpy(self, value):
        """The explicit conversion boundary out: a NativeTensorCore in,
        a fresh float64 NumPy array out (materialized, no shared state).

        Rejects anything that is not a NativeTensorCore — including a
        tensorforge.Tensor — with a clear TypeError."""
        return self._require_core(value, "to_numpy").to_numpy()

    def zeros(self, shape, dtype="float64", device="cpu"):
        return cpp.NativeTensorCore.zeros(shape, dtype=dtype, device=device)

    def full(self, shape, fill_value, dtype="float64", device="cpu"):
        return cpp.NativeTensorCore.full(shape, fill_value, dtype=dtype, device=device)

    def _require_core(self, value, op):
        if not isinstance(value, cpp.NativeTensorCore):
            raise TypeError(
                f"the native backend's {op} needs NativeTensorCore operands "
                f"(build one with tensor_from_array); got {type(value).__name__}"
            )
        return value

    def add(self, a, b):
        return self._require_core(a, "add").add(self._require_core(b, "add"))

    def relu(self, a):
        return self._require_core(a, "relu").relu()

    def matmul(self, a, b):
        return self._require_core(a, "matmul").matmul(self._require_core(b, "matmul"))

    def sum(self, a, axis=None, keepdims=False):
        return self._require_core(a, "sum").sum(axis=axis, keepdims=keepdims)

    def mean(self, a, axis=None, keepdims=False):
        return self._require_core(a, "mean").mean(axis=axis, keepdims=keepdims)

    def __repr__(self):
        return f"NativeBackend(available={self.available()})"
